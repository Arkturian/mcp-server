"""Gateway prueft JWTs mit agentos_jwt v1.0.1 (#3161, Teil von #2972/#2970).

Vorher eigene Pruefung in auth.py: Rueckfall auf den einzigen Schluessel bei Tokens
ohne kid, eigener Widerrufs-Abruf, aud-Tor in _token_abweisen, verify_aud=False.
Jetzt das gemeinsame Modul (vendort, sha256 unten geprueft). Hier alles mit echt
signierten Tokens: eigene RSA-Schluessel, JWKS und Widerrufsliste kommen aus einer
Attrappe (afetch), kein Netz.

Pruefpunkte aus #3152 (gleiche Liste fuer alle Dienste):
  unsigniertes Token mit type ["agent"] -> 401 statt 500, widerrufene jti -> 401,
  unbekannte kid -> 401, nicht angenommene Klasse -> 401, gueltiges Agenten-Token -> 200.
Gateway (Entscheid Cloud 060afe45, Uebergang B bis OBO #3004):
  oauth_access mit aud aus MCP_ERLAUBTE_AUD und is_superadmin im signierten Token ->
  "*"; aud nicht in der Liste -> 401; Schalter aus -> keine Rechte.
Uebernommen aus test_aud_tor_2976 / test_jwt_revoked / test_jwt_audience (die alten
Interna gibt es nicht mehr): Voice-Grant, fremde Zielgruppe, Liste als aud, Rotation
nach kid, min_agent_iat, Ablauf, Aussteller, fremde Signatur, Notnagel REVOKED_AGENT_JTIS.
"""
import asyncio
import hashlib
import json
import os
import stat
import time
from pathlib import Path

import httpx
import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from jwt.algorithms import RSAAlgorithm
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route

for _k in ("ARKTURIAN_API_KEY", "ONEAL_STORAGE_API_KEY", "JWT_PUBLIC_KEY", "COMM_API_KEY"):
    os.environ.setdefault(_k, "test-only")

import agentos_jwt  # noqa: E402
import auth  # noqa: E402

SHA256 = "ff39755338d7e96d24e2b6a5bdfb6f16e4b6b5f07e7cd541f706f2c6dee2063d"
CALL = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                   "params": {"name": "werkzeug", "arguments": {}}})
CHATGPT = "e03f38c4-0000-4000-8000-000000000001"          # Form wie in auth_db, ausgedacht
FREMD = "9d1c0de0-0000-4000-8000-0000000000ff"           # anonym per DCR registriert
ISS = "auth-api.arkturian.com"

_K1 = rsa.generate_private_key(public_exponent=65537, key_size=2048)
_K2 = rsa.generate_private_key(public_exponent=65537, key_size=2048)
_FREMDER = rsa.generate_private_key(public_exponent=65537, key_size=2048)


def _jwk(key, kid):
    d = json.loads(RSAAlgorithm.to_jwk(key.public_key()))
    d.update(kid=kid, alg="RS256", use="sig")
    return d


def _token(claims=None, key=_K1, kid="k1", **ueber):
    jetzt = int(time.time())
    c = {"iss": ISS, "sub": "agent:Probe", "iat": jetzt, "exp": jetzt + 600, "jti": f"j-{time.time_ns()}"}
    c.update(claims or {})
    c.update(ueber)
    kopf = {"kid": kid} if kid else {}
    return jwt.encode(c, key, algorithm="RS256", headers=kopf)


def _agent(**c):
    return {"type": "agent", "agent_name": "Probe", "permissions": ["mcp:cloud:*"], **c}


def _connector(aud=CHATGPT, superadmin=True, scope="openid", **c):
    return {"type": "oauth_access", "sub": "u-alex", "email": "alex@example.test", "aud": aud,
            "scope": scope, "is_superadmin": superadmin, **c}


def _app():
    async def ok(request):
        return JSONResponse({"ok": 1, "perms": request.state.agent.permissions,
                             "superadmin": auth.current_caller_is_superadmin()})
    app = Starlette(routes=[Route("/{srv}/", ok, methods=["POST"])])
    app.add_middleware(auth.JWTAuthMiddleware)
    return app


class Welt:
    """auth-api-Attrappe: JWKS und Widerrufsliste, abrufbar ueber afetch."""

    def __init__(self):
        self.keys = [_jwk(_K1, "k1")]
        self.revoked: list[str] = []
        self.min_iat = None
        self.jwks_weg = False
        self.rev_weg = False
        self.abrufe: list[str] = []
        self.uhr = 1_000_000.0

    async def afetch(self, url, params):
        self.abrufe.append(url.rsplit("/", 1)[-1])
        if url.endswith("/jwks.json"):
            if self.jwks_weg:
                raise httpx.ConnectError("auth-api weg")
            return {"keys": list(self.keys)}
        if url.endswith("/revoked-agent-jtis"):
            if self.rev_weg:
                raise httpx.ConnectError("auth-api weg")
            return {"revoked_jtis": [{"jti": j} for j in self.revoked], "as_of": "2026-11-16T10:00:00Z",
                    "min_agent_iat": self.min_iat}
        raise AssertionError(url)


@pytest.fixture
def gw(monkeypatch, tmp_path):
    monkeypatch.setattr(auth, "AUTH_REQUIRE_JWT", True)
    monkeypatch.setattr(auth, "AUTH_API_ISSUER", ISS)
    monkeypatch.setattr(auth, "_ERLAUBTE_AUD", [CHATGPT])
    monkeypatch.setattr(auth, "_SUPERADMIN_OHNE_SCOPE", True)
    monkeypatch.delenv("REVOKED_AGENT_JTIS", raising=False)
    welt = Welt()

    def neu(state_dir=None, **anders):
        art = {"state_dir": str(state_dir)} if state_dir else {"ephemeral_state": True}
        v = auth._verifier_bauen(afetch=welt.afetch, clock=lambda: welt.uhr, **art, **anders)
        monkeypatch.setattr(auth, "_verifier_obj", v)
        return v
    welt.neu = neu
    neu()

    def ruf(token, pfad="/cloud/"):
        async def go():
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=_app()), base_url="http://gw") as c:
                return await c.post(pfad, content=CALL, headers={
                    "Authorization": f"Bearer {token}", "Content-Type": "application/json"})
        return asyncio.run(go())
    welt.ruf = ruf
    return welt


# ── Das Modul selbst ────────────────────────────────────────────────────────────
def test_vendort_byte_gleich_zu_jwt_v1_0_1():
    roh = (Path(auth.__file__).parent / "agentos_jwt.py").read_bytes()
    assert hashlib.sha256(roh).hexdigest() == SHA256
    assert agentos_jwt.__version__ == "1.0.1"


def test_angenommene_klassen_und_issuer(monkeypatch):
    monkeypatch.setattr(auth, "_ERLAUBTE_AUD", [CHATGPT, "e03f*"])
    v = auth._verifier_bauen(ephemeral_state=True)
    assert v.accept == {"agent": None, "service": None, "user": None, "oauth_access": frozenset({CHATGPT})}
    assert v.issuer == "auth-api.arkturian.com"
    monkeypatch.setattr(auth, "_ERLAUBTE_AUD", [])
    assert "oauth_access" not in auth._verifier_bauen(ephemeral_state=True).accept


def test_ohne_state_dir_kein_verifier(monkeypatch):
    """Pflicht, sobald agent/service angenommen werden (README)."""
    with pytest.raises(ValueError):
        auth._verifier_bauen(state_dir=None)


def test_adressen_aus_base_url_oder_jwks_url(monkeypatch):
    monkeypatch.setattr(auth, "MCP_JWT_BASE_URL", "http://localhost:8050")
    v = auth._verifier_bauen(ephemeral_state=True)
    assert v.jwks_url == "http://localhost:8050/api/v1/auth/.well-known/jwks.json"
    assert v.revocation_url == "http://localhost:8050/api/v1/auth/.well-known/revoked-agent-jtis"
    monkeypatch.setattr(auth, "MCP_JWT_BASE_URL", "")
    monkeypatch.setattr(auth, "AUTH_API_JWKS_URL", "https://a.test/api/v1/auth/.well-known/jwks.json")
    v = auth._verifier_bauen(ephemeral_state=True)
    assert v.revocation_url == "https://a.test/api/v1/auth/.well-known/revoked-agent-jtis"


# ── Pruefpunkte aus #3152 ───────────────────────────────────────────────────────
def test_gueltiges_agenten_token_200(gw):
    r = gw.ruf(_token(_agent()))
    assert r.status_code == 200 and r.json()["perms"] == ["mcp:cloud:*"]


def test_unsigniertes_token_mit_typ_als_liste_401_statt_500(gw):
    roh = jwt.encode({"iss": ISS, "sub": "x", "type": ["agent"], "iat": int(time.time()),
                      "exp": int(time.time()) + 600}, key=None, algorithm="none")
    r = gw.ruf(roh)
    assert r.status_code == 401 and r.json()["reason"] in ("unknown_kid", "malformed")


def test_signiertes_token_mit_typ_als_liste_401(gw):
    r = gw.ruf(_token(_agent(type=["agent"])))
    assert r.status_code == 401 and r.json()["reason"] == "malformed"


def test_widerrufene_jti_401(gw):
    gw.revoked = ["weg-1"]
    r = gw.ruf(_token(_agent(), jti="weg-1"))
    assert r.status_code == 401 and r.json()["reason"] == "jti_revoked"


def test_unbekannte_kid_401(gw):
    r = gw.ruf(_token(_agent(), key=_K2, kid="k9"))
    assert r.status_code == 401 and r.json()["reason"] == "unknown_kid"


@pytest.mark.parametrize("typ", ["realtime_grant", "ephemeral_capability", "delegated", "irgendwas"])
def test_nicht_angenommene_klasse_401(gw, typ):
    r = gw.ruf(_token({"type": typ, "sub": "x"}))
    assert r.status_code == 401 and r.json()["reason"] == "type_not_accepted"


def test_token_ohne_type_401(gw):
    """Alte Tokens ohne type: kein untyped_as (Vorgabe des Moduls)."""
    r = gw.ruf(_token({"sub": "u-alex", "is_superadmin": True}))
    assert r.status_code == 401 and r.json()["reason"] == "type_not_accepted"


# ── Connectoren, Uebergang (B) ──────────────────────────────────────────────────
def test_connector_aus_der_liste_ohne_scope_superadmin_alles(gw):
    r = gw.ruf(_token(_connector()))
    assert r.status_code == 200
    assert r.json()["perms"] == ["*"] and r.json()["superadmin"] is True


def test_connector_nicht_in_der_liste_401(gw):
    r = gw.ruf(_token(_connector(aud=FREMD)))
    assert r.status_code == 401 and r.json()["reason"] == "aud_mismatch"


def test_schalter_aus_connector_ohne_scope_keine_rechte(gw, monkeypatch):
    monkeypatch.setattr(auth, "_SUPERADMIN_OHNE_SCOPE", False)
    r = gw.ruf(_token(_connector()))
    assert r.status_code == 403


def test_connector_ohne_superadmin_nur_seine_scopes(gw):
    r = gw.ruf(_token(_connector(superadmin=False, scope="openid mcp:cloud")))
    assert r.status_code == 200 and r.json()["perms"] == ["mcp:cloud:*"] and r.json()["superadmin"] is False
    r = gw.ruf(_token(_connector(superadmin=False)))
    assert r.status_code == 403


def test_connector_mit_scope_bleibt_bei_seinen_scopes_auch_als_superadmin(gw):
    r = gw.ruf(_token(_connector(scope="mcp:content")), pfad="/content/")
    assert r.json()["perms"] == ["mcp:content:*"]


def test_aud_liste_im_token(gw):
    r = gw.ruf(_token(_connector(aud=[FREMD, CHATGPT])))
    assert r.status_code == 200 and r.json()["perms"] == ["*"]


def test_leere_liste_jeder_connector_401(gw, monkeypatch):
    monkeypatch.setattr(auth, "_ERLAUBTE_AUD", [])
    gw.neu()
    r = gw.ruf(_token(_connector()))
    assert r.status_code == 401 and r.json()["reason"] == "type_not_accepted"


def test_muster_in_der_liste_gilt_nicht_mehr(gw, monkeypatch, caplog):
    monkeypatch.setattr(auth, "_ERLAUBTE_AUD", ["e03f38c4-*"])
    gw.neu()
    r = gw.ruf(_token(_connector()))
    assert r.status_code == 401
    assert "gilt seit #3161 nicht mehr" in caplog.text


@pytest.mark.parametrize("aud", ["opencloud-web", "ai-realtime:alex"])
def test_fremde_zielgruppen_401(gw, monkeypatch, aud):
    """OpenCloud-Web und Voice-Grants -- auch wenn jemand sie in die Liste schriebe."""
    monkeypatch.setattr(auth, "_ERLAUBTE_AUD", [CHATGPT, aud])
    gw.neu()
    r = gw.ruf(_token({"type": "realtime_grant", "sub": "u", "aud": aud}))
    assert r.status_code == 401
    r = gw.ruf(_token(_connector(aud="opencloud-desktop")))
    assert r.status_code == 401 and r.json()["reason"] == "aud_mismatch"


# ── Superadmin nur aus dem Modul fuer alle anderen Klassen ─────────────────────
def test_agent_mit_is_superadmin_im_token_ist_kein_superadmin(gw):
    r = gw.ruf(_token(_agent(is_superadmin=True)))
    assert r.json()["superadmin"] is False and r.json()["perms"] == ["mcp:cloud:*"]


def test_eigenes_login_superadmin_alles(gw):
    r = gw.ruf(_token({"type": "user", "sub": "u-alex", "is_superadmin": True}))
    assert r.status_code == 200 and r.json()["perms"] == ["*"] and r.json()["superadmin"] is True


def test_eigenes_login_ohne_superadmin_keine_rechte(gw):
    assert gw.ruf(_token({"type": "user", "sub": "u-b", "is_superadmin": False})).status_code == 403


def test_login_superadmin_als_text_ist_kein_superadmin(gw):
    r = gw.ruf(_token({"type": "user", "sub": "u-alex", "is_superadmin": "true"}))
    assert r.status_code == 403


def test_login_mit_aud_401(gw):
    r = gw.ruf(_token({"type": "user", "sub": "u-alex", "aud": CHATGPT, "is_superadmin": True}))
    assert r.status_code == 401 and r.json()["reason"] == "aud_mismatch"


# ── Schluessel, Ablauf, Aussteller ──────────────────────────────────────────────
def test_token_ohne_kid_401_auch_bei_einem_schluessel(gw):
    """Der alte Rueckfall (einziger Schluessel) ist weg."""
    r = gw.ruf(_token(_agent(), kid=None))
    assert r.status_code == 401 and r.json()["reason"] == "unknown_kid"


def test_rotation_zweiter_schluessel_nach_kid(gw):
    assert gw.ruf(_token(_agent())).status_code == 200
    gw.keys.append(_jwk(_K2, "k2"))
    gw.uhr += 11                              # Nachladen hoechstens alle 10 s
    assert gw.ruf(_token(_agent(), key=_K2, kid="k2")).status_code == 200


def test_unbekannte_kid_laedt_hoechstens_alle_10_s_nach(gw):
    gw.ruf(_token(_agent()))
    vorher = gw.abrufe.count("jwks.json")
    for _ in range(5):
        assert gw.ruf(_token(_agent(), key=_K2, kid="k9")).status_code == 401
    assert gw.abrufe.count("jwks.json") == vorher


@pytest.mark.parametrize("ueber,grund", [
    ({"exp": int(time.time()) - 3600}, "expired"),
    ({"iss": "https://boese.example"}, "bad_issuer"),
])
def test_ablauf_und_aussteller(gw, ueber, grund):
    r = gw.ruf(_token(_agent(), **ueber))
    assert r.status_code == 401 and r.json()["reason"] == grund


def test_fremde_signatur_401(gw):
    r = gw.ruf(_token(_agent(), key=_FREMDER, kid="k1"))
    assert r.status_code == 401 and r.json()["reason"] == "bad_signature"


# ── Widerruf: min_agent_iat, Zustand, Ausfall der auth-api ─────────────────────
def test_min_agent_iat(gw):
    gw.min_iat = "2099-01-01T00:00:00Z"
    r = gw.ruf(_token(_agent()))
    assert r.status_code == 401 and r.json()["reason"] == "iat_before_cutoff"


def test_jwks_nicht_erreichbar_503(gw):
    gw.jwks_weg = True
    r = gw.ruf(_token(_agent()))
    assert r.status_code == 503 and r.json()["reason"] == "KeysUnavailable"


def test_widerruf_unbekannt_503_fuer_agenten_login_geht(gw):
    gw.rev_weg = True
    r = gw.ruf(_token(_agent()))
    assert r.status_code == 503 and r.json()["reason"] == "RevocationUnknown"
    r = gw.ruf(_token({"type": "user", "sub": "u-alex", "is_superadmin": True}))
    assert r.status_code == 200


def test_zustand_auf_platte_ueberlebt_neustart_ohne_auth_api(gw, tmp_path):
    ort = tmp_path / "agentos_jwt"
    gw.neu(state_dir=ort)
    gw.revoked = ["weg-1"]
    assert gw.ruf(_token(_agent())).status_code == 200
    datei = ort / "revocation.json"
    assert stat.S_IMODE(os.stat(ort).st_mode) == 0o700 and stat.S_IMODE(os.stat(datei).st_mode) == 0o600
    gw.rev_weg = True                         # Neustart, auth-api weg
    gw.neu(state_dir=ort)
    assert gw.ruf(_token(_agent())).status_code == 200
    assert gw.ruf(_token(_agent(), jti="weg-1")).json()["reason"] == "jti_revoked"


def test_notnagel_revoked_agent_jtis(gw, monkeypatch):
    monkeypatch.setenv("REVOKED_AGENT_JTIS", "a, notfall-1")
    r = gw.ruf(_token(_agent(), jti="notfall-1"))
    assert r.status_code == 401 and r.json()["reason"] == "jti_revoked"
    assert gw.ruf(_token(_agent(), jti="frei")).status_code == 200


def test_ohne_strenge_bei_ausfall_anonym_wie_bisher(gw, monkeypatch):
    monkeypatch.setattr(auth, "AUTH_REQUIRE_JWT", False)
    gw.jwks_weg = True
    r = gw.ruf(_token(_agent()))
    # wie vorher bei fehlender JWKS: weiter als anonymer Aufrufer, ohne jedes Recht
    assert r.status_code == 200 and r.json()["perms"] == [] and r.json()["superadmin"] is False
