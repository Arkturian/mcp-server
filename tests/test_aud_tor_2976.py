"""#2976 (Folge #2970 Befund 76, #2972 Befund 78): welches Token gilt am Gateway?

Befund 76: Jedes Token mit is_superadmin bekam ["*"] auf alle Werkzeuge, ohne dass
`aud` geprueft wurde. Ein anonym per DCR registrierter Client mit scope=openid reichte,
sobald sich ein Superadmin dort anmeldete. Voice-Grants (aud ai-realtime:*) und
Capability-Tokens (type ephemeral_capability) wurden gar nicht abgewiesen.
Befund 78: min_agent_iat der auth-api blieb unbeachtet, die JWKS galt nur mit keys[0].

Jetzt (Entscheid Cloud 15.11.2026):
- `aud` muss in MCP_ERLAUBTE_AUD stehen (je Gateway, fnmatch), sonst 401, auch bei
  leerer Liste. Ein Token aus der Liste gilt wie ein eigenes Login.
- Token-Arten: ohne type, agent, service, user, oauth_access. Alles andere 401.
- ai-realtime:* nie, auch nicht ueber die Liste.
- min_agent_iat fuer Agenten- und Dienst-Tokens.
- Schluessel je `kid`.

Alles gegen Vorrichtungen: kein Netz, keine echten Tokens.
"""
import asyncio
import base64
import json
import os
import time

import httpx
import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route

for _k in ("ARKTURIAN_API_KEY", "ONEAL_STORAGE_API_KEY", "JWT_PUBLIC_KEY", "COMM_API_KEY"):
    os.environ.setdefault(_k, "test-only")

import auth  # noqa: E402

CALL = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                   "params": {"name": "werkzeug", "arguments": {}}})
CHATGPT = "e03f38c4-0000-4000-8000-000000000001"           # Form wie in auth_db, ausgedacht
FREMD = "9d1c0de0-0000-4000-8000-0000000000ff"            # anonym per DCR registriert


def _app():
    async def ok(request):
        return JSONResponse({"ok": 1, "perms": request.state.agent.permissions})
    app = Starlette(routes=[Route("/{srv}/", ok, methods=["POST"])])
    app.add_middleware(auth.JWTAuthMiddleware)
    return app


@pytest.fixture
def gw(monkeypatch, tmp_path):
    monkeypatch.setattr(auth, "AUTH_REQUIRE_JWT", True)
    monkeypatch.setattr(auth, "_REVOKED_STATE", str(tmp_path / "revoked.json"))
    monkeypatch.setattr(auth, "_revoked_loaded", True)
    monkeypatch.setattr(auth, "_revoked_polled", set())
    monkeypatch.setattr(auth, "_min_agent_iat", None, raising=False)
    monkeypatch.setattr(auth, "_ERLAUBTE_AUD", [], raising=False)
    monkeypatch.setattr(auth, "_SUPERADMIN_OHNE_SCOPE", True)

    async def key():
        return "k"
    monkeypatch.setattr(auth, "_get_public_key", key)
    stand = {}
    monkeypatch.setattr(auth, "_decode_jwt", lambda token, key: dict(stand["claims"]))

    def ruf(claims, pfad="/cloud/"):
        stand["claims"] = claims

        async def go():
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=_app()), base_url="http://gw") as c:
                return await c.post(pfad, content=CALL, headers={
                    "Authorization": "Bearer t", "Content-Type": "application/json"})
        return asyncio.run(go())
    return ruf


SUPERADMIN_OPENID = {"sub": "u-alex", "is_superadmin": True, "scope": "openid"}


# --- aud ------------------------------------------------------------------------------

def test_fremder_dcr_client_mit_superadmin_wird_abgewiesen(gw, monkeypatch):
    """Befund 76: Superadmin meldet sich an einem fremden Client an, scope=openid."""
    monkeypatch.setattr(auth, "_ERLAUBTE_AUD", [CHATGPT], raising=False)
    r = gw({**SUPERADMIN_OPENID, "aud": FREMD})
    assert r.status_code == 401 and "audience not allowed" in r.text, r.text


def test_leere_liste_jede_aud_ist_fremd(gw):
    r = gw({**SUPERADMIN_OPENID, "aud": CHATGPT})
    assert r.status_code == 401, r.text


def test_zugelassener_connector_ohne_scope_wie_eigenes_login(gw, monkeypatch):
    """ChatGPT-Connectoren tragen keine Scopes (Cloud 84ce89dc): Superadmin-* bleibt."""
    monkeypatch.setattr(auth, "_ERLAUBTE_AUD", [CHATGPT], raising=False)
    r = gw({**SUPERADMIN_OPENID, "aud": CHATGPT})
    assert r.status_code == 200 and r.json()["perms"] == ["*"], r.text


def test_zugelassener_client_ohne_superadmin_nur_seine_scopes(gw, monkeypatch):
    monkeypatch.setattr(auth, "_ERLAUBTE_AUD", ["6eddccfb-*"], raising=False)
    c = {"sub": "u1", "aud": "6eddccfb-1111", "scope": "openid mcp:cloud mcp:content"}
    assert gw(c, "/cloud/").status_code == 200
    assert gw(c, "/business/").status_code == 403


def test_ohne_aud_wie_bisher(gw):
    assert gw({"sub": "u-alex", "is_superadmin": True}).json()["perms"] == ["*"]
    agent = {"type": "agent", "agent_name": "A", "permissions": ["mcp:cloud:*"], "iat": int(time.time())}
    assert gw(agent).status_code == 200


def test_aud_liste_als_liste_im_token(gw, monkeypatch):
    monkeypatch.setattr(auth, "_ERLAUBTE_AUD", [CHATGPT], raising=False)
    assert gw({**SUPERADMIN_OPENID, "aud": [CHATGPT, "x"]}).status_code == 200
    assert gw({**SUPERADMIN_OPENID, "aud": [FREMD]}).status_code == 401


# --- Voice-Grant und Token-Arten ------------------------------------------------------

def test_voice_grant_nie_auch_nicht_ueber_die_liste(gw, monkeypatch):
    monkeypatch.setattr(auth, "_ERLAUBTE_AUD", ["*"], raising=False)
    r = gw({"sub": "u-alex", "is_superadmin": True, "aud": "ai-realtime:profil-1"})
    assert r.status_code == 401 and "voice grant" in r.text, r.text


@pytest.mark.parametrize("typ", ["ephemeral_capability", "realtime_grant", "irgendwas"])
def test_fremde_token_art_wird_abgewiesen(gw, typ):
    r = gw({"type": typ, "sub": "agent:X", "agent_name": "X", "permissions": ["*"]})
    assert r.status_code == 401 and "token type" in r.text, r.text


@pytest.mark.parametrize("typ", ["user", "oauth_access"])
def test_kuenftige_menschen_arten_gehen(gw, typ):
    assert gw({"type": typ, "sub": "u-alex", "is_superadmin": True}).status_code == 200


# --- min_agent_iat ------------------------------------------------------------------

class _Antwort:
    def __init__(self, daten):
        self.status_code, self._d = 200, daten

    def json(self):
        return self._d


class _Client:
    def __init__(self, daten):
        self.daten = daten

    async def get(self, url, params=None):
        return _Antwort(self.daten)


def test_min_agent_iat_aus_dem_abruf(gw):
    jetzt = int(time.time())
    asyncio.run(auth.revoked_poll_once(_Client({"revoked_jtis": [], "as_of": "a1",
                                                 "min_agent_iat": jetzt - 60})))
    alt = {"type": "agent", "agent_name": "A", "permissions": ["mcp:cloud:*"], "iat": jetzt - 3600}
    neu = dict(alt, iat=jetzt)
    r = gw(alt)
    assert r.status_code == 401 and "min_agent_iat" in r.text, r.text
    assert gw(neu).status_code == 200
    # Menschen-Tokens betrifft es nicht.
    assert gw({"sub": "u-alex", "is_superadmin": True, "iat": jetzt - 3600}).status_code == 200
    # Fehlt der Schluessel (alte auth-api), bleibt der Stand; ein Absenken gilt.
    asyncio.run(auth.revoked_poll_once(_Client({"revoked_jtis": [], "as_of": "a2"})))
    assert gw(alt).status_code == 401
    asyncio.run(auth.revoked_poll_once(_Client({"revoked_jtis": [], "as_of": "a3", "min_agent_iat": None})))
    assert gw(alt).status_code == 200
    # Der Spiegel auf Platte traegt ihn mit.
    asyncio.run(auth.revoked_poll_once(_Client({"revoked_jtis": [], "as_of": "a4",
                                                 "min_agent_iat": "2026-11-15T00:00:00Z"})))
    assert json.loads(open(auth._REVOKED_STATE).read())["min_agent_iat"] == 1794700800


# --- Schluessel je kid ----------------------------------------------------------------

def _jwk(priv, kid):
    pub = priv.public_key().public_numbers()

    def b64(n):
        return base64.urlsafe_b64encode(n.to_bytes((n.bit_length() + 7) // 8, "big")).rstrip(b"=").decode()
    return {"kty": "RSA", "kid": kid, "n": b64(pub.n), "e": b64(pub.e), "alg": "RS256", "use": "sig"}


@pytest.fixture
def jwks(monkeypatch):
    k1 = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    k2 = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    stand = {"keys": [_jwk(k1, "auth-1")], "abrufe": 0}

    class _AsyncClient:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def get(self, url):
            stand["abrufe"] += 1
            return httpx.Response(200, json={"keys": list(stand["keys"])},
                                  request=httpx.Request("GET", url))
    monkeypatch.setattr(auth.httpx, "AsyncClient", _AsyncClient)
    monkeypatch.setattr(auth, "_public_key", None)
    monkeypatch.setattr(auth, "_public_key_fetched_at", 0.0)
    monkeypatch.setattr(auth, "_jwks_nach_kid", {}, raising=False)
    monkeypatch.setattr(auth, "_revoked_loaded", True)
    monkeypatch.setattr(auth, "_revoked_polled", set())
    return k1, k2, stand


def _tok(priv, kid=None, **extra):
    claims = {"iss": auth.AUTH_API_ISSUER, "sub": "agent:A", "exp": int(time.time()) + 600, **extra}
    return jwt.encode(claims, priv, algorithm="RS256", headers={"kid": kid} if kid else None)


def test_rotation_zweiter_schluessel_nach_kid(jwks):
    k1, k2, stand = jwks
    stand["keys"].append(_jwk(k2, "auth-2"))
    pk = asyncio.run(auth._get_public_key())
    assert auth._decode_jwt(_tok(k2, "auth-2"), pk)["sub"] == "agent:A"      # vorher: nur keys[0]
    assert auth._decode_jwt(_tok(k1, "auth-1"), pk)["sub"] == "agent:A"
    with pytest.raises(jwt.InvalidSignatureError):
        auth._decode_jwt(_tok(k2, "auth-1"), pk)                         # kid zeigt auf k1
    with pytest.raises(jwt.InvalidTokenError, match="without kid"):
        auth._decode_jwt(_tok(k1), pk)                                   # zwei Schluessel, kein kid


def test_unbekannter_kid_holt_einmal_nach_dann_abgewiesen(jwks, monkeypatch):
    k1, k2, stand = jwks
    pk = asyncio.run(auth._get_public_key())
    assert stand["abrufe"] == 1
    tok = _tok(k2, "auth-2")
    monkeypatch.setattr(auth, "_public_key_fetched_at", time.time() - 60)
    stand["keys"].append(_jwk(k2, "auth-2"))                              # Rotation bei auth-api
    asyncio.run(auth._jwks_fuer_kid(auth._kid_aus(tok)))
    assert stand["abrufe"] == 2
    assert auth._decode_jwt(tok, pk)["sub"] == "agent:A"
    # Unbekannt und gerade erst geholt: kein weiterer Abruf, abgewiesen.
    fremd = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    t3 = _tok(fremd, "auth-9")
    asyncio.run(auth._jwks_fuer_kid(auth._kid_aus(t3)))
    assert stand["abrufe"] == 2
    with pytest.raises(jwt.InvalidTokenError, match="unknown signing key"):
        auth._decode_jwt(t3, pk)


def test_ein_schluessel_ohne_kid_geht_weiter(jwks):
    k1, _k2, _stand = jwks
    pk = asyncio.run(auth._get_public_key())
    assert auth._decode_jwt(_tok(k1), pk)["sub"] == "agent:A"
