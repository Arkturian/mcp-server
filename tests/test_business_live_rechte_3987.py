"""#3987: Fuer `business` zaehlen beim Werkzeugaufruf zusaetzlich die AKTUELLEN Rechte aus auth-api.

auth-api hat mcp:business:* aus 11 Grundrollen genommen. Agenten-Tokens tragen die Rechte vom
Praegen und gelten 8760 h, Claude Code holt sie nur beim Verbinden. Ohne Live-Abgleich riefe ein
laufender Agent documents_mark_paid bis zu seinem naechsten Neustart weiter auf.
Echte JWTAuthMiddleware mit festen Claims; auth-api ist eine Attrappe (_live_rechte_abruf bzw.
httpx.MockTransport). Keine business-api.
"""
import asyncio
import json
import os

import httpx
import pytest
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route

for _k in ("ARKTURIAN_API_KEY", "ONEAL_STORAGE_API_KEY", "JWT_PUBLIC_KEY", "COMM_API_KEY"):
    os.environ.setdefault(_k, "test-only")

import auth  # noqa: E402
from pruef_helfer import claims_fest  # noqa: E402

# Der echte Abruf, bevor conftest ihn fuer jeden Test sperrt (_live_rechte_ohne_netz).
_ECHTER_ABRUF = auth._live_rechte_abruf


def _app():
    async def ok(request):
        return JSONResponse({"ok": 1})
    app = Starlette(routes=[Route("/{srv}/", ok, methods=["POST"])])
    app.add_middleware(auth.JWTAuthMiddleware)
    return app


@pytest.fixture
def gw(monkeypatch):
    """Setzt Token-Rechte und Live-Rechte; zaehlt die Abrufe bei auth-api."""
    monkeypatch.setattr(auth, "EXPLICIT_ONLY_SERVERS", set())
    monkeypatch.setattr(auth, "OFFENE_SERVER", set())
    monkeypatch.setattr(auth, "LIVE_RECHTE_SERVER", frozenset({"business"}))
    monkeypatch.setattr(auth, "_LIVE_RECHTE", {})
    zustand = {"live": [], "abrufe": [], "fehler": None, "typ": "agent", "token": []}

    def setze(token_rechte, live=None, typ="agent"):
        zustand["token"] = token_rechte
        zustand["typ"] = typ
        if live is not None:
            zustand["live"] = live
        monkeypatch.setattr(auth, "_pruefen", claims_fest(lambda t: {
            "type": zustand["typ"], "agent_name": "Probe-3987", "sub": "agent:Probe-3987",
            "role": "developer", "permissions": zustand["token"]}))

    async def abruf(name, token):
        zustand["abrufe"].append((name, token))
        if zustand["fehler"]:
            raise auth.LiveRechteUnbekannt(zustand["fehler"])
        return list(zustand["live"])
    monkeypatch.setattr(auth, "_live_rechte_abruf", abruf)
    zustand["setze"] = setze
    return zustand


def _post(srv, body, token="t-probe"):
    async def go():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=_app()), base_url="http://gw") as c:
            return await c.post(f"/{srv}/", content=json.dumps(body),
                                headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"})
    return asyncio.run(go())


def _call(*werkzeuge, srv="business"):
    teile = [{"jsonrpc": "2.0", "id": i, "method": "tools/call", "params": {"name": w, "arguments": {}}}
             for i, w in enumerate(werkzeuge)]
    return _post(srv, teile[0] if len(teile) == 1 else teile)


# ---------------------------------------------------------------------------
# Die Gegenprobe aus dem Issue

def test_altes_token_neues_recht_entzogen_ergibt_403(gw):
    """Token aus der Rolle (mcp:business:*), in auth-api ist das Recht weg."""
    gw["setze"](["mcp:business:*"], live=["mcp:issue:*"])
    r = _call("documents_mark_paid")
    assert r.status_code == 403
    assert "documents_mark_paid" in r.json()["error"]["message"]
    assert _call("documents_list").status_code == 403


def test_business_mit_recht_arbeitet_weiter(gw):
    gw["setze"](["mcp:business:*"], live=["mcp:business:*"])
    assert _call("documents_mark_paid").status_code == 200
    assert _call("transactions_import").status_code == 200


def test_nur_lesend_vergeben(gw):
    """Cloud/Assistant: einzelne Lesewerkzeuge in auth-api, altes Token mit mcp:business:*."""
    gw["setze"](["mcp:business:*"], live=["mcp:business:documents_list", "mcp:business:clients_list"])
    assert _call("documents_list").status_code == 200
    assert _call("clients_list").status_code == 200
    assert _call("documents_mark_paid").status_code == 403
    assert _call("paypal_sync").status_code == 403


def test_platzhalter_in_auth_api_gilt(gw):
    for live in (["*"], ["mcp:*:*"], ["mcp:*"], ["mcp:business:*"]):
        gw["setze"](["mcp:business:*"], live=live)
        auth._LIVE_RECHTE.clear()
        assert _call("documents_mark_paid").status_code == 200, live


def test_nur_einengend_live_oeffnet_nichts(gw):
    """auth-api darf nichts oeffnen, was das Token nicht hat."""
    gw["setze"](["mcp:business:documents_list"], live=["mcp:business:*"])
    assert _call("documents_list").status_code == 200
    assert _call("documents_mark_paid").status_code == 403


def test_stapel_mit_einem_verbotenen_wird_abgewiesen(gw):
    gw["setze"](["mcp:business:*"], live=["mcp:business:documents_list"])
    assert _call("documents_list", "documents_mark_paid").status_code == 403


def test_agent_in_auth_api_geloescht(gw):
    gw["setze"](["mcp:business:*"], live=[])
    assert _call("documents_list").status_code == 403


# ---------------------------------------------------------------------------
# Was NICHT live geprueft wird

def test_tools_list_bleibt_beim_token(gw):
    """Ein 403 auf tools/list hiesse "MCP server failed" und weckte den MCP-Waechter."""
    gw["setze"](["mcp:business:*"], live=[])
    r = _post("business", {"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
    assert r.status_code == 200
    assert gw["abrufe"] == []


def test_anderer_server_ohne_abruf(gw):
    gw["setze"](["mcp:issue:*"], live=[])
    assert _call("issue_list", srv="issue").status_code == 200
    assert gw["abrufe"] == []


def test_dienst_nicht_live_geprueft(gw):
    gw["setze"](["mcp:business:*"], live=[], typ="service")
    assert _call("documents_list").status_code == 200
    assert gw["abrufe"] == []


def test_mensch_nicht_live_geprueft(gw):
    """Menschen-Tokens leiten ihre Rechte anders ab (_rechte_aus_claims); ob sie durchkommen,
    entscheidet das bisherige Tor. Der Live-Abgleich gilt nur fuer type=agent."""
    gw["setze"](["mcp:business:*"], live=[], typ="user")
    _call("documents_list")
    assert gw["abrufe"] == []


def test_ohne_recht_im_token_kein_abruf(gw):
    gw["setze"](["mcp:issue:*"], live=["mcp:business:*"])
    assert _call("documents_mark_paid").status_code == 403
    assert gw["abrufe"] == []


def test_abruf_mit_dem_token_des_aufrufers(gw):
    gw["setze"](["mcp:business:*"], live=["mcp:business:*"])
    _call("documents_list")
    assert gw["abrufe"] == [("Probe-3987", "t-probe")]


# ---------------------------------------------------------------------------
# Zwischenspeicher und Ausfall von auth-api

def test_zwischenspeicher_je_agent(gw, monkeypatch):
    uhr = [1000.0]
    monkeypatch.setattr(auth, "LIVE_RECHTE_TTL_S", 60.0)
    import time
    monkeypatch.setattr(time, "monotonic", lambda: uhr[0])
    gw["setze"](["mcp:business:*"], live=["mcp:business:*"])
    _call("documents_list")
    _call("documents_list")
    assert len(gw["abrufe"]) == 1
    gw["live"] = []                       # Recht entzogen
    uhr[0] += 61
    assert _call("documents_list").status_code == 403
    assert len(gw["abrufe"]) == 2


def test_ausfall_ohne_stand_weist_ab(gw):
    gw["setze"](["mcp:business:*"], live=["mcp:business:*"])
    gw["fehler"] = "ConnectError"
    r = _call("documents_mark_paid")
    assert r.status_code == 503
    assert "nicht pruefbar" in r.json()["error"]["message"]


def test_unerwarteter_fehler_weist_ab(gw, monkeypatch):
    """Fail-closed: wirft die Pruefung etwas Unerwartetes, geht der Aufruf nicht durch."""
    gw["setze"](["mcp:business:*"], live=["mcp:business:*"])

    async def kaputt(name, token):
        raise RuntimeError("unerwartet")
    monkeypatch.setattr(auth, "_live_rechte_abruf", kaputt)
    assert _call("documents_list").status_code == 403


def test_ausfall_mit_stand_innerhalb_notstand(gw, monkeypatch):
    uhr = [1000.0]
    import time
    monkeypatch.setattr(time, "monotonic", lambda: uhr[0])
    monkeypatch.setattr(auth, "LIVE_RECHTE_TTL_S", 60.0)
    monkeypatch.setattr(auth, "LIVE_RECHTE_NOTSTAND_S", 900.0)
    gw["setze"](["mcp:business:*"], live=["mcp:business:documents_list"])
    assert _call("documents_list").status_code == 200
    gw["fehler"] = "HTTP 502"
    uhr[0] += 300                          # TTL vorbei, Notstand nicht
    assert _call("documents_list").status_code == 200
    assert _call("documents_mark_paid").status_code == 403     # der alte Stand engt weiter ein
    uhr[0] += 700                          # Notstand vorbei
    assert _call("documents_list").status_code == 503


# ---------------------------------------------------------------------------
# Der Abruf selbst

def _mock_client(monkeypatch, handler):
    echt = httpx.AsyncClient

    def fabrik(*a, **k):
        k["transport"] = httpx.MockTransport(handler)
        return echt(*a, **k)
    monkeypatch.setattr(httpx, "AsyncClient", fabrik)


def test_abruf_liest_effective_permissions(monkeypatch):
    monkeypatch.setattr(auth, "MCP_JWT_BASE_URL", "http://auth.invalid:8050")
    gesehen = {}

    def handler(req):
        gesehen["url"] = str(req.url)
        gesehen["auth"] = req.headers.get("authorization")
        return httpx.Response(200, json={"agent": "A b", "extra_permissions": ["x"],
                                         "effective_permissions": ["mcp:business:documents_list", "", 7]})
    _mock_client(monkeypatch, handler)
    rechte = asyncio.run(_ECHTER_ABRUF("A b/../c", "tok"))
    assert rechte == ["mcp:business:documents_list"]
    assert gesehen["url"] == "http://auth.invalid:8050/api/v1/auth/rbac/agents/A%20b%2F..%2Fc/permissions"
    assert gesehen["auth"] == "Bearer tok"


def test_abruf_404_heisst_keine_rechte(monkeypatch):
    monkeypatch.setattr(auth, "MCP_JWT_BASE_URL", "http://auth.invalid:8050")
    _mock_client(monkeypatch, lambda req: httpx.Response(404, json={"detail": "agent_not_found:A"}))
    assert asyncio.run(_ECHTER_ABRUF("A", "tok")) == []


@pytest.mark.parametrize("antwort", [
    httpx.Response(500, text="kaputt"), httpx.Response(401, json={}), httpx.Response(403, json={}),
    httpx.Response(200, text="kein json"), httpx.Response(200, json={"extra_permissions": []}),
])
def test_abruf_ohne_verwertbaren_stand(monkeypatch, antwort):
    monkeypatch.setattr(auth, "MCP_JWT_BASE_URL", "http://auth.invalid:8050")
    _mock_client(monkeypatch, lambda req: antwort)
    with pytest.raises(auth.LiveRechteUnbekannt):
        asyncio.run(_ECHTER_ABRUF("A", "tok"))


def test_basis_ohne_mcp_jwt_base_url(monkeypatch):
    monkeypatch.setattr(auth, "MCP_JWT_BASE_URL", "")
    monkeypatch.setattr(auth, "AUTH_API_JWKS_URL", "https://auth.invalid/api/v1/auth/.well-known/jwks.json")
    assert auth._auth_api_basis() == "https://auth.invalid"


def test_vorgabe_nur_business(monkeypatch):
    monkeypatch.delenv("MCP_LIVE_RECHTE_SERVER", raising=False)
    assert auth._live_server_vorgabe() == frozenset({"business"})
    monkeypatch.setenv("MCP_LIVE_RECHTE_SERVER", " Business , tree ,")
    assert auth._live_server_vorgabe() == frozenset({"business", "tree"})
    monkeypatch.setenv("MCP_LIVE_RECHTE_SERVER", "")
    assert auth._live_server_vorgabe() == frozenset()


def test_business_ist_hauptschluessel_server():
    """Am Gateway zaehlen fuer business nur mcp:business:-Rechte (EXPLICIT_ONLY_SERVERS):
    `*` und `mcp:*:*` oeffnen es nicht, weder im Token noch in auth-api."""
    assert "business" in auth.EXPLICIT_ONLY_SERVERS
    assert auth._live_verweigert(["*", "mcp:*:*"], "business", ["documents_mark_paid"]) == "documents_mark_paid"
