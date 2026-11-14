"""#2893 (Cloud-Security 14.11.2026, Befund 22): nur tools/call war angemeldet.

GET, andere HTTP-Verben, Rumpf ohne JSON und tools/list, resources/read,
prompts/get liefen ohne JWT durch -- auch an EXPLICIT_ONLY-Servern und durch die
Durchreiche. GET / nannte interne Upstream-Adressen, /health probte bei jedem
anonymen Abruf mit Betreiber-Schluesseln, /docs und /openapi.json waren offen.

Jetzt: ohne Token nur der Handshake (initialize, notifications/initialized, ping),
OPTIONS und die oeffentlichen Pfade; alles andere braucht ein gueltiges JWT mit
Recht am Server. Ein GET ohne Token bekommt 405 wie von einem zustandslosen
MCP-Server, sonst 403 (kein 401: das wuerde in Claude Code OAuth ausloesen).
"""
import asyncio
import json
import os

import httpx
import pytest
from fastapi import HTTPException
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route

for _k in ("ARKTURIAN_API_KEY", "ONEAL_STORAGE_API_KEY", "JWT_PUBLIC_KEY", "COMM_API_KEY"):
    os.environ.setdefault(_k, "test-only")

import auth  # noqa: E402
import server  # noqa: E402


def _rpc(method, **extra):
    return json.dumps({"jsonrpc": "2.0", "id": 1, "method": method, "params": {}, **extra})


def _app():
    async def ok(request):
        return JSONResponse({"ok": 1})
    app = Starlette(routes=[Route("/{srv}/", ok, methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"]),
                            Route("/.well-known/{x}", ok, methods=["GET"])])
    app.add_middleware(auth.JWTAuthMiddleware)
    return app


@pytest.fixture
def gw(monkeypatch):
    monkeypatch.setattr(auth, "AUTH_REQUIRE_JWT", True)

    async def key():
        return "k"
    monkeypatch.setattr(auth, "_get_public_key", key)
    stand = {"claims": {"type": "agent", "agent_name": "A", "permissions": ["mcp:cloud:*"]}}
    monkeypatch.setattr(auth, "_decode_jwt", lambda token, key: dict(stand["claims"]))

    def ruf(verb, pfad, body=None, token=None, typ="application/json"):
        kopf = {"Content-Type": typ}
        if token:
            kopf["Authorization"] = f"Bearer {token}"

        async def go():
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=_app()), base_url="http://gw") as c:
                return await c.request(verb, pfad, content=body, headers=kopf)
        return asyncio.run(go()).status_code
    ruf.stand = stand
    return ruf


@pytest.mark.parametrize("methode", ["tools/list", "resources/list", "resources/read", "prompts/list",
                                     "prompts/get", "completion/complete", "unbekannt"])
def test_methoden_ausser_handshake_brauchen_token(gw, methode):
    assert gw("POST", "/cloud/", _rpc(methode)) == 403
    assert gw("POST", "/cloud/", _rpc(methode), token="t") == 200


@pytest.mark.parametrize("methode", ["initialize", "notifications/initialized", "ping"])
def test_handshake_ohne_token(gw, methode):
    assert gw("POST", "/cloud/", _rpc(methode)) == 200


def test_stapel_mit_mehr_als_handshake_braucht_token(gw):
    stapel = json.dumps([{"jsonrpc": "2.0", "id": 1, "method": "initialize"},
                         {"jsonrpc": "2.0", "id": 2, "method": "resources/read"}])
    assert gw("POST", "/cloud/", stapel) == 403


@pytest.mark.parametrize("verb,code", [("GET", 405), ("PUT", 403), ("DELETE", 403)])
def test_andere_verben_ohne_token(gw, verb, code):
    assert gw(verb, "/cloud/") == code
    assert gw(verb, "/cloud/", token="t") == 200


def test_rumpf_ohne_json_braucht_token(gw):
    assert gw("POST", "/cloud/", b"kein json", typ="text/plain") == 403


def test_recht_am_server_gilt_auch_fuer_tools_list(gw):
    gw.stand["claims"] = {"type": "agent", "agent_name": "A", "permissions": ["mcp:content:*"]}
    assert gw("POST", "/cloud/", _rpc("tools/list"), token="t") == 403
    assert gw("GET", "/cloud/", token="t") == 403


def test_oeffentliche_pfade_und_preflight(gw):
    assert gw("GET", "/.well-known/oauth-protected-resource") == 200
    assert gw("OPTIONS", "/cloud/") == 200


def test_tools_list_wird_nicht_als_aufruf_gezaehlt(gw, monkeypatch):
    gezaehlt = []
    monkeypatch.setattr(auth, "_aufrufe_vermerken", lambda *a: gezaehlt.append(a))
    gw("POST", "/cloud/", _rpc("tools/list"), token="t")
    assert gezaehlt == []
    gw("POST", "/cloud/", json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                                      "params": {"name": "x", "arguments": {}}}), token="t")
    assert len(gezaehlt) == 1


# --- Oeffentliche Routen des Gateways -------------------------------------------

def test_root_nennt_keine_internen_adressen():
    out = asyncio.run(server.root_route())
    assert set(out) == {"name", "version", "servers"}
    assert "http" not in json.dumps(out)


def test_health_ohne_fehlertext_und_zwischengespeichert(monkeypatch):
    rufe = []

    async def details():
        rufe.append(1)
        raise HTTPException(status_code=207, detail={
            "status": "degraded", "content": {"ok": 1},
            "storage_arkturian_error": "Upstream 401 from GET http://intern:8001/x: Invalid API key"})
    monkeypatch.setattr(server, "health", details)
    monkeypatch.setattr(server, "_HEALTH_CACHE", {"bis": 0.0, "wert": None})
    r1 = asyncio.run(server.health_route())
    r2 = asyncio.run(server.health_route())
    assert r1.status_code == 207 and rufe == [1]                     # zweiter Abruf aus dem Speicher
    body = json.loads(r1.body)
    assert body == {"status": "degraded", "checks": {"content": "ok", "storage_arkturian": "fehler"}}
    assert "intern" not in r1.body.decode() and "API key" not in r2.body.decode()


def test_docs_sind_aus():
    assert server.app.openapi_url is None and server.app.docs_url is None and server.app.redoc_url is None
    assert "/docs" not in auth.PUBLIC_PATHS and "/openapi.json" not in auth.PUBLIC_PATHS
