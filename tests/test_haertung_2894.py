"""#2894 (Cloud-Security 14.11.2026, Befund 23) und Nachtrag von Comm.

1. CORS war allow_origins=["*"] mit allow_credentials=True. Jetzt nur Herkuenfte
   aus MCP_CORS_ORIGINS, ohne Anmeldedaten; ohne Eintrag keine.
2. Eingebaute Ersatzschluessel (ONEAL_API_KEY, STORY_API_KEY) sind weg: ohne
   Schluessel scheitert ein Aufruf mit klarer Meldung.
3. PDF-Exporte senden den Ausweis des Aufrufers, den Dienstschluessel nur ohne ihn.
4. Die business-Werkzeuge haben keinen Parameter api_key mehr. Weil FastMCP
   unbekannte Argumente still verwirft, lehnt das Gateway einen Aufruf mit api_key ab.
5. outlook_get_attachment hat keinen Parameter save_to mehr.
"""
import asyncio
import inspect
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
import server  # noqa: E402


def _werkzeuge(mcp):
    return {t.name: t for t in asyncio.run(mcp.list_tools())}


# --- 4. business ohne api_key ---------------------------------------------------------

def test_kein_business_werkzeug_bietet_api_key_an():
    ws = _werkzeuge(server.business_mcp)
    assert len(ws) >= 40
    mit = [n for n, t in ws.items() if "api_key" in (t.inputSchema or {}).get("properties", {})]
    assert mit == []


def test_business_aufruf_mit_dem_schluessel_des_gateways(monkeypatch):
    gesehen = []

    async def _fetch(method, url, **kw):
        gesehen.append(kw.get("headers"))
        return {}
    monkeypatch.setattr(server, "_fetch_json", _fetch)
    monkeypatch.setattr(server, "BUSINESS_API_KEY", "GATEWAY")
    asyncio.run(server.call_business_api("GET", "/api/v1/clients"))
    assert gesehen == [{"X-API-Key": "GATEWAY"}]
    assert "api_key" not in inspect.signature(server.call_business_api).parameters


def _app():
    async def ok(request):
        return JSONResponse({"ok": 1})
    app = Starlette(routes=[Route("/{srv}/", ok, methods=["POST"])])
    app.add_middleware(auth.JWTAuthMiddleware)
    return app


@pytest.fixture
def gw(monkeypatch):
    monkeypatch.setattr(auth, "AUTH_REQUIRE_JWT", True)

    async def key():
        return "k"
    monkeypatch.setattr(auth, "_get_public_key", key)
    monkeypatch.setattr(auth, "_decode_jwt", lambda token, key: {
        "type": "agent", "agent_name": "Business", "permissions": ["mcp:business:*", "mcp:cloud:*"]})
    monkeypatch.setattr(auth, "_aufrufe_vermerken", lambda *a: None)

    def ruf(srv, args):
        body = json.dumps({"jsonrpc": "2.0", "id": 9, "method": "tools/call",
                           "params": {"name": "clients_list", "arguments": args}})

        async def go():
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=_app()), base_url="http://gw") as c:
                return await c.post(f"/{srv}/", content=body,
                                    headers={"Content-Type": "application/json", "Authorization": "Bearer t"})
        return asyncio.run(go())
    return ruf


def test_api_key_im_aufruf_wird_abgelehnt_statt_still_verworfen(gw):
    r = gw("business", {"api_key": "biz_fremd", "status": "active"})
    assert r.status_code == 400
    d = r.json()
    assert d["id"] == 9 and d["error"]["code"] == -32602 and "api_key" in d["error"]["message"]
    assert "biz_fremd" not in r.text
    assert gw("business", {"status": "active"}).status_code == 200
    assert gw("cloud", {"api_key": "egal"}).status_code == 200          # nur business hat die Regel


# --- 2. keine eingebauten Ersatzschluessel --------------------------------------------

def test_keine_eingebauten_ersatzschluessel():
    quelle = inspect.getsource(server)
    assert "oneal_demo_token" not in quelle and "story_ark_secret_2025" not in quelle


@pytest.mark.parametrize("ruf,konst", [(lambda: server.call_oneal_api("GET", "/x"), "ONEAL_API_KEY"),
                                       (lambda: server.call_story_api("GET", "/x"), "STORY_API_KEY")])
def test_ohne_schluessel_klare_meldung_und_kein_aufruf(monkeypatch, ruf, konst):
    aufrufe = []

    async def _fetch(*a, **k):
        aufrufe.append(1)
    monkeypatch.setattr(server, "_fetch_json", _fetch)
    monkeypatch.setattr(server, konst, "")
    with pytest.raises(RuntimeError, match=konst):
        asyncio.run(ruf())
    assert aufrufe == []


# --- 3. PDF-Export mit dem Ausweis des Aufrufers ---------------------------------------

@pytest.mark.parametrize("werkzeug", ["content_posts_export_pdf", "content_posts_export_themed_pdf"])
def test_pdf_export_mit_aufrufer_jwt(monkeypatch, werkzeug):
    gesehen = []

    class C:
        def __init__(self, **kw): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def get(self, url, headers=None, params=None):
            gesehen.append(dict(headers or {}))
            return httpx.Response(200, content=b"%PDF", headers={"content-type": "application/pdf"},
                                  request=httpx.Request("GET", url))
    monkeypatch.setattr(server.httpx, "AsyncClient", C)
    monkeypatch.setattr(server, "CONTENT_API_KEY", "DIENST")
    fn = getattr(server, werkzeug)
    t = (auth._current_jwt.set("JWT-DES-AUFRUFERS"), auth._current_agent_name.set("A"),
         auth._current_is_agent.set(True))
    try:
        asyncio.run(fn(post_id=5))
    except Exception:
        pass                                   # nur die Kopfzeilen zaehlen
    finally:
        auth._current_jwt.reset(t[0]); auth._current_agent_name.reset(t[1]); auth._current_is_agent.reset(t[2])
    assert gesehen and gesehen[0].get("Authorization") == "Bearer JWT-DES-AUFRUFERS"
    assert "X-API-KEY" not in gesehen[0]


# --- 1. CORS --------------------------------------------------------------------------

def test_cors_ohne_eintrag_keine_fremde_herkunft():
    from starlette.middleware.cors import CORSMiddleware
    [cors] = [m for m in server.app.user_middleware if m.cls is CORSMiddleware]
    opts = cors.kwargs if hasattr(cors, "kwargs") else cors.options
    assert opts["allow_credentials"] is False
    assert "*" not in opts["allow_origins"]
    assert opts["allow_origins"] == [o for o in server._CORS_ORIGINS]


# --- 5. outlook ohne save_to ------------------------------------------------------------

def test_outlook_anhang_ohne_save_to():
    assert "save_to" not in inspect.signature(server.comm_outlook_get_attachment).parameters
    t = _werkzeuge(server.comm_mcp)["outlook_get_attachment"]
    assert "save_to" not in (t.inputSchema or {}).get("properties", {})
