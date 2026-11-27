"""Offene Server (Alex 08.11.2026, #2679): jedes gueltige Agenten-JWT genuegt.

MCP_OFFENE_SERVER je Gateway; Vorgabe leer. Ohne Token bleibt es zu, und ein
Server, der zugleich nur mit ausdruecklichem Recht geht, bleibt zu.
"""
import asyncio
from rollensperre import GESPERRT, code  # noqa: E402  (#4171)
import json

import httpx
import pytest
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route

import auth
from pruef_helfer import claims_fest  # noqa: E402

CALL = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                   "params": {"name": "forecast_when", "arguments": {}}})


def _app():
    async def ok(request):
        return JSONResponse({"ok": 1})
    app = Starlette(routes=[Route("/{srv}/", ok, methods=["POST"])])
    app.add_middleware(auth.JWTAuthMiddleware)
    return app


@pytest.fixture
def agent(monkeypatch):
    monkeypatch.setattr(auth, "OFFENE_SERVER", {"llmtime"})
    monkeypatch.setattr(auth, "EXPLICIT_ONLY_SERVERS", set())

    async def key():
        return "k"
    monkeypatch.setattr(auth, "_pruefen", claims_fest(lambda token: {
        "type": "agent", "agent_name": "X", "role": "developer", "permissions": ["mcp:issue:*"]}))
    return monkeypatch


def _post(path, token=True):
    async def go():
        h = {"Content-Type": "application/json"}
        if token:
            h["Authorization"] = "Bearer t"
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=_app()), base_url="http://gw") as c:
            return await c.post(path, content=CALL, headers=h)
    return code(asyncio.run(go()))


def test_offener_server_ohne_eigenes_recht(agent):
    assert _post("/llmtime/") == 200


def test_anderer_server_weiter_mit_recht(agent):
    assert _post("/storage/") == 403


def test_explicit_only_gewinnt(agent):
    agent.setattr(auth, "EXPLICIT_ONLY_SERVERS", {"llmtime"})
    assert _post("/llmtime/") == 403


def test_vorgabe_ist_leer():
    import importlib, os
    alt = os.environ.pop("MCP_OFFENE_SERVER", None)
    try:
        assert importlib.reload(auth).OFFENE_SERVER == set()
    finally:
        if alt is not None:
            os.environ["MCP_OFFENE_SERVER"] = alt
        importlib.reload(auth)



# #3857: Fuer Projektleiter (Rolle projektleitung) gelten offene Server nicht, nur ihre eigenen Rechte.

def _projektleiter_aufruf(monkeypatch, werkzeug, server="projekt", rolle="projektleitung"):
    monkeypatch.setattr(auth, "OFFENE_SERVER", {"llmtime", "projekt"})
    monkeypatch.setattr(auth, "EXPLICIT_ONLY_SERVERS", set())
    rechte = ["mcp:projekt:projekt_stand", "mcp:projekt:projekt_entscheidungen", "mcp:projekt:deliverable",
              "mcp:projekt:projekt_entscheidung", "mcp:projekt:projektleitung_absichten"]
    monkeypatch.setattr(auth, "_pruefen", claims_fest(lambda token: {
        "type": "agent", "agent_name": "Projektleitung-v10", "role": rolle,
        "permissions": rechte if rolle == "projektleitung" else ["mcp:issue:*"]}))
    aufruf = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                         "params": {"name": werkzeug, "arguments": {"vorgabe_id": 0}}})

    async def go():
        h = {"Content-Type": "application/json", "Authorization": "Bearer t"}
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=_app()), base_url="http://gw") as c:
            return await c.post(f"/{server}/", content=aufruf, headers=h)
    return code(asyncio.run(go()))


def test_projektleiter_ruft_nur_seine_werkzeuge(monkeypatch):
    assert _projektleiter_aufruf(monkeypatch, "projekt_stand") == 200
    assert _projektleiter_aufruf(monkeypatch, "projektleitung_absichten") == 200
    for werkzeug in ("plan_absegnen", "tor_oeffnen", "vorgabe_loeschen", "deliverable_abnehmen"):
        assert _projektleiter_aufruf(monkeypatch, werkzeug) == GESPERRT, werkzeug


def test_projektleiter_hat_keinen_offenen_llmtime(monkeypatch):
    assert _projektleiter_aufruf(monkeypatch, "forecast_when", server="llmtime") == 403


def test_andere_agenten_bleiben_offen(monkeypatch):
    assert _projektleiter_aufruf(monkeypatch, "plan_absegnen", rolle="developer") == 200


def test_gegriffenes_recht_nennt_fuer_projektleiter_das_eigene_recht(monkeypatch):
    monkeypatch.setattr(auth, "OFFENE_SERVER", {"projekt"})
    monkeypatch.setattr(auth, "EXPLICIT_ONLY_SERVERS", set())
    assert auth.gegriffenes_recht(["mcp:projekt:projekt_stand"], "projekt", "projekt_stand") == "offen"
    assert auth.gegriffenes_recht(["mcp:projekt:projekt_stand"], "projekt", "projekt_stand",
                                  "projektleitung") == "mcp:projekt:projekt_stand"
