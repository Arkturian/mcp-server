"""Offene Server (Alex 08.11.2026, #2679): jedes gueltige Agenten-JWT genuegt.

MCP_OFFENE_SERVER je Gateway; Vorgabe leer. Ohne Token bleibt es zu, und ein
Server, der zugleich nur mit ausdruecklichem Recht geht, bleibt zu.
"""
import asyncio
import json

import httpx
import pytest
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route

import auth

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
    monkeypatch.setattr(auth, "_get_public_key", key)
    monkeypatch.setattr(auth, "_decode_jwt", lambda token, key: {
        "type": "agent", "agent_name": "X", "role": "developer", "permissions": ["mcp:issue:*"]})
    return monkeypatch


def _post(path, token=True):
    async def go():
        h = {"Content-Type": "application/json"}
        if token:
            h["Authorization"] = "Bearer t"
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=_app()), base_url="http://gw") as c:
            return await c.post(path, content=CALL, headers=h)
    return asyncio.run(go()).status_code


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
