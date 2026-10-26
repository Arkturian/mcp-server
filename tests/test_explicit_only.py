"""Durchreichen nur mit ausdruecklichem Recht (Entscheid Alex 26.10.2026).

An einem Server in auth.EXPLICIT_ONLY_SERVERS zaehlen "*", "mcp:*" und
"mcp:*:*" nicht; ein ausdrueckliches mcp:<server>:<tool> gibt nur dieses
Werkzeug frei. Andere Server pruefen unveraendert.
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
                   "params": {"name": "projektkosten", "arguments": {}}})


def _app():
    async def ok(request):
        return JSONResponse({"ok": 1})
    app = Starlette(routes=[Route("/{srv}/", ok, methods=["POST"])])
    app.add_middleware(auth.JWTAuthMiddleware)
    return app


@pytest.fixture
def mit_rechten(monkeypatch):
    monkeypatch.setattr(auth, "EXPLICIT_ONLY_SERVERS", {"p3-buchhaltung"})

    async def key():
        return "k"
    monkeypatch.setattr(auth, "_get_public_key", key)

    def setze(perms):
        monkeypatch.setattr(auth, "_decode_jwt", lambda token, key: {
            "type": "agent", "agent_name": "X", "role": "r", "permissions": perms})
    return setze


def _post(path, body=CALL):
    async def go():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=_app()),
                                     base_url="http://gw") as c:
            return await c.post(path, content=body, headers={
                "Authorization": "Bearer t", "Content-Type": "application/json"})
    return asyncio.run(go()).status_code


@pytest.mark.parametrize("perms", [["*"], ["mcp:*"], ["mcp:*:*"], ["mcp:*:projektkosten"],
                                   ["mcp:storage:*"], ["mcp:p3-buchhaltung"], []])
def test_platzhalter_zaehlen_nicht(mit_rechten, perms):
    mit_rechten(perms)
    assert _post("/p3-buchhaltung/") == 403


@pytest.mark.parametrize("perms", [["mcp:p3-buchhaltung:*"], ["*", "mcp:p3-buchhaltung:*"],
                                   ["mcp:p3-buchhaltung:projektkosten"]])
def test_ausdrueckliches_recht_kommt_durch(mit_rechten, perms):
    mit_rechten(perms)
    assert _post("/p3-buchhaltung/") == 200


def test_werkzeugrecht_gibt_nur_dieses_werkzeug_frei(mit_rechten):
    mit_rechten(["*", "mcp:p3-buchhaltung:anderes"])
    assert _post("/p3-buchhaltung/") == 403


@pytest.mark.parametrize("perms", [["*"], ["mcp:*"], ["mcp:*:*"]])
def test_andere_server_unveraendert(mit_rechten, perms):
    mit_rechten(perms)
    assert _post("/storage/") == 200


def test_ohne_tools_call_keine_pruefung(mit_rechten):
    mit_rechten([])
    body = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
    assert _post("/p3-buchhaltung/", body) == 200
