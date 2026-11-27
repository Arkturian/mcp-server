"""#3130: Tree-Stufen im Gateway gestaffelt wie in tree-api (write ⊇ status ⊇ read).

Befund von Tree nach dem #3121-Release (mcp-server 18a0bd9): tree-api app/auth.py
tree_scopes staffelt die Stufen, das Gateway nahm sie woertlich. Nur mcp:tree:status
sperrte alle 9 Lese-Werkzeuge, nur mcp:tree:write dazu nodes_update -- ueber REST
waere beides erlaubt. Gegen die echte JWTAuthMiddleware mit gefaelschtem JWT, keine
tree-api. person bleibt woertlich (person-api staffelt nicht).
"""
import asyncio
from rollensperre import GESPERRT, code  # noqa: E402  (#4171)
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
import server  # noqa: E402
from person_tools import WERKZEUG_STUFEN as PERSON_STUFEN  # noqa: E402

LESEN = {w for w, s in server.TREE_WERKZEUG_STUFEN.items() if s == "read"}
STATUS = {w for w, s in server.TREE_WERKZEUG_STUFEN.items() if s == "status"}
SCHREIBEN = {w for w, s in server.TREE_WERKZEUG_STUFEN.items() if s == "write"}


@pytest.fixture
def rechte(monkeypatch):
    monkeypatch.setattr(auth, "EXPLICIT_ONLY_SERVERS", set())
    monkeypatch.setattr(auth, "OFFENE_SERVER", set())
    monkeypatch.setattr(auth, "WERKZEUG_STUFEN", {"tree": dict(server.TREE_WERKZEUG_STUFEN),
                                                  "person": dict(PERSON_STUFEN)})
    # raising=False/getattr: die Gegenprobe auf dem alten Stand misst so das Verhalten,
    # nicht nur das Fehlen der neuen Naht.
    monkeypatch.setattr(auth, "STUFEN_REIHE",
                        {"tree": getattr(server, "TREE_STUFEN_REIHE", ("read", "status", "write"))},
                        raising=False)

    async def key():
        return "k"

    def setze(perms):
        monkeypatch.setattr(auth, "_pruefen", claims_fest(lambda token: {
            "type": "agent", "agent_name": "Projektmanager", "role": "pm", "permissions": perms}))
    return setze


def _app():
    async def ok(request):
        return JSONResponse({"ok": 1})
    app = Starlette(routes=[Route("/{srv}/", ok, methods=["POST"])])
    app.add_middleware(auth.JWTAuthMiddleware)
    return app


def _post(srv, werkzeug):
    body = {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": werkzeug, "arguments": {}}}

    async def go():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=_app()), base_url="http://gw") as c:
            return await c.post(f"/{srv}/", content=json.dumps(body),
                                headers={"Content-Type": "application/json", "Authorization": "Bearer t"})
    return asyncio.run(go())


def _erlaubt(srv, werkzeuge):
    return {w for w in werkzeuge if code(_post(srv, w)) == 200}


def test_reihe_wie_tree_api():
    assert server.TREE_STUFEN_REIHE == ("read", "status", "write")
    assert len(LESEN) == 9 and STATUS == {"nodes_update"}


def test_status_deckt_die_lese_werkzeuge(rechte):
    rechte(["mcp:tree:status"])
    assert _erlaubt("tree", LESEN | STATUS | SCHREIBEN) == LESEN | STATUS
    assert {"tree_get", "tree_outline"} <= LESEN


def test_write_deckt_alles(rechte):
    rechte(["mcp:tree:write"])
    assert _erlaubt("tree", LESEN | STATUS | SCHREIBEN) == LESEN | STATUS | SCHREIBEN


def test_read_deckt_weder_status_noch_write(rechte):
    rechte(["mcp:tree:read"])
    assert _erlaubt("tree", LESEN | STATUS | SCHREIBEN) == LESEN
    r = _post("tree", "nodes_update")
    assert code(r) == GESPERRT                                     # #4171
    msg = r.json()["result"]["content"][0]["text"]
    assert "mcp:tree:nodes_update | mcp:tree:status | mcp:tree:write" in msg


def test_protokoll_nennt_das_greifende_recht(rechte):
    assert auth.gegriffenes_recht(["mcp:tree:write"], "tree", "tree_get") == "mcp:tree:write"
    assert auth.gegriffenes_recht(["mcp:tree:status"], "tree", "tree_outline") == "mcp:tree:status"
    # eigene Stufe vor hoeherer, auch wenn die hoehere laenger ist
    assert auth.gegriffenes_recht(["mcp:tree:write", "mcp:tree:read"], "tree", "tree_get") == "mcp:tree:read"
    assert auth.gegriffenes_recht(["mcp:tree:status", "mcp:tree:read"], "tree", "tree_outline") == "mcp:tree:read"
    assert auth.gegriffenes_recht(["mcp:tree:*", "mcp:tree:write"], "tree", "tree_get") == "mcp:tree:write"
    assert auth.gegriffenes_recht(["mcp:tree:read"], "tree", "nodes_create") == "?"


def test_person_bleibt_woertlich(rechte):
    """person-api staffelt nicht: write liest dort nicht mit (Gegenstueck)."""
    lesen = {w for w, s in PERSON_STUFEN.items() if s == "read"}
    schreiben = {w for w, s in PERSON_STUFEN.items() if s == "write"}
    rechte(["mcp:person:write"])
    assert _erlaubt("person", lesen | schreiben) == schreiben
