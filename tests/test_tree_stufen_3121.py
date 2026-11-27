"""Tree-Werkzeuge mit Rechte-Stufen im Gateway (#3121, Projektmanager-Rolle Beitrag 6162).

tree-api trennt mcp:tree:* auf in read / status / write, die Projekt-Bindung ist
tree:projekt:<id> (ohne mcp:, Beitrag 6162). Das Gateway prueft je
Werkzeug mcp:tree:<werkzeug> oder mcp:tree:<stufe> (WERKZEUG_STUFEN["tree"]); ohne
Anmeldung wiese es einen Agenten mit nur read und status ab, bevor tree-api ihn sieht.
Die Stufen sind gestaffelt wie in tree-api (#3130: write ⊇ status ⊇ read); mcp:tree:*
und Platzhalter decken alles.
Nur gegen die echte JWTAuthMiddleware mit gefaelschtem JWT -- keine tree-api.
"""
import asyncio
from rollensperre import GESPERRT, code  # noqa: E402  (#4171)
import json
import os
import subprocess
import sys

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

LESEN = {"service_health", "projects_list", "projects_get", "tree_get", "tree_outline",
         "tree_compact", "tree_export", "nodes_get", "persons_list"}
STATUS = {"nodes_update"}
SCHREIBEN = {"projects_create", "projects_delete", "tree_import", "nodes_create", "nodes_move",
             "nodes_delete", "persons_create", "persons_update", "persons_delete",
             "assign_person", "unassign_person"}


def _werkzeuge():
    return {t.name for t in server.tree_mcp._tool_manager.list_tools()}


# ---------------------------------------------------------------------------
# Zuordnung

def test_jedes_tree_werkzeug_hat_genau_seine_stufe():
    """Ein neues Tree-Werkzeug ohne Stufe faellt hier auf (es ginge sonst nur mit mcp:tree:*)."""
    assert set(server.TREE_WERKZEUG_STUFEN) == _werkzeuge() == LESEN | STATUS | SCHREIBEN
    stufen = {}
    for w, s in server.TREE_WERKZEUG_STUFEN.items():
        stufen.setdefault(s, set()).add(w)
    assert stufen == {"read": LESEN, "status": STATUS, "write": SCHREIBEN}


def test_server_meldet_tree_am_gateway_an():
    """Eigener Prozess: test_offene_server laedt `auth` neu und verwirft damit, was
    server.py beim Import angemeldet hat -- im Dienst geschieht das nie."""
    code = ("import json, auth, server; print(json.dumps({"
            "'stufen': auth.WERKZEUG_STUFEN.get('tree') == server.TREE_WERKZEUG_STUFEN,"
            "'reihe': auth.STUFEN_REIHE.get('tree') == ('read', 'status', 'write'),"
            "'explicit': 'tree' in auth.EXPLICIT_ONLY_SERVERS,"
            "'werkzeuge': sorted(t.name for t in server.tree_mcp._tool_manager.list_tools())}))")
    wurzel = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    r = subprocess.run([sys.executable, "-c", code], cwd=wurzel, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr[-2000:]
    stand = json.loads(r.stdout.strip().splitlines()[-1])
    # Kein reiner Ausdruecklich-Server: mcp:*, mcp:*:* und * gelten fuer tree weiter.
    assert stand == {"stufen": True, "reihe": True, "explicit": False,
                     "werkzeuge": sorted(LESEN | STATUS | SCHREIBEN)}


# ---------------------------------------------------------------------------
# Gateway: Recht je Werkzeug (echte JWTAuthMiddleware)

def _app():
    async def ok(request):
        return JSONResponse({"ok": 1})
    app = Starlette(routes=[Route("/{srv}/", ok, methods=["POST"])])
    app.add_middleware(auth.JWTAuthMiddleware)
    return app


@pytest.fixture
def rechte(monkeypatch):
    monkeypatch.setattr(auth, "EXPLICIT_ONLY_SERVERS", set())
    monkeypatch.setattr(auth, "OFFENE_SERVER", set())
    monkeypatch.setattr(auth, "WERKZEUG_STUFEN", {"tree": dict(server.TREE_WERKZEUG_STUFEN)})
    monkeypatch.setattr(auth, "STUFEN_REIHE", {"tree": server.TREE_STUFEN_REIHE})

    async def key():
        return "k"

    def setze(perms):
        monkeypatch.setattr(auth, "_pruefen", claims_fest(lambda token: {
            "type": "agent", "agent_name": "Projektmanager", "role": "pm", "permissions": perms}))
    return setze


def _call(*werkzeuge):
    teile = [{"jsonrpc": "2.0", "id": i, "method": "tools/call", "params": {"name": w, "arguments": {}}}
             for i, w in enumerate(werkzeuge)]
    body = teile[0] if len(teile) == 1 else teile

    async def go():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=_app()), base_url="http://gw") as c:
            return await c.post("/tree/", content=json.dumps(body),
                                headers={"Content-Type": "application/json", "Authorization": "Bearer t"})
    return code(asyncio.run(go()))


def _erlaubt():
    return {w for w in LESEN | STATUS | SCHREIBEN if _call(w) == 200}


def test_nur_read(rechte):
    rechte(["mcp:tree:read"])
    assert _erlaubt() == LESEN
    assert _call("nodes_update") == GESPERRT
    assert _call("nodes_create") == GESPERRT


def test_read_und_status(rechte):
    """Die Projektmanager-Rolle: lesen und Status setzen, nichts anlegen."""
    rechte(["mcp:tree:read", "mcp:tree:status"])
    assert _erlaubt() == LESEN | STATUS
    assert _call("nodes_update") == 200
    assert _call("nodes_create") == GESPERRT


def test_mit_projekt_begrenzung(rechte):
    """tree:projekt:<id> (Beitrag 6162, ohne mcp:) wertet nur tree-api aus; am Gateway
    oeffnet es kein Werkzeug und stoert die Stufen nicht. Auch eine versehentlich mit
    mcp: gepraegte Form oeffnet nichts, nicht einmal als Platzhalter."""
    rechte(["mcp:tree:read", "mcp:tree:status", "tree:projekt:5"])
    assert _erlaubt() == LESEN | STATUS
    rechte(["tree:projekt:5"])
    assert _erlaubt() == set()
    rechte(["tree:projekt:*"])
    assert _erlaubt() == set()
    rechte(["mcp:tree:projekt:5"])
    assert _erlaubt() == set()
    rechte(["mcp:tree:projekt:*"])
    assert _erlaubt() == set()


@pytest.mark.parametrize("perms,erlaubt", [
    (["mcp:tree:status"], LESEN | STATUS),                 # gestaffelt (#3130): status liest auch
    (["mcp:tree:write"], LESEN | STATUS | SCHREIBEN),      # gestaffelt (#3130): write deckt alles
    (["mcp:tree:read", "mcp:tree:write"], LESEN | STATUS | SCHREIBEN),
    (["mcp:tree:read", "mcp:tree:status", "mcp:tree:write"], LESEN | STATUS | SCHREIBEN),
    (["mcp:tree:nodes_get"], {"nodes_get"}),               # einzelnes Werkzeug ausdruecklich
    (["mcp:tree:*"], LESEN | STATUS | SCHREIBEN),          # bestehende Agenten: unveraendert
    (["mcp:*"], LESEN | STATUS | SCHREIBEN),
    (["mcp:*:*"], LESEN | STATUS | SCHREIBEN),
    (["*"], LESEN | STATUS | SCHREIBEN),
    (["mcp:content:read"], set()),                         # Stufe eines anderen Servers
    ([], set()),
])
def test_stufen_gestaffelt(rechte, perms, erlaubt):
    rechte(perms)
    assert _erlaubt() == erlaubt


def test_stapel_wird_je_werkzeug_geprueft(rechte):
    rechte(["mcp:tree:read", "mcp:tree:status"])
    assert _call("tree_get", "nodes_update") == 200
    assert _call("tree_get", "nodes_delete") == 403


def test_protokoll_nennt_die_stufe(monkeypatch):
    """gegriffenes_recht (#2806) nennt die Stufe statt eines Platzhalters."""
    monkeypatch.setattr(auth, "EXPLICIT_ONLY_SERVERS", set())
    monkeypatch.setattr(auth, "OFFENE_SERVER", set())
    monkeypatch.setattr(auth, "WERKZEUG_STUFEN", {"tree": dict(server.TREE_WERKZEUG_STUFEN)})
    monkeypatch.setattr(auth, "STUFEN_REIHE", {"tree": server.TREE_STUFEN_REIHE})
    perms = ["mcp:tree:read", "mcp:tree:status"]
    assert auth.gegriffenes_recht(perms, "tree", "tree_outline") == "mcp:tree:read"
    assert auth.gegriffenes_recht(perms, "tree", "nodes_update") == "mcp:tree:status"
    assert auth.gegriffenes_recht(perms, "tree", "nodes_create") == "?"
