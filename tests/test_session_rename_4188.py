"""#4188 (Alex 27.11.): Manuels Agenten benennen sich selbst um -- Werkzeug session_rename im cloud-MCP.

- Nur der eigene Name; das Tor in cloud-api (lifecycle_gate: der Agent selbst) prueft es noch einmal.
- Nur mit dem ausdruecklichen Recht mcp:cloud:session_rename: mcp:cloud:*, mcp:* und * oeffnen es nicht
  (NUR_AUSDRUECKLICH_WERKZEUGE), denn die Rolle customer-agent traegt mcp:cloud:* auch in anderen
  Mandanten. AuthAPI traegt das Recht nur fuer die Agenten des Mandanten lesser ein.
Gateway gegen die echte JWTAuthMiddleware mit gefaelschtem JWT, Werkzeug ohne cloud-api.
"""
import asyncio
import json
import os
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route

for _k in ("ARKTURIAN_API_KEY", "ONEAL_STORAGE_API_KEY", "JWT_PUBLIC_KEY", "COMM_API_KEY"):
    os.environ.setdefault(_k, "test-only")

import auth  # noqa: E402
from pruef_helfer import claims_fest  # noqa: E402
from rollensperre import GESPERRT, code  # noqa: E402
import server  # noqa: E402


# ── Werkzeug ────────────────────────────────────────────────────────────────

def _als(name, agent=True):
    return auth._current_agent_name.set(name), auth._current_is_agent.set(agent)


def _zurueck(t):
    auth._current_agent_name.reset(t[0])
    auth._current_is_agent.reset(t[1])


def _rufen(name, session_name, new_name, agent=True):
    t = _als(name, agent)
    try:
        with patch.object(server, "call_cloud_api", new=AsyncMock(return_value={"status": "renamed"})) as call:
            out = asyncio.run(server.cloud_session_rename(session_name, new_name))
    finally:
        _zurueck(t)
    return out, call


def test_eigener_name_geht_ueber_den_portalweg():
    out, call = _rufen("r_cubic", "r_cubic", "r-cubic_Development")
    assert out == {"status": "renamed"}
    assert call.call_args.args == ("POST", "/api/sessions/r_cubic/rename")
    assert call.call_args.kwargs["json_body"] == {"new_name": "r-cubic_Development"}
    assert call.call_args.kwargs["timeout"] >= 120


@pytest.mark.parametrize("name, ziel, agent, fehler", [
    ("r_cubic", "r-cubic admin", True, "nur_eigener_name"),       # fremder Agent
    ("", "r_cubic", False, "nur_agenten"),                         # Mensch / kein Agenten-Token
    ("r_cubic", "r_cubic", False, "nur_agenten"),                  # Name im Token, aber kein Agent
    ("r_cubic", "r_cubic", True, "neuer_name_fehlt"),
])
def test_alles_andere_wird_abgelehnt_ohne_aufruf(name, ziel, agent, fehler):
    neu = "r_cubic" if fehler == "neuer_name_fehlt" else "x-neu"
    out, call = _rufen(name, ziel, neu, agent)
    assert out["error"] == fehler and out["status"] == "error"
    call.assert_not_called()


def test_werkzeug_ist_registriert():
    assert "session_rename" in {t.name for t in asyncio.run(server.cloud_mcp.list_tools())}


# ── Gateway: nur ausdrueckliches Recht ──────────────────────────────────────

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
    monkeypatch.setattr(auth, "NUR_AUSDRUECKLICH_WERKZEUGE", {("cloud", "session_rename")})

    def setze(perms):
        monkeypatch.setattr(auth, "_pruefen", claims_fest(lambda token: {
            "type": "agent", "agent_name": "r_cubic", "role": "customer-agent", "permissions": perms}))
    return setze


def _call(werkzeug, srv="cloud"):
    body = {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": werkzeug, "arguments": {}}}

    async def go():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=_app()), base_url="http://gw") as c:
            return await c.post(f"/{srv}/", content=json.dumps(body),
                                headers={"Content-Type": "application/json", "Authorization": "Bearer t"})
    return code(asyncio.run(go()))


@pytest.mark.parametrize("perms", [["mcp:cloud:*"], ["mcp:*"], ["*"], ["mcp:*:*"]])
def test_platzhalter_oeffnet_session_rename_nicht(rechte, perms):
    rechte(perms)
    assert _call("session_rename") == GESPERRT
    assert _call("health") == 200, "die uebrigen Werkzeuge bleiben wie bisher offen"


@pytest.mark.parametrize("perms", [["mcp:cloud:*", "mcp:cloud:session_rename"], ["mcp:cloud:session_rename"]])
def test_ausdrueckliches_recht_oeffnet_es(rechte, perms):
    rechte(perms)
    assert _call("session_rename") == 200


def test_gegenprobe_ohne_liste_wie_bisher(rechte, monkeypatch):
    monkeypatch.setattr(auth, "NUR_AUSDRUECKLICH_WERKZEUGE", set())
    rechte(["mcp:cloud:*"])
    assert _call("session_rename") == 200


def test_anderer_server_gleichen_werkzeugnamens_unberuehrt(rechte):
    rechte(["mcp:content:*"])
    assert _call("session_rename", srv="content") == 200


def test_vorgabe_enthaelt_session_rename():
    assert ("cloud", "session_rename") in {
        (t.split(":", 1)[0], t.split(":", 1)[1]) for t in auth._NUR_AUSDRUECKLICH_VORGABE.split(",")}


def test_protokoll_nennt_das_woertliche_recht():
    assert auth.gegriffenes_recht(["mcp:cloud:*", "mcp:cloud:session_rename"], "cloud", "session_rename") \
        == "mcp:cloud:session_rename"
    assert auth.gegriffenes_recht(["mcp:cloud:*"], "cloud", "session_rename") == "?"


def test_nur_ausdruecklich_gilt_je_server_und_werkzeug(monkeypatch):
    monkeypatch.setattr(auth, "NUR_AUSDRUECKLICH_WERKZEUGE", {("cloud", "session_rename")})
    assert auth.nur_ausdruecklich("cloud", "session_rename") is True
    assert auth.nur_ausdruecklich("content", "session_rename") is False
    assert auth.nur_ausdruecklich("cloud", "session_restart") is False


def test_neuer_name_ohne_raender_und_pfad_aus_dem_token():
    out, call = _rufen("r_cubic", "r_cubic", "  r-cubic_Development ")
    assert call.call_args.args == ("POST", "/api/sessions/r_cubic/rename")
    assert call.call_args.kwargs["json_body"] == {"new_name": "r-cubic_Development"}


def _stapel(teile, srv="cloud"):
    async def go():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=_app()), base_url="http://gw") as c:
            return await c.post(f"/{srv}/", content=json.dumps(teile),
                                headers={"Content-Type": "application/json", "Authorization": "Bearer t"})
    return code(asyncio.run(go()))


def test_stapel_mit_session_rename_ist_gesperrt(rechte):
    """Cloud 76d91ba6: ein Stapel [health, session_rename] mit mcp:cloud:* -- der zweite Aufruf sperrt."""
    rechte(["mcp:cloud:*"])
    teile = [{"jsonrpc": "2.0", "id": i, "method": "tools/call", "params": {"name": w, "arguments": {}}}
             for i, w in enumerate(["health", "session_rename"])]
    assert _stapel(teile) == 403                      # Stapel: gesperrt als HTTP 403, nicht als Werkzeugergebnis
    rechte(["mcp:cloud:*", "mcp:cloud:session_rename"])
    assert _stapel(teile) == 200


@pytest.mark.parametrize("methode", ["initialize", "tools/list", "notifications/initialized", "ping"])
def test_andere_methoden_mit_platzhalter_bleiben_offen(rechte, methode):
    """Der Rumpf wird jetzt fuer alle cloud-Aufrufe gelesen -- Handshake und Liste bleiben 200."""
    rechte(["mcp:cloud:*"])
    assert _stapel({"jsonrpc": "2.0", "id": 1, "method": methode, "params": {}}) == 200
