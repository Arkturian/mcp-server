"""#4171 (Cloud 14267): Eine Sperre nach Werkzeugrecht bei GUELTIGEM Token meldet sich klar als solche.

Projektleitung-v24 bekam fuer `posts_create`/`posts_list` (content) eine Ablehnung, die wie ein Fehler bei den
Anmeldedaten aussah, und meldete sie als Stoerung. Das Gateway antwortete mit HTTP 403; den Text im
JSON-RPC-Rumpf sah die Sitzung nicht. Jetzt: HTTP 200 mit Werkzeugergebnis `isError: true` und dem Text
`rolle_gesperrt: Werkzeug <mcp:server:werkzeug> ist fuer die Rolle <rolle> gesperrt ... Nicht wiederholen,
nicht melden.` Fehlende Tokens, die Sperre am Server, ein Stapel und eine gescheiterte Live-Pruefung bleiben
bei 401/403.
"""
import asyncio
import json

import httpx
import pytest
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route

import auth
from pruef_helfer import claims_fest  # noqa: E402

PL_RECHTE = ["mcp:content:sections_get", "mcp:content:sections_list", "mcp:content:sections_reply"]


def _app():
    async def ok(request):
        return JSONResponse({"jsonrpc": "2.0", "id": 1, "result": {"content": [{"type": "text", "text": "ok"}]}})
    app = Starlette(routes=[Route("/{srv}/", ok, methods=["POST"])])
    app.add_middleware(auth.JWTAuthMiddleware)
    return app


@pytest.fixture
def pl(monkeypatch):
    monkeypatch.setattr(auth, "OFFENE_SERVER", set())
    monkeypatch.setattr(auth, "EXPLICIT_ONLY_SERVERS", set())
    monkeypatch.setattr(auth, "_pruefen", claims_fest(lambda token: {
        "type": "agent", "agent_name": "Projektleitung-v24", "role": "projektleitung", "permissions": PL_RECHTE}))
    return monkeypatch


def _post(rumpf, server="content", token="t"):
    async def go():
        h = {"Content-Type": "application/json"}
        if token:
            h["Authorization"] = f"Bearer {token}"
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=_app()), base_url="http://gw") as c:
            return await c.post(f"/{server}/", content=json.dumps(rumpf), headers=h)
    return asyncio.run(go())


def _ruf(name, i=7):
    return {"jsonrpc": "2.0", "id": i, "method": "tools/call", "params": {"name": name, "arguments": {}}}


@pytest.mark.parametrize("werkzeug", ["posts_create", "posts_list"])
def test_gesperrtes_werkzeug_ist_ein_werkzeugergebnis_mit_code(pl, werkzeug):
    r = _post(_ruf(werkzeug))
    assert r.status_code == 200
    j = r.json()
    assert j["jsonrpc"] == "2.0" and j["id"] == 7 and "error" not in j
    assert j["result"]["isError"] is True
    text = j["result"]["content"][0]["text"]
    assert text.startswith("rolle_gesperrt: ")
    assert f"mcp:content:{werkzeug}" in text and "Rolle projektleitung" in text
    assert "Nicht wiederholen, nicht melden." in text


def test_erlaubtes_werkzeug_geht_durch(pl):
    r = _post(_ruf("sections_get"))
    assert r.status_code == 200 and r.json()["result"]["content"][0]["text"] == "ok"


def test_ohne_token_wie_bisher(pl):
    r = _post(_ruf("posts_create"), token=None)
    assert r.status_code in (401, 403) and "rolle_gesperrt" not in r.text


def test_server_ohne_jedes_recht_bleibt_403(pl):
    r = _post(_ruf("issue_create"), server="issue")
    assert r.status_code == 403 and "rolle_gesperrt" not in r.text


def test_stapel_bleibt_403_mit_demselben_code(pl):
    r = _post([_ruf("sections_get", 1), _ruf("posts_create", 2)])
    assert r.status_code == 403
    fehler = r.json()["error"]
    assert fehler["message"].startswith("rolle_gesperrt: ")
    assert fehler["data"] == {"code": "rolle_gesperrt", "werkzeug": "mcp:content:posts_create",
                              "rolle": "projektleitung"}


def test_hilfsfunktion_nennt_noetige_rechte():
    r = auth._rolle_gesperrt({"id": 3}, "developer", "mcp:tree:nodes_update", "noetig: a | b")
    text = json.loads(r.body)["result"]["content"][0]["text"]
    assert text == ("rolle_gesperrt: Werkzeug mcp:tree:nodes_update ist fuer die Rolle developer gesperrt "
                    "(noetig: a | b). Nicht wiederholen, nicht melden.")
    assert json.loads(auth._rolle_gesperrt({"id": 3}, "", "mcp:x:y").body)["result"]["content"][0]["text"] \
        .startswith("rolle_gesperrt: Werkzeug mcp:x:y ist fuer die Rolle ? gesperrt.")
