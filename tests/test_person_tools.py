"""MCP-Server `person` (#2441, Auftrag Alex ueber Business).

* Werkzeuge formen die Anfragen der person-api (/api/v1, Felder aus ihrem Quellcode).
* Transport: das JWT des aufrufenden Agenten als Bearer, kein Dienstschluessel;
  ohne Aufrufer-Kontext kein Aufruf, sondern ein klarer Fehler.
* Gateway: kein offener Server, Platzhalter zaehlen nicht; je Werkzeug die Stufe
  mcp:person:read|write|admin (woertlich, wie in der person-api) oder mcp:person:*.
Nur gegen Fakes -- die person-api ist vom Gateway noch nicht erreichbar.
"""
import asyncio
import json
import os
import subprocess
import sys
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route

os.environ.setdefault("MCP_SERVERS", "person")
import auth  # noqa: E402
import person_tools as pt  # noqa: E402
import server  # noqa: E402
from mcp.server.fastmcp import FastMCP  # noqa: E402


# ---------------------------------------------------------------------------
# Werkzeuge -> Anfragen

@pytest.fixture
def aufrufe():
    gesehen = []

    async def transport(method, path, **kw):
        gesehen.append((method, path, kw))
        return {"ok": True}
    mcp = FastMCP(name="person-test")
    pt.register_person_tools(mcp, transport)
    return mcp, gesehen


def _rufe(mcp, name, args):
    return asyncio.run(mcp.call_tool(name, args))


ERWARTET = [
    ("tenants_list", {}, ("GET", "/api/v1/tenants", {})),
    ("tenant_get", {"tenant_id": "t 1"}, ("GET", "/api/v1/tenants/t%201", {})),
    ("persons_list", {"tenant_id": "t1", "include_owner": False},
     ("GET", "/api/v1/persons", {"params": {"tenant_id": "t1", "include_owner": "false"}})),
    ("person_get", {"person_id": "p/1"}, ("GET", "/api/v1/persons/p%2F1", {})),
    ("resolve", {"tenant_id": "t1", "channel": "email", "value": "a@b.c"},
     ("GET", "/api/v1/resolve", {"params": {"tenant_id": "t1", "channel": "email", "value": "a@b.c"}})),
    ("merge_candidates", {"tenant_id": "t1"}, ("GET", "/api/v1/merge-candidates", {"params": {"tenant_id": "t1"}})),
    ("skip_rules_list", {"tenant_id": "t1"}, ("GET", "/api/v1/skip-rules", {"params": {"tenant_id": "t1"}})),
    ("person_create", {"tenant_id": "t1", "canonical_name": "Peter W", "is_owner": False,
                       "identifiers": [{"channel": "email", "value": "p@x.at"}]},
     ("POST", "/api/v1/persons", {"json_body": {"tenant_id": "t1", "canonical_name": "Peter W", "is_owner": False,
                                                "identifiers": [{"channel": "email", "value": "p@x.at"}]}})),
    ("person_update", {"person_id": "p1", "org": "DIHA"},
     ("PUT", "/api/v1/persons/p1", {"json_body": {"org": "DIHA"}})),
    ("resolve_or_create", {"tenant_id": "t1", "channel": "email", "value": "a@b.c", "name_hint": "A"},
     ("POST", "/api/v1/resolve-or-create", {"json_body": {"tenant_id": "t1", "channel": "email",
                                                          "value": "a@b.c", "name_hint": "A"}})),
    ("identifier_add", {"person_id": "p1", "channel": "phone", "value": "+43 1"},
     ("POST", "/api/v1/persons/p1/identifiers", {"json_body": {"channel": "phone", "value": "+43 1"}})),
    ("persons_merge", {"target_id": "p1", "source_id": "p2", "reason": "doppelt"},
     ("POST", "/api/v1/persons/p1/merge/p2", {"json_body": {"reason": "doppelt"}})),
    ("skip_rule_add", {"tenant_id": "t1", "value": "noreply@x.at"},
     ("POST", "/api/v1/skip-rules", {"json_body": {"tenant_id": "t1", "value": "noreply@x.at", "added_via": "mcp"}})),
    ("skip_rule_delete", {"rule_id": 7}, ("DELETE", "/api/v1/skip-rules/7", {})),
    ("person_skip", {"person_id": "p1"}, ("POST", "/api/v1/persons/p1/skip", {})),
]


@pytest.mark.parametrize("name,args,erwartet", ERWARTET, ids=[e[0] for e in ERWARTET])
def test_werkzeug_formt_die_anfrage(aufrufe, name, args, erwartet):
    mcp, gesehen = aufrufe
    _rufe(mcp, name, args)
    assert gesehen == [erwartet]


def test_jedes_werkzeug_hat_seine_stufe(aufrufe):
    mcp, _ = aufrufe
    namen = {t.name for t in asyncio.run(mcp.list_tools())}
    assert namen == set(pt.WERKZEUG_STUFEN) == {e[0] for e in ERWARTET}
    stufen = {}
    for n, s in pt.WERKZEUG_STUFEN.items():
        stufen.setdefault(s, set()).add(n)
    assert stufen == {
        "read": {"tenants_list", "tenant_get", "persons_list", "person_get", "resolve",
                 "merge_candidates", "skip_rules_list"},
        "write": {"person_create", "person_update", "resolve_or_create", "identifier_add"},
        "admin": {"persons_merge", "skip_rule_add", "skip_rule_delete", "person_skip"}}


def test_server_meldet_person_am_gateway_an():
    """Eigener Prozess: test_offene_server laedt `auth` neu und verwirft damit, was
    server.py beim Import angemeldet hat -- im Dienst geschieht das nie."""
    code = ("import json, auth, server, person_tools as pt; print(json.dumps({"
            "'explicit': 'person' in auth.EXPLICIT_ONLY_SERVERS, 'offen': auth._offen('person'),"
            "'stufen': auth.WERKZEUG_STUFEN.get('person') == pt.WERKZEUG_STUFEN,"
            "'werkzeuge': sorted(t.name for t in server.person_mcp._tool_manager.list_tools())}))")
    wurzel = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    r = subprocess.run([sys.executable, "-c", code], cwd=wurzel, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr[-2000:]
    stand = json.loads(r.stdout.strip().splitlines()[-1])
    assert stand == {"explicit": True, "offen": False, "stufen": True, "werkzeuge": sorted(pt.WERKZEUG_STUFEN)}


# ---------------------------------------------------------------------------
# Transport: Bearer des Aufrufers, sonst Fehler

def test_bearer_wird_durchgereicht():
    with patch.object(auth, "current_caller_jwt", return_value="jwt-business"), \
         patch.object(auth, "current_caller_agent_name", return_value="Business"), \
         patch.object(server, "_fetch_json", new=AsyncMock(return_value={"persons": []})) as f:
        asyncio.run(server.call_person_api("GET", "/api/v1/persons", params={"tenant_id": "t1"}))
    args, kw = f.call_args
    assert args == ("GET", "https://person-api.arkserver.arkturian.com/api/v1/persons")
    assert kw["headers"]["Authorization"] == "Bearer jwt-business"
    assert kw["headers"]["X-Agent-Name"] == "Business"
    assert not any(k.lower() in ("x-api-key", "x-iacp-token") for k in kw["headers"])


def test_ohne_aufrufer_kein_aufruf():
    with patch.object(auth, "current_caller_jwt", return_value=None), \
         patch.object(auth, "current_caller_agent_name", return_value=None), \
         patch.object(server, "_fetch_json", new=AsyncMock()) as f:
        with pytest.raises(ValueError, match="person_requires_caller"):
            asyncio.run(server.call_person_api("GET", "/api/v1/tenants"))
    f.assert_not_called()


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
    monkeypatch.setattr(auth, "EXPLICIT_ONLY_SERVERS", {"person"})
    monkeypatch.setattr(auth, "OFFENE_SERVER", set())
    monkeypatch.setattr(auth, "WERKZEUG_STUFEN", {"person": dict(pt.WERKZEUG_STUFEN)})

    async def key():
        return "k"
    monkeypatch.setattr(auth, "_get_public_key", key)

    def setze(perms):
        monkeypatch.setattr(auth, "_decode_jwt", lambda token, key: {
            "type": "agent", "agent_name": "X", "role": "r", "permissions": perms})
    return setze


def _post(body, token="t"):
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    async def go():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=_app()), base_url="http://gw") as c:
            return await c.post("/person/", content=json.dumps(body), headers=headers)
    return asyncio.run(go()).status_code


def _call(*werkzeuge, token="t"):
    teile = [{"jsonrpc": "2.0", "id": i, "method": "tools/call", "params": {"name": w, "arguments": {}}}
             for i, w in enumerate(werkzeuge)]
    return _post(teile[0] if len(teile) == 1 else teile, token=token)


@pytest.mark.parametrize("perms,werkzeug,code", [
    (["mcp:person:read"], "persons_list", 200),
    (["mcp:person:read"], "person_create", 403),
    (["mcp:person:read"], "persons_merge", 403),
    (["mcp:person:write"], "person_create", 200),
    (["mcp:person:write"], "persons_list", 403),          # woertlich, wie die person-api
    (["mcp:person:admin"], "persons_merge", 200),
    (["mcp:person:admin"], "person_update", 403),
    (["mcp:person:*"], "persons_merge", 200),
    (["mcp:person:persons_list"], "persons_list", 200),   # einzelnes Werkzeug ausdruecklich
    (["mcp:person:contact"], "persons_list", 403),        # Sicht-Recht der person-api, kein Werkzeug
    (["mcp:*"], "persons_list", 403),                     # kein offener Server, Platzhalter zaehlen nicht
    (["*"], "persons_list", 403),
    ([], "persons_list", 403),
])
def test_recht_je_werkzeug(rechte, perms, werkzeug, code):
    rechte(perms)
    assert _call(werkzeug) == code


def test_stapel_wird_je_werkzeug_geprueft(rechte):
    rechte(["mcp:person:read"])
    assert _call("persons_list", "person_get") == 200
    assert _call("persons_list", "persons_merge") == 403     # vorher: Liste umging die Pruefung


def test_stapel_ohne_token_wird_abgewiesen(rechte, monkeypatch):
    """Vorher: `.get` auf der Liste warf, der Stapel galt nicht als Werkzeugaufruf und lief
    ohne JWT weiter (heute faengt erst die mcp-Bibliothek ihn mit 400 ab)."""
    monkeypatch.setattr(auth, "AUTH_REQUIRE_JWT", True)
    assert _call("persons_list", "persons_merge", token=None) == 403


def test_ohne_werkzeugaufruf_bleibt_offen(rechte, monkeypatch):
    monkeypatch.setattr(auth, "AUTH_REQUIRE_JWT", True)
    assert _post({"jsonrpc": "2.0", "id": 1, "method": "tools/list"}, token=None) == 200
    assert _post([{"jsonrpc": "2.0", "method": "notifications/initialized"}], token=None) == 200
