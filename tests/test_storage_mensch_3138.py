"""#3138: storage calls of a non-agent caller carry X-Grant-Scope-Restrict too.

Before, `_storage_headers()` narrowed only agents (#3030). A human token
(login or OAuth connector with `mcp:storage`), a service token or an anonymous
caller sent no header and saw, with the shared gateway key, everything that
account sees. Since 16.11. the tenant gate lets `edera` through to storage, so
that included humans of edera. Now only a superadmin goes unrestricted; every
other non-agent caller is narrowed to a scope that matches no grant.
"""
import asyncio
import base64
import json
import os
import re
import sys
from pathlib import Path

import httpx
import pytest
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
for _k in ("ARKTURIAN_API_KEY", "ONEAL_STORAGE_API_KEY", "JWT_PUBLIC_KEY", "COMM_API_KEY"):
    os.environ.setdefault(_k, "test-only")

import auth  # noqa: E402
from pruef_helfer import claims_fest  # noqa: E402

# Abschrift aus storage-api storage/signing.py (origin/main 231c53b). Jede
# Freigabe geht dort durch _validate_scope mit genau diesem Muster.
STORAGE_SCOPE_RE = re.compile(r"^[a-z][a-z0-9-]{0,40}:[A-Za-z0-9._-]{1,120}$")


@pytest.fixture
def aufrufer():
    """Setzt die Identitaet, wie JWTAuthMiddleware sie aus einem gepruften JWT setzt."""
    tokens = []

    def setze(name="", agent=False, superadmin=False):
        tokens.append((auth._current_agent_name.set(name), auth._current_is_agent.set(agent),
                       auth._current_is_superadmin.set(superadmin)))
    yield setze
    for n, a, s in reversed(tokens):
        auth._current_is_superadmin.reset(s)
        auth._current_is_agent.reset(a)
        auth._current_agent_name.reset(n)


def _capture(calls, answer=None):
    async def _f(method, url, **kw):
        calls.append((method, url, kw))
        return answer if answer is not None else {"ok": True}
    return _f


LESEN = [
    lambda s: s.storage_assets_get(id=5),
    lambda s: s.storage_assets_list(),
    lambda s: s.storage_assets_similar(id=5),
    lambda s: s.storage_kg_search(query="x"),
    lambda s: s.storage_assets_get_embedding_text(id=5),
    lambda s: s.storage_kg_embed(id=5),
    lambda s: s.storage_assets_refs(),
]


def _restrict(calls):
    import server
    werte = [kw["headers"].get("X-Grant-Scope-Restrict") for _, url, kw in calls
             if url.startswith(server.STORAGE_API_BASE)]
    assert werte, "no storage call made"
    return werte


def test_sperrwert_passt_auf_keine_freigabe():
    import server
    assert not STORAGE_SCOPE_RE.match(server.STORAGE_RESTRICT_OHNE_SCOPE)
    assert not server.STORAGE_RESTRICT_OHNE_SCOPE.startswith(("cloud-session", "cloud-user", "cloud-tenant"))
    assert len(server.STORAGE_RESTRICT_OHNE_SCOPE) <= 200        # storage-api kappt bei 200


@pytest.mark.parametrize("call", LESEN)
def test_mensch_ohne_superadmin_eingeschraenkt(monkeypatch, aufrufer, call):
    """OAuth-Connector eines Menschen: im Namensfeld steht sein sub, kein Agent."""
    import server
    aufrufer(name="7f1c2b9e-0000-4000-8000-000000000001")
    calls = []
    monkeypatch.setattr(server, "_fetch_json", _capture(calls))
    asyncio.run(call(server))
    assert set(_restrict(calls)) == {server.STORAGE_RESTRICT_OHNE_SCOPE}


@pytest.mark.parametrize("call", LESEN)
def test_superadmin_uneingeschraenkt(monkeypatch, aufrufer, call):
    import server
    aufrufer(name="alex-sub", superadmin=True)
    calls = []
    monkeypatch.setattr(server, "_fetch_json", _capture(calls))
    asyncio.run(call(server))
    assert set(_restrict(calls)) == {None}


def test_agent_unveraendert_eigener_scope(monkeypatch, aufrufer):
    import server
    aufrufer(name="EDERA-HERMINE (CFO)", agent=True)
    calls = []
    monkeypatch.setattr(server, "_fetch_json", _capture(calls))
    asyncio.run(server.storage_assets_list())
    assert _restrict(calls) == [server.storage_grant_scope_for("EDERA-HERMINE (CFO)")]


def test_anonym_eingeschraenkt(monkeypatch):
    """Ohne JWT (nur mit AUTH_REQUIRE_JWT=false erreichbar): fail-closed."""
    import server
    calls = []
    monkeypatch.setattr(server, "_fetch_json", _capture(calls))
    asyncio.run(server.storage_assets_get(id=5))
    assert _restrict(calls) == [server.STORAGE_RESTRICT_OHNE_SCOPE]


def test_mensch_upload_ohne_grant_scope(monkeypatch, aufrufer):
    """Der Sperrwert ist nie eine Freigabe: kein grant_scope bei Uploads eines Menschen."""
    import server
    aufrufer(name="7f1c2b9e-0000-4000-8000-000000000001")
    calls = []
    monkeypatch.setattr(server, "_fetch_json", _capture(calls, {"upload_url": "u"}))
    asyncio.run(server.storage_assets_fetch(url="https://x.example/a.png", analyze=False))
    asyncio.run(server.storage_assets_upload(file_base64="TICKET", filename="a.png"))
    payload = base64.b64encode(b"x").decode()
    asyncio.run(server.storage_assets_upload(file_base64=f"CHUNK:u1:0:1:{payload}", filename="a.png"))
    for _, _, kw in calls:
        assert "grant_scope" not in (kw.get("json_body") or {})
        assert "grant_scope" not in (kw.get("params") or {})
    assert set(_restrict(calls)) == {server.STORAGE_RESTRICT_OHNE_SCOPE}


def test_mensch_multipart_upload_eingeschraenkt(monkeypatch, aufrufer):
    import server
    aufrufer(name="7f1c2b9e-0000-4000-8000-000000000001")
    gesehen = {}

    class _Antwort:
        def raise_for_status(self):
            return None

        def json(self):
            return {"id": 1}

    class _Client:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, url, headers=None, files=None, data=None):
            gesehen.update(headers=headers, data=data)
            return _Antwort()
    monkeypatch.setattr(server.httpx, "AsyncClient", _Client)
    asyncio.run(server.call_storage_upload(b"x", "a.png", form_fields={"is_public": "false"}))
    assert gesehen["headers"]["X-Grant-Scope-Restrict"] == server.STORAGE_RESTRICT_OHNE_SCOPE
    assert "grant_scope" not in gesehen["data"]


# ── durch die echte Middleware ──────────────────────────────────────────────

def _app(seen):
    async def ok(request):
        import server
        seen.append((server._storage_restrict_scope(), auth.current_caller_is_superadmin()))
        return JSONResponse({"ok": 1})
    app = Starlette(routes=[Route("/{srv}/", ok, methods=["POST"])])
    app.add_middleware(auth.JWTAuthMiddleware)
    return app


CALL = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                   "params": {"name": "assets_list", "arguments": {}}})


@pytest.fixture
def gateway(monkeypatch):
    monkeypatch.setattr(auth, "EXPLICIT_ONLY_SERVERS", set())
    monkeypatch.setattr(auth, "MANDANTEN_SCHRANKE", auth._schranke_laden("storage:arkturian,edera"))

    async def key():
        return "k"

    def post(claims):
        monkeypatch.setattr(auth, "_pruefen", claims_fest(lambda token: claims))
        seen = []

        async def go():
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=_app(seen)),
                                         base_url="http://gw") as c:
                return await c.post("/storage/", content=CALL, headers={
                    "Authorization": "Bearer t", "Content-Type": "application/json"})
        r = asyncio.run(go())
        return r.status_code, seen
    return post


MENSCH_EDERA = {"type": "user", "sub": "7f1c2b9e-0000-4000-8000-000000000001", "email": "m@edera.example",
                "scope": "mcp:storage", "is_superadmin": False,
                "roles": [{"sys": "cloud", "role": "member", "tid": "edera"}]}


def test_middleware_mensch_von_edera_durch_schranke_aber_eingeschraenkt(gateway):
    import server
    status, seen = gateway(MENSCH_EDERA)
    assert status == 200
    assert seen == [(server.STORAGE_RESTRICT_OHNE_SCOPE, False)]


def test_middleware_superadmin_uneingeschraenkt(gateway):
    status, seen = gateway(dict(MENSCH_EDERA, is_superadmin=True,
                                roles=[{"sys": "cloud", "role": "admin", "tid": "arkturian"}]))
    assert status == 200 and seen == [("", True)]


def test_middleware_superadmin_nur_mit_echtem_true(gateway):
    import server
    status, seen = gateway(dict(MENSCH_EDERA, is_superadmin="true"))
    assert status == 200 and seen == [(server.STORAGE_RESTRICT_OHNE_SCOPE, False)]


def test_middleware_agent_eigener_scope(gateway):
    status, seen = gateway({"type": "agent", "agent_name": "Business", "role": "developer",
                            "tenant": "arkturian", "permissions": ["mcp:storage:*"]})
    assert status == 200 and seen == [("cloud-session:Business", False)]


def test_middleware_dienst_eingeschraenkt(gateway):
    import server
    status, seen = gateway({"type": "service", "sub": "service:comm", "tenant": "arkturian",
                            "permissions": ["mcp:storage:*"]})
    assert status == 200 and seen == [(server.STORAGE_RESTRICT_OHNE_SCOPE, False)]


def test_middleware_kein_nachlaufen_zwischen_anfragen(gateway):
    """Nach einer Superadmin-Anfrage traegt der Kontext keine Markierung mehr."""
    gateway(dict(MENSCH_EDERA, is_superadmin=True))
    assert auth.current_caller_is_superadmin() is False
