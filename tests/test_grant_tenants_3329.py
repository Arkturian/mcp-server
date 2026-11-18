"""#3329 (Storage df15d25a, Cloud): Wo das Gateway X-Grant-Scope-Restrict an storage-api
schickt, geht X-Grant-Scope-Tenants mit, sonst kennt die Grant-Einschraenkung beim
Vorab-Lesen den Mandanten nicht, und assets_download auf personal-/intern-Objekte endet 404.

Der Wert kommt NUR aus dem geprueften JWT: Agent = `tenant`-Claim, Mensch = Mitgliedschaften
`roles[].tid` ohne Gast. Kommagetrennt, hoechstens 100, nur [a-z0-9_-]. Kein
Werkzeug-Parameter und kein Zusatzkopf kann ihn veraendern.
"""
import asyncio
import json
import os
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

AGENT = {"type": "agent", "agent_name": "Business", "role": "developer", "tenant": "arkturian",
         "tenant_ids": ["arkturian", "edera"], "permissions": ["mcp:storage:*"]}
MENSCH = {"type": "user", "sub": "7f1c2b9e-0000-4000-8000-000000000001", "email": "m@edera.example",
          "scope": "mcp:storage", "is_superadmin": False, "roles": [{"sys": "cloud", "role": "member", "tid": "edera"},
                                             {"sys": "content", "role": "viewer", "tid": "edera"},
                                             {"sys": "cloud", "role": "member", "tid": "oneal_eu"}]}


# ── Mandanten aus den Claims ───────────────────────────────────────────────

@pytest.mark.parametrize("claims,erwartet", [
    (AGENT, ("arkturian",)),                                         # nur tenant, nie tenant_ids
    (dict(AGENT, tenant=""), ()),
    (MENSCH, ("edera", "oneal_eu")),                                 # Mitgliedschaften, ohne Doppel
    (dict(MENSCH, type="oauth_access"), ("edera", "oneal_eu")),
    (dict(MENSCH, guest_tenant="oneal_eu"), ("edera",)),             # Gast ist keine Mitgliedschaft
    (dict(MENSCH, roles=[{"tid": "Edera"}, {"tid": "a b"}, {"tid": "x;y"}, {"tid": "ok-1"}]), ("ok-1",)),
    ({"type": "service", "sub": "service:comm", "tenant": "arkturian"}, ()),
    ({"sub": "alt", "tenant": "arkturian"}, ()),                     # ohne type: keine
    (dict(AGENT, agent_name=""), ()),
])
def test_mandanten_aus_claims(claims, erwartet):
    assert auth.mandanten_aus_claims(claims) == erwartet


def test_hoechstens_100():
    viele = dict(MENSCH, roles=[{"tid": f"m{i}"} for i in range(150)])
    m = auth.mandanten_aus_claims(viele)
    assert len(m) == 100 and m[0] == "m0" and m[-1] == "m99"


# ── durch die echte Middleware bis zum Kopf an storage-api ──────────────────

def _app(seen):
    async def ok(request):
        import server
        seen.append(server._storage_headers())
        return JSONResponse({"ok": 1})
    app = Starlette(routes=[Route("/{srv}/", ok, methods=["POST"])])
    app.add_middleware(auth.JWTAuthMiddleware)
    return app


def _call(argumente):
    return json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                       "params": {"name": "assets_list", "arguments": argumente}})


@pytest.fixture
def gateway(monkeypatch):
    monkeypatch.setattr(auth, "EXPLICIT_ONLY_SERVERS", set())
    monkeypatch.setattr(auth, "MANDANTEN_SCHRANKE", {})

    def post(claims, argumente=None):
        monkeypatch.setattr(auth, "_pruefen", claims_fest(lambda token: claims))
        seen = []

        async def go():
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=_app(seen)),
                                         base_url="http://gw") as c:
                return await c.post("/storage/", content=_call(argumente or {}), headers={
                    "Authorization": "Bearer t", "Content-Type": "application/json",
                    "X-Grant-Scope-Tenants": "fremd"})
        r = asyncio.run(go())
        assert r.status_code == 200
        return seen[0]
    return post


def test_agent_traegt_seinen_mandanten(gateway):
    h = gateway(AGENT)
    assert h["X-Grant-Scope-Restrict"] == "cloud-session:Business"
    assert h["X-Grant-Scope-Tenants"] == "arkturian"


def test_mensch_traegt_seine_mitgliedschaften(gateway):
    import server
    h = gateway(MENSCH)
    assert h["X-Grant-Scope-Restrict"] == server.STORAGE_RESTRICT_OHNE_SCOPE
    assert h["X-Grant-Scope-Tenants"] == "edera,oneal_eu"


def test_werkzeug_parameter_und_eingehender_kopf_aendern_nichts(gateway):
    h = gateway(AGENT, {"tenant": "edera", "tenants": "edera", "X-Grant-Scope-Tenants": "edera"})
    assert h["X-Grant-Scope-Tenants"] == "arkturian"


def test_superadmin_ohne_einschraenkung_ohne_mandanten(gateway):
    h = gateway(dict(MENSCH, is_superadmin=True))
    assert "X-Grant-Scope-Restrict" not in h and "X-Grant-Scope-Tenants" not in h


def test_dienst_und_agent_ohne_mandant_ohne_kopf(gateway):
    import server
    h = gateway({"type": "service", "sub": "service:comm", "tenant": "arkturian",
                 "permissions": ["mcp:storage:*"]})
    assert h["X-Grant-Scope-Restrict"] == server.STORAGE_RESTRICT_OHNE_SCOPE
    assert "X-Grant-Scope-Tenants" not in h
    assert "X-Grant-Scope-Tenants" not in gateway(dict(AGENT, tenant=""))


def test_kein_nachlaufen_zwischen_anfragen(gateway):
    gateway(AGENT)
    assert auth.current_caller_tenants() == ()


# ── echte Werkzeuge: der Kopf geht an jeden storage-Aufruf ──────────────────

@pytest.fixture
def als_agent():
    t = (auth._current_agent_name.set("Business"), auth._current_is_agent.set(True),
         auth._current_tenants.set(("arkturian",)))
    yield
    auth._current_tenants.reset(t[2])
    auth._current_is_agent.reset(t[1])
    auth._current_agent_name.reset(t[0])


def _capture(calls, answer=None):
    async def _f(method, url, **kw):
        calls.append((method, url, kw))
        return answer if answer is not None else {"ok": True, "upload_url": "u"}
    return _f


def test_assets_download_und_lesen_tragen_den_mandanten(monkeypatch, als_agent):
    import server
    calls = []
    monkeypatch.setattr(server, "_fetch_json", _capture(calls))
    asyncio.run(server.storage_assets_get(id=5))
    asyncio.run(server.storage_assets_list())
    asyncio.run(server.storage_assets_upload_ticket(filename="a.png"))
    koepfe = [kw["headers"] for _, url, kw in calls if url.startswith(server.STORAGE_API_BASE)]
    assert koepfe and all(k.get("X-Grant-Scope-Tenants") == "arkturian" for k in koepfe)


def test_zusatzkoepfe_ueberschreiben_die_identitaet_nicht(monkeypatch, als_agent):
    import server
    calls = []
    monkeypatch.setattr(server, "_fetch_json", _capture(calls))
    asyncio.run(server.call_storage_api("GET", "/storage/objects/5", extra_headers={
        "x-grant-scope-tenants": "edera", "X-Grant-Scope-Restrict": "cloud-session:Anders",
        "X-API-KEY": "anderer", "X-Filename": "a.png"}))
    h = calls[0][2]["headers"]
    assert h["X-Grant-Scope-Tenants"] == "arkturian" and h["X-Grant-Scope-Restrict"] == "cloud-session:Business"
    assert h["X-API-KEY"] == server.STORAGE_API_KEY and h["X-Filename"] == "a.png"
    assert [k for k in h if k.lower() == "x-grant-scope-tenants"] == ["X-Grant-Scope-Tenants"]
