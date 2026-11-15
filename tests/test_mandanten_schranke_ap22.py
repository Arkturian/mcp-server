"""Mandanten-Schranke (Cloud-Security AP 2.2, 15.11.2026).

An Hauptschluessel-Servern zaehlt neben dem Recht der Mandant aus dem JWT.
Vorgabe leer (Kundeninstanzen), auf dem Betreiber-Gateway per Env gesetzt.
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
                   "params": {"name": "documents_list", "arguments": {}}})
SPEC = "business,storage,artrack,codepilot,story,review:arkturian;oneal,oneal-storage:arkturian,oneal,oneal-europe"


def _app():
    async def ok(request):
        return JSONResponse({"ok": 1})
    app = Starlette(routes=[Route("/{srv}/", ok, methods=["POST"])])
    app.add_middleware(auth.JWTAuthMiddleware)
    return app


@pytest.fixture
def gateway(monkeypatch):
    monkeypatch.setattr(auth, "EXPLICIT_ONLY_SERVERS", set())

    async def key():
        return "k"
    monkeypatch.setattr(auth, "_get_public_key", key)

    def setze(tenant, schranke=SPEC, perms=("*",)):
        monkeypatch.setattr(auth, "MANDANTEN_SCHRANKE", auth._schranke_laden(schranke))
        claims = {"type": "agent", "agent_name": "X", "role": "developer", "permissions": list(perms)}
        if tenant is not None:
            claims["tenant"] = tenant
        monkeypatch.setattr(auth, "_decode_jwt", lambda token, key: claims)
    return setze


def _post(path):
    async def go():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=_app()),
                                     base_url="http://gw") as c:
            return await c.post(path, content=CALL, headers={
                "Authorization": "Bearer t", "Content-Type": "application/json"})
    return asyncio.run(go()).status_code


@pytest.mark.parametrize("tenant", ["edera", "martin", "pdrei", "weissc3", "user-x", "EDERA"])
@pytest.mark.parametrize("srv", ["business", "storage", "artrack", "codepilot", "story", "review", "oneal", "oneal-storage"])
def test_fremder_mandant_abgewiesen(gateway, tenant, srv):
    gateway(tenant)
    assert _post(f"/{srv}/") == 403


@pytest.mark.parametrize("srv", ["business", "storage", "oneal", "oneal-storage"])
def test_betreiber_durch(gateway, srv):
    gateway("arkturian")
    assert _post(f"/{srv}/") == 200


@pytest.mark.parametrize("tenant", ["oneal", "oneal-europe"])
def test_oneal_nur_an_oneal(gateway, tenant):
    gateway(tenant)
    assert _post("/oneal/") == 200
    assert _post("/oneal-storage/") == 200
    assert _post("/business/") == 403
    assert _post("/storage/") == 403


def test_ohne_mandanten_claim_unveraendert(gateway):
    gateway(None)
    assert _post("/business/") == 200


def test_andere_server_unberuehrt(gateway):
    gateway("edera")
    assert _post("/content/") == 200
    assert _post("/cloud/") == 200


def test_vorgabe_leer_kundeninstanz(gateway):
    gateway("p3", schranke="")
    assert _post("/business/") == 200
    assert _post("/storage/") == 200


def test_recht_bleibt_pflicht(gateway):
    gateway("arkturian", perms=())
    assert _post("/business/") == 403


# --- Menschen-/OAuth-Tokens ohne tenant-Claim (Cloud-Security AP 2.2, Restluecke) ---
@pytest.fixture
def mensch(monkeypatch):
    monkeypatch.setattr(auth, "EXPLICIT_ONLY_SERVERS", set())

    async def key():
        return "k"
    monkeypatch.setattr(auth, "_get_public_key", key)
    monkeypatch.setattr(auth, "MANDANTEN_SCHRANKE", auth._schranke_laden(SPEC))

    def setze(roles, superadmin=False):
        claims = {"sub": "u@x", "email": "u@x", "roles": roles, "is_superadmin": superadmin,
                  "scope": "openid mcp:business mcp:storage mcp:oneal", "aud": "connector"}
        monkeypatch.setattr(auth, "_decode_jwt", lambda token, key: claims)
    return setze


def test_mensch_fremder_mandant_abgewiesen(mensch):
    mensch([{"sys": "content", "role": "editor", "tid": "edera"}])
    assert _post("/business/") == 403
    assert _post("/storage/") == 403


def test_mensch_ohne_rollen_abgewiesen(mensch):
    mensch([])
    assert _post("/business/") == 403


def test_mensch_mit_betreiberrolle_durch(mensch):
    mensch([{"sys": "engineering", "role": "developer", "tid": "arkturian"}])
    assert _post("/business/") == 200


def test_mensch_oneal_nur_oneal(mensch):
    mensch([{"sys": "x", "role": "y", "tid": "oneal"}])
    assert _post("/oneal/") == 200
    assert _post("/business/") == 403


def test_superadmin_durch(mensch):
    mensch([], superadmin=True)
    assert _post("/business/") == 200
