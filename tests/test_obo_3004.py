"""#3004: on-behalf-of-Tausch fuer Menschen-Tokens an die Backends (RFC 8693).

Vertrag: auth-api ead0b2c, Issue #3004 Kommentar 8430. Nichts Echtes: auth-api ist
eine Vorrichtung, kein Netz.
"""
import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
for _k in ("ARKTURIAN_API_KEY", "ONEAL_STORAGE_API_KEY", "JWT_PUBLIC_KEY", "COMM_API_KEY"):
    os.environ.setdefault(_k, "test-only")

import httpx  # noqa: E402
import pytest  # noqa: E402

import auth  # noqa: E402
import obo  # noqa: E402
import server  # noqa: E402


class _AuthApi:
    """Vorrichtung fuer POST /api/v1/auth/oauth/token."""

    def __init__(self):
        self.aufrufe = []
        self.tausch_status = 200
        self.tausch_fehler = "invalid_scope"

    def antwort(self, daten):
        self.aufrufe.append(dict(daten))
        if daten.get("grant_type") == "client_credentials":
            return 200, {"access_token": "dienst-jwt", "expires_in": 3600}
        if self.tausch_status != 200:
            return self.tausch_status, {"error": self.tausch_fehler}
        assert daten["actor_token"] == "dienst-jwt" and daten["subject_token"]
        return 200, {"access_token": f"delegiert:{daten['audience']}:{len(self.aufrufe)}",
                     "issued_token_type": obo.AKTEUR_TYP, "token_type": "Bearer", "expires_in": 300}


@pytest.fixture
def welt(monkeypatch):
    obo._reset_for_tests()
    api = _AuthApi()

    class _Client:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, url, data=None, **k):
            assert url == "https://auth.test/api/v1/auth/oauth/token"
            status, js = api.antwort(data)
            return httpx.Response(status, json=js, request=httpx.Request("POST", url))
    monkeypatch.setattr(obo.httpx, "AsyncClient", _Client)
    monkeypatch.setenv("AUTH_API_BASE", "https://auth.test")
    monkeypatch.setenv("MCP_OBO_CLIENT_ID", "mcp-gateway")
    monkeypatch.setenv("MCP_OBO_CLIENT_SECRET", "geheim-test")
    monkeypatch.setenv("MCP_OBO", "1")
    monkeypatch.setenv("MCP_OBO_ZIELE", "cloud-api,content-api")
    yield api
    obo._reset_for_tests()


def _als(typ, token="roh-token"):
    return auth._current_jwt.set(token), auth._current_token_typ.set(typ)


def test_schalter_vorgabe_aus(monkeypatch):
    monkeypatch.delenv("MCP_OBO", raising=False)
    assert obo.aktiv("cloud-api") is False
    monkeypatch.setenv("MCP_OBO", "1")
    monkeypatch.setenv("MCP_OBO_ZIELE", "cloud-api")
    assert obo.aktiv("cloud-api") and not obo.aktiv("content-api")
    monkeypatch.setenv("MCP_OBO_ZIELE", "*")
    assert obo.aktiv("issue-api")


def test_tausch_je_ziel_mit_zwischenspeicher(welt):
    a = asyncio.run(obo.tausche("roh-token", "cloud-api"))
    b = asyncio.run(obo.tausche("roh-token", "cloud-api"))
    c = asyncio.run(obo.tausche("roh-token", "content-api"))
    assert a == b and a.startswith("delegiert:cloud-api") and c.startswith("delegiert:content-api")
    arten = [x["grant_type"] for x in welt.aufrufe]
    assert arten == ["client_credentials", obo.GRANT, obo.GRANT], "Dienst-JWT einmal, je Ziel ein Tausch"
    t = [x for x in welt.aufrufe if x["grant_type"] == obo.GRANT][0]
    assert t["subject_token_type"] == obo.SUBJECT_TYP and t["actor_token_type"] == obo.AKTEUR_TYP


def test_tausch_fehler_ohne_rueckfall(welt, monkeypatch):
    welt.tausch_status = 400
    with pytest.raises(obo.OboFehler) as e:
        asyncio.run(obo.tausche("roh-token", "cloud-api"))
    assert "invalid_scope" in str(e.value)
    monkeypatch.delenv("MCP_OBO_CLIENT_SECRET")
    obo._reset_for_tests()
    with pytest.raises(obo.OboFehler):
        asyncio.run(obo.tausche("roh-token", "cloud-api"))


def test_ziel_nach_basis_url():
    assert server._obo_ziel(f"{server.CLOUD_API_BASE}/api/agents/route") == "cloud-api"
    assert server._obo_ziel(f"{server.CONTENT_API_BASE}/api/v1/posts") == "content-api"
    assert server._obo_ziel("https://unbekannt.example/x") == ""


def test_mensch_wird_getauscht_agent_nicht(welt):
    url = f"{server.CLOUD_API_BASE}/api/agents/route"
    t1, t2 = _als("oauth_access")
    try:
        h = asyncio.run(server._obo_kopf(url, {"Authorization": "Bearer roh-token", "X-Foo": "1"}))
        assert h["Authorization"].startswith("Bearer delegiert:cloud-api") and h["X-Foo"] == "1"
        # Kein Aufrufer-Token im Kopf (Dienstschluessel): unveraendert.
        assert asyncio.run(server._obo_kopf(url, {"X-API-KEY": "k"})) == {"X-API-KEY": "k"}
        # Ziel nicht freigeschaltet: unveraendert.
        h2 = asyncio.run(server._obo_kopf(f"{server.ISSUE_API_BASE}/x", {"Authorization": "Bearer roh-token"}))
        assert h2 == {"Authorization": "Bearer roh-token"}
    finally:
        auth._current_jwt.reset(t1); auth._current_token_typ.reset(t2)
    t1, t2 = _als("agent")
    try:
        h = asyncio.run(server._obo_kopf(url, {"Authorization": "Bearer roh-token"}))
        assert h == {"Authorization": "Bearer roh-token"}, "Agenten-Token bleibt"
    finally:
        auth._current_jwt.reset(t1); auth._current_token_typ.reset(t2)


def test_gescheiterter_tausch_stoppt_den_aufruf(welt):
    welt.tausch_status = 400
    t1, t2 = _als("user")
    try:
        with pytest.raises(obo.OboFehler):
            asyncio.run(server._obo_kopf(f"{server.CLOUD_API_BASE}/api/x", {"Authorization": "Bearer roh-token"}))
    finally:
        auth._current_jwt.reset(t1); auth._current_token_typ.reset(t2)
