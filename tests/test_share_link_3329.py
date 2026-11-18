"""#3329: assets_share_link / assets_share_links / assets_unshare_link reichen an cloud-api durch.

Die Identitaet (und damit der Storage-Scope) kommt aus dem JWT, das cloud-api prueft; das MCP
schickt keinen Namen und keinen Scope mit. Nur Agenten duerfen die Werkzeuge rufen.
"""
import asyncio
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
for _k in ("ARKTURIAN_API_KEY", "ONEAL_STORAGE_API_KEY", "JWT_PUBLIC_KEY", "COMM_API_KEY"):
    os.environ.setdefault(_k, "test-only")


@pytest.fixture
def als_agent():
    import auth
    t = auth._current_agent_name.set("Business"), auth._current_is_agent.set(True)
    yield
    auth._current_agent_name.reset(t[0])
    auth._current_is_agent.reset(t[1])


def _fake(calls, antwort):
    async def _f(method, url, **kw):
        calls.append({"method": method, "url": url, "json": kw.get("json_body"), "params": kw.get("params")})
        return antwort
    return _f


def test_link_anlegen_reicht_durch_ohne_scope(monkeypatch, als_agent):
    import server
    calls = []
    monkeypatch.setattr(server, "_fetch_json", _fake(calls, {"id": 41, "url": "https://files.test/s/x/a.pdf"}))
    r = asyncio.run(server.storage_assets_share_link(id=7))
    assert r["id"] == 41
    [c] = calls
    assert c["method"] == "POST" and c["url"].endswith("/api/agents/self/share-link")
    assert c["json"] == {"object_id": 7}                       # kein scope, kein Name
    asyncio.run(server.storage_assets_share_link(id=7, gueltig_bis="2026-12-01T12:00:00Z"))
    assert calls[-1]["json"] == {"object_id": 7, "gueltig_bis": "2026-12-01T12:00:00Z"}


def test_liste_und_widerruf(monkeypatch, als_agent):
    import server
    calls = []
    monkeypatch.setattr(server, "_fetch_json", _fake(calls, {"ok": True}))
    asyncio.run(server.storage_assets_share_links(id=7))
    assert calls[-1]["method"] == "GET" and calls[-1]["url"].endswith("/api/agents/self/share-links")
    assert calls[-1]["params"] == {"object_id": 7}
    asyncio.run(server.storage_assets_unshare_link(id=7, link_id=41))
    assert calls[-1]["method"] == "POST" and calls[-1]["url"].endswith("/api/agents/self/share-link-revoke")
    assert calls[-1]["json"] == {"object_id": 7, "link_id": 41}


@pytest.mark.parametrize("werkzeug,args", [("storage_assets_share_link", {"id": 7}),
                                            ("storage_assets_share_links", {"id": 7}),
                                            ("storage_assets_unshare_link", {"id": 7, "link_id": 1})])
def test_nur_agenten(monkeypatch, werkzeug, args):
    import server
    calls = []
    monkeypatch.setattr(server, "_fetch_json", _fake(calls, {}))
    with pytest.raises(ValueError, match="requires_agent"):
        asyncio.run(getattr(server, werkzeug)(**args))
    assert calls == []


def test_beschreibungen_nennen_die_regel():
    import server
    tools = {t.name: t for t in asyncio.run(server.storage_mcp.list_tools())}
    d = tools["assets_share_link"].description
    assert "stays private" in d and "30 days" in d and "only" in d and "once" in d
    assert "assets_unshare_link" in d and "assets_share_links" in d
    assert "Never make a file public just to share it" in tools["assets_get"].description
    assert "readable link to hand to people" not in tools["assets_get"].description
