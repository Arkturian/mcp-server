"""Post 5721: assets_share / assets_download reichen an cloud-api durch.

Die Identität kommt aus dem JWT, das cloud-api prüft; das MCP schickt keinen
Namen mit. Metadaten holt assets_download erst nach erfolgreicher Signatur.
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


def _fake(calls, antworten):
    async def _f(method, url, **kw):
        calls.append((method, url, kw.get("json_body")))
        for teil, antwort in antworten.items():
            if teil in url:
                return antwort
        raise AssertionError(f"unerwartet: {url}")
    return _f


def test_share_reicht_durch(monkeypatch, als_agent):
    import server
    calls = []
    monkeypatch.setattr(server, "_fetch_json", _fake(calls, {"/api/agents/self/media-share": {"ok": True}}))
    r = asyncio.run(server.storage_assets_share(id=132599, share_with=["Business"]))
    assert r == {"ok": True}
    m, url, body = calls[0]
    assert m == "POST" and url.endswith("/api/agents/self/media-share")
    assert body == {"object_id": 132599, "share_with": ["Business"]}


def test_download_signiert_dann_metadaten(monkeypatch, als_agent):
    import server
    calls = []
    monkeypatch.setattr(server, "_fetch_json", _fake(calls, {
        "/api/agents/self/media-urls": {"expires_at": "t", "items": [{"id": 7, "url": "https://s/m/7?sig=x"}]},
        "/storage/objects/7": {"original_filename": "v.pdf", "mime_type": "application/pdf",
                               "file_size_bytes": 3, "checksum": "ab"},
    }))
    r = asyncio.run(server.storage_assets_download(id=7))
    assert r == {"id": 7, "url": "https://s/m/7?sig=x", "expires_at": "t", "filename": "v.pdf",
                 "mime_type": "application/pdf", "size": 3, "sha256": "ab"}
    assert calls[0][1].endswith("/api/agents/self/media-urls") and calls[0][2] == {"ids": [7]}


def test_download_ohne_bindung_verraet_keine_metadaten(monkeypatch, als_agent):
    import server
    calls = []
    monkeypatch.setattr(server, "_fetch_json", _fake(calls, {
        "/api/agents/self/media-urls": {"items": [{"id": 7, "error": "not_granted"}]}}))
    r = asyncio.run(server.storage_assets_download(id=7))
    assert r == {"id": 7, "error": "not_granted"}
    assert not any("/storage/objects/" in u for _, u, _ in calls)


@pytest.mark.parametrize("werkzeug", ["share", "download"])
def test_ohne_agenten_identitaet_abgelehnt(monkeypatch, werkzeug):
    import server
    calls = []
    monkeypatch.setattr(server, "_fetch_json", _fake(calls, {}))
    with pytest.raises(ValueError, match="requires_agent"):
        if werkzeug == "share":
            asyncio.run(server.storage_assets_share(id=1, share_with=["X"]))
        else:
            asyncio.run(server.storage_assets_download(id=1))
    assert calls == []
