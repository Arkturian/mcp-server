"""media_view: das Bild selbst als MCP-Bildinhalt, gleiche Berechtigung wie assets_download.

Geholt wird nur der signierte Link, ohne Dienstschlüssel (Post 5782).
"""
import asyncio
import os
import sys
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
for _k in ("ARKTURIAN_API_KEY", "ONEAL_STORAGE_API_KEY", "JWT_PUBLIC_KEY", "COMM_API_KEY"):
    os.environ.setdefault(_k, "test-only")

JPEG = b"\xff\xd8\xff\xe0" + b"x" * 100


@pytest.fixture
def als_agent():
    import auth
    t = auth._current_agent_name.set("Business"), auth._current_is_agent.set(True)
    yield
    auth._current_agent_name.reset(t[0])
    auth._current_is_agent.reset(t[1])


def _setup(monkeypatch, *, meta, item=None, antwort=None):
    import server
    base = httpx.URL(server.STORAGE_API_BASE)
    item = item if item is not None else {"id": 7, "url": f"https://{base.host}/storage/media/7?exp=1&kid=k&sig=s"}
    calls, geholt = [], []

    async def _fake_json(method, url, **kw):
        calls.append((method, url, kw))
        if "/api/agents/self/media-urls" in url:
            return {"expires_at": "t", "items": [item]}
        if "/storage/objects/7" in url:
            return meta
        raise AssertionError(url)

    def _handler(request):
        geholt.append(request)
        return antwort or httpx.Response(200, content=JPEG, headers={"content-type": "image/jpeg"})

    echt = httpx.AsyncClient
    monkeypatch.setattr(server, "_fetch_json", _fake_json)
    monkeypatch.setattr(server.httpx, "AsyncClient",
                        lambda **kw: echt(transport=httpx.MockTransport(_handler), **kw))
    return server, calls, geholt


def test_bild_kommt_als_bildinhalt(monkeypatch, als_agent):
    from mcp.server.fastmcp import Image
    server, calls, geholt = _setup(monkeypatch, meta={
        "mime_type": "image/png", "width": 2160, "height": 3840, "original_filename": "s.png"})
    r = asyncio.run(server.storage_media_view(id=7, max_px=1600))
    assert isinstance(r[0], Image) and r[0].data == JPEG
    assert "s.png" in r[1]
    q = dict(geholt[0].url.params)
    assert q["height"] == "1600" and "width" not in q and q["format"] == "jpg" and q["sig"] == "s"
    assert "x-api-key" not in {k.lower() for k in geholt[0].headers}


def test_kleines_bild_wird_nicht_hochskaliert(monkeypatch, als_agent):
    server, _, geholt = _setup(monkeypatch, meta={"mime_type": "image/jpeg", "width": 800, "height": 600})
    asyncio.run(server.storage_media_view(id=7))
    q = dict(geholt[0].url.params)
    assert "width" not in q and "height" not in q


def test_querformat_begrenzt_breite(monkeypatch, als_agent):
    server, _, geholt = _setup(monkeypatch, meta={"mime_type": "video/mp4", "width": 3840, "height": 2160})
    asyncio.run(server.storage_media_view(id=7, max_px=5000))
    assert dict(geholt[0].url.params)["width"] == "2000"


def test_ohne_bindung_nichts_geholt(monkeypatch, als_agent):
    server, calls, geholt = _setup(monkeypatch, meta={}, item={"id": 7, "error": "not_granted"})
    r = asyncio.run(server.storage_media_view(id=7))
    assert r == {"id": 7, "error": "not_granted"}
    assert not geholt and not any("/storage/objects/" in c[1] for c in calls)


def test_fremder_host_wird_nicht_geholt(monkeypatch, als_agent):
    server, calls, geholt = _setup(monkeypatch, meta={"mime_type": "image/png"},
                                   item={"id": 7, "url": "https://evil.example/storage/media/7"})
    assert asyncio.run(server.storage_media_view(id=7)) == {"id": 7, "error": "unexpected_media_url"}
    assert not geholt


def test_audio_nicht_unterstuetzt(monkeypatch, als_agent):
    server, _, geholt = _setup(monkeypatch, meta={"mime_type": "audio/mpeg"})
    r = asyncio.run(server.storage_media_view(id=7))
    assert r["error"] == "unsupported_type" and not geholt


def test_pdf_ohne_renderer(monkeypatch, als_agent):
    server, _, _ = _setup(monkeypatch, meta={"mime_type": "application/pdf"},
                          antwort=httpx.Response(200, content=b"%PDF-1.4", headers={"content-type": "application/pdf"}))
    r = asyncio.run(server.storage_media_view(id=7))
    assert r["error"] == "pdf_preview_unavailable"


def test_zu_gross(monkeypatch, als_agent):
    server, _, _ = _setup(monkeypatch, meta={"mime_type": "image/png"},
                          antwort=httpx.Response(200, content=b"x" * (server_max() + 1),
                                                 headers={"content-type": "image/jpeg"}))
    assert asyncio.run(server.storage_media_view(id=7))["error"] == "too_large"


def server_max():
    import server
    return server._MEDIA_VIEW_MAX_BYTES


def test_ohne_agent_abgelehnt(monkeypatch):
    server, _, _ = _setup(monkeypatch, meta={})
    with pytest.raises(ValueError, match="media_view_requires_agent"):
        asyncio.run(server.storage_media_view(id=7))


def test_mcp_liefert_image_content(monkeypatch, als_agent):
    server, _, _ = _setup(monkeypatch, meta={"mime_type": "image/png", "width": 10, "height": 10})
    out = asyncio.run(server.storage_mcp.call_tool("media_view", {"id": 7}))
    inhalte = out[0] if isinstance(out, tuple) else out
    arten = [c.type for c in inhalte]
    assert arten == ["image", "text"]
    assert inhalte[0].mimeType == "image/jpeg"
