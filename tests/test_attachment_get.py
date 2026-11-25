"""attachment_get: WhatsApp-/Discord-Anhang über comm öffnen (r-cubic admin, 25.11.).

comm prüft die Zustellung an den Aufrufer und signiert; das Gateway holt nur den
signierten Link, ohne Dienstschlüssel und ohne JWT. Läuft nicht über das storage-MCP.
"""
import asyncio
import base64
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
    t = auth._current_agent_name.set("r-cubic admin"), auth._current_is_agent.set(True)
    yield
    auth._current_agent_name.reset(t[0])
    auth._current_is_agent.reset(t[1])


def _setup(monkeypatch, *, info=None, antwort=None, comm_fehler=None):
    import server
    base = httpx.URL(server.STORAGE_API_BASE)
    if info is None:
        info = {"url": f"https://{base.host}/storage/media/9?exp=1&kid=k&sig=s&scope=x",
                "ttl_seconds": 900, "mime_type": "image/jpeg", "size_bytes": 104, "filename": "wa.jpg"}
    calls, geholt = [], []

    async def _fake_comm(method, endpoint, **kw):
        calls.append((method, endpoint, kw))
        if comm_fehler:
            raise comm_fehler
        return info

    def _handler(request):
        geholt.append(request)
        return antwort or httpx.Response(200, content=JPEG, headers={"content-type": "image/jpeg"})

    echt = httpx.AsyncClient
    monkeypatch.setattr(server, "call_comm_api", _fake_comm)
    monkeypatch.setattr(server.httpx, "AsyncClient",
                        lambda **kw: echt(transport=httpx.MockTransport(_handler), **kw))
    return server, calls, geholt


def test_bild_kommt_als_bildinhalt(monkeypatch, als_agent):
    from mcp.server.fastmcp import Image
    server, calls, geholt = _setup(monkeypatch)
    r = asyncio.run(server.comm_attachment_get("att_AbC-1", max_px=1200))
    assert isinstance(r[0], Image) and r[0].data == JPEG and "wa.jpg" in r[1]
    assert calls == [("GET", "/api/v1/attachments/att_AbC-1", {"params": {"mode": "signed_url"}})]
    q = dict(geholt[0].url.params)
    assert q["sig"] == "s" and q["format"] == "jpg" and q["height"] == "1200"
    kopf = {k.lower() for k in geholt[0].headers}
    assert "x-api-key" not in kopf and "authorization" not in kopf


def test_andere_datei_als_base64(monkeypatch, als_agent):
    import server as s
    base = httpx.URL(s.STORAGE_API_BASE)
    info = {"url": f"https://{base.host}/storage/media/9?sig=s", "mime_type": "application/zip",
            "size_bytes": 3, "filename": "a.zip"}
    server, _, geholt = _setup(monkeypatch, info=info,
                               antwort=httpx.Response(200, content=b"abc", headers={"content-type": "application/zip"}))
    r = asyncio.run(server.comm_attachment_get("a1"))
    assert r["filename"] == "a.zip" and base64.b64decode(r["data"]) == b"abc"
    assert "format" not in dict(geholt[0].url.params)


def test_zu_grosse_datei(monkeypatch, als_agent):
    import server as s
    base = httpx.URL(s.STORAGE_API_BASE)
    info = {"url": f"https://{base.host}/storage/media/9?sig=s", "mime_type": "application/zip"}
    gross = b"x" * (s._ANHANG_DATEI_MAX + 1)
    server, _, _ = _setup(monkeypatch, info=info,
                          antwort=httpx.Response(200, content=gross, headers={"content-type": "application/zip"}))
    assert asyncio.run(server.comm_attachment_get("a1"))["error"] == "too_large"


def test_fremder_host_wird_nicht_geholt(monkeypatch, als_agent):
    server, _, geholt = _setup(monkeypatch, info={"url": "https://evil.example/storage/media/9", "mime_type": "image/jpeg"})
    assert asyncio.run(server.comm_attachment_get("a1"))["error"] == "unexpected_media_url"
    assert not geholt


def test_kein_storage_pfad_wird_nicht_geholt(monkeypatch, als_agent):
    import server as s
    base = httpx.URL(s.STORAGE_API_BASE)
    server, _, geholt = _setup(monkeypatch, info={"url": f"https://{base.host}/storage/other/9", "mime_type": "image/jpeg"})
    assert asyncio.run(server.comm_attachment_get("a1"))["error"] == "unexpected_media_url"
    assert not geholt


def test_ungueltige_id_ruft_comm_nicht(monkeypatch, als_agent):
    server, calls, _ = _setup(monkeypatch)
    for bad in ("../x", "a/b", "", "a b"):
        assert asyncio.run(server.comm_attachment_get(bad))["error"] == "invalid_attachment_id"
    assert not calls


def test_nicht_zugestellt_kommt_von_comm(monkeypatch, als_agent):
    fehler = RuntimeError("404 attachment_not_found_or_not_for_caller")
    server, _, geholt = _setup(monkeypatch, comm_fehler=fehler)
    with pytest.raises(RuntimeError, match="not_for_caller"):
        asyncio.run(server.comm_attachment_get("a1"))
    assert not geholt


def test_pdf_ohne_vorschau(monkeypatch, als_agent):
    import server as s
    base = httpx.URL(s.STORAGE_API_BASE)
    info = {"url": f"https://{base.host}/storage/media/9?sig=s", "mime_type": "application/pdf"}
    server, _, _ = _setup(monkeypatch, info=info,
                          antwort=httpx.Response(200, content=b"%PDF", headers={"content-type": "application/pdf"}))
    assert asyncio.run(server.comm_attachment_get("a1"))["error"] == "pdf_preview_unavailable"


def test_nur_fuer_agenten(monkeypatch):
    server, calls, _ = _setup(monkeypatch)
    with pytest.raises(Exception):
        asyncio.run(server.comm_attachment_get("a1"))
    assert not calls
