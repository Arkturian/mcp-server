"""#3234: verschluesselte Objekte im storage-MCP (Vertrag storage-api #3230). Storage und cloud-api
sind Attrappen; belegt werden: encrypted erzwingt privat/ohne Ablauf/ohne KI, der Name bleibt aus
der URL, ohne Bestaetigung wird verworfen, Anmeldung bei cloud-api, echter Name aus dem signierten
Abruf, Hinweis im Ergebnis, assets_fetch lehnt ab."""
import asyncio
import json
import logging
import os
import sys
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
for _k in ("ARKTURIAN_API_KEY", "ONEAL_STORAGE_API_KEY", "JWT_PUBLIC_KEY", "COMM_API_KEY"):
    os.environ.setdefault(_k, "test-only")

GEHEIM = "Gehaltsliste Martina 2026.pdf"
PNG = b"\x89PNG\r\n\x1a\n" + b"x" * 64


@pytest.fixture
def als_agent():
    import auth
    t = auth._current_agent_name.set("Business"), auth._current_is_agent.set(True)
    yield
    auth._current_agent_name.reset(t[0])
    auth._current_is_agent.reset(t[1])


@pytest.fixture
def welt(monkeypatch, caplog):
    import server
    base = httpx.URL(server.STORAGE_API_BASE)
    w = {"json": [], "http": [], "antwort": {"id": 55, "encrypted": True, "is_public": False},
         "meta": {"encrypted": True, "mime_type": "application/pdf", "original_filename": "encrypted-55"},
         "registriert": True, "base": base, "item_shared_by": None}

    async def _fake_json(method, url, **kw):
        w["json"].append((method, url, kw))
        if "/api/agents/self/media-urls" in url:
            return {"expires_at": "t", "items": [{"id": 55, "url": f"https://{base.host}/storage/media/55?exp=1&kid=k&sig=s",
                                                  "shared_by": w["item_shared_by"]}]}
        if "/api/agents/self/encrypted-objects" in url:
            return {"id": 55, "registriert": w["registriert"]}
        if url.endswith("/storage/objects/55") and method == "GET":
            return w["meta"]
        if url.endswith("/storage/objects/55") and method == "DELETE":
            return {}
        if "/storage/upload-ticket" in url:
            return {"upload_url": "https://x/put/abc", "expires_in": 900}
        raise AssertionError((method, url))

    def _handler(request):
        w["http"].append(request)
        if request.url.path == "/storage/upload":
            return httpx.Response(200, json=w["antwort"])
        cd = "attachment; filename*=UTF-8''" + "Gehaltsliste%20Martina%202026.pdf"
        if request.method == "HEAD":
            return httpx.Response(200, headers={"content-disposition": cd, "content-length": "12345"})
        return httpx.Response(200, content=PNG, headers={"content-type": "image/png",
                                                         "content-disposition": cd.replace(".pdf", ".png")})

    echt = httpx.AsyncClient
    monkeypatch.setattr(server, "_fetch_json", _fake_json)
    monkeypatch.setattr(server.httpx, "AsyncClient", lambda **kw: echt(transport=httpx.MockTransport(_handler), **kw))
    caplog.set_level(logging.DEBUG)
    w["caplog"], w["server"], w["echt"] = caplog, server, echt
    return w


def _felder(req):
    roh = req.content.decode("latin-1")
    import re
    return dict(re.findall(r'name="([^"]+)"\r\n\r\n([^\r]*)', roh))


def test_upload_verschluesselt_privat_ohne_ablauf_ohne_ki(welt, als_agent):
    import base64
    s = welt["server"]
    res = asyncio.run(s.storage_assets_upload(file_base64=base64.b64encode(b"%PDF-1").decode(), filename=GEHEIM,
                                              is_public=True, ai_mode="full", ttl_hours=24, encrypted=True))
    (req,) = welt["http"]
    f = _felder(req)
    assert f["encrypted"] == "true" and f["private"] == "true" and f["is_public"] == "false"
    assert f["ai_mode"] == "none" and f["reuse_existing"] == "false" and "ttl_hours" not in f
    assert GEHEIM not in str(req.url)
    assert res["encrypted"] is True and res["mitteilungen"] is True and "5 Minuten" in res["hinweis"]
    assert any("/api/agents/self/encrypted-objects" in u for _, u, _ in welt["json"])


def test_ohne_bestaetigung_wird_verworfen(welt, als_agent):
    import base64
    s = welt["server"]
    welt["antwort"] = {"id": 55, "is_public": False}          # alte Storage ohne encrypted
    with pytest.raises(RuntimeError, match="encryption_not_confirmed"):
        asyncio.run(s.storage_assets_upload(file_base64=base64.b64encode(b"x").decode(), filename=GEHEIM,
                                            encrypted=True))
    assert ("DELETE", f"{s.STORAGE_API_BASE}/storage/objects/55") in [(m, u) for m, u, _ in welt["json"]]


def test_fehler_ohne_name_im_log(welt, als_agent, monkeypatch):
    import base64
    s = welt["server"]

    def _handler(request):
        return httpx.Response(503, json={"detail": {"code": "encryption_key_unavailable", "name": GEHEIM}})
    echt = welt["echt"]
    monkeypatch.setattr(s.httpx, "AsyncClient", lambda **kw: echt(transport=httpx.MockTransport(_handler), **kw))
    with pytest.raises(httpx.HTTPStatusError):
        asyncio.run(s.storage_assets_upload(file_base64=base64.b64encode(b"x").decode(), filename=GEHEIM,
                                            encrypted=True))
    assert GEHEIM not in welt["caplog"].text and "503" in welt["caplog"].text


def test_ticket_name_nur_im_kopf(welt, als_agent):
    s = welt["server"]
    res = asyncio.run(s.storage_assets_upload_ticket(filename=GEHEIM, ttl_hours=5, ai_mode="vision",
                                                     is_public=True, encrypted=True))
    (m, url, kw) = [c for c in welt["json"] if "/storage/upload-ticket" in c[1]][0]
    assert "filename" not in (kw.get("params") or {}) and kw["params"]["encrypted"] == "true"
    assert kw["params"]["private"] == "true" and kw["params"]["ai_mode"] == "none" and "ttl_hours" not in kw["params"]
    assert kw["headers"]["X-Filename"] == "Gehaltsliste%20Martina%202026.pdf"
    assert "hinweis" in res
    # Derselbe Weg ueber die TICKET-Hintertuer von assets_upload; CHUNK geht nicht.
    welt["json"].clear()
    asyncio.run(s.storage_assets_upload(file_base64="TICKET", filename=GEHEIM, encrypted=True))
    assert any("/storage/upload-ticket" in u for _, u, _ in welt["json"])
    with pytest.raises(ValueError, match="stückweise"):
        asyncio.run(s.storage_assets_upload(file_base64="CHUNK:u:0:1:eA==", filename=GEHEIM, encrypted=True))


def test_fetch_verschluesselt_wird_abgelehnt(welt, als_agent):
    with pytest.raises(ValueError, match="encrypted_fetch_unsupported"):
        asyncio.run(welt["server"].storage_assets_fetch(url="https://example.org/a.pdf", encrypted=True))
    assert welt["json"] == []


def test_download_nimmt_den_echten_namen_aus_dem_signierten_link(welt, als_agent):
    res = asyncio.run(welt["server"].storage_assets_download(id=55))
    assert res["filename"] == GEHEIM and res["size"] == 12345 and res["encrypted"] is True
    assert "sha256" not in res and "$HOME/tmp" in res["hinweis"] and res["mitteilungen"] is True
    (head,) = welt["http"]
    assert head.method == "HEAD" and "x-api-key" not in {k.lower() for k in head.headers}
    # Geteiltes Objekt: kein Anmelden (Besitzer ist der Teilende).
    welt["item_shared_by"] = "Bob"
    welt["json"].clear()
    asyncio.run(welt["server"].storage_assets_download(id=55))
    assert not any("/encrypted-objects" in u for _, u, _ in welt["json"])


def test_media_view_verschluesselt_ohne_umrechnung(welt, als_agent):
    from mcp.server.fastmcp import Image
    welt["meta"] = {"encrypted": True, "mime_type": "image/png", "original_filename": "encrypted-55"}
    r = asyncio.run(welt["server"].storage_media_view(id=55, max_px=400))
    assert isinstance(r[0], Image) and r[0].data == PNG
    (get,) = welt["http"]
    assert "width" not in str(get.url) and "format" not in str(get.url) and "sig=s" in str(get.url)
    assert "Gehaltsliste Martina 2026.png" in r[1] and "verschlüsselt" in r[1]
    welt["meta"] = {"encrypted": True, "mime_type": "application/pdf"}
    assert asyncio.run(welt["server"].storage_media_view(id=55))["error"] == "encrypted_no_preview"


def test_content_disposition():
    from server import _cd_name
    assert _cd_name("attachment; filename*=UTF-8''Zo%C3%AB%20x.pdf") == "Zoë x.pdf"
    assert _cd_name('inline; filename="a b.png"') == "a b.png"
    assert _cd_name("inline") is None and _cd_name(None) is None
