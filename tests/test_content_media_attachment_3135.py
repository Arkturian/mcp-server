"""media_add mit role='attachment' holt die Angaben vom Storage-Objekt (#3135)."""
import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
for _k in ("ARKTURIAN_API_KEY", "ONEAL_STORAGE_API_KEY", "JWT_PUBLIC_KEY", "COMM_API_KEY"):
    os.environ.setdefault(_k, "test-only")

SHA = "c" * 64


def _wire(monkeypatch, storage_obj):
    import server
    content_calls, storage_calls = [], []

    async def fake_content(method, path, **kw):
        content_calls.append((method, path, kw.get("json_body")))
        return {"id": 1}

    async def fake_storage(method, path, **kw):
        storage_calls.append((method, path))
        return storage_obj

    monkeypatch.setattr(server, "call_content_api", fake_content)
    monkeypatch.setattr(server, "call_storage_api", fake_storage)
    return server, content_calls, storage_calls


def test_attachment_takes_metadata_from_storage(monkeypatch):
    server, cc, sc = _wire(monkeypatch, {"original_filename": "Rechnung.pdf",
                                         "mime_type": "application/pdf",
                                         "checksum": SHA, "file_size_bytes": 4711})
    asyncio.run(server.content_media_add(post_id=7, storage_id=55, role="attachment"))
    assert sc == [("GET", "/storage/objects/55")]
    body = cc[0][2]
    assert body["filename"] == "Rechnung.pdf" and body["content_type"] == "application/pdf"
    assert body["sha256"] == SHA and body["size_bytes"] == 4711 and body["role"] == "attachment"


def test_attachment_without_checksum_is_not_sent(monkeypatch):
    server, cc, _ = _wire(monkeypatch, {"original_filename": "a.pdf", "mime_type": "application/pdf"})
    r = asyncio.run(server.content_media_add(post_id=7, storage_id=55, role="attachment"))
    assert r["error"] == "attachment_storage_metadata_missing" and r["missing"] == ["sha256"]
    assert cc == []


def test_gallery_does_not_ask_storage(monkeypatch):
    server, cc, sc = _wire(monkeypatch, {})
    asyncio.run(server.content_media_add(post_id=7, storage_id=55, role="gallery"))
    assert sc == [] and "sha256" not in cc[0][2]
