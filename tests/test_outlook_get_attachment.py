"""Davids Issue #31: outlook_get_attachment im comm-MCP — Zugehoerigkeit
Nachricht<->Anhang, Ausgabe als Storage-Objekt/Datei statt Base64 im Kontext."""
import asyncio
import inspect
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
for _k in ("ARKTURIAN_API_KEY", "ONEAL_STORAGE_API_KEY", "JWT_PUBLIC_KEY", "COMM_API_KEY"):
    os.environ.setdefault(_k, "test-only")

_MSG = {"message": {"attachments": [
    {"attachment_id": "A1", "filename": "../rechnung.pdf", "mime_type": "application/pdf", "size": 3},
]}}
_ATT = {"attachment_id": "A1", "size": 3, "data": "YWJj"}  # "abc"


def _fake_comm(calls):
    async def _call(method, endpoint, **kw):
        calls.append(endpoint)
        if endpoint.endswith("/attachments/A1"):
            return _ATT
        if endpoint.endswith("/messages/M1"):
            return _MSG
        raise AssertionError(endpoint)
    return _call


def test_registered_and_defaults_to_storage():
    import server
    names = {t.name for t in server.comm_mcp._tool_manager.list_tools()}
    assert "outlook_get_attachment" in names
    sig = inspect.signature(server.comm_outlook_get_attachment)
    assert sig.parameters["deliver"].default == "storage"


def test_foreign_attachment_id_is_refused(monkeypatch):
    import server
    calls = []
    monkeypatch.setattr(server, "call_comm_api", _fake_comm(calls))
    r = asyncio.run(server.comm_outlook_get_attachment("steiner", "M1", "A9"))
    assert r["error"] == "attachment_not_in_message"
    assert not any(e.endswith("/attachments/A9") for e in calls)  # Bytes nie geholt


def test_storage_delivery_keeps_base64_out_of_context(monkeypatch):
    import server
    calls, uploads = [], []
    monkeypatch.setattr(server, "call_comm_api", _fake_comm(calls))

    async def _upload(file_bytes, filename, *, form_fields=None):
        uploads.append((file_bytes, filename, form_fields))
        return {"id": 77, "file_url": "https://s/x"}
    monkeypatch.setattr(server, "call_storage_upload", _upload)
    r = asyncio.run(server.comm_outlook_get_attachment("steiner", "M1", "A1"))
    assert "data" not in r
    assert r["storage"]["id"] == 77 and r["storage"]["media_url"].endswith("/storage/media/77")
    assert uploads[0][0] == b"abc" and uploads[0][1] == "rechnung.pdf"  # Pfadanteil entfernt
    assert uploads[0][2]["ai_mode"] == "none" and uploads[0][2]["is_public"] == "false"


def test_file_delivery_writes_into_directory(tmp_path, monkeypatch):
    import server
    monkeypatch.setattr(server, "call_comm_api", _fake_comm([]))
    r = asyncio.run(server.comm_outlook_get_attachment("steiner", "M1", "A1", deliver="file", save_to=str(tmp_path)))
    assert Path(r["path"]).read_bytes() == b"abc" and r["path"].endswith("rechnung.pdf")
    bad = asyncio.run(server.comm_outlook_get_attachment("steiner", "M1", "A1", deliver="file", save_to="relative"))
    assert bad["error"] == "save_to_required"
