"""#3030 (Befund 105): storage calls of an agent carry X-Grant-Scope-Restrict
from the verified identity; non-public agent uploads are private_media."""
import asyncio
import base64
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


def _capture(calls, answer=None):
    async def _f(method, url, **kw):
        calls.append((method, url, kw))
        return answer if answer is not None else {"ok": True}
    return _f


@pytest.mark.parametrize("call", [
    lambda s: s.storage_assets_get(id=5),
    lambda s: s.storage_assets_list(),
    lambda s: s.storage_assets_similar(id=5),
    lambda s: s.storage_kg_search(query="x"),
    lambda s: s.storage_assets_get_embedding_text(id=5),
    lambda s: s.storage_kg_embed(id=5),
    lambda s: s.storage_assets_refs(),
])
def test_agent_reads_are_restricted(monkeypatch, als_agent, call):
    import server
    calls = []
    monkeypatch.setattr(server, "_fetch_json", _capture(calls))
    asyncio.run(call(server))
    assert calls, "no storage call made"
    for _, url, kw in calls:
        if url.startswith(server.STORAGE_API_BASE):
            assert kw["headers"]["X-Grant-Scope-Restrict"] == "cloud-session:Business"


def test_non_agent_has_no_restriction(monkeypatch):
    import server
    calls = []
    monkeypatch.setattr(server, "_fetch_json", _capture(calls))
    asyncio.run(server.storage_assets_get(id=5))
    assert "X-Grant-Scope-Restrict" not in calls[0][2]["headers"]


def test_scope_not_from_arguments(monkeypatch, als_agent):
    import server
    calls = []
    monkeypatch.setattr(server, "_fetch_json", _capture(calls))
    asyncio.run(server.storage_assets_list(context="cloud-session:Other"))
    assert calls[0][2]["headers"]["X-Grant-Scope-Restrict"] == "cloud-session:Business"


def test_fetch_private_for_agent(monkeypatch, als_agent):
    import server
    calls = []
    monkeypatch.setattr(server, "_fetch_json", _capture(calls))
    asyncio.run(server.storage_assets_fetch(url="https://x.example/a.png", analyze=False))
    body = calls[0][2]["json_body"]
    assert body["private"] is True and body["grant_scope"] == "cloud-session:Business"
    calls.clear()
    asyncio.run(server.storage_assets_fetch(url="https://x.example/a.png", is_public=True, analyze=False))
    assert "private" not in calls[0][2]["json_body"]


def test_upload_ticket_path_private_for_agent(monkeypatch, als_agent):
    import server
    calls = []
    monkeypatch.setattr(server, "_fetch_json", _capture(calls, {"upload_url": "u"}))
    asyncio.run(server.storage_assets_upload(file_base64="TICKET", filename="a.png"))
    assert calls[0][2]["params"]["private"] == "true"


def test_upload_chunk_path_private_for_agent(monkeypatch, als_agent):
    import server
    calls = []
    monkeypatch.setattr(server, "_fetch_json", _capture(calls))
    payload = base64.b64encode(b"x").decode()
    asyncio.run(server.storage_assets_upload(file_base64=f"CHUNK:u1:0:1:{payload}", filename="a.png"))
    assert calls[0][2]["params"]["private"] == "true"
    assert calls[0][2]["headers"]["X-Grant-Scope-Restrict"] == "cloud-session:Business"


def test_upload_base64_path_private_for_agent(monkeypatch, als_agent):
    import server
    seen = {}

    async def fake_upload(file_bytes, filename, *, form_fields=None, share_scopes=None):
        seen.update(form_fields or {})
        return {"id": 1}
    monkeypatch.setattr(server, "call_storage_upload", fake_upload)
    asyncio.run(server.storage_assets_upload(file_base64=base64.b64encode(b"x").decode(), filename="a.png", ai_mode="none"))
    assert seen["private"] == "true"
