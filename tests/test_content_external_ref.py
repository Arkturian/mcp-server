"""posts_create/posts_update reichen external_ref an content-api durch (#2811)."""
import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
for _k in ("ARKTURIAN_API_KEY", "ONEAL_STORAGE_API_KEY", "JWT_PUBLIC_KEY", "COMM_API_KEY"):
    os.environ.setdefault(_k, "test-only")


def _capture(monkeypatch):
    import server
    calls = []

    async def fake(method, path, **kw):
        calls.append((method, path, kw.get("json_body")))
        return {"ok": True}

    monkeypatch.setattr(server, "call_content_api", fake)
    return server, calls


def test_create_passes_external_ref(monkeypatch):
    server, calls = _capture(monkeypatch)
    asyncio.run(server.content_posts_create(title="Notiz", content="x",
                                            external_ref={"path": "/docs/notiz.md"}))
    assert calls[0][0] == "POST" and calls[0][2]["external_ref"] == {"path": "/docs/notiz.md"}


def test_create_without_external_ref_sends_none(monkeypatch):
    server, calls = _capture(monkeypatch)
    asyncio.run(server.content_posts_create(title="Notiz"))
    assert "external_ref" not in calls[0][2]


def test_update_link_and_unlink(monkeypatch):
    server, calls = _capture(monkeypatch)
    asyncio.run(server.content_posts_update(post_id=5, external_ref={"path": "/docs/a.md"}))
    assert calls[-1][2] == {"external_ref": {"path": "/docs/a.md"}}
    asyncio.run(server.content_posts_update(post_id=5, external_ref_loesen=True))
    assert calls[-1][1] == "/api/v1/posts/5" and calls[-1][2] == {"external_ref": None}
    asyncio.run(server.content_posts_update(post_id=5, title="t"))
    assert "external_ref" not in calls[-1][2]


def test_update_both_is_rejected(monkeypatch):
    server, calls = _capture(monkeypatch)
    r = asyncio.run(server.content_posts_update(post_id=5, external_ref={"path": "/docs/a.md"},
                                                external_ref_loesen=True))
    assert r["error"] == "external_ref_conflict" and calls == []
