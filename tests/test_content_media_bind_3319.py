"""media_bind und media_unbind reichen an content-api durch (#3319).
Die Regeln (eigene Bilder, Schreibrecht) prüfen content-api und storage-api."""
import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
for _k in ("ARKTURIAN_API_KEY", "ONEAL_STORAGE_API_KEY", "JWT_PUBLIC_KEY", "COMM_API_KEY"):
    os.environ.setdefault(_k, "test-only")


def test_bind_und_unbind(monkeypatch):
    import server
    calls = []

    async def fake(method, url, **kw):
        calls.append((method, url, kw.get("json_body")))
        return {"storage_id": 134668, "scope": "content-post:6228", "created": True}

    monkeypatch.setattr(server, "_fetch_json", fake)
    asyncio.run(server.content_media_bind(post_id=6228, storage_id=134668))
    asyncio.run(server.content_media_unbind(post_id=6228, storage_id=134668))
    assert calls[0][0] == "POST" and calls[0][1].endswith("/api/v1/posts/6228/media-grants")
    assert calls[0][2] == {"storage_id": 134668}
    assert calls[1][0] == "DELETE" and calls[1][1].endswith("/api/v1/posts/6228/media-grants/134668")
