"""posts_notify_level / posts_notify_levels reichen an content-api durch (#2674).

Die Regeln (nur eigene Stufe, gueltige Werte, Mandant) prueft content-api.
"""
import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
for _k in ("ARKTURIAN_API_KEY", "ONEAL_STORAGE_API_KEY", "JWT_PUBLIC_KEY", "COMM_API_KEY"):
    os.environ.setdefault(_k, "test-only")


def _fake(calls, antwort):
    async def _f(method, url, **kw):
        calls.append((method, url, kw.get("json_body")))
        return antwort
    return _f


def test_stufe_beitrag_und_abschnitt(monkeypatch):
    import server
    calls = []
    monkeypatch.setattr(server, "_fetch_json", _fake(calls, {"level": "aus"}))
    asyncio.run(server.content_posts_notify_level(post_id=5, level="aus"))
    asyncio.run(server.content_posts_notify_level(post_id=5, level="gefragt", section_id="q-1"))
    assert calls[0][0] == "PUT" and calls[0][1].endswith("/api/v1/posts/5/notify-level")
    assert calls[0][2] == {"level": "aus"}
    assert calls[1][2] == {"level": "gefragt", "section_id": "q-1"}


def test_stufen_lesen(monkeypatch):
    import server
    calls = []
    monkeypatch.setattr(server, "_fetch_json", _fake(calls, {"members": []}))
    asyncio.run(server.content_posts_notify_levels(post_id=5))
    assert calls[0][0] == "GET" and calls[0][1].endswith("/api/v1/posts/5/notify-levels")


def test_reply_leitet_bei_privat_um(monkeypatch):
    import server
    calls = []

    async def _f(method, url, **kw):
        calls.append((method, url, kw.get("json_body")))
        if url.endswith("/reply"):
            raise RuntimeError('Upstream 409 from POST x: {"error": "section_replies_private"}')
        return {"id": 1, "count": 1}

    monkeypatch.setattr(server, "_fetch_json", _f)
    r = asyncio.run(server.content_sections_reply(post_id=5, section_id="q-1", text="Ich"))
    assert calls[1][1].endswith("/sections/q-1/private-replies") and calls[1][2] == {"text": "Ich"}
    assert r["private"] is True


def test_reply_andere_fehler_bleiben(monkeypatch):
    import pytest
    import server

    async def _f(method, url, **kw):
        raise RuntimeError("Upstream 403 from POST x: forbidden")

    monkeypatch.setattr(server, "_fetch_json", _f)
    with pytest.raises(RuntimeError):
        asyncio.run(server.content_sections_reply(post_id=5, section_id="q-1", text="Ich"))
