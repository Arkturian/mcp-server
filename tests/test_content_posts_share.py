"""posts_share / posts_shares / posts_unshare reichen an content-api durch (Alex 30.10.2026).

Die Regeln (nur lesend, nur Autor, kein Platzhalter) prueft content-api.
Das MCP schickt immer permission=read und nie eine Einladungsmail.
"""
import asyncio
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
for _k in ("ARKTURIAN_API_KEY", "ONEAL_STORAGE_API_KEY", "JWT_PUBLIC_KEY", "COMM_API_KEY"):
    os.environ.setdefault(_k, "test-only")


def _fake(calls, antwort):
    async def _f(method, url, **kw):
        calls.append((method, url, kw.get("json_body"), kw.get("params")))
        if callable(antwort):
            return antwort(method, url, kw)
        return antwort
    return _f


def test_share_je_empfaenger_lesend_ohne_mail(monkeypatch):
    import server
    calls = []

    def antwort(method, url, kw):
        b = kw["json_body"]
        if b["principal_id"] == "agent:Kaputt":
            raise RuntimeError("403 Not authorized")
        return {"id": 1, "principal_id": b["principal_id"], "permission": "read",
                "share_token": "geheim", "share_url": "https://x/redeem/geheim"}

    monkeypatch.setattr(server, "_fetch_json", _fake(calls, antwort))
    r = asyncio.run(server.content_posts_share(
        post_id=5741, agents=["Business", "Kaputt"], emails=["apopovic.aut@gmail.com"], note="Bericht"))
    assert [c[2]["principal_id"] for c in calls] == [
        "agent:Business", "agent:Kaputt", "email:apopovic.aut@gmail.com"]
    assert all(c[0] == "POST" and c[1].endswith("/api/v1/posts/5741/permissions") for c in calls)
    assert all(c[2]["permission"] == "read" and c[2]["send_email"] is False for c in calls)
    assert [g["principal_id"] for g in r["granted"]] == ["agent:Business", "email:apopovic.aut@gmail.com"]
    assert all("share_token" not in g and "share_url" not in g for g in r["granted"])
    assert r["failed"][0]["principal_id"] == "agent:Kaputt"


def test_share_ohne_empfaenger():
    import server
    with pytest.raises(ValueError):
        asyncio.run(server.content_posts_share(post_id=1, agents=[" "], emails=[]))


def test_shares_und_unshare(monkeypatch):
    import server
    calls = []
    monkeypatch.setattr(server, "_fetch_json", _fake(calls, [{"id": 3, "share_token": "t"}]))
    rows = asyncio.run(server.content_posts_shares(post_id=7, include_revoked=True))
    assert rows == [{"id": 3}]
    assert calls[0][0] == "GET" and calls[0][3] == {"include_revoked": "true"}
    calls.clear()
    monkeypatch.setattr(server, "_fetch_json", _fake(calls, {}))
    assert asyncio.run(server.content_posts_unshare(post_id=7, share_id=3))["revoked"] is True
    assert calls[0][0] == "DELETE" and calls[0][1].endswith("/api/v1/posts/7/permissions/3")


def test_visibility_wird_durchgereicht(monkeypatch):
    import server
    calls = []
    monkeypatch.setattr(server, "_fetch_json", _fake(calls, {"id": 9}))
    asyncio.run(server.content_posts_update(post_id=9, visibility="restricted"))
    asyncio.run(server.content_posts_create(title="T", visibility="restricted"))
    assert [c[2]["visibility"] for c in calls] == ["restricted", "restricted"]
