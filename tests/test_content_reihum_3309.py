"""sections_pass, sections_reihum und targets an sections_reply reichen an
content-api durch (#3309). Die Regeln prüft content-api."""
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


def test_pass(monkeypatch):
    import server
    calls = []
    monkeypatch.setattr(server, "_fetch_json", _fake(calls, {"am_zug": "B"}))
    asyncio.run(server.content_sections_pass(post_id=5, section_id="q-1", grund="bereits abgedeckt"))
    assert calls[0][0] == "POST" and calls[0][1].endswith("/api/v1/posts/5/sections/q-1/pass")
    assert calls[0][2] == {"grund": "bereits abgedeckt"}


def test_reihum_eingriff(monkeypatch):
    import server
    calls = []
    monkeypatch.setattr(server, "_fetch_json", _fake(calls, {"am_zug": "A"}))
    asyncio.run(server.content_sections_reihum(post_id=5, section_id="q-1", aktion="reihenfolge",
                                               reihenfolge=["B", "A"]))
    asyncio.run(server.content_sections_reihum(post_id=5, section_id="q-1", aktion="zurueckholen", agent="C"))
    assert calls[0][1].endswith("/sections/q-1/reihum")
    assert calls[0][2] == {"aktion": "reihenfolge", "reihenfolge": ["B", "A"]}
    assert calls[1][2] == {"aktion": "zurueckholen", "agent": "C"}


def test_reply_mit_targets(monkeypatch):
    import server
    calls = []
    monkeypatch.setattr(server, "_fetch_json", _fake(calls, {"section_id": "q-1"}))
    asyncio.run(server.content_sections_reply(post_id=5, section_id="q-1", text="x", targets="Cloud"))
    asyncio.run(server.content_sections_reply(post_id=5, section_id="q-1", text="y"))
    assert calls[0][2] == {"text": "x", "targets": "Cloud"}
    assert calls[1][2] == {"text": "y"}
