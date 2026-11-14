"""Kalender-Werkzeuge (#2839): ein Satz für alle Anbieter, richtige comm-api-Aufrufe."""
import asyncio
import os

for _k in ("ARKTURIAN_API_KEY", "ONEAL_STORAGE_API_KEY", "JWT_PUBLIC_KEY", "COMM_API_KEY"):
    os.environ.setdefault(_k, "test")

import server  # noqa: E402


def _calls(monkeypatch):
    seen = []

    async def fake(method, endpoint, **kw):
        seen.append((method, endpoint, kw))
        return {"ok": True}
    monkeypatch.setattr(server, "call_comm_api", fake)
    return seen


def test_tool_set_complete():
    names = {t.name for t in asyncio.run(server.comm_mcp.list_tools())}
    assert {"calendar_list_sources", "calendar_list_calendars", "calendar_list_events",
            "calendar_get_event", "calendar_create_event", "calendar_update_event",
            "calendar_delete_event", "calendar_respond", "calendar_freebusy"} <= names


def test_update_is_patch_with_only_given_fields(monkeypatch):
    seen = _calls(monkeypatch)
    asyncio.run(server.comm_calendar_update_event("rec2.ics~20261208T080000Z", source="icloud",
                                                  start="2026-12-08T15:00:00", etag='"e"'))
    method, path, kw = seen[0]
    assert method == "PATCH"
    assert path == "/api/v1/calendar/icloud/events/rec2.ics~20261208T080000Z"
    assert kw["json_body"] == {"start": "2026-12-08T15:00:00", "send_updates": "all", "etag": '"e"'}
    assert kw["params"] == {"calendar_id": "primary"}


def test_create_keeps_legacy_args(monkeypatch):
    seen = _calls(monkeypatch)
    asyncio.run(server.comm_calendar_create_event("T", "2026-12-01T10:00:00", "2026-12-01T11:00:00",
                                                  timezone="Europe/Berlin", google_meet=True))
    body = seen[0][2]["json_body"]
    assert body["time_zone"] == "Europe/Berlin" and body["online_meeting"] is True


def test_delete_and_respond(monkeypatch):
    seen = _calls(monkeypatch)
    asyncio.run(server.comm_calendar_delete_event("ev1", source="edera", send_updates="none"))
    asyncio.run(server.comm_calendar_respond("ev1", "declined", source="edera", comment="nein"))
    assert seen[0][0] == "DELETE" and seen[0][2]["params"]["send_updates"] == "none"
    assert seen[1][1].endswith("/edera/events/ev1/respond")
    assert seen[1][2]["json_body"]["comment"] == "nein"
