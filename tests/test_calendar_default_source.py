"""Kalender-Werkzeuge: Vorgabe-Quelle je Instanz (COMM_CALENDAR_DEFAULT_SOURCE).

Anlass (Cloud, P3 16.11.2026): fest "apopovic" gab auf Kundeninstanzen
ohne ausdrückliche Quelle immer 403.
"""
import asyncio
import os

import pytest

for _k in ("ARKTURIAN_API_KEY", "ONEAL_STORAGE_API_KEY", "JWT_PUBLIC_KEY", "COMM_API_KEY"):
    os.environ.setdefault(_k, "test")

import server  # noqa: E402


@pytest.fixture
def seen(monkeypatch):
    calls = []

    async def fake(method, endpoint, **kw):
        calls.append((method, endpoint))
        return {"ok": True}
    monkeypatch.setattr(server, "call_comm_api", fake)
    return calls


def test_unset_keeps_operator_default(monkeypatch, seen):
    monkeypatch.delenv("COMM_CALENDAR_DEFAULT_SOURCE", raising=False)
    asyncio.run(server.comm_calendar_list_events())
    assert seen[-1] == ("GET", "/api/v1/calendar/apopovic/events")


def test_instance_default(monkeypatch, seen):
    monkeypatch.setenv("COMM_CALENDAR_DEFAULT_SOURCE", "jascha")
    asyncio.run(server.comm_calendar_list_calendars())
    asyncio.run(server.comm_calendar_get_event("e1"))
    assert seen == [("GET", "/api/v1/calendar/jascha/calendars"),
                    ("GET", "/api/v1/calendar/jascha/events/e1")]


def test_explicit_source_wins(monkeypatch, seen):
    monkeypatch.setenv("COMM_CALENDAR_DEFAULT_SOURCE", "jascha")
    asyncio.run(server.comm_calendar_delete_event("e1", source="icloud", send_updates="none"))
    assert seen[-1] == ("DELETE", "/api/v1/calendar/icloud/events/e1")


@pytest.mark.parametrize("call", [
    lambda: server.comm_calendar_list_calendars(),
    lambda: server.comm_calendar_list_events(),
    lambda: server.comm_calendar_get_event("e1"),
    lambda: server.comm_calendar_create_event("T", "2026-12-01T10:00:00"),
    lambda: server.comm_calendar_update_event("e1", summary="x"),
    lambda: server.comm_calendar_delete_event("e1"),
    lambda: server.comm_calendar_respond("e1", "accepted"),
])
def test_empty_default_requires_source(monkeypatch, seen, call):
    monkeypatch.setenv("COMM_CALENDAR_DEFAULT_SOURCE", "")
    with pytest.raises(ValueError, match="calendar_source_required"):
        asyncio.run(call())
    assert seen == []


def test_schema_has_no_fixed_mailbox_default():
    tools = {t.name: t for t in asyncio.run(server.comm_mcp.list_tools())}
    for name in ("calendar_list_calendars", "calendar_list_events", "calendar_get_event",
                 "calendar_create_event", "calendar_update_event", "calendar_delete_event",
                 "calendar_respond"):
        prop = tools[name].inputSchema["properties"]["source"]
        assert prop.get("default") is None, name
