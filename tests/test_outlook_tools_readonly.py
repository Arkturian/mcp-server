"""#1455 + steiner Issue #60: comm-MCP traegt Outlook-LESE-Tools plus
`outlook_create_draft` — und sonst keinen Schreibpfad.

Steward (David) will Sent-Mails auswerten (Lesen). Fuer steiner's David
kommt `outlook_create_draft` dazu (Cloud 2026-10-21): Entwuerfe landen
im Drafts-Ordner, der Mensch sendet selbst in Outlook — der Code-Pfad
hier wie in comm-api darf NIEMALS einen Send ausloesen (comm-api hat
den adversarial Test dafuer, hier reicht der Namens-Blacklist-Check).
"""
import asyncio
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
for _k in ("ARKTURIAN_API_KEY", "ONEAL_STORAGE_API_KEY", "JWT_PUBLIC_KEY", "COMM_API_KEY"):
    os.environ.setdefault(_k, "test-only")


def _tool_names():
    import server
    mgr = getattr(server.comm_mcp, "_tool_manager", None)
    if mgr is not None and hasattr(mgr, "list_tools"):
        return {t.name for t in mgr.list_tools()}
    return {t.name for t in asyncio.run(server.comm_mcp.list_tools())}


def test_outlook_read_tools_are_registered():
    names = _tool_names()
    assert {"outlook_list_accounts", "outlook_list_messages", "outlook_get_message"} <= names


def test_no_outlook_send_tools():
    """`send`/`mark`/`delete`/`move` bleiben verboten. `create_draft` ist
    erlaubt (Entwurf != Send, steiner #60) — der Blacklist-Check prueft
    jetzt gezielt die wirkenden Verben, nicht pauschal 'draft'."""
    names = {n for n in _tool_names() if n.startswith("outlook_")}
    # #3403 (Steiner fakturace@, Freigabe Alex über Cloud): genau ein
    # benanntes Werkzeug darf „mark“ tragen. Es setzt nur
    # flag.flagStatus=complete; comm-api prüft Quelle und Agent.
    erlaubt = {"outlook_mark_complete"}
    forbidden = {
        n for n in names - erlaubt
        if any(k in n for k in ("send", "mark", "delete", "move"))
    }
    assert forbidden == set(), forbidden


def test_outlook_create_draft_is_registered():
    """steiner #60: Entwuerfe-Tool muss unter diesem Namen laufen, damit
    der Prompt-Pointer in server.py:7883 nicht ins Leere zeigt."""
    names = _tool_names()
    assert "outlook_create_draft" in names


def test_outlook_create_draft_signature():
    """idempotency_key ist Pflicht und muss im Schema sichtbar sein —
    sonst kann der Agent ihn nicht uebergeben. subject/cc/bcc optional."""
    import inspect
    import server
    sig = inspect.signature(server.comm_outlook_create_draft)
    required = {"source", "body"}
    for p in required:
        assert p in sig.parameters, p
    assert "idempotency_key" in sig.parameters
    # Optional fields present and defaulted
    for p in ("subject", "cc", "bcc", "reply_to_message_id",
             "attachments", "html", "to"):
        assert p in sig.parameters, p


def test_folder_is_only_sent_when_given():
    import inspect
    import server
    sig = inspect.signature(server.comm_outlook_list_messages)
    assert sig.parameters["folder"].default == ""


def test_cloud_admin_tools_registered():
    import server
    names = {t.name for t in server.cloud_mcp._tool_manager.list_tools()}
    assert {"session_mcp_servers", "session_set_mcp_servers", "session_restart"} <= names


def test_list_messages_exposes_cursor_and_timerange():
    """Ein Agent kann nur uebergeben, was das Schema deklariert (#1455)."""
    import inspect
    import server
    sig = inspect.signature(server.comm_outlook_list_messages)
    for p in ("cursor", "since", "until"):
        assert p in sig.parameters, p
        assert sig.parameters[p].default == ""


def test_sections_patch_registered_and_shaped():
    """Section-Attribute (Status/Assignee) brauchen einen MCP-Weg (#23 David)."""
    import inspect
    import server
    names = {t.name for t in server.content_mcp._tool_manager.list_tools()}
    assert "sections_patch" in names
    sig = inspect.signature(server.content_sections_patch)
    assert set(sig.parameters) == {"post_id", "section_id", "attrs", "status"}
    assert sig.parameters["attrs"].default is None and sig.parameters["status"].default is None


def test_outlook_mark_complete_ruft_nur_die_eine_route(monkeypatch):
    """#3403: Der Wrapper ruft genau POST …/mark-complete/{id}, nichts sonst."""
    import server
    calls = []

    async def _fake(method, endpoint, **kw):
        calls.append((method, endpoint, kw))
        return {"success": True}

    monkeypatch.setattr(server, "call_comm_api", _fake)
    out = asyncio.run(server.comm_outlook_mark_complete("steiner_fakturace", "AAMk-1"))
    assert out == {"success": True}
    assert calls == [("POST", "/api/v1/outlook/steiner_fakturace/mark-complete/AAMk-1", {})]
    import inspect
    assert list(inspect.signature(server.comm_outlook_mark_complete).parameters) == ["source", "message_id"]


def test_outlook_create_draft_idempotency_key_required_in_schema():
    """Steiner-Nachforderung `4baee9ff` §2: Ohne Default weist das MCP-Schema
    `idempotency_key` als Pflichtfeld aus."""
    import inspect
    import server
    sig = inspect.signature(server.comm_outlook_create_draft)
    assert sig.parameters["idempotency_key"].default is inspect.Parameter.empty


def test_outlook_create_draft_signature_order_body_before_idempotency():
    import inspect
    import server
    sig = inspect.signature(server.comm_outlook_create_draft)
    names = list(sig.parameters.keys())
    assert names[:3] == ["source", "body", "idempotency_key"]
    for name in names[3:]:
        assert sig.parameters[name].default is not inspect.Parameter.empty, name


@pytest.mark.parametrize("source,message_id", [
    ("steiner/mark-read/AAMk123#", "x"),
    ("steiner", "AAMk123/../../mark-read/y"),
    ("steiner", "AAMk123?x=1#"),
])
def test_outlook_mark_complete_cannot_leave_its_route(monkeypatch, source, message_id):
    """Cloud-Security 236faa34: `/`, `?`, `#` bleiben im Segment, der Aufruf
    wird nie zu mark-read oder mark-unread."""
    import urllib.parse
    import server
    calls = []

    async def _fake(method, endpoint, **kw):
        calls.append((method, endpoint))
        return {"success": True}

    monkeypatch.setattr(server, "call_comm_api", _fake)
    asyncio.run(server.comm_outlook_mark_complete(source, message_id))
    q, m = urllib.parse.quote(source, safe=""), urllib.parse.quote(message_id, safe="")
    assert calls == [("POST", f"/api/v1/outlook/{q}/mark-complete/{m}")]
    assert "#" not in calls[0][1] and "?" not in calls[0][1]


@pytest.mark.parametrize("wert", ["", ".", ".."])
def test_outlook_mark_complete_rejects_empty_and_dot_segments(monkeypatch, wert):
    import server

    async def _fake(*a, **k):
        raise AssertionError("darf nicht aufgerufen werden")

    monkeypatch.setattr(server, "call_comm_api", _fake)
    with pytest.raises(ValueError):
        asyncio.run(server.comm_outlook_mark_complete("steiner", wert))
