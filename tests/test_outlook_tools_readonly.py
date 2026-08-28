"""#1455: comm-MCP traegt Outlook-LESE-Tools — und nur die.

Steward (David) will Sent-Mails auswerten; ein Schreibpfad (send/mark/delete)
darf ueber diesen Spiegel nicht entstehen.
"""
import asyncio
import os
import sys
from pathlib import Path

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


def test_no_outlook_write_tools():
    names = {n for n in _tool_names() if n.startswith("outlook_")}
    forbidden = {n for n in names if any(k in n for k in ("send", "mark", "delete", "move", "draft"))}
    assert forbidden == set(), forbidden


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
