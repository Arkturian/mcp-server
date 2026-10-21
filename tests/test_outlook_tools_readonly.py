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
    forbidden = {
        n for n in names
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


def test_outlook_create_draft_idempotency_key_required_in_schema():
    """Steiner-Nachforderung 2026-10-21 (`4baee9ff` §2): `idempotency_key`
    hatte `str = ""` als Default → MCP-Schema zeigte das Feld als optional,
    obwohl die Beschreibung "MANDATORY" sagt. Clients, die nach Schema
    arbeiten, konnten den Key weglassen. Fix: Default raus."""
    import inspect
    import server
    sig = inspect.signature(server.comm_outlook_create_draft)
    param = sig.parameters["idempotency_key"]
    # Kein Default → im Schema als "required" ausgewiesen.
    assert param.default is inspect.Parameter.empty, (
        f"idempotency_key hat Default {param.default!r} — das MCP-Schema "
        f"zeigt es dann als optional, obwohl es Pflicht ist."
    )


def test_outlook_create_draft_signature_order_body_before_idempotency():
    """Positional-Argumente: nach `source` und `body` muss
    `idempotency_key` kommen (vor den anderen required-less Fields),
    damit es im Schema als required gelistet wird. Steiner-Review
    `4baee9ff` §2."""
    import inspect
    import server
    sig = inspect.signature(server.comm_outlook_create_draft)
    names = list(sig.parameters.keys())
    # source, body, idempotency_key sind die drei required-Params; alle
    # anderen haben Defaults und sind optional.
    assert names[:3] == ["source", "body", "idempotency_key"]
    for name in names[3:]:
        assert sig.parameters[name].default is not inspect.Parameter.empty, (
            f"{name!r} hat keinen Default — würde als zweiter required-"
            f"Param im Schema stehen und die MCP-Erwartung brechen."
        )


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
