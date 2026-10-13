"""#2440: exercise the registered tool, auth forwarding and retry intent."""
import asyncio
import os

import pytest
from mcp.server.fastmcp.exceptions import ToolError

os.environ.setdefault("MCP_SERVERS", "comm")

import auth
import server


TOOL = "whatsapp_send_group_message"
ARGS = {"group_token": "grp-TEST1234", "body": "Hallo", "request_key": "intent-12345"}


@pytest.fixture
def capture(monkeypatch):
    calls = []

    async def fetch(method, url, **kwargs):
        calls.append((method, url, kwargs))
        return {"wa_rest_id": "message-1", "status": "queued", "is_replay": False}

    monkeypatch.setattr(server, "_fetch_json", fetch)
    monkeypatch.setattr(auth, "current_caller_jwt", lambda: "verified.fixture.jwt")
    monkeypatch.setattr(auth, "current_caller_verified_agent", lambda: "Whatsapp")
    monkeypatch.setattr(auth, "current_caller_agent_name", lambda: "Whatsapp")
    return calls


def invoke(args):
    return asyncio.run(server.comm_mcp.call_tool(TOOL, args))


def test_discovery_exposes_required_intent_and_opaque_group():
    tools = asyncio.run(server.comm_mcp.list_tools())
    schema = next(t.inputSchema for t in tools if t.name == TOOL)
    assert set(schema["required"]) == {"group_token", "body", "request_key"}
    assert schema["properties"]["group_token"]["pattern"] == r"^grp-[A-Z0-9]+$"
    for field in ("agent_session", "owner_email", "account_id", "group_id"):
        assert field not in schema["properties"]


def test_real_tool_forwards_caller_jwt_and_same_key_on_retry(capture):
    args = {**ARGS, "in_reply_to": "inbox-123"}
    invoke(args)
    invoke(args)
    assert len(capture) == 2
    assert capture[0] == capture[1]
    method, url, kwargs = capture[0]
    assert method == "POST" and url.endswith("/api/v1/whatsapp/group-send")
    assert kwargs["json_body"] == args
    assert kwargs["headers"]["Authorization"] == "Bearer verified.fixture.jwt"
    assert "X-API-KEY" not in kwargs["headers"]


def test_new_intent_same_text_keeps_new_key(capture):
    invoke(ARGS)
    invoke({**ARGS, "request_key": "intent-67890"})
    assert [c[2]["json_body"]["request_key"] for c in capture] == ["intent-12345", "intent-67890"]
    assert all("in_reply_to" not in c[2]["json_body"] for c in capture)


@pytest.mark.parametrize("agent,jwt", [("", ""), ("", "human.fixture.jwt"), ("Whatsapp", "")])
def test_no_legacy_or_human_auth_fallback(capture, monkeypatch, agent, jwt):
    monkeypatch.setattr(server, "COMM_API_KEY", "must-not-be-used")
    monkeypatch.setattr(auth, "current_caller_verified_agent", lambda: agent)
    monkeypatch.setattr(auth, "current_caller_jwt", lambda: jwt)
    with pytest.raises(ToolError, match="agent_identity_required"):
        invoke(ARGS)
    assert not capture


@pytest.mark.parametrize("patch", [
    {"group_token": "120000000000001@g.us"},
    {"request_key": "bad key with spaces"},
    {"attachments": [{"storage_id": 123}]},
])
def test_bad_input_or_media_does_not_send_partial_text(capture, patch):
    with pytest.raises(ToolError):
        invoke({**ARGS, **patch})
    assert not capture


STATUS_TOOL = "whatsapp_group_message_status"
STATUS_KEY = "wa-group:" + "a" * 64


def status_invoke(key=STATUS_KEY):
    return asyncio.run(server.comm_mcp.call_tool(STATUS_TOOL, {"idempotency_key": key}))


def test_status_discovery_uses_returned_key_only():
    tools = asyncio.run(server.comm_mcp.list_tools())
    schema = next(t.inputSchema for t in tools if t.name == STATUS_TOOL)
    assert schema["required"] == ["idempotency_key"]
    assert set(schema["properties"]) == {"idempotency_key"}


@pytest.mark.parametrize("agent", ["Whatsapp", ""])
def test_status_forwards_verified_agent_or_human_jwt(capture, monkeypatch, agent):
    monkeypatch.setattr(auth, "current_caller_verified_agent", lambda: agent)
    status_invoke()
    assert len(capture) == 1
    method, url, kwargs = capture[0]
    assert method == "GET"
    assert url.endswith(f"/api/v1/whatsapp/group-send/{STATUS_KEY}/status")
    assert kwargs["headers"]["Authorization"] == "Bearer verified.fixture.jwt"
    assert "X-API-KEY" not in kwargs["headers"]
    assert kwargs.get("json_body") is None


def test_status_never_falls_back_to_service_key(capture, monkeypatch):
    monkeypatch.setattr(auth, "current_caller_jwt", lambda: "")
    monkeypatch.setattr(server, "COMM_API_KEY", "must-not-be-used")
    with pytest.raises(ToolError, match="caller_identity_required"):
        status_invoke()
    assert not capture


@pytest.mark.parametrize("key", ["intent-12345", "grp-TEST1234", "../status", STATUS_KEY + "?other=1"])
def test_status_rejects_wrong_identity_or_path(capture, key):
    with pytest.raises(ToolError):
        status_invoke(key)
    assert not capture


def test_status_preserves_unknown_and_partial_receipts(capture, monkeypatch):
    import json

    payload = {
        "status": "failed",
        "send_error": "group_replay_roster_changed_acceptance_unknown",
        "delivery_scope": "any_participant",
        "group_delivery": {
            "expected_participants": 3, "delivered_participants": 1,
            "read_participants": 0, "decrypt_failed_participants": 1,
        },
    }

    async def fetch(method, url, **kwargs):
        capture.append(method)
        return payload

    monkeypatch.setattr(server, "_fetch_json", fetch)
    result = status_invoke()
    # FastMCP versions may also return structured content beside text blocks.
    content = result[0] if isinstance(result, tuple) else result
    assert json.loads(content[0].text) == payload
    assert capture == ["GET"]
