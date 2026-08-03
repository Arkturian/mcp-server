"""Regression tests for the normal-agent Arcturian reply bridge (#838)."""

from __future__ import annotations

import asyncio
import os
import unittest
from unittest.mock import AsyncMock, patch

os.environ.setdefault("MCP_SERVERS", "cloud")

import auth  # noqa: E402
import server  # noqa: E402


class CloudReplyBridgeTests(unittest.TestCase):
    def test_legacy_send_message_body_is_byte_shape_compatible(self):
        body = server._cloud_message_body(
            from_session="AppDev",
            to_session="Cloud",
            message="Hallo",
            timeout=0,
            wait=False,
        )
        self.assertEqual(
            body,
            {
                "from": "AppDev",
                "to": "Cloud",
                "message": "Hallo",
                "wait": False,
                "timeout": 0,
            },
        )

    def test_arcturian_reply_fields_are_additive(self):
        body = server._cloud_message_body(
            from_session="AppDev",
            to_session="Arcturian",
            message="Welche Variante?",
            timeout=0,
            wait=False,
            in_reply_to="msg-action-a",
            event_type="question",
            artifacts=[{"kind": "content_post", "id": 4431}],
            options=["A", "B"],
            retryable=True,
            expires_at=200.0,
        )
        self.assertEqual(body["in_reply_to"], "msg-action-a")
        self.assertEqual(body["event_type"], "question")
        self.assertEqual(body["options"], ["A", "B"])
        self.assertTrue(body["retryable"])
        self.assertEqual(body["artifacts"][0]["id"], 4431)

    def test_verified_jwt_context_is_the_only_agent_header_source(self):
        jwt_token = auth._current_jwt.set("signed.jwt")
        name_token = auth._current_agent_name.set("AppDev")
        previous_iacp_token = server.IACP_TOKEN
        server.IACP_TOKEN = "shared-secret"
        try:
            self.assertEqual(
                server._caller_auth_headers(),
                {"Authorization": "Bearer signed.jwt", "X-Agent-Name": "AppDev"},
            )
            self.assertEqual(
                server._cloud_request_headers(),
                {
                    "Authorization": "Bearer signed.jwt",
                    "X-Agent-Name": "AppDev",
                    "X-Agent-Gateway": "mcp",
                    "X-IACP-Token": "shared-secret",
                },
            )
        finally:
            server.IACP_TOKEN = previous_iacp_token
            auth._current_agent_name.reset(name_token)
            auth._current_jwt.reset(jwt_token)

        self.assertEqual(server._caller_auth_headers(), {})
        self.assertEqual(server._cloud_request_headers(), {})

    def test_tool_forwards_reply_wire_without_deriving_actor_from_claim(self):
        mocked = AsyncMock(return_value={"status": "sent"})
        with patch.object(server, "call_cloud_api", mocked):
            result = asyncio.run(
                server.cloud_send_message(
                    "Claimed-Name",
                    "Arcturian",
                    "Fertig",
                    timeout=0,
                    in_reply_to="msg-action-a",
                    event_type="result",
                )
            )
        self.assertEqual(result, {"status": "sent"})
        body = mocked.await_args.kwargs["json_body"]
        self.assertEqual(body["from"], "Claimed-Name")
        self.assertNotIn("X-Agent-Name", body)
        self.assertEqual(body["in_reply_to"], "msg-action-a")
        self.assertTrue(mocked.await_args.kwargs["agent_gateway_proof"])


if __name__ == "__main__":
    unittest.main()
