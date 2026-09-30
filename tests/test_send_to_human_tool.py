"""send_to_human (cloud-MCP, Alex 30.09.): „schick Tommy eine Nachricht" ohne Kanal
heisst das interne System — POST /api/user-messages mit der Anmeldung des Agenten.
Den Absender setzt cloud-api aus dem JWT; das Werkzeug reicht nie ein `from` durch."""
from __future__ import annotations
import asyncio
import os
import unittest
from unittest.mock import AsyncMock, patch

os.environ.setdefault("MCP_SERVERS", "cloud")
import server  # noqa: E402


class SendToHumanToolTests(unittest.TestCase):
    def test_ruft_user_messages_ohne_absender(self):
        with patch.object(server, "call_cloud_api",
                          new=AsyncMock(return_value={"id": "m1", "delivered_to": 2})) as call:
            out = asyncio.run(server.cloud_send_to_human(" t.saier@edera-safety.com ", "Hallo Tommy"))
        self.assertEqual(out["delivered_to"], 2)
        args, kwargs = call.call_args
        self.assertEqual(args, ("POST", "/api/user-messages"))
        self.assertEqual(kwargs["json_body"], {"to": "t.saier@edera-safety.com", "text": "Hallo Tommy"})
        self.assertNotIn("from", kwargs["json_body"])

    def test_message_id_wird_durchgereicht(self):
        with patch.object(server, "call_cloud_api", new=AsyncMock(return_value={})) as call:
            asyncio.run(server.cloud_send_to_human("a@b.at", "x", message_id="idem-1"))
        self.assertEqual(call.call_args.kwargs["json_body"]["message_id"], "idem-1")

    def test_werkzeug_ist_registriert(self):
        namen = {t.name for t in asyncio.run(server.cloud_mcp.list_tools())}
        self.assertIn("send_to_human", namen)
