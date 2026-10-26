"""bridge_message_status (cloud-MCP): Stand einer Bruecken-Nachricht ueber
GET /api/bridge/outbox/{id}, MIT Gateway-Nachweis — cloud-api zeigt den Stand
nur dem bezeugten Absender (Alex 26.10.2026: Zustellklarheit)."""
from __future__ import annotations
import asyncio
import os
import unittest
from unittest.mock import AsyncMock, patch

os.environ.setdefault("MCP_SERVERS", "cloud")
import server  # noqa: E402


class BridgeMessageStatusToolTests(unittest.TestCase):
    def test_pfad_und_nachweis(self):
        with patch.object(server, "call_cloud_api", new=AsyncMock(return_value={"status": "delivered"})) as call:
            out = asyncio.run(server.cloud_bridge_message_status("71f420d152df"))
        self.assertEqual(out, {"status": "delivered"})
        args, kwargs = call.call_args
        self.assertEqual(args, ("GET", "/api/bridge/outbox/71f420d152df"))
        self.assertTrue(kwargs["agent_gateway_proof"])

    def test_kennung_wird_kodiert(self):
        with patch.object(server, "call_cloud_api", new=AsyncMock(return_value={})) as call:
            asyncio.run(server.cloud_bridge_message_status("../x"))
        self.assertEqual(call.call_args.args[1], "/api/bridge/outbox/..%2Fx")

    def test_echter_aufrufpfad(self):
        with patch.object(server, "_fetch_json", new=AsyncMock(return_value={"status": "queued"})) as fetch:
            out = asyncio.run(server.cloud_bridge_message_status("abc"))
        self.assertEqual(out, {"status": "queued"})
        self.assertTrue(fetch.call_args.args[1].endswith("/api/bridge/outbox/abc"))
