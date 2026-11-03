"""token_lage (cloud-MCP, #2640): GET /api/usage/lage mit dem JWT des Aufrufers."""
from __future__ import annotations
import asyncio
import os
import unittest
from unittest.mock import AsyncMock, patch

os.environ.setdefault("MCP_SERVERS", "cloud")
import server  # noqa: E402


class TokenLageToolTests(unittest.TestCase):
    def test_reicht_agent_durch(self):
        with patch.object(server, "_fetch_json", new=AsyncMock(return_value={"konten": []})) as fetch:
            out = asyncio.run(server.cloud_token_lage(agent="Cloud"))
        self.assertEqual(out, {"konten": []})
        args, kwargs = fetch.call_args
        self.assertEqual(args[0], "GET")
        self.assertTrue(args[1].endswith("/api/usage/lage"))
        self.assertEqual(kwargs["params"], {"agent": "Cloud"})
        self.assertEqual(kwargs["timeout"], 90)

    def test_ohne_agent_leer(self):
        with patch.object(server, "_fetch_json", new=AsyncMock(return_value={})) as fetch:
            asyncio.run(server.cloud_token_lage())
        self.assertEqual(fetch.call_args.kwargs["params"], {"agent": ""})
