"""history_export (cloud-MCP): reicht Fenster, Format, Zustellart und den Aufrufer an
GET /api/sessions/{name}/history/export weiter — gleiches Tor wie read_history."""
from __future__ import annotations
import asyncio
import os
import unittest
from unittest.mock import AsyncMock, patch

os.environ.setdefault("MCP_SERVERS", "cloud")
import server  # noqa: E402


class HistoryExportToolTests(unittest.TestCase):
    def test_defaults_und_requester(self):
        with patch.object(server, "call_cloud_api", new=AsyncMock(return_value={"turns": 3})) as call:
            out = asyncio.run(server.cloud_history_export("CloudV2", "Cloud", since="6h"))
        self.assertEqual(out, {"turns": 3})
        args, kwargs = call.call_args
        self.assertEqual(args[:2], ("GET", "/api/sessions/Cloud/history/export"))
        self.assertEqual(kwargs["params"], {
            "since": "6h", "until": "", "format": "txt", "tools": "labels",
            "thinking": 0, "deliver": "auto", "requester": "CloudV2"})

    def test_optionen_werden_durchgereicht(self):
        with patch.object(server, "call_cloud_api", new=AsyncMock(return_value={})) as call:
            asyncio.run(server.cloud_history_export("A", "B", since="3d", until="1d", format="md",
                                                    tools="full", thinking=True, deliver="storage"))
        p = call.call_args.kwargs["params"]
        self.assertEqual((p["format"], p["tools"], p["thinking"], p["deliver"], p["until"]),
                         ("md", "full", 1, "storage", "1d"))
