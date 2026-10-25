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

    def test_echter_aufrufpfad_mit_timeout(self):
        """Issue #2585: der Mock auf call_cloud_api verdeckte, dass dessen Signatur
        kein timeout kannte — hier laeuft der Aufruf durch die echte Funktion."""
        with patch.object(server, "_fetch_json", new=AsyncMock(return_value={"turns": 1})) as fetch:
            out = asyncio.run(server.cloud_history_export("Whatsapp", "Whatsapp", since="2h"))
        self.assertEqual(out, {"turns": 1})
        self.assertEqual(fetch.call_args.kwargs["timeout"], 90)
        self.assertTrue(fetch.call_args.args[1].endswith("/api/sessions/Whatsapp/history/export"))
