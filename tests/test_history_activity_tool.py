"""history_activity (cloud-MCP): inhaltsfreie Aktivitaet, gleiches Tor wie history_export."""
from __future__ import annotations
import asyncio
import os
import unittest
from unittest.mock import AsyncMock, patch

os.environ.setdefault("MCP_SERVERS", "cloud")
import server  # noqa: E402


class HistoryActivityToolTests(unittest.TestCase):
    def test_reicht_fenster_und_aufrufer_weiter(self):
        with patch.object(server, "call_cloud_api", new=AsyncMock(return_value={"count": 2})) as call:
            out = asyncio.run(server.cloud_history_activity("TIME-LLM", "Cloud", since="3d", until="1d"))
        self.assertEqual(out, {"count": 2})
        args, kwargs = call.call_args
        self.assertEqual(args[:2], ("GET", "/api/sessions/Cloud/history/activity"))
        self.assertEqual(kwargs["params"], {"since": "3d", "until": "1d", "requester": "TIME-LLM"})


if __name__ == "__main__":
    unittest.main()
