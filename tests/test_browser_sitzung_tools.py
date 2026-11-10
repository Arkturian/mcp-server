"""browser_sitzungen / browser_freigabe / browser_aufruf / browser_freigabe_beenden (Post 5857)."""
from __future__ import annotations
import asyncio
import os
import unittest
from unittest.mock import AsyncMock, patch

os.environ.setdefault("MCP_SERVERS", "cloud")
import server  # noqa: E402


class BrowserSitzungToolTests(unittest.TestCase):
    def _ruf(self, coro):
        with patch.object(server, "call_cloud_api", new=AsyncMock(return_value={})) as call:
            asyncio.run(coro())
        return call.call_args

    def test_sitzungen(self):
        a = self._ruf(lambda: server.cloud_browser_sitzungen())
        self.assertEqual(a.args, ("GET", "/api/browser/sitzungen"))

    def test_freigabe(self):
        a = self._ruf(lambda: server.cloud_browser_freigabe("steuern", "Formular", 10))
        self.assertEqual(a.args, ("POST", "/api/browser/freigaben"))
        self.assertEqual(a.kwargs["json_body"], {"stufe": "steuern", "grund": "Formular", "minuten": 10, "sitzung": "aktiv"})

    def test_aufruf(self):
        a = self._ruf(lambda: server.cloud_browser_aufruf("click", {"selector": "#ok"}, "S1"))
        self.assertEqual(a.args, ("POST", "/api/browser/aufrufe"))
        self.assertEqual(a.kwargs["json_body"], {"methode": "click", "parameter": {"selector": "#ok"}, "sitzung": "S1"})
        self.assertGreaterEqual(a.kwargs["timeout"], 40)

    def test_beschreibung_nennt_devtools(self):
        tools = asyncio.run(server.cloud_mcp.list_tools()) if hasattr(server.cloud_mcp, "list_tools") else []
        beschr = {t.name: t.description for t in tools}.get("browser_aufruf", "")
        if beschr:
            for m in ("performance", "network", "profile", "coverage", "heap", "evaluate"):
                self.assertIn(m, beschr)

    def test_beenden(self):
        a = self._ruf(lambda: server.cloud_browser_freigabe_beenden())
        self.assertEqual(a.args, ("POST", "/api/browser/freigaben/beenden"))


if __name__ == "__main__":
    unittest.main()
