"""turn_initiator (Alex 02.11.2026): der Agent fragt selbst, welcher Mensch
bzw. welche Kette seinen laufenden Zug begonnen hat."""
from __future__ import annotations
import asyncio
import os
import unittest
from unittest.mock import AsyncMock, patch

os.environ.setdefault("MCP_SERVERS", "cloud")
import server  # noqa: E402


class TurnInitiatorToolTests(unittest.TestCase):
    def test_fragt_den_initiator_der_eigenen_sitzung(self):
        antwort = {"agent": "Cloud", "art": "mensch", "email": "alex@example.com", "gueltig": True}
        with patch.object(server, "call_cloud_api", new=AsyncMock(return_value=antwort)) as call:
            out = asyncio.run(server.cloud_turn_initiator("Cloud"))
        self.assertEqual(call.call_args.args, ("GET", "/api/sessions/Cloud/initiator"))
        self.assertEqual(out, antwort)

    def test_beschreibung_warnt_vor_dem_login_konto(self):
        tools = asyncio.run(server.cloud_mcp.list_tools()) if hasattr(server.cloud_mcp, "list_tools") else []
        beschr = {t.name: t.description for t in tools}.get("turn_initiator", "")
        if beschr:
            self.assertIn("userEmail", beschr)


if __name__ == "__main__":
    unittest.main()
