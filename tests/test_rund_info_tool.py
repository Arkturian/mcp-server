"""rund_info (cloud-MCP, #2797): Details einer Rundnachricht ueber GET /api/rund/{id},
mit dem Bearer des Aufrufers -- cloud-api entscheidet, wer lesen darf (sonst 404)."""
from __future__ import annotations
import asyncio
import os
import unittest
from unittest.mock import AsyncMock, patch

os.environ.setdefault("MCP_SERVERS", "cloud")
import server  # noqa: E402


class RundInfoToolTests(unittest.TestCase):
    def _rufen(self, rid):
        antwort = {"id": rid, "absender": "Cloud", "zeit": 1.0, "n": 2,
                   "empfaenger": [{"name": "A", "status": "zugestellt"}, {"name": "B", "status": "wartet"}]}
        with patch.object(server, "call_cloud_api", new=AsyncMock(return_value=antwort)) as call:
            out = asyncio.run(server.cloud_rund_info(rid))
        return out, call

    def test_pfad_und_antwort(self):
        out, call = self._rufen("rund-0123456789ab")
        self.assertEqual(call.call_args.args, ("GET", "/api/rund/rund-0123456789ab"))
        self.assertEqual(out["n"], 2)

    def test_kaputte_kennung_ruft_nichts(self):
        for rid in ("../sessions", "a b", "", "x" * 41):
            with patch.object(server, "call_cloud_api", new=AsyncMock()) as call:
                with self.assertRaises(ValueError):
                    asyncio.run(server.cloud_rund_info(rid))
            call.assert_not_called()

    def test_werkzeug_ist_registriert(self):
        namen = {t.name for t in asyncio.run(server.cloud_mcp.list_tools())}
        self.assertIn("rund_info", namen)


if __name__ == "__main__":
    unittest.main()
