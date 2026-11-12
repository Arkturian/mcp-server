"""kanal_antworten (comm-MCP): Agent antwortet ausdrücklich auf eine
eingegangene Kanalnachricht statt per Fenster-Text (Beitrag 5899)."""
from __future__ import annotations
import asyncio
import os
import unittest
from unittest.mock import AsyncMock, patch

os.environ.setdefault("MCP_SERVERS", "comm")
import server  # noqa: E402


class KanalAntwortenToolTests(unittest.TestCase):
    def test_pfad_und_rumpf(self):
        with patch.object(server, "call_comm_api", new=AsyncMock(return_value={"gesendet": []})) as call:
            out = asyncio.run(server.comm_kanal_antworten(
                "wa-abc", text="Hallo", anhaenge=[{"storage_id": 7}], idempotency_key="k1"))
        self.assertEqual(out, {"gesendet": []})
        args, kwargs = call.call_args
        self.assertEqual(args, ("POST", "/api/v1/kanal/antworten"))
        self.assertEqual(kwargs["json_body"], {
            "handle": "wa-abc", "text": "Hallo",
            "anhaenge": [{"storage_id": 7}], "idempotency_key": "k1"})

    def test_ohne_text_nur_anhang(self):
        with patch.object(server, "call_comm_api", new=AsyncMock(return_value={})) as call:
            asyncio.run(server.comm_kanal_antworten("wa-abc", anhaenge=[{"storage_id": 7}]))
        self.assertEqual(call.call_args.kwargs["json_body"],
                         {"handle": "wa-abc", "anhaenge": [{"storage_id": 7}]})


if __name__ == "__main__":
    unittest.main()
