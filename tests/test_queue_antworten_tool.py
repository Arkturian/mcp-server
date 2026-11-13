"""queue_antworten (cloud-MCP, #2740): der Agent gibt die Antwort auf seine Queue-Nachricht
ausdruecklich ab -> POST /api/queue/<er selbst>/messages/<msg_id>/antwort.

Die Session kommt aus der geprueften Gateway-Identitaet, nie aus einem Argument."""
from __future__ import annotations
import asyncio
import os
import unittest
from unittest.mock import AsyncMock, patch

os.environ.setdefault("MCP_SERVERS", "cloud")
import auth  # noqa: E402
import server  # noqa: E402


class QueueAntwortenToolTests(unittest.TestCase):
    def _rufen(self, agent, *args, **kwargs):
        with patch.object(auth, "current_caller_verified_agent", return_value=agent), \
             patch.object(server, "call_cloud_api",
                          new=AsyncMock(return_value={"ok": True, "status": "done"})) as call:
            out = asyncio.run(server.cloud_queue_antworten(*args, **kwargs))
        return out, call

    def test_pfad_aus_der_identitaet_und_rumpf(self):
        anh = [{"storage_id": 131124, "filename": "Rechnung.pdf", "content_type": "application/pdf"}]
        out, call = self._rufen("Arkturian", 4711, "Anbei die Rechnung.", attachments=anh)
        self.assertEqual(out["status"], "done")
        args, kwargs = call.call_args
        self.assertEqual(args, ("POST", "/api/queue/Arkturian/messages/4711/antwort"))
        self.assertEqual(kwargs["json_body"], {"content": "Anbei die Rechnung.", "attachments": anh})

    def test_ohne_anhaenge_kein_feld(self):
        _, call = self._rufen("Arkturian", 5, "ok")
        self.assertEqual(call.call_args.kwargs["json_body"], {"content": "ok"})

    def test_name_wird_kodiert(self):
        _, call = self._rufen("Bauleiter Größe", 7, "ok")
        self.assertEqual(call.call_args.args[1], "/api/queue/Bauleiter%20Gr%C3%B6%C3%9Fe/messages/7/antwort")

    def test_ohne_agenten_identitaet_kein_aufruf(self):
        with patch.object(auth, "current_caller_verified_agent", return_value=""), \
             patch.object(server, "call_cloud_api", new=AsyncMock()) as call:
            with self.assertRaises(ValueError):
                asyncio.run(server.cloud_queue_antworten(1, "x"))
        call.assert_not_called()

    def test_werkzeug_ist_registriert(self):
        namen = {t.name for t in asyncio.run(server.cloud_mcp.list_tools())}
        self.assertIn("queue_antworten", namen)


if __name__ == "__main__":
    unittest.main()
