"""ai-Werkzeuge tragen die Aufrufer-Identitaet (AiApi, Auftrag Alex ueber CloudV2:
Bilder je Agent). call_ai_api setzt dieselben Kopfzeilen wie _caller_auth_headers
(Bearer <jwt>, X-Agent-Name); ohne Aufrufer-Kontext bleibt es beim X-API-Key;
extra_headers des Aufrufers gehen vor."""
from __future__ import annotations
import asyncio
import os
import unittest
from unittest.mock import AsyncMock, patch

os.environ.setdefault("MCP_SERVERS", "ai")
import auth  # noqa: E402
import server  # noqa: E402


class AiKopfzeilenTests(unittest.TestCase):
    def _genimage(self, jwt, agent):
        with patch.object(auth, "current_caller_jwt", return_value=jwt), \
             patch.object(auth, "current_caller_agent_name", return_value=agent), \
             patch.object(auth, "current_caller_verified_agent", return_value=agent), \
             patch.object(server, "AI_API_KEY", "ai-schluessel"), \
             patch.object(server, "_with_grant_scope", side_effect=lambda b: b), \
             patch.object(server, "_fetch_json", new=AsyncMock(return_value={"image_url": "u"})) as f:
            asyncio.run(server.ai_genimage("ein Haus"))
        return f.call_args.kwargs["headers"], f.call_args.args

    def test_genimage_traegt_agent_und_ausweis(self):
        h, args = self._genimage("jwt-123", "Bauleiter")
        self.assertTrue(args[1].endswith("/ai/genimage"))
        self.assertEqual(h["Authorization"], "Bearer jwt-123")
        self.assertEqual(h["X-Agent-Name"], "Bauleiter")
        self.assertEqual(h["X-API-Key"], "ai-schluessel")

    def test_ohne_aufrufer_nur_der_schluessel(self):
        h, _ = self._genimage(None, None)
        self.assertEqual(h, {"X-API-Key": "ai-schluessel"})

    def test_mitgegebene_kopfzeilen_gehen_vor(self):
        with patch.object(auth, "current_caller_jwt", return_value="jwt-123"), \
             patch.object(auth, "current_caller_agent_name", return_value="Bauleiter"), \
             patch.object(auth, "current_caller_verified_agent", return_value="Bauleiter"), \
             patch.object(server, "AI_API_KEY", ""), \
             patch.object(server, "_fetch_json", new=AsyncMock(return_value={})) as f:
            asyncio.run(server.call_ai_api("POST", "/ai/x", extra_headers={"X-Agent-Name": "Anders"}))
        h = f.call_args.kwargs["headers"]
        self.assertEqual(h["X-Agent-Name"], "Anders")
        self.assertEqual(h["Authorization"], "Bearer jwt-123")


if __name__ == "__main__":
    unittest.main()
