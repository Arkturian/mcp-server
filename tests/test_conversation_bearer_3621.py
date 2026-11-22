"""#3621 (Fund Comm/Conversation 21.11.): conversation-Werkzeuge endeten mit 401 missing_bearer.

call_conversation_api schickte nur X-API-KEY; conversation-api liest aber nur
`Authorization: Bearer` oder ihr Session-Cookie. Jetzt geht das JWT des Aufrufers mit, damit
sie ihre eigenen Rechte- und Besitzpruefungen anwendet. Ohne Aufrufer geht nichts mit: kein
Dienstschluessel, kein Dienst-Token (sonst laese jeder mit den Rechten des Gateways).
"""
from __future__ import annotations

import asyncio
import os
import unittest
from unittest.mock import AsyncMock, patch

os.environ.setdefault("MCP_SERVERS", "conversation")
import auth  # noqa: E402
import server  # noqa: E402


class ConversationBearerTests(unittest.TestCase):
    def _get(self, jwt, agent):
        with patch.object(auth, "current_caller_jwt", return_value=jwt), \
             patch.object(auth, "current_caller_agent_name", return_value=agent), \
             patch.object(auth, "current_caller_verified_agent", return_value=agent), \
             patch.object(server, "CONVERSATION_API_KEY", "alter-schluessel", create=True), \
             patch.object(server, "_fetch_json", new=AsyncMock(return_value={"id": "c_1"})) as f:
            asyncio.run(server.conversation_get("c_1304c46ca105", include_body=False))
        return f.call_args

    def test_werkzeug_reicht_das_jwt_des_aufrufers_weiter(self):
        aufruf = self._get("jwt-comm", "Comm")
        h = aufruf.kwargs["headers"]
        self.assertEqual(h["Authorization"], "Bearer jwt-comm")
        self.assertEqual(h["X-Agent-Name"], "Comm")
        self.assertTrue(aufruf.args[1].endswith("/api/v1/conversations/c_1304c46ca105"))
        self.assertEqual(aufruf.kwargs["params"], {"include_body": "false"})

    def test_kein_schluessel_neben_dem_jwt(self):
        h = self._get("jwt-comm", "Comm").kwargs["headers"]
        self.assertFalse([k for k in h if k.lower() == "x-api-key"], h)

    def test_ohne_aufrufer_keine_anmeldung(self):
        """conversation-api antwortet dann 401; ein Gateway-Ausweis waere der falsche Ersatz."""
        self.assertEqual(self._get(None, None).kwargs["headers"], {})

    def test_gesetzte_umgebungsvariable_wirkt_nicht(self):
        self.assertFalse(hasattr(server, "CONVERSATION_API_KEY"))


if __name__ == "__main__":
    unittest.main()
