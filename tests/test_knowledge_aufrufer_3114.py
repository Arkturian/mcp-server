"""#3114 (#3002 Teil 3): call_knowledge_api reicht den Aufrufer-JWT an knowledge-api.

Bisher headers={} ("no auth required"): knowledge-api sah bei keinem MCP-Aufruf,
wer fragt, und konnte nicht auf den Mandanten begrenzen. Kein echter Aufruf:
_fetch_json ist eine Attrappe.
"""
from __future__ import annotations

import asyncio
import os
import unittest
from unittest.mock import AsyncMock, patch

os.environ.setdefault("MCP_SERVERS", "knowledge")
import auth  # noqa: E402
import server  # noqa: E402


def _rufen(jwt="", agent=""):
    with patch.object(auth, "current_caller_jwt", return_value=jwt), \
         patch.object(auth, "current_caller_agent_name", return_value=agent), \
         patch.object(auth, "current_caller_verified_agent", return_value=agent), \
         patch.object(server, "IACP_TOKEN", "gateway-nachweis"), \
         patch.object(server, "_fetch_json", new=AsyncMock(return_value={"ok": True})) as f:
        asyncio.run(server.call_knowledge_api("GET", "/knowledge/query", params={"q": "x"}))
    return f.call_args


class KnowledgeAufruferTests(unittest.TestCase):
    def test_mit_aufrufer_geht_der_bearer_mit(self):
        a = _rufen("jwt-abc", "Tschepp")
        self.assertEqual(a.args[1], f"{server.KNOWLEDGE_API_BASE}/knowledge/query")
        kopf = a.kwargs["headers"]
        self.assertEqual(kopf["Authorization"], "Bearer jwt-abc")
        self.assertEqual(kopf.get("X-Agent-Name"), "Tschepp")

    def test_ohne_aufrufer_kein_kopf(self):
        self.assertEqual(_rufen().kwargs["headers"], {})

    def test_nie_ein_schluessel_und_nie_der_gateway_nachweis(self):
        for jwt, agent in (("jwt-abc", "Tschepp"), ("", ""), ("jwt-mensch", "")):
            kopf = _rufen(jwt, agent).kwargs["headers"]
            self.assertNotIn("X-API-KEY", kopf)
            self.assertNotIn("X-IACP-Token", kopf)
            self.assertNotIn("X-Agent-Gateway", kopf)
            self.assertNotIn("gateway-nachweis", " ".join(kopf.values()))

    def test_benutzer_token_nennt_keinen_agenten(self):
        """#2890: nur ein Agenten-Token nennt einen Agenten."""
        kopf = _rufen("jwt-mensch", "").kwargs["headers"]
        self.assertEqual(kopf, {"Authorization": "Bearer jwt-mensch"})


if __name__ == "__main__":
    unittest.main()
