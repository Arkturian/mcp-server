"""send_message als Rundnachricht (#2765, Beitrag 5921): `to_session` als Liste und
`antwort`. Der Server prueft jeden Empfaenger einzeln; das Werkzeug reicht nur durch.
Ohne die neuen Argumente bleibt der Aufruf wie bisher."""
from __future__ import annotations
import asyncio
import os
import unittest
from unittest.mock import AsyncMock, patch

os.environ.setdefault("MCP_SERVERS", "cloud")
import server  # noqa: E402


class RundnachrichtToolTests(unittest.TestCase):
    def _rufen(self, **kw):
        with patch.object(server, "call_cloud_api", new=AsyncMock(return_value={"gesendet": 2})) as call:
            out = asyncio.run(server.cloud_send_message(**kw))
        return out, call

    def test_liste_und_antwort_gehen_an_route(self):
        out, call = self._rufen(from_session="Cloud", to_session=[" Login", "Comm", ""],
                                message="Bitte X pruefen", timeout=0, antwort="keine")
        args, kwargs = call.call_args
        self.assertEqual(args, ("POST", "/api/agents/route"))
        self.assertEqual(kwargs["json_body"]["to"], ["Login", "Comm"])
        self.assertEqual(kwargs["json_body"]["antwort"], "keine")
        self.assertEqual(out, {"gesendet": 2})

    def test_einzelner_empfaenger_wie_bisher(self):
        _, call = self._rufen(from_session="Cloud", to_session="Login", message="hallo", timeout=0)
        body = call.call_args.kwargs["json_body"]
        self.assertEqual(body, {"from": "Cloud", "to": "Login", "message": "hallo", "wait": False, "timeout": 0})

    def test_schema_nimmt_liste_und_antwort(self):
        tools = {t.name: t for t in asyncio.run(server.cloud_mcp.list_tools())}
        schema = tools["send_message"].inputSchema
        self.assertIn("antwort", schema["properties"])
        to = schema["properties"]["to_session"]
        typen = {v.get("type") for v in to.get("anyOf", [to])}
        self.assertTrue({"string", "array"} <= typen, to)


if __name__ == "__main__":
    unittest.main()
