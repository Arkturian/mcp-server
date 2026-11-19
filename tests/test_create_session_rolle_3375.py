"""#3375 (Cloud 2b4b29f5, Fall Mia 18.11.): create_session hatte role="developer" und
department="engineering" als Vorgabe. Jetzt: ohne Angabe geht kein role/department hinaus
(cloud-api setzt die Vorgabe des Mandanten), role="" geht als "" hinaus (keine Rolle)."""
from __future__ import annotations

import asyncio
import os
import unittest
from unittest.mock import AsyncMock, patch

os.environ.setdefault("MCP_SERVERS", "cloud")
import server  # noqa: E402


def _rumpf(**kw):
    call = AsyncMock(return_value={"status": "created", "name": "Mia"})
    with patch.object(server, "call_cloud_api", new=call):
        asyncio.run(server.cloud_create_session("Mia", **kw))
    method, endpoint = call.call_args.args
    assert (method, endpoint) == ("POST", "/api/sessions")
    return call.call_args.kwargs["json_body"]


class CreateSessionRolleTests(unittest.TestCase):
    def test_ohne_angabe_keine_rolle_und_keine_abteilung_im_rumpf(self):
        rumpf = _rumpf()
        self.assertNotIn("role", rumpf)
        self.assertNotIn("department", rumpf)

    def test_leere_rolle_geht_als_leer_hinaus(self):
        self.assertEqual(_rumpf(role="")["role"], "")

    def test_angegebene_rolle_und_abteilung_unveraendert(self):
        rumpf = _rumpf(role="guide-bot", department="external")
        self.assertEqual((rumpf["role"], rumpf["department"]), ("guide-bot", "external"))

    def test_clone_schickt_keine_rolle(self):
        call = AsyncMock(return_value={"status": "cloned", "name": "Neu"})
        with patch.object(server, "call_cloud_api", new=call):
            asyncio.run(server.cloud_clone_session("Alt", "Neu"))
        rumpf = call.call_args.kwargs["json_body"]
        self.assertNotIn("role", rumpf)
        self.assertNotIn("department", rumpf)


if __name__ == "__main__":
    unittest.main()
