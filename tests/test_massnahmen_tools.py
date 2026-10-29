"""Massnahmen- und Regel-Werkzeuge (cloud-MCP, Alex 29.10.2026): reichen nur
weiter, Rechte prueft cloud-api."""
from __future__ import annotations
import asyncio
import os
import unittest
from unittest.mock import AsyncMock, patch

os.environ.setdefault("MCP_SERVERS", "cloud")
import server  # noqa: E402


def _ruf(fn, *a, **k):
    with patch.object(server, "call_cloud_api", new=AsyncMock(return_value={"ok": True})) as call:
        asyncio.run(fn(*a, **k))
    return call.call_args


class MassnahmenToolTests(unittest.TestCase):
    def test_anlegen_agenten(self):
        c = _ruf(server.cloud_massnahme_create, "konto_umstellen", "Kontowechsel", agenten=["A"],
                 ziel_profil="Profile 7")
        self.assertEqual(c.args, ("POST", "/api/system/massnahmen"))
        self.assertEqual(c.kwargs["json_body"], {"aktion": "konto_umstellen", "grund": "Kontowechsel",
                                                 "ausloeser": "zugende", "agenten": ["A"],
                                                 "ziel_profil": "Profile 7"})

    def test_anlegen_ganzer_knoten(self):
        c = _ruf(server.cloud_massnahme_create, "tmux_neustart", "Wartung", whole_node=True)
        self.assertTrue(c.kwargs["json_body"]["knoten"])
        self.assertNotIn("agenten", c.kwargs["json_body"])

    def test_steuern_und_lesen_kodiert(self):
        self.assertEqual(_ruf(server.cloud_massnahme_release, "m-1").args,
                         ("POST", "/api/system/massnahmen/m-1/freigeben"))
        self.assertEqual(_ruf(server.cloud_massnahme_cancel, "../x").args[1],
                         "/api/system/massnahmen/..%2Fx/abbrechen")
        self.assertEqual(_ruf(server.cloud_massnahme_get, "m-2").args, ("GET", "/api/system/massnahmen/m-2"))

    def test_liste_lage_regeln(self):
        self.assertEqual(_ruf(server.cloud_massnahmen_list).kwargs["params"], {"knoten": "alle"})
        self.assertIsNone(_ruf(server.cloud_agent_lage).kwargs["params"])
        c = _ruf(server.cloud_regeln_set, {"token_erneuern": {"modus": "melden"}}, node="mac")
        self.assertEqual(c.args, ("PUT", "/api/system/regeln"))
        self.assertEqual(c.kwargs["params"], {"knoten": "mac"})
        self.assertIsNone(_ruf(server.cloud_regeln_get).kwargs["params"])

    def test_aufrufer_token_nicht_gateway(self):
        """Rechte haengen am Aufrufer: kein Gateway-Nachweis, der eigene JWT geht mit."""
        self.assertNotIn("agent_gateway_proof", _ruf(server.cloud_massnahmen_list).kwargs)


if __name__ == "__main__":
    unittest.main()
