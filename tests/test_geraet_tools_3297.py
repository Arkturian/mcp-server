"""geraet_sitzungen / geraet_aufruf / geraet_ergebnis / geraet_mitteilung (#3297, Vertrag Beitrag 6215).
Duenne Huellen um die Agenten-Routen der cloud-api; die Pruefung macht die cloud-api."""
from __future__ import annotations

import asyncio
import os
import unittest
from unittest.mock import AsyncMock, patch

os.environ.setdefault("MCP_SERVERS", "cloud")
import server  # noqa: E402


class GeraetToolTests(unittest.TestCase):
    def _ruf(self, coro):
        with patch.object(server, "call_cloud_api", new=AsyncMock(return_value={})) as call:
            asyncio.run(coro())
        return call.call_args

    def test_sitzungen(self):
        a = self._ruf(lambda: server.cloud_geraet_sitzungen())
        self.assertEqual(a.args, ("GET", "/api/agents/self/geraete"))

    def test_aufruf(self):
        a = self._ruf(lambda: server.cloud_geraet_aufruf("datei_zeigen", {"storage_id": 7}))
        self.assertEqual(a.args, ("POST", "/api/agents/self/geraet-aufrufe"))
        self.assertEqual(a.kwargs["json_body"], {"methode": "datei_zeigen", "parameter": {"storage_id": 7}})
        self.assertGreaterEqual(a.kwargs["timeout"], 40)
        a = self._ruf(lambda: server.cloud_geraet_aufruf("geraet_info", None, "8d3f2c1a-0b4e-4c7d-9a1f-2e3d4c5b6a7f"))
        self.assertEqual(a.kwargs["json_body"]["geraet"], "8d3f2c1a-0b4e-4c7d-9a1f-2e3d4c5b6a7f")

    def test_ergebnis(self):
        a = self._ruf(lambda: server.cloud_geraet_ergebnis("a1b2c3d4e5f6"))
        self.assertEqual(a.args, ("GET", "/api/agents/self/geraet-aufrufe/a1b2c3d4e5f6"))

    def test_mitteilung(self):
        a = self._ruf(lambda: server.cloud_geraet_mitteilung("Angebot", "Bitte ansehen", 77, ["oeffnen", "spaeter"]))
        self.assertEqual(a.args, ("POST", "/api/agents/self/geraet-mitteilung"))
        self.assertEqual(a.kwargs["json_body"], {"titel": "Angebot", "text": "Bitte ansehen",
                                                 "datei": {"storage_id": 77}, "aktionen": ["oeffnen", "spaeter"]})
        a = self._ruf(lambda: server.cloud_geraet_mitteilung("Hinweis", "Fertig"))
        self.assertEqual(a.kwargs["json_body"], {"titel": "Hinweis", "text": "Fertig"})

    def test_beschreibung_nennt_die_methoden(self):
        tools = asyncio.run(server.cloud_mcp.list_tools())
        beschr = {t.name: t.description for t in tools}["geraet_aufruf"]
        for m in ("datei_zeigen", "datei_oeffnen", "ordner_zeigen", "url_oeffnen", "mitteilung",
                  "zwischenablage_setzen", "geraet_info", "status=offen", "geraet_waehlen", "kein_zugriff"):
            self.assertIn(m, beschr)


if __name__ == "__main__":
    unittest.main()
