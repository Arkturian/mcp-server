"""browser_anfrage / browser_ergebnis (Alex 10.11.2026, Post 5856)."""
from __future__ import annotations
import asyncio
import os
import unittest
from unittest.mock import AsyncMock, patch

os.environ.setdefault("MCP_SERVERS", "cloud")
import server  # noqa: E402


class BrowserAnfrageToolTests(unittest.TestCase):
    def test_anfrage_geht_an_cloud_api(self):
        with patch.object(server, "call_cloud_api", new=AsyncMock(return_value={"status": "offen"})) as call:
            asyncio.run(server.cloud_browser_anfrage("abfrage", "Formular pruefen", "form input"))
        self.assertEqual(call.call_args.args, ("POST", "/api/browser/requests"))
        self.assertEqual(call.call_args.kwargs["json_body"],
                         {"art": "abfrage", "grund": "Formular pruefen", "selector": "form input"})
        self.assertGreaterEqual(call.call_args.kwargs["timeout"], 30)

    def test_ohne_selector_kein_feld(self):
        with patch.object(server, "call_cloud_api", new=AsyncMock(return_value={})) as call:
            asyncio.run(server.cloud_browser_anfrage("screenshot", "Layout ansehen"))
        self.assertNotIn("selector", call.call_args.kwargs["json_body"])

    def test_datei_mit_adresse(self):
        with patch.object(server, "call_cloud_api", new=AsyncMock(return_value={})) as call:
            asyncio.run(server.cloud_browser_anfrage("datei", "PDF lesen", adresse="https://x.at/a.pdf"))
        self.assertEqual(call.call_args.kwargs["json_body"]["adresse"], "https://x.at/a.pdf")

    def test_sitzung_nur_wenn_nicht_aktiv(self):
        """#2868: sitzung='agenten' bzw. eine sitzung_id geht mit; 'aktiv' bleibt der alte Rumpf."""
        with patch.object(server, "call_cloud_api", new=AsyncMock(return_value={})) as call:
            asyncio.run(server.cloud_browser_anfrage("seite", "Lesen", sitzung="agenten"))
        self.assertEqual(call.call_args.kwargs["json_body"]["sitzung"], "agenten")
        with patch.object(server, "call_cloud_api", new=AsyncMock(return_value={})) as call:
            asyncio.run(server.cloud_browser_anfrage("seite", "Lesen"))
        self.assertNotIn("sitzung", call.call_args.kwargs["json_body"])

    def test_beschreibungen_nennen_den_agenten_chrome(self):
        tools = {t.name: t for t in asyncio.run(server.cloud_mcp.list_tools())}
        for name in ("browser_anfrage", "browser_sitzungen", "browser_freigabe", "browser_aufruf",
                     "browser_freigabe_beenden"):
            self.assertIn("agenten", tools[name].description, name)
        self.assertIn("agenten_chrome", tools["browser_sitzungen"].description)
        self.assertIn("eigener_tab_noetig", tools["browser_aufruf"].description)

    def test_ergebnis_liest_eigene_anfrage(self):
        with patch.object(server, "call_cloud_api", new=AsyncMock(return_value={"status": "fertig"})) as call:
            asyncio.run(server.cloud_browser_ergebnis("abc123"))
        self.assertEqual(call.call_args.args, ("GET", "/api/browser/requests/abc123"))


if __name__ == "__main__":
    unittest.main()
