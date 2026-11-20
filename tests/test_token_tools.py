"""token_verbrauch und token_auslastung (cloud-MCP, #3532, Post 6351).

Die Berichtsform kommt aus tests/daten/token_bericht_form_3532.json, erzeugt mit der
Server-Funktion (cloud-api usage_verdichtung), nicht von Hand nachgebaut. Gemessen wird,
was durch FastMCP hinausgeht (wie test_zeit_tools)."""
import asyncio
import copy
import json
import unittest
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from token_tools import ZIEL_BYTES, register_token_tools

FORM = json.loads((Path(__file__).parent / "daten" / "token_bericht_form_3532.json").read_text())
# gruppe=aufgabe fuer einen lesenden Agenten, erzeugt mit usage_aufgaben.zusammenstellen + fuer_leser
# (Cloud-Security b2ebb368: die alte, handgebaute Attrappe hatte `anfragen` in `tokens`).
AUFGABE = json.loads((Path(__file__).parent / "daten" / "token_aufgabe_form_3532.json").read_text())


def _geliefert(ergebnis) -> str:
    teile = ergebnis[0] if isinstance(ergebnis, tuple) else ergebnis
    return "".join(getattr(t, "text", "") for t in teile)


def _gross(n: int) -> dict:
    """Die echte Form, mit n Schluesseln im Rang und 48 Zeitpunkten."""
    d = copy.deepcopy(FORM)
    v = d["verdichtung"]
    vorlage = v["rang"][0]
    v["rang"] = [{**vorlage, "schluessel": f"Agent-mit-langem-Namen-{i:05d}", "menge": n - i} for i in range(n)]
    v["stapel"] = [{"t": f"2026-11-{18 + h // 24:02d}T{h % 24:02d}:00:00+00:00", "teile": {}, "summe": h}
                   for h in range(48)]
    return d


def _fusion(n_zeilen: int = 3, n_konten: int = 1) -> dict:
    zeilen = [{"t": f"2026-11-20T{h % 24:02d}:00:00+00:00", "pct_anstieg": 1.0, "pct_gemessen": 0.8,
               "pct_ungemessen": 0.1, "pct_rest": 0.1,
               "anteile": [{"agent": f"A{i}", "node": "mac", "pct": 0.04} for i in range(20)]}
              for h in range(n_zeilen)]
    claude = [{"konto": "arkturian.com@gmail.com" if i == 0 else f"konto{i}@x.at", "konto_uuid": "u",
               "profile": "Profile 32", "quelle": "claude", "fenster": "seven_day", "gewichte": {},
               "eichung": {"quelle": "regression", "k_pct_je_mio_aequivalente": 0.5, "tage": 9,
                           "r2": 0.8, "stand": "s"}, "zeilen": copy.deepcopy(zeilen)} for i in range(n_konten)]
    return {"von": "a", "bis": "b", "takt": "stunde", "gefiltert": True, "lieferungen": {"mac": "x"},
            "konten": claude + [
                       {"konto": "apopovic.aut@gmail.com", "konto_uuid": None, "profile": None, "quelle": "codex",
                        "fenster": None, "gewichte": {}, "eichung": None, "zeilen": []}]}


class Werkzeuge(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.calls = []
        self.antwort = copy.deepcopy(FORM)

        async def transport(method, path, **kwargs):
            self.calls.append((method, path, kwargs))
            return self.antwort
        self.mcp = FastMCP("isolated-token")
        register_token_tools(self.mcp, transport)

    async def _rufe(self, name, args):
        return json.loads(_geliefert(await self.mcp.call_tool(name, args)))

    async def test_verbrauch_ruft_den_bericht_mit_mass_und_wiener_tagen(self):
        d = await self._rufe("token_verbrauch", {})
        m, path, kw = self.calls[0]
        p = kw["params"]
        self.assertEqual((m, path), ("GET", "/api/usage/report"))
        self.assertEqual((p["gruppe"], p["mass"], p["tagesgrenze"], p["takt"]), ("agent", "output", "wien", "stunde"))
        from datetime import datetime
        self.assertEqual((datetime.fromisoformat(p["bis"]) - datetime.fromisoformat(p["von"])).total_seconds(), 86400)
        self.assertTrue(d["gefiltert"])
        self.assertEqual(d["gesamt"], FORM["verdichtung"]["gesamt"])
        self.assertEqual(d["rang"][0]["schluessel"], "A11")
        self.assertEqual(set(d["rang"][0]) >= {"input", "output", "cache_read", "cache_write", "menge", "anteil"}, True)
        self.assertGreater(d["verlauf"][0]["t"], d["verlauf"][-1]["t"], "juengster zuerst")

    async def test_tage_summieren_tokens_getrennt(self):
        d = await self._rufe("token_verbrauch", {"gruppe": "tag", "mass": "eingabe"})
        p = self.calls[0][2]["params"]
        self.assertEqual((p["gruppe"], p["takt"]), ("quelle", "tag"))
        t = d["tage"][0]
        n = sum(1 for z in FORM["zeilen"] if z["t"] == t["t"])
        self.assertEqual((t["input"], t["cache_read"], t["cache_write"]), (10 * n, 1000 * n, 5 * n))
        self.assertEqual(t["menge"], t["input"] + t["cache_read"] + t["cache_write"])

    async def test_projekt_aus_gruppe_aufgabe(self):
        self.antwort = copy.deepcopy(AUFGABE)
        d = await self._rufe("token_verbrauch", {"gruppe": "projekt"})
        self.assertEqual(self.calls[0][2]["params"]["gruppe"], "aufgabe")
        self.assertEqual([(r["projekt"], r["output"], r["anfragen"], r["pp"]) for r in d["rang"]],
                         [("Cloud", 1_000_000, 7, 29.0), ("Portal", 400_000, 4, 19.0),
                          ("ohne Aufgabe", 100_000, 1, 2.5)])
        self.assertTrue(d["gefiltert"])

    async def test_je_aufgabe_mit_pp_und_anfragen(self):
        self.antwort = copy.deepcopy(AUFGABE)
        d = await self._rufe("token_verbrauch", {"gruppe": "aufgabe"})
        self.assertEqual([(r["aufgabe"], r["projekt"], r["anfragen"], r["pp"]) for r in d["rang"]],
                         [("2", "Cloud", 7, 29.0), ("1", "Portal", 4, 19.0), ("ohne", None, 1, 2.5)])

    async def test_kombinierte_gruppe_und_max_reihen(self):
        await self._rufe("token_verbrauch", {"gruppe": "agent, model", "max_reihen": 5})
        p = self.calls[0][2]["params"]
        self.assertEqual((p["gruppe"], p["max_reihen"]), ("agent,model", 5))
        for falsch in ("agent,agent", "agent,tag", "agent,person"):
            self.assertIn("fehler", await self._rufe("token_verbrauch", {"gruppe": falsch}))
        self.assertEqual(len(self.calls), 1)

    async def test_unbekannte_gruppe_oder_mass_ohne_aufruf(self):
        for args in ({"gruppe": "person"}, {"mass": "reasoning"}, {"von": "gestern"}):
            self.assertIn("fehler", await self._rufe("token_verbrauch", args))
        self.assertEqual(self.calls, [])

    async def test_verbrauch_gross_bleibt_unter_20_kb_und_sagt_es(self):
        self.antwort = _gross(3000)
        text = _geliefert(await self.mcp.call_tool("token_verbrauch", {"limit": 1000}))
        self.assertLessEqual(len(text.encode()), ZIEL_BYTES)
        d = json.loads(text)
        self.assertTrue(d["gekuerzt"])
        self.assertIn("Eingrenzen", d["hinweis"])
        self.assertEqual(d["schluessel_gesamt"], 3000)
        self.assertEqual(d["rang"][0]["schluessel"], "Agent-mit-langem-Namen-00000", "die groessten bleiben")
        self.assertIn("rang", d["weggelassen"])

    async def test_klein_ohne_kuerzung(self):
        d = await self._rufe("token_verbrauch", {})
        self.assertNotIn("gekuerzt", d)

    async def test_auslastung_verdichtet_je_konto(self):
        self.antwort = _fusion()
        d = await self._rufe("token_auslastung", {"konto": "arkturian.com@gmail.com"})
        self.assertEqual(self.calls[0][1], "/api/usage/fusion")
        self.assertEqual(self.calls[0][2]["params"]["konto"], "arkturian.com@gmail.com")
        k = d["konten"][0]
        self.assertEqual((k["pct_anstieg"], k["pct_gemessen"]), (3.0, 2.4))
        self.assertEqual(len(k["agenten"]), 15)
        self.assertEqual(k["agenten_weitere"]["anzahl"], 5)
        self.assertEqual(d["konten"][1], {"konto": "apopovic.aut@gmail.com", "quelle": "codex", "profile": None})
        self.assertTrue(d["gefiltert"])

    async def test_auslastung_gross_unter_20_kb(self):
        self.antwort = _fusion(24 * 14, n_konten=3)          # 14 Tage stuendlich, drei Konten
        text = _geliefert(await self.mcp.call_tool("token_auslastung", {"takt": "stunde"}))
        self.assertLessEqual(len(text.encode()), ZIEL_BYTES)
        d = json.loads(text)
        self.assertTrue(d["gekuerzt"])
        self.assertEqual(d["konten"][0]["verlauf"][0][0], "2026-11-20T23:00:00+00:00", "juengster bleibt")

    async def test_fehlerantwort_wird_durchgereicht(self):
        self.antwort = {"detail": "nur fuer Admins"}
        self.assertEqual(await self._rufe("token_verbrauch", {}), {"detail": "nur fuer Admins"})
        self.assertEqual(await self._rufe("token_auslastung", {}), {"detail": "nur fuer Admins"})


class ImServer(unittest.TestCase):
    def test_im_cloud_mcp_registriert(self):
        import os
        os.environ.setdefault("MCP_SERVERS", "cloud")
        import server
        namen = {t.name for t in asyncio.run(server.cloud_mcp.list_tools())}
        self.assertIn("token_verbrauch", namen)
        self.assertIn("token_auslastung", namen)
        self.assertIn("token_lage", namen)


if __name__ == "__main__":
    unittest.main()
