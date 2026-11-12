"""zeit_tag (cloud-MCP, Post #5880): Tagesantwort von cloud-api in Summen.

Die Groessengrenze wird gegen einen ERZEUGTEN Tag belegt (fester Seed), nie gegen
echte Daten: Agentennamen, Ordner und Zeiten einer echten Antwort sind Kunden- und
Arbeitsdaten und gehoeren in kein Repository, auch nicht pseudonymisiert (Cloud
11.11.2026). Der Erzeuger hat die Form der echten Antwort vom 11.11. bis 22:17:
1882 Zuege, 374 Bloecke, 122 Abschnitte, 71 Ordner, 36 Agenten, Feldlaengen wie
dort oder laenger (Namen, Pfade, Zahlen), also eher zu gross als zu klein.
"""
from __future__ import annotations

import asyncio
import json
import random
import unittest
from datetime import datetime, timedelta, timezone

from mcp.server.fastmcp import FastMCP

from zeit_tools import (MAX_AGENTEN, MAX_ZUEGE, TOP_ORDNER, ZIEL_BYTES, register_zeit_tools,
                        tag_zusammenfassen)

TAG_BEGINN = datetime(2026, 11, 10, 23, 0, tzinfo=timezone.utc)      # 11.11. 00:00 in Wien
ARTEN = (("agent", 619), ("mensch", 521), ("direkt", 487), (None, 136), ("system", 95), ("queue", 24))


def _iso(t: datetime) -> str:
    return t.isoformat(timespec="microseconds")


def _laenge(text: str, n: int, fuell: str = "x") -> str:
    return (text + "-" + fuell * n)[:n] if len(text) < n else text


def erzeugter_tag(zuege: int = 1882, bloecke: int = 374, abschnitte: int = 122, ordner: int = 71,
                  agenten: int = 36, seed: int = 11112026) -> dict:
    """Ein voller Tag in der Form von GET /api/usage/tag, ohne echte Daten."""
    r = random.Random(seed)
    namen = [_laenge(f"agent-{i + 1:02d}", 8 + (i * 7) % 12) for i in range(agenten)]   # 8..19 Zeichen
    gewicht = [1 / (i + 1) ** 1.1 for i in range(agenten)]                            # wenige viel, viele wenig
    personen = ["person1@beispiel.test", "person2@beispiel.test"]
    arten = [a for a, _ in ARTEN]
    zl = []
    for _ in range(zuege):
        agent = r.choices(namen, gewicht)[0]
        art = r.choices(arten, [n for _, n in ARTEN])[0]
        beginn = TAG_BEGINN + timedelta(seconds=r.uniform(0, 22 * 3600))
        dauer = None if r.random() < 0.001 else min(9000.0, r.lognormvariate(3.0, 1.6))
        tokens = None if r.random() < 0.001 else {
            "anfragen": r.randint(1, 40), "input": r.randint(0, 99_999), "output": int(r.lognormvariate(7.2, 1.0)),
            "cache_read": r.randint(0, 24_999_999), "cache_write": r.randint(0, 999_999), "reasoning": r.randint(0, 9_999)}
        hinzu = r.randint(1, 814) if r.random() < 0.05 else 0
        zl.append({"turn_id": f"p3:{r.getrandbits(128):032x}"[:39], "agent": agent, "node": "knoten-01",
                   "beginn": _iso(beginn), "ende": _iso(beginn + timedelta(seconds=dauer or 0)),
                   "dauer_s": dauer, "ausloeser_art": art,
                   "ausloeser": r.choice(personen) if art == "mensch" else (r.choice(namen) if art == "agent" else None),
                   "zeilen": {"dateien": 1 if hinzu else 0, "hinzu": hinzu, "geaendert": 0, "entfernt": 0},
                   "tokens": tokens})
    zl.sort(key=lambda z: z["beginn"])
    bl = [{"quelle": r.choice(("portal", "extension")), "geraet": f"geraet{r.randint(1, 6)}",
           "beginn": _iso(TAG_BEGINN + timedelta(minutes=900 + i * 3)),
           "ende": _iso(TAG_BEGINN + timedelta(minutes=902 + i * 3)), "zustand": "aktiv",
           "kontext": {"agenten": [r.choice(namen)], "host": f"host{r.randint(1, 12)}.beispiel.test"},
           "herkunft": "gemessen", "mandant": "", } for i in range(bloecke)]
    ab = [{"beginn": _iso(TAG_BEGINN + timedelta(minutes=900 + i * 8)),
           "ende": _iso(TAG_BEGINN + timedelta(minutes=905 + i * 8)),
           "zustand": "passiv" if i % 20 == 19 else "aktiv"} for i in range(abschnitte)]
    od = sorted(({"ordner": _laenge(f"/home/nutzer/code/projekt-{i + 1:02d}/src", 32 + (i * 13) % 46, "y"),
                  "zuege": r.randint(1, 17), "agenten": r.randint(1, 2), "dateien": r.randint(1, 26),
                  "hinzu": r.randint(0, 1100), "geaendert": r.randint(0, 99), "entfernt": r.randint(0, 99)}
                 for i in range(ordner)), key=lambda o: -(o["hinzu"] + o["geaendert"] + o["entfernt"]))
    return {"person": personen[0], "datum": "2026-11-11", "zeitzone": "Europe/Vienna",
            "von": _iso(TAG_BEGINN), "bis": _iso(TAG_BEGINN + timedelta(days=1)), "hinweis": None,
            "mensch": {"bloecke": bl, "vereinigt": {"aktiv_s": 16783, "passiv_s": 45, "abschnitte": ab}},
            "agenten": zl, "ordner": od,
            "summen": {"mensch_aktiv_s": 16783, "mensch_passiv_s": 45, "zuege": zuege, "zuege_von_person": 153,
                       "agenten": agenten, "agent_s": 119397,
                       "tokens": {"anfragen": 7546, "input": 4547259, "output": 5071074, "cache_read": 2985283558,
                                  "cache_write": 29194117, "reasoning": 158709},
                       "zeilen": {"dateien": 194, "hinzu": 14976, "geaendert": 255, "entfernt": 61}}}


def _bytes(o) -> int:
    return len(json.dumps(o, ensure_ascii=False).encode("utf-8"))


def _zug(tid, agent, beginn, dauer, art=None, wer=None, output=0, hinzu=0):
    return {"turn_id": tid, "agent": agent, "node": "mac", "beginn": beginn, "ende": None,
            "dauer_s": dauer, "ausloeser_art": art, "ausloeser": wer,
            "zeilen": {"dateien": 1, "hinzu": hinzu, "geaendert": 0, "entfernt": 0},
            "tokens": {"anfragen": 1, "input": 10, "output": output, "cache_read": 0,
                       "cache_write": 0, "reasoning": 0}}


def _kleiner_tag(zuege=None, ordner=None) -> dict:
    return {
        "person": "alex@x.at", "datum": "2026-11-11", "zeitzone": "Europe/Vienna",
        "von": "2026-11-10T23:00:00+00:00", "bis": "2026-11-11T23:00:00+00:00", "hinweis": None,
        "mensch": {"bloecke": [{"quelle": "portal"}],
                   "vereinigt": {"aktiv_s": 600, "passiv_s": 60, "abschnitte": [
                       {"beginn": "2026-11-11T14:00:00+00:00", "ende": "2026-11-11T14:10:00+00:00",
                        "zustand": "aktiv"}]}},
        "agenten": zuege if zuege is not None else [
            _zug("u1", "A", "2026-11-11T08:00:00+00:00", 60, "mensch", "alex@x.at", output=5, hinzu=3),
            _zug("u2", "A", "2026-11-11T09:00:00+00:00", 30, "agent", "Cloud", output=1),
            _zug("u3", "B", "2026-11-11T10:00:00+00:00", 300, "mensch", "tommy@x.at")],
        "ordner": ordner if ordner is not None else [{"ordner": "/r", "zuege": 1, "agenten": 1, "dateien": 1,
                                                      "hinzu": 3, "geaendert": 0, "entfernt": 0}],
        "summen": {"zuege": 3},
    }


class Verdichten(unittest.TestCase):
    def test_agenten_je_name_mit_summen(self):
        out = tag_zusammenfassen(_kleiner_tag())
        self.assertEqual([a["agent"] for a in out["agenten"]], ["B", "A"])        # nach Dauer
        a = out["agenten"][1]
        self.assertEqual((a["zuege"], a["von_person"], a["dauer_s"], a["zeilen"]), (2, 1, 90, 3))
        self.assertEqual(a["tokens"]["output"], 6)
        self.assertEqual(out["agenten"][0]["von_person"], 0)                       # Tommys Zug zaehlt nicht

    def test_mensch_als_vereinigung_in_ortszeit(self):
        m = tag_zusammenfassen(_kleiner_tag())["mensch"]
        self.assertEqual(m["abschnitte"], [["15:00", "15:10", "aktiv"]])          # UTC+1 im November
        self.assertEqual((m["aktiv_s"], m["passiv_s"], m["bloecke"]), (600, 60, 1))

    def test_top_ordner_und_gesamtzahl(self):
        ordner = [{"ordner": f"/o{i}", "zuege": 1, "agenten": 1, "dateien": 1, "hinzu": 100 - i,
                   "geaendert": 0, "entfernt": 0} for i in range(25)]
        out = tag_zusammenfassen(_kleiner_tag(ordner=ordner))
        self.assertEqual([o["ordner"] for o in out["ordner"]], [f"/o{i}" for i in range(TOP_ORDNER)])
        self.assertEqual(out["ordner_gesamt"], 25)

    def test_zuege_nur_auf_wunsch(self):
        self.assertNotIn("zuege", tag_zusammenfassen(_kleiner_tag()))
        out = tag_zusammenfassen(_kleiner_tag(), zuege=True)
        self.assertEqual([z["agent"] for z in out["zuege"]], ["A", "A", "B"])
        self.assertEqual(out["zuege_hinweis"], "alle 3 Zuege")
        self.assertEqual(out["zuege"][0]["beginn"], "09:00:00")

    def test_zuege_gekuerzt_auf_die_laengsten(self):
        viele = [_zug(f"u{i}", "A", f"2026-11-11T{8 + i // 60:02d}:{i % 60:02d}:00+00:00", i)
                 for i in range(MAX_ZUEGE + 30)]
        out = tag_zusammenfassen(_kleiner_tag(zuege=viele), zuege=True)
        self.assertEqual(len(out["zuege"]), MAX_ZUEGE)
        self.assertEqual(min(z["dauer_s"] for z in out["zuege"]), 30)               # die 50 laengsten
        self.assertEqual([z["dauer_s"] for z in out["zuege"]], list(range(30, 80)))  # nach Beginn = nach Dauer hier
        self.assertEqual(out["zuege_hinweis"], f"{MAX_ZUEGE} von {MAX_ZUEGE + 30} Zuegen, "
                                               "die laengsten, nach Beginn sortiert")


class VollerTag(unittest.TestCase):
    """Groessengrenze gegen einen erzeugten vollen Tag (Cloud 11.11.2026: unter ~20 KB)."""

    @classmethod
    def setUpClass(cls):
        cls.tag = erzeugter_tag()

    def test_erzeuger_hat_die_form_eines_vollen_tages(self):
        self.assertEqual(len(self.tag["agenten"]), 1882)
        self.assertEqual(len(self.tag["mensch"]["bloecke"]), 374)
        self.assertEqual(len(self.tag["ordner"]), 71)
        self.assertEqual(len({z["agent"] for z in self.tag["agenten"]}), 36)
        self.assertGreater(_bytes(self.tag), 800_000)                     # echt: 964 KB
        self.assertEqual(erzeugter_tag(), self.tag)                        # fester Seed

    def test_doppelt_so_viele_agenten_bleiben_unter_der_grenze(self):
        """Die Grenze darf nicht an der Zahl der Agenten haengen."""
        tag = erzeugter_tag(agenten=72, zuege=3000, seed=7)
        out = tag_zusammenfassen(tag, zuege=True)
        self.assertEqual(len(out["agenten"]), MAX_AGENTEN + 1)
        self.assertLess(_bytes(out), ZIEL_BYTES)

    def test_summen_unter_der_grenze(self):
        self.assertLess(_bytes(tag_zusammenfassen(self.tag)), ZIEL_BYTES)

    def test_mit_zuegen_unter_der_grenze(self):
        out = tag_zusammenfassen(self.tag, zuege=True)
        self.assertEqual(len(out["zuege"]), MAX_ZUEGE)
        self.assertIn(f"von {len(self.tag['agenten'])} Zuegen", out["zuege_hinweis"])
        self.assertLess(_bytes(out), ZIEL_BYTES)

    def test_nichts_geht_beim_verdichten_verloren(self):
        out = tag_zusammenfassen(self.tag)
        self.assertEqual(len(out["agenten"]), MAX_AGENTEN + 1)
        self.assertEqual(out["agenten"][-1]["agent"], f"+{36 - MAX_AGENTEN} weitere")
        self.assertEqual(sum(a["zuege"] for a in out["agenten"]), len(self.tag["agenten"]))
        self.assertEqual(sum(a["zeilen"] for a in out["agenten"]),
                         sum(sum(z["zeilen"].values()) - z["zeilen"]["dateien"] for z in self.tag["agenten"]))
        self.assertEqual(out["summen"], self.tag["summen"])
        self.assertEqual(len(out["mensch"]["abschnitte"]),
                         len(self.tag["mensch"]["vereinigt"]["abschnitte"]))


def _geliefert(ergebnis) -> str:
    """Der Text, den der Agent wirklich bekommt (call_tool liefert je nach mcp-Version
    eine Liste oder ein Paar aus Liste und strukturiertem Inhalt)."""
    teile = ergebnis[0] if isinstance(ergebnis, tuple) else ergebnis
    return "".join(getattr(t, "text", "") for t in teile)


class GeliefertesFormat(unittest.IsolatedAsyncioTestCase):
    """Gemessen wird, was durch FastMCP hinausgeht, nicht ein json.dumps im Test:
    Am 11.11. lag die Pruefung bei 17,4 KB, geliefert wurden 27,5 KB (eingerueckt)."""

    async def asyncSetUp(self):
        tag = erzeugter_tag()

        async def transport(method, path, **kwargs):
            return tag
        self.mcp = FastMCP("isolated-zeit-format")
        register_zeit_tools(self.mcp, transport)

    async def test_geliefert_unter_der_grenze(self):
        for zuege in (False, True):
            text = _geliefert(await self.mcp.call_tool("zeit_tag", {"person": "p@x.at", "zuege": zuege}))
            self.assertLess(len(text.encode("utf-8")), ZIEL_BYTES, f"zuege={zuege}")

    async def test_geliefert_ist_kompaktes_json(self):
        text = _geliefert(await self.mcp.call_tool("zeit_tag", {"person": "p@x.at"}))
        self.assertNotIn("\n", text)
        self.assertEqual(json.loads(text)["summen"]["zuege"], 1882)


class Werkzeug(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.calls = []
        self.antwort = _kleiner_tag()

        async def transport(method, path, **kwargs):
            self.calls.append((method, path, kwargs))
            return self.antwort

        self.mcp = FastMCP("isolated-zeit")
        register_zeit_tools(self.mcp, transport)

    async def test_ruft_die_tagesroute_mit_person_und_datum(self):
        await self.mcp.call_tool("zeit_tag", {"datum": " 2026-11-11 ", "person": "alex@x.at"})
        self.assertEqual(self.calls, [("GET", "/api/usage/tag",
                                       {"params": {"datum": "2026-11-11", "person": "alex@x.at"},
                                        "timeout": 90})])

    async def test_leere_angaben_gehen_nicht_mit(self):
        await self.mcp.call_tool("zeit_tag", {})
        self.assertEqual(self.calls[0][2]["params"], {})

    async def test_fremde_antwort_unveraendert(self):
        self.antwort = {"detail": "x"}
        out = str(await self.mcp.call_tool("zeit_tag", {"person": "a@b.at"}))
        self.assertIn("detail", out)
        self.assertNotIn("agenten", out)                 # nicht verdichtet, nichts erfunden

    async def test_beschreibung_nennt_personendaten_und_pflicht_person(self):
        tool = {t.name: t for t in await self.mcp.list_tools()}["zeit_tag"]
        self.assertIn("Personendatum", tool.description)
        self.assertIn("`person`", tool.description)


def _quellen():
    """`quellen` in der Form von cloud-api (Post #5898, Nachtrag Zugriffsregel)."""
    return [{"foederation": "arkturian", "status": "ok", "umfang": "alles"},
            {"foederation": "p3", "status": "ok", "umfang": "eigene", "bloecke": 12, "zuege": 70},
            {"foederation": "ele", "status": "verweigert", "grund": "usage_lesen_aus"},
            {"foederation": "x", "status": "keine_antwort", "grund": "keine Antwort in 20 s"}]


class Foederationen(unittest.IsolatedAsyncioTestCase):
    """Andere Foederationen ueber die Bruecke (Post #5898)."""

    async def asyncSetUp(self):
        self.calls = []
        self.antwort = {**erzeugter_tag(), "quellen": _quellen()}

        async def transport(method, path, **kwargs):
            self.calls.append(kwargs)
            return self.antwort
        self.mcp = FastMCP("isolated-zeit-foed")
        register_zeit_tools(self.mcp, transport)

    async def test_foederationen_geht_mit(self):
        await self.mcp.call_tool("zeit_tag", {"person": "p@x.at", "foederationen": " p3 "})
        self.assertEqual(self.calls[0]["params"], {"person": "p@x.at", "foederationen": "p3"})

    async def test_quellen_mit_status_und_grund(self):
        q = json.loads(_geliefert(await self.mcp.call_tool("zeit_tag", {"person": "p@x.at"})))["quellen"]
        self.assertEqual([(x["foederation"], x["status"]) for x in q],
                         [("arkturian", "ok"), ("p3", "ok"), ("ele", "verweigert"), ("x", "keine_antwort")])
        self.assertEqual(q[2]["grund"], "usage_lesen_aus")
        self.assertEqual(q[1], {"foederation": "p3", "status": "ok", "umfang": "eigene", "bloecke": 12, "zuege": 70})

    async def test_umfang_geht_mit(self):
        self.antwort = {**erzeugter_tag(), "umfang": "alles"}
        self.assertEqual(json.loads(_geliefert(await self.mcp.call_tool("zeit_tag", {"person": "p@x.at"})))["umfang"],
                         "alles")

    async def test_beschreibung_nennt_das_recht(self):
        tool = {t.name: t for t in await self.mcp.list_tools()}["zeit_tag"]
        self.assertIn("zeit:lesen", tool.description)
        self.assertNotIn("benannten Agenten", tool.description)

    async def test_mit_quellen_unter_der_grenze(self):
        for zuege in (False, True):
            text = _geliefert(await self.mcp.call_tool("zeit_tag", {"person": "p@x.at", "zuege": zuege}))
            self.assertLess(len(text.encode("utf-8")), ZIEL_BYTES, f"zuege={zuege}")

    async def test_ohne_quellen_kein_feld(self):
        self.antwort = erzeugter_tag()
        self.assertNotIn("quellen", json.loads(_geliefert(await self.mcp.call_tool("zeit_tag", {"person": "p@x.at"}))))


class ImServer(unittest.TestCase):
    def test_im_cloud_mcp_registriert(self):
        import os
        os.environ.setdefault("MCP_SERVERS", "cloud")
        import server
        namen = {t.name for t in asyncio.run(server.cloud_mcp.list_tools())}
        self.assertIn("zeit_tag", namen)
        self.assertIn("token_lage", namen)


if __name__ == "__main__":
    unittest.main()
