import unittest

from mcp.server.fastmcp import FastMCP

from projekt_tools import RECHT, register_projekt_tools


class ProjektToolsTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.calls = []
        self.antworten = {}

        async def transport(method, path, **kwargs):
            self.calls.append((method, path, kwargs))
            return self.antworten.get((method, path), {"ok": True})

        self.mcp = FastMCP("isolated-projekt")
        register_projekt_tools(self.mcp, transport)

    async def rufe(self, name, args):
        return await self.mcp.call_tool(name, args)

    async def test_tool_set(self):
        names = {t.name for t in await self.mcp.list_tools()}
        self.assertEqual(names, {
            "vorgaben_liste", "vorgabe", "vorgabe_anlegen", "projekt_starten", "plan_ueberarbeiten",
            "vorgabe_ablehnen", "projekt_stand", "plan_absegnen", "projektleitung_schalten",
            "projektleitung_antworten", "tor_oeffnen", "meine_aufgaben", "vorgabe_loeschen",
            "paket_fragen", "paket_verlauf", "deliverable", "deliverable_nachtragen", "deliverable_abnehmen",
            "projekt_entscheidung", "projekt_entscheidungen", "projektleitung_absichten",
            "plan_zeichnen", "plan_skizzieren", "projekt_aendern", "spaces_liste", "space_posten",
        })

    async def test_freigeber_tools_carry_the_rule(self):
        tools = {t.name: t for t in await self.mcp.list_tools()}
        for name in ("vorgabe_anlegen", "projekt_starten", "plan_ueberarbeiten", "vorgabe_ablehnen",
                     "plan_absegnen", "projektleitung_schalten", "projektleitung_antworten", "tor_oeffnen",
                     "vorgabe_loeschen", "paket_fragen", "deliverable_nachtragen", "deliverable_abnehmen",
                     "plan_zeichnen", "plan_skizzieren", "projekt_aendern"):
            self.assertIn(RECHT, tools[name].description, name)

    async def test_vorgabe_mit_analyse(self):
        await self.rufe("vorgabe_anlegen", {"text": "Verbrauchsansicht", "analyse_partner": ["Cloud-worker"]})
        self.assertEqual(self.calls, [("POST", "/api/v1/vorgaben", {"json_body": {
            "text": "Verbrauchsansicht", "analyse": {"partner": ["Cloud-worker"], "fragen": []}}})])

    async def test_projekt_starten_nimmt_die_aktuelle_fassung(self):
        self.antworten[("GET", "/api/v1/vorgaben/8")] = {"status": "zur_pruefung", "aktuelle_version": 2,
                                                         "projekt_post_id": None}
        await self.rufe("projekt_starten", {"vorgabe_id": 8})
        self.assertEqual(self.calls[-1], ("POST", "/api/v1/projekt/vorgaben/8/umsetzen",
                                          {"json_body": {"version": 2}}))

    async def test_projekt_starten_segnet_in_der_phase_absegnen_ab(self):
        self.antworten[("GET", "/api/v1/vorgaben/9")] = {"status": "freigegeben", "aktuelle_version": 1,
                                                         "projekt_post_id": 6406, "phase": "absegnen"}
        await self.rufe("projekt_starten", {"vorgabe_id": 9})
        self.assertEqual(self.calls[-1], ("POST", "/api/v1/projekt/projekte/6406/plan/absegnen", {"json_body": {}}))

    async def test_projekt_starten_ohne_plan_sagt_woran_es_haengt(self):
        self.antworten[("GET", "/api/v1/vorgaben/3")] = {"status": "plant", "aktuelle_version": None,
                                                         "projekt_post_id": None}
        await self.rufe("projekt_starten", {"vorgabe_id": 3})
        self.assertEqual([c[0] for c in self.calls], ["GET"])

    async def test_projektleitung_schalten_behaelt_die_intelligenz(self):
        self.antworten[("GET", "/api/v1/vorgaben/10")] = {"leitung": {"modus": "an", "intelligenz": "gruendlich"}}
        await self.rufe("projektleitung_schalten", {"vorgabe_id": 10, "modus": "trocken"})
        self.assertEqual(self.calls[-1], ("PUT", "/api/v1/vorgaben/10/leitung",
                                          {"json_body": {"modus": "trocken", "intelligenz": "gruendlich"}}))

    async def test_projektleitung_schalten_mit_klon_quelle(self):
        await self.rufe("projektleitung_schalten", {"vorgabe_id": 10, "modus": "an", "intelligenz": "normal",
                                                    "quelle_agent": "Marketing"})
        self.assertEqual(self.calls, [("PUT", "/api/v1/vorgaben/10/leitung", {"json_body": {
            "modus": "an", "intelligenz": "normal", "quelle_agent": "Marketing"}})])

    async def test_projektleitung_schalten_mit_art_agent(self):
        await self.rufe("projektleitung_schalten", {"vorgabe_id": 21, "modus": "an", "art": "agent",
                                                    "anbieter": "claude", "modell": "opus", "aufwand": "high"})
        # Mit Art kein Nachlesen der alten Intelligenz: die Art bestimmt Gehirn und Stufe auf dem Server.
        self.assertEqual(self.calls, [("PUT", "/api/v1/vorgaben/21/leitung", {"json_body": {
            "modus": "an", "art": "agent", "anbieter": "claude", "modell": "opus", "aufwand": "high"}})])

    async def test_projektleitung_schalten_zurueck_auf_zustandslos(self):
        await self.rufe("projektleitung_schalten", {"vorgabe_id": 21, "modus": "an", "art": "zustandslos"})
        self.assertEqual(self.calls, [("PUT", "/api/v1/vorgaben/21/leitung", {"json_body": {
            "modus": "an", "art": "zustandslos"}})])

    async def test_projektleitung_schalten_leeres_modell_heisst_vorgabe(self):
        self.antworten[("GET", "/api/v1/vorgaben/21")] = {"leitung": {"intelligenz": "normal"}}
        await self.rufe("projektleitung_schalten", {"vorgabe_id": 21, "modus": "an", "modell": "", "aufwand": ""})
        self.assertEqual(self.calls[-1], ("PUT", "/api/v1/vorgaben/21/leitung", {"json_body": {
            "modus": "an", "modell": "", "aufwand": "", "intelligenz": "normal"}}))

    async def test_projekt_stand_nennt_art_und_anbieter(self):
        self.antworten[("GET", "/api/v1/vorgaben/21")] = {
            "titel": "T", "projekt_post_id": 6509,
            "leitung": {"modus": "an", "gehirn": "sitzung", "art": "agent", "anbieter": "claude", "modell": "opus"}}
        self.antworten[("GET", "/api/v1/projekt/projekte/6509")] = {"aufgaben": []}
        self.antworten[("GET", "/api/v1/projekt/projekte/6509/leitung")] = {"modus": "an", "gehirn": "sitzung"}
        stand = await self.mcp._tool_manager.call_tool("projekt_stand", {"vorgabe_id": 21})
        pl = stand["projektleitung"]
        self.assertEqual((pl["art"], pl["anbieter"], pl["modell"], pl["aufwand"]), ("agent", "claude", "opus", ""))

    async def test_projekt_stand_kuerzt_analyse_und_befund(self):
        lang = "x" * 5000
        phasen = {"phase": "ausfuehrung",
                  "gespraeche": [{"partner": "Cloud", "section_id": "q-1",
                                  "antworten": [{"von": "Cloud", "am": "a", "text": lang},
                                                {"von": "Cloud", "am": "b", "text": "kurz"}]},
                                 {"partner": "Issue", "section_id": "q-2", "antworten": []}],
                  "befund": {"section_id": "p-1", "status": "approved", "text": lang}}
        self.antworten[("GET", "/api/v1/vorgaben/23")] = {"projekt_post_id": 6533, "leitung": {"gehirn": "kette"}}
        self.antworten[("GET", "/api/v1/projekt/projekte/6533")] = {"aufgaben": []}
        self.antworten[("GET", "/api/v1/projekt/projekte/6533/leitung")] = {"gehirn": "kette", "phasen": phasen}
        stand = await self.mcp._tool_manager.call_tool("projekt_stand", {"vorgabe_id": 23})
        g = stand["phasen"]["gespraeche"]
        self.assertEqual((g[0]["partner"], g[0]["section_id"], g[0]["antworten"]), ("Cloud", "q-1", 2))
        self.assertEqual(g[0]["letzte"], {"von": "Cloud", "am": "b", "text": "kurz"})
        self.assertEqual((g[1]["antworten"], g[1]["letzte"]), (0, None))
        self.assertTrue(stand["phasen"]["befund"]["text"].endswith(" …"))
        self.assertLess(len(stand["phasen"]["befund"]["text"]), 900)
        self.assertEqual(stand["phasen"]["befund"]["status"], "approved")
        voll = await self.mcp._tool_manager.call_tool("projekt_stand", {"vorgabe_id": 23, "voll": True})
        self.assertEqual(voll["phasen"], phasen)

    async def test_projekt_stand_alter_rahmen_ohne_art(self):
        self.antworten[("GET", "/api/v1/vorgaben/14")] = {"projekt_post_id": 6465, "leitung": {"gehirn": "kette"}}
        self.antworten[("GET", "/api/v1/projekt/projekte/6465")] = {"aufgaben": []}
        self.antworten[("GET", "/api/v1/projekt/projekte/6465/leitung")] = {"gehirn": "kette"}
        stand = await self.mcp._tool_manager.call_tool("projekt_stand", {"vorgabe_id": 14})
        self.assertEqual(stand["projektleitung"]["art"], "zustandslos")
        self.assertIsNone(stand["projektleitung"]["anbieter"])

    async def test_vorgabe_loeschen(self):
        await self.rufe("vorgabe_loeschen", {"vorgabe_id": 9})
        self.assertEqual(self.calls, [("DELETE", "/api/v1/vorgaben/9", {})])

    async def test_vorgabe_endgueltig_loeschen(self):
        await self.rufe("vorgabe_loeschen", {"vorgabe_id": 9, "endgueltig": True})
        self.assertEqual(self.calls, [("DELETE", "/api/v1/vorgaben/9", {"params": {"endgueltig": "true"}})])

    async def test_paket_fragen_und_verlauf(self):
        self.antworten[("GET", "/api/v1/vorgaben/7")] = {"projekt_post_id": 500}
        self.antworten[("GET", "/api/v1/projekt/projekte/500")] = {"aufgaben": [
            {"plan": "I2", "schluessel": "500/t-2"}]}
        await self.rufe("paket_fragen", {"vorgabe_id": 7, "paket": "I2", "text": "Was soll ich klicken?"})
        self.assertEqual(self.calls[-1], ("POST", "/api/v1/projekt/aufgaben/500/t-2/frage",
                                          {"json_body": {"text": "Was soll ich klicken?"}}))
        await self.rufe("paket_verlauf", {"vorgabe_id": 7, "paket": "I2"})
        self.assertEqual(self.calls[-1], ("GET", "/api/v1/projekt/aufgaben/500/t-2/verlauf", {}))

    async def test_tor_oeffnen_findet_das_paket(self):
        self.antworten[("GET", "/api/v1/vorgaben/7")] = {"projekt_post_id": 500}
        self.antworten[("GET", "/api/v1/projekt/projekte/500")] = {"aufgaben": [
            {"plan": "I1", "schluessel": "500/t-1"}, {"plan": "I2", "schluessel": "500/t-2"}]}
        await self.rufe("tor_oeffnen", {"vorgabe_id": 7, "paket": "I2"})
        self.assertEqual(self.calls[-1], ("POST", "/api/v1/projekt/aufgaben/500/t-2/tor", {"json_body": {}}))


if __name__ == "__main__":
    unittest.main()


class DeliverableToolsTests(ProjektToolsTests):
    async def test_vorgabe_mit_typ_und_deliverable(self):
        await self.rufe("vorgabe_anlegen", {"text": "Neue Ansicht", "typ": "funktion",
                                            "deliverable": "Screenshot der neuen Ansicht"})
        self.assertEqual(self.calls[-1], ("POST", "/api/v1/vorgaben", {"json_body": {
            "text": "Neue Ansicht", "typ": "funktion", "deliverable": "Screenshot der neuen Ansicht"}}))

    async def test_deliverable_nachtragen_und_abnehmen(self):
        await self.rufe("deliverable_nachtragen", {"vorgabe_id": 10, "hinweis": "nur Pool-Mitglieder"})
        self.assertEqual(self.calls[-1], ("PUT", "/api/v1/vorgaben/10/deliverable",
                                          {"json_body": {"zuordnen": True, "hinweis": "nur Pool-Mitglieder"}}))
        await self.rufe("deliverable_abnehmen", {"vorgabe_id": 10, "ergebnis": "fehlt", "text": "Link fehlt"})
        self.assertEqual(self.calls[-1], ("POST", "/api/v1/vorgaben/10/abnahme",
                                          {"json_body": {"ergebnis": "fehlt", "text": "Link fehlt"}}))

    async def test_entscheidung_mit_quelle(self):
        await self.rufe("projekt_entscheidung", {"vorgabe_id": 10, "text": "Martin sieht nur seine Konten",
                                                 "quelle_art": "portal", "quelle_ref": "Cloud 15:34 UTC"})
        self.assertEqual(self.calls[-1], ("POST", "/api/v1/vorgaben/10/entscheidungen", {"json_body": {
            "text": "Martin sieht nur seine Konten", "quelle": {"art": "portal", "ref": "Cloud 15:34 UTC"}}}))
        await self.rufe("projekt_entscheidungen", {"vorgabe_id": 10})
        self.assertEqual(self.calls[-1][:2], ("GET", "/api/v1/vorgaben/10/entscheidungen"))


class AbsichtenToolTests(ProjektToolsTests):
    async def test_absichten_gehen_an_den_projektbeitrag(self):
        self.antworten[("GET", "/api/v1/vorgaben/10")] = {"projekt_post_id": 6417}
        await self.rufe("projektleitung_absichten", {"vorgabe_id": 10, "absichten": [{"art": "logbuch", "text": "x"}]})
        self.assertEqual(self.calls[-1], ("POST", "/api/v1/projekt/projekte/6417/absichten", {"json_body": {
            "absichten": [{"art": "logbuch", "text": "x"}], "lage": "", "notiz": ""}}))


class DesigntoolTests(ProjektToolsTests):
    """Plan zeichnen, skizzieren und laufend ändern (Alex, 24.11.2026)."""

    async def test_vorgabe_zum_skizzieren(self):
        await self.rufe("vorgabe_anlegen", {"text": "Testseite", "skizzieren": True})
        self.assertEqual(self.calls, [("POST", "/api/v1/vorgaben", {"json_body": {"text": "Testseite", "planen": False}})])

    async def test_plan_zeichnen_nimmt_die_aktuelle_fassung(self):
        self.antworten[("GET", "/api/v1/vorgaben/8")] = {"aktuelle_version": 3, "plan": {"version": 3}}
        pakete = [{"schluessel": "I1", "titel": "Bauen", "agent": "CloudV2"}]
        await self.rufe("plan_zeichnen", {"vorgabe_id": 8, "pakete": pakete})
        self.assertEqual(self.calls[-1], ("PUT", "/api/v1/vorgaben/8/plan",
                                          {"json_body": {"version": 3, "issues": pakete, "anmerkung": ""}}))

    async def test_skizze_ohne_plan_hat_keine_fassung(self):
        self.antworten[("GET", "/api/v1/vorgaben/9")] = {"aktuelle_version": None, "plan": None}
        schritte = [{"schluessel": "I1", "titel": "Bauen", "abhaengig_von": []}]
        await self.rufe("plan_skizzieren", {"vorgabe_id": 9, "schritte": schritte})
        self.assertEqual(self.calls[-1], ("POST", "/api/v1/vorgaben/9/skizze",
                                          {"json_body": {"schritte": schritte, "ziele": [], "anmerkung": ""}}))
        self.antworten[("GET", "/api/v1/vorgaben/9")] = {"aktuelle_version": 2, "plan": {"version": 2}}
        await self.rufe("plan_skizzieren", {"vorgabe_id": 9, "schritte": schritte})
        self.assertEqual(self.calls[-1][2]["json_body"]["version"], 2)

    async def test_projekt_aendern(self):
        aenderungen = [{"schluessel": "I4", "neu": True, "titel": "Domain", "agent": "Cloud-worker"},
                       {"schluessel": "I2", "leitung": "eigen"}]
        await self.rufe("projekt_aendern", {"vorgabe_id": 10, "aenderungen": aenderungen})
        self.assertEqual(self.calls[-1], ("POST", "/api/v1/vorgaben/10/aenderungen",
                                          {"json_body": {"aenderungen": aenderungen, "anmerkung": ""}}))


class SpacePostenTests(ProjektToolsTests):
    async def test_space_posten_nach_name_wortlich_als_idee(self):
        self.antworten[("GET", "/api/v1/spaces")] = [{"id": 0, "name": "Allgemein"}, {"id": 3, "name": "Insecta"}]
        self.antworten[("POST", "/api/v1/spaces/3/ideen")] = {"id": 41, "titel": None, "status": "idee",
                                                             "space_id": 3, "quelle": "gespraech:Insecta"}
        await self.rufe("space_posten", {"text": "Gasmodell gegen Messdaten prüfen", "space": "insecta",
                                         "quelle": "gespraech:Insecta"})
        self.assertEqual(self.calls[-1], ("POST", "/api/v1/spaces/3/ideen", {"json_body": {
            "text": "Gasmodell gegen Messdaten prüfen", "quelle": "gespraech:Insecta"}}))

    async def test_space_posten_ohne_space_geht_nach_allgemein(self):
        self.antworten[("GET", "/api/v1/spaces")] = [{"id": 0, "name": "Allgemein"}, {"id": 3, "name": "Insecta"}]
        await self.rufe("space_posten", {"text": "Eine Idee für später"})
        self.assertEqual(self.calls[-1], ("POST", "/api/v1/spaces/0/ideen", {"json_body": {"text": "Eine Idee für später"}}))

    async def test_space_posten_mitglied_ohne_allgemein(self):
        # Ein Mitglied (6516) sieht „Allgemein“ nicht: mit genau einem Space geht es dorthin.
        self.antworten[("GET", "/api/v1/spaces")] = [{"id": 3, "name": "Insecta"}]
        await self.rufe("space_posten", {"text": "Notiz"})
        self.assertEqual(self.calls[-1], ("POST", "/api/v1/spaces/3/ideen", {"json_body": {"text": "Notiz"}}))
        # Mit mehreren Spaces fragt das Werkzeug zurück, statt zu raten.
        self.calls.clear()
        self.antworten[("GET", "/api/v1/spaces")] = [{"id": 3, "name": "Insecta"}, {"id": 5, "name": "Labor"}]
        ergebnis = await self.rufe("space_posten", {"text": "Notiz"})
        self.assertEqual([c[0] for c in self.calls], ["GET"])
        self.assertIn("space_noetig", str(ergebnis))

    async def test_space_posten_name_mit_leerzeichen_und_fehlerantwort(self):
        self.antworten[("GET", "/api/v1/spaces")] = [{"id": 0, "name": "Allgemein"}, {"id": 4, "name": "Live  Board"}]
        await self.rufe("space_posten", {"text": "x", "space": " live board "})
        self.assertEqual(self.calls[-1][1], "/api/v1/spaces/4/ideen")
        self.antworten[("GET", "/api/v1/spaces")] = {"error": "kein_recht"}
        ergebnis = await self.rufe("space_posten", {"text": "x", "space": "Insecta"})
        self.assertIn("kein_recht", str(ergebnis))

    async def test_space_posten_unbekannter_space_nennt_die_liste(self):
        self.antworten[("GET", "/api/v1/spaces")] = [{"id": 0, "name": "Allgemein"}]
        ergebnis = await self.rufe("space_posten", {"text": "x y z", "space": "Gibtsnicht"})
        self.assertNotIn(("POST", "/api/v1/spaces/0/ideen"), [(c[0], c[1]) for c in self.calls])
        self.assertIn("space_unbekannt", str(ergebnis))
