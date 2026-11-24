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
        })

    async def test_freigeber_tools_carry_the_rule(self):
        tools = {t.name: t for t in await self.mcp.list_tools()}
        for name in ("vorgabe_anlegen", "projekt_starten", "plan_ueberarbeiten", "vorgabe_ablehnen",
                     "plan_absegnen", "projektleitung_schalten", "projektleitung_antworten", "tor_oeffnen",
                     "vorgabe_loeschen", "paket_fragen", "deliverable_nachtragen", "deliverable_abnehmen"):
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
