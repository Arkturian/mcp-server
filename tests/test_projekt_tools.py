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
            "projektleitung_antworten", "tor_oeffnen", "meine_aufgaben",
        })

    async def test_freigeber_tools_carry_the_rule(self):
        tools = {t.name: t for t in await self.mcp.list_tools()}
        for name in ("vorgabe_anlegen", "projekt_starten", "plan_ueberarbeiten", "vorgabe_ablehnen",
                     "plan_absegnen", "projektleitung_schalten", "projektleitung_antworten", "tor_oeffnen"):
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

    async def test_tor_oeffnen_findet_das_paket(self):
        self.antworten[("GET", "/api/v1/vorgaben/7")] = {"projekt_post_id": 500}
        self.antworten[("GET", "/api/v1/projekt/projekte/500")] = {"aufgaben": [
            {"plan": "I1", "schluessel": "500/t-1"}, {"plan": "I2", "schluessel": "500/t-2"}]}
        await self.rufe("tor_oeffnen", {"vorgabe_id": 7, "paket": "I2"})
        self.assertEqual(self.calls[-1], ("POST", "/api/v1/projekt/aufgaben/500/t-2/tor", {"json_body": {}}))


if __name__ == "__main__":
    unittest.main()
