import json
import unittest

from mcp.server.fastmcp import FastMCP

from event_tools import DATA_RULE, STATUSES, register_event_tools


def _json(result):
    """FastMCP-Ergebnis (Inhaltsliste oder Tupel) als JSON lesen."""
    content = result[0] if isinstance(result, tuple) else result
    return json.loads(content[0].text)


class EventToolsTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.calls = []
        self.exports = []
        self.rows = [{"id": i, "reference": f"ADIC-2027-{i:04d}"} for i in range(1, 8)]

        async def transport(method, path, **kwargs):
            self.calls.append((method, path, kwargs))
            if path.endswith("/registrations"):
                return list(self.rows)
            return {"ok": True}

        async def exporter(tenant, fmt, params, share_with):
            self.exports.append((tenant, fmt, params, share_with))
            return {"id": 4711, "private": True}

        self.mcp = FastMCP("isolated-event")
        register_event_tools(self.mcp, transport, exporter)

    async def test_tool_set(self):
        names = {t.name for t in await self.mcp.list_tools()}
        self.assertEqual(names, {
            "event_access", "events_list", "registrations_list", "registration_get",
            "registration_update_status", "event_stats", "registrations_export",
        })

    async def test_no_delete_tool(self):
        names = {t.name for t in await self.mcp.list_tools()}
        self.assertFalse(any("delete" in n for n in names))

    async def test_personal_data_tools_carry_the_rule(self):
        tools = {t.name: t for t in await self.mcp.list_tools()}
        for name in ("registrations_list", "registration_get", "registrations_export"):
            self.assertIn(DATA_RULE, tools[name].description, name)

    async def test_list_filters_quote_tenant_and_limit(self):
        await self.mcp.call_tool("registrations_list", {"tenant": "inno 4def", "event": "adic-2027", "status": "confirmed", "limit": 3})
        method, path, kw = self.calls[0]
        self.assertEqual((method, path), ("GET", "/api/v1/admin/inno%204def/registrations"))
        self.assertEqual(kw["params"], {"event": "adic-2027", "status": "confirmed"})

    async def test_list_result_is_capped_with_total(self):
        res = _json(await self.mcp.call_tool("registrations_list", {"tenant": "inno4def", "limit": 3}))
        self.assertEqual(res["total"], 7)
        self.assertEqual(len(res["items"]), 3)
        self.assertTrue(res["truncated"])

    async def test_status_is_validated_before_any_call(self):
        with self.assertRaises(Exception):
            await self.mcp.call_tool("registration_update_status", {"tenant": "inno4def", "id": 3, "status": "deleted"})
        self.assertEqual(self.calls, [])

    async def test_update_status_patches(self):
        await self.mcp.call_tool("registration_update_status", {"tenant": "clarityprocess", "id": 12, "status": "paid"})
        self.assertEqual(self.calls, [("PATCH", "/api/v1/admin/clarityprocess/registrations/12", {"json_body": {"status": "paid"}})])

    async def test_stats_and_events_paths(self):
        await self.mcp.call_tool("event_stats", {"tenant": "inno4def", "event": "adic-2027"})
        await self.mcp.call_tool("events_list", {"tenant": "clarityprocess", "offering": "clarity-retreat"})
        self.assertEqual(self.calls[0][:2], ("GET", "/api/v1/admin/inno4def/stats"))
        self.assertEqual(self.calls[0][2]["params"], {"event": "adic-2027"})
        self.assertEqual(self.calls[1][:2], ("GET", "/api/v1/public/clarityprocess/events"))

    async def test_export_goes_through_exporter_only(self):
        res = _json(await self.mcp.call_tool("registrations_export", {"tenant": "inno4def", "format": "XLSX", "event": "adic-2027", "share_with": ["Inno4def"]}))
        self.assertEqual(self.exports, [("inno4def", "xlsx", {"event": "adic-2027"}, ["Inno4def"])])
        self.assertEqual(self.calls, [])
        self.assertEqual(res["id"], 4711)

    async def test_export_rejects_unknown_format(self):
        with self.assertRaises(Exception):
            await self.mcp.call_tool("registrations_export", {"tenant": "inno4def", "format": "pdf"})
        self.assertEqual(self.exports, [])

    def test_statuses_match_event_api(self):
        self.assertEqual(set(STATUSES), {"requested", "confirmed", "paid", "waitlist", "cancelled"})


if __name__ == "__main__":
    unittest.main()
