import unittest

from mcp.server.fastmcp import FastMCP

from llmtime_tools import AUSSCHLUSS, FORECAST_RULE, register_llmtime_tools


class LlmtimeToolsTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.calls = []

        async def transport(method, path, **kwargs):
            self.calls.append((method, path, kwargs))
            return {"ok": True}

        self.mcp = FastMCP("isolated-llmtime")
        register_llmtime_tools(self.mcp, transport)

    async def test_tool_set(self):
        names = {t.name for t in await self.mcp.list_tools()}
        self.assertEqual(names, {
            "item_add", "item_start", "item_finish", "item_discard", "items_list",
            "forecast_when", "forecast_how_many", "forecast_aufwand",
            "import_merges", "rules_get", "rules_set",
        })

    async def test_forecast_tools_carry_the_rule(self):
        tools = {t.name: t for t in await self.mcp.list_tools()}
        for name in ("forecast_when", "forecast_how_many", "forecast_aufwand"):
            self.assertIn(FORECAST_RULE, tools[name].description, name)

    async def test_exclusion_is_described_as_one_off_events_not_quiet_days(self):
        tools = {t.name: t for t in await self.mcp.list_tools()}
        for name in ("forecast_when", "forecast_how_many"):
            self.assertIn(AUSSCHLUSS, tools[name].description, name)
            self.assertNotIn("Tage ohne Arbeit", tools[name].description, name)

    async def test_item_add_drops_only_unset_fields(self):
        await self.mcp.call_tool("item_add", {"projekt": "demo", "meilenstein": "M1"})
        self.assertEqual(self.calls, [("POST", "/api/v1/items",
                                       {"json_body": {"projekt": "demo", "meilenstein": "M1"}})])

    async def test_finish_keeps_zero_tokens_and_quotes_id(self):
        await self.mcp.call_tool("item_finish", {"id": "a/b", "tokens": 0})
        self.assertEqual(self.calls, [("POST", "/api/v1/items/a%2Fb/finish",
                                       {"json_body": {"tokens": 0}})])

    async def test_finish_passes_zeilen_and_aufwand_names_new_sizes(self):
        await self.mcp.call_tool("item_finish", {"id": "x", "zeilen": 42})
        self.assertEqual(self.calls[-1][2]["json_body"], {"zeilen": 42})
        tools = {t.name: t for t in await self.mcp.list_tools()}
        for groesse in ("aktiv_stunden", "zeilen", "tokens", "stunden"):
            self.assertIn(groesse, tools["forecast_aufwand"].description)

    async def test_when_joins_exclusions(self):
        await self.mcp.call_tool("forecast_when", {
            "projekt": "demo", "rest": 0,
            "bis_ausschluss": ["2026-12-24..2026-12-26", "2026-12-31"],
        })
        method, path, kw = self.calls[0]
        self.assertEqual((method, path), ("GET", "/api/v1/forecast/when"))
        self.assertEqual(kw["params"], {"projekt": "demo", "rest": 0,
                                        "bis_ausschluss": "2026-12-24..2026-12-26,2026-12-31"})

    async def test_when_passes_zufluss_only_when_set(self):
        await self.mcp.call_tool("forecast_when", {"projekt": "demo", "rest": 3, "zufluss": False})
        self.assertEqual(self.calls[-1][2]["params"], {"projekt": "demo", "rest": 3, "zufluss": "false"})
        await self.mcp.call_tool("forecast_when", {"projekt": "demo"})
        self.assertNotIn("zufluss", self.calls[-1][2]["params"])
        tools = {t.name: t for t in await self.mcp.list_tools()}
        self.assertIn("nicht konvergent", tools["forecast_when"].description)

    async def test_how_many_requires_bis(self):
        await self.mcp.call_tool("forecast_how_many", {"projekt": "demo", "bis": "2026-11-21"})
        self.assertEqual(self.calls[0][2]["params"], {"projekt": "demo", "bis": "2026-11-21"})

    async def test_import_merges_sends_flag_explicitly(self):
        merges = [{"commit": "aaa", "nachricht": "fix(x): y (#1)",
                   "merged_at": "2026-11-05T12:00:00+00:00",
                   "first_commit_at": "2026-11-03T12:00:00+00:00", "meilenstein": "M8"}]
        await self.mcp.call_tool("import_merges", {"projekt": "p", "merges": merges})
        body = self.calls[0][2]["json_body"]
        self.assertIs(body["aktualisieren"], False)
        self.assertNotIn("regeln", body)  # absent = stored project rules apply
        self.assertEqual(body["merges"], merges)

    async def test_rules_set_sends_list_body_even_when_empty(self):
        await self.mcp.call_tool("rules_set", {"projekt": "p q", "regeln": []})
        self.assertEqual(self.calls, [("PUT", "/api/v1/projects/p%20q/rules", {"json_body": []})])


class LlmtimeHeadersTests(unittest.TestCase):
    def test_instance_key_wins_over_caller_jwt(self):
        import server

        old = server.LLMTIME_API_KEY
        try:
            server.LLMTIME_API_KEY = "k"
            self.assertEqual(server._llmtime_headers(), {"X-API-Key": "k"})
        finally:
            server.LLMTIME_API_KEY = old

    def test_without_key_caller_headers_are_used(self):
        import server

        old = server.LLMTIME_API_KEY
        try:
            server.LLMTIME_API_KEY = ""
            self.assertNotIn("X-API-Key", server._llmtime_headers())
        finally:
            server.LLMTIME_API_KEY = old


if __name__ == "__main__":
    unittest.main()
