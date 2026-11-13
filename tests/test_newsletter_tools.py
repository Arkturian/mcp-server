import unittest

from mcp.server.fastmcp import FastMCP

from newsletter_tools import LIMITS, TEMPLATE_RULES, register_newsletter_tools


class NewsletterToolsTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.calls = []

        async def transport(method, path, **kwargs):
            self.calls.append((method, path, kwargs))
            return {"ok": True}

        self.mcp = FastMCP("isolated-newsletter")
        register_newsletter_tools(self.mcp, transport)

    async def test_tool_set_has_no_real_send(self):
        names = {t.name for t in await self.mcp.list_tools()}
        self.assertEqual(names, {
            "tenants_list", "templates_list", "template_get", "templates_upsert",
            "template_preview", "campaigns_list", "campaign_get", "campaign_create",
            "campaign_update", "campaign_duplicate", "campaign_delete", "campaign_hold",
            "campaign_preview", "campaign_test_send", "mail_usage",
        })
        self.assertFalse(any("send" in n and "test" not in n for n in names))

    async def test_descriptions_carry_the_rules(self):
        tools = {t.name: t for t in await self.mcp.list_tools()}
        self.assertIn(LIMITS, tools["tenants_list"].description)
        self.assertIn(TEMPLATE_RULES, tools["templates_upsert"].description)

    async def test_upsert_is_a_put_by_slug_with_escaped_path(self):
        await self.mcp.call_tool("templates_upsert", {
            "tenant": "clarityprocess", "slug": "clarity momente", "name": "N",
            "subject_template": "S", "text_template": "T", "html_template": "H",
        })
        method, path, kwargs = self.calls[-1]
        self.assertEqual(method, "PUT")
        self.assertEqual(path, "/api/v1/admin/tenants/clarityprocess/templates/clarity%20momente")
        self.assertEqual(kwargs["json_body"]["html_template"], "H")

    async def test_update_sends_only_given_fields(self):
        await self.mcp.call_tool("campaign_update", {"campaign_id": "c1", "name": "Neu"})
        self.assertEqual(self.calls[-1], ("PATCH", "/api/v1/admin/campaigns/c1",
                                          {"json_body": {"name": "Neu"}}))

    async def test_preview_keeps_explicit_null_first_name(self):
        await self.mcp.call_tool("campaign_preview", {"campaign_id": "c1", "first_name": None})
        self.assertEqual(self.calls[-1][2]["json_body"], {"first_name": None})

    async def test_mail_usage_without_range_sends_no_params(self):
        await self.mcp.call_tool("mail_usage", {"tenant": "clarityprocess"})
        self.assertEqual(self.calls[-1], ("GET", "/api/v1/admin/tenants/clarityprocess/mail-usage",
                                          {"params": None}))
