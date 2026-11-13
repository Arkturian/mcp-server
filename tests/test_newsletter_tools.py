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


# --- Gateway: only an explicit mcp:newsletter right opens the server ----------

import asyncio  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402

import httpx  # noqa: E402
import pytest  # noqa: E402
from starlette.applications import Starlette  # noqa: E402
from starlette.responses import JSONResponse  # noqa: E402
from starlette.routing import Route  # noqa: E402

import auth  # noqa: E402

CALL = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                   "params": {"name": "templates_list", "arguments": {"tenant": "x"}}})


def test_server_registers_newsletter_as_explicit_only():
    """Own process: other tests reload `auth`, which drops what server.py registered."""
    code = ("import json, auth, server; print(json.dumps({"
            "'explicit': 'newsletter' in auth.EXPLICIT_ONLY_SERVERS,"
            "'offen': auth._offen('newsletter')}))")
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    r = subprocess.run([sys.executable, "-c", code], cwd=root, capture_output=True,
                       text=True, timeout=120)
    assert r.returncode == 0, r.stderr[-2000:]
    assert json.loads(r.stdout.strip().splitlines()[-1]) == {"explicit": True, "offen": False}


def _gateway():
    async def ok(request):
        return JSONResponse({"ok": 1})
    app = Starlette(routes=[Route("/{srv}/", ok, methods=["POST"])])
    app.add_middleware(auth.JWTAuthMiddleware)
    return app


@pytest.fixture
def perms(monkeypatch):
    monkeypatch.setattr(auth, "EXPLICIT_ONLY_SERVERS", {"newsletter"})
    monkeypatch.setattr(auth, "OFFENE_SERVER", {"newsletter"})

    async def key():
        return "k"
    monkeypatch.setattr(auth, "_get_public_key", key)

    def set_perms(values):
        monkeypatch.setattr(auth, "_decode_jwt", lambda token, key: {
            "type": "agent", "agent_name": "X", "role": "r", "permissions": values})
    return set_perms


def _post():
    async def go():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=_gateway()),
                                     base_url="http://gw") as c:
            return await c.post("/newsletter/", content=CALL, headers={
                "Authorization": "Bearer t", "Content-Type": "application/json"})
    return asyncio.run(go()).status_code


@pytest.mark.parametrize("values", [["*"], ["mcp:*"], ["mcp:*:*"], ["mcp:*:templates_list"],
                                    ["mcp:storage:*"], []])
def test_wildcards_do_not_open_newsletter(perms, values):
    perms(values)
    assert _post() == 403


@pytest.mark.parametrize("values", [["mcp:newsletter:*"], ["mcp:*", "mcp:newsletter:templates_list"]])
def test_explicit_right_opens_newsletter(perms, values):
    perms(values)
    assert _post() == 200
