import json
import unittest
from mcp.server.fastmcp import FastMCP
from story_tools import OPERATIONS, register_story_tools


class StoryToolsTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.calls = []

        async def transport(method, path, **kwargs):
            self.calls.append((method, path, kwargs))
            return {"echo": kwargs}

        self.mcp = FastMCP("isolated-story")
        register_story_tools(self.mcp, transport)

    async def test_tools_and_explicit_payload_passthrough(self):
        tools = await self.mcp.list_tools()
        self.assertEqual(len(tools), len(OPERATIONS) + 1)
        schema = next(t.inputSchema for t in tools if t.name == "production_execute")
        self.assertEqual(schema["required"], ["project_id", "run_id", "body"])
        body = {
            "confirm_api_billing": False,
            "expected_revision": 4,
            "expected_fingerprint": "a" * 64,
            "phase": "samples",
            "item_ids": ["image-1"],
        }
        await self.mcp.call_tool(
            "production_execute", {"project_id": 5, "run_id": "run-00001", "body": body}
        )
        self.assertEqual(
            self.calls,
            [
                (
                    "POST",
                    "/api/v1/projects/5/production-runs/run-00001/execute",
                    {"json_body": body},
                )
            ],
        )
        # No truthy filtering and no invented billing permission.
        self.assertIs(self.calls[0][2]["json_body"]["confirm_api_billing"], False)

    async def test_existing_contract_action_exposes_chat_workflow(self):
        async def transport(method, path, **kwargs):
            if path == "/openapi.json":
                return {
                    "paths": {
                        "/api/v1/projects/{project_id}/production-preview": {
                            "post": {"summary": "Preview"}
                        }
                    }
                }
            self.assertEqual(path, "/api/v1/workflow")
            return {
                "version": "v2",
                "guide": "Complete chat guide",
                "interaction": {"portal_required": False},
                "field_usage": {"text": "spoken"},
            }

        mcp = FastMCP("old-connector-contract")
        register_story_tools(mcp, transport)
        output = await mcp.call_tool(
            "contracts_get", {"operation": "production_preview"}
        )
        result = output[1] if isinstance(output, tuple) else json.loads(output[0].text)
        self.assertTrue(result["available"])
        self.assertEqual(result["workflow"]["guide"], "Complete chat guide")
        self.assertFalse(result["workflow"]["interaction"]["portal_required"])
        self.assertEqual(result["field_usage"], {"text": "spoken"})

    async def test_recovery_is_separate_and_path_injection_is_rejected(self):
        await self.mcp.call_tool(
            "production_recover", {"project_id": 5, "run_id": "run-00001"}
        )
        self.assertEqual(
            self.calls[0],
            ("POST", "/api/v1/projects/5/production-runs/run-00001/recover", {}),
        )
        with self.assertRaises(Exception):
            await self.mcp.call_tool(
                "production_runs_get", {"project_id": 5, "run_id": "../../ai/genimage"}
            )
        self.assertEqual(len(self.calls), 1)

    async def test_image_feedback_uses_media_binding_without_production_fields(self):
        body = {"expected_revision": 0, "expected_media_fingerprint": "a" * 64,
                "rating": "like", "liked": "", "change_requested": ""}
        await self.mcp.call_tool("image_feedback_save", {"media_id": 42, "body": body})
        self.assertEqual(self.calls, [("PUT", "/api/v1/media/42/feedback", {"json_body": body})])
        tool = next(t for t in await self.mcp.list_tools() if t.name == "image_feedback_save")
        self.assertIn("NEVER production/billing approval", tool.description)
        self.assertEqual(tool.inputSchema["required"], ["media_id", "body"])

    async def test_single_image_scope_is_explicit_and_forwarded(self):
        body = {"request_id": "single-image-test", "expected_source_fingerprint": "b" * 64,
                "target_media_id": 42, "instruction": "Preserve light, improve sensors"}
        await self.mcp.call_tool("visual_development_analyse", {"project_id": 7, "body": body})
        self.assertEqual(self.calls, [("POST", "/api/v1/projects/7/visual-development", {"json_body": body})])
        tool = next(t for t in await self.mcp.list_tools() if t.name == "visual_development_analyse")
        self.assertIn("target_media_id", tool.description)
        self.assertIn("Never infer global scope", tool.description)


if __name__ == "__main__":
    unittest.main()
