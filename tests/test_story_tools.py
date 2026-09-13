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


if __name__ == "__main__":
    unittest.main()
