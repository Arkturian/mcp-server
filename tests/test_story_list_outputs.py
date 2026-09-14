"""Exercise actual legacy tool registrations without loading other services."""

import ast
import pathlib
import typing
import unittest

from mcp.server.fastmcp import FastMCP


class StoryListOutputTests(unittest.IsolatedAsyncioTestCase):
    async def test_actual_list_tools_validate_empty_and_populated_arrays(self):
        names = {
            "story_" + key + "_list"
            for key in ("characters", "beats", "scenes", "shots")
        }
        source = ast.parse(
            (pathlib.Path(__file__).parents[1] / "server.py").read_text()
        )
        selected = [
            n
            for n in source.body
            if isinstance(n, ast.AsyncFunctionDef) and n.name in names
        ]
        self.assertEqual(len(selected), 4)
        calls, response = [], []

        async def transport(method, path):
            calls.append((method, path))
            return response

        mcp = FastMCP("story-list-test")
        ns = {"story_mcp": mcp, "call_story_api": transport, **vars(typing)}
        exec(
            compile(
                ast.Module(body=selected, type_ignores=[]),
                "actual_story_list_tools",
                "exec",
            ),
            ns,
        )
        tools = {t.name: t for t in await mcp.list_tools()}
        for key, param, path in [
            ("characters", "project_id", "/projects/4/characters"),
            ("beats", "project_id", "/projects/4/beats"),
            ("scenes", "beat_id", "/beats/4/scenes"),
            ("shots", "scene_id", "/scenes/4/shots"),
        ]:
            self.assertEqual(
                tools[key + "_list"].outputSchema["properties"]["result"]["type"],
                "array",
            )
            for response in [
                [],
                [{"id": 9, "name": "fixture", "metadata_json": {"preserved": True}}],
            ]:
                content, structured = await mcp.call_tool(key + "_list", {param: 4})
                self.assertEqual(structured["result"], response)
                self.assertEqual(calls[-1], ("GET", "/api/v1" + path))
        self.assertEqual(len(calls), 8)
