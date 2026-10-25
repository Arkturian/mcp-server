"""Projektordner fuer Agenten im cloud-MCP (Jascha/Alex 25.10.2026): anlegen,
zuweisen, und create_session/clone_session legen den neuen Agenten gleich ab."""
from __future__ import annotations
import asyncio
import os
import unittest
from unittest.mock import AsyncMock, patch

os.environ.setdefault("MCP_SERVERS", "cloud")
import server  # noqa: E402


class AgentProjectsToolTests(unittest.TestCase):
    def test_create_mit_project_weist_danach_zu(self):
        calls = []

        async def fake(method, endpoint, **kw):
            calls.append((method, endpoint, kw.get("json_body")))
            return {"status": "created", "name": "Bau-1"} if method == "POST" else {"ok": True, "project_id": "p1"}
        with patch.object(server, "call_cloud_api", new=fake):
            out = asyncio.run(server.cloud_create_session("Bau-1", project_id="p1"))
        self.assertEqual(calls[1], ("PUT", "/api/sessions/Bau-1/project", {"project_id": "p1"}))
        self.assertEqual(out["project"], {"ok": True, "project_id": "p1"})

    def test_ohne_project_oder_bei_fehler_keine_zuweisung(self):
        for result in ({"status": "created"}, {"error": "name taken"}):
            with patch.object(server, "call_cloud_api", new=AsyncMock(return_value=dict(result))) as call:
                asyncio.run(server.cloud_create_session("X", project_id="" if "status" in result else "p1"))
            self.assertEqual(call.await_count, 1)

    def test_clone_weist_dem_klon_zu(self):
        with patch.object(server, "call_cloud_api",
                          new=AsyncMock(side_effect=[{"status": "cloned"}, {"ok": True}])) as call:
            asyncio.run(server.cloud_clone_session("Bau-1", "Bau-2", project_id="p9"))
        self.assertEqual(call.call_args_list[1].args, ("PUT", "/api/sessions/Bau-2/project"))
        self.assertEqual(call.call_args_list[1].kwargs["json_body"], {"project_id": "p9"})

    def test_setzen_und_herausnehmen(self):
        with patch.object(server, "call_cloud_api", new=AsyncMock(return_value={})) as call:
            asyncio.run(server.cloud_session_set_project("Bau-1", ""))
        self.assertEqual(call.call_args.kwargs["json_body"], {"project_id": None})

    def test_update_parent_strich_heisst_oberste_ebene(self):
        with patch.object(server, "call_cloud_api", new=AsyncMock(return_value={})) as call:
            asyncio.run(server.cloud_agent_projects_update("p1", parent_id="-"))
        self.assertEqual(call.call_args.kwargs["json_body"], {"parent_id": ""})
        out = asyncio.run(server.cloud_agent_projects_update("p1"))
        self.assertIn("error", out)


if __name__ == "__main__":
    unittest.main()
