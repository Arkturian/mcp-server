"""/health auf Mandanten-Gateways (Steward/Cloud 02.10.): Davids Gateway mountet per
MCP_SERVERS nur seine Dienste, /health probte trotzdem oneal-storage ohne Schluessel.
_fetch_json warf RuntimeError, gefangen wurde nur httpx.HTTPError -> 500.
Erwartet: nur gemountete Dienste werden geprobt, jeder Probenfehler wird 'degraded'."""
from __future__ import annotations
import asyncio
import os
import unittest
from unittest.mock import AsyncMock, patch

os.environ.setdefault("MCP_SERVERS", "cloud")
import server  # noqa: E402
from fastapi import HTTPException  # noqa: E402

_PROBEN = ("storage_kg_stats", "oneal_service_ping", "oneal_storage_kg_stats",
           "artrack_service_health", "content_service_health", "tree_service_health")


def _alle_gruen():
    return {n: AsyncMock(return_value={"ok": True}) for n in _PROBEN}


class HealthMountedOnlyTests(unittest.TestCase):
    def _lauf(self, enabled, mocks):
        with patch.object(server, "ENABLED_MCPS", enabled), \
             patch.multiple(server, **mocks):
            return asyncio.run(server.health())

    def test_nicht_gemountete_dienste_werden_nicht_geprobt(self):
        mocks = _alle_gruen()
        mocks["oneal_storage_kg_stats"] = AsyncMock(side_effect=RuntimeError("Upstream 401: Invalid API key"))
        out = self._lauf({"cloud", "content", "storage", "tree"}, mocks)
        self.assertEqual(out["status"], "healthy")
        mocks["oneal_storage_kg_stats"].assert_not_awaited()
        mocks["artrack_service_health"].assert_not_awaited()
        self.assertEqual(set(out) - {"status"}, {"storage_arkturian", "content", "tree"})

    def test_runtime_error_wird_degraded_statt_500(self):
        mocks = _alle_gruen()
        mocks["storage_kg_stats"] = AsyncMock(side_effect=RuntimeError("Upstream 401: Invalid API key"))
        with self.assertRaises(HTTPException) as ctx:
            self._lauf({"storage", "content"}, mocks)
        self.assertEqual(ctx.exception.status_code, 207)
        self.assertIn("Invalid API key", ctx.exception.detail["storage_arkturian_error"])

    def test_ohne_filter_wird_alles_geprobt(self):
        mocks = _alle_gruen()
        out = self._lauf(None, mocks)
        self.assertEqual(out["status"], "healthy")
        for m in mocks.values():
            m.assert_awaited()


if __name__ == "__main__":
    unittest.main()
