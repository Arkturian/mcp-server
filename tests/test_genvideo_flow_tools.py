"""genvideo_flow-Werkzeuge (AiApi, Alex 14.11.2026): Rumpf nur mit gesetzten
Feldern, Aufrufer-Kennung wie bei genimage, Job-ID sicher im Pfad."""
from __future__ import annotations
import asyncio
import os
import unittest
from unittest.mock import AsyncMock, patch

os.environ.setdefault("MCP_SERVERS", "ai")
import auth  # noqa: E402
import server  # noqa: E402


class GenvideoFlowTests(unittest.TestCase):
    def _lauf(self, coro_fn, *a, **k):
        with patch.object(auth, "current_caller_jwt", return_value="jwt-1"), \
             patch.object(auth, "current_caller_agent_name", return_value="Bauleiter"), \
             patch.object(auth, "current_caller_verified_agent", return_value="Bauleiter"), \
             patch.object(server, "_fetch_json", new=AsyncMock(return_value={"ok": 1})) as f:
            asyncio.run(coro_fn(*a, **k))
        return f.call_args

    def test_abgabe_rumpf_und_kennung(self):
        c = self._lauf(server.ai_genvideo_flow, "Katze", confirm_api_billing=True,
                       aspect_ratio="portrait", start_image=4711)
        self.assertEqual(c.args[0], "POST")
        self.assertTrue(c.args[1].endswith("/ai/genvideo/flow"))
        self.assertEqual(c.kwargs["json_body"], {"prompt": "Katze", "confirm_api_billing": True,
                                                 "aspect_ratio": "portrait", "start_image": "4711"})
        self.assertEqual(c.kwargs["headers"]["X-Agent-Name"], "Bauleiter")

    def test_ohne_freigabe_geht_false_mit(self):
        c = self._lauf(server.ai_genvideo_flow, "x")
        self.assertIs(c.kwargs["json_body"]["confirm_api_billing"], False)

    def test_status_kodiert_schraegstrich(self):
        c = self._lauf(server.ai_genvideo_flow_status, "j1-user:3-a/../b")
        self.assertTrue(c.args[1].endswith("/ai/genvideo/flow/status/j1-user:3-a%2F..%2Fb"))


if __name__ == "__main__":
    unittest.main()


class GenimageFlowTests(GenvideoFlowTests):
    def test_bild_rumpf(self):
        c = self._lauf(server.ai_genimage_flow, "Haus", confirm_api_billing=True,
                       aspect_ratio="16:9", reference_images=[4711, "https://x/y.png"])
        self.assertTrue(c.args[1].endswith("/ai/genimage/flow"))
        self.assertEqual(c.kwargs["json_body"], {"prompt": "Haus", "confirm_api_billing": True,
                                                 "aspect_ratio": "16:9",
                                                 "reference_images": ["4711", "https://x/y.png"]})
        self.assertEqual(c.kwargs["headers"]["X-Agent-Name"], "Bauleiter")
