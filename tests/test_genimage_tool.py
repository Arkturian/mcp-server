"""genimage im ai-MCP (Alex 24.10.2026, P3): Agenten muessen den Abo-Weg
(codex-imagegen), Referenzbilder und die uebrigen api-ai-Felder erreichen.
Ausgefuehrt, nicht gelesen: der Rumpf, der bei api-ai ankaeme, wird geprueft."""
from __future__ import annotations
import asyncio
import os
import unittest
from unittest.mock import AsyncMock, patch

os.environ.setdefault("MCP_SERVERS", "cloud")
import server  # noqa: E402


def _werkzeug():
    f = server.ai_genimage
    return getattr(f, "fn", f)


class GenimageToolTests(unittest.TestCase):
    def _rumpf(self, **kw):
        with patch.object(server, "call_ai_api", AsyncMock(return_value={"ok": True})) as m:
            asyncio.run(_werkzeug()(**kw))
        args, kwargs = m.call_args
        self.assertEqual(args, ("POST", "/ai/genimage"))
        return kwargs["json_body"]

    def test_abo_rendering_mit_vorlage_full_hd(self):
        b = self._rumpf(prompt="fotorealistisch", model="codex-imagegen", width=1920, height=1080,
                        reference_image_urls=["https://example.org/storage/media/1"])
        self.assertEqual(b["model"], "codex-imagegen")
        self.assertEqual((b["width"], b["height"]), (1920, 1080))
        self.assertEqual(b["reference_image_urls"], ["https://example.org/storage/media/1"])

    def test_vorgabe_ist_kein_abgeschaltetes_modell_und_schickt_nur_gesetztes(self):
        b = self._rumpf(prompt="x")
        self.assertEqual(b["model"], "gpt-image-2")
        self.assertEqual(set(b), {"prompt", "model", "width", "height"})

    def test_route_privat_und_abrechnung_werden_durchgereicht(self):
        b = self._rumpf(prompt="x", route="subscription", private=True, confirm_api_billing=True,
                        background="transparent", quality="high", negative_prompt="Text")
        for k, v in (("route", "subscription"), ("private", True), ("confirm_api_billing", True),
                     ("background", "transparent"), ("quality", "high"), ("negative_prompt", "Text")):
            self.assertEqual(b[k], v)

    def test_private_false_wird_nicht_verschluckt(self):
        b = self._rumpf(prompt="x", private=False)
        self.assertIs(b["private"], False)

    def test_ergebnis_wird_an_den_aufrufer_gebunden(self):
        import auth
        with patch.object(auth, "current_caller_verified_agent", lambda: "PlanMaster"):
            b = self._rumpf(prompt="x", private=True)
        self.assertEqual(b["grant_scope"], "cloud-session:PlanMaster")

    def test_ohne_gepruefte_identitaet_keine_bindung(self):
        import auth
        with patch.object(auth, "current_caller_verified_agent", lambda: ""):
            b = self._rumpf(prompt="x", private=True)
        self.assertNotIn("grant_scope", b)

    def test_ein_scope_genuegt_der_api_ai_pruefung(self):
        # api-ai prueft genau EINEN Scope (_SCOPE_RE, kein Komma) vor der
        # Erzeugung; auch ein Name mit Leerzeichen muss durchgehen.
        import re
        import auth
        muster = re.compile(r"^[a-z][a-z0-9-]{0,40}:[A-Za-z0-9._-]{1,120}$")
        with patch.object(auth, "current_caller_verified_agent", lambda: "EDERA-HERMINE (CFO)"):
            b = self._rumpf(prompt="x")
        self.assertRegex(b["grant_scope"], muster)


if __name__ == "__main__":
    unittest.main()
