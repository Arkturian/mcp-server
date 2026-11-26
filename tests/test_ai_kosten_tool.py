"""#4127 (Auftrag Alex ueber Cloud an AiApi): Werkzeug ai_kosten reicht GET /ai/kosten unveraendert durch.

Geprueft: nur die Identitaet des Aufrufers geht mit (Bearer <jwt>, X-Agent-Name), NIE der Betreiber-Schluessel
(sonst liesse /ai/kosten jeden durch); ohne Aufrufer kein Kopf, AiApi antwortet 401; leere Parameter fallen
weg; Antwort und Fehler von AiApi kommen unveraendert; die Beschreibung warnt vor dem Summieren von Abo-Werten."""
from __future__ import annotations
import ast
import asyncio
import os
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx

os.environ.setdefault("MCP_SERVERS", "ai")
import auth  # noqa: E402
import server  # noqa: E402

TEXT = (Path(__file__).resolve().parents[1] / "server.py").read_text(encoding="utf-8")


def _beschreibung():
    fn = next(n for n in ast.parse(TEXT).body if isinstance(n, ast.AsyncFunctionDef) and n.name == "ai_kosten")
    for kw in fn.decorator_list[0].keywords:
        if kw.arg == "description":
            return ast.literal_eval(kw.value)
    raise AssertionError("keine description")


class AiKostenTests(unittest.TestCase):
    def _rufen(self, jwt, agent, **kw):
        antwort = {"gruppe": "weg", "zeilen": [{"weg": "stt", "art": "geld", "betrag": 0.04, "waehrung": "USD"}]}
        with patch.object(auth, "current_caller_jwt", return_value=jwt), \
             patch.object(auth, "current_caller_agent_name", return_value=agent), \
             patch.object(auth, "current_caller_verified_agent", return_value=agent), \
             patch.object(server, "AI_API_KEY", "betreiber-schluessel"), \
             patch.object(server, "_fetch_json", new=AsyncMock(return_value=antwort)) as f:
            out = asyncio.run(server.ai_kosten(**kw))
        return out, antwort, f

    def test_nur_der_ausweis_des_aufrufers_nie_der_schluessel(self):
        out, antwort, f = self._rufen("jwt-123", "Bauleiter", von="2026-11-20", bis="2026-11-26", gruppe="aufrufer",
                                      takt="tag")
        self.assertIs(out, antwort)                                           # unveraendert
        args, kw = f.call_args.args, f.call_args.kwargs
        self.assertEqual(args[0], "GET")
        self.assertTrue(args[1].endswith("/ai/kosten"))
        self.assertEqual(kw["headers"], {"Authorization": "Bearer jwt-123", "X-Agent-Name": "Bauleiter"})
        self.assertNotIn("X-API-Key", kw["headers"])
        self.assertNotIn("X-API-KEY", kw["headers"])
        self.assertEqual(kw["params"], {"von": "2026-11-20", "bis": "2026-11-26", "gruppe": "aufrufer", "takt": "tag"})

    def test_ohne_aufrufer_kein_kopf_und_ohne_parameter_keine(self):
        _, _, f = self._rufen(None, None, von="", gruppe="  ")
        self.assertEqual(f.call_args.kwargs["headers"], {})
        self.assertIsNone(f.call_args.kwargs["params"])

    def test_401_von_aiapi_kommt_durch(self):
        def antwort(request):
            assert "x-api-key" not in {k.lower() for k in request.headers}
            return httpx.Response(401, json={"detail": "anmeldung_noetig"})
        echt = httpx.AsyncClient
        with patch.object(auth, "current_caller_jwt", return_value=None), \
             patch.object(auth, "current_caller_verified_agent", return_value=None), \
             patch.object(server, "AI_API_KEY", "betreiber-schluessel"), \
             patch.object(httpx, "AsyncClient", lambda **k: echt(transport=httpx.MockTransport(antwort))):
            with self.assertRaises(RuntimeError) as ex:
                asyncio.run(server.ai_kosten())
        self.assertIn("Upstream 401", str(ex.exception))
        self.assertIn("anmeldung_noetig", str(ex.exception))

    def test_beschreibung(self):
        d = _beschreibung()
        for teil in ("Kosten und Kontingent der KI-Dienste dieser Instanz: Geld je Topf, Abo-Mengen, Aufrufe je "
                     "Weg/Aufrufer, Lücken. Recht ai:kosten:lesen.",
                     "art=geld", "gegenwert_usd", "nicht summieren", "weg | aufrufer | anbieter | tag", "JJJJ-MM-TT"):
            self.assertIn(teil, d)

    def test_kein_call_ai_api(self):
        fn = next(n for n in ast.parse(TEXT).body if isinstance(n, ast.AsyncFunctionDef) and n.name == "ai_kosten")
        code = ast.get_source_segment(TEXT, fn).split('"""', 2)[-1]
        self.assertNotIn("call_ai_api(", code)
        self.assertNotIn("AI_API_KEY", code)


if __name__ == "__main__":
    unittest.main()
