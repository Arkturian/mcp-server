"""#3658: Werkzeug kette_text -> POST /ai/kette (Vertrag AiApi 11327, 11330, 11331).

Die Funktion wird wie in test_chatgpt_text_contract.py aus dem Quelltext gebaut und mit
einer Attrappe fuer call_ai_api ausgefuehrt; server.py wird nicht importiert, nichts geht ins Netz.
"""
import ast
import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

SOURCE = Path(__file__).resolve().parents[1] / "server.py"
TEXT = SOURCE.read_text(encoding="utf-8")
TREE = ast.parse(TEXT)
KONSTANTEN = ("KETTE_BUDGET_VORGABE_S", "KETTE_BUDGET_HOECHSTENS_S", "KETTE_ZUSCHLAG_S")


def _fn(name):
    return next(n for n in TREE.body if isinstance(n, ast.AsyncFunctionDef) and n.name == name)


def _werkzeug():
    gesendet = []

    async def call_ai_api(method, path, params=None, json_body=None, timeout=None, **kw):
        gesendet.append({"method": method, "path": path, "params": params, "body": json_body,
                         "timeout": timeout, "kw": kw})
        return {"response": "ok", "routed_to": "agy"}

    knoten = [n for n in TREE.body if isinstance(n, ast.Assign)
              and any(getattr(t, "id", None) in KONSTANTEN for t in n.targets)]
    fn = _fn("ai_kette_text")
    fn.decorator_list = []
    modul = ast.Module(body=knoten + [fn], type_ignores=[])
    ns = {"call_ai_api": call_ai_api, "Optional": Optional, "Dict": Dict, "Any": Any, "List": List}
    exec(compile(modul, "server.py", "exec"), ns)
    return lambda **kw: (asyncio.run(ns["ai_kette_text"](**kw)), gesendet[-1])


def _beschreibung():
    deko = _fn("ai_kette_text").decorator_list[0]
    return {kw.arg: ast.literal_eval(kw.value) for kw in deko.keywords}


def test_name_und_pfad():
    d = _beschreibung()
    assert d["name"] == "kette_text"
    for teil in ("/ai/kette", "routed_to", "versucht", "kontingent", "wege_fehler", "cli_fehlt",
                 "kette_aus", "budget"):
        assert teil in d["description"], teil


def test_nur_prompt_minimaler_rumpf_und_frist_vorgabe():
    antwort, g = _werkzeug()(prompt="x")
    assert antwort == {"response": "ok", "routed_to": "agy"}            # unveraendert weiter
    assert g["method"] == "POST" and g["path"] == "/ai/kette" and g["params"] is None
    assert g["body"] == {"prompt": "x"}
    assert g["timeout"] == 40.0                                          # 30 + 10


def test_alle_felder_gehen_unveraendert_durch():
    wege = [{"weg": "agy", "model": "gemini-4-hydrogen-medium"}, {"weg": "codex", "model": "gpt-6-luna",
                                                                  "effort": "low"}]
    _, g = _werkzeug()(prompt="x", system="s", wege=wege, budget_s=45, nur_erster_weg=True,
                       max_tokens=400, temperature=0.2)
    assert g["body"] == {"prompt": "x", "system": "s", "wege": wege, "budget_s": 45,
                         "nur_erster_weg": True, "max_tokens": 400, "temperature": 0.2}
    assert g["timeout"] == 55.0


def test_false_und_null_werden_nicht_verschluckt():
    _, g = _werkzeug()(prompt="x", nur_erster_weg=False, temperature=0.0)
    assert g["body"]["nur_erster_weg"] is False and g["body"]["temperature"] == 0.0


@pytest.mark.parametrize("budget,frist", [(500, 130.0), (120, 130.0), (5, 15.0), ("kaputt", 40.0),
                                          (float("nan"), 40.0), (-3, 40.0), (0, 40.0)])
def test_frist_ist_endlich_und_gedeckelt(budget, frist):
    _, g = _werkzeug()(prompt="x", budget_s=budget)
    assert g["timeout"] == frist
    assert g["body"]["budget_s"] == budget or budget != budget           # api-ai prueft und weist ab


def test_keine_eigene_reihenfolge_und_keine_werkzeuge():
    """Die Reihenfolge kommt aus KI_KETTE; das Werkzeug kennt keine Parameter dafuer."""
    namen = [a.arg for a in _fn("ai_kette_text").args.args]
    assert namen == ["prompt", "system", "wege", "budget_s", "nur_erster_weg", "max_tokens", "temperature"]
