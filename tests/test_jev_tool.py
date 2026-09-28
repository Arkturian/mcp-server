"""Jev-Tool im ai-MCP (Content #5083): Pflichttexte in der Beschreibung,
Durchreichen an /ai/jev, Aufrufer-Zuordnung aus dem geprueften JWT."""
import ast
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1] / "server.py"
TEXT = SOURCE.read_text(encoding="utf-8")
TREE = ast.parse(TEXT)


def _fn(name):
    return next(n for n in TREE.body if isinstance(n, ast.AsyncFunctionDef) and n.name == name)


def _beschreibung():
    fn = _fn("ai_jev")
    deko = fn.decorator_list[0]
    for kw in deko.keywords:
        if kw.arg == "description":
            return ast.literal_eval(kw.value)
    raise AssertionError("keine description")


def test_signatur_und_aufruf():
    fn = _fn("ai_jev")
    assert [a.arg for a in fn.args.args] == ["state", "questions", "model"]
    src = ast.get_source_segment(TEXT, fn)
    assert '"/ai/jev"' in src and "extra_headers" in src and "current_caller_agent_name" in src


def test_pflichttexte():
    d = _beschreibung()
    for fragment in ("noul", "choice", "score",
                     "Jev generiert keinen Text, nur typisierte Urteile. Rechnen, Zählen und Datumsvergleiche gehören in den Code.",
                     "Keine echten personenbezogenen Daten (Kunden-, Behörden-, Personenmails) ohne Alex' Freigabe. TypeSafe bietet ohne Enterprise-Vertrag keine Zero Data Retention.",
                     "Jev-Urteile sind nie ein Berechtigungs-Tor. Ein Wahrscheinlichkeitsmodell darf nicht über Rechte entscheiden.",
                     "https://docs.typesafe.ai/llms.txt"):
        assert fragment in d, fragment


def test_transport_kennt_extra_headers():
    fn = _fn("call_ai_api")
    assert "extra_headers" in [a.arg for a in fn.args.kwonlyargs]
