"""decisions-Tool im ai-MCP (Post #5694): nur fuer Bildfragen und lange
gecachte Kontexte, Durchreichen an /ai/decisions mit dem JWT des Aufrufers."""
import ast
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1] / "server.py"
TEXT = SOURCE.read_text(encoding="utf-8")
TREE = ast.parse(TEXT)


def _fn(name):
    return next(n for n in TREE.body if isinstance(n, ast.AsyncFunctionDef) and n.name == name)


def _beschreibung():
    for kw in _fn("ai_decisions").decorator_list[0].keywords:
        if kw.arg == "description":
            return ast.literal_eval(kw.value)
    raise AssertionError("keine description")


def test_signatur_und_aufruf():
    fn = _fn("ai_decisions")
    assert [a.arg for a in fn.args.args] == ["input", "questions", "prompt_cache_key"]
    src = ast.get_source_segment(TEXT, fn)
    assert '"/ai/decisions"' in src and "current_caller_jwt" in src and '"Authorization"' in src


def test_pflichttexte():
    d = _beschreibung()
    for fragment in ("Standard für typisierte Urteile bleibt jev", "#5694", "refusal",
                     "predicate", "choice", "score",
                     "Keine echten personenbezogenen Daten",
                     "Urteile sind nie ein Berechtigungs-Tor"):
        assert fragment in d, fragment
