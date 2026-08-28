"""Das Gateway ist ohne ausdrueckliche Gegenanweisung fail-closed (#24).

David/Steward 28.08.: ein anonymer JSON-RPC `tools/call` auf sections_patch
lieferte HTTP 200 und schrieb wirklich — AUTH_REQUIRE_JWT stand per Default
auf "false", und der Content-Helfer faellt ohne JWT auf den Service-Key
zurueck. Der Default gehoert auf "true".
"""
import importlib
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _reload(monkeypatch, value):
    if value is None:
        monkeypatch.delenv("AUTH_REQUIRE_JWT", raising=False)
    else:
        monkeypatch.setenv("AUTH_REQUIRE_JWT", value)
    sys.modules.pop("auth", None)
    return importlib.import_module("auth")


def test_default_is_strict(monkeypatch):
    assert _reload(monkeypatch, None).AUTH_REQUIRE_JWT is True


def test_explicit_false_still_possible(monkeypatch):
    assert _reload(monkeypatch, "false").AUTH_REQUIRE_JWT is False


def test_explicit_true(monkeypatch):
    assert _reload(monkeypatch, "true").AUTH_REQUIRE_JWT is True
