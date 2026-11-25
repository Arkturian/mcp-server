"""6465 I11: Werkzeuge fuer Agenten im Live-Whiteboard (content-MCP)."""
import asyncio
import os

import pytest

for _k in ("ARKTURIAN_API_KEY", "ONEAL_STORAGE_API_KEY", "JWT_PUBLIC_KEY", "COMM_API_KEY"):
    os.environ.setdefault(_k, "test-only")

import server  # noqa: E402


@pytest.fixture
def aufrufe(monkeypatch):
    gesehen = []

    async def _api(method, endpoint, *, params=None, json_body=None):
        gesehen.append((method, endpoint, params, json_body))
        return {"ok": True}
    monkeypatch.setattr(server, "call_content_api", _api)
    return gesehen


def _rufe(name, argumente):
    return asyncio.run(server.content_mcp.call_tool(name, argumente))


def test_vier_werkzeuge_rufen_die_board_routen(aufrufe):
    _rufe("board_get", {"post_id": 7})
    obj = [{"art": "notiz", "x": 1, "y": 2, "w": 3, "h": 4, "text": "Hallo"}]
    _rufe("board_objekte_anlegen", {"post_id": 7, "objekte": obj})
    _rufe("board_objekt_aendern", {"post_id": 7, "objekt_id": "o-1", "felder": {"x": 9},
                                   "entfernen": ["farbe"]})
    _rufe("board_objekt_aendern", {"post_id": 7, "objekt_id": "a/b", "felder": {"text": "x"}})
    _rufe("board_objekt_loeschen", {"post_id": 7, "objekt_id": "o-1"})
    _rufe("board_objekt_loeschen", {"post_id": 7, "objekt_id": "o-1", "pfeile": False})
    assert aufrufe == [
        ("GET", "/api/v1/posts/7/board", None, None),
        ("POST", "/api/v1/posts/7/board/objekte", None, {"objekte": obj}),
        ("PATCH", "/api/v1/posts/7/board/objekte/o-1", None, {"felder": {"x": 9}, "entfernen": ["farbe"]}),
        ("PATCH", "/api/v1/posts/7/board/objekte/a%2Fb", None, {"felder": {"text": "x"}}),
        ("DELETE", "/api/v1/posts/7/board/objekte/o-1", {"pfeile": "true"}, None),
        ("DELETE", "/api/v1/posts/7/board/objekte/o-1", {"pfeile": "false"}, None),
    ]


def test_werkzeuge_sind_angemeldet():
    namen = {t.name for t in asyncio.run(server.content_mcp.list_tools())}
    assert {"board_get", "board_objekte_anlegen", "board_objekt_aendern",
            "board_objekt_loeschen"} <= namen
