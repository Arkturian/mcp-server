"""Issue-Werkzeuge reichen projekt_id an die issue-api durch (Beitrag 6533, Paket N1).

Grundlage ist issue-api 24c8f8a (I11a): Spalte projekt_id, Filter
GET /issues?projekt_id=, PATCH mit null entfernt. Hier wird nur geprueft, dass
die Werkzeuge das Feld unveraendert weitergeben und bestehende Aufrufe ohne
projekt_id genau dieselbe Anfrage schicken wie vorher.
"""
import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
for _k in ("ARKTURIAN_API_KEY", "ONEAL_STORAGE_API_KEY", "JWT_PUBLIC_KEY", "COMM_API_KEY"):
    os.environ.setdefault(_k, "test-only")


def _attrappe(monkeypatch):
    import server
    calls = []

    async def _f(method, path, json_body=None, params=None, **kw):
        calls.append({"method": method, "path": path, "json": json_body, "params": params})
        return {"ok": True}

    monkeypatch.setattr(server, "call_issue_api", _f)
    return server, calls


def test_create_structured_mit_und_ohne(monkeypatch):
    server, calls = _attrappe(monkeypatch)
    asyncio.run(server.issue_create_structured(title="t", description="d", projekt_id=23))
    asyncio.run(server.issue_create_structured(title="t", description="d"))
    assert calls[0]["method"] == "POST" and calls[0]["path"] == "/api/v1/issues"
    assert calls[0]["json"]["projekt_id"] == 23
    assert "projekt_id" not in calls[1]["json"]
    assert calls[1]["json"] == {"title": "t", "description": "d", "severity": "minor", "status": "open"}


def test_create_freitext(monkeypatch):
    server, calls = _attrappe(monkeypatch)
    asyncio.run(server.issue_create(description="Login kaputt", projekt_id=23))
    asyncio.run(server.issue_create(description="Login kaputt"))
    assert calls[0]["path"] == "/api/v1/issues/intake"
    assert calls[0]["json"] == {"text": "Login kaputt", "projekt_id": 23}
    assert calls[1]["json"] == {"text": "Login kaputt"}


def test_update_setzt_und_entfernt(monkeypatch):
    server, calls = _attrappe(monkeypatch)
    asyncio.run(server.issue_update(id=7, projekt_id=23))
    asyncio.run(server.issue_update(id=7, projekt_id=0))
    asyncio.run(server.issue_update(id=7, title="neu"))
    assert calls[0]["method"] == "PATCH" and calls[0]["path"] == "/api/v1/issues/7"
    assert calls[0]["json"] == {"projekt_id": 23}
    assert calls[1]["json"] == {"projekt_id": None}  # 0 entfernt die Zuordnung
    assert calls[2]["json"] == {"title": "neu"}


def test_list_filter(monkeypatch):
    server, calls = _attrappe(monkeypatch)
    asyncio.run(server.issue_list(projekt_id=23))
    asyncio.run(server.issue_list(assignee="agent:Issue"))
    assert calls[0]["method"] == "GET" and calls[0]["path"] == "/api/v1/issues"
    assert calls[0]["params"]["projekt_id"] == 23
    assert "projekt_id" not in calls[1]["params"]
