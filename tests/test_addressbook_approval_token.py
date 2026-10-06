"""#5245 Phase 1: Einmal-Freigabe fuer das Adressbuch. Das Token aus dem
IACP-Bescheid reicht der Agent als `approval_token` mit; Comm erwartet es
als Kopf `X-Approval-Token`. Ohne Token entsteht kein Kopf."""
import asyncio
import inspect
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
for _k in ("ARKTURIAN_API_KEY", "ONEAL_STORAGE_API_KEY", "JWT_PUBLIC_KEY", "COMM_API_KEY"):
    os.environ.setdefault(_k, "test-only")

TOOLS = ("comm_addressbook_search", "comm_addressbook_get",
         "comm_addressbook_count", "comm_addressbook_import_vcard")


def _capture(monkeypatch):
    import server
    seen = []

    async def _fetch(method, url, *, headers=None, **kw):
        seen.append(dict(headers or {}))
        return {"ok": True}

    monkeypatch.setattr(server, "_fetch_json", _fetch)
    return server, seen


def test_alle_adressbuch_werkzeuge_nehmen_approval_token():
    import server
    for name in TOOLS:
        p = inspect.signature(getattr(server, name)).parameters["approval_token"]
        assert p.default is None


def test_token_wird_als_kopf_weitergereicht(monkeypatch):
    server, seen = _capture(monkeypatch)
    asyncio.run(server.comm_addressbook_search(q="Peter", approval_token=" OST.jwt "))
    asyncio.run(server.comm_addressbook_get(contact_id=7, approval_token="OST.jwt"))
    asyncio.run(server.comm_addressbook_count(approval_token="OST.jwt"))
    asyncio.run(server.comm_addressbook_import_vcard(storage_id=1, approval_token="OST.jwt"))
    assert [h.get("X-Approval-Token") for h in seen] == ["OST.jwt"] * 4


def test_ohne_token_kein_kopf(monkeypatch):
    server, seen = _capture(monkeypatch)
    asyncio.run(server.comm_addressbook_search(q="Peter"))
    assert "X-Approval-Token" not in seen[0]
