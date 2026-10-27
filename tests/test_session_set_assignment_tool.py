"""#2590: MCP-Werkzeug session_set_assignment."""
import asyncio
import os
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
for _k in ("ARKTURIAN_API_KEY", "ONEAL_STORAGE_API_KEY", "JWT_PUBLIC_KEY", "COMM_API_KEY"):
    os.environ.setdefault(_k, "test-only")


def test_nur_genannte_felder_und_pfad(monkeypatch):
    import server
    gesehen = []

    async def _call(method, path, json_body=None, **kw):
        gesehen.append((method, path, json_body))
        return {"state": "freigabe_angefragt"}
    monkeypatch.setattr(server, "call_cloud_api", _call)
    r = asyncio.run(server.cloud_session_set_assignment("Laptop", owner_email="j@p3", project_id=""))
    assert r == {"state": "freigabe_angefragt"}
    assert gesehen == [("POST", "/api/sessions/Laptop/zuordnung",
                        {"restart": "nein", "owner_email": "j@p3", "project_id": ""})]


def test_fehler_kommt_mit_begruendung(monkeypatch):
    import server
    req = httpx.Request("POST", "http://x/api/sessions/Laptop/zuordnung")
    resp = httpx.Response(409, json={"error": "no_approver"}, request=req)

    async def _call(*a, **k):
        raise httpx.HTTPStatusError("409", request=req, response=resp)
    monkeypatch.setattr(server, "call_cloud_api", _call)
    r = asyncio.run(server.cloud_session_set_assignment("Laptop", role="developer"))
    assert r == {"status": 409, "error": "no_approver"}
