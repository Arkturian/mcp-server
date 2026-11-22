"""#3695: MCP-Werkzeug session_delete ruft DELETE /api/sessions/{name} der cloud-api.

Die Rechte entscheidet allein cloud-api am JWT des Aufrufers (owner_or_admin); das Werkzeug
reicht nur durch, kodiert den Namen als EIN Pfadsegment und gibt Fehler mit Grund zurueck.
"""
import asyncio
import os
import sys
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
for _k in ("ARKTURIAN_API_KEY", "ONEAL_STORAGE_API_KEY", "JWT_PUBLIC_KEY", "COMM_API_KEY"):
    os.environ.setdefault(_k, "test-only")


@pytest.fixture
def aufrufe(monkeypatch):
    import server
    gesehen = []

    async def _call(method, path, params=None, json_body=None, **kw):
        gesehen.append((method, path, params, json_body, kw))
        return {"status": "purged", "name": "x", "actions": ["tmux_killed", "registry_purged"]}
    monkeypatch.setattr(server, "call_cloud_api", _call)
    return gesehen


def test_loescht_ueber_die_cloud_api_mit_purge(aufrufe):
    import server
    r = asyncio.run(server.cloud_session_delete("C-Weiss-Praesentationen"))
    assert r["actions"] == ["tmux_killed", "registry_purged"]
    assert aufrufe == [("DELETE", "/api/sessions/C-Weiss-Praesentationen", {"purge": "true"}, None, {})]


def test_purge_false_nur_beenden(aufrufe):
    import server
    asyncio.run(server.cloud_session_delete("Demo", purge=False))
    assert aufrufe[0][:3] == ("DELETE", "/api/sessions/Demo", {"purge": "false"})


@pytest.mark.parametrize("name, pfad", [
    ("Project Manager", "/api/sessions/Project%20Manager"),
    ("X/restart", "/api/sessions/X%2Frestart"),
    ("../config", "/api/sessions/..%2Fconfig"),
    ("a?purge=false", "/api/sessions/a%3Fpurge%3Dfalse"),
    ("Jürgen", "/api/sessions/J%C3%BCrgen"),
])
def test_name_bleibt_ein_pfadsegment(aufrufe, name, pfad):
    import server
    asyncio.run(server.cloud_session_delete(name))
    assert aufrufe[0][1] == pfad and aufrufe[0][2] == {"purge": "true"}


@pytest.mark.parametrize("status, rumpf", [
    (404, {"detail": "session not found"}),
    (403, {"detail": {"error": "nicht_erlaubt"}}),
])
def test_fehler_kommt_mit_status_und_grund(monkeypatch, status, rumpf):
    import server
    req = httpx.Request("DELETE", "http://x/api/sessions/Fremd")
    resp = httpx.Response(status, json=rumpf, request=req)

    async def _call(*a, **k):
        raise httpx.HTTPStatusError(str(status), request=req, response=resp)
    monkeypatch.setattr(server, "call_cloud_api", _call)
    assert asyncio.run(server.cloud_session_delete("Fremd")) == {"status": status, **rumpf}


def test_mit_der_identitaet_des_aufrufers_nicht_des_gateways(monkeypatch):
    """call_cloud_api ohne agent_gateway_proof: es zaehlt das JWT des Aufrufers."""
    import server
    gesehen = []

    async def _call(method, path, **kw):
        gesehen.append(kw)
        return {}
    monkeypatch.setattr(server, "call_cloud_api", _call)
    asyncio.run(server.cloud_session_delete("Demo"))
    assert "agent_gateway_proof" not in gesehen[0]


def test_beschreibung_nennt_endgueltig_und_auftrag():
    import server
    werkzeug = next(t for t in server.cloud_mcp._tool_manager.list_tools() if t.name == "session_delete")
    text = werkzeug.description
    assert "Cannot be undone" in text and "human has asked" in text and "404" in text
