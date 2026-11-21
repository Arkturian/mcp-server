"""#3654: documents_send reicht nur das geprüfte Agenten-JWT an business-api."""
import asyncio
import os

import pytest

for _k in ("ARKTURIAN_API_KEY", "ONEAL_STORAGE_API_KEY", "JWT_PUBLIC_KEY", "COMM_API_KEY"):
    os.environ.setdefault(_k, "test-only")

import auth  # noqa: E402
import server  # noqa: E402


@pytest.fixture
def anfragen(monkeypatch):
    gesehen = []

    async def _fetch(method, url, **kw):
        gesehen.append({"method": method, "url": url, **kw})
        return {"id": 1}
    monkeypatch.setattr(server, "_fetch_json", _fetch)
    monkeypatch.setattr(server, "BUSINESS_API_KEY", "GATEWAY")
    monkeypatch.setattr(server, "BUSINESS_API_BASE", "https://business.test")
    return gesehen


def _rufer(monkeypatch, agent, jwt):
    monkeypatch.setattr(auth, "current_caller_verified_agent", lambda: agent)
    monkeypatch.setattr(auth, "current_caller_jwt", lambda: jwt)


def _sende():
    asyncio.run(server.business_mcp.call_tool("documents_send", {"doc_id": 7}))


def test_agent_token_geht_mit(anfragen, monkeypatch):
    _rufer(monkeypatch, "Business", "a.b.c")
    _sende()
    k = anfragen[0]["headers"]
    assert k["X-Agent-Token"] == "a.b.c" and k["X-API-Key"] == "GATEWAY"


@pytest.mark.parametrize("agent,jwt", [("", "user.jwt"), ("", ""), ("Business", "")])
def test_ohne_geprueften_agenten_keine_kopfzeile(anfragen, monkeypatch, agent, jwt):
    _rufer(monkeypatch, agent, jwt)
    _sende()
    assert "X-Agent-Token" not in anfragen[0]["headers"]


def test_andere_business_werkzeuge_bekommen_es_nicht(anfragen, monkeypatch):
    _rufer(monkeypatch, "Business", "a.b.c")
    asyncio.run(server.business_mcp.call_tool("documents_cancel", {"doc_id": 7}))
    assert "X-Agent-Token" not in anfragen[0]["headers"]
