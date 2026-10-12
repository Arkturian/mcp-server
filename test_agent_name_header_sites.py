"""#2402: die drei Sendestellen des Gateways kodieren den Namen (ausgefuehrt)."""
import os
import types

import httpx

os.environ.setdefault("MCP_SERVERS", "cloud")
os.environ.setdefault("ARKTURIAN_API_KEY", "nur-fuer-den-test")

import auth  # noqa: E402
import server  # noqa: E402
from agent_identity_header import get_agent_name


def test_caller_auth_headers_mit_umlaut_gehen_durch_httpx(monkeypatch):
    monkeypatch.setattr(auth, "current_caller_jwt", lambda: "signed.jwt")
    monkeypatch.setattr(auth, "current_caller_agent_name", lambda: "Kunde-Handbücher")
    h = server._caller_auth_headers()
    req = httpx.Request("GET", "http://x.test/", headers=h)        # warf vorher UnicodeEncodeError
    assert get_agent_name(req.headers) == "Kunde-Handbücher"
    assert req.headers["Authorization"] == "Bearer signed.jwt"


def test_caller_auth_headers_ascii_unveraendert(monkeypatch):
    monkeypatch.setattr(auth, "current_caller_jwt", lambda: "signed.jwt")
    monkeypatch.setattr(auth, "current_caller_agent_name", lambda: "AppDev")
    assert server._caller_auth_headers() == {"Authorization": "Bearer signed.jwt", "X-Agent-Name": "AppDev"}


def test_get_agent_headers_kodiert_und_behaelt_die_uebrigen(monkeypatch):
    agent = auth.AgentContext()
    agent.authenticated = True
    agent.name, agent.role, agent.department, agent.tenant = "Büro Ost", "developer", "engineering", "kunde"
    req = types.SimpleNamespace(state=types.SimpleNamespace(agent=agent))
    h = auth.get_agent_headers(req)
    httpx.Request("GET", "http://x.test/", headers=h)
    assert get_agent_name(h) == "Büro Ost" and h["X-Agent-Role"] == "developer" and h["X-Agent-Tenant"] == "kunde"
