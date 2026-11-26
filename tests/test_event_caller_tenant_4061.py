"""#4061: event-api calls carry the verified agent name and its home tenant.

The event-api grants access only for name *and* home tenant (X-Caller-Tenant,
the `tenant` claim). A user token's sub is not an agent name.
"""
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
for _k in ("ARKTURIAN_API_KEY", "ONEAL_STORAGE_API_KEY", "JWT_PUBLIC_KEY", "COMM_API_KEY"):
    os.environ.setdefault(_k, "test-only")

import auth  # noqa: E402
import server  # noqa: E402


@pytest.fixture
def caller(monkeypatch):
    monkeypatch.setattr(server, "EVENT_API_KEY", "gw-test")
    tokens = []

    def setze(home, name="Inno4def", is_agent=True):
        tokens.append((auth._current_home_tenant.set(home), auth._current_agent_name.set(name),
                       auth._current_is_agent.set(is_agent)))
    yield setze
    for h, n, a in reversed(tokens):
        auth._current_is_agent.reset(a)
        auth._current_agent_name.reset(n)
        auth._current_home_tenant.reset(h)


def test_agent_name_and_home_tenant_are_sent(caller):
    caller("lesser")
    h = server._event_headers()
    assert h["X-API-Key"] == "gw-test"
    assert h["X-Agent-Name"] == "Inno4def"
    assert h["X-Caller-Tenant"] == "lesser"
    assert "Authorization" not in h


def test_no_home_tenant_header_without_tenant_claim(caller):
    caller("")
    assert "X-Caller-Tenant" not in server._event_headers()


def test_user_token_sub_is_not_an_agent_name(caller):
    caller("arkturian", name="someone@example.org", is_agent=False)
    h = server._event_headers()
    assert "X-Agent-Name" not in h


def test_non_ascii_agent_name_is_encoded(caller):
    caller("arkturian", name="Förderungen")
    h = server._event_headers()
    assert h["X-Agent-Name"] == "F%C3%B6rderungen" and h["X-Agent-Name-Encoding"] == "pct"
    assert h["X-Caller-Tenant"] == "arkturian"


def test_without_gateway_key_nothing_is_sent(caller, monkeypatch):
    monkeypatch.setattr(server, "EVENT_API_KEY", "")
    caller("arkturian")
    with pytest.raises(ValueError, match="event_unavailable"):
        server._event_headers()
