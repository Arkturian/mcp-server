"""#3811: storage calls carry the caller's home tenant as X-Caller-Tenant.

Only the verified `tenant` claim counts (never `tenant_ids`), and tool
arguments or extra headers can never set it.
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


@pytest.mark.parametrize("claims,expected", [
    ({"type": "agent", "agent_name": "Blender", "tenant": "lesser", "tenant_ids": ["lesser", "arkturian"]}, "lesser"),
    ({"type": "agent", "agent_name": "Storage", "tenant": "arkturian"}, "arkturian"),
    ({"type": "agent", "agent_name": "Old", "tenant": None}, ""),
    ({"type": "agent", "agent_name": "Old"}, ""),
    ({"type": "agent", "agent_name": "X", "tenant": "Les Ser"}, ""),
    ({"type": "agent", "agent_name": "X", "tenant": "lesser,edera"}, ""),
    ({"type": "agent", "agent_name": "X", "tenant_ids": ["edera"]}, ""),
])
def test_home_tenant_is_exactly_the_tenant_claim(claims, expected):
    assert auth.heimat_mandant_aus_claims(claims) == expected


@pytest.fixture
def heimat():
    tokens = []

    def setze(value, agent="Blender"):
        tokens.append((auth._current_home_tenant.set(value), auth._current_agent_name.set(agent),
                       auth._current_is_agent.set(bool(agent))))
    yield setze
    for h, n, a in reversed(tokens):
        auth._current_is_agent.reset(a)
        auth._current_agent_name.reset(n)
        auth._current_home_tenant.reset(h)


def test_header_is_sent_from_the_verified_identity(heimat):
    heimat("lesser")
    assert server._storage_headers()["X-Caller-Tenant"] == "lesser"


def test_no_header_without_a_home_tenant(heimat):
    heimat("")
    assert "X-Caller-Tenant" not in server._storage_headers()


@pytest.mark.parametrize("name", ["X-Caller-Tenant", "x-caller-tenant", "X-CALLER-TENANT"])
def test_extra_headers_cannot_set_or_override_it(heimat, name):
    heimat("")
    assert not any(k.lower() == "x-caller-tenant" for k in server._storage_headers({name: "edera"}))
    heimat("lesser")
    h = server._storage_headers({name: "edera"})
    assert [v for k, v in h.items() if k.lower() == "x-caller-tenant"] == ["lesser"]
