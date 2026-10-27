"""P3 27.10.2026: mcp >= ~1.24 laesst hinter nginx nur localhost zu (421).
MCP_ALLOWED_HOSTS gibt die oeffentlichen Namen frei, der Schutz bleibt an."""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
for _k in ("ARKTURIAN_API_KEY", "ONEAL_STORAGE_API_KEY", "JWT_PUBLIC_KEY", "COMM_API_KEY"):
    os.environ.setdefault(_k, "test-only")


def test_ohne_variable_unveraendert(monkeypatch):
    import server
    monkeypatch.delenv("MCP_ALLOWED_HOSTS", raising=False)
    assert server._transport_security() is None


def test_mit_variable_name_erlaubt_schutz_an(monkeypatch):
    import pytest
    import server
    pytest.importorskip("mcp.server.transport_security")
    monkeypatch.setenv("MCP_ALLOWED_HOSTS", "mcp.p3.arkturian.com, x.example")
    ts = server._transport_security()
    assert ts.enable_dns_rebinding_protection is True
    assert "mcp.p3.arkturian.com" in ts.allowed_hosts and "x.example:*" in ts.allowed_hosts
    assert "localhost:*" in ts.allowed_hosts
    assert "https://mcp.p3.arkturian.com" in ts.allowed_origins


def test_fabrik_reicht_es_durch(monkeypatch):
    import pytest
    import server
    pytest.importorskip("mcp.server.transport_security")
    monkeypatch.setenv("MCP_ALLOWED_HOSTS", "mcp.p3.arkturian.com")
    m = server.FastMCP(name="probe", streamable_http_path="/", stateless_http=True)
    assert "mcp.p3.arkturian.com" in m.settings.transport_security.allowed_hosts
