"""#3888 Punkt 2: inbox(msg_id=…) prueft bei cloud-api, ob eine IACP-Nachricht wirklich beim Agenten
ankam (GET /api/agents/self/eingang/{msg_id}, Agent aus dem JWT). Ohne msg_id wie bisher die Inbox."""
import asyncio
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
for _k in ("ARKTURIAN_API_KEY", "ONEAL_STORAGE_API_KEY", "JWT_PUBLIC_KEY", "COMM_API_KEY"):
    os.environ.setdefault(_k, "test-only")


@pytest.fixture
def als_agent():
    import auth
    t = auth._current_agent_name.set("Neon"), auth._current_is_agent.set(True)
    yield
    auth._current_agent_name.reset(t[0])
    auth._current_is_agent.reset(t[1])


def _fake(calls, antwort):
    async def _f(method, url, **kw):
        calls.append((method, url))
        return antwort
    return _f


def test_mit_msg_id_fragt_den_eingang(monkeypatch, als_agent):
    import server
    calls = []
    monkeypatch.setattr(server, "_fetch_json", _fake(calls, {"angekommen": False}))
    r = asyncio.run(server.cloud_inbox("Neon", msg_id=" e869c18a "))
    assert r == {"angekommen": False}
    assert calls[0][0] == "GET" and calls[0][1].endswith("/api/agents/self/eingang/e869c18a")


def test_id_wird_kodiert(monkeypatch, als_agent):
    import server
    calls = []
    monkeypatch.setattr(server, "_fetch_json", _fake(calls, {}))
    asyncio.run(server.cloud_inbox("Neon", msg_id="a/../b"))
    assert calls[0][1].endswith("/api/agents/self/eingang/a%2F..%2Fb")


def test_ohne_msg_id_wie_bisher(monkeypatch, als_agent):
    import server
    calls = []
    monkeypatch.setattr(server, "_fetch_json", _fake(calls, {"messages": []}))
    asyncio.run(server.cloud_inbox("Neon"))
    assert calls[0][1].endswith("/api/agents/inbox/Neon")
