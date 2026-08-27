import jwt, pytest
import auth


def test_feste_liste_leer(monkeypatch):
    monkeypatch.delenv("REVOKED_AGENT_JTIS", raising=False)
    monkeypatch.setattr(auth, "_revoked_polled", set()); monkeypatch.setattr(auth, "_revoked_loaded", True)
    auth._reject_if_revoked({"jti": "c9a8dc32-9150-4891-b949-5e184a7dcff3"})


def test_env_override_und_normalfall(monkeypatch):
    monkeypatch.setenv("REVOKED_AGENT_JTIS", "x1,x2")
    with pytest.raises(jwt.InvalidTokenError):
        auth._reject_if_revoked({"jti": "x2"})
    auth._reject_if_revoked({"jti": "ok"})
    auth._reject_if_revoked({})


class _Resp:
    def __init__(self, code, data): self.status_code, self._d = code, data
    def json(self): return self._d


class _Client:
    def __init__(self, pages): self.pages, self.calls = list(pages), []
    async def get(self, url, params=None):
        self.calls.append((url, dict(params or {})))
        return self.pages.pop(0) if self.pages else _Resp(200, {"revoked_jtis": [], "as_of": "t9"})
    async def aclose(self): pass


def test_poll_fuellt_deny_liste(monkeypatch, tmp_path):
    import asyncio
    monkeypatch.setattr(auth, "_REVOKED_STATE", str(tmp_path / "r.json"))
    monkeypatch.setattr(auth, "_revoked_polled", set()); monkeypatch.setattr(auth, "_revoked_as_of", ""); monkeypatch.setattr(auth, "_revoked_loaded", False)
    c = _Client([_Resp(200, {"revoked_jtis": [{"jti": "polled-1", "revoked_at": "t1"}], "as_of": "t1"})])
    assert asyncio.run(auth.revoked_poll_once(c)) == 1
    assert c.calls[0][0].endswith("/revoked-agent-jtis")
    with pytest.raises(jwt.InvalidTokenError):
        auth._reject_if_revoked({"jti": "polled-1"})
    assert asyncio.run(auth.revoked_poll_once(c)) == 0 and c.calls[1][1] == {"since": "t1"}
    asyncio.run(auth.revoked_poll_once(_Client([_Resp(503, {})])))
    with pytest.raises(jwt.InvalidTokenError):
        auth._reject_if_revoked({"jti": "polled-1"})
