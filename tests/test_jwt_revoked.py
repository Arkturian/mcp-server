import jwt, pytest
import auth


def test_revoked_jti_wird_abgewiesen():
    with pytest.raises(jwt.InvalidTokenError):
        auth._reject_if_revoked({"jti": "c9a8dc32-9150-4891-b949-5e184a7dcff3", "sub": "agent:Wave"})


def test_env_override_und_normalfall(monkeypatch):
    monkeypatch.setenv("REVOKED_AGENT_JTIS", "x1,x2")
    with pytest.raises(jwt.InvalidTokenError):
        auth._reject_if_revoked({"jti": "x2"})
    auth._reject_if_revoked({"jti": "ok"})
    auth._reject_if_revoked({})
