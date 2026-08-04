"""Token validation: audience must not gate, everything else must (2026-08-04).

ChatGPT could not use any MCP server: every call came back
"401 Invalid token: Invalid audience", while Claude Code worked fine.

The asymmetry has a simple cause. PyJWT raises InvalidAudienceError when a
token *carries* an `aud` claim and the verifier passes no expected audience —
an unexpected audience is an error, not something silently ignored. auth-api
stamps `aud=client_id` on OAuth access tokens, and that id is minted per client
by dynamic registration, so this resource server cannot know the valid values
ahead of time. Agents with a static JWT carry no `aud` at all and sailed
through; every OAuth client was rejected.

These tests pin both halves of the contract: the audience must not decide
access, and dropping it must not have loosened anything else.

Run: python3 -m pytest tests/ -q   (pytest is not wired into CI for this repo)
"""

import sys
import time
from pathlib import Path

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import auth as A  # noqa: E402

_KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)
_FOREIGN = rsa.generate_private_key(public_exponent=65537, key_size=2048)


def _token(overrides=None, key=None):
    claims = {
        "iss": A.AUTH_API_ISSUER,
        "sub": "agent:Probe",
        "exp": int(time.time()) + 600,
    }
    claims.update(overrides or {})
    return jwt.encode(claims, key or _KEY, algorithm="RS256")


def test_token_with_audience_is_accepted():
    """THE regression: an OAuth client's token carries aud=client_id."""
    claims = A._decode_jwt(_token({"aud": "client-abc123"}), _KEY.public_key())
    assert claims["sub"] == "agent:Probe"


def test_token_without_audience_still_accepted():
    """Static agent JWTs carry no aud — must keep working unchanged."""
    claims = A._decode_jwt(_token(), _KEY.public_key())
    assert claims["sub"] == "agent:Probe"


def test_expired_token_still_rejected():
    with pytest.raises(jwt.ExpiredSignatureError):
        A._decode_jwt(_token({"exp": int(time.time()) - 10}), _KEY.public_key())


def test_wrong_issuer_still_rejected():
    """Only auth-api may mint tokens for this server."""
    with pytest.raises(jwt.InvalidIssuerError):
        A._decode_jwt(_token({"iss": "https://evil.example"}), _KEY.public_key())


def test_foreign_signature_still_rejected():
    """The signature over the JWKS key is the actual trust anchor — with the
    audience check gone it carries the whole weight, so it had better hold."""
    with pytest.raises(jwt.InvalidSignatureError):
        A._decode_jwt(_token({"aud": "x"}, key=_FOREIGN), _KEY.public_key())
