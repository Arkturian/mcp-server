"""Post #5250: storage-MCP bindet neue Objekte an den Scope des Agenten.

Der Scope kommt aus der geprüften Gateway-Identität (type=agent), nie aus
Werkzeug-Argumenten. Benutzer-Tokens und anonyme Aufrufe bekommen keinen.
"""
import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
for _k in ("ARKTURIAN_API_KEY", "ONEAL_STORAGE_API_KEY", "JWT_PUBLIC_KEY", "COMM_API_KEY"):
    os.environ.setdefault(_k, "test-only")


def _als(name: str, is_agent: bool):
    import auth
    return auth._current_agent_name.set(name), auth._current_is_agent.set(is_agent)


def _zurueck(tokens):
    import auth
    auth._current_agent_name.reset(tokens[0])
    auth._current_is_agent.reset(tokens[1])


def _fake_fetch(calls):
    async def _f(method, url, **kw):
        calls.append((url, kw))
        return {"ok": True}
    return _f


def test_scope_abbildung():
    import server
    assert server.storage_grant_scope_for("Cloud") == "cloud-session:Cloud"
    assert server.storage_grant_scope_for("K.I.T.T.") == "cloud-session:K.I.T.T."
    assert server.storage_grant_scope_for("Förderungen") == \
        "cloud-session-x:" + "Förderungen".encode().hex()
    assert server.storage_grant_scope_for("Professor T").startswith("cloud-session-x:")
    assert server.storage_grant_scope_for("") == ""
    lang = "Ä" * 31  # 62 Bytes -> 124 Hex-Zeichen, ueber der Grenze
    sc = server.storage_grant_scope_for(lang)
    assert sc.startswith("cloud-session-x:h") and len(sc.split(":", 1)[1]) == 65
    assert len(server.storage_grant_scope_for("a" * 121).split(":", 1)[1]) <= 120
    for n in ("Cloud", "Förderungen", lang, "a" * 200):
        art, ident = server.storage_grant_scope_for(n).split(":", 1)
        assert len(ident) <= 120 and server._STORAGE_SCOPE_ID_RE.fullmatch(ident)


def test_ticket_und_fetch_bekommen_scope_des_agenten(monkeypatch):
    import server
    calls = []
    monkeypatch.setattr(server, "_fetch_json", _fake_fetch(calls))
    t = _als("Cloud", True)
    try:
        asyncio.run(server.storage_assets_upload_ticket(filename="a.png"))
        asyncio.run(server.storage_assets_fetch(url="https://x/y.png"))
    finally:
        _zurueck(t)
    assert calls[0][1]["params"]["grant_scope"] == "cloud-session:Cloud"
    assert calls[1][1]["json_body"]["grant_scope"] == "cloud-session:Cloud"


def test_fremder_scope_wird_ueberschrieben(monkeypatch):
    import server
    calls = []
    monkeypatch.setattr(server, "_fetch_json", _fake_fetch(calls))
    t = _als("Cloud", True)
    try:
        asyncio.run(server.call_storage_api(
            "POST", "/storage/upload-ticket",
            params={"filename": "a", "grant_scope": "cloud-session:Storage"}))
    finally:
        _zurueck(t)
    assert calls[0][1]["params"]["grant_scope"] == "cloud-session:Cloud"


def test_benutzer_token_bekommt_keinen_scope(monkeypatch):
    """Gegenprobe: gleicher Aufruf, nur type != agent -> kein Scope, auch kein fremder."""
    import server
    calls = []
    monkeypatch.setattr(server, "_fetch_json", _fake_fetch(calls))
    t = _als("user:abc", False)
    try:
        asyncio.run(server.call_storage_api(
            "POST", "/storage/upload-ticket",
            params={"filename": "a", "grant_scope": "cloud-session:Cloud"}))
    finally:
        _zurueck(t)
    assert "grant_scope" not in calls[0][1]["params"]


def test_andere_endpunkte_unberuehrt(monkeypatch):
    import server
    calls = []
    monkeypatch.setattr(server, "_fetch_json", _fake_fetch(calls))
    t = _als("Cloud", True)
    try:
        asyncio.run(server.call_storage_api("GET", "/storage/list", params={"limit": 1}))
    finally:
        _zurueck(t)
    assert "grant_scope" not in calls[0][1]["params"]


def test_multipart_upload_bekommt_scope(monkeypatch):
    import server
    gesendet = {}

    class _Resp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"id": 1}

    class _Client:
        def __init__(self, *a, **kw):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, url, headers=None, files=None, data=None):
            gesendet.update(data or {})
            return _Resp()

    monkeypatch.setattr(server.httpx, "AsyncClient", _Client)
    t = _als("Förderungen", True)
    try:
        asyncio.run(server.storage_assets_upload(file_base64="YWJj", filename="a.txt"))
    finally:
        _zurueck(t)
    assert gesendet["grant_scope"] == "cloud-session-x:" + "Förderungen".encode().hex()
