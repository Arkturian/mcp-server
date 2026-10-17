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


def test_stueck_upload_bekommt_scope(monkeypatch):
    import server
    calls = []
    monkeypatch.setattr(server, "_fetch_json", _fake_fetch(calls))
    t = _als("Cloud", True)
    try:
        asyncio.run(server.storage_assets_upload_chunked(
            upload_id="u1", index=0, total_chunks=1, filename="a.txt", chunk_base64="YWJj"))
    finally:
        _zurueck(t)
    assert calls[0][0].endswith("/storage/upload-chunk")
    assert calls[0][1]["params"]["grant_scope"] == "cloud-session:Cloud"


# ── share_with (Post 5484): weitere Agenten beim Upload freigeben ──────────

_AGENTEN = {
    "DRONE-CHALLENGE": {"name": "DRONE-CHALLENGE", "tenant_id": "arkturian", "is_active": True},
    "Comm": {"name": "Comm", "tenant_id": "arkturian", "is_active": True},
    "Förderungen": {"name": "Förderungen", "tenant_id": "arkturian", "is_active": True},
    "Kunde": {"name": "Kunde", "tenant_id": "ele", "is_active": True},
    "Alt": {"name": "Alt", "tenant_id": "arkturian", "is_active": False},
    "Ohnefeld": {"name": "Ohnefeld", "tenant_id": "arkturian"},
}


def _fake_auth(monkeypatch, gesehen):
    import server
    from urllib.parse import unquote

    class _R:
        def __init__(self, row):
            self.row = row
            self.status_code = 200 if row else 404

        def raise_for_status(self):
            pass

        def json(self):
            return self.row

    class _Client:
        def __init__(self, *a, **kw):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def get(self, url, headers=None):
            gesehen.append((url, headers))
            return _R(_AGENTEN.get(unquote(url.rsplit("/", 1)[1])))

    monkeypatch.setattr(server.httpx, "AsyncClient", _Client)


def _mit_jwt(name, is_agent=True):
    import auth
    return _als(name, is_agent) + (auth._current_jwt.set("jwt-of-caller"),)


def _ohne_jwt(tokens):
    import auth
    _zurueck(tokens[:2])
    auth._current_jwt.reset(tokens[2])


def test_share_with_bildet_namen_ab(monkeypatch):
    import server
    gesehen = []
    _fake_auth(monkeypatch, gesehen)
    t = _mit_jwt("DRONE-CHALLENGE")
    try:
        scopes = asyncio.run(server._share_with_scopes(["Comm", "Förderungen", "Comm"]))
    finally:
        _ohne_jwt(t)
    assert scopes == ["cloud-session:Comm", "cloud-session-x:" + "Förderungen".encode().hex()]
    assert all(h == {"Authorization": "Bearer jwt-of-caller"} for _, h in gesehen)


def test_share_with_lehnt_ab(monkeypatch):
    import pytest
    import server
    _fake_auth(monkeypatch, [])
    for liste, code in ((["Comn"], "share_with_unknown_agent: Comn"),
                        (["Kunde"], "share_with_foreign_tenant: Kunde"),
                        (["Alt"], "share_with_unknown_agent: Alt"),
                        (["Ohnefeld"], "share_with_unknown_agent: Ohnefeld"),
                        ([f"a{i}" for i in range(20)], "share_with_too_many")):
        t = _mit_jwt("DRONE-CHALLENGE")
        try:
            with pytest.raises(ValueError, match=code):
                asyncio.run(server._share_with_scopes(liste))
        finally:
            _ohne_jwt(t)
    t = _mit_jwt("user:abc", is_agent=False)
    try:
        with pytest.raises(ValueError, match="share_with_requires_agent"):
            asyncio.run(server._share_with_scopes(["Comm"]))
        assert asyncio.run(server._share_with_scopes(None)) == []
    finally:
        _ohne_jwt(t)


def test_share_with_landet_im_grant_scope(monkeypatch):
    import server
    calls = []
    monkeypatch.setattr(server, "_fetch_json", _fake_fetch(calls))

    async def _scopes(sw):
        return ["cloud-session:Comm"] if sw else []
    monkeypatch.setattr(server, "_share_with_scopes", _scopes)
    t = _als("DRONE-CHALLENGE", True)
    try:
        asyncio.run(server.storage_assets_upload_ticket(filename="a.pdf", share_with=["Comm"]))
        asyncio.run(server.storage_assets_fetch(url="https://x/y.pdf", share_with=["Comm"]))
        asyncio.run(server.storage_assets_upload_chunked(
            upload_id="u", index=0, total_chunks=1, filename="a.pdf",
            chunk_base64="QQ==", share_with=["Comm"]))
        asyncio.run(server.storage_assets_upload_ticket(filename="b.pdf"))
    finally:
        _zurueck(t)
    want = "cloud-session:DRONE-CHALLENGE,cloud-session:Comm"
    assert calls[0][1]["params"]["grant_scope"] == want
    assert calls[1][1]["json_body"]["grant_scope"] == want
    assert calls[2][1]["params"]["grant_scope"] == want
    assert calls[3][1]["params"]["grant_scope"] == "cloud-session:DRONE-CHALLENGE"


def test_share_with_im_multipart_upload(monkeypatch, tmp_path):
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
            gesendet.update(data)
            return _Resp()

    async def _scopes(sw):
        return ["cloud-session:Comm"] if sw else []
    monkeypatch.setattr(server.httpx, "AsyncClient", _Client)
    monkeypatch.setattr(server, "_share_with_scopes", _scopes)
    f = tmp_path / "a.pdf"
    f.write_bytes(b"%PDF")
    t = _als("DRONE-CHALLENGE", True)
    try:
        asyncio.run(server.storage_assets_upload_file(path=str(f), private=True, share_with=["Comm"]))
    finally:
        _zurueck(t)
    assert gesendet["grant_scope"] == "cloud-session:DRONE-CHALLENGE,cloud-session:Comm"
    assert gesendet["private"] == "true"


def test_unbekannter_name_speichert_nichts(monkeypatch, tmp_path):
    """Die Prüfung läuft vor dem Upload: bei einem Fehler geht nichts an storage-api."""
    import pytest
    import server
    _fake_auth(monkeypatch, [])
    calls = []
    monkeypatch.setattr(server, "_fetch_json", _fake_fetch(calls))
    f = tmp_path / "a.pdf"
    f.write_bytes(b"%PDF")
    t = _mit_jwt("DRONE-CHALLENGE")
    try:
        with pytest.raises(ValueError, match="share_with_unknown_agent"):
            asyncio.run(server.storage_assets_upload_file(path=str(f), share_with=["Comn"]))
        with pytest.raises(ValueError, match="share_with_unknown_agent"):
            asyncio.run(server.storage_assets_upload_ticket(filename="a.pdf", share_with=["Comn"]))
    finally:
        _ohne_jwt(t)
    assert calls == []
