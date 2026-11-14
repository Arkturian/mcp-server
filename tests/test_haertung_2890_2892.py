"""Gateway-Haertung nach Cloud-Security (14.11.2026, AP 1.3).

#2890  Fehlte der `permissions`-Claim, galt ["*"]: jedes Benutzer-, OIDC- oder
       OAuth-Token hatte alle Rechte. Ausserdem nannte `_caller_auth_headers` bei
       Benutzer-Tokens deren `sub` als X-Agent-Name, und `_cloud_request_headers`
       haengte daran den Gateway-Nachweis samt X-IACP-Token.
#2891  assets_upload_file las jeden Pfad des Gateway-Hosts (root), Outlook
       deliver='file' schrieb an jeden Pfad.
#2892  send_to_role: ohne Aufrufer, pauschales X-IACP-Token, `role` unkodiert im
       Pfad (httpx normalisiert `..` vor dem Versand).

Alles gegen Vorrichtungen: kein Netz, kein echtes Token, keine Datei ausserhalb
von tmp_path.
"""
import asyncio
import json
import os

import httpx
import pytest
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route

for _k in ("ARKTURIAN_API_KEY", "ONEAL_STORAGE_API_KEY", "JWT_PUBLIC_KEY", "COMM_API_KEY"):
    os.environ.setdefault(_k, "test-only")

import auth  # noqa: E402
import server  # noqa: E402

CALL = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                   "params": {"name": "werkzeug", "arguments": {}}})


# --- #2890: Rechte aus den Claims -------------------------------------------------

@pytest.mark.parametrize("claims,erwartet", [
    ({"type": "agent", "agent_name": "A", "permissions": ["mcp:cloud:*"]}, ["mcp:cloud:*"]),
    ({"type": "agent", "agent_name": "A"}, []),                                   # ohne Claim: nichts
    ({"type": "service", "sub": "service:comm-api", "permissions": ["x:y"]}, ["x:y"]),
    ({"sub": "u1", "email": "mensch@example.test", "is_superadmin": False}, []),   # Login-Token
    ({"sub": "u1", "is_superadmin": True}, ["*"]),                                 # Betreiber
    ({"sub": "u1", "aud": "cid", "scope": "openid mcp:cloud mcp:content"}, ["mcp:cloud:*", "mcp:content:*"]),
    ({"sub": "u1", "aud": "cid", "scope": "openid profile email"}, []),
    ({"sub": "u1", "aud": "cid", "scope": "mcp:cloud mcp:*:x room:read"}, ["mcp:cloud:*"]),
    ({"sub": "u1", "aud": "cid", "scope": "mcp:cloud", "is_superadmin": True}, ["mcp:cloud:*"]),
])
def test_rechte_aus_claims(claims, erwartet):
    assert auth._rechte_aus_claims(claims) == erwartet


def test_superadmin_oauth_ohne_scope_schaltbar(monkeypatch):
    c = {"sub": "u1", "aud": "cid", "scope": "openid", "is_superadmin": True}
    monkeypatch.setattr(auth, "_SUPERADMIN_OHNE_SCOPE", True)
    assert auth._rechte_aus_claims(c) == ["*"]
    monkeypatch.setattr(auth, "_SUPERADMIN_OHNE_SCOPE", False)
    assert auth._rechte_aus_claims(c) == []


def test_kontext_ohne_rechte_hat_keine():
    assert auth.AgentContext().permissions == []
    assert not auth.AgentContext().has_permission("mcp:cloud:send_message")


@pytest.mark.parametrize("aud,abgelehnt", [("opencloud-web", True), (["x", "OpenCloud-Desktop"], True),
                                           ("chatgpt-cid", False), (None, False)])
def test_fremde_zielgruppe(aud, abgelehnt):
    assert auth._aud_abgelehnt({"aud": aud} if aud is not None else {}) is abgelehnt


def _app():
    async def ok(request):
        return JSONResponse({"ok": 1, "perms": request.state.agent.permissions})
    app = Starlette(routes=[Route("/{srv}/", ok, methods=["POST"])])
    app.add_middleware(auth.JWTAuthMiddleware)
    return app


@pytest.fixture
def gateway(monkeypatch):
    async def key():
        return "k"
    monkeypatch.setattr(auth, "_get_public_key", key)

    def mit(claims):
        monkeypatch.setattr(auth, "_decode_jwt", lambda token, key: dict(claims))

    def post(pfad):
        async def go():
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=_app()), base_url="http://gw") as c:
                return await c.post(pfad, content=CALL, headers={
                    "Authorization": "Bearer t", "Content-Type": "application/json"})
        return asyncio.run(go())
    return mit, post


def test_benutzer_token_ohne_rechte_kommt_nicht_durch(gateway):
    mit, post = gateway
    mit({"sub": "u1", "email": "mensch@example.test", "is_superadmin": False})
    assert post("/cloud/").status_code == 403


def test_oauth_scope_oeffnet_nur_seinen_server(gateway):
    mit, post = gateway
    mit({"sub": "u1", "aud": "chatgpt-cid", "scope": "openid mcp:cloud"})
    assert post("/cloud/").status_code == 200
    assert post("/content/").status_code == 403


def test_opencloud_token_wird_abgewiesen(gateway):
    mit, post = gateway
    mit({"sub": "u1", "aud": "opencloud-web", "is_superadmin": True})
    r = post("/cloud/")
    assert r.status_code == 401 and "audience" in r.text


@pytest.mark.parametrize("srv", ["business", "oneal-storage", "codepilot", "artrack", "story"])
def test_hauptschluessel_server_nur_ausdruecklich(gateway, srv):
    mit, post = gateway
    assert srv in auth.EXPLICIT_ONLY_SERVERS
    mit({"type": "agent", "agent_name": "Admin", "permissions": ["*"]})
    assert post(f"/{srv}/").status_code == 403
    mit({"sub": "u1", "is_superadmin": True})                                     # "*" als Betreiber
    assert post(f"/{srv}/").status_code == 403
    mit({"type": "agent", "agent_name": "Fach", "permissions": [f"mcp:{srv}:*"]})
    assert post(f"/{srv}/").status_code == 200


def test_agent_mit_rechten_wie_bisher(gateway):
    mit, post = gateway
    mit({"type": "agent", "agent_name": "Dev", "permissions": ["mcp:*:*"]})
    assert post("/cloud/").status_code == 200


# --- #2890: Kopfzeilen nur fuer gepruefte Agenten ---------------------------------

def _kontext(jwt="signed.jwt", name="", agent=False):
    return auth._current_jwt.set(jwt), auth._current_agent_name.set(name), auth._current_is_agent.set(agent)


def _zurueck(t):
    auth._current_jwt.reset(t[0]); auth._current_agent_name.reset(t[1]); auth._current_is_agent.reset(t[2])


def test_benutzer_token_ist_kein_agent_und_bekommt_kein_iacp_token(monkeypatch):
    monkeypatch.setattr(server, "IACP_TOKEN", "foederation")
    t = _kontext(name="mensch@example.test", agent=False)
    try:
        h = server._cloud_request_headers()
    finally:
        _zurueck(t)
    assert h == {"Authorization": "Bearer signed.jwt"}


def test_agent_token_traegt_namen_und_nachweis(monkeypatch):
    monkeypatch.setattr(server, "IACP_TOKEN", "foederation")
    t = _kontext(name="AppDev", agent=True)
    try:
        h = server._cloud_request_headers()
    finally:
        _zurueck(t)
    assert h["X-Agent-Name"] == "AppDev" and h["X-Agent-Gateway"] == "mcp" and h["X-IACP-Token"] == "foederation"


# --- #2892: send_to_role und Pfadsegmente ----------------------------------------

@pytest.fixture
def gesendet(monkeypatch):
    rufe = []

    async def fake(method, url, **kw):
        rufe.append((method, url, kw))
        return {"id": "m1", "status": "queued"}
    monkeypatch.setattr(server, "_fetch_json", fake)
    monkeypatch.setattr(server, "IACP_TOKEN", "foederation")
    return rufe


def test_send_to_role_nur_fuer_agenten(gesendet):
    t = _kontext(name="mensch@example.test", agent=False)
    try:
        r = asyncio.run(server.cloud_send_to_role("finance", "hallo"))
    finally:
        _zurueck(t)
    assert r["error"] == "send_to_role_nur_fuer_agenten" and gesendet == []


@pytest.mark.parametrize("rolle", ["../../api/sessions/X/restart", "a/b", "fin?x=1", "", "a b", "x" * 65, "..", "%2e%2e"])
def test_send_to_role_rolle_nach_muster(gesendet, rolle):
    t = _kontext(name="AppDev", agent=True)
    try:
        r = asyncio.run(server.cloud_send_to_role(rolle, "hallo"))
    finally:
        _zurueck(t)
    assert r["error"] == "role_invalid" and gesendet == []


def test_send_to_role_mit_absender_aus_dem_token(gesendet):
    t = _kontext(name="AppDev", agent=True)
    try:
        asyncio.run(server.cloud_send_to_role("finance", "hallo", user_id="gefaelscht"))
    finally:
        _zurueck(t)
    [(method, url, kw)] = gesendet
    assert method == "POST" and url.endswith("/api/iacp/send/finance")
    assert kw["json_body"] == {"text": "hallo", "user_id": "AppDev", "from_session": "AppDev"}
    assert kw["headers"]["Authorization"] == "Bearer signed.jwt" and kw["headers"]["X-Agent-Name"] == "AppDev"


@pytest.mark.parametrize("url", ["https://c/api/a/../b", "https://c/api/%2e%2e/b", "https://c/api/%252E%252e/b",
                                 "https://c/api/./b", "https://c/api/x/.."])
def test_pfadwaechter_weist_punktsegmente_ab(url):
    with pytest.raises(ValueError):
        server._pfad_sicher(url)


@pytest.mark.parametrize("url", ["https://c/api/sessions/Cloud-worker", "https://c/api/x%2Fy/z",
                                 "https://c/api/a.b/..c", "https://c/api/v1/outlook/s/messages/AAMk%2Bx%3D"])
def test_pfadwaechter_laesst_normale_pfade(url):
    server._pfad_sicher(url)


def test_fetch_json_schickt_keinen_ausbruch(monkeypatch):
    erzeugt = []

    class _Nie:
        def __init__(self, *a, **k):
            erzeugt.append(1)
    monkeypatch.setattr(server.httpx, "AsyncClient", _Nie)
    with pytest.raises(ValueError):
        asyncio.run(server._fetch_json("POST", "https://c/api/iacp/send/../../sessions/X/restart", headers={}))
    assert erzeugt == []


@pytest.mark.parametrize("wert,ok", [("finance", "finance"), ("a/b", "a%2Fb"), ("x?y#z", "x%3Fy%23z"),
                                     ("Kunde-Handbücher", "Kunde-Handb%C3%BCcher"), (42, "42")])
def test_seg_kodiert_ein_segment(wert, ok):
    assert server._seg(wert) == ok


@pytest.mark.parametrize("wert", ["", ".", "..", None, "a\nb"])
def test_seg_weist_ab(wert):
    with pytest.raises(ValueError):
        server._seg(wert)


# --- #2891: assets_upload_file nur unter erlaubten Wurzeln ----------------------

@pytest.fixture
def upload(monkeypatch, tmp_path):
    wurzel = tmp_path / "erlaubt"
    wurzel.mkdir()
    monkeypatch.setenv("MCP_UPLOAD_WURZELN", str(wurzel))
    hoch = []

    async def fake_upload(data, name, form_fields=None, share_scopes=None):
        hoch.append((data, name))
        return {"id": 1}

    async def keine_scopes(share_with):
        return []
    monkeypatch.setattr(server, "call_storage_upload", fake_upload)
    monkeypatch.setattr(server, "_share_with_scopes", keine_scopes)
    return wurzel, hoch


def test_upload_aus_der_wurzel(upload):
    wurzel, hoch = upload
    (wurzel / "bild.png").write_bytes(b"PNG")
    asyncio.run(server.storage_assets_upload_file(str(wurzel / "bild.png")))
    assert hoch == [(b"PNG", "bild.png")]


def test_upload_ausserhalb_wird_nicht_gelesen(upload, tmp_path):
    wurzel, hoch = upload
    geheim = tmp_path / "private.pem"
    geheim.write_bytes(b"GEHEIM")
    with pytest.raises(ValueError, match="not allowed"):
        asyncio.run(server.storage_assets_upload_file(str(geheim)))
    with pytest.raises(ValueError, match="not allowed"):
        asyncio.run(server.storage_assets_upload_file(str(wurzel / ".." / "private.pem")))
    assert hoch == []


def test_upload_verweis_aus_der_wurzel_heraus(upload, tmp_path):
    wurzel, hoch = upload
    geheim = tmp_path / ".env"
    geheim.write_bytes(b"GEHEIM")
    (wurzel / "harmlos.txt").symlink_to(geheim)
    with pytest.raises(ValueError, match="not allowed"):
        asyncio.run(server.storage_assets_upload_file(str(wurzel / "harmlos.txt")))
    assert hoch == []


def test_upload_abgeschaltet(monkeypatch, tmp_path):
    monkeypatch.setenv("MCP_UPLOAD_WURZELN", "")
    f = tmp_path / "x.txt"
    f.write_bytes(b"x")
    with pytest.raises(ValueError, match="aus"):
        asyncio.run(server.storage_assets_upload_file(str(f)))


def test_upload_vorgabe_ist_eigenes_verzeichnis(monkeypatch):
    monkeypatch.delenv("MCP_UPLOAD_WURZELN", raising=False)
    assert [str(w) for w in server._upload_wurzeln()] == ["/tmp/mcp-upload"]
