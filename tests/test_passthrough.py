"""MCP-Durchreiche (P3 26.10.2026): fremder MCP-Server hinter dem Gateway.

Geprueft wird, was die Sicherheit traegt: das Agenten-JWT geht unveraendert
durch, Gateway-Geheimnisse und Cookies nicht, eigene Servernamen lassen sich
nicht ueberschreiben, und ein toter Zieldienst ergibt 502 statt 500.
"""
from __future__ import annotations

import asyncio
import json

import httpx
from starlette.applications import Starlette
from starlette.routing import Mount

import passthrough as pt


def _client_mit(app_ziel_handler, name="p3-buchhaltung", basis="http://ziel.local/mcp"):
    gesehen = {}

    def handler(req: httpx.Request):
        gesehen["url"] = str(req.url)
        gesehen["headers"] = dict(req.headers)
        gesehen["body"] = req.content
        r = app_ziel_handler(req)
        # Response(json=…) gilt in httpx als schon gelesen; ein echter
        # Zieldienst liefert einen Strom, also hier ebenso.
        return httpx.Response(r.status_code, headers=r.headers,
                              stream=httpx.ByteStream(r.content))

    d = pt.Durchreiche(name, basis, transport=httpx.MockTransport(handler))
    app = Starlette(routes=[Mount(f"/{name}", app=d)])
    return app, gesehen


def _lauf(app, method, path, **kw):
    async def go():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                     base_url="http://gw") as c:
            return await c.request(method, path, **kw)
    return asyncio.run(go())


def test_parse_gueltig_und_verworfen():
    r = pt.parse("p3-buchhaltung=http://127.0.0.1:8031/mcp/, 1bad=http://x, y=ftp://x, "
                 "z=http://u:p@h/mcp, issue=http://h/mcp", reserved={"issue"})
    assert r == {"p3-buchhaltung": "http://127.0.0.1:8031/mcp"}
    assert pt.parse("") == {} and pt.parse(None) == {}


def test_jwt_geht_durch_geheimnisse_nicht():
    app, gesehen = _client_mit(lambda req: httpx.Response(200, json={"ok": 1}))
    body = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
    r = _lauf(app, "POST", "/p3-buchhaltung/", content=body, headers={
        "Authorization": "Bearer agent-jwt", "Mcp-Session-Id": "s1",
        "X-IACP-Token": "geheim", "Cookie": "a=b", "Content-Type": "application/json"})
    assert r.status_code == 200 and r.json() == {"ok": 1}
    h = gesehen["headers"]
    assert h["authorization"] == "Bearer agent-jwt"
    assert h["mcp-session-id"] == "s1"
    assert "x-iacp-token" not in h and "cookie" not in h
    assert h["host"] == "ziel.local"
    assert gesehen["url"] == "http://ziel.local/mcp"
    assert gesehen["body"] == body.encode()


def test_unterpfad_und_query():
    app, gesehen = _client_mit(lambda req: httpx.Response(204))
    _lauf(app, "GET", "/p3-buchhaltung/x/y?a=1")
    assert gesehen["url"] == "http://ziel.local/mcp/x/y?a=1"


def test_antwortkoepfe_und_status_kommen_zurueck():
    app, _ = _client_mit(lambda req: httpx.Response(
        202, headers={"Mcp-Session-Id": "neu", "Set-Cookie": "x=y"}, text="data: 1\n\n"))
    r = _lauf(app, "POST", "/p3-buchhaltung/", content=b"{}")
    assert r.status_code == 202
    assert r.headers["mcp-session-id"] == "neu"
    assert "set-cookie" not in r.headers
    assert r.text == "data: 1\n\n"


def test_401_wird_403_ohne_oauth_kopf():
    app, _ = _client_mit(lambda req: httpx.Response(401, headers={
        "WWW-Authenticate": 'Bearer resource_metadata="http://127.0.0.1:8031/.well-known/x"'},
        json={"error": "invalid_token"}))
    r = _lauf(app, "POST", "/p3-buchhaltung/", content=b"{}")
    assert r.status_code == 403
    assert "www-authenticate" not in r.headers
    assert r.json() == {"error": "invalid_token"}


def test_toter_zieldienst_502():
    def weg(req):
        raise httpx.ConnectError("refused")
    app, _ = _client_mit(weg)
    r = _lauf(app, "POST", "/p3-buchhaltung/", content=b"{}")
    assert r.status_code == 502


def test_server_mountet_aus_umgebung_ohne_eigene_namen(monkeypatch):
    monkeypatch.setenv("MCP_PASSTHROUGH", "p3-buchhaltung=http://127.0.0.1:8031/mcp,cloud=http://boese/mcp")
    d = pt.aus_umgebung(reserved={"cloud"})
    assert list(d) == ["p3-buchhaltung"] and d["p3-buchhaltung"].basis == "http://127.0.0.1:8031/mcp"
