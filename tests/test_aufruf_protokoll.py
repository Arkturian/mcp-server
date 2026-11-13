"""#2806: Das Gateway zaehlt erlaubte tools/call je Agent, Rolle, Server, Werkzeug
und gegriffenem Recht -- ohne Argumente, mit Aufbewahrung ueber Tage.

Anlass 13.11.2026: Ob SWFME sein Sonderrecht mcp:*:* nutzt, war nicht
beantwortbar; erlaubte Aufrufe hinterliessen keine Spur, das Journal rotiert.
"""
import asyncio
import json
import logging
import sqlite3

import httpx
import pytest
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route

import auth
import aufruf_protokoll as ap

GEHEIM = "streng-geheimes-argument-4711"


def _call(*namen, args=None):
    teile = [{"jsonrpc": "2.0", "id": i, "method": "tools/call",
              "params": {"name": n, "arguments": args or {"text": GEHEIM}}} for i, n in enumerate(namen)]
    return json.dumps(teile[0] if len(teile) == 1 else teile)


def _app():
    async def ok(request):
        return JSONResponse({"ok": 1})
    app = Starlette(routes=[Route("/{srv}/", ok, methods=["POST"])])
    app.add_middleware(auth.JWTAuthMiddleware)
    return app


def _post(path, body):
    async def lauf():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=_app()), base_url="http://t") as c:
            return await c.post(path, content=body, headers={"authorization": "Bearer x",
                                                              "content-type": "application/json"})
    return asyncio.run(lauf())


@pytest.fixture
def agent(monkeypatch):
    """JWT des Agenten SWFME mit wechselbaren Rechten."""
    rechte = {"p": ["mcp:*:*"]}
    monkeypatch.setattr(auth, "OFFENE_SERVER", set())
    monkeypatch.setattr(auth, "EXPLICIT_ONLY_SERVERS", set())
    monkeypatch.setattr(auth, "WERKZEUG_STUFEN", {})

    async def key():
        return "k"
    monkeypatch.setattr(auth, "_get_public_key", key)
    monkeypatch.setattr(auth, "_decode_jwt", lambda token, key: {
        "type": "agent", "agent_name": "SWFME", "role": "service", "permissions": rechte["p"]})
    return rechte


def _zeilen():
    return ap.protokoll().abfrage()


def test_platzhalter_aufruf_ergibt_genau_einen_eintrag_ohne_argumente(agent, caplog):
    caplog.set_level(logging.INFO, logger="mcp-aufrufe")
    assert _post("/storage/", _call("assets_list")).status_code == 200
    z = _zeilen()
    assert [(e["agent"], e["rolle"], e["server"], e["werkzeug"], e["recht"], e["anzahl"]) for e in z] == \
        [("SWFME", "service", "storage", "assets_list", "mcp:*:*", 1)]
    roh = open(ap.protokoll().pfad, "rb").read()
    assert GEHEIM.encode() not in roh and GEHEIM not in caplog.text, "nie Argumente"
    assert "recht=mcp:*:* (Platzhalter)" in caplog.text


def test_genauestes_recht_greift(agent):
    agent["p"] = ["*", "mcp:*:*", "mcp:storage:*", "mcp:storage:assets_list"]
    _post("/storage/", _call("assets_list"))
    _post("/storage/", _call("assets_get"))
    _post("/content/", _call("posts_get"))
    r = {(e["server"], e["werkzeug"]): e["recht"] for e in _zeilen()}
    assert r == {("storage", "assets_list"): "mcp:storage:assets_list",
                 ("storage", "assets_get"): "mcp:storage:*",
                 ("content", "posts_get"): "mcp:*:*"}


def test_stufe_und_offener_server(agent, monkeypatch):
    agent["p"] = ["mcp:person:read"]
    monkeypatch.setattr(auth, "WERKZEUG_STUFEN", {"person": {"persons_list": "read"}})
    monkeypatch.setattr(auth, "OFFENE_SERVER", {"llmtime"})
    assert _post("/person/", _call("persons_list")).status_code == 200
    assert _post("/llmtime/", _call("forecast_when")).status_code == 200
    r = {e["server"]: e["recht"] for e in _zeilen()}
    assert r == {"person": "mcp:person:read", "llmtime": "offen"}


def test_abgelehnt_wird_nicht_gezaehlt(agent):
    agent["p"] = ["mcp:issue:*"]
    assert _post("/storage/", _call("assets_list")).status_code == 403
    assert _zeilen() == []


def test_stapel_zaehlt_jeden_aufruf(agent):
    _post("/storage/", _call("assets_list", "assets_get"))
    assert sorted(e["werkzeug"] for e in _zeilen()) == ["assets_get", "assets_list"]


def test_zaehlt_und_meldet_je_tag_einmal(agent, caplog):
    caplog.set_level(logging.INFO, logger="mcp-aufrufe")
    for _ in range(3):
        _post("/storage/", _call("assets_list"))
    assert _zeilen()[0]["anzahl"] == 3
    assert caplog.text.count("tools/call erlaubt:") == 1, "Journal knapp: eine Zeile je Tag und Schluessel"


# -- Ablage ohne Gateway --------------------------------------------------------

def test_ablage_summiert_ueber_schreibvorgaenge_und_loescht_alte_tage(tmp_path):
    uhr = [1_800_000_000.0]
    p = ap.Protokoll(str(tmp_path / "a.db"), tage=14, jetzt=lambda: uhr[0])
    p.vermerken("SWFME", "service", "storage", "assets_list", "mcp:*:*")
    assert p.schreiben() == 1
    p.vermerken("SWFME", "service", "storage", "assets_list", "mcp:*:*")
    p.schreiben()
    assert p.abfrage(agent="SWFME")[0]["anzahl"] == 2
    uhr[0] += 10 * 86400
    p.vermerken("Cloud", "architect", "cloud", "send_message", "mcp:*:*")
    assert {e["agent"] for e in p.abfrage(tage=14)} == {"SWFME", "Cloud"}, "zehn Tage spaeter noch da"
    uhr[0] += 6 * 86400
    p.vermerken("Login", "admin", "cloud", "health", "mcp:cloud:*")   # Schreiben MIT offenen Zaehlern
    p.schreiben()
    with sqlite3.connect(p.pfad) as c:                                 # die Datei selbst, nicht abfrage()
        agenten = {z[0] for z in c.execute("SELECT agent FROM aufruf")}
    assert agenten == {"Cloud", "Login"}, "aelter als 14 Tage geloescht"
    assert p.abfrage(recht="mcp:storage:*") == []


def test_schreiben_gescheitert_behaelt_die_zaehler(tmp_path):
    p = ap.Protokoll(str(tmp_path / "gibtsnicht" / "\0kaputt.db"))
    p.vermerken("A", "r", "s", "w", "mcp:*:*")
    assert p.schreiben() == 0
    p.pfad = str(tmp_path / "gut.db")
    assert p.schreiben() == 1 and p.abfrage()[0]["anzahl"] == 1


def test_genauigkeit():
    assert sorted(["*", "mcp:*:*", "mcp:s:*", "mcp:s:t", "mcp:*"], key=ap.genauigkeit) == \
        ["*", "mcp:*", "mcp:*:*", "mcp:s:*", "mcp:s:t"]
    assert ap.ist_platzhalter("mcp:*:*") and not ap.ist_platzhalter("mcp:s:t")


def test_kein_echter_pfad_in_tests():
    assert "/var/lib" not in ap.protokoll().pfad


def test_server_startet_den_schreiber_und_schreibt_beim_beenden(monkeypatch):
    import os
    os.environ.setdefault("MCP_SERVERS", "cloud")
    import server
    gestartet, geschrieben = [], []

    class _P:
        async def schreib_schleife(self):
            gestartet.append(1)

        def schreiben(self):
            geschrieben.append(1)
            return 0
    monkeypatch.setattr(ap, "_instanz", _P())

    async def still():
        return None
    monkeypatch.setattr(auth, "revoked_poll_loop", still)
    monkeypatch.setattr(server, "ENABLED_MCPS", set())
    monkeypatch.setattr(server, "_started_mcps", [])

    async def lauf():
        await server.startup()
        await asyncio.sleep(0)
        await server.shutdown()
    asyncio.run(lauf())
    assert gestartet == [1] and geschrieben == [1]
