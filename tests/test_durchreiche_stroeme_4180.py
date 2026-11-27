"""#4180 (config 27.11.): GET-Stroeme belegten den ganzen Verbindungsvorrat eines Ziels der Durchreiche.

Seit 11:15Z antwortete die Durchreiche `config` auf jede Anfrage mit PoolTimeout: 100 MCP-Sitzungen hielten je
einen GET-Strom offen, der eine Client je Ziel hat httpx' Vorgabe max_connections=100, ohne Lese-Timeout. Jetzt
haben GET-Stroeme einen eigenen Vorrat und eine Grenze (MCP_PASSTHROUGH_STROEME, Vorgabe 32); darueber kommt
sofort 405 (fuer den MCP-Client: kein Strom vom Server). POST scheitert nie an offenen Stroemen.

Gemessen gegen einen ECHTEN Zieldienst auf 127.0.0.1 (asyncio-Server, HTTP/1.1): nur ein echter Transport hat
einen Verbindungsvorrat, MockTransport und ASGITransport nicht. Die Durchreiche wird direkt als ASGI-App gerufen,
damit die GET-Stroeme wirklich offen bleiben.
"""
from __future__ import annotations

import asyncio
import json
import logging

import pytest

import passthrough as pt


async def _zieldienst():
    """Haelt jeden GET als SSE-Strom offen, bis `halten` gesetzt wird; beantwortet POST sofort (keep-alive)."""
    halten = asyncio.Event()
    zaehler = {"get": 0, "post": 0}

    async def bedienen(reader, writer):
        try:
            while True:
                kopf = await reader.readuntil(b"\r\n\r\n")
                zeilen = kopf.decode("latin-1").split("\r\n")
                methode = zeilen[0].split(" ")[0]
                laenge = next((int(z.split(":", 1)[1]) for z in zeilen[1:]
                               if z.lower().startswith("content-length:")), 0)
                if laenge:
                    await reader.readexactly(laenge)
                if methode == "GET":
                    zaehler["get"] += 1
                    writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
                                 b"Transfer-Encoding: chunked\r\n\r\n8\r\n: ping\n\n\r\n")
                    await writer.drain()
                    await halten.wait()
                    return
                zaehler["post"] += 1
                rumpf = b'{"jsonrpc":"2.0","id":1,"result":{"ok":true}}'
                writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: "
                             + str(len(rumpf)).encode() + b"\r\n\r\n" + rumpf)
                await writer.drain()
        except (asyncio.IncompleteReadError, ConnectionError):
            pass
        finally:
            writer.close()

    server = await asyncio.start_server(bedienen, "127.0.0.1", 0)
    return server, server.sockets[0].getsockname()[1], halten, zaehler


def _aufruf(d, methode, rumpf=b""):
    """Ein Aufruf an die Durchreiche als ASGI; Antwortbeginn und Rumpf landen in `aus`. `trennen` beendet ihn."""
    aus = {"start": None, "body": b""}
    begonnen, trennen = asyncio.Event(), asyncio.Event()
    gelesen = {"n": 0}

    async def receive():
        if gelesen["n"] == 0:
            gelesen["n"] = 1
            return {"type": "http.request", "body": rumpf, "more_body": False}
        await trennen.wait()
        return {"type": "http.disconnect"}

    async def send(msg):
        if msg["type"] == "http.response.start":
            aus["start"] = msg
            begonnen.set()
        elif msg["type"] == "http.response.body":
            aus["body"] += msg.get("body", b"")

    scope = {"type": "http", "method": methode, "path": "/", "root_path": "", "query_string": b"",
             "headers": [(b"content-type", b"application/json"), (b"accept", b"text/event-stream"),
                         (b"authorization", b"Bearer agent-jwt")],
             "http_version": "1.1", "scheme": "http", "server": ("gw", 80), "client": ("127.0.0.1", 9)}
    task = asyncio.create_task(d(scope, receive, send))
    return task, aus, begonnen, trennen


async def _hundert_stroeme_dann_post(stroeme, anzahl=100, post_frist=15.0):
    server, port, halten, zaehler = await _zieldienst()
    d = pt.Durchreiche("config", f"http://127.0.0.1:{port}/mcp") if stroeme is None else \
        pt.Durchreiche("config", f"http://127.0.0.1:{port}/mcp", stroeme=stroeme)
    gets = [_aufruf(d, "GET") for _ in range(anzahl)]
    try:
        await asyncio.wait_for(asyncio.gather(*(b.wait() for _, _, b, _ in gets)), 20)
        status = [a["start"]["status"] for _, a, _, _ in gets]
        offen = d.offene_stroeme
        t, a, _, _ = _aufruf(d, "POST", json.dumps({"jsonrpc": "2.0", "id": 1, "method": "ping"}).encode())
        await asyncio.wait_for(t, post_frist)
        return status, offen, a, zaehler
    finally:
        for task, _, _, trennen in gets:
            trennen.set()
        halten.set()
        await asyncio.wait([g[0] for g in gets], timeout=10)
        await d.schliessen()
        server.close()
        await server.wait_closed()


def test_hundert_offene_stroeme_ein_post_geht_trotzdem_durch():
    status, offen, post, zaehler = asyncio.run(_hundert_stroeme_dann_post(None))
    assert status.count(200) == pt.STROEME_VORGABE == offen == zaehler["get"]
    assert status.count(405) == 100 - pt.STROEME_VORGABE
    assert post["start"]["status"] == 200 and json.loads(post["body"])["result"] == {"ok": True}


def test_grenze_aus_der_umgebung(monkeypatch):
    monkeypatch.setenv("MCP_PASSTHROUGH_STROEME", "5")
    status, offen, post, _ = asyncio.run(_hundert_stroeme_dann_post(None, anzahl=8))
    assert status.count(200) == 5 == offen and status.count(405) == 3 and post["start"]["status"] == 200


def test_ohne_stroeme_nur_post():
    status, offen, post, zaehler = asyncio.run(_hundert_stroeme_dann_post(0, anzahl=3))
    assert status == [405] * 3 and offen == 0 and zaehler["get"] == 0 and post["start"]["status"] == 200


def test_geschlossener_strom_gibt_den_platz_frei():
    async def lauf():
        server, port, halten, _ = await _zieldienst()
        d = pt.Durchreiche("config", f"http://127.0.0.1:{port}/mcp", stroeme=1)
        try:
            t1, a1, b1, x1 = _aufruf(d, "GET")
            await asyncio.wait_for(b1.wait(), 10)
            t2, a2, b2, _ = _aufruf(d, "GET")
            await asyncio.wait_for(b2.wait(), 10)
            x1.set()                                              # der Agent geht, sein Strom endet
            await asyncio.wait_for(t1, 10)
            belegt_nach = d.offene_stroeme
            t3, a3, b3, x3 = _aufruf(d, "GET")
            await asyncio.wait_for(b3.wait(), 10)
            x3.set()
            await asyncio.wait_for(t3, 10)
            return a1["start"]["status"], a2["start"]["status"], belegt_nach, a3["start"]["status"]
        finally:
            halten.set()
            await d.schliessen()
            server.close()
            await server.wait_closed()
    assert asyncio.run(lauf()) == (200, 405, 0, 200)


def test_toter_zieldienst_gibt_den_platz_frei():
    async def lauf():
        d = pt.Durchreiche("config", "http://127.0.0.1:9/mcp", stroeme=1)   # Port 9: niemand hoert
        try:
            t, a, _, _ = _aufruf(d, "GET")
            await asyncio.wait_for(t, 10)
            return a["start"]["status"], d.offene_stroeme
        finally:
            await d.schliessen()
    assert asyncio.run(lauf()) == (502, 0)


def test_grenze_wird_einmal_gemeldet(caplog):
    caplog.set_level(logging.WARNING, logger="mcp-passthrough")
    asyncio.run(_hundert_stroeme_dann_post(2, anzahl=6))
    zeilen = [r for r in caplog.records if "Grenze fuer GET-Stroeme" in r.getMessage()]
    assert len(zeilen) == 1 and "config" in zeilen[0].getMessage()


@pytest.mark.parametrize("roh,erwartet", [(None, 32), ("", 32), ("7", 7), (" 0 ", 0), ("-3", 0), ("viele", 32)])
def test_stroeme_grenze(monkeypatch, roh, erwartet):
    if roh is None:
        monkeypatch.delenv("MCP_PASSTHROUGH_STROEME", raising=False)
    else:
        monkeypatch.setenv("MCP_PASSTHROUGH_STROEME", roh)
    assert pt.stroeme_grenze() == erwartet


def test_auch_mit_grenze_hundert_bleibt_post_frei():
    """Die Grenze allein schuetzt nicht, wenn sie den Vorrat der Aufrufe erreicht: GET-Stroeme haben deshalb
    einen eigenen Vorrat. Mit 100 erlaubten Stroemen (= httpx-Vorgabe je Client) geht der POST trotzdem durch."""
    status, offen, post, _ = asyncio.run(_hundert_stroeme_dann_post(100, anzahl=100, post_frist=8.0))
    assert status.count(200) == 100 == offen and post["start"]["status"] == 200
