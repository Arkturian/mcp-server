"""aufruf_protokoll.py - Wer ruft welches Werkzeug ueber welches Recht? (#2806)

Anlass 13.11.2026: SWFME trug das Sonderrecht ``mcp:*:*``. Die Frage, ob es das
ueberhaupt nutzt, liess sich nicht beantworten: das Gateway protokollierte nur
Ablehnungen, ein erlaubter ``tools/call`` hinterliess keinen Aufrufer, und das
Journal reicht wegen der Rotation nur Stunden zurueck.

Hier wird je erlaubtem Aufruf gezaehlt: Tag (UTC), Agent, Rolle, Server,
Werkzeug und das Recht, das gegriffen hat. Gegriffen hat das GENAUESTE Recht, das
den Aufruf erlaubt (``mcp:storage:assets_list`` vor ``mcp:storage:*`` vor
``mcp:*:*`` vor ``*``). Steht dort ein Platzhalter, war er noetig -- genau die
Frage "nutzt X sein mcp:*:*". Das JWT fuehrt Rollen- und Sonderrechte in einer
Liste; ob ein Recht ein Sonderrecht ist, sagt der Abgleich mit der Rolle.

Nie gespeichert: Argumente, Inhalte, Tokens. Die Zaehler liegen im Speicher und
werden alle ``SCHREIB_ABSTAND_S`` in SQLite geschrieben (nicht im Anfragepfad);
Zeilen aelter als ``MCP_AUFRUF_TAGE`` (Vorgabe 14) fallen beim Schreiben weg.
Ins Journal kommt je Tag und (Agent, Server, Werkzeug, Recht) eine Zeile, beim
ersten Aufruf -- das Journal bleibt knapp, die Zahlen stehen in der Datei.

Abfrage auf dem Gateway-Rechner::

    python3 aufruf_protokoll.py --agent SWFME
    python3 aufruf_protokoll.py --recht 'mcp:*:*' --tage 14
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sqlite3
import threading
import time
from contextlib import closing
from datetime import datetime, timezone
from typing import Callable, Optional

logger = logging.getLogger("mcp-aufrufe")

PFAD = os.getenv("MCP_AUFRUF_DB", "/var/lib/mcp-server/aufrufe.db")
TAGE = int(os.getenv("MCP_AUFRUF_TAGE", "14") or 14)
SCHREIB_ABSTAND_S = float(os.getenv("MCP_AUFRUF_SCHREIB_S", "30") or 30)
_LAENGE = 160                     # Namen werden gekappt, nie ganze Rohwerte


def genauigkeit(recht: str) -> tuple:
    """Fuehrende Teile ohne Platzhalter (``*``=0, ``mcp:*:*``=1, ``mcp:s:*``=2,
    ``mcp:s:t``=3), bei Gleichstand das laengere Recht."""
    n = 0
    for teil in (recht or "").split(":"):
        if teil == "*":
            break
        n += 1
    return (n, len(recht or ""))


def ist_platzhalter(recht: str) -> bool:
    return "*" in (recht or "")


def _tag(t: float) -> str:
    return datetime.fromtimestamp(t, tz=timezone.utc).strftime("%Y-%m-%d")


class Protokoll:
    def __init__(self, pfad: str = PFAD, tage: int = TAGE, jetzt: Callable[[], float] = time.time):
        self.pfad = pfad
        self.tage = max(1, int(tage))
        self._jetzt = jetzt
        self._lock = threading.Lock()
        # (tag, agent, rolle, server, werkzeug, recht) -> [anzahl, erst, zuletzt]
        self._offen: dict[tuple, list] = {}
        self._heute_gemeldet: set[tuple] = set()
        self._gemeldet_tag = ""

    # -- im Anfragepfad: nur Speicher ---------------------------------------
    def vermerken(self, agent: str, rolle: str, server: str, werkzeug: str, recht: str) -> None:
        t = self._jetzt()
        tag = _tag(t)
        k = (tag, (agent or "?")[:_LAENGE], (rolle or "")[:_LAENGE], (server or "")[:_LAENGE],
             (werkzeug or "")[:_LAENGE], (recht or "?")[:_LAENGE])
        with self._lock:
            e = self._offen.get(k)
            if e is None:
                self._offen[k] = [1, t, t]
            else:
                e[0] += 1
                e[2] = t
            if self._gemeldet_tag != tag:
                self._gemeldet_tag, self._heute_gemeldet = tag, set()
            neu = k[1:] not in self._heute_gemeldet
            if neu:
                self._heute_gemeldet.add(k[1:])
        if neu:
            logger.info("tools/call erlaubt: agent=%s rolle=%s mcp:%s tool=%s recht=%s%s",
                        k[1], k[2], k[3], k[4], k[5], " (Platzhalter)" if ist_platzhalter(k[5]) else "")

    # -- Ablage ----------------------------------------------------------------
    def _db(self) -> sqlite3.Connection:
        d = os.path.dirname(self.pfad)
        if d:
            os.makedirs(d, exist_ok=True)
        c = sqlite3.connect(self.pfad, timeout=5.0)
        c.execute("""CREATE TABLE IF NOT EXISTS aufruf (
                         tag TEXT NOT NULL, agent TEXT NOT NULL, rolle TEXT NOT NULL, server TEXT NOT NULL,
                         werkzeug TEXT NOT NULL, recht TEXT NOT NULL, anzahl INTEGER NOT NULL,
                         erst REAL NOT NULL, zuletzt REAL NOT NULL,
                         PRIMARY KEY (tag, agent, rolle, server, werkzeug, recht))""")
        return c

    def schreiben(self) -> int:
        """Zaehler in die Datei; alte Tage loeschen. Scheitert das Schreiben,
        bleiben die Zaehler im Speicher fuer den naechsten Versuch."""
        with self._lock:
            offen, self._offen = self._offen, {}
        if not offen:
            self._aufraeumen()
            return 0
        try:
            with closing(self._db()) as c, c:
                c.executemany("""INSERT INTO aufruf (tag, agent, rolle, server, werkzeug, recht, anzahl, erst, zuletzt)
                                 VALUES (?,?,?,?,?,?,?,?,?)
                                 ON CONFLICT (tag, agent, rolle, server, werkzeug, recht) DO UPDATE SET
                                   anzahl = anzahl + excluded.anzahl,
                                   erst = min(erst, excluded.erst), zuletzt = max(zuletzt, excluded.zuletzt)""",
                              [(*k, *v) for k, v in offen.items()])
                c.execute("DELETE FROM aufruf WHERE tag < ?", (self._grenze(),))
        except Exception:
            logger.exception("aufruf-protokoll: schreiben gescheitert (%d Zeilen bleiben)", len(offen))
            with self._lock:
                for k, v in offen.items():
                    e = self._offen.get(k)
                    if e is None:
                        self._offen[k] = v
                    else:
                        e[0] += v[0]
                        e[1], e[2] = min(e[1], v[1]), max(e[2], v[2])
            return 0
        return len(offen)

    def _grenze(self) -> str:
        return _tag(self._jetzt() - self.tage * 86400)

    def _aufraeumen(self) -> None:
        if not os.path.exists(self.pfad):
            return
        try:
            with closing(self._db()) as c, c:
                c.execute("DELETE FROM aufruf WHERE tag < ?", (self._grenze(),))
        except Exception:
            logger.exception("aufruf-protokoll: aufraeumen gescheitert")

    async def schreib_schleife(self) -> None:
        while True:
            await asyncio.sleep(SCHREIB_ABSTAND_S)
            try:
                await asyncio.to_thread(self.schreiben)
            except Exception:
                logger.exception("aufruf-protokoll: Schleife")

    # -- Abfrage ---------------------------------------------------------------
    def abfrage(self, agent: Optional[str] = None, recht: Optional[str] = None,
                server: Optional[str] = None, tage: Optional[int] = None) -> list[dict]:
        """Summen je (Agent, Rolle, Server, Werkzeug, Recht) ueber die letzten Tage."""
        self.schreiben()
        if not os.path.exists(self.pfad):
            return []
        bis = _tag(self._jetzt() - max(1, int(tage or self.tage)) * 86400)
        sql = ["SELECT agent, rolle, server, werkzeug, recht, sum(anzahl), min(erst), max(zuletzt), "
               "min(tag), max(tag) FROM aufruf WHERE tag >= ?"]
        werte: list = [bis]
        for spalte, wert in (("agent", agent), ("recht", recht), ("server", server)):
            if wert:
                sql.append(f"AND {spalte} = ?")
                werte.append(wert)
        sql.append("GROUP BY agent, rolle, server, werkzeug, recht ORDER BY sum(anzahl) DESC")
        with closing(self._db()) as c:
            zeilen = c.execute(" ".join(sql), werte).fetchall()
        return [{"agent": z[0], "rolle": z[1], "server": z[2], "werkzeug": z[3], "recht": z[4],
                 "anzahl": z[5], "erst": z[6], "zuletzt": z[7], "von_tag": z[8], "bis_tag": z[9]}
                for z in zeilen]


_instanz: Optional[Protokoll] = None


def protokoll() -> Protokoll:
    global _instanz
    if _instanz is None:
        _instanz = Protokoll()
    return _instanz


def _main() -> None:
    p = argparse.ArgumentParser(description="Erlaubte MCP-Werkzeugaufrufe je Agent und Recht (#2806)")
    p.add_argument("--agent")
    p.add_argument("--recht")
    p.add_argument("--server")
    p.add_argument("--tage", type=int, default=TAGE)
    p.add_argument("--pfad", default=PFAD)
    a = p.parse_args()
    print(json.dumps(Protokoll(a.pfad).abfrage(a.agent, a.recht, a.server, a.tage), indent=1, ensure_ascii=False))


if __name__ == "__main__":
    _main()
