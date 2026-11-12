"""Zeit-Werkzeuge: der Tag einer Person in Summen (cloud-api GET /api/usage/tag).

Vertrag: Post #5880 (Uebersicht „Zeit heute“), cloud-api ad399af (Lesen durch
benannte Agenten, Alex 11.11.2026). Wer lesen darf, entscheidet cloud-api: die
Person selbst, ein Superadmin und die Agenten in ZEIT_LESE_AGENTEN. Dieses
Werkzeug reicht nur den Aufruf durch und verdichtet die Antwort.

Warum verdichten: Die Tagesantwort ist roh rund 900 KB (1882 Zuege am 11.11. bis 22:17).
Ein Agent braucht Summen, nicht jeden Zug. Einzelne Zuege nur auf Wunsch und
hoechstens MAX_ZUEGE, Agenten einzeln hoechstens MAX_AGENTEN. Ziel: unter ZIEL_BYTES,
belegt in tests/test_zeit_tools.py gegen einen erzeugten Tag in der Form eines
echten (keine echten Daten im Repo, Cloud 11.11.2026).

Registrierung getrennt von server.py, damit das Werkzeug gegen einen falschen
Transport geprueft werden kann.
"""

import json
from datetime import datetime
from typing import Any, Awaitable, Callable, Dict, List, Optional
from zoneinfo import ZoneInfo

Transport = Callable[..., Awaitable[Any]]

MAX_ZUEGE = 50
MAX_FREMDE_AGENTEN = 15     # je andere Foederation in `quellen` (Post #5898)
MAX_AGENTEN = 25            # danach eine Zeile „+N weitere“ mit Summen (Cloud 11.11.2026)
TOP_ORDNER = 10
ZIEL_BYTES = 20_000
TOKEN_FELDER = ("anfragen", "input", "output", "cache_read", "cache_write", "reasoning")
ZEILEN_FELDER = ("hinzu", "geaendert", "entfernt")


def _uhr(iso: Optional[str], tz: ZoneInfo, sekunden: bool = False) -> Optional[str]:
    """ISO-Zeit als Uhrzeit des Tages in der Zeitzone der Antwort."""
    if not iso:
        return None
    try:
        t = datetime.fromisoformat(str(iso).replace("Z", "+00:00"))
    except ValueError:
        return str(iso)
    return t.astimezone(tz).strftime("%H:%M:%S" if sekunden else "%H:%M")


def _zeilen(z: Dict[str, Any]) -> int:
    q = z.get("zeilen") or {}
    return sum(int(q.get(k) or 0) for k in ZEILEN_FELDER)


def _von_person(z: Dict[str, Any], person: str) -> bool:
    return z.get("ausloeser_art") == "mensch" and (z.get("ausloeser") or "") == person


def _agenten(zuege: List[Dict[str, Any]], person: str) -> List[Dict[str, Any]]:
    je: Dict[str, Dict[str, Any]] = {}
    for z in zuege:
        a = je.setdefault(z.get("agent") or "?", {
            "agent": z.get("agent") or "?", "zuege": 0, "von_person": 0, "dauer_s": 0.0,
            "zeilen": 0, "tokens": {k: 0 for k in TOKEN_FELDER}})
        a["zuege"] += 1
        a["von_person"] += 1 if _von_person(z, person) else 0
        a["dauer_s"] += float(z.get("dauer_s") or 0)
        a["zeilen"] += _zeilen(z)
        for k, v in (z.get("tokens") or {}).items():
            if k in a["tokens"]:
                a["tokens"][k] += int(v or 0)
    alle = sorted(je.values(), key=lambda a: (-a["dauer_s"], a["agent"]))
    out, rest = alle[:MAX_AGENTEN], alle[MAX_AGENTEN:]
    if rest:
        # Die Grenze haengt sonst an der Zahl der Agenten des Tages, nicht an der Bedeutung.
        weitere = {"agent": f"+{len(rest)} weitere", "weitere": len(rest), "zuege": 0, "von_person": 0,
                   "dauer_s": 0.0, "zeilen": 0, "tokens": {k: 0 for k in TOKEN_FELDER}}
        for a in rest:
            for k in ("zuege", "von_person", "dauer_s", "zeilen"):
                weitere[k] += a[k]
            for k, v in a["tokens"].items():
                weitere["tokens"][k] += v
        out.append(weitere)
    for a in out:
        a["dauer_s"] = round(a["dauer_s"])
        a["tokens"] = {k: v for k, v in a["tokens"].items() if v}        # Nullen tragen nichts bei
    return out


def _zug_kurz(z: Dict[str, Any], tz: ZoneInfo) -> Dict[str, Any]:
    """Ein Zug in ~150 Bytes: leere Felder fallen weg (50 Zuege bleiben so unter 8 KB)."""
    t = z.get("tokens") or {}
    voll = {"agent": z.get("agent"), "beginn": _uhr(z.get("beginn"), tz, sekunden=True),
            "dauer_s": round(float(z.get("dauer_s") or 0)),
            "ausloeser_art": z.get("ausloeser_art"), "ausloeser": z.get("ausloeser"),
            "output": int(t.get("output") or 0), "zeilen": _zeilen(z)}
    return {k: v for k, v in voll.items() if v not in (None, 0, "")}


def _quelle_kurz(q: Dict[str, Any]) -> Dict[str, Any]:
    """Eine Foederation in `quellen`: Status, Grund, Umfang und die laengsten Agenten
    als Summen (ohne Personen). 38 Agenten auf P3 waeren sonst rund 8 KB."""
    out = {k: q.get(k) for k in ("foederation", "status", "grund", "umfang", "verweigert", "bloecke", "zuege")
           if q.get(k) not in (None, [], {})}
    summen = q.get("agenten_summen")
    if isinstance(summen, list):
        kurz = []
        for a in sorted(summen, key=lambda a: -float(a.get("dauer_s") or 0))[:MAX_FREMDE_AGENTEN]:
            zl = a.get("zeilen") or {}
            kurz.append({"agent": a.get("agent"), "zuege": a.get("zuege"), "dauer_s": a.get("dauer_s"),
                         "zeilen": sum(int(zl.get(k) or 0) for k in ZEILEN_FELDER),
                         "output": int((a.get("tokens") or {}).get("output") or 0)})
        out["agenten_summen"] = kurz
        out["agenten_summen_gesamt"] = len(summen)
    return out


def tag_zusammenfassen(d: Dict[str, Any], zuege: bool = False) -> Dict[str, Any]:
    """Die Tagesantwort von cloud-api in Summen. Rein, ohne Netz."""
    tz = ZoneInfo(d.get("zeitzone") or "Europe/Vienna")
    person = (d.get("person") or "").lower()
    mensch = d.get("mensch") or {}
    ver = mensch.get("vereinigt") or {}
    alle = d.get("agenten") or []
    ordner = d.get("ordner") or []
    out: Dict[str, Any] = {
        "person": d.get("person"), "datum": d.get("datum"), "zeitzone": d.get("zeitzone"),
        "hinweis": d.get("hinweis"),
        "summen": d.get("summen"),
        "mensch": {
            "aktiv_s": ver.get("aktiv_s", 0), "passiv_s": ver.get("passiv_s", 0),
            "bloecke": len(mensch.get("bloecke") or []),
            # [von, bis, zustand] in Ortszeit: ein Tag mit 100 Abschnitten bleibt so bei ~3 KB.
            "abschnitte": [[_uhr(a.get("beginn"), tz), _uhr(a.get("ende"), tz), a.get("zustand")]
                           for a in ver.get("abschnitte") or []],
        },
        "agenten": _agenten(alle, person),
        "ordner": [{k: o.get(k) for k in ("ordner", "zuege", "agenten", "dateien", *ZEILEN_FELDER)}
                   for o in ordner[:TOP_ORDNER]],
        "ordner_gesamt": len(ordner),
    }
    if d.get("quellen") is not None:
        out["quellen"] = [_quelle_kurz(q) for q in d.get("quellen") or [] if isinstance(q, dict)]
    if zuege:
        laengste = sorted(alle, key=lambda z: -float(z.get("dauer_s") or 0))[:MAX_ZUEGE]
        laengste.sort(key=lambda z: str(z.get("beginn") or ""))
        out["zuege"] = [_zug_kurz(z, tz) for z in laengste]
        out["zuege_hinweis"] = (f"{len(laengste)} von {len(alle)} Zuegen, die laengsten, nach Beginn sortiert"
                                if len(alle) > MAX_ZUEGE else f"alle {len(alle)} Zuege")
    return out


def register_zeit_tools(mcp, transport: Transport) -> None:
    """``transport(method, path, params=?, timeout=?)`` ruft cloud-api mit dem JWT des Aufrufers."""

    @mcp.tool(
        name="zeit_tag",
        description=(
            "Ein Tag einer Person in Summen (Post #5880, Uebersicht „Zeit heute“). "
            "Lesen duerfen die Person selbst, Superadmins und die von Alex benannten Agenten "
            "(alle, die am 11.11.2026 mit „Cloud“ anfingen); sonst 403. Agenten MUESSEN "
            "`person` (Mail) nennen. `datum` = Kalendertag in Wien, JJJJ-MM-TT, leer = heute. "
            "Liefert `summen`, die Bahn `mensch` (aktive/passive Bildschirmzeit, Abschnitte "
            "[von, bis, zustand] in Ortszeit), `agenten` je Name (zuege, von_person = von dieser "
            "Person ausgeloest, dauer_s, zeilen, tokens), die 25 laengsten einzeln, der Rest als "
            "eine Zeile „+N weitere“ mit Summen, und die Top-10-`ordner` nach Umfang. "
            "Einzelne Zuege nur mit zuege=true, hoechstens 50 (die laengsten), mit Gesamtzahl. "
            "Andere Foederationen (z.B. P3) fragt cloud-api ueber die Bruecke mit ab (Post #5898): "
            "`foederationen` leer = alle freigeschalteten, `keine` = nur diese, sonst Komma-Liste. "
            "`quellen` nennt je Foederation den Status (ok, keine_antwort, verweigert, fehler, "
            "nicht_abgefragt) mit Grund; fehlt eine, steht es dort. Zuege von drueben tragen den "
            "Knoten `<bruecke>-<knoten>`. "
            "Anwesenheit ist ein Personendatum: nicht in Beitraege, Issues oder an Dritte kopieren, "
            "nur dem Menschen selbst oder seinem Admin zeigen."
        ),
    )
    async def zeit_tag(datum: str = "", person: str = "", zuege: bool = False, foederationen: str = "") -> Any:
        params = {k: v for k, v in (("datum", datum.strip()), ("person", person.strip()),
                                    ("foederationen", foederationen.strip())) if v}
        d = await transport("GET", "/api/usage/tag", params=params, timeout=90)
        if not isinstance(d, dict) or "mensch" not in d:
            return d
        # Kompakt als Text: Ein dict gibt FastMCP eingerueckt aus, das machte am 11.11.
        # aus 15,7 KB live 27,5 KB (gemessen am Gateway mit zuege=true).
        return json.dumps(tag_zusammenfassen(d, zuege=zuege), ensure_ascii=False, separators=(",", ":"))
