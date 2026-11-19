"""Zeit-Werkzeuge: der Tag einer Person in Summen (cloud-api GET /api/usage/tag), dazu die
Aufgaben des Mini-Hirns lesen und umordnen (GET /api/aufgaben/tag, POST /api/aufgaben/umordnen,
Alex ueber CloudV2-time 13.11.2026), und eine Aufgabe nach ID (GET /api/aufgaben/{id}/schritte, #3430).

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
MAX_AGENTEN = 25            # danach eine Zeile „+N weitere“ mit Summen (Cloud 11.11.2026)
TOP_ORDNER = 10
ZIEL_BYTES = 20_000
MAX_SCHRITTE = 150          # aufgaben_tag: so viele Schritte einzeln, der Rest nur in den Summen
MAX_AUFGABE_SCHRITTE = 30   # aufgabe: die juengsten so vielen Schritte, mit Texten unter ZIEL_BYTES
TOKEN_FELDER = ("anfragen", "input", "output", "cache_read", "cache_write", "reasoning")
ZEILEN_FELDER = ("hinzu", "geaendert", "entfernt")


def _zeitpunkt(iso: str) -> datetime:
    return datetime.fromisoformat(str(iso).replace("Z", "+00:00"))


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
    """Eine Foederation in `quellen`: Status, Grund, Umfang (alles/eigene), Zahl der
    Bloecke und Zuege von dort."""
    return {k: q.get(k) for k in ("foederation", "status", "grund", "umfang", "bloecke", "zuege")
            if q.get(k) not in (None, [], {})}


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
        "umfang": d.get("umfang"),
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


def entwurf_zusammenfassen(d: Dict[str, Any]) -> Dict[str, Any]:
    """Der Tagesentwurf von cloud-api kompakt: je Block eine Zeile. Rein, ohne Netz."""
    def _pfad(p: Optional[Dict[str, Any]]) -> Optional[str]:
        return " › ".join(p.get("pfad") or [p.get("name") or ""]) if p else None

    eintraege = []
    for e in d.get("eintraege") or []:
        z = e.get("zuordnung") or {}
        zeile = {"von": e.get("von_lokal"), "bis": e.get("bis_lokal"),
                 "min": int(round((e.get("dauer_s") or 0) / 60)),
                 "projekt": _pfad(e.get("projekt")), "art": z.get("art"),
                 "sicherheit": z.get("sicherheit"), "notiz": e.get("notiz") or None}
        if e.get("nebenbei"):
            zeile["nebenbei"] = [_pfad(n) for n in e["nebenbei"]]
        v = e.get("vorschlag")
        if v:
            zeile["vorschlag"] = {"projekt": _pfad(v.get("projekt")), "sicherheit": v.get("sicherheit")}
        eintraege.append({k: w for k, w in zeile.items() if w not in (None, [], "")})
    s = d.get("summen") or {}
    return {"person": d.get("person"), "datum": d.get("datum"), "umfang": d.get("umfang"),
            "summen": {**{k: s.get(k) for k in ("aktiv_s", "zugeordnet_s", "ohne_ordner_s", "offen_s", "passiv_s")},
                       "je_projekt": [{"projekt": _pfad(p), "art": p.get("art"),
                                       "min": int(round((p.get("sekunden") or 0) / 60))}
                                      for p in s.get("je_projekt") or []]},
            "eintraege": eintraege,
            "quellen": [_quelle_kurz(q) for q in d.get("quellen") or [] if isinstance(q, dict)],
            "jev": d.get("jev")}


def _arbeit_kurz(a: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Agentenarbeit einer Gruppe (direkt/kette): None bleibt None (keine Messung), sonst
    zuege, h, output und zeilen; Zeilen nur, wenn welche geaendert wurden."""
    if a is None:
        return None
    out = {"zuege": a.get("zuege"), "h": a.get("dauer_h"), "output": a.get("output")}
    z = a.get("zeilen") or {}
    if any(z.get(k) for k in ZEILEN_FELDER):
        out["zeilen"] = {k: z.get(k) or 0 for k in ZEILEN_FELDER}
    return out


def woche_zusammenfassen(d: Dict[str, Any]) -> Dict[str, Any]:
    """Der Wochenrueckblick von cloud-api (#3089) kompakt. Rein, ohne Netz.

    Projekte als Pfad, Gruppen ohne Zuege fallen weg. Ein Tag ohne Messung behaelt None:
    Das ist „keine Messung“, nicht 0."""
    projekte = []
    for p in d.get("projekte") or []:
        zeile: Dict[str, Any] = {"projekt": _pfad_text(p.get("projekt")), "art": (p.get("projekt") or {}).get("art")}
        if p.get("mensch_aktiv_h"):
            zeile["mensch_h"] = p["mensch_aktiv_h"]
        for gruppe in ("direkt", "kette"):
            a = p.get(gruppe) or {}
            if a.get("zuege"):
                zeile[gruppe] = _arbeit_kurz(a)
        # #2837: Mails gezaehlt, Termine mit Dauer, Gesamtzeit ohne Doppelzaehlung (nur wenn da)
        mails = {k: v for k, v in (p.get("mails") or {}).items() if v}
        if mails:
            zeile["mails"] = mails
        if (p.get("termine") or {}).get("anzahl"):
            zeile["termine"] = p["termine"]
        if (p.get("gesamt") or {}).get("gesamt_h"):
            zeile["gesamt"] = p["gesamt"]
        projekte.append(zeile)
    tage = []
    for t in d.get("tage") or []:
        m = t.get("mensch")
        tage.append({"datum": t.get("datum"), "wochentag": t.get("wochentag"), "messung": t.get("messung"),
                     "mensch": ({k: m.get(k) for k in ("aktiv_h", "passiv_h", "zugeordnet_h", "ohne_ordner_h",
                                                        "offen_h")} if m is not None else None),
                     "direkt": _arbeit_kurz(t.get("direkt")), "kette": _arbeit_kurz(t.get("kette")),
                     **{k: t.get(k) for k in ("mails", "termine", "gesamt") if k in t}})
    w = d.get("woche") or {}
    return {"person": d.get("person"), "von": d.get("von"), "bis": d.get("bis"), "umfang": d.get("umfang"),
            "einheit": d.get("einheit"),
            "woche": {"tage": w.get("tage"), "tage_gemessen": w.get("tage_gemessen"), "mensch": w.get("mensch"),
                      "direkt": _arbeit_kurz(w.get("direkt")), "kette": _arbeit_kurz(w.get("kette")),
                      **{k: w.get(k) for k in ("mails", "termine", "gesamt") if k in w}},
            "projekte": projekte, "tage": tage,
            "quellen": d.get("quellen"), "jev": d.get("jev"), "regeln": d.get("regeln")}


def _pfad_text(p: Optional[Dict[str, Any]]) -> Optional[str]:
    if not p:
        return None
    return " › ".join(p.get("pfad") or [p.get("name") or p.get("id") or ""])


def aufgaben_zusammenfassen(d: Dict[str, Any], texte: bool = False) -> Dict[str, Any]:
    """Die Schritte eines Tages (GET /api/aufgaben/tag) kompakt, mit Summen je Aufgabe und
    Projekt und dem Projektkatalog. `texte`: je Schritt auch Eingabe und Schlusssatz, soweit
    die cloud-api sie liefert (nur der Person selbst und den Agenten ihres Besitzers). Rein."""
    tz = ZoneInfo("Europe/Vienna")
    schritte = d.get("schritte") or []
    je_aufgabe: Dict[Any, Dict[str, Any]] = {}
    je_projekt: Dict[Any, Dict[str, Any]] = {}
    zeilen = []
    for s in schritte:
        # zeit_s ist gekappt (cloud-api #3173) und darf 0 sein; nur wenn es fehlt, gilt dauer_s.
        z = s.get("zeit_s")
        sek = float(z) if z is not None else float(s.get("dauer_s") or 0)
        projekt = _pfad_text(s.get("projekt"))
        tok = s.get("tokens") if isinstance(s.get("tokens"), dict) else None
        lauf = s.get("durchlauf_s")
        zeilen.append({k: v for k, v in {
            "turn_id": s.get("turn_id"), "uhr": _uhr(s.get("beginn"), tz, sekunden=tok is not None),
            "agent": s.get("agent"),
            "min": round(sek / 60, 1), "aufgabe_id": s.get("aufgabe_id"), "titel": s.get("titel"),
            "projekt": projekt, "quelle": s.get("quelle"), "projekt_quelle": s.get("projekt_quelle"),
            "geerbt": s.get("geerbt") or None,
            "tokens": {k: v for k, v in tok.items() if v} if tok is not None else None,
            "ende": _uhr(s.get("ende_letzter"), tz, sekunden=True) if tok is not None else None,
            "durchlauf_min": round(float(lauf) / 60, 1) if tok is not None and lauf is not None else None,
            "offen": s.get("zuege_offen") or None,
            "eingabe": s.get("eingabe_kurz") if texte else None,
            "satz": s.get("satz") if texte else None}.items() if v not in (None, "")})
        a = je_aufgabe.setdefault(s.get("aufgabe_id"), {"aufgabe_id": s.get("aufgabe_id"), "titel": s.get("titel"),
                                                        "projekt": projekt, "schritte": 0, "s": 0.0,
                                                        "agenten": set(), "tokens": None, "von": None,
                                                        "bis": None, "offen": 0})
        a["schritte"] += 1
        a["s"] += sek
        a["agenten"].add(s.get("agent") or "?")
        if tok is not None:
            a["tokens"] = a["tokens"] or {}
            for k, v in tok.items():
                a["tokens"][k] = a["tokens"].get(k, 0) + int(v or 0)
            a["von"] = min(filter(None, (a["von"], s.get("beginn"))), default=None, key=_zeitpunkt)
            a["bis"] = max(filter(None, (a["bis"], s.get("ende_letzter"))), default=None, key=_zeitpunkt)
            a["offen"] += int(s.get("zuege_offen") or 0)
        pr = je_projekt.setdefault(projekt, {"projekt": projekt, "s": 0.0, "schritte": 0})
        pr["s"] += sek
        pr["schritte"] += 1
    aufgaben = []
    for a in sorted(je_aufgabe.values(), key=lambda a: -a["s"]):
        zeile = {"aufgabe_id": a["aufgabe_id"], "titel": a["titel"], "projekt": a["projekt"],
                 "schritte": a["schritte"], "min": round(a["s"] / 60), "agenten": sorted(a["agenten"])}
        if a["tokens"] is not None:
            zeile["tokens"] = {k: v for k, v in a["tokens"].items() if v}
            zeile["von"], zeile["bis"] = _uhr(a["von"], tz, sekunden=True), _uhr(a["bis"], tz, sekunden=True)
            if a["von"] and a["bis"]:
                zeile["durchlauf_min"] = round((_zeitpunkt(a["bis"]) - _zeitpunkt(a["von"])).total_seconds() / 60, 1)
            if a["offen"]:
                zeile["offen"] = a["offen"]
        aufgaben.append(zeile)
    out: Dict[str, Any] = {
        "person": d.get("person"), "datum": d.get("datum"), "noch_offen": d.get("noch_offen"),
        "aufgaben": aufgaben,
        "projekte": [{"projekt": p["projekt"], "schritte": p["schritte"], "min": round(p["s"] / 60)}
                     for p in sorted(je_projekt.values(), key=lambda p: -p["s"])],
        "schritte": zeilen[:MAX_SCHRITTE],
    }
    if len(zeilen) > MAX_SCHRITTE:
        out["schritte_hinweis"] = f"{MAX_SCHRITTE} von {len(zeilen)} Schritten einzeln, Summen ueber alle"
    if d.get("katalog") is not None:
        out["katalog"] = [{"id": k.get("id"), "pfad": " › ".join(k.get("pfad") or [])} for k in d["katalog"]]
    return out


def _datum_uhr(iso: Optional[str], tz: ZoneInfo) -> Optional[str]:
    if not iso:
        return None
    return _zeitpunkt(iso).astimezone(tz).strftime("%d.%m. %H:%M")


def aufgabe_zusammenfassen(d: Dict[str, Any], texte: bool = True) -> Dict[str, Any]:
    """Eine Aufgabe (GET /api/aufgaben/{id}/schritte) kompakt: Kopf wie im Aufgaben-Ueberblick des
    Portals und die juengsten MAX_AUFGABE_SCHRITTE Schritte, aelteste zuerst. Texte nur, soweit
    die cloud-api sie liefert (#3430). Rein."""
    tz = ZoneInfo("Europe/Vienna")
    a = d.get("aufgabe") or {}
    zeit = a.get("zeit_s") or {}
    kopf = {k: v for k, v in {
        "id": a.get("id"), "titel": a.get("titel"), "kurz": a.get("kurz"), "satz": a.get("satz"),
        "mehrere_ziele": a.get("mehrere_ziele"), "projekt": _pfad_text(a.get("projekt")),
        "projekt_quelle": a.get("projekt_quelle"), "zustand": a.get("zustand"),
        "zustand_grund": a.get("zustand_grund"), "agenten": a.get("agenten") or None,
        "issue": a.get("issue"), "angelegt_durch": a.get("angelegt_durch"),
        "erledigt_am": _datum_uhr(a.get("erledigt_am"), tz),
        "min_heute": round(float(zeit.get("heute") or 0) / 60) or None,
        "min_woche": round(float(zeit.get("woche") or 0) / 60) or None}.items() if v not in (None, "", [])}
    alle = sorted(d.get("schritte") or [], key=lambda s: s.get("ts") or "")
    zeilen = []
    for s in alle[-MAX_AUFGABE_SCHRITTE:]:
        tok = s.get("tokens") if isinstance(s.get("tokens"), dict) else None
        zeilen.append({k: v for k, v in {
            "turn_id": s.get("turn_id"), "wann": _datum_uhr(s.get("ts"), tz), "agent": s.get("agent"),
            "quelle": s.get("quelle"), "korrigiert_von": s.get("korrigiert_von"),
            "min": round(float(s.get("zeit_s") or 0) / 60, 1), "zuege": s.get("zuege") or None,
            "tokens": {k: v for k, v in tok.items() if v} if tok is not None else None,
            "eingabe": s.get("eingabe_kurz") if texte else None,
            "satz": s.get("satz") if texte else None}.items() if v not in (None, "")})
    out: Dict[str, Any] = {"aufgabe": kopf, "schritte_gesamt": len(alle), "schritte": zeilen}
    if len(alle) > MAX_AUFGABE_SCHRITTE:
        out["schritte_hinweis"] = (f"die juengsten {MAX_AUFGABE_SCHRITTE} von {len(alle)} Schritten; "
                                   "aeltere im Portal oder mit aufgaben_tag je Tag")
    return out


def register_zeit_tools(mcp, transport: Transport) -> None:
    """``transport(method, path, params=?, timeout=?)`` ruft cloud-api mit dem JWT des Aufrufers."""

    @mcp.tool(
        name="zeit_tag",
        description=(
            "Ein Tag einer Person in Summen (Post #5880, Uebersicht „Zeit heute“). "
            "Wer das Recht `zeit:lesen` hat (Superadmin immer, Agenten mit dem Recht im Token), "
            "liest alle Menschen und Agenten; sonst nur die eigene Zeit (`umfang`: alles/eigene). "
            "Agenten MUESSEN `person` (Mail) nennen. `datum` = Kalendertag in Wien, JJJJ-MM-TT, leer = heute. "
            "Liefert `summen`, die Bahn `mensch` (aktive/passive Bildschirmzeit, Abschnitte "
            "[von, bis, zustand] in Ortszeit), `agenten` je Name (zuege, von_person = von dieser "
            "Person ausgeloest, dauer_s, zeilen, tokens), die 25 laengsten einzeln, der Rest als "
            "eine Zeile „+N weitere“ mit Summen, und die Top-10-`ordner` nach Umfang. "
            "Einzelne Zuege nur mit zuege=true, hoechstens 50 (die laengsten), mit Gesamtzahl. "
            "Andere Foederationen (z.B. P3) fragt cloud-api ueber die Bruecke mit ab (Post #5898): "
            "`foederationen` leer = alle freigeschalteten, `keine` = nur diese, sonst Komma-Liste. "
            "Drueben entscheidet die andere Foederation nach IHREN Rechten fuer den fragenden "
            "Menschen (bei Agenten der Besitzer). `quellen` nennt je Foederation Status (ok, "
            "keine_antwort, verweigert, fehler, nicht_abgefragt), Grund und Umfang; fehlt eine, "
            "steht es dort. Zuege von drueben tragen den Knoten `<bruecke>-<knoten>`. "
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

    @mcp.tool(
        name="zeit_tagesentwurf",
        description=(
            "Stundenzettel-Entwurf eines Tages (Alex 12.11.2026): die aktive Zeit einer Person in "
            "Bloecken, je Block ein Projekt aus den Projektordnern des Portals und eine kurze Notiz. "
            "Gleiches Recht und gleiche Parameter wie zeit_tag (Agenten MUESSEN `person` nennen). "
            "`art` sagt, woher die Zuordnung kommt: agent (Projektordner des Agenten, den die Person "
            "offen hatte oder angeschrieben hat), bruecke (andere Foederation, z.B. P3), host "
            "(Website gehoert zu einem Agenten mit Projekt), jev (Jev hat zugeordnet, mit "
            "`sicherheit`), ohne_ordner (Agent liegt in keinem Projektordner, seine Zeit steht "
            "unter seinem Namen), offen (keine Zuordnung, ggf. mit `vorschlag`). `nebenbei` nennt "
            "parallele Projekte ab 20 % Anteil. `summen`: zugeordnet_s (Ordner und Foederationen), "
            "ohne_ordner_s, offen_s, je_projekt in Minuten. jev=false schaltet Jev ab. Jev bekommt "
            "Uhrzeiten, Websites und Agentennamen: auf der Betreiber-Instanz fuer alle Menschen "
            "(Alex 12.11.2026), auf einer Kundeninstanz nur fuer den Tag des Besitzers, sonst "
            "`jev.aus` = keine_freigabe. "
            "Anwesenheit ist ein Personendatum: nicht in Beitraege, Issues oder an Dritte kopieren, "
            "nur dem Menschen selbst oder seinem Admin zeigen."
        ),
    )
    async def zeit_tagesentwurf(datum: str = "", person: str = "", foederationen: str = "",
                                jev: bool = True) -> Any:
        params = {k: v for k, v in (("datum", datum.strip()), ("person", person.strip()),
                                    ("foederationen", foederationen.strip())) if v}
        if not jev:
            params["jev"] = "false"
        d = await transport("GET", "/api/usage/tagesentwurf", params=params, timeout=90)
        if not isinstance(d, dict) or "eintraege" not in d:
            return d
        return json.dumps(entwurf_zusammenfassen(d), ensure_ascii=False, separators=(",", ":"))

    @mcp.tool(
        name="zeit_woche",
        description=(
            "Wochenrueckblick einer Person (#3089, Alex 15.11.2026): Summen je Projektordner und je "
            "Tag, ohne Bloecke und ohne Uhrzeiten. Je Tag derselbe Rechenweg wie zeit_tagesentwurf, "
            "gleiches Recht (Agenten MUESSEN `person` nennen). `von`/`bis` = Kalendertage in Wien "
            "(JJJJ-MM-TT), `bis` zaehlt mit, hoechstens 31 Tage; leer = Montag dieser Woche bis heute "
            "(am Sonntag die ganze Woche). Je Projekt: `mensch_h` (aktive Bildschirmzeit der Person), "
            "`direkt` (Zuege, die die Person selbst ausgeloest hat) und `kette` (Zuege, die aus ihren "
            "Auftraegen folgen: Agent-zu-Agent und Folgezuege, laut Mini-Hirn von einem ihrer Schritte "
            "geerbt), je mit zuege, h (Agentenstunden, parallele Agenten zaehlen einzeln), output "
            "(Ausgabe-Tokens) und zeilen. Ein Zug zaehlt beim Projektordner seines Agenten; ein Agent "
            "ohne Ordner steht unter seinem Namen (art agent). `tage`: dieselben Summen je Tag. "
            "null heisst KEINE MESSUNG, nicht 0: `messung` nennt je Tag, ob Anwesenheit (mensch), "
            "Agentenzuege (agenten) und Ketten (kette, Mini-Hirn seit 11.11.2026) gemessen wurden. "
            "`woche`: Summen und `tage_gemessen`. `quellen`: andere Foederationen (z.B. P3) und an "
            "welchen Tagen sie fehlten. Mails und Kalender (#2837): `mails` zaehlt je Projekt ein, "
            "aus_selbst und aus_agent (keine Dauer); `termine` = Kalendertermine (anzahl, dauer_h bis "
            "jetzt); `gesamt` = Zeit des Menschen ohne Doppelzaehlung (bildschirm_h ohne die Minuten in "
            "Terminen, termine_h, mail_annahme_h = ANGENOMMENE Minuten je Mail ausserhalb gemessener "
            "Zeit, gesamt_h). `mensch_h` bleibt die reine Bildschirmzeit. null = nicht erfasst. "
            "Dauert je nach Bereich einige Sekunden. "
            "Anwesenheit ist ein Personendatum: nicht in Beitraege, Issues oder an Dritte kopieren, "
            "nur dem Menschen selbst oder seinem Admin zeigen."
        ),
    )
    async def zeit_woche(person: str = "", von: str = "", bis: str = "", foederationen: str = "",
                         jev: bool = True) -> Any:
        params = {k: v for k, v in (("person", person.strip()), ("von", von.strip()), ("bis", bis.strip()),
                                    ("foederationen", foederationen.strip())) if v}
        if not jev:
            params["jev"] = "false"
        d = await transport("GET", "/api/usage/woche", params=params, timeout=180)
        if not isinstance(d, dict) or "projekte" not in d:
            return d
        return json.dumps(woche_zusammenfassen(d), ensure_ascii=False, separators=(",", ":"))

    @mcp.tool(
        name="aufgaben_tag",
        description=(
            "Mini-Hirn (Alex 13.11.2026): Welcher Aufgabe und welchem Projekt sind die Eingaben einer "
            "Person an einem Tag zugeordnet? Je Schritt (Eingabe des Menschen samt der Zuege, die "
            "davon erben) Uhrzeit, Agent, Minuten, aufgabe_id, Titel, Projektpfad, `quelle` und "
            "`projekt_quelle` (mensch = vom Menschen festgelegt; luna, folge, regel = automatisch; "
            "ohne = keine Aufgabe). Dazu Summen je Aufgabe und je Projekt und mit katalog=true alle "
            "Projektordner (id und Pfad), damit du einen Namen wie „INSECTA“ in eine projekt_id "
            "aufloest. `datum` = Kalendertag in Wien (JJJJ-MM-TT, leer = heute), `agent` = nur dieser "
            "Agent. `texte=true`: je Schritt auch `eingabe` (Anfang der Eingabe) und `satz` (Schluss "
            "des Agenten), um eine Zuordnung zu beurteilen. Die Texte liefert die cloud-api nur, wenn "
            "die Person dein Besitzer ist, oder wenn du ohne Leserecht in ihrem eigenen Zug liest; "
            "ein Leserecht allein (`zeit:lesen`, `zeit:lesen:<mail>`) gibt keine Texte. Lesen darf, "
            "wer `zeit:lesen` hat, oder ein Agent, dessen laufenden Zug genau diese Person selbst "
            "begonnen hat. Personendaten: nur dem Menschen selbst zeigen. "
            "`tokens=true` (Aufwand einer Anweisung, #3175): je Schritt und je Aufgabe auch `tokens` "
            "(anfragen, input = neue Eingabe, output, cache_read, cache_write), `ende` des letzten Zugs, "
            "`durchlauf_min` (Eingabe bis Ende des letzten Zugs) und `offen` (laufende Zuege, die Zahlen "
            "wachsen noch); `uhr` dann mit Sekunden. `min` ist die aktive Zeit, gekappt wie in der "
            "Uebersicht. Nenne Tokens immer getrennt nach Ausgabe, neuer Eingabe und Cache, nie als "
            "eine Summe: der Cache ist meist ueber 99 %. Ein Zug erscheint erst nach seinem Ende in "
            "der Ablage, die juengste Arbeit fehlt also noch."
        ),
    )
    async def aufgaben_tag(person: str, datum: str = "", agent: str = "", katalog: bool = True,
                           texte: bool = False, tokens: bool = False) -> Any:
        params = {k: v for k, v in (("person", person.strip()), ("datum", datum.strip()),
                                    ("agent", agent.strip())) if v}
        if katalog:
            params["katalog"] = "true"
        if tokens:
            params["tokens"] = "true"
        d = await transport("GET", "/api/aufgaben/tag", params=params, timeout=60)
        if not isinstance(d, dict) or "schritte" not in d:
            return d
        return json.dumps(aufgaben_zusammenfassen(d, texte=texte), ensure_ascii=False, separators=(",", ":"))

    @mcp.tool(
        name="aufgabe",
        description=(
            "Mini-Hirn (#3430): eine Aufgabe nach ihrer ID lesen, so wie der Mensch sie im "
            "Aufgaben-Ueberblick des Portals sieht. Schreibt dir ein Mensch aus einer Aufgabe heraus, "
            "steht vorne in seiner Nachricht eine Marke wie `[Aufgabe 308 · <Name> · <Projekt> · "
            "<Zustand> · … · Details: aufgabe(308)]`; dann rufst du dieses Werkzeug mit der ID, bevor "
            "du antwortest. Liefert den Kopf (titel, kurz, satz, mehrere_ziele, Projektpfad, zustand, "
            "agenten, issue, Minuten heute und in der Woche) und die juengsten 30 Schritte, aelteste "
            "zuerst: `wann`, Agent, `quelle` (mensch, luna, folge, regel), Minuten und mit texte=true "
            "(Vorgabe) die Eingabe des Menschen und den Schlusssatz des Agenten. Die Texte liefert die "
            "cloud-api nur, wenn die Person dein Besitzer ist oder deinen laufenden Zug selbst begonnen "
            "hat. Lesen darf, wer `zeit:lesen` hat, oder ein Agent im eigenen Zug der Person; sonst "
            "403. Personendaten: nur dem Menschen selbst zeigen. `tokens=true`: je Schritt auch Tokens."
        ),
    )
    async def aufgabe(id: int, texte: bool = True, tokens: bool = False) -> Any:  # noqa: A002 - Name aus der Marke
        params = {"tokens": "true"} if tokens else None
        d = await transport("GET", f"/api/aufgaben/{int(id)}/schritte", params=params, timeout=60)
        if not isinstance(d, dict) or "aufgabe" not in d:
            return d
        return json.dumps(aufgabe_zusammenfassen(d, texte=texte), ensure_ascii=False, separators=(",", ":"))

    @mcp.tool(
        name="aufgaben_umordnen",
        description=(
            "Mini-Hirn: Aufgaben oder Schritte einer Person umordnen, wenn sie es dir im Gespraech "
            "sagt („alles von heute war Projekt X“, „das gehoert zu Aufgabe 53“). Erlaubt NUR, wenn "
            "genau diese Person deinen laufenden Zug selbst begonnen hat (kein anderer Mensch im "
            "Zug, keine Kette ueber Agenten); sonst 403 mit `grund`. Nie aus eigenem Antrieb und "
            "nie fuer jemand anderen. Gespeichert wird „mensch:<mail> über <dein Name>“. "
            "Ziel genau eines: `projekt_id` (aus aufgaben_tag katalog), `aufgabe_id` oder `neu_titel` "
            "(neue Aufgabe, optional mit projekt_id). Auswahl genau eines: `aufgaben` (Liste von "
            "aufgabe_id), `turn_ids` (Schritte aus aufgaben_tag) oder `agent` + `datum` (alle "
            "Schritte dieses Agenten an dem Tag). Ganze Aufgaben + projekt_id: die Aufgaben wandern "
            "samt allen Schritten, ohne Modell; das ist der Weg fuer „das war alles Projekt X“. "
            "Schritte + aufgabe_id/neu_titel: verschoben, Quelle mensch. Schritte + nur projekt_id "
            "(z.B. agent + datum, wie der Knopf in der Tagesansicht des Portals): eine Aufgabe, deren "
            "Schritte alle ausgewaehlt sind, wandert ganz; eine Aufgabe mit Schritten auch anderswo "
            "gibt die ausgewaehlten an eine gleichnamige Aufgabe im Projekt ab; nur Schritte ohne "
            "Aufgabe ordnet Luna in dem Projekt ein (hoechstens 40). Pruefe vorher mit aufgaben_tag "
            "und sag dem Menschen danach, was du geaendert hast."
        ),
    )
    async def aufgaben_umordnen(person: str, projekt_id: str = "", aufgabe_id: int = 0, neu_titel: str = "",
                                aufgaben: Optional[List[int]] = None, turn_ids: Optional[List[str]] = None,
                                agent: str = "", datum: str = "") -> Any:
        if neu_titel.strip():
            ziel: Dict[str, Any] = {"neu": {"titel": neu_titel.strip(), "projekt_id": projekt_id.strip() or None}}
        elif aufgabe_id:
            ziel = {"aufgabe_id": int(aufgabe_id)}
        else:
            ziel = {"projekt_id": projekt_id.strip()}
        if aufgaben:
            auswahl: Dict[str, Any] = {"aufgaben": [int(i) for i in aufgaben]}
        elif turn_ids:
            auswahl = {"turn_ids": [str(t) for t in turn_ids]}
        else:
            auswahl = {"agent": agent.strip(), "datum": datum.strip()}
        return await transport("POST", "/api/aufgaben/umordnen", timeout=300,
                               json_body={"person": person.strip(), "ziel": ziel, "auswahl": auswahl})

    @mcp.tool(
        name="zeit_projekt",
        description=(
            "Projektsicht fuer einen Projektmanager-Agenten (#3122, Beitrag 6162): Aufgaben, Schritte "
            "und Zeit EINES Projekts. Mit dem Recht `zeit:projekt` ist das Projekt der Portal-Ordner, "
            "in dem du in der Registry liegst, samt Unterordnern (nie der Elternordner, nie private "
            "Ordner); `projekt` grenzt dann nur auf einen Unterordner ein. Ohne das Recht nennt ein "
            "Leser mit `zeit:lesen` das Projekt (`projekt` = Ordner-Id). `von`/`bis` = Kalendertage in "
            "Wien (JJJJ-MM-TT), `bis` zaehlt mit, hoechstens 31 Tage, leer = die letzten 7. "
            "`aufgaben`: aller Personen des Projekts, ohne Personenfeld, je mit zeit_s (Agentenzeit "
            "im Bereich), zuege, schritte, mails, knoten (Tree-Knoten der Aufgabe, Vorgabe) und `eigen` "
            "(Aufgabe deines Besitzers; nur dann steht der Text des letzten Schritts dabei). "
            "`ohne_aufgabe`: Schritte im Projekt ohne Aufgabe. `knoten`: je Tree-Knoten Agentenzeit, "
            "Zuege, Schritte und Mails, gezaehlt mit schritt.knoten ?? aufgabe.knoten ?? Regel (null = "
            "ohne Knoten; was bewusst ausserhalb des Plans liegt, steht dort als Teilsumme `ausserhalb`), "
            "je mit `je_quelle` (schritt, aufgabe, regel). `regeln`: die Regeln des Plans mit `treffer` "
            "im Bereich (abgeschaltete mit aktiv=false), `regeln_stand`: `fassung` und `geaendert_am` "
            "der aktiven Regeln, fuer Berichte. `tree_projekt` = der Tree-Plan des Ordners (null: noch "
            "nicht verknuepft). "
            "`menschenzeit`: Zeit deines Besitzers je Tag und Ordner (bildschirm_h, termine_h, "
            "mail_annahme_h = ANGENOMMENE Minuten je Mail, gesamt_h), nie die anderer Menschen; "
            "hoechstens 14 Tage je Abfrage (sonst 422 menschenzeit_zu_lang, laengere Zeitraeume in "
            "Abschnitten), denn je Tag wird der ganze Tagesentwurf gerechnet (7 Tage etwa 30 s). Nach "
            "einem Abbruch dieselbe Abfrage wiederholen: das fertige Ergebnis liegt einige Minuten "
            "bereit, eine laufende Rechnung wird geteilt. menschenzeit=false spart die Rechnung. "
            "`darf_zuordnen`: ob du zeit_projekt_zuordnen "
            "nutzen darfst. `aufgabe_id`: statt der Uebersicht die Schritte EINER Aufgabe deines "
            "Projekts, je mit turn_id, Uhrzeit, Agent, zuege, zeit_s, quelle und knoten/knoten_quelle "
            "(Knoten des Schritts, geht vor dem der Aufgabe); Eingabe und Satz nur bei Aufgaben deines "
            "Besitzers, eine Aufgabe ausserhalb deines Projekts ergibt 404. "
            "Menschenzeit ist ein Personendatum: nicht an andere Projektmitglieder."
        ),
    )
    async def zeit_projekt(projekt: str = "", von: str = "", bis: str = "", menschenzeit: bool = True,
                           aufgabe_id: int = 0) -> Any:
        # Ein Werkzeug fuer Uebersicht und Schritte: die Freigabe 6162 v3 nennt genau
        # mcp:cloud:zeit_projekt und mcp:cloud:zeit_projekt_zuordnen.
        if aufgabe_id:
            params = {"projekt": projekt.strip()} if projekt.strip() else {}
            d = await transport("GET", f"/api/usage/projekt/aufgaben/{int(aufgabe_id)}/schritte",
                                params=params, timeout=60)
            if not isinstance(d, dict) or "schritte" not in d:
                return d
            return json.dumps(d, ensure_ascii=False, separators=(",", ":"))
        params = {k: v for k, v in (("projekt", projekt.strip()), ("von", von.strip()), ("bis", bis.strip())) if v}
        if not menschenzeit:
            params["menschenzeit"] = "false"
        d = await transport("GET", "/api/usage/projekt", params=params, timeout=180)
        if not isinstance(d, dict) or "aufgaben" not in d:
            return d
        return json.dumps(d, ensure_ascii=False, separators=(",", ":"))

    @mcp.tool(
        name="zeit_projekt_zuordnen",
        description=(
            "Zuordnung im eigenen Projekt korrigieren (#3122), nur mit dem Recht `zeit:projekt:zuordnen`. "
            "`aenderungen` (1 bis 200), je eine von: {\"art\": \"knoten\", \"aufgabe\": <id>, \"knoten\": "
            "<Id des Tree-Knotens (ganze Zahl) oder null>} setzt die Vorgabe einer Aufgabe; {\"art\": \"knoten\", \"schritt\": "
            "<turn_id>, \"knoten\": …} den Knoten eines Schritts (geht vor); {\"art\": \"verschieben\", "
            "\"schritt\": <turn_id>, \"aufgabe\": <id>} haengt einen Schritt an eine andere Aufgabe "
            "desselben Projekts und derselben Person. Statt eines Knotens geht auch \"ausserhalb\" "
            "(bewusst kein Planknoten). Regeln (Beitrag 6169): {\"art\": \"regel\", \"wenn\": {agent, "
            "titel_enthaelt, ordner, ab, bis}, \"knoten\": <Id oder \"ausserhalb\">, \"rang\": 1..1000 "
            "(Vorgabe 100)} ordnet jeden Schritt zu, auf den alle genannten Bedingungen passen: agent und "
            "ab/bis (Tag in Wien) gelten fuer den Eingabezug des Schritts, Folgezuege erben; Mails einer "
            "Aufgabe ueber titel_enthaelt, ordner und ab/bis, nie ueber agent. Von Hand geht vor (Schritt "
            "vor Aufgabe vor Regel), Regeln eines Menschen vor deinen, dann der kleinere rang. Eine "
            "Regel aendert sich nie: {\"art\": \"regel_entfernen\", \"regel\": <id>} schaltet sie ab, "
            "dann legst du eine neue an (bei einem Phasenwechsel: alte mit bis, neue mit ab). "
            "Je Aenderung ein `ergebnis`: ok, unveraendert, "
            "unbekannt (nicht in deinem Projekt), ziel_unbekannt, andere_person, mensch_hat_gesetzt "
            "(was ein Mensch festgelegt hat, aenderst du nicht), knoten_nicht_im_plan (tree-api kennt "
            "den Knoten nicht im Tree-Plan deines Ordners), knoten_ist_zeitpunkt (Meilenstein oder "
            "Deliverable), ordner_unbekannt (wenn.ordner nicht im Projekt oder mit eigenem Plan), "
            "plan_ausserhalb_des_projekts; eine neue Regel nennt ihre `regel`-Id. Knoten und Regeln nur, "
            "wenn der Ordner mit einem "
            "Tree-Plan verknuepft ist (sonst 409 kein_tree_projekt). Den Status am Knoten setzt du "
            "direkt in tree-api, nicht hier. Aus dem Projekt hinaus geht nichts: "
            "Gehoert etwas nicht ins Projekt, frag den Menschen."
        ),
    )
    async def zeit_projekt_zuordnen(aenderungen: List[Dict[str, Any]], projekt: str = "") -> Any:
        body: Dict[str, Any] = {"aenderungen": list(aenderungen or [])}
        if projekt.strip():
            body["projekt"] = projekt.strip()
        return await transport("POST", "/api/usage/projekt/zuordnen", timeout=120, json_body=body)
