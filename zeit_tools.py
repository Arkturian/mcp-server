"""Zeit-Werkzeuge: der Tag einer Person in Summen (cloud-api GET /api/usage/tag), dazu die
Aufgaben des Mini-Hirns lesen und umordnen (GET /api/aufgaben/tag, POST /api/aufgaben/umordnen,
Alex ueber CloudV2-time 13.11.2026).

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


def _pfad_text(p: Optional[Dict[str, Any]]) -> Optional[str]:
    if not p:
        return None
    return " › ".join(p.get("pfad") or [p.get("name") or p.get("id") or ""])


def aufgaben_zusammenfassen(d: Dict[str, Any]) -> Dict[str, Any]:
    """Die Schritte eines Tages (GET /api/aufgaben/tag) kompakt, mit Summen je Aufgabe und
    Projekt und dem Projektkatalog. Rein, ohne Netz."""
    tz = ZoneInfo("Europe/Vienna")
    schritte = d.get("schritte") or []
    je_aufgabe: Dict[Any, Dict[str, Any]] = {}
    je_projekt: Dict[Any, Dict[str, Any]] = {}
    zeilen = []
    for s in schritte:
        sek = float(s.get("zeit_s") or s.get("dauer_s") or 0)
        projekt = _pfad_text(s.get("projekt"))
        zeilen.append({k: v for k, v in {
            "turn_id": s.get("turn_id"), "uhr": _uhr(s.get("beginn"), tz), "agent": s.get("agent"),
            "min": round(sek / 60, 1), "aufgabe_id": s.get("aufgabe_id"), "titel": s.get("titel"),
            "projekt": projekt, "quelle": s.get("quelle"), "projekt_quelle": s.get("projekt_quelle"),
            "geerbt": s.get("geerbt") or None}.items() if v not in (None, "")})
        a = je_aufgabe.setdefault(s.get("aufgabe_id"), {"aufgabe_id": s.get("aufgabe_id"), "titel": s.get("titel"),
                                                        "projekt": projekt, "schritte": 0, "s": 0.0,
                                                        "agenten": set()})
        a["schritte"] += 1
        a["s"] += sek
        a["agenten"].add(s.get("agent") or "?")
        pr = je_projekt.setdefault(projekt, {"projekt": projekt, "s": 0.0, "schritte": 0})
        pr["s"] += sek
        pr["schritte"] += 1
    aufgaben = [{"aufgabe_id": a["aufgabe_id"], "titel": a["titel"], "projekt": a["projekt"],
                 "schritte": a["schritte"], "min": round(a["s"] / 60), "agenten": sorted(a["agenten"])}
                for a in sorted(je_aufgabe.values(), key=lambda a: -a["s"])]
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
        name="aufgaben_tag",
        description=(
            "Mini-Hirn (Alex 13.11.2026): Welcher Aufgabe und welchem Projekt sind die Eingaben einer "
            "Person an einem Tag zugeordnet? Je Schritt (Eingabe des Menschen samt der Zuege, die "
            "davon erben) Uhrzeit, Agent, Minuten, aufgabe_id, Titel, Projektpfad, `quelle` und "
            "`projekt_quelle` (mensch = vom Menschen festgelegt; luna, folge, regel = automatisch; "
            "ohne = keine Aufgabe). Dazu Summen je Aufgabe und je Projekt und mit katalog=true alle "
            "Projektordner (id und Pfad), damit du einen Namen wie „INSECTA“ in eine projekt_id "
            "aufloest. Ohne Eingabetexte. `datum` = Kalendertag in Wien (JJJJ-MM-TT, leer = heute), "
            "`agent` = nur dieser Agent. Lesen darf, wer `zeit:lesen` hat, oder ein Agent, dessen "
            "laufenden Zug genau diese Person selbst begonnen hat. Personendaten: nur dem Menschen "
            "selbst zeigen."
        ),
    )
    async def aufgaben_tag(person: str, datum: str = "", agent: str = "", katalog: bool = True) -> Any:
        params = {k: v for k, v in (("person", person.strip()), ("datum", datum.strip()),
                                    ("agent", agent.strip())) if v}
        if katalog:
            params["katalog"] = "true"
        d = await transport("GET", "/api/aufgaben/tag", params=params, timeout=60)
        if not isinstance(d, dict) or "schritte" not in d:
            return d
        return json.dumps(aufgaben_zusammenfassen(d), ensure_ascii=False, separators=(",", ":"))

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
            "Schritte + aufgabe_id/neu_titel: verschoben, Quelle mensch. Schritte + nur projekt_id: "
            "Luna waehlt je Schritt die Aufgabe in dem Projekt (hoechstens 40, dauert Sekunden je "
            "Schritt). Pruefe vorher mit aufgaben_tag und sag dem Menschen danach, was du geaendert hast."
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
