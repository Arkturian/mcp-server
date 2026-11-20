"""Token-Werkzeuge: Verbrauch und Auslastung fuer Agenten (#3532, Post 6351).

token_verbrauch liest cloud-api GET /api/usage/report mit `mass` (die Verdichtung rechnet
cloud-api, dieselbe Funktion wie das Portal), token_auslastung GET /api/usage/fusion.
Wer lesen darf und welche Konten ein Agent sieht, entscheidet cloud-api
(USAGE_REPORT_AGENTEN / USAGE_FUSION_AGENTEN, Erlaubnisliste USAGE_AGENT_KONTEN). Fuer
Agenten traegt die Antwort `gefiltert: true`: Summen und Anteile gelten dann nur fuer die
sichtbaren Konten, nicht fuer Alex' Gesamtzahl.

Antwort hoechstens ZIEL_BYTES; wird es mehr, kuerzt das Werkzeug von hinten (kleinste
Schluessel, aelteste Zeitpunkte), setzt `gekuerzt: true` und sagt, wie man eingrenzt.
Tokens immer getrennt nach input, output, cache_read und cache_write. `menge` ist nur
die Groesse, nach der sortiert wird (je `mass`).

Registrierung getrennt von server.py, damit die Werkzeuge gegen einen falschen Transport
geprueft werden koennen (wie zeit_tools).
"""

import json
from datetime import datetime, timedelta, timezone
from typing import Any, Awaitable, Callable, Dict, List, Optional

Transport = Callable[..., Awaitable[Any]]

ZIEL_BYTES = 20_000
GRUPPEN_BERICHT = ("agent", "konto", "node", "model", "quelle")
GRUPPEN = GRUPPEN_BERICHT + ("tag", "projekt")
MASSE = ("output", "eingabe", "alle", "zeilen", "aktiv")
TOKENARTEN = ("input", "output", "cache_read", "cache_write")
HINWEIS_KUERZEN = ("Antwort gekuerzt (hoechstens 20 KB). Eingrenzen: kuerzerer Zeitraum (von/bis), "
                   "agent= oder konto=, oder kleineres limit.")


def _iso(v: str) -> Optional[datetime]:
    if not v:
        return None
    t = datetime.fromisoformat(v.strip().replace("Z", "+00:00"))
    return t if t.tzinfo else t.replace(tzinfo=timezone.utc)


def zeitraum(von: str, bis: str, jetzt: Optional[datetime] = None) -> tuple:
    """Vorgabe: die letzten 24 h bis jetzt."""
    jetzt = jetzt or datetime.now(timezone.utc)
    t_bis = _iso(bis) or jetzt
    t_von = _iso(von) or (t_bis - timedelta(hours=24))
    return t_von, t_bis


def _takt(takt: str, t_von: datetime, t_bis: datetime) -> str:
    if takt in ("stunde", "tag"):
        return takt
    return "stunde" if (t_bis - t_von) <= timedelta(days=2) else "tag"


def _kompakt(d: Any) -> int:
    return len(json.dumps(d, ensure_ascii=False, separators=(",", ":")).encode())


def _kuerzen(out: Dict[str, Any], listen: List[str]) -> Dict[str, Any]:
    """Von hinten kuerzen, bis die Antwort unter ZIEL_BYTES liegt. Zaehlt je Liste mit."""
    if _kompakt(out) <= ZIEL_BYTES:
        return out
    out["gekuerzt"] = True
    out["hinweis"] = HINWEIS_KUERZEN
    weg: Dict[str, int] = {}
    for name in listen:
        liste = out.get(name)
        while isinstance(liste, list) and liste and _kompakt(out) > ZIEL_BYTES:
            liste.pop()
            weg[name] = weg.get(name, 0) + 1
            out["weggelassen"] = weg
    return out


def _r(x: Any, n: int = 4) -> Any:
    return round(x, n) if isinstance(x, float) else x


def _menge(z: Dict[str, Any], mass: str) -> float:
    aus = z.get("output") or 0
    ein = (z.get("input") or 0) + (z.get("cache_read") or 0) + (z.get("cache_write") or 0)
    return aus if mass == "output" else ein if mass == "eingabe" else aus + ein


def _kopf(d: Dict[str, Any], gruppe: str, mass: str) -> Dict[str, Any]:
    return {"von": d.get("von"), "bis": d.get("bis"), "gruppe": gruppe, "mass": mass,
            "takt": d.get("takt"), "tagesgrenze": d.get("tagesgrenze"),
            "gefiltert": bool(d.get("gefiltert")), "lieferungen": d.get("lieferungen") or {}}


def verbrauch_zusammenfassen(d: Dict[str, Any], gruppe: str, mass: str, limit: int = 200) -> Dict[str, Any]:
    """Bericht mit `verdichtung` (gruppe agent|konto|node|model|quelle) bzw. Tage (gruppe tag)."""
    out = _kopf(d, gruppe, mass)
    limit = max(1, min(int(limit or 200), 1000))
    if gruppe == "tag":
        je: Dict[str, Dict[str, Any]] = {}
        for z in d.get("zeilen") or []:
            t = je.setdefault(z["t"], {"t": z["t"], "antworten": 0, **{a: 0 for a in TOKENARTEN}})
            t["antworten"] += z.get("antworten") or 0
            for a in TOKENARTEN:
                t[a] += z.get(a) or 0
        tage = sorted(je.values(), key=lambda x: x["t"], reverse=True)       # juengster zuerst
        for t in tage:
            t["menge"] = _menge(t, mass)
        out["gesamt"] = sum(t["menge"] for t in tage)
        out["tage"] = tage[:limit]
        return _kuerzen(out, ["tage"])
    v = d.get("verdichtung") or {}
    out["gefiltert"] = bool(v.get("gefiltert", out["gefiltert"]))
    out["gesamt"] = v.get("gesamt", 0)
    out["schluessel_gesamt"] = len(v.get("rang") or [])
    out["rang"] = [{k: _r(x) for k, x in r.items()} for r in (v.get("rang") or [])[:limit]]
    out["verlauf"] = [{"t": s["t"], "summe": s["summe"]} for s in reversed(v.get("stapel") or [])]
    if mass == "zeilen" and v.get("zeilen_rang") is not None:
        out["zeilen_rang"] = [{k: _r(x) for k, x in r.items()} for r in v["zeilen_rang"][:limit]]
    if mass == "aktiv" and v.get("aktiv_rang") is not None:
        out["aktiv_rang"] = [{k: _r(x, 1) for k, x in r.items()} for r in v["aktiv_rang"][:limit]]
    if v.get("grund"):
        out["grund"] = v["grund"]
    return _kuerzen(out, ["verlauf", "zeilen_rang", "aktiv_rang", "rang"])


def projekte_zusammenfassen(d: Dict[str, Any], mass: str, limit: int = 200) -> Dict[str, Any]:
    """gruppe=aufgabe des Berichts je Projekt summiert (pp = Prozentpunkte des 7-Tage-Fensters)."""
    out = _kopf(d, "projekt", mass)
    je: Dict[str, Dict[str, Any]] = {}
    for z in d.get("zeilen") or []:
        p = z.get("projekt") or {}
        name = p.get("name") or ("ohne Aufgabe" if z.get("art") == "ohne" else "ohne Projekt")
        r = je.setdefault(name, {"projekt": name, "projekt_id": p.get("id"), "anfragen": 0, "pp": 0.0,
                                 **{a: 0 for a in TOKENARTEN}})
        tok = z.get("tokens") or {}
        r["anfragen"] += tok.get("anfragen") or 0
        r["pp"] += z.get("pp") or 0.0
        for a in TOKENARTEN:
            r[a] += tok.get(a) or 0
    rang = list(je.values())
    for r in rang:
        r["menge"] = _menge(r, mass if mass in ("output", "eingabe", "alle") else "output")
        r["pp"] = round(r["pp"], 3)
    rang.sort(key=lambda r: (-r["menge"], r["projekt"].casefold()))
    gesamt = sum(r["menge"] for r in rang)
    for r in rang:
        r["anteil"] = round(r["menge"] / gesamt, 4) if gesamt else 0.0
    out.update(gesamt=gesamt, schluessel_gesamt=len(rang), rang=rang[:max(1, min(int(limit or 200), 1000))])
    return _kuerzen(out, ["rang"])


def auslastung_zusammenfassen(d: Dict[str, Any], limit_agenten: int = 15) -> Dict[str, Any]:
    """Fusion je Konto: Summen ueber den Zeitraum, groesste Agenten, Verlauf (juengster zuerst)."""
    konten = []
    for k in d.get("konten") or []:
        e: Dict[str, Any] = {"konto": k.get("konto"), "quelle": k.get("quelle"), "profile": k.get("profile")}
        if k.get("quelle") != "claude":
            konten.append(e)
            continue
        ei = k.get("eichung") or {}
        e["eichung"] = {"quelle": ei.get("quelle"), "k_pct_je_mio_aequivalente": ei.get("k_pct_je_mio_aequivalente"),
                        "tage": ei.get("tage"), "r2": ei.get("r2")}
        zeilen = k.get("zeilen") or []
        summe = {f: round(sum(z.get(f) or 0 for z in zeilen), 3)
                 for f in ("pct_anstieg", "pct_gemessen", "pct_ungemessen", "pct_rest")}
        agenten: Dict[tuple, float] = {}
        for z in zeilen:
            for a in z.get("anteile") or []:
                schl = (a.get("agent"), a.get("node"))
                agenten[schl] = agenten.get(schl, 0.0) + (a.get("pct") or 0.0)
        top = sorted(agenten.items(), key=lambda x: -x[1])
        e.update(summe)
        e["agenten"] = [{"agent": a, "node": n, "pct": round(p, 3)} for (a, n), p in top[:limit_agenten]]
        if len(top) > limit_agenten:
            e["agenten_weitere"] = {"anzahl": len(top) - limit_agenten,
                                    "pct": round(sum(p for _, p in top[limit_agenten:]), 3)}
        e["verlauf"] = [[z["t"], z.get("pct_anstieg"), z.get("pct_gemessen"), z.get("pct_ungemessen")]
                        for z in reversed(zeilen)]
        konten.append(e)
    out = {"von": d.get("von"), "bis": d.get("bis"), "takt": d.get("takt"), "gefiltert": bool(d.get("gefiltert")),
           "verlauf_spalten": ["t", "pct_anstieg", "pct_gemessen", "pct_ungemessen"],
           "konten": konten, "lieferungen": d.get("lieferungen") or {}}
    if _kompakt(out) > ZIEL_BYTES:
        out["gekuerzt"] = True
        out["hinweis"] = HINWEIS_KUERZEN
        while _kompakt(out) > ZIEL_BYTES and any(len(k.get("verlauf") or []) > 1 for k in konten):
            for k in konten:
                if len(k.get("verlauf") or []) > 1:
                    k["verlauf"].pop()
        while _kompakt(out) > ZIEL_BYTES and any(k.get("agenten") for k in konten):
            for k in konten:
                if k.get("agenten"):
                    k["agenten"].pop()
    return out


def register_token_tools(mcp, transport: Transport) -> None:
    """``transport(method, path, params=?, timeout=?)`` ruft cloud-api mit dem JWT des Aufrufers."""

    @mcp.tool(
        name="token_verbrauch",
        description=(
            "Token-Verbrauch mit denselben Zahlen wie die Sicht Token-Verbrauch im Portal (#3532). "
            "`gruppe`: agent | konto | node | model | quelle | tag | projekt. `mass` bestimmt die "
            "Reihenfolge (`menge`): output (Vorgabe), eingabe (input + cache_read + cache_write), alle, "
            "zeilen (geaenderte Dateizeilen, nur agent/node), aktiv (aktive Sekunden, nur agent/node). "
            "Tokens stehen immer getrennt (input, output, cache_read, cache_write); reasoning steckt bei "
            "Codex schon in output. `von`/`bis` ISO, Vorgabe die letzten 24 h; Tage in Wiener Zeit. "
            "Agenten sehen nur die freigegebenen Konten: `gefiltert: true` heisst, Summen und Anteile "
            "gelten fuer diese Konten, nicht fuer den Gesamtverbrauch. Hoechstens 20 KB; sonst "
            "`gekuerzt: true` mit Hinweis zum Eingrenzen. Lesen nur fuer benannte Agenten und Admins."
        ),
    )
    async def token_verbrauch(von: str = "", bis: str = "", gruppe: str = "agent", mass: str = "output",
                              agent: str = "", konto: str = "", takt: str = "", limit: int = 200) -> Any:
        gruppe = (gruppe or "agent").strip().lower()
        mass = (mass or "output").strip().lower()
        if gruppe not in GRUPPEN:
            return {"fehler": f"gruppe: eine von {', '.join(GRUPPEN)}"}
        if mass not in MASSE:
            return {"fehler": f"mass: eine von {', '.join(MASSE)}"}
        try:
            t_von, t_bis = zeitraum(von, bis)
        except ValueError:
            return {"fehler": "von/bis: ISO-Zeit, z. B. 2026-11-20T00:00:00+01:00"}
        params: Dict[str, Any] = {"von": t_von.isoformat(), "bis": t_bis.isoformat()}
        for k, v in (("agent", agent), ("konto", konto)):
            if v.strip():
                params[k] = v.strip()
        if gruppe == "projekt":
            params.update(gruppe="aufgabe", takt="stunde" if (t_bis - t_von) <= timedelta(days=14) else "tag")
            d = await transport("GET", "/api/usage/report", params=params, timeout=120)
            if not isinstance(d, dict) or "zeilen" not in d:
                return d
            erg = projekte_zusammenfassen(d, mass, limit)
        else:
            params.update(gruppe="quelle" if gruppe == "tag" else gruppe, mass=mass, tagesgrenze="wien",
                          takt="tag" if gruppe == "tag" else _takt(takt, t_von, t_bis))
            d = await transport("GET", "/api/usage/report", params=params, timeout=120)
            if not isinstance(d, dict) or "zeilen" not in d:
                return d
            erg = verbrauch_zusammenfassen(d, gruppe, mass, limit)
        return json.dumps(erg, ensure_ascii=False, separators=(",", ":"))

    @mcp.tool(
        name="token_auslastung",
        description=(
            "Auslastung der Konten zusammengefuehrt mit den gezaehlten Tokens (#3532, Fusion): je "
            "Claude-Konto Anstieg des 7-Tage-Fensters in Prozentpunkten (pct_anstieg), davon Agenten "
            "zugeordnet (pct_gemessen), nicht zuordenbar (pct_ungemessen), die groessten Agenten mit "
            "ihren Punkten und der Verlauf (juengster zuerst). Codex-Konten nur mit Namen. "
            "`takt` stunde|tag, `von`/`bis` ISO, Vorgabe die letzten 24 h, `konto` optional. "
            "Agenten sehen nur die freigegebenen Konten (`gefiltert: true`). Hoechstens 20 KB. "
            "Lesen nur fuer benannte Agenten und Admins."
        ),
    )
    async def token_auslastung(takt: str = "stunde", von: str = "", bis: str = "", konto: str = "") -> Any:
        try:
            t_von, t_bis = zeitraum(von, bis)
        except ValueError:
            return {"fehler": "von/bis: ISO-Zeit, z. B. 2026-11-20T00:00:00+01:00"}
        params = {"takt": takt if takt in ("stunde", "tag") else "stunde",
                  "von": t_von.isoformat(), "bis": t_bis.isoformat()}
        if konto.strip():
            params["konto"] = konto.strip()
        d = await transport("GET", "/api/usage/fusion", params=params, timeout=120)
        if not isinstance(d, dict) or "konten" not in d:
            return d
        return json.dumps(auslastung_zusammenfassen(d), ensure_ascii=False, separators=(",", ":"))
