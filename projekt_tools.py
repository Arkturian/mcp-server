"""project-api tools: das Projektmanagement von AgentOS für Agenten (Owner: Project Manager).

Alex, 23.11.2026: „Die Systeme, die wir bauen, sind für Menschen und Agenten.“ Was ein Mensch im
Project Canvas (project.arkturian.com) tun kann, kann ein Agent hier genauso: Vorgaben anlegen,
Projekte starten, Pläne absegnen, die Projektleitung steuern und ihr antworten.

Das JWT des aufrufenden Agenten geht an die project-api (`/agenten/api/v1/` auf project.arkturian.com,
ohne Google-Gate). Lesen darf jeder Agent. Handlungen eines Freigebers erlaubt die project-api nur,
wenn ein freigebender Mensch den laufenden Zug des Agenten begonnen hat oder der Zug einen gültigen
Auftragsstempel eines Freigebers aus der eigenen Föderation trägt (sie fragt dafür die cloud-api).
"""

from typing import Any, Awaitable, Callable, Dict, List, Optional
from urllib.parse import quote

Transport = Callable[..., Awaitable[Any]]

API = "/api/v1"

RECHT = (
    "Nur im Auftrag eines freigebenden Menschen: Er hat deinen laufenden Zug selbst begonnen "
    "(etwa „starte das Projekt“) oder dir per gestempeltem Auftrag geschrieben. Sonst antwortet "
    "die project-api mit 403 `nur_freigeber`; dann nicht umgehen, sondern den Menschen fragen."
)


def _seg(value: Any) -> str:
    return quote(str(value), safe="")


def _drop_none(d: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in d.items() if v is not None}


def _kurz(eintraege: List[dict], anzahl: int) -> List[dict]:
    return list(eintraege or [])[-anzahl:]


def register_projekt_tools(mcp, transport: Transport) -> None:
    """Register the project tools on ``mcp``; ``transport(method, path, **kw)``."""

    async def vorgabe_holen(vorgabe_id: int) -> dict:
        return await transport("GET", f"{API}/vorgaben/{_seg(vorgabe_id)}")

    async def projekt_von(vorgabe_id: int) -> tuple[dict, int]:
        v = await vorgabe_holen(vorgabe_id)
        post = v.get("projekt_post_id")
        if not post:
            raise ValueError(f"Vorgabe {vorgabe_id} hat noch kein Projekt (Status {v.get('status')}).")
        return v, int(post)

    @mcp.tool(
        name="vorgaben_liste",
        description="""Vorgaben im Projektmanagement, die neuesten zuerst: id, titel, status und bei
        Projekten mit Phasen die Phase (analyse, planung, abstimmung, absegnen, ausfuehrung).
        Nimm das, um die `vorgabe_id` zu finden, wenn ein Mensch ein Projekt beim Namen nennt.""",
    )
    async def vorgaben_liste(status: Optional[str] = None, limit: int = 30) -> Any:
        liste = await transport("GET", f"{API}/vorgaben", params=_drop_none({"status": status, "limit": limit}))
        return [
            {k: v.get(k) for k in ("id", "titel", "status", "phase", "aktuelle_version", "erstellt_von",
                                   "geaendert_am")}
            for v in liste
        ]

    @mcp.tool(
        name="vorgabe",
        description="""Eine Vorgabe mit aktuellem Plan (Ziele, Pakete, Abnahme, Prüfung), Verlauf,
        Projektbeitrag (`projekt_post_id`), Rahmen der Projektleitung, Analyse und Anhängen.""",
    )
    async def vorgabe(vorgabe_id: int) -> Any:
        return await vorgabe_holen(vorgabe_id)

    @mcp.tool(
        name="vorgabe_anlegen",
        description=f"""Legt eine Vorgabe an: Was soll erreicht werden, in den Worten des Menschen.

        - Ohne `analyse_partner` plant der Planer sofort; der Plan wartet dann auf Freigabe
          (`projekt_starten`).
        - Mit `analyse_partner` (1 bis 6 Agenten aus dem Katalog) klärt die Projektleitung zuerst mit
          ihnen den Stand und schreibt einen Befund; geplant wird danach, abgestimmt mit den
          Beteiligten, und Alex segnet ab. Optional `analyse_fragen` des Menschen.
          Das ist eine Handlung eines Freigebers. {RECHT}""",
    )
    async def vorgabe_anlegen(
        text: str,
        titel: Optional[str] = None,
        analyse_partner: Optional[List[str]] = None,
        analyse_fragen: Optional[List[str]] = None,
    ) -> Any:
        rumpf: Dict[str, Any] = _drop_none({"text": text, "titel": titel})
        if analyse_partner:
            rumpf["analyse"] = {"partner": list(analyse_partner), "fragen": list(analyse_fragen or [])}
        return await transport("POST", f"{API}/vorgaben", json_body=rumpf)

    @mcp.tool(
        name="projekt_starten",
        description=f"""Startet das Projekt einer Vorgabe, so wie „Projekt starten“ im Canvas.

        - Plan zur Prüfung (ohne Phasen): gibt die aktuelle Fassung frei und legt das Projekt an; die
          ersten Pakete werden geweckt.
        - Projekt in der Phase `absegnen` (alle Beteiligten haben zugestimmt): segnet den Plan ab, die
          Ausführung beginnt.
        - Sonst sagt die Antwort, woran es hängt (etwa: Analyse oder Abstimmung läuft noch).
        {RECHT}""",
    )
    async def projekt_starten(vorgabe_id: int) -> Any:
        v = await vorgabe_holen(vorgabe_id)
        post, phase = v.get("projekt_post_id"), v.get("phase")
        if post and phase == "absegnen":
            return await transport("POST", f"{API}/projekt/projekte/{_seg(post)}/plan/absegnen", json_body={})
        if post:
            return {"gestartet": False, "projekt_post_id": post, "phase": phase or "ausfuehrung",
                    "hinweis": "Das Projekt läuft schon. In der Analyse oder Abstimmung startet es nach "
                               "der Freigabe der Beteiligten; dann erneut aufrufen."}
        version = v.get("aktuelle_version")
        if not version or v.get("status") not in ("zur_pruefung", "freigegeben"):
            return {"gestartet": False, "status": v.get("status"),
                    "hinweis": "Es gibt noch keinen Plan zur Freigabe."}
        return await transport("POST", f"{API}/projekt/vorgaben/{_seg(vorgabe_id)}/umsetzen",
                               json_body={"version": version})

    @mcp.tool(
        name="plan_ueberarbeiten",
        description=f"""Der Planer erstellt aus der Anmerkung eine neue Fassung der aktuellen Planung
        (nur solange der Plan zur Prüfung steht). {RECHT}""",
    )
    async def plan_ueberarbeiten(vorgabe_id: int, anmerkung: str) -> Any:
        v = await vorgabe_holen(vorgabe_id)
        return await transport("POST", f"{API}/vorgaben/{_seg(vorgabe_id)}/ueberarbeiten",
                               json_body={"version": v.get("aktuelle_version"), "anmerkung": anmerkung})

    @mcp.tool(
        name="vorgabe_ablehnen",
        description=f"""Lehnt eine Vorgabe samt aktuellem Plan ab, mit Grund. {RECHT}""",
    )
    async def vorgabe_ablehnen(vorgabe_id: int, grund: str) -> Any:
        return await transport("POST", f"{API}/vorgaben/{_seg(vorgabe_id)}/ablehnen", json_body={"grund": grund})

    @mcp.tool(
        name="vorgabe_loeschen",
        description=f"""Löscht eine Vorgabe samt Projekt aus der Liste, wie „Löschen“ im Canvas: Die
        Projektleitung geht aus, niemand wird mehr geweckt, und wer schon an einem Paket arbeitet,
        bekommt Bescheid. Daten und Beiträge in Content bleiben stehen. {RECHT}""",
    )
    async def vorgabe_loeschen(vorgabe_id: int) -> Any:
        return await transport("DELETE", f"{API}/vorgaben/{_seg(vorgabe_id)}")

    @mcp.tool(
        name="projekt_stand",
        description="""Stand eines Projekts auf einen Blick: Phase, Pakete mit Zuständigen und Phase
        (wartet, bereit, aktiv, blockiert, tor, fertig), Analyse (Antworten je Gesprächspartner),
        Befund, Abstimmung über den Plan, die letzten Fragen der Projektleitung an Alex und die
        letzten Einträge ihres Logbuchs.""",
    )
    async def projekt_stand(vorgabe_id: int) -> Any:
        v, post = await projekt_von(vorgabe_id)
        graph = await transport("GET", f"{API}/projekt/projekte/{_seg(post)}")
        leitung = await transport("GET", f"{API}/projekt/projekte/{_seg(post)}/leitung")
        return {
            "vorgabe_id": vorgabe_id, "titel": v.get("titel"), "projekt_post_id": post,
            "phase": leitung.get("phase") or v.get("phase"),
            "pakete": [{k: a.get(k) for k in ("plan", "titel", "agent", "phase", "status", "tor", "schluessel")}
                       for a in graph.get("aufgaben", [])],
            "phasen": leitung.get("phasen"),
            "projektleitung": {"modus": leitung.get("modus"), "gehirn": leitung.get("gehirn"),
                               "austausch": _kurz(leitung.get("austausch"), 5),
                               "logbuch": _kurz(leitung.get("logbuch"), 5)},
        }

    @mcp.tool(
        name="plan_absegnen",
        description=f"""Segnet den abgestimmten Plan ab, wie „Freigeben“ auf der Karte „Plan absegnen“.
        Geht nur in der Phase `absegnen`, also wenn alle Beteiligten zugestimmt haben. {RECHT}""",
    )
    async def plan_absegnen(vorgabe_id: int) -> Any:
        _, post = await projekt_von(vorgabe_id)
        return await transport("POST", f"{API}/projekt/projekte/{_seg(post)}/plan/absegnen", json_body={})

    @mcp.tool(
        name="projektleitung_schalten",
        description=f"""Schalter je Projekt: `modus` an (handelt selbst), trocken (schreibt nur ins
        Logbuch, was sie täte) oder aus; `gehirn` kette (zustandslos) oder sitzung (eigene
        Opus-Sitzung). {RECHT}""",
    )
    async def projektleitung_schalten(vorgabe_id: int, modus: str, gehirn: str = "kette") -> Any:
        return await transport("PUT", f"{API}/vorgaben/{_seg(vorgabe_id)}/leitung",
                               json_body={"modus": modus, "gehirn": gehirn})

    @mcp.tool(
        name="projektleitung_antworten",
        description=f"""Antwort oder Hinweis an die Projektleitung eines Projekts, wie im Austausch des
        Canvas. Ein „Freigeben“ in der Phase `absegnen` segnet den Plan ab. {RECHT}""",
    )
    async def projektleitung_antworten(vorgabe_id: int, text: str) -> Any:
        _, post = await projekt_von(vorgabe_id)
        return await transport("POST", f"{API}/projekt/projekte/{_seg(post)}/leitung/antwort",
                               json_body={"text": text})

    @mcp.tool(
        name="tor_oeffnen",
        description=f"""Öffnet das Tor vor einem Paket (Geld, Rechte, Außenwirkung, Löschen,
        Kundendaten); das Paket startet sofort. `paket` ist der Planschlüssel, etwa I2. {RECHT}""",
    )
    async def tor_oeffnen(vorgabe_id: int, paket: str) -> Any:
        _, post = await projekt_von(vorgabe_id)
        graph = await transport("GET", f"{API}/projekt/projekte/{_seg(post)}")
        treffer = [a for a in graph.get("aufgaben", []) if str(a.get("plan")) == str(paket)]
        if not treffer:
            raise ValueError(f"Paket {paket} gibt es in diesem Projekt nicht.")
        return await transport("POST", f"{API}/projekt/aufgaben/{treffer[0]['schluessel']}/tor", json_body={})

    @mcp.tool(
        name="meine_aufgaben",
        description="""Deine Pakete in allen Projekten, mit Phase und letzter Meldung.""",
    )
    async def meine_aufgaben() -> Any:
        return await transport("GET", f"{API}/projekt/meine-aufgaben")
