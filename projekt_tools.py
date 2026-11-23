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

    async def paket_schluessel(vorgabe_id: int, paket: str) -> str:
        _, post = await projekt_von(vorgabe_id)
        graph = await transport("GET", f"{API}/projekt/projekte/{_seg(post)}")
        treffer = [a for a in graph.get("aufgaben", []) if str(a.get("plan")) == str(paket)]
        if not treffer:
            raise ValueError(f"Paket {paket} gibt es in diesem Projekt nicht.")
        return str(treffer[0]["schluessel"])

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
          Das ist eine Handlung eines Freigebers.
        - Ein Projekt dreht sich um ein Deliverable: was der Mensch am Ende hat. `typ` (frage, funktion,
          fehler, dokument, inhalt, betrieb) und `deliverable` (in seinen Worten, woran er erkennt, dass es
          fertig ist, etwa „Screenshot der neuen Ansicht“) sind freiwillig; sonst erkennt sie der Planer.
        {RECHT}""",
    )
    async def vorgabe_anlegen(
        text: str,
        titel: Optional[str] = None,
        analyse_partner: Optional[List[str]] = None,
        analyse_fragen: Optional[List[str]] = None,
        typ: Optional[str] = None,
        deliverable: Optional[str] = None,
    ) -> Any:
        rumpf: Dict[str, Any] = _drop_none({"text": text, "titel": titel, "typ": typ, "deliverable": deliverable})
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
        description=f"""Löscht eine Vorgabe samt Projekt, wie „Löschen“ im Canvas: Die Projektleitung
        geht aus, niemand wird mehr geweckt, und wer schon an einem Paket arbeitet, bekommt Bescheid.

        - Ohne `endgueltig` („Aus der Liste nehmen“) bleiben Daten und Beiträge in Content stehen.
        - Mit `endgueltig=true` („Endgültig löschen“) entfernt die project-api auch den Projekt- und den
          Leitungsbeitrag in Content, eine eigene Projektleitungs-Sitzung und alle Daten der Vorgabe.
          Das lässt sich nicht rückgängig machen: nur, wenn der Mensch ausdrücklich „endgültig“ oder
          „alles löschen“ gesagt hat. Hochgeladene Dateien bleiben privat im Storage.
        {RECHT}""",
    )
    async def vorgabe_loeschen(vorgabe_id: int, endgueltig: bool = False) -> Any:
        if endgueltig:
            return await transport("DELETE", f"{API}/vorgaben/{_seg(vorgabe_id)}", params={"endgueltig": "true"})
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
            "deliverable": graph.get("deliverable"),
            "pakete": [{k: a.get(k) for k in ("plan", "titel", "agent", "phase", "status", "tor", "schluessel",
                                              "angenommen", "wartet_auf", "wartet_was", "wofuer", "lieferung",
                                              "folgearbeit", "nachweis")}
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
        schluessel = await paket_schluessel(vorgabe_id, paket)
        return await transport("POST", f"{API}/projekt/aufgaben/{schluessel}/tor", json_body={})

    @mcp.tool(
        name="paket_fragen",
        description=f"""Frage an den Zuständigen eines Pakets, wie „Frage an <Agent>“ im Canvas. Die Frage
        steht im Abschnitt der Aufgabe und weckt nur ihn; die Antwort liest du mit `paket_verlauf`.
        `paket` ist der Planschlüssel, etwa I2. {RECHT}""",
    )
    async def paket_fragen(vorgabe_id: int, paket: str, text: str) -> Any:
        schluessel = await paket_schluessel(vorgabe_id, paket)
        return await transport("POST", f"{API}/projekt/aufgaben/{schluessel}/frage", json_body={"text": text})

    @mcp.tool(
        name="paket_verlauf",
        description="""Meldungen, Fragen und Antworten zu einem Paket, ohne den Weckruf. `paket` ist der
        Planschlüssel, etwa I2.""",
    )
    async def paket_verlauf(vorgabe_id: int, paket: str) -> Any:
        schluessel = await paket_schluessel(vorgabe_id, paket)
        return await transport("GET", f"{API}/projekt/aufgaben/{schluessel}/verlauf")

    @mcp.tool(
        name="meine_aufgaben",
        description="""Deine Pakete in allen Projekten, mit Phase und letzter Meldung.""",
    )
    async def meine_aufgaben() -> Any:
        return await transport("GET", f"{API}/projekt/meine-aufgaben")

    @mcp.tool(
        name="deliverable",
        description="""Was der Mensch am Ende eines Projekts hat: Deliverable mit Nachweis, je Lieferstufe der
        Zustand (offen, in_arbeit, geliefert, nachbesserung, abgenommen), die Pakete dazu und die Abnahmen;
        Folgearbeit zählt nicht mit.""",
    )
    async def deliverable(vorgabe_id: int) -> Any:
        return await transport("GET", f"{API}/vorgaben/{_seg(vorgabe_id)}/deliverable")

    @mcp.tool(
        name="deliverable_nachtragen",
        description=f"""Trägt für ein laufendes Projekt ohne Deliverable eines nach. Der Planer legt es aus
        der Vorgabe fest, ordnet jedes Paket einer Lieferstufe oder der Folgearbeit zu und schreibt, wofür es
        gut ist. `hinweis`: was der Mensch dazu gesagt hat, im Wortlaut. {RECHT}""",
    )
    async def deliverable_nachtragen(vorgabe_id: int, hinweis: str = "") -> Any:
        return await transport("PUT", f"{API}/vorgaben/{_seg(vorgabe_id)}/deliverable",
                               json_body={"zuordnen": True, "hinweis": hinweis})

    @mcp.tool(
        name="deliverable_abnehmen",
        description=f"""Der Mensch nimmt eine Lieferstufe ab (`ergebnis="abgenommen"`) oder sagt, was noch
        fehlt (`ergebnis="fehlt"`, `text` Pflicht; daraus macht die Projektleitung Nacharbeit). Ohne `stufe`
        gilt die erste noch offene. Nur, wenn der Mensch das selbst gesagt hat. {RECHT}""",
    )
    async def deliverable_abnehmen(vorgabe_id: int, ergebnis: str, text: str = "", stufe: Optional[str] = None) -> Any:
        return await transport("POST", f"{API}/vorgaben/{_seg(vorgabe_id)}/abnahme",
                               json_body=_drop_none({"ergebnis": ergebnis, "text": text, "stufe": stufe}))

    @mcp.tool(
        name="projekt_entscheidung",
        description="""Trägt eine Entscheidung oder neue Regel von Alex zu einem laufenden Projekt ein, im
        Wortlaut. Sie hat Vorrang vor Plan und Paketen: Die Projektleitung prüft die offenen Pakete dagegen und
        schreibt betroffene Abnahmen um. Du überbringst nur, du entscheidest nicht.

        `quelle_art` und `quelle_ref` müssen nachlesbar sein:
        - `auftrag`: Alex hat deinen laufenden Zug selbst begonnen (ref leer). Dann gilt sie sofort.
        - `karte`: message_id der Karte, auf die Alex so geantwortet hat.
        - `austausch`: Nummer des Ereignisses im Austausch des Projekts.
        - `portal`: Agent und Zeit eines Portal-Zugs, etwa „Cloud 23.11. 15:34 UTC“. Den kann die project-api
          nicht nachlesen: Die Entscheidung bleibt ungeprüft, Alex bekommt eine Karte und bestätigt sie.""",
    )
    async def projekt_entscheidung(vorgabe_id: int, text: str, quelle_art: str, quelle_ref: str = "") -> Any:
        return await transport("POST", f"{API}/vorgaben/{_seg(vorgabe_id)}/entscheidungen",
                               json_body={"text": text, "quelle": {"art": quelle_art, "ref": quelle_ref}})

    @mcp.tool(
        name="projekt_entscheidungen",
        description="""Die Entscheidungen von Alex zu einem Projekt, mit Quelle, Prüfung und Stand
        (gilt, abgelehnt).""",
    )
    async def projekt_entscheidungen(vorgabe_id: int) -> Any:
        return await transport("GET", f"{API}/vorgaben/{_seg(vorgabe_id)}/entscheidungen")
