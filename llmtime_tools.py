"""LLM-TIME-API tools: measured forecasts instead of guessed durations.

Contract: Content-Post #5801 (TIME-LLM), API state 504379d.

Registration is isolated from the other MCP domains so the tools can be
exercised against a fake transport. The transport decides the credential
(caller JWT inside the federation, an instance key on a customer gateway);
the tools only shape requests.
"""

from typing import Any, Awaitable, Callable, Dict, List, Optional
from urllib.parse import quote

Transport = Callable[..., Awaitable[Any]]

API = "/api/v1"

FORECAST_RULE = (
    "Prognose abfragen statt Dauer schätzen: Die Werte beruhen auf gemessener "
    "Erfahrung dieser Instanz. Gib P50/P85/P95 samt Annahmen weiter. Bei "
    "‚nicht konvergent‘ oder ‚keine Datengrundlage‘ nennst du den Grund "
    "(Feld `grund`) und kein Datum."
)


AUSSCHLUSS = (
    "`bis_ausschluss`: einmalige Sonderereignisse, die bewusst aus der "
    "Stichprobe fallen (je Eintrag `YYYY-MM-DD` oder `von..bis`). Normale Tage "
    "ohne Abschluss NICHT ausschließen, sie zählen als 0 — auch Ausfälle wie "
    "ein Kontolimit. Nur so misst das Fenster die echte Kapazität."
)


def _drop_none(d: Dict[str, Any]) -> Dict[str, Any]:
    """Keep explicit falsy values (0, False, ""), drop only unset ones."""
    return {k: v for k, v in d.items() if v is not None}


def _item_path(item_id: str, action: str = "") -> str:
    path = f"{API}/items/{quote(str(item_id), safe='')}"
    return f"{path}/{action}" if action else path


def _forecast_params(
    projekt: Optional[str],
    meilenstein: Optional[str],
    fenster_tage: Optional[int],
    bis_ausschluss: Optional[List[str]],
    stichtag: Optional[str],
    seed: Optional[int],
    stichproben: Optional[int],
) -> Dict[str, Any]:
    return _drop_none({
        "projekt": projekt,
        "meilenstein": meilenstein,
        "fenster_tage": fenster_tage,
        "bis_ausschluss": ",".join(bis_ausschluss) if bis_ausschluss else None,
        "stichtag": stichtag,
        "seed": seed,
        "stichproben": stichproben,
    })


def register_llmtime_tools(mcp, transport: Transport) -> None:
    """Register the LLM-TIME tools on ``mcp``; ``transport(method, path, **kw)``."""

    @mcp.tool(
        name="item_add",
        description=f"""Arbeitspaket anlegen (POST /items).

        Wann: sobald du eine Aufgabe, einen Fix oder Nacharbeit übernimmst oder
        erkennst. Bei Nacharbeit `rework_of` = Id des ursprünglichen Pakets.
        Pflicht ist nur `projekt`. Ohne `id` vergibt der Dienst eine; dieselbe
        `id` ein zweites Mal liefert das vorhandene Paket (nichts doppelt).
        `meilenstein` darf nicht leer oder nur Leerzeichen sein (422).
        `created` ISO-8601 mit Zeitzone, leer = jetzt.
        Antwort: das Paket mit `status: "offen"`.""",
    )
    async def item_add(
        projekt: str,
        meilenstein: Optional[str] = None,
        typ: Optional[str] = None,
        bereich: Optional[str] = None,
        agent: Optional[str] = None,
        ausloeser: Optional[str] = None,
        rework_of: Optional[str] = None,
        beleg: Optional[str] = None,
        id: Optional[str] = None,
        created: Optional[str] = None,
    ) -> Any:
        body = _drop_none({
            "projekt": projekt, "meilenstein": meilenstein, "typ": typ,
            "bereich": bereich, "agent": agent, "ausloeser": ausloeser,
            "rework_of": rework_of, "beleg": beleg, "id": id, "created": created,
        })
        return await transport("POST", f"{API}/items", json_body=body)

    @mcp.tool(
        name="item_start",
        description="""Arbeit an einem Paket beginnt (POST /items/{id}/start).

        Optional, aber ohne Start zählt die Durchlaufzeit ab dem Anlegen.
        `at` ISO-8601, leer = jetzt. 409 bei unerlaubtem Zustandswechsel.""",
    )
    async def item_start(id: str, at: Optional[str] = None) -> Any:
        return await transport("POST", _item_path(id, "start"), json_body=_drop_none({"at": at}))

    @mcp.tool(
        name="item_finish",
        description="""Paket fertig (POST /items/{id}/finish). Erst das macht es zum Durchsatz.

        `at` ISO-8601, leer = jetzt. `tokens` = Opus-Input-Äquivalente
        (Formel cloud-api usage_fusion), nur wenn du sie kennst. `zeilen` =
        geänderte Codezeilen (hinzu + entfernt, z. B. aus `git diff --numstat`),
        falls bekannt. 409 bei einem zweiten finish.""",
    )
    async def item_finish(id: str, at: Optional[str] = None, tokens: Optional[int] = None,
                          zeilen: Optional[int] = None) -> Any:
        return await transport(
            "POST", _item_path(id, "finish"),
            json_body=_drop_none({"at": at, "tokens": tokens, "zeilen": zeilen}),
        )

    @mcp.tool(
        name="item_discard",
        description="""Paket verwerfen (POST /items/{id}/discard).

        Für Arbeit, die nicht mehr gemacht wird. Ein verworfenes Paket senkt
        den Zufluss und verfälscht sonst die Prognose. `at` leer = jetzt.""",
    )
    async def item_discard(id: str, at: Optional[str] = None) -> Any:
        return await transport("POST", _item_path(id, "discard"), json_body=_drop_none({"at": at}))

    @mcp.tool(
        name="items_list",
        description="""Pakete auflisten (GET /items).

        Wann: um offene oder fertige Pakete zu sehen, etwa bevor du ein
        Duplikat anlegst. `status`: offen | begonnen | fertig | verworfen.
        `limit` Vorgabe 500, höchstens 10 000.""",
    )
    async def items_list(
        projekt: Optional[str] = None,
        meilenstein: Optional[str] = None,
        status: Optional[str] = None,
        agent: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> Any:
        params = _drop_none({
            "projekt": projekt, "meilenstein": meilenstein, "status": status,
            "agent": agent, "limit": limit,
        })
        return await transport("GET", f"{API}/items", params=params)

    @mcp.tool(
        name="forecast_when",
        description=f"""Wann ist der Rest fertig? (GET /forecast/when, Monte-Carlo über gemessenen Durchsatz)

        Wann: IMMER, wenn du sonst „das dauert etwa X Tage“ oder „bis Freitag“
        sagen würdest. {FORECAST_RULE}
        `rest` leer = offene Pakete im Projekt bzw. Meilenstein.
        `zufluss`: leer = aus bei angegebenem `rest` (feste Menge: wann sind
        diese N fertig), an bei offenen Paketen (wann ist der Rückstand leer,
        während neue Arbeit dazukommt). `true` kann „nicht konvergent“ ergeben.
        `fenster_tage` Vorgabe 14. `stichtag` Vorgabe heute.
        {AUSSCHLUSS}
        Ein Perzentil mit `tage: null` liegt jenseits des Horizonts (3650 Tage).""",
    )
    async def forecast_when(
        projekt: str,
        meilenstein: Optional[str] = None,
        rest: Optional[int] = None,
        zufluss: Optional[bool] = None,
        fenster_tage: Optional[int] = None,
        bis_ausschluss: Optional[List[str]] = None,
        stichtag: Optional[str] = None,
        seed: Optional[int] = None,
        stichproben: Optional[int] = None,
    ) -> Any:
        params = _forecast_params(projekt, meilenstein, fenster_tage, bis_ausschluss,
                                  stichtag, seed, stichproben)
        if rest is not None:
            params["rest"] = rest
        if zufluss is not None:
            params["zufluss"] = "true" if zufluss else "false"
        return await transport("GET", f"{API}/forecast/when", params=params)

    @mcp.tool(
        name="forecast_how_many",
        description=f"""Wie viele Pakete bis Datum X? (GET /forecast/how_many)

        Wann: bei „was schaffen wir bis …?“ (fester Termin, offener Umfang).
        {FORECAST_RULE}
        P85 ist die Anzahl, die in mindestens 85 % der Läufe erreicht wird,
        daher P95 ≤ P85 ≤ P50. `bis` Pflicht, `YYYY-MM-DD`.
        {AUSSCHLUSS}""",
    )
    async def forecast_how_many(
        projekt: str,
        bis: str,
        meilenstein: Optional[str] = None,
        fenster_tage: Optional[int] = None,
        bis_ausschluss: Optional[List[str]] = None,
        stichtag: Optional[str] = None,
        seed: Optional[int] = None,
        stichproben: Optional[int] = None,
    ) -> Any:
        params = _forecast_params(projekt, meilenstein, fenster_tage, bis_ausschluss,
                                  stichtag, seed, stichproben)
        params["bis"] = bis
        return await transport("GET", f"{API}/forecast/how_many", params=params)

    @mcp.tool(
        name="forecast_aufwand",
        description=f"""Wie viel Aufwand brauchen N Pakete dieser Art? (GET /forecast/aufwand)

        Wann: bevor du eine Aufgabe übernimmst oder ein Budget nennst.
        {FORECAST_RULE}
        `projekt`, `agent`, `typ`, `bereich` bilden die Referenzklasse.
        `groesse`: stunden (Vorgabe, Durchlaufzeit inkl. Warten) |
        aktiv_stunden (reine Agentenzeit aus den Turns, ohne Warten) |
        tokens | zeilen (geänderte Codezeilen). `anzahl` Vorgabe 1.
        P85 wird in 85 % der Läufe NICHT überschritten (Budgetsicht).
        Gib `quellen` mit aus: es zeigt, wie viel eigene und wie viel
        allgemeine Erfahrung eingeflossen ist. Tokens gibt es nur, wenn Pakete
        mit `tokens` abgeschlossen wurden, aktiv_stunden nur mit eingespielten
        Turns, zeilen nur mit `zeilen` am Paket.""",
    )
    async def forecast_aufwand(
        projekt: Optional[str] = None,
        agent: Optional[str] = None,
        typ: Optional[str] = None,
        bereich: Optional[str] = None,
        groesse: Optional[str] = None,
        anzahl: Optional[int] = None,
        seed: Optional[int] = None,
        stichproben: Optional[int] = None,
    ) -> Any:
        params = _drop_none({
            "projekt": projekt, "agent": agent, "typ": typ, "bereich": bereich,
            "groesse": groesse, "anzahl": anzahl, "seed": seed, "stichproben": stichproben,
        })
        return await transport("GET", f"{API}/forecast/aufwand", params=params)

    @mcp.tool(
        name="import_merges",
        description="""Repo-Historie einspielen (POST /import/merges), höchstens 20 000 Merges je Aufruf.

        Du liest selbst aus Git, z. B. `git log --first-parent --merges`.
        Je Merge: `commit` (Hash = Id), `nachricht`, `merged_at`,
        `first_commit_at` (erster Commit von merge-base..^2), optional
        `zweig`, `autor`, `meilenstein` (nicht leer; gilt vor den Regeln),
        `zeilen` (geänderte Zeilen des Merges, z. B. `git diff --numstat ^1..merge`).
        `commit` ohne '/' und Leerraum.
        `typ`, `bereich` und PR-Nummer liest der Dienst aus der Nachricht.
        `regeln`: [{muster, meilenstein, feld: nachricht|zweig|beide}];
        fehlt es, gelten die gespeicherten Regeln des Projekts.
        `aktualisieren` (Vorgabe false): bekannte Hashes im selben Projekt
        bekommen typ/bereich/beleg neu; der Meilenstein ändert sich nur durch
        ausdrückliches Mitschicken (sonst bleibt der vorhandene, Regeln
        greifen nur bei Paketen ohne Meilenstein). Zeiten/Zustand bleiben.
        Ohne Flag idempotent.
        Antwort: gelesen = neu + vorhanden + konflikt. `vorhanden` = bekannt im
        selben Projekt, `konflikt` = Hash gehört zu einem anderen Projekt
        (bleibt unverändert), `aktualisiert`, `ohne_meilenstein`.""",
    )
    async def import_merges(
        projekt: str,
        merges: List[Dict[str, Any]],
        regeln: Optional[List[Dict[str, Any]]] = None,
        aktualisieren: bool = False,
    ) -> Any:
        body: Dict[str, Any] = {"projekt": projekt, "merges": merges, "aktualisieren": aktualisieren}
        if regeln is not None:
            body["regeln"] = regeln
        return await transport("POST", f"{API}/import/merges", json_body=body)

    @mcp.tool(
        name="rules_get",
        description="""Gespeicherte Meilenstein-Regeln eines Projekts lesen (GET /projects/{projekt}/rules).""",
    )
    async def rules_get(projekt: str) -> Any:
        return await transport("GET", f"{API}/projects/{quote(projekt, safe='')}/rules")

    @mcp.tool(
        name="rules_set",
        description="""Meilenstein-Regeln eines Projekts ersetzen (PUT /projects/{projekt}/rules).

        `regeln`: [{muster (Regex), meilenstein (nicht leer), feld: nachricht|zweig|beide}],
        erster Treffer gewinnt. Die Liste ersetzt die gespeicherte vollständig;
        eine leere Liste löscht alle Regeln.""",
    )
    async def rules_set(projekt: str, regeln: List[Dict[str, Any]]) -> Any:
        return await transport(
            "PUT", f"{API}/projects/{quote(projekt, safe='')}/rules", json_body=list(regeln)
        )
