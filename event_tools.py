"""event-api tools: Termine, Anmeldungen, Auswertung und Export je Mandant.

Contract: event-api (Arkturian/event-api), Admin-Endpunkte unter /api/v1/admin.
Owner: agent Event.

Auth: The gateway sends its instance key (EVENT_API_KEY) plus the verified
caller name as X-Agent-Name. The agent JWT never reaches event-api. Which
tenant an agent may see is decided by event-api itself (agent_grants, only on
Alex' word); this server is registered EXPLICIT_ONLY, so wildcard roles do not
unlock it.

Registration is isolated from server.py so the tools can be exercised against
a fake transport and a fake exporter.
"""

from typing import Any, Awaitable, Callable, Dict, List, Optional
from urllib.parse import quote

Transport = Callable[..., Awaitable[Any]]
Exporter = Callable[..., Awaitable[Any]]

API = "/api/v1"
STATUSES = ("requested", "confirmed", "paid", "waitlist", "cancelled")

DATA_RULE = (
    "Anmeldungen sind Personendaten von Kunden. Nie in Beiträge, Issues, IACP "
    "oder Chats an Dritte kopieren. Für Menschen: Export als private Datei."
)


def _drop_none(d: Dict[str, Any]) -> Dict[str, Any]:
    """Keep explicit falsy values (0, False, ""), drop only unset ones."""
    return {k: v for k, v in d.items() if v is not None}


def _t(tenant: str) -> str:
    return quote(tenant.strip(), safe="")


def register_event_tools(mcp, transport: Transport, exporter: Exporter) -> None:
    """Register the event tools on ``mcp``.

    ``transport(method, path, params=?, json_body=?)`` calls event-api and returns JSON.
    ``exporter(tenant, fmt, params, share_with)`` fetches a CSV/XLSX export and stores it
    as a private storage file bound to the caller; returns the storage object.
    """

    @mcp.tool(
        name="event_access",
        description="""Welche Mandanten darfst du in der event-api sehen? (GET /admin/access)

        Wann: zuerst, bevor du einen `tenant` angibst. Liefert `tenants` mit
        `slug`, `name`, `can_write`. Leer heißt: keine Freigabe. Freigaben
        erteilt nur Alex, eingetragen vom Agenten Event.""",
    )
    async def event_access() -> Any:
        return await transport("GET", f"{API}/admin/access")

    @mcp.tool(
        name="events_list",
        description="""Termine eines Mandanten mit Paketen, freien Plätzen und Formularfeldern.

        Öffentliche Sicht (nur veröffentlichte Termine), z. B. tenant="inno4def".
        `offering` filtert nach Format (z. B. "clarity-retreat"). `formFields`
        nennt die kundenspezifischen Felder (key, label, type, options).""",
    )
    async def events_list(tenant: str, offering: Optional[str] = None) -> Any:
        return await transport("GET", f"{API}/public/{_t(tenant)}/events", params=_drop_none({"offering": offering}))

    @mcp.tool(
        name="registrations_list",
        description=f"""Anmeldungen eines Mandanten (GET /admin/{{tenant}}/registrations).

        Filter: `event` (Termin-slug), `status` ({' | '.join(STATUSES)}),
        `since`/`until` (ISO-8601, Anmeldezeitpunkt), `q` (Freitext in Name,
        E-Mail, Nachricht, Antworten). `limit` begrenzt die Antwort (Vorgabe 50,
        höchstens 500); `total` nennt die volle Trefferzahl.
        Antworten der Formularfelder stehen in `answers`. Nur mit Freigabe für
        den Mandanten (sonst 403 agent_not_granted). {DATA_RULE}""",
    )
    async def registrations_list(
        tenant: str,
        event: Optional[str] = None,
        status: Optional[str] = None,
        since: Optional[str] = None,
        until: Optional[str] = None,
        q: Optional[str] = None,
        limit: int = 50,
    ) -> Any:
        if status is not None and status not in STATUSES:
            raise ValueError(f"status muss einer von {', '.join(STATUSES)} sein")
        rows = await transport(
            "GET",
            f"{API}/admin/{_t(tenant)}/registrations",
            params=_drop_none({"event": event, "status": status, "since": since, "until": until, "q": q}),
        )
        limit = max(1, min(int(limit), 500))
        items: List[Any] = rows if isinstance(rows, list) else []
        return {"total": len(items), "items": items[:limit], "truncated": len(items) > limit}

    @mcp.tool(
        name="registration_get",
        description=f"""Eine Anmeldung mit allen Feldern (GET /admin/{{tenant}}/registrations/{{id}}). {DATA_RULE}""",
    )
    async def registration_get(tenant: str, id: int) -> Any:
        return await transport("GET", f"{API}/admin/{_t(tenant)}/registrations/{int(id)}")

    @mcp.tool(
        name="registration_update_status",
        description=f"""Status einer Anmeldung ändern (PATCH /admin/{{tenant}}/registrations/{{id}}).

        `status`: {' | '.join(STATUSES)}. `confirmed` und `paid` belegen einen
        Platz. Braucht Schreibrecht für den Mandanten (sonst 403
        write_not_granted). Jede Änderung wird mit deinem Namen protokolliert.
        Löschen geht über dieses Werkzeug nicht, das bleibt Menschen vorbehalten.""",
    )
    async def registration_update_status(tenant: str, id: int, status: str) -> Any:
        if status not in STATUSES:
            raise ValueError(f"status muss einer von {', '.join(STATUSES)} sein")
        return await transport("PATCH", f"{API}/admin/{_t(tenant)}/registrations/{int(id)}", json_body={"status": status})

    @mcp.tool(
        name="event_stats",
        description="""Auswertung je Termin (GET /admin/{tenant}/stats).

        Je Termin: Kapazität, freie Plätze, Anzahl je Status, Paket, Sprache,
        Newsletter- und Werbe-Einwilligungen sowie Zählung je Auswahlfeld
        (z. B. Rolle, Interesse, Fokus). Keine Personendaten. `event` filtert
        auf einen Termin.""",
    )
    async def event_stats(tenant: str, event: Optional[str] = None) -> Any:
        return await transport("GET", f"{API}/admin/{_t(tenant)}/stats", params=_drop_none({"event": event}))

    @mcp.tool(
        name="registrations_export",
        description=f"""Anmeldungen als Excel (xlsx) oder CSV exportieren, abgelegt als PRIVATE Storage-Datei.

        `format`: "xlsx" (Vorgabe) oder "csv". Filter `event`, `status`.
        Formularfelder werden eigene Spalten, leere Spalten entfallen.
        Die Datei ist an dich gebunden; `share_with` gibt sie zusätzlich
        einzelnen Agenten (Liste von Namen) lesend frei. Antwort: das
        Storage-Objekt (`id`); holen mit `assets_download(id)`. Der Export wird
        in der event-api protokolliert. {DATA_RULE}""",
    )
    async def registrations_export(
        tenant: str,
        format: str = "xlsx",
        event: Optional[str] = None,
        status: Optional[str] = None,
        share_with: Optional[List[str]] = None,
    ) -> Any:
        fmt = format.lower().strip()
        if fmt not in ("xlsx", "csv"):
            raise ValueError('format muss "xlsx" oder "csv" sein')
        if status is not None and status not in STATUSES:
            raise ValueError(f"status muss einer von {', '.join(STATUSES)} sein")
        return await exporter(tenant.strip(), fmt, _drop_none({"event": event, "status": status}), share_with)
