"""person-api tools: Personen, Kennungen und Zusammenfuehrungen je Mandant (#2441).

Auftrag Alex ueber Business; Vertrag Business/Cloud. Felder nach dem Quellcode der
person-api (apopovic77/person-api, app/routers/persons.py, Stand a337086).

Rechte: drei Stufen, so wie die person-api sie woertlich prueft (ohne Vererbung):
  read  -- lesen (Mandanten, Personen, Aufloesung, Kandidaten, Ausnahmen)
  write -- anlegen und aendern
  admin -- zusammenfuehren, Ausnahmen pflegen, Person ueberspringen
`mcp:person:contact` (Telefon-/Messenger-Kennungen sehen) prueft die person-api
selbst beim Antworten; kein Werkzeug haengt davon ab.

Die Werkzeuge formen nur Anfragen. Der Transport reicht das JWT des aufrufenden
Agenten als Bearer durch und verweigert ohne Aufrufer (server.call_person_api) --
die person-api prueft Rechte und Besitzer-Regel je Aufrufer selbst.
"""

from typing import Any, Awaitable, Callable, Dict, List, Optional
from urllib.parse import quote

Transport = Callable[..., Awaitable[Any]]

API = "/api/v1"
LESEN, SCHREIBEN, VERWALTEN = "read", "write", "admin"

# Werkzeug -> Stufe. server.py meldet das an auth.WERKZEUG_STUFEN["person"].
WERKZEUG_STUFEN: Dict[str, str] = {
    "tenants_list": LESEN, "tenant_get": LESEN, "persons_list": LESEN, "person_get": LESEN,
    "resolve": LESEN, "merge_candidates": LESEN, "skip_rules_list": LESEN,
    "person_create": SCHREIBEN, "person_update": SCHREIBEN, "resolve_or_create": SCHREIBEN,
    "identifier_add": SCHREIBEN,
    "persons_merge": VERWALTEN, "skip_rule_add": VERWALTEN, "skip_rule_delete": VERWALTEN,
    "person_skip": VERWALTEN,
}


def _id(v: Any) -> str:
    return quote(str(v), safe="")


def _drop_none(d: Dict[str, Any]) -> Dict[str, Any]:
    """Keep explicit falsy values (0, False, ""), drop only unset ones."""
    return {k: v for k, v in d.items() if v is not None}


def register_person_tools(mcp, transport: Transport) -> None:
    """Register the person tools on ``mcp``; ``transport(method, path, **kw)``."""

    # ── Lesen (mcp:person:read) ─────────────────────────────────────────────
    @mcp.tool(name="tenants_list",
              description="Mandanten, die du sehen darfst (GET /tenants). Recht: mcp:person:read.")
    async def tenants_list() -> Any:
        return await transport("GET", f"{API}/tenants")

    @mcp.tool(name="tenant_get", description="Einen Mandanten lesen (GET /tenants/{id}). Recht: mcp:person:read.")
    async def tenant_get(tenant_id: str) -> Any:
        return await transport("GET", f"{API}/tenants/{_id(tenant_id)}")

    @mcp.tool(name="persons_list", description="""Personen eines Mandanten (GET /persons).

    `include_owner=false` laesst die Besitzer-Person weg. Telefon- und Messenger-
    Kennungen liefert die person-api nur mit mcp:person:contact. Recht: mcp:person:read.""")
    async def persons_list(tenant_id: str, include_owner: bool = True) -> Any:
        return await transport("GET", f"{API}/persons",
                               params={"tenant_id": tenant_id, "include_owner": str(bool(include_owner)).lower()})

    @mcp.tool(name="person_get", description="Eine Person samt Kennungen (GET /persons/{id}). Recht: mcp:person:read.")
    async def person_get(person_id: str) -> Any:
        return await transport("GET", f"{API}/persons/{_id(person_id)}")

    @mcp.tool(name="resolve", description="""Kennung zu Person aufloesen (GET /resolve).

    `channel` z. B. email, phone, whatsapp; `value` die Kennung. Keine Person -> 404.
    Recht: mcp:person:read.""")
    async def resolve(tenant_id: str, channel: str, value: str) -> Any:
        return await transport("GET", f"{API}/resolve",
                               params={"tenant_id": tenant_id, "channel": channel, "value": value})

    @mcp.tool(name="merge_candidates",
              description="Moegliche Doppelte eines Mandanten (GET /merge-candidates). Recht: mcp:person:read.")
    async def merge_candidates(tenant_id: str) -> Any:
        return await transport("GET", f"{API}/merge-candidates", params={"tenant_id": tenant_id})

    @mcp.tool(name="skip_rules_list",
              description="Ausnahmeregeln eines Mandanten (GET /skip-rules). Recht: mcp:person:read.")
    async def skip_rules_list(tenant_id: str) -> Any:
        return await transport("GET", f"{API}/skip-rules", params={"tenant_id": tenant_id})

    # ── Schreiben (mcp:person:write) ────────────────────────────────────────
    @mcp.tool(name="person_create", description="""Person anlegen (POST /persons).

    Pflicht: `tenant_id`, `canonical_name`. `identifiers`: Liste von
    {channel, value, label?}. Recht: mcp:person:write.""")
    async def person_create(
        tenant_id: str,
        canonical_name: str,
        org: Optional[str] = None,
        role: Optional[str] = None,
        notes: Optional[str] = None,
        is_owner: Optional[bool] = None,
        identifiers: Optional[List[Dict[str, Any]]] = None,
    ) -> Any:
        body = _drop_none({"tenant_id": tenant_id, "canonical_name": canonical_name, "org": org,
                           "role": role, "notes": notes, "is_owner": is_owner,
                           "identifiers": identifiers})
        return await transport("POST", f"{API}/persons", json_body=body)

    @mcp.tool(name="person_update", description="""Person aendern (PUT /persons/{id}).

    Nur gesetzte Felder werden geaendert (canonical_name, org, role, notes).
    Recht: mcp:person:write.""")
    async def person_update(
        person_id: str,
        canonical_name: Optional[str] = None,
        org: Optional[str] = None,
        role: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> Any:
        body = _drop_none({"canonical_name": canonical_name, "org": org, "role": role, "notes": notes})
        return await transport("PUT", f"{API}/persons/{_id(person_id)}", json_body=body)

    @mcp.tool(name="resolve_or_create", description="""Kennung aufloesen oder Person anlegen
    (POST /resolve-or-create). `name_hint`/`org_hint` fuer eine neue Person.
    Recht: mcp:person:write.""")
    async def resolve_or_create(
        tenant_id: str,
        channel: str,
        value: str,
        name_hint: Optional[str] = None,
        org_hint: Optional[str] = None,
    ) -> Any:
        body = _drop_none({"tenant_id": tenant_id, "channel": channel, "value": value,
                           "name_hint": name_hint, "org_hint": org_hint})
        return await transport("POST", f"{API}/resolve-or-create", json_body=body)

    @mcp.tool(name="identifier_add", description="""Kennung an eine Person haengen
    (POST /persons/{id}/identifiers). Recht: mcp:person:write.""")
    async def identifier_add(person_id: str, channel: str, value: str, label: Optional[str] = None) -> Any:
        body = _drop_none({"channel": channel, "value": value, "label": label})
        return await transport("POST", f"{API}/persons/{_id(person_id)}/identifiers", json_body=body)

    # ── Verwalten (mcp:person:admin) ────────────────────────────────────────
    @mcp.tool(name="persons_merge", description="""Person `source_id` in `target_id` zusammenfuehren
    (POST /persons/{target}/merge/{source}). Nicht umkehrbar ueber dieses Werkzeug.
    Recht: mcp:person:admin.""")
    async def persons_merge(target_id: str, source_id: str, reason: Optional[str] = None) -> Any:
        return await transport("POST", f"{API}/persons/{_id(target_id)}/merge/{_id(source_id)}",
                               json_body=_drop_none({"reason": reason}))

    @mcp.tool(name="skip_rule_add", description="""Ausnahmeregel anlegen (POST /skip-rules):
    diese Kennung wird nicht als Person gefuehrt. `kind` Vorgabe identifier.
    Recht: mcp:person:admin.""")
    async def skip_rule_add(
        tenant_id: str,
        value: str,
        kind: Optional[str] = None,
        channel: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> Any:
        body = _drop_none({"tenant_id": tenant_id, "value": value, "kind": kind, "channel": channel,
                           "notes": notes, "added_via": "mcp"})
        return await transport("POST", f"{API}/skip-rules", json_body=body)

    @mcp.tool(name="skip_rule_delete",
              description="Ausnahmeregel loeschen (DELETE /skip-rules/{id}). Recht: mcp:person:admin.")
    async def skip_rule_delete(rule_id: int) -> Any:
        return await transport("DELETE", f"{API}/skip-rules/{int(rule_id)}")

    @mcp.tool(name="person_skip", description="""Person ueberspringen (POST /persons/{id}/skip).
    Recht: mcp:person:admin.""")
    async def person_skip(person_id: str) -> Any:
        return await transport("POST", f"{API}/persons/{_id(person_id)}/skip")
