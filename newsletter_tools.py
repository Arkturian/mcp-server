"""newsletter-api tools: templates, campaign drafts, preview, test sends.

Owner: Newsletter agent (newsletter-api, Stufe 1 of Post #5883).

The caller's agent JWT is forwarded. newsletter-api admits an agent only for
the tenants Alex released to it (``agent_tenant_access``) and never lets an
agent send for real, approve imports, fill the test list or read contacts.
Those steps stay with a logged-in human on newsletter.arkturian.com.
"""

from typing import Any, Awaitable, Callable, Dict, Optional
from urllib.parse import quote

Transport = Callable[..., Awaitable[Any]]

API = "/api/v1/admin"

LIMITS = (
    "Agenten bearbeiten nur Vorlagen und Entwürfe ihrer freigegebenen Mandanten. "
    "Echter Versand, Import-Freigabe, Testliste und Kontaktdaten bleiben beim "
    "Menschen (newsletter.arkturian.com); die API antwortet dort mit 403."
)

TEMPLATE_RULES = (
    "Jinja mit StrictUndefined und Autoescape: optionale Felder über `.get()` "
    "lesen, `|safe` nur auf HTML-Felder aus dem Kontext, nie auf `subscriber.*`. "
    "Text- und HTML-Fassung müssen `{{ unsubscribe_url }}` enthalten. Verfügbar "
    "sind der Kampagnen-Kontext, `subscriber.email|first_name|language`, "
    "`unsubscribe_url` und `website_url`."
)


def _seg(value: str) -> str:
    return quote(str(value), safe="")


def _drop_none(d: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in d.items() if v is not None}


def register_newsletter_tools(mcp, transport: Transport) -> None:
    """Register the newsletter tools on ``mcp``; ``transport(method, path, **kw)``."""

    @mcp.tool(
        name="tenants_list",
        description=f"""Mandanten, die du bearbeiten darfst, mit Listen und Vorlagen.

        {LIMITS}""",
    )
    async def tenants_list() -> Any:
        return await transport("GET", f"{API}/tenants")

    @mcp.tool(
        name="templates_list",
        description="Vorlagen eines Mandanten (slug, name, updated_at).",
    )
    async def templates_list(tenant: str) -> Any:
        return await transport("GET", f"{API}/tenants/{_seg(tenant)}/templates")

    @mcp.tool(
        name="template_get",
        description="Eine Vorlage mit Betreff-, Text- und HTML-Quelltext.",
    )
    async def template_get(tenant: str, slug: str) -> Any:
        return await transport(
            "GET", f"{API}/tenants/{_seg(tenant)}/templates/{_seg(slug)}"
        )

    @mcp.tool(
        name="templates_upsert",
        description=f"""Vorlage anlegen oder ersetzen (PUT, Schlüssel = slug).

        {TEMPLATE_RULES}
        Syntaxfehler → 422. Wird die Vorlage gerade von einer Kampagne im
        Versand benutzt → 409. Antwort enthält `created` (neu oder ersetzt).""",
    )
    async def templates_upsert(
        tenant: str,
        slug: str,
        name: str,
        subject_template: str,
        text_template: str,
        html_template: str,
    ) -> Any:
        return await transport(
            "PUT",
            f"{API}/tenants/{_seg(tenant)}/templates/{_seg(slug)}",
            json_body={
                "name": name,
                "subject_template": subject_template,
                "text_template": text_template,
                "html_template": html_template,
            },
        )

    @mcp.tool(
        name="template_preview",
        description="""Vorlage mit einem freien Kontext rendern, ohne etwas zu speichern.

        `first_name` leer ("") oder null prüft die neutrale Anrede.
        Liefert {subject, text, html} genau wie beim Versand. Fehlt ein Feld,
        das die Vorlage braucht → 422 mit dem Namen.""",
    )
    async def template_preview(
        tenant: str,
        slug: str,
        context: Dict[str, Any],
        first_name: Optional[str] = "",
    ) -> Any:
        return await transport(
            "POST",
            f"{API}/tenants/{_seg(tenant)}/templates/{_seg(slug)}/preview",
            json_body={"context": context, "first_name": first_name},
        )

    @mcp.tool(
        name="campaigns_list",
        description="""Letzte 20 Kampagnen eines Mandanten mit Status, dazu Zähler.

        Status: draft, on_hold (gegen Versand gesperrt), queued, sending, sent,
        completed_with_errors. Kontakte selbst siehst du nicht, nur Zahlen.""",
    )
    async def campaigns_list(tenant: str) -> Any:
        return await transport("GET", f"{API}/tenants/{_seg(tenant)}/overview")

    @mcp.tool(
        name="campaign_get",
        description="Eine Kampagne mit Kontext, Vorlage und Zustellungen je Status.",
    )
    async def campaign_get(campaign_id: str) -> Any:
        return await transport("GET", f"{API}/campaigns/{_seg(campaign_id)}")

    @mcp.tool(
        name="campaign_create",
        description="""Kampagne als Entwurf anlegen. Es wird nichts versendet.

        `slug` klein, Ziffern und Bindestriche, je Liste eindeutig (sonst 409).
        `context` liefert die Felder der Vorlage.""",
    )
    async def campaign_create(
        tenant: str,
        list_slug: str,
        template_slug: str,
        slug: str,
        name: str,
        context: Dict[str, Any],
    ) -> Any:
        return await transport(
            "POST",
            f"{API}/tenants/{_seg(tenant)}/campaigns",
            json_body={
                "list_slug": list_slug,
                "template_slug": template_slug,
                "slug": slug,
                "name": name,
                "context": context,
            },
        )

    @mcp.tool(
        name="campaign_update",
        description="""Entwurf ändern (Name, Kontext, Vorlage). Nur draft/on_hold, sonst 409.

        `context` ersetzt den ganzen Kontext, er wird nicht gemischt.""",
    )
    async def campaign_update(
        campaign_id: str,
        name: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        template_slug: Optional[str] = None,
    ) -> Any:
        return await transport(
            "PATCH",
            f"{API}/campaigns/{_seg(campaign_id)}",
            json_body=_drop_none(
                {"name": name, "context": context, "template_slug": template_slug}
            ),
        )

    @mcp.tool(
        name="campaign_duplicate",
        description="Kampagne als neuen Entwurf kopieren (Vorlage und Kontext).",
    )
    async def campaign_duplicate(
        campaign_id: str, slug: str, name: Optional[str] = None
    ) -> Any:
        return await transport(
            "POST",
            f"{API}/campaigns/{_seg(campaign_id)}/duplicate",
            json_body=_drop_none({"slug": slug, "name": name}),
        )

    @mcp.tool(
        name="campaign_delete",
        description="Entwurf löschen. Nur draft/on_hold ohne Zustellungen, sonst 409.",
    )
    async def campaign_delete(campaign_id: str) -> Any:
        return await transport("DELETE", f"{API}/campaigns/{_seg(campaign_id)}")

    @mcp.tool(
        name="campaign_hold",
        description="""Entwurf gegen Versand sperren (Status on_hold).

        Aufheben kann nur ein angemeldeter Mensch auf newsletter.arkturian.com.""",
    )
    async def campaign_hold(campaign_id: str) -> Any:
        return await transport("POST", f"{API}/campaigns/{_seg(campaign_id)}/hold")

    @mcp.tool(
        name="campaign_preview",
        description="""Kampagne so rendern, wie sie beim Versand rausginge.

        `first_name` "" oder null = neutrale Anrede. `context` wird nur für diese
        Vorschau über den gespeicherten gelegt. Liefert {subject, text, html}.""",
    )
    async def campaign_preview(
        campaign_id: str,
        first_name: Optional[str] = "",
        context: Optional[Dict[str, Any]] = None,
    ) -> Any:
        return await transport(
            "POST",
            f"{API}/campaigns/{_seg(campaign_id)}/preview",
            json_body=_drop_none({"context": context}) | {"first_name": first_name},
        )

    @mcp.tool(
        name="campaign_test_send",
        description="""Testmail an die Testliste des Mandanten, Betreff mit „[Test]“.

        Die Testliste pflegt nur ein Mensch; ist sie leer → 409. Echte Kontakte
        bekommen dabei nichts. Antwort: sent, failed und Ergebnis je Adresse.""",
    )
    async def campaign_test_send(campaign_id: str) -> Any:
        return await transport(
            "POST", f"{API}/campaigns/{_seg(campaign_id)}/test-send", json_body={}
        )

    @mcp.tool(
        name="mail_usage",
        description="""Versendete Mails je Monat (Wien) und Art: campaign, confirmation, test.

        `since` inklusive, `until` exklusiv, ISO-8601.""",
    )
    async def mail_usage(
        tenant: str, since: Optional[str] = None, until: Optional[str] = None
    ) -> Any:
        return await transport(
            "GET",
            f"{API}/tenants/{_seg(tenant)}/mail-usage",
            params=_drop_none({"since": since, "until": until}) or None,
        )
