"""#3434: business-MCP-Werkzeug documents_mark_sent -> POST /api/v1/documents/{id}/mark-sent.

Duenner Wrapper wie documents_mark_paid: verschickt nichts, verbucht einen Versand, der
ausserhalb stattgefunden hat. Pruefungen (Kanal, Comm-ID, Zeit, Telefonnummern) macht
business-api; ihre Fehler kommen beim Agenten mit Status und Grund an. Kein Netz."""
import asyncio
import json
import os

import httpx
import pytest

for _k in ("ARKTURIAN_API_KEY", "ONEAL_STORAGE_API_KEY", "JWT_PUBLIC_KEY", "COMM_API_KEY"):
    os.environ.setdefault(_k, "test-only")

import server  # noqa: E402


@pytest.fixture
def business(monkeypatch):
    """Attrappe der business-api: haelt Anfragen fest, antwortet wie vorgegeben."""
    gesehen = []
    antwort = {"status": 200, "json": {"id": 38, "status": "sent"}}

    def handler(request: httpx.Request) -> httpx.Response:
        gesehen.append(request)
        return httpx.Response(antwort["status"], json=antwort["json"])

    echt = httpx.AsyncClient

    def _client(*a, **kw):
        kw["transport"] = httpx.MockTransport(handler)
        return echt(*a, **kw)
    monkeypatch.setattr(server.httpx, "AsyncClient", _client)
    monkeypatch.setattr(server, "BUSINESS_API_BASE", "https://business.invalid")
    monkeypatch.setattr(server, "BUSINESS_API_KEY", "gateway-schluessel")
    return gesehen, antwort


def _rufen(**kw):
    return asyncio.run(server.business_documents_mark_sent(**kw))


def test_whatsapp_mit_comm_id(business):
    gesehen, _ = business
    erg = _rufen(doc_id=38, channel="whatsapp", comm_message_id="wamid.X1", sent_at="2026-11-19T14:15:00+01:00",
                 recipient_label="Praxis")
    assert erg == {"id": 38, "status": "sent"}
    [r] = gesehen
    assert r.method == "POST" and str(r.url) == "https://business.invalid/api/v1/documents/38/mark-sent"
    assert json.loads(r.content) == {"channel": "whatsapp", "comm_message_id": "wamid.X1",
                                     "sent_at": "2026-11-19T14:15:00+01:00", "recipient_label": "Praxis"}
    assert r.headers["X-API-Key"] == "gateway-schluessel"


def test_nur_gesetzte_felder(business):
    gesehen, _ = business
    _rufen(doc_id=36, channel="persoenlich")
    assert json.loads(gesehen[0].content) == {"channel": "persoenlich"}


@pytest.mark.parametrize("status,grund", [
    (409, "not_a_draft"),
    (422, "comm_id_required"),
    (422, "bad_channel"),
    (422, "sent_at_in_future"),
    (422, "no_phone_numbers"),
])
def test_fehler_der_business_api_kommen_an(business, status, grund):
    _, antwort = business
    antwort.update(status=status, json={"detail": grund})
    with pytest.raises(RuntimeError) as e:
        _rufen(doc_id=38, channel="telegram")
    assert f"Upstream {status}" in str(e.value) and grund in str(e.value)


def test_beschreibung_sagt_verschickt_nichts():
    tools = asyncio.run(server.business_mcp.list_tools())
    [t] = [t for t in tools if t.name == "documents_mark_sent"]
    assert "SENDS NOTHING" in t.description and "NEVER a phone number" in t.description
    props = t.inputSchema["properties"]
    assert set(props) == {"doc_id", "channel", "comm_message_id", "sent_at", "recipient_label"}
    assert set(t.inputSchema.get("required") or []) == {"doc_id", "channel"}
