"""#3194 (Business 16.11.2026): Im business-MCP fehlten die Angebote.

business-api hat POST /api/v1/documents/angebot und /{id}/accept|decline|convert
(seit 006_angebot_support, auf arkturian ausgerollt), das MCP bot nur create_honorarnote
und create_invoice. Agenten erreichen business-api nur ueber das MCP (#2986).

Geprueft wird ueber den echten Werkzeugaufruf (FastMCP call_tool) bis zur Anfrage an
business-api. Der Rumpf wird gegen eine Abschrift der Schemas aus business-api geprueft
(app/api/schemas/document.py, main b4893d84), streng: unbekannte Felder fallen auf.
"""
import asyncio
import json
import os
from datetime import date
from typing import List, Optional

import pytest
from pydantic import BaseModel, ConfigDict, ValidationError

for _k in ("ARKTURIAN_API_KEY", "ONEAL_STORAGE_API_KEY", "JWT_PUBLIC_KEY", "COMM_API_KEY"):
    os.environ.setdefault(_k, "test-only")

import server  # noqa: E402


# --- Abschrift aus business-api (b4893d84), extra="forbid" statt der Vorgabe "ignore" ---
class DocumentItemCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    description: str
    quantity: float = 1.0
    unit_price: float = 0.0
    vat_rate: Optional[float] = None


class AngebotCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    client_id: int
    items: List[DocumentItemCreate]
    vat_rate: Optional[float] = None
    issued_date: Optional[date] = None
    valid_days: int = 30
    notes: Optional[str] = None
    content_post_id: Optional[int] = None


class ConvertAngebotRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    target_type: str = "invoice"
    due_days: int = 14


@pytest.fixture
def anfragen(monkeypatch):
    gesehen = []

    async def _fetch(method, url, **kw):
        gesehen.append({"method": method, "url": url, **kw})
        return {"id": 77, "doc_type": "angebot", "status": "draft"}
    monkeypatch.setattr(server, "_fetch_json", _fetch)
    monkeypatch.setattr(server, "BUSINESS_API_KEY", "GATEWAY")
    monkeypatch.setattr(server, "BUSINESS_API_BASE", "https://business.test")
    return gesehen


def _rufe(name, argumente):
    return asyncio.run(server.business_mcp.call_tool(name, argumente))


def _werkzeuge():
    return {t.name: t for t in asyncio.run(server.business_mcp.list_tools())}


def test_werkzeuge_sind_da_ohne_api_key():
    ws = _werkzeuge()
    for n in ("create_angebot", "angebot_accept", "angebot_decline", "angebot_convert"):
        assert n in ws, n
        assert "api_key" not in (ws[n].inputSchema or {}).get("properties", {})
    assert ws["create_angebot"].inputSchema["required"] == ["client_id", "items"]


def test_create_angebot_wie_business_es_braucht(anfragen):
    """Der Fall aus dem Auftrag: Kunde 31, zwei Posten, 30 Tage gueltig."""
    _rufe("create_angebot", {
        "client_id": 31, "valid_days": 30, "issued_date": "2026-11-16", "notes": "Gilt ab Auftrag",
        "items": [{"description": "Konzept", "quantity": 4, "unit_price": 95.0},
                  {"description": "Umsetzung", "quantity": 12, "unit_price": 95.0}],
    })
    [a] = anfragen
    assert (a["method"], a["url"]) == ("POST", "https://business.test/api/v1/documents/angebot")
    assert a["headers"] == {"X-API-Key": "GATEWAY"}
    rumpf = AngebotCreate.model_validate(a["json_body"])        # wirft bei falschem Feld
    assert rumpf.client_id == 31 and rumpf.valid_days == 30 and rumpf.issued_date == date(2026, 11, 16)
    assert [i.description for i in rumpf.items] == ["Konzept", "Umsetzung"]


def test_vat_rate_nur_wenn_gesetzt(anfragen):
    """Ohne vat_rate waehlt business-api den Satz des Mandanten (_resolve_vat_rate);
    ein fest mitgeschicktes 0.0 wuerde ihn ueberschreiben. 0.0 ausdruecklich geht durch."""
    _rufe("create_angebot", {"client_id": 1, "items": [{"description": "x", "quantity": 1, "unit_price": 1}]})
    _rufe("create_angebot", {"client_id": 1, "vat_rate": 0.0,
                             "items": [{"description": "x", "quantity": 1, "unit_price": 1}]})
    _rufe("create_angebot", {"client_id": 1, "vat_rate": 20.0, "content_post_id": 6200,
                             "items": [{"description": "x", "quantity": 1, "unit_price": 1}]})
    ohne, null, zwanzig = (a["json_body"] for a in anfragen)
    assert "vat_rate" not in ohne and ohne["valid_days"] == 30 and "issued_date" not in ohne
    assert null["vat_rate"] == 0.0
    assert zwanzig["vat_rate"] == 20.0 and zwanzig["content_post_id"] == 6200
    for r in (ohne, null, zwanzig):
        AngebotCreate.model_validate(r)


@pytest.mark.parametrize("name,pfad", [("angebot_accept", "accept"), ("angebot_decline", "decline")])
def test_annehmen_ablehnen(anfragen, name, pfad):
    _rufe(name, {"doc_id": 77})
    [a] = anfragen
    assert (a["method"], a["url"]) == ("POST", f"https://business.test/api/v1/documents/77/{pfad}")
    assert a.get("json_body") is None


def test_umwandeln(anfragen):
    _rufe("angebot_convert", {"doc_id": 77})
    _rufe("angebot_convert", {"doc_id": 77, "target_type": "honorarnote", "due_days": 30})
    vorgabe, hn = anfragen
    assert vorgabe["url"] == "https://business.test/api/v1/documents/77/convert"
    assert ConvertAngebotRequest.model_validate(vorgabe["json_body"]).model_dump() == \
        {"target_type": "invoice", "due_days": 14}
    assert hn["json_body"] == {"target_type": "honorarnote", "due_days": 30}


def test_antwort_von_business_api_kommt_durch(anfragen):
    """Was business-api liefert (das Dokument samt valid_until), kommt unveraendert beim
    Agenten an: als Text und als strukturiertes Ergebnis."""
    inhalt, strukturiert = _rufe("create_angebot", {"client_id": 31, "items": [
        {"description": "x", "quantity": 1, "unit_price": 1}]})
    erwartet = {"id": 77, "doc_type": "angebot", "status": "draft"}
    assert json.loads(inhalt[0].text) == erwartet and strukturiert == {"result": erwartet}


def test_leitfaden_nennt_die_angebote():
    leitfaden = asyncio.run(server.business_guide())
    for wort in ("create_angebot", "angebot_accept", "angebot_convert", "AN-YYYY-NNN", "valid_until"):
        assert wort in leitfaden, wort


def test_abschrift_trifft_falsche_felder():
    """Die strenge Abschrift muss ein falsches Feld wirklich ablehnen, sonst prueft sie nichts."""
    with pytest.raises(ValidationError):
        AngebotCreate.model_validate({"client_id": 1, "items": [], "due_days": 14})
