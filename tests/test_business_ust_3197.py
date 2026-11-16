"""#3197: create_invoice und create_honorarnote ueberschrieben den USt-Satz des Mandanten.

Beide hatten `vat_rate: float = 0.0` und schickten den Wert immer mit. business-api nimmt
ohne Wert den Satz des Mandanten. Stand business-api 16.11.2026: Mandanten 2 und 3 mit 20 %
(nicht Kleinunternehmer), 1 und 4 Kleinunternehmer; Dokumente gibt es bisher nur bei 1.

Geprueft ueber den echten Werkzeugaufruf bis zur Anfrage, dann mit einer woertlichen
Abschrift von `_resolve_vat_rate` (business-api app/api/routes/documents.py, main b4893d84)
der Satz, den business-api daraus macht.
"""
import asyncio
import os
from types import SimpleNamespace
from typing import Optional

import pytest

for _k in ("ARKTURIAN_API_KEY", "ONEAL_STORAGE_API_KEY", "JWT_PUBLIC_KEY", "COMM_API_KEY"):
    os.environ.setdefault(_k, "test-only")

import server  # noqa: E402


def _resolve_vat_rate(explicit: Optional[float], tenant) -> float:
    """Abschrift aus business-api (b4893d84)."""
    if explicit is not None:
        return explicit
    return 0.0 if tenant.is_kleinunternehmer else tenant.default_vat_rate


UST_MANDANT = SimpleNamespace(is_kleinunternehmer=False, default_vat_rate=20.0)      # wie 2 und 3
KLEINUNTERNEHMER = SimpleNamespace(is_kleinunternehmer=True, default_vat_rate=0.0)    # wie 1

AUFRUFE = {
    "create_invoice": {"client_id": 1, "items": [{"description": "x", "quantity": 1, "unit_price": 100}]},
    "create_honorarnote": {"client_id": 1, "hours": 2, "rate": 95, "description": "Beratung"},
    "create_angebot": {"client_id": 1, "items": [{"description": "x", "quantity": 1, "unit_price": 100}]},
}


@pytest.fixture
def rumpf(monkeypatch):
    gesehen = []

    async def _fetch(method, url, **kw):
        gesehen.append(kw.get("json_body"))
        return {"id": 1}
    monkeypatch.setattr(server, "_fetch_json", _fetch)

    def rufe(name, **extra):
        asyncio.run(server.business_mcp.call_tool(name, {**AUFRUFE[name], **extra}))
        return gesehen[-1]
    return rufe


@pytest.mark.parametrize("name", list(AUFRUFE))
def test_ohne_angabe_kein_vat_rate_und_keine_vorgabe_im_schema(rumpf, name):
    assert "vat_rate" not in rumpf(name)
    ws = {t.name: t for t in asyncio.run(server.business_mcp.list_tools())}
    feld = ws[name].inputSchema["properties"]["vat_rate"]
    assert feld.get("default") is None and "vat_rate" not in ws[name].inputSchema.get("required", [])


@pytest.mark.parametrize("name", list(AUFRUFE))
def test_ust_mandant_bekommt_seinen_satz(rumpf, name):
    assert _resolve_vat_rate(rumpf(name).get("vat_rate"), UST_MANDANT) == 20.0


@pytest.mark.parametrize("name", list(AUFRUFE))
def test_ausdruecklich_null_bleibt_null(rumpf, name):
    b = rumpf(name, vat_rate=0.0)
    assert b["vat_rate"] == 0.0 and _resolve_vat_rate(b["vat_rate"], UST_MANDANT) == 0.0


@pytest.mark.parametrize("name", list(AUFRUFE))
def test_kleinunternehmer_unveraendert(rumpf, name):
    """Vorher kam 0.0 fest mit, jetzt entscheidet der Mandant: fuer Kleinunternehmer beides 0."""
    assert _resolve_vat_rate(rumpf(name).get("vat_rate"), KLEINUNTERNEHMER) == 0.0
    assert _resolve_vat_rate(0.0, KLEINUNTERNEHMER) == 0.0


@pytest.mark.parametrize("name", list(AUFRUFE))
def test_ausdruecklicher_satz_geht_durch(rumpf, name):
    assert rumpf(name, vat_rate=10.0)["vat_rate"] == 10.0


def test_leitfaden_verspricht_keine_feste_null_mehr():
    leitfaden = asyncio.run(server.business_guide())
    assert "VAT rate: 0% (Kleinunternehmerregelung" not in leitfaden
    assert "Pass vat_rate only to override" in leitfaden
