"""#3528 (Business 20.11.2026): Marke je Beleg an den create-Werkzeugen."""
import asyncio
import os

import pytest

for _k in ("ARKTURIAN_API_KEY", "ONEAL_STORAGE_API_KEY", "JWT_PUBLIC_KEY", "COMM_API_KEY"):
    os.environ.setdefault(_k, "test-only")

import server  # noqa: E402


@pytest.fixture
def anfragen(monkeypatch):
    gesehen = []

    async def _fetch(method, url, **kw):
        gesehen.append({"method": method, "url": url, **kw})
        return {"id": 1}
    monkeypatch.setattr(server, "_fetch_json", _fetch)
    monkeypatch.setattr(server, "BUSINESS_API_KEY", "GATEWAY")
    monkeypatch.setattr(server, "BUSINESS_API_BASE", "https://business.test")
    return gesehen


def _rufe(name, argumente):
    return asyncio.run(server.business_mcp.call_tool(name, argumente))


ITEMS = [{"description": "x", "quantity": 1, "unit_price": 1}]


@pytest.mark.parametrize("name,args", [
    ("create_angebot", {"client_id": 1, "items": ITEMS}),
    ("create_invoice", {"client_id": 1, "items": ITEMS}),
    ("create_honorarnote", {"client_id": 1, "hours": 1, "rate": 1, "description": "x"}),
])
def test_brand_wird_durchgereicht_und_sonst_weggelassen(anfragen, name, args):
    _rufe(name, {**args, "brand": "arkturian"})
    _rufe(name, args)
    mit, ohne = (a["json_body"] for a in anfragen)
    assert mit["brand"] == "arkturian"
    assert "brand" not in ohne  # ohne Angabe entscheidet business-api (Spread Your Wings)


def test_brand_steht_im_schema():
    ws = {t.name: t for t in asyncio.run(server.business_mcp.list_tools())}
    for n in ("create_angebot", "create_invoice", "create_honorarnote"):
        assert "brand" in ws[n].inputSchema["properties"], n
        assert "arkturian" in ws[n].description
