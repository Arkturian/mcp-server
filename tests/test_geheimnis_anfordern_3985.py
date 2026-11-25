"""#3985 (aus #3915): MCP-Werkzeug geheimnis_anfordern gegen POST /api/geheimnisse/anfragen.

Ein Mensch tippt den Wert im Portal ein, kein Agent sieht ihn. Das Werkzeug hat deshalb kein Feld
fuer einen Wert und gibt nur den Stand zurueck (feste Liste von Feldern). Freie Ziele (Pfade) gehen
gar nicht erst hinaus; ueber gueltige, aber unbekannte Ziele und die Berechtigung entscheidet die
cloud-api (422 ziel_unbekannt, 403 nicht_erlaubt), und diese Antworten kommen unveraendert beim Agenten an.

Der Aufruf laeuft durch den echten _fetch_json mit httpx.MockTransport. Adressen erfunden (.invalid).
"""
from __future__ import annotations

import asyncio
import inspect
import json
import os

import httpx
import pytest

os.environ.setdefault("MCP_SERVERS", "cloud")
import server  # noqa: E402

ZIEL = "arkserver:github-dispatch"
FUER = "mensch@arkturian.invalid"


@pytest.fixture
def cloud(monkeypatch):
    gesehen = []
    antwort = {"status": 201, "json": {
        "id": "a1", "zweck": "Release-Anzeige", "art": "github_pat", "ziel": ZIEL, "fuer": FUER,
        "von": "agent:Cloud", "status": "offen", "laeuft_ab": 1.0, "versuche_rest": 5,
        "ziel_text": "Lesezugang", "karte": "um-1", "knoten": "arkserver",
        "wert": "DARF-NIE-HINAUS", "pfad": "/var/lib/cloud-api/geheimnisse/ablage/x"}}

    def handler(request: httpx.Request) -> httpx.Response:
        gesehen.append(request)
        return httpx.Response(antwort["status"], json=antwort["json"])
    echt = httpx.AsyncClient

    def _client(*a, **kw):
        kw["transport"] = httpx.MockTransport(handler)
        return echt(*a, **kw)
    monkeypatch.setattr(server.httpx, "AsyncClient", _client)
    monkeypatch.setattr(server, "CLOUD_API_BASE", "https://cloud.invalid")
    monkeypatch.setattr(server, "_caller_auth_headers", lambda *a, **k: {"Authorization": "Bearer agent-jwt"})
    return gesehen, antwort


def _rufen(**kw):
    args = {"zweck": "Release-Anzeige", "art": "github_pat", "ziel": ZIEL, "fuer": FUER, **kw}
    return asyncio.run(server.cloud_geheimnis_anfordern(**args))


def test_anfrage_geht_mit_dem_ausweis_des_agenten(cloud):
    gesehen, _ = cloud
    out = _rufen(fuer=f"  {FUER} ")
    [r] = gesehen
    assert r.method == "POST" and str(r.url) == "https://cloud.invalid/api/geheimnisse/anfragen"
    assert r.headers["authorization"] == "Bearer agent-jwt"
    assert json.loads(r.content) == {"zweck": "Release-Anzeige", "art": "github_pat", "ziel": ZIEL, "fuer": FUER}
    assert out["id"] == "a1" and out["karte"] == "um-1" and out["status"] == "offen"


def test_antwort_nur_der_stand(cloud):
    out = _rufen()
    assert set(out) <= set(server._GEHEIMNIS_FELDER)
    assert "wert" not in out and "pfad" not in out
    assert "DARF-NIE-HINAUS" not in json.dumps(out)


def test_werkzeug_kennt_keinen_wert():
    assert set(inspect.signature(server.cloud_geheimnis_anfordern).parameters) == {"zweck", "art", "ziel", "fuer"}


@pytest.mark.parametrize("ziel", ["/etc/cloud-api-github-dispatch.token", "../geheim", "arkserver", "",
                                  "arkserver:/etc/x", "arkserver: x", "Arkserver:x", "a:b\nc"])
def test_freies_ziel_geht_nicht_hinaus(cloud, ziel):
    gesehen, _ = cloud
    with pytest.raises(ValueError, match="benanntes Ziel"):
        _rufen(ziel=ziel)
    assert gesehen == []


@pytest.mark.parametrize("ziel", [ZIEL, "comm:env:TELEGRAM_BOT_TOKEN", "comm:imap:office-1", "comm:ms_app"])
def test_benannte_ziele_gehen_hinaus(cloud, ziel):
    _rufen(ziel=ziel)
    assert json.loads(cloud[0][-1].content)["ziel"] == ziel


@pytest.mark.parametrize("status, code", [(422, "ziel_unbekannt"), (422, "art_passt_nicht"),
                                          (403, "nicht_erlaubt"), (409, "anderer_knoten")])
def test_antworten_der_cloud_api_kommen_an(cloud, status, code):
    """Unbekanntes Ziel 422, nicht berechtigter Agent 403: kein Erfolg, der Code steht in der Meldung."""
    _, antwort = cloud
    antwort.update(status=status, json={"detail": {"code": code, "message": "x"}})
    with pytest.raises(RuntimeError) as e:
        _rufen(ziel="arkserver:unbekannt")
    assert f"Upstream {status}" in str(e.value) and code in str(e.value)


def test_registriert_im_cloud_server():
    tools = asyncio.run(server.cloud_mcp.list_tools())
    [t] = [t for t in tools if t.name == "geheimnis_anfordern"]
    assert "never a value" in t.description and "nicht_erlaubt" in t.description
