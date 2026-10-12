"""#2402: Agentenname mit Nicht-ASCII-Zeichen im Kopf, ueber den echten HTTP-Stapel.

Beigesteuert vom Cloud-Agenten der ELE-Instanz (12.10.2026), ueber die Bruecke
als Text geliefert und hier unveraendert uebernommen.
"""
import httpx, pytest
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route
from agent_identity_header import set_agent_name, get_agent_name, NAME_HEADER, ENCODING_HEADER

NAMES = ["Cloud", "Projekt S1 Abc-Def", "Projekt A1 Ort - Stadt Ost",
         "Kunde-Handbücher", "Büro Ä/Ö%20 +x", "日本語-Agent", "a%2Fb"]

async def who(request):
    return JSONResponse({"name": get_agent_name(request.headers)})
app = Starlette(routes=[Route("/who", who)])

@pytest.mark.parametrize("name", NAMES)
def test_roundtrip_ueber_echten_http_stapel(name):
    h = {}; set_agent_name(h, name)
    httpx.Request("GET", "http://x/who", headers=h)   # warf vorher UnicodeEncodeError
    import anyio
    async def go():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://x") as c:
            return (await c.get("/who", headers=h)).json()["name"]
    assert anyio.run(go) == name

def test_alter_zustand_wirft():
    with pytest.raises(UnicodeEncodeError):
        httpx.Request("GET", "http://x", headers={NAME_HEADER: "Kunde-Handbücher"})

def test_ascii_unveraendert_und_ohne_markierung():
    h = {}; set_agent_name(h, "Projekt S1 Abc-Def")
    assert h == {NAME_HEADER: "Projekt S1 Abc-Def"}

def test_ascii_name_mit_prozent_wird_ohne_markierung_nicht_dekodiert():
    assert get_agent_name({NAME_HEADER: "a%2Fb"}) == "a%2Fb"

def test_kaputte_kodierung_ergibt_leer():
    assert get_agent_name({NAME_HEADER: "%FF%FE", ENCODING_HEADER: "pct"}) == ""

def test_alter_empfaenger_sieht_kodierten_namen_nicht_den_echten():
    h = {}; set_agent_name(h, "Kunde-Handbücher")
    assert h[NAME_HEADER] != "Kunde-Handbücher" and h[NAME_HEADER].isascii()

def test_leerer_name_setzt_nichts():
    h = {}; set_agent_name(h, ""); assert h == {}
