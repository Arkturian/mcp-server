"""#3786 (ELE-Cloud 32f54daf, 23.11.): session_config leert bzw. setzt den Fortsetzen-Faden.

Die Werkzeugfunktion wird aus server.py herausgeloest und ausgefuehrt, mit ersetztem
call_cloud_api (wie test_session_config_parent). Die Rechte (Besitzer, Admin, sessions:manage
im Mandanten) und die Pruefung der Faden-ID (422) entscheidet cloud-api; deren Tests stehen in
cloud-api tests/test_config_pin_3786.py.
"""
import ast
import asyncio
import pathlib
from typing import Any, Dict

QUELLE = pathlib.Path(__file__).resolve().parents[1] / "server.py"


def _werkzeug(antwort):
    baum = ast.parse(QUELLE.read_text())
    fn = next(n for n in baum.body if isinstance(n, ast.AsyncFunctionDef) and n.name == "cloud_session_config")
    fn.decorator_list = []
    rufe = []

    async def call_cloud_api(method, path, json_body=None, **k):
        rufe.append((method, path, json_body))
        return dict(antwort)
    ns = {"Dict": Dict, "Any": Any, "call_cloud_api": call_cloud_api}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])), str(QUELLE), "exec"), ns)
    return ns["cloud_session_config"], rufe


NEU = {"name": "Steward", "action": "updated", "pin_wirkt": "beim_neustart",
       "pinned_resume_session_id": {"alt": "019a-gesperrt", "neu": ""}}


def test_leeren_schickt_leeren_pin_und_tag_und_reicht_alt_neu_durch():
    f, rufe = _werkzeug(NEU)
    r = asyncio.run(f("Steward", pinned_resume_session_id=""))
    assert rufe == [("PATCH", "/api/sessions/Steward/config",
                     {"pinned_resume_session_id": "", "pinned_resume_tag": ""})]
    assert r["pinned_resume_session_id"] == {"alt": "019a-gesperrt", "neu": ""} and "hinweis" not in r


def test_neue_faden_id():
    f, rufe = _werkzeug({**NEU, "pinned_resume_session_id": {"alt": "", "neu": "019b-neu"}})
    asyncio.run(f("Steward", pinned_resume_session_id="019b-neu"))
    assert rufe[0][2] == {"pinned_resume_session_id": "019b-neu", "pinned_resume_tag": ""}


def test_ohne_angabe_kein_pin_im_rumpf():
    f, rufe = _werkzeug({"name": "X", "action": "updated"})
    r = asyncio.run(f("X", model="opus"))
    assert rufe == [("PATCH", "/api/sessions/X/config", {"model": "opus"})] and "hinweis" not in r


def test_aeltere_cloud_api_ohne_alt_neu_bekommt_einen_hinweis():
    f, _ = _werkzeug({"name": "Steward", "action": "updated"})
    r = asyncio.run(f("Steward", pinned_resume_session_id=""))
    assert "cloud-api ab #3786" in r["hinweis"]


def test_fehler_der_cloud_api_kommen_unveraendert_zurueck():
    for fehler in ({"error": "invalid_resume_id"}, {"detail": "forbidden"}):
        f, _ = _werkzeug(fehler)
        assert asyncio.run(f("Steward", pinned_resume_session_id="../x")) == fehler


def test_beschreibung_nennt_das_feld():
    assert "pinned_resume_session_id (str: conversation the agent resumes" in QUELLE.read_text()
