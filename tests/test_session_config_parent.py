"""session_config reicht `parent` an cloud-api durch (AgentOS auf O'Neal, 13.10.2026).

Der Server verlangt beim Import mehrere Dienst-Schluessel. Deshalb wird die
Werkzeugfunktion aus server.py herausgeloest und AUSGEFUEHRT (nicht nur gelesen),
mit ersetztem call_cloud_api.
"""
import ast
import asyncio
import pathlib
from typing import Any, Dict

QUELLE = pathlib.Path(__file__).resolve().parents[1] / "server.py"


def _werkzeug():
    baum = ast.parse(QUELLE.read_text())
    fn = next(n for n in baum.body if isinstance(n, ast.AsyncFunctionDef) and n.name == "cloud_session_config")
    fn.decorator_list = []
    rufe = []

    async def call_cloud_api(method, path, json_body=None, **k):
        rufe.append((method, path, json_body))
        return {"ok": True}
    ns = {"Dict": Dict, "Any": Any, "call_cloud_api": call_cloud_api}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])), str(QUELLE), "exec"), ns)
    return ns["cloud_session_config"], rufe


def test_parent_wird_gesetzt_und_geleert():
    f, rufe = _werkzeug()
    asyncio.run(f("BIUS-IT", parent="BIUS"))
    asyncio.run(f("BIUS-IT", parent=""))
    assert rufe == [("PATCH", "/api/sessions/BIUS-IT/config", {"parent": "BIUS"}),
                    ("PATCH", "/api/sessions/BIUS-IT/config", {"parent": ""})]


def test_ohne_parent_bleibt_der_rumpf_wie_bisher():
    f, rufe = _werkzeug()
    asyncio.run(f("X", model="opus"))
    assert rufe == [("PATCH", "/api/sessions/X/config", {"model": "opus"})]


def test_beschreibung_nennt_parent():
    assert 'parent (str: name of the parent agent' in QUELLE.read_text()


def _sammel():
    baum = ast.parse(QUELLE.read_text())
    fn = next(n for n in baum.body if isinstance(n, ast.AsyncFunctionDef) and n.name == "cloud_session_config_bulk")
    fn.decorator_list = []
    rufe = []

    async def call_cloud_api(method, path, json_body=None, **k):
        rufe.append((method, path, json_body))
        return {"state": "freigabe_angefragt", "interaction_id": "ix-1"}   # 202 kommt als Rumpf durch
    from typing import List, Optional
    ns = {"Dict": Dict, "Any": Any, "List": List, "Optional": Optional, "call_cloud_api": call_cloud_api}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])), str(QUELLE), "exec"), ns)
    return ns["cloud_session_config_bulk"], rufe


def test_sammel_mit_namen_und_mit_filter():
    f, rufe = _sammel()
    r = asyncio.run(f(agents=["A", "B"], effort="xhigh"))
    asyncio.run(f(model_filter="sonnet", model="opus"))
    assert rufe == [("POST", "/api/konfig-sammel", {"agenten": ["A", "B"], "effort": "xhigh"}),
                    ("POST", "/api/konfig-sammel", {"filter": {"model": "sonnet"}, "model": "opus"})]
    assert r["state"] == "freigabe_angefragt"
