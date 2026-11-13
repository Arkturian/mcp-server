"""Regression contract for explicit Codex model and reasoning selection."""

import ast
from pathlib import Path


SOURCE = Path(__file__).resolve().parents[1] / "server.py"
TREE = ast.parse(SOURCE.read_text(encoding="utf-8"))


def _async_function(name: str) -> ast.AsyncFunctionDef:
    return next(
        node
        for node in TREE.body
        if isinstance(node, ast.AsyncFunctionDef) and node.name == name
    )


def test_chatgpt_tool_exposes_model_and_effort_separately():
    function = _async_function("ai_chatgpt_text")
    argument_names = [argument.arg for argument in function.args.args]
    assert "model" in argument_names
    assert "effort" in argument_names

    source = ast.get_source_segment(SOURCE.read_text(encoding="utf-8"), function)
    assert 'body["effort"] = normalized_effort' in source
    assert 'params = {"model": model.strip()}' in source
    assert 'body["model"]' not in source
    assert 'params=params, json_body=body' in source


def test_ai_transport_forwards_query_parameters():
    function = _async_function("call_ai_api")
    argument_names = [argument.arg for argument in function.args.kwonlyargs]
    assert "params" in argument_names

    fetch_call = next(
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Await)
        and isinstance(node.value, ast.Call)
        and getattr(node.value.func, "id", None) == "_fetch_json"
    ).value
    keywords = {keyword.arg: keyword.value for keyword in fetch_call.keywords}
    assert isinstance(keywords["params"], ast.Name)
    assert keywords["params"].id == "params"


def test_chatgpt_tool_rejects_unknown_reasoning_effort():
    function = _async_function("ai_chatgpt_text")
    source = ast.get_source_segment(SOURCE.read_text(encoding="utf-8"), function)
    for effort in ("minimal", "low", "medium", "high", "xhigh"):
        assert f'"{effort}"' in source
    assert "raise ValueError" in source


def _chatgpt_mit_attrappe():
    """Fuehrt ai_chatgpt_text mit einer Attrappe fuer call_ai_api aus."""
    import asyncio
    from typing import Any, Dict, Optional

    gesendet = []

    async def call_ai_api(method, path, params=None, json_body=None):
        gesendet.append({"path": path, "params": params, "body": json_body})
        return {"ok": True}

    function = _async_function("ai_chatgpt_text")
    function.decorator_list = []
    modul = ast.Module(body=[function], type_ignores=[])
    ns = {"call_ai_api": call_ai_api, "Optional": Optional, "Dict": Dict, "Any": Any}
    exec(compile(modul, "server.py", "exec"), ns)
    return lambda **kw: (asyncio.run(ns["ai_chatgpt_text"](**kw)), gesendet[-1])[1]


def test_tools_false_geht_im_rumpf_an_ai_chatgpt():
    aufruf = _chatgpt_mit_attrappe()
    gesendet = aufruf(prompt="x", effort="low", tools=False, model="gpt-5.6-luna")
    assert gesendet["path"] == "/ai/chatgpt"
    assert gesendet["body"]["tools"] is False
    assert gesendet["params"] == {"model": "gpt-5.6-luna"}


def test_ohne_tools_bleibt_der_rumpf_wie_bisher():
    aufruf = _chatgpt_mit_attrappe()
    assert "tools" not in aufruf(prompt="x")["body"]
    assert aufruf(prompt="x", tools=True)["body"]["tools"] is True


def test_request_id_geht_nur_wenn_gesetzt_in_den_rumpf():
    """Cloud cd67b9a0: cloud-api gibt beim Vorlesen seine lauf_id als request_id mit."""
    aufruf = _chatgpt_mit_attrappe()
    gesendet = aufruf(prompt="x", effort="low", tools=False, request_id="3f9a1c07")
    assert gesendet["path"] == "/ai/chatgpt" and gesendet["body"]["request_id"] == "3f9a1c07"
    assert gesendet["params"] is None                      # kein Abfrageparameter


def test_ohne_request_id_bleibt_alles_wie_bisher():
    aufruf = _chatgpt_mit_attrappe()
    for kw in ({}, {"request_id": None}, {"request_id": ""}):
        gesendet = aufruf(prompt="x", system="s", max_tokens=200, temperature=0.2, effort="low", tools=False,
                          model="gpt-5.6-luna", **kw)
        assert gesendet == {"path": "/ai/chatgpt", "params": {"model": "gpt-5.6-luna"},
                            "body": {"prompt": "x", "system": "s", "max_tokens": 200, "temperature": 0.2,
                                     "effort": "low", "tools": False}}
