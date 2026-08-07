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
