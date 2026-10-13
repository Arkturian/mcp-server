"""Das Gateway ist ohne ausdrueckliche Gegenanweisung fail-closed (#24).

David/Steward 28.08.: ein anonymer JSON-RPC `tools/call` auf sections_patch
lieferte HTTP 200 und schrieb wirklich — AUTH_REQUIRE_JWT stand per Default
auf "false", und der Content-Helfer faellt ohne JWT auf den Service-Key
zurueck. Der Default gehoert auf "true".
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("value, expected", [(None, "True"), ("false", "False"), ("true", "True")])
def test_require_jwt_at_startup(value, expected):
    # Exercise import-time configuration in a fresh interpreter. Replacing
    # sys.modules["auth"] in the test runner splits its caller ContextVars from
    # references held by already-collected middleware/tool tests (#2454).
    env = os.environ.copy()
    if value is None:
        env.pop("AUTH_REQUIRE_JWT", None)
    else:
        env["AUTH_REQUIRE_JWT"] = value
    result = subprocess.run(
        [sys.executable, "-c", "import auth; print(auth.AUTH_REQUIRE_JWT)"],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        capture_output=True,
        text=True,
        check=True,
        timeout=15,
    )
    assert result.stdout.strip() == expected
