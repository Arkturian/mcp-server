"""#3201: start.sh startet uvicorn mit --timeout-keep-alive 120 (Vorgabe waere 5 s,
Codex-Clients halten ruhende Verbindungen bis 90 s). start.sh wird AUSGEFUEHRT, mit
einer uvicorn-Attrappe, die nur ihre Argumente aufschreibt; kein Server, kein Netz.
"""
import pathlib
import shutil
import subprocess

START = pathlib.Path(__file__).resolve().parents[1] / "start.sh"


def _argv(tmp_path):
    shutil.copy(START, tmp_path / "start.sh")
    bin_ = tmp_path / "venv" / "bin"
    bin_.mkdir(parents=True)
    log = tmp_path / "argv.txt"
    (bin_ / "uvicorn").write_text(f"#!/bin/sh\nfor a in \"$@\"; do echo \"$a\"; done > '{log}'\n")
    (bin_ / "uvicorn").chmod(0o755)
    r = subprocess.run(["bash", str(tmp_path / "start.sh")], capture_output=True, text=True, timeout=20,
                       env={"PATH": "/usr/bin:/bin", "HOME": str(tmp_path), "ARKTURIAN_API_KEY": "test-attrappe"})
    assert r.returncode == 0, r.stderr
    return log.read_text().splitlines()


def test_start_sh_setzt_keepalive_ueber_dem_client_leerlauf(tmp_path):
    a = _argv(tmp_path)
    assert a[0] == "server:app"
    i = a.index("--timeout-keep-alive")
    assert int(a[i + 1]) == 120 and int(a[i + 1]) > 90
    # der Rest wie bisher
    assert a[a.index("--host") + 1] == "127.0.0.1" and a[a.index("--port") + 1] == "8080"
    assert "--proxy-headers" in a and a[a.index("--log-level") + 1] == "info"
