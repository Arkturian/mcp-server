#!/usr/bin/env bash

# Wrapper for running the Arkturian MCP server. Designed for systemd usage.
set -euo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$BASE_DIR"

: "${ARKTURIAN_API_BASE:=https://api.arkturian.com}"
: "${MCP_HOST:=127.0.0.1}"
: "${MCP_PORT:=8080}"

if [[ -z "${ARKTURIAN_API_KEY:-}" ]]; then
  echo "ARKTURIAN_API_KEY must be set in the environment" >&2
  exit 1
fi

UVICORN_BIN="$BASE_DIR/venv/bin/uvicorn"
if [[ ! -x "$UVICORN_BIN" ]]; then
  echo "Expected uvicorn at $UVICORN_BIN (install deps first)" >&2
  exit 1
fi

# --timeout-keep-alive 120 (#3201): uvicorn closes idle connections after 5 s by
# default, Codex clients (rmcp/reqwest) keep them pooled for up to 90 s. A reuse right
# at the 5-s edge lost the request ("Transport send error"). Same value as the tenant
# gateway unit in cloud-api/.devops/tenant/agentos-tenant-bootstrap.sh.
exec "$UVICORN_BIN" server:app --timeout-keep-alive 120 --host "$MCP_HOST" --port "$MCP_PORT" --proxy-headers --log-level info
