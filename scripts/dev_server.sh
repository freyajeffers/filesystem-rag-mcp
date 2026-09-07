#!/usr/bin/env bash
set -euo pipefail

# Quick start script for developing and running filesystem-rag-mcp locally

ROOT_DIR="${1:-.}"
TRANSPORT="${2:-stdio}"
PORT="${3:-8000}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

if [ -f "$PROJECT_ROOT/.venv/bin/activate" ]; then
    source "$PROJECT_ROOT/.venv/bin/activate"
fi

echo "Starting filesystem-rag-mcp in dev mode..."
echo "  Root Directory: $ROOT_DIR"
echo "  Transport:      $TRANSPORT"

if [ "$TRANSPORT" = "stdio" ]; then
    python3 -m filesystem_rag_mcp.cli --transport stdio --root-dir "$ROOT_DIR" --log-level DEBUG
else
    echo "  Listening on:   http://127.0.0.1:$PORT"
    python3 -m filesystem_rag_mcp.cli --transport http --no-auth --host 127.0.0.1 --port "$PORT" --root-dir "$ROOT_DIR" --log-level DEBUG
fi
