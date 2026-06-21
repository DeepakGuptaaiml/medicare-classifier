#!/usr/bin/env bash
# Run Medicare Classifier API locally — classifier only (/predict, /health).
# RAG (/ask) runs on Azure only (Python 3.12 + HF_API_TOKEN in Container App).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "==> Project: $ROOT"
echo "==> Local mode: classifier only. Use Azure for Policy Q&A (/ask)."

PYTHON="python3"
if ! command -v "$PYTHON" >/dev/null 2>&1; then
  echo "ERROR: python3 not found"
  exit 1
fi

echo "==> Using: $($PYTHON --version)"

if [[ ! -d .venv ]]; then
  echo "==> Creating .venv..."
  "$PYTHON" -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate

echo "==> Installing classifier dependencies..."
pip install -q -r requirements-api-local.txt

echo "==> Starting API on http://127.0.0.1:8000"
echo "    Swagger: http://127.0.0.1:8000/docs"
echo "    RAG /ask: use Azure API (not configured locally)"
exec uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
