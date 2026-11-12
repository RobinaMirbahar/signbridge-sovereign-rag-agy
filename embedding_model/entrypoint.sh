#!/usr/bin/env bash
set -euo pipefail
ollama serve &
sleep 3
exec uvicorn app:app --host 0.0.0.0 --port "${PORT:-8080}"
