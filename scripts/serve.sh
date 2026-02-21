#!/usr/bin/env bash
# scripts/serve.sh — Start the API server
set -euo pipefail

HOST="${ATM_API_HOST:-0.0.0.0}"
PORT="${ATM_API_PORT:-8000}"
WORKERS="${ATM_API_WORKERS:-1}"

echo "=== Starting atm-forecast API ==="
echo "Host:    $HOST"
echo "Port:    $PORT"
echo "Workers: $WORKERS"
echo ""

uvicorn atm_forecast.api.main:app \
    --host "$HOST" \
    --port "$PORT" \
    --workers "$WORKERS"
