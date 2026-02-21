#!/usr/bin/env bash
# scripts/train.sh — Run the training pipeline
set -euo pipefail

DATA_PATH="${1:?Usage: scripts/train.sh <data-path> [target-column]}"
TARGET="${2:-temperature}"

echo "=== Training atmospheric forecast model ==="
echo "Data:   $DATA_PATH"
echo "Target: $TARGET"
echo ""

python -m atm_forecast.training \
    --data "$DATA_PATH" \
    --target "$TARGET" \
    --log-level INFO
