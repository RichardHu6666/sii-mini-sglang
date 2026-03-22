#!/usr/bin/env bash
set -euo pipefail

# Run EP A/B benchmark in background so it survives terminal close.
# Usage example:
#   bash benchmark/online/run_compare_ep_ab.sh \
#     --model /path/to/Qwen3-30B --tp 4 --num-prompts 64 --input-len 512 --output-len 128

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

OUT_DIR="benchmark/online/results"
mkdir -p "$OUT_DIR"
TS="$(date +%Y%m%d-%H%M%S)"
LOG_FILE="$OUT_DIR/ep_ab_runner_${TS}.log"
PID_FILE="$OUT_DIR/ep_ab_runner_${TS}.pid"

nohup python benchmark/online/compare_ep_ab.py "$@" >"$LOG_FILE" 2>&1 &
PID=$!
echo "$PID" >"$PID_FILE"

echo "Started background benchmark process"
echo "PID: $PID"
echo "Log: $LOG_FILE"
echo "PID file: $PID_FILE"
echo "Tail logs with: tail -f $LOG_FILE"
