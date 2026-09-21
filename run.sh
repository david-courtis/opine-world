#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "usage: ./run.sh <game> [extra args...]" >&2
  exit 1
fi
GAME="$1"; shift
cd "$(dirname "$0")"

MODEL="claude-opus-4-8[1m]"

uv run python play.py \
  --game "$GAME" \
  --max-actions 2000 \
  --model "$MODEL" \
  --effort high \
  --agentic-consumer-model "$MODEL" \
  --agentic-consumer-effort high \
  --critique --critique-interval 3 \
  --synthesis-defer-min-moves-after-divergence 5 \
  --synthesis-defer-max-errors 3 \
  --synthesis-defer-min-action-plans-after-divergence 3 \
  --planner-after-levels-completed 1 \
  --frames-only \
  --claude-isolation docker \
  "$@"
