#!/usr/bin/env bash
# Run OPINE-World on one ARC-AGI-3 game with the exact configuration used for the
# paper: Claude Opus 4.8 for both the action and synthesis agents, the critic, the
# deferred-CEGIS gate, the planner settings, the spriteless (frames-only) regime,
# and the Docker filtered-network sandbox. Any flag can be overridden by appending
# it (argparse takes the last value); e.g. `--claude-isolation bwrap`.
#
# Docker isolation needs the gateway up first: docker/gateway_up.sh
#
# Usage:
#   ./run.sh <game> [extra play.py args...]
#   ./run.sh ar25
#   ./run.sh m0r0 --max-actions 3000
#   ./run.sh ft09 --claude-isolation bwrap
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
