#!/usr/bin/env bash
# GPU telemetry from a terminal -- the same samples the portal's GPU panel charts.
#
#   scripts/gpu.sh                  # now, plus a 1-hour summary split into training vs idle
#   scripts/gpu.sh --window 6h      # 15m | 1h | 6h | 24h | all
#   scripts/gpu.sh watch            # one line a second, like `nvidia-smi -l 1`
#   scripts/gpu.sh daemon           # record samples without running the portal
#   scripts/gpu.sh raw              # one reading as JSON
#
# History comes from logs/gpu.jsonl, which scripts/portal.sh records by default (every 5s,
# tagged with whether a run was training). Without one of those running there is a "now"
# but no graph.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-.venv/bin/python}
exec "$PY" scripts/gpu.py "$@"
