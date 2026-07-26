#!/usr/bin/env bash
# Start the local training portal: a browser page that starts, stops and graphs a run.
#
# It is a *view* over the same files scripts/phase2.sh and scripts/stop.sh use, and it
# drives those very scripts -- so nothing here is a second way of doing things, and you can
# switch between the browser and the terminal mid-run without either getting confused.
#
#   scripts/portal.sh                 # http://127.0.0.1:8765
#   scripts/portal.sh --port 9000
#   scripts/portal.sh --open          # ...and open a browser
#   scripts/portal.sh --host 0.0.0.0 --allow-remote    # reachable from the LAN (no login!)
#
# Stopping the portal (Ctrl-C) never touches a training run: the trainer is a separate,
# detached process and the portal holds no state of its own.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-.venv/bin/python}
exec "$PY" -m aksharallm.portal "$@"
