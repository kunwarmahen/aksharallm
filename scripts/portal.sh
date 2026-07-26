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
#   scripts/portal.sh --lan           # reachable from other machines on your network;
#                                     # it prints the address to type on the phone/laptop.
#
# --lan has no login: anyone who can reach the address can start and stop training. That is
# usually what you want on a home LAN, and never what you want on the open internet.
#
# Stopping the portal (Ctrl-C) never touches a training run: the trainer is a separate,
# detached process and the portal holds no state of its own.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-.venv/bin/python}
exec "$PY" -m aksharallm.portal "$@"
