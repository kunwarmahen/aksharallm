#!/usr/bin/env bash
# Schedule starts and stops -- "train 22:00-06:30 on weeknights", several windows a day.
#
# The same rules the portal's Schedule panel edits (one schedule.json in the repo root), so
# you can add a window in the browser and pause it from here, or the other way round.
#
#   scripts/schedule.sh                                   # rules + when they next fire
#   scripts/schedule.sh window small-code 22:00 06:30 --days mon-fri
#   scripts/schedule.sh window small-code 13:00 17:30 --days sat,sun --steps 2000
#   scripts/schedule.sh add start small-code 09:00        # single rules, if you prefer
#   scripts/schedule.sh add stop  small-code 12:00
#   scripts/schedule.sh pause <id> | resume <id> | rm <id>
#   scripts/schedule.sh on | off                          # master switch
#   scripts/schedule.sh log                               # what it actually did
#
# Something has to watch the clock -- any one of:
#   scripts/portal.sh              the portal runs the scheduler by default
#   scripts/schedule.sh daemon     a foreground loop, no web server
#   * * * * * cd <repo> && scripts/schedule.sh check      (let cron do the ticking)
#
# All three take the same one-per-machine lock, so they never double-fire.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-.venv/bin/python}
exec "$PY" scripts/schedule.py "$@"
