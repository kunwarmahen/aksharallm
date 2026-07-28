#!/usr/bin/env bash
# Start the local training portal: a browser page that starts, stops, schedules and graphs
# a run.
#
# It is a *view* over the same files scripts/phase2.sh and scripts/stop.sh use, and it
# drives those very scripts -- so nothing here is a second way of doing things, and you can
# switch between the browser and the terminal mid-run without either getting confused.
#
#   scripts/portal.sh                 # http://127.0.0.1:8765 (foreground; Ctrl-C to stop)
#   scripts/portal.sh --lan           # ...also reachable from other machines on your
#                                     # network; it prints the address to type there.
#   scripts/portal.sh --open          # ...and open a browser
#   scripts/portal.sh --port 9000
#
# Lifecycle, from any terminal (it works off logs/portal.pid):
#
#   scripts/portal.sh --status        # running? which pid, which port
#   scripts/portal.sh --stop          # stop it
#   scripts/portal.sh --restart --lan # stop it, start it again in the background
#   scripts/portal.sh --bg --lan      # start in the background (log: logs/portal.log)
#
# Stopping or restarting the portal never touches a training run: the trainer is a separate,
# detached process and the portal holds no state of its own. A restart does briefly pause
# the *scheduler* (the portal runs it), so a rule due during those two seconds is missed --
# it stays missed rather than firing late.
#
# --lan has no login: anyone who can reach the address can start and stop training. That is
# usually what you want on a home LAN, and never what you want on the open internet.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-.venv/bin/python}
PID_FILE=logs/portal.pid
LOG=logs/portal.log

portal_pid() {   # a live portal, or nothing
    local p
    if [ -f "$PID_FILE" ]; then
        p=$(tr -dc '0-9' < "$PID_FILE")
        [ -n "$p" ] && kill -0 "$p" 2>/dev/null \
            && ps -p "$p" -o args= 2>/dev/null | grep -q 'aksharallm.portal' \
            && { echo "$p"; return 0; }
    fi
    # No usable pid file, but a portal may still be running: one started before pid files
    # existed, or -- the case that actually bit -- one whose file a *second* portal on
    # another port took over and then removed on its way out. Without this fallback the
    # lifecycle commands all report "not running" while the page is plainly still being
    # served, which is a maddening thing to debug.
    #
    # Matched on the module name and confined to this checkout, so it can only ever find a
    # portal, and only ours. The same belt-and-braces rule as identifying a trainer: the pid
    # file first, a narrow process match second.
    p=$(pgrep -f 'python.* -m aksharallm\.portal' 2>/dev/null | head -1) || true
    if [ -n "${p:-}" ] && [ "$(readlink -f "/proc/$p/cwd" 2>/dev/null)" = "$PWD" ]; then
        echo "$p"
        return 0
    fi
    return 1
}

stop_portal() {
    local p
    if ! p=$(portal_pid); then
        echo "portal is not running."
        rm -f "$PID_FILE"
        return 0
    fi
    # SIGTERM, not -9: the portal routes it through its Ctrl-C path, which releases the
    # scheduler lock. A killed -9 portal leaves logs/scheduler.pid behind and the next one
    # refuses to run the clock.
    kill -TERM "$p"
    for _ in $(seq 20); do kill -0 "$p" 2>/dev/null || break; sleep 0.25; done
    if kill -0 "$p" 2>/dev/null; then
        echo "portal pid $p did not stop; sending SIGKILL." >&2
        kill -9 "$p" 2>/dev/null || true
        rm -f "$PID_FILE" logs/scheduler.pid
    fi
    echo "portal stopped (pid $p). Training runs are unaffected."
}

usage() {
    # This wrapper owns --status/--stop/--restart/--bg; everything else is passed through to
    # the server. Without this, `--help` fell through to argparse, which knows nothing about
    # the four lifecycle flags -- so the script looked like it had no way to stop itself.
    sed -n '2,28p' "$0" | sed 's/^# \{0,1\}//'
    echo
    echo "Options this script handles itself:"
    echo "  --status      is a portal running? which pid, which address"
    echo "  --stop        stop it (SIGTERM, so the scheduler lock is released)"
    echo "  --restart     stop it, then start again in the background"
    echo "  --bg          start in the background (log: $LOG)"
    echo
    echo "Everything else is passed to the server itself:"
    echo
    "$PY" -m aksharallm.portal --help | sed 's/^/  /'
}

RESTART=0
BG=0
ARGS=()
while [ $# -gt 0 ]; do
    case "$1" in
        -h|--help) usage; exit 0 ;;
        --status)
            if p=$(portal_pid); then
                echo "portal is running as pid $p"
                ps -p "$p" -o args= | sed 's/^/    /'
                grep -m1 -E 'portal  ->|on your network' "$LOG" 2>/dev/null | sed 's/^/    /' || true
            else
                echo "portal is not running.  start it:  scripts/portal.sh --bg --lan"
            fi
            exit 0 ;;
        --stop)    stop_portal; exit 0 ;;
        --restart) RESTART=1; BG=1; shift ;;
        --bg)      BG=1; shift ;;
        *)         ARGS+=("$1"); shift ;;
    esac
done

[ "$RESTART" = "1" ] && stop_portal

if p=$(portal_pid); then
    echo "portal is already running as pid $p  (scripts/portal.sh --restart to replace it)" >&2
    exit 1
fi

if [ "$BG" = "1" ]; then
    mkdir -p logs
    nohup "$PY" -m aksharallm.portal "${ARGS[@]+"${ARGS[@]}"}" > "$LOG" 2>&1 &
    sleep 2
    if ! portal_pid > /dev/null; then
        echo "the portal did not come up. Last lines of $LOG:" >&2
        tail -20 "$LOG" >&2
        exit 1
    fi
    sed 's/^/    /' "$LOG"
    echo
    echo "    running in the background; log: $LOG"
    echo "    stop:    scripts/portal.sh --stop"
    exit 0
fi

exec "$PY" -m aksharallm.portal "${ARGS[@]+"${ARGS[@]}"}"
