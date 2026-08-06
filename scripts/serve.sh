#!/usr/bin/env bash
# Serve a checkpoint over HTTP: continuous batching, a paged KV cache, an OpenAI-shaped API.
#
#   scripts/serve.sh                       # serve the default checkpoint on :8770
#   scripts/serve.sh small-code            # ...a particular run's best checkpoint
#   scripts/serve.sh small-code --bg       # in the background (log: logs/serve/serve.log)
#   scripts/serve.sh --status              # running? which pid, which port, which model
#   scripts/serve.sh --stop                # stop it
#   scripts/serve.sh --restart small-code  # stop, start again in the background
#
#   PORT=9000 MAX_BATCH=64 DEVICE=cuda scripts/serve.sh small-code --bg
#
# The same shape as scripts/portal.sh on purpose: a pid file, a log, and lifecycle flags that
# work from any terminal — so the portal's Serve panel can drive *this script* rather than
# containing a second way to start a server. See docs/16-serving.md.
#
# It never fights a training run for the card: the server asks the same device policy the
# playground uses, and loads on the CPU while a run is training. `/health` says which.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-.venv/bin/python}
PORT=${PORT:-8770}
MAX_BATCH=${MAX_BATCH:-32}
DEVICE=${DEVICE:-}
POOL_BLOCKS=${POOL_BLOCKS:-}
DIR=logs/serve
PID_FILE=$DIR/serve.pid
META=$DIR/serve.meta
LOG=$DIR/serve.log

mkdir -p "$DIR"

serve_pid() {   # a live server, or nothing
    local p
    if [ -f "$PID_FILE" ]; then
        p=$(tr -dc '0-9' < "$PID_FILE")
        [ -n "$p" ] && kill -0 "$p" 2>/dev/null \
            && ps -p "$p" -o args= 2>/dev/null | grep -q 'aksharallm.serve' \
            && { echo "$p"; return 0; }
    fi
    # Same fallback as the portal's: a server started before this script existed, or one
    # whose pid file was lost, is still a server — and reporting "not running" while the
    # port answers is the most confusing thing this script could do.
    pgrep -f "aksharallm\.serve" 2>/dev/null | head -1
}

stop_serve() {
    local p; p=$(serve_pid || true)
    if [ -z "$p" ]; then echo "no server running"; return 0; fi
    echo "stopping server (pid $p)"
    kill "$p" 2>/dev/null || true
    for _ in $(seq 1 40); do kill -0 "$p" 2>/dev/null || break; sleep 0.25; done
    kill -0 "$p" 2>/dev/null && { echo "  did not stop; SIGKILL"; kill -9 "$p" 2>/dev/null || true; }
    rm -f "$PID_FILE"
}

status_serve() {
    local p; p=$(serve_pid || true)
    if [ -z "$p" ]; then echo "not running"; return 1; fi
    echo "running as pid $p"
    [ -f "$META" ] && sed 's/^/  /' "$META"
    return 0
}

CKPT=""
BG=0
for arg in "$@"; do
    case "$arg" in
        --status)  status_serve; exit $? ;;
        --stop)    stop_serve; exit 0 ;;
        --restart) stop_serve ;;
        --bg)      BG=1 ;;
        --port)    ;;   # handled through PORT=, kept here so it is not read as a checkpoint
        -*)        echo "unknown flag: $arg" >&2; exit 2 ;;
        *)         CKPT="$arg" ;;
    esac
done

if [ -n "$(serve_pid || true)" ]; then
    echo "a server is already running:" >&2
    status_serve >&2
    echo "stop it first: scripts/serve.sh --stop" >&2
    exit 1
fi

CKPT=${CKPT:-small-code}
ARGS=("$CKPT" --port "$PORT" --max-batch "$MAX_BATCH")
[ -n "$DEVICE" ] && ARGS+=(--device "$DEVICE")
[ -n "$POOL_BLOCKS" ] && ARGS+=(--pool-blocks "$POOL_BLOCKS")

{
    echo "checkpoint $CKPT"
    echo "port       $PORT"
    echo "max_batch  $MAX_BATCH"
    echo "device     ${DEVICE:-auto}"
    echo "started    $(date '+%Y-%m-%d %H:%M:%S')"
    echo "command    $PY -m aksharallm.serve ${ARGS[*]}"
} > "$META"

if [ "$BG" = "1" ]; then
    nohup "$PY" -m aksharallm.serve "${ARGS[@]}" >> "$LOG" 2>&1 &
    echo $! > "$PID_FILE"
    sleep 1
    echo "serving in the background: http://127.0.0.1:$PORT/v1  (pid $(cat "$PID_FILE"))"
    echo "  log:    $LOG"
    echo "  health: curl -s http://127.0.0.1:$PORT/health | $PY -m json.tool"
    echo "  stop:   scripts/serve.sh --stop"
else
    echo $$ > "$PID_FILE"
    trap 'rm -f "$PID_FILE"' EXIT
    exec "$PY" -m aksharallm.serve "${ARGS[@]}" 2>&1 | tee -a "$LOG"
fi
