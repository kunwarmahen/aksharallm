#!/usr/bin/env bash
# Stop a background training run *cleanly*: the trainer finishes the current step, writes
# ckpt_last.pt at that exact step, then exits. Re-running phase2.sh resumes with no loss
# spike, so stopping costs nothing but the seconds of the step in flight.
#
# It works off the pid file phase2.sh writes to checkpoints/<run>/train.pid.
#
# Usage:
#   scripts/stop.sh                    # stop the default run now (small-code; PURE=1 -> small)
#   scripts/stop.sh small              # stop a named run  (== checkpoints/small)
#   scripts/stop.sh small-code --at 20000   # finish step 20000, then save and exit
#   scripts/stop.sh small-code --after 500  # do 500 more steps, then save and exit
# Both are inclusive: the step you name is trained, logged and checkpointed; the resume
# picks up the step after it.
#   scripts/stop.sh --status           # is it running, and where is it? changes nothing
#   WAIT=900 scripts/stop.sh           # wait longer for the save (default 300s)
#   FORCE=1  scripts/stop.sh           # SIGKILL if still alive after WAIT (loses that step)
#
# --at/--after don't wait around: they leave the target in the STOP file and return, so you
# can queue a bounded finish and close the terminal.
set -euo pipefail
cd "$(dirname "$0")/.."

PURE=${PURE:-0}
DEFAULT_RUN=$([ "$PURE" = "1" ] && echo small || echo small-code)
WAIT=${WAIT:-300}
FORCE=${FORCE:-0}

RUN=""
AT=""
STATUS=0
while [ $# -gt 0 ]; do
    case "$1" in
        --at)     AT=${2:?--at needs a step number}; shift 2 ;;
        --after)  AFTER=${2:?--after needs a step count}; shift 2 ;;
        --status) STATUS=1; shift ;;
        -h|--help) sed -n '2,18p' "$0"; exit 0 ;;
        -*)      echo "unknown flag: $1" >&2; exit 2 ;;
        *)       RUN=$1; shift ;;
    esac
done
RUN=${RUN:-$DEFAULT_RUN}
RUN_DIR=checkpoints/$RUN
PID_FILE=$RUN_DIR/train.pid
STOP_FILE=$RUN_DIR/STOP
LOG_LINK=train_${RUN}.log   # symlink phase2.sh points at the current session's log

if [ ! -d "$RUN_DIR" ]; then
    echo "no such run: $RUN_DIR" >&2
    echo "runs found: $(ls -1 checkpoints 2>/dev/null | tr '\n' ' ')" >&2
    exit 1
fi

# ---- find the process ------------------------------------------------------------------
PID=""
[ -f "$PID_FILE" ] && PID=$(tr -dc '0-9' < "$PID_FILE")

# No pid file? The run may predate phase2.sh writing one, or have been launched by hand.
# Fall back to the command line (the config name matches the run name) and adopt the pid.
if [ -z "$PID" ]; then
    PID=$(pgrep -f "aksharallm.train.pretrain configs/${RUN}.yaml" | head -1 || true)
    if [ -n "$PID" ]; then
        echo "$PID" > "$PID_FILE"
        echo "found pid $PID by command line (no pid file); recorded it in $PID_FILE."
    fi
fi

if [ -z "$PID" ] || ! kill -0 "$PID" 2>/dev/null; then
    echo "run '$RUN' is not running${PID:+ (pid $PID is gone)}."
    if [ "$STATUS" = "1" ]; then
        [ -f "$RUN_DIR/run.meta" ] && sed 's/^/    /' "$RUN_DIR/run.meta"
        exit 0
    fi
    rm -f "$PID_FILE"
    # A leftover STOP file would make the *next* launch stop at step 0. Clear it.
    [ -f "$STOP_FILE" ] && { rm -f "$STOP_FILE"; echo "cleared a stale STOP file."; }
    exit 0
fi

# pids get recycled; make sure this one is still our trainer before signalling it.
if ! ps -p "$PID" -o args= 2>/dev/null | grep -q 'aksharallm.train'; then
    echo "pid $PID is alive but is not an aksharallm trainer -- refusing to touch it." >&2
    echo "(stale pid file; removing it)" >&2
    rm -f "$PID_FILE"
    exit 1
fi

# ---- status only: report and change nothing --------------------------------------------
if [ "$STATUS" = "1" ]; then
    echo "run '$RUN' is training as pid $PID (up $(ps -p "$PID" -o etime= | tr -d ' '))."
    [ -f "$RUN_DIR/run.meta" ] && sed 's/^/    /' "$RUN_DIR/run.meta"
    if [ -f "$STOP_FILE" ]; then
        echo "    STOP queued: $(cat "$STOP_FILE" | tr -d '\n') (empty = stopping now)"
    fi
    [ -e "$LOG_LINK" ] && { echo "    log: $(readlink -f "$LOG_LINK")"
                            echo "    last line:"; tail -1 "$LOG_LINK" | sed 's/^/      /'; }
    exit 0
fi

# ---- deferred stop: leave a target step in the STOP file and return --------------------
if [ -n "${AFTER:-}" ]; then
    LOG_JSONL=$RUN_DIR/train_log.jsonl
    CUR=$(grep -o '"step": *[0-9]*' "$LOG_JSONL" 2>/dev/null | tail -1 | tr -dc '0-9' || true)
    if [ -z "$CUR" ]; then
        echo "can't read a current step from $LOG_JSONL -- use --at <step> instead." >&2
        exit 1
    fi
    AT=$((CUR + AFTER))
    echo "last logged step is $CUR, so finishing step $AT."
fi

if [ -n "$AT" ]; then
    echo "$AT" > "$STOP_FILE"
    echo "queued: pid $PID will finish step $AT, save ckpt_last.pt at it, and exit"
    echo "        (so the resume starts at $((AT + 1)))."
    echo "  cancel:  rm $STOP_FILE"
    echo "  watch:   tail -f $LOG_LINK"
    exit 0
fi

# ---- stop now --------------------------------------------------------------------------
: > "$STOP_FILE"   # empty STOP file == stop after the current step
echo "asked pid $PID to stop after the current step. waiting up to ${WAIT}s for the save..."
waited=0
while [ "$waited" -lt "$WAIT" ] && kill -0 "$PID" 2>/dev/null; do
    sleep 2
    waited=$((waited + 2))
done

if kill -0 "$PID" 2>/dev/null; then
    if [ "$FORCE" = "1" ]; then
        echo "still alive after ${waited}s -- SIGKILL. The previous checkpoint stays intact;"
        echo "you lose at most the steps since it was written."
        kill -9 "$PID" 2>/dev/null || true
        sleep 2
        rm -f "$STOP_FILE"   # the trainer never got to clean it up
    else
        echo "still running after ${waited}s. Saving a 300M-param checkpoint takes ~30s, so it" >&2
        echo "is most likely mid-step. Give it longer (WAIT=900 scripts/stop.sh $RUN) or force" >&2
        echo "it (FORCE=1 scripts/stop.sh $RUN). The STOP file is in place either way." >&2
        exit 1
    fi
fi

rm -f "$PID_FILE"
echo "stopped after ${waited}s. tail of the log:"
[ -e "$LOG_LINK" ] && tail -3 "$LOG_LINK" | sed 's/^/    /'
echo
echo "resume:  scripts/phase2.sh   (picks up from ckpt_last.pt, no loss spike)"
echo "compare: scripts/sessions.py $RUN"
