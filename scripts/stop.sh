#!/usr/bin/env bash
# Stop a background training run *cleanly*: the trainer finishes the current step, writes
# ckpt_last.pt at that exact step, then exits. Re-running phase2.sh resumes with no loss
# spike, so stopping costs nothing but the seconds of the step in flight.
#
# It works off checkpoints/<run>/train.pid, which the trainer itself writes (so it is the
# pid of whoever is training into *that directory* -- never the 50-step smoke test, which
# has the same command line but a throwaway out_dir).
#
# If nothing is training yet but phase2.sh is still in pre-flight (checkpoints/<run>/
# launch.pid), a stop aborts that launch instead -- nothing has trained, so nothing is lost.
#
# Usage:
#   scripts/stop.sh                    # stop the default run now (small-code; PURE=1 -> small)
#   scripts/stop.sh small              # stop a named run  (== checkpoints/small)
#   scripts/stop.sh small-code --at 20000   # finish step 20000, then save and exit
#   scripts/stop.sh small-code --after 500  # do 500 more steps, then save and exit
# Both are inclusive: the step you name is trained, logged and checkpointed; the resume
# picks up the step after it.
#   scripts/stop.sh small-code --in 30m     # train 30 more minutes, then save and exit
#   scripts/stop.sh small-code --by 06:30   # train until 06:30 (tomorrow if it has passed)
# --in takes 30m / 90s / 2h / 1h30m, or a bare number read as minutes. The deadline goes
# into the STOP file as @<epoch> and the *trainer* honours it, so it survives this terminal
# closing, the portal restarting, and the run slowing down halfway through.
#   scripts/stop.sh small-code --cancel     # withdraw a queued stop; the run carries on
#   scripts/stop.sh --status           # is it running, and where is it? changes nothing
#   WAIT=900 scripts/stop.sh           # wait longer for the save (default 300s)
#   FORCE=1  scripts/stop.sh           # SIGKILL if still alive after WAIT (loses that step)
#
# --at/--after/--in/--by don't wait around: they leave the target in the STOP file and
# return, so you can queue a bounded finish and close the terminal.
#
# All of this is also available as buttons: scripts/portal.sh (it runs this very script).
set -euo pipefail
cd "$(dirname "$0")/.."

PURE=${PURE:-0}
DEFAULT_RUN=$([ "$PURE" = "1" ] && echo small || echo small-code)
WAIT=${WAIT:-300}
FORCE=${FORCE:-0}

# Seconds from "30m", "90s", "2h", "1h30m", or a bare number read as minutes -- "give it
# another 30" is half an hour in every conversation this script exists to serve.
parse_duration() {
    local d=${1,,} h=0 m=0 s=0
    if [[ $d =~ ^[0-9]+$ ]]; then echo $((d * 60)); return 0; fi
    [[ $d =~ ^(([0-9]+)h)?(([0-9]+)m)?(([0-9]+)s)?$ && $d != "" ]] || return 1
    h=${BASH_REMATCH[2]:-0}; m=${BASH_REMATCH[4]:-0}; s=${BASH_REMATCH[6]:-0}
    local total=$((h * 3600 + m * 60 + s))
    [ "$total" -ge 1 ] || return 1
    echo "$total"
}

RUN=""
AT=""
UNTIL=""
STATUS=0
CANCEL=0
while [ $# -gt 0 ]; do
    case "$1" in
        --at)     AT=${2:?--at needs a step number}; shift 2 ;;
        --after)  AFTER=${2:?--after needs a step count}; shift 2 ;;
        --in)     SECS=$(parse_duration "${2:?--in needs a duration}") || {
                      echo "cannot read '$2' as a duration -- try 30m, 90s, 2h or 1h30m" >&2
                      exit 2; }
                  UNTIL=$(( $(date +%s) + SECS )); shift 2 ;;
        --by)     [[ ${2:?--by needs HH:MM} =~ ^([01]?[0-9]|2[0-3]):[0-5][0-9]$ ]] || {
                      echo "--by takes a 24-hour HH:MM, not '$2'" >&2; exit 2; }
                  UNTIL=$(date -d "$2" +%s)
                  # A time that has already passed today means the one tomorrow morning --
                  # "--by 06:30" at midnight is the whole night, not a stop in the past.
                  [ "$UNTIL" -le "$(date +%s)" ] && UNTIL=$((UNTIL + 86400))
                  shift 2 ;;
        --cancel) CANCEL=1; shift ;;
        --status) STATUS=1; shift ;;
        -h|--help) sed -n '2,33p' "$0"; exit 0 ;;
        -*)      echo "unknown flag: $1" >&2; exit 2 ;;
        *)       RUN=$1; shift ;;
    esac
done
RUN=${RUN:-$DEFAULT_RUN}
RUN_DIR=checkpoints/$RUN
PID_FILE=$RUN_DIR/train.pid
STOP_FILE=$RUN_DIR/STOP
LAUNCH_PID_FILE=$RUN_DIR/launch.pid   # phase2.sh while it is still pre-flighting
LAUNCH_META=$RUN_DIR/launch.meta
LOG_LINK=train_${RUN}.log   # symlink phase2.sh points at the current session's log

launch_pid() {   # a live phase2.sh for this run (pre-flight: tests, data, smoke test)
    local p
    [ -f "$LAUNCH_PID_FILE" ] || return 1
    p=$(tr -dc '0-9' < "$LAUNCH_PID_FILE")
    [ -n "$p" ] && kill -0 "$p" 2>/dev/null && ps -p "$p" -o args= 2>/dev/null \
        | grep -q 'phase2.sh' && { echo "$p"; return 0; }
    return 1
}
launch_stage() { sed -n 's/^stage *//p' "$LAUNCH_META" 2>/dev/null | head -1; }

# What a STOP file is asking for, in words. The three forms are the trainer's contract
# (aksharallm/train/stopfile.py): empty, a step number, or @<epoch>.
describe_stop() {
    local raw
    raw=$(tr -d '\n' < "$1" 2>/dev/null | head -c 32)
    case "$raw" in
        "")   echo "stopping after the current step" ;;
        @*)   echo "stop at $(date -d "@${raw#@}" '+%H:%M' 2>/dev/null || echo "${raw}")" ;;
        *)    echo "stop after step $raw" ;;
    esac
}

if [ ! -d "$RUN_DIR" ]; then
    echo "no such run: $RUN_DIR" >&2
    echo "runs found: $(ls -1 checkpoints 2>/dev/null | tr '\n' ' ')" >&2
    exit 1
fi

# ---- cancel a queued stop ---------------------------------------------------------------
# Removing the STOP file is all it takes: the trainer only reads it, and reads it fresh
# every step. Works whether or not anything is running -- a leftover STOP on a dead run
# would otherwise end the *next* launch at step 0.
if [ "$CANCEL" = "1" ]; then
    if [ -f "$STOP_FILE" ]; then
        echo "cancelled the queued stop for '$RUN' (was: $(describe_stop "$STOP_FILE"))."
        rm -f "$STOP_FILE"
        echo "the run continues to its budget (or until you stop it again)."
    else
        echo "no stop is queued for '$RUN'."
    fi
    exit 0
fi

# ---- find the process ------------------------------------------------------------------
PID=""
[ -f "$PID_FILE" ] && PID=$(tr -dc '0-9' < "$PID_FILE")

# No pid file? Fall back to the command line and adopt the pid.
#
# The `$` anchor is not decoration. The 50-step smoke test inside phase2.sh runs the *same*
# command with `-o train.out_dir=/tmp/...` appended, so an unanchored match finds the smoke
# test and cheerfully aims a stop at it -- writing a STOP file the smoke test never reads,
# while reporting a pid that is about to vanish. Anchoring means we only ever adopt a plain,
# override-free launch; anything else is expected to have written its own train.pid
# (aksharallm/train/pretrain.py does this into its own out_dir).
if [ -z "$PID" ]; then
    PID=$(pgrep -f "aksharallm\.train\.pretrain configs/${RUN}\.yaml\$" | head -1 || true)
    if [ -n "$PID" ]; then
        echo "$PID" > "$PID_FILE"
        echo "found pid $PID by command line (no pid file); recorded it in $PID_FILE."
    fi
fi

if [ -z "$PID" ] || ! kill -0 "$PID" 2>/dev/null; then
    # Nothing is training -- but a launch may be in pre-flight, minutes away from training.
    # Treating that as "not running" is how you end up with two trainers on one GPU.
    if LPID=$(launch_pid); then
        STAGE=$(launch_stage)
        echo "run '$RUN' is not training yet: a launch is in pre-flight (pid $LPID, stage ${STAGE:-?})."
        [ -f "$LAUNCH_META" ] && sed 's/^/    /' "$LAUNCH_META"
        if [ "$STATUS" = "1" ]; then
            echo "    it will start training on its own; stop it again once it does."
            exit 0
        fi
        if [ "$STAGE" = "launching" ]; then
            echo "it is launching the trainer right now -- wait a few seconds and stop that instead." >&2
            exit 1
        fi
        echo "aborting the launch (nothing has trained yet, so nothing is lost)."
        # The launcher plus whatever it is running right now (pytest, a data build, the smoke
        # test). Signal the children too: bash defers a signal until its foreground child
        # exits, which for the smoke test is eight minutes away. Children only -- never a
        # process group, which for a terminal launch would include your shell.
        for child in $(pgrep -P "$LPID" 2>/dev/null); do kill -TERM "$child" 2>/dev/null || true; done
        kill -TERM "$LPID" 2>/dev/null || true
        waited=0
        while [ "$waited" -lt 30 ] && kill -0 "$LPID" 2>/dev/null; do sleep 1; waited=$((waited + 1)); done
        if kill -0 "$LPID" 2>/dev/null; then
            echo "launch pid $LPID is still alive after ${waited}s -- kill -9 $LPID if it stays." >&2
            exit 1
        fi
        echo "launch aborted after ${waited}s."
        [ "$STAGE" = "data" ] && echo "NOTE: it was building token files. Check the sizes of" \
            "data/*/*.bin before the next launch -- a half-written .bin is not obvious." >&2
        exit 0
    fi

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
ARGS=$(ps -p "$PID" -o args= 2>/dev/null || true)
if ! printf '%s' "$ARGS" | grep -q 'aksharallm.train'; then
    echo "pid $PID is alive but is not an aksharallm trainer -- refusing to touch it." >&2
    echo "(stale pid file; removing it)" >&2
    rm -f "$PID_FILE"
    exit 1
fi
# Belt and braces after the anchored pgrep above: never aim a stop at the smoke test. Its
# STOP file lives in /tmp/aksharallm_smoke, so a stop written here would be silently ignored
# while the terminal claimed success.
if printf '%s' "$ARGS" | grep -q 'aksharallm_smoke'; then
    echo "pid $PID is the 50-step smoke test, not the run -- refusing." >&2
    echo "wait for pre-flight to finish, then stop the real trainer." >&2
    rm -f "$PID_FILE"
    exit 1
fi

# ---- status only: report and change nothing --------------------------------------------
if [ "$STATUS" = "1" ]; then
    echo "run '$RUN' is training as pid $PID (up $(ps -p "$PID" -o etime= | tr -d ' '))."
    [ -f "$RUN_DIR/run.meta" ] && sed 's/^/    /' "$RUN_DIR/run.meta"
    [ -f "$STOP_FILE" ] && echo "    STOP queued: $(describe_stop "$STOP_FILE")"
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

# A deadline goes in as @<epoch>, and the trainer compares it against the clock on every
# step. Nothing here has to stay alive to make it happen -- which is the whole point, since
# the alternative (a timer in this shell, or in the portal) dies with whatever holds it.
if [ -n "$UNTIL" ]; then
    echo "@$UNTIL" > "$STOP_FILE"
    LEFT=$(( (UNTIL - $(date +%s) + 30) / 60 ))
    echo "queued: pid $PID will train until $(date -d "@$UNTIL" '+%H:%M') (~${LEFT}m from now),"
    echo "        then save ckpt_last.pt at whatever step it is on and exit."
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
