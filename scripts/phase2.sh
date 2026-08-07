#!/usr/bin/env bash
# Phase 2: the real base model. ~300M params, ~10B tokens, ~6 days on a 3090.
#
# Default = the BLENDED base: 85% FineWeb-Edu + 15% Python, so one run yields a base that
# both chats (after SFT/DPO) and codes (after Python continued-pretraining).
# Set PURE=1 for the non-blended FineWeb-Edu-only fallback.
#
# Run this ONCE, after Phase 1 works. It pre-flights, builds data, smoke-tests, then
# launches in the background. Safe to interrupt (scripts/stop.sh); re-run to resume.
#
#   data prep   ~2-4 hours (network bound)
#   pretraining ~6 days
#
# Env knobs:
#   PURE=1              FineWeb-Edu only, no code blend
#   STOP_AFTER=500      train 500 steps this launch, then save and exit (chunked training)
#   STOP_IN=30m         same, bounded by wall-clock instead: 30m / 90s / 2h / 1h30m, or a
#                       bare number read as minutes. Counted from the first training step,
#                       so pre-flight and torch.compile do not eat into it.
#   SKIP_SMOKE=1        skip the 50-step smoke test (see below -- resumes only)
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-.venv/bin/python}
NEED_GB=25
PURE=${PURE:-0}
STOP_AFTER=${STOP_AFTER:-}
STOP_IN=${STOP_IN:-}
SKIP_SMOKE=${SKIP_SMOKE:-0}
SKIP_TESTS=${SKIP_TESTS:-0}

# Seconds from "30m" / "90s" / "2h" / "1h30m", or a bare number read as minutes. Same
# grammar as scripts/stop.sh --in, because they are the same idea at two different moments.
if [ -n "$STOP_IN" ]; then
    d=${STOP_IN,,}
    if [[ $d =~ ^[0-9]+$ ]]; then
        STOP_IN_S=$((d * 60))
    elif [[ $d =~ ^(([0-9]+)h)?(([0-9]+)m)?(([0-9]+)s)?$ ]] && [ -n "$d" ]; then
        STOP_IN_S=$(( ${BASH_REMATCH[2]:-0} * 3600 + ${BASH_REMATCH[4]:-0} * 60 + ${BASH_REMATCH[6]:-0} ))
    else
        echo "cannot read STOP_IN='$STOP_IN' as a duration -- try 30m, 90s, 2h or 1h30m" >&2
        exit 2
    fi
    [ "${STOP_IN_S:-0}" -ge 1 ] || { echo "STOP_IN must be at least one second" >&2; exit 2; }
fi

if [ "$PURE" = "1" ]; then
    CFG=configs/small.yaml;      RUN=small
else
    CFG=configs/small-code.yaml; RUN=small-code
fi
RUN_DIR=checkpoints/$RUN
PID_FILE=$RUN_DIR/train.pid
# One log per session, kept forever, so sessions can be compared afterwards. `LOG_LINK` is a
# stable path (a symlink to the newest session) so `tail -f train_<run>.log` always works.
LOG_DIR=logs/$RUN
LOG=$LOG_DIR/train_$(date '+%Y%m%d-%H%M%S').log
LOG_LINK=train_${RUN}.log
mkdir -p "$RUN_DIR" "$LOG_DIR"

# ---- publish this launch ----------------------------------------------------------------
# Pre-flight takes minutes (tests, data check, a 50-step smoke test) before any trainer
# exists. Without a record of it, a second launch -- from another terminal, or from the
# portal -- sails straight past the "already training?" check below and you get two of
# everything on one GPU. So the launcher publishes *itself*, the same way it publishes the
# trainer, and both scripts/stop.sh and the portal read it.
LAUNCH_PID_FILE=$RUN_DIR/launch.pid
LAUNCH_META=$RUN_DIR/launch.meta
LAUNCH_LOG=${LAUNCH_LOG:-}          # the caller (e.g. the portal) can name the log it captures
STARTED=$(date '+%Y-%m-%d %H:%M:%S')

launch_stage() {   # so an aborted launch can say *what* it interrupted
    cat > "$LAUNCH_META" <<META
pid     $$
stage   $1
started $STARTED
config  $CFG
log     ${LAUNCH_LOG:-(terminal)}
META
}
# Clean up on every exit path -- success, error, or SIGTERM from `scripts/stop.sh`.
trap 'rm -f "$LAUNCH_PID_FILE"' EXIT
trap 'echo "[abort] launch cancelled during $(sed -n "s/^stage *//p" "$LAUNCH_META" 2>/dev/null)"; exit 130' TERM INT

echo "=== pre-flight ($([ "$PURE" = 1 ] && echo 'pure FineWeb-Edu' || echo 'blended 85/15 general+Python')) ==="

if [ -f "$LAUNCH_PID_FILE" ]; then
    OTHER=$(tr -dc '0-9' < "$LAUNCH_PID_FILE")
    if [ -n "$OTHER" ] && [ "$OTHER" != "$$" ] && kill -0 "$OTHER" 2>/dev/null; then
        echo "    ERROR: another launch of '$RUN' is already in pre-flight (pid $OTHER)." >&2
        sed 's/^/           /' "$LAUNCH_META" >&2 2>/dev/null || true
        echo "           abort it:  scripts/stop.sh $RUN   (or wait for it to start training)" >&2
        trap - EXIT   # that pid file is the *other* launch's -- do not delete it
        exit 1
    fi
fi
echo "$$" > "$LAUNCH_PID_FILE"
launch_stage preflight

# Two trainers sharing one GPU and one checkpoint dir ruin both runs, and the smoke test
# below would fight the live one for memory. Check this before anything expensive.
if [ -f "$PID_FILE" ] && kill -0 "$(tr -dc '0-9' < "$PID_FILE")" 2>/dev/null; then
    echo "    ERROR: run '$RUN' is already training as pid $(cat "$PID_FILE")." >&2
    echo "           watch it:  tail -f $LOG_LINK" >&2
    echo "           stop it:   scripts/stop.sh $RUN" >&2
    exit 1
fi
rm -f "$PID_FILE"
FREE_GB=$(df -BG --output=avail . | tail -1 | tr -dc '0-9')
echo "    free disk: ${FREE_GB} GB (need ~${NEED_GB} GB)"
if [ "$FREE_GB" -lt "$NEED_GB" ]; then
    echo "    ERROR: not enough disk. Free space or lower --max-train-tokens." >&2
    exit 1
fi
# The suite is the launch gate: a six-day run should not start on code that fails its own
# tests, and it has caught real breakage. `preflight_tests.py` runs exactly that suite and
# prints ONE LINE PER FILE — `pytest -q` prints 1,250 bare dots over 90 silent seconds,
# which reads as a hang, and the reasonable response to a hang is to cancel the launch.
if [ "$SKIP_TESTS" = "1" ] && [ -s "$RUN_DIR/ckpt_last.pt" ]; then
    echo "    SKIP_TESTS=1 and $RUN_DIR/ckpt_last.pt exists -- skipping the suite"
    echo "    (only defensible when RESUMING a config that has already trained for real)"
else
    [ "$SKIP_TESTS" = "1" ] && echo "    SKIP_TESTS=1 ignored: nothing to resume, so this is a first launch."
    echo "=== tests ($(ls tests/test_*.py | wc -l) files) ==="
    $PY scripts/preflight_tests.py tests/ || { echo "tests failed -- fix before a 6-day run" >&2; exit 1; }
fi

echo
echo "=== 1/3  building token files (~20 GB) ==="
launch_stage data
if [ "$PURE" = "1" ]; then
    if [ -s data/fineweb/train.bin ]; then
        echo "    already present, skipping"
    else
        $PY -m aksharallm.data.prepare fineweb-edu-10bt --out-dir data/fineweb \
            --vocab-size 32768 --val-tokens 10000000 --max-train-tokens 10000000000
    fi
    BINS="data/fineweb/train.bin"
else
    # Blended: prepare_blend writes data/blend/{fineweb-edu-10bt,codeparrot-python}.bin,
    # val.bin and tokenizer.json -- exactly the paths configs/small-code.yaml expects,
    # so no manual pasting of the train_sources block is needed.
    if [ -s data/blend/fineweb-edu-10bt.bin ] && [ -s data/blend/codeparrot-python.bin ]; then
        echo "    already present, skipping"
    else
        $PY -m aksharallm.data.prepare_blend --out-dir data/blend --vocab-size 32768 \
            --source fineweb-edu-10bt:0.85 --source codeparrot-python:0.15 \
            --val-tokens 10000000 --max-train-tokens 10000000000
    fi
    BINS="data/blend/fineweb-edu-10bt.bin data/blend/codeparrot-python.bin"
fi

for b in $BINS; do
    sz=$(stat -c%s "$b")
    echo "    $b: $((sz / 1000000000)) GB / $((sz / 2)) tokens"
    if [ "$sz" -lt 1000000000 ]; then
        echo "    ERROR: $b is far smaller than expected. Do not train on this." >&2
        exit 1
    fi
done

echo
echo "=== 2/3  smoke test (50 steps) ==="
launch_stage smoke
# Skipping is only defensible when *resuming* a config that has already trained for real:
# the thing the smoke test proves (model builds, data loads, memory fits, MFU is sane) was
# proved by the previous session. On a first launch, or after touching the config or the
# data, let it run -- 8 minutes here has repeatedly beaten discovering it at hour six.
if [ "$SKIP_SMOKE" = "1" ] && [ -s "$RUN_DIR/ckpt_last.pt" ]; then
    echo "    SKIP_SMOKE=1 and $RUN_DIR/ckpt_last.pt exists -- skipping (resume of a proven config)"
elif [ "$SKIP_SMOKE" = "1" ]; then
    echo "    SKIP_SMOKE=1 ignored: there is no checkpoint to resume, so this is a first"
    echo "    launch and the smoke test is exactly what you want. Running it."
    $PY -m aksharallm.train.pretrain "$CFG" \
        -o train.max_steps=50 -o train.out_dir=/tmp/aksharallm_smoke -o train.resume=null
else
    echo "    check: step-0 loss ~10.4, MFU > 35%, memory stable"
    $PY -m aksharallm.train.pretrain "$CFG" \
        -o train.max_steps=50 -o train.out_dir=/tmp/aksharallm_smoke -o train.resume=null
fi

echo
echo "=== 3/3  launching the real run ==="
launch_stage launching
rm -f "$RUN_DIR/STOP"   # a STOP left over from a previous stop would end this run at step 0

EXTRA=()
[ -n "$STOP_AFTER" ] && EXTRA+=(-o "train.stop_after=$STOP_AFTER")
[ -n "$STOP_IN" ] && EXTRA+=(-o "train.stop_after_s=$STOP_IN_S")

nohup $PY -m aksharallm.train.pretrain "$CFG" "${EXTRA[@]}" > "$LOG" 2>&1 &
PID=$!
ln -sfn "$LOG" "$LOG_LINK"   # stable path -> newest session

# The pid goes to a file so any later shell (or scripts/stop.sh) can find this run without
# the terminal that started it. run.meta is for humans reading it back days later.
echo "$PID" > "$PID_FILE"
cat > "$RUN_DIR/run.meta" <<META
pid     $PID
started $(date '+%Y-%m-%d %H:%M:%S')
config  $CFG
log     $LOG
cmd     $PY -m aksharallm.train.pretrain $CFG ${EXTRA[*]-}
META
# Append-only history of every session of this run, for comparing them later.
echo "$(date '+%Y-%m-%d %H:%M:%S')  pid $PID  $LOG" >> "$RUN_DIR/sessions.log"

sleep 5   # long enough to catch an immediate crash (bad config, no GPU, missing file)
if ! kill -0 "$PID" 2>/dev/null; then
    echo "    ERROR: the trainer died within 5s. Last lines of $LOG:" >&2
    tail -20 "$LOG" >&2
    rm -f "$PID_FILE"
    exit 1
fi

echo "    pid $PID  ->  $PID_FILE  (config: $CFG)"
[ -n "$STOP_AFTER" ] && echo "    will stop itself after $STOP_AFTER steps"
[ -n "$STOP_IN" ] && echo "    will stop itself after ${STOP_IN_S}s of training (~$((STOP_IN_S / 60))m)"
echo "    log $LOG  (this session; $LOG_LINK -> it)"
echo
echo "    watch:   tail -f $LOG_LINK"
echo "             scripts/portal.sh              (browser: progress, graphs, start/stop)"
echo "    stop:    scripts/stop.sh $RUN"
echo "    later:   scripts/stop.sh $RUN --after 500   (do 500 more steps, then stop)"
echo "             scripts/stop.sh $RUN --at 20000    (stop on reaching step 20000)"
echo "    resume:  re-run this script -- it picks up exactly where it left off"
echo "    compare: scripts/sessions.py $RUN           (per-session table)"
echo
echo "    Stopping is safe at any time: it saves at the current step and the resumed"
echo "    run continues with no loss spike. A crash costs at most ~20 min (ckpt_every)."
