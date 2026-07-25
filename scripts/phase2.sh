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
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-.venv/bin/python}
NEED_GB=25
PURE=${PURE:-0}
STOP_AFTER=${STOP_AFTER:-}

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

echo "=== pre-flight ($([ "$PURE" = 1 ] && echo 'pure FineWeb-Edu' || echo 'blended 85/15 general+Python')) ==="

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
$PY -m pytest tests/ -q || { echo "tests failed -- fix before a 6-day run" >&2; exit 1; }

echo
echo "=== 1/3  building token files (~20 GB) ==="
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
echo "    check: step-0 loss ~10.4, MFU > 35%, memory stable"
$PY -m aksharallm.train.pretrain "$CFG" \
    -o train.max_steps=50 -o train.out_dir=/tmp/aksharallm_smoke -o train.resume=null

echo
echo "=== 3/3  launching the real run ==="
mkdir -p "$RUN_DIR" "$LOG_DIR"
rm -f "$RUN_DIR/STOP"   # a STOP left over from a previous stop would end this run at step 0

EXTRA=()
[ -n "$STOP_AFTER" ] && EXTRA+=(-o "train.stop_after=$STOP_AFTER")

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
echo "    log $LOG  (this session; $LOG_LINK -> it)"
echo
echo "    watch:   tail -f $LOG_LINK"
echo "    stop:    scripts/stop.sh $RUN"
echo "    later:   scripts/stop.sh $RUN --after 500   (do 500 more steps, then stop)"
echo "             scripts/stop.sh $RUN --at 20000    (stop on reaching step 20000)"
echo "    resume:  re-run this script -- it picks up exactly where it left off"
echo "    compare: scripts/sessions.py $RUN           (per-session table)"
echo
echo "    Stopping is safe at any time: it saves at the current step and the resumed"
echo "    run continues with no loss spike. A crash costs at most ~20 min (ckpt_every)."
