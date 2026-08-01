#!/usr/bin/env bash
# Launch a Phase-1-scale experiment run: hours, not days, on data that already exists.
#
# scripts/phase2.sh is the launcher for the real base model — it builds 20 GB of tokens and
# pre-flights for a six-day run. The experiments (mixture of experts today, masked diffusion
# next) are a different shape: TinyStories is already tokenized, the run is a few hours, and
# the point is to compare against the dense baseline that already reached val 1.472.
#
# It publishes exactly the same contract phase2.sh does — launch.pid / launch.meta while
# pre-flighting, train.pid / run.meta / sessions.log once training, one log per session with
# a stable symlink — so scripts/stop.sh, scripts/sessions.py and the portal all drive it
# without knowing which launcher started it.
#
#   scripts/experiment.sh tiny-moe          # the MoE experiment (configs/tiny-moe.yaml)
#   scripts/experiment.sh tiny              # re-run the dense baseline
#
# Env knobs (same names and meanings as phase2.sh):
#   STOP_AFTER=500      train 500 steps this launch, then save and exit
#   STOP_IN=30m         same, bounded by wall clock: 30m / 90s / 2h / 1h30m, or a bare
#                       number read as minutes. Counted from the first training step.
#   SKIP_SMOKE=1        skip the 30-step smoke test (only honoured when resuming)
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-.venv/bin/python}
RUN=${1:-${RUN:-}}
STOP_AFTER=${STOP_AFTER:-}
STOP_IN=${STOP_IN:-}
SKIP_SMOKE=${SKIP_SMOKE:-0}

if [ -z "$RUN" ]; then
    echo "usage: scripts/experiment.sh <run>    (a configs/<run>.yaml at Phase-1 scale)" >&2
    echo "       e.g. scripts/experiment.sh tiny-moe" >&2
    exit 2
fi
# The name reaches a path and a command line, so it is whitelisted rather than escaped —
# the same rule aksharallm/portal/runs.py applies to a name arriving from a browser.
if ! [[ $RUN =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]]; then
    echo "bad run name: '$RUN'" >&2
    exit 2
fi

CFG=configs/$RUN.yaml
[ -f "$CFG" ] || { echo "no such config: $CFG" >&2; exit 2; }

# Same duration grammar as phase2.sh and scripts/stop.sh --in: they are one idea at three
# different moments, so they parse identically.
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

RUN_DIR=checkpoints/$RUN
PID_FILE=$RUN_DIR/train.pid
LOG_DIR=logs/$RUN
LOG=$LOG_DIR/train_$(date '+%Y%m%d-%H%M%S').log
LOG_LINK=train_${RUN}.log
mkdir -p "$RUN_DIR" "$LOG_DIR"

# ---- publish this launch ----------------------------------------------------------------
LAUNCH_PID_FILE=$RUN_DIR/launch.pid
LAUNCH_META=$RUN_DIR/launch.meta
LAUNCH_LOG=${LAUNCH_LOG:-}
STARTED=$(date '+%Y-%m-%d %H:%M:%S')

launch_stage() {
    cat > "$LAUNCH_META" <<META
pid     $$
stage   $1
started $STARTED
config  $CFG
log     ${LAUNCH_LOG:-(terminal)}
META
}
trap 'rm -f "$LAUNCH_PID_FILE"' EXIT
trap 'echo "[abort] launch cancelled during $(sed -n "s/^stage *//p" "$LAUNCH_META" 2>/dev/null)"; exit 130' TERM INT

echo "=== pre-flight ($RUN) ==="

if [ -f "$LAUNCH_PID_FILE" ]; then
    OTHER=$(tr -dc '0-9' < "$LAUNCH_PID_FILE")
    if [ -n "$OTHER" ] && [ "$OTHER" != "$$" ] && kill -0 "$OTHER" 2>/dev/null; then
        echo "    ERROR: another launch of '$RUN' is already in pre-flight (pid $OTHER)." >&2
        sed 's/^/           /' "$LAUNCH_META" >&2 2>/dev/null || true
        echo "           abort it:  scripts/stop.sh $RUN" >&2
        trap - EXIT
        exit 1
    fi
fi
echo "$$" > "$LAUNCH_PID_FILE"
launch_stage preflight

if [ -f "$PID_FILE" ] && kill -0 "$(tr -dc '0-9' < "$PID_FILE")" 2>/dev/null; then
    echo "    ERROR: run '$RUN' is already training as pid $(cat "$PID_FILE")." >&2
    echo "           stop it:   scripts/stop.sh $RUN" >&2
    exit 1
fi
rm -f "$PID_FILE"

# One GPU. A Phase-2 session is the expensive thing on this machine and an experiment must
# never be what killed it -- so this refuses rather than sharing.
for other in checkpoints/*/train.pid; do
    [ -f "$other" ] || continue
    opid=$(tr -dc '0-9' < "$other" 2>/dev/null || true)
    orun=$(basename "$(dirname "$other")")
    if [ -n "$opid" ] && [ "$orun" != "$RUN" ] && kill -0 "$opid" 2>/dev/null; then
        echo "    ERROR: '$orun' is training as pid $opid and this would share the card." >&2
        echo "           stop it first:  scripts/stop.sh $orun" >&2
        exit 1
    fi
done

$PY -m pytest tests/ -q || { echo "tests failed -- fix before spending GPU hours" >&2; exit 1; }

# ---- data -------------------------------------------------------------------------------
echo
echo "=== 1/3  data ==="
launch_stage data
TRAIN_BIN=$($PY - "$CFG" <<'EOF'
import sys, yaml
print((yaml.safe_load(open(sys.argv[1])) or {}).get("data", {}).get("train_bin", ""))
EOF
)
if [ -z "$TRAIN_BIN" ]; then
    echo "    ERROR: $CFG has no data.train_bin" >&2
    exit 1
fi
if [ -s "$TRAIN_BIN" ]; then
    sz=$(stat -c%s "$TRAIN_BIN")
    echo "    $TRAIN_BIN: $((sz / 1000000)) MB / $((sz / 2)) tokens"
else
    echo "    $TRAIN_BIN is missing. Building TinyStories (~10 minutes)..."
    $PY -m aksharallm.data.prepare tinystories --out-dir data/tinystories \
        --vocab-size 8192 --max-train-tokens 400000000
fi

# ---- smoke ------------------------------------------------------------------------------
echo
echo "=== 2/3  smoke test (30 steps) ==="
launch_stage smoke
if [ "$SKIP_SMOKE" = "1" ] && [ -s "$RUN_DIR/ckpt_last.pt" ]; then
    echo "    SKIP_SMOKE=1 and $RUN_DIR/ckpt_last.pt exists -- skipping (resume of a proven config)"
else
    [ "$SKIP_SMOKE" = "1" ] && echo "    SKIP_SMOKE=1 ignored: nothing to resume, so this is a first launch."
    echo "    check: step-0 loss ~9.0, memory stable, and for an MoE run the experts field"
    $PY -m aksharallm.train.pretrain "$CFG" \
        -o train.max_steps=30 -o train.out_dir=/tmp/aksharallm_smoke -o train.resume=null \
        -o train.eval_every=1000 -o train.log_every=10
fi

# ---- launch -----------------------------------------------------------------------------
echo
echo "=== 3/3  launching ==="
launch_stage launching
rm -f "$RUN_DIR/STOP"

EXTRA=()
[ -n "$STOP_AFTER" ] && EXTRA+=(-o "train.stop_after=$STOP_AFTER")
[ -n "$STOP_IN" ] && EXTRA+=(-o "train.stop_after_s=$STOP_IN_S")

nohup $PY -m aksharallm.train.pretrain "$CFG" "${EXTRA[@]}" > "$LOG" 2>&1 &
PID=$!
ln -sfn "$LOG" "$LOG_LINK"

echo "$PID" > "$PID_FILE"
cat > "$RUN_DIR/run.meta" <<META
pid     $PID
started $(date '+%Y-%m-%d %H:%M:%S')
config  $CFG
log     $LOG
cmd     $PY -m aksharallm.train.pretrain $CFG ${EXTRA[*]-}
META
echo "$(date '+%Y-%m-%d %H:%M:%S')  pid $PID  $LOG" >> "$RUN_DIR/sessions.log"

sleep 5
if ! kill -0 "$PID" 2>/dev/null; then
    echo "    ERROR: the trainer died within 5s. Last lines of $LOG:" >&2
    tail -20 "$LOG" >&2
    rm -f "$PID_FILE"
    exit 1
fi

echo "    pid $PID  ->  $PID_FILE  (config: $CFG)"
[ -n "$STOP_AFTER" ] && echo "    will stop itself after $STOP_AFTER steps"
[ -n "$STOP_IN" ] && echo "    will stop itself after ${STOP_IN_S}s of training"
echo "    log $LOG  ($LOG_LINK -> it)"
echo
echo "    watch:   tail -f $LOG_LINK"
echo "             scripts/portal.sh              (browser: progress, graphs, experts)"
echo "    stop:    scripts/stop.sh $RUN"
echo "    compare: scripts/sessions.py $RUN"
echo "             $PY -m aksharallm.eval $RUN --suite fast"
