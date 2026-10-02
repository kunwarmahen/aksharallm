#!/usr/bin/env bash
# Launch an audio run: the codec, the audio language model over its tokens, or the speech
# recogniser (docs/23).
#
# It publishes exactly the contract phase2.sh and experiment.sh publish -- launch.pid /
# launch.meta while pre-flighting, train.pid / run.meta / sessions.log once training, one log
# per session with a stable symlink -- so scripts/stop.sh, scripts/sessions.py and the portal
# drive it without knowing the model is not a transformer over text.
#
#   scripts/audio.sh codec-synth      # the codec on synthetic babble (minutes, no download)
#   scripts/audio.sh codec-lj         # the codec on LJSpeech (docs/21)
#   scripts/audio.sh audiolm-synth    # the audio LM over a trained codec's tokens
#   scripts/audio.sh asr-synth        # the Conformer-CTC recogniser on synthetic speech
#   scripts/audio.sh asr-libri100     # ...on LibriSpeech train-clean-100 (docs/23)
#
# Env knobs (same names and meanings as phase2.sh):
#   STOP_AFTER=500      train 500 steps this launch, then save and exit
#   STOP_IN=30m         same, bounded by wall clock: 30m / 90s / 2h / 1h30m, or a bare
#                       number read as minutes. Counted from the first training step.
#   SKIP_SMOKE=1        skip the smoke test (only honoured when resuming)
#   SKIP_TESTS=1        skip the test suite too (likewise: only when resuming)
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-.venv/bin/python}
RUN=${1:-${RUN:-}}
STOP_AFTER=${STOP_AFTER:-}
STOP_IN=${STOP_IN:-}
SKIP_SMOKE=${SKIP_SMOKE:-0}
SKIP_TESTS=${SKIP_TESTS:-0}

if [ -z "$RUN" ]; then
    echo "usage: scripts/audio.sh <run>    (a configs/<run>.yaml for the audio phase)" >&2
    echo "       e.g. scripts/audio.sh codec-synth" >&2
    exit 2
fi
# The name reaches a path and a command line, so it is whitelisted rather than escaped.
if ! [[ $RUN =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]]; then
    echo "bad run name: '$RUN'" >&2
    exit 2
fi

CFG=configs/$RUN.yaml
[ -f "$CFG" ] || { echo "no such config: $CFG" >&2; exit 2; }

# Which trainer? The config says so, by which top-level section it has. This is the same
# distinction aksharallm/portal/runs.py draws to decide what is a language-model run.
if grep -qE '^codec:' "$CFG"; then
    TRAINER=aksharallm.audio.train_codec
    KIND=codec
elif grep -qE '^audiolm:' "$CFG"; then
    TRAINER=aksharallm.audio.train_lm
    KIND=audiolm
elif grep -qE '^asr:' "$CFG"; then
    TRAINER=aksharallm.asr.train
    KIND=asr
else
    echo "$CFG has no 'codec:', 'audiolm:' or 'asr:' section -- not an audio run." >&2
    echo "For a language model over text, use scripts/phase2.sh or scripts/experiment.sh." >&2
    exit 2
fi

# Same duration grammar as phase2.sh and scripts/stop.sh --in.
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

OUT_DIR=$($PY - "$CFG" <<'EOF'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1])) or {}
print((cfg.get("train") or {}).get("out_dir", ""))
EOF
)
[ -n "$OUT_DIR" ] || { echo "$CFG has no train.out_dir" >&2; exit 2; }

RUN_DIR=$OUT_DIR
PID_FILE=$RUN_DIR/train.pid
LOG_DIR=logs/$RUN
LOG=$LOG_DIR/train_$(date '+%Y%m%d-%H%M%S').log
LOG_LINK=train_${RUN}.log
mkdir -p "$RUN_DIR" "$LOG_DIR"

LAUNCH_PID_FILE=$RUN_DIR/launch.pid
LAUNCH_META=$RUN_DIR/launch.meta
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

echo "=== pre-flight ($RUN, $KIND) ==="

if [ -f "$LAUNCH_PID_FILE" ]; then
    OTHER=$(tr -dc '0-9' < "$LAUNCH_PID_FILE")
    if [ -n "$OTHER" ] && [ "$OTHER" != "$$" ] && kill -0 "$OTHER" 2>/dev/null; then
        echo "    ERROR: another launch of '$RUN' is already in pre-flight (pid $OTHER)." >&2
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

# One GPU. A Phase-2 session is the expensive thing on this machine and an audio run must
# never be what killed it -- so this refuses rather than sharing.
for other in checkpoints/*/train.pid; do
    [ -f "$other" ] || continue
    opid=$(tr -dc '0-9' < "$other" 2>/dev/null || true)
    orun=$(basename "$(dirname "$other")")
    if [ -n "$opid" ] && [ "$orun" != "$(basename "$RUN_DIR")" ] && kill -0 "$opid" 2>/dev/null; then
        echo "    ERROR: '$orun' is training as pid $opid and this would share the card." >&2
        echo "           stop it first:  scripts/stop.sh $orun" >&2
        exit 1
    fi
done

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
    $PY scripts/preflight_tests.py tests/ || { echo "tests failed -- fix before spending GPU hours" >&2; exit 1; }
fi

# ---- data -------------------------------------------------------------------------------
echo
echo "=== 1/3  data ==="
launch_stage data
CORPUS=$($PY - "$CFG" <<'EOF'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1])) or {}
print((cfg.get("data") or {}).get("corpus", ""))
EOF
)
if [ -n "$CORPUS" ] && [ ! -s "$CORPUS/audio.bin" ]; then
    if [ "$CORPUS" = "data/audio/synth" ]; then
        echo "    $CORPUS is missing. Building the synthetic corpus (~30 s)..."
        $PY -m aksharallm.audio corpus --out "$CORPUS" --clips 400
    else
        echo "    ERROR: $CORPUS/audio.bin does not exist." >&2
        echo "           build it:  $PY -m aksharallm.audio pack <wav-dir> --out $CORPUS" >&2
        echo "           or fetch:  $PY -m aksharallm.audio fetch ljspeech" >&2
        exit 1
    fi
fi
if [ "$KIND" = asr ]; then
    # A recogniser reads data.train (a list) and data.val, not data.corpus.
    $PY scripts/asr_data_check.py "$CFG" || exit 1
fi
if [ -n "$CORPUS" ]; then
    sz=$(stat -c%s "$CORPUS/audio.bin")
    echo "    $CORPUS/audio.bin: $((sz / 1000000)) MB = $((sz / 2 / 16000 / 60)) minutes at 16 kHz"
fi

# ---- smoke ------------------------------------------------------------------------------
echo
echo "=== 2/3  smoke test (30 steps) ==="
launch_stage smoke
if [ "$SKIP_SMOKE" = "1" ] && [ -s "$RUN_DIR/ckpt_last.pt" ]; then
    echo "    SKIP_SMOKE=1 and $RUN_DIR/ckpt_last.pt exists -- skipping (resume of a proven config)"
else
    [ "$SKIP_SMOKE" = "1" ] && echo "    SKIP_SMOKE=1 ignored: nothing to resume, so this is a first launch."
    echo "    check: the loss falls, and 'book' is NOT stuck near 1 -- codebook collapse is"
    echo "           invisible in the loss curve and fatal to everything downstream."
    SMOKE_EXTRA=(-o train.sample_every=0)
    [ "$KIND" = asr ] && SMOKE_EXTRA=()   # a recogniser writes no audio samples
    $PY -m $TRAINER "$CFG" \
        -o train.max_steps=30 -o train.out_dir=/tmp/aksharallm_audio_smoke \
        -o train.resume=null -o train.eval_every=0 "${SMOKE_EXTRA[@]}" \
        -o train.ckpt_every=0 -o train.log_every=10
fi

# ---- launch -----------------------------------------------------------------------------
echo
echo "=== 3/3  launching ==="
launch_stage launching
rm -f "$RUN_DIR/STOP"

EXTRA=()
[ -n "$STOP_AFTER" ] && EXTRA+=(-o "train.stop_after=$STOP_AFTER")
[ -n "$STOP_IN" ] && EXTRA+=(-o "train.stop_after_s=$STOP_IN_S")

# Unbuffered: redirected to a file, Python holds stdout in a 8 KB buffer, so the log -- and
# `tail -f`, and the startup check below, which greps it for the first step -- stayed EMPTY
# for minutes while the trainer was training. Found on the first LibriSpeech launch.
PYTHONUNBUFFERED=1 nohup $PY -m $TRAINER "$CFG" "${EXTRA[@]}" > "$LOG" 2>&1 &
PID=$!
ln -sfn "$LOG" "$LOG_LINK"

echo "$PID" > "$PID_FILE"
cat > "$RUN_DIR/run.meta" <<META
pid     $PID
started $(date '+%Y-%m-%d %H:%M:%S')
config  $CFG
log     $LOG
cmd     $PY -m $TRAINER $CFG ${EXTRA[*]-}
META
echo "$(date '+%Y-%m-%d %H:%M:%S')  pid $PID  $LOG" >> "$RUN_DIR/sessions.log"

# Watch the trainer until it has logged a training step, not for a fixed few seconds. A
# `sleep 5` here let a recogniser that crashed on RESUME -- ~30 s in, after loading 28k
# utterances -- print the success block below twice in one night, leaving a pid file naming a
# dead process and a dashboard still showing the previous run (gotcha 22, again: stage.sh had
# already been fixed for exactly this). CRASH_WINDOW caps the wait.
CRASH_WINDOW=${CRASH_WINDOW:-180}
started=0
for _ in $(seq "$CRASH_WINDOW"); do
    sleep 1
    if ! kill -0 "$PID" 2>/dev/null; then
        echo "    ERROR: the trainer died during startup. Last lines of $LOG:" >&2
        tail -20 "$LOG" >&2
        rm -f "$PID_FILE"
        exit 1
    fi
    if grep -qE '^step +[0-9]+ ' "$LOG" 2>/dev/null; then
        started=1
        break
    fi
done
[ "$started" = 1 ] && echo "    first step logged -- the trainer is past startup" \
    || echo "    still starting after ${CRASH_WINDOW}s (no step logged yet) -- watch the log"

echo "    pid $PID  ->  $PID_FILE  (config: $CFG)"
[ -n "$STOP_AFTER" ] && echo "    will stop itself after $STOP_AFTER steps"
[ -n "$STOP_IN" ] && echo "    will stop itself after ${STOP_IN_S}s of training"
echo "    log $LOG  ($LOG_LINK -> it)"
echo
echo "    watch:   tail -f $LOG_LINK"
echo "             scripts/portal.sh              (browser: the Audio tab)"
echo "    listen:  $RUN_DIR/samples/              (original.wav against stepNNNNNN.wav)"
echo "    stop:    scripts/stop.sh $(basename "$RUN_DIR")"
if [ "$KIND" = codec ]; then
    echo "    measure: $PY -m aksharallm.audio report $RUN_DIR/ckpt_best.pt"
elif [ "$KIND" = asr ]; then
    echo "    measure: $PY -m aksharallm.asr eval $(basename "$RUN_DIR") --corpus data/asr/test-clean"
fi
