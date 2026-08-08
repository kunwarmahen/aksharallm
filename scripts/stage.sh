#!/usr/bin/env bash
# Post-training in one command: SFT, DPO, or GRPO on a base model.
#
#   scripts/stage.sh sft   [base_run]     # base -> chat model     (needs a base checkpoint)
#   scripts/stage.sh dpo   [base_run]     # align it               (needs an SFT checkpoint)
#   scripts/stage.sh grpo  [base_run]     # RL on the code sandbox (needs an SFT checkpoint)
#
# base_run defaults to `small-code`. This is the post-training twin of `scripts/phase2.sh`:
# it prepares the stage's data if missing, enforces the prerequisite (you cannot run GRPO
# without an SFT model), launches the trainer detached, and writes the SAME pid/meta/log
# files phase2.sh writes -- so `scripts/stop.sh` and the portal drive it with no special
# cases. "Everything through scripts": the portal only ever shells out to this.
#
# The dependency chain it enforces:
#     base (ckpt_best.pt) --> SFT (sft_best.pt) --> DPO / GRPO
#
# Env knobs:
#   TOK=path/to/tokenizer.json   override the tokenizer (else inferred from base_run)
#   DATA=smoltalk|ultrafeedback  override the dataset recipe
#   SEQ=1024   EPOCHS=2   LR=...   extra trainer args passed through
#   BS=8  ACCUM=8                  SFT micro-batch and accumulation (BS*ACCUM*SEQ = tokens/step)
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-.venv/bin/python}
STAGE=${1:-}
BASE_RUN=${2:-small-code}
SEQ=${SEQ:-1024}

case "$STAGE" in
    sft|dpo|grpo) ;;
    *) echo "usage: scripts/stage.sh <sft|dpo|grpo> [base_run]" >&2; exit 2 ;;
esac

# ---- resolve paths -----------------------------------------------------------------------
BASE_CKPT=checkpoints/$BASE_RUN/ckpt_best.pt
SFT_RUN=$BASE_RUN-sft
SFT_CKPT=checkpoints/$SFT_RUN/sft_best.pt
RUN=$BASE_RUN-$STAGE                 # e.g. small-code-grpo
RUN_DIR=checkpoints/$RUN
LOG_DIR=logs/$RUN
LOG=$LOG_DIR/${STAGE}_$(date '+%Y%m%d-%H%M%S').log
LOG_LINK=train_${RUN}.log
PID_FILE=$RUN_DIR/train.pid
# The same path scripts/stop.sh writes and the portal's Stop button drives. Post-training
# stages get their own run directory, so the plain name is unambiguous here — the trainers'
# own default of <stage>_STOP exists only for the LoRA path, which writes into the *base*
# run's directory where a pretraining run would read a file called STOP.
STOP_FILE=$RUN_DIR/STOP
LAUNCH_PID_FILE=$RUN_DIR/launch.pid
LAUNCH_META=$RUN_DIR/launch.meta

# tokenizer: inferred from the base run unless TOK is set. Resolved BEFORE we create any
# directories, so an unknown base fails without littering checkpoints/ with an empty dir.
if [ -z "${TOK:-}" ]; then
    case "$BASE_RUN" in
        small-code) TOK=data/blend/tokenizer.json ;;
        small)      TOK=data/fineweb/tokenizer.json ;;
        tiny)       TOK=data/tinystories/tokenizer.json ;;
        *) echo "ERROR: cannot infer tokenizer for base '$BASE_RUN'; set TOK=..." >&2; exit 1 ;;
    esac
fi
mkdir -p "$RUN_DIR" "$LOG_DIR"

# ---- publish this launch (same contract as phase2.sh) ------------------------------------
STARTED=$(date '+%Y-%m-%d %H:%M:%S')
launch_stage() {
    cat > "$LAUNCH_META" <<META
pid     $$
stage   $1
started $STARTED
config  $STAGE on $BASE_RUN
log     ${LAUNCH_LOG:-(terminal)}
META
}
trap 'rm -f "$LAUNCH_PID_FILE"' EXIT
trap 'echo "[abort] $STAGE launch cancelled"; exit 130' TERM INT

echo "=== $STAGE on $BASE_RUN (-> $RUN) ==="

# refuse a second launch of this stage
if [ -f "$LAUNCH_PID_FILE" ]; then
    OTHER=$(tr -dc '0-9' < "$LAUNCH_PID_FILE")
    if [ -n "$OTHER" ] && [ "$OTHER" != "$$" ] && kill -0 "$OTHER" 2>/dev/null; then
        echo "    ERROR: a launch of '$RUN' is already in pre-flight (pid $OTHER)." >&2
        trap - EXIT; exit 1
    fi
fi
if [ -f "$PID_FILE" ] && kill -0 "$(tr -dc '0-9' < "$PID_FILE" 2>/dev/null)" 2>/dev/null; then
    echo "    ERROR: '$RUN' is already training as pid $(cat "$PID_FILE")." >&2
    echo "           stop it:  scripts/stop.sh $RUN" >&2
    exit 1
fi
rm -f "$PID_FILE"
# A STOP left over from the last run would end this one at step 0, and that reads as a
# broken launcher rather than a stale file. phase2.sh clears it for the same reason.
if [ -f "$STOP_FILE" ]; then
    echo "    cleared a stale STOP from the previous run."
    rm -f "$STOP_FILE"
fi
echo "$$" > "$LAUNCH_PID_FILE"
launch_stage preflight

# ---- gating: the prerequisite must exist ------------------------------------------------
require() {   # require <file> <human message>
    if [ ! -s "$1" ]; then
        echo "    ERROR: missing prerequisite: $1" >&2
        echo "           $2" >&2
        exit 1
    fi
}
case "$STAGE" in
    sft)  require "$BASE_CKPT" "train a base model first (scripts/phase2.sh)." ;;
    dpo)  require "$SFT_CKPT"  "run SFT first: scripts/stage.sh sft $BASE_RUN" ;;
    grpo) require "$SFT_CKPT"  "run SFT first: scripts/stage.sh sft $BASE_RUN" ;;
esac

# ---- prepare the stage's data (if missing) and build the trainer command ----------------
launch_stage data
case "$STAGE" in
    sft)
        DATA=${DATA:-smoltalk}
        [ -s data/sft/train_tokens.npy ] || $PY -m aksharallm.data.prepare_sft "$DATA" \
            --tokenizer "$TOK" --out-dir data/sft --seq-len "$SEQ"
        # The trainer's own default (16 x 4) was sized for the tiny models and OOMs a 300M
        # model on a 24 GB card: 16 x 1024 of activations on top of AdamW's fp32 states
        # leaves nothing, and it dies in the first forward pass. 8 x 8 is the same 65,536
        # tokens/step, measured at ~21 GB peak on a 3090. Raise BS on a bigger card.
        CMD=($PY -m aksharallm.train.sft --base "$BASE_CKPT" --data-dir data/sft
             --tokenizer "$TOK" --out-dir "$RUN_DIR" --epochs "${EPOCHS:-2}" --lr "${LR:-1e-5}"
             --batch-size "${BS:-8}" --grad-accum "${ACCUM:-8}"
             --stop-file "$STOP_FILE" --resume "${RESUME:-auto}")
        ;;
    dpo)
        DATA=${DATA:-ultrafeedback}
        [ -s data/dpo/train_chosen_tokens.npy ] || $PY -m aksharallm.data.prepare_dpo "$DATA" \
            --tokenizer "$TOK" --out-dir data/dpo --seq-len "$SEQ"
        CMD=($PY -m aksharallm.train.dpo --sft "$SFT_CKPT" --data-dir data/dpo
             --tokenizer "$TOK" --out-dir "$RUN_DIR" --beta "${BETA:-0.1}" --lr "${LR:-5e-7}"
             --stop-file "$STOP_FILE")
        ;;
    grpo)
        # Code reward uses the built-in sandbox tasks -- no dataset to prepare.
        CMD=($PY -m aksharallm.train.grpo --init "$SFT_CKPT" --tokenizer "$TOK"
             --out-dir "$RUN_DIR" --reward "${REWARD:-code}" --group-size "${GROUP:-8}"
             --lr "${LR:-1e-6}" --steps "${STEPS:-500}"
             --stop-file "$STOP_FILE")
        ;;
esac

# ---- launch detached, record it, catch an immediate crash -------------------------------
launch_stage launching
nohup "${CMD[@]}" > "$LOG" 2>&1 &
PID=$!
ln -sfn "$LOG" "$LOG_LINK"
echo "$PID" > "$PID_FILE"
cat > "$RUN_DIR/run.meta" <<META
pid     $PID
started $(date '+%Y-%m-%d %H:%M:%S')
config  $STAGE on $BASE_RUN
log     $LOG
cmd     ${CMD[*]}
META
echo "$(date '+%Y-%m-%d %H:%M:%S')  pid $PID  $LOG" >> "$RUN_DIR/sessions.log"

# Watch it long enough to catch the crashes that happen on the way up. Five seconds used to
# be the window, and an OOM in the first forward pass landed just past it: the script
# declared success, left a pid file behind for a dead process, and the portal quietly went
# back to "ready" with the traceback unread. Allocating the model, compiling and reaching
# step 1 takes ~20s on the 300M model, so watch for 30 and stop as soon as it dies.
CRASH_WINDOW=${CRASH_WINDOW:-30}
for _ in $(seq "$CRASH_WINDOW"); do
    sleep 1
    if ! kill -0 "$PID" 2>/dev/null; then
        echo "    ERROR: the $STAGE trainer died during startup. Last lines of $LOG:" >&2
        tail -20 "$LOG" >&2
        rm -f "$PID_FILE"
        exit 1
    fi
done

echo "    pid $PID  ->  $PID_FILE"
echo "    watch:   tail -f $LOG_LINK"
echo "    stop:    scripts/stop.sh $RUN"
echo "    result:  checkpoints/$RUN/${STAGE}_best.pt"
