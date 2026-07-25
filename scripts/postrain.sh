#!/usr/bin/env bash
# Phase 3: post-training. Turns a base model into a chat assistant.
#
#   SFT ~2 hours, DPO ~3 hours on an RTX 3090.
#
# Usage:  scripts/postrain.sh checkpoints/small/ckpt_best.pt data/fineweb/tokenizer.json
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-.venv/bin/python}
BASE=${1:?usage: postrain.sh <base_checkpoint> <tokenizer.json>}
TOK=${2:?usage: postrain.sh <base_checkpoint> <tokenizer.json>}
SEQ=${SEQ:-1024}

echo "=== 1/4  preparing SFT data (SmolTalk) ==="
[ -s data/sft/train_tokens.npy ] || $PY -m aksharallm.data.prepare_sft smoltalk \
    --tokenizer "$TOK" --out-dir data/sft --seq-len "$SEQ"

echo
echo "=== 2/4  supervised fine-tuning ==="
$PY -m aksharallm.train.sft \
    --base "$BASE" --data-dir data/sft --tokenizer "$TOK" \
    --out-dir checkpoints/sft --epochs 2 --lr 1e-5

echo
echo "=== 3/4  preparing preference data (UltraFeedback) ==="
[ -s data/dpo/train_chosen_tokens.npy ] || $PY -m aksharallm.data.prepare_dpo ultrafeedback \
    --tokenizer "$TOK" --out-dir data/dpo --seq-len "$SEQ"

echo
echo "=== 4/4  DPO ==="
$PY -m aksharallm.train.dpo \
    --sft checkpoints/sft/sft_best.pt --data-dir data/dpo --tokenizer "$TOK" \
    --out-dir checkpoints/dpo --beta 0.1 --lr 5e-7

echo
echo "done. chat with it:"
echo "  $PY -m aksharallm.infer.cli checkpoints/dpo/dpo_best.pt --mode chat"
