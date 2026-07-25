#!/usr/bin/env bash
# Phase 1 end to end: data -> pretrain -> generate. ~30 minutes on an RTX 3090.
# This is the "does everything work?" script. Run it after any change.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-.venv/bin/python}

echo "=== 1/3  building TinyStories token files ==="
if [ -s data/tinystories/train.bin ]; then
    echo "    already present ($(du -h data/tinystories/train.bin | cut -f1)), skipping"
else
    $PY -m aksharallm.data.prepare tinystories \
        --out-dir data/tinystories \
        --vocab-size 8192 \
        --val-tokens 5000000 \
        --max-train-tokens 400000000
fi

echo
echo "=== 2/3  pretraining (resumes automatically if interrupted) ==="
$PY -m aksharallm.train.pretrain configs/tiny.yaml

echo
echo "=== 3/3  generating from the best checkpoint ==="
$PY -m aksharallm.infer.cli checkpoints/tiny/ckpt_best.pt \
    --prompt "Once upon a time" --max-new-tokens 150

echo
echo "done. chat with it:"
echo "  $PY -m aksharallm.infer.cli checkpoints/tiny/ckpt_best.pt"
