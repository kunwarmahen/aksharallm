#!/usr/bin/env bash
# Phase 2: the real base model. ~300M params, 10B tokens of FineWeb-Edu.
#
#   data prep   ~2-4 hours (network bound)
#   pretraining ~6 days
#
# Safe to interrupt and re-run at any point -- both stages resume.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-.venv/bin/python}
NEED_GB=25

echo "=== pre-flight ==="
FREE_GB=$(df -BG --output=avail . | tail -1 | tr -dc '0-9')
echo "    free disk: ${FREE_GB} GB (need ~${NEED_GB} GB)"
if [ "$FREE_GB" -lt "$NEED_GB" ]; then
    echo "    ERROR: not enough disk. Free some space or lower --max-train-tokens." >&2
    exit 1
fi
$PY -m pytest tests/ -q || { echo "tests failed -- fix before a 6-day run" >&2; exit 1; }

echo
echo "=== 1/3  building FineWeb-Edu token files (~20 GB) ==="
if [ -s data/fineweb/train.bin ]; then
    echo "    already present ($(du -h data/fineweb/train.bin | cut -f1)), skipping"
else
    $PY -m aksharallm.data.prepare fineweb-edu-10bt \
        --out-dir data/fineweb \
        --vocab-size 32768 \
        --val-tokens 10000000 \
        --max-train-tokens 10000000000
fi

ACTUAL=$(stat -c%s data/fineweb/train.bin)
echo "    train.bin: $((ACTUAL / 1000000000)) GB / $((ACTUAL / 2)) tokens"
if [ "$ACTUAL" -lt 1000000000 ]; then
    echo "    ERROR: train.bin is far smaller than expected. Do not train on this." >&2
    exit 1
fi

echo
echo "=== 2/3  smoke test (50 steps) ==="
echo "    check: step-0 loss ~10.4, MFU > 35%, memory stable"
$PY -m aksharallm.train.pretrain configs/small.yaml \
    -o train.max_steps=50 -o train.out_dir=/tmp/aksharallm_smoke -o train.resume=null

echo
echo "=== 3/3  launching the real run ==="
nohup $PY -m aksharallm.train.pretrain configs/small.yaml > train_small.log 2>&1 &
PID=$!
echo "    pid $PID"
echo
echo "    watch:   tail -f train_small.log"
echo "    stop:    kill $PID        (or: touch checkpoints/small/STOP)"
echo "    resume:  re-run this script -- it picks up exactly where it left off"
echo
echo "    Stopping is safe at any time: it saves at the current step and the resumed"
echo "    run continues with no loss spike. A crash costs at most ~20 min (ckpt_every)."
