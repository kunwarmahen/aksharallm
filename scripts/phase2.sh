#!/usr/bin/env bash
# Phase 2: the real base model. ~300M params, ~10B tokens, ~6 days on a 3090.
#
# Default = the BLENDED base: 85% FineWeb-Edu + 15% Python, so one run yields a base that
# both chats (after SFT/DPO) and codes (after Python continued-pretraining).
# Set PURE=1 for the non-blended FineWeb-Edu-only fallback.
#
# Run this ONCE, after Phase 1 works. It pre-flights, builds data, smoke-tests, then
# launches in the background. Safe to interrupt (kill / STOP-file); re-run to resume.
#
#   data prep   ~2-4 hours (network bound)
#   pretraining ~6 days
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-.venv/bin/python}
NEED_GB=25
PURE=${PURE:-0}

if [ "$PURE" = "1" ]; then
    CFG=configs/small.yaml;      RUN=small
else
    CFG=configs/small-code.yaml; RUN=small-code
fi

echo "=== pre-flight ($([ "$PURE" = 1 ] && echo 'pure FineWeb-Edu' || echo 'blended 85/15 general+Python')) ==="
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
nohup $PY -m aksharallm.train.pretrain "$CFG" > "train_${RUN}.log" 2>&1 &
PID=$!
echo "    pid $PID  (config: $CFG)"
echo
echo "    watch:   tail -f train_${RUN}.log"
echo "    stop:    kill $PID        (or: touch checkpoints/${RUN}/STOP)"
echo "    resume:  re-run this script -- it picks up exactly where it left off"
echo
echo "    Stopping is safe at any time: it saves at the current step and the resumed"
echo "    run continues with no loss spike. A crash costs at most ~20 min (ckpt_every)."
