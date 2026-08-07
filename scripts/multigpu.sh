#!/usr/bin/env bash
# Launch a training run across several GPUs -- or across CPU processes, to test the path.
#
# There is one card in this machine. That is not a reason for the distributed path to be
# untested: the gloo backend runs the whole thing across CPU processes, exercising the
# process group, the rank split, the all-reduce and the rank-0-only writes with no CUDA
# anywhere. What a second card adds is speed, not coverage.
#
#   scripts/multigpu.sh 2 configs/tiny.yaml                    # 2 GPUs
#   NPROC=2 DEVICE=cpu scripts/multigpu.sh 2 configs/tiny.yaml # 2 CPU processes (gloo)
#
# Everything else -- the STOP file, resume, the session log -- works unchanged, because
# `train/distributed.py` is a no-op when torchrun has not set RANK/WORLD_SIZE.
#
# NOTE: the per-rank batch_size stays what the config says. The GLOBAL batch is
# `batch_size x grad_accum x seq_len x world_size`, which is what the trainer reports and
# what the token budget is spent against -- so N GPUs finish `train.max_steps` in 1/N the
# time having seen N times the tokens. Halve `grad_accum` to keep the global batch fixed.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-.venv/bin/python}
NPROC=${1:-${NPROC:-2}}
CFG=${2:-configs/tiny.yaml}
shift 2 2>/dev/null || true

[ -f "$CFG" ] || { echo "no such config: $CFG" >&2; exit 2; }
if ! [[ $NPROC =~ ^[0-9]+$ ]] || [ "$NPROC" -lt 1 ]; then
    echo "usage: scripts/multigpu.sh <n_processes> <config> [-o key=value ...]" >&2
    exit 2
fi

HAVE=$($PY -c "import torch; print(torch.cuda.device_count())")
if [ "${DEVICE:-}" != "cpu" ] && [ "$NPROC" -gt "$HAVE" ]; then
    echo "ERROR: asked for $NPROC processes but this machine has $HAVE CUDA device(s)." >&2
    echo "       Two ranks sharing one card is slower than one rank using it -- they" >&2
    echo "       serialise on the same SMs and each holds its own copy of the model." >&2
    echo "       To exercise the distributed path anyway:  DEVICE=cpu $0 $NPROC $CFG" >&2
    exit 1
fi

[ "${DEVICE:-}" = "cpu" ] && export CUDA_VISIBLE_DEVICES=""

echo "=== $NPROC processes, $CFG, ${DEVICE:-cuda} ==="
exec $PY -m torch.distributed.run --nproc_per_node="$NPROC" --standalone \
    -m aksharallm.train.pretrain "$CFG" "$@"
