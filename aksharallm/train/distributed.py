"""Training on more than one GPU — and the four ways it is quietly wrong.

Data-parallel training is one idea: give every rank a *different* slice of the batch, let
each compute gradients on its own, average the gradients across ranks, and step. Every rank
holds a full copy of the model, so the parameters stay identical forever and the only thing
that moves between processes is a gradient.

```mermaid
flowchart LR
    D["one batch"] --> A["rank 0<br/>micro-batch A"]
    D --> B["rank 1<br/>micro-batch B"]
    A --> GA["grad A"]
    B --> GB["grad B"]
    GA --> R["all-reduce:<br/>every rank ends up with (A+B)/2"]
    GB --> R
    R --> S["identical step<br/>on every rank"]
```

**There is one GPU on this machine and there will not be a second one.** That is not a
reason to leave this untestable: the **gloo** backend runs the whole thing across CPU
processes, exercising the real code path — the process group, the rank split, the
all-reduce, the rank-0-only writes — with no CUDA anywhere. `tests/test_distributed.py`
spawns two processes and checks the arithmetic. What a second card would add is speed, not
coverage.

Four things that are wrong by default
--------------------------------------
**1. Every rank must see different data.** The obvious bug is not a crash: if two ranks draw
the same batch, the all-reduce averages two copies of one gradient, and you have bought
nothing but heat. The data loader is seeded `seed + rank`, and a test asserts two ranks draw
different batches from the same corpus.

**2. Gradient accumulation must not all-reduce every micro-batch.** DDP synchronises on
every `backward()` by default, so `grad_accum: 4` on 2 ranks does four all-reduces per step
where one would do. `no_sync()` on all but the last micro-step is the fix, and it is worth
a factor of `grad_accum` on the communication bill. The *result* is identical either way,
which is why nobody notices.

**3. The stop file must be decided by one rank and broadcast.** This is the one that hangs.
If rank 0 reads a STOP file a millisecond before rank 1 does, rank 0 leaves the loop and
rank 1 blocks forever inside the next all-reduce waiting for a peer that has gone. Any
per-rank decision — a stop, a NaN guard, an early exit — has to be turned into a collective
before it is acted on. `agree()` is that, and it is the function to reach for.

**4. The numbers scale, and reporting them unscaled is a lie.** `tokens_per_step` is
`batch_size × grad_accum × seq_len × world_size`; a loss printed from rank 0 alone is a
noisier estimate than the one that was actually stepped on. Both are handled here rather
than left for each trainer to remember.

Read with: docs/05-pretraining.md -- the chapter this implements; it ends with the order to
read these files in. See also docs/08-scaling.md.
"""

from __future__ import annotations

import contextlib
import os
from dataclasses import dataclass

import torch
import torch.distributed as dist

#: What `torchrun` sets. Their absence is the signal that this is an ordinary single-process
#: run, which must stay the default and must cost nothing.
_ENV = ("RANK", "WORLD_SIZE", "LOCAL_RANK")


@dataclass(frozen=True)
class Dist:
    """Where this process sits in the group. `world_size == 1` means "not distributed"."""

    rank: int = 0
    world_size: int = 1
    local_rank: int = 0
    backend: str = ""

    @property
    def enabled(self) -> bool:
        return self.world_size > 1

    @property
    def is_main(self) -> bool:
        """Rank 0. **The only rank that writes anything** — checkpoints, logs, the pid file.

        Not because the others have nothing to say, but because they would say the same
        thing at the same instant into the same file.
        """
        return self.rank == 0

    def describe(self) -> str:
        if not self.enabled:
            return "single process"
        return f"rank {self.rank}/{self.world_size} on {self.backend}"


def from_env() -> Dist:
    """Read `torchrun`'s environment. Returns a single-process `Dist` when it is absent."""
    if not all(k in os.environ for k in ("RANK", "WORLD_SIZE")):
        return Dist()
    return Dist(
        rank=int(os.environ["RANK"]),
        world_size=int(os.environ["WORLD_SIZE"]),
        local_rank=int(os.environ.get("LOCAL_RANK", 0)),
    )


def setup(device: str = "cuda") -> Dist:
    """Join the process group if `torchrun` started us. Idempotent, and free when it did not.

    The backend follows the device: **nccl** for CUDA (it talks GPU to GPU, over NVLink or
    PCIe, without going through host memory) and **gloo** for CPU. Gloo is not a fallback —
    it is how this code is tested here, where there is one card.
    """
    info = from_env()
    if not info.enabled:
        return info
    backend = "nccl" if device.startswith("cuda") and torch.cuda.is_available() else "gloo"
    if not dist.is_initialized():
        dist.init_process_group(backend=backend)
    if backend == "nccl":
        # Pin this process to its own card before anything allocates, or every rank piles
        # onto cuda:0 and the run OOMs at a memory total that looks impossible.
        torch.cuda.set_device(info.local_rank)
    return Dist(info.rank, info.world_size, info.local_rank, backend)


def teardown() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def wrap(model, info: Dist):
    """`DistributedDataParallel`, or the model unchanged on one process.

    Returns `(wrapped, inner)` — the wrapped module to run and the original to save from.
    Saving a DDP-wrapped `state_dict()` prefixes every key with `module.`, which loads
    nowhere and is discovered days later by someone trying to use the checkpoint.
    """
    if not info.enabled:
        return model, model
    from torch.nn.parallel import DistributedDataParallel

    ids = [info.local_rank] if info.backend == "nccl" else None
    return DistributedDataParallel(model, device_ids=ids), model


def unwrap(model):
    """The bare `Transformer` inside whatever wrappers are on it.

    Two can be stacked: `torch.compile` hides the module behind `_orig_mod`, and DDP behind
    `module`. **DDP wraps the compiled model, so `module` comes off first.** Anything that
    reaches for a method the wrappers do not forward — `estimate_mfu`, `moe_aux_loss`,
    `state_dict` for a checkpoint — has to go through here, and the failure mode is an
    `AttributeError` at the first log line rather than anything subtle.
    """
    model = getattr(model, "module", model)
    return getattr(model, "_orig_mod", model)


def device_for(info: Dist, requested: str) -> str:
    """`cuda:LOCAL_RANK` under nccl, so two ranks do not share one card."""
    if info.enabled and info.backend == "nccl":
        return f"cuda:{info.local_rank}"
    return requested


# ---------------------------------------------------------------------------------------
# the collectives that keep ranks in step
# ---------------------------------------------------------------------------------------


def agree(flag: bool, info: Dist) -> bool:
    """True on **every** rank if it is true on **any**. The deadlock guard.

    Any decision taken per-rank — stop now, this loss is NaN, the budget is spent — must
    become a collective before it is acted on. Without this, one rank leaves the training
    loop and every other rank blocks forever inside the next all-reduce waiting for a peer
    that has exited. The symptom is a run that is not dead, not progressing, and shows no
    error at all.

    `any` rather than `all` is deliberate: a stop request that only rank 0 can see is still a
    stop request, and the safe reading of disagreement is to stop.
    """
    if not info.enabled:
        return flag
    t = torch.tensor([1.0 if flag else 0.0])
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return bool(t.item() > 0)


def mean(value: float, info: Dist) -> float:
    """The average of a scalar across ranks — for logging a loss that means something.

    Rank 0's loss is computed on `1/world_size` of the batch that was actually stepped on,
    so printing it unaveraged reports a noisier number than the one the optimizer saw. It
    is not *wrong*, exactly, and it is not what happened either.
    """
    if not info.enabled:
        return value
    t = torch.tensor([value], dtype=torch.float64)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return float(t.item() / info.world_size)


def no_sync(model, info: Dist, last: bool):
    """Suppress DDP's all-reduce on every micro-batch but the last of an accumulation.

    DDP synchronises inside `backward()`, so `grad_accum: 4` costs four all-reduces per
    optimizer step where one would do. The gradients are identical either way — accumulation
    is a sum and averaging commutes with it — so the only symptom of getting this wrong is
    that the run is slower than it should be, by a factor of `grad_accum` on communication.
    """
    if not info.enabled or last or not hasattr(model, "no_sync"):
        return contextlib.nullcontext()
    return model.no_sync()


def tokens_per_step(batch_size: int, grad_accum: int, seq_len: int, info: Dist) -> int:
    """The **global** batch. Every rank contributes a full micro-batch every micro-step.

    Reporting the per-rank figure makes throughput, the token budget, the ETA and the cost
    per million tokens all wrong by exactly `world_size`, in the flattering direction.
    """
    return batch_size * grad_accum * seq_len * info.world_size


def barrier(info: Dist) -> None:
    """Wait for every rank. Used once, after rank 0 has written a checkpoint, so no rank
    races ahead and starts a save while the previous one is half-written."""
    if info.enabled:
        dist.barrier()
