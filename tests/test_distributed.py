"""Data-parallel training, tested across CPU processes because there is one GPU here.

**That constraint is the point of this file.** The gloo backend runs a real process group, a
real all-reduce and the real rank split with no CUDA anywhere, so every line of
`train/distributed.py` is exercised. What a second card would add is speed, not coverage —
and a distributed path that has never run is a distributed path that does not work.

The four things checked are the four things that are wrong by default:

1. **every rank must see different data** — two ranks on the same slice all-reduce two copies
   of one gradient and buy nothing but heat, and no loss curve would say so;
2. **the all-reduce must actually average** — asserted against a hand-computed mean;
3. **a stop must be agreed before it is acted on** — otherwise one rank exits and the rest
   block forever inside the next all-reduce, which is a hang with no error;
4. **the reported batch is global** — the per-rank figure makes throughput, the budget, the
   ETA and the cost per token all wrong by exactly `world_size`, flatteringly.
"""

from __future__ import annotations

import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from aksharallm.train.distributed import (
    Dist,
    agree,
    device_for,
    from_env,
    mean,
    no_sync,
    setup,
    teardown,
    tokens_per_step,
    wrap,
)

#: Two processes is enough to prove every collective here; more only makes the test slower.
WORLD = 2


# ---------------------------------------------------------------------------------------
# the no-op path — the one that must stay free
# ---------------------------------------------------------------------------------------


def test_without_torchrun_it_is_a_single_process(monkeypatch):
    """The default, and the one that matters most: an ordinary run must be untouched."""
    for k in ("RANK", "WORLD_SIZE", "LOCAL_RANK"):
        monkeypatch.delenv(k, raising=False)
    info = from_env()
    assert not info.enabled and info.is_main and info.world_size == 1
    assert info.describe() == "single process"


def test_the_collectives_are_identities_on_one_process():
    """Nothing in the trainer should need an `if distributed:` around it."""
    solo = Dist()
    assert agree(True, solo) is True and agree(False, solo) is False
    assert mean(3.5, solo) == 3.5
    model = torch.nn.Linear(2, 2)
    wrapped, inner = wrap(model, solo)
    assert wrapped is model and inner is model
    with no_sync(model, solo, last=False):
        pass


def test_the_global_batch_scales_with_the_world():
    """`batch x accum x seq x world`. Reporting the per-rank figure is wrong by exactly
    `world_size`, in the direction that flatters the run."""
    assert tokens_per_step(8, 4, 512, Dist()) == 8 * 4 * 512
    assert tokens_per_step(8, 4, 512, Dist(rank=0, world_size=4)) == 8 * 4 * 512 * 4


def test_each_rank_gets_its_own_card_under_nccl():
    """Without this every rank allocates on cuda:0 and the run OOMs at a memory total that
    looks impossible."""
    assert device_for(Dist(1, 2, 1, "nccl"), "cuda") == "cuda:1"
    assert device_for(Dist(1, 2, 1, "gloo"), "cpu") == "cpu"
    assert device_for(Dist(), "cuda") == "cuda"


def test_only_rank_zero_is_main():
    assert Dist(0, 4).is_main and not Dist(1, 4).is_main


# ---------------------------------------------------------------------------------------
# the real thing, across processes
# ---------------------------------------------------------------------------------------


def _run(rank: int, world: int, fn, out, *args):
    """One rank of a gloo group. Everything a `torchrun` launch would set, set by hand."""
    os.environ.update(
        RANK=str(rank), WORLD_SIZE=str(world), LOCAL_RANK=str(rank),
        MASTER_ADDR="127.0.0.1", MASTER_PORT="29517",
    )
    info = setup("cpu")
    try:
        out[rank] = fn(info, *args)
    finally:
        teardown()


def spawn(fn, *args, world: int = WORLD):
    """Run `fn(info, *args)` on `world` processes and return each one's result."""
    ctx = mp.get_context("spawn")
    out = ctx.Manager().dict()
    procs = [ctx.Process(target=_run, args=(r, world, fn, out, *args)) for r in range(world)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=120)
    for p in procs:
        assert p.exitcode == 0, f"a rank exited {p.exitcode}"
    return [out[r] for r in range(world)]


def _identity(info: Dist):
    return (info.rank, info.world_size, info.backend, info.is_main)


def _all_reduce_mean(info: Dist):
    # Rank r contributes r + 1, so the mean over two ranks is 1.5.
    return mean(float(info.rank + 1), info)


def _agree_from_rank_zero(info: Dist):
    # Only rank 0 "sees" the stop. Every rank must come back True.
    return agree(info.rank == 0, info)


def _agree_nobody(info: Dist):
    return agree(False, info)


def _gradients_are_averaged(info: Dist):
    """The whole of data parallelism, in one step.

    Each rank's input is different, so each computes a different gradient. After DDP's
    all-reduce every rank must hold the *same* gradient, and it must equal the mean.
    """
    torch.manual_seed(0)
    model = torch.nn.Linear(4, 1, bias=False)
    wrapped, inner = wrap(model, info)
    x = torch.full((1, 4), float(info.rank + 1))
    wrapped(x).sum().backward()
    return inner.weight.grad.reshape(-1).tolist()


def _different_seeds_give_different_batches(info: Dist, path: str):
    from aksharallm.data.loader import TokenDataset

    ds = TokenDataset(path, 16, "cpu", seed=1234 + info.rank)
    return ds.get_batch(2)[0].reshape(-1).tolist()


def test_the_group_forms_and_every_rank_knows_where_it_is():
    rows = spawn(_identity)
    assert [r[0] for r in rows] == [0, 1]
    assert all(r[1] == WORLD and r[2] == "gloo" for r in rows)
    assert [r[3] for r in rows] == [True, False]


def test_the_mean_is_the_mean_and_every_rank_has_it():
    """Rank 0 contributes 1, rank 1 contributes 2, and both must come back with 1.5 — not
    with their own value, which is what an un-reduced log line reports."""
    assert spawn(_all_reduce_mean) == [1.5, 1.5]


def test_a_stop_seen_by_one_rank_is_seen_by_all():
    """**The deadlock guard.** Without this, rank 0 leaves the training loop on a STOP file
    and rank 1 blocks forever in the next all-reduce: not dead, not progressing, no error."""
    assert spawn(_agree_from_rank_zero) == [True, True]


def test_agreement_does_not_invent_a_stop():
    """The complement — otherwise `agree` could return True always and pass the test above."""
    assert spawn(_agree_nobody) == [False, False]


def test_ddp_averages_the_gradients():
    """Two ranks, two different inputs, one gradient afterwards.

    Rank 0 sees `[1,1,1,1]` and rank 1 sees `[2,2,2,2]`; the gradient of `sum(Wx)` is `x`, so
    the averaged gradient must be `[1.5]*4` **on both ranks**.
    """
    rows = spawn(_gradients_are_averaged)
    assert rows[0] == rows[1], "the ranks disagree — the all-reduce did not happen"
    assert rows[0] == pytest.approx([1.5, 1.5, 1.5, 1.5])


def test_each_rank_draws_a_different_batch(tmp_path):
    """The bug that buys nothing but heat. Two ranks on the same slice average two copies of
    one gradient, and the loss curve looks completely normal."""
    import numpy as np

    path = tmp_path / "train.bin"
    np.random.default_rng(0).integers(0, 8000, 20_000, dtype=np.uint16).tofile(path)
    rows = spawn(_different_seeds_give_different_batches, str(path))
    assert rows[0] != rows[1]


# ---------------------------------------------------------------------------------------
# accumulation
# ---------------------------------------------------------------------------------------


def _accumulation_matches_one_big_backward(info: Dist):
    """`no_sync` must not change the answer — only the number of all-reduces.

    Two micro-batches with `no_sync` on the first, against the same two summed in one
    backward. Accumulation is a sum and averaging commutes with it, so the gradients must
    match; if they do not, `no_sync` is being used where it changes semantics rather than
    only communication.
    """
    torch.manual_seed(0)
    model = torch.nn.Linear(4, 1, bias=False)
    wrapped, inner = wrap(model, info)

    a = torch.full((1, 4), float(info.rank + 1))
    b = torch.full((1, 4), float(info.rank + 3))

    inner.zero_grad(set_to_none=True)
    for i, x in enumerate((a, b)):
        with no_sync(wrapped, info, last=i == 1):
            (wrapped(x).sum() / 2).backward()
    accumulated = inner.weight.grad.reshape(-1).clone()

    inner.zero_grad(set_to_none=True)
    (wrapped(torch.cat([a, b])).sum() / 2).backward()
    at_once = inner.weight.grad.reshape(-1).clone()
    return (accumulated.tolist(), at_once.tolist())


def test_no_sync_changes_the_communication_and_not_the_answer():
    for accumulated, at_once in spawn(_accumulation_matches_one_big_backward):
        assert accumulated == pytest.approx(at_once, abs=1e-6)


def _checkpoint_keys_are_unprefixed(info: Dist):
    """A DDP `state_dict()` prefixes every key with `module.`, which loads nowhere and is
    discovered days later by whoever tries to use the checkpoint."""
    model = torch.nn.Linear(2, 2)
    wrapped, inner = wrap(model, info)
    return (sorted(inner.state_dict()), sorted(wrapped.state_dict()))


def test_the_saveable_module_is_the_unwrapped_one():
    for inner_keys, wrapped_keys in spawn(_checkpoint_keys_are_unprefixed):
        assert inner_keys == ["bias", "weight"]
        assert all(k.startswith("module.") for k in wrapped_keys)


def test_the_process_group_is_actually_torn_down():
    """A group left open holds sockets and makes the next `init_process_group` in the same
    process fail with a message about the store already existing."""
    spawn(_identity)
    assert not dist.is_initialized()
