---
id: quantization
title: Four bits per weight, and the group that collapses
doc: docs/10-quantization.md
files:
  - aksharallm/quant/qtensor.py
  - aksharallm/quant/qlinear.py
verify: tests/test_quant.py::test_the_zero_point_is_always_a_representable_code
prereqs: [training-loop]
minutes: 35
summary: Storing weights in 4 bits instead of 16, why the scale is per group of 64, and a real bug found by a test rather than by a measurement.
---

# 9. Four bits per weight, and the group that collapses

A trained weight is a 16-bit float. Most of that precision is not doing anything: the values
in any small slice of a matrix are clustered tightly, and what matters is where each one
sits *within that cluster*.

So store a cheap map instead. For each group of 64 weights, record the range, then keep each
weight as a small integer code:

```
group of 64 weights, all in [-0.08, 0.11]
scale      = (max - min) / 15          <- 4 bits = 16 codes
zero_point = the code that means 0.0
stored     = round((w - min) / scale)  <- a number from 0 to 15
```

Two 4-bit codes pack into one byte. A 599 MB model becomes 213 MB — and *not* 150 MB, because
the embedding table stays in float: it is a lookup, not a matmul, and quantizing it costs
accuracy while saving nothing at inference.

**Group size is the whole trade.** One scale for a whole matrix is cheap and terrible — a
single outlier stretches the range and every other weight loses resolution. One scale per 64
weights costs a little metadata and keeps the resolution where the values actually are.

## Smaller is not faster

Worth internalising, because it is the opposite of what everyone expects: at this scale, 4-bit
inference is *slower* than 16-bit unless you have a fused kernel, because every matmul now
has to unpack and rescale first. Quantization buys **memory**, not speed. `docs/10` measures
it honestly, including the fused Triton kernel that wins some of it back.

---

## Exercise: exclude zero from the range

This one is a real bug from this repo, and it was found by a *test*, not by a benchmark.

An asymmetric range is fitted to the group's actual `[min, max]`. If a group happens to live
entirely in `[1, 2]`, the zero point wants to be a code of `-15` — which is not representable,
gets clamped to 0, and then **every weight in the group saturates to a single code**. The
group is destroyed. The fix is to stretch the range to always include zero.

1. Run the check. It passes — it quantizes groups that never cross zero and asserts the
   zero-point is a code that actually exists, i.e. inside `[qmin, qmax]`.
2. In `aksharallm/quant/qtensor.py`, find where the range is stretched to include zero and
   remove that adjustment, fitting the raw min and max instead.
3. Run the check. **It should fail** for the groups that do not straddle zero.
4. Put it back. Green.

> **What you just saw, and why it is the best lesson here.** Re-running the full benchmark
> after the fix gave *identical* numbers — because trained weights are near zero-centred and
> almost every real group straddles zero anyway. The bug was invisible to measurement and
> obvious to a test that asked a question about a property rather than about an average.
> Benchmarks tell you how it performs; tests tell you what it means.
>
> **A confession, and the reason this lesson is worth doing.** The check above did not exist
> until this lesson was written. The exercise was tried against the suite as it stood, and
> **nothing went red** — the stretch had a paragraph of comment explaining why it was
> load-bearing and not one test holding it in place. So the test was written, and it is the
> one you just ran. An invariant with an explanation and no check is one edit away from
> being quietly untrue, and this is exactly how you find that out: try to break it on
> purpose and see whether anything notices.

## Measure it yourself

```bash
python -m aksharallm.quant checkpoints/tiny/ckpt_best.pt --compare
```

Every method and group size against the bf16 baseline on the same batches — the only honest
way to read a quantization number. A perplexity with nothing beside it means nothing.
