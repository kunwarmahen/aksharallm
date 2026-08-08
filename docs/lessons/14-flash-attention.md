---
id: flash-attention
title: Writing the kernel, and the decision that makes it safe to ship
doc: docs/04-model.md
files:
  - aksharallm/model/flash.py
  - aksharallm/model/transformer.py
verify: tests/test_flash.py::test_the_model_accepts_the_flash_impl_on_any_device
prereqs: [attention]
minutes: 35
summary: Attention without ever building the score matrix — and why the interesting line is not in the kernel but in the routing decision in front of it.
---

# 14. Writing the kernel, and the decision that makes it safe to ship

[Lesson 3](docs/lessons/03-attention.md) built attention as three steps: compute `S = QK^T`, softmax it,
multiply by `V`. That middle matrix is the problem. At 4,096 tokens with 16 heads it is
**4 GB** in fp32, and every byte of it is written to memory and read back twice.

FlashAttention never builds it. It walks the keys in blocks and keeps three small running
numbers per query row:

```
m   the largest score seen so far
l   the sum of exp(score - m) so far
o   the weighted sum of values so far, expressed relative to that same m
```

When a block arrives with a larger maximum, the two accumulators are **rescaled** by
`exp(m_old - m_new)` and the new block folded in. That is the *online softmax*, and it is
the whole trick — it makes softmax associative, so the sum can be computed in one pass over
blocks that never leave the chip.

```
exp(s - m_new) = exp(s - m_old) · exp(m_old - m_new)
```

The output is **exact**. Not an approximation, not a windowed attention — the same numbers,
on a different memory schedule. `aksharallm/model/flash.py` is that, ours, in Triton, forward
and backward.

## What it is actually worth

| | our kernel vs `F.scaled_dot_product_attention` |
|---|---|
| forward | parity from T=2048 (1.02x) |
| backward | ~20% behind, by design — two kernels, scores recomputed twice, no atomics |
| end to end on the 300M | 51.6% → **50.5%** MFU |
| memory at T=8192 | **422 MB**, where the naive `(T,S)` version runs out |

So `attn_impl` stays `sdpa` by default, and this file is here to be **read**, not to be
switched on for a six-day run. That is an honest result and worth more than a fake win.

## The line that matters is not in the kernel

A hand-written kernel only handles the shapes it was written for. Ours refuses an arbitrary
mask, dropout inside attention, an unusual head dimension, and a query block too short to
fill a tensor-core tile. It also cannot run at all on a machine with no CUDA.

`flash.usable()` is that whole decision, and it **never raises**. Everything it says no to
has a correct `F.scaled_dot_product_attention` path immediately behind it, so choosing the
kernel is a routing decision rather than a promise.

---

## Exercise: take the guard off

1. Run the check. It passes: a model configured with `attn_impl: flash` gives an answer on
   **any** device, including a laptop with no GPU at all.
2. In `aksharallm/model/transformer.py`, find the branch in `Attention.forward` that reads
   `if self.attn_impl == "flash" and flash.usable(...)`. Delete the `flash.usable(...)` half
   of the condition, so the kernel is called whenever the config asks for it.
3. Run the check. **It should fail.** Without a CUDA device there is no kernel to call, and
   the model that used to answer now raises.
4. Put it back. Green.

> **What you just saw.** The kernel is the impressive part and the guard is the shippable
> part. Notice also which failure you got: a loud one. The *quiet* version of this bug is a
> `usable()` that says yes to a call carrying an attention mask — the kernel does not take
> masks, so the mask would simply be dropped, and a sliding-window model
> ([lesson 15](docs/lessons/15-long-context.md)) would silently attend to everything with no error
> anywhere.

## Going further

The other trap in this file is the one that has now bitten this repo twice: causal masking
is **bottom-right aligned**. Query `m` of `T` may see key `n` of `S` when `n <= m + (S - T)`.
When queries and keys are the same length that offset is zero and you never notice; when
they are not — several tokens verified against a warm cache, which is exactly what
[lesson 16](docs/lessons/16-speculative.md) does — a top-left triangle hides most of the prompt from
every query, trains fine, and generates fluent nonsense.

`tests/test_flash.py::test_causal_is_bottom_right_aligned` is the defence, and it needs a
GPU to run. If you have one, break `DIAG` in the kernel and watch it.
