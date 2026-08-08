---
id: lora
title: Fine-tuning without touching the model
doc: docs/12-lora.md
files:
  - aksharallm/lora/layer.py
  - aksharallm/lora/inject.py
verify: tests/test_lora.py::test_the_adapter_is_the_identity_at_initialisation
prereqs: [sft-mask]
minutes: 30
summary: Two skinny matrices beside every frozen one, why one of them starts at exactly zero, and how 4,791 MB of fine-tuning budget becomes 327 MB.
---

# 10. Fine-tuning without touching the model

Fine-tuning normally means training every weight, which means holding every weight's
gradient and two optimiser moments as well. For a 300M model that is about **4,791 MB** — for
a model whose weights are 600 MB.

LoRA's observation: the *change* a fine-tune makes to a big matrix is low rank. It does not
need the freedom of a full matrix to express "be better at Python". So freeze the original
and learn a small correction beside it:

```
h = W x  +  (B A) x        W frozen (600 MB, no gradients)
                           A: r x in     B: out x r        r = 8
```

`A` and `B` together are ~2.5% of the parameters. Only they have gradients and optimiser
state, which is where the memory goes.

| | to fine-tune | the artifact |
|---|---|---|
| full fine-tune | 4,791 MB | a new 1.2 GB checkpoint |
| LoRA r=8 | 1,253 MB | a 14 MB adapter |
| **QLoRA r=8** | **327 MB** | a 14 MB adapter |

QLoRA is the same thing with the frozen base stored in 4-bit NF4 — which is why the
quantization lesson comes first. **There is no separate QLoRA class in this repo:**
`LoRALinear` wraps an `nn.Linear` or a `QuantLinear`, and that is the entire difference.

## Why B starts at zero

`B` is initialised to **exactly zero**, so `BA = 0` and the adapted model is *identical* to
the base model at step 0. Training then moves away from a model that already works, rather
than from a random perturbation of one.

If both matrices started random, step 0 would be a damaged model and the first thing training
would have to do is undo the damage. And `A` must *not* be zero — with both at zero the
gradient of the product is zero and nothing would ever move.

You have met this idea before: sparse upcycling in the MoE lesson is the same trick, and it
is worth noticing the pattern. **Start from the trained thing, not near it.**

---

## Exercise: break the identity

1. Run the check. It passes — it asserts the adapted layer's output exactly equals the base
   layer's before any training.
2. In `aksharallm/lora/layer.py`, find where `B` is initialised and give it small random
   values instead of zeros.
3. Run the check. **It should fail**, with outputs that differ before a single step.
4. Put it back. Green.

> **What you just saw.** This one *would* still train. The loss would start higher, come down
> anyway, and end up somewhere slightly worse — a difference you would never attribute to an
> initialiser. Silent-quality bugs are the theme of this whole path.

## The other thing an adapter buys

A specialisation becomes a **file**, not a model. One base plus a chat adapter plus a Python
adapter, swapped at inference, instead of two 1.2 GB checkpoints. The Playground's adapter
picker is one dropdown away from base-versus-adapted, which is the fastest way to see what a
fine-tune actually did.

```bash
python -m aksharallm.lora budget checkpoints/small-code/ckpt_best.pt
```

That prints the table above for your own checkpoint, before spending any GPU time on it.
