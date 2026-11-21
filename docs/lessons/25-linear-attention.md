---
id: linear-attention
title: A memory of fixed size instead of a KV cache, and the three inputs it must not forget
doc: docs/04-model.md
files:
  - aksharallm/model/deltanet.py
  - aksharallm/infer/generate.py
  - configs/tiny-hybrid.yaml
verify: tests/test_deltanet.py::test_a_recurrent_state_continues_exactly_where_the_full_forward_would
prereqs: [kv-cache]
minutes: 30
summary: A Gated DeltaNet layer keeps one small matrix per head instead of every past key and value. Generating with it means carrying that matrix, and the last few inputs of its convolution, from one token to the next.
---

# 25. Linear attention: a memory that does not grow

In [lesson 4](docs/lessons/04-kv-cache.md) the KV cache kept every past key and value, so its
size grows with the text. A **Gated DeltaNet** layer keeps something else: one matrix `S` per
head, the same size however long the text gets. Each new token writes into it (the delta rule
overwrites what a key used to point at) and reads out of it.

`tiny-hybrid` puts DeltaNet in four of its six blocks and keeps ordinary attention in two
([doc 4](docs/04-model.md) § "Linear attention").

```
token t ──► short convolution over the last 4 inputs ──► q, k, v
                                                          │
                     S_t = gate · S_{t-1} + write(k, v)  ◄┘     read: o = S_t · q
```

## What a cache has to carry

To generate one token at a time without re-reading the prefix, a DeltaNet block's cache
(`RecurrentState`) needs two things from the previous step:

* **`S`**, the memory matrix.
* **the convolution's tail**: the last 3 inputs. The convolution looks 4 tokens back, so the
  first token of a new step still needs the 3 before it.

Forget either and nothing crashes. The model still writes fluent-looking text, just not the
text the full forward pass would have written.

---

## Exercise: forget the convolution's tail

1. Run the check. It reads the same 40 tokens in four different splits (all at once, in
   pieces, one at a time) and requires every position to match the full forward pass to 1e-9.
2. In `aksharallm/model/deltanet.py`, in `GatedDeltaNet.forward`, find the line that saves
   `cache.conv` after the cached convolution. Replace it with `cache.conv = None`.
3. Run the check. **It should fail** on every split that reads in more than one piece: each
   new piece starts its convolution from zeros instead of the inputs it saw last time.
4. Put it back. Green.

> **What you just saw.** One-shot generation (a single piece) still passes, which is why a
> test that only compares "generate 20 tokens" against itself would never notice. Splitting
> the same sequence several ways is what exposes state that was not carried.

## Try it

```bash
python -m aksharallm.infer.cli checkpoints/tiny-hybrid/ckpt_best.pt --prompt "Once upon a time"
```

Measured on the CPU: 160 tokens/s at 500 tokens against 30 when the prefix is re-read, with
identical output.
