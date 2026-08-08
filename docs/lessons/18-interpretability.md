---
id: interp
title: Asking the model where it decided
doc: docs/18-interpretability.md
files:
  - aksharallm/interp/capture.py
  - aksharallm/interp/lens.py
  - aksharallm/interp/patch.py
verify: tests/test_interp.py::test_the_recomputed_attention_reproduces_the_layers_real_output
prereqs: [sampling]
minutes: 35
summary: Attention maps, the logit lens and activation patching — and the rule that keeps a picture from being a drawing.
---

# 18. Asking the model where it decided

Every lesson so far has treated the model as a function from tokens to tokens. This one
opens it.

Three tools, and the reason they belong together is that they answer the same question in
different ways — **a story only one of them supports is not a story.**

## The logit lens: *when* did it decide?

Every block *adds* to the residual stream. Nothing overwrites it. So the output head can be
pointed at the running total halfway up and asked what the model would have said if it had
stopped there.

On our own 300M, `"The capital of France is"` only becomes `' Paris'` at **block 20 of 24** —
and it changes its mind eleven times on the way.

## Activation patching: *what* carries it?

Run the model on `"The capital of Italy is"`, then force **one** activation back to the value
it had on the France prompt, and see whether `' Paris'` comes back. Whatever restores the
answer is where the information was.

It agrees with the lens: the country information sits on the country token through blocks
10–19 and **moves to the last position at block 20**. A third method — per-head tracing —
puts half of that move in head 1 of block 20. Three methods, one answer.

## Sparse autoencoders: *what* is a direction?

A 384-dimensional residual stream carries far more than 384 concepts, so concepts share
directions — *superposition*. A wide, sparsely-activating autoencoder pulls them apart. The
sparsity penalty `α` is the whole knob: at 0.003 the dictionary fires 200 features per
token and explains everything uselessly; at 0.02 half the dictionary is dead. At **0.008**
it fires 13.7 features, explains 94.1% of the variance, and 3% of it is dead.

## The rule for this whole area

**A picture that is wrong is still a picture.** An attention map with the mask applied
incorrectly looks exactly like an attention map. A lens row computed from the wrong tensor
still reads as plausible English. Nothing here fails loudly, so every tool is pinned to
something unarguable:

| tool | pinned to |
|---|---|
| attention map | `weights @ V`, projected by `wo`, **equals the layer's real output** |
| logit lens | the last row equals the model's actual prediction |
| activation patching | patching the final layer restores exactly 100% |
| per-head tracing | the heads' outputs sum to the layer's output |
| sparse autoencoder | decoder columns stay unit-norm |

---

## Exercise: make a plausible, wrong picture

The fused attention kernel never stores its attention matrix, so the map you look at is
**recomputed** from the layer's own inputs. Recomputed means it can disagree with what the
model did.

1. Run the check. It passes: the recomputed weights, multiplied by V and projected by `wo`,
   reproduce the attention module's real output to 1e-5.
2. In `aksharallm/interp/capture.py`, find `attention_maps` and remove the `apply_rope` line
   — compute the scores from the raw `q` and `k`.
3. Run the check. **It should fail.** Now look at what you made: the map is still a valid
   probability distribution per row, still zero above the diagonal, still shows heads
   attending to plausible-looking places. It is a perfectly convincing picture of something
   the model never did.
4. Put it back. Green.

> **What you just saw.** This is why interpretability work is pinned to an identity rather
> than to "does the heatmap look sensible". You cannot eyeball the difference — and every
> conclusion drawn from that picture would have been about a model that does not exist.

## Try it on the real model

```bash
python -m aksharallm.interp lens small-code --prompt "The capital of France is"
python -m aksharallm.interp patch small-code --clean "The capital of France is" \
    --corrupt "The capital of Italy is" --answer " Paris" --other " Rome"
```

or the portal's **Interp** tab, which drives the same functions on the resident model.
