---
id: diffusion
title: Filling in blanks instead of predicting the next word
doc: docs/19-diffusion.md
files:
  - aksharallm/diffusion/corrupt.py
  - aksharallm/diffusion/generate.py
  - aksharallm/train/pretrain.py
verify: tests/test_diffusion.py::test_the_weight_makes_mask_rates_comparable
prereqs: [training-loop]
minutes: 35
summary: The other way to build a language model — and the one-character weight that decides whether it learns anything useful.
---

# 19. Filling in blanks instead of predicting the next word

Everything up to here writes strictly left to right. Token 5 is chosen knowing tokens 1–4
and nothing else, and once chosen it is never revisited.

That is not a law of language modelling. It is a **design decision**, and it buys two
specific things: because a causal model may never look right, it can be trained on every
position of every sequence at once, and it can cache its own past while generating
([lesson 4](docs/lessons/04-kv-cache.md)).

A **masked diffusion** model gives both up. It is trained to fill in blanks with attention
running in both directions, and it generates by unmasking the positions it is most confident
about first — in whatever order that turns out to be.

```
▁ ▁ ▁ ▁ ▁ ▁            step 0    everything blank
▁ ▁ cat ▁ ▁ .          step 1    the two it was surest about
The ▁ cat sat ▁ .      step 2
The big cat sat on .   step 3    done
```

Look at step 1 again. It committed the full stop before it had decided what the sentence was
about, because a full stop was the easiest position to be right about — and once committed,
it constrains everything else.

## The objective, complete

```python
t      = t_min + (1 - t_min) * rand(B)          # one mask rate per sequence, uniform
masked = rand(B, T) < t[:, None]                # an independent coin per position
x_t    = where(masked, MASK, x)                 # the corrupted input
loss   = (ce(model(x_t), x) * masked).sum(1) / (t * T)      # then .mean()
```

That is all of it. There is no noise schedule to tune and no second network — for discrete
tokens, "noise" just means "replaced by `[MASK]`".

Compare it with [lesson 5](docs/lessons/05-training-loop.md)'s four lines. **Only the loss changed.** The
optimiser, the schedule, gradient accumulation, checkpoint/resume, the stop file, the
throughput counter — none of that knows what paradigm it is training, which is why there is
no second trainer in this repo. `train/pretrain.py` asks an *objective* for a batch and a
loss, and there are two objectives.

## What it costs and what it buys

It is **less data-efficient**, by a lot: next-token prediction gets a signal at every
position of every sequence, this gets one only at the masked positions. Published
comparisons put the gap at 3–16x the compute for equal quality, so it runs at 13.8M here and
will never be the main model.

There is also **no KV cache and there cannot be one**. A cache works because position *n*'s
keys are settled the moment it is generated; here any position may be rewritten on any step,
so a cached key would belong to a token that no longer exists. `Transformer.forward` raises
rather than letting it try.

What it buys is two things autoregression structurally cannot do: **infilling** — give it a
prefix *and* a suffix and it writes the middle, because "the middle" is just the positions
still masked — and a **compute dial**, 48 tokens in 16 forward passes or in 4.

---

## Exercise: drop the `1/t`

The loss divides by `t`, the fraction of tokens hidden. Take it out and the model still
trains, the loss still falls, and the samples still look like English.

1. Run the check. It passes: with uniform logits, the weighted loss is `log(V)` whether 10%
   or 90% of the sequence was masked. Every mask rate contributes equally.
2. In `aksharallm/diffusion/corrupt.py`, find `diffusion_loss` and remove the `c.t` from the
   denominator — divide by `T` alone.
3. Run the check. **It should fail**, and read the three numbers: they now scale with the
   mask rate, so a sequence masked at 90% counts nine times a sequence masked at 10%.
4. Put it back. Green.

> **What you just saw.** Without the weight, the model spends nearly all of its capacity on
> near-blank sequences — the hardest and least informative corruptions — and comparatively
> none on the light ones it will actually be asked to infill. It would still converge. It
> would just be worse, at a task you never measured, and nothing in the loss curve would
> mention it. (The weight is also exactly what makes the loss a valid bound on the
> likelihood, which is the better reason to keep it.)

## One number to be careful with

A diffusion run's validation loss is an **ELBO — an upper bound** on the negative log
likelihood. An autoregressive run's is the exact cross-entropy. They are different
quantities, so "diffusion 1.9 against the baseline's 1.472" is **not** "0.43 worse" — it is
"at most this bad, by an unknown margin".

Everything guarding that is naming: the trainer prints `ppl <=` rather than `ppl`, and the
measurement returns a key called `ppl_upper_bound` rather than `ppl`. If you add a metric
that is a bound, put it in the name — a comment is read once and the key is read forever.

```bash
scripts/experiment.sh tiny-diffusion    # or the portal's Diffusion tab, which animates it
```
