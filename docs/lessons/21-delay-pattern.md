---
id: delay-pattern
title: Eight integers per position, and the shift that keeps them honest
doc: docs/21-audio.md
files:
  - aksharallm/audio/delay.py
  - aksharallm/audio/lm.py
  - aksharallm/model/transformer.py
verify: tests/test_audiolm.py::test_undelay_inverts_delay_exactly
prereqs: [codec]
minutes: 30
summary: The codec hands the language model eight integers per frame, not one. Three ways to deal with that, two of them wrong, and the third is a diagonal shift.
---

# 21. Eight integers per position, and the shift that keeps them honest

[Lesson 20](docs/lessons/20-codec.md) turned a waveform into integers. There is a catch: it
produced **eight of them per frame**, and they are not independent. Codebook 2 quantizes the
error codebook 1 left behind, which is the entire point of a residual.

A language model wants one token per position. Three ways to bridge that.

## Flatten

```
[c⁰₀ c¹₀ c²₀ … c⁷₀ c⁰₁ c¹₁ …]
```

Correct, and it makes the sequence **eight times longer**. Ten seconds of speech goes from
500 positions to 4,000, and attention is quadratic. This is the option that works and that
you cannot afford.

## Predict all eight at once

Take one position per frame, and put eight heads on it predicting `c⁰ₜ … c⁷ₜ` from frames
before `t`. Fast, cheap, and **wrong**: it assumes the eight are conditionally independent
given the past, and they are the opposite of independent — each one is defined as a
correction to the last.

## Delay

```
book 0:  c⁰₀  c⁰₁  c⁰₂  c⁰₃
book 1:   ·   c¹₀  c¹₁  c¹₂
book 2:   ·    ·   c²₀  c²₁
book 3:   ·    ·    ·   c³₀
```

Shift codebook *k* right by *k* frames. Now everything in one column can be predicted in
parallel — because by the time `c¹₀` is predicted, `c⁰₀` is in the past and **visible in the
context**. The dependency is carried by attention rather than assumed away.

The sequence grows from `T` to `T + N − 1`. Eight extra positions on five hundred, instead of
eight times five hundred.

## What the model actually changed

Almost nothing, and that is the lesson. `Transformer.forward` gained two optional arguments,
both exact no-ops on every path you have used so far:

- `inputs_embeds`, because a position carries eight integers and the model sums eight
  embeddings. Concatenating instead would make `d_model` depend on the codebook count;
  summing works because the tables are *separate*, so code 5 of book 0 and code 5 of book 3
  are different vectors — a composition, not a collision.
- `return_hidden`, because it needs **eight heads**. One head over `8 × 1024` would let the
  model put probability on a book-3 code while predicting book 0, which is not a possible
  answer.

Blocks, RoPE, GQA, the causal mask: untouched.

---

## Exercise: reverse which codebook leads

Which end of the stack goes first is a choice, and writing it the other way round is the
easiest mistake in the file. It belongs to the family that trains fine and generates garbage.

1. Run the check. It passes — `undelay(delay(x)) == x`, exactly, for four shapes.
2. In `aksharallm/audio/delay.py`, find `delay` and reverse the shift so codebook 0 is the
   one delayed furthest instead of the one that leads:

   ```python
   out[:, k, (n - 1 - k) : (n - 1 - k) + t] = codes[:, k]      # <- wrong
   ```

3. Run the check. **It should fail** — three of the four cases, with an equality assertion.
   Note what it takes to see it. Nothing about the *shape* is wrong, nothing raises, and
   every value in the tensor is a valid code index.
4. Now look at which case still passes: **`[1-5]`, the one-codebook shape**. With a single
   codebook there is no delay to get backwards, so that configuration is blind to this bug
   entirely. It is exactly the shape you would reach for while debugging.
5. Put it back. Green.

> **What you just saw.** Both versions return a tensor of exactly the right shape, full of
> perfectly valid code indices. Train with it and the loss falls normally, because the model
> simply learns whatever consistent scrambling you gave it. You find out at generation time,
> when the decoder is handed codebook 1 of frame 5 beside codebook 0 of frame 4 and
> reconstructs an interleaving of two different moments. It sounds like speech that has been
> through a shredder — and the fastest way to conclude that the *codec* is broken.
>
> This is why the check is parameterised over four shapes rather than one. A single shape
> can be blind to a real bug, as step 4 just showed, and picking the *simplest* shape is
> what makes that most likely.

## The other check worth knowing

An untrained model over 512 codes scores exactly `ln 512 = 6.238` nats, because a uniform
distribution is what "knows nothing" means. The trainer prints that expectation in its header
and marks the step-0 line when it matches:

```
expect     step-0 loss ~= ln(512) = 6.2383
step      0  loss  6.3145 (ema  6.3145)  ...   <- uniform
```

If the delay pattern, the target masking or the eight heads were wired wrongly, that number
would not land — a mask that ignores real cells, or heads reading the wrong axis, both move
it. It is the same trick as the DPO loop's `ln 2 = 0.6931`
([lesson 8](docs/lessons/08-sft-mask.md) territory), and it costs nothing to look at.

## And the number that does not tell you anything

The audio LM's loss falls smoothly while it generates fluent nonsense, because most of the
entropy lives in the **high codebooks** — which are nearly noise, and which nobody, model or
otherwise, can predict. Do not read the curve. Listen to `checkpoints/<run>/samples/`, which
the trainer writes as it goes for exactly this reason.

```bash
.venv/bin/python -m aksharallm.audio encode checkpoints/codec-synth/ckpt_best.pt \
    --corpus data/audio/synth --out data/audio/synth-codes
scripts/audio.sh audiolm-synth
```
