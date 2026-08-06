---
id: long-context
title: Reading further than you were trained to
doc: docs/18-long-context.md
files:
  - aksharallm/model/rope.py
  - aksharallm/longctx/extend.py
verify: tests/test_longctx.py::test_ntk_barely_touches_the_fast_pairs_and_fully_slows_the_slow_ones
prereqs: [attention]
minutes: 30
summary: A trained model's context can be multiplied without training anything — and the obvious way to do it is the one that ruins the model.
---

# 15. Reading further than you were trained to

Our 300M model was trained with a 1,024-token window. Ask it about token 4,000 and it does
not get gradually vaguer — it **falls off a cliff**.

The reason is in [lesson 3](docs/lessons/03-attention.md). Position is not a number the model looks up;
it is an *angle*. RoPE rotates each pair of channels by an amount proportional to the
token's position, so the model learns to read relative distance out of the angle between two
rotations. Past the trained window it is being handed angles it has never seen, and the
whole geometry stops meaning anything.

The remarkable part: **RoPE has no parameters.** So extending the context is arithmetic on a
config, not training. `longctx extend` writes a checkpoint whose tensors are byte-for-byte
the ones it read.

## Three ways to stretch the ladder

Each channel pair rotates at its own frequency — fast pairs encode local position, slow
pairs encode global position.

| method | what it does | measured on our 300M |
|---|---|---|
| **linear** | divide **every** frequency by the factor | works, and wrecks short contexts: in-window loss 2.356 → **3.035** at 2x |
| **ntk** | raise the RoPE base instead, which tilts the ladder — slow pairs interpolated fully, fast pairs barely touched | 2x costs **0.009 nats** (2.356 → 2.365). Effectively free |
| **yarn** | interpolate per pair with a ramp between the two regimes, plus an attention temperature | the one that holds furthest: 2.464 at 4x, where NTK grows a cliff of its own at 3,584 |

The intuition for why linear is bad: squashing the *fast* pairs destroys the model's ability
to tell "one token back" from "two tokens back" — which is most of what it uses attention
for — in order to buy range it mostly does not need at that resolution.

And it works. Extended 4x with YaRN, the 300M finds a fact hidden anywhere in a 4,096-token
haystack **92.5% of the time against a 25% chance line**. The 13.8M model, extended
identically, sits at chance: the positions became legible for both, but *retrieval* is a
capability only the larger one ever learned.

---

## Exercise: turn NTK back into the thing it improves on

1. Run the check. It passes: NTK leaves the fastest channel pair almost untouched while
   slowing the slowest one by the full factor.
2. In `aksharallm/model/rope.py`, find the `ntk` branch of the frequency computation. It
   raises the base — `theta * factor ** (head_dim / (head_dim - 2))`. Replace the whole
   branch with the `linear` one: divide the base frequencies by `factor` instead.
3. Run the check. **It should fail**, and read the numbers it prints. The fast pair, which
   NTK left alone, has now been slowed by the full factor along with everything else.
4. Put it back. Green.

> **What you just saw.** The two methods are one line apart and produce the same *range*.
> The difference is entirely in which channel pairs pay for it, and that difference is worth
> 0.68 nats of in-window loss on a real model. A benchmark that only measured "can it read
> 2,048 tokens" would have scored them the same.

## The measurement is half the lesson

Two numbers disagree, and you need both:

* **loss by position** says whether the model is still *fluent* out there;
* **needle-in-a-haystack** says whether it can still *retrieve*.

A sliding window scores the best perplexity of anything we tried and is structurally blind
past its own window. Perplexity alone would have recommended it.

One methodology note, learned the hard way here: the first version of the numbers above was
sampled over **two** windows and reported an in-window baseline of 0.990 where thirty-two
windows say 2.356. Every conclusion survived the correction and not one number did. Use at
least sixteen windows before quoting a position curve.

```bash
python -m aksharallm.longctx sweep tiny        # or the portal's Context tab
```
