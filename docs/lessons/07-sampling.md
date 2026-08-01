---
id: sampling
title: Choosing the next token
doc: docs/06-inference.md
files:
  - aksharallm/infer/generate.py
verify: tests/test_generate.py::test_top_p_keeps_the_nucleus
prereqs: [kv-cache]
minutes: 25
play: story
summary: Temperature, top-k and top-p are three knobs on the same distribution, and the reason repetition penalty is switched off here.
---

# 7. Choosing the next token

The model does not output a token. It outputs a **score for every token in the vocabulary** —
32,768 numbers — and something has to choose. That something is not part of the model, is not
trained, and changes the output completely.

**Always taking the highest score** (greedy) sounds right and reads terribly: it loops,
because the safest continuation of a sentence is often a sentence you have already written.

So we sample, with three knobs:

| knob | what it does | at its extreme |
|---|---|---|
| **temperature** | divides the scores before the softmax | `0` = greedy; `2` = incoherent |
| **top-k** | keep only the k highest-scoring tokens | `1` = greedy; `vocab` = no filter |
| **top-p** (nucleus) | keep the smallest set whose probability sums to p | `0.0` = greedy; `1.0` = no filter |

Top-p is the subtle one and usually the best default. Top-k always keeps 50 candidates, even
when the model is *certain* — after "New York" the correct next token might carry 0.99
probability, and top-k still admits 49 alternatives. Top-p keeps one candidate there and
forty in a genuinely open position. It adapts to how confident the model is.

## Why repetition penalty is off here

Most sampling code ships with a repetition penalty on by default, which suppresses tokens
that have already appeared. In this project it is deliberately `1.0` — off — and the reason
is diagnostic: **looping is the symptom that tells you a base model is undertrained.** A
penalty hides it, and you lose the clearest signal you have that the run needs more steps.

Ship it on. Debug with it off.

---

## Exercise: break the nucleus boundary

1. Run the check. It passes — it builds a distribution by hand and asserts exactly which
   tokens survive a given `p`.
2. In `aksharallm/infer/generate.py`, find the top-p filter and change the comparison that
   decides where the cumulative sum cuts off (`>` and `>=` are one character apart and both
   look correct).
3. Run the check. **It should fail** — note *which* token moved across the boundary.
4. Put it back. Green.

> **What you just saw.** An off-by-one in a filter does not crash and does not look wrong.
> It slightly changes which tokens are reachable, which slightly changes the character of
> everything the model writes — and you would never find it by reading output.

## Feel it

In the **Playground**, run the same prompt at temperature 0.2 and at 1.2, then at top-p 0.5
and 0.95. Every generation is recorded in `logs/playground.jsonl` with the checkpoint's step
and loss, so you can put them side by side afterwards.
