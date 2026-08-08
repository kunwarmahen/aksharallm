---
id: kv-cache
title: The KV cache, and the one-line bug that ruins it
doc: docs/07-inference.md
files:
  - aksharallm/model/transformer.py
  - aksharallm/infer/generate.py
verify: tests/test_model.py::test_kv_cache_matches_full_forward
prereqs: [attention]
minutes: 30
play: repetition
summary: Why generating 100 tokens without a cache costs 100x too much, and the real bug where the model trained perfectly and generated garbage.
---

# 4. The KV cache, and the one-line bug that ruins it

Generation is a loop: predict one token, append it, predict the next. Done naively, step *n*
re-runs the whole model over all *n* tokens — so writing 100 tokens does the work of
100 + 99 + 98 + … forward passes. Quadratic, for no reason.

The reason there *is* no reason: for every token already in the sequence, the keys and values
are **the same every step**. They depend on that token and its position, neither of which
changes. So compute them once and keep them.

```
step 1   [The]                     -> cache K,V for 1 token
step 2   [The, cat]                -> compute K,V for "cat" only, append
step 3   [The, cat, sat]           -> compute K,V for "sat" only, append
```

Each step processes **one** token against a growing cache. Linear, not quadratic — and the
memory it costs is why grouped-query attention exists: fewer key/value heads means a smaller
cache, which at long context is the difference between fitting and not.

---

## Exercise: the bug that actually happened here

This one is real, it is in `PLAN.md`'s bug list, and it cost hours.

During *training* the model sees the whole sequence, so attention must be causal — the mask
from the last lesson. During *decoding with a cache* the model sees exactly one new token,
and every cached position is legitimately in its past. Applying a causal mask to a
single-token query masks away **the entire cache**, because PyTorch aligns the triangle to
the end of the sequence: the one query is treated as position 0.

The model still trains perfectly. It just generates garbage — each token conditioned on
nothing but itself.

1. Run the check. It passes: it generates with the cache and without it, and asserts the
   logits match.
2. In `aksharallm/model/transformer.py`, find where `is_causal` is decided in
   `Attention.forward` and make it always `True` — deleting the guard that turns it off for
   a single-token step.
3. Run the check. **It should fail**, with the two paths disagreeing.
4. Put it back. Green.

> **What you just saw.** No test of the *training* path would have caught this — training
> never takes the single-token branch. The bug lives in the gap between two code paths, which
> is exactly where "it trains fine, why is the output nonsense?" bugs live. The test that
> catches it compares the two paths against each other rather than either against a
> hand-written expectation.

## See it in the model

Once Phase 2 has a checkpoint, open the **Playground** and run the `repetition` probe. A
half-trained model loops — *"the the the"* — and that is normal and informative. A model with
a broken cache does something different and unmistakable: every token is unrelated to the
last, because nothing is conditioning on anything.

Learning to tell those two failures apart by eye is worth more than either explanation.
