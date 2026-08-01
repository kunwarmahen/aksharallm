---
id: attention
title: Attention, and the mask that makes it prediction
doc: docs/03-model.md
files:
  - aksharallm/model/transformer.py
verify: tests/test_model.py::test_causality
prereqs: [tokenizer]
minutes: 35
summary: How tokens look at each other, and why one triangular mask is the difference between predicting the future and copying it.
---

# 3. Attention, and the mask that makes it prediction

Every token in the sequence produces three vectors:

| | | |
|---|---|---|
| **query** | what am I looking for? |
| **key** | what do I have to offer? |
| **value** | what do I pass on if chosen? |

Token *i* compares its query against every key, turns those scores into weights with a
softmax, and takes the weighted sum of the corresponding values. That is the entire
mechanism. "The cat sat on the ___" works because the query at `___` matches the key at
`cat`, so the value at `cat` dominates the sum.

Everything else in the block — RMSNorm, RoPE, SwiGLU, the residual stream — is support for
this one operation. Read `docs/03-model.md` for what each of them does and why.

## The mask is not a detail

Training runs on the whole sequence at once: every position predicts its next token *in
parallel*. That is what makes it efficient, and it is also a trap, because position 3 can
see position 4 — and position 4 **is the answer**.

So the score matrix is masked: everything above the diagonal becomes `-inf` before the
softmax, so each position sees only itself and what came before.

```
        the   cat   sat   on
the      ✓     ·     ·     ·
cat      ✓     ✓     ·     ·
sat      ✓     ✓     ✓     ·
on       ✓     ✓     ✓     ✓          · = masked to -inf
```

Without it the model scores brilliantly on training data and cannot generate a word, because
at generation time there is no future to copy. It is the single highest-consequence line in
the file.

---

## Exercise: let the model cheat

1. Run the check. It passes — it feeds two sequences that differ only *after* position `t`
   and asserts the output at `t` is identical.
2. In `aksharallm/model/transformer.py`, find where `is_causal` is passed to
   `scaled_dot_product_attention` and set it to `False`.
3. Run the check. **It should fail**, and the failure is the interesting part: the outputs at
   position `t` now differ, which means information has flowed backwards in time.
4. Put it back. Green.

> **What you just saw.** The test compares two forward passes rather than checking a mask
> exists, because that is the property that matters and it holds however the mask is
> implemented. A test that asserted "the code contains `is_causal=True`" would pass while
> the model cheated through some other path.

## Where to look next

Open `aksharallm/model/transformer.py` in the **Code** tab and read `Attention.forward` with
the local model's help — ask it why `n_rep` exists, and why keys and values have their own
smaller head count. That is grouped-query attention, and the next lesson is about the cache
it makes affordable.
