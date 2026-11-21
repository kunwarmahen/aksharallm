---
id: recall
title: A task that is nothing but looking things up, and the off-by-one that would score it wrong
doc: docs/19-long-context.md
files:
  - aksharallm/longctx/recall.py
  - configs/recall-hybrid-easy.yaml
  - configs/recall-hybrid.yaml
verify: tests/test_recall.py::test_the_scorer_gives_a_real_lookup_everything_and_a_copier_nothing
prereqs: [long-context]
minutes: 30
summary: Key-value pairs, then queries. A model scores only by finding each key earlier in the context and reading what followed. Reading the prediction from the wrong position turns a perfect model into a failing one and a copier into a perfect one.
---

# 28. Associative recall

The needle test of [lesson 15](docs/lessons/15-long-context.md) asks a model to find one fact
in a lot of text. At 13.8M parameters every model fails it, so it cannot compare designs.
**Associative recall** ([doc 19](docs/19-long-context.md) § "A task that is all retrieval")
can, because the task is nothing but lookup:

```
K17 V3  K40 V9  K2 V51   F F F ... F   ?   K40 V9   K2 V51
\______ pairs ________/  \_ gap ___/       \_ queries _/
```

Keys are distinct and values random, so the value after a query key can only come from
finding that key earlier. The scorer sweeps how many pairs a model holds and how far back it
can reach, past its trained window too.

It is also hard to learn. On the full task a small model stayed at chance; on easy documents
(2–8 pairs, no gap) the lookup formed suddenly. So each arm trains in two stages: easy first,
then the full task from those weights.

## The off-by-one

A model's logits at position `p` predict the token at `p + 1`. So the prediction for the
value at position `p` is read from the logits at `p − 1`. The test holds the scorer to two
models whose answers are known: a model that really looks each key up must score 100%, and a
model that copies the previous token must score 0%.

---

## Exercise: read from the wrong position

1. Run the check. Lookup 100%, copier 0%.
2. In `aksharallm/longctx/recall.py`, in `accuracy`, change `pred[row, pos - 1]` to
   `pred[row, pos]`.
3. Run the check. **It should fail**: the lookup model now scores near 0, because at `pos` it
   predicts the *next* query key, not this value.
4. Put it back. Green.

> **What you just saw.** With the bug, a real model's recall would read as zero and the
> conclusion would be "the architecture cannot retrieve". The same off-by-one family as the
> needle's `logits[len(prefix) - 1 : -1]`: always pin a scorer to a model whose score you know.

## Run it yourself

```bash
python -m aksharallm.longctx recall-data
scripts/experiment.sh recall-hybrid-easy && scripts/experiment.sh recall-hybrid
python -m aksharallm.longctx recall recall-hybrid --device cuda
```

Or press "Score recall" in the Context tab with a recall checkpoint selected.
