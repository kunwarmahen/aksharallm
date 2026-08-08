---
id: tokenizer
title: Why the model reads "tokens" and not letters
doc: docs/03-tokenizer.md
files:
  - aksharallm/tokenizer/tokenizer.py
verify: tests/test_pipeline.py::test_roundtrip_including_unicode
prereqs: [data]
minutes: 25
summary: Byte-pair encoding from scratch, why a 32k vocabulary is a size choice and not a detail, and the one thing that must never change after training starts.
---

# 2. Why the model reads "tokens" and not letters

Two obvious designs both fail:

* **One number per character.** The vocabulary is tiny, but "international" is 13 steps of
  work instead of one, and the model spends its capacity learning to spell.
* **One number per word.** Sequences get short, but the vocabulary is unbounded — every
  typo, name and number is a new word, and anything unseen becomes `<unk>`.

**Byte-pair encoding** sits between them. Start with the 256 possible bytes, then repeatedly
find the most frequent adjacent pair in the corpus and merge it into a new token. Do that
32,000 times and common words end up as one token, rare words as a handful of pieces, and
*nothing is ever unknown* — worst case, a string falls back to its bytes.

```
"tokenization"  ->  ["token", "ization"]
"aksharallm"    ->  ["aks", "har", "all", "m"]
```

The merges are learned **from our corpus**, which is the point of doing it ourselves: a
vocabulary trained on our blend of educational web text and Python spends its slots on the
things we actually train on.

## The thing you can never change afterwards

The tokenizer decides that token 5,142 means `" model"`. The embedding table's row 5,142 is
then trained to mean exactly that. **Swap the tokenizer and every row points at the wrong
thing** — and nothing crashes. You get a model that emits fluent-looking nonsense, and
nothing in the stack will tell you why.

This is why every post-training stage in this repo is passed the *base model's* tokenizer,
not a fresh one.

---

## Exercise: break the round trip

`encode` then `decode` must return exactly what went in — for ASCII, for accents, for emoji,
for a Python snippet with tabs.

1. Run the check. It passes.
2. In `aksharallm/tokenizer/tokenizer.py`, find `decode` and make it join the pieces with a
   space instead of concatenating them.
3. Run the check. **It should fail** — look at how the expected and actual strings differ.
4. Put it back. Green.

> **What you just saw.** Nothing about the model changed. A tokenizer bug produces a model
> that is *trained correctly on the wrong text*, and the only symptom is that the output
> reads slightly oddly — spacing that is not quite right, words that are almost words.

## Try it for real

The Playground and the Code tab both use this tokenizer. A quick way to feel the vocabulary:

```bash
.venv/bin/python -c "
from aksharallm.tokenizer.tokenizer import Tokenizer
t = Tokenizer('data/blend/tokenizer.json')
for s in ['the', 'transformer', 'kubernetes', 'def train(model):', '🙂']:
    ids = t.encode(s)
    print(f'{s!r:24} {len(ids):>2} tokens  {[t.decode([i]) for i in ids]}')"
```

Words the corpus is full of cost one token. Words it has never seen cost five. That ratio is
the tokenizer's whole job, and it is why vocabulary size is a *modelling* decision: at 32k
on a 300M model the embedding table is 11% of the parameters, and at 128k it would be a
third of them.
