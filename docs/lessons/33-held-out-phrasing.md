---
id: held-out-phrasing
title: Reading a sentence, and a held-out test that training must never see
doc: docs/26-pictures.md
files:
  - aksharallm/pictures/phrases.py
  - aksharallm/pictures/text.py
  - aksharallm/pictures/train_flow.py
verify: tests/test_flow.py::test_every_phrasing_says_its_facts_and_the_held_out_ones_are_never_trained_on
prereqs: [latent]
minutes: 25
summary: G4a reads the caption with a frozen language model of our own, trained on ten phrasings of every caption and tested on two it never saw. A held-out phrasing is only a test while no training batch ever contains it. Let one leak in, and the score on it goes up while the thing it measures disappears.
---

# 33. Held-out phrasings

G3b ([lesson 32](docs/lessons/32-latent.md)) read three fixed words. G4a reads a sentence: the
frozen `tiny` model of Phase 1 turns it into one vector per token (`pictures/text.py`), and the
flow model reads those vectors where the three words used to be ([doc 26](docs/26-pictures.md)
§ G4a). Nothing in `tiny` is trained; the flow model learns only how to use what it says.

## Two held-out axes

The test has two things the model never saw in training:

* **a combination**: purple triangles, as in every lesson since 29;
* **a phrasing**: "show me two red circles" and "two circles in red". Ten other phrasings are
  used in training (`phrases.TRAIN`); these two (`phrases.HELD_OUT`) only in reports.

A model that only learned "these strings make these pictures" draws nothing useful from a
phrasing it never saw. One that reads the words still can. Measured: "two circles in red" scored
99.0%, the same as the trained phrasings, and "show me …" 97.4%.

That measurement means something only if `train_flow.phrased`, which picks a phrasing for every
training example, can never pick a held-out one. The check draws thousands of training
phrasings and requires none of them to be held out.

---

## Exercise: let the held-out phrasings leak

1. Run the check. Green.
2. In `aksharallm/pictures/train_flow.py`, inside `phrased`, change `names = list(TRAIN)` to
   `names = list(TRAIN) + list(HELD_OUT)` (and add `from .phrases import HELD_OUT`).
3. Run the check. **It should fail.**
4. Put it back. Green.

> **What you just saw.** With the leak, training is just as fast, the loss looks the same, and
> the score on "show me …" would go *up*: the model would have practised it. The number that
> was supposed to say "it reads sentences it has never seen" would then say nothing at all, and
> it would say it with more confidence than before. A held-out set is a promise about training,
> not a property of the test, so the test has to check the training side.

## Run it yourself

```bash
scripts/audio.sh flow-text-shapes
python -m aksharallm.pictures gen-report flow-text-shapes --guidance 1 --k 32
python -m aksharallm.pictures gen-report flow-text-shapes --guidance 1 --k 32 --phrasing show_me
python -m aksharallm.pictures gen-compare \
    logs/pictures/gen-flow-text-shapes-step29999-g1-k32.json \
    logs/pictures/gen-flow-text-shapes-step29999-g1-k32-pshow_me.json --by-fact
```

Or open the **Pictures** tab, pick a "G4 ·" model, and type a sentence of your own.
