---
id: guidance
title: Drawing from a sentence, and the one line that makes it obey
doc: docs/26-pictures.md
files:
  - aksharallm/pictures/codes.py
  - aksharallm/pictures/gen.py
  - aksharallm/pictures/score.py
verify: tests/test_picgen.py::test_guidance_mixes_the_captioned_and_the_unconditional_halves
prereqs: [pictures]
minutes: 25
summary: G1 writes a picture's codes after its caption, with the same transformer that writes text. Classifier-free guidance runs it twice per step, with the caption and without, and pushes the logits away from the captionless ones. Swap the two halves and the knob that should make it obey makes it ignore the words.
---

# 30. Guidance

Once a picture is 128 integers ([lesson 29](docs/lessons/29-pictures.md)), drawing one from
a sentence is next-token prediction: the sequence is `[three] [red] [circles] [IMG]` and then
the codes, and the transformer of [lesson 3](docs/lessons/03-attention.md) learns it unchanged
([doc 26](docs/26-pictures.md) § From a sentence to a picture). The tokenizer's decoder paints
the codes, and the reader checks the picture against the caption.

## Classifier-free guidance

During training, one caption in ten is replaced by `NULL`, so the same model learns two
things: what a picture of *these words* looks like, and what a picture from this corpus
looks like with no words at all. When it draws, `pictures/gen.py`'s `sample` runs both at
every step and mixes them:

```
logits = uncond + w * (cond - uncond)
```

`cond - uncond` is exactly what the caption changes. `w = 1` is the model as trained, `w = 0`
throws the caption away, and `w = 3` exaggerates the caption's effect, so the drawing obeys
the words more and varies less. Imagen and Stable Diffusion have the same line, applied to
predicted noise instead of logits.

The test uses a constructed model that prefers code 5 when it has a caption and code 7 when
it has `NULL`. Then `w = 1` and `w = 3` must give 5, and `w = 0` must give 7.

---

## Exercise: swap the halves

1. Run the check. Green.
2. In `aksharallm/pictures/gen.py`, inside `sample`, change
   `lg = unc + guidance * (cond - unc)` to `lg = cond + guidance * (unc - cond)`.
3. Run the check. **It should fail**: at `w = 0` the model now follows the caption, and at
   `w = 3` it is pushed *away* from it.
4. Put it back. Green.

> **What you just saw.** The swapped line is still a valid mix of two distributions and
> still draws pictures, so nothing crashes. With it, raising the guidance makes the model
> ignore the caption more. Only a measurement that knows what the caption asked for (the
> reader, here) would ever notice, and a gallery of nice-looking shapes would not.

## What it did on the real model

On `imggen-shapes` (docs/26 § Measured), without guidance the never-seen "purple triangles"
came out the right shape in the colours triangles were trained in: 35% purple, the rest red,
blue or yellow. At guidance 3 they were 84% purple and the held-out score doubled (21.9% →
43.0%, paired p 0.0007), while captions it already knew fell a little (66.2% → 62.0%,
p 0.006). Guidance 5 broke the shapes.

Then the same model trained on 8 times the pictures (docs/26 § Fixing count) drew the purple
triangles right 74% of the time **without** guidance, and guidance 3 no longer helped them
(p 0.79) while still costing a little on the captions it knew. Guidance had been making up for
a model that had not learned the words well enough: it is a knob for pushing a weak model,
and its value has to be measured again every time the model changes. G2 (lesson 31), which revises the
whole picture at every step, draws every never-seen purple triangle right with or without it.

## Run it yourself

```bash
python -m aksharallm.pictures codes imgtok-shapes --books 2          # the corpus as codes
scripts/audio.sh imggen-shapes                                       # train G1 (on the card)
python -m aksharallm.pictures draw imggen-shapes "three purple triangles" --guidance 3
python -m aksharallm.pictures gen-report imggen-shapes --guidance 1
python -m aksharallm.pictures gen-report imggen-shapes --guidance 3
python -m aksharallm.pictures gen-compare logs/pictures/gen-imggen-shapes-step7999-g1.json \
    logs/pictures/gen-imggen-shapes-step7999-g3.json
```

Or open the **Pictures** tab, pick three words, and press **Draw**: each picture is bordered
green or red by the reader's verdict.
