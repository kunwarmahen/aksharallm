---
id: pictures
title: Turning a picture into integers, and a judge that has to be right before it can judge
doc: docs/26-pictures.md
files:
  - aksharallm/pictures/reader.py
  - aksharallm/pictures/tokenizer.py
  - aksharallm/pictures/measure.py
verify: tests/test_pictures.py::test_the_reader_sees_the_damage_a_tokenizer_does
prereqs: [codec]
minutes: 30
summary: An image tokenizer is the audio codec in two dimensions, with the same quantizer. It is scored by reading each reconstruction back with a rule-based reader, which is only worth trusting if it is exact on the originals and strict enough to see a blur. Loosen it and a tokenizer that paints circles for squares scores perfectly.
---

# 29. Pictures as integers

A transformer models integers. To make it draw, a picture has to become a short list of
integers that turns back into the picture: the **image tokenizer** ([doc 26](docs/26-pictures.md)).
It is the codec of [lesson 20](docs/lessons/20-codec.md) with 2-D convolutions: an encoder
shrinks a 64×64 picture to an 8×8 grid, `ResidualVQ` (imported from `audio/vq.py`, unchanged)
replaces each cell with four codebook numbers, and a decoder paints it back.

## Scoring a reconstruction

Pixel error (PSNR) is a weak score for shapes pictures: most of the picture is background,
so a square repainted as a round blob costs only a few dB. The pictures were drawn from
descriptions we chose ("two green squares"), so a better score is: does the reconstruction
still say that? `pictures/reader.py` reads it back by rule: blobs far from the background
grey, their colour, their count, and their shape from how much of their bounding box they
fill (square about 1.0, circle about 0.79, triangle about 0.5).

A judge has two ways to be wrong, and the test checks both:

* **too strict**: it must be right on every original, or its errors become the tokenizer's;
* **too lenient**: it must see the damage. A light blur (σ = 1) still reads as a square; a
  heavy one (σ = 3), which is what a half-trained decoder paints, must not.

---

## Exercise: make the judge lenient

1. Run the check. Green.
2. In `aksharallm/pictures/reader.py`, change `SQUARE_ABOVE, CIRCLE_ABOVE = 0.90, 0.64` to
   `SQUARE_ABOVE, CIRCLE_ABOVE = 0.60, 0.40`.
3. Run the check. **It should fail**: the heavily blurred squares now read as squares.
4. Put it back. Green.

> **What you just saw.** With the loose thresholds, a blurry round blob fills 0.69 of its box
> and counts as a square. The smoke run's tokenizer, which turned every square into a soft
> circle, would then have scored 100% on shape instead of 27%. A judge is a measurement
> instrument: calibrate it against inputs whose answer you know, in both directions, before
> believing what it says about anything else.

## Run it yourself

```bash
python -m aksharallm.pictures read                     # the reader on the corpus: 100%
scripts/audio.sh imgtok-shapes                         # train the tokenizer (on the card)
python -m aksharallm.pictures report imgtok-shapes     # the ladder and the gallery
```

Or open the **Pictures** tab, pick a report, and look at the gallery beside the table.
