---
id: latent
title: Drawing in a smaller space, and the scale that has to come back
doc: docs/26-pictures.md
files:
  - aksharallm/pictures/latents.py
  - aksharallm/pictures/flow.py
verify: tests/test_flow.py::test_a_latent_drawing_is_mapped_back_before_the_decoder_paints_it
prereqs: [flow-matching]
minutes: 25
summary: G3 is G2's flow matching run on the image tokenizer's 8x8x64 latent instead of pixels, and the tokenizer's decoder paints the result. The latents are normalised to unit scale so they match the noise the flow starts from. Forget to undo that before the decoder, and every drawing is painted at the wrong scale.
---

# 32. Latent flow

G2 ([lesson 31](docs/lessons/31-flow-matching.md)) draws 12,288 numbers a picture. Almost all of
them are predictable from their neighbours, so real image models draw something smaller and
let a decoder fill in the pixels. G3 uses the image tokenizer of
[lesson 29](docs/lessons/29-pictures.md): its encoder turns a picture into an 8×8 grid of 64
numbers, the flow model draws such a grid, and the tokenizer's decoder paints it
([doc 26](docs/26-pictures.md) § G3). The flow model has 64 tokens to attend over instead of 256.

## Normalising, and undoing it

Flow matching starts from noise with mean 0 and variance 1. The tokenizer's latent channels do
not look like that (their spreads range from about 0.1 to 0.5), so `pictures/latents.py`
subtracts each channel's mean and divides by its standard deviation, measured on the training
pictures, before the flow model ever sees a latent. The model therefore draws latents in
normalised units, and `flow.py`'s `draw` must multiply by the standard deviation and add the
mean back before the decoder paints them. The statistics are saved inside every G3 checkpoint
for exactly this reason.

The check uses an oracle that flows to a known normalised latent and requires the picture to
be the decoder's picture of the **denormalised** latent.

---

## Exercise: forget to map it back

1. Run the check. Green.
2. In `aksharallm/pictures/flow.py`, inside `draw`, change
   `x = decode_latent(tokenizer, denormalise(x, m, s), snap=snap)` to
   `x = decode_latent(tokenizer, x, snap=snap)`.
3. Run the check. **It should fail.**
4. Put it back. Green.

> **What you just saw.** On the real tokenizer, decoding the stored latents without mapping
> them back still produces pictures of coloured shapes, and the reader still calls 62% of them
> right. Nothing crashes, the training loss is untouched (training never decodes anything), and
> the drawings look roughly plausible. Only a measurement against a known answer shows that a
> third of them are wrong, and why.

## What it did on the real model

`flow-latent-shapes` trained in 16 minutes where G2 took about 42 (64 tokens a step instead of
256) and kept colour and shape (≥ 99.8%), but drew 87.6% of seen captions right against G2's
99.2%: it sometimes puts two shapes touching, because it places them on an 8×8 grid of latent
cells rather than a 16×16 grid of patches (docs/26 § Measured: G3 against G2). Cheaper per
step, coarser in where things go: the trade every latent model makes. A finer latent (the 16×16
tokenizer's, which reconstructs just as perfectly) made it *worse*, about 50%, whatever the
number of tokens: a latent can decode perfectly and still be hard to generate, which is why
Stable Diffusion's autoencoder is trained to keep its latent smooth. Doing exactly that here (a
continuous autoencoder with a light KL term, 8×8×8 = 512 numbers) brought latent flow up to
98.9%, level with the pixel model, in a third of the time.

## Run it yourself

```bash
python -m aksharallm.pictures latents imgtok-shapes --corpus data/vision/shapes64k \
    --out data/pictures/latents-imgtok-shapes-64k
scripts/audio.sh flow-latent-shapes
python -m aksharallm.pictures gen-report flow-latent-shapes --guidance 1 --k 32
python -m aksharallm.pictures gen-compare logs/pictures/gen-flow-shapes-step29999-g1-k32.json \
    logs/pictures/gen-flow-latent-shapes-step29999-g1-k32.json     # G2 against G3, paired
```

Or open the **Pictures** tab: G3 models are listed beside G1 and G2 in the Draw panel.

Next, [lesson 33](docs/lessons/33-held-out-phrasing.md): the same model reading a whole
sentence through a frozen language model, and a held-out test that training must never see.
