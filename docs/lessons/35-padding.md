---
id: padding
title: Reading 590,000 captions in batches, and the side the padding goes on
doc: docs/26-pictures.md
files:
  - aksharallm/pictures/text.py
  - aksharallm/pictures/coco.py
  - aksharallm/pictures/photos.py
verify: tests/test_coco.py::test_a_padded_batch_encodes_each_caption_exactly_as_alone
prereqs: [contrastive]
minutes: 20
summary: G4c draws photos from COCO's people-written captions, read by the frozen 300M. There are too many captions to cache, so each batch's are encoded together, padded to one length. Pad at the end and a causal model gives every caption exactly the vectors it would get alone. Pad at the front and every caption is read after a run of padding tokens, and the vectors change while nothing errors.
---

# 35. The side the padding goes on

G4c ([doc 26](docs/26-pictures.md) § G4c) is the first corpus nobody drew for us: COCO's
118,287 photos, each with five sentences written by people. The caption is read by the frozen
300M `small-code-py` (`tiny` only ever read children's stories), and its vectors go into the flow
model where G4a's went.

Shapes had a few thousand distinct sentences, so `pictures/text.py` encoded each once and kept
it. COCO has 590,000. They cannot all be kept (`cache_limit`), so most are encoded every step,
and one at a time a 300M would be the slowest part of training. So the uncached captions of a
batch go through the model **together**, padded to the longest.

## Why the end, and not the front

The text model is causal: the vector at token *j* is computed from tokens 0..*j* only. Put the
padding **after** a caption and its real tokens never see it, so each row of the batch is exactly
the caption read alone; the pad positions are then zeroed and masked. Put the padding **before**
the caption and every real token has read a run of padding first: different positions (RoPE
rotates by position), different context, different vectors. No error, no NaN; the flow model
would simply train on one thing and, drawing a single caption, read another.

The check encodes three captions of different lengths together and each one alone, with a real
(small, random) transformer, and requires the vectors to match.

---

## Exercise: pad at the front

1. Run the check. Green.
2. In `aksharallm/pictures/text.py`, inside `_many`, change
   `batch[r, : len(i)] = torch.tensor(i, dtype=torch.long)` to
   `batch[r, longest - len(i):] = torch.tensor(i, dtype=torch.long)`.
3. Run the check. **It should fail.**
4. Put it back. Green.

> **What you just saw.** Left-padding is what you would write for a model that generates after
> the prompt, and it is wrong here, where the vectors at the caption's own positions are what
> is read. Nothing downstream can tell: the loss still falls. The only thing that knows is a
> test comparing the batch with the caption alone.

## Run it yourself

```bash
python -m aksharallm.pictures coco                      # after the three COCO zips are in data/coco/raw/
scripts/audio.sh ae-coco && scripts/audio.sh clip-coco && scripts/audio.sh flow-coco
python -m aksharallm.pictures coco-report flow-coco --clip clip-coco --guidance 3
```

Or open the **Pictures** tab: "From a sentence to a photo (G4c)".
