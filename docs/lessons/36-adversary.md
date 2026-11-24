---
id: adversary
title: Sharper photos, and the sign that decides which way the decoder learns
doc: docs/26-pictures.md
files:
  - aksharallm/pictures/adversarial.py
  - aksharallm/pictures/train_ae.py
  - aksharallm/pictures/measure.py
verify: tests/test_sharper.py::test_a_decoder_step_on_the_adversarial_loss_makes_its_patches_read_as_more_real
prereqs: [padding]
minutes: 20
summary: An autoencoder trained on pixel error alone learns that a smear is the cheapest guess, and keeps about half a real photo's fine detail. A patch discriminator makes a smear expensive, since real photos have texture. The decoder's adversarial loss is minus the discriminator's score on its own patches. Drop the minus and the decoder learns to look more fake, while every loss curve still moves.
---

# 36. The sign that decides which way the decoder learns

`ae-coco` ([doc 26](docs/26-pictures.md) § Sharper photos) reconstructs COCO photos at 27.7 dB,
which sounds fine, and keeps only **54%** of their fine detail (`measure.sharpness`: power above a
quarter cycle per pixel, against the real photos'). L1 + L2 asks for exactly that. Where the
decoder cannot tell where an edge falls, the average of the places it might be costs least, and an
average of edges is a smear. Every drawing `flow-coco` makes goes through this decoder, so every
drawing inherits the smear.

`pictures/adversarial.py` adds a second network, a **patch discriminator**: strided convolutions
to an 8×8 map of scores, each one judging a 38-pixel patch as real or reconstructed. It is trained
to tell the two apart (`d_loss`, a hinge). The decoder gets one more term, `g_loss`, which says
"make your patches score as real". A smear does not score as real: real photos have texture.

## Which way is "real"?

The discriminator gives real patches high scores. So the decoder's loss is

    g_loss = -mean(score of its own patches)

and minimising it *raises* that score. One character, a minus sign, decides whether the decoder
learns to look more like a photo or less. Without it the decoder still trains, the discriminator
still trains (it gets an easier job), and the logged numbers still move. The run would simply be
learning to be obviously fake, and the reconstruction loss beside it would hide that, because it
keeps pulling the pixels back.

The check takes a small discriminator, one fake picture, and one gradient step on `g_loss`
alone, and requires the discriminator's score on that picture to have gone **up**.

---

## Exercise: drop the minus

1. Run the check. Green.
2. In `aksharallm/pictures/adversarial.py`, inside `g_loss`, change `return -fake_logits.mean()`
   to `return fake_logits.mean()`.
3. Run the check. **It should fail.**
4. Put it back. Green.

> **What you just saw.** An adversarial term is two networks pulling against each other, and
> each pull has a direction that nothing else in the run checks. The discriminator's hinge has
> the same trap (swap its two signs and it learns that reconstructions are real);
> `test_the_hinge_losses_point_the_right_way` pins that one. Two more things keep the adversary
> useful rather than harmful, both in `adversarial.py`: its weight adapts to the reconstruction
> gradient at the decoder's last layer, and it starts only once the autoencoder already
> reconstructs (`start_step`).

## Run it yourself

```bash
python -m aksharallm.pictures ae-report ae-coco                 # how sharp is the pixel-loss one?
scripts/audio.sh ae-coco-gan                                    # the same, plus the discriminator
python -m aksharallm.pictures ae-report ae-coco-gan --against ae-coco
```

Or open the **Pictures** tab: "From a sentence to a photo (G4c)" → *How sharp is it?*
