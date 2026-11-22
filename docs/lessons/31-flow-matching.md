---
id: flow-matching
title: Drawing by flow matching, and the direction that decides whether it draws at all
doc: docs/26-pictures.md
files:
  - aksharallm/pictures/flow.py
  - aksharallm/pictures/train_flow.py
  - aksharallm/pictures/drawer.py
verify: tests/test_flow.py::test_the_loss_is_zero_for_the_true_velocity_and_not_for_its_negative
prereqs: [guidance]
minutes: 30
summary: G2 moves pure noise to a picture along a straight line, and the model learns the line's direction, its velocity. The whole objective is one subtraction, x1 minus x0. Write it the other way round and every loss curve still falls while the model learns to turn pictures into noise.
---

# 31. Flow matching

G1 ([lesson 30](docs/lessons/30-guidance.md)) wrote a picture one code at a time. G2 draws the
way Stable Diffusion 3 and the Veo family do: start from pure noise and move the **whole**
picture a little at a time until it is a drawing ([doc 26](docs/26-pictures.md) § G2).

## The rule

Take a training picture `x1` and a sample of noise `x0`. Join them with a straight line:

```
x_t = (1 - t) * x0 + t * x1        t = 0 is noise, t = 1 is the picture
```

Along that line, every point moves with the same velocity, `x1 - x0`. So the training example
is: here is `x_t` and `t` and the caption, predict `x1 - x0`. The loss is the squared error
(`pictures/flow.py`, `flow_loss`). To draw, start at noise and take 50 steps of
`x += velocity / 50` (`sample`). If the model is perfect, the steps land on a picture exactly;
the test `test_a_perfect_velocity_field_lands_exactly_on_the_picture` checks the sampler with
a velocity field that knows the answer.

The model is the transformer of [lesson 3](docs/lessons/03-attention.md) again, run with
`causal: false` over the time step, the three caption words and 256 patches of the noisy
picture, so every patch sees every other one.

The check for this lesson builds a "model" that returns the true velocity and requires its loss
to be zero, and one that returns the negative and requires its loss to be large.

---

## Exercise: reverse the direction

1. Run the check. Green.
2. In `aksharallm/pictures/flow.py`, inside `flow_loss`, change `(x1 - x0)` to `(x0 - x1)`.
3. Run the check. **It should fail**: the true velocity now scores a large loss.
4. Put it back. Green.

> **What you just saw.** With the subtraction reversed, training still works perfectly well:
> the loss falls and every curve looks healthy. But the model has learned the velocity from
> picture to noise, and the sampler, which starts at noise and steps forward, walks away from
> every picture. Nothing about the loss can tell you which way round it is. Only a test that
> knows the true answer (or the reader looking at the result) can.

## What it did on the real model

`flow-shapes` (docs/26 § Measured: G2 against G1) drew 99.2% of the seen captions right and
**all 128** drawings of the never-seen purple triangles, against 89.0% and 70.3% for the best
G1 trained on the same pictures for the same steps (paired p 2e-43 and 7e-12). Ten Euler steps
already give 98%; two give 40%. G1 writes the picture cell by cell and cannot go back to add the
triangle it forgot; G2 revises the whole picture at every step, and counting is a question
about the whole picture.

## Run it yourself

```bash
python -m aksharallm.vision corpus --out data/vision/shapes64k --images 64000
scripts/audio.sh flow-shapes                                      # train G2 (on the card)
python -m aksharallm.pictures draw flow-shapes "two red circles" --guidance 3
python -m aksharallm.pictures gen-report flow-shapes --guidance 3 --k 32
python -m aksharallm.pictures gen-compare \
    logs/pictures/gen-imggen-shapes64k-pos-long-step29999-g3-k32.json \
    logs/pictures/gen-flow-shapes-step29999-g3-k32.json            # G1 against G2, paired
```

Or open the **Pictures** tab: G2 models are listed beside G1 in the Draw panel.
