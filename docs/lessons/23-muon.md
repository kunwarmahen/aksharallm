---
id: muon
title: A second optimizer, and the one factor that lets it share AdamW's learning rate
doc: docs/05-pretraining.md
files:
  - aksharallm/train/muon.py
  - configs/tiny-adamw.yaml
  - configs/tiny-muon.yaml
verify: tests/test_muon.py::test_a_muon_step_has_the_rms_adamw_would_have
prereqs: [training-loop]
minutes: 30
summary: AdamW scales each number in a gradient on its own. Muon treats a weight matrix as a matrix and gives every direction the same step. One small factor keeps that step the size AdamW's would have been.
---

# 23. A second optimizer: Muon

The [training loop](docs/05-pretraining.md) has used AdamW since lesson 5. AdamW looks at
every number in a gradient on its own and divides it by that number's running size. It never
notices that the numbers form a **matrix**.

A weight matrix's gradient usually has a few strong directions and many weak ones. So a step
along it moves the weights a lot in a few directions and barely in the rest. Muon fixes that
with **orthogonalisation**. It keeps the gradient's directions and sets every one of them to
the same strength.

```
gradient G ──► momentum M ──► Newton-Schulz (5 steps) ──► O ≈ U·Vᵀ   (every singular value ≈ 1)
                                                            │
             W ◄── W − lr · 0.2·√max(rows, cols) · O ◄──────┘
```

* **Newton-Schulz** is five matrix multiplications that push every singular value towards 1.
  It does the job of an SVD, and it is cheap enough to run on every step.
* **Only the block matrices use Muon.** The embedding, the RMSNorm gains and the MoE router
  stay on AdamW. `split_for_muon` decides which is which, and both kinds of parameter live in
  one optimizer object, so the schedule, checkpoint and resume code are unchanged.

## Why one learning rate can serve both

An orthogonal matrix with `max(rows, cols) = n` has entries of size about `1/√n`. A 64-wide
matrix gives entries of 0.125; a 1,024-wide one gives 0.03. Left like that, the step size
would depend on the layer's shape, and no single `lr` would suit all of them.

The `0.2 · √max(rows, cols)` factor (from Moonlight, 2025) rescales the step so its RMS is
about `0.2 · lr` whatever the shape. That is roughly the size of an AdamW step. So the Muon
run can reuse AdamW's learning rate, warmup and decay unchanged, and the two configs differ
in exactly one key.

---

## Exercise: take the scale away

1. Run the check. It passes: a Muon step on a 64×64 and a 256×64 matrix both have an RMS of
   about `0.2 · lr`.
2. In `aksharallm/train/muon.py`, find the line that computes `scale` inside the Muon step
   and replace it with `scale = 1.0`.
3. Run the check. **It should fail.** The 64×64 step is now `lr / 8` and the 256×64 one is
   `lr / 16`: the same `lr` gives each layer a different step.
4. Put it back. Green.

> **What you just saw.** Nothing crashes without the factor. A run still trains, just with
> every block taking a step of a different, smaller size, and with an `lr` that was tuned for
> AdamW and now means something else. A comparison against AdamW would then be measuring the
> learning rate, not the optimizer.

## Run the comparison yourself

```bash
scripts/experiment.sh tiny-adamw      # the control: tiny, AdamW, seeded batches
scripts/experiment.sh tiny-muon       # the same run with optim.name: muon
```

Or press Start on either run in the portal. Compare their **val** losses at the same step.
Measured here: 1.4418 for Muon against 1.4752 for AdamW. The same pair at ~100M
(`small-student` / `small-student-muon`) ended 3.267 against 3.371, and Muon also won on
ARC-Easy and PIQA ([doc 5](docs/05-pretraining.md)).
The control is a re-run rather than `tiny`'s old 1.472, because that run's batches were not
seeded, so the comparison would not be like for like.
