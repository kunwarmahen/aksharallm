---
id: distillation
title: A small model taught by a big one
doc: docs/24-distillation.md
files:
  - aksharallm/train/distil.py
  - configs/tiny-distil.yaml
verify: tests/test_distil.py::test_the_t_squared_factor_keeps_the_soft_gradient_on_one_scale
prereqs: [quantization]
minutes: 30
summary: Train a student on the teacher's whole prediction, not just the next word — and the one factor that keeps a temperature from quietly switching the teacher off.
---

# 22. A small model taught by a big one

[Quantization](docs/11-quantization.md) kept the model and stored it in fewer bits. Distillation
makes a **different, smaller model** and trains it to imitate the big one.

Ordinary training shows the model one right answer per position: after "The capital of
France is", the next token was " Paris", and every other token counts as equally wrong. A
trained model knows more than that. It might give " Paris" 0.6, " a" 0.1 and " banana"
almost nothing. Distillation shows the student that whole spread, so it learns which wrong
answers were *nearly* right.

The loss has two parts, mixed by `alpha`:

```
soft = T² · KL( softmax(teacher / T) ‖ softmax(student / T) )
hard = cross_entropy(student, real next token)
loss = alpha · soft + (1 − alpha) · hard
```

`T` is a temperature. Dividing the logits by `T = 2` flattens both distributions, so the
teacher's small probabilities carry real weight instead of rounding to nothing.

## Why only between our own models

The soft loss compares the teacher's probability for token id 8,412 with the student's
probability for the same id. That only works if id 8,412 is the same string to both, so they
must share one tokenizer. `check_teacher` compares the tokenizer **files**, not their paths
or their sizes. Two different 8,192-token vocabularies have the same size, and a run between
them would train happily and learn nonsense.

---

## Exercise: take out the T²

Softening by `T` also shrinks the soft loss's gradient, by `1/T²`. The `T²` in front puts it
back. Without it, raising the temperature would quietly turn the teacher down relative to
the real next token, and `alpha` would no longer mean what the config says.

1. Run the check. It passes: the soft gradient at `T = 4` is within 4x of its size at `T = 1`.
2. In `aksharallm/train/distil.py`, find `_chunk_losses` and delete `* temperature ** 2`
   from the line that computes `soft`.
3. Run the check. **It should fail**: the gradient at `T = 4` is now about 16x smaller.
4. Put it back. Green.

> **What you just saw.** Nothing crashes without the `T²`. The run trains, and the student
> gets a little better. It just learns mostly from the hard labels, which is ordinary
> pretraining with an expensive teacher attached. The loss curve cannot tell you that.

## Run the experiment yourself

```bash
scripts/experiment.sh tiny-student      # a ~5M student from scratch (the control)
scripts/experiment.sh tiny-distil       # the same student, taught by the 13.8M tiny
```

Or press Start on either run in the portal. Compare the two **val** losses: both are the
student's plain cross-entropy. Do not compare the training losses, because the distilled
run's is a mix of both terms. Its step line shows `ce`, the hard term alone, which is the
comparable number.
