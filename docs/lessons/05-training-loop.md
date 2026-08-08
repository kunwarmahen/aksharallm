---
id: training-loop
title: The training loop, and the learning rate that has to warm up
doc: docs/05-pretraining.md
files:
  - aksharallm/train/pretrain.py
  - aksharallm/train/schedule.py
verify: tests/test_pipeline.py::test_warmup_is_linear_and_peaks_at_base_lr
prereqs: [kv-cache]
minutes: 30
summary: Forward, loss, backward, step — plus gradient accumulation, and why the learning rate starts near zero and ends near zero.
---

# 5. The training loop, and the learning rate that has to warm up

Stripped of everything else, this is all of it:

```python
logits, loss = model(x, targets=y)   # forward
loss.backward()                      # gradients
optimizer.step()                     # move the weights
optimizer.zero_grad()                # forget the gradients
```

Everything around it in `pretrain.py` is there for one of three reasons: **it makes this
fit in memory**, **it makes it survive being interrupted**, or **it tells you what is
happening**.

## Gradient accumulation: a batch bigger than the card

Big batches train more stably, and a 300M model with a 1,024-token window will not fit a
large batch on a 24 GB card. So the batch is split: run several small ones, let the
gradients pile up, and step once at the end.

```
batch_size 12  x  grad_accum 20  x  seq_len 1024  =  245,760 tokens per step
```

The optimiser cannot tell the difference. This is the single most useful trick for training
a model larger than your GPU "allows".

## The learning rate is a curve, not a number

```
lr
 |      ______
 |     /      \____
 |    /            \______
 |___/                    \____
     warmup      cosine decay
```

**Warmup** — the first few hundred steps start near zero and ramp up. At initialisation the
weights are random and the gradients are enormous; taking full-size steps immediately can put
the model somewhere it never recovers from. This is one of the most common causes of a loss
curve that spikes and never comes back.

**Cosine decay** — the rate falls smoothly to a floor. Big steps early to find the right
region, small steps late to settle into it.

Read `docs/05-pretraining.md` for what mixed precision and gradient clipping are doing, then
open `aksharallm/train/schedule.py` in the **Code** tab.

---

## Exercise: remove the warmup

1. Run the check. It passes — it asserts the rate rises linearly and reaches exactly the
   configured maximum at the end of warmup.
2. In `aksharallm/train/schedule.py`, find the warmup branch and make it return the full base
   rate immediately.
3. Run the check. **It should fail**, showing the rate at an early step is already at
   maximum.
4. Put it back. Green.

> **What you just saw.** A missing warmup does not fail — it trains. It just sometimes
> produces a model that is worse than it should be, and sometimes produces a loss spike at
> step 30 that looks like bad data. Schedule bugs are diagnosed by their *shape*, which is
> why the portal draws the learning rate as a curve you can look at.

## Watch a real one

```bash
scripts/experiment.sh tiny        # ~35 minutes on a free GPU, or press Start in the portal
```

Watch the **Learning rate** chart climb and then fall, and the loss drop fastest while the
rate is at its peak.
