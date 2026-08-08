---
id: moe
title: More parameters than you compute with
doc: docs/15-moe.md
files:
  - aksharallm/model/moe.py
verify: tests/test_moe.py::test_upcycled_model_is_exactly_the_dense_model_at_init
prereqs: [lora, quantization]
minutes: 35
summary: A router and N experts in place of one feed-forward network — and the failure that is completely invisible in the loss curve.
---

# 13. More parameters than you compute with

Every model up to here spends the same arithmetic on every token: knowing more means each
token costs more. A **mixture of experts** breaks that link. The feed-forward network in each
block becomes N of them plus a small **router**, and each token is sent to the top-k.

It fits this project because of where the parameters already are: in the 300M model the
feed-forward layers are **68%** of everything. That is the part MoE replaces.

Measured here at Phase 1 scale, with each expert `d_ff/k` wide so that top-2 costs exactly
what the dense layer cost — identical FLOPs per token, 35.0M parameters stored against 7.1M
used:

| | dense | 8 experts, top-2 |
|---|---|---|
| val loss | 1.4764 | **1.4081** |

Better, at the same compute per token, on the same data and seed. That is the claim MoE
makes, and it held.

## The failure you cannot see

Nothing in the training objective wants the experts to be *used*. A few win slightly early,
receive more gradient, get better, and win more. The rest never train.

The model quietly becomes a smaller dense one carrying dead weight — **and the loss curve
looks completely normal while it happens.** It is a little worse than it should be, which is
indistinguishable from a model that is simply a little worse.

Hence a load-balancing loss, a router z-loss, per-expert token counts on every step line, and
a chart in the portal with one line per expert. Watch `experts 0.96 bal (12-13%)`: `balance`
is 1.0 when every expert gets an equal share and 1/N when one takes everything.

---

## Exercise: break identity-at-init

Getting an MoE out of an existing 300M model is done by **sparse upcycling**: copy the trained
feed-forward network into N experts, add a router, keep training. It works because it is an
*exact identity* at step 0 — identical experts, a zero router, and top-k weights renormalised
to sum to 1 mean the upcycled model computes precisely what the dense model computed.

You have seen this before. It is LoRA's `B = 0` again: **start from the trained thing, not
near it.**

1. Run the check. It passes — and note it asserts `torch.equal`, not `allclose`. An
   approximate identity would mean the copy is subtly wrong, and nothing else would say so.
2. In `aksharallm/model/moe.py`, find `upcycle_state_dict` and initialise the router's gate
   with small random values instead of zeros.
3. Run the check. **It should fail**: the upcycled model no longer reproduces the dense one.
4. Put it back. Green.

> **What you just saw.** A random gate still routes, still trains, still converges — to
> something slightly worse, having spent its first thousand steps recovering from a
> perturbation you introduced for no reason. There is no error message for that.

## Run the experiment yourself

```bash
scripts/experiment.sh tiny-moe      # ~37 minutes on a free GPU, or Start in the portal
```

Watch the **Expert routing** chart. Eight lines that stay together are a healthy router; one
climbing while the others sink is the failure above, and it is worth stopping the run for.
