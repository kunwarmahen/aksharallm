# 15. Mixture of experts: more parameters than you compute with

Every model in this project so far spends the same arithmetic on every token. Doubling its
knowledge means doubling what each token costs. A **mixture of experts** breaks that link:
the feed-forward network in each block becomes N of them plus a small **router**, and each
token is sent to only the top-k. Parameters go up by roughly N/k; compute per token does
not move.

The reason it fits *this* project is where the parameters already are:

|                  | 300M blended base | 13.8M TinyStories |
|------------------|-------------------|-------------------|
| embedding (tied) | 33.6M (11%)       | —                 |
| attention        | 62.9M (21%)       | —                 |
| **FFN**          | **202.9M (68%)**  | **7.1M (51%)**    |

Two thirds of the real model is the thing MoE replaces.

```mermaid
flowchart LR
  X["token<br/>(one row of the batch)"] --> R{{"router<br/>d_model x N"}}
  R -->|"top-2 of 8"| E3["expert 3"]
  R -->|"top-2 of 8"| E6["expert 6"]
  R -. "not run" .-> EX["experts 0,1,2,4,5,7"]
  E3 --> S(("weighted<br/>sum"))
  E6 --> S
  S --> O["output"]
```

---

## The one thing that actually goes wrong

Nothing in the training objective wants the experts to be *used*. A few of them win
slightly early, get more gradient, get better, and win more. The rest never train.

The model quietly becomes a smaller dense one carrying dead weight — **and the loss curve
looks fine while it happens.** It is a little worse than it should be, which is
indistinguishable from a model that is simply a little worse. That is the whole reason this
chapter has an auxiliary loss and a chart.

```mermaid
flowchart TD
  A["step 100<br/>expert 3 is 2% ahead"] --> B["it gets more tokens"]
  B --> C["so it gets more gradient"]
  C --> D["so it gets better"]
  D --> B
  D --> E["step 2000: experts 0,1,2,4-7 receive nothing<br/>loss curve: <b>normal</b>"]
```

Two things prevent it, and both are in `model/moe.py`:

### The load-balancing loss

```
L_aux = alpha * N * sum_i  f_i * P_i
```

* `f_i` — the **fraction of tokens dispatched** to expert i. Discrete, so no gradient flows
  through it.
* `P_i` — the **mean router probability** for expert i. This is where the gradient lives.

The product is minimised when both are uniform, and the `N` factor makes the value
scale-free: it is 1.0 at perfect balance whether you have 4 experts or 64, so `alpha` means
the same thing everywhere. Using `f` alone would have no gradient at all; using `P` alone
lets the router keep a flat *average* while still sending every individual token to one
expert.

### The z-loss

Softmax is shift-invariant, so nothing in the main loss stops the router's logits growing
without bound — and once they do, one probability saturates at 1 and the router stops being
trainable at all. `z_alpha * mean(logsumexp(logits)^2)` pins the scale. It is a leash, not
an objective: 1e-3.

### And the chart

Per-expert token counts are computed **every step** — it costs a `bincount` — printed on the
step line and written to `train_log.jsonl`, so the portal draws one line per expert:

```
experts 0.96 bal (12-13%)
```

`balance` is `(1/N) / max_share`: **1.0** when every expert gets an equal share, **1/N** when
one takes everything. If a line peels away from the others on that chart, stop the run.

---

## Two shapes, answering different questions

### 1. Matched *active* parameters — the honest experiment

Each expert is `d_ff / k` wide, so routing to k of them costs exactly what the dense FFN
cost. Identical FLOPs per token, more total capacity. **This is the claim MoE actually
makes**, so it is the thing to test; a config with 8 full-width experts would be a bigger
model winning by being bigger, which proves nothing.

| N | k | expert d_ff | FFN total | FFN active | model total |
|---|---|---|---|---|---|
| dense | — | 1024 | 7.1M | 7.1M | 13.8M |
| 4 | 2 | 512 | 14.2M | 7.1M | 20.8M |
| **8** | **2** | **512** | **28.3M** | **7.1M** | **35.0M** |
| 16 | 2 | 512 | 56.6M | 7.1M | 63.3M |

`configs/tiny-moe.yaml` is the 8/2 row, and it is identical to `configs/tiny.yaml` in every
other field — same data, same depth and width, same optimiser, same batch, same steps, same
seed — because the dense run of that config already exists and reached **val 1.472**.

### 2. Sparse upcycling — the affordable one

Clone a *trained* FFN into N experts, add a router, continue pretraining for a fraction of
the original budget.

**This is the only affordable route to an MoE at 300M**, and the constraint that decides it
is tokens, not memory: Chinchilla-optimal for the resulting 1.7B model is ~34B tokens, the
blend is 10B, and Phase 2 has spent half of them. A fresh MoE pretrain would produce a badly
undertrained model *and* burn the run in progress.

**Upcycling is identity-at-init, exactly like LoRA's `B = 0`** (docs/12). If every expert is
a copy of the trained FFN, the router's gate is zeros, and the top-k weights are renormalised
to sum to 1, then the upcycled model computes *exactly* what the dense model computed on
step 0:

```mermaid
flowchart LR
  D["trained dense FFN"] -->|"copy x N"| E["N identical experts"]
  D2["router gate = 0"] --> U["uniform probabilities"]
  U -->|"top-k, renormalised<br/>weights sum to 1"| I["output = FFN(x), exactly"]
  E --> I
  I -->|"one gradient step<br/>breaks the symmetry"| T["experts differentiate"]
```

There is a test that asserts `torch.equal`, not `allclose` — an approximate identity would
mean the copy is subtly wrong, and you would never find out.

> **A trap this exposes:** a zero router is *only* safe when the experts are identical. With
> a zero gate every probability is equal, `topk` breaks the exact ties by index, and every
> token picks the same k experts — so a *from-scratch* MoE initialised that way would train
> k experts and leave the rest dead. From scratch the gate gets the normal 0.02 init.

> **A second trap, only at top-1 (found 2026-10-20, planning the 300M upcycle):**
> renormalising *one* weight over itself makes it the constant 1, and a constant has no
> gradient. The router then learns only from the balancing loss — it routes *evenly*, never
> *well*, and the loss curve cannot tell you. For k = 1 the weight is `p / stopgrad(p)`:
> still exactly 1 going forward, so the identity above still holds, with gradient
> ∂ log p — the chosen expert's logit rises when scaling its output up would have helped.
> Measured on the test setup: router gradient norm **4.2e-7** with plain renormalisation
> (float rounding, not signal) against **9.8**. `tests/test_moe.py` asserts > 1e-2, because a
> `> 0` check passed the broken version.

**Upcycling a real checkpoint is one config key.** `train.init` pointed at a *dense*
checkpoint from an MoE config upcycles it in `pretrain.init_weights` — each FFN copied into
every expert, the router at zero — so `scripts/experiment.sh small-code-py-moe` needs no
conversion step. On the 300M Python specialist (`configs/small-code-py-moe.yaml`: 4 experts,
top-1, full-width): **908M parameters, 299.5M active**, and over 1,024 tokens of never-seen
Python the upcycled model's loss and every argmax match the dense model's exactly
(difference 0.0). Phase 4c trains it against a dense arm on identical data (PLAN.md).

**Step 0 of an upcycled top-1 run reads `experts 0.25 bal (0-100%, 3 dead)`, and that is
fine.** With a zero gate all four probabilities tie, so every token goes to whichever expert
`topk` breaks the tie towards (expert 0 on the GPU, expert 2 on the CPU; the order is not
specified). The three unpicked experts stay bit-identical, and so do their gate rows, so
they can only split off one at a time, when the tied group wins some token's argmax. That
happens almost immediately. Measured 2026-10-21 on `checkpoints/tiny` upcycled the same way
(4 experts, top-1, lr 1e-4 with 200 warmup steps): **all four in use by step 25**, while
the LR was still under 1.3e-5, with balance 0.87–0.97 through step 500 and no layer below
0.43. A run still showing dead experts a few hundred steps in *is* a problem.

**What it costs in memory, measured on the 3090.** 908M parameters is 13.5 GiB of fp32
weights, gradients and AdamW state before a single activation. At micro-batch 4×1024 the
first launch OOMed on every attempt: 20.7 GiB allocated plus 0.6 GiB stranded in
fragments, and a 512 MiB fp32 logits cast refused on the *second* micro-batch, the first one
that runs with gradients already allocated. A CPU count of the tensors autograd saves had
predicted 7.9 GiB of activations (0.305 GiB a layer against dense's 0.226), ~21.5 GiB in
all. The arms now run 3×80 (the same 245,760 tokens a step) with
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. That fits: 22.2 GB peak over 30 steps,
balance 0.94 by step 20.

**At this margin, anything else on the card is fatal, and "unloaded" means nothing if
something keeps asking.** The second night's real launches all died on step 1 while their
smoke tests passed. Every OOM message named one more process, holding 0.7 to 5.6 GiB with a
fresh PID each time. It was an Ollama runner, reloaded every ~37 s by another service's
scheduled requests and fitting itself into whatever VRAM was free while the trainer was
still allocating. Read the *other processes* in an OOM message before shrinking the batch.
The third night's OOMs named no other process. Every *resume* carried 3.4 GiB more than a
fresh run at the same point: a second copy of all 908M weights, kept on the GPU by the trainer (docs/05 §
checkpointing, fixed 2026-10-23). An upcycled MoE is where that shows first: its total
parameters, not its active ones, set every one of these bills.

**The first full night (2026-10-24), steps 278 → 844:** 22.2 GB flat, ~27 s/step. The
router stayed balanced the whole way (0.92–0.98, every expert between 23% and 27% of tokens,
none dead). That's what the balancing loss is for, and it held on a real 908M model as
it had on `tiny`. Nights 5 and 6 (to step 1,982) were the same: 0.93–0.99, with no expert
drifting. A daytime session finished the arm on 2026-10-26 (step 2,999, val 1.1462) with the
shares still between 24% and 26%: 3,000 steps and no collapse.

**Balanced is not the same as better.** The dense control (same data, same batches,
same schedule) led at every step measured so far: val 1.1833 vs 1.1914 at step 500 and
1.1780 vs 1.1871 at 1,000, never-seen Python 1.4827 vs 1.4923 at 1,000. It ran at 10.35
s/step against the MoE's ~26. Two things not to misread along the way: the MoE trainer's
loss lines include the balance and z-losses (~0.29 at 24 layers, exactly the step-0 gap
between the arms), so only val and held-out numbers compare; and the full comparison waits
for both arms' step 3,000 (PLAN.md § 4c).

**The result at step 3,000 (2026-10-29): a tie on what it was for, at 2.5× the cost.**
Never-seen Python 1.4466 (MoE) vs 1.4470 (dense), from 1.465 for both. MBPP 15 vs 19, sampled
HumanEval pass@10 4.08% vs 4.91%. No paired test separates the arms (MBPP p 0.34, pass@1 p 0.14,
pass@10 p 0.40). The dense arm did it at 10.4 s/step and 8.5 GB against the MoE's 25.7 s and
22.6 GB. So on the 13.8M from scratch, matched *active* parameters won by 4.6% (above), and
on the 300M, upcycled for 737M tokens of continued training, three times the parameters bought
nothing measurable. The likely reason is too few tokens for copies of one FFN to diverge
into specialists: the router stayed near-perfectly balanced, so each expert saw a random
quarter of the data. It's a hypothesis, untested here. Full table: PLAN.md § 4c result.

---

## What it measured

Both runs: 13.8M-scale TinyStories, 8,000 steps, same data, seed, batch, optimiser and
schedule. The only difference is the six MoE lines. **37m27s** on a 3090.

| step | dense val | MoE val | delta |
|---|---|---|---|
| 1,000 | 1.9083 | 1.8547 | −0.054 |
| 3,000 | 1.6532 | 1.5940 | −0.059 |
| 5,000 | 1.5470 | 1.4830 | −0.064 |
| 7,000 | 1.4847 | 1.4165 | −0.068 |
| **7,500 (best)** | **1.4764** | **1.4081** | **−0.068 (−4.6%)** |

**The mixture of experts wins by 0.068 validation loss at identical FLOPs per token**, and
the gap *widens* monotonically rather than closing — the extra capacity keeps paying as the
dense model starts to saturate. On the harness, same 200 items:

| | dense | MoE |
|---|---|---|
| perplexity | 4.364 | **4.071** (−6.7%) |
| ARC-Easy | 24.0% ±3.0 | 27.0% ±3.1 |
| PIQA | 54.5% ±3.5 | 57.5% ±3.5 |

Perplexity confirms the val-loss result. **The two multiple-choice numbers do not mean
anything here** — chance is 25% and 50%, the error bars are ±3, and a 13.8M model trained on
children's stories has no business answering either. They are listed because leaving them
out would be picking the flattering half.

What it cost: **MFU 52% against the dense run's ~57%**, and 35.0M parameters stored to
compute with 7.1M of them. That is the trade in one line — *memory for quality at fixed
compute*, which is the opposite of the trade quantization makes (docs/11) and the same shape
as the one LoRA makes (docs/12).

The router stayed balanced the whole run — `experts 0.97 bal (12-13%)`, against a perfect
12.5% — which is what the auxiliary loss is for and is not something to take for granted.

---

## Why the dispatch is a sort

The obvious implementation runs every expert on every token and multiplies by a 0/1 mask.
It is trivially correct and it throws away the entire point — you pay N experts' compute per
token.

What `MoEFeedForward.forward` does instead:

```mermaid
flowchart LR
  A["(B,T,d) -> (N,d)<br/>flatten the batch"] --> B["router: top-k<br/>(N,k) weights + ids"]
  B --> C["flatten to N*k pairs<br/>argsort by expert id"]
  C --> D["expert e owns a<br/>CONTIGUOUS slice"]
  D --> E["one matmul per expert<br/>over its slice"]
  E --> F["scatter-add back,<br/>weighted by the router"]
```

The Python loop is over **experts** (8 of them), not over tokens, and each iteration is one
real matmul. No token is dropped — there is no capacity factor, so an expert that receives
90% of a batch still processes all of it.

The honest cost: **MFU falls**. A sort plus N smaller matmuls uses the card less well than
one big matmul. Measured on the 13.8M model: the dense run does ~57% MFU, this does **~52%**,
and `torch.compile` is *off* in `configs/tiny-moe.yaml` because dynamo recompiles on the
changing slice shapes and costs more than it saves at this size. The expert weights are also
stacked `nn.Parameter`s rather than `nn.Linear`s — that is what makes the grouped matmul
possible, and it has consequences below.

---

## What MoE breaks

| area | what happens | what the code does |
|---|---|---|
| **quantization** | experts are `nn.Parameter`, so `linear_layers()` never sees them: 68% of the 300M would stay float while the report claims 2.8x. And the router's gate *is* an `nn.Linear`, so it would be quantized — the one layer that must never be, because a wrong route sends the token to a **different expert**, not to a slightly wrong number. | `quantize_model` **refuses** an MoE checkpoint with that explanation |
| **LoRA** | `apply_lora` would silently adapt attention only | allowed, but the report lists the experts and the router as skipped, with the reason |
| **eval** | reporting one parameter count is misleading | `num_params()` and `num_active_params()` are both available; quote both |
| **inference** | nothing. Routing is per token, the KV cache is untouched | a test drives prefill *and* a single-token decode step |

---

## Reading a run

```bash
scripts/experiment.sh tiny-moe          # or the portal's Start button
tail -f train_tiny-moe.log
```

The step line grows one field:

```
step 760 | loss 1.9901 (ema 2.0265) | ppl 7.6 | ... | experts 0.93 bal (12-13%)
```

and the portal's dashboard grows one chart — **Expert routing**, one line per expert with a
rule at the even share — which appears only for MoE runs.

Two things to know when reading the numbers:

* **The training loss of an MoE run includes the auxiliary term** (~0.01 with the default
  alpha). The *validation* loss does not: `Transformer.forward` adds the aux only while
  `self.training`, precisely so val stays a plain cross-entropy and stays comparable with the
  dense baseline's 1.472. Compare val, not train.
* **`experts 0.96 bal (12-13%)` is healthy.** `0.5 bal` with half the experts near zero is
  collapse, and it is worth stopping for.

---

## The code, in reading order

One new file, and five places that had to notice it. [doc 4](04-model.md) first — this
replaces exactly one component of it.

| # | file | what to look for |
|---|---|---|
| 1 | [`configs/tiny-moe.yaml`](../configs/tiny-moe.yaml) | six lines different from `configs/tiny.yaml`. Diff them — that is the whole experiment |
| 2 | [`model/moe.py`](../aksharallm/model/moe.py) → `Router.forward` | logits → softmax → top-k → renormalise (and, at k = 1, `p / stopgrad(p)` so the router still learns), and the counts it returns. Then `_balance_loss` (`f_i · P_i`, and why neither term works alone) and `_z_loss` |
| 3 | same file → `MoEFeedForward.forward` | the sort: flatten to `N*k` pairs, `argsort` by expert id, one matmul per **contiguous** slice, scatter-add back. The loop is over 8 experts, not over tokens |
| 4 | same file | `upcycle_state_dict` — copy the trained FFN N times, zero the gate; identity at init, exactly like LoRA's `B = 0`. Then `moe_stats` and `MoEStats.shares` |
| 5 | [`aksharallm/config.py`](../aksharallm/config.py) | `n_experts`, `moe_top_k`, `moe_expert_d_ff`, `moe_aux_alpha`, `moe_z_alpha`, `moe_every`, plus `is_moe()` / `moe_layer()` |
| 6 | [`model/transformer.py`](../aksharallm/model/transformer.py) | the one line in `Block.__init__`, then `moe_aux_loss` and where `forward` adds it **only while `self.training`** — which is why val loss stays comparable with the dense baseline. Also `num_active_params` |
| 7 | [`train/pretrain.py`](../aksharallm/train/pretrain.py) | the `experts` field on the step line and `moe` in the jsonl — the chart's data source |
| 8 | [`scripts/experiment.sh`](../scripts/experiment.sh) | the launcher for Phase-1-scale runs, same contract as `phase2.sh` |
| 9 | [`quant/convert.py`](../aksharallm/quant/convert.py) · [`lora/inject.py`](../aksharallm/lora/inject.py) | `_refuse_moe`, and the skipped-layer reporting — the "what MoE breaks" table above, in code |

What pins it: `tests/test_moe.py` — 25 tests, including the upcycling one that asserts
`torch.equal` rather than `allclose`, and the sorted dispatch checked against a naive masked
implementation. Break identity-at-init on purpose in [lesson 13](lessons/13-moe.md).
