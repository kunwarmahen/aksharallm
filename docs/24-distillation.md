# 24. Distillation: a small model taught by a big one

> [Doc 11](11-quantization.md) made a trained model smaller by storing its weights in fewer
> bits. This chapter makes a *different*, smaller model: one with fewer weights, trained to
> imitate the big one. The big model is the **teacher** and the small one is the
> **student**. The student learns from the teacher's whole prediction at every position, not
> just from the one token that actually came next.

```mermaid
flowchart LR
    X["a batch of text"] --> T["teacher<br/>(frozen, trained)"]
    X --> S["student<br/>(small, training)"]
    T --> P["teacher's distribution<br/>softened by T"]
    S --> Q["student's distribution<br/>softened by T"]
    P --> KL["soft loss<br/>T² · KL(p ‖ q)"]
    Q --> KL
    S --> CE["hard loss<br/>cross-entropy vs<br/>the real next token"]
    KL --> L["loss = α·soft + (1−α)·hard"]
    CE --> L
```

## Why a distribution teaches more than a label

Ordinary pretraining ([doc 5](05-pretraining.md)) shows the model one right answer per
position: after "The capital of France is", the next token was " Paris", and every other
token is equally wrong. A trained model knows more than that. It puts, say, 0.6 on " Paris",
0.1 on " a", 0.05 on " the" and almost nothing on " banana". That spread says which wrong
answers are *nearly* right, which is something a single label cannot say.

Distillation hands the student that spread. Two things make it work:

- **A temperature `T`.** Both distributions are computed from `logits / T`. At `T = 1` the
  teacher's tail is tiny, so the student barely sees it. At `T = 2` or `4` it flattens out,
  and the small probabilities carry real weight. This repo uses `T = 2`.
- **A factor of `T²` on the soft loss.** Softening also shrinks the soft loss's gradient,
  by `1/T²`. Without the factor, raising the temperature would quietly turn the soft term
  down, and `α` would stop meaning what it says. `tests/test_distil.py` checks that the
  gradient's size at `T = 4` stays within 4x of its size at `T = 1`. Without the factor it
  would be about 16x smaller.

The hard term stays, weighted `1 − α`, because the teacher is wrong sometimes and the real
text is not. This repo uses `α = 0.5`. With `α = 0` the loss is exactly ordinary
pretraining, which is how the experiment's control run is defined.

## Why only between our own models

The soft loss compares the teacher's probability for token id *n* with the student's
probability for the same id. That only means something if **id *n* is the same string to
both models**, so they must share one tokenizer ([gotcha 3](../AGENTS.md)). No Ollama model
uses our tokenizer, which is why "learn from a 31B model" is done through generated data
instead ([doc 14](14-synthetic-data.md)). Distillation here is from our 300M to a smaller
model, or from `tiny` to a smaller one.

A mismatch does not crash. The loss falls, the curve looks normal, and the student learns
nonsense. So the check is a refusal, not a warning, and it compares the **tokenizer files
themselves**. Comparing paths would reject the same file copied elsewhere. Comparing
vocabulary sizes would accept two different 8,192-token vocabularies. A test plants exactly
that second case.

The teacher also must not be a diffusion model or a tagger (their outputs are not next-token
distributions), and must have a context window at least as long as the student trains on.

## The logits are the memory

At the 300M scale a batch of `(8, 1024)` positions over a 32,768-token vocabulary gives a
logit tensor of 268M numbers: 0.5 GB in bf16, 1 GB in fp32. The KL needs the student's
softmax, the teacher's, and their logs, all in fp32, because it subtracts two
log-probabilities that are often nearly equal and bf16 loses that difference.

So the loss is computed `distil.chunk` positions at a time (128 by default). Chunking alone
saves nothing: autograd keeps every chunk's intermediates for backward. Each chunk therefore
runs under `torch.utils.checkpoint`, and backward recomputes it, so only one chunk's fp32
work is ever alive. A test checks that the chunked loss and every gradient equal the
one-piece computation, with a ragged last chunk and ignored positions in it.

The teacher's weights are the smaller problem. They can be loaded **quantized with our own
code** ([doc 11](11-quantization.md)): `distil.teacher_quant: nf4` holds a 300M teacher in
about 0.2 GB instead of 0.6. That also makes a question this repo is well placed to answer:
does a 4-bit teacher teach worse?

## It is an objective, not a trainer

There is no separate training loop. [Doc 5](05-pretraining.md) says how a new kind of
training arrives: as a new *objective* that the one loop runs. `DistilObjective` is the
fourth, after next-token prediction, masked diffusion ([doc 20](20-diffusion.md)) and the
punctuation tagger ([doc 23](23-speech-recognition.md)). A config with a `distil.teacher`
gets it; nothing else changes. Accumulation, the schedule, mixed precision, resume, the stop
file, the session log, the run report and the portal's Start and Stop buttons all work
unchanged.

The teacher is loaded on the first batch, not at startup, because only then is the device
known. Under multi-GPU that is a different card on every rank. It is frozen and in eval
mode, and a test checks its weights are unchanged after a run.

## The experiment: does a teacher help?

A controlled pair at the TinyStories scale, the same shape as the MoE experiment
([doc 15](15-moe.md)):

| run | config | student | teacher |
|---|---|---|---|
| control | `configs/tiny-student.yaml` | ~5.3M (d 256, 4 layers) | none |
| distilled | `configs/tiny-distil.yaml` | the same | `tiny` (13.8M, val 1.472) |

Everything else is identical: data, seed (and therefore the batches, gotcha 16), schedule,
8,000 steps. Both report the student's **plain cross-entropy** as their validation number,
the same number an ordinary run reports, so the two val curves go on one axis. The gap
between them is what the teacher bought.

```bash
scripts/experiment.sh tiny-student      # the control
scripts/experiment.sh tiny-distil       # the same student, taught by tiny
```

Or press **Start** on either run in the portal's dashboard. Its Configuration panel shows
the teacher, temperature and α for the distilled run.

**Not run yet.** The CPU smoke run behaves as expected: the hard term starts at 9.09
against ln(8,192) = 9.01, which is what an untrained student should score. The first real
result belongs here.

### Reading a run

The step line carries both halves: `| kd 1.234 | ce 2.345`.

- **`ce` is the number to compare with the control's training loss.** It is the hard term
  alone, which is exactly what an ordinary run calls its loss.
- **`loss` (and its `ppl`) is the mix**, so it sits on a different scale from the control's.
  Do not put the two training-loss curves on one chart. The val curves are the comparison.
- **`kd` should fall quickly and then flatten.** It cannot reach zero: the student is too
  small to copy the teacher exactly, and that gap is the cost of the smaller model.
- **Throughput and MFU count the student only.** Every batch also costs a teacher forward
  pass, so tokens per second are lower than the control's for the same student.

## What comes after the TinyStories pair

1. **The real one: the 300M base → a ~100M student**, then SFT on the student. This was
   agreed in this order on 2026-08-01: distil the base first, because the base is what the
   quantized 300M is compared against.
2. **Distilled ~100M vs `gptq-nf4-g64` of the 300M at a matched ~205 MB** (doc 11). Two ways
   to make the same model small, compared at the same file size.
3. **A 4-bit teacher vs a bf16 teacher** for the same student (`teacher_quant: nf4`).

Not built: initialising the student from the teacher's layers (worth testing against from
scratch), and an MoE student (refused, because its load-balancing loss lives on a path this
objective does not take, and a router without it can collapse unseen).

---

## The code, in reading order

One new file and four places that had to notice it.

| # | file | what to look for |
|---|---|---|
| 1 | [`configs/tiny-distil.yaml`](../configs/tiny-distil.yaml) | diff it against `configs/tiny-student.yaml`: the `distil:` block is the whole experiment |
| 2 | [`train/distil.py`](../aksharallm/train/distil.py) → `distil_loss` and `_chunk_losses` | the two terms, the `T²`, the fp32 cast, the mask on ignored positions, and `checkpoint` around each chunk |
| 3 | same file → `check_teacher` | the refusals: vocabulary, tokenizer *contents*, a non-next-token teacher, a shorter window |
| 4 | same file → `load_teacher` and `DistilObjective` | frozen, eval mode, optional `int8`/`nf4` through `quant/convert.py`; loaded at the first batch; the `kd`/`ce` stats |
| 5 | [`aksharallm/config.py`](../aksharallm/config.py) → `DistilConfig` | `teacher`, `temperature`, `alpha`, `teacher_quant`, `chunk` |
| 6 | [`train/pretrain.py`](../aksharallm/train/pretrain.py) → `objective_for` | the one branch that picks this objective when `distil.teacher` is set |
| 7 | [`portal/runs.py`](../aksharallm/portal/runs.py) | the two `LAUNCHERS` entries, and the `teacher` row in `_config_summary` |

What pins it: `tests/test_distil.py`, 19 tests. They cover the loss against the KL written
out by hand, chunked equal to unchunked, the `T²` scale, ignored positions dropping out of
both terms, every refusal, a frozen teacher in float/int8/nf4, and one real run through
`pretrain.main`. Every test that guards a mutation was seen to fail against it first, and
one gap was found that way: no test covered the soft term ignoring masked positions, so one
was added. Break the `T²` on purpose in [lesson 22](lessons/22-distillation.md).
