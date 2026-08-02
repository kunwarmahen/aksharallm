# aksharallm

> **akshara** (अक्षर), Sanskrit — a *letter*, *syllable*, or *character*: the smallest unit
> of written language. It also means *imperishable*. A fitting name for a model built up from
> the smallest pieces of text, one token at a time.

**Build a language model from scratch — data, tokenizer, architecture, pretraining, and
post-training — on one RTX 3090.**

Every core piece is written by hand in PyTorch and commented to explain *why*, not just
what. Libraries are used only for plumbing (downloading datasets, the BPE trainer). If
you read this repo top to bottom you will know how an LLM actually works.

```
                    ┌──────────┐
  raw web text ───▶ │ TOKENIZE │ ───▶ 400M–10B integers on disk
                    └──────────┘
                          │
                          ▼
                    ┌──────────┐
                    │ PRETRAIN │ ───▶ base model: completes text, knows facts,
                    └──────────┘      cannot hold a conversation
                          │
                          ▼
                    ┌──────────┐
                    │   SFT    │ ───▶ chat model: follows instructions
                    └──────────┘
                          │
                          ▼
                    ┌──────────┐
                    │   DPO    │ ───▶ aligned model: prefers better answers
                    └──────────┘
                          │
                          ▼
                    ┌──────────┐
                    │  GRPO    │ ───▶ specialist: RL on a reward you can *run*
                    └──────────┘      (code passes its tests)
```

---

## Does it work?

Phase 1 (a 13.8M-parameter model trained on TinyStories for ~25 minutes) writes this:

> *Once upon a time, there was a little girl named Lily. She loved to play outside in the
> sunshine. One day, she went for a walk with her mommy and saw a big hill.*
> *"Mommy, what's that?" she asked.*
> *"It's a mountain," her mommy replied.*

Nothing in that sentence was programmed. The model learned grammar, dialogue punctuation,
and narrative structure purely from predicting the next token.

---

## Quickstart

Requires: Linux, an NVIDIA GPU (24 GB recommended), Python 3.11+, and
[uv](https://github.com/astral-sh/uv).

```bash
git clone <this repo> && cd aksharallm
uv venv --python 3.12
uv pip install -e .

# 1. Download + tokenize ~400M tokens of TinyStories   (~90 seconds)
python -m aksharallm.data.prepare tinystories \
    --out-dir data/tinystories --vocab-size 8192 --max-train-tokens 400000000

# 2. Pretrain a 13.8M-param model                       (~25 minutes on a 3090)
python -m aksharallm.train.pretrain configs/tiny.yaml

# 3. Talk to it
python -m aksharallm.infer.cli tiny --prompt "Once upon a time"

# 4. See what it can and cannot do yet
python -m aksharallm.infer.cli tiny --probes
```

That's the entire loop, end to end. Everything after this is the same code with bigger
numbers.

---

## The documentation

Read these in order. They assume no prior knowledge of machine learning.

Every chapter ends with **"The code, in reading order"** — the `.py` files it explains,
listed in the order that makes them make sense, with what to look at inside each and the
test that pins it. It points the other way too: every module's docstring names the chapter
it belongs to (`Read with: docs/03-model.md`), so a file you land in from a traceback or a
grep is one line away from the prose. `tests/test_docs.py` fails if either pointer rots.

| # | Doc | What you'll learn |
|---|-----|-------------------|
| 0 | [What we're building](docs/00-overview.md) | What an LLM is, the four training stages, why each exists |
| 1 | [Data](docs/01-data.md) | Where text comes from, why quality beats quantity, the on-disk format |
| 2 | [Tokenizer](docs/02-tokenizer.md) | Why models read "tokens" not letters, how BPE works |
| 3 | [The model](docs/03-model.md) | Attention, RoPE, SwiGLU, residual streams — every line of the transformer |
| 4 | [Pretraining](docs/04-pretraining.md) | The training loop, mixed precision, LR schedules, reading the logs |
| 5 | [Post-training](docs/05-posttraining.md) | SFT, DPO, and **GRPO** (RL on a verifiable reward) — text completer → assistant → code/math specialist |
| 6 | [Inference](docs/06-inference.md) | KV caches, sampling, and how to tell whether a half-trained model is learning |
| 7 | [Scaling up](docs/07-scaling.md) | Phase 2: a 300M model on 10B tokens, and how to size your own |
| 8 | [Troubleshooting](docs/08-troubleshooting.md) | Loss spikes, NaNs, OOM, slow training |
| 9 | [Running & watching it](docs/09-running-and-watching.md) | The scripts, stop/resume, the gated post-training stages, and the **portal** — with diagrams |
| 10 | [Quantization](docs/10-quantization.md) | Storing weights in 4 bits: group scales, RTN/GPTQ/AWQ/QAT, **NF4**, a fused Triton kernel, and why smaller isn't faster |
| 11 | [LoRA & QLoRA](docs/11-lora.md) | Fine-tuning without training the model: low-rank adapters, a 4-bit frozen base, one base + many skills, and a free DPO reference model |
| 12 | [Evaluation](docs/12-eval.md) | Is the model actually any good? MMLU/ARC/HellaSwag/PIQA scored by log-likelihood, GSM8K, HumanEval executed for real, an LLM-judge — and why 25% on MMLU is not a failure |
| 15 | [The learning path](docs/15-learning-path.md) | The repo as a course: thirteen lessons that each end in breaking real code and watching a real test go red — and why a lesson only counts once the check has been red *and then* green |
| 14 | [Mixture of experts](docs/14-moe.md) | More parameters than you compute with: a router, N experts, top-k per token — the load-balancing loss, why upcycling is an identity at init, and the collapse that is invisible in the loss curve |
| 13 | [Synthetic data](docs/13-synthetic-data.md) | Making the training set with a local teacher instead of downloading it: a seed grid instead of a temperature, tests that are **executed twice**, near-duplicate detection, and why the rejection tally is the quality signal |

---

## Repo layout

```
aksharallm/
├── configs/              YAML run configs — the only thing that changes between runs
│   ├── tiny.yaml         Phase 1: 13.8M params, TinyStories
│   ├── small.yaml        Phase 2 (pure): 300M params, FineWeb-Edu only
│   ├── small-code.yaml   Phase 2 (blended): 300M, 85% FineWeb-Edu + 15% Python
│   └── tiny-moe.yaml     the MoE experiment: tiny.yaml + 8 experts, matched active params
├── aksharallm/
│   ├── config.py         dataclass config loading + CLI overrides
│   ├── tokenizer/        byte-level BPE training and the chat template
│   ├── data/
│   │   ├── prepare.py        pretraining corpus  -> uint16 token stream
│   │   ├── prepare_blend.py  several corpora     -> blended tokenizer + per-source bins
│   │   ├── prepare_sft.py    chat corpus         -> packed blocks + loss mask
│   │   ├── prepare_dpo.py    preference corpus   -> (chosen, rejected) pairs
│   │   └── loader.py         memmap batch sampling (TokenDataset, MixedTokenDataset)
│   ├── model/
│   │   ├── transformer.py    the whole architecture, ~300 lines
│   │   └── moe.py            mixture of experts: router, sorted dispatch, upcycling — docs/14
│   ├── quant/            int8/int4/NF4 from scratch — see docs/10
│   ├── lora/             LoRA + QLoRA adapters from scratch — see docs/11
│   │   ├── qtensor.py        group scales, zero-points, 4-bit packing
│   │   ├── qlinear.py        QuantLinear: a drop-in for nn.Linear that stores bytes
│   │   ├── rtn.py            round-to-nearest baseline
│   │   ├── calib.py          forward hooks: Hessians and per-channel activation energy
│   │   ├── gptq.py           Hessian-guided error compensation
│   │   ├── awq.py            activation-aware scaling, folded into the preceding op
│   │   ├── qat.py            straight-through estimator + fine-tune loop
│   │   ├── kernels.py        fused dequantize-and-matmul, in Triton
│   │   └── convert.py        model surgery, save/load quantized checkpoints
│   ├── train/
│   │   ├── pretrain.py       next-token prediction (single- or blended-source)
│   │   ├── sft.py            instruction tuning
│   │   ├── dpo.py            preference tuning
│   │   ├── schedule.py       learning-rate schedules
│   │   ├── stopfile.py       the STOP contract: stop now / at step N / at a wall-clock time
│   │   │                     — shared by pretraining, SFT and QAT
│   │   └── runlog.py         reads train_log.jsonl back (sessions, series) — shared by
│   │                         scripts/sessions.py and the portal
│   ├── portal/           local web portal: start/stop a run, watch it, test it, read it
│   │   ├── runs.py           run state on disk; drives phase2.sh / stop.sh
│   │   ├── schedule.py       recurring start/stop windows + the clock loop
│   │   ├── gpu.py            nvidia-smi sampling, history, training-vs-idle summary
│   │   ├── cost.py           energy ledger + what each run cost in electricity
│   │   ├── explain.py        source browser + a local Ollama model that explains it
│   │   ├── evals.py          benchmark jobs; shells out to `python -m aksharallm.eval`
│   │   ├── synth.py          generation jobs; shells out to `python -m aksharallm.synth`
│   │   ├── learn.py          the Learn tab: lessons, gating, and running their checks
│   │   ├── server.py         stdlib http.server + a small JSON API
│   │   └── static/           the client: no build step, no framework, no dependencies
│   │       ├── index.html        the shell; server fills its <!--#include --> markers
│   │       ├── parts/<tab>.html  markup, one file per view
│   │       ├── js/<tab>.js       ES modules, one per view (+ core/state/router/charts)
│   │       └── css/<tab>.css     rules, one per view (+ base/chrome/controls/narrow)
│   ├── learn/            the learning path: lessons, gating, checks — see docs/15
│   │   ├── lessons.py        frontmatter, the prereq graph, and the anti-rot validation
│   │   ├── progress.py       learning/progress.json + the red-then-green completion rule
│   │   └── check.py          run one pytest node id, report what it said
│   ├── synth/            generating training data with a local teacher — see docs/13
│   │   ├── prompts.py        the seed grid: 480 / 1,296 structurally different prompts
│   │   ├── recipes.py        python / chat / preference: prompt, parser, export
│   │   ├── filters.py        validity checks + shingle-based near-duplicate detection
│   │   ├── verify.py         run the generated tests, then run them against a stub
│   │   ├── dataset.py        data/synth/<name>/: samples, rejects, provenance
│   │   └── run.py            the loop, its budgets, and the STOP file
│   ├── eval/             the benchmark harness — see docs/12
│   │   ├── sources.py        download benchmarks once into data/eval/, then work offline
│   │   ├── suites.py         what each benchmark asks, and how an answer is judged right
│   │   ├── scoring.py        batched log-likelihood, greedy decode-until, perplexity
│   │   ├── judge.py          twelve open prompts graded 1-5 by a local Ollama model
│   │   ├── runner.py         run suites against a checkpoint; one JSON per evaluation
│   │   └── report.py         every result ever, and the trend across training steps
│   └── infer/            talking to a checkpoint, and judging what comes back
│       ├── generate.py       KV-cache sampling loop (streaming + one-shot)
│       ├── checkpoints.py    what has been trained: step, loss, stage, tokens seen
│       ├── engine.py         one model kept warm; CPU while a run has the GPU
│       ├── tasks.py          the fixed probes and the graded Python tasks
│       ├── sandbox.py        runs the Python the model wrote, under limits
│       ├── history.py        every generation + the training state that produced it
│       ├── playground.py     the four above in the order both front ends use
│       └── cli.py            completion / chat / code, probes, tasks, comparisons
├── docs/lessons/         the course: one markdown file per lesson, with frontmatter
├── scripts/
│   ├── phase1.sh         Phase 1 end to end (data -> pretrain -> generate), ~30 min
│   ├── phase2.sh         Phase 2: pre-flight, build data, smoke test, background launch
│   ├── stop.sh           stop a background run cleanly: now, after N steps, or at a time
│   ├── portal.sh         the web portal (progress, graphs, start/stop); --lan to share
│   ├── schedule.sh       recurring start/stop windows ("22:00-06:30, mon-fri")
│   ├── gpu.sh            GPU utilisation/memory/temp/power, now and over time
│   ├── sessions.py       per-session summary of a run trained over many evenings
│   └── postrain.sh       Phase 3: SFT then DPO
├── tests/                correctness tests (KV cache, causality, RoPE, mixing, DPO)
└── docs/                 the guide above
```

---

## The plan: one blended base → two models

**Phase 1 — `configs/tiny.yaml`.** 13.8M params, TinyStories, ~25 minutes. The point is
not the model; it's proving every stage of the pipeline works before you spend a week of
GPU time. Always start here after changing anything.

**Phase 2 — `configs/small-code.yaml`.** ~300M params, ~10B tokens of **85% FineWeb-Edu +
15% Python**, roughly 6 days on a 3090. Blending code into pretraining means one run yields
a base that both chats (after SFT/DPO) and codes (after Python continued-pretraining) — and
code also improves general reasoning. See [docs/07-scaling.md](docs/07-scaling.md).
(`configs/small.yaml` is the pure-FineWeb-Edu fallback.)

```mermaid
flowchart LR
    B["blended base<br/>85% web + 15% Python"] --> C["general chat<br/>SFT + DPO"]
    B --> P["Python specialist<br/>continued-pretrain + code SFT"]
```

Both use identical code. Only the config differs.

Six days of compute needn't be six days of calendar. `scripts/phase2.sh` launches in the
background and records its pid; `scripts/stop.sh` stops it cleanly — now, after a set number
of steps, or at a time you name — and every stop saves at the exact current step, so
re-running resumes with no loss spike:

```bash
scripts/phase2.sh                       # launch (pid -> checkpoints/<run>/train.pid)
STOP_IN=3h scripts/phase2.sh            # ...for one evening, then save and exit
scripts/stop.sh small-code --status     # alive? at what step?
scripts/stop.sh small-code --after 500  # do 500 more steps, then save and exit
scripts/stop.sh small-code --in 20m     # stop twenty minutes from now
scripts/stop.sh small-code --by 06:30   # stop at half six (tomorrow, if it has passed)
scripts/stop.sh small-code              # stop now, gracefully
scripts/phase2.sh                       # resume where it left off
scripts/sessions.py small-code          # compare the sessions afterwards
```

A timed stop is a **deadline written into the run's STOP file**, which the trainer checks
every step — so it survives closing the terminal or restarting the portal, and stays true
if the run slows down. Nothing has to sit and watch the clock.

Each session gets its own `logs/<run>/train_<timestamp>.log` (never overwritten), and
`train_<run>.log` symlinks to the newest one.

### Making it smaller: quantization

A trained model is a few hundred million numbers stored in 16 bits each. Store them in 4
instead and the 300M model goes from **599 MB to 213 MB**. The trick is that weights come
in tightly clustered groups, so one shared scale per 64 weights lets each individual weight
be a 4-bit integer.

There are four ways to choose those integers, all written from scratch here — round to
nearest, [GPTQ](docs/10-quantization.md) (push each rounding error into the columns you
haven't done yet), AWQ (scale the channels that matter up before rounding), and
quantization-aware training (put the rounding inside the training loop). Measured on the
300M model, against a bf16 perplexity of 13.519:

| | size | perplexity |
|---|---|---|
| bf16 | 599 MB | 13.519 |
| int8 | 342 MB | 13.519 — **free** |
| int4, round-to-nearest | 213 MB | +0.253 |
| int4, GPTQ | 213 MB | **+0.163** |

There is also a hand-written Triton kernel that unpacks the 4-bit weights *inside* the
matmul, which makes decoding 1.6× faster than the naive quantized path. It still does not
beat plain bf16 — and the reason turned out to be the most interesting thing in the
chapter, so [it is written up honestly](docs/10-quantization.md#the-fused-kernel-and-an-honest-performance-story)
rather than quietly omitted.

A later addition, **NF4**: instead of spacing the 16 levels evenly, put them at the
quantiles of a normal distribution — which is what trained weights actually look like. It
is more accurate *and* smaller than int4 (no zero-point to store), and the levels are
derived from `erfinv` rather than pasted in from the paper.

### Changing it cheaply: LoRA and QLoRA

Fine-tuning a model normally costs far more than storing it, because Adam keeps two fp32
moments per trainable parameter. For the 300M model that is 4.8 GB — before activations,
and on a card that also has to hold the forward pass.

[LoRA](docs/11-lora.md) freezes the model and trains a small low-rank correction beside it
(`y = Wx + BAx`), so ~1% of the parameters train and the optimiser state shrinks with them.
**QLoRA** then holds the frozen base in 4-bit NF4 — safe precisely because it is frozen and
never receives a gradient. Measured on the 300M checkpoint:

| | to fine-tune | the artifact |
|---|---|---|
| full fine-tune | 4,791 MB | a whole new 1.2 GB checkpoint |
| LoRA r=8 | 1,253 MB | a 14 MB adapter |
| **QLoRA r=8** | **327 MB** | a 14 MB adapter |

And it costs almost nothing in quality: on the 13.8M model, identical data and schedule,
full fine-tuning reached val **1.2364**, LoRA **1.2425**, QLoRA **1.2433** — while training
2.5% of the parameters. (LoRA is *slower* in wall clock, though. It buys memory, not time,
and the chapter says so.)

Two consequences change how the project is shaped. A specialisation becomes a **file**, not
a model — one base plus a chat adapter and a Python adapter, swapped at inference, instead
of two 1.2 GB checkpoints. And DPO's frozen reference model becomes **free**: switch the
adapter off and the model you are already holding *is* the model you started from.

### What comes after that

In order, all written from scratch: **GRPO** ✅ (RL on a reward the code sandbox actually
verifies) → **quantization** ✅ → **LoRA/QLoRA** ✅ → a **real eval harness** ✅ (MMLU, ARC,
HellaSwag, PIQA, GSM8K, HumanEval and a model-judged suite — [docs/12](docs/12-eval.md)) →
**synthetic data** ✅ ([docs/13](docs/13-synthetic-data.md)) → **mixture of experts** ✅
([docs/14](docs/14-moe.md)) → **distillation** → **diffusion training** → export and
serving.

The mixture of experts is the first of those with a measured answer. Run at Phase 1 scale
against the dense baseline — same data, seed, batch, steps, and **the same FLOPs per token**,
because each expert is `d_ff/k` wide rather than a full copy — 8 experts at top-2 reached
val **1.4081** against the dense **1.4764**, a 4.6% improvement that *widened* throughout
training. It stores 35.0M parameters to compute with 7.1M of them: memory traded for quality
at fixed compute, which is the exact opposite of the trade [quantization](docs/10-quantization.md)
makes. The cost is MFU falling from ~57% to 52%, because a sort and eight small matmuls use
the card less well than one big one.

What makes it work is the part that has nothing to do with experts: a **load-balancing loss**
and a routing chart. Without them a few experts win early, take all the gradient, and the
rest never train — and the loss curve looks completely normal while it happens.

The harness came before the last four deliberately. Mixture of experts, synthetic data and
distillation are all changes to model *quality*, and validation loss either cannot see them
or reports them backwards — training on generated text is the easiest way to improve a loss
curve while making a model worse. You need the instrument before you run the experiment.

**Synthetic data**, built next, turned out to be mostly *filters*. A local teacher writing
exercises is four lines; the package around it exists because generated data is the easiest
way to make a model worse while its loss improves. So the Python recipe runs the tests the
teacher wrote — and then runs them again against a stubbed solution, because tests that pass
with the function's body removed prove nothing and look identical to real ones. Diversity
comes from a grid of 480 structurally different prompts rather than from a temperature, and
every dropped sample is counted by reason, because "30% survived" means three unrelated
problems depending on *which* filter took the rest. On this machine that loop paid for
itself immediately: reading the rejects showed a teacher writing correct solutions with
wrong expected values in the tests, one rule in the template fixed it, and the pass rate
went 25% → 58%.

Two of the others are worth explaining, because they are not the usual list.

**Diffusion training.** Everything above is autoregressive: predict the next token from the
ones before it. A masked diffusion language model throws that away — it corrupts a sequence
by masking tokens at random, learns to denoise it with *bidirectional* attention, and
generates by unmasking positions in any order until none are left. The whole training
objective is: mask each token with probability `t ~ U(0,1)`, run the model with the causal
mask off, and take cross-entropy on the masked positions weighted by `1/t`.

It will be trained at Phase 1 scale (13.8M params, TinyStories) against the autoregressive
baseline that is already there — same data, same size, same budget. Not because it will win:
masked diffusion needs several times the compute of autoregression to reach the same
quality. It is there because it can do two things autoregression structurally cannot —
**infilling** (given a prefix *and* a suffix, write the middle) and **parallel generation**
(watch a whole sequence resolve from all-masked to text in about 32 steps).

**A learning path.** This repo is meant to be learned from, and right now the docs are a
reading order with nothing to do. The plan is a set of lessons that each pair a doc section
with an exercise and a check that has to go from red to green — usually *break this on
purpose and watch what fails.* The best material is already in the repo's own history: the
silent data-truncation bug, and the masking bug that trained perfectly and generated
nonsense. Lessons will reference files rather than line numbers, and each one's check is a
real test in the suite — so if a lesson goes stale, the test run says so.

### Watching it in a browser

```bash
scripts/portal.sh --bg --lan    # background, reachable from your phone; prints the address
scripts/portal.sh --status      # running? which pid, which address
scripts/portal.sh --restart     # stop and start again (never touches a training run)
```

A local page with the progress against the budget and an ETA, live loss / throughput /
gradient-norm / LR curves (drag sideways across one to zoom into that stretch of steps —
the y-axis refits to the window; double-click to go back), the per-session table, the tail
of the log, and buttons for start (with a budget for the session), stop now, and a **Stop
at…** dialog where you pick steps or time from presets — 1 / 5 / 10 / 30 min, 1 / 2 / 4 h —
or a logarithmic dial, and read what it means before committing: which step it lands on,
what time it finishes, how many checkpoints happen on the way. Five tabs, each with its own
address so a view can be bookmarked: **Dashboard** (the run), **Playground** (talk to the
checkpoint it is producing), **Code** (have the source explained to you), **Quantize**
(make it 4-bit and measure what that cost) and **Docs** (read this guide in the browser,
diagrams and all).

It is a **view over the same files**, and it presses the same buttons: it starts runs with
`scripts/phase2.sh` and stops them with `scripts/stop.sh`, so a run launched from a terminal
appears in the portal and vice versa — including while it is still in pre-flight — and
closing the portal never stops training. Standard library only: the server is `http.server`,
the charts are hand-written SVG.

### Testing the model while it is still training

A loss curve tells you the number is going down. It does not tell you whether the model can
finish a sentence. The portal's **Playground** tab — and the same thing from a terminal —
lets you ask it:

```bash
python -m aksharallm.infer.cli                    # what has been trained so far
python -m aksharallm.infer.cli small-code         # talk to it
python -m aksharallm.infer.cli small-code --probes   # the fixed prompt suite
python -m aksharallm.infer.cli small-code --tasks    # Python tasks, actually executed
python -m aksharallm.infer.cli --compare fluency     # that prompt, across every step
```

Three things make it useful rather than a toy:

**It will not cost you the run.** A Phase-2 run holds ~21 GB of a 24 GB card. The model
would fit in the gap, but a CUDA context is half a gigabyte before any weights land and the
failure mode is a six-day run dying overnight. So while a run is training, inference loads
on the **CPU**, automatically, and the tab says why. The card is used the moment it is free.

**It knows what the checkpoint can do.** A base model has never seen a chat turn, so chat is
disabled on one — with that sentence, rather than letting you conclude the model is broken.

**The record outlives the checkpoint.** `ckpt_last.pt` is overwritten every 500 steps.
Rather than archive 1.2 GB of weights forty times, every generation appends a line to
`logs/playground.jsonl` carrying the step, validation loss and tokens seen of the model that
produced it — so the same prompt at step 7,000 and step 30,000 sit side by side for about a
kilobyte. That comparison is what `--compare` and the tab's history panel show.

The Python tasks are graded by **executing** the generated function against asserts, in a
subprocess with CPU-time and memory limits, isolated from this project's code, in a
throwaway directory. It is not a container — see `aksharallm/infer/sandbox.py`, which is
honest about what the containment is and is not — and `infer.run_tests: false` turns it off.

### Reading the code with a local model

The portal's third tab is a source browser for this repo with an explainer attached: pick a
file, highlight a line or a block, and a model running on your own machine through
[Ollama](https://ollama.com) tells you what it does, why it is written that way, and what
would bite you if you changed it. Follow-up questions keep the thread; the answer streams in
as it is generated.

```bash
ollama serve && ollama pull gemma4:12b   # once
scripts/portal.sh --open                 # then click "Code"
```

The model gets the selection *and* the whole enclosing file *and* a primer on the project,
which is what lets it answer "why" rather than just paraphrasing syntax. Which model, where
Ollama lives and how much of the machine it may use are all in `configs/portal.yaml`.

One thing to know: the explainer shares your GPU with training. A 12B model is ~8 GB of VRAM
against a Phase-2 run's ~21 GB of a 24 GB card, so the tab warns you when a run is live, and
`num_gpu: 0` keeps the explainer on the CPU where it cannot touch it. Details and the other
sharp edges are in `docs/07-scaling.md`.

### Watching the GPU

```bash
scripts/gpu.sh                  # now + a 1-hour summary, split into training vs idle
scripts/gpu.sh watch            # one line a second
```

The portal samples `nvidia-smi` every five seconds into `logs/gpu.jsonl` and charts
utilisation, memory, temperature and power, banding the periods when a run was training.
Since each sample records whether a trainer was alive, the summary splits into *while
training* vs *idle* — which is how you notice that the GPU sat at 40% all night, or that
something else was resident on it.

### What it cost

```bash
python -m aksharallm.portal.cost            # per run, today, all time, per 1M tokens
python -m aksharallm.portal.cost backfill   # fold telemetry taken before the ledger existed
```

The same power readings, integrated: energy per run, folded as it arrives into permanent
ten-minute buckets in `logs/energy.jsonl` (the telemetry file itself is a rolling buffer, so
a total read back from it would quietly shrink as old samples were trimmed). Give it a rate
in `configs/portal.yaml` — `cost.per_kwh`, plus `host_watts`/`psu_efficiency` if you want the
number to match a plug meter rather than the card alone — and every run gains a price, a cost
per million tokens, and a **coverage** figure saying how much of the run was actually
recorded. With no rate set it shows kilowatt-hours and says so. Portal: the **Cost** panel.

### Training on a schedule

```bash
scripts/schedule.sh window small-code 22:00 06:30 --days mon-fri   # train overnight
scripts/schedule.sh window small-code 13:00 17:30 --days sat,sun   # and weekend afternoons
scripts/schedule.sh                                                # what's next
```

Or the portal's **Schedule** panel — pick a run, two times, click the days. Both edit the
same `schedule.json`, and the clock loop runs inside the portal (or as
`scripts/schedule.sh daemon`, or from cron via `scripts/schedule.sh check`). Firing a rule
calls the same `phase2.sh` / `stop.sh` the buttons do, so a scheduled start is
indistinguishable from one you typed. Starting when it is already training is a no-op, and
a fire missed while the machine slept stays missed rather than going off nine hours late.

Everything either side needs is on disk: the trainer writes `train.pid` into its own
checkpoint dir (so the 50-step smoke test, which shares its command line, can never be
mistaken for the run), `phase2.sh` publishes `launch.pid` + `launch.meta` while it
pre-flights, and a stop during pre-flight aborts the launch from either side. `--lan` serves
it to the rest of your network; there is no login, so keep that to networks you trust.

```mermaid
flowchart LR
    UI["portal page<br/>charts · playground · code"] <-->|JSON + SSE| SRV["aksharallm.portal"]
    SRV -->|reads| F[("train_log.jsonl<br/>train.pid · STOP · logs/")]
    SRV -->|runs| P["phase2.sh / stop.sh"] --> T["trainer"] -->|appends| F
    T -->|saves| C[("checkpoints/&lt;run&gt;/*.pt")]
    SRV -->|generates from| C
    SRV -->|appends| H[("logs/playground.jsonl<br/>output + the step that made it")]
```

---

## Hardware notes

Measured on an RTX 3090 (24 GB):

| | Phase 1 (13.8M) | Phase 2 (~300M) |
|---|---|---|
| throughput | 460k tok/s | ~19k tok/s (est.) |
| MFU | 50% | ~45% (est.) |
| VRAM | 3.6 GB | ~20 GB |
| wall clock | 25 min | ~6 days |

`torch.compile` is worth 1.7× throughput *and* lowers memory — leave it on.

**Disk is usually the real constraint**, not VRAM. Tokenized data is 2 bytes/token:
10B tokens = 20 GB, and the raw download is larger. Check `df -h` before Phase 2.

---

## Running the tests

```bash
uv pip install pytest && python -m pytest tests/ -q
```

The KV-cache test is the one that matters: it asserts that token-by-token generation
reproduces a full-sequence forward pass exactly. A bug there gives you a model that
trains perfectly and generates garbage.

---

## Credits and further reading

This follows the path laid down by Karpathy's
[nanoGPT](https://github.com/karpathy/nanoGPT), with a modern Llama-style architecture.
Key papers: [Attention Is All You Need](https://arxiv.org/abs/1706.03762),
[RoFormer (RoPE)](https://arxiv.org/abs/2104.09864),
[GLU Variants (SwiGLU)](https://arxiv.org/abs/2002.05202),
[Chinchilla](https://arxiv.org/abs/2203.15556),
[DPO](https://arxiv.org/abs/2305.18290).
