# aksharallm

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
python -m aksharallm.infer.cli checkpoints/tiny/ckpt_best.pt \
    --prompt "Once upon a time"
```

That's the entire loop, end to end. Everything after this is the same code with bigger
numbers.

---

## The documentation

Read these in order. They assume no prior knowledge of machine learning.

| # | Doc | What you'll learn |
|---|-----|-------------------|
| 0 | [What we're building](docs/00-overview.md) | What an LLM is, the four training stages, why each exists |
| 1 | [Data](docs/01-data.md) | Where text comes from, why quality beats quantity, the on-disk format |
| 2 | [Tokenizer](docs/02-tokenizer.md) | Why models read "tokens" not letters, how BPE works |
| 3 | [The model](docs/03-model.md) | Attention, RoPE, SwiGLU, residual streams — every line of the transformer |
| 4 | [Pretraining](docs/04-pretraining.md) | The training loop, mixed precision, LR schedules, reading the logs |
| 5 | [Post-training](docs/05-posttraining.md) | SFT and DPO — turning a text completer into an assistant |
| 6 | [Inference](docs/06-inference.md) | KV caches, sampling, temperature and top-p |
| 7 | [Scaling up](docs/07-scaling.md) | Phase 2: a 300M model on 10B tokens, and how to size your own |
| 8 | [Troubleshooting](docs/08-troubleshooting.md) | Loss spikes, NaNs, OOM, slow training |

---

## Repo layout

```
aksharallm/
├── configs/              YAML run configs — the only thing that changes between runs
│   ├── tiny.yaml         Phase 1: 13.8M params, TinyStories
│   ├── small.yaml        Phase 2 (pure): 300M params, FineWeb-Edu only
│   └── small-code.yaml   Phase 2 (blended): 300M, 85% FineWeb-Edu + 15% Python
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
│   │   └── transformer.py    the whole architecture, ~300 lines
│   ├── train/
│   │   ├── pretrain.py       next-token prediction (single- or blended-source)
│   │   ├── sft.py            instruction tuning
│   │   ├── dpo.py            preference tuning
│   │   ├── schedule.py       learning-rate schedules
│   │   └── runlog.py         reads train_log.jsonl back (sessions, series) — shared by
│   │                         scripts/sessions.py and the portal
│   ├── portal/           local web portal: start/stop a run, watch the curves
│   │   ├── runs.py           run state on disk; drives phase2.sh / stop.sh
│   │   ├── schedule.py       recurring start/stop windows + the clock loop
│   │   ├── server.py         stdlib http.server + a small JSON API
│   │   └── static/           one page, hand-written SVG charts, no dependencies
│   ├── eval/evaluate.py  perplexity, HellaSwag, sample generations
│   └── infer/
│       ├── generate.py   KV-cache sampling loop
│       └── cli.py        interactive chat / completion
├── scripts/
│   ├── phase1.sh         Phase 1 end to end (data -> pretrain -> generate), ~30 min
│   ├── phase2.sh         Phase 2: pre-flight, build data, smoke test, background launch
│   ├── stop.sh           stop a background run cleanly, now or after N more steps
│   ├── portal.sh         the web portal (progress, graphs, start/stop); --lan to share
│   ├── schedule.sh       recurring start/stop windows ("22:00-06:30, mon-fri")
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
background and records its pid; `scripts/stop.sh` stops it cleanly — now, or after a set
number of steps — and every stop saves at the exact current step, so re-running resumes with
no loss spike:

```bash
scripts/phase2.sh                       # launch (pid -> checkpoints/<run>/train.pid)
scripts/stop.sh small-code --status     # alive? at what step?
scripts/stop.sh small-code --after 500  # do 500 more steps, then save and exit
scripts/stop.sh small-code              # stop now, gracefully
scripts/phase2.sh                       # resume where it left off
scripts/sessions.py small-code          # compare the sessions afterwards
```

Each session gets its own `logs/<run>/train_<timestamp>.log` (never overwritten), and
`train_<run>.log` symlinks to the newest one.

### Watching it in a browser

```bash
scripts/portal.sh --bg --lan    # background, reachable from your phone; prints the address
scripts/portal.sh --status      # running? which pid, which address
scripts/portal.sh --restart     # stop and start again (never touches a training run)
```

A local page with the progress against the budget and an ETA, live loss / throughput /
gradient-norm / LR curves, the per-session table, the tail of the log, and buttons for
start, stop, "stop after N more steps" and "stop at step N".

It is a **view over the same files**, and it presses the same buttons: it starts runs with
`scripts/phase2.sh` and stops them with `scripts/stop.sh`, so a run launched from a terminal
appears in the portal and vice versa — including while it is still in pre-flight — and
closing the portal never stops training. Standard library only: the server is `http.server`,
the charts are hand-written SVG.

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
    UI["portal page<br/>charts + buttons"] <-->|JSON| SRV["aksharallm.portal"]
    SRV -->|reads| F[("train_log.jsonl<br/>train.pid · STOP · logs/")]
    SRV -->|runs| P["phase2.sh / stop.sh"] --> T["trainer"] -->|appends| F
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
