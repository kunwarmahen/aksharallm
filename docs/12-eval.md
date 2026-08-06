# 12. Evaluation: is the model actually any good?

Up to this point every number in this project has been a **loss**. Loss is the right thing
to watch while training — it is smooth, it is cheap, and it moves every hundred steps. It
is also the wrong thing to trust, for a reason that is easy to say and easy to forget:

> Cross-entropy measures how well the model predicts the *next token of a corpus*.
> Nothing you want the model to do is that.

The gap does not matter much while a model is getting better at everything at once. It
matters enormously the moment you start making trade-offs — and the next three things this
project builds are all trade-offs:

| what we are about to build | how loss misleads |
|---|---|
| **Mixture of experts** | more total parameters at the same active cost. Loss barely moves; capacity is the claim, and capacity shows up on knowledge benchmarks. |
| **Synthetic data** | training on generated text improves loss *and* degrades the model. This is the classic failure, and val loss reports it as progress. |
| **Distillation** | a smaller model matched to a bigger one's outputs. "Is 100M-distilled better than 300M-int4 at the same file size?" is not a question loss can be asked. |
| **Quantization** | int4 costs +0.17 perplexity. Is that 0.17 worth anything? Perplexity cannot say. MMLU can. |

That is why the harness was built **before** those four, not after. It is the instrument;
they are the experiments.

```mermaid
flowchart LR
  A[training run] -->|val loss| B(is training working?)
  A --> C[checkpoint]
  C --> D[eval harness]
  D -->|MMLU, ARC, HellaSwag, PIQA| E(does it know things?)
  D -->|GSM8K| F(can it reason?)
  D -->|HumanEval| G(can it write working code?)
  D -->|LLM-judge| H(is it any good to talk to?)
  D -->|perplexity| B
```

---

## Five kinds of measurement

They differ in what they can see and in what they cost. Roughly: the cheaper it is, the
less it tells you.

```mermaid
flowchart TD
  subgraph scored["scored — nothing is generated"]
    P[perplexity<br/>held-out loss] --- MC[multiple choice<br/>MMLU · ARC · HellaSwag · PIQA]
  end
  subgraph generated["generated — then checked by something objective"]
    G[GSM8K<br/>regex on the final number] --- H[HumanEval<br/>run the hidden tests]
  end
  subgraph judged["judged — no right answer exists"]
    J[12 open prompts<br/>graded 1-5 by a local model]
  end
  scored -->|seconds to minutes| generated
  generated -->|minutes to an hour| judged
```

### 1. Perplexity

`exp(mean cross-entropy)` on held-out text. Kept because it is continuous with the
training curve, and because when it disagrees with everything else that is itself
information. **Not comparable across tokenizers** — a tokenizer that splits text into more,
easier pieces gets a better perplexity for free.

### 2. Multiple choice, scored by log-likelihood

This is the one worth understanding, because it is not what most people assume.

You do **not** ask the model "is the answer A, B, C or D?" and read what it says. A 300M
base model cannot follow that instruction; it has never been taught to. Instead you put
each candidate answer after the question, one at a time, and measure how surprised the
model is by it:

```
context:   "Question: What do plants need to make food?\nAnswer:"

           + " sunlight"     ->  sum log P(tokens)  =  -8.4     <- least surprised
           + " darkness"     ->                        -14.1
           + " metal"        ->                        -16.7
           + " silence"      ->                        -15.9
```

Argmax of those four numbers is the model's answer. Nothing is sampled, so the score is
exactly reproducible, and a model far too small to *write* an answer can still show a
signal.

Three accuracies come out of this and they disagree on purpose:

| number | what it is | when to read it |
|---|---|---|
| `acc` | argmax of the raw summed log-probability | biased toward **short** answers — every extra token can only make a sequence less likely |
| `acc_norm` | the same, divided by the answer's length **in characters** | the headline. HellaSwag's wrong endings are adversarially long, so raw `acc` is close to meaningless there |
| `acc_greedy` | how often the right answer is also what the model *would have generated* | much stricter; near zero for a small model even when `acc_norm` is respectable |

The reported `score` is `acc_norm`, which is what published figures quote.

### 3. Generative, checked by a regex — GSM8K

Grade-school maths word problems, five-shot with chain-of-thought examples drawn from the
**train** split (never the test split — showing the model a test question as a worked
example is contamination, and the two splits are one argument apart). Each shot ends with
GSM8K's own `#### 42` marker, so the model is taught the output format at the same time as
the reasoning style, and extraction is unambiguous rather than "the last number anywhere".

Decoding is **greedy**. A benchmark that samples gives a different score every run, and the
entire purpose here is comparing a checkpoint against its own earlier self.

### 4. Executed — HumanEval

164 Python functions with hidden tests. The model writes the body, the code is cut down by
the same `extract_code` the Playground uses, and then it is **actually run** in
`aksharallm/infer/sandbox.py` — a separate isolated interpreter with CPU-time and memory
limits and no access to this project. (That module's docstring is honest that this is a
limit, not a container. Read it before pointing this at a model you did not train.)

The dataset's test block ends with a `check(entry_point)` call that the runner has to add.
Forget it and every assert is *defined* and never *run*, and every model scores 100%.

### 5. Judged by another model

Everything above has an objective answer. That covers a lot and misses the thing you would
actually notice about a model: whether its answers are any good. Twelve fixed open-ended
prompts — explanation, instruction-following, summarisation, reasoning, code, honesty,
writing — are answered by our model and graded 1–5 by a local Ollama model against a
**written rubric** that ships with the prompt.

An LLM-judge is a **consistent** reader, not a correct one. It is biased toward long
answers, toward its own style, toward confident prose — in the same directions every time,
which is exactly what makes a comparison between two of *our* checkpoints meaningful even
though the absolute number is not. Three things hold it steady:

- **temperature 0** and a fixed prompt, so re-grading the same answers gives the same
  grades;
- **a rubric per prompt**, written when the prompt was, so the standard cannot drift;
- **the judge never learns which checkpoint produced the answer**, or sees any other
  answer to compare against.

A judge that fails to answer is recorded as **ungraded**, not as a 1. Scoring it 1 would
punish the model being tested for the judge's mistake.

---

## What a score means at our scale

This is the most important table in the chapter, and the reason every suite carries an
`expect` line that the CLI prints beside its result.

| suite | chance | what a 300M base should do | if it does not |
|---|---|---|---|
| **PIQA** | 50% | the first to move — it needs plausibility, not knowledge | something is wrong |
| **ARC-Easy** | 25% | clearly above chance by the end of Phase 2 | undertrained, or the prompt format broke |
| **HellaSwag** | 25% | a few points over chance, noisily | normal below ~1B |
| **MMLU** | 25% | **sits on 25%** | that *is* the result. Movement above ~27% is the first sign of world knowledge |
| **ARC-Challenge** | 25% | chance | expected until well past 1B |
| **GSM8K** | 0% | **0%** | correct. Multi-step arithmetic needs a model an order of magnitude bigger |
| **HumanEval** | 0% | **0/164** | correct, for a long time |

Two of those deserve saying out loud:

**25% on MMLU is not a failure, it is a coin flip.** Four-way multiple choice pays 25% for
guessing. A reader who does not know that concludes the model is broken; this is the single
commonest way to misread the table, which is why the chance line is drawn on every chart
and printed in every row.

**0% on GSM8K is still worth measuring.** Not for the number — for watching the *failure*
change. "No numbers at all" → "numbers, no method" → "right method, wrong arithmetic" is
real progress that the score never shows. The same argument as the code tasks in
`aksharallm/infer/tasks.py`, where the error type (`SyntaxError` → `NameError` →
`AssertionError`) is the progress meter.

### The error bar is part of the number

At n=500 the binomial standard error is about 2%. A two-point move between checkpoints is
noise. The harness reports `stderr` on every multiple-choice suite, the portal only colours
a score when it clears chance by more than its own error bar, and neither will let a coin
flip be dressed up as a result.

---

## First real measurement

Phase 2 at **step 18,000 of 40,000** (45%), on the CPU because a run owned the card:

| suite | tiny (13.8M, TinyStories) | small-code (299M, blend) | chance |
|---|---|---|---|
| perplexity | 4.337 | – | – |
| ARC-Easy | 20.0% (n=40) | **46.7%** ± 6.4 (n=60) | 25% |
| PIQA | 55.0% (n=40) | **65.0%** ± 6.2 (n=60) | 50% |

Both 300M numbers are clear of chance by more than their error bars, at 45% of the training
budget. The TinyStories model sits at chance on ARC-Easy, which is exactly right — it was
trained on children's stories and has never seen a science question.

The perplexity of 4.337 on `tiny` is worth noting for a different reason: **it matches the
4.36 recorded from the training run itself**. The harness and the training curve agree,
which is the cheapest possible check that the scoring path is not quietly wrong.

---

## How it is put together

```mermaid
flowchart TD
  CLI["python -m aksharallm.eval"] --> R[runner.Harness]
  PORTAL["portal Eval tab"] -->|subprocess| CLI
  R -->|loads the model| E[infer.Engine]
  E -->|device policy, adapters,<br/>quantized checkpoints| M[(checkpoint)]
  R --> S[suites.py<br/>what each benchmark asks]
  S --> SRC[sources.py<br/>data/eval/*.jsonl]
  R --> SC[scoring.py<br/>log-likelihood · greedy decode]
  R --> J[judge.py] -->|shares the client| OL[portal.explain.Ollama]
  R --> SB[infer.sandbox]
  R -->|one JSON per run| OUT[(logs/eval/)]
  OUT --> REP[report.py<br/>trend across steps]
  REP --> PORTAL
```

Four decisions in there are load-bearing.

**The harness loads models through `infer.Engine`, not itself.** That inherits, for free
and without a second copy: the device policy (the CPU whenever a run is training, with the
reason stated — which matters more here than in the Playground because an evaluation is
*minutes* of sustained work), LoRA adapters, and quantized checkpoints. So "what did int4
cost on MMLU?" and "did the fine-tune help?" are both one flag.

**Data is downloaded once into `data/eval/`, never streamed.** Streaming breaks
reproducibility twice over: the Hub's copy can change, and "the first 500 rows of a stream"
is not reliably the same 500 rows twice. Each fetch writes a `.meta.json` recording repo,
config, split, row count and date — the answer to "which MMLU is this?", worth being able
to give three model generations later. Only the columns a scorer reads are kept.

**Results are a folder of JSON files, not a database.** Same decision as the quantization
panel and the Playground's history: a result is a few hundred kilobytes, `grep` works on
it, a job started in a terminal shows up in the browser, and there is no schema to migrate
the day a suite is added.

**The portal shells out to the CLI.** The Eval tab never evaluates anything itself — it
runs the command you would have typed. So the browser and the terminal cannot disagree
about what was measured or where it ran.

---

## Two things in the scoring that are easy to get wrong

**The continuation is tokenized separately from the context.** `encode(ctx + cont)` is not
`encode(ctx) + encode(cont)`, because BPE merges across the join. Counting continuation
tokens backwards off the joined string is off by one wherever a merge crosses the boundary
— which is most of the time. Every choice in a question is affected equally, so it is
*invisible in the accuracy* and wrong in the numbers.

**Sequences are right-padded, with no attention mask.** Under a causal mask a position can
only attend to positions before it, so padding after the real tokens cannot influence any
real position. Left-padding would need a mask, and getting that subtly wrong gives you a
harness that scores every model a little too low and never says so. There is a test that
scores a mixed-length batch against the same pairs one at a time.

Batches are sized by a **token budget**, not a row count: the logits tensor is
`(batch, tokens, 32768)` and that is where all the memory goes. 2,048 tokens per forward is
~270 MB in float32 whether that is 32 short MMLU prompts or 2 long HellaSwag ones. Lower
`--batch-tokens` if scoring runs out of memory.

`Transformer.forward` gained one keyword for this: **`full_logits=True`** returns every
position's logits with no loss. Passing `targets` just to get the full tensor would make
the model compute a mean cross-entropy the harness throws away, at the cost of a second
float32 copy of a quarter-gigabyte tensor — on a device that is usually the CPU. There is a
test that `full_logits` returns exactly what the training path returns, because if those
ever diverge every benchmark number is computed on different logits from the ones the model
trains with, and nothing would say so.

---

## Running it

```bash
# what can be measured, and what to expect at this scale — start here
python -m aksharallm.eval suites

# download the benchmarks. Once, ~19 MB, then it works offline forever
python -m aksharallm.eval fetch --all
python -m aksharallm.eval fetch --list

# the default set on a run's best checkpoint
python -m aksharallm.eval small-code

# everything, whole splits, no caps — hours on the CPU
python -m aksharallm.eval small-code --suite all --limit 0

# one suite, more items, a name that goes in the filename
python -m aksharallm.eval small-code --suite mmlu --limit 2000 --label mid-phase2

# a LoRA adapter against its own base — "did the fine-tune actually help?"
python -m aksharallm.eval small-code --adapter small-code/sft_best.lora.pt --suite default

# what int4 cost on a benchmark rather than on perplexity
python -m aksharallm.eval small-code/ckpt_best-gptq-nf4-g64.pt --suite mmlu,arc-easy

# open-ended, graded by a local model
python -m aksharallm.eval small-code --suite judge --judge-model gemma4:31b

# every evaluation so far, and one suite across every step
python -m aksharallm.eval report
python -m aksharallm.eval report --suite arc-easy
```

Or the portal's **Eval** tab (`scripts/portal.sh`, then `#evals`), which is a view over
exactly those commands: pick a checkpoint, tick suites, press Evaluate. It leads with the
trend chart rather than the Run button, for the same reason the Finetune tab leads with the
memory budget — one benchmark score in isolation is close to meaningless.

## Configuration

The judge reads `judge:` in `configs/portal.yaml`, which is the Code tab's `explain:`
config with a different section name and different defaults. They share one client and one
set of error messages, and must not share a model: the explainer wants something small
enough to run beside a training run, the judge wants the best model on the machine and does
not mind waiting.

```yaml
judge:
  model: qwen3.5:27b     # the biggest thing you have; quality matters more than speed here
  temperature: 0.0       # a judge that samples is not a judge
  think: false           # same trap as the explainer — see docs/07
  num_predict: 400
  timeout_s: 600
```

`AKSHARALLM_JUDGE_MODEL` and `AKSHARALLM_JUDGE_NUM_GPU` override it for one session;
`AKSHARALLM_OLLAMA_HOST` is shared with the explainer.

## Gotchas worth keeping

1. **Never change a prompt format without renaming the suite.** A base model's MMLU score
   moves several points on whether the options are `A.` or `(A)`. The formats here are the
   ones the standard harnesses use; changing one makes every number measured before the
   change a different benchmark, silently.
2. **The few-shot examples must not come from the split being scored.** GSM8K's shots come
   from `train`, MMLU's from `dev`. Both are one argument away from the test split.
3. **HellaSwag's test split ships unlabelled.** Scoring it would count every item wrong and
   report a plausible-looking low number. The builder skips unlabelled rows; validation is
   what everyone scores.
4. **`ybisk/piqa` stopped loading on `datasets` >= 5** — it is a dataset *script*, and
   those were removed. Every source can list fallback repositories for exactly this reason.
5. **A stopped evaluation writes nothing, deliberately.** Half a benchmark looks like a
   whole one in a table. To spend less time, lower `--limit`; do not stop early.
6. **`logs/eval/current.json` is job state, not a result.** It sits beside the results
   because the portal writes it there; the reader excludes it by name. Without that the
   running job appears in the table as an empty evaluation.

## Is the benchmark trustworthy? (contamination, and the number hiding two)

Every score above assumes two things nobody had checked. This section checks them.

### 1. Did the test leak into the training data?

If a question and its answer already sit somewhere in the ten billion tokens the model was
trained on, a right answer means nothing — the model may simply be remembering. In the
chapter's own words from earlier: **a benchmark number nobody has checked for leakage is a
rumour.**

The published method is **n-gram overlap**. An item is *dirty* if any run of 13 consecutive
tokens from it also appears anywhere in the training corpus. Thirteen is the standard choice
and it is a Goldilocks number: at 8, ordinary English shares n-grams by accident and
everything looks contaminated; at 25, a reformatted whitespace run or a one-word paraphrase
breaks the streak and nothing does.

```mermaid
flowchart LR
    E["the benchmark<br/>~10^5 tokens"] --> H["every 13-gram,<br/>hashed and sorted"]
    T["the training corpus<br/>~10^10 tokens"] -->|"chunks of 32M"| R["every 13-gram,<br/>hashed"]
    R --> S{"seen<br/>before?"}
    H --> S
    S -->|yes| D["this item is dirty"]
    S -->|no| C["this item is clean"]
    style D fill:#9d0208,color:#fff
    style C fill:#2d6a4f,color:#fff
```

The asymmetry is what makes it tractable: build the tiny side into a sorted array, then
stream the huge side past it once. Both sides are hashed with the same rolling polynomial
computed over whole chunks in numpy — a Python loop over ten billion positions is not a
program that finishes — and membership is a vectorised binary search.

**The one distinction the report exists to draw** is between the question and the answer:

| part | what it means | how worried to be |
|---|---|---|
| `question` | the question text appears in training | usually **not at all**. Benchmark questions are public text; a web crawl is expected to contain them |
| `answered` | the question **with its correct answer attached** appears | **this is the one.** It is what a contaminated corpus memorises, and what makes a score meaningless |

Getting that distinction right took a second attempt, and the mistake is instructive.
"Question plus answer" contains every n-gram of the question, so the first version lit up
`answered` for any corpus that merely held the public question — the two columns were the
same column, and the more alarming one was the useless one. The fix is to keep only the
n-grams that **reach into the answer**: the last 12 tokens of the question and everything
after it. A test plants a question-only corpus and requires `answered` to stay at zero.

**The output that matters is not the percentage, it is the clean score.** Re-read a
benchmark result, drop the contaminated items, and see whether the number moves:

```bash
python -m aksharallm.eval contaminate --suite mc --verify     --against logs/eval/20260806-small-code-eval.json
```

A suite that is 8% dirty and scores the same either way is fine. One that is 3% dirty and
gains four points on exactly those three percent is telling you something. This needs the
run's per-item verdicts, so it does not work on a result recorded with `--no-items` — and
it says so rather than inventing a number.

Three things the implementation is careful about, all of which would otherwise make the
report **quietly optimistic**, which is the wrong direction for a contamination check to be
wrong in:

- **Chunks overlap by n-1 tokens.** An n-gram straddling a chunk boundary is still an n-gram
  in the corpus; forget the overlap and you silently lose one window per chunk.
- **Items shorter than 13 tokens are counted as *unchecked*, not as clean.** A suite of
  one-line questions must not report 0% dirty when the truth is 0% checked.
- **Hash collisions are acknowledged and can be removed.** Two different 13-grams can share
  a 64-bit hash; with ~10⁶ probe hashes and ~10¹⁰ lookups that is about one spurious hit per
  two thousand full scans. Small enough to ignore in a summary, too large to leave unstated
  in a finding somebody will quote — so `--verify` re-reads the actual tokens behind every
  hit and drops the ones that do not hold up.

### 2. One validation number is hiding two

`configs/small-code.yaml` trains on **85% prose and 15% Python** and reports a single val
loss. That average is 85% prose by construction, so the model's Python ability is nearly
invisible in it, and either half can move without the total saying so.

Split it and the two halves are not remotely alike:

| source | tokens | weight | loss | perplexity |
|---|---|---|---|---|
| fineweb-edu (prose) | 8,500,000 | 0.85 | **2.7696** | 15.95 |
| codeparrot-python | 1,500,000 | 0.15 | **1.2558** | **3.51** |
| *weight-blended* | | | *2.5425* | |

**Python is more than twice as predictable as prose** — perplexity 3.5 against 16.0. That is
not the model being good at code so much as code being repetitive, but either way it is a
fact the single number 2.54 completely conceals, and it is the number to watch when the
Python specialist of Phase 4 starts training.

The blended figure is also the check. The run's own best val loss at this step was **2.5552**
and blending the parts gives **2.5425** — agreement to 0.013, which is what says the split
was taken in the right place.

**Where the boundaries come from, and why they are verified.** `prepare_blend` writes
`val.bin` by concatenating one part per source, each capped at `val_tokens × weight`:

```mermaid
flowchart LR
    A["fineweb-edu<br/>0 .. 8,499,999"] --> C["val.bin<br/>10,000,000 tokens"]
    B["codeparrot-python<br/>8,500,000 .. 9,999,999"] --> C
```

Nothing recorded those offsets, so for the existing blend they are **derived** from the
weights — and a derived number nobody checks is how a report becomes confidently wrong. So
each span's content is read back and asked whether it matches the source it claims to be: a
span called `codeparrot-python` had better look like code, and one called `fineweb-edu` had
better not. A span that fails prints **MISMATCH** and the split is declared meaningless
rather than used, because a prose/Python split with the split in the wrong place produces
two plausible numbers that are both averages of the same mixture. A source name the check
has no opinion about prints *unverified*, never *ok*.

Going forward there is nothing to derive: `prepare_blend` now writes `val.manifest.json`
beside the bin, and the manifest always wins.

```bash
python -m aksharallm.eval domains small-code --device cpu
python -m aksharallm.eval contaminate --suite mc --verify
```

Both are also buttons in the portal's **Eval** tab, under "Is the benchmark trustworthy?",
and they share that tab's one-job-at-a-time lock — a contamination scan streams ten billion
tokens and a per-domain split runs the model, and neither wants to be doing that while an
evaluation is trying to produce a number.

## The code, in reading order

| # | file | what to look for |
|---|---|---|
| 1 | [`eval/suites.py`](../aksharallm/eval/suites.py) | **start here.** `Suite` and the `SUITES` table — every benchmark's chance line, its `expect` sentence and its builder. Then `build_mmlu` / `build_hellaswag` / `build_gsm8k` / `build_humaneval` to see exactly what the model is shown |
| 2 | [`eval/sources.py`](../aksharallm/eval/sources.py) | `Source` and `fetch` — downloaded once into `data/eval/`, with a `.meta.json` recording which copy of the dataset this is. Note the fallback repositories |
| 3 | [`eval/scoring.py`](../aksharallm/eval/scoring.py) | `_encode_pair` (the continuation tokenized *with* the context — the subtle one), `loglikelihood`, `score_mc` (`acc` vs `acc_norm` vs `acc_greedy`), `generate_until`, `perplexity`. `_batches` is the token-budget batcher |
| 4 | [`eval/runner.py`](../aksharallm/eval/runner.py) | `Harness` — one suite at a time, loading the model through `infer.Engine` rather than itself, which is where the device policy, adapters and quantized checkpoints come from free |
| 5 | [`eval/judge.py`](../aksharallm/eval/judge.py) | `build_messages` / `parse_grade` / `run` — temperature 0, a rubric per prompt, and `Grade` with ungraded distinct from 1 |
| 6 | [`eval/report.py`](../aksharallm/eval/report.py) | `Results` → `summary_table` / `compare_table` — a folder of JSON, no database, and the trend across steps |
| 7 | [`eval/__main__.py`](../aksharallm/eval/__main__.py) | `cmd_suites` / `cmd_fetch` / `cmd_run` / `cmd_report` — thin, by design |
| 8 | [`aksharallm/infer/sandbox.py`](../aksharallm/infer/sandbox.py) | HumanEval's scorer is the same sandbox the Playground and GRPO use. Note who adds the `check(entry_point)` call |
| 9 | [`eval/contamination.py`](../aksharallm/eval/contamination.py) | `ngram_hashes` (the Horner trick that makes ten billion tokens tractable), `build_probe` (and the trim that makes `answered` mean something), `scan_bin`'s chunk overlap, `clean_score` |
| 10 | [`eval/domains.py`](../aksharallm/eval/domains.py) | `derive_spans` then `verify_spans` — the derivation and the check on it. `blended()` is the end-to-end test: it has to agree with the run's own val loss |
| 11 | [`aksharallm/portal/evals.py`](../aksharallm/portal/evals.py) | `EvalJobs` — the tab runs the CLI in a subprocess and reads the same result files; `start_audit` adds the two checks above to the same job lock |

What pins it: `tests/test_contamination.py` leads with a **positive control** — a known item
planted in a fake corpus that the scanner must find — because a checker that reports 0%
because it is broken looks exactly like a clean corpus, and it is the more comfortable of the
two answers. Then `tests/test_eval.py` — the mixed-length batch scored against the same pairs
one at a time (right-padding under a causal mask), `full_logits` returning exactly what the
training path returns, and the BPE-boundary test written *for*
[lesson 11](lessons/11-eval.md) after the original one turned out to be unable to fail.

## Where this sits

```
prepare-data -> pretrain -> SFT -> DPO/GRPO -> quantize -> serve
                    |         |        |          |
                    +---------+--------+----------+---> EVALUATE, at every stage
```

Evaluation is not a phase. It is the thing you do after each of the others, which is why
the harness takes any checkpoint, any adapter, quantized or not, and keeps every result
forever in the same folder.

Next in the build order are the synthetic-data pipeline and mixture of experts. Both are
changes to model *quality* rather than to model *cost*, and neither could have been judged
before this chapter existed — which is why they were deliberately scheduled after it.
