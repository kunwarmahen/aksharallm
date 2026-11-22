# 19. Long context — teaching a model to read further than it was trained to

> Our 300M model was trained on windows of **1,024 tokens** — about a page and a half. This
> chapter is about what happens when you show it eight pages instead, why it fails in a very
> specific way, the three one-line fixes for it, and how to tell which of them actually
> worked. Everything here is measured on our own checkpoints, and none of it required a
> single minute of retraining.
>
> The short version: **one config line takes it to 4,096 tokens**, where it still finds a fact
> hidden in the text **92.5% of the time** against a 25% chance line — and no weight changes.

---

## Start here: what "context" actually is

A language model does not have a memory. Every time it produces a token it re-reads the
entire conversation from the beginning. The **context window** is simply the largest number
of tokens it is able to re-read at once.

So "extending the context" is not about giving the model more storage. It is about fixing
something much narrower and stranger, and to see what, you need one idea about how the model
knows *where* a word is.

### Position is an angle

The model has no built-in notion of order — to the attention mechanism a sentence is a bag
of words. Order is added by **RoPE**, which encodes a token's position by *rotating* its
query and key vectors. Token 5 gets rotated a little, token 500 a lot.

Each head's channels are split into pairs, and each pair spins at its own speed:

```mermaid
flowchart TB
    subgraph L["one head's channel pairs, fastest to slowest"]
        direction LR
        P1["pair 1<br/>a full turn<br/>every ~6 tokens"]
        P2["pair 8<br/>a full turn<br/>every ~200"]
        P3["pair 16<br/>a full turn<br/>every ~3,000"]
        P4["pair 32<br/>a fraction of a turn<br/>across the whole window"]
    end
    P1 -.->|"encodes<br/>'the previous word'"| U1["local detail"]
    P4 -.->|"encodes<br/>'near the start / near the end'"| U2["global position"]
```

Fast pairs tell the model about neighbours. Slow pairs tell it roughly where in the document
it is. Training on 1,024-token windows shows it every angle the *fast* pairs can make —
they go all the way round, many times — but only a **thin slice** of what the slow pairs can
make, because they never complete even one rotation in 1,024 steps.

### So the failure is not forgetting, it is nonsense

Ask that model about token 4,000 and the slow pairs are four times further round the circle
than anything in training. The model is not straining to remember. It is being handed
coordinates in a system it has never seen, and every attention score it computes from them is
meaningless.

This is what that looks like on our own 13.8M model, trained on 512 tokens and then simply
asked for 2,048 — nothing changed but the length:

| positions | loss | perplexity |
|---|---|---|
| 0–511 | 1.26 | 3.5 |
| 512–1023 | 3.78 | 43.8 |
| 1024–1535 | 5.03 | 153.4 |
| 1536–2047 | 4.98 | 144.8 |

Perplexity 3.5 → 153. That is not degradation, it is a **cliff**, and it is the single most
important picture in this chapter:

```mermaid
flowchart LR
    A["trained window<br/>loss ~1.3"] -->|"one token past the edge"| B["loss 3.8"]
    B --> C["loss 5.0<br/>ppl 153"]
    style A fill:#2d6a4f,color:#fff
    style B fill:#bc6c25,color:#fff
    style C fill:#9d0208,color:#fff
```

---

## The three fixes, and they are all one line

Here is the good news, and it is genuinely surprising the first time: **RoPE has no
parameters**. The rotation angles are computed from the position and a fixed frequency
ladder. So changing how a model addresses position changes *no weights at all* — it is
arithmetic, applied at load time.

All three methods change exactly one thing: the frequency ladder `inv_freq`. They differ
only in *which* pairs they slow down.

```mermaid
flowchart TB
    subgraph N["none — the cliff"]
        direction LR
        NA["fast pairs<br/>unchanged"] --- NB["slow pairs<br/>unchanged → off the map ✗"]
    end
    subgraph L["linear — squash everything"]
        direction LR
        LA["fast pairs<br/>4x slower ✗<br/>local detail lost"] --- LB["slow pairs<br/>4x slower ✓"]
    end
    subgraph K["ntk — tilt the ladder"]
        direction LR
        KA["fast pairs<br/>~untouched ✓"] --- KB["slow pairs<br/>4x slower ✓"]
    end
    subgraph Y["yarn — per-pair, with a ramp"]
        direction LR
        YA["fast pairs<br/>untouched ✓"] --- YC["middle<br/>smooth ramp"] --- YB["slow pairs<br/>4x slower ✓"]
    end
```

**`linear`** — *position interpolation*. Pretend token 4,000 is token 1,000; divide every
position by the factor. Every angle is now one the model has seen, which is why it works at
all. But it squashes the fast pairs too, and those were carrying "the previous word" — so it
buys range by throwing away local resolution.

**`ntk`** — instead of dividing positions, **raise `theta`**, the number the frequency ladder
is built from. Because the ladder is geometric, nudging its base tilts the whole thing: the
slowest pair ends up interpolated by almost exactly the factor while the fastest is barely
touched. One number, and it keeps what `linear` destroys.

**`yarn`** — does per-pair and explicitly what NTK does as a side effect. For each pair it
asks *how many full rotations does this complete inside the original window?* Many (≥32) means
the model has seen the whole circle, so leave it alone. Fewer than one means it has only ever
seen a sliver, so interpolate fully. In between, ramp smoothly. It also nudges the attention
temperature, because a longer context spreads attention thinner.

**`dynamic`** — NTK with the factor computed from the length actually being processed. Below
the original window the factor is 1 and the model is bit-for-bit unscaled, so short prompts
pay nothing. Reach for this when one checkpoint has to serve both.

---

## What they are actually worth (measured, our own models)

`python -m aksharallm.longctx sweep <ckpt>` runs every method over the same windows — same
weights, same data, one variable.

### The 300M, trained on 1,024

**At 2x** (measured over 2,048 tokens, 32 windows):

| method | overall loss | inside the window | past it | cliff |
|---|---|---|---|---|
| **none** | 3.264 | 2.356 | 4.173 | **at 1,280** |
| linear | 3.004 | **3.035** ← ruined | 2.974 | none |
| **ntk** | **2.323** | 2.365 | 2.282 | none |
| **yarn** | **2.310** | 2.376 | 2.245 | none |

The in-window baseline is **2.356**. NTK lands at 2.365 inside the window — **nine
thousandths of a nat**, which is nothing — and 2.282 *past* it. Doubling this model's context
cost effectively nothing: no fine-tune, no weights touched, one config line.

(Its *overall* loss of 2.323 sitting **below** the in-window baseline is not an error, and it
is worth understanding. Loss normally improves with position, because later tokens have more
context to condition on. A working extension keeps that slope going — which is what "it is
actually using the extra room" looks like as a number.)

Now read the `linear` row. It took the in-window loss from 2.356 to **3.035** to buy the same
range. That is the local-resolution cost, and it is not subtle.

**At 4x** (measured over 4,096 tokens, 24 windows) the methods separate hard:

| method | overall loss | inside the window | past it | cliff |
|---|---|---|---|---|
| none | 4.740 | 2.578 | 5.461 | at 1,536 |
| linear | 4.275 | **4.379** | 4.241 | none |
| ntk | 2.895 | 2.623 | 2.986 | **at 3,584** |
| **yarn** | **2.464** | 2.650 | **2.402** | none |

**NTK grows a cliff of its own at 3,584.** Its tilt buys roughly 3.5x and then runs out.
YaRN's per-pair ramp does not, and holds 2.402 all the way to the end.

### The 13.8M, trained on 512 — the same shape, one tenth the size

| | 2x (measured at 1,024) | 4x (measured at 2,048) |
|---|---|---|
| none | 2.558 | 3.744 |
| linear | 1.511 | 2.185 |
| ntk | 1.526 | 2.366 |
| **yarn** | **1.372** | **1.512** |

Same story at a scale that runs in a couple of minutes: everything works at 2x, and at 4x NTK
*loses* to linear while YaRN stays close to its baseline.

**Rule of thumb, now confirmed at both scales: NTK to 2x, YaRN beyond it.**

> **A methodology warning, from getting this wrong in this very chapter.** The first version
> of the 300M table was run on the CPU with **two** windows and reported an in-window baseline
> of 0.990 where 32 windows say **2.356**. Two windows of 2,048 tokens is four thousand tokens
> of one or two documents; land on something repetitive and the whole curve moves by more than
> a nat. Every *conclusion* survived the correction and not one *number* did. Use at least 16
> windows before quoting anything.

---

## The other approach: don't look so far back

Scaling makes distant positions legible. A **sliding window** sidesteps the problem entirely
— just refuse to attend more than *w* tokens back. Attention stops being O(T²) and the KV
cache stops growing.

```mermaid
flowchart LR
    subgraph W["sliding window, w = 512"]
        S["sinks<br/>tokens 0-3"]:::sink
        G["... 1,000 tokens the model cannot see ..."]:::gone
        V["the last 512 tokens"]:::vis
        Q["the token being predicted"]:::q
    end
    S --> Q
    V --> Q
    G -.->|blocked| Q
    classDef sink fill:#2d6a4f,color:#fff
    classDef gone fill:#6c757d,color:#fff
    classDef vis fill:#1d3557,color:#fff
    classDef q fill:#bc6c25,color:#fff
```

On our 13.8M at 2,048 tokens, a 512-token window gives **loss 1.253** — flat all the way out,
no cliff, and *better than YaRN's 1.512*. With no scaling and no fine-tune.

Which sounds like it wins, and it does not, for a reason worth being precise about.

> **A sliding window buys perplexity by giving up the thing you wanted.** The model is now
> structurally incapable of using anything more than 512 tokens back. Its next-token
> prediction is excellent, because next-token prediction is mostly local. Its ability to
> answer a question about page one is exactly zero — by construction, not by accident.

This is the single most important trap in long context, and it is why the next section
exists.

### Attention sinks, and a finding of our own

The published wisdom (StreamingLLM) is that a sliding window needs **attention sinks** — keep
the first few tokens permanently visible. The reason is elegant: attention is a softmax, so
the weights must sum to one whether or not anything in the window deserves them. Models learn
to dump the remainder on the first few tokens, which every position can see. Slide the window
past them and that overflow lands on real tokens instead.

We measured it, and **on our model sinks made it worse**:

| configuration | loss at 2,048 |
|---|---|
| window 512, **no** sinks | **1.253** |
| window 512, 4 sinks | 1.778 |
| window 512, 4 sinks, **+ YaRN 4x** | 1.470 |

That third row is the explanation. Our sinks live at absolute positions 0–3, so a query at
position 1,500 reaching back to them is asking about a **relative distance of 1,500** — far
outside the trained window, and precisely the illegible angle the sliding window had just
finished eliminating. Sinks reintroduce the disease. Making those angles legible with YaRN
recovers most of the loss, which confirms the diagnosis.

The published implementations avoid this by assigning RoPE positions **by slot in the cache**
rather than by absolute position, so the maximum relative distance is bounded by the window.
**We do not do that**, because our KV cache rotates keys once when they are written and never
re-rotates them. Fixing it properly means position-at-read-time, which is a real change to
the cache. It is written down here rather than papered over: on this codebase today, use a
sliding window **without** sinks, or with sinks *and* a scaling method.

---

## Measuring it honestly: perplexity is not enough

Two questions hide inside "does the long context work?", and conflating them is the most
common mistake in this area.

```mermaid
flowchart TB
    Q["does the long context work?"] --> A["1. is it still fluent out there?<br/><b>curve.py</b> — loss by position"]
    Q --> B["2. can it still <i>retrieve</i> from out there?<br/><b>haystack.py</b> — needle in a haystack"]
    A --> AR["fixed by RoPE scaling.<br/>A sliding window scores<br/>well here while being blind."]
    B --> BR["not fixed by anything here.<br/>Needs a model that learned<br/>to use long range —<br/>our 300M has, our 13.8M has not."]
    style AR fill:#1d3557,color:#fff
    style BR fill:#9d0208,color:#fff
```

**Loss by position** (`curve.py`) is the cheap one and it runs on any checkpoint, from the
run's own `val.bin`, with no download and no fine-tune. Its shape is the whole result: flat
is healthy, a cliff is a naive extension, a step-then-flat is a working scaling method. It is
the first thing to run before believing any long-context claim.

**Needle in a haystack** (`haystack.py`) is the honest one. Hide a sentence — *"The secret
code for Bengaluru is 7431."* — at a chosen depth in a lot of ordinary text, then ask for it
back, and sweep length × depth into the grid everyone recognises.

Two choices make it work on a model this small. Nothing is **generated**: the true code and
some distractors are each scored by log-probability and the trial is correct when the true one
wins — the same machinery as our ARC and PIQA suites, exactly reproducible, and it works on a
model far too small to answer out loud. And the filler is the model's **own validation data**,
so a failure means "could not retrieve", not "was confused by our noise".

### The 300M, extended to 4x with YaRN — and it works

6 trials per cell, four-way choice, chance 25%:

| needle depth | 512 | 1,024 | 2,048 | 4,096 |
|---|---|---|---|---|
| 0% (very front) | 100% | 100% | 83% | **33%** |
| 25% | 100% | 100% | 83% | 83% |
| 50% | 100% | 83% | 100% | 83% |
| 75% | 100% | 100% | 100% | 100% |
| 100% (very end) | 100% | 100% | 100% | 100% |

**Overall 92.5% ± 2.4% against a 25% chance line**, at lengths up to **four times the window
the weights were ever trained on**, with no fine-tune and no weights changed. This is the
result that justifies the whole chapter: the perplexity curve said the positions had become
legible, and this says the model can actually *use* them.

The grid also has a shape, and it is the published one. Read the last two rows: a needle near
the **end** is found every single time at every length. Read the first row: a needle at the
very **front** of a 4,096-token context drops to 33%, barely above chance. Information nearest
the question is easiest, information furthest from it is hardest, and the gap opens up exactly
as the context gets long. Nothing here was tuned to produce that.

### The 13.8M, for contrast — at chance

The same test on the 13.8M scores **16.7% ± 10.8% against 25%** — i.e. nothing, at every
length and depth, and the CLI says "NOT distinguishable from chance" rather than printing a
number that reads like partial credit.

That is not a broken test, it is the answer, and the contrast with the 300M is the lesson.
Both models have legible positions after scaling. Only one of them ever learned to retrieve
from far away. **Scaling makes the positions legible; using them is a capability, and
capabilities come from training** — which is why both numbers are published rather than only
the good one.

### A task that is all retrieval, so a small model can be asked (built and run 2026-11-21)

The needle cannot compare small architectures: at 13.8M every model is at chance, so it says
nothing about whether a design *can* retrieve. And the position curve is fluency, not
retrieval: the hybrid of [doc 4](04-model.md) § "Linear attention" showed no loss cliff past
its 512-token window, which leaves open whether its DeltaNet memory gives back anything it
was told. `longctx/recall.py` is the version that can tell at any size, because the task is
nothing but looking things up:

```
<|endoftext|>  K17 V3  K40 V9  ...  K2 V51     F F F ... F      ?   K40 V9  K2 V51 ...
               \_____ pairs: a key, its value _____/  \_ gap _/      \_ queries _/
```

Keys are distinct and values random, so a query's value can only come from finding its key
earlier and reading what followed. The vocabulary is 148 words. Training documents are short
(4–48 pairs, gaps mostly under 32 tokens), and scoring sweeps two things: **how many pairs**
it can hold (8–48) and **how far back** it can reach, with gaps out to 1,536 tokens — three
times past the 512 positions training ever showed.

```mermaid
flowchart LR
    D["recall-dense<br/>attention in all 6"] --> Q{"retrieves past<br/>512 tokens?"}
    H["recall-hybrid<br/>DeltaNet in 4, attention in 2"] --> Q
    N["recall-deltanet<br/>DeltaNet in all 6<br/>(no positions at all)"] --> Q
```

Three configs one model key apart (`full_attn_every` absent / 3 / 7, where 7 > 6 layers means
no block is attention), same data, same budget. The expectation going in: dense attention
retrieves perfectly inside its window and falls off past it (the RoPE angles it never saw);
an all-DeltaNet model has no positions, so distance should not matter to it, but its memory
is a fixed 64 × 64 matrix per head, so the number of pairs should; the hybrid is both. The
first two held and the third did not (measured below). The scorer is held to an oracle that looks keys up (100%) and a copier (0%), which pins the
off-by-one that matters: the value at position `p` is read from the logits at `p − 1`.

Retrieval is learned suddenly, when an induction circuit forms, and often late, and getting
it to form at all took four tries on the CPU (a 2-layer d128 model, so the GPU runs are not
spent finding out). Every pair is queried, so ~18% of tokens sit behind a lookup; the loss
falls by about 0.74 nats when it forms, from the no-lookup floor of ~3.71 to ~2.69 (an
estimate; the GPU runs with every lookup right end at 2.907, see "Measured" below).

| tried on the CPU | what happened |
|---|---|
| the full range only (4–48 pairs, gaps to 200), 4-layer d192, 18M tokens | at chance, loss on the no-lookup floor |
| easy documents only (2–8 pairs, no gap), 2-layer, 1,500 steps | **formed**: loss 3.88 → 2.96, 52% at 4 pairs, 34% at 8 |
| easy mixed in from step 0, even 50% of tokens | nothing formed by 4,000 steps |
| **easy first, then the full range from those weights** | recall on the full task: 25% at 8 pairs / gap 32 (guessing among the document's values: 12.5%), 8–10% at gaps of 128–384 with 16 pairs |

So each arm is a **curriculum of two runs**: `recall-<arm>-easy` (2,000 steps on `easy.bin`),
then `recall-<arm>` (6,000 steps from its weights via `train.init`, with 20% easy documents
kept in by tokens, because the CPU model forgot the easy case without them). 8,000 steps in
all, the same for every arm.

```bash
python -m aksharallm.longctx recall-data                    # data/recall (+ easy.bin), ~3 min on the CPU
scripts/experiment.sh recall-hybrid-easy                    # stage 1: where the lookup forms
scripts/experiment.sh recall-hybrid                         # stage 2: from its weights; same for -dense, -deltanet
python -m aksharallm.longctx recall recall-hybrid --device cuda
python -m aksharallm.longctx recall-compare logs/longctx/recall-recall-dense-step5999.json \
    logs/longctx/recall-recall-hybrid-step5999.json         # paired, query by query, per cell
```

#### Measured (2026-11-21/22): position-free memory reaches three times the window

All six runs on the 3090 (`logs/queue/night-1122.sh`), 128 documents × 8 queries = **1,024
queries per cell**, the same seeded documents for every model, so every comparison below is
paired query by query (`longctx recall-compare`, also the Context tab's "pair against"
picker). Scores are on the CPU in fp32 for all arms. Chance is 1.6% (any of 64 values), or
1/pairs for a model that knows only that the answer is one of this document's values.

| recall, % of 1,024 queries | dense (6 attention) | hybrid (4 DeltaNet + 2 attention) | all DeltaNet |
|---|---|---|---|
| **after stage 1** (easy documents only, `ckpt_last`, step 1,999) | | | |
| 16 pairs, no gap | 63.2 | **100.0** | 99.9 |
| 8 / 16 / 32 / 48 pairs, gap 32 | 2.0 / 1.9 / 2.5 / 2.7 | 99.7 / 96.9 / 91.7 / 79.2 | **99.7 / 99.5 / 96.6 / 88.7** |
| 16 pairs, gap 128 / 384 (inside 512) | 0.2 / 0.8 | 83.4 / 65.0 | **99.1 / 98.7** |
| 16 pairs, gap 768 / 1,536 (past 512) | 0.3 / 0.0 | 28.3 / 1.7 | **99.0 / 98.6** |
| **after stage 2** (full range, step 5,999) | | | |
| every by-pairs cell (8–48 pairs, gap 32) | 100 | 100 | 100 |
| 16 pairs, gap 0 / 128 | 100 / 100 | 100 / 100 | 100 / 100 |
| 16 pairs, gap 384 (length 434, inside) | **97.9** | 100 | 100 |
| **16 pairs, gap 768 (length 818, past 512)** | **35.6** | **100** | **100** |
| **16 pairs, gap 1,536 (length 1,586, 3× the window)** | **7.0** | **100** | **100** |

Paired, stage 2, dense against either other arm: gap 384 22–0 (p 5e-7), gap 768 **659–0**,
gap 1,536 **952–0** (p < 1e-280). Hybrid against all-DeltaNet: 0–0 in every cell.

What it says:

- **Dense attention retrieves perfectly and only where it trained.** 100% to length 178, a
  first slip at 434, and past 512 it falls to 36% then 7%: the RoPE angles it never saw. The
  same cliff [doc 4](04-model.md)'s loss-by-position showed, now as retrieval.
- **The hybrid retrieves at three times its window**, with two RoPE attention blocks still in
  it. The position-free DeltaNet blocks carry the lookup; whatever the attention blocks do
  past 512 does not break it.
- **The capacity limit I expected did not appear.** 48 pairs is 100% for every arm after
  stage 2. Six layers × six heads of 64 × 64 memory is more room than 48 pairs need. Finding
  the limit needs more pairs or a smaller head; the sweep stops at 48.
- **Stage 1 is where the arms differ, and it is the most interesting row.** On easy documents
  (no filler, ever), the all-DeltaNet model learned a rule that works at *any* gap: 98.6% at
  1,536 tokens of filler it never saw. Dense attention learned one that works only with no
  gap (63% at gap 0, 2% with 32 filler tokens). A plausible reading, not checked inside the
  model: attention found a rule that leans on where things sit, which filler breaks, while
  the delta rule can only store and read "the value under this key", which filler cannot
  touch. Paired: 1,010–0 at gap 1,536. The hybrid is in between
  (100% at gap 0, 28% at 768).
- **Validation loss cannot tell the arms apart.** All three end at **2.907** on `val.bin`
  (dense 2.9074, hybrid 2.9072, DeltaNet 2.9073), because validation is read in 512-token
  windows of documents shaped like training ones, where dense attention is perfect too. With every lookup right, 2.907 is this data's floor; the ~2.69 estimated above
  for "perfect lookup" was wrong about the rest of the document's entropy. Only the gap sweep
  separates a model that will fail at 818 tokens from one that will not.
- **Cost**, same 8,000 steps: stage 1 + 2 = 18 min dense, 25 min hybrid, 31 min all-DeltaNet
  (plain PyTorch chunked DeltaNet, no kernel).

A trap paid for while measuring: the easy stage's `ckpt_best.pt` is chosen on the full-range
`val.bin`, which means nothing for a model that has only seen easy documents (val 6–9 nats).
The queue scored those `ckpt_best` files (steps 500 and 1,000), which caught DeltaNet
*before* its lookup formed (19.7% at 8 pairs, against 99.7% at step 1,999). The table uses
`ckpt_last`, the weights stage 2 starts from.

---

## Doing it

```bash
# 1. where does this model actually stop working? (saved to logs/longctx/ like the others,
#    since 2026-11-20; it used to only print, so the Context tab never saw a curve)
python -m aksharallm.longctx curve small-code --len 2048 --device cpu

# 2. which fix is best for it? (same windows, one variable)
python -m aksharallm.longctx sweep small-code --len 2048 --factor 2 --device cpu

# 3. bake the winner into a checkpoint — weights untouched
python -m aksharallm.longctx extend small-code --method ntk --factor 2 \
    --out checkpoints/small-code/ckpt_ntk2x.pt

# 4. and be honest about what it can do with the room
python -m aksharallm.longctx needle small-code --lengths 512 1024 2048 --device cpu
```

Or press the buttons in the portal's **Context** tab, which drives exactly these functions —
including **Write the extended checkpoint**, which names the file it is about to create
before it creates one, and adds it to the picker when it lands. The original is never
touched: an extension is a second file, so it stays a hypothesis you can delete.

In a config:

```yaml
model:
  max_seq_len: 4096
  rope_scaling: {type: yarn, factor: 4.0, original_max_seq_len: 1024}
  attn_window: 1024        # optional; sliding window
  attn_sinks: 0            # see the finding above before setting this
```

### Three things that will bite

1. **`original_max_seq_len` is the whole bookkeeping.** Once `max_seq_len` is 4,096 nothing
   else in the config remembers the weights were trained on 1,024, and every method needs
   that number. `extend` records it, which is what makes an extended checkpoint
   self-describing — it reloads correctly in the Playground, the harness and the server with
   no flags. Extending a second time compounds against the *original*, not against 4,096.
2. **`dynamic` is stateful.** After one 8k sequence, a later 1k sequence is served by the 8k
   factor. Every published implementation behaves this way. Worse, if it grows *during*
   generation the keys already in the KV cache were rotated with the old factor and disagree
   with the new queries — call `model.pin_rope(max_len)` before a generation loop.
3. **`--device cuda` is refused while a run is training.** These are long forward passes;
   4,096 tokens of logits is half a gigabyte before the loss is computed, and the training
   run has about 3 GB spare. Use `--device cpu`, or `--force-gpu` if you know there is room.

---

## What we did not do

**No fine-tune.** Every number here is a training-free extension, which is the interesting
regime and the one that fits an evening. A short fine-tune at the extended length is the
standard next step; on this model the place it would show is the top-left of the needle grid,
where a needle at the very front of a 4,096-token context is still only found a third of the
time.

**No block-skipping for the sliding window in the Triton kernel.** Our own
[FlashAttention kernel](04-model.md) understands `window` and `sinks` — passed as two
integers rather than as a mask, which matters because the equivalent bool tensor is 64 MB at
T=8192. But it still *walks* the skipped key blocks and masks them, rather than never loading
them, so the window costs no memory and saves no time. The causal diagonal already does the
real thing; extending it to a window is bounded work and is the obvious next optimisation.

---

## The code, in reading order

| # | file | what to look for |
|---|---|---|
| 1 | [`aksharallm/model/rope.py`](../aksharallm/model/rope.py) | **start here.** The module docstring is the derivation; then `plan()`, which is the whole chapter in forty lines — one branch per method, each changing only `inv_freq` |
| 2 | `_yarn_ramp` | rotations-per-original-window as the thing being thresholded. This is what makes YaRN per-channel rather than global |
| 3 | [`aksharallm/config.py`](../aksharallm/config.py) → `RopeScaling`, `attn_window`, `attn_sinks` | the three config surfaces, and the `__post_init__` coercion that lets ten `ModelConfig(**ckpt[...])` call sites keep working |
| 4 | [`transformer.py`](../aksharallm/model/transformer.py) → `sliding_window_mask` | the three-clause rule, and the docstring on why the sink clause exists |
| 5 | `Transformer._rope` · `pin_rope` | the only stateful thing in the model, and the two traps that come with it |
| 6 | [`longctx/extend.py`](../aksharallm/longctx/extend.py) | `plan_extension` — a pure function, so the portal can show what *would* change without writing 1.2 GB to find out |
| 7 | [`longctx/curve.py`](../aksharallm/longctx/curve.py) | `reduction="none"` is the point of the file; then `cliff()`, and why it anchors to the in-window mean rather than position 0 |
| 8 | [`longctx/haystack.py`](../aksharallm/longctx/haystack.py) | `build_context` (the needle goes in at a *token* boundary) and `_score_candidates` (the off-by-one that would score every candidate on the wrong tokens) |
| 8b | [`longctx/recall.py`](../aksharallm/longctx/recall.py) | `document` (an answer is hidden behind its key), `accuracy` (the value at `p` is read from the logits at `p − 1`), and `load_for_recall`, which sizes the RoPE table for the longest document without touching a weight. Read from the wrong position in [lesson 28](lessons/28-recall.md) |
| 9 | [`longctx/__main__.py`](../aksharallm/longctx/__main__.py) | `sweep` is the command the chapter exists for; `_check_device` is the courtesy that stops a measurement killing a run |
| 10 | [`aksharallm/portal/longctx.py`](../aksharallm/portal/longctx.py) | the Context tab's server side. `plan` is free (metadata only, no weights) and `start` is a detached job, one at a time; note that no path ever arrives from the browser — `default_out_name` decides where an extension lands |

What pins it: `tests/test_longctx.py`. The four worth reading are
`test_none_is_bit_for_bit_the_old_cache` (every existing checkpoint depends on the default
not moving a single angle), `test_a_window_wider_than_the_sequence_changes_nothing` and
`test_changing_a_token_outside_the_window_cannot_reach_the_last_position` (the window does
nothing when it should and something when it should), and `test_extending_twice_compounds`. `tests/test_recall.py` holds the recall scorer to an
oracle that looks every key up (100%) and a copier (0%).

---

Next: [4. The model →](04-model.md) for the attention kernel this builds on.
