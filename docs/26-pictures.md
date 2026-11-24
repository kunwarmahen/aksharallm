# 26. Making pictures: a picture as integers, and drawing one from a sentence

Everything this repo has generated so far is a sequence of integers: text tokens, codec
codes for speech, the 65 tokens of a game frame. To make it *draw*, a picture first has to
become such a sequence, and that sequence has to turn back into the picture. That is the
**image tokenizer**, the first piece (G0) of the picture line planned in `PLAN.md`. The
second (G1, [below](#from-a-sentence-to-a-picture-g1)) is a transformer that reads "two red
circles" and writes those integers. The rest of the chapter goes the way real image models went:

| step | what it is | best result on the shapes | section |
|---|---|---|---|
| G0 | image tokenizer (VQ) | every picture reads back right | [The tokenizer](#the-tokenizer) |
| G1 | a transformer writes the codes | 89% (after more pictures fixed counting) | [G1](#from-a-sentence-to-a-picture-g1) |
| G2 | flow matching in pixels | 99.2% | [G2](#g2-drawing-by-flow-matching) |
| G3 / G3b | the same in a latent: a VQ latent, then a continuous one | 87.6% / **98.9% at a third of the cost** | [G3](#g3-flow-matching-in-the-latent) |
| G4a | the caption read by our own frozen language model | 99.3%, 99.0% on a phrasing never seen | [G4a](#g4a-reading-a-sentence-2026-11-23) |
| G4b | our own contrastive judge, and scenes that also say size and place | judge agrees with the reader at kappa ~0.9; scenes 91% seen (the autoencoder's ceiling), 85% held out at guidance 3 | [G4b](#g4b-a-judge-that-learned-2026-11-23) |
| G4c | real COCO photos from people's captions | own caption in the top 5 of 1,000: 18% (g1) / 60% (g3), real photos 34%, chance 0.5% | [G4c](#g4c-real-photos-built-2026-11-23) |

Lessons 29–36 break one piece each: the judge, guidance, the velocity, the latent's scale, a
held-out phrasing, the contrastive targets, the side the padding goes on, and the sign of the
adversary.

```mermaid
flowchart LR
    G0["G0 image tokenizer<br/>(built)"] --> G1["G1 a caption to<br/>the codes (built)"]
    G0 --> G3["G3 flow in the<br/>latent (built)"]
    G2["G2 flow matching<br/>in pixels (built)"] --> G3
    G1 --> G4["G4 text to image<br/>(G4a-c built)"]
    G3 --> G4
    G4 --> G5["G5 short videos<br/>of moving shapes"]
```

It is the audio codec of [chapter 21](21-audio.md) turned sideways: strided convolutions
shrink the input, residual vector quantization turns each cell of what is left into a few
integers, and a mirrored decoder paints the input again. The quantizer
(`audio/vq.py`'s `ResidualVQ`) is **imported unchanged**. Its EMA codebooks, first-batch
seeding, dead-code restart and quantizer dropout were written for speech and carry over to
pictures without an edit.

## The pictures, and why they are drawn rather than downloaded

The corpus is the one the vision chapter ([chapter 22](22-vision.md)) already draws: 64×64
pictures of one to four shapes (circle, square or triangle) in one of five colours, on a
near-black background, each made from a description we chose ("three red circles").
8,000 pictures, 98 MB, rendered in seconds. Purple triangles are never drawn; a separate
set of 64 of them is the **held-out combination**.

Drawing them ourselves is what makes the tokenizer **measurable** rather than merely
demonstrable. A reconstruction of a photograph can only be looked at. A reconstruction of
"three red circles" can be checked: does it still show three, still red, still circles?

## Reading a picture back, with no model

`pictures/reader.py` answers that question by rule:

```mermaid
flowchart LR
    I["picture"] --> F["foreground: pixels far<br/>from the background grey"]
    F --> C["colour: nearest of the<br/>5 palette colours, majority"]
    F --> L["connected blobs,<br/>20+ pixels each"]
    L --> N["count"]
    L --> S["shape = how much of its bounding<br/>box a blob fills: square ~1.0,<br/>circle ~0.79, triangle ~0.5"]
```

A learned reader (the vision model) could do this too, but it has its own error rate, and a
tokenizer's mistakes would then be mixed with the reader's. The rule-based reader is
**exact on every original** (all 8,000 and all 64 held-out pictures; a test checks every
count × colour × shape on fresh renders), so whatever it gets wrong on a reconstruction is
the reconstruction's fault. That row, *the originals*, is printed first in every report as
the positive control.

It is also strict in the useful direction: a light blur (σ = 1 pixel) still reads as a
square, a heavy one (σ = 3) reads as a circle, which is exactly what an undertrained
decoder paints (below). A test pins both sides.

## The tokenizer

```mermaid
flowchart LR
    X["64x64x3"] --> E["conv, then 3 x<br/>(res block, 4x4 conv stride 2)"]
    E --> Z["8x8 x 64"]
    Z --> Q["ResidualVQ: 4 codebooks x 512<br/>(the audio quantizer, unchanged)"]
    Q --> I["4 x 8 x 8 = 256 integers"]
    Q --> D["3 x (nearest upsample,<br/>conv, res block)"]
    D --> Y["64x64x3"]
```

| choice | why |
|---|---|
| 4×4 kernels at stride 2 down | every input pixel is covered exactly twice, so no checkerboard of pixels seen once and twice |
| nearest upsample + conv up, not a transposed conv | a stride-2 transposed conv overlaps unevenly and paints a faint checkerboard on flat regions, and a shapes picture is almost all flat region |
| GroupNorm, not BatchNorm | a picture's encoding must not depend on what else is in its training batch (the test checks the encoder in *train* mode, the only mode where BatchNorm would leak) |
| L1 + L2 pixel loss | L2 alone blurs edges, L1 alone is slow to fix a whole region's colour; no LPIPS or adversarial loss, because both need a network that is not ours |
| the latent flattened to 64 positions for `ResidualVQ` | raster order, so the quantizer sees `(B, D, T)` exactly as it did for audio |
| quantizer dropout | each batch uses 1 to 4 codebooks, so every prefix decodes (otherwise "1 codebook" is never trained) |

4.55M parameters. An 8×8 grid of 4 codes from 512 is 256 integers, or 2,304 bits, for a
picture of 98,304 bits. `configs/imgtok-shapes16.yaml` is the same model with two halvings
instead of three: a 16×16 grid, 1,024 integers. The pair asks how much the coarser grid
loses, read back by the reader, before G1 has to model 4× longer sequences.

## What a report says

`python -m aksharallm.pictures report <run>` reconstructs the val pictures at 1, 2 and 4
codebooks and writes `logs/pictures/report-<run>-step<N>.json` plus a gallery of PNG rows
(the original, then each codebook count):

| column | meaning |
|---|---|
| PSNR | pixel error in dB. The usual number and a weak one: most of a shapes picture is background, so a lost shape costs little PSNR |
| count / colour / shape / all three | what the reader sees in the reconstruction, against the description it was drawn from |
| held-out: all three | the same on purple triangles, a combination the tokenizer never saw |
| codebook use | distinct entries used and the effective size (perplexity) per codebook. A codebook of 512 with an effective size of 20 is a 20-entry codebook, and **the loss curve cannot show that** |

Each row keeps its per-picture verdicts, so two reports pair picture by picture:
`pictures compare A.json B.json` prints b-only / a-only and an exact sign test, for the val
pictures and the held-out ones. That is how the 8×8 and 16×16 grids are compared.

### The smoke run, 300 steps on the CPU

Not a result (the real run is 20,000 steps on the card), but it shows what the reader is
for. At step 299 the loss had fallen from 1.54 to 0.053, every codebook was in use (339–359
entries of 512), and the reader said:

| codebooks | PSNR | count | colour | shape | all three |
|---|---|---|---|---|---|
| 1 | 25.3 dB | 96.9% | 100% | 26.6% | 26.6% |
| 4 | 26.0 dB | 95.3% | 100% | 26.6% | 26.6% |

Colour and count were already there; **shape was not**. The gallery showed why: every
square came back as a soft round blob. 26 dB of PSNR says "close"; the reader says "a
quarter of these pictures now show the wrong shape". Held-out purple triangles: 0%.

### The full run, 20,000 steps on the card (2026-11-22)

Both grids trained to the end in about a quarter of an hour each (`imgtok-shapes` 4.55M
parameters, 16 min; `imgtok-shapes16` 1.28M, 14 min). 400 val pictures and the 64 held-out
purple triangles, reports from the best checkpoints:

| grid | codebooks | numbers / picture | PSNR (val) | all three (val) | PSNR (held out) | all three (held out) |
|---|---|---|---|---|---|---|
| 8×8 | 1 | 64 | 56.1 dB | 100% | 41.3 dB | 95.3% |
| 8×8 | 2 | 128 | 62.1 dB | 100% | 44.3 dB | 100% |
| 8×8 | 4 | 256 | 63.5 dB | 100% | 46.4 dB | 100% |
| 16×16 | 1 | 256 | 60.2 dB | 100% | 53.3 dB | 100% |
| 16×16 | 4 | 1,024 | 66.7 dB | 100% | 55.1 dB | 100% |

Every codebook is in use (495–511 of 512 entries, effective sizes 38–146). Paired picture by
picture at 4 codebooks the two grids disagree on **nothing** (0 / 0, val and held out), so the
coarser grid loses nothing the reader can see, and G1 models the 8×8 one. The held-out PSNR
is ~15 dB lower than val: purple triangles are reconstructed less exactly, but at one
codebook only three of 64 are misread (their colour), and at two none are.

## Run it

```bash
# G0, the tokenizer
python -m aksharallm.vision corpus                      # data/vision/shapes (seconds), if missing
python -m aksharallm.pictures read                      # the reader on the corpus: must be 100%
scripts/audio.sh imgtok-shapes                          # train (the non-text launcher)
python -m aksharallm.pictures report imgtok-shapes      # ladder + held-out + gallery
python -m aksharallm.pictures reports                   # every report, one line each
python -m aksharallm.pictures compare logs/pictures/report-imgtok-shapes-stepN.json \
    logs/pictures/report-imgtok-shapes16-stepM.json

# G1, a caption to a picture
python -m aksharallm.pictures codes imgtok-shapes --books 2     # data/pictures/codes-imgtok-shapes
# ...or the 64,000-picture corpus the counting fix trains on (docs/26 § Fixing count)
python -m aksharallm.vision corpus --out data/vision/shapes64k --images 64000
python -m aksharallm.pictures codes imgtok-shapes --books 2 --corpus data/vision/shapes64k \
    --out data/pictures/codes-imgtok-shapes-64k
scripts/audio.sh imggen-shapes64k-pos-long                      # the best G1 so far

# G2, flow matching in pixels (same 64k corpus)
scripts/audio.sh flow-shapes
python -m aksharallm.pictures gen-report flow-shapes --guidance 3 --k 32   # --steps N to vary
python -m aksharallm.pictures draw flow-shapes "three purple triangles"
scripts/audio.sh imggen-shapes                                  # train (encodes the codes if missing)
python -m aksharallm.pictures draw imggen-shapes "two red circles" --guidance 3
python -m aksharallm.pictures gen-report imggen-shapes --guidance 3   # all 60 captions, judged
python -m aksharallm.pictures gen-compare A.json B.json               # paired per drawing

# G4c, real photos (COCO 2017: the three archives into data/coco/raw/ first)
python -m aksharallm.pictures coco                              # data/vision/coco64 (+ coco64-val), ~2 min
scripts/audio.sh ae-coco && scripts/audio.sh clip-coco && scripts/audio.sh flow-coco
python -m aksharallm.pictures coco-report flow-coco --clip clip-coco --guidance 3
python -m aksharallm.pictures coco-compare A.json B.json        # same captions, paired

# G4b, our own contrastive model, and captions that say size and place
python -m aksharallm.pictures scenes                            # data/vision/scenes64k (+ -holdout)
scripts/audio.sh clip-shapes                                    # small: CPU is fine (CUDA_VISIBLE_DEVICES=)
python -m aksharallm.pictures clip-calibrate clip-shapes        # the none-fits floor (1% of clean originals)
python -m aksharallm.pictures clip-report clip-shapes           # beside the reader, originals
python -m aksharallm.pictures clip-report clip-shapes --through ae-shapes   # damaged pictures
python -m aksharallm.pictures gen-report imggen-shapes-gpu --clip clip-shapes
scripts/audio.sh ae-scenes && scripts/audio.sh flow-text-scenes
python -m aksharallm.pictures gen-report flow-text-scenes --guidance 1 --clip clip-shapes
```

`scripts/audio.sh` is the launcher for every run whose config is not a text model; it reads
the `picture:` section, renders the corpus if missing, checks the reader on 500 pictures
before spending the card, and publishes the same pid/meta/log contract as every launcher, so
`scripts/stop.sh`, the scheduler and the dashboard's Start drive it unchanged.

In the portal: **Pictures** (under *Reach further*) shows every report with the originals'
row first, the gallery, a paired comparison of two reports, and the jobs (a report, on the
CPU, the reader check, encoding the corpus for G1, a G1 report), each with its CLI command
printed. For G1 it has **Draw**: pick the three words and a guidance, and it draws on the CPU
in a second or two, each picture bordered green or red by the reader's verdict; and **Every
caption, judged**, with the held-out captions first in the gallery.

## From a sentence to a picture (G1)

The tokenizer turns out to be solved on this corpus (the full run, 20,000 steps): every val
picture and every held-out purple triangle reads back right at 4 codebooks, and **one codebook
is already enough for 100% of the val pictures** (56 dB), with 95% on the held-out ones. So G1
models the first **two** codebooks: 64 cells × 2 = 128 codes a picture, all 100% when read back.

G1 is the next-token transformer of [chapter 4](04-model.md), imported unchanged. A picture is
written as a sequence after its caption:

```mermaid
flowchart LR
    W["three red circles"] --> S["[three] [red] [circles] [IMG]"]
    S --> T["transformer<br/>(model/transformer.py, unchanged)"]
    T --> C["128 codes, one at a time:<br/>cell 0 book 0, cell 0 book 1, cell 1 book 0, ..."]
    C --> D["tokenizer's decoder (G0)"]
    D --> P["picture"] --> R["reader: right count,<br/>colour, shape?"]
```

| choice | why |
|---|---|
| one vocabulary: 3 specials + 15 words + 2 × 512 codes | a code from book 1 is a different id from the same number in book 0, so the model always knows which codebook a slot holds |
| position-major order (each cell's two codes together) | the second code refines the first, so it comes straight after it; codebook-major would ask for all 64 refinements before the model has seen any |
| the slot mask in the sampler | at each step only one codebook's 512 ids are legal; the sampler sets the rest to −∞. Training has no mask, so a model that had not learned the layout would show it in the loss |
| loss on codes only | the caption and the IMG marker carry no loss, so the val loss is cross-entropy **per code** (step 0 reads ln 1042 = 6.95, uniform over the whole vocabulary; ln 512 = 6.24 is a model that has learned only which codebook each slot holds) |
| words from the corpus's constants, not its captions | "purple" and "triangles" are words although no picture shows both |
| 10% of captions dropped to `NULL` | the same model then also draws with no caption, which is what guidance needs |

### Guidance

**Classifier-free guidance** is the knob every text-to-image system has. At each sampling step
the model is run twice, with the caption and with `NULL`, and the logits are pushed away from
the captionless ones:

```
logits = uncond + w * (cond - uncond)
```

`w = 1` is the model as trained, `w = 0` ignores the caption, and `w > 1` exaggerates whatever
the caption changes. In a diffusion model the same line is applied to the predicted noise;
here it is applied to the next code's logits. Both halves share one KV cache (2B rows, the
same length).

### How a drawing is scored

`pictures gen-report <run>` draws `k` pictures for each of the **60 captions** the words can
make (4 counts × 5 colours × 3 shapes), reads every one, and reports:

| column | meaning |
|---|---|
| seen / held out | the 56 captions with training pictures, and the 4 purple-triangle captions with none |
| count / colour / shape / all three | the reader's verdict against the caption |
| copies | drawings identical, code for code, to a training picture: memory, not drawing |
| variety | distinct pictures among one caption's `k`: what guidance trades for obedience |

Every (caption, sample) keeps its verdict, and sampling is seeded, so two reports (two guidance
settings, two checkpoints) pair drawing by drawing: `pictures gen-compare A.json B.json`.

### Measured: the first run (2026-11-22)

`imggen-shapes`, 5.09M parameters, 8,000 steps of 64 pictures, trained **on the CPU** (the
launcher refused to share the card with a queued run; ~2–3 s/step, 4 h). Step 0 read 6.99
against ln 1042 = 6.95.

**The val loss and the drawings disagree.** Val loss bottomed at **1.946 at step 2,000** and
then rose steadily to 2.52 at step 7,999 while the training loss kept falling: on likelihood,
7,600 pictures are memorised by 5M parameters. But the drawings did not get worse. Paired on
the same 480 drawings, step 2,000 and step 7,999 are indistinguishable (guidance 3: 83 vs 84
discordant, p 1.0; guidance 1: p 0.50), and neither copies a single training picture. Val
loss scores every possible picture; the sampler only ever visits the likely ones.

Every number below is from `ckpt_best.pt` (step 2,000), 8 drawings per caption (32 where
marked), seed 0:

| guidance | captions | n | count | colour | shape | all three | copies |
|---|---|---|---|---|---|---|---|
| 1 (none) | seen | 1,792 | 68.8% | 98.0% | 98.7% | **66.2%** | 0% |
| 1 (none) | held out | 128 | 68.8% | **35.2%** | 94.5% | **21.9%** | 0% |
| 3 | seen | 1,792 | 65.2% | 98.9% | 94.9% | **62.0%** | 0% |
| 3 | held out | 128 | 57.0% | **84.4%** | 90.6% | **43.0%** | 0% |
| 5 | seen | 448 | 47.5% | 98.0% | 79.7% | 36.8% | 0% |
| 5 | held out | 32 | 37.5% | 87.5% | 84.4% | 21.9% | 0% |

(32 per caption for guidance 1 and 3; variety is 100% everywhere: no caption ever drew the
same picture twice.)

**Guidance is what makes the unseen combination purple.** Without it, "three purple triangles"
comes out as triangles of the right shape in the *wrong* colour: of 128 held-out drawings, 45
are purple, 47 red, 21 blue and 14 yellow, and some single pictures mix two colours, which no
training picture ever does. The word "triangle" pulls towards the colours triangles were seen
in. Guidance exaggerates what the caption changes, and "purple" is part of that, so at
guidance 3 colour rises to 84% and the held-out score doubles, **21.9% → 43.0% (44 drawings
better, 17 worse, p 0.0007)**. It costs a little on captions the model knows, **66.2% → 62.0%
(330 better, 406 worse, p 0.006)**, and guidance 5 breaks shapes (seen 36.8%). That is the
trade every text-to-image system makes with this knob, measured exactly: obedience on the
unfamiliar against fidelity on the familiar.

**Count is the bottleneck.** Colour and shape are 94–99% on seen captions; count is 65–69%,
and its errors are almost all off by one in either direction (guidance 1, 32 per caption: asked for three, 113 drew two,
65 drew four). Counting needs the whole picture at once rather than one region, which is the
same order the vision model learned in (colour, then shape, counting last; chapter 22).


### Fixing count: four arms, one key apart (2026-11-22)

Two hypotheses for why count lagged. **Not enough pictures**: 7,600 pictures for 56 captions is
~136 each, the model memorised them by step 2,000 (val loss rose for the next 6,000 steps), and
counting is the attribute that needs the most examples, because it is a property of the whole
picture rather than of any one region. **No sense of where**: RoPE gives the transformer only a
1-D distance along the sequence, and two codes four steps apart in raster order may be side by
side or on different rows. The fix for the second is a learned *slot embedding*, row + column +
codebook of the code about to be written, added to each input (`picgen.pos2d`,
`PicGenTransformer`, a subclass that leaves `Transformer` untouched). A free check first:
lowering the sampling temperature to 0.4 moved count from 66% to 70%, so the fix had to be in
training.

The bigger corpus is `vision corpus --images 64000` with the same seed, so its first 8,000
pictures **are** the original corpus (checked byte for byte) and the reader is exact on all
64,000. All four arms trained on the GPU, 8,000 steps of 64, same seed, ~3 minutes each
(`logs/queue/g1-count.sh`); `test_the_counting_arms_are_each_one_key_from_the_control` pins
the configs. Final checkpoints, 32 drawings per caption, paired per drawing:

| arm | val (best / final) | seen: count | seen: all | held out: all (g1) | held out: all (g3) |
|---|---|---|---|---|---|
| control, 8k pictures | 1.945 / 2.677 | 65.6% | 63.8% | 14.8% | 48.4% |
| + slot embedding | 1.925 / 2.621 | 67.9% | 67.1% | 35.9% | 45.3% |
| 64k pictures | 1.280 / 1.280 | 83.4% | 83.2% | 68.8% | 67.2% |
| 64k + slot embedding | 1.261 / 1.261 | **86.3%** | **86.0%** | **74.2%** | 71.9% |

(seen columns at guidance 1; colour and shape are ≥ 98% on seen captions in every arm.)

* **More pictures is the fix.** Control → 64k: seen 63.8% → 83.2% (526 drawings better, 179
  worse, p 2e-40), the unseen purple triangles 14.8% → 68.8% (74–5, p 8e-17). The val loss no
  longer turns upward: best and final are the same number, so 8,000 steps under-trains it.
* **The slot embedding helps a little, every time.** +3 points on seen captions at 8k (372–314,
  p 0.03) and at 64k (241–190, p 0.016); its held-out gains are not significant at 64k
  (32–25, p 0.43). A small, real effect on top of the data, not instead of it.
* **The count errors changed shape.** The control mostly drew *too few* (asked minus drawn:
  +1 in 518 drawings, −1 in 129); the 64k + slot model errs both ways and rarely (142 / 120).
* **With enough pictures, guidance is no longer what draws the unseen combination.** At 8k,
  guidance 3 lifted the purple triangles 14.8% → 48.4% (49–6). At 64k + slot it does nothing
  for them (74.2% → 71.9%, p 0.79) and costs 3 points on seen captions (205–266, p 0.006): the
  model already binds "purple" to the triangles without being pushed. Guidance was
  compensating for a model that had not learned the words well enough.
* Copies stay at 0–0.3%, variety at 100%.

**And longer.** Because the 64k arms were still improving at 8,000 steps, the winning arm ran
again for 30,000 (`imggen-shapes64k-pos-long`, 11 minutes): val 1.261 → **1.125**, seen
captions **89.0%** at guidance 1 (count 89.1%; paired against the 8,000-step arm 195–143,
p 0.006), the purple triangles 70.3% (g1) / 77.3% (g3), not different from the shorter run
(p 0.58). What is left of the counting error grows with the number: asked for one shape it
draws one 98% of the time, two 92%, three 86%, four 80%, and it now errs almost only by drawing
one too few. From the first run to this one, count went **65.6% → 89.1%** and the never-seen
combination **14.8% → 70.3%** without guidance. It is the model the portal's Draw now picks
first (the newest checkpoint).

## G2: drawing by flow matching

G1 writes a picture one code at a time. **G2 starts from pure noise and moves the whole picture
towards a drawing**, in 50 small steps, steered by the caption at every one: the way Stable
Diffusion 3, Flux and the Veo/Sora family generate. It works in pixels (the latent version is
G3), on the same 64,000 pictures, and is scored by the same reader on the same 60 captions,
so G1 and G2 compare drawing for drawing.

```mermaid
flowchart LR
    N["noise<br/>x0 ~ N(0, I)"] -->|"50 Euler steps<br/>x += v / 50"| P["picture x1"]
    subgraph step["one step at time t"]
        X["x_t (noisy picture)"] --> T["transformer, causal: false<br/>[time] [3 words] [256 patches]"]
        T --> V["velocity v<br/>(where to move each pixel)"]
    end
```

**The training rule is four lines** (`pictures/flow.py`, `flow_loss`):

```
x0 ~ N(0, I)                     noise
x_t = (1 - t) x0 + t x1          a straight line from noise (t = 0) to a training picture (t = 1)
target = x1 - x0                 the direction along that line: its velocity
loss = | model(x_t, t, caption) - target |^2
```

This is **flow matching** (rectified flow). There is no noise schedule to tune (DDPM's betas and
alphas), and because the paths are straight, drawing takes tens of steps instead of a thousand.
Times are drawn from a logit-normal (as Stable Diffusion 3 does): most training happens where
the picture is half formed, which is where the decisions are.

**The model is the same transformer a third time.** A picture is cut into 16×16 patches of 4×4
pixels; each patch becomes a token (plus a row and column embedding), and the time step and
the three caption words become four more tokens in front. `model/transformer.py` runs over
them unchanged with `causal: false` (every patch sees every other patch, as in masked
diffusion, chapter 20), and a linear head reads a velocity off each patch's hidden state. The
caption enters **in context**, as tokens, not through a new kind of block (DiT's adaLN): the
transformer is told nothing. The config says so in a new key, `embeds_only: true`, because a
bidirectional model with no mask token used to be refused as having nothing to train on, and
here the objective is outside it.

| choice | why |
|---|---|
| straight paths, predict velocity | the simplest objective that samples in few steps; the loss is a plain MSE |
| output layer initialised to zero | step 0 predicts zero velocity, so the first loss is a known number, 1 + mean(x1²) ≈ 1.72 here |
| logit-normal times | little training at t ≈ 0 (pure noise) and t ≈ 1 (finished picture), where there is nothing to learn |
| fixed seed for val | val uses the same pictures, noise and times every time, so "best" is never the kindest noise |
| guidance on velocities | `v = v_uncond + w (v_cond − v_uncond)`, the same line as G1's, on a different quantity |
| copies and variety through the tokenizer | drawn pixels are encoded by G0 and checked against the training pictures' codes, exactly as G1's drawings are |

`pictures/drawer.py` is one door for both models: `draw`, `gen-report` and the portal's Draw
panel load a G1 or a G2 checkpoint the same way, and `gen-compare` pairs a G2 report against a
G1 one when they drew the same captions with the same k and seed.

### Measured: G2 against G1 (2026-11-22)

`flow-shapes`, 14.5M parameters, 30,000 steps of 64 pictures on the GPU in 56 minutes (0.17
s/step beside another run, 0.08 alone). Step 0 read 1.716 (zero velocity, as predicted); val
fell to **0.0047** and never turned upward. During training (2 drawings per caption, guidance
3) the score went 56% → 83% → 68% → 98% → 99% → 100% at steps 5k–30k.

On the same 60 captions, 32 drawings each, same seed, against the best G1
(`imggen-shapes64k-pos-long`, same 64,000 pictures, same 30,000 steps):

| model | guidance | seen: count | seen: all three | held out: all three | copies |
|---|---|---|---|---|---|
| G1 (codes, one at a time) | 1 | 89.1% | 89.0% | 70.3% | 0% |
| **G2 (flow matching, pixels)** | 1 | **99.2%** | **99.2%** | **100%** | 0% |
| G1 | 3 | 86.4% | 84.5% | 77.3% | 0% |
| **G2** | 3 | 99.1% | 99.1% | 100% | 0% |

Paired drawing by drawing at guidance 1: seen **197 better, 13 worse (p 2e-43)**, the purple
triangles **38 better, 0 worse (p 7e-12)**. Colour and shape are 100% everywhere, count 99%,
every one of the 128 held-out drawings is right, no drawing is a copy of a training picture,
and no caption ever drew the same picture twice. Guidance changes nothing (99.2% vs 99.1%):
the model already obeys the words.

**Why G2 counts where G1 could not.** G1 commits to the top-left cell first and can never take
it back; by the time it learns there should have been one more triangle, the cells where it
could go are written. G2 decides the whole picture together and revises it at every step:
"how many shapes are there" is a question about the picture as a whole, and every step of G2
looks at the picture as a whole.

**How few steps it needs** (guidance 1, 8 per caption):

| Euler steps | 2 | 5 | 10 | 25 | 50 |
|---|---|---|---|---|---|
| seen: all three | 40.0% | 94.9% | 98.0% | 99.1% | 99.2% |
| held out: all three | 9.4% | 87.5% | 96.9% | 100% | 100% |

Straight paths are what make this possible: 10 steps give 98%, where DDPM-style diffusion
needs hundreds. G2's drawing costs 50 forward passes of a 14.5M model; G1's costs 128 passes of
a 5M one, so the two are within a factor of about two in compute per picture.

The shapes corpus is now solved by G2. What it cannot show is detail, texture or real
captions; that is what G3 (the same thing in G0's latent, far cheaper per step) and G4 (real
text, real pictures) are for.

## G3: flow matching in the latent

G2 works on 12,288 numbers a picture. Most of them are predictable from their neighbours (a
pixel inside a red square is red), so every real image model, Stable Diffusion, Flux and Veo
included, works in a smaller space and lets a decoder paint the pixels. G3 does exactly that
with what this chapter already has: G0's encoder gives an **8×8×64 latent** (4,096 numbers),
G2's flow model draws one of those instead of a picture, and G0's decoder paints it.

```mermaid
flowchart LR
    N["noise<br/>8x8x64"] -->|"50 Euler steps,<br/>64 tokens each"| Z["latent"]
    Z --> D["× std + mean<br/>(undo the normalising)"]
    D --> S{"snap?"}
    S -->|no| DEC["G0's decoder"]
    S -->|yes| Q["re-quantize onto<br/>the codebooks"] --> DEC
    DEC --> P["64x64 picture"] --> R["reader"]
```

Nothing new is trained except the flow model: `pictures/flow.py` with `space: latent`,
`patch: 1`, `channels: 64` (one token per latent cell, 64 tokens against G2's 256), the same
width, depth, steps, schedule and seed as G2 (`configs/flow-latent-shapes.yaml`, one block
apart from `flow-shapes.yaml`).

Three decisions in `pictures/latents.py`:

| decision | why |
|---|---|
| the latent is `q`, the quantized sum at all 4 codebooks, not the encoder's raw `z` | the decoder only ever saw `q`; `z` is a point it was never trained to paint |
| normalised per channel, statistics from the training split | flow matching starts from unit-variance noise; a latent whose channels range from 0.1 to 0.5 in scale would make the straight line mostly a change of scale. The mean and std go into every checkpoint |
| optional **snap** before decoding | re-quantizing a drawn latent onto the codebooks removes small off-grid errors the decoder never saw; measured both ways |

The round trip is checked before anything trains: the 300 val latents, mapped back and decoded,
read 100% right (with or without snapping). Decoded **without** mapping them back, the same
latents read 62% right, which is the silent bug lesson 32 plants.

`pictures latents imgtok-shapes --corpus data/vision/shapes64k` writes the corpus (524 MB of
float16, 5 minutes on the CPU); `scripts/audio.sh flow-latent-shapes` encodes it if missing.

### Measured: G3 against G2 (2026-11-22)

`flow-latent-shapes`, 14.5M parameters (the same transformer as G2), 30,000 steps, run on the
card with the hybrid paused. **16 minutes**, 0.030–0.037 s/step against G2's 0.083 alone: 64
tokens instead of 256 makes every step ~2.5× cheaper. Val (fixed noise, in normalised latent
units, so not comparable with G2's pixel val) 0.110 → 0.0386, still falling at the end.

32 drawings per caption, paired with G2 on the same captions and seed:

| model | guidance | seen: count | seen: all three | held out: all three | copies |
|---|---|---|---|---|---|
| G2 (pixels) | 1 | 99.2% | **99.2%** | **100%** | 0% |
| G3 (latent) | 1 | 87.7% | 87.6% | 79.7% | 0.7% |
| G3, snapped | 1 | 87.7% | 87.6% | 78.9% | 0.8% |
| G3 | 3 | 85.4% | 85.3% | 74.2% | 1.2% |

Paired at guidance 1: G2 is right where G3 is wrong 216 times, the reverse 8 (p 1e-53); on the
purple triangles 26 to 0 (p 3e-8). **Colour and shape survive the move to the latent (≥ 99.8%);
count does not.** It is 96% for one shape and 80% for four, and the gallery shows why: G3 now
and then puts two shapes **touching**, which no training picture does (the corpus spaces them on
a grid), and the reader rightly sees one blob. Its count errors go both ways (one too many in
159 drawings, one too few in 84). Snapping to the codebooks changes nothing; guidance 3 hurts.

The likeliest reason is resolution: a latent cell covers 8×8 pixels, so G3 decides where a
shape goes on a 64-cell grid, where G2 decides on 256 cells of 4×4 pixels, and a misplaced cell
is a shape eight pixels out. The copies (0.7%) are all one-square captions, which have only
about two hundred possible layouts; identical code grids there are coincidence, not memory.

So the trade is the one latent models make, measured: **~2.5× cheaper per step for a loss in
placement precision**, at the same number of steps.

### Is it the latent's resolution? No (2026-11-22)

The obvious test of "8×8 cells are too coarse" is the 16×16 tokenizer, `imgtok-shapes16`, which
is already trained and reconstructs every picture perfectly (G0). Two arms on its 16×16×64
latent, each one config block from G3 (`configs/flow-latent16p2-shapes.yaml`,
`flow-latent16-shapes.yaml`; same width, depth, steps, schedule and seed):

| arm | latent | tokens | 30k steps | seen: count | seen: all three | held out |
|---|---|---|---|---|---|---|
| G2 | pixels, 12,288 numbers | 256 | ~42 min | 99.2% | **99.2%** | **100%** |
| G3 | 8×8×64, 4,096 numbers | 64 | 16 min | 87.7% | **87.6%** | **79.7%** |
| G3-16, 2×2 cells per token | 16×16×64, 16,384 numbers | 64 | 16 min | 52.3% | 51.4% | 32.0% |
| G3-16, 1 cell per token | 16×16×64, 16,384 numbers | 256 | 44 min | 50.8% | 50.3% | 27.3% |

The finer latent is **far worse**, not better (G3 against either arm: 748–99 and 761–92 drawings,
p < 1e-120), and the two 16×16 arms are the same as each other (358–378, p 0.48): four times the
tokens bought nothing. The drawings are clean, crisp shapes of the right colour and kind in the
wrong number, with errors both ways (asked for three: two, four, five).

So the explanation in the previous section was wrong, or at least not the main thing. What the
numbers do say:

* **A latent that reconstructs perfectly is not automatically one a generator can learn.** Both
  tokenizers decode every picture right. The 16×16 one is shallower (two halvings, 1.28M
  parameters against 4.55M) and its latent holds *more* numbers than the picture (16,384 against
  12,288), so it is less a summary of the picture than a re-encoding of its pixels.
* **It is not the token count**, since 64 and 256 tokens score the same on it, and not raw size
  alone, since G2 draws 12,288 pixel values with 256 tokens at 99%.
* Every latent here is a sum of four codebook vectors, so its values cluster rather than vary
  smoothly the way pixels do; a flow learns a smooth path to a target, and a clustered target
  may be what makes the bigger latent hard. That is a hypothesis, not a measurement.

This is why real latent models do not use a plain VQ tokenizer's latent: Stable Diffusion's
autoencoder is trained with a small KL penalty that keeps its latent smooth and roughly
Gaussian, precisely so that a diffusion model can learn it. The next section is that experiment.

### A latent made to be learned: G3b (2026-11-22)

`pictures/autoencoder.py` keeps the tokenizer's encoder and decoder and replaces the codebooks
with a **Gaussian latent**: the encoder predicts a mean and a log variance per latent number,
training decodes a sample (mean + spread × noise), and a small KL term (weight 1e-3) pulls every
latent towards a unit Gaussian. The latent is **8×8×8 = 512 numbers**, against the VQ latents'
4,096 and 16,384. For drawing, the latent is the encoder's mean.

```mermaid
flowchart LR
    X["picture"] --> E["encoder<br/>(the tokenizer's)"] --> M["mean, log variance<br/>8x8x8"]
    M -->|"training: mean + spread x noise<br/>drawing: the mean"| Z["latent"]
    Z --> D["decoder<br/>(the tokenizer's)"] --> Y["picture"]
    M -.->|"KL to N(0, 1), weight 1e-3"| K["keeps the latent smooth"]
```

`ae-shapes` trained in 16 minutes and reconstructs **100%** of the val pictures and of the
held-out purple triangles (56 dB). Three of its eight channels carry almost nothing (spread
0.02 against 0.4–0.7): the KL term turned them off, which the decoder ignores. Then G3's flow
model, unchanged, on its latent (`flow-ae-shapes`, one block from `flow-latent-shapes`; copies
are still checked with the VQ tokenizer's codes, `flow.score_tokenizer`):

| model | what it draws | numbers | 30k steps | seen: all three | held out |
|---|---|---|---|---|---|
| G2 | pixels | 12,288 | ~42 min | 99.2% | 100% |
| G3 | VQ latent, 8×8×64 | 4,096 | 16 min | 87.6% | 79.7% |
| G3 | VQ latent, 16×16×64 | 16,384 | 16–44 min | ~51% | ~30% |
| **G3b** | **continuous latent, 8×8×8** | **512** | **15.5 min** | **98.9%** | **97.7%** |

G3b against G2, paired: 13 better, 18 worse on seen captions (**p 0.47, a tie**), 0–3 on the
held-out ones (p 0.25); against G3: 220–17 (p 4e-46). Colour, shape and count are all ≥ 97.7%,
no copies, every drawing different.

**So the explanation holds**: the VQ latents were hard because of what they were, not how big.
A smooth latent made for the purpose, a twenty-fourth the size of the picture, draws as well as
the pixels at a third of the cost, which is the whole case for latent diffusion. This
autoencoder is what G4 (real sentences and photos) will build on.

## G4a: reading a sentence (2026-11-23)

Every model so far read exactly three words from a vocabulary of fifteen. A text-to-image
model reads a sentence. G4a keeps G3b (the flow model on the continuous latent) and changes
only how the caption gets in: a **frozen language model of our own** reads it, the way Imagen
reads its prompt with T5 and Stable Diffusion with CLIP's text model.

```mermaid
flowchart LR
    S["'a picture of two red circles'"] --> TOK["tiny's tokenizer"] --> LM["tiny, 13.8M<br/>(frozen, never trained here)"]
    LM --> H["12 vectors of 384<br/>(final hidden states, zero-padded)"]
    H --> P["linear projection"] --> F["G3b's flow model:<br/>[time] [12 caption vectors] [64 latent cells]"]
    F --> D["the autoencoder's decoder"] --> PIC["picture"]
```

`tiny` is the TinyStories model of Phase 1. It has never seen these pictures, and nothing in
it is trained: `pictures/text.py` runs it once per distinct sentence and caches the vectors.
The flow model learns a projection from its 384-wide vectors to its own width, and nothing
else changes (`configs/flow-text-shapes.yaml` is `flow-ae-shapes.yaml` plus four text keys).

**The sentences** (`pictures/phrases.py`) are ten phrasings of every caption, used in
training at random ("a picture of two red circles", "2 red circles", "two circles, all red",
"red circles, two of them", "there are two red circles", "two circles coloured red", "an image
showing …", "… on a dark background", "draw …"), and **two phrasings never used in training**:
"show me two red circles" and "two circles in red". That is a second held-out axis beside the
purple triangles. Reports pass the facts each sentence was made from, so the reader is never
checked against a parsed caption (`gen-report --phrasing NAME`; two phrasings are paired with
`gen-compare --by-fact`).

**Measured** (`flow-text-shapes`, 30,000 steps, 16 minutes on the card; 32 drawings per fact,
guidance 1, the same seed throughout):

| caption said as | seen: all three | held out (purple triangles) |
|---|---|---|
| G3b, three fixed words | 98.9% | 97.7% |
| **G4a**, the same three words | **99.3%** (vs G3b 19–12, p 0.28) | 97.7% |
| G4a, "a picture of …" (trained) | 99.5% | 98.4% |
| G4a, "… , N of them" (trained) | 99.2% | 98.4% |
| G4a, digits "2 red circles" (trained) | 99.5% | 98.4% |
| G4a, "two circles in red" (**never trained**) | 99.0% (vs three words 8–14, p 0.29) | 99.2% |
| G4a, "show me two red circles" (**never trained**) | 97.4% (8–42, p 1e-6) | 95.3% |

Reading through a frozen language model costs nothing against three fixed words; every
training phrasing scores the same; one new phrasing scores the same and the other loses two
points, a real but small cost of an imperative it never saw. In the portal a G4 model takes
any sentence: "could you paint three yellow squares please", a phrasing nowhere in training,
drew 4 of 4 right. A sentence with no count, colour or shape in it ("a happy dog") is drawn
(as shapes, which is all it knows) and shown unjudged.

What this does **not** show yet: understanding beyond these three attributes. Every sentence
here still names exactly one count, one colour and one shape. G4b (our own CLIP-style scorer)
and G4c (real photos and captions) are where that gets tested.

## G4b: a judge that learned (2026-11-23)

Everything above was scored by the reader, which is exact only because we drew the pictures.
Photos (G4c) have no such judge. Real image models are scored by a **contrastive** model that
has learned which captions go with which pictures: OpenAI's CLIP, and the scores built on it.
G4b trains one of our own, and earns it trust the only way a learned judge can: by agreeing
with the exact one, picture by picture, where both apply.

```mermaid
flowchart LR
    P["picture"] --> V["ViT (docs/22)<br/>mean of 64 patches"] --> A["128 numbers,<br/>unit length"]
    C["caption"] --> T["tiny, frozen"] --> H["one masked layer,<br/>masked mean"] --> B["128 numbers,<br/>unit length"]
    A --> S["cosine"]
    B --> S
    S --> Q["which caption, of all<br/>the corpus can say,<br/>is this picture's?"]
    Q --> K["compared with the reader:<br/>agreement and kappa"]
```

**The loss counts every caption of the same fact as a match.** CLIP's loss is a cross-entropy
over the batch in both directions, with picture *i* matching caption *i* only. With 60 shapes
facts (or 270 scene facts) a batch of 128 repeats facts, often in different phrasings, and the
diagonal would push two pictures of "two red circles" apart. `clip.clip_loss` spreads each
row's target over every row with the same fact; with one match per row it is CLIP's loss
exactly. [Lesson 34](lessons/34-contrastive.md) puts the diagonal back.

**The score it gives** (`own_clip` in a report, `pictures gen-report --clip clip-shapes`): a
drawing counts as right when, of every caption its corpus can say in canonical wording (60 or
270), the one it was drawn for is the closest. Beside it, per picture, the reader's verdict, and
how often the two agree, with Cohen's kappa (agreement beyond what their two rates would give by
chance: 1 always, 0 a coin). It is our feature space, so it is not CLIPScore and not FID.

### Captions that say more: scenes

The second half of G4b is a corpus whose captions say two more things, the two that real
text-to-image models are known to get wrong (`pictures/scenes.py`, `pictures scenes`):

    two big red circles on the left
    one small blue square in the middle

1–3 shapes in one column of the same 3×3 grid, small (half-size 4.8 px) or big (7 px). The reader
gained two readings (`size_of`: the blob's bounding box across, 12 px between the two;
`where_of`: the column every blob's centre is in, else "spread") and is exact on 10,800 renders
of all 270 facts; `judge` checks only what a fact names, so shapes facts are judged exactly as
before. **Two combinations are never drawn in training**: purple triangles, and **circles on
the right**, the held-out combination of the new attribute (circles everywhere else, other
shapes on the right). The ten phrasings say size with the shape word and the place at the end
("two big circles, all red, on the left"); for a shapes fact they give exactly G4a's strings.

`ae-shapes`, trained only on shapes, reads scenes back right 91% of the time (small circles
come back square-ish), so the scenes flow model gets its own autoencoder, `ae-scenes`, then
`flow-text-scenes` (G4a's recipe on it, `text_len` 16).

### Measured: our judge against the reader (2026-11-23, CPU)

`clip-shapes`: 2.4M trainable parameters (ViT 192×4, a one-layer text head over frozen `tiny`),
batch 128, 3,000 steps, **19 minutes on the CPU** (~2 s/step while the text cache filled, then
0.3–1 s). Best checkpoint at step 2,750. Its closest caption is the picture's own on:

| pictures (known facts) | n | reader | our model | agree | kappa |
|---|---|---|---|---|---|
| shapes, val originals | 512 | 100% | **100%** | 100% | – |
| scenes, val originals | 512 | 100% | **100%** | 100% | – |
| purple triangles (never in training) | 64 | 100% | **100%** | 100% | – |
| scenes held out: purple triangles, circles on the right | 128 | 100% | **100%** | 100% | – |
| G1 `imggen-shapes-gpu` drawings (8 per caption, all 60 captions) | 480 | 60.4% | 63.7% | **95.4%** | **0.90** |
| G1 `imggen-shapes64k-pos-long` drawings | 480 | 88.5% | 90.8% | **97.7%** | **0.88** |
| G4a `flow-text-shapes` drawings | 480 | 99.8% | 100% | 99.8% | (one disagreement) |
| scenes through `ae-shapes` (damaged) | 512 | 91.4% | 97.3% | 90.2% | **0.10** |

On a held-out phrasing ("show me …") it is 99.4–100% on originals, so it reads sentences the way
G4a does. Kappa is blank where every verdict is "right": with no wrong pictures there is
nothing to agree about beyond chance.

**Where they disagree, it is almost always the model accepting what the reader rejects** (19
against 3, 10 against 0, 1 against 0 on the three drawing sets), and every one of those on the G1
drawings is **five or six shapes drawn for "four"**. That is not a slip; it is what a
closed-set judge is. It answers "which of these captions fits best?", and no caption it is
offered says "five", so "four" is the best fit. It cannot say "none of these". CLIP-style
scoring has the same limit, which is why a count it was never offered is the place to distrust
it.

**On damaged scenes the two judges part ways (kappa 0.10)**, and looking at the pictures shows
why: `ae-shapes` repaints small squares with rounded tops (the reader, by bounding-box fill,
calls them circles; the model still calls them squares) and small circles as bell shapes (the
reader still says circle; the model says triangle). Neither judge is right about a shape that
is half one thing and half another; they are simply strict in different places. So the learned
judge is trustworthy on clean pictures and on the mistakes generators actually make (wrong
count, wrong colour), and not a substitute for the reader on borderline shapes.

### None fits: a floor, and what it costs (2026-11-23)

A closed-set judge picks the best caption it is offered, so five shapes come out "four". The fix
is a **cosine floor** (`pictures clip-calibrate clip-shapes`, written to
`checkpoints/clip-shapes/none_fits.json` with the step it belongs to): the 1st percentile of how
well clean val originals (the last 2,048 of each corpus, never the ones a report reads) match
their own canonical caption: 0.920 for shapes, 0.948 for scenes. A picture whose best caption
fits worse than that is judged **none fits**. It applies to canonical captions only; another
phrasing sits on its own scale.

The numbers that decided it (cosine with the asked caption):

| pictures | 1st pct | median |
|---|---|---|
| clean seen originals | 0.921 | 0.961 |
| G1 drawings the reader calls right | 0.80–0.92 | 0.96 |
| five or six shapes drawn for "four" (28) | 0.77 | **0.90** |
| clean originals of the **held-out** combinations | 0.88–0.90 | 0.96–0.97 |

So the floor catches most of the five-for-four pictures, and it also refuses the held-out
originals that fit a little less well because no training picture looked like them. The margin
(best caption's lead over the runner-up) was tried as well and separates them worse.

| | closed set | with the floor |
|---|---|---|
| G1 `imggen-shapes-gpu`, agree / kappa / model-only / reader-only | 95.4% / 0.90 / 19 / 3 | **96.7% / 0.93 / 6 / 10** |
| G1 `imggen-shapes64k-pos-long` | 97.7% / 0.88 / 11 / 0 | 97.5% / 0.88 / 2 / 10 |
| clean held-out originals, shapes / scenes | 100% / 100% | **92.2% / 77.3%** |
| clean seen originals | 100% | 99.0–99.2% (1% by construction) |

The floor turns a lenient judge into a strict one: its disagreements with the reader move from
"accepts what is wrong" to "refuses what is right", and on the pictures that matter most, the
never-seen combinations, it refuses up to a quarter. **A cutoff calibrated on what training
showed also refuses what is new**, which is exactly what a held-out test exists to accept. Both
verdicts are reported (`own_clip.closed` beside the floored one, and a "closed set" column in
`clip-report`), so a reader can see which kind of error a number carries. Offering the captions
that are missing (an open vocabulary, real for G4c's photos, where every val caption is a
candidate) is the better fix than any floor.

### Measured: drawing scenes (2026-11-23)

`ae-scenes` (24 min) is the scenes corpus's own autoencoder, because `ae-shapes` read scenes back
only 91% right; `ae-scenes` reads back **93%** (94% on the held-out combinations), so 93% is the
ceiling for anything drawn through it. `flow-text-scenes` (G4a's recipe on it, 18 min on the card)
then drew every one of the 270 facts 8 times, judged by the reader and by `clip-shapes`:

| report | seen, all right | held out, all right | circles on the right: shape | purple triangles |
|---|---|---|---|---|
| canonical, guidance 1 | 91.4% | 66.7% | 59.6% | 83.3% |
| `show_me` (never trained), g1 | 91.2% | 66.4% | 59.6% | 84.0% |
| `in_colour` (never trained), g1 | 91.2% | 64.6% | 65.0% | 69.4% |
| canonical, **guidance 3** | 91.1% | **84.9%** | **100%** | 85.4% |

* **Seen captions sit at the autoencoder's ceiling**: count and place are 92% each, size, colour
  and shape 99–100%. The 9% that is lost is mostly two shapes drawn touching, the same failure the
  autoencoder makes reading them back.
* **The new attribute generalises; the new combination needs guidance.** A circle has never been
  shown on the right, and at guidance 1 only 60% of those circles come out round (the rest are the
  squares and triangles training put there). Guidance 3 makes it 100%, and held out goes 66.7 →
  84.9% (paired 73–3, p 2e-18) while seen captions do not move (18–24, p 0.44). On the shapes,
  guidance stopped mattering once the data was enough; on scenes, where a held-out combination
  conflicts with a place, it still does.
* **Phrasings it never saw cost nothing measurable** (paired against canonical: `show_me` 11–12,
  p 1.0; `in_colour` 19–27, p 0.30), except that `in_colour` loses some purple (83% colour).
* **The two judges agree on seen pictures (98%, kappa 0.88) and part on held-out ones** (81%,
  kappa 0.63, and nearly every disagreement is the reader accepting a picture our model refuses):
  the floor at work, refusing what is new (see above).

## G4c: real photos (built 2026-11-23)

The first corpus nobody drew for us. COCO 2017: 118,287 training and 5,000 validation photos of
everyday scenes, each with five sentences written by people. `pictures coco` reads the three
downloaded archives member by member (never extracted), cuts each photo to its centre square
and resizes it to 64×64, and writes the same `images.bin` + manifest as every corpus here
(`kind: coco`, one fact per photo: its id and its five captions). 118k photos in two minutes;
1.4 GB.

```mermaid
flowchart LR
    C["a COCO caption"] --> T["the 300M small-code-py,<br/>frozen (it read web English)"]
    T --> F["flow model, 512 x 12,<br/>on an 8x8x16 latent"]
    F --> D["ae-coco's decoder"] --> P["64x64 photo"]
    P --> J["clip-coco: its own caption<br/>in the top 5 of 1,000?"]
    R["the real photo of<br/>that caption"] --> J
```

Three runs, each one of the earlier recipes pointed at photos:

| run | what | from |
|---|---|---|
| `ae-coco` | the continuous autoencoder, 8×8×**16** latent (a photo carries more than shapes on a flat ground), plain L1+L2: expect blur | G3b's `ae-shapes` |
| `clip-coco` | the contrastive judge: a 6-layer ViT and the frozen 300M; a photo's five captions are all its matches | G4b's `clip-shapes` |
| `flow-coco` | text to photo: the flow model, larger, on `ae-coco`'s latent, reading a random one of the photo's five captions through the frozen 300M | G4a's `flow-text-shapes` |

**The caption is read by the 300M, not `tiny`.** `tiny` only ever read children's stories;
"a giraffe standing next to a zebra" needs a model that has read the web. The text model runs on
the card (`text_device: auto`), and with 590,000 captions its vectors cannot all be cached
(`text_cache: 30000`); the rest are encoded every step, all the uncached captions of a batch in
one forward, padded **at the end**: a causal model's vector at a real token never sees what
follows, so a padded row equals the row alone (`tests/test_coco.py`, and
[lesson 35](lessons/35-padding.md) pads at the front).

**The report** (`pictures coco-report flow-coco --clip clip-coco`, `pictures/photos.py`): one
drawing for each of the first 1,000 val captions, and every number twice, for the drawings and
for the **real photos** of the same captions, with chance beside them: how well the drawing
matches its caption (and another photo's caption, the floor), whether its own caption is in the
top 1 / 5 / 10 of the 1,000, and the Fréchet distance between the drawings' and the real photos'
image vectors in our model's features, beside the same distance between two halves of the real
photos. It is FID's formula in our own features, so it is not FID. Two reports on the same
captions pair per caption (`pictures coco-compare`).

### Measured: real photos (2026-11-23)

All three runs by day on the card: `ae-coco` 33 min (40k steps, val **27.5 dB** PSNR: blurry, as
expected of plain L1+L2), `clip-coco` 50 min (15k steps; on its own val photos the right caption is
the closest in a batch 13.4% of the time, 37.8% in the top 5), `flow-coco` 4 h 40 min (60k steps,
0.26 s/step). `coco-report`, 1,000 val captions the flow model never saw, one drawing each:

| | match (cosine) | another photo's caption | own caption in top 1 / 5 / 10 of 1,000 |
|---|---|---|---|
| drawn, guidance 1 | 0.416 | 0.074 | 6.4 / 18.4 / 27.7% |
| drawn, guidance 3 | **0.578** | 0.037 | **26.1 / 59.7 / 76.1%** |
| the real photos | 0.489 | 0.059 | 11.5 / 33.5 / 46.8% |
| chance | | | 0.1 / 0.5 / 1.0% |

Distance to the real photos in our features: **0.08**, against **0.04** between two halves of the
real photos. Guidance 3 against 1, paired per caption on top 5: **436–23, p 5e-100**.

![Real COCO val photos above, flow-coco drawings of their captions below](img/coco-flow-g3.png)

*Real photo above, drawing below, guidance 3, the first 16 val captions.* The bear, the stop sign,
the bedroom, the kitchen, the baseball field, the skier and the surfer are recognisable at 64 px;
faces, bodies and lettering are smears ("STOP" comes out as "ƧMUP").

**Read the guidance-3 row with care: it beats the real photos, and that does not mean it is
better than real.** Guidance pushes each drawing toward what is most typical of its caption, which
is exactly what a contrastive judge rewards; a real photo is matched to one of its five captions
and also holds everything that caption leaves out. So past the real photos' ceiling the judge
measures how typical a drawing is, not how good. The two honest numbers are guidance 1 (well
above chance, below the ceiling) and the distance (twice the real photos' own). This is the
reason every number is printed beside the real photos': without that row, 59.7% would read as a
model that draws better than a camera. Two cautions not yet tested: the judge and the drawer read
the caption through the same frozen 300M, so their errors may agree; and the ceiling is the real
photos as photographed, not through `ae-coco`, whose blur the drawings inherit.

## Sharper photos (built 2026-11-23)

The photos above are blurry, and the first question is where the blur comes from. A drawing is
painted by `ae-coco`'s decoder, so it can be no sharper than that decoder. `pictures ae-report`
passes the 1,000 val photos through an autoencoder and judges what comes back with the same
contrastive model, so the autoencoder gets the same table as a drawing:

| `ae-coco`, 1,000 val photos | match | top 1 / 5 / 10 | sharpness | PSNR |
|---|---|---|---|---|
| the real photos | 0.489 | 11.5 / 33.5 / 46.8% | 1.00 | |
| through `ae-coco` | 0.455 | 9.3 / **25.0** / 37.1% | **0.54** | 27.75 dB |

**Sharpness** (`measure.sharpness`) is the power above a quarter of a cycle per pixel (fine
texture and sharp edges; Nyquist is half a cycle), against the real photos': 1.00 is as sharp,
below 1 is blurrier. `ae-coco` keeps 54% of it, 38% of the finest band. And the autoencoder alone
costs eight points of top 5 (33.5 → 25.0%). So the honest ceiling for a drawing through `ae-coco`
is 25%, not the real photos' 34%, and `coco-report` now prints that row too ("real photos through
its autoencoder").

**Why L1 + L2 blurs.** When the decoder cannot tell exactly where an edge or a blade of grass
falls, the average of the places it might be costs least, and an average of edges is a smear.
To a pixel loss a sharp edge one pixel off is wrong twice, once where it is and once where it
should have been. Nothing in the loss prefers it.

**The fix: a patch discriminator** (`pictures/adversarial.py`). A second, small network looks at
38-pixel patches of a photo and says whether each came from a real photo or from the decoder; the
decoder is also trained to make its patches score as real. A smear does not: real photos have
texture. This is how Stable Diffusion's autoencoder is trained (it also uses a pretrained
perceptual network, which this repo does not have, since everything here is trained by us).

```mermaid
flowchart LR
    X["photo"] --> E["encoder"] --> Z["latent 8x8x16"] --> D["decoder"] --> R["reconstruction"]
    R --> L["L1 + L2 + KL<br/>(as before)"]
    R --> P["patch discriminator:<br/>8x8 real/fake scores"]
    X --> P
    P --> G["decoder: make R's<br/>patches read as real"]
    P --> H["discriminator: tell<br/>X from R (hinge)"]
```

Four details, each pinned by a test (`tests/test_sharper.py`):

* **Hinge losses.** The discriminator pushes real scores above +1 and fake ones below −1 and then
  stops, so it cannot run away from a decoder still learning. The decoder's loss is *minus* the
  score of its own patches; drop the minus and it learns to look more fake while every curve
  still moves ([lesson 36](lessons/36-adversary.md)).
* **An adaptive weight.** The adversarial gradient at the decoder's last layer is scaled to the
  size of the reconstruction gradient there, times `weight` (0.5). A fixed weight is either too
  weak early or, once the discriminator is confident, strong enough to paint texture that is not
  in the photo.
* **A late start.** The discriminator switches on at step 10,000 of 40,000, once the autoencoder
  already reconstructs (`ae-coco` was at 26.3 dB by then).
* **Best is the latest checkpoint** once the adversary has started. The pixel loss rises as the
  decoder trades averages for texture, so any rule that ranks by it picks the least sharp
  checkpoint there is. The first version ranked by it "from the start on", picked step 10,000,
  one step in, and measured sharpness 0.47 for a run whose last step reached 0.77.

**The experiment** is two controlled pairs, each one change apart (a test asserts it):
`ae-coco` → `ae-coco-gan` (the `adversarial:` section), and `flow-coco` → `flow-coco-gan` (the
latent it draws in). The question is whether the detail the adversary adds survives being
*drawn* rather than reconstructed: a flow model learns the latent's distribution, and fine texture
is the part of it least determined by the caption.

```bash
python -m aksharallm.pictures ae-report ae-coco                      # the blurry one, measured
scripts/audio.sh ae-coco-gan                                         # + the discriminator
python -m aksharallm.pictures ae-report ae-coco-gan --against ae-coco   # paired per photo
scripts/audio.sh flow-coco-gan                                       # flow on its latent
python -m aksharallm.pictures coco-report flow-coco-gan --clip clip-coco --guidance 3
python -m aksharallm.pictures coco-compare A.json B.json             # same captions, paired
```

### Measured: the reconstructions (2026-11-23)

`ae-coco-gan` trained 40,000 steps in 39 minutes on the card (by day, the user's go-ahead). The
discriminator stayed contested throughout: right about 50–75% of the time on real patches and
55–95% on reconstructed ones, never pinned near 100% (it has won) or 50% (it learns nothing).
`ae-report`, 1,000 val photos, paired photo by photo against `ae-coco`:

| | sharpness | top 5 | distance to real | PSNR |
|---|---|---|---|---|
| the real photos | 1.00 | 33.5% | 0.041 (two halves) | |
| `ae-coco` | 0.54 | 25.0% | 0.054 | 27.75 dB |
| `ae-coco-gan` | **0.77** | 25.6% | **0.022** | 26.06 dB |

Closer to the real photo's fine detail on **992 of 1,000** photos (p 5e-282), and the price is
PSNR on every photo (1,000 of 1,000, −1.7 dB): the decoder now puts texture where the photo has
texture, not always in exactly the same pixels, and a pixel metric counts that as error twice.
The judge's top 5 did not fall, which is the check that the texture is the photo's and not
invented. The distance to the real photos fell **below** that between two halves of the real
photos: in our judge's features a set of these reconstructions is as close to the real set as
the real set is to itself.

![Real photos (top), through ae-coco (middle), through ae-coco-gan (bottom)](img/ae-coco-gan.png)

*Top: real val photos. Middle: through `ae-coco`. Bottom: through `ae-coco-gan`.* The bookcase,
the bear's fur, the grass and the letters on the stop sign come back; faces at this size do not.

**Status**: `flow-coco-gan` (the drawing model on this latent) is training on the card
(`logs/queue/sharp-1124.sh`: by day until 23:50, then the 00:30–05:30 window), then its
reports at guidance 1 and 3, each paired with `flow-coco`'s.

## Traps to keep

1. **A score that borrows a name it does not earn.** There is no FID here: FID needs a
   pretrained Inception network. When G1+ need a distribution score, it will be a distance
   in the feature space of *our own* classifier, named as such (gotcha 12).
2. **Codebook collapse is silent.** Printed every logged step (`used a/b/c/d`) and in every
   report.
3. **`ImageCaptions(val_frac=0)` still keeps one picture back** (`max(1, …)`), so the first
   reader check said "7,999 of 7,999". The reader and the held-out scores read the whole set
   as one split (`split="val", val_frac=1.0`).
4. **A report started from the browser runs on the CPU**: a button must never put work on a
   card someone else is using. The night queue passes `--device cuda` itself.
5. **A learned judge is trusted only where it agrees with an exact one.** G4b's contrastive
   model is checked against the reader on pictures whose facts are known, originals and damaged,
   before its number is read on its own. Its score is `own_clip`: our feature space, not
   CLIPScore.
6. **A contrastive loss must know which pictures say the same thing.** With 60 or 270 facts, a
   batch repeats facts; CLIP's diagonal would count those as mismatches and train anyway
   (lesson 34).

## The code, in reading order

| # | file | what to look for |
|---|---|---|
| 1 | [`pictures/reader.py`](../aksharallm/pictures/reader.py) → `read`, `judge` | foreground by distance from the background, blobs by flood fill, shape by bounding-box fill; why the reader must be exact on the originals |
| 2 | [`pictures/tokenizer.py`](../aksharallm/pictures/tokenizer.py) → `PictureConfig`, `PictureTokenizer` | the 2-D codec: encoder, `ResidualVQ` on the flattened grid, decoder; `encode`/`decode` and the codebook-prefix dial |
| 3 | [`audio/vq.py`](../aksharallm/audio/vq.py) → `ResidualVQ` | the quantizer, unchanged from audio: EMA, seeding, restart, dropout |
| 4 | [`pictures/train.py`](../aksharallm/pictures/train.py) → `main` | the trainer contract (pid, STOP, session records, resume with the data generator's state), codebook use every logged step |
| 5 | [`pictures/measure.py`](../aksharallm/pictures/measure.py) → `ladder`, `compare`, `gallery` | PSNR, the reader at each codebook count, codebook use, per-picture verdicts paired by a sign test |
| 6 | [`pictures/__main__.py`](../aksharallm/pictures/__main__.py) | `read`, `report`, `compare`, `reports` |
| 7 | [`pictures/codes.py`](../aksharallm/pictures/codes.py) → `WORDS`, `flatten`, `CodesDataset.rows` | G1's vocabulary of words and codes, position-major order, the loss on codes only, the `NULL` caption |
| 8 | [`pictures/gen.py`](../aksharallm/pictures/gen.py) → `sample`, `draw`, `PicGenTransformer` | the slot mask, classifier-free guidance over one shared KV cache, decoding through the tokenizer; the (row, column, codebook) slot embedding, taken from the cache's offset while decoding |
| 9 | [`pictures/train_gen.py`](../aksharallm/pictures/train_gen.py) → `main` | the same trainer contract; refuses codes from a different tokenizer; draws all 60 captions at each sample |
| 10 | [`pictures/score.py`](../aksharallm/pictures/score.py) → `report`, `compare` | seen vs held-out captions, copies of training pictures, variety, per-drawing verdicts paired |
| 11 | [`pictures/flow.py`](../aksharallm/pictures/flow.py) → `flow_loss`, `sample`, `FlowModel` | G2: the straight path and its velocity, Euler sampling with guidance, the transformer run bidirectionally over [time, words, patches] |
| 12 | [`pictures/train_flow.py`](../aksharallm/pictures/train_flow.py) → `evaluate`, `main` | the trainer contract again; val on fixed pictures, noise and times |
| 13 | [`pictures/latents.py`](../aksharallm/pictures/latents.py) → `picture_latent`, `encode_latents`, `decode_latent` | G3's corpus: the quantized latent, per-channel normalising from the training split, decoding with or without snapping |
| 14 | [`pictures/autoencoder.py`](../aksharallm/pictures/autoencoder.py) → `PictureAutoencoder` · [`train_ae.py`](../aksharallm/pictures/train_ae.py) | G3b's coder: the tokenizer's encoder and decoder around a Gaussian latent, the KL term, drawing from the mean; its trainer judges reconstructions with the reader |
| 15 | [`pictures/phrases.py`](../aksharallm/pictures/phrases.py) · [`pictures/text.py`](../aksharallm/pictures/text.py) | G4a: ten training phrasings and two held out, facts recovered only for the portal's free text; the frozen text model, zero padding and its cache |
| 16 | [`pictures/drawer.py`](../aksharallm/pictures/drawer.py) → `load_drawer` | one door for G1, G2 and G3: draw, score, copies through the tokenizer; `corpus_kind` picks 60 shapes or 270 scene facts |
| 17 | [`pictures/scenes.py`](../aksharallm/pictures/scenes.py) → `render`, `HELD_OUT` | G4b's second corpus: size and place in one column, two held-out combinations; the reader's `size_of` and `where_of` read them back |
| 18 | [`pictures/clip.py`](../aksharallm/pictures/clip.py) → `clip_loss`, `ClipModel`, `rank`, `agreement` · [`train_clip.py`](../aksharallm/pictures/train_clip.py) | G4b: every same-fact caption a match, padding masked out of the text head, a picture's closest caption among every caption its corpus can say, Cohen's kappa against the reader |
| 19 | [`pictures/coco.py`](../aksharallm/pictures/coco.py) → `pack`, `square` · [`pictures/photos.py`](../aksharallm/pictures/photos.py) → `score`, `frechet` | G4c: COCO packed from the archives in order with its captions; the photo report, every number beside the real photos' and chance, the Fréchet distance by two `eigh` calls |
| 20 | [`pictures/adversarial.py`](../aksharallm/pictures/adversarial.py) → `PatchDiscriminator`, `d_loss`, `g_loss`, `adaptive_weight` · [`measure.py`](../aksharallm/pictures/measure.py) → `sharpness` · [`photos.py`](../aksharallm/pictures/photos.py) → `recon_report`, `through` | sharper photos: the discriminator beside the pixel loss, its late start and adaptive weight in `train_ae.py`, and the fine-detail measure that says whether it worked |
| 21 | [`portal/pictures.py`](../aksharallm/portal/pictures.py) · [`static/js/pictures.js`](../aksharallm/portal/static/js/pictures.js) | the Pictures tab: reports with the gallery inlined (paths re-checked to sit inside `logs/pictures/`), compare, inline Draw, jobs through `portal/jobs.py` |

Tests: `tests/test_pictures.py`. They cover the reader on every count × colour × shape, the
reader seeing blur and recolouring, the codebook prefix ignoring later codebooks, batch
independence in train mode, the ladder and the paired compare, the trainer's save and
resume, and the portal refusing a gallery path outside `logs/pictures/`. Six mutants were
checked red: loose shape thresholds, swapped compare columns, a missing containment check,
BatchNorm in place of GroupNorm, a resume restarting at step 0, and a prefix decode reading
every codebook. [Lesson 29](lessons/29-pictures.md) breaks the reader.

`tests/test_picgen.py` covers G1: the flatten order and its inverse, the loss on codes only and
the dropped caption, the slot mask, guidance (a constructed model that prefers one code with a
caption and another without, so `w = 0`, `1` and `3` must each pick the right one), the paired
comparison refusing different captions, the trainer's resume and its refusal of another
tokenizer's codes, and the portal's inline draw. Six mutants were checked red: guidance halves
swapped, guidance ignoring the unconditional half, the IMG marker left in the loss, codebook-major
flattening, the wrong slot range, and the tokenizer check removed.
The slot embedding adds three more: the slot names the right cell and codebook, cached decoding
matches the full forward (red if the offset is taken as 0, if the slot is off by one, or if row
and column are swapped), and a `pos2d` checkpoint round-trips.
[Lesson 30](lessons/30-guidance.md) breaks guidance.

`tests/test_flow.py` covers G2 with exact oracles: patches cut and put back in raster order; a
velocity field that knows the picture must be integrated onto it exactly; the true velocity
scores zero loss and its negative does not; guidance must combine a model that says +0.1 with
a caption and −0.1 without in the right proportions; patch 0 must change when only the last
patch does (bidirectional); val loss is repeatable; and the drawing door and the portal take a
G2 checkpoint. Six mutants were checked red: times one step late, the target's sign flipped,
guidance halves swapped, a causal transformer, stepping backwards, and patches out of order.
[Lesson 31](lessons/31-flow-matching.md) flips the velocity.

G3 adds three tests: latents normalised on the training split decode back to the tokenizer's own
`q` (float16 storage aside); an oracle that flows to a known normalised latent must produce the
decoder's picture of the *denormalised* latent (and a latent model without its tokenizer is
refused); and a G3 run carries its statistics into the checkpoint, draws through the door as
G3, and refuses latents written by a different tokenizer. Four mutants were checked red: no
denormalising, the latent clamped like pixels, no tokenizer check, statistics not saved.
[Lesson 32](lessons/32-latent.md) skips the denormalising.

G3b adds three: the autoencoder draws from its mean in eval mode and its KL is zero at the prior;
one loader opens either coder and G3's latents come from the autoencoder's mean (with nothing to
snap to); and a G3b run scores copies with the VQ tokenizer, not the autoencoder. Three mutants
were checked red: sampling in eval mode, ignoring the autoencoder's latent, and scoring with the
autoencoder.

G4b adds `tests/test_clip.py` (13): the reader on every scene and on a wrong size and place,
a corpus that never draws its held-out combinations, scene captions round-tripping while shapes
phrasings stay G4a's, a scene report broken out by attribute and held-out combination, CLIP's
loss exactly at one match per row, two pictures of one fact never pushed apart, padding never
changing an embedding, rank, agreement and kappa, the none-fits floor (below it nothing fits,
the closed verdict kept; a quantile of clean originals; one checkpoint's floor never another's),
and the trainer's save and resume. Sixteen mutants were checked red. G4c adds
`tests/test_coco.py` (7): photos packed in order with their own captions, a photo's five
captions as one match, a padded batch encoding each caption exactly as alone (and the cache
bounded), the Fréchet distance against its closed form, the photo report's ceiling and floor,
the flow trainer reading one of a photo's own captions, and the photo autoencoder's PSNR-only
report. Eight mutants were checked red. Sharper photos add `tests/test_sharper.py` (11): sharpness
is 1 for the same photos, far below 1 for a blur, and ordered; the hinge losses point the right
way; one decoder step on the adversarial loss raises the discriminator's score on its patches;
the adaptive weight matches the two gradients and cancels a louder adversary; with the adversary
off nothing about a run changes; it starts on its step and best is chosen after it; a resume
carries the discriminator on and refuses to restart it; two reconstruction reports pair photo by
photo; both new configs are one change from their controls; the portal lists reconstruction
reports. Ten mutants were checked red (one, the rule for best, only after a test was added for
it; and the decoder-step check first passed with its sign flipped, on a flat test picture where
GroupNorm had no variance to work with).

G4a adds five: every phrasing says its facts and the held-out phrasings are never drawn for
training; a text model reads vectors and its empty caption is zeros; the text encoder pads with
zeros, truncates and caches; sentences are judged against their facts, not parsed; and two
phrasings pair by fact but refuse to pair by accident. Four mutants were checked red: held-out
phrasings in training, a NULL id for the empty caption, captions parsed instead of facts, and
the padding filled. [Lesson 33](lessons/33-held-out-phrasing.md) leaks the held-out phrasings.
