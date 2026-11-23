---
id: contrastive
title: A judge that learned, and the matches its loss must count
doc: docs/26-pictures.md
files:
  - aksharallm/pictures/clip.py
  - aksharallm/pictures/train_clip.py
  - aksharallm/pictures/scenes.py
verify: tests/test_clip.py::test_two_pictures_of_one_fact_are_not_pushed_apart
prereqs: [held-out-phrasing]
minutes: 25
summary: G4b trains our own contrastive picture-caption model, the kind of judge real image models are scored with, and checks it against the rule-based reader. Its loss has to know that several pictures in a batch say the same thing. Give it CLIP's diagonal instead, and it learns to tell apart pictures that should match, while training looks perfectly healthy.
---

# 34. A judge that learned

Every drawing so far was judged by the rule-based reader ([lesson 29](docs/lessons/29-pictures.md)),
which works only because we drew the corpus ourselves. Real text-to-image models are judged by a
**contrastive** model that has learned which captions go with which pictures (OpenAI's CLIP, and
the scores built on it). G4b trains one of our own ([doc 26](docs/26-pictures.md) § G4b):
docs/22's vision tower reads the picture, the frozen `tiny` reads the caption, and each side ends
in a 128-number unit vector. A picture and its caption should point the same way.

## The loss, and what it counts as a match

CLIP's loss looks at a batch of pictures and their captions and asks two questions: for each
picture, which caption is its own? For each caption, which picture? With one match per row
(picture *i* goes with caption *i*) both questions are a cross-entropy over the batch.

Our corpus has 60 shapes facts and 270 scene facts. A batch of 128 pictures holds the same fact
several times, in different phrasings. CLIP's diagonal would call "two red circles" and "there
are two red circles" a **mismatch** whenever they sit on different rows, and push the two
pictures apart. `clip.clip_loss` takes a `same` matrix (1 where two rows state the same fact)
and spreads each row's target over all of its matches. With one match per row it is CLIP's loss
exactly (`test_one_positive_per_row_is_clips_loss_exactly`).

The check compares two arrangements of six pictures, two facts with three pictures each: one
where every picture of a fact sits at the same point, and one where every picture has its own.
The right loss must prefer the first.

---

## Exercise: give it CLIP's diagonal

1. Run the check. Green.
2. In `aksharallm/pictures/clip.py`, inside `clip_loss`, change
   `same = same.to(logits.dtype)` to `same = torch.eye(len(same)).to(logits.dtype)`.
3. Run the check. **It should fail.**
4. Put it back. Green.

> **What you just saw.** With the diagonal, training still runs and the loss still falls: the
> model can always lower it by learning to tell two pictures of two red circles apart, by their
> exact positions. That is effort spent on the opposite of what a judge needs. Nothing in the
> loss curve says so. The check says so, because it asks which arrangement each loss prefers.

## Run it yourself

```bash
python -m aksharallm.pictures scenes                     # the scenes corpus, size and place too
scripts/audio.sh clip-shapes                             # or on the CPU: CUDA_VISIBLE_DEVICES=
python -m aksharallm.pictures clip-report clip-shapes    # beside the reader, on known pictures
python -m aksharallm.pictures clip-report clip-shapes --through ae-shapes   # damaged pictures
python -m aksharallm.pictures gen-report imggen-shapes-gpu --clip clip-shapes
```

Or open the **Pictures** tab: "A judge that learned" lists the reports, and "Also judged by"
adds our model's verdict to a drawing report.
