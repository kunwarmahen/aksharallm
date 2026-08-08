---
id: sft-mask
title: Teaching it to answer, not to continue
doc: docs/06-posttraining.md
files:
  - aksharallm/tokenizer/tokenizer.py
  - aksharallm/data/prepare_sft.py
verify: tests/test_pipeline.py::test_chat_mask_covers_only_assistant_content
prereqs: [training-loop]
minutes: 30
summary: Pretraining and fine-tuning differ in exactly one thing — which tokens count towards the loss — and getting that mask wrong teaches the model to ask questions.
---

# 8. Teaching it to answer, not to continue

A base model is a text *continuer*. Ask it "What is the capital of France?" and a good one
may well reply "What is the capital of Germany? What is the capital of Spain?" — because on
the internet, questions are usually followed by more questions. It is not broken. It is doing
exactly what it was trained to do.

Supervised fine-tuning changes the behaviour, and it differs from pretraining in **one
thing**:

```
pretrain:  every token is a target.        "learn what text looks like"
SFT:       only the assistant's tokens.    "learn what an answer looks like"
```

Same model, same loop, same optimiser. What changes is a parallel array of 0s and 1s.

```
tokens  <|im_start|>user \n What is 2+2? <|im_end|> <|im_start|>assistant \n Four <|im_end|>
mask         0      0   0  0   0  0  0  0     0          0          0     0   1   1
             \___________ context, not trained on ___________/        \_ trained on _/
```

Train on the user's tokens too and you get a model that has learned to write plausible
*questions* — which is what it will do when you talk to it.

## Packing, and why the boundary is fine

Examples are packed end to end into fixed windows rather than padded. Padding a 1,024-token
window to hold a 60-token exchange wastes 94% of the compute. The cost is that one window can
contain the end of one conversation and the start of the next — which is a boundary the model
needs to learn anyway.

---

## Exercise: train on the question

1. Run the check. It passes — it renders a chat and asserts the mask covers the assistant's
   content and nothing else.
2. In `aksharallm/tokenizer/tokenizer.py`, find `render_chat` and make the user turn's tokens
   trainable as well (mark them `1`).
3. Run the check. **It should fail**, naming positions that should not be trained.
4. Put it back. Green.

> **What you just saw.** The mask is data, not code — nothing type-checks it, nothing crashes,
> and a run with a wrong mask trains smoothly to a lower loss than the correct one, because
> predicting the user's next word is an easier problem than answering. Lower loss, worse
> model.

## The number to check on real data

`prepare_sft` prints the trainable fraction:

```
trained on 27,832,411 tokens (56.2% of the total)
```

Expect **30–50%** for a chat corpus. Much lower means the conversations are mostly user text;
much higher means the mask is wrong. Our SmolTalk set is 56.2%, which is high-ish and
explained by long assistant answers. That one printed line is the cheapest check that the
most invisible part of post-training is behaving.
