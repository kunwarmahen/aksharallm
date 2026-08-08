---
id: eval
title: Is it any good? — and why the loss cannot say
doc: docs/13-eval.md
files:
  - aksharallm/eval/scoring.py
  - aksharallm/eval/suites.py
verify: tests/test_eval.py::test_a_real_bpe_merge_across_the_boundary_does_not_shift_the_score
prereqs: [sampling]
minutes: 35
summary: How a multiple-choice benchmark is actually scored, why 25% on MMLU is not a failure, and a tokenizer trap that is invisible in the accuracy.
---

# 11. Is it any good? — and why the loss cannot say

Every number so far has been a **loss**. Loss is the right thing to watch while training —
smooth, cheap, moves every hundred steps — and the wrong thing to trust, for a reason that is
easy to say and easy to forget:

> Cross-entropy measures how well the model predicts the next token of a corpus.
> Nothing you want the model to do is that.

It matters most exactly when you start making trade-offs. Does int4 quantization cost
anything a person would notice? Did the synthetic data help? Perplexity cannot answer either.

## How a multiple-choice question is actually scored

Not by asking the model to output "B". A small base model has never been told what that
means. Instead, each option is scored as a **continuation**:

```
context:  "The capital of France is"
option A: " Berlin"   -> sum of log P(each token)  =  -8.4
option B: " Paris"    -> sum of log P(each token)  =  -2.1   <- highest wins
```

No generation, no parsing, no format sensitivity — it works on a model that cannot follow
instructions at all, which is the whole point at this stage.

## Read the chance line, always

Four-way multiple choice pays **25%** for guessing. A model scoring 25% on MMLU has learned
nothing about MMLU — and it has not failed, either, because a 300M model was never going to.
PIQA is two-way: chance is 50%. A score is only interesting when it clears chance **by more
than its own error bar**, which is why the harness prints one and the portal only colours a
number that does.

Our Phase 2 model at step 18,000: ARC-Easy **46.7% ± 6.4** against 25% chance. That is real.

---

## Exercise: tokenize the continuation with the context

This trap is worth more than it looks.

To score just the continuation, you need to know which tokens are the continuation. The
obvious approach — encode `context + continuation`, then count backwards by the length of
`encode(continuation)` — is **wrong**, because BPE merges across the join. If the last token
of the context and the first of the continuation merge into one token, the count is off by
one and you have scored a token of the context as part of the answer.

The damage is subtle: it shifts **every** option's score, so accuracy barely moves. It is
invisible in the number and wrong in every number.

1. Run the check. It passes.
2. In `aksharallm/eval/scoring.py`, find where the continuation's token count is determined
   and change it to encode the two together and subtract, instead of encoding the
   continuation on its own against the context.
3. Run the check. **It should fail**, on the case where a merge crosses the boundary.
4. Put it back. Green.

> **What you just saw.** A scoring bug does not fail — it produces a plausible number, wrong
> in a consistent direction, forever. That is why the harness has a test comparing
> log-likelihood against a hand-computed value: the only defence against a number that looks
> right is to know what it should be.
>
> **And a second lesson, from writing this one.** The check for this lesson had to be
> *added*. The obvious existing test asserts the same property — but against a fake
> tokenizer that is one byte per token, so it has no merges, so both implementations look
> identical to it and the break above left it green. A test that cannot fail is not a test.
> The new one trains a real BPE tokenizer, picks a pair that provably merges across the
> boundary, and asserts *that* first — so if the merge ever stops happening, the test says so
> instead of passing for the wrong reason.

## Run it

```bash
python -m aksharallm.eval tiny --suite fast --limit 200
python -m aksharallm.eval report --suite arc-easy      # every score, across training steps
```

Or the portal's **Eval** tab, which leads with the trend chart rather than the Run button —
because one benchmark score in isolation is close to meaningless, and the same suite across
ten checkpoints is not.
