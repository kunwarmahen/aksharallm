---
id: speculative
title: Guessing ahead, and the arithmetic that makes it free
doc: docs/07-inference.md
files:
  - aksharallm/infer/speculative.py
verify: tests/test_speculative.py::test_greedy_speculative_decoding_is_token_for_token_the_target_alone
prereqs: [sampling]
minutes: 30
summary: A second, worse model writes the next few tokens and the good one checks them — and the output is provably the good model's, not a compromise.
---

# 16. Guessing ahead, and the arithmetic that makes it free

Generating one token means moving every weight in the model from memory to the chip. For a
300M model that is ~600 MB of traffic to produce **two bytes**. The arithmetic is nothing;
the memory bandwidth is everything. Which means a forward pass over *eight* positions costs
almost exactly what a forward pass over one costs.

Speculative decoding sells that spare capacity. Something cheap guesses the next few tokens;
the real model checks all of them in a single pass, keeps the prefix it agrees with, and
throws the rest away.

```
draft:   the cat sat on the mat        (cheap, probably right about "the", "cat")
target:  ────✓────✓────✓────✗          one forward pass over all six
emit:    the cat sat  +  its own correction
```

## The part that sounds too good to be true

The output is **not** an approximation of the big model. It is exactly what the big model
would have produced on its own — same tokens, same distribution — and that is a theorem, not
a hope.

Accept a drafted token `x` with probability `min(1, p(x)/q(x))`, where `p` is the target's
distribution and `q` the draft's. On rejection, emit a sample from `norm(max(p - q, 0))`.
Then:

```
P(emit x) = q(x)·min(1, p(x)/q(x))  +  P(reject)·norm(max(p-q,0))(x)
          = min(q(x), p(x))         +  max(p(x) - q(x), 0)
          = p(x)
```

Both terms are needed. Rejecting and falling back to a plain sample from `p` would
double-count the mass the draft already got right, and bias the output towards tokens both
models happen to like — a bias no test of "does the text look fine?" would ever catch.

## Measured here

Our 300M, greedy, output verified identical to the unassisted model:

| gamma (tokens guessed per round) | speedup |
|---|---|
| 2 | 1.43x |
| 4 | 1.56x |
| 8 | **2.01x** |

The drafter shipped is **model-free** — an n-gram lookup in the text so far. The obvious
choice, our trained 13.8M model, is *refused*: it has an 8k TinyStories vocabulary against
the 300M's 32k blend vocabulary, so token id 4,001 means different strings to the two models
and the acceptance test would be comparing nonsense. Same argument as cross-tokenizer
distillation, and it is a hard refusal rather than a warning.

---

## Exercise: accept everything

1. Run the check. It passes: with greedy decoding, speculative and ordinary generation emit
   token for token the same sequence.
2. In `aksharallm/infer/speculative.py`, find `accept_or_correct` — the rule in one place.
   Make it always accept: return `True, None` before it compares anything.
3. Run the check. **It should fail.** Every guess the draft made is now in the output,
   including the ones the target disagreed with, so the two sequences diverge at the first
   mistake.
4. Put it back. Green.

> **What you just saw.** The failure mode of a broken acceptance rule is not a crash and not
> gibberish — it is *the draft model's text, delivered at the draft model's quality, while
> your logs still say you are running the 300M*. The only thing standing between those two
> outcomes is one comparison, which is why it lives in its own function with its own test.

## Where it actually pays

Inside a batch. `BatchEngine(speculate=N)` took our server from 238 to **372 tokens/s** at
batch 32 — 52.5 tokens per model pass, 39% of guesses accepted — because the ragged batch
step already handles rows of uneven length and paging makes a rejected token free: it simply
sits past the sequence's `cached` mark and is never read. See [lesson 17](docs/lessons/17-serving.md).

One trap that comes with it: a round emits several tokens at once, so `max_tokens` and the
stop token have to be clipped **inside** the round. Ask for 16 and you get 17 otherwise.
