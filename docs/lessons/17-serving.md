---
id: serving
title: Thirty conversations, one pass over the weights
doc: docs/16-serving.md
files:
  - aksharallm/serve/paged.py
  - aksharallm/serve/batch.py
verify: tests/test_serve.py::test_the_paged_path_produces_the_same_logits_as_a_contiguous_cache
prereqs: [speculative]
minutes: 35
summary: Turning a checkpoint into something other people can use — where the memory actually goes, and the two bugs that made a working model look broken.
---

# 17. Thirty conversations, one pass over the weights

[Lesson 4](docs/lessons/04-kv-cache.md) gave every sequence a cache sized for the **whole** context
window, allocated up front. That is right for one conversation and hopeless for thirty: a
1,024-token window reserved for a request that turns out to be 40 tokens long wastes 96% of
what it took, and the server runs out of memory while the card sits mostly empty.

Two ideas fix it, and neither touches the model.

## Paging: memory bounded by what is used

Cut the cache into fixed **blocks** of 16 tokens. A sequence owns a list of block ids — its
*block table* — and grows by taking one more block when it fills the last. Nothing is
contiguous and nothing is reserved.

```
seq A  ▸ blocks [7, 2, 9]        tokens 0..47
seq B  ▸ blocks [7, 2, 4, 1]     tokens 0..63   ← shares 7 and 2 with A: same prompt prefix
```

Two sequences that begin with the same system prompt can **point at the same blocks**, so
that prefix is stored once and computed once. Blocks are reference-counted; the last holder
to finish gives them back.

## Continuous batching: one step, many sequences

A step prefills whoever just arrived and decodes everyone already running, in the same
ragged pass over the weights. A sequence that finishes leaves on the step it finishes and
its blocks are free immediately — nobody waits for the slowest member of a batch.

Measured on our 300M, 64 tokens per request, greedy:

| batch | tokens/s |
|---|---|
| 1 | 50 |
| 8 | 134 |
| 32 | 236 |
| 64 | **272** |

Read the trade honestly: **per-request latency rises with batch size.** Batching wins
throughput, not responsiveness, which is exactly why the Playground does not batch.

---

## Exercise: break the addressing

1. Run the check. It passes: a paged cache produces the same logits, to floating-point
   tolerance, as the contiguous cache from lesson 4. That equivalence is the whole claim —
   paging is an *addressing* change, not a modelling one.
2. In `aksharallm/serve/paged.py`, find `BlockTable.write`. It computes `slots` and writes
   `pool_k[:, slots] = k`. Change the write to land at the wrong place — for example write
   to `slots` shifted by one (`pool_k[:, [s + 1 for s in slots]] = k`).
3. Run the check. **It should fail**: every key is now filed under its neighbour's address,
   so attention reads the wrong past.
4. Put it back. Green.

> **What you just saw.** Paging is bookkeeping, and bookkeeping fails quietly. The real
> version of this bug in this repo was worse than the one you just made: the pool was shaped
> `(n_blocks, n_kv_heads, ...)`, whose natural view is `pool.transpose(0,1).reshape(...)` —
> and **reshape on a transposed tensor returns a copy**. Every single write went into a
> temporary. The pool stayed full of zeros, the model attended to nothing but the token it
> had just been handed, and it repeated that token forever. It looked exactly like an
> undertrained model. The fix was to choose a shape whose slot view needs no copy.

## The other bug worth knowing

Gathering blocks **per sequence** meant 768 indexing operations per step at 24 layers × 32
sequences, which cost more than batching saved — batch 8 ran at 54 tokens/s, barely better
than one at a time. Building one address table for the whole batch took it to 134. If a
batched implementation is not faster than the unbatched one, look for a loop over sequences
before you look at the model.

## A test lesson, which cost real time twice

An **untrained** model's output barely depends on its input. So a test that checks
positions, masks or RoPE against a randomly initialised model passes even when those things
are wrong — the outputs are near-identical either way. The test model in `tests/test_serve.py`
is deliberately trained for about two seconds first. Without that, mutating the position
handling left every test green.

```bash
scripts/serve.sh tiny            # or the portal's Serve panel
```
