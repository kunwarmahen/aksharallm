---
id: data
title: Text becomes numbers on disk
doc: docs/02-data.md
files:
  - aksharallm/data/prepare.py
  - aksharallm/data/loader.py
verify: tests/test_pipeline.py::test_loader_shift_and_bounds
prereqs: []
minutes: 25
summary: How a corpus becomes a flat file of 16-bit numbers, and why the training batch is the same array twice, one position apart.
---

# 1. Text becomes numbers on disk

A language model never sees text. Before any training happens, a corpus is turned into a
long list of integers and written to a single flat file — `train.bin` — with **no structure
at all**: no records, no delimiters, no index. Just numbers, end to end.

```
"Once upon a time"  ->  [7594, 3011, 261, 1216]  ->  train.bin
```

Two bytes per number (`uint16`), which is why the vocabulary is capped at 65,536. Ours is
32,768, so every token fits with room to spare.

## Why a flat file and nothing cleverer

Because training reads it *randomly*, forever, and never changes it. `np.memmap` hands the
file to the operating system's page cache and reads become pointer arithmetic — no parsing,
no decompression, no dataloader workers, no format to maintain. A 20 GB corpus is opened
instantly on a machine with 62 GB of RAM because nothing is loaded until it is touched.

## The batch is one array, twice

This is the part worth having in your fingers. To train "predict the next token", you need
inputs and targets. They are **the same slice of the file, one position apart**:

```
tokens   [ 7594, 3011,  261, 1216,  982 ]
x        [ 7594, 3011,  261, 1216 ]        <- what the model sees
y        [       3011,  261, 1216,  982 ]  <- what it should predict
```

Every position is a training example: given `7594`, predict `3011`; given `7594, 3011`,
predict `261`. A window of 512 tokens is 512 examples, not one — which is why this is such
an efficient way to learn.

Read `docs/02-data.md`, then open [`aksharallm/data/loader.py`](aksharallm/data/loader.py)
in the **Code** tab and find where `x` and `y` are cut from the memmap.

---

## Exercise: break the shift

The whole thing hangs on that one-position offset. Get it wrong and the model learns to
predict the token it was just given — a task it can solve perfectly, at which point the loss
falls beautifully and the model is worthless.

1. Run the check below. It passes.
2. In `aksharallm/data/loader.py`, find where the target slice is taken and make it start at
   the same place as the input instead of one later.
3. Run the check again. **It should fail** — read what it says about the values it expected.
4. Put it back. Run the check. Green.

> **What you just saw.** The failure is loud here because a test asserts the exact
> relationship. In a training run it would not be: the loss would drop *faster* than normal,
> which looks like good news.

## The bug this file actually had

Worse than an off-by-one, and it happened here. The tokenizer ran across several processes
with `pool.imap`, which iterates the input generator on multiprocessing's internal
task-handler thread. The HuggingFace streaming reader raised there, the exception was
**swallowed**, and the run finished with an empty `train.bin` and **exit code 0**.

Nothing failed. There was simply no data. The fix is in `prepare.py`: iterate on the main
thread with bounded `apply_async`, and hard-check for zero tokens before writing anything.

The lesson generalises past this repo: *a pipeline that reports success is not the same as a
pipeline that produced output.* Check the size of what you made.

```bash
ls -la data/*/train.bin        # bytes / 2 = tokens. Is that the number you expected?
```
