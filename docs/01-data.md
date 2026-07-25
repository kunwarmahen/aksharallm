# 1. Data

> **The rule:** at fixed compute, better data beats a better architecture, every time.
> This is the stage where a hobbyist has the most leverage, and it's the stage most
> tutorials skip.

## Where text comes from

| Dataset | Size | What it is | Good for |
|---|---|---|---|
| **TinyStories** | 0.5B tok | Synthetic children's stories, ~1500-word vocabulary | Phase 1. Tiny models can actually master it. |
| **FineWeb-Edu** | 1.3T tok | CommonCrawl filtered by a classifier for "educational value" | Phase 2. The best open pretraining data per token. |
| FineWeb | 15T tok | CommonCrawl, deduplicated, lightly filtered | When you need volume over quality |
| The Stack v2 | 900B tok | Permissively-licensed source code | Code models |
| Wikipedia | 4B tok | Encyclopedia | Facts, but too small alone |

We use TinyStories then FineWeb-Edu. The recipes live in
[`aksharallm/data/prepare.py`](../aksharallm/data/prepare.py):

```python
RECIPES = {
    "tinystories":      ("roneneldan/TinyStories",   None,           "text"),
    "fineweb-edu-10bt": ("HuggingFaceFW/fineweb-edu", "sample-10BT", "text"),
}
```

Adding your own is one line — any HuggingFace dataset with a text column works.

### Why TinyStories first?

A 13.8M-parameter model does not have the capacity to learn all of English. Trained on
web text it produces plausible-looking gibberish. Trained on TinyStories — which
deliberately uses only words a 4-year-old knows — it produces genuinely coherent prose,
because the task is small enough to actually fit in the model.

This makes it perfect for validating a pipeline: if your code is correct, you *will* see
coherent output in 25 minutes. If you don't, something is broken, and you've learned that
cheaply instead of six days in.

---

## Quality filtering

Raw CommonCrawl is mostly SEO spam, boilerplate navigation, and adult content. The
standard pipeline:

```mermaid
flowchart TD
    A[raw HTML crawl] --> B[extract text, drop markup]
    B --> C[language filter: keep English]
    C --> D[heuristics: drop pages with too few words,<br/>too many symbols, no punctuation]
    D --> E[dedup: near-identical pages appear thousands of times]
    E --> F[quality classifier: is this educational?]
    F --> G[final corpus]
```

**Deduplication matters more than people expect.** The same article syndicated across
5,000 sites means the model sees it 5,000 times, memorises it verbatim, and wastes
capacity. FineWeb-Edu has all of this done already, which is why we use it rather than
building our own crawl — that's a project in itself.

---

## The on-disk format

After tokenization, the entire corpus is **one flat file of unsigned 16-bit integers**:

```
train.bin:  [doc1 tokens] 0 [doc2 tokens] 0 [doc3 tokens] 0 ...
                              ↑
                    <|endoftext|> separator
```

No headers. No record boundaries. No JSON. Just numbers.

**Why `uint16`?** Our vocabulary is ≤ 65,536, so every token fits in 2 bytes. 10 billion
tokens = 20 GB. In `int32` it would be 40 GB, for zero benefit.

**Why one flat file?** Because training doesn't want documents, it wants *random windows*:

```python
i = random.randint(0, n_tokens - seq_len - 1)
x = data[i     : i + seq_len]        # what the model sees
y = data[i + 1 : i + seq_len + 1]    # what it must predict
```

`x` and `y` are the same slice, offset by one. That single-token shift **is** the
pretraining objective — position *t* sees `x[t]` and must produce `y[t] == x[t+1]`.

```
data:   [ 791, 6864,  315, 9822,  374, 12366 ]
             ↓     ↓     ↓     ↓     ↓
x:      [ 791, 6864,  315, 9822,  374 ]
y:      [ 6864, 315, 9822,  374, 12366 ]
```

A window will often straddle a document boundary. That's fine, and actually useful: it
teaches the model what `<|endoftext|>` means — "topic over, start something new".

**Why `np.memmap`?** The file stays on disk; the OS pages in only the bytes you touch and
caches them in free RAM. A 20 GB corpus works on a machine with 8 GB of RAM, with no
DataLoader, no worker processes, and no collate function. See
[`loader.py`](../aksharallm/data/loader.py) — it's 60 lines.

---

## Train/validation split

The validation set is carved from the *start* of the stream, and training skips those
documents:

```python
# doc 0 .. 50,000        -> validation
# doc 50,000 .. end      -> training
```

This matters. If a validation document also appears in training, the model has memorised
it and your val loss becomes a lie — it will keep dropping while the model gets no better.
That's **contamination**, and it's the most common way people fool themselves.

---

## Running it

```bash
python -m aksharallm.data.prepare tinystories \
    --out-dir data/tinystories \
    --vocab-size 8192 \
    --max-train-tokens 400000000
```

Output:
```
[1/3] training 8192-token BPE on 200,000 docs
[2/3] writing validation split (5,000,000 tokens)
[3/3] writing train split
done. train=400,000,000 tok  val=5,000,000 tok  (0.81 GB on disk)
```

Roughly 90 seconds on a 16-core machine — tokenization is CPU-parallel across processes.

### Options worth knowing

| Flag | Why you'd use it |
|---|---|
| `--max-train-tokens N` | Disk budget. Also makes the run terminate deterministically instead of draining a slow stream. |
| `--vocab-size N` | 8k for tiny models, 32k for Phase 2. See [doc 2](02-tokenizer.md). |
| `--tokenizer path` | Reuse an existing tokenizer instead of fitting a new one. **Required** if you're adding data to an existing model. |
| `--n-proc N` | Tokenizer processes. Defaults to cores − 2. |

> ⚠️ **Never change the tokenizer without re-tokenizing everything and retraining from
> scratch.** Token id 5051 means a different string under a different tokenizer, and the
> model's embedding table is indexed by id. Mixing them produces fluent nonsense.

---

## Disk budget

| Corpus | Tokens | `train.bin` |
|---|---|---|
| TinyStories | 400M | 0.8 GB |
| FineWeb-Edu `sample-10BT` | 10B | 20 GB |
| FineWeb-Edu `sample-100BT` | 100B | 200 GB |

Check `df -h` before starting Phase 2. Tokenization streams (raw text is never all held
at once) but the output still has to land somewhere.

---

## An implementation detail that cost real debugging time

The tokenizer runs in a process pool. The obvious way to feed it is:

```python
for arr in pool.imap(tokenize, batches):   # ❌ silently truncates
```

But `imap` iterates `batches` on multiprocessing's internal task-handler **thread**, and
the HuggingFace streaming reader raises from a non-main thread. That exception is
swallowed — `imap` just ends early, and you get a **truncated or empty dataset with a
zero exit code**. We hit exactly this and wrote 0 tokens while the script reported success.

The fix is to keep stream iteration on the main thread and dispatch explicitly:

```python
for batch in batches:                       # ✅ errors propagate
    pending.append(pool.apply_async(tokenize, (batch,)))
    if len(pending) >= max_pending:
        write(pending.popleft().get())
```

Plus a hard check at the end: if 0 tokens were written, raise. **Always verify your data
files are the size you expect before starting a multi-day run.**

---

Next: [2. Tokenizer →](02-tokenizer.md)
