# 8. Troubleshooting

## Quick diagnosis

| symptom | most likely cause |
|---|---|
| step-0 loss ≠ `ln(vocab_size)` | broken init, or targets not shifted |
| loss flat from step 0 | LR far too low, or gradients not flowing |
| loss → NaN | LR too high; fp16 instead of bf16 |
| loss drops then explodes | LR too high, or no gradient clipping |
| val loss rises while train falls | overfitting — too many epochs |
| val loss suspiciously low | train/val contamination |
| MFU < 20% | `compile` off, batch too small, or fp32 |
| generates garbage but trains fine | **KV cache bug** |
| OOM | reduce `batch_size`, raise `grad_accum` |

---

## Training issues

### Step-0 loss is wrong

A fresh model should score `ln(vocab_size)`:

| vocab | expected loss |
|---|---|
| 8,192 | 9.01 |
| 32,768 | 10.40 |

**Much lower** → your targets aren't shifted. Verify:

```python
x, y = ds.get_batch(2)
assert (x[0, 1:] == y[0, :-1]).all()    # y is x shifted by one
```

If you accidentally pass `targets=x` (unshifted), the model can cheat via the residual
stream and tied embeddings — it predicts the *current* token, and loss starts around 6.9
instead of 9.0. (We hit this in our own first smoke test.)

**Much higher** → initialisation is wrong. Check the `std=0.02` init is being applied.

### Loss becomes NaN

In order of likelihood:

1. **LR too high.** Halve it. If a run NaNs at step 3000, resume from the last good
   checkpoint at a lower LR.
2. **Using fp16 instead of bf16.** fp16 needs a gradient scaler. Use bf16 — your 3090
   supports it.
3. **Gradient clipping disabled.** `grad_clip: 1.0` should always be on.
4. **Corrupt data** — a token id ≥ `vocab_size` indexes out of the embedding table:

```python
import numpy as np
d = np.memmap('data/fineweb/train.bin', dtype=np.uint16, mode='r')
print(d.max())     # must be < vocab_size
```

### Loss spikes then recovers

Normal. A single unusual batch. Gradient clipping is doing its job — that's what it's for.

### Loss spikes and never recovers

Stop. Resume from the last good checkpoint with a lower LR (try 0.5×). If it happens
repeatedly at the same step, you have bad data at that position in the stream.

### Loss is flat from step 0

- LR too low (check the warmup isn't longer than your whole run)
- `optimizer.step()` not being called
- `grad_norm` logging exactly `0.0` — no gradient is reaching the parameters

### Val loss looks too good

Almost always **contamination** — validation text also appears in training. Our
`prepare.py` skips the val documents when writing the train split. If you built data
yourself, verify no overlap.

---

## Memory

### CUDA out of memory

In order of preference:

1. **Lower `batch_size`, raise `grad_accum`** to keep tokens/step constant. Same math,
   less memory, marginally slower.
2. **Shorten `seq_len`.** Attention memory grows with sequence length.
3. **Turn `compile` ON.** Counter-intuitive but real: fusion removes intermediate tensors.
   We measured 6.3 GB → 3.6 GB.
4. **Gradient checkpointing** — recompute activations in the backward pass instead of
   storing them. ~60% less activation memory, ~30% slower.

### Memory grows over time

Usually a Python-side leak: something holds a reference to a tensor that's part of the
autograd graph. Log `loss.item()`, never `loss` itself:

```python
losses.append(loss.item())     # ✅ a float
losses.append(loss)            # ❌ keeps the entire graph alive
```

---

## Speed

### MFU below 20%

Check in this order:

1. `compile: true`? (1.7× on our runs)
2. bf16 autocast actually active?
3. Batch size large enough to keep the GPU fed? Check `nvidia-smi` — utilisation should
   be >90%.
4. TF32 enabled? `torch.backends.cuda.matmul.allow_tf32 = True`

### MFU above 100% / tok/s spikes on the first line after a resume

Instrumentation, not hardware — MFU over 100% is arithmetically impossible. `tok/s` divides
by the number of steps in the log window, and the first window after a resume is partial
(resume at 620, `log_every: 50`, first line at 650 = 31 steps, not 50). Fixed by measuring
the window instead of assuming `log_every`; older logs still show the inflated first line.
Ignore that one line and read the second.

### Throughput drops mid-run

- **Thermal throttling.** `nvidia-smi -q -d TEMPERATURE`. A 3090 throttles around 83°C.
- Another process on the GPU.
- Disk thrashing if your token file doesn't fit in the page cache (rare; sequential-ish
  reads are cheap).

Every step line is timestamped, so find *when* it changed rather than guessing: `grep -n
'tok/s' train_<run>.log` and look at the clock beside the drop (03:00 → a nightly backup
or cron job; a slow creep → heat). Across sessions, `scripts/sessions.py <run>` gives the
mean tok/s per launch, which is the fastest way to see "last night was 20% slower".

### A session's log disappeared

Relaunching with `> train.log` truncates it. Use `scripts/phase2.sh`, which writes
`logs/<run>/train_<timestamp>.log` per session and symlinks `train_<run>.log` to the newest;
the numbers also survive in the append-only `checkpoints/<run>/train_log.jsonl`
(`scripts/sessions.py <run>`).

### The run vanished and I don't know why

Read the end of that session's log. A clean stop always says why:

```
[stop] signal -- saving ckpt_last.pt at step 619 and exiting     # Ctrl-C, kill, or stop.sh
[stop] STOP file asked for step 2000 -- saving ...               # a queued bounded stop
[stop] reached stop step 2000 -- saving ...                      # train.stop_after/stop_at
```

No `[stop]` line at all means it died without warning — OOM (check `dmesg | tail`), a CUDA
error (in the log above the last step line), or the machine rebooted. You lose at most the
steps since the last `ckpt_every` save; relaunch and it resumes.

### It stopped at step 0 immediately after launching

A leftover `checkpoints/<run>/STOP` file. A clean stop deletes it, a `kill -9` does not.
`rm checkpoints/<run>/STOP` (or `scripts/stop.sh <run>`, which clears stale ones).

### The first step takes forever

That's `torch.compile`, ~60s. Only once per run. Set `compile: false` while debugging.

---

## Generation issues

### Trains fine, generates garbage

**This is almost always the KV cache.** Run the test:

```bash
python -m pytest tests/test_model.py::test_kv_cache_matches_full_forward -q
```

It asserts cached decoding reproduces a full forward pass exactly. The classic bug is
passing `is_causal=True` during single-token decode, which masks away the entire context.

Second suspect: a tokenizer mismatch. If the checkpoint was trained with a different
`tokenizer.json` than you're loading, every id means something else. Output will be
fluent-looking nonsense.

### It repeats itself endlessly

```
The cat sat on the mat. The cat sat on the mat. The cat sat on the mat.
```

- Temperature too low (0 is greedy and loops readily). Try 0.8.
- Add `--repetition-penalty 1.1`.
- Undertrained model — this is normal early in a run.

### It stops immediately

Generating EOS as the first token. Either the model is very undertrained, or your prompt
is being encoded with an unexpected BOS. Check `tok.encode(prompt, bos=True)`.

### Output is incoherent at high temperature

Working as intended. Lower to 0.7–0.8.

---

## Data pipeline issues

### `train.bin` is 0 bytes (or much smaller than expected)

We hit this one. The cause was feeding a lazy HuggingFace stream into `pool.imap`, which
iterates it on multiprocessing's internal thread — exceptions there are swallowed and the
iterator silently ends. Fixed by iterating on the main thread with `apply_async`, plus a
hard check that raises if 0 tokens were written.

**Always verify file sizes before a long run:**

```bash
ls -la data/fineweb/
python -c "print(open('data/fineweb/train.bin','rb').seek(0,2)//2, 'tokens')"
```

### The prep script hangs at the end

The HF streaming reader stalls when draining a dataset to completion. Pass
`--max-train-tokens` with a value it will actually reach, so it terminates on the cap
instead of waiting for the stream to end.

### Segfault / "PyGILState_Release" at exit

Cosmetic — the `datasets` streaming HTTP thread dying during interpreter shutdown, after
all data is written. We call `os._exit(0)` after flushing to skip finalization entirely.
If you see this in your own script, your data is fine; check the file sizes.

---

## Sanity checklist before a multi-day run

```bash
# 1. tests pass
python -m pytest tests/ -q

# 2. data is the expected size
ls -la data/fineweb/

# 3. no token id exceeds the vocab
python -c "
import numpy as np
d = np.memmap('data/fineweb/train.bin', dtype=np.uint16, mode='r')
print('tokens:', len(d), 'max id:', d.max())"

# 4. 50-step smoke test (throwaway dir so it can't pollute the real run's resume:auto)
python -m aksharallm.train.pretrain configs/small.yaml \
    -o train.max_steps=50 -o train.out_dir=/tmp/aksharallm_smoke -o train.resume=null

# 5. confirm resume works — run the REAL config twice; the second should say "resumed from ..."
#    (cheapest version: STOP_AFTER=5 scripts/phase2.sh, twice — the second must start at 5)
```

Five minutes here saves six days.
