# 9. Troubleshooting

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
| a portal stage flashes "running", then "ready" | the trainer died on startup — read the log named in `checkpoints/<run>/run.meta` |
| SFT OOMs on a model whose pretraining fit | SFT's defaults are its own, not the model's YAML — lower `BS=`, raise `ACCUM=` |

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

**Post-training is not exempt.** SFT and DPO train every weight, so they need the same
memory per micro-batch as pretraining — but they take their batch size from `sft.py`'s own
defaults (`16 × 4`), not from `configs/<run>.yaml`. A model whose pretraining you tuned to
fit will still OOM under SFT if you never set it. Use `BS=` / `ACCUM=` on
`scripts/stage.sh` and keep their product constant; see
[doc 6](06-posttraining.md#hyperparameters--and-why-they-differ-from-pretraining).

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
[stop] reached stop time 06:30 -- saving ...                     # stop.sh --in / --by
[stop] reached stop step 2000 -- saving ...                      # train.stop_after/stop_at
[stop] reached this session's 30m00s time budget -- saving ...   # train.stop_after_s / STOP_IN
```

Each names which of the five it was, because they send you to different places: a `stop
time` you did not set means someone (or the portal) queued one; a `time budget` means the
*launch* was bounded and the next one will be too unless you drop `STOP_IN`.

No `[stop]` line at all means it died without warning — OOM (check `dmesg | tail`), a CUDA
error (in the log above the last step line), or the machine rebooted. You lose at most the
steps since the last `ckpt_every` save; relaunch and it resumes.

### It stopped at step 0 immediately after launching

A leftover `checkpoints/<run>/STOP` file. A clean stop deletes it, a `kill -9` does not.
`rm checkpoints/<run>/STOP` (or `scripts/stop.sh <run>`, which clears stale ones).

`scripts/stop.sh <run> --status` prints what a queued stop is asking for in words (`stop at
06:30`, `stop after step 20000`), which is worth checking before assuming the file is stale
— a deadline that has already passed looks identical to a leftover until you read it.

The same trap exists for fine-tunes and QAT, and is handled the same way: their stop files
are `logs/finetune/STOP` and `logs/quant/STOP`, cleared when a job starts.

### It stopped sooner (or later) than the time I asked for

It stops at the **first step boundary past the deadline**, so a slow step overshoots by up
to one step — seconds on a small model, ~20s on a 300M one. It never stops early: the
deadline is compared against the clock, not estimated from throughput.

If it stopped much later, check that anything is running at all — a deadline in the STOP
file only fires while the trainer is polling it. A stop queued against a run that has
already exited sits in the file and ends the *next* launch immediately (see above).

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

## The portal

### `cannot bind 127.0.0.1:8765`

Another portal is already running (they are harmless, but only one can hold the port):
`scripts/portal.sh --port 8766`, or stop the other one. Nothing about a training run is
affected either way — the portal holds no state.

### The Start button is greyed out

Either that run is already training (the badge says so), or it has no launcher.
`scripts/phase2.sh` knows how to build `small-code` and `small`; any other run is read-only
in the portal and must be started from a terminal. Hover the button for the reason.

### I pressed Start and the phase says `pre-flight` for ten minutes

That is `phase2.sh` doing its job: tests, disk check, data check, then a 50-step smoke test
before the real run. The log panel is streaming that launcher log; the phase turns to
`training` when the trainer's pid appears. On a resume you can tick **skip smoke test**
(`SKIP_SMOKE=1`), which is honoured only when `ckpt_last.pt` exists.

### A stop said "queued: pid NNN will finish step N" but nothing stopped

Check whether that pid was the **smoke test**: `ps -p NNN -o args=`. During pre-flight,
`phase2.sh` runs the identical trainer command with `-o train.out_dir=/tmp/aksharallm_smoke`,
so anything identifying a run by command line alone finds it — and a STOP file written to
`checkpoints/<run>/` is never read by a process whose `out_dir` is `/tmp`. Fixed by the
trainer writing `train.pid` into its own `out_dir` and by anchoring the command-line
fallback; if you see this again, the pid file is stale and something is matching too loosely.

### The portal says idle but `nvidia-smi` shows a busy GPU

Something is training that this run's `train.pid` doesn't point at — a hand-launched trainer
for a different config, or another program. `scripts/stop.sh <run> --status` and
`nvidia-smi` together will say which. The portal deliberately refuses to signal a pid whose
command line isn't an `aksharallm` trainer.

### Closing the portal / the browser

Neither stops training. The trainer is detached (`nohup`, its own session), the portal is a
reader, and the buttons work by writing the same `STOP` file `scripts/stop.sh` writes.

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

---

## The code, in reading order

This chapter has no order of its own — a symptom sends you straight to one place. So this
table is by symptom, and each row is the first file to open:

| symptom | open | what to look at |
|---|---|---|
| step-0 loss is wrong | [`aksharallm/data/loader.py`](../aksharallm/data/loader.py) → [`model/transformer.py`](../aksharallm/model/transformer.py) | `get_batch`'s one-position shift; then `_init_weights` and the `0.02/√(2·n_layers)` scaling |
| NaN, spikes, or a flat loss | [`aksharallm/train/pretrain.py`](../aksharallm/train/pretrain.py) · [`train/schedule.py`](../aksharallm/train/schedule.py) | the `ctx` autocast (bf16, no scaler), `clip_grad_norm_`, and `get_lr` — check the warmup is not longer than the run |
| OOM, or MFU below 20% | [`aksharallm/config.py`](../aksharallm/config.py) · `pretrain.py` | `TrainConfig.batch_size` / `grad_accum` / `seq_len` / `compile`, and `estimate_mfu` in `transformer.py` for what the number means |
| MFU above 100% after a resume | `pretrain.py`, the logging block | `step - prev_log_step` — the window is measured, not assumed. Instrumentation, not hardware |
| trains fine, generates garbage | [`aksharallm/model/transformer.py`](../aksharallm/model/transformer.py) | `is_causal = cache is None or T > 1` in `Attention.forward`. Second suspect: the tokenizer path in the checkpoint |
| it repeats, or stops immediately | [`aksharallm/infer/generate.py`](../aksharallm/infer/generate.py) | `_filter_logits` and the repetition-penalty sign handling; then whether EOS is being encoded into the prompt |
| `train.bin` is empty or short | [`aksharallm/data/prepare.py`](../aksharallm/data/prepare.py) | `tokenize_to_bin` — the `apply_async` loop and the zero-token hard check |
| a run vanished, or stopped at step 0 | [`aksharallm/train/stopfile.py`](../aksharallm/train/stopfile.py) · [`scripts/stop.sh`](../scripts/stop.sh) | `parse` / `reached` / `describe` — a leftover STOP file, or a deadline you forgot you queued |
| the portal disagrees with `nvidia-smi` | [`aksharallm/portal/runs.py`](../aksharallm/portal/runs.py) | `_alive` / `_cmdline` — the pid file first, the anchored command line as fallback, and why it refuses to signal anything else |

And the check that comes before all of them: `python -m pytest tests/ -q`, about five
seconds. `tests/test_model.py::test_kv_cache_matches_full_forward` alone rules out the
single most confusing failure on this page.

---

Next: [10. Running and watching it →](10-running-and-watching.md)
