---
id: stop-resume
title: Stopping a six-day run without losing it
doc: docs/09-running-and-watching.md
files:
  - aksharallm/train/stopfile.py
  - scripts/stop.sh
verify: tests/test_pipeline.py::test_garbage_stop_file_is_treated_as_stop_now
prereqs: [training-loop]
minutes: 20
summary: A run that must survive being interrupted, and the one-line file that lets a terminal, a script and a browser all mean the same thing.
---

# 6. Stopping a six-day run without losing it

Phase 2 is roughly 60 hours of GPU time. Nobody has 60 uninterrupted hours, the machine is
also somebody's desktop, and a power cut is not a special case — it is Tuesday. So the run
is built to be **stopped and resumed**, and every part of this project assumes that.

The pieces are small:

* **checkpoints** — the weights, the optimiser state and the step number, written
  periodically. The optimiser state matters: Adam keeps running averages per parameter, and
  resuming without them causes a visible loss spike.
* **`resume: auto`** — a new launch loads `ckpt_last.pt` and carries on at the next step.
* **the STOP file** — how you ask.

## Why a file and not a signal

You want three things to be able to stop a run: a terminal, a script, and a button in a
browser. Signals need a process id and a parent-child relationship; a file needs neither.
The trainer reads `<out_dir>/STOP` fresh on **every step**, so all three routes mean exactly
the same thing to the process, and a stop queued from a portal that then restarts is still
queued.

One line, three forms:

```
(empty)        stop after the current step
20000          stop on reaching step 20,000   (inclusive: 20,000 is trained)
@1753985400    stop at this wall-clock time
```

The third is why "stop in twenty minutes" works without anything watching a clock on the
trainer's behalf. A timer in the portal dies with the portal; a duration converted to a step
count at the moment you press the button is wrong the moment throughput changes. A deadline
in a file is still true after a reboot.

## Anything unreadable means stop

If the file exists but cannot be parsed, the trainer stops. That is the safe reading of an
ambiguous request: a trainer that cannot tell whether it was asked to stop should stop.

---

## Exercise: make an unreadable stop file ambiguous

1. Run the check. It passes — it writes nonsense into a STOP file and asserts the result is
   "stop now".
2. In `aksharallm/train/stopfile.py`, find `parse` and make the unparseable case return a
   request that is *not* an immediate stop (for example, treat it as no stop at all).
3. Run the check. **It should fail.**
4. Put it back. Green.

> **What you just saw.** The test encodes a *decision*, not a behaviour: when the request is
> ambiguous, err toward stopping. Tests that pin down decisions like this are the ones that
> stop a future change from quietly reversing something that was thought about once.

## Do it for real

```bash
scripts/experiment.sh tiny-moe &          # or press Start in the portal
scripts/stop.sh tiny-moe --status         # alive? at what step? what stop is queued?
scripts/stop.sh tiny-moe --in 2m          # queue one, in wall clock
scripts/stop.sh tiny-moe --cancel         # change your mind
```

Then look at the run again: it saved at the step it stopped on, and starting it again picks
up from the next one with no spike in the loss curve.
