"""Continuing a stopped run: one small contract that every post-training loop obeys.

The twin of `stopfile.py`. That module says how a run *ends* early; this one says how the
next launch picks it back up, so "stop it tonight, continue tomorrow" is the same two
commands for SFT, DPO and GRPO as it already is for pretraining.

    scripts/stage.sh sft small-code     # start, or continue -- same command
    RESUME=none scripts/stage.sh sft small-code   # ...or deliberately start over

Three things make a resume *correct* rather than merely possible, and each one is a silent
bug when it is missing -- no crash, no warning, just a worse model:

**1. The optimizer, not only the weights.** Adam's moments are most of what a warm run
knows. Reloading weights into a fresh optimizer throws that away and produces a visible
loss bump at every resume, which then gets misread as a data problem.

**2. The position in the data.** Pretraining samples random windows from a stream, so a
restarted sampler costs only exactness. SFT and DPO iterate a *shuffled epoch*: a resume
that re-shuffles shows the model some examples twice within one epoch and others not at
all -- precisely the overfitting post-training is most exposed to, and nothing in the loss
curve reveals it. So the checkpoint records the rng state as of the **start of the current
epoch** (the permutation is drawn once, there) plus how many micro-batches of that epoch
were consumed; the resume replays the same permutation and skips forward.

**3. The reference model must NOT move.** This is the one that is unique to post-training
and the easiest to get wrong. DPO and GRPO hold two models: a *policy* that trains and a
frozen *reference* that says where the policy started. The reference is the whole
constraint -- the KL term measures drift away from it. A resume that reloads the reference
from the same checkpoint as the policy re-anchors it to the drifted weights, the KL term
collapses toward zero, and the run wanders arbitrarily far from the SFT model **while
reporting a small KL**. It looks like a well-behaved run right up until you read the
samples.

    --init / --sft   the reference, always. Never reloaded from a resume.
    --resume         the policy and its optimizer, and nothing else.

Also carried: the best metric seen so far (`best_val`, `best_reward`). Letting it reset
means the first step of the next session is "the best so far" and overwrites the best
checkpoint with a worse model -- the one failure here that destroys work rather than
wasting it.

Read with: docs/06-posttraining.md -- the chapter this implements. See also
docs/10-running-and-watching.md for the stop/resume loop from the outside.
"""

from __future__ import annotations

from pathlib import Path

import torch

#: Spellings of "do not resume". This arrives from `RESUME=` in the environment, where
#: unsetting a variable is awkward and the natural thing to type is a word. Without this,
#: `RESUME=none` goes looking for a checkpoint named "none".
OFF = ("", "none", "off", "no", "false")


def resolve(spec: str | None, default: Path) -> Path | None:
    """Which checkpoint to continue from: a path, or None to start fresh.

    `auto` is the value the launchers pass, and it means "continue if there is something to
    continue, otherwise start". That is what lets one command serve both, which is what
    makes stop/resume usable from a button rather than from a decision.
    """
    if spec is None:
        return None
    text = str(spec).strip()
    if text.lower() in OFF:
        return None
    if text.lower() == "auto":
        return default if default.exists() else None
    return Path(text)


def load(path: Path, model, optimizer=None, device="cpu") -> dict:
    """Restore the *policy* and its optimizer. Never the reference -- see the module docstring.

    Returns the whole checkpoint so the caller can read `step`, `best_*` and its own
    progress payload out of it.
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt


def epoch_progress(epoch: int, batches_done: int, epoch_rng_state, best: float) -> dict:
    """Where an epoch-iterating trainer (SFT, DPO) had got to.

    `epoch_rng_state` is the generator state *before* this epoch's permutation was drawn,
    and `batches_done` counts micro-batches pulled from it since. Together they name one
    exact position in the shuffle.
    """
    return {"epoch": epoch, "batches_done": batches_done,
            "epoch_rng_state": epoch_rng_state, "best": best}


def step_progress(step: int, rng_state, best: float) -> dict:
    """Where a step-iterating trainer (GRPO) had got to.

    No epochs to replay -- the loop is `range(steps)` -- but the sampler that picks which
    prompts to work on each step is stateful, and letting it restart means every session
    grinds the same prompts in the same order.
    """
    return {"step": step, "rng_state": rng_state, "best": best}


def restore_rng(rng, state, what: str) -> bool:
    """Put a numpy Generator back where it was. False (with a note) if the state is unusable.

    A checkpoint written before this existed has no state to restore, which is not an error
    -- it just means this one resume is not identical to an uninterrupted run. Saying so is
    better than either crashing or pretending.
    """
    if state is None:
        return False
    try:
        rng.bit_generator.state = state
        return True
    except (KeyError, TypeError, ValueError) as exc:
        print(f"  note: could not restore {what} ({exc}); continuing from the seed")
        return False
