"""Activation patching: the difference between watching a model and interrogating one.

`lens.py` observes. It can tell you the answer appeared at layer 14, and it cannot tell you
whether layer 14 *caused* it — a residual stream that decodes to "Paris" might be carrying
information that some later layer was going to use anyway. Correlation, and a well-known way
to fool yourself.

Patching intervenes. Run the model on a **clean** prompt and a **corrupted** one that differs
in a single meaningful way, then run the corrupted prompt again while forcing one activation
back to its clean value. If the answer comes back, that activation *carries* the difference.
If it does not, it does not — regardless of what the lens showed.

```
clean       "The capital of France is"     ->  Paris
corrupted   "The capital of Italy  is"     ->  Rome
patched     corrupted, but layer L position P taken from the clean run
                                           ->  Paris again?  then L,P carries "which country"
```

The measurement is a **logit difference**, not a probability: `logit(clean answer) -
logit(corrupted answer)`. Differences of logits are what the model actually computes with,
they are unaffected by the softmax's normalisation over 32,000 irrelevant tokens, and a
single number per patch is what makes a grid of layers × positions readable.

Reported as a *fraction restored*: 0 means the patch changed nothing, 1 means it fully
recovered the clean behaviour. Values above 1 happen and are not a bug — a patched activation
can push harder than the clean run did, usually because the rest of the corrupted context is
now agreeing with it.

**The prompts must tokenize to the same length.** Otherwise position 7 means different things
in the two runs and every number in the grid is a comparison of unrelated things. That is a
refusal here, not a warning.

Read with: docs/17-interpretability.md -- the chapter this implements; it ends with the order
to read these files in.
"""

from __future__ import annotations

import torch

from .capture import Capture, run


class PatchError(Exception):
    """The two prompts cannot be compared."""


def check_pair(clean_ids: list[int], corrupt_ids: list[int]) -> None:
    if len(clean_ids) != len(corrupt_ids):
        raise PatchError(
            f"the prompts tokenize to different lengths ({len(clean_ids)} vs "
            f"{len(corrupt_ids)}). Position 7 would mean different things in the two runs, so "
            f"every patch would compare unrelated activations. Reword one of them.")
    if clean_ids == corrupt_ids:
        raise PatchError("the two prompts are identical — there is no difference to trace.")


@torch.no_grad()
def logit_diff(logits: torch.Tensor, answer: int, other: int, position: int = -1) -> float:
    """`logit(answer) - logit(other)` at one position. The whole measurement."""
    row = logits[position].float()
    return float(row[answer] - row[other])


@torch.no_grad()
def patch_grid(model, clean_ids: list[int], corrupt_ids: list[int], answer: int,
               other: int, device: str = "cpu", position: int | None = None) -> dict:
    """Patch every (layer, position) in turn and report how much each one restores.

    One forward pass per cell — layers × positions of them — which is why this is a small-model
    and short-prompt tool. A 24-layer model on a 12-token prompt is 288 passes: seconds on a
    GPU, a minute on a CPU.
    """
    check_pair(clean_ids, corrupt_ids)
    clean = run(model, clean_ids, device=device)
    corrupt = run(model, corrupt_ids, device=device)

    base = logit_diff(corrupt.logits, answer, other)
    target = logit_diff(clean.logits, answer, other)
    span = target - base
    positions = list(range(len(clean_ids))) if position is None else [position]

    ids = torch.tensor([corrupt_ids], dtype=torch.long, device=device)
    grid: list[list[float]] = []
    for layer in range(len(model.blocks)):
        row: list[float] = []
        for pos in positions:
            donor = clean.residual[layer][pos]

            def hook(_module, _inputs, output, _pos=pos, _donor=donor):
                # The block's output *is* the residual stream at that depth, so replacing one
                # row of it is exactly "make this position believe what the clean run
                # believed", with everything downstream free to react.
                patched = output.clone()
                patched[0, _pos] = _donor
                return patched

            handle = model.blocks[layer].register_forward_hook(hook)
            try:
                logits, _ = model(ids, full_logits=True)
            finally:
                handle.remove()
            restored = (logit_diff(logits[0], answer, other) - base) / span if span else 0.0
            row.append(restored)
        grid.append(row)

    return {
        "grid": grid,                     # [layer][position] -> fraction restored
        "positions": positions,
        "layers": len(model.blocks),
        "clean_diff": target,
        "corrupt_diff": base,
        "span": span,
        "best": _best(grid, positions),
    }


def _best(grid: list[list[float]], positions: list[int]) -> dict | None:
    best = None
    for li, row in enumerate(grid):
        for pi, value in enumerate(row):
            if best is None or value > best["restored"]:
                best = {"layer": li, "position": positions[pi], "restored": value}
    return best


def summarise(result: dict, tokens: list[str]) -> str:
    """One sentence a person can act on, because a 24x12 grid of numbers is not a finding."""
    best = result.get("best")
    if not best:
        return "nothing to report: the grid is empty."
    if result["span"] == 0:
        return ("the clean and corrupted prompts produce the same logit difference, so there "
                "is nothing for a patch to restore. Pick answers the two runs disagree about.")
    token = tokens[best["position"]] if best["position"] < len(tokens) else "?"
    return (f"the difference is carried by block {best['layer']} at position "
            f"{best['position']} ({token!r}): patching it alone restores "
            f"{best['restored'] * 100:.0f}% of the clean logit difference.")
