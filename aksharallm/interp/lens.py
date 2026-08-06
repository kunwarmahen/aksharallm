"""The logit lens: what the model would have said if it had stopped at layer *n*.

A pre-norm transformer never rewrites its state — each block *adds* to a running total called
the residual stream, and the output head reads that total at the end. So the same head can be
pointed at the total **halfway through**, and it answers a question nothing else in this repo
can: not "what did the model predict?" but "*when did it decide?*"

    prediction at layer n  =  lm_head( final_norm( residual_after_layer_n ) )

Applying the final norm before the head is the part people leave out and then wonder why the
early layers look like noise: the head was trained to read normalised vectors, and the
residual stream's magnitude grows with depth. It is one line, and without it every layer's
top token is whichever one happens to have the largest raw dot product.

What it is good for, in order of usefulness on a model this size:

* **Where an answer forms.** A fact that is present from layer 6 and one that only appears at
  layer 22 are different kinds of knowledge.
* **Whether a layer changed its mind.** `flips` counts the layers whose top token differs
  from the layer before, which is a compact way to read a whole prompt at once.
* **Confidence over depth.** A probability that climbs steadily is a model that is sure; one
  that jumps at the last layer is a model that was talked into it.

The honest caveat: this is a *reading*, not a measurement. Nothing says an intermediate
residual is supposed to decode to anything sensible — the model is free to hold information
in a form the output head cannot read until later layers rotate it. Treat a clean logit-lens
story as a hypothesis to check with `patch.py`, which intervenes rather than observes.

Read with: docs/17-interpretability.md -- the chapter this implements; it ends with the order
to read these files in.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .capture import Capture


@torch.no_grad()
def logit_lens(model, cap: Capture, position: int = -1, top: int = 5) -> list[dict]:
    """What each layer would predict for the token after `position`.

    Returns one row per layer (plus row 0 for the embedding, before any block), each with its
    top-`k` tokens and their probabilities.
    """
    rows: list[dict] = []
    states = [cap.embedding] + list(cap.residual)
    for depth, state in enumerate(states):
        if state is None:
            continue
        hidden = state[position]
        logits = model.lm_head(model.norm(hidden.to(next(model.parameters()).dtype)))
        probs = F.softmax(logits.float(), dim=-1)
        vals, idx = torch.topk(probs, min(top, probs.numel()))
        rows.append({
            "layer": depth,                       # 0 = embedding only, 1 = after block 0
            "label": "embedding" if depth == 0 else f"block {depth - 1}",
            "top": [{"id": int(i), "prob": float(p)} for i, p in zip(idx, vals)],
            "entropy": float(-(probs * probs.clamp_min(1e-12).log()).sum()),
        })
    return rows


def lens_story(rows: list[dict], decode) -> dict:
    """The lens as a sentence: where the final answer first appeared, and how often the top
    token changed on the way.

    `decode` turns a token id into text — passed in rather than imported, so this module never
    needs a tokenizer and the same function serves the CLI and the portal.
    """
    if not rows:
        return {"answer": None, "settled_at": None, "flips": 0, "rows": rows}
    answer = rows[-1]["top"][0]["id"]
    settled = None
    flips = 0
    previous = None
    for row in rows:
        top = row["top"][0]["id"]
        if previous is not None and top != previous:
            flips += 1
        previous = top
        if top == answer and settled is None:
            settled = row["layer"]
        elif top != answer:
            settled = None                        # it changed its mind again; keep looking
    return {
        "answer": answer,
        "answer_text": decode([answer]),
        "settled_at": settled,
        "settled_label": next((r["label"] for r in rows if r["layer"] == settled), None),
        "flips": flips,
        "layers": len(rows),
        "rows": rows,
    }


@torch.no_grad()
def layer_contributions(model, cap: Capture, position: int = -1) -> list[dict]:
    """How much each block *moved* the residual stream, and whether it helped the final answer.

    Two numbers per layer. `norm_delta` is the size of what the block added, which says where
    the work happens. `answer_delta` is how much that addition raised (or lowered) the final
    answer's logit — the same reading as the lens, but attributed to one block rather than
    accumulated, so a layer that quietly argues *against* the eventual answer is visible.
    """
    states = [cap.embedding] + list(cap.residual)
    dtype = next(model.parameters()).dtype
    answer = int(model.lm_head(model.norm(states[-1][position].to(dtype))).argmax())
    out = []
    for i in range(1, len(states)):
        before, after = states[i - 1][position], states[i][position]
        delta = after - before
        lo = float(model.lm_head(model.norm(before.to(dtype)))[answer])
        hi = float(model.lm_head(model.norm(after.to(dtype)))[answer])
        out.append({
            "layer": i - 1,
            "norm_delta": float(delta.float().norm()),
            "norm_after": float(after.float().norm()),
            "answer_delta": hi - lo,
        })
    return out
