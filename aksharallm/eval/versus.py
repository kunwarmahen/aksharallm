"""Head to head: which of two models answered better, prompt by prompt.

Absolute grades are too coarse to separate two small models. On the twelve-prompt judge set
our SFT, DPO and GRPO models scored 1.42, 1.25 and 1.42 out of 5 — nearly every answer a 1
or a 2, so a real difference between two of them mostly lands on the same grade and
vanishes. A judge shown *both* answers can still say which is better when both are poor, and
"which is better" is the question we actually have.

Two things make the verdict worth trusting:

* **Both orders.** Judges prefer whichever answer they read first (or second — it varies by
  model, which is worse). Every prompt is judged twice, A-then-B and B-then-A; a model wins
  the prompt only if it is preferred **both** times. Anything else is a tie. Position bias
  then shows up as ties, never as wins, and the `inconsistent` count says how much of it
  there was.
* **A sign test, not a percentage.** 9 wins to 5 sounds decisive and is not (p ≈ 0.42).
  `p_value` is the exact two-sided binomial test on wins vs losses, ties excluded, so a
  result reads as "B is better" only when chance is an unlikely explanation.

It reuses answers already generated: each side is a result JSON written by
`python -m aksharallm.eval <ckpt> --suite judge48`, whose `items` keep every answer. So a
comparison costs judge calls and no generation, and any two evaluated checkpoints can be
compared after the fact — including ones evaluated weeks apart, as long as it was the same
suite. Written to `logs/eval/versus-<a>-vs-<b>-<when>.json` under `comparisons`, never
`suites`, so the result readers (which go by shape) do not mistake it for a benchmark row.

Read with: docs/13-eval.md -- the chapter this implements; it ends with the order to read
these files in.
"""

from __future__ import annotations

import json
import math
import time
from datetime import datetime
from pathlib import Path

from . import judge as judge_mod
from . import suites as suites_mod
from .sources import EvalError

SYSTEM = """\
You compare two answers written by small language models that are being trained from
scratch. Both may be poor; your job is only to say which one is BETTER for the prompt, judged
against the rubric. Consistency matters more than anything else.

Rules:
* Prefer the answer that is more correct and does more exactly what the prompt asked.
* A confidently invented fact is worse than admitting ignorance.
* Do not prefer an answer for being longer.
* Say "tie" only if neither is better in any way that matters.

Reply with JSON only, on one line, in exactly this form:
{"better": "A" | "B" | "tie", "reason": "<one short sentence>"}
"""


def build_messages(prompt: str, rubric: str, first: str, second: str) -> list[dict]:
    body = (f"PROMPT:\n{prompt}\n\nRUBRIC — what a good answer contains:\n{rubric}\n\n"
            f"ANSWER A:\n{first if first.strip() else '(empty)'}\n\n"
            f"ANSWER B:\n{second if second.strip() else '(empty)'}\n\n"
            "Which is better? JSON only.")
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": body}]


def parse_choice(text: str) -> str | None:
    """'A', 'B', 'tie', or None when nothing usable came back (recorded, never guessed)."""
    found = judge_mod.JSON_RE.search(text or "")
    if found:
        try:
            got = str(json.loads(found.group(0)).get("better", "")).strip().lower()
        except (ValueError, TypeError, AttributeError):
            got = ""
        if got in ("a", "b"):
            return got.upper()
        if got == "tie":
            return "tie"
    return None


def verdict(first: str | None, second: str | None) -> str:
    """Combine the two orderings into one result for model 1 vs model 2.

    `first` is the judge's choice with model 1 shown as A; `second` with model 2 shown as A.
    So model 1 is preferred by `first == "A"` and by `second == "B"`.
    """
    if first is None or second is None:
        return "ungraded"
    one = {"A": "m1", "B": "m2"}.get(first, "tie")
    two = {"A": "m2", "B": "m1"}.get(second, "tie")
    if one == two == "m1":
        return "win"
    if one == two == "m2":
        return "loss"
    return "tie"


def sign_test(wins: int, losses: int) -> float | None:
    """Exact two-sided binomial p-value for wins vs losses under p = 0.5. Ties excluded."""
    n = wins + losses
    if n == 0:
        return None
    k = min(wins, losses)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def summarise(rows: list[dict]) -> dict:
    counts = {k: sum(r["verdict"] == k for r in rows) for k in ("win", "loss", "tie", "ungraded")}
    decided = counts["win"] + counts["loss"]
    return {**counts, "n": len(rows),
            "win_rate": counts["win"] / decided if decided else None,
            "p_value": sign_test(counts["win"], counts["loss"]),
            "inconsistent": sum(r.get("inconsistent", False) for r in rows)}


def load_side(ref: str, suite: str, root: Path) -> dict:
    """A result file, or the newest result for a checkpoint ref that ran `suite`."""
    path = Path(ref)
    if path.suffix == ".json" and (path.exists() or (root / path).exists()):
        data = json.loads((path if path.exists() else root / path).read_text())
        if suite not in (data.get("suites") or {}):
            raise EvalError(f"{ref} has no {suite!r} result")
        data["_file"] = str(path)
        return data
    from ..infer.checkpoints import CheckpointStore
    ident = CheckpointStore(root).identify(ref)
    best = None
    for f in sorted((root / "logs" / "eval").glob("*.json")):
        try:
            data = json.loads(f.read_text())
        except (OSError, ValueError):
            continue
        s = data.get("suites")
        if (isinstance(s, dict) and data.get("checkpoint") == ident
                and (s.get(suite) or {}).get("items")):
            data["_file"] = str(f)
            best = data                       # sorted by timestamped name: last is newest
    if best is None:
        raise EvalError(f"no {suite!r} result for {ident} — run "
                        f"`python -m aksharallm.eval {ident} --suite {suite}` first")
    return best


def compare(cfg, side1: dict, side2: dict, suite: str, model: str | None = None,
            progress=None) -> dict:
    """Judge every prompt both sides answered, in both orders."""
    items = {i.id: i for i in suites_mod.JUDGE_SETS[suite]}
    a1 = {r["id"]: r.get("answer", "") for r in side1["suites"][suite]["items"]}
    a2 = {r["id"]: r.get("answer", "") for r in side2["suites"][suite]["items"]}
    ids = [i for i in items if i in a1 and i in a2]
    if not ids:
        raise EvalError("the two results share no prompts — were they the same suite?")
    ok, why = judge_mod.available(cfg)
    if not ok:
        raise EvalError(why)

    def ask(item, first, second) -> tuple[str | None, str]:
        client = judge_mod.Ollama(cfg)
        reply, stream = "", client.chat(build_messages(item.prompt, item.rubric, first, second),
                                        model=model)
        try:
            for kind, piece in stream:
                if kind == "delta":
                    reply += piece
        finally:
            stream.close()
        return parse_choice(reply), reply.strip()[:300]

    rows = []
    for n, pid in enumerate(ids):
        item = items[pid]
        c1, r1 = ask(item, a1[pid], a2[pid])
        c2, r2 = ask(item, a2[pid], a1[pid])
        v = verdict(c1, c2)
        rows.append({"id": pid, "group": item.group, "verdict": v,
                     "inconsistent": v == "tie" and None not in (c1, c2) and c1 != "tie"
                     and c2 != "tie",
                     "first": c1, "second": c2, "reason_first": r1, "reason_second": r2})
        if progress:
            progress(n + 1, len(ids), "versus")
    judge_mod.release(cfg, model or cfg.model)

    groups = {}
    for g in sorted({r["group"] for r in rows}):
        groups[g] = summarise([r for r in rows if r["group"] == g])
    return {"suite": suite, "judge_model": model or cfg.model,
            "model1": side1.get("checkpoint"), "model2": side2.get("checkpoint"),
            "label1": (side1.get("options") or {}).get("label"),
            "label2": (side2.get("options") or {}).get("label"),
            "file1": side1.get("_file"), "file2": side2.get("_file"),
            "overall": summarise(rows), "groups": groups, "comparisons": rows,
            "when": datetime.now().isoformat(timespec="seconds")}


def reading(res: dict) -> str:
    o = res["overall"]
    if o["p_value"] is None:
        return "no prompt was decided either way — the two are indistinguishable here"
    better = res["model1"] if o["win"] > o["loss"] else res["model2"]
    if o["p_value"] < 0.05:
        return f"{better} is better (p = {o['p_value']:.3f}, sign test on decided prompts)"
    return (f"no significant difference (p = {o['p_value']:.2f}); {o['tie']} of {o['n']} "
            "prompts were ties")


def write(res: dict, root: Path) -> Path:
    def slug(s):
        return (s or "x").replace("/", "_").replace(".pt", "")
    out = root / "logs" / "eval" / (f"versus-{slug(res['label1'] or res['model1'])}-vs-"
                                    f"{slug(res['label2'] or res['model2'])}-"
                                    f"{time.strftime('%Y%m%d-%H%M%S')}.json")
    out.write_text(json.dumps(res, indent=1))
    return out
