"""Every evaluation that has ever been run, and what changed between them.

A single benchmark score is close to useless. 25.4% on MMLU means nothing until you know
that chance is 25%, that the same checkpoint scored 25.1% ten thousand steps ago, and that
the standard error is 2%. So the harness keeps every result and this module is the part
that puts them side by side.

The store is a folder of JSON files (`logs/eval/*.json`), not a database. Same decision as
the quantization panel and the playground's history: a result is a few hundred kilobytes,
`git` and `grep` both work on it, a run started in a terminal shows up in the browser, and
there is no schema migration to get wrong the day a suite is added.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from ..infer.checkpoints import repo_root
from .suites import SUITES

#: Files in `logs/eval/` that are job bookkeeping rather than results. The portal's Eval
#: panel writes its "what is running" state beside the results, the same way the quantize
#: panel does; without this the running job appears in the table as an empty evaluation.
NOT_RESULTS = {"current.json"}


def results_dir(root: Path | str | None = None) -> Path:
    """Where evaluations are recorded. Deliberately under `logs/`, not `data/`: a result is
    something this project produced, not something it downloaded."""
    return (Path(root) if root else repo_root()) / "logs" / "eval"


class Results:
    """Read-only view over `logs/eval/`.

    Nothing here imports torch, so the portal can render the results table (and the whole
    Eval tab) without a model or a CUDA context anywhere near the web server.
    """

    def __init__(self, root: Path | str | None = None):
        self.dir = results_dir(root)

    def files(self) -> list[Path]:
        if not self.dir.is_dir():
            return []
        return sorted((p for p in self.dir.glob("*.json") if p.name not in NOT_RESULTS),
                      key=lambda p: p.stat().st_mtime, reverse=True)

    def load(self, limit: int = 50, run: str | None = None) -> list[dict]:
        out = []
        for path in self.files():
            try:
                data = json.loads(path.read_text())
            except (OSError, ValueError):
                continue
            prov = data.get("provenance") or {}
            if run and prov.get("run") != run:
                continue
            data["_file"] = path.name
            data["_when"] = path.stat().st_mtime
            out.append(data)
            if len(out) >= limit:
                break
        return out

    def rows(self, limit: int = 50, run: str | None = None) -> list[dict]:
        """One flat row per evaluation, for a table. Suite scores as a `{name: score}` map.

        Deliberately loses the per-item detail: this is the "did it move?" view, and the
        file on disk is there for the "which questions?" view.
        """
        rows = []
        for data in self.load(limit, run):
            prov = data.get("provenance") or {}
            scores = {}
            for name, res in (data.get("suites") or {}).items():
                if res.get("skipped"):
                    continue
                scores[name] = {"score": res.get("score"),
                                "kind": res.get("kind") or SUITES[name].kind if name in SUITES else None,
                                "n": res.get("n"),
                                "stderr": res.get("stderr"),
                                "baseline": res.get("baseline")}
            rows.append({
                "file": data.get("_file"), "when": data.get("_when"),
                "checkpoint": data.get("checkpoint"), "run": prov.get("run"),
                "step": prov.get("step"), "best_val": prov.get("best_val"),
                "tokens_seen": prov.get("tokens_seen"), "params": prov.get("params"),
                "stage": data.get("stage"), "adapter": data.get("adapter"),
                "device": data.get("device"), "seconds": data.get("seconds"),
                "label": (data.get("options") or {}).get("label"),
                "scores": scores,
            })
        return rows

    def compare(self, suite: str, run: str | None = None, limit: int = 200) -> dict:
        """One suite's score at every step it was ever measured at, oldest first.

        This is the shape the whole harness exists to produce: a column of numbers against
        training step that says whether the model is getting better at something a person
        would notice, rather than at predicting the next token.
        """
        points = []
        for row in self.rows(limit, run):
            entry = (row["scores"] or {}).get(suite)
            if not entry or entry.get("score") is None:
                continue
            points.append({
                "step": row["step"], "score": entry["score"], "n": entry["n"],
                "stderr": entry.get("stderr"), "checkpoint": row["checkpoint"],
                "adapter": row["adapter"], "when": row["when"], "file": row["file"],
                "best_val": row["best_val"], "tokens_seen": row["tokens_seen"],
            })
        points.sort(key=lambda p: (p["step"] is None, p["step"], p["when"]))
        suite_info = SUITES.get(suite)
        return {
            "suite": suite,
            "baseline": suite_info.baseline if suite_info else None,
            "expect": suite_info.expect if suite_info else None,
            "kind": suite_info.kind if suite_info else None,
            "points": points,
        }

    def latest(self, run: str | None = None) -> dict | None:
        rows = self.load(1, run)
        return rows[0] if rows else None


# --------------------------------------------------------------------------------------
# text rendering, shared by the CLI and the job log the portal tails
# --------------------------------------------------------------------------------------

def fmt_score(name: str, res: dict) -> str:
    if res.get("skipped"):
        return "skipped"
    kind = res.get("kind") or (SUITES[name].kind if name in SUITES else "")
    if kind == "ppl":
        return f"{res['perplexity']:.3f}"
    if kind == "judge":
        return f"{res['mean']:.2f}/5" if res.get("mean") is not None else "–"
    score = res.get("score")
    return f"{score * 100:.1f}%" if score is not None else "–"


def summary_table(result: dict) -> str:
    """The block the CLI prints at the end of a run.

    Every row carries its own "what to expect" line. That is not padding: the single most
    common way to misread this table is to see 25% on MMLU and conclude the model is
    broken, when 25% is what four-way multiple choice pays for guessing.
    """
    prov = result.get("provenance") or {}
    head = [
        "",
        f"  {result.get('checkpoint')}"
        + (f"  + {result['adapter']}" if result.get("adapter") else ""),
        f"  step {prov.get('step'):,}" if prov.get("step") is not None else "  step –",
    ]
    if prov.get("best_val") is not None:
        head[-1] += f"   val {prov['best_val']:.4f}"
    if prov.get("params"):
        head[-1] += f"   {prov['params'] / 1e6:.0f}M params"
    head[-1] += f"   on {result.get('device')}"

    lines = list(head) + ["", f"  {'suite':<14} {'score':>9}  {'n':>6}  {'chance':>7}   note"]
    lines.append("  " + "-" * 74)
    for name, res in (result.get("suites") or {}).items():
        if res.get("skipped"):
            lines.append(f"  {name:<14} {'skipped':>9}  {'':>6}  {'':>7}   {res.get('error', '')[:44]}")
            continue
        base = res.get("baseline")
        lines.append(
            f"  {name:<14} {fmt_score(name, res):>9}  {str(res.get('n', '')):>6}  "
            f"{(f'{base * 100:.0f}%' if base else ''):>7}   "
            + (f"± {res['stderr'] * 100:.1f}" if res.get("stderr") is not None else ""))
    lines.append("")
    for name, res in (result.get("suites") or {}).items():
        if res.get("expect"):
            lines.append(f"  {name}: {_wrap(res['expect'], 76, '    ')}")
    lines.append("")
    lines.append(f"  total {result.get('seconds', 0):.0f}s"
                 f"   {time.strftime('%Y-%m-%d %H:%M', time.localtime(result.get('started')))}")
    return "\n".join(lines)


def _wrap(text: str, width: int, indent: str) -> str:
    words, line, out = text.split(), "", []
    for word in words:
        if len(line) + len(word) + 1 > width:
            out.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    out.append(line)
    return ("\n" + indent).join(out)


def compare_table(data: dict) -> str:
    """`suite` across every evaluation, as a column with a delta."""
    points = data.get("points") or []
    if not points:
        return f"  no results recorded for {data['suite']} yet."
    lines = [f"", f"  {data['suite']}"
             + (f"   (chance {data['baseline'] * 100:.0f}%)" if data.get("baseline") else ""),
             "", f"  {'step':>9}  {'val':>7}  {'score':>8}  {'Δ':>7}  {'n':>6}  checkpoint",
             "  " + "-" * 74]
    prev = None
    for p in points:
        score = p["score"]
        shown = f"{score * 100:.1f}%" if data.get("kind") != "ppl" else f"{score:.3f}"
        delta = "" if prev is None else (
            f"{(score - prev) * 100:+.1f}" if data.get("kind") != "ppl"
            else f"{score - prev:+.3f}")
        step = p["step"] if p["step"] is not None else "–"
        val = "–" if p.get("best_val") is None else format(p["best_val"], ".4f")
        lines.append(
            f"  {step:>9}  {val:>7}  {shown:>8}  {delta:>7}  {str(p['n'] or ''):>6}  "
            f"{p['checkpoint']}" + (f" + {p['adapter']}" if p.get("adapter") else ""))
        prev = score
    if data.get("expect"):
        lines += ["", "  " + _wrap(data["expect"], 76, "  ")]
    return "\n".join(lines)
