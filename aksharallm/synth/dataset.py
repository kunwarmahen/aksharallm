"""Where generated data lands, what is recorded beside it, and how it reaches a trainer.

Generated data is kept in its own tree, `data/synth/<name>/`, and never mixed silently into
an existing set. That is not tidiness. Six weeks after a fine-tune goes strange, the only
question worth asking is "what was in the data", and the answer has to be a file rather than
a memory of which model was running that evening.

    data/synth/py-v1/
      samples.jsonl   one kept sample per line
      rejects.jsonl   what was thrown away and why — capped, but the tally in meta is exact
      meta.json       the provenance record

`meta.json` holds the teacher and its host, the template version, every sampling parameter,
the counts at each stage of the funnel, and one entry per generation session. It is written
after every sample rather than at the end, because a run that is stopped — which is normal
here, the same as training — must leave a dataset that can be described and resumed.

**Rejects are kept.** The tally is the quality signal (see `filters.REJECT_REASONS`), and
the rejected text is what tells you *why* the tally looks like that: a 30% pass rate caused
by the teacher writing untestable functions and one caused by near-duplicates need opposite
fixes, and the counts alone cannot distinguish them. The file is capped so a long run cannot
fill a disk with failed attempts.

Exports are deliberately thin. `prepare_sft` and `prepare_dpo` already know how to tokenize
messages and preference triples, so a dataset exports to *their* input shape — a JSONL of
`{"messages": [...]}` or `{"prompt", "chosen", "rejected"}` — and the existing tokenizing,
packing and masking code is reused untouched. Nothing here writes a `.bin`.

Read with: docs/14-synthetic-data.md -- the chapter this implements; it ends with the order to
read these files in.
"""

from __future__ import annotations

import json
import time
from pathlib import Path


class SynthError(Exception):
    """Anything a person could have done differently: an unknown recipe, a teacher that is
    not pulled, an unparseable reply. Carries the reason key for rejections."""


#: A long run rejects thousands of samples; the file is for reading, not for archiving.
MAX_REJECTS = 500
#: Rejected text is trimmed — the first part of a bad reply is where the problem is visible.
REJECT_CHARS = 2000


def synth_root(root: Path | None = None) -> Path:
    from ..portal.runs import repo_root

    return (Path(root) if root else repo_root()) / "data" / "synth"


class Dataset:
    """One generated dataset on disk: append-only samples plus a provenance record."""

    def __init__(self, name: str, root: Path | None = None):
        if not name or "/" in name or "\\" in name or name.startswith("."):
            raise SynthError(f"bad dataset name {name!r} — one path segment, no slashes.")
        self.name = name
        self.dir = synth_root(root) / name
        self.meta: dict = {}
        if self.meta_path.exists():
            self.meta = self._read_meta()

    # ---- paths --------------------------------------------------------------------------
    @property
    def samples_path(self) -> Path:
        return self.dir / "samples.jsonl"

    @property
    def rejects_path(self) -> Path:
        return self.dir / "rejects.jsonl"

    @property
    def meta_path(self) -> Path:
        return self.dir / "meta.json"

    @property
    def exists(self) -> bool:
        return self.meta_path.exists() or self.samples_path.exists()

    # ---- provenance ---------------------------------------------------------------------
    def _read_meta(self) -> dict:
        try:
            return json.loads(self.meta_path.read_text())
        except (OSError, ValueError):
            return {}

    def open(self, recipe: str, teacher: str, host: str, options: dict,
             template_version: int) -> dict:
        """Start (or continue) a dataset and begin a session.

        Appending to an existing dataset with a *different recipe* is refused: two recipes
        produce different sample shapes, and a file that holds both cannot be exported. A
        different teacher is allowed and recorded — "the first 2,000 from qwen2.5:14b and the
        rest from gemma4:31b" is a legitimate dataset, but only if it says so.
        """
        self.dir.mkdir(parents=True, exist_ok=True)
        if self.meta and self.meta.get("recipe") not in (None, recipe):
            raise SynthError(
                f"'{self.name}' already holds {self.meta['recipe']} samples; a dataset is "
                f"one recipe. Use a different name for {recipe}.")
        now = time.time()
        counts = self.meta.get("counts") or {}
        self.meta = {
            **self.meta,
            "name": self.name,
            "recipe": recipe,
            "teacher": teacher,
            "host": host,
            "template_version": template_version,
            "options": options,
            "created": self.meta.get("created", now),
            "updated": now,
            "counts": {
                "asked": int(counts.get("asked", 0)),
                "parsed": int(counts.get("parsed", 0)),
                "kept": int(counts.get("kept", 0)),
                "rejected": dict(counts.get("rejected", {})),
            },
            "teachers": sorted(set(self.meta.get("teachers", []) + [teacher])),
            # Both are lists for the same reason: a dataset appended to over several
            # evenings can legitimately span two teachers or two versions of the prompt, and
            # a single field would quietly report only the last one.
            "template_versions": sorted(set(self.meta.get("template_versions", [])
                                            + [template_version])),
            "sessions": list(self.meta.get("sessions", [])) + [
                {"started": now, "teacher": teacher, "kept": 0, "asked": 0}],
        }
        self.save()
        return self.meta

    def save(self) -> None:
        self.meta["updated"] = time.time()
        self.dir.mkdir(parents=True, exist_ok=True)
        # Written whole each time (the file is a few kB) and atomically, so a stop between
        # the truncate and the write cannot leave a dataset with no provenance at all.
        tmp = self.meta_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.meta, indent=2))
        tmp.replace(self.meta_path)

    def close(self, reason: str = "finished") -> None:
        if self.meta.get("sessions"):
            self.meta["sessions"][-1] |= {"ended": time.time(), "reason": reason}
        self.save()

    # ---- writing ------------------------------------------------------------------------
    def append(self, sample: dict) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        with self.samples_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(sample, ensure_ascii=False) + "\n")
        counts = self.meta.setdefault("counts", {})
        counts["kept"] = int(counts.get("kept", 0)) + 1
        if self.meta.get("sessions"):
            self.meta["sessions"][-1]["kept"] = \
                int(self.meta["sessions"][-1].get("kept", 0)) + 1

    def reject(self, reason: str, seed_id: str, detail: str = "", text: str = "") -> None:
        counts = self.meta.setdefault("counts", {}).setdefault("rejected", {})
        counts[reason] = int(counts.get(reason, 0)) + 1
        if self._reject_lines() >= MAX_REJECTS:
            return                       # the tally stays exact; the examples stop
        with self.rejects_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"seed": seed_id, "reason": reason, "detail": detail,
                                 "text": (text or "")[:REJECT_CHARS],
                                 "when": time.time()}, ensure_ascii=False) + "\n")

    def _reject_lines(self) -> int:
        return sum(int(v) for v in
                   (self.meta.get("counts", {}).get("rejected") or {}).values())

    def count_asked(self, n: int = 1) -> None:
        counts = self.meta.setdefault("counts", {})
        counts["asked"] = int(counts.get("asked", 0)) + n
        if self.meta.get("sessions"):
            self.meta["sessions"][-1]["asked"] = \
                int(self.meta["sessions"][-1].get("asked", 0)) + n

    def count_parsed(self, n: int = 1) -> None:
        counts = self.meta.setdefault("counts", {})
        counts["parsed"] = int(counts.get("parsed", 0)) + n

    # ---- reading ------------------------------------------------------------------------
    def samples(self, limit: int | None = None) -> list[dict]:
        return list(_read_jsonl(self.samples_path, limit))

    def rejects(self, limit: int | None = None) -> list[dict]:
        return list(_read_jsonl(self.rejects_path, limit))

    def n_samples(self) -> int:
        """Counted from the file, not from `meta`.

        The two can disagree — a process killed between the append and the metadata write —
        and when they do, the file is the truth. A dataset that claims 400 samples and holds
        399 is the kind of small lie that ends up in a paper.
        """
        try:
            with self.samples_path.open("rb") as fh:
                return sum(1 for line in fh if line.strip())
        except OSError:
            return 0

    def stats(self) -> dict:
        counts = self.meta.get("counts", {})
        asked = int(counts.get("asked", 0))
        kept = self.n_samples()
        rejected = counts.get("rejected", {}) or {}
        return {
            "name": self.name,
            "recipe": self.meta.get("recipe"),
            "teacher": self.meta.get("teacher"),
            "teachers": self.meta.get("teachers", []),
            "template_version": self.meta.get("template_version"),
            "template_versions": self.meta.get("template_versions", []),
            "created": self.meta.get("created"),
            "updated": self.meta.get("updated"),
            "asked": asked,
            "parsed": int(counts.get("parsed", 0)),
            "kept": kept,
            "rejected": rejected,
            "rejected_total": sum(int(v) for v in rejected.values()),
            # The number that matters: of everything the teacher was asked for, how much
            # survived every filter. Anything under ~20% means the prompt needs work, not
            # that the teacher is bad.
            "pass_rate": (kept / asked) if asked else None,
            "sessions": self.meta.get("sessions", []),
            "options": self.meta.get("options", {}),
            "dir": str(self.dir),
        }

    # ---- export -------------------------------------------------------------------------
    def export(self, path: Path | None = None) -> dict:
        """Write the shape `prepare_sft` / `prepare_dpo` read, and say which command to run.

        The export is regenerated from `samples.jsonl` every time rather than being kept in
        step with it, so it can never be stale relative to the data it came from.
        """
        from .recipes import get_recipe

        recipe = get_recipe(self.meta.get("recipe") or "")
        rows = self.samples()
        if not rows:
            raise SynthError(f"'{self.name}' has no samples to export yet.")
        out = Path(path) if path else (self.dir / f"{recipe.consumer}.jsonl")
        out.parent.mkdir(parents=True, exist_ok=True)
        written = 0
        with out.open("w", encoding="utf-8") as fh:
            for row in rows:
                if recipe.consumer == "dpo":
                    payload = recipe.to_dpo(row)
                else:
                    msgs = recipe.to_sft(row)
                    payload = {"messages": msgs} if msgs else None
                if not payload:
                    continue
                fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
                written += 1
        module = "prepare_sft" if recipe.consumer == "sft" else "prepare_dpo"
        return {
            "path": str(out), "rows": written, "consumer": recipe.consumer,
            "next": (f"python -m aksharallm.data.{module} jsonl --file {out} "
                     f"--tokenizer data/blend/tokenizer.json "
                     f"--out-dir data/{recipe.consumer}-synth"),
        }


def _read_jsonl(path: Path, limit: int | None = None):
    try:
        with path.open("r", encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                if limit is not None and i >= limit:
                    return
                line = line.strip()
                if line:
                    try:
                        yield json.loads(line)
                    except ValueError:
                        continue
    except OSError:
        return


def list_datasets(root: Path | None = None) -> list[dict]:
    """Every generated dataset, newest first — for the CLI's `list` and the portal."""
    base = synth_root(root)
    out = []
    if not base.is_dir():
        return out
    for path in sorted(base.iterdir()):
        if not path.is_dir():
            continue
        try:
            ds = Dataset(path.name, root=base.parent.parent)
        except SynthError:
            continue
        if not ds.exists:
            continue
        out.append(ds.stats())
    out.sort(key=lambda s: s.get("updated") or 0, reverse=True)
    return out
