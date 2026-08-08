"""Where benchmark data comes from, and why it is copied to disk before use.

A benchmark is only a benchmark if the same questions are asked every time. Streaming a
dataset from the Hub on every run breaks that in two quiet ways: the Hub's copy can change
under you, and "the first 500 rows of a stream" is not guaranteed to be the same 500 rows
twice. So every suite is **fetched once and written to `data/eval/<name>.jsonl`**, and
every run after that reads the local file. The number you measured last month stays
reproducible, and evaluating works with the network unplugged.

Each fetch also writes `<name>.meta.json` — which repository, which config, which split,
how many rows, and when. That file is the answer to "which MMLU is this?", which is a
question worth being able to answer three model generations later.

Only the columns a suite actually uses are kept. HellaSwag's validation split carries
`source_id`, `split_type` and friends that no scorer ever reads; dropping them takes the
cache from tens of megabytes to a few, and makes the file readable with `head`.

Read with: docs/13-eval.md -- the chapter this implements; it ends with the order to read these
files in.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from ..infer.checkpoints import repo_root


class EvalError(Exception):
    """Anything the harness can explain in a sentence: missing data, unknown suite."""


@dataclass(frozen=True)
class Source:
    """One split of one dataset, and the local file it becomes."""

    name: str
    #: Candidate Hub repositories, tried in order. A list rather than a string because the
    #: canonical repo is not always loadable: `datasets` 5.0 removed dataset *scripts*, so
    #: `ybisk/piqa` (a script dataset) raises, and the working copy is a parquet mirror.
    #: Trying in order means the canonical name is still recorded when it works.
    repos: tuple[str, ...]
    config: str | None
    split: str
    #: Columns kept in the cache. Everything else is dropped — see the module docstring.
    fields: tuple[str, ...]
    #: Rows to keep at most. The cache is meant to be committed-to-disk, not comprehensive:
    #: nothing here evaluates more than a few thousand items in a sitting.
    max_rows: int = 20000
    note: str = ""


SOURCES: dict[str, Source] = {
    "mmlu": Source(
        "mmlu", ("cais/mmlu",), "all", "test",
        ("question", "subject", "choices", "answer"),
        note="57 subjects, 14,042 questions. The standard general-knowledge exam.",
    ),
    # MMLU is conventionally 5-shot, and the shots come from its own `dev` split — five
    # worked examples per subject, which is why the dev split exists at all. Keeping it as a
    # separate source means the shots are subject-matched rather than random.
    "mmlu-dev": Source(
        "mmlu-dev", ("cais/mmlu",), "all", "dev",
        ("question", "subject", "choices", "answer"),
        note="5 worked examples per subject — the few-shot prompt for MMLU.",
    ),
    "gsm8k": Source(
        "gsm8k", ("openai/gsm8k",), "main", "test",
        ("question", "answer"),
        note="1,319 grade-school word problems with worked solutions.",
    ),
    "gsm8k-train": Source(
        "gsm8k-train", ("openai/gsm8k",), "main", "train",
        ("question", "answer"), max_rows=64,
        note="The few-shot examples for GSM8K, taken from the train split so no test "
             "question is ever shown to the model.",
    ),
    "arc-easy": Source(
        "arc-easy", ("allenai/ai2_arc",), "ARC-Easy", "test",
        ("id", "question", "choices", "answerKey"),
        note="2,376 grade-school science questions, the easy half.",
    ),
    "arc-challenge": Source(
        "arc-challenge", ("allenai/ai2_arc",), "ARC-Challenge", "test",
        ("id", "question", "choices", "answerKey"),
        note="1,172 science questions a retrieval baseline gets wrong.",
    ),
    "hellaswag": Source(
        "hellaswag", ("Rowan/hellaswag",), None, "validation",
        ("ctx", "endings", "label", "activity_label"),
        note="10,042 sentence completions, three of four adversarially wrong.",
    ),
    "piqa": Source(
        "piqa", ("baber/piqa", "nthngdy/piqa", "ybisk/piqa"), None, "validation",
        ("goal", "sol1", "sol2", "label"),
        note="1,838 physical-commonsense pairs. The canonical `ybisk/piqa` is a dataset "
             "script and no longer loads on datasets>=5; the mirrors are the same rows.",
    ),
    "humaneval": Source(
        "humaneval", ("openai/openai_humaneval",), None, "test",
        ("task_id", "prompt", "canonical_solution", "test", "entry_point"),
        note="164 Python functions with hidden tests. The objective code number.",
    ),
}


def eval_dir(root: Path | str | None = None) -> Path:
    return (Path(root) if root else repo_root()) / "data" / "eval"


def cache_path(name: str, root: Path | str | None = None) -> Path:
    return eval_dir(root) / f"{name}.jsonl"


def meta_path(name: str, root: Path | str | None = None) -> Path:
    return eval_dir(root) / f"{name}.meta.json"


def spec(name: str) -> Source:
    try:
        return SOURCES[name]
    except KeyError:
        raise EvalError(f"unknown dataset {name!r}. Known: {', '.join(sorted(SOURCES))}")


def is_cached(name: str, root: Path | str | None = None) -> bool:
    path = cache_path(name, root)
    return path.is_file() and path.stat().st_size > 0


def meta(name: str, root: Path | str | None = None) -> dict:
    try:
        return json.loads(meta_path(name, root).read_text())
    except (OSError, ValueError):
        return {}


def fetch(name: str, root: Path | str | None = None, refresh: bool = False,
          log=print) -> Path:
    """Download `name` and write it to `data/eval/`. A no-op when already cached.

    Returns the path to the jsonl. Raises `EvalError` with a usable sentence if the data
    cannot be had — an eval harness that fails with a stack trace from inside a third-party
    loader is one you stop running.
    """
    src = spec(name)
    path = cache_path(name, root)
    if path.is_file() and path.stat().st_size > 0 and not refresh:
        return path

    try:
        from datasets import load_dataset       # noqa: PLC0415 — optional, and slow to import
    except ImportError:
        raise EvalError(
            "the `datasets` package is needed to fetch benchmark data (it is only needed "
            "once per suite — after that the harness reads data/eval/). "
            "Install it with:  uv pip install datasets")

    errors: list[str] = []
    rows: list[dict] | None = None
    used = None
    for repo in src.repos:
        try:
            log(f"[eval] fetching {name}: {repo}"
                + (f" ({src.config})" if src.config else "") + f" split={src.split}")
            ds = load_dataset(repo, src.config, split=src.split)
            rows = [{k: r[k] for k in src.fields if k in r} for r in ds]
            used = repo
            break
        except Exception as exc:                # noqa: BLE001 — any loader failure, same answer
            errors.append(f"{repo}: {type(exc).__name__}: {exc}")

    if rows is None:
        raise EvalError(
            f"could not fetch {name} from any of {', '.join(src.repos)}.\n  "
            + "\n  ".join(errors))
    if not rows:
        raise EvalError(f"{name} came back empty from {used} — refusing to cache it.")

    rows = rows[: src.max_rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".jsonl.tmp")
    with tmp.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    tmp.replace(path)                            # atomic: never leave a half file behind
    meta_path(name, root).write_text(json.dumps({
        "name": name, "repo": used, "config": src.config, "split": src.split,
        "rows": len(rows), "fields": list(src.fields), "fetched_at": time.time(),
        "note": src.note,
    }, indent=2))
    log(f"[eval] cached {len(rows):,} rows → {path}")
    return path


def load(name: str, root: Path | str | None = None, limit: int | None = None,
         auto_fetch: bool = True) -> list[dict]:
    """The rows of `name`, from the local cache, fetching it first if allowed.

    `limit` takes the **first** N rows rather than a random sample. Deterministic beats
    representative here: the point is that two checkpoints answer the same questions, and a
    seeded sample would still change the moment anyone touched the seed.
    """
    path = cache_path(name, root)
    if not (path.is_file() and path.stat().st_size > 0):
        if not auto_fetch:
            raise EvalError(
                f"{name} is not cached. Fetch it once with:  "
                f"python -m aksharallm.eval fetch {name}")
        fetch(name, root)
    rows = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    if not rows:
        raise EvalError(f"{path} is empty — refetch it with --refresh.")
    return rows


def status(root: Path | str | None = None) -> list[dict]:
    """What is on disk, for the CLI's `fetch --list` and the portal's Data panel."""
    out = []
    for name, src in SOURCES.items():
        path = cache_path(name, root)
        cached = path.is_file() and path.stat().st_size > 0
        info = meta(name, root) if cached else {}
        out.append({
            "name": name, "repos": list(src.repos), "config": src.config,
            "split": src.split, "note": src.note, "cached": cached,
            "path": str(path), "bytes": path.stat().st_size if cached else 0,
            "rows": info.get("rows"), "repo": info.get("repo"),
            "fetched_at": info.get("fetched_at"),
        })
    return out


@dataclass
class Missing:
    """Datasets a requested set of suites needs but does not have."""

    names: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return bool(self.names)

    def describe(self) -> str:
        return ("not cached: " + ", ".join(self.names) + ".  Fetch with:  "
                "python -m aksharallm.eval fetch " + " ".join(self.names))
