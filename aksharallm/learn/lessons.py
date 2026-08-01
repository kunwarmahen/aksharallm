"""The lessons: what they are, what order they go in, and how they avoid going stale.

A learning path that is only a reading order would add nothing here — the docs are already
numbered 00 to 14. What makes this worth building is that every lesson is a **triple**:

    read the doc  ->  open the file  ->  break it, and watch a real test go red

The third part is the whole point. Reading that a causal mask stops a model seeing the
future is not the same as deleting `is_causal=True`, watching `test_causality` fail, and
seeing exactly which assertion catches it. Most of the exercises are real bugs that actually
happened in this repo — `pool.imap` swallowing a stream exception and writing an empty
`train.bin` *with exit code 0* is a better lesson than anything a tutorial would invent.

The rot problem, and the two rules that solve it
------------------------------------------------
A second surface that describes the code will drift from the code, and a stale lesson is
worse than stale prose: it sends someone to break a line that has moved and leaves them
convinced they have misunderstood. Two rules make this self-checking:

1. **Lessons reference files, never line numbers.** A file that is renamed fails loudly;
   line 47 silently becomes something else.
2. **Every `verify` is a real pytest node id.** `tests/test_lessons.py` collects them all and
   fails if one no longer exists — so the existing suite is the drift detector, and a lesson
   that has rotted shows up as a failing test rather than as a confused reader.

Frontmatter
-----------
Each `docs/lessons/*.md` opens with a YAML block:

```yaml
---
id: kv-cache                       # stable, used in URLs and progress.json
title: The KV cache, and the mask that ruins it
doc: docs/06-inference.md          # the deep dive to read first
files:                             # what to open in the Code tab
  - aksharallm/model/transformer.py
verify: tests/test_model.py::test_kv_cache_matches_full_forward
prereqs: [attention]               # lesson ids that must be complete first
minutes: 25
summary: One sentence for the list.
---
```
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

#: Ids reach URLs, filenames and a progress file, so they are whitelisted rather than escaped.
ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,40}$")

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n(.*)$", re.DOTALL)


class LearnError(Exception):
    """A lesson that cannot be read, or a request for one that does not exist."""


@dataclass
class Lesson:
    id: str
    title: str
    path: Path
    body: str
    doc: str | None = None
    files: list[str] = field(default_factory=list)
    verify: str | None = None
    prereqs: list[str] = field(default_factory=list)
    minutes: int | None = None
    summary: str = ""
    #: An optional hand-off into the Playground: the probe or prompt this lesson is about.
    play: str | None = None
    order: int = 0

    def as_dict(self, body: bool = False) -> dict:
        out = {
            "id": self.id, "title": self.title, "doc": self.doc, "files": self.files,
            "verify": self.verify, "prereqs": self.prereqs, "minutes": self.minutes,
            "summary": self.summary, "play": self.play, "order": self.order,
            "path": str(self.path),
        }
        if body:
            out["body"] = self.body
        return out


def lessons_dir(root: Path | None = None) -> Path:
    from ..portal.runs import repo_root

    return (Path(root) if root else repo_root()) / "docs" / "lessons"


def parse(path: Path, order: int = 0) -> Lesson:
    """One lesson file. A missing frontmatter block is an error, not a default.

    Silently accepting a lesson with no `verify` would produce something that looks like a
    lesson and cannot be completed — the failure mode this whole module is arranged against.
    """
    text = path.read_text(encoding="utf-8")
    found = _FRONTMATTER_RE.match(text)
    if not found:
        raise LearnError(f"{path.name} has no YAML frontmatter block")
    try:
        meta = yaml.safe_load(found.group(1)) or {}
    except yaml.YAMLError as exc:
        raise LearnError(f"{path.name}: frontmatter is not valid YAML ({exc})")
    if not isinstance(meta, dict):
        raise LearnError(f"{path.name}: frontmatter must be a mapping")

    lesson_id = str(meta.get("id") or "").strip()
    if not ID_RE.match(lesson_id):
        raise LearnError(f"{path.name}: bad or missing id {lesson_id!r} "
                         "(lowercase letters, digits and dashes)")
    files = meta.get("files") or []
    if isinstance(files, str):
        files = [files]
    prereqs = meta.get("prereqs") or []
    if isinstance(prereqs, str):
        prereqs = [prereqs]

    return Lesson(
        id=lesson_id,
        title=str(meta.get("title") or lesson_id),
        path=path,
        body=found.group(2).strip(),
        doc=str(meta["doc"]) if meta.get("doc") else None,
        files=[str(f) for f in files],
        verify=str(meta["verify"]) if meta.get("verify") else None,
        prereqs=[str(p) for p in prereqs],
        minutes=int(meta["minutes"]) if meta.get("minutes") else None,
        summary=str(meta.get("summary") or "").strip(),
        play=str(meta["play"]) if meta.get("play") else None,
        order=order,
    )


def load_all(root: Path | None = None) -> list[Lesson]:
    """Every lesson, in filename order.

    Filename order *is* the curriculum order — `01-data.md`, `02-tokenizer.md` — because a
    reading order that lives in a separate index file is an index file that goes stale. The
    prereq graph is the real constraint; the numbering is the suggestion.
    """
    base = lessons_dir(root)
    if not base.is_dir():
        return []
    out: list[Lesson] = []
    for i, path in enumerate(sorted(base.glob("*.md"))):
        if path.name.lower() == "readme.md":
            continue
        out.append(parse(path, order=i))
    return out


def by_id(root: Path | None = None) -> dict[str, Lesson]:
    return {lesson.id: lesson for lesson in load_all(root)}


def get(lesson_id: str, root: Path | None = None) -> Lesson:
    lessons = by_id(root)
    if lesson_id not in lessons:
        raise LearnError(f"no lesson {lesson_id!r}. Known: {', '.join(sorted(lessons))}")
    return lessons[lesson_id]


def validate(root: Path | None = None) -> list[str]:
    """Everything that would make the path lie to a reader. Empty list means healthy.

    Run by `tests/test_lessons.py`, which is what turns the ordinary test suite into the
    drift detector: rename a source file or delete a test, and a lesson pointing at it fails
    the suite rather than failing a person.
    """
    from ..portal.runs import repo_root

    base = Path(root) if root else repo_root()
    problems: list[str] = []
    seen: dict[str, Path] = {}
    lessons = load_all(root)

    for lesson in lessons:
        where = lesson.path.name
        if lesson.id in seen:
            problems.append(f"{where}: duplicate id '{lesson.id}' (also {seen[lesson.id].name})")
        seen[lesson.id] = lesson.path

        if not lesson.verify:
            problems.append(f"{where}: no `verify` — a lesson with no check cannot be done")
        if not lesson.summary:
            problems.append(f"{where}: no `summary` — the list has nothing to show")
        if lesson.doc and not (base / lesson.doc).exists():
            problems.append(f"{where}: doc '{lesson.doc}' does not exist")
        for rel in lesson.files:
            if not (base / rel).exists():
                problems.append(f"{where}: file '{rel}' does not exist")
        # Line numbers rot silently; this is the rule that keeps lessons pointing at files.
        for rel in lesson.files:
            if ":" in rel and not rel.endswith((".py", ".md", ".sh", ".yaml", ".js", ".css")):
                problems.append(f"{where}: '{rel}' looks like a line reference — use the file")

    known = set(seen)
    for lesson in lessons:
        for prereq in lesson.prereqs:
            if prereq not in known:
                problems.append(f"{lesson.path.name}: prereq '{prereq}' is not a lesson")
            elif prereq == lesson.id:
                problems.append(f"{lesson.path.name}: is its own prereq")

    for cycle in _cycles({l.id: [p for p in l.prereqs if p in known] for l in lessons}):
        problems.append(f"prereq cycle: {' -> '.join(cycle)}")
    return problems


def _cycles(graph: dict[str, list[str]]) -> list[list[str]]:
    """Any prereq loop, which would lock every lesson in it forever."""
    found: list[list[str]] = []
    state: dict[str, int] = {}

    def walk(node: str, trail: list[str]):
        state[node] = 1
        for nxt in graph.get(node, []):
            if state.get(nxt) == 1:
                found.append(trail[trail.index(nxt):] + [nxt] if nxt in trail
                             else [nxt, node, nxt])
            elif state.get(nxt, 0) == 0:
                walk(nxt, trail + [nxt])
        state[node] = 2

    for node in graph:
        if state.get(node, 0) == 0:
            walk(node, [node])
    return found
