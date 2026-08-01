"""Who has done what — and the rule that makes "done" mean something.

A lesson here is complete when its check has been seen **red and then green**: failing once,
and passing afterwards. Not "the test passes" — the test passes on a clean checkout, so that
would complete the entire path for someone who never opened a file.

The exercise *is* breaking the code. Requiring the red run means the record says you broke it
and put it back, which is the only evidence this design can actually collect. It also fails
in the safe direction: someone who genuinely did the work and then reverted before pressing
the button just presses it once more.

    run 1: green   -> attempted, not complete   ("break it first")
    run 2: RED     -> the exercise happened
    run 3: green   -> complete

State lives in `learning/progress.json` — outside `docs/`, because it is personal and
changes constantly, and gitignored for the same reason a shell history is. Losing it costs
some ticks in a list, so it is written whole and atomically rather than carefully.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

#: A long tail of runs is not interesting; the shape of the last few is.
MAX_RUNS = 20


@dataclass
class LessonProgress:
    id: str
    runs: list[dict] = field(default_factory=list)
    completed_at: float | None = None

    @property
    def attempts(self) -> int:
        return len(self.runs)

    @property
    def seen_red(self) -> bool:
        return any(not r.get("passed") for r in self.runs)

    @property
    def last_passed(self) -> bool | None:
        return bool(self.runs[-1]["passed"]) if self.runs else None

    @property
    def complete(self) -> bool:
        return self.completed_at is not None

    def state(self) -> str:
        """One word for the list. `started` covers "green but never broken", which is the
        state the red-then-green rule exists to distinguish."""
        if self.complete:
            return "complete"
        if not self.runs:
            return "todo"
        return "broken" if self.last_passed is False else "started"

    def as_dict(self) -> dict:
        return {"id": self.id, "runs": self.runs[-MAX_RUNS:], "attempts": self.attempts,
                "seen_red": self.seen_red, "last_passed": self.last_passed,
                "complete": self.complete, "completed_at": self.completed_at,
                "state": self.state()}


class Progress:
    """`learning/progress.json`, read and written whole."""

    def __init__(self, root: Path | None = None):
        from ..portal.runs import repo_root

        self.root = Path(root) if root else repo_root()
        self.path = self.root / "learning" / "progress.json"
        self._data = self._read()

    def _read(self) -> dict:
        try:
            data = json.loads(self.path.read_text())
        except (OSError, ValueError):
            return {"lessons": {}}
        if not isinstance(data, dict) or not isinstance(data.get("lessons"), dict):
            return {"lessons": {}}
        return data

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self._data, indent=2))
        tmp.replace(self.path)

    # ---- reading -------------------------------------------------------------------------
    def of(self, lesson_id: str) -> LessonProgress:
        raw = self._data["lessons"].get(lesson_id) or {}
        return LessonProgress(id=lesson_id, runs=list(raw.get("runs") or []),
                              completed_at=raw.get("completed_at"))

    def all(self) -> dict[str, LessonProgress]:
        return {k: self.of(k) for k in self._data["lessons"]}

    def is_complete(self, lesson_id: str) -> bool:
        return self.of(lesson_id).complete

    # ---- writing -------------------------------------------------------------------------
    def record(self, lesson_id: str, passed: bool, detail: str = "",
               duration_s: float = 0.0) -> LessonProgress:
        """Add a check run, and complete the lesson if this green follows a red.

        Completion is decided here rather than at read time so the moment is *stamped* — a
        rule that changes later must not silently un-complete somebody's finished lesson.
        """
        entry = self.of(lesson_id)
        entry.runs.append({"when": time.time(), "passed": bool(passed),
                           "detail": detail[:400], "duration_s": round(duration_s, 2)})
        entry.runs = entry.runs[-MAX_RUNS:]
        if passed and entry.seen_red and not entry.complete:
            entry.completed_at = time.time()
        self._data["lessons"][lesson_id] = {"runs": entry.runs,
                                            "completed_at": entry.completed_at}
        self.save()
        return entry

    def reset(self, lesson_id: str | None = None) -> None:
        """Forget one lesson, or all of them. Deliberately available: someone coming back to
        a chapter a year later should be able to do it properly again."""
        if lesson_id is None:
            self._data = {"lessons": {}}
        else:
            self._data["lessons"].pop(lesson_id, None)
        self.save()


def gate(lessons, progress: Progress) -> dict[str, dict]:
    """Which lessons are open, and for the locked ones, exactly what is missing.

    The same shape as the post-training panel's gating, and for the same reason: a disabled
    button with no explanation is indistinguishable from a broken one.
    """
    done = {lid for lid, p in progress.all().items() if p.complete}
    out = {}
    for lesson in lessons:
        missing = [p for p in lesson.prereqs if p not in done]
        titles = {l.id: l.title for l in lessons}
        out[lesson.id] = {
            "open": not missing,
            "missing": missing,
            "reason": None if not missing else
            ("finish " + ", ".join(f"'{titles.get(m, m)}'" for m in missing) + " first"),
        }
    return out
