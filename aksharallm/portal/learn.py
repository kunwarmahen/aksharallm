"""The Learn tab's server side.

Thinner than the other panels, because there is no job to supervise: a lesson's check is one
pytest node and finishes in a couple of seconds, so it runs inline and answers the request
rather than being detached and polled. The quantize and eval panels shell out to a
long-running CLI; this one calls three functions in :mod:`aksharallm.learn` and hands back
what they said.

What it does own is the **shape the browser needs**: lessons, their gate state, the reader's
progress, and — for a locked lesson — the sentence saying what is missing. The same gating
idea as the post-training panel, for the same reason: a disabled button with no explanation
is indistinguishable from a broken one.

Read with: docs/15-learning-path.md -- the chapter this implements; it ends with the order to
read these files in.
"""

from __future__ import annotations

from pathlib import Path

from ..learn import check as check_mod
from ..learn import lessons as lessons_mod
from ..learn.progress import Progress, gate
from .runs import RunError, repo_root


class Learn:
    def __init__(self, root: Path | None = None):
        self.root = Path(root) if root else repo_root()

    # ---- reading -------------------------------------------------------------------------
    def status(self) -> dict:
        """Every lesson with its gate and progress — one call, because the list is small and
        the tab wants all of it to draw a single row per lesson."""
        lessons = lessons_mod.load_all(self.root)
        progress = Progress(self.root)
        gates = gate(lessons, progress)
        rows = []
        for lesson in lessons:
            entry = progress.of(lesson.id)
            rows.append({**lesson.as_dict(), **gates[lesson.id],
                         "progress": entry.as_dict()})
        done = sum(1 for r in rows if r["progress"]["complete"])
        return {
            "lessons": rows,
            "total": len(rows),
            "complete": done,
            # Where to go next: the first open, unfinished lesson. The tab opens on it, so
            # arriving at the Learn tab never means choosing from a list of thirteen.
            "next": next((r["id"] for r in rows
                          if r["open"] and not r["progress"]["complete"]), None),
            "problems": lessons_mod.validate(self.root),
        }

    def lesson(self, lesson_id: str) -> dict:
        """One lesson, with its body — the markdown the tab renders."""
        try:
            lesson = lessons_mod.get(lesson_id, self.root)
        except lessons_mod.LearnError as exc:
            raise RunError(str(exc))
        progress = Progress(self.root)
        gates = gate(lessons_mod.load_all(self.root), progress)[lesson.id]
        return {**lesson.as_dict(body=True), **gates,
                "progress": progress.of(lesson.id).as_dict()}

    # ---- doing ---------------------------------------------------------------------------
    def check(self, lesson_id: str, force: bool = False) -> dict:
        """Run a lesson's check, record the result, and say what it means.

        The `note` is the whole point of doing this here rather than in the browser: green
        after red is a finished lesson, green on its own is "you have not broken it yet", and
        red is the middle of the exercise rather than a mistake. Three outcomes, three
        different sentences.
        """
        try:
            lesson = lessons_mod.get(lesson_id, self.root)
        except lessons_mod.LearnError as exc:
            raise RunError(str(exc))
        if not lesson.verify:
            raise RunError(f"'{lesson_id}' has no check")

        progress = Progress(self.root)
        gates = gate(lessons_mod.load_all(self.root), progress)[lesson.id]
        if not gates["open"] and not force:
            raise RunError(f"'{lesson_id}' is locked — {gates['reason']}")

        try:
            result = check_mod.run(lesson.verify, self.root)
        except check_mod.CheckError as exc:
            raise RunError(str(exc))

        was_complete = progress.of(lesson_id).complete
        entry = progress.record(lesson_id, result.passed, result.summary, result.duration_s)
        if not result.passed:
            note = ("Red — which is the middle of the exercise, not a mistake. Put the code "
                    "back and run this again to finish the lesson.")
        elif entry.complete and not was_complete:
            note = "Lesson complete: you broke it and put it back."
        elif entry.complete:
            note = "Still green. This lesson is already complete."
        else:
            note = ("Green, but nothing has been broken yet — that is the exercise. This "
                    "lesson completes once the check has gone red and then green again.")
        return {"ok": True, **result.as_dict(), "note": note,
                "progress": entry.as_dict()}

    def reset(self, lesson_id: str | None = None) -> dict:
        progress = Progress(self.root)
        if lesson_id:
            try:
                lessons_mod.get(lesson_id, self.root)
            except lessons_mod.LearnError as exc:
                raise RunError(str(exc))
        progress.reset(lesson_id)
        return {"ok": True,
                "note": f"reset {lesson_id or 'every lesson'} — do it properly again."}
