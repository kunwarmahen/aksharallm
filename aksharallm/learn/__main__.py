"""The learning path, from a terminal.

    python -m aksharallm.learn                       # where you are, and what is open next
    python -m aksharallm.learn show kv-cache         # read one lesson
    python -m aksharallm.learn check kv-cache        # run its check, record the result
    python -m aksharallm.learn validate              # would any lesson lie to a reader?
    python -m aksharallm.learn reset [id]            # do a lesson properly again

The portal's Learn tab is a view over exactly these functions, and the progress file is the
same one, so a lesson checked here is ticked there.

Read with: docs/16-learning-path.md -- the chapter this implements; it ends with the order to
read these files in.
"""

from __future__ import annotations

import argparse
import sys

from .check import CheckError, run as run_check
from .lessons import LearnError, get, load_all, validate
from .progress import Progress, gate


def _bar(done: int, total: int, width: int = 24) -> str:
    filled = 0 if not total else round(width * done / total)
    return "[" + "#" * filled + "." * (width - filled) + "]"


def cmd_list(args) -> int:
    lessons = load_all(args.root)
    if not lessons:
        print("no lessons found under docs/lessons/")
        return 1
    progress = Progress(args.root)
    gates = gate(lessons, progress)
    done = sum(1 for l in lessons if progress.is_complete(l.id))

    print(f"\n  {_bar(done, len(lessons))}  {done}/{len(lessons)} complete\n")
    marks = {"complete": "[x]", "broken": "[!]", "started": "[~]", "todo": "[ ]"}
    for lesson in lessons:
        state = progress.of(lesson.id).state()
        open_ = gates[lesson.id]["open"]
        mark = marks[state] if open_ else "[-]"
        print(f"  {mark} {lesson.id:16} {lesson.title}")
        if not open_:
            print(f"      locked: {gates[lesson.id]['reason']}")
        elif state == "started":
            print("      check passes, but you have not broken it yet — that is the exercise")
        elif state == "broken":
            print("      currently red: fix it and run the check again to finish the lesson")
    print("\n  [x] done  [~] started  [!] red now  [ ] not started  [-] locked\n")
    return 0


def cmd_show(args) -> int:
    lesson = get(args.id, args.root)
    print(f"\n{'=' * 78}\n{lesson.title}\n{'=' * 78}")
    print(f"read    {lesson.doc}")
    for rel in lesson.files:
        print(f"open    {rel}")
    print(f"check   {lesson.verify}")
    if lesson.prereqs:
        print(f"after   {', '.join(lesson.prereqs)}")
    print()
    print(lesson.body)
    print()
    return 0


def cmd_check(args) -> int:
    lesson = get(args.id, args.root)
    progress = Progress(args.root)
    gates = gate(load_all(args.root), progress)[lesson.id]
    if not gates["open"] and not args.force:
        print(f"\n  '{lesson.id}' is locked: {gates['reason']}  (--force to run it anyway)\n")
        return 1

    print(f"\n  {lesson.verify}")
    try:
        result = run_check(lesson.verify, args.root)
    except CheckError as exc:
        print(f"  {exc}\n", file=sys.stderr)
        return 2

    entry = progress.record(lesson.id, result.passed, result.summary, result.duration_s)
    print(f"  {'PASS' if result.passed else 'FAIL'}  {result.summary}  "
          f"({result.duration_s:.1f}s)")
    if not result.passed:
        print("\n" + "\n".join("  " + ln for ln in result.output.splitlines()[-12:]))
        print("\n  Red is the middle of the exercise, not a mistake. Put the code back and "
              "run this again.\n")
    elif entry.complete and entry.completed_at:
        print("\n  Lesson complete — you broke it and put it back.\n")
    else:
        print("\n  Green, but the exercise has not happened yet: this lesson completes once "
              "the check has gone red and then green again.\n")
    return 0 if result.passed else 1


def cmd_validate(args) -> int:
    problems = validate(args.root)
    if not problems:
        lessons = load_all(args.root)
        print(f"\n  {len(lessons)} lessons, no problems.\n")
        return 0
    print(f"\n  {len(problems)} problem(s):")
    for p in problems:
        print(f"    {p}")
    print()
    return 1


def cmd_reset(args) -> int:
    progress = Progress(args.root)
    progress.reset(args.id)
    print(f"reset {args.id or 'every lesson'}.")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m aksharallm.learn",
                                 description="The repo as a course: read, break, verify.")
    ap.add_argument("--root", default=None)
    sub = ap.add_subparsers(dest="cmd")
    sub.add_parser("list", help="every lesson and where you are")
    show = sub.add_parser("show", help="read one lesson")
    show.add_argument("id")
    check = sub.add_parser("check", help="run a lesson's check and record it")
    check.add_argument("id")
    check.add_argument("--force", action="store_true", help="ignore the prereq gate")
    sub.add_parser("validate", help="check the lessons against the code they describe")
    reset = sub.add_parser("reset", help="forget progress for one lesson, or all")
    reset.add_argument("id", nargs="?")

    args = ap.parse_args(argv)
    try:
        return {"list": cmd_list, "show": cmd_show, "check": cmd_check,
                "validate": cmd_validate, "reset": cmd_reset,
                None: cmd_list}[args.cmd](args)
    except LearnError as exc:
        print(f"\n  {exc}\n", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
