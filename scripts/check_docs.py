#!/usr/bin/env python
"""Is the writing still true? One command, run before calling any change done.

The tests already pin most of this — `tests/test_docs.py` for the chapter/module pointers,
`tests/test_journeys.py` for the route map's commands, `tests/test_portal_css.py` for the
columns that only look scrollable. This script runs those *and* the checks that live outside
pytest because they cover files pytest does not own: the local, gitignored `.claude/skills/`
playbooks, which rot silently because nothing imports them and nobody greps them.

That gap was not hypothetical. The first run of this script found **twenty broken links**
across sixteen skills -- every one a `../docs/...` written from the wrong depth -- and a
"nineteen lessons" claim in four files two lessons after the count changed.

    python scripts/check_docs.py            # everything
    python scripts/check_docs.py --fast     # skip pytest, just the prose checks

Read with: docs/09-running-and-watching.md -- the chapter on running things here.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKILLS = ROOT / ".claude" / "skills"
DOCS = ROOT / "docs"

#: The pytest files that already guard the writing. Run as one invocation: they are fast,
#: and a partial answer is what lets the next thing slip through.
GUARD_TESTS = ["tests/test_docs.py", "tests/test_journeys.py",
               "tests/test_lessons.py", "tests/test_portal_css.py"]

LINK = re.compile(r"\[([^\]]*)\]\(([^)]+)\)")

WORDS = {"nineteen": 19, "twenty": 20, "twenty-one": 21, "twenty-two": 22,
         "twenty-three": 23, "twenty-four": 24, "twelve": 12, "thirteen": 13,
         "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18}


def problems_in_skill_links() -> list[str]:
    """Every relative link in a skill must resolve.

    Skills live at `.claude/skills/<name>/SKILL.md`, so the repo root is `../../../` — and
    the depth was written three different ways across sixteen files. Nothing else notices:
    a skill is prose loaded into a prompt, so a dead link is a dead end at exactly the
    moment someone is trying to follow the playbook.
    """
    out = []
    for skill in sorted(SKILLS.glob("*/SKILL.md")):
        for _, target in LINK.findall(skill.read_text()):
            base = target.split("#")[0]
            if not base or base.startswith(("http", "mailto")):
                continue
            if not (skill.parent / base).resolve().exists():
                out.append(f"{skill.parent.name}/SKILL.md -> {base}")
    return out


def problems_in_lesson_count() -> list[str]:
    """A spelled-out count of the lessons has to match how many there are.

    Prose counts are the classic silent drift: the course went from nineteen to twenty-one
    when the audio lessons landed, and four files went on saying nineteen -- including the
    README table and the `learn` skill's own description, which is what the model reads to
    decide whether the skill applies.
    """
    real = len(list((DOCS / "lessons").glob("*.md")))
    if not real:
        return ["docs/lessons/ is empty — has it moved?"]
    out = []
    pattern = re.compile(rf"({'|'.join(WORDS)})[ -]lessons", re.I)
    for path in [*DOCS.glob("*.md"), ROOT / "README.md", *SKILLS.glob("*/SKILL.md")]:
        for found in pattern.finditer(path.read_text(errors="replace")):
            said = WORDS[found.group(1).lower()]
            if said != real:
                out.append(f"{path.relative_to(ROOT)} says {found.group(0)!r}, "
                           f"but docs/lessons/ has {real}")
    return out


def problems_in_skill_doc_refs() -> list[str]:
    """A `docs/NN-name.md` named in a skill's prose must exist, link or not."""
    out = []
    ref = re.compile(r"docs/(\d\d-[a-z-]+\.md)")
    for skill in sorted(SKILLS.glob("*/SKILL.md")):
        for name in set(ref.findall(skill.read_text())):
            if not (DOCS / name).is_file():
                out.append(f"{skill.parent.name}/SKILL.md names docs/{name}, which is gone")
    return out


def problems_in_chapter_coverage() -> list[str]:
    """Every chapter should be reachable from the README's table, or a reader never finds it."""
    listed = set(re.findall(r"\(docs/(\d\d-[a-z-]+\.md)\)", (ROOT / "README.md").read_text()))
    return [f"docs/{p.name} is in docs/ but not in the README table"
            for p in sorted(DOCS.glob("[0-9][0-9]-*.md")) if p.name not in listed]


CHECKS = [
    ("skill links resolve", problems_in_skill_links),
    ("skills name real chapters", problems_in_skill_doc_refs),
    ("lesson counts agree", problems_in_lesson_count),
    ("every chapter is in the README", problems_in_chapter_coverage),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--fast", action="store_true", help="skip pytest, prose checks only")
    args = ap.parse_args()

    failed = 0
    for name, check in CHECKS:
        found = check()
        print(f"  {'FAIL' if found else 'ok  '}  {name}")
        for line in found:
            print(f"          {line}")
        failed += bool(found)

    if not args.fast:
        print(f"\n  running {' '.join(GUARD_TESTS)}")
        proc = subprocess.run([sys.executable, "-m", "pytest", *GUARD_TESTS, "-q",
                               "--no-header"], cwd=ROOT)
        failed += proc.returncode != 0

    print("\n  the writing still matches the code.\n" if not failed
          else f"\n  {failed} check(s) failed — fix before calling the change done.\n")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
