"""The route map's commands must be real, because the route map exists to be copied.

`docs/22-journeys.md` is the page a newcomer follows from raw text to a working model. Its
entire value is that the commands in it can be pasted. One invented flag costs the reader
their trust in the whole page, and they have no way to tell which line was the wrong one.

`tests/test_docs.py` keeps the *links* honest. Nothing kept the *commands* honest, and the
first draft of chapter 22 shipped three that do not work:

  - `scripts/serve.sh small-code --speculate 4` — speculation is `SPECULATE=`, an env var,
    and serve.sh treats an unknown `-*` as fatal, so this exits 2 rather than being ignored;
  - `python -m aksharallm.vision caption` — the required `checkpoint` positional is missing;
  - `python -m aksharallm.learn show 01` — lesson ids are names (`data`), never numbers.

All three look right. Two of them are wrong only in the interface, which is exactly what a
human reviewer skims past. So they are checked here against the real `--help`.

Read with: docs/22-journeys.md -- the chapter this guards.
"""

from __future__ import annotations

import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CHAPTER = ROOT / "docs" / "22-journeys.md"

#: A placeholder the reader is meant to substitute, not a literal to check.
PLACEHOLDER = re.compile(r"^<.*>$")


def commands() -> list[str]:
    """Every shell line in the chapter's ```bash blocks, continuations joined."""
    out: list[str] = []
    for block in re.findall(r"```bash\n(.*?)```", CHAPTER.read_text(), re.S):
        joined = block.replace("\\\n", " ")
        for line in joined.splitlines():
            line = re.sub(r"\s+#.*$", "", line).strip()
            if line and not line.startswith("#"):
                out.append(line)
    return out


def inline_commands() -> list[str]:
    """Commands quoted in prose and tables, e.g. `python -m aksharallm.eval domains x`.

    The tables were where two of the three broken commands lived — a table cell reads as
    reference material, so it gets skimmed harder than a code block.
    """
    text = CHAPTER.read_text()
    found = re.findall(r"`(python -m aksharallm\.[^`]+|scripts/[a-z_]+\.sh[^`]*)`", text)
    return [f.strip() for f in found]


def split_env(cmd: str) -> list[str]:
    """Drop leading `VAR=value` assignments — `SPECULATE=4 scripts/serve.sh ...`."""
    parts = cmd.split()
    while parts and re.match(r"^[A-Z_][A-Z0-9_]*=", parts[0]):
        parts.pop(0)
    return parts


def required_positionals(usage: str) -> list[str]:
    """The positionals argparse would refuse to run without, read off its usage line.

    Two things here are not obvious, and both produced false failures first time round:

    - **The program prefix is everything before the first `[`.** Trying to strip it by
      name counted the subcommand as a positional, so `eval run [-h] ... checkpoint` looked
      like it needed two arguments when it needs one.
    - **A required *option* still appears unbracketed**, metavar and all:
      `prepare.py [-h] --out-dir OUT_DIR ... {tinystories,...}`. Counting `OUT_DIR` as a
      positional made a correct command look short of an argument.

    `{a,b,c}` groups survive as one token on purpose — a subcommand slot must still be
    filled by something.
    """
    body = usage[usage.index("[") :] if "[" in usage else usage
    out, depth = [], 0
    for ch in body:
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth = max(0, depth - 1)
        elif depth == 0:
            out.append(ch)

    tokens, skip = [], False
    for tok in "".join(out).split():
        if skip:                       # the metavar belonging to the option before it
            skip = False
            continue
        if tok.startswith("-"):
            skip = True                # `--out-dir OUT_DIR`
            continue
        tokens.append(tok)
    return [t for t in tokens if t != "..."]


def help_for(argv: list[str]) -> tuple[int, str]:
    proc = subprocess.run([sys.executable, "-m", argv[0], *argv[1:], "--help"],
                          cwd=ROOT, capture_output=True, text=True, timeout=180)
    return proc.returncode, proc.stdout + proc.stderr


def usage_block(text: str) -> str:
    m = re.search(r"usage:(.*?)(\n\n|\noptions:|\npositional)", text, re.S)
    return m.group(1) if m else ""


ALL = sorted(set(commands() + inline_commands()))
PY_CMDS = [c for c in ALL if split_env(c)[:2] == ["python", "-m"]]
SH_CMDS = [c for c in ALL if split_env(c) and split_env(c)[0].startswith("scripts/")]


def test_the_chapter_actually_contains_commands():
    """If the extraction broke, every test below would pass on an empty list."""
    assert len(PY_CMDS) >= 10, PY_CMDS
    assert len(SH_CMDS) >= 4, SH_CMDS


@pytest.mark.parametrize("cmd", SH_CMDS)
def test_a_script_exists_and_handles_every_flag_it_is_given(cmd: str):
    """`scripts/serve.sh` ends an unknown `-*` with `exit 2`, so a flag it does not parse is
    not a harmless extra — the command in the docs simply fails."""
    parts = split_env(cmd)
    script = ROOT / parts[0]
    assert script.is_file(), f"{parts[0]} does not exist"
    body = script.read_text()

    # Must be matched by the argument-parsing `case`, NOT merely present somewhere in the
    # file. `--speculate` appears in serve.sh — in the line that builds the *python* command
    # from `$SPECULATE` — so a substring test passes while the script still rejects the flag.
    # That is exactly how the bad command survived review the first time.
    handled = set()
    for case_body in re.findall(r"case\s+\"?\$\w+\"?\s+in(.*?)esac", body, re.S):
        for pattern in re.findall(r"^\s*([^)\n]+?)\)", case_body, re.M):
            handled.update(p.strip() for p in pattern.split("|"))
    # `-*)` and `*)` are the fall-through arms — in serve.sh `-*)` is precisely the one that
    # prints "unknown flag" and exits 2. Counting a catch-all as a handler would make this
    # test pass for every flag ever written, which is worse than not having it.
    handled -= {"*", "-*"}

    for flag in [p for p in parts[1:] if p.startswith("-")]:
        assert flag in handled, (
            f"{parts[0]} does not parse {flag} — documented in docs/22-journeys.md as "
            f"`{cmd}`, but its `case` handles {sorted(handled)}. Check whether it is an "
            f"environment variable instead (e.g. SPECULATE=4).")


def test_a_documented_lesson_id_is_a_real_lesson():
    """`learn show` takes a free-text id, so argparse accepts anything and the tests above
    cannot see the difference. The chapter first said `learn show 01`; lesson ids are names
    (`data`, `tokenizer`), and a number gets a "no such lesson" for a reader's first move
    into the course."""
    real = {lesson.id for lesson in __import__(
        "aksharallm.learn.lessons", fromlist=["load_all"]).load_all()}
    assert real, "no lessons found — has the loader moved?"
    for cmd in ALL:
        parts = split_env(cmd)
        if parts[:3] != ["python", "-m", "aksharallm.learn"]:
            continue
        if len(parts) > 4 and parts[3] in ("show", "check"):
            assert parts[4] in real, (
                f"`{cmd}` names lesson {parts[4]!r}, which does not exist. "
                f"Ids are names, not numbers — e.g. {sorted(real)[:3]}")


@pytest.fixture(scope="module")
def helps() -> dict[str, tuple[int, str]]:
    """`--help` for every documented python command, fetched in parallel.

    Each one costs ~1.2s of import, so serially this would add half a minute to the suite;
    together they land in a few. `--help` runs no model and writes nothing.
    """
    jobs = {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        for cmd in PY_CMDS:
            parts = split_env(cmd)[2:]
            sub = [p for p in parts[1:] if not p.startswith("-")
                   and not PLACEHOLDER.match(p)]
            # Ask for help at the subcommand level when the first word is one.
            jobs[cmd] = pool.submit(help_for, [parts[0]] + sub[:1])
    return {k: v.result() for k, v in jobs.items()}


@pytest.mark.parametrize("cmd", PY_CMDS)
def test_a_documented_command_names_a_real_module_and_subcommand(cmd: str, helps):
    code, text = helps[cmd]
    assert "No module named" not in text, f"`{cmd}`: module does not exist"
    assert "invalid choice" not in text, (
        f"`{cmd}`: not a real subcommand.\n{text.strip().splitlines()[-1] if text else ''}")
    assert code == 0, f"`{cmd}`: --help failed\n{text[-400:]}"


@pytest.mark.parametrize("cmd", PY_CMDS)
def test_a_documented_command_supplies_every_required_argument(cmd: str, helps):
    """`python -m aksharallm.vision caption` is missing its checkpoint. argparse would exit
    2; a reader would assume the tool was broken rather than the doc."""
    _, text = helps[cmd]
    usage = usage_block(text)
    if not usage:
        pytest.skip("no usage line to read")
    parts = split_env(cmd)[2:]
    # Values belonging to the doc's own options (`--out-dir data/blend`) are not positionals.
    given, skip = [], False
    for tok in parts[1:]:
        if skip:
            skip = False
            continue
        if tok.startswith("-"):
            skip = "=" not in tok
            continue
        given.append(tok)

    need = required_positionals(usage)
    # `--help` was asked at the subcommand level, so its usage no longer lists the
    # subcommand word — but the doc's command still spends one token on it.
    if given and need and not any(n.startswith("{") for n in need):
        sub = split_env(cmd)[2:][1:2]
        if sub and sub[0] == given[0] and helps_subcommand(cmd):
            given = given[1:]

    assert len(given) >= len(need), (
        f"`{cmd}` is missing a required argument.\nusage:{usage.strip()}\n"
        f"needs {need}, the doc supplies {given}")


def helps_subcommand(cmd: str) -> bool:
    """Did we ask `--help` of a subcommand rather than of the module itself?"""
    parts = split_env(cmd)[2:]
    first = [p for p in parts[1:] if not p.startswith("-")]
    if not first:
        return False
    code, text = help_for([parts[0]])
    return re.search(rf"^\s+{re.escape(first[0])}\b", text, re.M) is not None
