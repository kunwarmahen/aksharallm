"""Running code the model wrote, without letting it run away with the machine.

This is the only part of the project that *executes* a generation instead of reading it,
and that deserves to be stated plainly: **a language model writes a program and this module
runs it on your computer.** That is the point — "does it code?" has an actual answer, and
reading a plausible-looking function is not that answer — but it is worth knowing exactly
what the containment is and what it is not.

What it is:

* a **separate process**, so an infinite loop, a segfault or `sys.exit(1)` costs a
  subprocess and nothing else;
* **`-I` isolated mode** — no `PYTHONPATH`, no `site-packages` from the user, no current
  directory on `sys.path`, so `import aksharallm` fails and there is nothing of this
  project's to reach;
* a **CPU-time limit** via `RLIMIT_CPU`, which a `while True:` cannot escape the way it can
  escape a wall-clock timeout it never yields to;
* an **address-space limit** via `RLIMIT_AS`, so `[0] * 10**12` raises `MemoryError`
  instead of taking the machine into swap while a training run is on it;
* **no new processes** (`RLIMIT_NPROC`) and **no core dumps**;
* a **fresh empty temp directory** as the working directory, removed afterwards, so a file
  the program writes lands there and nowhere near the repo;
* a **stripped environment**, so nothing in the process inherits credentials.

What it is **not**: a security boundary. It is not a container, not a namespace, not seccomp.
The code cannot loop forever, exhaust memory, fork-bomb or scribble on the repo, and it
cannot import this project — but it *can* open a socket, and a determined program could read
a world-readable file. That is an acceptable trade for running a 300M model's attempt at
`is_palindrome`, and it would not be acceptable for running a stranger's code. If this is
ever pointed at output from a model you did not train, put it in a container first.

Turn the whole thing off with `infer.run_tests: false` in `configs/portal.yaml`; the code
tab still generates and shows the function, it just never runs it.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

#: Prepended to every program. Unbuffered-ish output and a recursion cap, so a model that
#: writes the naive recursive `fibonacci` and is asked for n=10,000 raises rather than
#: dying by C-stack overflow (which would look like a crash rather than a wrong answer).
PREAMBLE = "import sys\nsys.setrecursionlimit(2000)\n"

#: Hard ceiling on captured output. A program that prints in a loop until its CPU limit
#: fires can produce hundreds of MB, and none of it is worth reading.
MAX_OUTPUT = 16_000


@dataclass
class Result:
    """What happened when the program ran."""

    ok: bool
    status: str            # pass | fail | error | timeout | syntax | disabled | unsupported
    detail: str
    stdout: str = ""
    stderr: str = ""
    duration_s: float = 0.0
    program: str = field(default="", repr=False)

    def as_dict(self) -> dict:
        return {"ok": self.ok, "status": self.status, "detail": self.detail,
                "stdout": self.stdout, "stderr": self.stderr,
                "duration_s": self.duration_s, "program": self.program}


def _limits(cpu_s: int, memory_mb: int):
    """The `preexec_fn` that shrinks the child before it runs a line of Python.

    Runs in the forked child between `fork` and `exec`, which is why it only uses
    async-signal-safe calls. `resource` is POSIX-only; on anything else the caller has
    already refused to run at all.
    """
    import resource

    def apply():
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_s, cpu_s))
        as_bytes = memory_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (as_bytes, as_bytes))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        resource.setrlimit(resource.RLIMIT_FSIZE, (8 * 1024 * 1024, 8 * 1024 * 1024))
        try:
            resource.setrlimit(resource.RLIMIT_NPROC, (0, 0))
        except (ValueError, OSError):
            pass                      # not enforceable everywhere; the rest still holds
        os.setsid()                   # own process group, so a timeout kills the whole tree

    return apply


def available() -> tuple[bool, str]:
    """Whether this machine can run the sandbox at all."""
    if os.name != "posix":
        return False, ("running generated code needs POSIX resource limits, which this "
                       "platform does not have. Generation still works; the tests do not "
                       "run.")
    try:
        import resource  # noqa: F401
    except ImportError:
        return False, "the `resource` module is unavailable, so limits cannot be applied."
    return True, ""


def run_program(program: str, timeout_s: float = 10.0, memory_mb: int = 512,
                enabled: bool = True) -> Result:
    """Execute `program` under the limits above and report what happened.

    The exit code carries the verdict: an `assert` that fails raises `AssertionError`, which
    is a non-zero exit and a traceback on stderr. So "did every test pass" is simply
    "did it exit 0", and the traceback's last line is the sentence worth showing.
    """
    if not enabled:
        return Result(False, "disabled", "running generated code is turned off "
                                         "(infer.run_tests: false in configs/portal.yaml).")
    ok, why = available()
    if not ok:
        return Result(False, "unsupported", why)

    full = PREAMBLE + program
    try:
        compile(full, "<model>", "exec")
    except SyntaxError as exc:
        # Worth catching here rather than in the child: for a base model this is the single
        # most common outcome, and "SyntaxError: line 4" is a more useful answer than a
        # traceback from a subprocess.
        return Result(False, "syntax", f"the generated code is not valid Python: {exc.msg} "
                                       f"(line {exc.lineno})", program=program)

    workdir = tempfile.mkdtemp(prefix="aksharallm-sandbox-")
    script = Path(workdir) / "program.py"
    script.write_text(full)
    # -I: isolated. No PYTHONPATH, no user site-packages, no cwd on sys.path. -S skips
    # site.py entirely. Together they mean the child cannot import anything of ours.
    argv = [sys.executable, "-I", "-S", str(script)]
    # A minimal environment: enough to run, nothing to leak.
    env = {"PATH": "/usr/bin:/bin", "HOME": workdir, "TMPDIR": workdir,
           "LC_ALL": "C.UTF-8", "PYTHONIOENCODING": "utf-8", "PYTHONDONTWRITEBYTECODE": "1"}

    import time
    t0 = time.monotonic()
    try:
        proc = subprocess.run(
            argv, cwd=workdir, env=env, capture_output=True, text=True,
            stdin=subprocess.DEVNULL, timeout=timeout_s,
            # CPU seconds are rounded up from the wall-clock budget: the wall-clock timeout
            # catches a program that sleeps, RLIMIT_CPU catches one that spins.
            preexec_fn=_limits(max(1, int(timeout_s) + 1), memory_mb))
        out, err, code = proc.stdout, proc.stderr, proc.returncode
    except subprocess.TimeoutExpired as exc:
        return Result(False, "timeout",
                      f"the program was still running after {timeout_s:.0f}s and was killed "
                      "— an infinite loop, or an algorithm too slow for the test.",
                      stdout=_trim(exc.stdout), stderr=_trim(exc.stderr),
                      duration_s=timeout_s, program=program)
    except (OSError, ValueError) as exc:
        return Result(False, "error", f"could not start the sandbox: {exc}", program=program)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    duration = time.monotonic() - t0
    out, err = _trim(out), _trim(err)
    if code == 0:
        return Result(True, "pass", "every assertion passed.", stdout=out, stderr=err,
                      duration_s=duration, program=program)

    last = _last_line(err)
    if "AssertionError" in err:
        return Result(False, "fail", f"an assertion failed — the function runs but is "
                                     f"wrong. {last}", stdout=out, stderr=err,
                      duration_s=duration, program=program)
    if "MemoryError" in err:
        return Result(False, "error", f"the program ran out of its {memory_mb} MB budget.",
                      stdout=out, stderr=err, duration_s=duration, program=program)
    if code < 0:
        return Result(False, "timeout", f"killed by signal {-code} — usually the CPU-time "
                                        "limit catching an infinite loop.",
                      stdout=out, stderr=err, duration_s=duration, program=program)
    return Result(False, "error", last or f"exited with status {code}.", stdout=out,
                  stderr=err, duration_s=duration, program=program)


def _trim(text: str | bytes | None) -> str:
    if not text:
        return ""
    if isinstance(text, bytes):
        text = text.decode(errors="replace")
    if len(text) <= MAX_OUTPUT:
        return text
    return text[:MAX_OUTPUT] + f"\n… {len(text) - MAX_OUTPUT} more characters not shown"


def _last_line(text: str) -> str:
    """The final line of a traceback — the exception and its message, which is the part
    a person reads."""
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    return lines[-1] if lines else ""


def run_task(task, completion: str, chat: bool = False, timeout_s: float = 10.0,
             memory_mb: int = 512, enabled: bool = True) -> Result:
    """Grade one `tasks.CodeTask` against what the model produced."""
    from .tasks import assemble
    return run_program(assemble(task, completion, chat=chat), timeout_s=timeout_s,
                       memory_mb=memory_mb, enabled=enabled)
