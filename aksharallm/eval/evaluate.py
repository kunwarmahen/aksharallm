"""The old entry point. Kept so that `python -m aksharallm.eval.evaluate <ckpt>` still
works, and forwards to the harness.

This module used to *be* the evaluation — 150 lines of perplexity and HellaSwag. The real
harness replaced it (see `aksharallm/eval/__init__.py` for the map), and the one thing worth
preserving was the command anyone had in their shell history. It translates the old flags
and hands over; there is no second implementation behind it.

    old:  python -m aksharallm.eval.evaluate small-code --tasks perplexity,hellaswag
    new:  python -m aksharallm.eval small-code --suite perplexity,hellaswag

Read with: docs/13-eval.md -- the chapter this implements; it ends with the order to read these
files in.
"""

from __future__ import annotations

import sys

#: Old `--tasks` names to suite names. `samples` has no equivalent and never should: a
#: benchmark suite that prints three generations is a playground, and there is one of those
#: already (`python -m aksharallm.infer.cli <run> --probes`), with a history file.
TASK_MAP = {"perplexity": "perplexity", "hellaswag": "hellaswag"}


def main(argv: list[str] | None = None) -> int:
    from .__main__ import main as harness_main

    argv = list(sys.argv[1:] if argv is None else argv)
    out: list[str] = []
    dropped: list[str] = []
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "--tasks" and i + 1 < len(argv):
            names, unknown = [], []
            for task in argv[i + 1].split(","):
                task = task.strip()
                (names if task in TASK_MAP else unknown).append(TASK_MAP.get(task, task))
            if names:
                out += ["--suite", ",".join(names)]
            dropped += unknown
            i += 2
            continue
        if arg == "--n-batches" and i + 1 < len(argv):
            out += ["--limit", argv[i + 1]]
            i += 2
            continue
        if arg == "--out" and i + 1 < len(argv):
            out += ["--json", argv[i + 1]]
            i += 2
            continue
        if arg == "--tokenizer" and i + 1 < len(argv):
            # The harness reads the tokenizer the checkpoint recorded and refuses to guess:
            # the BPE vocabulary *is* the embedding index, so the wrong one produces fluent
            # nonsense rather than an error.
            print("  note: --tokenizer is ignored; the harness uses the one the checkpoint "
                  "was trained with.", file=sys.stderr)
            i += 2
            continue
        out.append(arg)
        i += 1

    if dropped:
        print(f"  note: dropped '{', '.join(dropped)}' — no longer a suite. "
              "See:  python -m aksharallm.eval suites", file=sys.stderr)
    print("  (this entry point now forwards to `python -m aksharallm.eval`)",
          file=sys.stderr)
    return harness_main(out)


if __name__ == "__main__":
    raise SystemExit(main())
