"""The eval harness: is the model actually any good?

Validation loss answers "is training working". It cannot answer "is the model better than
last week at anything a person would notice", and it is actively misleading for the three
things this project builds next — mixture-of-experts, synthetic data and distillation all
change quality in ways a loss curve either cannot see or reports backwards. That is why the
harness was built before them rather than after.

Five kinds of measurement, in increasing order of how much they tell you and how long they
take:

    perplexity            held-out loss. Smooth, cheap, not comparable across tokenizers.
    multiple choice       MMLU, ARC, HellaSwag, PIQA — scored by log-likelihood, never
                          generated, so the number is exactly reproducible.
    generative            GSM8K — the model writes an answer and a regex checks the number.
    executed              HumanEval — the model writes code and it is run against tests.
    judged                twelve open-ended prompts graded 1-5 by a local Ollama model.

    aksharallm.eval.sources    downloading and caching the data (data/eval/)
    aksharallm.eval.suites     what each benchmark asks, and how it is scored
    aksharallm.eval.scoring    the model-side primitives (log-likelihood, greedy decode)
    aksharallm.eval.judge      the LLM-judge
    aksharallm.eval.runner     running suites against a checkpoint
    aksharallm.eval.report     every result so far, and the deltas between them

    python -m aksharallm.eval suites          # start here

The submodules are imported lazily, through `__getattr__`: `aksharallm.eval.sources` must
be importable to *list* the benchmarks without pulling in torch, which the portal does on
every page load.

Read with: docs/12-eval.md -- the chapter this implements; it ends with the order to read these
files in.
"""

from __future__ import annotations

_LAZY = {
    "Harness": "runner", "Options": "runner", "describe": "runner",
    "Results": "report", "compare_table": "report", "summary_table": "report",
    "EvalError": "sources",
    "SUITES": "suites", "catalogue": "suites", "resolve": "suites",
}

__all__ = list(_LAZY)


def __getattr__(name: str):
    import importlib

    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(f".{module}", __name__), name)
