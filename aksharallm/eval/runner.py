"""Running a set of suites against one checkpoint, and recording what happened.

The harness deliberately reuses :class:`~aksharallm.infer.engine.Engine` to load the model
rather than loading it itself. That buys, for free and without a second copy of the logic:

* the **device policy** — the CPU whenever a training run owns the card, with the reason
  stated, which matters more here than in the playground because an evaluation is minutes
  of sustained work rather than one generation;
* **adapters** — a LoRA adapter can be evaluated against its own base, which is the only
  way to answer "did the fine-tune help?";
* **quantized checkpoints** — an int4 `.pt` loads through the same path, so "what did
  quantization cost on MMLU, not on perplexity" is one command.

Every run writes one JSON file to `logs/eval/`. That file *is* the record: there is no
database, the portal reads the same files, and a run started from a terminal appears in the
browser. It is the same arrangement the quantization panel uses, for the same reason.

Ordering inside a run is deliberate: the cheap deterministic suites first, generation last.
An evaluation that is going to fail on a missing dataset or a bad checkpoint should fail in
the first ten seconds, not after twenty minutes of HumanEval.

Read with: docs/13-eval.md -- the chapter this implements; it ends with the order to read these
files in.
"""

from __future__ import annotations

import json
import platform
import time
from dataclasses import dataclass, field
from pathlib import Path

from ..infer.checkpoints import InferError, repo_root
from ..infer.engine import Engine, InferConfig
from . import judge as judge_mod
from . import scoring, sources, suites as suites_mod
from .report import results_dir
from .sources import EvalError

#: Suites are run in this order regardless of how they were listed: fast and certain first.
KIND_ORDER = {"ppl": 0, "mc": 1, "gen": 2, "code": 3, "judge": 4}


@dataclass
class Options:
    """Everything that changes what a run measures. Recorded in the result file verbatim,
    because a score without the settings that produced it is not comparable to anything."""

    suites: list[str] = field(default_factory=lambda: list(suites_mod.DEFAULT_SUITES))
    #: Per-suite item caps. `0` or `None` means the whole cached split.
    limit: int | None = None
    shots: int | None = None
    device: str | None = None
    adapter: str | None = None
    batch_tokens: int = scoring.DEFAULT_BATCH_TOKENS
    max_new_tokens: int = 256
    judge_model: str | None = None
    #: Keep every item's verdict in the JSON. Useful (it is how you find out *which*
    #: HumanEval problems pass) and the reason a result file is ~200 kB rather than 2 kB.
    keep_items: bool = True
    label: str | None = None

    def as_dict(self) -> dict:
        return {"suites": list(self.suites), "limit": self.limit, "shots": self.shots,
                "device": self.device, "adapter": self.adapter,
                "batch_tokens": self.batch_tokens, "max_new_tokens": self.max_new_tokens,
                "judge_model": self.judge_model, "label": self.label}


class Harness:
    """One checkpoint, several suites, one JSON file out."""

    def __init__(self, root: Path | str | None = None, engine: Engine | None = None,
                 cfg: InferConfig | None = None, log=print):
        self.root = Path(root).resolve() if root else repo_root()
        self.engine = engine or Engine(self.root, cfg=cfg)
        self.log = log

    # ---- pre-flight --------------------------------------------------------------------
    def missing_data(self, names: list[str]) -> list[str]:
        """Cached datasets a suite list needs and does not have."""
        return [n for n in suites_mod.datasets_for(names)
                if not sources.is_cached(n, self.root)]

    def preflight(self, ckpt_id: str, opts: Options) -> dict:
        """Everything checkable without loading a model. Raises on anything fatal.

        Called by the CLI *and* by the portal before it launches a job, so the browser can
        say "GSM8K is not downloaded" in the panel instead of in a log file ten seconds
        later.
        """
        names = suites_mod.resolve(opts.suites)
        info = self.engine.store.get(load_checkpoint_or_raise(self.engine, ckpt_id))
        if info.error:
            raise EvalError(f"{info.rel} cannot be loaded: {info.error}")

        notes: list[str] = []
        missing = self.missing_data(names)
        if "perplexity" in names and not self._val_bin(info):
            notes.append("perplexity: this checkpoint does not record a validation split, "
                         "so it will be skipped. Pass --val-bin to name one.")
        if "judge" in names:
            ok, why = judge_mod.available(self._judge_cfg(opts))
            if not ok:
                notes.append(f"judge: {why}")
        plan = self.engine.plan()
        return {"checkpoint": info.as_dict(), "suites": names, "missing": missing,
                "notes": notes, "device": plan.as_dict()}

    def _judge_cfg(self, opts: Options):
        cfg = judge_mod.default_config(self.root)
        if opts.judge_model:
            cfg.model = opts.judge_model
        return cfg

    def _val_bin(self, info) -> str | None:
        if not info.val_bin:
            return None
        path = Path(info.val_bin)
        if not path.is_absolute():
            path = self.root / info.val_bin
        return str(path) if path.is_file() else None

    # ---- the run -----------------------------------------------------------------------
    def run(self, ckpt_id: str, opts: Options | None = None, progress=None,
            val_bin: str | None = None) -> dict:
        opts = opts or Options()
        names = suites_mod.resolve(opts.suites)
        names.sort(key=lambda n: KIND_ORDER.get(suites_mod.get(n).kind, 9))

        missing = self.missing_data(names)
        if missing:
            raise EvalError(sources.Missing(missing).describe())

        started = time.time()
        t0 = time.monotonic()
        # `identify` turns "small-code" into "small-code/ckpt_best.pt" and a path into an id.
        # The engine wants the id; the caller should be able to type the run name.
        loaded = self.engine.load(load_checkpoint_or_raise(self.engine, ckpt_id),
                                  device=opts.device, adapter=opts.adapter)
        self.log(f"[eval] {loaded.info.rel} on {loaded.device} — {loaded.plan.reason}")
        if loaded.plan.slow:
            self.log("[eval] this will be slow. That is the correct trade: a benchmark "
                     "must never be the reason a training run dies.")

        out: dict = {
            "checkpoint": loaded.info.rel,
            "provenance": loaded.info.provenance(),
            "adapter": loaded.adapter.rel if loaded.adapter else None,
            "stage": loaded.stage,
            "device": loaded.device,
            "device_reason": loaded.plan.reason,
            "started": started,
            "host": platform.node(),
            "options": opts.as_dict(),
            "suites": {},
        }

        for name in names:
            suite = suites_mod.get(name)
            self.log(f"[eval] {name}: {suite.blurb}")
            began = time.monotonic()
            try:
                result = self._run_suite(name, loaded, opts, progress, val_bin)
            except EvalError as exc:
                # One suite failing must not lose the others. A missing judge model should
                # not cost you the MMLU number you waited twenty minutes for.
                self.log(f"[eval] {name}: skipped — {exc}")
                out["suites"][name] = {"error": str(exc), "skipped": True}
                continue
            if result is None:
                out["suites"][name] = {"skipped": True,
                                       "error": "nothing to measure for this checkpoint"}
                continue
            result["seconds"] = time.monotonic() - began
            result["kind"] = suite.kind
            result["expect"] = suite.expect
            result["baseline"] = suite.baseline
            if not opts.keep_items:
                result.pop("items", None)
            out["suites"][name] = result
            self.log(f"[eval] {name}: {describe(name, result)}  ({result['seconds']:.1f}s)")

        out["seconds"] = time.monotonic() - t0
        return out

    def _run_suite(self, name: str, loaded, opts: Options, progress, val_bin: str | None):
        suite = suites_mod.get(name)
        limit = opts.limit if opts.limit is not None else suite.default_limit
        limit = None if not limit else limit

        if suite.kind == "ppl":
            path = val_bin or self._val_bin(loaded.info)
            if not path:
                return None
            seq_len = loaded.info.seq_len or loaded.model.cfg.max_seq_len
            return scoring.perplexity(
                loaded.model, path, seq_len, n_batches=limit or 200,
                device=loaded.device, progress=progress)

        if suite.kind == "judge":
            items = suites_mod.JUDGE_PROMPTS[:limit] if limit else suites_mod.JUDGE_PROMPTS
            answers = []
            for i, item in enumerate(items):
                answers.append(self._answer(loaded, item.prompt, opts))
                if progress:
                    progress(i + 1, len(items), "judge:generate")
            return judge_mod.run(self._judge_cfg(opts), items, answers,
                                 progress=progress)

        rows = sources.load(suite.source, self.root, limit=limit, auto_fetch=False)
        shot_rows = (sources.load(suite.shot_source, self.root, auto_fetch=False)
                     if suite.shot_source else None)
        items = suites_mod.build(name, rows, shot_rows=shot_rows, shots=opts.shots)

        if suite.kind == "mc":
            return scoring.score_mc(loaded.model, loaded.tokenizer, items,
                                    device=loaded.device, batch_tokens=opts.batch_tokens,
                                    progress=progress, label=name)
        if suite.kind == "gen":
            return self._run_gen(name, items, loaded, opts, progress)
        if suite.kind == "code":
            return self._run_code(items, loaded, opts, progress)
        raise EvalError(f"suite {name!r} has an unknown kind {suite.kind!r}")

    def _answer(self, loaded, prompt: str, opts: Options) -> str:
        """One answer, in whatever form this checkpoint understands. **Judge suite only.**

        A chat model is asked through its chat template; a base model is handed the prompt
        as text. Sending ChatML to a base model produces noise — the same reason the
        Playground refuses to chat with one — so the stage decides, not the caller.

        The scope matters, because the obvious reading of that paragraph is wrong: this is
        called from `_run_judge` and nowhere else. `_run_gen` (GSM8K) and `_run_code`
        (HumanEval) also generate, and they hand the model `item.prompt` verbatim even when
        `loaded.stage` is `sft` — deliberately. GSM8K is a 5-shot chain-of-thought prompt and
        HumanEval is a signature to complete; wrapping either in ChatML makes it a *different
        benchmark*, and gotcha 1 in docs/13-eval.md is that a prompt format may not change
        without renaming the suite. Keeping them raw is what makes a base-vs-SFT comparison
        like-for-like. The judge suite is open-ended, has no base-model score worth
        preserving, and is the one measurement that is *about* being a chat model — so it is
        the one that speaks the template.

        See docs/13-eval.md § "Evaluating a chat model: what changes, and what must not".
        """
        if loaded.stage in ("sft", "dpo", "chat"):
            text = self.engine.build_prompt(loaded, "chat", prompt=prompt)
        else:
            text = prompt
        got = scoring.generate_until(loaded.model, loaded.tokenizer, text,
                                     stop=["\n\nUser:", "<|im_end|>"],
                                     max_new_tokens=opts.max_new_tokens,
                                     device=loaded.device)
        return got["text"]

    def _run_gen(self, name: str, items, loaded, opts: Options, progress) -> dict:
        correct, records = 0, []
        for i, item in enumerate(items):
            got = scoring.generate_until(
                loaded.model, loaded.tokenizer, item.prompt, stop=item.stop,
                max_new_tokens=opts.max_new_tokens, device=loaded.device)
            hit, extracted = suites_mod.gsm_correct(got["text"], item.gold)
            correct += int(hit)
            records.append({"id": item.id, "gold": item.gold, "got": extracted,
                            "correct": hit, "answer": got["text"][:1500]})
            if progress:
                progress(i + 1, len(items), name)
        n = max(1, len(items))
        return {"n": len(items), "correct": correct, "accuracy": correct / n,
                "score": correct / n, "items": records}

    def _run_code(self, items, loaded, opts: Options, progress) -> dict:
        """HumanEval, executed. `pass@1` with greedy decoding — one sample per problem.

        The generated code runs in `infer/sandbox.py`: a separate, isolated interpreter with
        CPU and memory limits and no access to this project. Read that module's docstring
        before trusting it with a model you did not train; it is honest that it is a limit,
        not a container.
        """
        from ..infer import sandbox
        from ..infer.tasks import extract_code

        ok, why = sandbox.available()
        if not ok:
            raise EvalError(f"the sandbox cannot run here ({why}), and HumanEval without "
                            "execution is not a measurement.")

        passed, records = 0, []
        for i, item in enumerate(items):
            got = scoring.generate_until(
                loaded.model, loaded.tokenizer, item.prompt,
                # A base model continues the file: it writes the function and then starts
                # another one. Everything from the next top-level statement is dropped by
                # extract_code, and these stops keep the generation from running that far.
                stop=["\ndef ", "\nclass ", "\nprint(", "\nif __name__"],
                max_new_tokens=opts.max_new_tokens, device=loaded.device)
            body = extract_code(got["text"], item.prompt, item.entry_point)
            program = (body if body.lstrip().startswith("def ")
                       else item.prompt.rstrip("\n") + "\n" + body)
            verdict = sandbox.run_program(f"{program}\n\n{item.tests}", timeout_s=10.0)
            passed += int(verdict.ok)
            records.append({"id": item.id, "passed": verdict.ok, "status": verdict.status,
                            "detail": (verdict.detail or "")[:300],
                            "code": program[:2000]})
            if progress:
                progress(i + 1, len(items), "humaneval")
        n = max(1, len(items))
        # The histogram of *how* it failed is the real signal while pass@1 is pinned at
        # zero: syntax errors give way to name errors, which give way to assertion errors,
        # which is a model that writes runnable code that is merely wrong. See the
        # `evaluate` skill for how to read this progression.
        failures: dict[str, int] = {}
        for rec in records:
            if not rec["passed"]:
                failures[rec["status"]] = failures.get(rec["status"], 0) + 1
        return {"n": len(items), "passed": passed, "pass@1": passed / n,
                "score": passed / n, "failures": dict(sorted(failures.items())),
                "items": records}

    # ---- writing it down ---------------------------------------------------------------
    def save(self, result: dict, path: Path | str | None = None) -> Path:
        run = (result.get("provenance") or {}).get("run") or "model"
        label = (result.get("options") or {}).get("label") or "eval"
        stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(result.get("started")))
        out = Path(path) if path else results_dir(self.root) / f"{stamp}-{run}-{label}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2, default=str))
        return out


def describe(name: str, result: dict) -> str:
    """One line per suite, the way the CLI and the log print it."""
    if result.get("skipped"):
        return f"skipped — {result.get('error', 'no reason recorded')}"
    kind = result.get("kind") or suites_mod.get(name).kind
    if kind == "ppl":
        return (f"perplexity {result['perplexity']:.3f}  "
                f"(loss {result['loss']:.4f} over {result['tokens']:,} tokens)")
    if kind == "mc":
        base = result.get("baseline")
        return (f"{result['acc_norm'] * 100:.1f}%  "
                f"(± {result['stderr'] * 100:.1f}, n={result['n']}"
                + (f", chance {base * 100:.0f}%" if base else "") + ")")
    if kind == "gen":
        return f"{result['accuracy'] * 100:.1f}%  ({result['correct']}/{result['n']})"
    if kind == "code":
        return f"pass@1 {result['pass@1'] * 100:.1f}%  ({result['passed']}/{result['n']})"
    if kind == "judge":
        mean = result.get("mean")
        return (f"{mean:.2f}/5 from {result.get('judge_model')} "
                f"({result.get('graded')}/{result.get('n')} graded)"
                if mean is not None else "no answers were graded")
    return json.dumps({k: v for k, v in result.items() if k != "items"})[:120]


def load_checkpoint_or_raise(engine: Engine, ref: str):
    try:
        return engine.store.identify(ref)
    except InferError as exc:
        raise EvalError(str(exc))
