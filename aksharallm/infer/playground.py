"""Ask a checkpoint something, grade the answer, keep the record.

`engine.py` knows how to make tokens. `tasks.py` knows what to ask. `sandbox.py` knows how
to run what came back. `history.py` knows how to remember it. This is the one object that
uses all four, so that the terminal and the browser do the same thing in the same order —
the same rule the portal already follows for starting and stopping runs.

The order matters and is always this:

    generate  ->  (code mode) run the tests  ->  write the history record

The tests run *before* the record is written, because a record of a code generation without
its verdict is the least useful kind: "the model wrote this" is worth much less than "the
model wrote this and it passed".

Read with: docs/06-inference.md -- the chapter this implements; it ends with the order to read
these files in.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterator

from . import sandbox
from .checkpoints import CheckpointStore, InferError, repo_root
from .engine import Engine, InferConfig, SamplingParams
from .history import History, record_from
from .tasks import TASKS_BY_ID, catalogue


class Playground:
    """Everything the Playground tab and `aksharallm.infer.cli` need, in one place."""

    def __init__(self, root: Path | str | None = None, cfg: InferConfig | None = None,
                 busy_cb: Callable[[], list[str]] | None = None):
        self.root = Path(root).resolve() if root else repo_root()
        self.cfg = cfg or InferConfig.load(self.root)
        self.engine = Engine(self.root, cfg=self.cfg, busy_cb=busy_cb)
        self.store: CheckpointStore = self.engine.store
        self.adapters = self.engine.adapters
        self.history = History(self.root, max_records=self.cfg.history_max)

    # ---- what is available -------------------------------------------------------------
    def overview(self) -> dict:
        """One response describing everything the tab needs to render itself."""
        self.cfg.reload_if_changed()
        self.history.max_records = self.cfg.history_max
        checkpoints = [c.as_dict() for c in self.store.list()]
        default = self.store.default()
        sandbox_ok, sandbox_why = sandbox.available()
        return {
            "checkpoints": checkpoints,
            # Adapters are offered alongside checkpoints rather than instead of them: the
            # picker is "which model" + "which specialisation on top of it", which is the
            # shape LoRA actually gives you.
            "adapters": [a.as_dict() for a in self.adapters.list()],
            "default": default.rel if default else None,
            "status": self.engine.status(),
            **catalogue(),
            "sandbox": {"enabled": self.cfg.run_tests, "available": sandbox_ok,
                        "note": sandbox_why,
                        "timeout_s": self.cfg.sandbox_timeout_s,
                        "memory_mb": self.cfg.sandbox_memory_mb},
            "history": self.history.stats(),
            "probes_seen": self.history.probes_seen(),
        }

    def status(self) -> dict:
        return self.engine.status()

    # ---- generating --------------------------------------------------------------------
    def stream(self, *, ckpt_id: str, mode: str, prompt: str = "",
               messages: list[dict] | None = None, system: str | None = None,
               params: SamplingParams | None = None, device: str | None = None,
               probe: str | None = None, task: str | None = None,
               adapter: str | None = None,
               record: bool = True) -> Iterator[tuple[str, object]]:
        """Generate, then grade and record. Yields the engine's events plus two of its own.

        Events: `("start", meta)`, `("delta", text)…`, optionally `("test", result)`, then
        `("done", stats)` — with `record_id` on the final payload once it has been written.

        Validation happens eagerly (see `Engine.stream`), so a bad request raises here
        rather than inside the stream.
        """
        code_task = None
        if task:
            code_task = TASKS_BY_ID.get(task)
            if code_task is None:
                raise InferError(f"no such task: {task} "
                                 f"(known: {', '.join(sorted(TASKS_BY_ID))})")
            if not prompt.strip():
                # Which phrasing depends on what the model can read: a base model gets the
                # bare signature to continue, a chat model gets an instruction.
                chat_mode = mode == "chat"
                prompt = code_task.instruction if chat_mode else code_task.prompt

        stream = self.engine.stream(ckpt_id, mode, prompt=prompt, messages=messages,
                                    system=system, params=params, device=device,
                                    adapter=adapter)
        return self._wrap(stream, mode=mode, prompt=prompt, system=system, probe=probe,
                          task=code_task, record=record)

    def _wrap(self, stream, *, mode: str, prompt: str, system: str | None,
              probe: str | None, task, record: bool) -> Iterator[tuple[str, object]]:
        text = ""
        try:
            for kind, payload in stream:
                if kind == "delta":
                    text += payload
                    yield kind, payload
                    continue
                if kind != "done":
                    yield kind, payload
                    continue

                stats = dict(payload)
                test = None
                if task is not None:
                    result = sandbox.run_task(
                        task, stats.get("text", text), chat=(mode == "chat"),
                        timeout_s=self.cfg.sandbox_timeout_s,
                        memory_mb=self.cfg.sandbox_memory_mb,
                        enabled=self.cfg.run_tests)
                    test = result.as_dict()
                    stats["test"] = test
                    yield "test", test

                if record:
                    row = record_from(stats, mode=mode, prompt=prompt,
                                      output=stats.get("text", text), probe=probe,
                                      task=task.id if task else None, test=test,
                                      system=system)
                    stats["record_id"] = self.history.append(row)["id"]
                yield "done", stats
        finally:
            # Closing the outer generator (the browser went away) must close the inner one,
            # or the decode loop keeps running and the engine's lock is never released.
            stream.close()

    def generate(self, **kw) -> dict:
        """The whole answer at once. Same recording and grading as `stream`."""
        text, stats = "", {}
        for kind, payload in self.stream(**kw):
            if kind == "delta":
                text += payload
            elif kind == "done":
                stats = dict(payload)
        stats.setdefault("text", text)
        return stats

    # ---- the fixed suites --------------------------------------------------------------
    def run_tasks(self, ckpt_id: str, *, mode: str = "complete",
                  task_ids: list[str] | None = None,
                  params: SamplingParams | None = None,
                  device: str | None = None, adapter: str | None = None,
                  on_result: Callable[[dict], None] | None = None) -> dict:
        """Every code task against one checkpoint: the project's pass@1, in miniature.

        `on_result` is called after each task so a caller can print a line as it goes — ten
        tasks on a CPU-bound 300M model is minutes, and a progress-free wait that long looks
        like a hang.
        """
        ids = task_ids or list(TASKS_BY_ID)
        rows, passed = [], 0
        for task_id in ids:
            task = TASKS_BY_ID.get(task_id)
            if task is None:
                raise InferError(f"no such task: {task_id}")
            stats = self.generate(ckpt_id=ckpt_id, mode=mode, task=task_id, params=params,
                                  device=device, adapter=adapter)
            test = stats.get("test") or {}
            passed += bool(test.get("ok"))
            row = {"task": task_id, "title": task.title, "difficulty": task.difficulty,
                   "status": test.get("status"), "ok": bool(test.get("ok")),
                   "detail": test.get("detail"), "output": stats.get("text", ""),
                   # What was actually executed, after `tasks.extract_code` trimmed the
                   # generation at the next top-level statement. Worth keeping separate
                   # from `output`: when a task fails you need to know whether the model
                   # wrote bad code or the extraction cut it in the wrong place.
                   "program": test.get("program", ""),
                   "tokens": stats.get("tokens"), "tok_per_s": stats.get("tok_per_s")}
            rows.append(row)
            if on_result:
                on_result(row)
        info = self.store.get(ckpt_id)
        return {"checkpoint": info.as_dict(), "provenance": info.provenance(),
                "passed": passed, "total": len(rows),
                "pass_rate": (passed / len(rows)) if rows else None, "rows": rows}

    def run_probes(self, ckpt_id: str, *, mode: str = "complete",
                   probe_ids: list[str] | None = None,
                   params: SamplingParams | None = None,
                   device: str | None = None, adapter: str | None = None,
                   on_result: Callable[[dict], None] | None = None) -> dict:
        """The fixed prompts, all of them, recorded — so this checkpoint has a row in the
        comparison next time you run it at a later step."""
        from .tasks import CHAT_PROMPTS, PROBES
        pool = {p.id: p for p in (CHAT_PROMPTS if mode == "chat" else PROBES)}
        ids = probe_ids or list(pool)
        rows = []
        for probe_id in ids:
            probe = pool.get(probe_id)
            if probe is None:
                raise InferError(f"no such probe: {probe_id} "
                                 f"(known: {', '.join(sorted(pool))})")
            stats = self.generate(ckpt_id=ckpt_id, mode=mode, prompt=probe.prompt,
                                  probe=probe_id, params=params, device=device,
                                  adapter=adapter)
            row = {"probe": probe_id, "group": probe.group, "prompt": probe.prompt,
                   "expect": probe.expect, "output": stats.get("text", ""),
                   "tokens": stats.get("tokens"), "tok_per_s": stats.get("tok_per_s")}
            rows.append(row)
            if on_result:
                on_result(row)
        info = self.store.get(ckpt_id)
        return {"checkpoint": info.as_dict(), "provenance": info.provenance(), "rows": rows}

    def close(self):
        self.engine.close()
