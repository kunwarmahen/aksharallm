"""The Tools tab's back end: does the chat model reach for a Python call, and is it right?

* **results** — every `tools eval` on disk (`logs/tools/*.json`), by kind, with and without
  the tool; the held-out kinds of the report's problem set (v1 or v2) flagged, because it is the column that says
  whether the model learned an idea or six templates.
* **compare** — two reports on the same problems, paired per problem (before vs after the
  tool SFT), through `tools.agent.compare`, the same function `tools compare` prints.
* **ask** — one question to a chat checkpoint, inline, with or without the tool, and the
  whole transcript back: the call it wrote, what the sandbox printed, the answer. One
  300M model is cached; the device is the Playground's (CPU while a run trains).
* **jobs** — `tools data` and `tools eval`, detached, through `CliJobs`.

Read with: docs/06-posttraining.md -- the chapter this implements; it ends with the order to
read these files in.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import torch

from ..audio.ttsreports import finite
from .jobs import CliJobs, JobError, training

RESULTS = Path("logs/tools")
TOKENIZER = "data/blend/tokenizer.json"
MAX_QUESTION = 400
LABEL_RE = re.compile(r"^[A-Za-z0-9_-]{1,40}$")


class ToolsError(RuntimeError):
    pass


class ToolUse:
    def __init__(self, root: Path, device_for=None):
        self.root = Path(root)
        self._device_for = device_for
        self._cache: dict = {}
        self.jobs = CliJobs(self.root, "logs/tools/jobs", "aksharallm.tools")

    def device(self) -> tuple[str, str]:
        if self._device_for is not None:
            return self._device_for()
        return ("cuda", "the card is free") if torch.cuda.is_available() else ("cpu", "no CUDA")

    def checkpoints(self) -> list[dict]:
        """Chat checkpoints (the stages that apply the chat template); a base model has never
        seen a turn, let alone a tool turn."""
        out = []
        for stage in ("sft", "dpo", "grpo"):
            for path in self.root.glob(f"checkpoints/*/{stage}_best.pt"):
                if (path.parent / "archive.meta").is_file():
                    continue
                out.append({"rel": str(path.relative_to(self.root)), "stage": stage,
                            "tool_trained": "tools" in path.parent.name,
                            "mtime": path.stat().st_mtime})
        return sorted(out, key=lambda r: (not r["tool_trained"], -r["mtime"]))

    def _rows(self) -> list[tuple[Path, dict]]:
        out = []
        for p in sorted((self.root / RESULTS).glob("*.json"), key=lambda p: p.stat().st_mtime,
                        reverse=True):
            try:
                out.append((p, json.loads(p.read_text())))
            except (OSError, ValueError):
                continue
        return out

    def results(self) -> list[dict]:
        from ..tools.problems import HELD_OUT_KIND
        return [{"name": p.name, "checkpoint": d.get("checkpoint"), "n": d.get("n"),
                 "tool": d.get("tool"), "direct": d.get("direct"), "called": d.get("called"),
                 "paired": d.get("paired"), "by_kind": d.get("by_kind"),
                 "set": d.get("set", "v1"), "held_out": d.get("held_out") or [HELD_OUT_KIND]}
                for p, d in self._rows() if "by_kind" in d]

    def overview(self) -> dict:
        from ..tools.problems import HELD_OUT_KIND, KINDS
        device, why = self.device()
        return finite({"checkpoints": self.checkpoints(), "results": self.results(),
                "kinds": list(KINDS), "held_out": HELD_OUT_KIND,
                "data": (self.root / "data/tools/sft.jsonl").is_file(),
                "device": device, "device_reason": why, "jobs": self.jobs.status()})

    def _load(self, name: str) -> dict:
        if "/" in name or not name.endswith(".json"):
            raise ToolsError("not a report name")
        p = self.root / RESULTS / name
        if not p.is_file():
            raise ToolsError(f"no report {name!r}")
        return json.loads(p.read_text())

    def report(self, name: str) -> dict:
        d = self._load(name)
        return {"name": name, "rows": d.get("rows", []), "by_kind": d.get("by_kind")}

    def compare(self, a: str, b: str) -> dict:
        from ..tools.agent import compare
        try:
            return {"a": a, "b": b, **compare(self._load(a), self._load(b)),
                    "command": f"python -m aksharallm.tools compare {RESULTS}/{a} {RESULTS}/{b}"}
        except ValueError as e:
            raise ToolsError(str(e)) from e

    # ---- ask ------------------------------------------------------------------------------

    def ask(self, checkpoint: str, question: str, use_tool: bool = True) -> dict:
        from ..infer.cli import load_model
        from ..tokenizer.tokenizer import Tokenizer
        from ..tools.agent import final_number, solve
        question = (question or "").strip()
        if not question:
            raise ToolsError("ask it something")
        if len(question) > MAX_QUESTION:
            raise ToolsError(f"keep the question under {MAX_QUESTION} characters")
        if checkpoint not in {c["rel"] for c in self.checkpoints()}:
            raise ToolsError(f"not a chat checkpoint: {checkpoint!r}")
        device, why = self.device()
        key = f"{checkpoint}@{device}"
        if self._cache.get("key") != key:
            self._cache = {}            # drop the old model before loading the next
            model, _ = load_model(str(self.root / checkpoint), device)
            model.eval()
            self._cache = {"key": key, "model": model, "tok": Tokenizer(self.root / TOKENIZER)}
        r = solve(self._cache["model"], self._cache["tok"], question, use_tool=use_tool,
                  device=device)
        return {"messages": r["messages"], "calls": r["calls"], "final": r["final"],
                "number": final_number(r["final"]), "device": device, "device_reason": why,
                "command": f"python -m aksharallm.tools ask {checkpoint} "
                           + json.dumps(question) + ("" if use_tool else " --no-tool")}

    # ---- jobs -----------------------------------------------------------------------------

    def start(self, spec: dict) -> dict:
        kind = spec.get("kind")
        cpu = ["--cpu"] if training(self.root) else []
        pset = spec.get("set") or "v1"
        if pset not in ("v1", "v2"):
            raise JobError(f"unknown problem set {pset!r}")
        if kind == "data":
            return self.jobs.start(["data", "--set", pset],
                                   f"write 30,000 teaching conversations (set {pset})")
        if kind == "eval":
            rel = str(spec.get("checkpoint") or "")
            if rel not in {c["rel"] for c in self.checkpoints()}:
                raise JobError(f"not a chat checkpoint: {rel!r}")
            label = str(spec.get("label") or "tools")
            if not LABEL_RE.match(label):
                raise JobError("a label is letters, digits, - and _ only")
            per = int(spec.get("per_kind") or 20)
            if not 1 <= per <= 100:
                raise JobError("problems per kind must be 1-100")
            return self.jobs.start(["eval", rel, "--set", pset, "--label", label,
                                    "--per-kind", str(per), *cpu], f"score {rel} ({label}, set {pset})")
        raise JobError(f"unknown job {kind!r}")
