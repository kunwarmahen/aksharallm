"""The World tab's back end: the toy world model's scores, and a game played beside the truth.

Three things, in the order a reader needs them:

* **results** — every `world eval` on disk (`logs/world/*.json`): exact next frames by outcome
  beside what copying the last frame would score, and open-loop rollouts with impossible
  frames counted. The copy column is what keeps a 36% from looking like a model.
Two games (docs/25): `sokoban`, deterministic, scored on exact frames; and `keys`, with a
hidden key and a slippery floor, scored on calibration (`world/calibrate.py`). Its results
carry a `calibration` block the tab draws as a second table.

* **play** — one game run open loop, inline, the model's frames next to the true ones. A
  13.8M model and 64 cells a frame: a handful of seconds even on the CPU, so no job runner.
* **jobs** — `world prepare` and `world eval`, detached, through `CliJobs`: the exact CLI
  command, printed in the panel. Training itself is the dashboard's Start (`world-sokoban`
  is in `LAUNCHERS`).

Device policy is the repo's: the CPU while a training run holds the card.

Read with: docs/25-world-model.md -- the chapter this implements; it ends with the order to
read these files in.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from ..audio.ttsreports import finite
from .jobs import CliJobs, JobError, training

RESULTS = Path("logs/world")
MAX_HORIZON = 12


class WorldError(RuntimeError):
    pass


class World:
    def __init__(self, root: Path, device_for=None):
        self.root = Path(root)
        self._device_for = device_for
        self._cache: dict = {}
        self.jobs = CliJobs(self.root, "logs/world/jobs", "aksharallm.world")

    def device(self) -> tuple[str, str]:
        if self._device_for is not None:
            return self._device_for()
        return ("cuda", "the card is free") if torch.cuda.is_available() else ("cpu", "no CUDA")

    # ---- discovery ------------------------------------------------------------------------

    def checkpoints(self) -> list[dict]:
        """World models are recognised by their vocabulary (15 words for sokoban, 17 for
        keys), not by name; the size also says which game the model plays."""
        from ..world.data import game_for_vocab
        out = []
        for path in sorted(self.root.glob("checkpoints/*/ckpt_*.pt")):
            try:
                blob = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
            except Exception:
                continue
            game = game_for_vocab((blob.get("model_config") or {}).get("vocab_size"))
            if game is None:
                continue
            out.append({"rel": str(path.relative_to(self.root)), "step": blob.get("step"),
                        "game": game,
                        "best_val": blob.get("best_val"), "mtime": path.stat().st_mtime})
        return sorted(out, key=lambda r: r["mtime"], reverse=True)

    def data(self) -> list[dict]:
        """Every prepared world on disk: `data/world` (sokoban) and `data/world-<game>`."""
        out = []
        for meta in sorted(self.root.glob("data/world*/meta.json")):
            try:
                m = json.loads(meta.read_text())
            except (OSError, ValueError):
                continue
            out.append({"dir": str(meta.parent.relative_to(self.root)),
                        "game": m.get("game", "sokoban"),
                        **{k: m.get(k) for k in ("tokens", "outcomes", "episodes", "steps")}})
        return out

    def results(self) -> list[dict]:
        out = []
        folder = self.root / RESULTS
        for p in sorted(folder.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                d = json.loads(p.read_text())
                out.append({"name": p.name, "checkpoint": d.get("checkpoint"),
                            "step": d.get("step"), "seconds": d.get("seconds"),
                            "game": d.get("game", "sokoban"),
                            "next_frame": d["next_frame"], "rollout": d["rollout"],
                            "calibration": d.get("calibration")})
            except (OSError, ValueError, KeyError):
                continue
        return out

    def overview(self) -> dict:
        device, why = self.device()
        # finite(): an outcome with no samples is NaN, and one NaN makes the browser reject
        # the whole response.
        return finite({"checkpoints": self.checkpoints(), "data": self.data(),
                       "results": self.results(), "device": device, "device_reason": why,
                       "jobs": self.jobs.status()})

    # ---- play -----------------------------------------------------------------------------

    def _model(self, rel: str):
        from ..infer.cli import load_model
        games = {c["rel"]: c["game"] for c in self.checkpoints()}
        if rel not in games:
            raise WorldError(f"not a world-model checkpoint: {rel!r}")
        device, _ = self.device()
        key = f"{rel}@{device}"
        if self._cache.get("key") != key:
            model, _ = load_model(str(self.root / rel), device)
            model.eval()
            self._cache = {"key": key, "model": model}
        return self._cache["model"], device, games[rel]

    def play(self, checkpoint: str, horizon: int = 6, seed: int = 0) -> dict:
        from ..world.evaluate import play
        horizon = max(1, min(int(horizon), MAX_HORIZON))
        model, device, game = self._model(checkpoint)
        steps = play(model, horizon=horizon, seed=int(seed), device=device, game=game)
        return {"checkpoint": checkpoint, "seed": seed, "steps": steps, "device": device,
                "game": game,
                "size": 8,
                "command": f"python -m aksharallm.world play {checkpoint} "
                           f"--horizon {horizon} --seed {seed}"}

    # ---- jobs -----------------------------------------------------------------------------

    def start(self, spec: dict) -> dict:
        kind = spec.get("kind")
        cpu = ["--cpu"] if training(self.root) else []
        if kind == "prepare":
            eps = int(spec.get("episodes") or 60_000)
            if not 100 <= eps <= 200_000:
                raise JobError("episodes must be between 100 and 200,000")
            game = spec.get("game") or "sokoban"
            if game not in ("sokoban", "keys"):
                raise JobError(f"unknown game {game!r}")
            return self.jobs.start(["prepare", "--game", game, "--episodes", str(eps)],
                                   f"prepare {eps:,} {game} episodes")
        if kind == "eval":
            rel = str(spec.get("checkpoint") or "")
            if rel not in {c["rel"] for c in self.checkpoints()}:
                raise JobError(f"not a world-model checkpoint: {rel!r}")
            return self.jobs.start(["eval", rel, *cpu], f"evaluate {rel}")
        raise JobError(f"unknown job {kind!r}")
