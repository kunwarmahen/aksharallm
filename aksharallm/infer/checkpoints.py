"""Which trained models exist on disk, and what each one actually is.

A checkpoint is a 1.2 GB file with an opaque name. Before you can sensibly *test* one you
need to know three things about it, and none of them are in the filename:

  * **how far it trained** — step 500 and step 7,000 of the same run are different models,
    and a model that writes gibberish at step 500 is behaving correctly.
  * **what it has been taught to do** — a base checkpoint has never seen a single ChatML
    token, so talking to it in chat format produces nonsense. That is not a bug to debug,
    it is Phase 3 not having run yet, and the UI should say so instead of letting you
    conclude the model is broken.
  * **which tokenizer built it** — the BPE vocabulary fixes the embedding index. Pair a
    checkpoint with the wrong `tokenizer.json` and you get fluent-looking garbage.

All three are inside the file, which is why this module exists: it opens each checkpoint
*without reading its weights* (`mmap=True` maps the file instead of loading 1.2 GB into
RAM, and tensor shapes are metadata) and caches the answer against the file's mtime. A
directory of six checkpoints is described in milliseconds, which is what lets the portal
poll it.

Reading a checkpoint the trainer is actively writing is safe, and deliberately so:
`save_checkpoint` writes `ckpt_last.tmp` and then `Path.replace()`s it, which is atomic.
You either see the whole previous checkpoint or the whole new one, never a half-written
file — so the playground can watch a live run improve simply by re-reading `ckpt_last.pt`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import torch

from ..train import runlog

#: Checkpoint names arrive from a browser and end up in a filesystem path, so they are
#: whitelisted rather than escaped — and the `.pt` suffix is required, which alone rules
#: out every traversal attempt.
CKPT_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*\.pt$")

#: Run names, same rule as `portal.runs.RUN_NAME_RE`. Duplicated rather than imported: this
#: layer is underneath the portal and must not depend on it (the CLI uses it with no web
#: server anywhere in the process).
RUN_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")

#: Filename prefix -> what the model has been trained to do. Order matters: longest match
#: first. The training stage is not recorded *inside* the checkpoint (it predates this
#: module), but each trainer writes its own filenames, and those are unambiguous.
STAGE_PREFIXES = (
    ("dpo_", "dpo"),      # aligned chat model  (aksharallm.train.dpo)
    ("sft_", "sft"),      # instruction-tuned   (aksharallm.train.sft)
    ("code_", "code"),    # Python specialist   (Phase 4 continued-pretraining)
    ("ckpt_", "base"),    # raw pretrained base (aksharallm.train.pretrain)
)

#: What each stage can be asked to do. This is the whole reason the playground knows to
#: grey out the Chat tab: a base model completes text and nothing else.
STAGE_INFO = {
    "base": {
        "label": "base",
        "modes": ["complete", "code"],
        "note": "A pretrained base model: it continues text, and that is all it has ever "
                "been asked to do. It has never seen a chat turn, so chat is disabled — "
                "that arrives with SFT in Phase 3.",
    },
    "sft": {
        "label": "chat (SFT)",
        "modes": ["complete", "chat", "code"],
        "note": "Instruction-tuned: it has been trained to answer as `assistant` inside "
                "the ChatML template, so chat works. Not yet preference-aligned.",
    },
    "dpo": {
        "label": "chat (SFT+DPO)",
        "modes": ["complete", "chat", "code"],
        "note": "Instruction-tuned and preference-aligned. This is the model to judge the "
                "chat side of the project on.",
    },
    "code": {
        "label": "Python specialist",
        "modes": ["complete", "chat", "code"],
        "note": "Continued-pretrained on Python. Judge it on code completion and the "
                "task suite rather than on prose.",
    },
    "unknown": {
        "label": "unknown",
        "modes": ["complete", "chat", "code"],
        "note": "The filename does not match any trainer's convention, so what this model "
                "can do is a guess. Everything is enabled; believe the output, not the UI.",
    },
}


def repo_root() -> Path:
    """The repo root, from this file's location (aksharallm/infer/checkpoints.py).

    Deliberately not imported from `portal.runs`, which computes the identical thing: this
    package sits *underneath* the portal. The CLI must be usable with no web server, no
    scheduler and no `http.server` anywhere in the process.
    """
    return Path(__file__).resolve().parents[2]


class InferError(Exception):
    """A request that cannot be honoured: no such checkpoint, a mode the model can't do,
    a tokenizer that has gone missing.

    Separate from `portal.runs.RunError` because this layer knows nothing about the web —
    the portal catches both and turns either into a 4xx carrying this message, so the
    sentence you read in the browser is the sentence the CLI would have printed.
    """


def stage_for(name: str) -> str:
    for prefix, stage in STAGE_PREFIXES:
        if name.startswith(prefix):
            return stage
    return "unknown"


def _param_count(state: dict, tied: bool) -> int:
    """Parameters, from tensor *shapes* only — no weight data is touched.

    With `tie_embeddings` the output projection shares storage with the token embedding, so
    it is either absent from the state dict or the same tensor twice; counting it would
    overstate a 300M model by 33M.
    """
    total, seen = 0, set()
    for key, tensor in state.items():
        if not hasattr(tensor, "numel"):
            continue
        if tied and key.endswith("lm_head.weight"):
            continue
        ident = id(tensor.untyped_storage()) if hasattr(tensor, "untyped_storage") else None
        if ident is not None and ident in seen:
            continue
        if ident is not None:
            seen.add(ident)
        total += tensor.numel()
    return total


@dataclass
class Checkpoint:
    """One `.pt` file, described. Everything here is cheap to obtain and safe to send to a
    browser — no tensors, no absolute paths beyond the repo-relative one."""

    run: str
    name: str
    path: Path
    rel: str
    size: int
    mtime: float
    stage: str
    step: int | None
    best_val: float | None
    max_steps: int | None
    tokens_seen: int | None
    tokens_per_step: int | None
    params: int | None
    vocab_size: int | None
    max_seq_len: int | None
    arch: str | None
    tokenizer: str | None
    tokenizer_ok: bool
    train_loss: float | None      # the run's ema at (or just before) this checkpoint's step
    error: str | None = None

    @property
    def modes(self) -> list[str]:
        return STAGE_INFO.get(self.stage, STAGE_INFO["unknown"])["modes"]

    def as_dict(self) -> dict:
        info = STAGE_INFO.get(self.stage, STAGE_INFO["unknown"])
        return {
            "run": self.run, "name": self.name, "rel": self.rel, "size": self.size,
            "mtime": self.mtime, "stage": self.stage, "stage_label": info["label"],
            "stage_note": info["note"], "modes": info["modes"],
            "step": self.step, "best_val": self.best_val, "max_steps": self.max_steps,
            "progress": ((self.step + 1) / self.max_steps
                         if self.step is not None and self.max_steps else None),
            "tokens_seen": self.tokens_seen, "tokens_per_step": self.tokens_per_step,
            "params": self.params, "vocab_size": self.vocab_size,
            "max_seq_len": self.max_seq_len, "arch": self.arch,
            "tokenizer": self.tokenizer, "tokenizer_ok": self.tokenizer_ok,
            "train_loss": self.train_loss, "error": self.error,
            "id": f"{self.run}/{self.name}",
        }

    def provenance(self) -> dict:
        """The subset worth stamping onto every generation this model produces.

        This is the answer to "what was the model when it said that?" — and it is the whole
        reason a text log can replace keeping a copy of the weights. `ckpt_last.pt` is
        overwritten every 500 steps, but a record saying *step 7,000, val 2.89* next to the
        text it produced is permanent, and costs a few hundred bytes.
        """
        return {"run": self.run, "checkpoint": self.name, "stage": self.stage,
                "step": self.step, "best_val": self.best_val,
                "train_loss": self.train_loss, "tokens_seen": self.tokens_seen,
                "max_steps": self.max_steps, "params": self.params,
                "ckpt_mtime": self.mtime}


class CheckpointStore:
    """Every checkpoint under `<root>/checkpoints/`, described and cached.

    The cache key is `(size, mtime)` rather than the path, so `ckpt_last.pt` being rewritten
    mid-run invalidates its own entry and the next read picks up the new step — which is
    exactly what you want when you are watching a live run get better.
    """

    def __init__(self, root: Path | str | None = None):
        self.root = Path(root).resolve() if root else repo_root()
        self._cache: dict[Path, tuple[tuple[int, float], Checkpoint]] = {}

    # ---- discovery ---------------------------------------------------------------------
    def dirs(self) -> list[Path]:
        base = self.root / "checkpoints"
        if not base.is_dir():
            return []
        return sorted(p for p in base.iterdir()
                      if p.is_dir() and RUN_NAME_RE.match(p.name))

    def list(self, run: str | None = None) -> list[Checkpoint]:
        """Every readable checkpoint, best-trained first within each run.

        Sorted by step descending so the picker's first entry is the most trained model —
        the one you almost always want — rather than whatever sorts first alphabetically.
        """
        out: list[Checkpoint] = []
        for d in self.dirs():
            if run and d.name != run:
                continue
            for path in sorted(d.glob("*.pt")):
                if not CKPT_NAME_RE.match(path.name):
                    continue
                out.append(self.describe(path))
        out.sort(key=lambda c: (c.run, -(c.step or -1), c.name))
        return out

    def resolve(self, run: str, name: str) -> Path:
        """Turn a browser-supplied `(run, name)` pair into a real file, or raise.

        Both halves are whitelisted and the result is re-checked for containment, because
        the portal can be served on a LAN and this is the only place a request gets to name
        a file to open.
        """
        if not RUN_NAME_RE.match(run or ""):
            raise InferError(f"invalid run name: {run!r}")
        if not CKPT_NAME_RE.match(name or ""):
            raise InferError(f"invalid checkpoint name: {name!r} (expected something.pt)")
        path = (self.root / "checkpoints" / run / name).resolve()
        base = (self.root / "checkpoints").resolve()
        if base not in path.parents:
            raise InferError(f"'{run}/{name}' is outside checkpoints/")
        if not path.is_file():
            known = ", ".join(c.rel for c in self.list()) or "none yet"
            raise InferError(f"no such checkpoint: {run}/{name}. Available: {known}")
        return path

    def get(self, ckpt_id: str) -> Checkpoint:
        """`"small-code/ckpt_best.pt"` -> the described checkpoint."""
        run, _, name = (ckpt_id or "").partition("/")
        return self.describe(self.resolve(run, name))

    def identify(self, ref: str) -> str:
        """Whatever a person typed -> a canonical `run/name.pt` id.

        Accepts the three things that are natural to type, in this order:

            checkpoints/small-code/ckpt_best.pt   a path, absolute or relative — what tab
                                                  completion gives you
            small-code/ckpt_best.pt               the id itself
            small-code                            a run: takes its best checkpoint, or its
                                                  last if it has never been evaluated

        The bare-run form is the one worth having. `ckpt_best.pt` is nearly always what you
        want and remembering to type it every time is friction for no reason.
        """
        ref = (ref or "").strip().rstrip("/")
        if not ref:
            raise InferError("no checkpoint given")

        path = Path(ref)
        if path.suffix == ".pt":
            if not path.is_absolute():
                # Try it as a path from here and from the repo root before assuming it is
                # already an id: `checkpoints/x/y.pt` is both a valid path and a plausible
                # (wrong) id, and the file on disk is the better answer.
                for candidate in (Path.cwd() / path, self.root / path):
                    if candidate.is_file():
                        path = candidate
                        break
            if path.is_file():
                resolved = path.resolve()
                base = (self.root / "checkpoints").resolve()
                if base in resolved.parents:
                    return f"{resolved.parent.name}/{resolved.name}"
                raise InferError(f"{ref} is outside checkpoints/ — move it under "
                                 f"checkpoints/<run>/ so its run is unambiguous.")
            if "/" in ref:
                return ref                     # an id for a file that does not exist yet
            raise InferError(f"no such checkpoint file: {ref}")

        run = ref.split("/")[0]
        found = [c for c in self.list(run) if not c.error]
        if not found:
            known = ", ".join(sorted(d.name for d in self.dirs())) or "none"
            raise InferError(f"no checkpoints for run '{run}' (runs with checkpoints: "
                             f"{known})")
        by_name = {c.name: c for c in found}
        for name in ("ckpt_best.pt", "ckpt_last.pt"):
            if name in by_name:
                return by_name[name].rel
        return found[0].rel

    # ---- description -------------------------------------------------------------------
    def describe(self, path: Path) -> Checkpoint:
        try:
            st = path.stat()
            key = (st.st_size, st.st_mtime)
        except OSError as exc:
            raise InferError(f"cannot stat {path.name}: {exc}")
        cached = self._cache.get(path)
        if cached and cached[0] == key:
            return cached[1]
        info = self._read(path, st.st_size, st.st_mtime)
        self._cache[path] = (key, info)
        return info

    def _read(self, path: Path, size: int, mtime: float) -> Checkpoint:
        run = path.parent.name
        base = dict(run=run, name=path.name, path=path,
                    rel=f"{run}/{path.name}", size=size, mtime=mtime,
                    stage=stage_for(path.name), step=None, best_val=None, max_steps=None,
                    tokens_seen=None, tokens_per_step=None, params=None, vocab_size=None,
                    max_seq_len=None, arch=None, tokenizer=None, tokenizer_ok=False,
                    train_loss=None)
        try:
            # mmap: the file is mapped, not read. Shapes and dtypes are in the zip header,
            # so everything below costs a few page faults rather than 1.2 GB of I/O.
            # weights_only=False because the payload carries the run's config dicts too.
            ckpt = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        except Exception as exc:
            # A checkpoint from a kill -9 mid-save, or a file that is not a checkpoint at
            # all. It still belongs in the list — with the reason it cannot be used.
            return Checkpoint(**base, error=f"unreadable ({type(exc).__name__}: {exc})")

        mcfg = ckpt.get("model_config") or {}
        cfg = ckpt.get("config") or {}
        tcfg = cfg.get("train") or {}
        dcfg = cfg.get("data") or {}

        tps = None
        if all(tcfg.get(k) for k in ("batch_size", "grad_accum", "seq_len")):
            tps = tcfg["batch_size"] * tcfg["grad_accum"] * tcfg["seq_len"]
        step = ckpt.get("step")
        tok = dcfg.get("tokenizer")
        tok_path = (self.root / tok) if tok and not Path(tok).is_absolute() else (
            Path(tok) if tok else None)

        arch = None
        if mcfg:
            arch = (f"d={mcfg.get('d_model')} L={mcfg.get('n_layers')} "
                    f"H={mcfg.get('n_heads')} KV={mcfg.get('n_kv_heads')} "
                    f"ctx={mcfg.get('max_seq_len')}")

        return Checkpoint(
            **{**base,
               "step": step,
               "best_val": ckpt.get("best_val"),
               "max_steps": tcfg.get("max_steps"),
               "tokens_per_step": tps,
               "tokens_seen": (tps * (step + 1)) if tps and step is not None else None,
               "params": _param_count(ckpt.get("model") or {},
                                      bool(mcfg.get("tie_embeddings"))),
               "vocab_size": mcfg.get("vocab_size"),
               "max_seq_len": mcfg.get("max_seq_len"),
               "arch": arch,
               "tokenizer": tok,
               "tokenizer_ok": bool(tok_path and tok_path.is_file()),
               "train_loss": self._loss_at(path.parent, step)},
            error=None)

    def _loss_at(self, run_dir: Path, step: int | None) -> float | None:
        """The run's smoothed training loss at the moment this checkpoint was saved.

        `best_val` is the best validation loss the run has *ever* reached, which for
        `ckpt_last.pt` may be from thousands of steps earlier. The ema at this step is the
        number that actually describes this file, and it is what makes two records in the
        history comparable.
        """
        if step is None:
            return None
        best = None
        for rec in runlog.load_records(run_dir / "train_log.jsonl"):
            rec_step = rec.get("step")
            if rec_step is None or rec_step > step:
                continue
            if rec.get("ema") is not None:
                best = rec["ema"]
        return best

    # ---- the default choice ------------------------------------------------------------
    def default(self, prefer_chat: bool = False) -> Checkpoint | None:
        """The checkpoint to open the playground on.

        Prefers the most-aligned model that exists (DPO > SFT > base), because once Phase 3
        has run that is the interesting one. Within a stage it takes the most *recently
        written* file, not the highest step: the toy `tiny` run finished at step 7,999 and
        the real 300M run is only a fifth of the way through its 40,000, so ranking by step
        would open the playground on a 13M-parameter TinyStories model every time.
        Returns None on a fresh clone with nothing trained yet.
        """
        usable = [c for c in self.list() if not c.error]
        if not usable:
            return None
        order = {"dpo": 0, "sft": 1, "code": 2, "base": 3, "unknown": 4}
        rank = lambda c: (order.get(c.stage, 9), -c.mtime)  # noqa: E731
        if prefer_chat:
            chat = [c for c in usable if c.stage in ("dpo", "sft", "code")]
            if chat:
                return min(chat, key=rank)
        return min(usable, key=rank)
