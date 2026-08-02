"""Reading this codebase with a local model: the source browser behind the portal's Code tab.

The training dashboard answers "what is the run doing?". This module answers the other
question a person has in front of a from-scratch LLM: *what is this code doing, and why is
it written this way?* You pick a file, highlight some lines, and a model running on your own
machine through Ollama explains them.

Three separable pieces, in the order a request touches them:

* **:class:`SourceTree`** — which files are readable at all. An allowlist of directories and
  extensions, resolved against the repo root, so the portal (which can be served on a LAN)
  can never be talked into reading `~/.ssh/id_rsa`, a 20 GB `train.bin`, or a checkpoint.
* **:func:`build_messages`** — the prompt. The selection alone is not enough to explain
  *why*, so the model also gets the whole enclosing file (windowed if it is large) and a
  short primer on what this project is and how it is laid out.
* **:class:`Ollama`** — the client. Streaming NDJSON over `urllib`, no dependency, so the
  portal stays stdlib-only like the rest of it.

Nothing here writes anything, and nothing here touches the GPU directly — but note that the
model Ollama loads *does* sit in the same VRAM the trainer is using. `configs/portal.yaml`
keeps the context window and `keep_alive` deliberately small for that reason.

Read with: docs/07-scaling.md -- the chapter this implements; it ends with the order to read
these files in.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import yaml

from .runs import RunError, repo_root

# --------------------------------------------------------------------------------------
# what is readable
# --------------------------------------------------------------------------------------

#: The Code tab browses the tree under the directory the portal was started in — the repo
#: root, or whatever `--root` pointed at. You can walk up and down inside it freely; you
#: cannot walk *out* of it (see :meth:`SourceTree.resolve`), which is the boundary that
#: matters when the portal is served on a LAN.
#:
#: Two things are pruned on the way down. Directories whose name starts with a dot (`.git`,
#: `.venv`, `.pytest_cache`) are environment and history, not code, and `.venv` alone is
#: tens of thousands of files — walking it would make the file list slow *and* useless.
SKIP_DIRS = {"__pycache__", "node_modules", "site-packages", "egg-info"}

#: Text we know how to show. A `.bin`, `.pt` or a checkpoint is not code and would be a very
#: expensive mistake to open, so the artifact directories simply have nothing in them to
#: list and never appear.
SOURCE_SUFFIXES = {".py", ".sh", ".yaml", ".yml", ".toml", ".md", ".txt", ".cfg", ".ini",
                   ".js", ".css", ".html", ".json"}

#: Even an allowed file gets a ceiling: a generated JSON tokenizer is legal by extension and
#: useless (and slow) to render as numbered lines.
MAX_SOURCE_BYTES = 400_000

LANGS = {".py": "python", ".sh": "bash", ".yaml": "yaml", ".yml": "yaml", ".toml": "toml",
         ".md": "markdown", ".js": "javascript", ".css": "css", ".html": "html",
         ".json": "json"}

#: Where the human-written explanation of an area lives. The model is *told* the path so it
#: can point the reader at it; the doc's text is not sent (the file plus the primer is
#: already the context that matters, and this keeps the prompt small enough to stay fast).
#:
#: Longest prefix wins, so the specific entries have to come before the general ones -- the
#: two `train/` post-training files before `train/` itself. Each chapter named here ends
#: with a "The code, in reading order" section covering the files it is mapped from, and
#: every module carries the reverse pointer in its docstring; keep the three in step.
DOC_HINTS = (
    ("aksharallm/data/prepare_sft", "docs/05-posttraining.md"),
    ("aksharallm/data/prepare_dpo", "docs/05-posttraining.md"),
    ("aksharallm/data", "docs/01-data.md"),
    ("aksharallm/tokenizer", "docs/02-tokenizer.md"),
    ("aksharallm/model/moe", "docs/14-moe.md"),
    ("aksharallm/model", "docs/03-model.md"),
    ("aksharallm/train/sft", "docs/05-posttraining.md"),
    ("aksharallm/train/dpo", "docs/05-posttraining.md"),
    ("aksharallm/train/grpo", "docs/05-posttraining.md"),
    ("aksharallm/train/runlog", "docs/09-running-and-watching.md"),
    ("aksharallm/train", "docs/04-pretraining.md"),
    ("aksharallm/eval", "docs/12-eval.md"),
    ("aksharallm/quant", "docs/10-quantization.md"),
    ("aksharallm/lora", "docs/11-lora.md"),
    ("aksharallm/synth", "docs/13-synthetic-data.md"),
    ("aksharallm/learn", "docs/15-learning-path.md"),
    ("aksharallm/infer", "docs/06-inference.md"),
    ("aksharallm/portal/explain", "docs/07-scaling.md"),
    ("aksharallm/portal/evals", "docs/12-eval.md"),
    ("aksharallm/portal/quantize", "docs/10-quantization.md"),
    ("aksharallm/portal/finetune", "docs/11-lora.md"),
    ("aksharallm/portal/synth", "docs/13-synthetic-data.md"),
    ("aksharallm/portal/learn", "docs/15-learning-path.md"),
    ("aksharallm/portal", "docs/09-running-and-watching.md"),
    ("docs/lessons", "docs/15-learning-path.md"),
    ("scripts", "docs/09-running-and-watching.md"),
    ("configs", "docs/04-pretraining.md"),
    ("tests", "docs/08-troubleshooting.md"),
)

#: What the model needs to know before it can say anything project-specific. Kept short and
#: factual: a primer that speculates is worse than none, because the model will repeat the
#: speculation back as if the codebase said it.
PRIMER = """\
You are reading `aksharallm`, a large language model built from scratch as a teaching
artifact: the BPE tokenizer, the transformer, the training loop, evaluation and inference
are all hand-written in PyTorch, with no modelling framework, and trained on a single
NVIDIA RTX 3090.

Layout:
  aksharallm/tokenizer/  byte-level BPE (trainer + encoder); it fixes the embedding index
  aksharallm/data/       corpus download, tokenisation to flat uint16 .bin files, blending
  aksharallm/model/      the transformer: RoPE, RMSNorm, grouped-query attention, SwiGLU
  aksharallm/train/      the pretraining loop, LR schedule, checkpointing, run logging,
                         and post-training: SFT, DPO, GRPO
  aksharallm/eval/       the benchmark harness: MMLU, ARC, HellaSwag, PIQA, GSM8K,
                         HumanEval, an LLM-judge, perplexity
  aksharallm/infer/      KV-cache generation and sampling
  aksharallm/quant/      int8/int4/NF4 from scratch: RTN, GPTQ, AWQ, QAT, a Triton kernel
  aksharallm/lora/       LoRA and QLoRA adapters from scratch
  aksharallm/synth/      generating training data with a local teacher, and checking it
  aksharallm/learn/      the learning path: thirteen lessons over this repo
  aksharallm/portal/     this local web portal (stdlib HTTP server, no dependencies)
  configs/*.yaml         one YAML per run; a run = that file plus `-o key=value` overrides
  scripts/*.sh           the launchers a human would type
  docs/00-15             the human-written deep dives; each ends with the order to read
                         the files it covers, and each module names its chapter

Conventions that explain a lot of the code:
  * Everything is config-driven. Nothing about a run is hardcoded.
  * Long runs are expected to be interrupted: the trainer writes `train.pid`, watches for a
    `STOP` file, saves `ckpt_last.pt` and resumes exactly where it left off.
  * The portal never starts or stops training itself — it shells out to the same scripts a
    human would run, so the button and the terminal can never disagree.
  * The project deliberately avoids dependencies where the dependency would hide the very
    thing the code exists to show.
"""

DEFAULT_ASK = ("Explain this selection: what it does, why it is written this way, and any "
               "nuance or gotcha I should know about it.")


def lang_for(rel: str) -> str:
    return LANGS.get(Path(rel).suffix, "")


def doc_for(rel: str) -> str | None:
    """The deep-dive doc covering this file, longest prefix first."""
    for prefix, doc in DOC_HINTS:
        if rel == prefix or rel.startswith(prefix + "/") or rel.startswith(prefix):
            return doc
    return "docs/00-overview.md" if rel.endswith((".py", ".sh", ".yaml")) else None


class SourceTree:
    """The tree the Code tab may read, rooted where the portal is running, and safe access
    to it.

    The root is the directory the server was started against — the repo root, or `--root`.
    Inside it you can go up and down anywhere; outside it you cannot go at all. Every path
    that arrives from the browser goes through :meth:`resolve`, which is the only place that
    turns a string into a `Path`, and which resolves symlinks *before* checking containment
    — so neither `../../etc/passwd` nor a symlink planted inside `docs/` gets out.
    """

    def __init__(self, root: Path | None = None):
        self.root = Path(root).resolve() if root else repo_root()

    # -- listing -------------------------------------------------------------------------
    def _prune(self, path: Path) -> bool:
        """True if we should not descend into this directory."""
        return path.name.startswith(".") or path.name in SKIP_DIRS \
            or path.name.endswith(".egg-info")

    def files(self) -> dict:
        """The whole readable tree at once: every file, and every folder containing one.

        One walk, one response, because the browser wants both at the same time — the folder
        view for navigating and the flat list for the filter box, and a filter that only
        searched the folder you happen to be in would be useless.

        A directory that holds no readable file at any depth is left out rather than shown
        empty: `data/` and `checkpoints/` contain nothing but `.bin` and `.pt`, and an empty
        folder in the tree reads as a bug.
        """
        files: list[dict] = []
        dirs: set[str] = set()

        def walk(base: Path):
            try:
                entries = sorted(base.iterdir(), key=lambda p: p.name.lower())
            except OSError:
                return
            for path in entries:
                if path.is_symlink():
                    # Listed only if it stays inside the root; a link out is not ours to
                    # publish, and a link to a parent would make the walk infinite.
                    try:
                        target = path.resolve()
                        if self.root not in target.parents and target != self.root:
                            continue
                    except OSError:
                        continue
                if path.is_dir():
                    if not self._prune(path):
                        walk(path)
                elif path.suffix in SOURCE_SUFFIXES:
                    try:
                        size = path.stat().st_size
                    except OSError:
                        continue
                    if size > MAX_SOURCE_BYTES:
                        continue
                    entry = self._entry(path, size)
                    files.append(entry)
                    # Every ancestor of a readable file is worth navigating into.
                    parts = entry["dir"].split("/") if entry["dir"] else []
                    for i in range(len(parts)):
                        dirs.add("/".join(parts[:i + 1]))

        walk(self.root)
        files.sort(key=lambda e: (e["dir"], e["name"].lower()))
        return {"root": str(self.root), "files": files, "dirs": sorted(dirs)}

    def _entry(self, path: Path, size: int | None = None) -> dict:
        rel = path.relative_to(self.root).as_posix()
        parent = str(Path(rel).parent)
        return {"path": rel, "dir": "" if parent == "." else parent, "name": path.name,
                "size": size if size is not None else path.stat().st_size,
                "lang": lang_for(rel)}

    # -- reading -------------------------------------------------------------------------
    def resolve(self, rel: str) -> Path:
        """Turn a browser-supplied path into a real file, or raise.

        The checks are ordered cheapest-first, and the error messages say which rule was
        broken — a reader who clicked something under `data/` should be told that binary
        artifacts are out of scope, not given a bare 404.
        """
        rel = (rel or "").strip().lstrip("/")
        if not rel or "\x00" in rel:
            raise RunError("no file given")
        candidate = (self.root / rel).resolve()
        if candidate != self.root and self.root not in candidate.parents:
            raise RunError(f"'{rel}' is outside {self.root.name}/ — the Code tab only reads "
                           "the tree the portal was started in.")
        norm = candidate.relative_to(self.root).as_posix()
        parts = Path(norm).parts
        if any(p.startswith(".") or p in SKIP_DIRS or p.endswith(".egg-info")
               for p in parts[:-1]):
            raise RunError(f"'{norm}' is inside a hidden, build or environment directory.")
        if candidate.suffix not in SOURCE_SUFFIXES:
            raise RunError(f"'{norm}' is not a text file the Code tab can show "
                           f"(allowed: {', '.join(sorted(SOURCE_SUFFIXES))}).")
        if not candidate.is_file():
            raise RunError(f"no such file: {norm}")
        if candidate.stat().st_size > MAX_SOURCE_BYTES:
            raise RunError(f"'{norm}' is larger than "
                           f"{MAX_SOURCE_BYTES // 1000} kB — too big to read line by line.")
        return candidate

    def read(self, rel: str) -> dict:
        path = self.resolve(rel)
        norm = path.relative_to(self.root).as_posix()
        text = path.read_text(encoding="utf-8", errors="replace")
        return {"path": norm, "lang": lang_for(norm), "text": text,
                "lines": text.count("\n") + (0 if text.endswith("\n") or not text else 1),
                "size": path.stat().st_size, "doc": doc_for(norm)}


# --------------------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------------------

@dataclass
class ExplainConfig:
    """Where the model lives and how much of the machine it may use.

    Loaded from `configs/portal.yaml` (`explain:`), overridden by environment, overridden
    again per-request by the model picker in the page. It reloads itself when the file's
    mtime changes, so editing the YAML does not mean restarting the portal — the same
    contract `Schedule` offers for rules.

    `SECTION` and `ENV_PREFIX` are class attributes rather than constants so that a second
    Ollama-backed feature can be the same config with a different section: the eval
    harness's LLM-judge subclasses this to read `judge:` and `AKSHARALLM_JUDGE_*`. One
    client, one set of failure messages, two callers.
    """

    SECTION = "explain"
    ENV_PREFIX = "AKSHARALLM_EXPLAIN"

    host: str = "http://127.0.0.1:11434"
    model: str = "gemma4:12b"
    temperature: float = 0.2
    num_ctx: int = 8192
    num_predict: int = 800
    keep_alive: str = "5m"
    #: Layers to put on the GPU. `None` lets Ollama decide (all of them, if they fit); `0`
    #: keeps the explainer entirely on the CPU, which is slow but cannot take VRAM away from
    #: a training run that is 20 GB into a 24 GB card.
    num_gpu: int | None = None
    #: Whether a reasoning model may think before answering. Off by default, and that is not
    #: a stylistic choice: a thinking model spends `num_predict` on its reasoning first, so
    #: with thinking on and a modest budget you get a complete train of thought and an empty
    #: answer. `None` leaves the field out of the request entirely.
    think: bool | None = False
    max_file_chars: int = 12000
    timeout_s: float = 300.0
    max_history: int = 12
    note: str | None = None          # why the file was ignored, if it was
    path: Path | None = field(default=None, repr=False)
    _mtime: float = field(default=0.0, repr=False)

    @classmethod
    def load(cls, root: Path | None = None) -> "ExplainConfig":
        root = Path(root).resolve() if root else repo_root()
        cfg = cls(path=root / "configs" / "portal.yaml")
        cfg.reload()
        return cfg

    def reload(self) -> "ExplainConfig":
        data: dict = {}
        self.note = None
        if self.path and self.path.is_file():
            try:
                self._mtime = self.path.stat().st_mtime
                loaded = yaml.safe_load(self.path.read_text()) or {}
                data = (loaded.get(self.SECTION) or {}) if isinstance(loaded, dict) else {}
            except (OSError, yaml.YAMLError) as exc:
                # A broken YAML must not take the whole portal down; the defaults are
                # usable, and the page shows why the file was ignored.
                self.note = f"{self.path.name} could not be read ({exc}); using defaults."
                data = {}
        for key in ("host", "model", "keep_alive"):
            if data.get(key) is not None:
                setattr(self, key, str(data[key]))
        if "think" in data:
            self.think = None if data["think"] is None else bool(data["think"])
        for key, cast in (("temperature", float), ("num_ctx", int), ("num_predict", int),
                          ("max_file_chars", int), ("timeout_s", float),
                          ("max_history", int), ("num_gpu", int)):
            if data.get(key) is not None:
                try:
                    setattr(self, key, cast(data[key]))
                except (TypeError, ValueError):
                    pass
        # Environment wins over the file: it is how you point at a model on another machine
        # for one session without editing a file that is checked in.
        self.host = os.environ.get("AKSHARALLM_OLLAMA_HOST", self.host).rstrip("/")
        self.model = os.environ.get(f"{self.ENV_PREFIX}_MODEL", self.model)
        if os.environ.get(f"{self.ENV_PREFIX}_NUM_GPU"):
            try:
                self.num_gpu = int(os.environ[f"{self.ENV_PREFIX}_NUM_GPU"])
            except ValueError:
                pass
        return self

    def reload_if_changed(self) -> "ExplainConfig":
        try:
            if self.path and self.path.stat().st_mtime != self._mtime:
                self.reload()
        except OSError:
            pass
        return self

    def as_dict(self) -> dict:
        return {"host": self.host, "model": self.model, "temperature": self.temperature,
                "num_ctx": self.num_ctx, "num_predict": self.num_predict,
                "keep_alive": self.keep_alive, "max_file_chars": self.max_file_chars,
                "num_gpu": self.num_gpu, "think": self.think, "note": self.note}


# --------------------------------------------------------------------------------------
# the prompt
# --------------------------------------------------------------------------------------

SYSTEM = """\
You explain source code to a curious engineer who is reading this project to learn how a
language model is actually built. You are looking at real code from their repository.

How to answer:
* Start with one sentence saying what the selection does. Then the detail.
* Explain the *why*, not just the *what*: why this shape, this order, this data type. If the
  code is written a particular way to avoid a specific failure, say which failure.
* Call out nuances explicitly — silent truncation, off-by-one risk, dtype and device
  choices, memory cost, anything whose absence would bite the reader later.
* Refer to real line numbers from the listing when you point at something.
* Use short paragraphs and bullets. Backticks for identifiers. No preamble, no summary of
  what you are about to do, no closing pep talk.
* If the selection is trivial, say so in a sentence rather than padding it out.
* You may only use what is in the listing and the project notes. If something depends on
  code you cannot see, say which file you would need to look at rather than guessing. Never
  invent an API, a flag or a file path.
"""


def number_lines(text: str, first: int = 1) -> str:
    lines = text.splitlines()
    width = max(4, len(str(first + len(lines) - 1)))
    return "\n".join(f"{i:>{width}} | {ln}" for i, ln in enumerate(lines, start=first))


def window_file(text: str, start: int, end: int, max_chars: int) -> tuple[str, bool]:
    """A numbered listing that fits in `max_chars`, always containing lines `start..end`.

    Returns `(listing, truncated)`. The window grows outward from the selection in whole
    lines so the model sees the enclosing function and its imports where possible; what was
    dropped is marked, because a model that cannot tell it is looking at a fragment will
    confidently explain a function that "has no error handling" when the handler is 40 lines
    above the cut.
    """
    lines = text.splitlines()
    if len(text) <= max_chars:
        return number_lines(text), False

    lo, hi = max(0, start - 1), min(len(lines), end)   # 0-based, half-open
    budget = max_chars - sum(len(lines[i]) + 8 for i in range(lo, hi))
    while budget > 0 and (lo > 0 or hi < len(lines)):
        if lo > 0:
            lo -= 1
            budget -= len(lines[lo]) + 8
        if budget > 0 and hi < len(lines):
            budget -= len(lines[hi]) + 8
            hi += 1
    body = number_lines("\n".join(lines[lo:hi]), first=lo + 1)
    head = f"      | … {lo} earlier line(s) not shown …\n" if lo else ""
    tail = (f"\n      | … {len(lines) - hi} later line(s) not shown …") if hi < len(lines) else ""
    return head + body + tail, True


def slice_lines(text: str, start: int, end: int) -> str:
    lines = text.splitlines()
    lo = max(1, min(start, len(lines) or 1))
    hi = max(lo, min(end, len(lines)))
    return "\n".join(lines[lo - 1:hi])


def build_messages(cfg: ExplainConfig, *, path: str, text: str, start: int, end: int,
                   question: str | None = None, snippet: str | None = None,
                   doc: str | None = None, history: list[dict] | None = None) -> list[dict]:
    """System + the context turn + any follow-up turns, in Ollama's chat format.

    The context turn is rebuilt from the file on every request rather than carried in the
    history, so a follow-up asked twenty minutes later still quotes the file as it is on
    disk now — not as it was when the tab was opened.
    """
    selection = slice_lines(text, start, end)
    listing, truncated = window_file(text, start, end, max(2000, cfg.max_file_chars))
    lang = lang_for(path) or ""

    parts = [PRIMER, "", f"File: `{path}`"]
    if doc:
        parts.append(f"The human-written deep dive for this area is `{doc}` — you may point "
                     "the reader at it, but you have not read it.")
    if truncated:
        parts.append("The listing below is a window around the selection, not the whole "
                     "file. Do not claim anything about the parts you cannot see.")
    parts += ["", "Here is the file, with line numbers:", "",
              f"```{lang}", listing, "```", ""]
    if start == end:
        parts.append(f"The reader has selected line {start}:")
    else:
        parts.append(f"The reader has selected lines {start}–{end}:")
    parts += ["", f"```{lang}", selection, "```"]
    if snippet and snippet.strip() and snippet.strip() != selection.strip():
        # A drag that ends mid-line: the whole lines give the model context, this says which
        # characters the reader actually cared about.
        parts += ["", "Within those lines they highlighted exactly this text:",
                  "", f"```{lang}", snippet.strip()[:2000], "```"]
    parts += ["", (question or DEFAULT_ASK).strip()]

    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": "\n".join(parts)}]
    for turn in (history or [])[-cfg.max_history:]:
        role = str(turn.get("role", ""))
        content = str(turn.get("content", "")).strip()
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content})
    return messages


# --------------------------------------------------------------------------------------
# the model
# --------------------------------------------------------------------------------------

class Ollama:
    """A minimal streaming client for a local Ollama server.

    Two calls are used: `GET /api/tags` for the model picker and `POST /api/chat` with
    `stream: true`, which answers with one JSON object per line. Errors are translated into
    `RunError` with a sentence that says what to do — "connection refused" on its own has
    sent more than one person looking for a bug in this file.
    """

    def __init__(self, cfg: ExplainConfig):
        self.cfg = cfg

    def _url(self, suffix: str) -> str:
        return f"{self.cfg.host.rstrip('/')}{suffix}"

    def _friendly(self, exc: Exception) -> RunError:
        host = self.cfg.host
        if isinstance(exc, urllib.error.HTTPError):
            detail = ""
            try:
                detail = json.loads(exc.read() or b"{}").get("error", "")
            except Exception:
                pass
            if exc.code == 404 and "model" in detail.lower():
                return RunError(f"{detail} — pull it first:  ollama pull {self.cfg.model}")
            return RunError(f"Ollama answered {exc.code}{': ' + detail if detail else ''}")
        return RunError(f"cannot reach Ollama at {host} ({exc}). Is `ollama serve` running? "
                        "Set AKSHARALLM_OLLAMA_HOST or configs/portal.yaml to point "
                        "somewhere else.")

    def models(self) -> list[dict]:
        req = urllib.request.Request(self._url("/api/tags"),
                                     headers={"Accept": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read() or b"{}")
        except (urllib.error.URLError, OSError, ValueError) as exc:
            raise self._friendly(exc)
        out = []
        for m in data.get("models", []):
            out.append({"name": m.get("name") or m.get("model", ""),
                        "size": m.get("size"),
                        "family": (m.get("details") or {}).get("family", "")})
        return [m for m in out if m["name"]]

    def chat(self, messages: list[dict],
             model: str | None = None) -> Iterator[tuple[str, str]]:
        """Yield `("delta", text)` as the answer is generated, `("thinking", text)` for a
        reasoning model's scratchpad.

        The two are kept apart on purpose. A reasoning model emits its thinking in a
        separate field, and a reader who wanted "what does this line do" should not have to
        scroll past the model talking to itself to find the sentence they asked for.

        The generator owns the connection: closing it (which is what happens when the
        browser navigates away and the server's write fails) closes the HTTP response, and
        Ollama stops generating instead of burning VRAM for nobody.
        """
        cfg = self.cfg
        options = {"temperature": cfg.temperature, "num_ctx": cfg.num_ctx,
                   "num_predict": cfg.num_predict}
        if cfg.num_gpu is not None:
            options["num_gpu"] = cfg.num_gpu
        payload = {
            "model": model or cfg.model,
            "messages": messages,
            "stream": True,
            "keep_alive": cfg.keep_alive,
            "options": options,
        }
        if cfg.think is not None:
            payload["think"] = cfg.think

        try:
            resp = self._open(payload)
        except urllib.error.HTTPError as exc:
            # Not every model understands `think` at all. Rather than keep a list of which
            # ones do — which would be wrong within a month — ask, and drop the field if
            # the server says it is not applicable.
            detail = ""
            try:
                detail = json.loads(exc.read() or b"{}").get("error", "")
            except Exception:
                pass
            if "think" not in detail.lower():
                raise self._friendly(urllib.error.HTTPError(
                    exc.url, exc.code, detail or exc.reason, exc.headers, None))
            payload.pop("think", None)
            try:
                resp = self._open(payload)
            except (urllib.error.URLError, OSError, ValueError) as exc2:
                raise self._friendly(exc2)
        except (urllib.error.URLError, OSError, ValueError) as exc:
            raise self._friendly(exc)

        with resp:
            for raw in resp:
                line = raw.strip()
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                except ValueError:
                    continue
                if chunk.get("error"):
                    raise RunError(str(chunk["error"]))
                message = chunk.get("message") or {}
                if message.get("thinking"):
                    yield "thinking", message["thinking"]
                if message.get("content"):
                    yield "delta", message["content"]
                if chunk.get("done"):
                    break

    def _open(self, payload: dict):
        req = urllib.request.Request(self._url("/api/chat"),
                                     data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
        return urllib.request.urlopen(req, timeout=self.cfg.timeout_s)
