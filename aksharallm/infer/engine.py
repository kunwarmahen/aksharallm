"""A checkpoint, loaded and kept warm, that you can talk to.

`aksharallm.infer.cli` already knew how to load a checkpoint and generate from it. What it
did not know was how to do that *while the same GPU is six days into training a 300M
model* — which is precisely when you most want to ask "is this thing learning anything?".
This module is that: one resident model, a device policy that will not cost you a run, and
a generation loop that streams.

Three decisions worth understanding before changing anything here.

**Where the model runs.** The card has 24 GB and a Phase-2 run holds about 21 of them. The
model itself is small — 300M parameters in bf16 is 0.6 GB, and grouped-query attention
makes the KV cache almost free (25 MB for a 1024-token context) — so it *would* fit in the
gap. It is still not worth it. A CUDA context is half a gigabyte before a single weight
lands, PyTorch's allocator will happily fragment what is left, and the failure mode is not
"the playground is slow", it is "the training run died at step 22,000 overnight". So the
default policy is: **if a run is training, load on the CPU and say so.** When the card is
idle you get it at full speed, automatically. `device: cuda` in the config overrides this
for someone who has read this paragraph and means it.

**Staying loaded.** Reading 1.2 GB off disk and building the module takes several seconds —
fine once, absurd per prompt when you are trying twelve variations of a prompt. So the model
stays resident and an idle timer unloads it (`idle_unload_s`), returning the memory the way
`keep_alive` does for the Code tab's Ollama model. Switching checkpoints swaps it.

**One generation at a time.** Two concurrent generations on one model would each allocate
their own KV cache and halve each other's speed for no reason. A lock serialises them, and
a caller who cannot get it is *told* rather than left hanging.

Read with: docs/06-inference.md -- the chapter this implements; it ends with the order to read
these files in.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Iterator

import torch
import yaml

from ..config import ModelConfig
from ..model.transformer import Transformer
from ..tokenizer.tokenizer import Tokenizer
from .checkpoints import (
    Adapter,
    AdapterStore,
    Checkpoint,
    CheckpointStore,
    InferError,
    repo_root,
)
from .generate import IncrementalDecoder, stream_generate

#: Free VRAM (bytes) below which `auto` will not use the card even when nothing is
#: training. The model plus its cache is ~0.7 GB; the rest is the CUDA context and enough
#: slack that the allocator never has to fight another process for the last hundred MB.
CUDA_HEADROOM = 2 * 1024 ** 3

DEFAULT_SYSTEM = "You are a helpful assistant."


# ---------------------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------------------

@dataclass
class SamplingParams:
    """The knobs the playground exposes. Defaults chosen for *inspecting* a model rather
    than showing it off: temperature 0.8 with nucleus sampling is what the model will
    actually feel like, and a repetition penalty above 1 hides exactly the degenerate
    looping that tells you a base model is undertrained."""

    max_new_tokens: int = 256
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.95
    repetition_penalty: float = 1.0
    seed: int | None = None

    #: Hard ceilings, so a browser cannot ask for a million tokens and hold the lock for an
    #: hour. Clamping (rather than rejecting) keeps the UI simple; the response says what
    #: was actually used.
    def clamp(self, ctx_len: int = 4096) -> "SamplingParams":
        return SamplingParams(
            max_new_tokens=max(1, min(int(self.max_new_tokens), ctx_len)),
            temperature=max(0.0, min(float(self.temperature), 5.0)),
            top_k=max(0, min(int(self.top_k), 100_000)),
            top_p=max(0.0, min(float(self.top_p), 1.0)),
            repetition_penalty=max(0.5, min(float(self.repetition_penalty), 3.0)),
            seed=self.seed,
        )

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class InferConfig:
    """`configs/portal.yaml` -> `infer:`. Reloaded when the file changes, the same contract
    `ExplainConfig` and `Schedule` offer: editing the YAML never means restarting anything.
    """

    #: auto | cuda | cpu. See the module docstring — `auto` means "the GPU, unless a run is
    #: training or the card is too full to be safe".
    device: str = "auto"
    #: Seconds of inactivity before the model is unloaded and its memory handed back.
    idle_unload_s: float = 300.0
    #: How long a caller waits for the generation lock before being told someone else is
    #: using the model.
    busy_wait_s: float = 2.0
    system: str = DEFAULT_SYSTEM
    sampling: SamplingParams = field(default_factory=SamplingParams)
    #: Longest prompt accepted from a client, in characters. The real limit is the model's
    #: 1024-token context; this only stops a paste of a whole file arriving over HTTP.
    max_prompt_chars: int = 20_000
    #: Generations kept in `logs/playground.jsonl`. See `history.py`.
    history_max: int = 2000
    #: Executing model-written Python — off unless asked for. See `sandbox.py`.
    run_tests: bool = True
    sandbox_timeout_s: float = 10.0
    sandbox_memory_mb: int = 512

    note: str | None = None
    path: Path | None = field(default=None, repr=False)
    _mtime: float = field(default=0.0, repr=False)

    @classmethod
    def load(cls, root: Path | None = None) -> "InferConfig":
        root = Path(root).resolve() if root else repo_root()
        cfg = cls(path=root / "configs" / "portal.yaml")
        cfg.reload()
        return cfg

    def reload(self) -> "InferConfig":
        data: dict = {}
        self.note = None
        if self.path and self.path.is_file():
            try:
                self._mtime = self.path.stat().st_mtime
                loaded = yaml.safe_load(self.path.read_text()) or {}
                data = (loaded.get("infer") or {}) if isinstance(loaded, dict) else {}
            except (OSError, yaml.YAMLError) as exc:
                # Same rule as the explainer: a half-edited YAML degrades to defaults and
                # says why, rather than taking the portal down.
                self.note = f"{self.path.name} could not be read ({exc}); using defaults."
                data = {}

        if data.get("device"):
            self.device = str(data["device"]).lower()
        if data.get("system") is not None:
            self.system = str(data["system"])
        for key, cast in (("idle_unload_s", float), ("busy_wait_s", float),
                          ("max_prompt_chars", int), ("history_max", int),
                          ("sandbox_timeout_s", float), ("sandbox_memory_mb", int)):
            if data.get(key) is not None:
                try:
                    setattr(self, key, cast(data[key]))
                except (TypeError, ValueError):
                    pass
        if data.get("run_tests") is not None:
            self.run_tests = bool(data["run_tests"])

        sampling = data.get("sampling") or {}
        if isinstance(sampling, dict):
            for key, cast in (("max_new_tokens", int), ("temperature", float),
                              ("top_k", int), ("top_p", float),
                              ("repetition_penalty", float)):
                if sampling.get(key) is not None:
                    try:
                        setattr(self.sampling, key, cast(sampling[key]))
                    except (TypeError, ValueError):
                        pass

        # Environment wins over the file, for a one-session override that does not touch a
        # checked-in file — `AKSHARALLM_INFER_DEVICE=cpu` is the one you actually reach for.
        if os.environ.get("AKSHARALLM_INFER_DEVICE"):
            self.device = os.environ["AKSHARALLM_INFER_DEVICE"].lower()
        if self.device not in ("auto", "cuda", "cpu"):
            self.note = f"unknown device {self.device!r}; falling back to auto."
            self.device = "auto"
        return self

    def reload_if_changed(self) -> "InferConfig":
        try:
            if self.path and self.path.stat().st_mtime != self._mtime:
                self.reload()
        except OSError:
            pass
        return self

    def as_dict(self) -> dict:
        return {"device": self.device, "idle_unload_s": self.idle_unload_s,
                "system": self.system, "sampling": self.sampling.as_dict(),
                "max_prompt_chars": self.max_prompt_chars, "run_tests": self.run_tests,
                "sandbox_timeout_s": self.sandbox_timeout_s,
                "sandbox_memory_mb": self.sandbox_memory_mb, "note": self.note}


# ---------------------------------------------------------------------------------------
# where to run
# ---------------------------------------------------------------------------------------

def training_runs(root: Path | None = None) -> list[str]:
    """Which runs are training right now, from the pid files the trainers write.

    This reads the same contract everything else in the project reads: the trainer writes
    `train.pid` into its own `out_dir`, so the file answers "who is training into this
    directory" — the question that matters — rather than "is there a process whose command
    line looks like training", which is the question that once got a stop request aimed at
    the 50-step smoke test. Hence the two guards on the command line: it must be a trainer,
    and it must not be the smoke test writing to /tmp.

    The portal replaces this with `RunStore`-backed detection, which knows about launches in
    pre-flight too. This is the version the CLI gets, with no web server in the process.
    """
    root = Path(root).resolve() if root else repo_root()
    base = root / "checkpoints"
    if not base.is_dir():
        return []
    live = []
    for pid_file in sorted(base.glob("*/train.pid")):
        try:
            pid = int(pid_file.read_text().strip())
            os.kill(pid, 0)
            args = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(
                errors="replace")
        except (OSError, ValueError, ProcessLookupError):
            continue
        if "aksharallm.train" in args and "aksharallm_smoke" not in args:
            live.append(pid_file.parent.name)
    return live


def cuda_free_bytes() -> int | None:
    """Free VRAM on the current device, or None if there is no CUDA at all.

    `mem_get_info` asks the driver, so it sees *every* process on the card — the trainer,
    an Ollama model the Code tab left resident, a stray notebook. That is the number worth
    deciding on; `torch.cuda.memory_allocated()` would only ever describe this process and
    would cheerfully report 24 GB free on a full card.
    """
    if not torch.cuda.is_available():
        return None
    try:
        return torch.cuda.mem_get_info()[0]
    except Exception:
        return None


@dataclass
class DevicePlan:
    """Where a generation will run, and why — the 'why' being the point.

    The reason string is shown in the tab *before* you press Generate, because "this is
    slow" and "this is slow because your run is training and I moved it to the CPU to
    protect it" are very different experiences.
    """

    device: str
    reason: str
    forced: bool = False           # the config asked for this device explicitly
    training: list[str] = field(default_factory=list)
    free_vram: int | None = None

    @property
    def slow(self) -> bool:
        """The CPU path is correct, safe and roughly two orders of magnitude slower. The
        page uses this to warn before you press Generate rather than after."""
        return self.device == "cpu"

    def as_dict(self) -> dict:
        return {"device": self.device, "reason": self.reason, "forced": self.forced,
                "training": self.training, "free_vram": self.free_vram, "slow": self.slow}


def plan_device(cfg: InferConfig, training: list[str] | None = None) -> DevicePlan:
    """Decide where to load, given what else is using the machine."""
    training = list(training or [])
    free = cuda_free_bytes()
    have_cuda = free is not None

    if cfg.device == "cpu":
        return DevicePlan("cpu", "configs/portal.yaml pins inference to the CPU "
                                 "(infer.device: cpu).", forced=True, training=training,
                          free_vram=free)

    if cfg.device == "cuda":
        if not have_cuda:
            return DevicePlan("cpu", "infer.device is cuda, but this machine has no CUDA "
                                     "device — falling back to the CPU.",
                              training=training, free_vram=free)
        warn = (f" A run is training ({', '.join(training)}) and you have asked for the GPU "
                "anyway — watch the GPU panel." if training else "")
        return DevicePlan("cuda", f"infer.device: cuda.{warn}", forced=True,
                          training=training, free_vram=free)

    # auto
    if not have_cuda:
        return DevicePlan("cpu", "no CUDA device on this machine.",
                          training=training, free_vram=free)
    if training:
        return DevicePlan(
            "cpu",
            f"{', '.join(training)} is training on this card, so inference runs on the CPU "
            "to keep the run safe. It will be slow — a few tokens a second. The GPU is used "
            "automatically once the run stops.",
            training=training, free_vram=free)
    if free is not None and free < CUDA_HEADROOM:
        return DevicePlan(
            "cpu",
            f"only {free / 1024 ** 3:.1f} GB of VRAM is free — something else is holding the "
            "card (an Ollama model from the Code tab unloads after a few minutes). Running "
            "on the CPU rather than risking an out-of-memory error.",
            training=training, free_vram=free)
    return DevicePlan("cuda", "the card is free.", training=training, free_vram=free)


# ---------------------------------------------------------------------------------------
# the engine
# ---------------------------------------------------------------------------------------

@dataclass
class Loaded:
    """What is currently in memory."""

    info: Checkpoint
    model: Transformer
    tokenizer: Tokenizer
    device: str
    plan: DevicePlan
    loaded_at: float
    load_s: float
    #: The adapter attached on top, if any. `None` means the plain base model.
    adapter: Adapter | None = None

    @property
    def stage(self) -> str:
        """What this *combination* has been trained to do.

        The adapter wins when there is one, and that is the whole point of adapters here:
        a base checkpoint carrying an SFT adapter is a chat model, even though the
        checkpoint's own filename still says `ckpt_`. Reading the stage off the file alone
        would refuse to chat with exactly the thing we built adapters to produce.
        """
        if self.adapter is not None and self.adapter.stage != "unknown":
            return self.adapter.stage
        return self.info.stage


class Engine:
    """One resident model, swapped on demand, unloaded when idle.

    `busy_cb` answers "which runs are training right now?". It defaults to
    :func:`training_runs` — the pid files — which is enough for anyone. The portal injects a
    `RunStore`-backed version instead, because that one also sees a launch still in
    pre-flight, which is minutes away from wanting the whole card.
    """

    def __init__(self, root: Path | str | None = None, cfg: InferConfig | None = None,
                 busy_cb: Callable[[], list[str]] | None = None):
        self.root = Path(root).resolve() if root else repo_root()
        self.cfg = cfg or InferConfig.load(self.root)
        self.store = CheckpointStore(self.root)
        self.adapters = AdapterStore(self.root)
        self.busy_cb = busy_cb or (lambda: training_runs(self.root))
        self._loaded: Loaded | None = None
        self._lock = threading.RLock()          # serialises generation
        self._state = threading.Lock()          # protects `_loaded` itself
        self._last_used = time.monotonic()
        self._reaper: threading.Thread | None = None
        self._stop = threading.Event()

    # ---- what is going on --------------------------------------------------------------
    def training(self) -> list[str]:
        try:
            return list(self.busy_cb()) if self.busy_cb else []
        except Exception:
            return []               # a broken probe must never block inference

    def plan(self) -> DevicePlan:
        return plan_device(self.cfg.reload_if_changed(), self.training())

    def status(self) -> dict:
        loaded = self._loaded
        plan = self.plan()
        idle = time.monotonic() - self._last_used
        return {
            "loaded": loaded.info.as_dict() if loaded else None,
            "adapter": loaded.adapter.as_dict() if loaded and loaded.adapter else None,
            "stage": loaded.stage if loaded else None,
            "device": loaded.device if loaded else None,
            "loaded_at": loaded.loaded_at if loaded else None,
            "load_s": loaded.load_s if loaded else None,
            "idle_s": idle if loaded else None,
            "unload_in_s": (max(0.0, self.cfg.idle_unload_s - idle) if loaded else None),
            "plan": plan.as_dict(),
            "busy": self._lock_held(),
            "config": self.cfg.as_dict(),
        }

    def _lock_held(self) -> bool:
        if self._lock.acquire(blocking=False):
            self._lock.release()
            return False
        return True

    # ---- loading -----------------------------------------------------------------------
    def load(self, ckpt_id: str, device: str | None = None,
             adapter: str | None = None) -> Loaded:
        """Make `ckpt_id` (optionally + `adapter`) the resident model.

        A no-op if it already is, on the same device, with the same adapter. The reload
        check includes the checkpoint's mtime: `ckpt_last.pt` is rewritten every 500 steps
        by a live run, and "the model I loaded twenty minutes ago" is not the model on
        disk. Re-selecting it in the picker therefore picks up the newer weights, which is
        the whole point of testing during a run. The adapter's mtime is in the key for the
        same reason — a fine-tune in progress rewrites `sft_best.lora.pt`.

        Swapping *only* the adapter still reloads the base. It could be made to swap
        adapters in place on the same weights, and at 11 MB against 1.2 GB that would be a
        real speedup — but it would mean tracking whether the resident model's adapters
        were injected with the same rank and targets, and getting that wrong loads an
        adapter into the wrong shapes. Correct and a second slower wins here.
        """
        info = self.store.get(ckpt_id)
        if info.error:
            raise InferError(f"{info.rel} cannot be loaded: {info.error}")
        ad: Adapter | None = None
        if adapter:
            ad = self.adapters.get(self.adapters.identify(adapter))
            if ad.error:
                raise InferError(f"{ad.rel} cannot be loaded: {ad.error}")
        plan = plan_device(self.cfg.reload_if_changed(), self.training())
        want = device or plan.device
        if want not in ("cuda", "cpu"):
            raise InferError(f"unknown device: {want!r}")
        if want == "cuda" and not torch.cuda.is_available():
            raise InferError("this machine has no CUDA device")

        key = (info.rel, info.mtime, want, ad.rel if ad else None, ad.mtime if ad else None)
        with self._state:
            cur = self._loaded
            if cur and self._key_of(cur) == key:
                self._last_used = time.monotonic()
                return cur

        with self._lock:                       # never swap under a running generation
            self._unload_locked()
            t0 = time.monotonic()
            model, tokenizer = self._build(info, want, ad)
            loaded = Loaded(info=info, model=model, tokenizer=tokenizer, device=want,
                            plan=plan, loaded_at=time.time(), load_s=time.monotonic() - t0,
                            adapter=ad)
            with self._state:
                self._loaded = loaded
            self._last_used = time.monotonic()
        self._start_reaper()
        return loaded

    @staticmethod
    def _key_of(loaded: Loaded):
        a = loaded.adapter
        return (loaded.info.rel, loaded.info.mtime, loaded.device,
                a.rel if a else None, a.mtime if a else None)

    def _build(self, info: Checkpoint, device: str,
               adapter: Adapter | None = None) -> tuple[Transformer, Tokenizer]:
        # weights_only=False: the payload carries the run's config dicts alongside the
        # tensors. These files are written by this project's own trainer, on this machine.
        ckpt = torch.load(info.path, map_location="cpu", weights_only=False)
        # A quantized checkpoint needs its QuantLinears built *before* the weights load:
        # `qweight/scales/qzeros` do not fit an nn.Linear's `weight` slot. Without this
        # branch a quantized .pt dropped into checkpoints/ is listed by the picker and
        # then fails on load, which is a confusing way to find out.
        from ..quant.convert import build_from_checkpoint, is_quantized_checkpoint

        if is_quantized_checkpoint(ckpt):
            model = build_from_checkpoint(ckpt, device=device)
        else:
            model = Transformer(ModelConfig(**ckpt["model_config"]))
            model.load_state_dict(ckpt["model"])
            # bf16 on the card (what it trained in, and half the memory); float32 on the
            # CPU, where bf16 matmuls are emulated and *slower* than the wider type.
            model = model.to(device=device, dtype=torch.bfloat16 if device == "cuda"
                             else torch.float32)
            model.eval()

        if adapter is not None:
            self._attach(model, adapter, ckpt, device)

        tok_path = self._tokenizer_path(info, ckpt)
        del ckpt                                # release the 1.2 GB staging copy promptly
        return model, Tokenizer(tok_path)

    def _attach(self, model: Transformer, adapter: Adapter, ckpt: dict, device: str):
        """Put an adapter on the loaded model.

        Strict by default: an adapter is a delta, and applied to the wrong base it does not
        fail, it just quietly makes the model worse. The check compares architecture and
        tokenizer against what the adapter recorded at training time.
        """
        from ..lora.adapter import AdapterError, attach_adapter, load_adapter_file

        try:
            payload = load_adapter_file(adapter.path)
            attach_adapter(model, payload, ckpt=ckpt, strict=True)
        except AdapterError as exc:
            raise InferError(str(exc))
        # The adapters arrive in fp32; match the base so the two halves of every layer are
        # the same dtype and no matmul silently upcasts.
        target = torch.bfloat16 if device == "cuda" else torch.float32
        for p in model.parameters():
            if p.is_floating_point():
                p.data = p.data.to(target)
        model.eval()

    def _tokenizer_path(self, info: Checkpoint, ckpt: dict) -> Path:
        """The tokenizer this checkpoint was trained with — non-negotiable.

        The BPE vocabulary *is* the embedding index. A checkpoint decoded with a different
        tokenizer produces confident, fluent nonsense and no error, which is a genuinely
        horrible thing to debug. So this refuses rather than guesses.
        """
        rel = ((ckpt.get("config") or {}).get("data") or {}).get("tokenizer")
        if not rel:
            raise InferError(
                f"{info.rel} does not record which tokenizer it was trained with, so it "
                "cannot be decoded safely. Generate from it with the CLI and an explicit "
                "--tokenizer if you know which one it was.")
        path = Path(rel)
        if not path.is_absolute():
            path = self.root / rel
        if not path.is_file():
            raise InferError(
                f"{info.rel} was trained with `{rel}`, which is not on disk. The tokenizer "
                "fixes the embedding index, so decoding with a different one would produce "
                "fluent nonsense rather than an error — refusing to guess.")
        return path

    def unload(self) -> bool:
        with self._lock:
            return self._unload_locked()

    def _unload_locked(self) -> bool:
        with self._state:
            loaded, self._loaded = self._loaded, None
        if loaded is None:
            return False
        was_cuda = loaded.device == "cuda"
        del loaded
        if was_cuda:
            # Without this the freed blocks stay in PyTorch's caching allocator and
            # `nvidia-smi` keeps showing them as used — which, on a card being watched by
            # the GPU panel, looks exactly like a leak.
            torch.cuda.empty_cache()
        return True

    # ---- the idle timer ----------------------------------------------------------------
    def _start_reaper(self):
        if self._reaper and self._reaper.is_alive():
            return
        self._stop.clear()
        self._reaper = threading.Thread(target=self._reap, name="infer-idle", daemon=True)
        self._reaper.start()

    def _reap(self):
        """Unload after `idle_unload_s` of no generations.

        A daemon thread that wakes once a second and mostly does nothing. It exits once it
        has unloaded, so an engine nobody is using costs a dead thread object rather than a
        timer that runs forever.
        """
        while not self._stop.wait(1.0):
            if self._loaded is None:
                return
            if time.monotonic() - self._last_used < self.cfg.idle_unload_s:
                continue
            # Skip this round rather than block: a generation running past the idle timeout
            # is, by definition, not idle.
            if self._lock.acquire(blocking=False):
                try:
                    if (self._loaded is not None
                            and time.monotonic() - self._last_used >= self.cfg.idle_unload_s):
                        self._unload_locked()
                        return
                finally:
                    self._lock.release()

    def close(self):
        self._stop.set()
        self.unload()

    # ---- prompting ---------------------------------------------------------------------
    def build_prompt(self, loaded: Loaded, mode: str, *, prompt: str = "",
                     messages: list[dict] | None = None,
                     system: str | None = None) -> tuple[list[int], int | None, str]:
        """Turn a request into token ids. Returns `(ids, stop_id, rendered)`.

        `rendered` is the exact text the model was shown, kept so the history record can
        answer "what did I actually ask it?" — for chat that is the whole ChatML transcript,
        not just the last thing you typed.
        """
        tok = loaded.tokenizer
        if mode in ("complete", "code"):
            if not prompt.strip():
                raise InferError("nothing to complete — type a prompt first.")
            # BOS matters: every training document began with it, so a continuation that
            # starts without one is asking the model to resume mid-document.
            return tok.encode(prompt, bos=True), tok.eos_id, prompt

        if mode == "chat":
            # `loaded.stage`, not `loaded.info.stage`: an SFT adapter on a base checkpoint
            # is a chat model, and this is the gate that has to know it.
            if loaded.stage == "base":
                raise InferError(
                    f"{loaded.info.rel} is a base model: it has only ever been trained to "
                    "continue text and has never seen a chat turn, so chat would return "
                    "noise. Use Complete, run Phase 3 (scripts/postrain.sh), or attach a "
                    "chat adapter (the Finetune tab).")
            turns = list(messages or [])
            if prompt.strip():
                turns.append({"role": "user", "content": prompt})
            if not turns:
                raise InferError("nothing to say — type a message first.")
            sys_prompt = self.cfg.system if system is None else system
            if sys_prompt:
                turns = [{"role": "system", "content": sys_prompt}] + turns
            ids, _ = tok.render_chat(turns, add_generation_prompt=True)
            return ids, tok.im_end_id, self._render_text(turns)

        raise InferError(f"unknown mode: {mode!r} (complete, chat or code)")

    @staticmethod
    def _render_text(turns: list[dict]) -> str:
        return "\n".join(f"<|im_start|>{t['role']}\n{t['content']}<|im_end|>" for t in turns)

    # ---- generation --------------------------------------------------------------------
    def stream(self, ckpt_id: str, mode: str, *, prompt: str = "",
               messages: list[dict] | None = None, system: str | None = None,
               params: SamplingParams | None = None, device: str | None = None,
               adapter: str | None = None) -> Iterator[tuple[str, object]]:
        """Generate, yielding `("start", meta)`, then `("delta", text)`, then `("done", stats)`.

        The tuple shape mirrors `portal.explain.Ollama.chat`, so the page consumes a local
        checkpoint and a local Ollama model through the same few lines of JavaScript.

        Closing the generator early — which is what happens when the browser goes away —
        stops the decode loop and releases the lock.

        Everything that can fail with a useful message — no such checkpoint, chat asked of
        a base model, a missing tokenizer — is checked *here*, before the generator is
        returned, so the portal can answer with an ordinary 4xx instead of an error buried
        inside a stream it has already committed to. (Which is why this is a plain function
        returning a generator, and not itself a generator: in a generator, nothing runs
        until the first `next()`.)
        """
        if len(prompt) > self.cfg.max_prompt_chars:
            raise InferError(f"prompt is longer than {self.cfg.max_prompt_chars} characters; "
                             "the model's context is only "
                             f"{self.store.get(ckpt_id).max_seq_len or '?'} tokens anyway.")
        loaded = self.load(ckpt_id, device=device, adapter=adapter)
        sp = (params or self.cfg.sampling).clamp(loaded.model.cfg.max_seq_len)
        ids, stop_id, rendered = self.build_prompt(loaded, mode, prompt=prompt,
                                                   messages=messages, system=system)

        # Truncate from the *left*: the newest context is the part that matters, and a chat
        # whose history has outgrown a 1024-token window should lose its oldest turns
        # rather than its current question.
        ctx = loaded.model.cfg.max_seq_len
        room = ctx - sp.max_new_tokens
        truncated = 0
        if room < 1:
            room, sp.max_new_tokens = ctx // 2, ctx // 2
        if len(ids) > room:
            truncated, ids = len(ids) - room, ids[-room:]

        # A cheap look at the lock, so "someone else is generating" is a 409 with a sentence
        # rather than an error event inside a stream. The real acquisition happens in the
        # generator; the gap between the two is a few microseconds and losing that race
        # costs a `busy_wait_s` wait, not a wrong answer.
        if self._lock_held():
            raise InferError("the model is busy with another generation — one at a time, so "
                             "they do not halve each other's speed.")

        return self._stream(loaded, mode, ids, stop_id, rendered, sp, truncated)

    def _stream(self, loaded: Loaded, mode: str, ids: list[int], stop_id: int | None,
                rendered: str, sp: SamplingParams,
                truncated: int) -> Iterator[tuple[str, object]]:
        if not self._lock.acquire(timeout=self.cfg.busy_wait_s):
            raise InferError("the model is busy with another generation — one at a time, so "
                             "they do not halve each other's speed.")
        try:
            yield "start", {
                "checkpoint": loaded.info.as_dict(),
                "adapter": loaded.adapter.as_dict() if loaded.adapter else None,
                "device": loaded.device,
                "plan": loaded.plan.as_dict(), "mode": mode,
                "prompt_tokens": len(ids), "truncated_tokens": truncated,
                "params": sp.as_dict(), "rendered": rendered,
                "load_s": loaded.load_s,
            }
            if sp.seed is not None:
                torch.manual_seed(sp.seed)      # reproducible sampling, for A/B-ing a prompt

            decoder = IncrementalDecoder(loaded.tokenizer,
                                         skip_ids={stop_id} if stop_id is not None else set())
            n, t0, first_at = 0, time.monotonic(), None
            finish = "length"
            for token in stream_generate(
                    loaded.model, ids, max_new_tokens=sp.max_new_tokens,
                    temperature=sp.temperature, top_k=sp.top_k or None, top_p=sp.top_p,
                    repetition_penalty=sp.repetition_penalty, eos_id=stop_id,
                    device=loaded.device):
                if token == stop_id:
                    finish = "stop"
                    break
                n += 1
                if first_at is None:
                    first_at = time.monotonic() - t0
                piece = decoder.push(token)
                if piece:
                    yield "delta", piece
            tail = decoder.flush()
            if tail:
                yield "delta", tail

            elapsed = time.monotonic() - t0
            yield "done", {
                "text": decoder.text, "tokens": n, "elapsed_s": elapsed,
                "tok_per_s": (n / elapsed) if elapsed > 0 else None,
                "first_token_s": first_at, "finish": finish,
                "prompt_tokens": len(ids), "truncated_tokens": truncated,
                "device": loaded.device, "params": sp.as_dict(),
                "adapter": loaded.adapter.as_dict() if loaded.adapter else None,
                "provenance": {**loaded.info.provenance(),
                               "adapter": loaded.adapter.rel if loaded.adapter else None,
                               "stage": loaded.stage},
            }
        finally:
            self._last_used = time.monotonic()
            self._lock.release()

    def generate(self, ckpt_id: str, mode: str, **kw) -> dict:
        """The whole answer at once — what the CLI's one-shot mode and the task runner use."""
        text, stats = "", {}
        for kind, payload in self.stream(ckpt_id, mode, **kw):
            if kind == "delta":
                text += payload
            elif kind == "done":
                stats = dict(payload)
        stats.setdefault("text", text)
        return stats
