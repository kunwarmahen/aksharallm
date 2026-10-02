"""The Dictation tab's back end: speak, and see what our own recogniser heard.

Phase 7 is "dictation that is still right on day two", and the tab is built in that order:

1. **Say something** — the browser records the microphone (or decodes a file you drop on
   it), sends 16-bit samples at whatever rate it has, and gets back what the recogniser
   wrote. It is the only tab whose input is *you*, which is the point: LibriSpeech is read
   audiobooks, and a recogniser's test set is the most forgiving speech it will ever hear.
2. **The five day-two checks**, as they come in. Silence first, because it is the one that is
   already measured on every eval (`asr/measure.py`).
3. **Every evaluation so far** — WER with its error bar and the worst speakers beside the
   mean, from `logs/asr/*.json`. Never the mean alone (day-two problem 2).

**Resampling is ours and is declared.** A browser microphone runs at 44.1 or 48 kHz; the model
hears 16. The conversion is `audio/io.resample` — the windowed-sinc resampler built and tested
for the audio phase — and the response says the rate it came in at, because a silent
conversion is exactly what trap 5 of the audio phase warns about.

Runs inline, like `vision` and `audio`: a few seconds of speech is one forward pass of a 20M
encoder. Device policy is the repo's standing one, borrowed from the Playground: the CPU
while a training run holds the card.

Read with: docs/23-speech-recognition.md -- the chapter this implements; it ends with the
order to read these files in.
"""

from __future__ import annotations

import base64
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch

#: Thirty seconds of speech is a long dictated paragraph and a few hundred milliseconds of
#: compute. Past that, a request is more likely a mistake than a sentence.
MAX_SECONDS = 30.0
#: The request body limit for this route only: 30 s at 48 kHz as base64 int16 is ~3.9 MB.
MAX_BODY = 8 * 1024 * 1024
#: The word LM `asr lm build` writes. Optional: without it the tab offers greedy only.
LM_PATH = Path("data/asr/lm/trigram.npz")
#: A personal dictionary can be long, but not a novel.
MAX_DICT_WORDS = 2000


class DictationError(RuntimeError):
    """Something to show as a message rather than a stack trace."""


def _alive(pid_file: Path) -> bool:
    try:
        os.kill(int(pid_file.read_text().strip()), 0)
        return True
    except (OSError, ValueError):
        return False


class Dictation:
    def __init__(self, root: Path | None = None, device_for=None):
        self.root = Path(root) if root else Path.cwd()
        self._device_for = device_for
        self._cache: dict = {}
        self._lm = None
        self.jobs = AsrJobs(self)

    def device(self) -> tuple[str, str]:
        if self._device_for is not None:
            return self._device_for()
        if torch.cuda.is_available():
            return "cuda", "the card is free"
        return "cpu", "no CUDA device"

    # ---- discovery --------------------------------------------------------------------

    def checkpoints(self) -> list[dict]:
        """Recognisers, identified by `stage == "asr"` rather than by a filename convention."""
        out = []
        for path in sorted(self.root.glob("checkpoints/*/ckpt_*.pt")):
            try:
                blob = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
            except Exception:
                continue
            if blob.get("stage") != "asr":
                continue
            a = blob.get("asr") or {}
            out.append({
                "rel": str(path.relative_to(self.root)),
                "run": path.parent.name,
                "step": blob.get("step"),
                "best_wer": blob.get("best_val") if math.isfinite(blob.get("best_val") or math.inf) else None,
                "shape": f"d={a.get('d_model')} x{a.get('n_layers')}",
                "mtime": path.stat().st_mtime,
            })
        # Best checkpoints first, then newest: `ckpt_best` is what you meant to talk to.
        return sorted(out, key=lambda r: (not r["rel"].endswith("ckpt_best.pt"), -r["mtime"]))

    def results(self) -> list[dict]:
        """Every `asr eval` result, newest first, without the per-utterance examples."""
        out = []
        for p in sorted((self.root / "logs/asr").glob("*.json")):
            try:
                d = json.loads(p.read_text())
            except (OSError, ValueError):
                continue
            if d.get("kind") != "asr_eval":
                continue
            sp = d.get("speakers") or {}
            rates = sorted(v["wer"] for v in sp.values()) if sp else []
            out.append({
                "file": p.name, "run": d.get("run"), "step": d.get("step"),
                "corpus": Path(d.get("corpus", "")).name, "time": d.get("time"),
                "wer": d.get("wer"), "wer_pm": d.get("wer_pm"), "cer": d.get("cer"),
                "words": d.get("words"), "utts": d.get("utts"),
                "speakers": len(rates),
                "median_speaker": rates[len(rates) // 2] if rates else None,
                "worst": (d.get("worst_speakers") or [])[:3],
                "silence_chars": (d.get("silence") or {}).get("chars"),
                "decoder": d.get("decoder", "greedy"),
                "decoder_desc": d.get("decoder_desc"),
                "dictionary_size": d.get("dictionary_size", 0),
                "names": d.get("names"),
            })
        return sorted(out, key=lambda r: r.get("time") or "", reverse=True)

    def runs(self) -> list[dict]:
        """Recogniser runs and where they have got to: step, latest val WER, silence chars."""
        out = []
        for cfg_path in sorted(self.root.glob("configs/*.yaml")):
            if not cfg_path.read_text().startswith("asr:") and "\nasr:" not in cfg_path.read_text():
                continue
            name = cfg_path.stem
            run_dir = self.root / "checkpoints" / name
            step = wer = silence = None
            log = run_dir / "train_log.jsonl"
            if log.is_file():
                for line in reversed(log.read_text().splitlines()[-600:]):
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if wer is None and "val_wer" in rec:
                        wer, silence = rec["val_wer"], rec.get("silence_chars")
                    if step is None and "step" in rec:
                        step = rec["step"]
                    if step is not None and wer is not None:
                        break
            out.append({"name": name, "training": _alive(run_dir / "train.pid"),
                        "step": step, "val_wer": wer, "silence_chars": silence})
        return out

    def tuned(self) -> dict | None:
        """The weights the latest `asr tune` chose on dev-clean, or None. The beam option uses
        these rather than defaults: alpha and beta are a property of a model + LM pair."""
        best = None
        for p in sorted((self.root / "logs/asr").glob("tune-*.json")):
            try:
                d = json.loads(p.read_text())
            except (OSError, ValueError):
                continue
            if Path(d.get("lm") or "").name != LM_PATH.name or not d.get("best"):
                continue
            # The best dev WER across every tuning run for this LM -- not "the newest file",
            # which a narrower follow-up grid would win just by being written last.
            if best is None or d["best"]["wer"] < best["wer"]:
                best = {**d["best"], "beam": d.get("beam", 16), "file": p.name,
                        "greedy_wer": d.get("greedy_wer")}
        return best

    def overview(self) -> dict:
        device, why = self.device()
        return {"checkpoints": self.checkpoints(), "results": self.results(), "runs": self.runs(),
                "device": device, "device_reason": why, "max_seconds": MAX_SECONDS,
                "lm": str(LM_PATH) if (self.root / LM_PATH).is_file() else None,
                "tuned": self.tuned()}

    def _language_model(self):
        """Loaded once and kept: it is a few hundred MB of arrays and a second to load."""
        if self._lm is None:
            from ..asr.ngram import TrigramLM
            path = self.root / LM_PATH
            if not path.is_file():
                raise DictationError("no word LM yet — build one: python -m aksharallm.asr lm build")
            self._lm = TrigramLM.load(path)
        return self._lm

    # ---- the model ----------------------------------------------------------------------

    def _model(self, rel: str):
        from ..asr.train import load_recognizer

        device, _ = self.device()
        key = f"{rel}@{device}"
        if self._cache.get("key") != key:
            path = (self.root / rel).resolve()
            if not str(path).startswith(str(self.root.resolve())):
                raise DictationError("checkpoint path escapes the repository")
            if not path.is_file():
                raise DictationError(f"no such checkpoint: {rel}")
            try:
                model, blob = load_recognizer(path, device)
            except ValueError as e:
                raise DictationError(str(e)) from e
            self._cache = {"key": key, "model": model, "step": blob.get("step")}
        return self._cache["model"], self._cache["step"], device

    def transcribe(self, checkpoint: str, pcm_b64: str, sample_rate: int,
                   decoder: str = "greedy", dictionary: str = "") -> dict:
        """Base64 little-endian int16 mono at `sample_rate` -> what the recogniser wrote.

        `decoder="beam"` runs the prefix beam search with the word LM and the dictionary (one
        word per line or space); the response carries the greedy text too, so the difference
        the language model made is on screen rather than asserted.
        """
        from ..asr import vocab
        from ..asr.ctc import greedy_decode
        from ..asr.decode import BeamDecoder
        from ..asr.measure import log_probs
        from ..audio.io import resample

        if not checkpoint:
            raise DictationError("pick a recogniser first")
        try:
            raw = base64.b64decode(pcm_b64, validate=True)
        except (ValueError, TypeError) as e:
            raise DictationError("the audio did not arrive as base64 int16") from e
        if not 8_000 <= int(sample_rate) <= 192_000:
            raise DictationError(f"a sample rate of {sample_rate} Hz is not a microphone's")
        x = np.frombuffer(raw[: len(raw) // 2 * 2], dtype="<i2").astype(np.float32) / 32768.0
        seconds_in = len(x) / sample_rate
        if seconds_in < 0.2:
            raise DictationError("that was under 0.2 seconds of audio — hold the button while you speak")
        if seconds_in > MAX_SECONDS:
            raise DictationError(f"{seconds_in:.0f} s is past the {MAX_SECONDS:.0f} s this tab accepts")
        model, step, device = self._model(checkpoint)
        sr = model.cfg.sample_rate
        if sample_rate != sr:
            x = resample(x, int(sample_rate), sr).astype(np.float32)
        peak = float(np.abs(x).max()) if x.size else 0.0
        rms = float(np.sqrt(np.mean(x ** 2))) if x.size else 0.0
        t0 = time.time()
        lp = log_probs(model, [x], device)[0]
        greedy = " ".join(vocab.decode(greedy_decode(torch.from_numpy(lp)[None],
                                                     torch.tensor([len(lp)]))[0]).split())
        text, how, words = greedy, "greedy", []
        if decoder == "beam":
            words = [w for w in dictionary.replace(",", " ").split() if w.strip()][:MAX_DICT_WORDS]
            t = self.tuned() or {}
            dec = BeamDecoder(lm=self._language_model(), alpha=t.get("alpha", 0.3),
                              beta=t.get("beta", 1.5), unk_penalty=t.get("unk_penalty", -6.0),
                              beam=t.get("beam", 16), dictionary=set(words))
            text = dec.decode(lp)
            how = (f"beam {dec.beam} + word LM (alpha {dec.alpha}, beta {dec.beta}"
                   + (", tuned on dev-clean" if t else ", untuned defaults") + ")"
                   + (f" + {len(dec.dictionary)} dictionary words" if dec.dictionary else ""))
        ms = (time.time() - t0) * 1000
        return {
            "text": text, "greedy": greedy, "decoder": how,
            "seconds": round(len(x) / sr, 2), "ms": round(ms, 1),
            "realtime": round(len(x) / sr / max(ms / 1000, 1e-9), 1),
            "rate_in": int(sample_rate), "rate_model": sr, "resampled": sample_rate != sr,
            "peak_dbfs": round(20 * math.log10(max(peak, 1e-9)), 1),
            "rms_dbfs": round(20 * math.log10(max(rms, 1e-9)), 1),
            "device": device, "step": step, "checkpoint": checkpoint,
        }

    def silence(self, checkpoint: str) -> dict:
        """Day-two check 1, live: what it writes on five clips with no speech in them."""
        from ..asr.measure import silence_check

        model, step, device = self._model(checkpoint)
        r = silence_check(model, device)
        return {"checkpoint": checkpoint, "step": step, "device": device, **r}


# ---------------------------------------------------------------------------------------
# jobs: everything `python -m aksharallm.asr` does, runnable from the tab
# ---------------------------------------------------------------------------------------

import re as _re  # noqa: E402
import subprocess as _subprocess  # noqa: E402
import sys as _sys  # noqa: E402

_WORD = _re.compile(r"^[a-z']{1,40}$")


class AsrJobs:
    """Fetch, pack, build the LM, tune and evaluate — from the browser, through the CLI.

    **Every job is the exact `python -m aksharallm.asr ...` command a terminal would run**, and
    the panel prints it. There is no second implementation to drift: the browser chooses the
    arguments, the CLI does the work, and the same JSON lands in `logs/asr/` either way.

    One job at a time (`logs/asr/jobs/asr.pid`), detached so closing the page or restarting
    the portal does not kill it. **Success is the exit code**, written by a wrapper shell into
    `<job>.rc` — not "an output file appeared", which is the check that reported every Eval
    audit as failed on success (gotcha 20): five kinds of job write five kinds of file, and an
    exit code is one thing to read for all of them.

    Device policy is the Playground's, like the rest of the tab: `--device cpu` while a run is
    training, so pressing Evaluate cannot be what killed a training run.
    """

    KINDS = ("fetch", "pack", "lm_fetch", "lm", "tune", "eval")

    def __init__(self, dictation: Dictation):
        self.d = dictation
        self.root = dictation.root

    @property
    def dir(self) -> Path:
        p = self.root / "logs" / "asr" / "jobs"
        p.mkdir(parents=True, exist_ok=True)
        return p

    def _current(self) -> dict:
        try:
            return json.loads((self.dir / "current.json").read_text())
        except (OSError, ValueError):
            return {}

    def _pid(self) -> int | None:
        cur = self._current()
        pid = cur.get("pid")
        if not pid:
            return None
        try:
            os.kill(int(pid), 0)
            live = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
        except (OSError, ValueError):
            return None
        # A recycled pid would not reproduce the command line recorded at launch.
        return int(pid) if live == cur.get("cmdline") else None

    # ---- what can be acted on ------------------------------------------------------------

    def corpora(self) -> list[dict]:
        out = []
        for man in sorted(self.root.glob("data/asr/*/manifest.json")) + \
                sorted(self.root.glob("data/audio/synth-asr/manifest.json")):
            if (man.parent / "audio.bin").is_file() and (man.parent / "transcripts.json").is_file():
                try:
                    secs = json.loads(man.read_text()).get("seconds", 0)
                except (OSError, ValueError):
                    secs = 0
                out.append({"rel": str(man.parent.relative_to(self.root)), "hours": round(secs / 3600, 2)})
        return out

    def splits(self) -> list[dict]:
        from ..asr.data import LIBRISPEECH_SIZES_GB
        out = []
        for split, gb in LIBRISPEECH_SIZES_GB.items():
            out.append({"split": split, "gb": gb,
                        "downloaded": (self.root / "data/asr/librispeech/LibriSpeech" / split).is_dir(),
                        "packed": (self.root / "data/asr" / split / "audio.bin").is_file()})
        return out

    def lm_info(self) -> dict | None:
        side = self.root / LM_PATH.with_suffix(".json")
        if not side.is_file():
            return None
        try:
            m = json.loads(side.read_text())["meta"]
        except (OSError, ValueError, KeyError):
            return None
        return {k: m.get(k) for k in ("vocab", "words", "every", "oov_rate", "D2", "D3", "built",
                                      "seconds", "perplexity", "overlap")}

    def status(self, tail: int = 40) -> dict:
        cur = self._current()
        pid = self._pid()
        if cur and cur.get("state") == "running" and pid is None:
            rc = self._rc(cur.get("job", ""))
            cur = {**cur, "state": "done" if rc == 0 else ("failed" if rc is not None else "lost"),
                   "rc": rc}
        log = []
        if cur.get("job"):
            try:
                log = (self.dir / f"{cur['job']}.log").read_text(errors="replace").splitlines()[-tail:]
            except OSError:
                pass
        corpus_text = (self.root / "data/asr/lm/librispeech-lm-norm.txt.gz").is_file()
        return {"running": pid is not None, "current": cur or None, "log": log,
                "corpora": self.corpora(), "splits": self.splits(), "lm": self.lm_info(),
                "lm_text": corpus_text, "device": self.d.device()[0]}

    def _rc(self, job: str) -> int | None:
        try:
            return int((self.dir / f"{job}.rc").read_text().strip())
        except (OSError, ValueError):
            return None

    # ---- starting one ---------------------------------------------------------------------

    def _ckpt(self, rel: str) -> str:
        if rel not in {c["rel"] for c in self.d.checkpoints()}:
            raise DictationError(f"not a recogniser checkpoint: {rel!r}")
        return rel

    def _corpus(self, rel: str) -> str:
        if rel not in {c["rel"] for c in self.corpora()}:
            raise DictationError(f"not a packed corpus with transcripts: {rel!r}")
        return rel

    @staticmethod
    def _num(v, lo: float, hi: float, name: str) -> str:
        try:
            x = float(v)
        except (TypeError, ValueError):
            raise DictationError(f"{name} must be a number") from None
        if not lo <= x <= hi:
            raise DictationError(f"{name} must be between {lo} and {hi}")
        return f"{x:g}"

    def command(self, spec: dict) -> tuple[list[str], str]:
        """`(argv after "python -m aksharallm.asr", a label)` for a job spec, validated.

        Everything the browser sends is checked against what exists (a checkpoint from the
        list, a corpus that is packed, a split LibriSpeech has) or parsed as a bounded number
        — nothing from the request reaches a command line unexamined.
        """
        from ..asr.data import LIBRISPEECH_SIZES_GB
        kind = spec.get("kind")
        if kind not in self.KINDS:
            raise DictationError(f"unknown job {kind!r}")
        device = "cpu" if self.d.device()[0] == "cpu" else "cuda"
        if kind in ("fetch", "pack"):
            split = spec.get("split")
            if split not in LIBRISPEECH_SIZES_GB:
                raise DictationError(f"unknown LibriSpeech split {split!r}")
            if kind == "pack" and not (self.root / "data/asr/librispeech/LibriSpeech" / split).is_dir():
                raise DictationError(f"{split} is not downloaded yet — fetch it first")
            return [kind, split], f"{kind} {split}"
        if kind == "lm_fetch":
            return ["lm", "fetch"], "download the LM text (1.5 GB)"
        if kind == "lm":
            if not (self.root / "data/asr/lm/librispeech-lm-norm.txt.gz").is_file():
                raise DictationError(
                    "the LM text is not downloaded: data/asr/lm/librispeech-lm-norm.txt.gz "
                    "(OpenSLR 11, 1.5 GB) — see docs/23 § Running it")
            every = int(self._num(spec.get("every", 1), 1, 1000, "every"))
            argv = ["lm", "build", "--every", str(every), "--check", "data/asr/dev-clean"]
            if spec.get("overlap"):
                argv.append("--overlap")
            if every > 1:
                # Never overwrite the full LM with a sample of it from a button.
                argv += ["--out", f"data/asr/lm/trigram-every{every}.npz"]
            return argv, f"build the word LM (1 line in {every})"
        ckpt = self._ckpt(str(spec.get("checkpoint") or ""))
        if kind == "tune":
            corpus = self._corpus(str(spec.get("corpus") or "data/asr/dev-clean"))
            if "test" in corpus:
                raise DictationError("tune on dev, never on test")
            limit = str(int(self._num(spec.get("limit", 800), 50, 5000, "limit")))
            return (["tune", ckpt, "--corpus", corpus, "--limit", limit, "--device", device],
                    f"tune the beam on {Path(corpus).name}")
        # eval
        corpus = self._corpus(str(spec.get("corpus") or "data/asr/test-clean"))
        argv = ["eval", ckpt, "--corpus", corpus, "--device", device]
        if spec.get("decoder") == "beam":
            if not (self.root / LM_PATH).is_file():
                raise DictationError("no word LM yet — build it first")
            t = self.d.tuned() or {}
            argv += ["--decoder", "beam", "--lm", str(LM_PATH),
                     "--alpha", self._num(spec.get("alpha", t.get("alpha", 0.8)), 0, 5, "alpha"),
                     "--beta", self._num(spec.get("beta", t.get("beta", 2.0)), -5, 20, "beta"),
                     "--unk-penalty", self._num(spec.get("unk_penalty", t.get("unk_penalty", -24)),
                                                -100, 0, "unk penalty")]
            words = [w.lower() for w in str(spec.get("dictionary") or "").replace(",", " ").split()]
            bad = [w for w in words if not _WORD.match(w)]
            if bad:
                raise DictationError(f"dictionary words may only use a-z and ': {bad[:3]}")
            if words:
                p = self.dir / f"dict-{int(time.time())}.txt"
                p.write_text("\n".join(sorted(set(words))[:MAX_DICT_WORDS]) + "\n")
                argv += ["--dict", str(p.relative_to(self.root))]
        return argv, f"evaluate on {Path(corpus).name}" + (" (beam)" if "--decoder" in argv else "")

    def start(self, spec: dict) -> dict:
        if self._pid() is not None:
            raise DictationError("a job is already running — wait for it or stop it")
        argv, label = self.command(spec)
        job = time.strftime("%Y%m%d-%H%M%S") + "-" + argv[0]
        py = _sys.executable
        cli = [py, "-u", "-m", "aksharallm.asr", *argv]
        rc = self.dir / f"{job}.rc"
        # A wrapper shell records the CLI's exit code, which is the whole success signal.
        script = " ".join(_shquote(a) for a in cli) + f"; echo $? > {_shquote(str(rc))}"
        with open(self.dir / f"{job}.log", "wb") as fh:
            proc = _subprocess.Popen(["/bin/sh", "-c", script], cwd=self.root,
                                     stdin=_subprocess.DEVNULL, stdout=fh,
                                     stderr=_subprocess.STDOUT, start_new_session=True)
        try:
            cmdline = Path(f"/proc/{proc.pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
        except OSError:
            cmdline = ""
        cur = {"job": job, "kind": argv[0], "label": label, "state": "running", "pid": proc.pid,
               "started": time.time(), "cmdline": cmdline,
               # What to type to do the same thing from a terminal — shown in the panel.
               "command": "python -m aksharallm.asr " + " ".join(_shquote(a) for a in argv)}
        (self.dir / "current.json").write_text(json.dumps(cur))
        return {"ok": True, **cur}

    def stop(self) -> dict:
        pid = self._pid()
        if pid is None:
            raise DictationError("no job is running")
        os.killpg(pid, 15)   # the shell and the CLI under it
        cur = self._current()
        (self.dir / "current.json").write_text(json.dumps({**cur, "state": "stopped"}))
        return {"ok": True, "stopped": pid}


def _shquote(s: str) -> str:
    import shlex
    return shlex.quote(s)
