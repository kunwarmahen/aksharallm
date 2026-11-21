"""The Audio tab's TTS panel: the saved TTS reports, paired comparisons, and Speak.

* **reports** — every `audio tts-report` on disk (`logs/audio/tts-*.json`) as one row: which
  speakers (seen / unseen / the author), how the voice was given (embedding, or cloned from a
  prompt), WER beside the original's and the codec's, and the voice cosine against its
  ceilings. Through `audio.ttsreports`, the same reader `audio tts-reports` prints.
* **compare** — two reports on the same sentences, paired per sentence (words and voice),
  through `audio.ttsreports.compare`, the function `audio tts-compare` prints. The tab offers
  the natural pairs (a `-prompt` report against its `-promptitems` control) first.
* **clips** — one sentence's original / codec / TTS WAVs, inlined as `data:` URIs.
* **speak / tts-report jobs** — detached `audio speak` and `audio tts-report` through
  `CliJobs`. Speak can clone a voice from any sentence a report already holds (its original
  WAV and its transcript), so nothing has to be uploaded.

Read with: docs/21-audio.md -- the chapter this implements; it ends with the order to read
these files in.
"""

from __future__ import annotations

import base64
import json
import re
import time
from pathlib import Path

import torch

from .jobs import CliJobs, JobError, training

REPORTS = Path("logs/audio")
SPEAK = Path("logs/audio/speak")
MAX_TEXT = 300
SPEAKER_RE = re.compile(r"^(LJ|\d{1,6}|me)$")


class TtsError(RuntimeError):
    pass


def _uri(path: Path) -> str | None:
    if not path.is_file():
        return None
    return "data:audio/wav;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


class Tts:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.jobs = CliJobs(self.root, "logs/audio/jobs", "aksharallm.audio")

    def checkpoints(self) -> list[dict]:
        """Audio LMs trained for TTS (`stage: audiolm`, `task: tts`)."""
        out = []
        for path in sorted(self.root.glob("checkpoints/*/ckpt_best.pt")):
            try:
                blob = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
            except Exception:
                continue
            if blob.get("stage") != "audiolm" or blob.get("task") != "tts":
                continue
            lm = blob.get("audiolm") or {}
            out.append({"rel": str(path.relative_to(self.root)), "step": blob.get("step"),
                        "multi": bool(lm.get("speaker_dim")),
                        "stop_head": bool(lm.get("predict_stop")),
                        "mtime": path.stat().st_mtime})
        return sorted(out, key=lambda r: r["mtime"], reverse=True)

    def reports(self) -> list[dict]:
        from ..audio.ttsreports import load_all
        rows = load_all(self.root)
        names = {r["name"] for r in rows}
        for r in rows:
            # The control a cloned report should be read against, if it is on disk.
            r["control"] = (r["name"].replace("-prompt.json", "-promptitems.json")
                            if r["name"].endswith("-prompt.json") else None)
            if r["control"] not in names:
                r["control"] = None
        return rows

    def spoken(self) -> list[dict]:
        out = []
        for p in sorted((self.root / SPEAK).glob("*.wav"), key=lambda p: p.stat().st_mtime,
                        reverse=True)[:12]:
            side = p.with_suffix(".json")
            meta = {}
            if side.is_file():
                try:
                    meta = json.loads(side.read_text())
                except (OSError, ValueError):
                    pass
            out.append({"name": p.name, "text": meta.get("text"), "voice": meta.get("voice"),
                        "checkpoint": meta.get("checkpoint"), "when": int(p.stat().st_mtime)})
        return out

    def overview(self) -> dict:
        return {"checkpoints": self.checkpoints(), "reports": self.reports(),
                "spoken": self.spoken(), "codes": self.codes(), "jobs": self.jobs.status(),
                "training": training(self.root)}

    # ---- one report -----------------------------------------------------------------------

    def _load(self, name: str) -> tuple[Path, dict]:
        if "/" in name or not (name.startswith("tts-") and name.endswith(".json")):
            raise TtsError("not a TTS report name")
        p = self.root / REPORTS / name
        if not p.is_file():
            raise TtsError(f"no report {name!r}")
        return p, json.loads(p.read_text())

    def report(self, name: str) -> dict:
        from ..audio.ttsreports import wav_dir
        p, blob = self._load(name)
        rows = blob.get("rows") or {}
        items = (rows.get("tts") or {}).get("items") or []
        orig = (rows.get("original") or {}).get("items") or []
        voice = (blob.get("voice") or {}).get("per_item") or []
        from ..audio.ttsreports import finite
        return finite({"name": name, "wavs": (self.root / wav_dir(p, blob)).is_dir(),
                "sentences": [{"index": j, "ref": it["ref"], "heard": it["hyp"],
                               "original_heard": orig[j]["hyp"] if j < len(orig) else None,
                               "speaker": voice[j]["speaker"] if j < len(voice) else None,
                               "voice": voice[j]["tts_own"] if j < len(voice) else None}
                              for j, it in enumerate(items)]})

    def clip(self, name: str, index: int) -> dict:
        from ..audio.ttsreports import wav_dir
        p, blob = self._load(name)
        d = self.root / wav_dir(p, blob)
        j = int(index)
        return {kind: _uri(d / f"{j:02d}-{kind}.wav") for kind in ("original", "codec", "tts")}

    def compare(self, a: str, b: str) -> dict:
        from ..audio.ttsreports import compare
        try:
            return {"a": a, "b": b, **compare(self._load(a)[1], self._load(b)[1]),
                    "command": f"python -m aksharallm.audio tts-compare {REPORTS}/{a} {REPORTS}/{b}"}
        except ValueError as e:
            raise TtsError(str(e)) from e

    def spoken_clip(self, name: str) -> dict:
        if "/" in name or not name.endswith(".wav"):
            raise TtsError("not a clip name")
        return {"wav": _uri(self.root / SPEAK / name)}

    # ---- jobs -----------------------------------------------------------------------------

    def _ckpt(self, rel: str) -> dict:
        for c in self.checkpoints():
            if c["rel"] == rel:
                return c
        raise JobError(f"not a TTS checkpoint: {rel!r}")

    def start(self, spec: dict) -> dict:
        from ..audio.ttsreports import wav_dir
        kind = spec.get("kind")
        cpu = ["--cpu"] if training(self.root) else []
        ck = self._ckpt(str(spec.get("checkpoint") or ""))
        if kind == "speak":
            text = " ".join(str(spec.get("text") or "").split())
            if not text:
                raise JobError("give it something to say")
            if len(text) > MAX_TEXT:
                raise JobError(f"keep it under {MAX_TEXT} characters (one or two sentences)")
            out = SPEAK / f"portal-{time.strftime('%Y%m%d-%H%M%S')}.wav"
            argv = ["speak", ck["rel"], text, "--out", str(out), *cpu]
            voice = "its single voice"
            if ck["multi"]:
                src = spec.get("clone") or {}
                if src.get("report"):
                    # Clone from a sentence a report already holds: its original WAV + words.
                    p, blob = self._load(str(src["report"]))
                    j = int(src.get("index") or 0)
                    items = blob["rows"]["original"]["items"]
                    if not 0 <= j < len(items):
                        raise JobError("no such sentence in that report")
                    wav = wav_dir(p, blob) / f"{j:02d}-original.wav"
                    if not (self.root / wav).is_file():
                        raise JobError("that report's WAVs are not on disk")
                    argv += ["--prompt", str(wav), "--prompt-text", items[j]["ref"]]
                    voice = f"cloned from sentence {j} of {src['report']}"
                else:
                    who = str(spec.get("speaker") or "LJ")
                    if not SPEAKER_RE.match(who):
                        raise JobError("a speaker is LJ or a LibriSpeech id")
                    argv += ["--speaker-id", who]
                    voice = f"speaker {who} (embedding)"
            (self.root / out).parent.mkdir(parents=True, exist_ok=True)
            (self.root / out).with_suffix(".json").write_text(json.dumps(
                {"text": text, "voice": voice, "checkpoint": ck["rel"]}))
            return self.jobs.start(argv, f"speak: {text[:40]}")
        if kind == "tts-report":
            n = int(spec.get("n") or 20)
            if not 1 <= n <= 100:
                raise JobError("sentences must be 1-100")
            argv = ["tts-report", ck["rel"], "--n", str(n), "--duration", "own", *cpu]
            codes = str(spec.get("codes") or "")
            if codes:
                if codes not in {c["rel"] for c in self.codes()}:
                    raise JobError(f"not a codes directory: {codes!r}")
                argv += ["--codes", codes]
            mode = spec.get("voicing")
            if mode in ("prompt", "prompt-items"):
                if not ck["multi"]:
                    raise JobError("cloning needs a multi-speaker checkpoint")
                argv.append(f"--{mode}")
            return self.jobs.start(argv, f"TTS report on {ck['rel']}")
        raise JobError(f"unknown job {kind!r}")

    def codes(self) -> list[dict]:
        """Codes directories a report can score: other speakers, encoded by the multi-speaker
        codec (`encode-many` writes `speaker_emb.npy` beside them). The training set itself is
        left out, because scoring it with --codes is what the default already does."""
        out = []
        for p in sorted(self.root.glob("data/audio/*-codes/speaker_emb.npy")):
            if p.parent.name == "many-codes":
                continue
            out.append({"rel": str(p.parent.relative_to(self.root))})
        return out
