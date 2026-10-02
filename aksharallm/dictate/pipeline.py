"""Audio in, finished text out — the one pipeline the hotkey, the CLI and the browser share.

```mermaid
flowchart LR
    A["16 kHz audio"] --> G{"anything<br/>said?"}
    G -->|"quieter than<br/>min_rms_dbfs"| N["nothing<br/>(no text on silence)"]
    G --> S["split long audio<br/>at its quietest moments"]
    S --> E["Conformer encoder<br/>asr/model.py"]
    E --> B["beam search + word LM<br/>+ YOUR dictionary<br/>asr/decode.py"]
    B --> C["cleanup<br/>dictate/cleanup.py"]
    C --> T["text"]
    T --> H["history.jsonl"]
```

There is exactly one of these. The desktop daemon (`dictate/daemon.py`), `python -m
aksharallm.dictate file`, and the portal's Dictation tab all construct a `Dictator` and call
`dictate()`, so a correction made in the browser changes what the hotkey types next, and a
number measured from the CLI is the number the hotkey gets.

**Settings** live in `configs/portal.yaml` under `dictate:` (the portal's file, not a run's —
see `Settings`). The beam weights default to whatever `asr tune` chose on dev-clean, read from
`logs/asr/tune-*.json`: α and β belong to a model + LM pair, and a hand-copied number goes
stale the day either is retrained.

**Every dictation is appended to `logs/dictate/history.jsonl`** — what was heard, what was
written and every step in between — because the correction loop needs the *shown* text to
diff against, and because "what did it hear when it wrote that?" should never be a mystery.

Read with: docs/23-speech-recognition.md -- the chapter this implements; it ends with the
order to read these files in.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
import yaml

from .cleanup import clean
from .personal import Personal

SAMPLE_RATE = 16_000


@dataclass
class Settings:
    #: A run name or a `.pt`. The recogniser is the ear (docs/23).
    recognizer: str = "asr-libri100"
    #: The punctuation tagger. Missing file = rules only (capital first, full stop last), and
    #: every result says which it got.
    punctuator: str = "checkpoints/punct/ckpt_best.pt"
    #: The word LM for beam search. Missing or `decoder: greedy` = no spelling help.
    lm: str = "data/asr/lm/trigram.npz"
    decoder: str = "beam"
    #: None = the best `asr tune` result for this LM.
    alpha: float | None = None
    beta: float | None = None
    unk_penalty: float | None = None
    beam: int | None = None
    #: The CPU by default: a few seconds of speech is cheap, and a dictation tool that holds
    #: the card would be the thing that kills a training run.
    device: str = "cpu"
    #: How the desktop daemon hands over text: type | paste | clipboard | none.
    output: str = "type"
    #: Recording stops itself after this, in case the second key press never comes.
    max_seconds: float = 120.0
    #: Quieter than this over the whole recording, and it was not speech. The recogniser
    #: already writes nothing on silence (docs/23); this saves the work and says so.
    min_rms_dbfs: float = -55.0
    #: Where the dictionary, corrections and history live.
    personal_dir: str = "logs/dictate"

    @classmethod
    def load(cls, root: str | Path = ".", **overrides) -> "Settings":
        raw = {}
        p = Path(root) / "configs" / "portal.yaml"
        try:
            raw = (yaml.safe_load(p.read_text()) or {}).get("dictate") or {}
        except OSError:
            pass
        known = {f.name for f in fields(cls)}
        bad = set(raw) - known
        if bad:
            raise ValueError(f"unknown key(s) under dictate: in {p}: {sorted(bad)}")
        raw.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**raw)


def best_tuning(root: str | Path, lm_path: str | Path) -> dict | None:
    """The lowest dev WER any `asr tune` run found for this LM, or None."""
    best = None
    for p in sorted((Path(root) / "logs/asr").glob("tune-*.json")):
        try:
            d = json.loads(p.read_text())
        except (OSError, ValueError):
            continue
        if Path(d.get("lm") or "").name != Path(lm_path).name or not d.get("best"):
            continue
        # The best across every tuning run -- not "the newest file", which a narrower
        # follow-up grid would win just by being written last.
        if best is None or d["best"]["wer"] < best["wer"]:
            best = {**d["best"], "beam": d.get("beam", 16), "file": p.name,
                    "greedy_wer": d.get("greedy_wer")}
    return best


def resolve_recognizer(root: Path, name: str) -> Path:
    p = Path(name)
    if p.suffix == ".pt":
        return p if p.is_absolute() else root / p
    for f in ("ckpt_best.pt", "ckpt_last.pt"):
        q = root / "checkpoints" / name / f
        if q.is_file():
            return q
    raise FileNotFoundError(f"no recogniser checkpoint for {name!r} under checkpoints/")


def split_points(x: np.ndarray, sr: int = SAMPLE_RATE, target_s: float = 25.0,
                 search_s: float = 5.0) -> list[int]:
    """Cut points for audio longer than `target_s`: the quietest 100 ms within `search_s`
    before each boundary. Cutting mid-word costs that word; cutting in a pause costs nothing."""
    n = len(x)
    step = int(target_s * sr)
    if n <= step + int(search_s * sr):
        return []
    hop = sr // 20
    win = sr // 10
    cuts, at = [], 0
    while n - at > step + int(search_s * sr):
        lo, hi = at + step - int(search_s * sr), at + step
        best, best_e = hi, math.inf
        for s in range(lo, hi - win, hop):
            e = float(np.mean(x[s : s + win] ** 2))
            if e < best_e:
                best, best_e = s + win // 2, e
        cuts.append(best)
        at = best
    return cuts


class Dictator:
    def __init__(self, settings: Settings | None = None, root: str | Path = "."):
        self.root = Path(root).resolve()
        self.s = settings or Settings.load(self.root)
        self._model = None
        self._lm = None
        self._punct = None
        self._punct_tried = False
        self.personal = Personal(self.root / self.s.personal_dir, known=self._known)

    # ---- components, loaded on first use ------------------------------------------------

    def recognizer(self):
        if self._model is None:
            from ..asr.train import load_recognizer
            path = resolve_recognizer(self.root, self.s.recognizer)
            self._model, blob = load_recognizer(path, self.s.device)
            self.recognizer_info = {"path": str(path.relative_to(self.root)) if path.is_relative_to(self.root) else str(path),
                                    "step": blob.get("step")}
        return self._model

    def lm(self):
        if self._lm is None and self.s.decoder == "beam":
            p = self.root / self.s.lm
            if p.is_file():
                from ..asr.ngram import TrigramLM
                self._lm = TrigramLM.load(p)
        return self._lm

    def punctuator(self):
        if not self._punct_tried:
            self._punct_tried = True
            p = self.root / self.s.punctuator
            if p.is_file():
                from .tagger import Punctuator
                self._punct = Punctuator.load(p, self.s.device)
        return self._punct

    def _known(self, w: str) -> bool:
        lm = self._lm
        return lm is not None and w in lm.index

    def describe(self) -> dict:
        t = best_tuning(self.root, self.s.lm) or {}
        p = self.punctuator()
        return {
            "recognizer": self.s.recognizer,
            "decoder": "greedy" if self.s.decoder != "beam" or self.lm() is None else "beam",
            "lm": self.s.lm if self.lm() is not None else None,
            "tuning": t or None,
            "punctuator": (f"{self.s.punctuator} (step {getattr(p, 'step', '?')})" if p
                           else "rules only — no tagger checkpoint at " + self.s.punctuator),
            "personal": self.personal.summary(),
            "device": self.s.device,
        }

    def _decoder(self):
        from ..asr.decode import BeamDecoder
        lm = self.lm()
        if lm is None:
            return None
        t = best_tuning(self.root, self.s.lm) or {}

        def pick(name, default):
            v = getattr(self.s, name)
            return v if v is not None else t.get(name, default)
        return BeamDecoder(lm=lm, alpha=pick("alpha", 0.8), beta=pick("beta", 2.0),
                           unk_penalty=pick("unk_penalty", -24.0), beam=int(pick("beam", 16)),
                           dictionary=self.personal.dictionary())

    # ---- the pipeline -----------------------------------------------------------------------

    def hear(self, x: np.ndarray) -> dict:
        """Audio -> the recogniser's words, greedy and (if available) beam."""
        from ..asr import vocab
        from ..asr.ctc import greedy_decode
        from ..asr.measure import log_probs
        import torch

        model = self.recognizer()
        cuts = split_points(x)
        pieces = np.split(x, cuts) if cuts else [x]
        t0 = time.time()
        lps = log_probs(model, [p.astype(np.float32) for p in pieces], self.s.device)
        t_enc = time.time() - t0
        greedy = " ".join(" ".join(vocab.decode(greedy_decode(torch.from_numpy(lp)[None],
                                                               torch.tensor([len(lp)]))[0]).split())
                          for lp in lps).strip()
        dec = self._decoder()
        t0 = time.time()
        heard = " ".join(dec.decode(lp) for lp in lps).strip() if dec else greedy
        return {"greedy": greedy, "heard": heard, "segments": len(pieces),
                "encoder_ms": round(t_enc * 1000, 1), "decoder_ms": round((time.time() - t0) * 1000, 1),
                "decoder": "beam" if dec else "greedy",
                "dictionary": len(dec.dictionary) if dec else 0}

    def dictate(self, x: np.ndarray, sr: int = SAMPLE_RATE, source: str = "api") -> dict:
        """Audio (float32 mono) -> a result record, also appended to the history."""
        from ..audio.io import resample
        x = np.asarray(x, dtype=np.float32)
        if sr != SAMPLE_RATE:
            x = resample(x, int(sr), SAMPLE_RATE).astype(np.float32)
        seconds = len(x) / SAMPLE_RATE
        rms = float(np.sqrt(np.mean(x ** 2))) if x.size else 0.0
        rms_db = 20 * math.log10(max(rms, 1e-9))
        t0 = time.time()
        rec = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "id": f"{time.time():.3f}",
               "source": source, "seconds": round(seconds, 2), "rms_dbfs": round(rms_db, 1)}
        if seconds < 0.2 or rms_db < self.s.min_rms_dbfs:
            rec.update(text="", heard="", greedy="", steps=[],
                       note=f"nothing heard ({rms_db:.0f} dBFS, gate {self.s.min_rms_dbfs:.0f})")
        else:
            h = self.hear(x)
            c = clean(h["heard"], self.punctuator(), self.personal)
            rec.update(h, text=c.text, steps=c.steps,
                       punctuator="tagger" if self.punctuator() else "rules")
        rec["ms"] = round((time.time() - t0) * 1000, 1)
        rec["realtime"] = round(seconds / max(rec["ms"] / 1000, 1e-9), 1)
        self._history(rec)
        return rec

    def clean_text(self, transcript: str) -> dict:
        """Cleanup alone, on words you supply — what the tagger does without the ear."""
        c = clean(transcript, self.punctuator(), self.personal)
        return {"text": c.text, "steps": c.steps,
                "punctuator": "tagger" if self.punctuator() else "rules"}

    # ---- history -------------------------------------------------------------------------

    @property
    def history_path(self) -> Path:
        return self.root / self.s.personal_dir / "history.jsonl"

    def _history(self, rec: dict) -> None:
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.history_path, "a") as f:
            f.write(json.dumps(rec) + "\n")

    def history(self, limit: int = 30) -> list[dict]:
        try:
            lines = self.history_path.read_text().splitlines()[-limit:]
        except OSError:
            return []
        out = []
        for line in reversed(lines):
            try:
                out.append(json.loads(line))
            except ValueError:
                pass
        return out

    def correct(self, shown: str, corrected: str, heard: str | None = None) -> dict:
        """The person fixed the text: learn from it (dictate/personal.py)."""
        self.lm()   # so replacements between two known words are refused
        return self.personal.learn(shown, corrected, heard=heard)
