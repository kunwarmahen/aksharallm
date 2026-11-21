"""Reading and comparing saved TTS reports (`audio tts-report` -> logs/audio/tts-*.json).

A report holds 20-40 sentences, so two reports are compared PAIRED, per sentence: which
sentence came out with fewer word errors, and which came out closer to its own speaker's
voice, then a sign test. A difference in two totals over 40 sentences is mostly noise, and
the pairing is what makes "cloning costs words" a finding rather than an impression.

`compare` refuses two reports on different sentences: the pairing would then be between
unrelated sentences and the counts would mean nothing.

Read with: docs/21-audio.md -- the chapter this implements; it ends with the order to read
these files in.
"""

from __future__ import annotations

import json
import re
from math import comb
from pathlib import Path

from ..asr.measure import edits

REPORTS = Path("logs/audio")


def sign_test(a: int, b: int) -> float:
    n, k = a + b, min(a, b)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0


def wav_dir(path: Path, blob: dict) -> Path:
    """Where a report's WAVs are. Recorded since 2026-11-20; older reports are named
    `tts-<run>-step<N>-<rest>.json` beside a `tts-<run>-<rest>/` folder."""
    if blob.get("out_dir"):
        return Path(blob["out_dir"])
    return path.parent / re.sub(r"-step\d+", "", path.stem, count=1)


def finite(x):
    """NaN/inf -> None, recursively. A one-speaker report has no "different speaker" to
    compare with, so `tts_other` is NaN, and a browser's JSON.parse rejects a whole response
    with one NaN in it (Python's json writes it happily). A gap is shown as a gap."""
    if isinstance(x, float):
        return x if x == x and abs(x) != float("inf") else None
    if isinstance(x, dict):
        return {k: finite(v) for k, v in x.items()}
    if isinstance(x, list):
        return [finite(v) for v in x]
    return x


def summary(path: Path, blob: dict) -> dict:
    rows = blob.get("rows") or {}
    v = blob.get("voice") or {}
    pr = blob.get("prompt")
    return finite({
        "name": path.name, "checkpoint": blob.get("checkpoint"), "step": blob.get("step"),
        "n": blob.get("n"), "duration": blob.get("duration"),
        "speakers": Path(blob["codes"]).name if blob.get("codes") else "seen",
        "voicing": ("cloned (prompt)" if pr is not None
                    else "embedding" if v else "single voice"),
        "wer": {k: r.get("wer") for k, r in rows.items()},
        "cer": {k: r.get("cer") for k, r in rows.items()},
        "runaway": (rows.get("tts") or {}).get("runaway"),
        "length_median": (blob.get("length_ratio") or {}).get("median"),
        "voice": {k: v.get(k) for k in ("tts_own", "tts_other", "codec_own", "same_speaker",
                                        "own_closer", "pairs", "p")} if v else None,
    })


def load_all(root: Path) -> list[dict]:
    out = []
    for p in sorted((root / REPORTS).glob("tts-*.json"), key=lambda p: p.stat().st_mtime,
                    reverse=True):
        try:
            out.append(summary(p, json.loads(p.read_text())))
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return out


def _errors(blob: dict) -> list[int]:
    return [edits(i["ref"].split(), i["hyp"].split()) for i in blob["rows"]["tts"]["items"]]


def compare(a: dict, b: dict) -> dict:
    """b against a, per sentence. `fewer_errors`: sentences where b's TTS was transcribed
    with fewer word errors than a's; `closer_voice`: where b's TTS is closer to the
    speaker's own recording."""
    ra = [i["ref"] for i in a["rows"]["tts"]["items"]]
    rb = [i["ref"] for i in b["rows"]["tts"]["items"]]
    if ra != rb:
        raise ValueError("the two reports are on different sentences, so they cannot be "
                         "paired (different --codes, --n, or prompt filtering)")
    ea, eb = _errors(a), _errors(b)
    better = sum(y < x for x, y in zip(ea, eb))
    worse = sum(y > x for x, y in zip(ea, eb))
    out = {"n": len(ra),
           "wer": {"a": a["rows"]["tts"]["wer"], "b": b["rows"]["tts"]["wer"],
                   "b_better": better, "b_worse": worse, "p": sign_test(better, worse)}}
    va, vb = a.get("voice"), b.get("voice")
    if va and vb and va.get("per_item") and vb.get("per_item"):
        pa = [x["tts_own"] for x in va["per_item"]]
        pb = [x["tts_own"] for x in vb["per_item"]]
        closer = sum(y > x for x, y in zip(pa, pb))
        further = sum(y < x for x, y in zip(pa, pb))
        out["voice"] = {"a": va["tts_own"], "b": vb["tts_own"], "b_closer": closer,
                        "b_further": further, "p": sign_test(closer, further),
                        "ceiling": va.get("codec_own"), "same_speaker": va.get("same_speaker")}
    return finite(out)
