"""`audio/transfer.py`: does a one-voice codec carry other voices?

The report's whole reading is per clip, paired: the recogniser on THIS clip's original
against the recogniser on THIS clip's reconstruction. Two ways it could quietly lie are
pinned here — pairing the wrong clips, and counting the direction backwards — plus the clip
picker, which decides whether "forty voices" really is forty voices.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from aksharallm.audio.transfer import measure_corpus, paired, pick_clips


class StubCodec(torch.nn.Module):
    """Identity at every rung except on clips whose first sample is marked: those it
    silences, so a recogniser can no longer hear them."""

    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(1))
        self.cfg = SimpleNamespace(sample_rate=16_000, n_codebooks=8)

    def encode(self, x):
        return x

    def decode(self, codes, n_codebooks=8):
        return codes * 0 if float(codes[0, 0]) == 0.5 else codes


def clip(i, damaged):
    rng = np.random.default_rng(i)
    w = (0.1 * rng.standard_normal(16_000)).astype(np.float32)
    w[0] = 0.5 if damaged else 0.0
    return {"name": f"c{i}", "speaker": f"s{i % 2}", "text": f"word{i} again", "wave": w}


def test_the_paired_count_finds_exactly_the_clips_the_codec_damaged():
    damaged = {1, 4, 5}
    clips = [clip(i, i in damaged) for i in range(8)]

    def transcribe(waves):
        # Originals and reconstructions arrive as separate lists in clip order; a silent
        # reconstruction is heard as nothing, everything else word-perfectly.
        return [c["text"] if np.abs(w[1:]).any() else "" for c, w in zip(clips, waves)]

    rep = measure_corpus(StubCodec(), clips, transcribe, rungs=(4, 8), judged=(8,))
    pw = rep["paired_wer"]["codec8"]
    assert (pw["worse"], pw["better"], pw["same"]) == (3, 0, 5)
    worse = [it["name"] for it in rep["items"] if it["wer"]["codec8"] > it["wer"]["original"]]
    assert worse == ["c1", "c4", "c5"]
    json.dumps(rep)  # the CLI writes this to logs/audio


def test_paired_counts_lower_as_better():
    assert paired([0.0, 0.5, 0.2], [0.3, 0.1, 0.2]) == \
        {"worse": 1, "better": 1, "same": 1, "p": 1.0}
    with pytest.raises(ValueError):
        paired([0.1], [0.1, 0.2])


def write_corpus(d, names, seconds=3.0, sr=16_000):
    n = int(seconds * sr)
    d.mkdir()
    (np.zeros(n * len(names), dtype="<i2") + 100).tofile(d / "audio.bin")
    (d / "manifest.json").write_text(json.dumps({
        "sample_rate": sr, "offsets": [i * n for i in range(len(names) + 1)],
        "names": names, "sources": [{}] * len(names), "seconds": n * len(names) / sr,
        "built": "test"}))
    (d / "transcripts.json").write_text(json.dumps({k: "HELLO THERE" for k in names}))


def test_per_speaker_takes_every_voice_not_the_first_ones(tmp_path):
    names = [f"{s}-1-{u}.flac" for s in (11, 22, 33) for u in range(5)]
    write_corpus(tmp_path / "ls", names)
    got = pick_clips(tmp_path / "ls", per_speaker=2)
    assert [c["speaker"] for c in got] == ["11", "11", "22", "22", "33", "33"]
    assert got[0]["text"] == "hello there"


def test_last_takes_the_held_out_tail(tmp_path):
    names = [f"LJ001-{i:04d}" for i in range(10)]
    write_corpus(tmp_path / "lj", names)
    got = pick_clips(tmp_path / "lj", last=3)
    assert [c["name"] for c in got] == names[-3:]
    with pytest.raises(ValueError, match="Hz"):
        pick_clips(tmp_path / "lj", last=3, sample_rate=24_000)


def test_the_portal_reads_the_newest_report_without_its_items(tmp_path):
    from aksharallm.portal.audio import Audio

    assert Audio(tmp_path).transfer()["report"] is None
    (tmp_path / "logs/audio").mkdir(parents=True)
    side = {"n": 1, "items": [{"name": "x"}], "mcd": {"8": {"mean": 2.0}}}
    (tmp_path / "logs/audio/codec-transfer-codec-lj-step1.json").write_text(
        json.dumps({"checkpoint": "c.pt", "step": 1, "judge": "j", "home": side, "other": side}))
    r = Audio(tmp_path).transfer()["report"]
    assert r["step"] == 1 and r["other"]["mcd"]["8"]["mean"] == 2.0
    assert "items" not in r["home"]


def test_a_smaller_sample_never_overwrites_the_full_report():
    from aksharallm.audio.__main__ import transfer_suffix

    full = SimpleNamespace(home_clips=64, per_speaker=3, home="data/audio/lj",
                           other="data/asr/test-clean")
    assert transfer_suffix(full) == ""
    quick = SimpleNamespace(**{**vars(full), "home_clips": 6, "per_speaker": 1})
    assert transfer_suffix(quick) == "-lj6-test-cleanx1"
    other = SimpleNamespace(**{**vars(full), "other": "data/asr/dev-clean"})
    assert transfer_suffix(other) != ""


def test_the_portal_shows_full_reports_one_per_codec_and_never_a_quick_check(tmp_path):
    import os

    from aksharallm.portal.audio import Audio

    d = tmp_path / "logs/audio"
    d.mkdir(parents=True)

    def write(name, ckpt, n, t):
        side = {"n": n, "mcd": {"8": {"mean": float(n)}}}
        (d / name).write_text(json.dumps({"checkpoint": ckpt, "step": 1, "home": side,
                                          "other": side}))
        os.utime(d / name, (t, t))

    write("codec-transfer-codec-lj-step1.json", "checkpoints/codec-lj/ckpt_best.pt", 120, 1)
    write("codec-transfer-codec-many-step1.json", "checkpoints/codec-many/ckpt_best.pt", 121, 2)
    write("codec-transfer-codec-many-step1-lj6-test-cleanx1.json",
          "checkpoints/codec-many/ckpt_best.pt", 7, 3)               # newest, but a quick check
    res = Audio(tmp_path).transfer()
    assert res["report"]["other"]["n"] == 121
    assert [c["other"]["n"] for c in res["codecs"]] == [120, 121]
