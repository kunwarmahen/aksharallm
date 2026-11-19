"""`audio/speaker.py`: a voice-identity score, and the numbers that give it a meaning.

What could quietly lie: an EER computed backwards (higher score read as "different"), labels
that do not follow the speaker (a classifier of clips, not people), a held-out split taken
inside a speaker's clips the wrong way, and a similarity column with no reference beside it.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch
import yaml

from aksharallm.audio.speaker import SpeakerCorpus, SpeakerModelConfig, eer
from aksharallm.audio.transfer import voice_columns


def test_eer_is_zero_when_separated_one_when_reversed_and_half_at_random():
    same = np.array([1, 1, 1, 0, 0, 0])
    assert eer(np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1]), same) == 0.0
    assert eer(np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9]), same) == 1.0
    rng = np.random.default_rng(0)
    big = rng.integers(0, 2, 4000)
    assert eer(rng.standard_normal(4000), big) == pytest.approx(0.5, abs=0.03)
    with pytest.raises(ValueError):
        eer(np.array([0.1, 0.2]), np.array([1, 1]))


def corpus(d, speakers, utts=3, seconds=2.5, sr=16_000):
    n = int(seconds * sr)
    names = [f"{s}-1-{u}" for s in speakers for u in range(utts)]
    d.mkdir(parents=True)
    # Each speaker a different tone, so a model can tell them apart in a few steps.
    t = np.arange(n) / sr
    audio = np.concatenate([(3000 * np.sin(2 * np.pi * (120 + 40 * k) * t)).astype("<i2")
                            for k, s in enumerate(speakers) for _ in range(utts)])
    audio.tofile(d / "audio.bin")
    (d / "manifest.json").write_text(json.dumps({
        "sample_rate": sr, "offsets": [i * n for i in range(len(names) + 1)], "names": names,
        "sources": [{}] * len(names), "seconds": n * len(names) / sr, "built": "test"}))
    (d / "transcripts.json").write_text(json.dumps({k: "A WORD" for k in names}))
    return str(d)


def test_labels_follow_the_speaker_and_the_held_out_clip_is_each_speakers_last(tmp_path):
    c = corpus(tmp_path / "ls", ["11", "22", "33"])
    train = SpeakerCorpus([c], crop=16_000, val_clips=1)
    val = SpeakerCorpus([c], crop=16_000, val_clips=1, split="val")
    assert train.speakers == ["ls/11", "ls/22", "ls/33"]
    assert sorted({lab for lab, _, _ in train.items}) == [0, 1, 2]
    assert [i for _, _, i in val.items] == [2, 5, 8]          # utt 2 of every speaker
    assert not {i for _, _, i in train.items} & {i for _, _, i in val.items}
    x, y = train.batch(8)
    assert x.shape == (8, 16_000) and y.max() < 3


def test_a_speaker_model_must_not_see_loudness():
    from aksharallm.asr.model import AsrModelConfig

    with pytest.raises(ValueError, match="utterance"):
        SpeakerModelConfig(encoder=AsrModelConfig(normalise="global"))


def test_the_voice_column_comes_with_its_two_references():
    items = [{"speaker": s} for s in ("a", "a", "b", "b")]
    e = torch.nn.functional.normalize(torch.tensor(
        [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]]), dim=-1)
    recon = torch.nn.functional.normalize(e + torch.tensor([0.0, 0.3]), dim=-1)
    out = voice_columns(items, {"original": e, "codec8": recon})
    ref = out["voice_ref"]
    assert ref["pairs"] == [2, 4]
    assert ref["same_speaker"] > 0.95 and ref["different_speaker"] < 0.3
    assert out["voice"]["codec8"] == pytest.approx(
        float((e * recon).sum(-1).mean()))
    assert items[0]["voice"]["codec8"] == pytest.approx(float(e[0] @ recon[0]))


def tiny_config(tmp_path, train, eval_, out, **kw):
    cfg = {"name": "spk",
           "speaker": {"encoder": {"d_model": 16, "n_layers": 1, "n_heads": 2,
                                   "conv_kernel": 3, "subsample_channels": 8,
                                   "max_frames": 256, "normalise": "utterance"},
                       "emb_dim": 8},
           "data": {"train": [train], "eval": eval_, "eval_per_speaker": 2,
                    "crop_seconds": 1.0},
           "optim": {"lr": 3e-3, "warmup_steps": 1},
           "train": {"out_dir": str(out), "batch_size": 8, "max_steps": 40, "eval_every": 39,
                     "eval_batches": 1, "ckpt_every": 0, "log_every": 1, **kw}}
    p = tmp_path / "spk.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return str(p)


def test_training_learns_the_speakers_and_the_checkpoint_round_trips(tmp_path):
    from aksharallm.audio import train_speaker
    from aksharallm.audio.speaker import load_speaker

    train = corpus(tmp_path / "train", ["1", "2", "3", "4"], utts=4)
    unseen = corpus(tmp_path / "dev", ["7", "8"], utts=2)
    out = tmp_path / "run"
    torch.manual_seed(0)
    assert train_speaker.main([tiny_config(tmp_path, train, unseen, out)]) == 0
    steps = [json.loads(line) for line in open(out / "train_log.jsonl") if '"acc"' in line]
    assert np.mean([s["loss"] for s in steps[-5:]]) < np.mean([s["loss"] for s in steps[:5]])
    evals = [json.loads(line) for line in open(out / "train_log.jsonl") if '"val_eer"' in line]
    assert evals and 0.0 <= evals[-1]["val_eer"] <= 1.0
    m = load_speaker(out / "ckpt_last.pt")
    e = m(torch.zeros(2, 16_000) + 0.01)
    assert e.shape == (2, 8) and torch.allclose(e.norm(dim=-1), torch.ones(2))
    with pytest.raises(ValueError, match="not a speaker model"):
        torch.save({"stage": "codec"}, tmp_path / "c.pt")
        load_speaker(tmp_path / "c.pt")

    # A different speaker list is a different classifier; resuming onto it is refused.
    more = corpus(tmp_path / "train2", ["1", "2", "3", "5"], utts=4)
    with pytest.raises(SystemExit, match="speaker list"):
        train_speaker.main([tiny_config(tmp_path, more, unseen, out)])
