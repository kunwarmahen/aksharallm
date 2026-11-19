"""The many-voices codec: weighted corpora, per-corpus validation, and a warm start.

Three ways it could quietly be a different experiment from the one the config describes:
the mix not being the weights (LJ drifts to its 5% share by hours, or LibriSpeech never
appears); one blended val number hiding that the old voice got worse; and a warm start that
silently loads into a different architecture, or loses to a resume the wrong way round.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch
import yaml

from aksharallm.audio.dataset import MixedAudioDataset


def corpus(d, value, n_clips=12, seconds=1.5, sr=16_000):
    """A packed corpus whose every sample is `value`, so a batch row says where it came from."""
    n = int(seconds * sr)
    d.mkdir(parents=True)
    (np.full(n * n_clips, value, dtype="<i2")).tofile(d / "audio.bin")
    names = [f"{d.name}-{i}" for i in range(n_clips)]
    (d / "manifest.json").write_text(json.dumps({
        "sample_rate": sr, "offsets": [i * n for i in range(n_clips + 1)], "names": names,
        "sources": [{}] * n_clips, "seconds": n * n_clips / sr, "built": "test"}))
    (d / "transcripts.json").write_text(json.dumps({k: "A" for k in names}))
    return str(d)


def test_every_batch_holds_each_corpus_at_its_weight(tmp_path):
    a, b, c = (corpus(tmp_path / n, v) for n, v in (("a", 100), ("b", 200), ("c", 300)))
    ds = MixedAudioDataset([{"corpus": a, "weight": 0.2}, {"corpus": b, "weight": 0.17},
                            {"corpus": c, "weight": 0.63}], window=8000, seed=0)
    seen = np.zeros(3)
    for _ in range(400):
        x = ds.batch(16)
        assert x.shape == (16, 8000)
        ids = np.round(x[:, 0].numpy() * 32768 / 100).astype(int) - 1
        k = np.bincount(ids, minlength=3)
        # floor(16 * w) rows of each, every batch: 3, 2 and 10, plus the leftovers.
        assert (k >= [3, 2, 10]).all() and k.sum() == 16
        seen += k
    assert seen / seen.sum() == pytest.approx([0.2, 0.17, 0.63], abs=0.01)


def test_sources_at_different_rates_are_refused(tmp_path):
    a = corpus(tmp_path / "a", 1)
    b = corpus(tmp_path / "b", 1, sr=24_000)
    with pytest.raises(ValueError, match="sample rate"):
        MixedAudioDataset([{"corpus": a}, {"corpus": b}], window=8000)


def tiny_config(tmp_path, sources, out, **train):
    cfg = {
        "name": "t", "codec": {"channels": 4, "strides": [2, 4, 5, 8], "dim": 8,
                               "n_codebooks": 2, "codebook_size": 16, "n_residual": 1},
        "data": {"window_seconds": 0.2, "val_clips": 2, "sources": sources},
        "optim": {"lr": 1e-3, "warmup_steps": 1},
        "loss": {"scales": [512, 256]},
        "train": {"out_dir": str(out), "batch_size": 4, "max_steps": 6, "eval_every": 3,
                  "eval_batches": 1, "sample_every": 0, "ckpt_every": 0, "log_every": 0,
                  **train},
    }
    p = tmp_path / f"{out.name}.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return str(p)


def val_records(out):
    return [json.loads(line) for line in open(out / "train_log.jsonl")
            if '"val_loss"' in line and '"step"' in line]


def test_validation_is_reported_per_corpus_and_weighted(tmp_path):
    from aksharallm.audio import train_codec

    rng = np.random.default_rng(0)
    srcs = []
    for name, w in (("lj", 0.25), ("ls", 0.75)):
        d = tmp_path / name
        corpus(d, 0)
        (rng.integers(-3000, 3000, size=24_000 * 12) .astype("<i2")).tofile(d / "audio.bin")
        srcs.append({"corpus": str(d), "weight": w})
    out = tmp_path / "run"
    assert train_codec.main([tiny_config(tmp_path, srcs, out)]) == 0
    rec = val_records(out)[-1]
    assert rec["val_loss"] == pytest.approx(0.25 * rec["val_loss_lj"] + 0.75 * rec["val_loss_ls"])


def test_a_warm_start_loads_the_weights_and_refuses_another_shape(tmp_path):
    from aksharallm.audio import train_codec

    src = [{"corpus": corpus(tmp_path / "lj", 0), "weight": 1.0}]
    np.random.default_rng(1).integers(-3000, 3000, size=24_000 * 12).astype("<i2").tofile(
        tmp_path / "lj" / "audio.bin")
    first = tmp_path / "first"
    assert train_codec.main([tiny_config(tmp_path, src, first, ckpt_every=5)]) == 0
    init = str(first / "ckpt_last.pt")
    w0 = torch.load(init, weights_only=False)["model"]

    # Zero steps from the init: the weights it starts from are the init's, exactly.
    second = tmp_path / "second"
    cfg = tiny_config(tmp_path, src, second, init=init, max_steps=1, eval_every=0)
    assert train_codec.main([cfg, "-o", "optim.lr=0.0"]) == 0
    w1 = torch.load(second / "ckpt_last.pt", weights_only=False)["model"]
    for k, v in w0.items():
        if k.startswith("quantizer."):
            # The codebooks are an EMA, not optimizer parameters: lr 0 does not freeze them,
            # and one step moves them ~1%. Close to the init's, far from a fresh codec's.
            if v.is_floating_point():
                assert torch.allclose(v, w1[k], rtol=0.2, atol=0.05), k
        else:
            assert torch.equal(v, w1[k]), k
    assert torch.load(second / "ckpt_last.pt", weights_only=False)["step"] == 0

    # A different codebook size is a different codec, not a warm start.
    third = tmp_path / "third"
    bad = tiny_config(tmp_path, src, third, init=init, max_steps=1)
    with pytest.raises(SystemExit, match="codebook_size"):
        train_codec.main([bad, "-o", "codec.codebook_size=32"])


def test_the_many_voices_config_matches_codec_lj_in_shape():
    lj = yaml.safe_load(open("configs/codec-lj.yaml"))
    many = yaml.safe_load(open("configs/codec-many.yaml"))
    assert many["codec"] == lj["codec"]
    assert many["train"]["init"] == "checkpoints/codec-lj/ckpt_best.pt"
    assert sum(s["weight"] for s in many["data"]["sources"]) == pytest.approx(1.0)
    assert many["data"]["sources"][0]["corpus"] == lj["data"]["corpus"]


# ---------------------------------------------------------------------------------------
# the content codebook (audio/content.py)
# ---------------------------------------------------------------------------------------

def tiny_recognizer(path):
    from dataclasses import asdict

    from aksharallm.asr import vocab
    from aksharallm.asr.model import AsrModelConfig, Recognizer

    cfg = AsrModelConfig(d_model=16, n_layers=1, n_heads=2, subsample_channels=8, max_frames=256)
    torch.manual_seed(0)
    m = Recognizer(cfg)
    torch.save({"stage": "asr", "alphabet": vocab.ALPHABET, "asr": asdict(cfg),
                "model": m.state_dict()}, path)
    return str(path)


def test_the_first_output_is_codebook_zeros_own_vector():
    from aksharallm.audio.vq import ResidualVQ

    torch.manual_seed(0)
    rvq = ResidualVQ(8, 3, 16, dropout=False).eval()
    z = torch.randn(2, 8, 10)
    q, idx, _, _, q0 = rvq(z, first=True)
    assert torch.allclose(q0, rvq.layers[0].decode(idx[:, 0]), atol=1e-6)
    assert not torch.allclose(q0, q)                     # the sum, not just the first


def test_the_content_loss_is_zero_on_the_teachers_direction_and_one_when_unrelated():
    from aksharallm.audio.content import ContentHead, content_loss

    torch.manual_seed(0)
    head = ContentHead(8, 8)
    with torch.no_grad():
        head.proj.weight.copy_(torch.eye(8)), head.proj.bias.zero_()
    q0 = torch.randn(4, 8, 50)
    target = 3.0 * q0.transpose(1, 2)                   # same direction, other norm
    assert float(content_loss(head, q0, target)) == pytest.approx(0.0, abs=1e-5)
    assert float(content_loss(head, q0, -target)) == pytest.approx(2.0, abs=1e-5)
    unrelated = torch.randn(4, 50, 8)
    assert float(content_loss(head, q0, unrelated)) == pytest.approx(1.0, abs=0.1)


def test_teacher_states_come_at_the_codecs_frame_rate(tmp_path):
    from aksharallm.audio.content import load_teacher, teacher_states

    rec = load_teacher(tiny_recognizer(tmp_path / "asr.pt"), "cpu")
    assert not any(p.requires_grad for p in rec.parameters())
    h = teacher_states(rec, 0.1 * torch.randn(3, 16_000), frames=50)
    assert h.shape == (3, 50, 16)


def test_a_content_run_trains_its_head_and_leaves_a_plain_codec(tmp_path):
    from aksharallm.audio import train_codec
    from aksharallm.audio.codec import load_codec

    src = [{"corpus": corpus(tmp_path / "lj", 0), "weight": 1.0}]
    np.random.default_rng(2).integers(-3000, 3000, size=24_000 * 12).astype("<i2").tofile(
        tmp_path / "lj" / "audio.bin")
    teacher = tiny_recognizer(tmp_path / "asr.pt")
    out = tmp_path / "content"
    cfg = tiny_config(tmp_path, src, out, max_steps=30, eval_every=0, log_every=1)
    over = ["-o", "loss.content_weight=1.0", "-o", f"loss.content_teacher={teacher}",
            "-o", "optim.lr=0.01"]
    assert train_codec.main([cfg, *over]) == 0
    steps = [json.loads(line) for line in open(out / "train_log.jsonl") if '"content"' in line]
    assert len(steps) == 30
    assert steps[-1]["content"] < steps[0]["content"] - 0.1
    blob = torch.load(out / "ckpt_last.pt", weights_only=False)
    assert "content_head" in blob and not any("proj" in k for k in blob["model"])
    load_codec(str(out / "ckpt_last.pt"), "cpu")          # still an ordinary codec

    # Switching the loss on or off across a resume is refused, not silently half-applied.
    with pytest.raises(SystemExit, match="with the content loss"):
        train_codec.main([cfg, "-o", "train.max_steps=31"])
