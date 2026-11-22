"""G2, flow matching in pixels: patches, the loss, the sampler, guidance, the model, the
trainer and the shared drawing door.

What could be quietly wrong: patches cut in a different order from how they are put back (the
picture comes out scrambled and the loss still falls); the velocity target with its sign
flipped (it trains towards noise); the sampler stepping in the wrong direction or at the wrong
times (it would never land on a picture even with a perfect model); guidance with its halves
swapped; and a transformer that is still causal (patch 0 would never see the rest of the
picture). Each of these has an exact test, built on an oracle where one exists.
"""

from __future__ import annotations

import json
from dataclasses import asdict

import numpy as np
import pytest
import torch
import torch.nn as nn

from aksharallm.config import ModelConfig
from aksharallm.pictures.codes import NULL, word_ids
from aksharallm.pictures.flow import (FlowConfig, FlowModel, flow_loss, patchify, sample,
                                      unpatchify)
from aksharallm.vision.image import synth_corpus


def tiny_fc(**kw) -> FlowConfig:
    return FlowConfig(**({"image_size": 16, "patch": 4, "d_model": 32, "n_layers": 2,
                          "n_heads": 2, "sample_steps": 4} | kw))


def test_patches_are_cut_in_raster_order_and_put_back_exactly():
    x = torch.randn(2, 3, 16, 16)
    p = patchify(x, 4)
    assert p.shape == (2, 16, 48)
    assert torch.equal(p[0, 0].reshape(3, 4, 4), x[0, :, :4, :4])      # top-left first
    assert torch.equal(p[0, 1].reshape(3, 4, 4), x[0, :, :4, 4:8])     # then to its right
    assert torch.equal(p[0, 4].reshape(3, 4, 4), x[0, :, 4:8, :4])     # then the next row
    assert torch.equal(unpatchify(p, 4, 16), x)


class Oracle(nn.Module):
    """Knows the picture: on the straight path to x1 the velocity at (x, t) is
    (x1 - x) / (1 - t). A correct sampler integrates it onto x1 exactly."""

    def __init__(self, x1):
        super().__init__()
        self.fc, self.x1 = tiny_fc(), x1

    def forward(self, x, t, words):
        return (self.x1[: len(x)] - x) / (1 - t)[:, None, None, None]


def test_a_perfect_velocity_field_lands_exactly_on_the_picture():
    x1 = torch.rand(2, 3, 16, 16) * 1.6 - 0.8
    out = sample(Oracle(x1), ["one red circle", "two blue squares"], steps=5)
    assert torch.allclose(out, x1, atol=1e-5)


def test_the_loss_is_zero_for_the_true_velocity_and_not_for_its_negative():
    x1 = torch.rand(4, 3, 16, 16) * 2 - 1
    words = torch.tensor([word_ids("one red circle")] * 4)

    class Truth(nn.Module):
        def forward(self, x_t, t, w):              # x0 recovered from x_t; v = x1 - x0
            tt = t[:, None, None, None]
            return (x1 - x_t) / (1 - tt)

    class Backwards(Truth):
        def forward(self, x_t, t, w):
            return -super().forward(x_t, t, w)
    gen = lambda: torch.Generator().manual_seed(0)
    assert float(flow_loss(Truth(), x1, words, gen())) < 1e-8
    assert float(flow_loss(Backwards(), x1, words, gen())) > 1.0


class Opinionated(nn.Module):
    """+0.1 everywhere with a caption, -0.1 with NULL: guidance is visible in the sign."""

    def __init__(self):
        super().__init__()
        self.fc = tiny_fc()

    def forward(self, x, t, words):
        s = torch.where(words[:, 0] == NULL, -0.1, 0.1)
        return s[:, None, None, None].expand_as(x)


def test_guidance_mixes_the_captioned_and_the_unconditional_velocities():
    x0 = torch.randn((1, 3, 16, 16), generator=torch.Generator().manual_seed(7))
    for w, v in ((1.0, 0.1), (0.0, -0.1), (3.0, -0.1 + 3 * 0.2)):
        out = sample(Opinionated(), ["one red circle"], steps=1, guidance=w, seed=7)
        assert torch.allclose(out, (x0 + v).clamp(-1, 1), atol=1e-6), w


def test_every_patch_sees_the_whole_picture_and_the_caption():
    torch.manual_seed(0)
    m = FlowModel(tiny_fc()).eval()
    nn.init.normal_(m.out.weight, std=0.5)            # zero-init would hide everything
    x = torch.randn(1, 3, 16, 16)
    t = torch.tensor([0.5])
    w = torch.tensor([word_ids("one red circle")])
    base = m(x, t, w)
    y = x.clone()
    y[..., 12:, 12:] += 1.0                            # change only the LAST patch
    assert not torch.allclose(m(y, t, w)[..., :4, :4], base[..., :4, :4])   # patch 0 moved
    other = m(x, t, torch.tensor([word_ids("four green squares")]))
    assert not torch.allclose(other, base)             # and the caption matters


def test_a_bidirectional_body_needs_an_objective_said_out_loud():
    with pytest.raises(ValueError, match="embeds_only"):
        ModelConfig(vocab_size=4, causal=False)
    ModelConfig(vocab_size=1, causal=False, embeds_only=True)


def test_the_trainer_trains_saves_resumes_and_its_val_loss_is_repeatable(tmp_path):
    from aksharallm.pictures.flow import load_flow
    from aksharallm.pictures.train_flow import evaluate, main
    from aksharallm.vision.image import ImageCaptions
    synth_corpus(tmp_path / "c", n_images=40, size=16, progress=None)
    cfg = tmp_path / "f.yaml"
    cfg.write_text(f"""
name: f
flow: {{image_size: 16, patch: 4, d_model: 32, n_layers: 1, n_heads: 2, sample_steps: 2}}
data: {{corpus: {tmp_path / 'c'}, codes: {tmp_path / 'none'}}}
optim: {{lr: 1.0e-3, warmup_steps: 1}}
train: {{out_dir: {tmp_path / 'run'}, batch_size: 4, max_steps: 3, eval_every: 2, eval_batches: 1,
         sample_every: 0, sample_k: 0, ckpt_every: 0, log_every: 1}}
""")
    assert main([str(cfg)]) == 0
    model, fc, blob = load_flow(tmp_path / "run" / "ckpt_last.pt")
    assert blob["step"] == 2 and blob["stage"] == "flow" and fc.image_size == 16
    assert main([str(cfg), "-o", "train.max_steps=5"]) == 0
    log = [json.loads(x) for x in (tmp_path / "run" / "train_log.jsonl").read_text().splitlines()]
    assert [r["start_step"] for r in log if r.get("event") == "session_start"] == [0, 3]
    val = ImageCaptions(tmp_path / "c", split="val", val_frac=0.25)
    from contextlib import nullcontext
    a = evaluate(model, val, "cpu", 4, 2, nullcontext)
    assert a == evaluate(model, val, "cpu", 4, 2, nullcontext)   # same noise, same number


def test_the_drawing_door_and_the_portal_take_a_flow_model(tmp_path):
    from aksharallm.pictures.drawer import load_drawer
    from aksharallm.pictures.tokenizer import PictureConfig, PictureTokenizer
    from aksharallm.portal.pictures import Pictures
    pc = PictureConfig(channels=[8, 16], latent_dim=8, n_codebooks=2, codebook_size=16)
    (tmp_path / "checkpoints/tok").mkdir(parents=True)
    torch.save({"model": PictureTokenizer(pc).state_dict(), "picture": asdict(pc),
                "stage": "picture", "step": 1}, tmp_path / "checkpoints/tok/ckpt_best.pt")
    fc = FlowConfig(d_model=32, n_layers=1, n_heads=2, sample_steps=2,
                    tokenizer="checkpoints/tok/ckpt_best.pt", books=2)
    (tmp_path / "checkpoints/flow").mkdir(parents=True)
    torch.save({"model": FlowModel(fc).state_dict(), "flow": asdict(fc), "stage": "flow",
                "step": 9}, tmp_path / "checkpoints/flow/ckpt_best.pt")
    d = load_drawer(tmp_path / "checkpoints/flow/ckpt_best.pt", tmp_path)
    assert d.kind == "G2"
    r = d.report(k=1, captions=["one red circle", "two purple triangles"])
    assert r["model"] == "G2" and r["seen"]["n"] == 1 and r["held_out"]["n"] == 1
    p = Pictures(tmp_path)
    assert [g["kind"] for g in p.overview()["generators"]] == ["G2"]
    out = p.draw({"checkpoint": "checkpoints/flow/ckpt_best.pt", "caption": "one red circle",
                  "n": 2, "guidance": 3})
    assert out["kind"] == "G2" and len(out["pictures"]) == 2
    assert np.asarray(r["_pictures"]).shape == (2, 64, 64, 3)
