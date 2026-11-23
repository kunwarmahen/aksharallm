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


# ---- G3: the same flow, in the tokenizer's latent ---------------------------------------

def _tiny_tokenizer(root):
    from aksharallm.pictures.tokenizer import PictureConfig, PictureTokenizer
    torch.manual_seed(0)
    pc = PictureConfig(image_size=16, channels=[8, 16], latent_dim=8, n_codebooks=2,
                       codebook_size=16)
    tok = PictureTokenizer(pc).eval()
    (root / "checkpoints/tok").mkdir(parents=True, exist_ok=True)
    torch.save({"model": tok.state_dict(), "picture": asdict(pc), "stage": "picture", "step": 1},
               root / "checkpoints/tok/ckpt_best.pt")
    return tok


def test_latents_are_normalised_on_the_training_split_and_decode_back_exactly(tmp_path):
    from aksharallm.pictures.latents import (LatentDataset, decode_latent, denormalise,
                                             encode_latents, picture_latent, stats_tensors)
    from aksharallm.vision.image import ImageCaptions, to_tensor
    tok = _tiny_tokenizer(tmp_path)
    synth_corpus(tmp_path / "c", n_images=40, size=16, progress=None)
    meta = encode_latents(tmp_path / "checkpoints/tok/ckpt_best.pt", tmp_path / "c",
                          tmp_path / "lat", progress=None)
    assert meta["shape"] == [8, 4, 4]
    ds = LatentDataset(tmp_path / "lat", split="train")
    z, _ = ds.rows(ds.index)
    assert abs(z.mean()) < 0.05 and abs(z.std() - 1) < 0.1      # unit scale, like the noise
    pics = ImageCaptions(tmp_path / "c", split="val", val_frac=1.0)
    x = torch.from_numpy(to_tensor(np.asarray(pics.images[:4])))
    q = picture_latent(tok, x)
    m, s = stats_tensors(meta["mean"], meta["std"])
    back = denormalise(torch.from_numpy(np.load(tmp_path / "lat/latents.npy")[:4]).float(), m, s)
    assert torch.allclose(back, q, atol=2e-2)                   # float16 storage, nothing else
    assert torch.allclose(decode_latent(tok, q), tok.decoder(q))


def test_a_latent_drawing_is_mapped_back_before_the_decoder_paints_it(tmp_path):
    """An oracle that flows to a known normalised latent: the picture must be the decoder's
    picture of the DENORMALISED latent. Without the mapping back, the decoder paints a latent
    at the wrong scale (on the real tokenizer the reader then drops from 100% to 62%)."""
    from aksharallm.pictures.flow import draw
    from aksharallm.pictures.latents import denormalise, stats_tensors
    from aksharallm.vision.image import from_tensor
    tok = _tiny_tokenizer(tmp_path)
    mean, std = [0.3] * 8, [2.0] * 8
    z = torch.randn(1, 8, 4, 4)

    class LatentOracle(Oracle):
        def __init__(self):
            super().__init__(z)
            self.fc = FlowConfig(space="latent", channels=8, image_size=4, patch=1,
                                 latent_mean=mean, latent_std=std, sample_steps=3)
    got = draw(LatentOracle(), ["one red circle"], tokenizer=tok)
    m, s = stats_tensors(mean, std)
    want = from_tensor(tok.decoder(denormalise(z, m, s)).clamp(-1, 1).detach().numpy())
    assert np.abs(got.astype(int) - want.astype(int)).max() <= 1
    with pytest.raises(ValueError, match="tokenizer"):
        draw(LatentOracle(), ["one red circle"])


def test_g3_trains_from_latents_carries_their_statistics_and_draws_through_the_door(tmp_path):
    from aksharallm.pictures.drawer import load_drawer
    from aksharallm.pictures.flow import load_flow
    from aksharallm.pictures.latents import encode_latents
    from aksharallm.pictures.train_flow import main
    _tiny_tokenizer(tmp_path)
    synth_corpus(tmp_path / "c", n_images=40, size=16, progress=None)
    tok = tmp_path / "checkpoints/tok/ckpt_best.pt"
    meta = encode_latents(tok, tmp_path / "c", tmp_path / "lat", progress=None)
    cfg = tmp_path / "g3.yaml"
    body = f"""
name: g3
flow: {{space: latent, patch: 1, d_model: 32, n_layers: 1, n_heads: 2, sample_steps: 2,
        tokenizer: {tok}, books: 2}}
data: {{corpus: {tmp_path / 'c'}, latents: {tmp_path / 'lat'}, codes: {tmp_path / 'none'}}}
optim: {{lr: 1.0e-3, warmup_steps: 1}}
train: {{out_dir: {tmp_path / 'checkpoints/g3'}, batch_size: 4, max_steps: 2, eval_every: 1,
         eval_batches: 1, sample_every: 0, sample_k: 0, ckpt_every: 0, log_every: 1}}
"""
    cfg.write_text(body)
    assert main([str(cfg)]) == 0
    _, fc, _ = load_flow(tmp_path / "checkpoints/g3/ckpt_last.pt")
    assert (fc.channels, fc.image_size) == (8, 4) and fc.latent_mean == meta["mean"]
    d = load_drawer(tmp_path / "checkpoints/g3/ckpt_last.pt", tmp_path)
    assert d.kind == "G3"
    assert d.draw(["two red circles"]).shape == (1, 16, 16, 3)
    cfg.write_text(body.replace(f"tokenizer: {tok}", f"tokenizer: {tmp_path / 'other.pt'}")
                   .replace("checkpoints/g3", "checkpoints/g3b"))
    with pytest.raises(SystemExit, match="same tokenizer"):
        main([str(cfg)])


# ---- G3b: a continuous autoencoder's latent ---------------------------------------------

def _tiny_ae(root):
    from aksharallm.pictures.autoencoder import AEConfig, PictureAutoencoder
    torch.manual_seed(0)
    ac = AEConfig(image_size=16, channels=[8, 16], latent_dim=4)
    ae = PictureAutoencoder(ac).eval()
    (root / "checkpoints/ae").mkdir(parents=True, exist_ok=True)
    torch.save({"model": ae.state_dict(), "autoencoder": asdict(ac), "stage": "autoencoder",
                "step": 1}, root / "checkpoints/ae/ckpt_best.pt")
    return ae


def test_the_autoencoder_draws_from_its_mean_and_its_kl_is_zero_at_the_prior(tmp_path):
    ae = _tiny_ae(tmp_path)
    x = torch.rand(2, 3, 16, 16) * 2 - 1
    mean, logvar = ae.moments(x)
    assert mean.shape == (2, 4, 4, 4) and torch.equal(ae.latent(x), mean)
    r1, _ = ae(x)
    r2, _ = ae(x)
    assert torch.equal(r1, r2)                      # eval: the mean, no sampling
    assert torch.equal(r1, ae.decoder(mean))

    class Prior(type(ae)):
        def moments(self, x):
            return torch.zeros(2, 4, 4, 4), torch.zeros(2, 4, 4, 4)
    p = Prior(ae.cfg)
    assert float(p(x)[1]) == 0.0                    # KL(N(0,1) || N(0,1)) = 0


def test_one_loader_opens_either_coder_and_latents_use_the_autoencoders_mean(tmp_path):
    from aksharallm.pictures.latents import decode_latent, picture_latent
    from aksharallm.pictures.train import load_coder
    ae = _tiny_ae(tmp_path)
    _tiny_tokenizer(tmp_path)
    a, _ = load_coder(tmp_path / "checkpoints/ae/ckpt_best.pt")
    t, _ = load_coder(tmp_path / "checkpoints/tok/ckpt_best.pt")
    assert hasattr(a, "latent") and not hasattr(t, "latent")
    x = torch.rand(1, 3, 16, 16) * 2 - 1
    assert torch.equal(picture_latent(a, x), ae.latent(x))
    z = picture_latent(a, x)
    assert torch.equal(decode_latent(a, z, snap=True), a.decoder(z))   # nothing to snap to


def test_g3b_scores_copies_with_the_vq_tokenizer_not_the_autoencoder(tmp_path):
    from aksharallm.pictures.drawer import load_drawer
    from aksharallm.pictures.latents import encode_latents
    from aksharallm.pictures.train_flow import main
    _tiny_ae(tmp_path)
    _tiny_tokenizer(tmp_path)
    synth_corpus(tmp_path / "c", n_images=40, size=16, progress=None)
    ae_path = tmp_path / "checkpoints/ae/ckpt_best.pt"
    encode_latents(ae_path, tmp_path / "c", tmp_path / "lat", progress=None)
    cfg = tmp_path / "g3b.yaml"
    cfg.write_text(f"""
name: g3b
flow: {{space: latent, patch: 1, d_model: 32, n_layers: 1, n_heads: 2, sample_steps: 2,
        tokenizer: {ae_path}, score_tokenizer: {tmp_path / 'checkpoints/tok/ckpt_best.pt'},
        books: 2}}
data: {{corpus: {tmp_path / 'c'}, latents: {tmp_path / 'lat'}, codes: {tmp_path / 'none'}}}
optim: {{lr: 1.0e-3, warmup_steps: 1}}
train: {{out_dir: {tmp_path / 'checkpoints/g3b'}, batch_size: 4, max_steps: 2, eval_every: 0,
         sample_every: 0, sample_k: 0, ckpt_every: 0, log_every: 1}}
""")
    assert main([str(cfg)]) == 0
    d = load_drawer(tmp_path / "checkpoints/g3b/ckpt_last.pt", tmp_path)
    assert d.kind == "G3" and hasattr(d.tokenizer, "latent") and not hasattr(d.scorer, "latent")
    r = d.report(k=1, captions=["one red circle"])
    assert r["seen"]["n"] == 1 and len(r["per_caption"]) == 1
