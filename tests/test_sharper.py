"""Sharper photos: the sharpness measure, the patch discriminator's losses, the adaptive weight,
and the autoencoder trainer with an adversary beside its pixel loss.

What could be quietly wrong: a sharpness measure that reads the wrong band (blur would score
as sharp); hinge losses with a sign flipped (the decoder would learn to look MORE fake, and the
loss curves would still move); an adaptive weight inverted (the adversary would be loudest
exactly when it is least useful); a run with the adversary off that no longer trains as every
earlier one did; and a best checkpoint chosen from before the adversary started, which would
hand the blurry autoencoder to the flow model while the run's name says it is the sharp one.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch


def _photos(n=6, size=32, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(n, size, size, 3), dtype=np.uint8)


def _blur(x):
    y = x.astype(np.float64)
    for ax in (1, 2):
        y = (np.roll(y, 1, ax) + 2 * y + np.roll(y, -1, ax)) / 4
    return y.round().astype(np.uint8)


def test_sharpness_is_one_for_the_same_photos_below_one_for_a_blur_and_ordered():
    from aksharallm.pictures.measure import sharpness
    x = _photos()
    assert sharpness(x, x) == pytest.approx(1.0)
    b = _blur(x)
    assert sharpness(b, x) < 0.3                       # a [1 2 1] blur kills the top band
    assert sharpness(x, b) == pytest.approx(1 / sharpness(b, x))   # the reference divides


def test_the_hinge_losses_point_the_right_way():
    from aksharallm.pictures.adversarial import d_loss, g_loss
    sure = d_loss(torch.full((4,), 2.0), torch.full((4,), -2.0))
    fooled = d_loss(torch.full((4,), -2.0), torch.full((4,), 2.0))
    assert float(sure) == 0.0 and float(fooled) == pytest.approx(3.0)
    assert float(g_loss(torch.full((4,), 2.0))) < float(g_loss(torch.full((4,), -2.0)))


def test_a_decoder_step_on_the_adversarial_loss_makes_its_patches_read_as_more_real():
    from aksharallm.pictures.adversarial import PatchDiscriminator, g_loss
    torch.manual_seed(0)
    disc = PatchDiscriminator([8, 16]).eval()
    # a textured picture: on a flat one GroupNorm has no variance to work with, and a step in
    # either direction nudges the score up (this check once passed with the sign flipped)
    fake = (torch.rand(2, 3, 32, 32) * 2 - 1).requires_grad_()
    before = float(disc(fake).mean().detach())
    g_loss(disc(fake)).backward()
    with torch.no_grad():
        fake -= 0.5 * fake.grad / fake.grad.norm()
    assert float(disc(fake).mean().detach()) > before


def test_the_adaptive_weight_matches_the_two_gradients_at_the_last_layer():
    from aksharallm.pictures.adversarial import adaptive_weight
    w = torch.randn(5, requires_grad=True)
    rec = (w * torch.arange(5.0)).sum()          # grad = [0..4]
    adv = (3 * w).sum()                          # grad = [3]*5
    lam = adaptive_weight(rec, adv, w)
    assert float(lam) == pytest.approx(torch.arange(5.0).norm() / (3 * 5 ** 0.5), rel=1e-3)
    # an adversary ten times as loud gets a tenth of the weight: what reaches the decoder is
    # the same
    lam10 = adaptive_weight(rec, 10 * adv, w)
    assert float(lam10) == pytest.approx(float(lam) / 10, rel=1e-3)


def _run(tmp_path, name, **adv):
    from aksharallm.pictures.train_ae import main
    from aksharallm.vision.image import synth_corpus
    corpus = tmp_path / "corpus"
    if not (corpus / "manifest.json").is_file():
        synth_corpus(corpus, n_images=48, size=16)
    lines = [f"name: {name}", "autoencoder:", "  image_size: 16", "  channels: [8, 8, 8]",
             "  latent_dim: 2", "data:", f"  corpus: {corpus}", f"  holdout: {tmp_path}/none",
             "optim:", "  lr: 1.0e-3", "  warmup_steps: 2", "train:",
             f"  out_dir: {tmp_path / name}", "  batch_size: 4", "  max_steps: 12",
             "  eval_every: 2", "  eval_batches: 1", "  sample_every: 0", "  sample_images: 0",
             "  ckpt_every: 4", "  log_every: 1", "  seed: 3"]
    stop_at = adv.pop("stop_at", None)
    if stop_at is not None:
        lines.append(f"  stop_at: {stop_at}")
    if adv:
        lines += ["adversarial:"] + [f"  {k}: {v}" for k, v in adv.items()] + ["  channels: [8, 8]"]
    (tmp_path / f"{name}.yaml").write_text("\n".join(lines) + "\n")
    assert main([str(tmp_path / f"{name}.yaml")]) == 0
    return tmp_path / name


def test_with_the_adversary_off_nothing_about_a_run_changes(tmp_path):
    out = _run(tmp_path, "plain")
    last = torch.load(out / "ckpt_last.pt", weights_only=False)
    assert "disc" not in last
    steps = [json.loads(x) for x in (out / "train_log.jsonl").read_text().splitlines()]
    assert not any("g_loss" in r for r in steps)


def test_the_adversary_trains_resumes_and_best_is_chosen_after_it_starts(tmp_path):
    out = _run(tmp_path, "gan", weight=0.5, start_step=6)
    last = torch.load(out / "ckpt_last.pt", weights_only=False)
    assert "disc" in last and "disc_optimizer" in last
    best = torch.load(out / "ckpt_best.pt", weights_only=False)
    assert best["step"] >= 6 and "disc" not in best      # best is the autoencoder alone
    recs = [json.loads(x) for x in (out / "train_log.jsonl").read_text().splitlines()]
    trained = [r for r in recs if "g_loss" in r]
    assert trained and min(r["step"] for r in trained) == 6


def test_a_resume_carries_the_discriminator_on_and_refuses_to_restart_it(tmp_path):
    out = _run(tmp_path, "gan", weight=0.5, start_step=2, stop_at=7)
    last = torch.load(out / "ckpt_last.pt", weights_only=False)
    assert last["step"] == 7
    # mark the saved discriminator: after one more step from it, it must still be near the mark
    last["disc"] = {k: torch.zeros_like(v) for k, v in last["disc"].items()}
    torch.save(last, out / "ckpt_last.pt")
    _run(tmp_path, "gan", weight=0.5, start_step=2, stop_at=8)
    after = torch.load(out / "ckpt_last.pt", weights_only=False)
    assert after["step"] == 8
    assert after["disc"]["net.0.weight"].abs().max() < 0.01     # a fresh one is ~0.1
    # and a checkpoint past start_step with no discriminator is refused, not silently restarted
    del after["disc"], after["disc_optimizer"]
    torch.save(after, out / "ckpt_last.pt")
    with pytest.raises(SystemExit, match="no discriminator"):
        _run(tmp_path, "gan", weight=0.5, start_step=2, stop_at=9)


def test_a_reconstruction_report_pairs_two_autoencoders_photo_by_photo():
    from aksharallm.pictures.photos import recon_compare
    a = {"captions": ["x", "y", "z"], "psnr": 20.0, "sharpness": 0.5,
         "per_photo": {"psnr": [20, 21, 22], "sharpness": [0.5, 0.5, 2.0]}}
    b = {"captions": ["x", "y", "z"], "psnr": 21.0, "sharpness": 0.9,
         "per_photo": {"psnr": [21, 20, 23], "sharpness": [0.9, 0.4, 1.1]}}
    c = recon_compare(a, b)
    assert (c["psnr"]["b_better"], c["psnr"]["a_better"]) == (2, 1)
    # closer to 1 on a log scale: 0.9 beats 0.5, 0.4 loses to 0.5, 1.1 beats 2.0
    assert (c["sharpness"]["b_better"], c["sharpness"]["a_better"]) == (2, 1)
    with pytest.raises(ValueError, match="different photos"):
        recon_compare(a, b | {"captions": ["z", "y", "x"]})


def test_best_is_the_latest_checkpoint_the_adversary_trained_not_the_lowest_pixel_loss(
        tmp_path, monkeypatch):
    # The pixel loss rises once the decoder trades averages for texture; make that happen on
    # every eval. "Lowest val" would keep step 2 (before the adversary); "lowest val from the
    # start on" would keep step 6, its first step, which is what the first real run did.
    import aksharallm.pictures.train_ae as tae
    calls = iter(range(100))
    monkeypatch.setattr(tae, "evaluate", lambda *a, **k: 1.0 + next(calls))
    out = _run(tmp_path, "gan", weight=0.5, start_step=6)
    assert torch.load(out / "ckpt_best.pt", weights_only=False)["step"] == 11   # the last step


def test_the_sharp_pair_is_each_one_change_from_its_control():
    """The comparison is only readable if each arm changes exactly what it says."""
    from aksharallm.config import config_to_dict
    from aksharallm.pictures.train_ae import load_ae_config
    from aksharallm.pictures.train_flow import load_flow_config

    def flat(load, name):
        d = config_to_dict(load(f"configs/{name}.yaml"))
        d.pop("name")
        d["train"].pop("out_dir")
        return {f"{k}.{kk}": v for k, sub in d.items() for kk, v in sub.items()}
    a, b = flat(load_ae_config, "ae-coco"), flat(load_ae_config, "ae-coco-gan")
    assert {k for k in a if a[k] != b[k]} == {"adversarial.weight"}   # off -> on; the rest is ae-coco
    f, g = flat(load_flow_config, "flow-coco"), flat(load_flow_config, "flow-coco-gan")
    assert {k for k in f if f[k] != g[k]} == {"flow.tokenizer", "data.latents"}


def test_the_portal_lists_reconstruction_reports_and_only_runs_them_on_photo_coders(tmp_path):
    from aksharallm.portal.jobs import JobError
    from aksharallm.portal.pictures import Pictures
    (tmp_path / "logs/pictures").mkdir(parents=True)
    rec = {"n": 2, "run": "ae-x", "step": 5, "psnr": 25.0, "sharpness": 0.5,
           "drawn": {"top5": 0.2, "sharpness": 0.5}, "real": {"top5": 0.3, "sharpness": 1.0}}
    (tmp_path / "logs/pictures/recon-ae-x-step5-n2.json").write_text(json.dumps(rec))
    p = Pictures(tmp_path)
    r = p.coco_reports()
    assert [(x["name"], x["kind"], x["sharpness"]) for x in r] == [("recon-ae-x-step5-n2.json", "recon", 0.5)]
    assert p.report("recon-ae-x-step5-n2.json")["psnr"] == 25.0
    with pytest.raises(JobError, match="not a photo autoencoder"):
        p.start({"kind": "ae-report", "checkpoint": "checkpoints/ae-shapes/ckpt_best.pt",
                     "clip": "x"})
