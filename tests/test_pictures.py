"""The picture line, G0: the rule-based reader, the tokenizer, its measurements, the trainer
and the Pictures tab.

What could be quietly wrong: a reader that is not exact on the originals (then every score is
the reader's error as much as the tokenizer's); a reader too lenient to see damage (it would
call a blur a square); a codebook prefix that secretly depends on the later codebooks (then
"1 codebook" in the ladder is not what it says); a picture whose reconstruction depends on its
batch-mates; a paired comparison with its columns swapped; and a gallery route that can be
pointed at a file outside logs/pictures/.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from aksharallm.pictures.measure import compare, ladder, psnr
from aksharallm.pictures.reader import BACKGROUND, caption, judge, read
from aksharallm.pictures.tokenizer import PictureConfig, PictureTokenizer
from aksharallm.vision.image import COLOURS, SHAPES, render, synth_corpus


def tiny_cfg(**kw) -> PictureConfig:
    base = dict(channels=[8, 16], latent_dim=8, n_codebooks=3, codebook_size=16)
    return PictureConfig(**(base | kw))


# ---- the reader -----------------------------------------------------------------------

def test_the_reader_is_exact_on_every_kind_of_picture_the_corpus_draws():
    """The positive control: every count x colour x shape, several layouts each."""
    wrong = []
    for count in range(1, 5):
        for colour in COLOURS:
            for shape in SHAPES:
                for seed in range(4):
                    fact = {"count": count, "colour": colour, "shape": shape}
                    if not judge(render(fact, seed=1000 * count + seed), fact)["all"]:
                        wrong.append((fact, seed))
    assert not wrong, wrong[:5]


def test_the_reader_sees_the_damage_a_tokenizer_does():
    fact = {"count": 2, "colour": "green", "shape": "square"}
    img = render(fact, seed=3)
    # a blur turns squares into soft blobs: a light one (sigma 1) must still read as square,
    # a heavy one (sigma 3, what the 300-step smoke decoder painted) must not
    def blur(sigma):
        y, x = np.mgrid[-6:7, -6:7]
        ker = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
        k = np.zeros((64, 64))
        k[:13, :13] = ker / ker.sum()
        k = np.roll(k, (-6, -6), (0, 1))
        return np.stack([np.clip(np.real(np.fft.ifft2(np.fft.fft2(img[..., c].astype(float))
                                                      * np.fft.fft2(k))), 0, 255)
                         for c in range(3)], -1).astype(np.uint8)
    assert read(img)["shape"] == read(blur(1))["shape"] == "square"
    assert read(blur(3))["shape"] != "square" and read(blur(3))["count"] == 2
    # a colour drifted past half way reads as the other colour
    recoloured = img.copy()
    recoloured[(img != BACKGROUND).any(-1)] = COLOURS["blue"]
    assert judge(recoloured, fact)["colour"] is False
    # an empty picture says nothing
    blank = np.full_like(img, BACKGROUND)
    assert read(blank) == {"count": 0, "colour": None, "shape": None}
    assert judge(blank, fact)["all"] is False


def test_the_caption_matches_the_corpus_wording():
    assert caption({"count": 3, "colour": "red", "shape": "circle"}) == "three red circles"
    assert caption({"count": 1, "colour": "blue", "shape": "square"}) == "one blue square"
    assert caption({"count": 0, "colour": None, "shape": None}) == "nothing"


# ---- the tokenizer ---------------------------------------------------------------------

def test_codes_have_the_grid_and_decode_back_to_a_picture():
    m = PictureTokenizer(tiny_cfg()).eval()
    x = torch.rand(2, 3, 64, 64) * 2 - 1
    idx = m.encode(x)
    assert idx.shape == (2, 3, 16, 16) and m.cfg.grid == 16 and m.cfg.tokens == 3 * 256
    assert m.decode(idx).shape == x.shape
    with pytest.raises(ValueError):
        PictureTokenizer(tiny_cfg(image_size=62))


def test_a_codebook_prefix_ignores_the_codebooks_after_it():
    """`decode(idx, 1)` is "the 1-codebook picture" only if codebooks 2.. cannot reach it."""
    m = PictureTokenizer(tiny_cfg()).eval()
    m(torch.rand(4, 3, 64, 64) * 2 - 1)            # seeds the codebooks from real outputs
    idx = m.encode(torch.rand(1, 3, 64, 64) * 2 - 1)
    other = idx.clone()
    other[:, 1:] = (other[:, 1:] + 5) % m.cfg.codebook_size
    assert torch.equal(m.decode(idx, 1), m.decode(other, 1))
    assert not torch.equal(m.decode(idx, 3), m.decode(other, 3))


def test_a_picture_encodes_the_same_whatever_its_batch_mates():
    """In TRAIN mode, where batch statistics would leak (in eval BatchNorm uses running stats
    and would pass this too, so eval cannot tell the two apart). The quantizer is left out:
    its EMA moves the codebook on every training call by design."""
    m = PictureTokenizer(tiny_cfg()).train()
    x = torch.rand(1, 3, 64, 64) * 2 - 1
    alone = m.encoder(x)
    crowd = m.encoder(torch.cat([x, torch.rand(3, 3, 64, 64) * 2 - 1]))[:1]
    assert torch.allclose(alone, crowd, atol=1e-5)
    m.eval()
    assert torch.allclose(m.decode(m.encode(x)),
                          m.decode(m.encode(torch.cat([x, x * 0.5])))[:1], atol=1e-5)


# ---- measurements ----------------------------------------------------------------------

def test_psnr_is_capped_for_identical_pictures_and_falls_with_noise():
    a = np.full((8, 8, 3), 100, dtype=np.uint8)
    assert psnr(a, a) == 99.0
    assert psnr(a, a + 1) > psnr(a, a + 10)


def test_the_ladder_judges_every_count_of_codebooks(tmp_path):
    from aksharallm.vision.image import ImageCaptions
    synth_corpus(tmp_path / "c", n_images=40, progress=None)
    ds = ImageCaptions(tmp_path / "c", split="val", val_frac=0.5)
    r = ladder(PictureTokenizer(tiny_cfg()), ds, n=8)
    assert r["books"] == [1, 2, 3] and [row["books"] for row in r["ladder"]] == [1, 2, 3]
    assert r["originals"]["all"] == 1.0                     # the reader's control row
    assert all(len(row["correct"]) == 8 for row in r["ladder"])
    assert len(r["usage"]) == 3


def test_two_reports_pair_picture_by_picture():
    def rep(marks):
        return {"n": len(marks), "books": [4],
                "ladder": [{"books": 4, "all": marks.count("1") / len(marks), "correct": marks}]}
    a, b = rep("0011"), rep("1110")
    r = compare(a, b)
    assert (r["b_only"], r["a_only"]) == (2, 1)            # b right where a was wrong: #0, #1
    with pytest.raises(ValueError):
        compare(a, rep("011"))


# ---- the trainer -----------------------------------------------------------------------

def test_the_trainer_trains_saves_and_resumes(tmp_path):
    from aksharallm.pictures.train import load_tokenizer, main
    synth_corpus(tmp_path / "c", n_images=40, progress=None)
    cfg = tmp_path / "t.yaml"
    cfg.write_text(f"""
name: t
picture: {{channels: [8, 16], latent_dim: 8, n_codebooks: 2, codebook_size: 16}}
data: {{corpus: {tmp_path / 'c'}, holdout: {tmp_path / 'none'}}}
optim: {{lr: 1.0e-3, warmup_steps: 1}}
train: {{out_dir: {tmp_path / 'run'}, batch_size: 4, max_steps: 3, eval_every: 2, eval_batches: 1,
         sample_every: 0, sample_images: 0, ckpt_every: 0, log_every: 1}}
""")
    assert main([str(cfg)]) == 0
    model, blob = load_tokenizer(tmp_path / "run" / "ckpt_last.pt")
    assert blob["step"] == 2 and blob["stage"] == "picture" and model.cfg.n_codebooks == 2
    assert main([str(cfg), "-o", "train.max_steps=5"]) == 0
    log = [json.loads(line) for line in (tmp_path / "run" / "train_log.jsonl").read_text().splitlines()]
    starts = [r["start_step"] for r in log if r.get("event") == "session_start"]
    assert starts == [0, 3]                                 # the second session resumed
    assert all("books_used" in r for r in log if "recon" in r)


# ---- the portal ------------------------------------------------------------------------

def _report(root, name="report-x-step1.json", gallery=("report-x-step1/pic00.png",)):
    d = root / "logs/pictures"
    (d / "report-x-step1").mkdir(parents=True, exist_ok=True)
    (d / "report-x-step1" / "pic00.png").write_bytes(b"\x89PNG fake")
    (root / "secret.png").write_bytes(b"\x89PNG secret")
    row = {"books": 4, "psnr": 30.0, "count": 1, "colour": 1, "shape": 1, "all": 0.5,
           "correct": "10", "bits": 2304}
    (d / name).write_text(json.dumps({"n": 2, "books": [4], "ladder": [row], "usage": [],
                                      "originals": {"all": 1.0}, "grid": 8, "n_codebooks": 4,
                                      "gallery": list(gallery), "run": "x", "step": 1}))
    return name


def test_the_portal_lists_reports_and_inlines_only_gallery_files_inside_logs(tmp_path):
    from aksharallm.portal.pictures import Pictures, PicturesError
    p = Pictures(tmp_path)
    name = _report(tmp_path, gallery=("report-x-step1/pic00.png", "../../secret.png"))
    assert [r["name"] for r in p.overview()["reports"]] == [name]
    rep = p.report(name)
    assert len(rep["gallery_png"]) == 1                     # the escaping path is dropped
    assert rep["gallery_png"][0].startswith("data:image/png;base64,")
    with pytest.raises(PicturesError):
        p.report("../secret.json")
    out = p.compare(name, name)
    assert out["rows"][0]["b_only"] == 0 and "pictures compare" in out["command"]


def test_the_portal_refuses_a_report_job_for_something_that_is_not_a_tokenizer(tmp_path):
    from aksharallm.portal.jobs import JobError
    from aksharallm.portal.pictures import Pictures
    with pytest.raises(JobError):
        Pictures(tmp_path).start({"kind": "report", "checkpoint": "checkpoints/tiny/ckpt_best.pt"})
    with pytest.raises(JobError):
        Pictures(tmp_path).start({"kind": "train"})
