"""G1, a sentence to a picture: the code vocabulary, the dataset, the sampler, the scorer and
the trainer.

What could be quietly wrong: codes flattened in one order and unflattened in another (the
decoder would paint a scrambled grid and the loss would still fall); the caption's own words
counted in the loss (the val loss would no longer be per code); a sampler that emits a book-1
id into a book-0 slot; classifier-free guidance with its two halves swapped or the
unconditional half ignored (every guidance setting would then draw the same thing); and a
paired comparison of reports that drew different captions.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch
import torch.nn as nn

from aksharallm.config import ModelConfig
from aksharallm.pictures.codes import (CAPTION_LEN, CODE0, IMG, NULL, WORDS, CodesDataset,
                                       all_captions, fact_of, flatten, unflatten, word_ids)
from aksharallm.pictures.gen import PicGenConfig, build, sample, slot_range
from aksharallm.pictures.score import compare, is_held_out
from aksharallm.vision.image import synth_corpus

K = 16          # a small codebook for the tests


def test_codes_flatten_position_major_and_come_back():
    codes = torch.randint(0, K, (3, 2, 4, 4))
    t = flatten(codes.reshape(3, 2, 16), 2, K)
    # cell 0 book 0, then cell 0 book 1, then cell 1 book 0
    assert t[0, 0] == CODE0 + codes[0, 0, 0, 0]
    assert t[0, 1] == CODE0 + K + codes[0, 1, 0, 0]
    assert t[0, 2] == CODE0 + codes[0, 0, 0, 1]
    assert torch.equal(unflatten(t, 2, K, 4), codes)
    # numpy and torch agree
    assert np.array_equal(flatten(codes.reshape(3, 2, 16).numpy(), 2, K), t.numpy())


def test_the_vocabulary_has_both_words_of_the_held_out_combination():
    assert "purple" in WORDS and "triangles" in WORDS and "triangle" in WORDS
    caps = all_captions()
    assert len(caps) == 60 and len(set(caps)) == 60
    assert sum(is_held_out(c) for c in caps) == 4
    assert fact_of("two red circles") == {"count": 2, "colour": "red", "shape": "circle"}
    with pytest.raises(ValueError, match="knows only"):
        word_ids("two crimson circles")
    with pytest.raises(ValueError, match="three|3 words"):
        word_ids("red circles")


def _codes_dir(tmp_path, n=40, books=2, grid=4):
    synth_corpus(tmp_path / "c", n_images=n, progress=None)
    d = tmp_path / "codes"
    d.mkdir()
    rng = np.random.default_rng(0)
    np.save(d / "codes.npy", rng.integers(0, K, (n, books, grid * grid)).astype(np.int16))
    (d / "meta.json").write_text(json.dumps({
        "tokenizer": str(tmp_path / "tok.pt"), "tokenizer_step": 1, "books": books,
        "grid": grid, "codebook_size": K, "corpus": str(tmp_path / "c"), "n": n}))
    return d


def test_only_the_codes_carry_loss_and_the_caption_can_be_dropped(tmp_path):
    ds = CodesDataset(_codes_dir(tmp_path), books=2, seed=0)
    x, y = ds.batch(8)
    assert x.shape[1] == y.shape[1] == ds.seq_len - 1 == CAPTION_LEN + 2 * 16
    assert (y[:, :CAPTION_LEN] == -100).all()                  # words and the IMG marker
    assert (x[:, CAPTION_LEN] == IMG).all()
    assert (y[:, CAPTION_LEN:] >= CODE0).all()                 # first target = first code
    x, _ = ds.batch(64, drop=1.0)
    assert (x[:, :CAPTION_LEN] == NULL).all()
    x, _ = ds.batch(64, drop=0.0)
    assert (x[:, :CAPTION_LEN] != NULL).all()


def test_the_sampler_only_ever_writes_a_legal_id_for_each_slot():
    torch.manual_seed(0)
    pg = PicGenConfig(books=2, d_model=32, n_layers=1, n_heads=2, dropout=0.0)
    model = build(pg, K, 4)
    codes = sample(model, ["two red circles", "one blue square"], books=2, codebook_size=K,
                   grid=4, guidance=2.0, seed=1)
    assert codes.shape == (2, 2, 4, 4)
    assert int(codes.min()) >= 0 and int(codes.max()) < K
    assert slot_range(0, 2, K) == (CODE0, CODE0 + K)
    assert slot_range(1, 2, K) == (CODE0 + K, CODE0 + 2 * K)
    # seeded: the same call gives the same pictures (what makes two settings pairable)
    again = sample(model, ["two red circles", "one blue square"], books=2, codebook_size=K,
                   grid=4, guidance=2.0, seed=1)
    assert torch.equal(codes, again)


class Opinionated(nn.Module):
    """Prefers code 5 when it was given a caption and code 7 when it was given NULL, so the
    guidance arithmetic is visible in which code comes out."""

    def __init__(self):
        super().__init__()
        self.cfg = ModelConfig(vocab_size=CODE0 + 2 * K, d_model=8, n_layers=1, n_heads=1,
                               max_seq_len=CAPTION_LEN + 1 + 2 * 16)
        self.w = nn.Parameter(torch.zeros(1))
        self.null = None

    def forward(self, idx, caches=None):
        if idx.shape[1] > 1:                       # the prompt: remember which rows are NULL
            self.null = idx[:, 0] == NULL
        lg = torch.full((idx.shape[0], 1, self.cfg.vocab_size), -10.0)
        for b in range(2):
            lg[~self.null, 0, CODE0 + b * K + 5] = 0.0
            lg[self.null, 0, CODE0 + b * K + 7] = 0.0
        return lg, None


def test_guidance_mixes_the_captioned_and_the_unconditional_halves():
    def draw(w):
        return sample(Opinionated(), ["two red circles"], books=2, codebook_size=K, grid=4,
                      guidance=w, temperature=1e-4, seed=0)
    assert (draw(1.0) == 5).all()            # no guidance: the captioned model alone
    assert (draw(3.0) == 5).all()            # pushed further towards the caption
    assert (draw(0.0) == 7).all()            # w = 0 is the unconditional model


def test_two_reports_pair_only_when_they_drew_the_same_captions():
    a = {"captions": ["one red circle"], "k": 2,
         "seen": {"n": 2, "all": 0.5, "correct": "10"}, "held_out": {"n": 0}}
    b = {"captions": ["one red circle"], "k": 2,
         "seen": {"n": 2, "all": 0.5, "correct": "01"}, "held_out": {"n": 0}}
    r = compare(a, b)["seen"]
    assert (r["b_only"], r["a_only"]) == (1, 1)
    with pytest.raises(ValueError, match="different captions"):
        compare(a, b | {"captions": ["two red circles"]})


def test_the_trainer_trains_saves_and_resumes(tmp_path):
    from aksharallm.pictures.gen import load_generator
    from aksharallm.pictures.train_gen import main
    d = _codes_dir(tmp_path)
    cfg = tmp_path / "g.yaml"
    cfg.write_text(f"""
name: g
picgen: {{tokenizer: {tmp_path / 'tok.pt'}, books: 2, d_model: 32, n_layers: 1, n_heads: 2}}
data: {{codes: {d}}}
optim: {{lr: 1.0e-3, warmup_steps: 1}}
train: {{out_dir: {tmp_path / 'run'}, batch_size: 4, max_steps: 3, eval_every: 2, eval_batches: 1,
         sample_every: 0, sample_k: 0, ckpt_every: 0, log_every: 1}}
""")
    assert main([str(cfg)]) == 0
    model, pg, blob = load_generator(tmp_path / "run" / "ckpt_last.pt")
    assert blob["step"] == 2 and blob["stage"] == "picgen" and pg.books == 2
    assert main([str(cfg), "-o", "train.max_steps=5"]) == 0
    log = [json.loads(x) for x in (tmp_path / "run" / "train_log.jsonl").read_text().splitlines()]
    assert [r["start_step"] for r in log if r.get("event") == "session_start"] == [0, 3]
    # step 0's loss starts near "knows nothing", over the whole vocabulary
    first = next(r["loss"] for r in log if r.get("step") == 0 and "loss" in r)
    assert abs(first - np.log(model.cfg.vocab_size)) < 1.0


def test_codes_from_another_tokenizer_are_refused(tmp_path):
    from aksharallm.pictures.train_gen import main
    d = _codes_dir(tmp_path)
    cfg = tmp_path / "g.yaml"
    cfg.write_text(f"""
name: g
picgen: {{tokenizer: {tmp_path / 'other.pt'}, books: 2, d_model: 32, n_layers: 1, n_heads: 2}}
data: {{codes: {d}}}
train: {{out_dir: {tmp_path / 'run'}, max_steps: 1, sample_k: 0}}
""")
    with pytest.raises(SystemExit, match="same\\s+tokenizer"):
        main([str(cfg)])


# ---- the portal ------------------------------------------------------------------------

def _gen_report(root, name, correct="10"):
    d = root / "logs/pictures"
    d.mkdir(parents=True, exist_ok=True)
    (d / name).write_text(json.dumps({
        "captions": ["one red circle"], "k": 2, "guidance": 3.0, "variety": 1.0,
        "seen": {"n": 2, "all": 0.5, "count": 1, "colour": 1, "shape": 0.5, "copies": 0.0,
                 "correct": correct},
        "held_out": {"n": 0}, "per_caption": [], "gallery": [], "run": "g", "step": 1}))
    return name


def test_the_portal_lists_and_pairs_g1_reports_but_not_against_a_tokenizer_report(tmp_path):
    from aksharallm.portal.pictures import Pictures, PicturesError
    p = Pictures(tmp_path)
    a = _gen_report(tmp_path, "gen-g-step1-g1.json", "10")
    b = _gen_report(tmp_path, "gen-g-step1-g3.json", "01")
    ov = p.overview()
    assert {r["name"] for r in ov["gen_reports"]} == {a, b} and ov["reports"] == []
    out = p.compare(a, b)
    assert out["rows"][0]["b_only"] == 1 and "gen-compare" in out["command"]
    (tmp_path / "logs/pictures/report-x-step1.json").write_text(json.dumps(
        {"n": 2, "ladder": [{"books": 4, "all": 1, "correct": "11"}], "books": [4]}))
    with pytest.raises(PicturesError, match="not one of each"):
        p.compare(a, "report-x-step1.json")


def test_the_portal_draws_only_from_a_g1_checkpoint_and_known_words(tmp_path):
    from aksharallm.portal.jobs import JobError
    from aksharallm.portal.pictures import Pictures, PicturesError
    p = Pictures(tmp_path)
    with pytest.raises(PicturesError, match="not a G1"):
        p.draw({"checkpoint": "checkpoints/tiny/ckpt_best.pt", "caption": "two red circles"})
    with pytest.raises(JobError):
        p.start({"kind": "gen-report", "checkpoint": "checkpoints/tiny/ckpt_best.pt"})
    with pytest.raises(JobError):
        p.start({"kind": "codes", "checkpoint": "nope.pt"})


def test_a_drawing_goes_caption_to_codes_to_a_judged_picture(tmp_path):
    """The whole inline path on tiny untrained models: G1 writes 2 of the tokenizer's 3
    codebooks, the third is zero-filled and ignored by the decoder, and each picture comes back
    as a PNG with the reader's verdict."""
    from dataclasses import asdict

    from aksharallm.pictures.tokenizer import PictureConfig, PictureTokenizer
    from aksharallm.portal.pictures import Pictures
    pc = PictureConfig(channels=[8, 16], latent_dim=8, n_codebooks=3, codebook_size=K)
    (tmp_path / "checkpoints/tok").mkdir(parents=True)
    torch.save({"model": PictureTokenizer(pc).state_dict(), "picture": asdict(pc),
                "stage": "picture", "step": 1}, tmp_path / "checkpoints/tok/ckpt_best.pt")
    pg = PicGenConfig(tokenizer="checkpoints/tok/ckpt_best.pt", books=2, d_model=32,
                      n_layers=1, n_heads=2, dropout=0.0)
    model = build(pg, K, pc.grid)
    (tmp_path / "checkpoints/gen").mkdir(parents=True)
    torch.save({"model": model.state_dict(), "model_config": asdict(model.cfg),
                "picgen": asdict(pg), "stage": "picgen", "step": 7},
               tmp_path / "checkpoints/gen/ckpt_best.pt")
    p = Pictures(tmp_path)
    assert [g["rel"] for g in p.overview()["generators"]] == ["checkpoints/gen/ckpt_best.pt"]
    r = p.draw({"checkpoint": "checkpoints/gen/ckpt_best.pt", "caption": "Three  PURPLE triangles",
                "n": 3, "guidance": 2})
    assert r["caption"] == "three purple triangles" and r["held_out"] and r["step"] == 7
    assert len(r["pictures"]) == 3 and r["pictures"][0]["png"].startswith("data:image/png;base64,")
    assert {"count", "colour", "shape"} <= set(r["pictures"][0]["read"])
    assert "--guidance 2" in r["command"]
