"""G4b: the scenes corpus (size and place), the reader that reads them, and our own
contrastive picture-caption model.

What could be quietly wrong: the loss treating two pictures of the same fact as a mismatch
(plain CLIP's diagonal, wrong for this data, and the model trains anyway); padding leaking into
a caption's embedding (the same sentence would embed differently beside a longer one); a corpus
that contains its own held-out combinations; a reader that cannot tell big from small or left
from right (every place score would be noise); and the agreement statistic swapping its columns.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from aksharallm.pictures.clip import (ClipConfig, ClipModel, agreement, clip_loss, fact_key,
                                      rank, same_fact)
from aksharallm.pictures.reader import judge, read
from aksharallm.pictures.scenes import (HELD_OUT, all_facts, caption_of, held_out,
                                        holdout_corpus, render, scenes_corpus)


class StubText:
    """A frozen 'text model': one fixed random vector per word, zeros past the caption."""

    dim, max_len = 6, 8

    def encode(self, captions, with_mask=False):
        out = torch.zeros(len(captions), self.max_len, self.dim)
        mask = torch.zeros(len(captions), self.max_len, dtype=torch.bool)
        for i, c in enumerate(captions):
            for j, w in enumerate(c.split()[: self.max_len]):
                g = torch.Generator().manual_seed(sum(map(ord, w)))
                out[i, j] = torch.randn(self.dim, generator=g)
                mask[i, j] = True
        return (out, mask) if with_mask else out


def tiny_cc(**kw) -> ClipConfig:
    return ClipConfig(**({"image_size": 16, "patch": 4, "d_vision": 16, "n_layers": 1,
                          "n_heads": 2, "d_text": 16, "text_layers": 1, "d_embed": 8,
                          "text": "stub", "text_len": 8, "text_dim": 6} | kw))


# --------------------------------------------------------------------------------------
# the scenes corpus and the reader
# --------------------------------------------------------------------------------------


def test_the_reader_reads_every_scene_exactly_and_still_reads_shapes_as_before():
    rng = np.random.default_rng(0)
    for f in all_facts():
        for _ in range(2):
            v = judge(render(f, seed=int(rng.integers(1 << 30))), f)
            assert v["all"], (f, v["read"])
    from aksharallm.vision.image import render as render_shapes
    f = {"count": 3, "colour": "red", "shape": "circle"}
    v = judge(render_shapes(f, seed=1), f)
    assert v["all"] and set(v) == {"count", "colour", "shape", "all", "read"}


def test_the_reader_sees_a_wrong_size_and_a_wrong_place():
    f = {"count": 2, "size": "big", "colour": "blue", "shape": "square", "where": "left"}
    assert judge(render(f, seed=4), f)["all"]
    small = judge(render(dict(f, size="small"), seed=4), f)
    assert small["size"] is False and small["count"] and small["where"]
    right = judge(render(dict(f, where="right"), seed=4), f)
    assert right["where"] is False and right["size"] and right["count"]
    # two shapes in different columns are not "on the left"
    img = np.maximum(render(dict(f, count=1), seed=1), render(dict(f, count=1, where="right"), seed=2))
    assert read(img)["where"] == "spread" and read(img)["count"] == 2


def test_the_corpus_never_draws_a_held_out_combination(tmp_path):
    from aksharallm.vision.image import ImageManifest
    m = scenes_corpus(tmp_path / "s", n_images=400, progress=None)
    assert m.kind == "scenes" and not any(held_out(f) for f in m.facts)
    assert ImageManifest.load(tmp_path / "s" / "manifest.json").kind == "scenes"
    h = holdout_corpus(tmp_path / "h", per_combo=5)
    assert sorted(held_out(f) for f in h.facts) == sorted([*HELD_OUT] * 5)
    # both held-out combinations are made of attributes training does show, apart
    seen = m.facts
    assert any(f["shape"] == "circle" for f in seen) and any(f["where"] == "right" for f in seen)
    assert any(f["colour"] == "purple" for f in seen) and any(f["shape"] == "triangle" for f in seen)


def test_scene_captions_round_trip_and_shapes_phrasings_are_unchanged():
    from aksharallm.pictures.codes import fact_of
    from aksharallm.pictures.phrases import fact_from_text, phrase
    assert all(fact_of(caption_of(f)) == f for f in all_facts())
    f = {"count": 2, "size": "big", "colour": "red", "shape": "circle", "where": "left"}
    assert caption_of(f) == "two big red circles on the left"
    assert phrase(f, "in_colour") == "two big circles in red on the left"
    assert phrase(f, "of_them") == "big red circles, two of them, on the left"
    assert fact_from_text("could you draw two large red circles at the left") == f
    # a shapes fact: exactly the strings G4a trained on
    g = {"count": 1, "colour": "blue", "shape": "square"}
    assert [phrase(g, n) for n in ("canonical", "of_them", "there_are", "in_colour")] == [
        "one blue square", "a blue square, just one", "there is one blue square", "one square in blue"]


def test_a_scene_report_breaks_out_size_place_and_each_held_out_combination():
    from aksharallm.pictures.score import judge_drawings
    facts = [{"count": 1, "size": "big", "colour": "red", "shape": "square", "where": "middle"},
             {"count": 2, "size": "small", "colour": "purple", "shape": "triangle", "where": "left"},
             {"count": 1, "size": "big", "colour": "green", "shape": "circle", "where": "right"}]
    pics = np.stack([render(f, seed=3) for f in facts])
    pics[2] = render(dict(facts[2], where="left"), seed=3)          # drawn in the wrong place
    r = judge_drawings([caption_of(f) for f in facts], 1, pics, np.zeros((3, 4)), set(), facts)
    assert r["attributes"] == ["count", "size", "colour", "shape", "where"]
    assert r["seen"]["all"] == 1.0 and r["held_out"]["n"] == 2 and r["held_out"]["where"] == 0.5
    assert r["held_out_by"]["circles on the right"]["all"] == 0.0
    assert r["held_out_by"]["purple triangles"]["all"] == 1.0


# --------------------------------------------------------------------------------------
# the contrastive model
# --------------------------------------------------------------------------------------


def test_one_positive_per_row_is_clips_loss_exactly():
    torch.manual_seed(0)
    img, txt = F.normalize(torch.randn(6, 4), dim=-1), F.normalize(torch.randn(6, 4), dim=-1)
    s = torch.tensor(10.0)
    logits = s * img @ txt.t()
    want = (F.cross_entropy(logits, torch.arange(6)) + F.cross_entropy(logits.t(), torch.arange(6))) / 2
    assert torch.allclose(clip_loss(img, txt, s, torch.eye(6)), want)


def test_two_pictures_of_one_fact_are_not_pushed_apart():
    """Two arrangements of six pictures (two facts, three pictures each): A sends every picture
    and caption of a fact to one point; B gives every picture its own point. The loss used here
    prefers A, which is true of the data. Plain CLIP's diagonal prefers B: it learns to tell
    apart pictures that say the same thing, and trains happily while doing it."""
    facts = [{"count": n % 2 + 1, "colour": "red", "shape": "circle"} for n in range(6)]
    same = same_fact(facts)
    assert same.sum() == 18                                       # two facts, three each
    point = {1: torch.tensor([1.0, 0, 0, 0, 0, 0]), 2: torch.tensor([0, 1.0, 0, 0, 0, 0])}
    a = torch.stack([point[f["count"]] for f in facts])          # by fact
    b = torch.eye(6)                                              # every picture apart
    s = torch.tensor(10.0)
    assert clip_loss(a, a, s, same) < clip_loss(b, b, s, same)
    assert clip_loss(b, b, s, torch.eye(6)) < clip_loss(a, a, s, torch.eye(6))


def test_padding_never_changes_a_captions_embedding():
    torch.manual_seed(0)
    m = ClipModel(tiny_cc()).eval()
    m.attach_text(StubText())
    vecs, mask = StubText().encode(["two red circles", "a picture of one big blue square"],
                                   with_mask=True)
    a = m.embed_vectors(vecs, mask)
    junk = vecs.clone()
    junk[~mask] = torch.randn(int((~mask).sum()), 6) * 10      # whatever sits past the caption
    assert torch.allclose(m.embed_vectors(junk, mask), a, atol=1e-6)
    assert torch.allclose(a.norm(dim=-1), torch.ones(2))


def test_rank_finds_the_asked_caption_among_every_candidate():
    m = ClipModel(tiny_cc()).eval()
    m.attach_text(StubText())
    cands = [{"count": n, "colour": c, "shape": "circle"} for n in (1, 2) for c in ("red", "blue")]
    asked = [cands[2], cands[1]]
    from aksharallm.pictures.phrases import phrase
    txt = m.embed_texts([phrase(f, "canonical") for f in cands]).detach()

    # a stand-in picture tower: picture i embeds exactly as its asked caption, then as cand 0
    m.embed_images = lambda x: torch.stack([txt[2], txt[0]])[: len(x)]
    pics = np.zeros((2, 16, 16, 3), dtype=np.uint8)
    r = rank(m, pics, asked, cands)
    assert r["top1"] == [True, False] and r["best"] == [2, 0]
    assert r["margin"][0] > 0 > r["margin"][1]
    assert fact_key(cands[0]) != fact_key(cands[1])


def test_agreement_counts_each_cell_and_kappa_is_zero_for_a_coin():
    a = agreement([True, True, False, False], [True, False, True, False])
    assert (a["both"], a["model_only"], a["reader_only"], a["neither"]) == (1, 1, 1, 1)
    assert a["agree"] == 0.5 and abs(a["kappa"]) < 1e-12
    b = agreement([True, True, False], [True, True, False])
    assert b["agree"] == 1.0 and b["kappa"] == pytest.approx(1.0)
    c = agreement([True, False], [False, False])
    assert c["model_only"] == 1 and c["reader_only"] == 0


def test_the_trainer_trains_saves_resumes_and_scores_its_corpus(tmp_path, monkeypatch):
    import aksharallm.pictures.train_clip as tc
    from aksharallm.pictures.clip import load_clip
    from aksharallm.vision.image import synth_corpus
    synth_corpus(tmp_path / "c", n_images=80, size=16, progress=None)
    monkeypatch.setattr(tc, "text_encoder_for", lambda cc, device="cpu", root=None: StubText())
    cfg = tmp_path / "c.yaml"
    cfg.write_text(f"""
name: c
clip: {{image_size: 16, patch: 4, d_vision: 16, n_layers: 1, n_heads: 2, d_text: 16, d_embed: 8,
        text: stub, text_len: 8}}
data: {{corpora: [{tmp_path / 'c'}], holdouts: [{tmp_path / 'none'}]}}
optim: {{lr: 1.0e-3, warmup_steps: 1}}
train: {{out_dir: {tmp_path / 'run'}, batch_size: 8, max_steps: 3, eval_every: 2, eval_batches: 1,
         eval_pictures: 4, ckpt_every: 0, log_every: 1}}
""")
    assert tc.main([str(cfg)]) == 0
    import aksharallm.pictures.clip as cl
    monkeypatch.setattr(cl, "text_encoder_for", lambda cc, device="cpu", root=None: StubText())
    model, blob = load_clip(tmp_path / "run" / "ckpt_last.pt")
    assert blob["stage"] == "clip" and blob["step"] == 2 and blob["clip"]["text_dim"] == 6
    assert tc.main([str(cfg), "-o", "train.max_steps=5"]) == 0
    log = [json.loads(x) for x in (tmp_path / "run" / "train_log.jsonl").read_text().splitlines()]
    assert [r["start_step"] for r in log if r.get("event") == "session_start"] == [0, 3]
    evals = [r for r in log if "shapes_top1" in r]
    assert evals and all(0 <= r["shapes_top1"] <= 1 for r in evals)


def test_below_the_floor_no_caption_fits_and_the_closed_verdict_is_kept():
    m = ClipModel(tiny_cc()).eval()
    m.attach_text(StubText())
    cands = [{"count": n, "colour": "red", "shape": "circle"} for n in (1, 2, 3)]
    from aksharallm.pictures.phrases import phrase
    txt = m.embed_texts([phrase(f, "canonical") for f in cands]).detach()
    # picture 0 is exactly caption 0; picture 1 is half way between captions 1 and 2
    mid = F.normalize(txt[1] + txt[2], dim=-1)
    m.embed_images = lambda x: torch.stack([txt[0], mid])[: len(x)]
    pics = np.zeros((2, 16, 16, 3), dtype=np.uint8)
    cos_mid = float(mid @ txt[1])
    r = rank(m, pics, [cands[0], cands[1]], cands, floor=(cos_mid + 1.0) / 2)
    assert r["fits"] == [True, False]
    assert r["top1"][0] is True and r["top1"][1] is False
    assert r["closed_top1"][0] is True                      # closed set: picks the nearer one anyway
    assert rank(m, pics, [cands[0], cands[1]], cands)["top1"][0] is True     # no floor, no "fits"


def test_the_floor_is_a_quantile_of_clean_originals_and_belongs_to_one_checkpoint(tmp_path):
    from aksharallm.pictures.clip import NONE_FITS, calibrate, read_floor
    m = ClipModel(tiny_cc()).eval()
    m.attach_text(StubText())
    cands = [{"count": n, "colour": "red", "shape": "circle"} for n in (1, 2, 3)]
    from aksharallm.pictures.phrases import phrase
    txt = m.embed_texts([phrase(f, "canonical") for f in cands]).detach()
    vecs = torch.stack([F.normalize(txt[i % 3] + 0.1 * (i + 1) * txt[(i + 1) % 3], dim=-1)
                        for i in range(30)])
    m.embed_images = lambda x: vecs[: len(x)]
    facts = [cands[i % 3] for i in range(30)]
    import aksharallm.pictures.phrases as ph
    orig = ph.all_facts
    ph.all_facts = lambda kind="shapes": cands
    try:
        got = calibrate(m, [("shapes", np.zeros((30, 16, 16, 3), np.uint8), facts)], q=0.1)
    finally:
        ph.all_facts = orig
    own = [float(vecs[i] @ txt[i % 3]) for i in range(30)]
    assert got["shapes"] == pytest.approx(float(np.quantile(own, 0.1)))
    (tmp_path / NONE_FITS).write_text(json.dumps({"ckpt_best.pt": {"step": 7, "floors": {"shapes": 0.5}}}))
    assert read_floor(tmp_path / "ckpt_best.pt", 7) == {"shapes": 0.5}
    assert read_floor(tmp_path / "ckpt_best.pt", 8) is None        # another step's scale
    assert read_floor(tmp_path / "ckpt_last.pt", 7) is None        # another checkpoint
