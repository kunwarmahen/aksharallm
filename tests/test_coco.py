"""G4c: COCO photos packed from the archives, captions as matches, the batched text encoder,
the photo autoencoder's report, the flow trainer reading a photo's own captions, and the photo
report's numbers.

What could be quietly wrong: photos packed out of order with their captions (every caption
would describe another photo, and every loss would still fall); a padded batch of captions
encoding differently from each caption alone (training and drawing would read different
vectors); the five captions of one photo counted as mismatches; the Fréchet distance wrong by
a matrix square root; and a ceiling computed on the wrong pairs, which would make any drawing
look close to the real thing.
"""

from __future__ import annotations

import io
import json
import zipfile

import numpy as np
import pytest
import torch
import torch.nn as nn

from aksharallm.config import ModelConfig


def _fake_coco(raw, n=6):
    """Three zips shaped like COCO's: each photo a flat colour, its captions naming it."""
    from PIL import Image
    raw.mkdir(parents=True, exist_ok=True)
    colours = [(200, 30, 30), (30, 200, 30), (30, 30, 200), (200, 200, 30), (30, 200, 200), (90, 90, 90)]
    images, anns = [], []
    with zipfile.ZipFile(raw / "val2017.zip", "w") as z:
        for k in range(n):
            i = 100 - k                                  # ids out of file order on purpose
            name = f"{i:012d}.jpg"
            buf = io.BytesIO()
            Image.new("RGB", (96 + 10 * k, 64), colours[k]).save(buf, format="PNG")
            z.writestr(f"val2017/{name}", buf.getvalue())
            images.append({"id": i, "file_name": name})
            anns += [{"image_id": i, "caption": f"photo {i} caption {c}"} for c in range(5)]
    with zipfile.ZipFile(raw / "annotations_trainval2017.zip", "w") as z:
        z.writestr("annotations/captions_val2017.json", json.dumps({"images": images, "annotations": anns}))
    return colours


def test_photos_are_packed_in_order_with_their_own_captions(tmp_path):
    from aksharallm.pictures.coco import pack
    from aksharallm.vision.image import ImageCaptions
    colours = _fake_coco(tmp_path / "raw")
    m = pack(tmp_path / "raw", tmp_path / "c", "val", workers=2, progress=None)
    assert m.kind == "coco" and [f["id"] for f in m.facts] == sorted(f["id"] for f in m.facts)
    ds = ImageCaptions(tmp_path / "c", split="val", val_frac=1.0)
    for j, f in enumerate(m.facts):
        k = 100 - f["id"]                               # which colour photo id was drawn with
        assert np.abs(ds.images[j].astype(int) - colours[k]).max() <= 3
        assert f["captions"] == [f"photo {f['id']} caption {c}" for c in range(5)]
        assert m.captions[j] == f["captions"][0]


def test_a_photos_five_captions_are_one_match_and_other_photos_are_not():
    from aksharallm.pictures.clip import fact_key, same_fact, text_of
    a, b = {"id": 1, "captions": ["x", "y"]}, {"id": 2, "captions": ["z"]}
    assert fact_key(a) != fact_key(b) and fact_key(a) == fact_key(dict(a))
    assert same_fact([a, a, b]).tolist() == [[1, 1, 0], [1, 1, 0], [0, 0, 1]]
    assert text_of(a, "show_me", 1) == "y" and text_of(a, i=2) == "x"
    # a drawn picture's facts are not a photo, whatever their values
    assert fact_key({"count": 1, "colour": "red", "shape": "circle"})[0] != "photo"


def test_a_padded_batch_encodes_each_caption_exactly_as_alone(monkeypatch):
    import aksharallm.infer.cli as cli
    import aksharallm.tokenizer.tokenizer as tk
    from aksharallm.model.transformer import Transformer
    from aksharallm.pictures.text import TextEncoder
    torch.manual_seed(0)
    lm = Transformer(ModelConfig(vocab_size=40, d_model=16, n_heads=2, n_layers=2, max_seq_len=16)).eval()

    class Tok:
        def __init__(self, path):
            pass

        def encode(self, s):
            return [1 + (ord(ch) % 37) for ch in s.replace(" ", "")][:9]
    monkeypatch.setattr(cli, "load_model", lambda path, device: (lm, {}))
    monkeypatch.setattr(tk, "Tokenizer", Tok)
    caps = ["a cat", "a dog on a long sofa", "two red buses"]
    together = TextEncoder("x", "t", max_len=10, cache_limit=0).encode(caps, with_mask=True)
    alone = [TextEncoder("x", "t", max_len=10).encode([c], with_mask=True) for c in caps]
    for j in range(3):
        assert torch.allclose(together[0][j], alone[j][0][0], atol=1e-6)
        assert torch.equal(together[1][j], alone[j][1][0])
    e = TextEncoder("x", "t", max_len=10, cache_limit=1)
    e.encode(caps)
    assert len(e.cache) == 1                            # bounded


def test_the_frechet_distance_matches_its_closed_form():
    from aksharallm.pictures.photos import frechet
    rng = np.random.default_rng(0)
    a = rng.normal(size=(20000, 4))
    b = rng.normal(size=(20000, 4)) * 3 + 2
    # isotropic: |mu|^2 + d (1 + 9 - 2*3) = 4*4 + 4*4 = 32
    assert frechet(a, b) == pytest.approx(32.0, rel=0.03)
    assert abs(frechet(a, a)) < 1e-9


def test_the_photo_report_puts_a_ceiling_and_a_floor_beside_every_number():
    from aksharallm.pictures.clip import ClipConfig, ClipModel
    from aksharallm.pictures.photos import compare, score

    class StubText:
        dim, max_len = 4, 4

        def encode(self, captions, with_mask=False):
            out = torch.zeros(len(captions), 4, 4)
            for i, c in enumerate(captions):
                out[i, 0, int(c[-1])] = 1.0
            return out, torch.tensor([[True, False, False, False]] * len(captions))
    m = ClipModel(ClipConfig(image_size=16, patch=4, d_vision=8, n_layers=1, n_heads=2, d_text=8,
                             d_embed=4, text="s", text_len=4, text_dim=4)).eval()
    m.attach_text(StubText())
    caps = [f"c{i}" for i in range(4)]
    txt = m.embed_texts(caps).detach()
    # "real" photos embed exactly as their captions; "drawn" ones as the next caption's
    m.embed_images = lambda x: (txt if x[0, 0, 0, 0] > 0 else txt.roll(-1, 0))[: len(x)]
    real = np.full((4, 16, 16, 3), 255, np.uint8)
    drawn = np.zeros((4, 16, 16, 3), np.uint8)
    r = score(m, drawn, real, caps)
    assert r["real"]["top1"] == 1.0 and r["drawn"]["top1"] == 0.0
    assert r["real"]["match"] == pytest.approx(1.0, abs=1e-5)
    assert r["real"]["mismatch"] < r["real"]["match"]
    assert r["chance"]["top1"] == 0.25
    with pytest.raises(ValueError, match="different captions"):
        compare({"captions": caps, "drawn": r["drawn"]}, {"captions": caps[::-1], "drawn": r["drawn"]})


def test_the_flow_trainer_reads_a_photo_with_one_of_its_own_captions(tmp_path):
    from aksharallm.pictures.coco import pack
    from aksharallm.pictures.train_flow import batch_of
    from aksharallm.vision.image import ImageCaptions
    _fake_coco(tmp_path / "raw")
    pack(tmp_path / "raw", tmp_path / "c", "val", workers=1, progress=None)
    ds = ImageCaptions(tmp_path / "c", split="val", val_frac=1.0, seed=0)

    class Model:
        text_proj = nn.Linear(1, 1)
        said = []

        def condition(self, caps, device="cpu"):
            Model.said = caps
            return torch.zeros(len(caps), 2, 3)
    batch_of(ds, 16, 0.0, np.random.default_rng(0), Model())
    for i, c in zip(ds.last_picks, Model.said, strict=True):
        assert c in ds.manifest.facts[int(i)]["captions"]
    assert len(set(Model.said)) > 6                   # not always the first caption


def test_the_autoencoder_report_on_photos_is_psnr_only(tmp_path):
    from aksharallm.pictures.autoencoder import AEConfig, PictureAutoencoder
    from aksharallm.pictures.coco import pack
    from aksharallm.pictures.train_ae import judge_set
    from aksharallm.vision.image import ImageCaptions
    _fake_coco(tmp_path / "raw")
    pack(tmp_path / "raw", tmp_path / "c", "val", workers=1, progress=None)
    ds = ImageCaptions(tmp_path / "c", split="val", val_frac=1.0)
    r = judge_set(PictureAutoencoder(AEConfig(image_size=64, channels=[8, 8, 8], latent_dim=2)), ds, 4, "cpu")
    assert r["all"] is None and r["n"] == 4 and np.isfinite(r["psnr"])
