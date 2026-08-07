"""Vision: the bridge, and the two ways it silently is not what it claims to be.

Two tests here exist because the thing they check went wrong, and both failures produce a
model that trains beautifully:

* **the double shift.** `VisionLanguageModel.forward` slices the text hidden states one
  position early — that slice *is* the shift — and the batch builder shifted the targets
  again in the ordinary way. The model learned, perfectly, to emit the token *after* next:
  training loss 0.0027, and every caption came out as `'w green'`;
* **"frozen" that is not frozen.** `requires_grad = False` does not stop an optimizer built
  over `model.parameters()` from holding the weights, and weight decay moves them without
  consulting a gradient.

The rest pin the arithmetic that everything else assumes: patches are a strided convolution,
the image costs exactly `n_image_tokens` positions, and the score is three booleans rather
than one number.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
import torch.nn as nn

from aksharallm.config import ModelConfig
from aksharallm.model.transformer import Transformer
from aksharallm.vision.encoder import Projector, VisionConfig, VisionEncoder, VisionTower, patchify
from aksharallm.vision.image import (
    COLOURS,
    ImageCaptions,
    caption_of,
    from_tensor,
    render,
    synth_corpus,
    to_tensor,
)
from aksharallm.vision.lm import VisionLanguageModel, score_batch, score_caption

VOCAB = 512


def tiny_vision(**kw) -> VisionConfig:
    return VisionConfig(image_size=32, patch=8, d_vision=64, n_layers=2, n_heads=4, **kw)


def tiny_lm(d_model: int = 64) -> Transformer:
    return Transformer(ModelConfig(vocab_size=VOCAB, d_model=d_model, n_layers=2,
                                   n_heads=4, max_seq_len=64))


# ---------------------------------------------------------------------------------------
# patches
# ---------------------------------------------------------------------------------------


def test_patchify_is_a_strided_convolution():
    """The claim the module docstring makes, asserted. If these ever diverge, one of the two
    descriptions of what a patch embedding *is* has become wrong."""
    torch.manual_seed(0)
    images = torch.randn(2, 3, 32, 32)
    patch, d = 8, 16
    conv = nn.Conv2d(3, d, kernel_size=patch, stride=patch, bias=False)
    linear = nn.Linear(3 * patch * patch, d, bias=False)
    # The conv's kernel, flattened the way `patchify` flattens a patch: (C, p, p).
    linear.weight.data = conv.weight.data.reshape(d, -1)

    by_conv = conv(images).flatten(2).transpose(1, 2)  # (B, n_patches, d)
    by_patch = linear(patchify(images, patch))
    assert torch.allclose(by_conv, by_patch, atol=1e-5)


def test_patches_are_row_major():
    """Patch `i` must be at grid position `(i // grid, i % grid)`, because that is what the
    position embedding assumes. A column-major flatten trains fine and learns a transposed
    world."""
    img = torch.zeros(1, 3, 32, 32)
    img[0, :, 0:8, 8:16] = 1.0  # row 0, column 1
    p = patchify(img, 8)
    ink = [i for i in range(p.shape[1]) if p[0, i].abs().sum() > 0]
    assert ink == [1]


def test_a_partial_patch_is_refused_not_dropped():
    with pytest.raises(ValueError, match="not divisible"):
        patchify(torch.zeros(1, 3, 30, 30), 8)
    with pytest.raises(ValueError, match="not divisible"):
        _ = VisionConfig(image_size=30, patch=8).grid


def test_the_encoder_gives_one_vector_per_patch():
    cfg = tiny_vision()
    enc = VisionEncoder(cfg)
    out = enc(torch.randn(2, 3, 32, 32))
    assert out.shape == (2, cfg.n_patches, cfg.d_vision) == (2, 16, 64)


def test_attention_in_the_tower_is_bidirectional():
    """A patch at the top-left must see the bottom-right. With a causal mask, patch 0's
    output could not depend on the last patch at all."""
    torch.manual_seed(0)
    enc = VisionEncoder(tiny_vision()).eval()
    a = torch.randn(1, 3, 32, 32)
    b = a.clone()
    b[0, :, 24:, 24:] = -a[0, :, 24:, 24:]  # change only the last patch
    with torch.no_grad():
        first_a, first_b = enc(a)[0, 0], enc(b)[0, 0]
    assert not torch.allclose(first_a, first_b, atol=1e-4)


# ---------------------------------------------------------------------------------------
# the projector
# ---------------------------------------------------------------------------------------


def test_the_projector_lands_in_the_language_models_width():
    proj = Projector(64, 128)
    assert proj(torch.randn(2, 16, 64)).shape == (2, 16, 128)


def test_pooling_changes_how_much_context_an_image_costs():
    """The trade, as a number: 16 patches into 4 tokens is 4 positions of the window."""
    tower = VisionTower(tiny_vision(n_tokens=4), d_model=64)
    assert tower.n_image_tokens == 4
    assert tower(torch.randn(2, 3, 32, 32)).shape == (2, 4, 64)


def test_without_pooling_an_image_costs_one_position_per_patch():
    tower = VisionTower(tiny_vision(), d_model=64)
    assert tower.n_image_tokens == tower.cfg.n_patches == 16


# ---------------------------------------------------------------------------------------
# frozen means frozen
# ---------------------------------------------------------------------------------------


def test_the_language_model_is_bit_identical_after_a_training_step():
    """**The one that matters.** `requires_grad = False` does not stop an optimizer built
    over `model.parameters()` from holding the weights, and weight decay moves them without
    consulting a gradient. `trainable_parameters()` is the single place that decides."""
    torch.manual_seed(0)
    model = VisionLanguageModel(tiny_lm(), tiny_vision())
    before = {k: v.clone() for k, v in model.lm.state_dict().items()}

    opt = torch.optim.AdamW(model.trainable_parameters(), lr=0.1, weight_decay=0.5)
    images = torch.randn(2, 3, 32, 32)
    text = torch.randint(0, VOCAB, (2, 6))
    for _ in range(3):
        _, loss = model(images, text, targets=text)
        opt.zero_grad()
        loss.backward()
        opt.step()

    after = model.lm.state_dict()
    for k, v in before.items():
        assert torch.equal(v, after[k]), f"the frozen language model moved: {k}"


def test_the_tower_does_move():
    """The complement — otherwise "nothing changed" would pass for the wrong reason."""
    torch.manual_seed(0)
    model = VisionLanguageModel(tiny_lm(), tiny_vision())
    before = model.tower.projector.net[0].weight.clone()
    opt = torch.optim.AdamW(model.trainable_parameters(), lr=0.1)
    _, loss = model(torch.randn(2, 3, 32, 32), torch.randint(0, VOCAB, (2, 6)),
                    targets=torch.randint(0, VOCAB, (2, 6)))
    loss.backward()
    opt.step()
    assert not torch.equal(before, model.tower.projector.net[0].weight)


def test_only_the_tower_is_trainable_by_default():
    model = VisionLanguageModel(tiny_lm(), tiny_vision())
    counts = model.n_params()
    assert counts["trainable"] == counts["total"]
    assert counts["trainable"] < counts["language_model"]


def test_unfreezing_is_possible_and_says_so():
    model = VisionLanguageModel(tiny_lm(), tiny_vision(), freeze_language_model=False)
    assert model.n_params()["trainable"] > model.n_params()["language_model"]


# ---------------------------------------------------------------------------------------
# the shift
# ---------------------------------------------------------------------------------------


def test_the_logits_line_up_with_the_text_not_one_past_it():
    """**The `'w green'` regression.** The `n_img - 1` slice IS the shift, so `logits[:, i]`
    predicts `text[:, i]`. Train a model to output a fixed caption and check that greedy
    decoding reproduces it — a double shift makes the loss go to zero and this fail."""
    torch.manual_seed(0)
    # The language model is unfrozen HERE and only here: this test is about alignment, not
    # about whether a two-layer frozen model has the capacity to be steered into emitting
    # three specific tokens. Starving it of capacity would make the test fail for a reason
    # that has nothing to do with what it claims to check.
    model = VisionLanguageModel(tiny_lm(), tiny_vision(n_tokens=4),
                                freeze_language_model=False)
    images = torch.randn(1, 3, 32, 32)
    wanted = torch.tensor([[1, 7, 9, 11]])
    targets = wanted.clone()
    targets[:, 0] = -100  # position 0 is the BOS, which is context and not a target

    opt = torch.optim.AdamW(model.trainable_parameters(), lr=0.01)
    for _ in range(400):
        _, loss = model(images, wanted, targets=targets)
        opt.zero_grad()
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        # Feed the prefix and check each next token is the one that follows it.
        for i in range(1, wanted.shape[1]):
            logits, _ = model(images, wanted[:, :i])
            assert int(logits[0, -1].argmax()) == int(wanted[0, i]), i


def test_the_image_costs_exactly_n_image_tokens_of_context():
    model = VisionLanguageModel(tiny_lm(), tiny_vision(n_tokens=4))
    x = model.embed(torch.randn(2, 3, 32, 32), torch.randint(0, VOCAB, (2, 7)))
    assert x.shape[1] == 4 + 7


def test_an_untrained_model_is_near_or_above_uniform():
    """Not *equal* to `ln(V)`: an untrained projector hands the frozen model vectors from
    nowhere in its input distribution, so it starts a little worse than uniform. Measured on
    the real run: 11.27 against ln(8192) = 9.01."""
    torch.manual_seed(0)
    model = VisionLanguageModel(tiny_lm(), tiny_vision())
    _, loss = model(torch.randn(4, 3, 32, 32), torch.randint(0, VOCAB, (4, 6)),
                    targets=torch.randint(0, VOCAB, (4, 6)))
    assert float(loss.detach()) > math.log(VOCAB) - 0.5


# ---------------------------------------------------------------------------------------
# the corpus
# ---------------------------------------------------------------------------------------


def test_a_rendered_image_contains_its_colour_and_nothing_else():
    img = render({"count": 3, "colour": "red", "shape": "circle"}, 64, seed=1)
    ink = img.reshape(-1, 3)
    ink = ink[ink.sum(1) > 100]
    assert len(ink) > 0
    assert np.abs(ink.astype(int) - np.array(COLOURS["red"])).sum(1).max() < 30


@pytest.mark.parametrize("shape", ["circle", "square", "triangle"])
def test_every_shape_renders_something_distinct(shape):
    """A shape that renders as nothing, or as the same thing as another, makes the corpus
    unlearnable in a way that looks like a model problem."""
    others = {s: render({"count": 1, "colour": "red", "shape": s}, 64, seed=2)
              for s in ("circle", "square", "triangle")}
    mine = others[shape]
    assert (mine.sum(-1) > 100).sum() > 20, "renders as (nearly) nothing"
    for s, img in others.items():
        if s != shape:
            assert not np.array_equal(mine, img)


def test_more_shapes_means_more_ink():
    """The property counting depends on. If four circles drew no more pixels than one, the
    task would be impossible and the model would be blamed."""
    ink = [int((render({"count": n, "colour": "blue", "shape": "circle"}, 64, seed=3)
                .sum(-1) > 100).sum()) for n in (1, 2, 3, 4)]
    assert ink == sorted(ink) and ink[-1] > ink[0] * 2


def test_captions_are_grammatical():
    assert caption_of({"count": 1, "colour": "red", "shape": "circle"}) == "one red circle"
    assert caption_of({"count": 3, "colour": "red", "shape": "circle"}) == "three red circles"


def test_the_held_out_combination_is_absent_from_training(tmp_path):
    """Both attributes must stay common — it is the PAIR that is held out, or the test is
    about whether the model knows the word 'purple' rather than whether it can compose."""
    man = synth_corpus(tmp_path / "c", n_images=400, size=32,
                       hold_out=("purple", "triangle"), progress=None)
    pairs = {(f["colour"], f["shape"]) for f in man.facts}
    assert ("purple", "triangle") not in pairs
    assert any(c == "purple" for c, _ in pairs), "purple vanished entirely"
    assert any(s == "triangle" for _, s in pairs), "triangles vanished entirely"


def test_the_pixel_round_trip_is_exact():
    """`astype(uint8)` truncates; 17.999999 becomes 17. Rounding is what makes this true."""
    a = np.random.default_rng(0).integers(0, 256, (4, 16, 16, 3), dtype=np.uint8)
    assert np.array_equal(from_tensor(to_tensor(a)), a)


def test_a_corpus_whose_manifest_disagrees_with_its_bytes_is_refused(tmp_path):
    synth_corpus(tmp_path / "c", n_images=20, size=32, progress=None)
    (tmp_path / "c" / "images.bin").write_bytes(b"\x00" * 100)
    with pytest.raises(ValueError, match="Re-render"):
        ImageCaptions(tmp_path / "c")


# ---------------------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------------------


def test_the_score_is_three_booleans_not_one_number():
    """A model that gets colour and shape right and never counts is a specific, diagnosable
    failure that a single accuracy would average away."""
    fact = {"count": 3, "colour": "red", "shape": "circle"}
    assert score_caption(fact, "three red circles") == {"count": True, "colour": True,
                                                        "shape": True}
    partial = score_caption(fact, "two red circles")
    assert partial["colour"] and partial["shape"] and not partial["count"]


def test_the_batch_score_reports_all_three_separately():
    fact = {"count": 2, "colour": "blue", "shape": "square"}
    out = score_batch([(fact, "two blue squares"), (fact, "one blue square")])
    assert out["colour"] == 1.0 and out["shape"] == 1.0
    assert out["count"] == 0.5 and out["all_three"] == 0.5
