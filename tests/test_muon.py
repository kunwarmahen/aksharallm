"""`train/muon.py`: orthogonalised momentum for the block matrices, AdamW for the rest.

What could be quietly wrong: Newton-Schulz not orthogonalising (a Muon that is really SGD
with momentum still trains, just differently); the wrong parameters going to Muon (an
embedding updated by Muon still trains too); the update's scale (Muon at the wrong RMS is a
different learning rate, and the comparison with AdamW then measures that); and a resume
that silently swaps optimizer state.
"""

from __future__ import annotations

import math

import pytest
import torch

from aksharallm.config import ModelConfig
from aksharallm.model.transformer import Transformer
from aksharallm.train.muon import Muon, newton_schulz, split_for_muon


def small_model():
    return Transformer(ModelConfig(vocab_size=64, d_model=32, n_layers=2, n_heads=4,
                                   n_kv_heads=2, max_seq_len=16))


@pytest.mark.parametrize("shape", [(32, 32), (16, 48), (48, 16)])
def test_newton_schulz_sets_every_singular_value_near_one(shape):
    torch.manual_seed(0)
    # A low-rank-ish gradient: a few strong directions, which is what Muon is for.
    g = torch.randn(shape[0], 4) @ torch.randn(4, shape[1]) * 5 + 0.01 * torch.randn(shape)
    s_before = torch.linalg.svdvals(g)
    s = torch.linalg.svdvals(newton_schulz(g))
    assert s_before[0] / s_before[-1] > 100          # the input really is ill-conditioned
    k = min(shape)
    # The quintic trades exactness for speed: values land in roughly [0.7, 1.2], not at 1.
    assert s[:k].max() < 1.3 and s[:4].min() > 0.6
    u, _, vh = torch.linalg.svd(g, full_matrices=False)
    # ... and along the gradient's own directions, not some other orthogonal matrix.
    top = (u[:, :4].T @ newton_schulz(g) @ vh[:4].T).diagonal()
    assert (top > 0.6).all()


def test_the_split_sends_block_matrices_to_muon_and_the_embedding_to_adamw():
    m = small_model()
    muon, decay, plain = split_for_muon(m.named_parameters())
    names = {id(p): n for n, p in m.named_parameters()}
    assert {names[id(p)] for p in decay} == {"tok_emb.weight"}   # tied with the head
    assert all(p.dim() == 1 for p in plain)
    assert all(".attn." in names[id(p)] or ".ffn." in names[id(p)] for p in muon)
    assert len(muon) == 2 * 7                                    # wq wk wv wo w1 w2 w3


def test_a_muon_step_has_the_rms_adamw_would_have():
    # Moonlight's scaling is what lets one lr serve both kinds of group: the update's RMS is
    # ~0.2 * lr whatever the matrix's shape.
    torch.manual_seed(0)
    for shape in [(64, 64), (256, 64)]:
        w = torch.nn.Parameter(torch.randn(shape))
        before = w.detach().clone()
        opt = Muon([{"params": [w], "muon": True}], lr=1e-2, momentum=0.0)
        w.grad = torch.randn(shape)
        opt.step()
        rms = (w.detach() - before).pow(2).mean().sqrt().item()
        assert rms == pytest.approx(0.2 * 1e-2, rel=0.25), shape


def test_the_adamw_groups_match_torch_adamw():
    torch.manual_seed(0)
    a = torch.nn.Parameter(torch.randn(8, 8))
    b = torch.nn.Parameter(a.detach().clone())
    ours = Muon([{"params": [a], "weight_decay": 0.1}], lr=1e-2, betas=(0.9, 0.95))
    ref = torch.optim.AdamW([b], lr=1e-2, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)
    for _ in range(5):
        g = torch.randn(8, 8)
        a.grad, b.grad = g.clone(), g.clone()
        ours.step(), ref.step()
    assert torch.allclose(a, b, atol=1e-6)


def test_muon_trains_the_model_and_resumes_exactly():
    torch.manual_seed(0)
    m = small_model()
    opt, _ = m.configure_optimizers(0.1, 3e-3, (0.9, 0.95), "cpu", name="muon")
    x = torch.randint(0, 64, (8, 16))

    def step():
        opt.zero_grad()
        _, loss = m(x[:, :-1], x[:, 1:])
        loss.backward()
        opt.step()
        return loss.item()

    first = step()
    for _ in range(30):
        last = step()
    assert last < first - 1.0
    assert first == pytest.approx(math.log(64), abs=0.3)

    # A resume restores the momentum buffers; the next step is then identical.
    state = {k: v.clone() for k, v in m.state_dict().items()}
    saved = opt.state_dict()
    a = step()
    m.load_state_dict(state)
    opt2, _ = m.configure_optimizers(0.1, 3e-3, (0.9, 0.95), "cpu", name="muon")
    opt2.load_state_dict(saved)
    opt = opt2
    assert step() == a


def test_an_adamw_checkpoint_refuses_to_load_into_muon():
    m = small_model()
    adamw, _ = m.configure_optimizers(0.1, 1e-3, (0.9, 0.95), "cpu")
    muon, _ = m.configure_optimizers(0.1, 1e-3, (0.9, 0.95), "cpu", name="muon")
    with pytest.raises(ValueError):
        muon.load_state_dict(adamw.state_dict())
    with pytest.raises(ValueError):
        m.configure_optimizers(0.1, 1e-3, (0.9, 0.95), "cpu", name="sgd")


def test_the_two_arms_differ_in_exactly_one_key():
    import yaml

    a, b = (yaml.safe_load(open(f"configs/tiny-{n}.yaml")) for n in ("adamw", "muon"))
    assert a["optim"].pop("name") == "adamw" and b["optim"].pop("name") == "muon"
    for d, n in ((a, "adamw"), (b, "muon")):
        assert d.pop("name") == f"tiny-{n}"
        assert d["train"].pop("out_dir") == f"checkpoints/tiny-{n}"
    assert a == b
    tiny = yaml.safe_load(open("configs/tiny.yaml"))
    assert a["model"] == tiny["model"] and a["train"]["max_steps"] == tiny["train"]["max_steps"]


def test_the_100m_muon_arm_is_its_control_plus_one_key():
    """small-student (trained 2026-11-02) is the control; only the Muon arm is new."""
    import yaml

    a = yaml.safe_load(open("configs/small-student.yaml"))
    b = yaml.safe_load(open("configs/small-student-muon.yaml"))
    assert "name" not in a["optim"] and b["optim"].pop("name") == "muon"
    assert b.pop("name") == "small-student-muon" and a.pop("name") == "small-student"
    assert b["train"].pop("out_dir") == "checkpoints/small-student-muon"
    a["train"].pop("out_dir")
    assert a == b
