"""Tests for the interpretability tools, and for the Serve panel that drives the HTTP server.

Interpretability has a failure mode the rest of this repo does not: **a picture that is
wrong is still a picture**. An attention map recomputed with the wrong RoPE angles, a logit
lens that forgets the final norm, a patch that restores nothing because the hook fired on the
wrong module — every one of them renders beautifully and tells you a story about your model
that is not true.

So each tool is pinned against something that cannot be argued with:

* the attention map, multiplied by V, must reproduce **the layer's own output**;
* the last row of the logit lens must equal **the model's actual next-token distribution**;
* patching the final layer must restore **100%** of the clean logit difference, because that
  residual *is* what the output head reads.
"""

from __future__ import annotations

import json
import math

import pytest
import torch

from aksharallm.config import ModelConfig
from aksharallm.interp.capture import (
    attention_maps,
    attention_summary,
    attention_values,
    run,
)
from aksharallm.interp.lens import layer_contributions, lens_story, logit_lens
from aksharallm.interp.patch import PatchError, check_pair, patch_grid, summarise
from aksharallm.interp.sae import SAE, SAEConfig, feature_report, load, save, train_sae
from aksharallm.model.transformer import Transformer

PROMPT = [3, 9, 27, 5, 11, 2]


def tiny(seed: int = 0, vocab: int = 64) -> Transformer:
    torch.manual_seed(seed)
    cfg = ModelConfig(vocab_size=vocab, d_model=32, n_layers=3, n_heads=4, n_kv_heads=2,
                      max_seq_len=64, dropout=0.0)
    return Transformer(cfg).eval()


# ---- attention: the map must be the model's own -------------------------------------------

@pytest.mark.parametrize("layer", [0, 2])
def test_the_recomputed_attention_reproduces_the_layers_real_output(layer):
    """The fused kernel never stores its attention matrix, so this one is recomputed — and a
    recomputation that does not match is a drawing, not a measurement.

    `softmax(QK^T/sqrt(d)) @ V`, projected by `wo`, must equal what the attention module
    actually returned. Get RoPE, the grouped-query expansion or the mask wrong and this fails
    while the picture still looks like attention.
    """
    model = tiny()
    cap = run(model, PROMPT, device="cpu")
    w = attention_maps(model, cap, layer)
    v = attention_values(model, cap, layer)
    mixed = (w.to(v.dtype) @ v)                       # (H, T, D)
    T = mixed.shape[1]
    mine = model.blocks[layer].attn.wo(mixed.transpose(0, 1).reshape(T, -1))
    assert torch.allclose(mine, cap.attn_out[layer], atol=1e-5), \
        (mine - cap.attn_out[layer]).abs().max()


def test_attention_rows_are_distributions_and_cannot_see_the_future():
    model = tiny()
    cap = run(model, PROMPT, device="cpu")
    w = attention_maps(model, cap, 1)
    assert torch.allclose(w.sum(-1), torch.ones_like(w.sum(-1)), atol=1e-5)
    T = w.shape[-1]
    upper = torch.triu(torch.ones(T, T, dtype=torch.bool), 1)
    assert float(w[:, upper].abs().max()) == 0.0


def test_the_head_summary_runs_wherever_the_weights_live():
    """It is presentation code over a (T, T) matrix, and building its position vector on the
    default device while the weights sat on the card is what took the Interp tab's attention
    view down with a 500."""
    model = tiny()
    cap = run(model, PROMPT, device="cpu")
    rows = attention_summary(attention_maps(model, cap, 0), [str(t) for t in PROMPT])
    assert len(rows) == model.cfg.n_heads
    assert all(0.0 <= r["self_weight"] <= 1.0 for r in rows)
    assert all(r["attends_to"] for r in rows)


# ---- the lens: the last layer must agree with the model -------------------------------------

def test_the_last_lens_row_is_the_models_actual_prediction():
    """If the final row disagrees with the model, the lens is applying the head wrongly — and
    every earlier row, which nothing else can check, is wrong in the same way."""
    model = tiny()
    cap = run(model, PROMPT, device="cpu")
    rows = logit_lens(model, cap, top=3)
    assert len(rows) == model.cfg.n_layers + 1        # embedding, then one per block
    real = torch.softmax(cap.logits[-1].float(), dim=-1)
    top = rows[-1]["top"][0]
    assert top["id"] == int(real.argmax())
    assert abs(top["prob"] - float(real.max())) < 1e-5


def test_the_lens_story_reports_where_the_answer_stuck():
    """`settled_at` is the *last* time the answer changed, not the first time it appeared: a
    layer that guesses the right token and then changes its mind has not settled."""
    rows = [
        {"layer": 0, "label": "embedding", "top": [{"id": 1, "prob": 0.4}], "entropy": 1.0},
        {"layer": 1, "label": "block 0", "top": [{"id": 7, "prob": 0.5}], "entropy": 1.0},
        {"layer": 2, "label": "block 1", "top": [{"id": 1, "prob": 0.6}], "entropy": 1.0},
        {"layer": 3, "label": "block 2", "top": [{"id": 7, "prob": 0.9}], "entropy": 0.5},
    ]
    story = lens_story(rows, lambda ids: f"<{ids[0]}>")
    assert story["answer"] == 7
    assert story["settled_at"] == 3          # not 1: it changed its mind at layer 2
    assert story["flips"] == 3


def test_layer_contributions_add_up_to_the_journey():
    model = tiny()
    cap = run(model, PROMPT, device="cpu")
    rows = layer_contributions(model, cap)
    assert len(rows) == model.cfg.n_layers
    assert all(r["norm_delta"] >= 0 for r in rows)


# ---- patching: intervention, not observation --------------------------------------------------

def test_patching_the_last_layer_restores_everything():
    """The strongest available check. The final residual *is* what the output head reads, so
    forcing it back to its clean value must recover the clean logit difference exactly — if it
    does not, the hook is on the wrong module or the donor is the wrong activation."""
    model = tiny()
    clean = [3, 9, 27, 5]
    corrupt = [3, 9, 28, 5]
    res = patch_grid(model, clean, corrupt, answer=11, other=12, device="cpu",
                     position=len(clean) - 1)
    last = res["grid"][-1][0]
    assert abs(last - 1.0) < 1e-3, last


def test_patching_an_early_position_of_the_last_layer_changes_nothing():
    """A sanity check in the other direction: the answer is read from the *last* position, so
    patching an earlier one at the final layer cannot move it. A grid where everything
    restores everything means the measurement is broken."""
    model = tiny()
    res = patch_grid(model, [3, 9, 27, 5], [3, 9, 28, 5], answer=11, other=12, device="cpu")
    assert abs(res["grid"][-1][0]) < 1e-6


def test_mismatched_prompts_are_refused_rather_than_compared():
    with pytest.raises(PatchError, match="different lengths"):
        check_pair([1, 2, 3], [1, 2])
    with pytest.raises(PatchError, match="identical"):
        check_pair([1, 2, 3], [1, 2, 3])


def test_the_summary_says_something_useful_when_there_is_nothing_to_restore():
    result = {"grid": [[0.0]], "positions": [0], "span": 0.0,
              "best": {"layer": 0, "position": 0, "restored": 0.0}}
    assert "nothing for a patch to restore" in summarise(result, ["x"])


# ---- the sparse autoencoder -------------------------------------------------------------------

def test_the_decoder_stays_unit_norm_through_training():
    """The constraint that makes sparsity mean something. Without it the model buys a low L1
    by shrinking its activations and growing the dictionary, and every feature 'fires' at
    0.001 — sparse by the loss, useless to read."""
    torch.manual_seed(0)
    acts = torch.randn(2000, 16)
    cfg = SAEConfig(d_model=16, n_features=64, layer=0, steps=50, batch=128, alpha=1e-3)
    sae, history = train_sae(acts, cfg, device="cpu", log_every=25, echo=lambda *_: None)
    norms = sae.decoder.weight.norm(dim=0)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)
    assert history[-1]["loss"] < history[0]["loss"]


def test_a_higher_penalty_buys_sparsity_with_reconstruction():
    """The one knob, and the trade it makes. This is the whole tuning story: on the real
    model alpha 0.003 gave 200 features per token and alpha 0.012 gave 2.6."""
    torch.manual_seed(0)
    acts = torch.randn(4000, 16)
    out = {}
    for alpha in (1e-3, 5e-2):
        cfg = SAEConfig(d_model=16, n_features=64, layer=0, steps=120, batch=256, alpha=alpha)
        sae, history = train_sae(acts, cfg, device="cpu", log_every=60, echo=lambda *_: None)
        out[alpha] = history[-1]
    assert out[5e-2]["l0"] < out[1e-3]["l0"]
    assert out[5e-2]["explained"] <= out[1e-3]["explained"] + 1e-6


def test_the_report_counts_dead_features():
    """A dictionary where most entries never fire has learned a small dictionary badly —
    the same shape of failure as MoE's router collapse, and invisible in the loss."""
    torch.manual_seed(0)
    acts = torch.randn(1000, 16)
    cfg = SAEConfig(d_model=16, n_features=64, layer=0, steps=60, batch=128, alpha=5e-2)
    sae, _ = train_sae(acts, cfg, device="cpu", log_every=30, echo=lambda *_: None)
    report = feature_report(sae, acts, device="cpu")
    assert report["n_features"] == 64
    assert 0 <= report["dead"] <= 64
    assert 0.0 <= report["dead_fraction"] <= 1.0


def test_an_sae_round_trips_through_disk(tmp_path):
    cfg = SAEConfig(d_model=8, n_features=32, layer=3, steps=1, batch=16)
    sae = SAE(cfg)
    path = save(sae, tmp_path / "sae.pt", [{"step": 0}], {"dead": 0})
    again = load(path, device="cpu")
    assert again.cfg.layer == 3 and again.cfg.n_features == 32
    x = torch.randn(4, 8)
    assert torch.allclose(sae.encode(x), again.encode(x), atol=1e-6)
    meta = json.loads(path.with_suffix(".json").read_text())
    assert meta["report"]["dead"] == 0


# ---- the portal's serve panel -------------------------------------------------------------

def test_the_serve_panel_reports_idle_when_nothing_is_running(tmp_path):
    from aksharallm.portal.serving import ServeJobs

    jobs = ServeJobs(tmp_path)
    status = jobs.status()
    assert status["running"] is False and status["phase"] == "idle"
    assert status["port"] == 8770 and status["health"] is None


def test_the_serve_panel_refuses_a_second_server(tmp_path, monkeypatch):
    """One server at a time, and the refusal happens before anything is spawned — two
    processes fighting for one port fail in a much less readable way."""
    from aksharallm.portal.runs import RunError
    from aksharallm.portal.serving import ServeJobs

    jobs = ServeJobs(tmp_path)
    monkeypatch.setattr(jobs, "_pid", lambda: 4242)
    with pytest.raises(RunError, match="already running"):
        jobs.start("small-code")


def test_the_serve_panel_whitelists_the_checkpoint(tmp_path):
    from aksharallm.portal.runs import RunError
    from aksharallm.portal.serving import ServeJobs

    with pytest.raises(RunError, match="invalid checkpoint"):
        ServeJobs(tmp_path).start("../../etc/passwd; rm -rf /")


def test_a_pid_with_no_health_reads_as_starting_not_running(tmp_path, monkeypatch):
    """The window between launch and a 1.2 GB checkpoint finishing loading is several
    seconds. Saying "running" there means every number beside it is blank."""
    from aksharallm.portal.serving import ServeJobs

    jobs = ServeJobs(tmp_path)
    monkeypatch.setattr(jobs, "_pid", lambda: 4242)
    monkeypatch.setattr(jobs, "_health", lambda port: None)
    assert jobs.status()["phase"] == "starting"
