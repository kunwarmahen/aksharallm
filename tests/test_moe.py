"""Tests for the mixture of experts.

Two of these matter more than the rest, because they cover the failures that do not announce
themselves:

* **identity at init** — sparse upcycling is only worth doing if the upcycled model starts as
  *exactly* the dense model. If the copy or the router init is subtly wrong you get a model
  that trains anyway, from a worse starting point, and nothing says so.
* **the aux loss is training-only** — a validation loss carrying a balancing term is not a
  cross-entropy, and the entire 13.8M experiment is a comparison of val losses against the
  dense baseline's 1.472.
"""

import pytest
import torch

from aksharallm.config import ModelConfig
from aksharallm.model.moe import MoEFeedForward, Router, moe_stats, upcycle_state_dict
from aksharallm.model.transformer import SwiGLU, Transformer


def cfg(**kw):
    base = dict(vocab_size=64, d_model=32, n_layers=2, n_heads=4, max_seq_len=16, d_ff=48)
    base.update(kw)
    return ModelConfig(**base)


@pytest.fixture
def dense():
    torch.manual_seed(0)
    return Transformer(cfg()).eval()


# ---- configuration ----------------------------------------------------------------------

def test_dense_by_default():
    model = Transformer(cfg())
    assert not cfg().is_moe
    assert all(isinstance(b.ffn, SwiGLU) for b in model.blocks)


def test_expert_width_defaults_to_matched_active_params():
    """The default is the honest comparison: k experts of d_ff/k cost what the dense FFN
    cost, so the experiment changes capacity and not FLOPs."""
    c = cfg(n_experts=8, moe_top_k=2)
    assert c.moe_expert_d_ff == c.d_ff // 2
    model = Transformer(c)
    dense_model = Transformer(cfg())
    assert model.num_params() > dense_model.num_params()
    # Active is the dense count plus the routers, which are d_model x n_experts per layer.
    routers = c.n_layers * c.d_model * c.n_experts
    assert model.num_active_params() == dense_model.num_params() + routers


def test_moe_every_leaves_the_other_layers_dense():
    model = Transformer(cfg(n_layers=4, n_experts=4, moe_every=2))
    kinds = [type(b.ffn).__name__ for b in model.blocks]
    assert kinds == ["MoEFeedForward", "SwiGLU", "MoEFeedForward", "SwiGLU"]


def test_top_k_must_fit_the_expert_count():
    with pytest.raises(ValueError):
        cfg(n_experts=4, moe_top_k=5)


# ---- routing ----------------------------------------------------------------------------

def test_router_weights_sum_to_one_over_the_chosen_k():
    """Renormalising over the top-k is what makes upcycling an identity; it is not taste."""
    torch.manual_seed(0)
    router = Router(d_model=16, n_experts=8, top_k=2)
    w, idx, aux, stats = router(torch.randn(20, 16))
    assert w.shape == (20, 2) and idx.shape == (20, 2)
    assert torch.allclose(w.sum(-1), torch.ones(20), atol=1e-5)
    assert stats.counts.sum() == 40          # 20 tokens x top-2 assignments


def test_every_token_is_routed_and_none_are_dropped():
    """No capacity factor: an expert that gets 90% of the batch still processes all of it."""
    torch.manual_seed(0)
    moe = MoEFeedForward(cfg(n_experts=4, moe_top_k=2))
    x = torch.randn(2, 8, 32)
    out = moe(x)
    assert out.shape == x.shape
    assert moe.stats.counts.sum() == 2 * 8 * 2
    assert torch.isfinite(out).all()


def test_balance_loss_is_lowest_when_routing_is_uniform():
    """`alpha * N * sum f_i P_i` — 1.0 x alpha at uniform, and higher when it is not.

    The input is all-positive on purpose: the gate has no bias (by design — a bias is a
    per-expert constant and exactly the cheap route to collapse), so "make expert 0 always
    win" needs a weight row that is large *and* an input whose sign is known.
    """
    router = Router(d_model=4, n_experts=4, top_k=1, aux_alpha=1.0, z_alpha=0.0)
    with torch.no_grad():
        router.gate.weight.zero_()           # uniform probabilities
    x = torch.rand(64, 4)
    _, _, uniform_aux, _ = router(x)

    with torch.no_grad():
        router.gate.weight[0] = 10.0         # expert 0 now wins every token
    _, _, collapsed_aux, stats = router(x)
    assert float(collapsed_aux) > float(uniform_aux)
    assert pytest.approx(float(uniform_aux), abs=0.05) == 1.0
    assert stats.as_dict()["max_share"] == 1.0
    assert float(collapsed_aux) == pytest.approx(4.0, abs=0.05), "N x 1 x 1 at full collapse"


def test_a_collapsed_router_is_visible_in_the_stats():
    """The number the trainer prints, checked against the two extremes it has to separate.

    Collapse is invisible in the loss curve — a model whose experts died is simply a bit
    worse than it should be — so this metric is the whole early-warning system.
    """
    from aksharallm.model.moe import MoEStats

    collapsed = MoEStats(torch.tensor([0.0, 0.0, 32.0, 0.0]), 32, 0.0, 0.0).as_dict()
    assert collapsed["max_share"] == 1.0
    assert collapsed["balance"] == pytest.approx(0.25)      # 1/N: one expert took it all
    assert collapsed["dead"] == 3

    healthy = MoEStats(torch.tensor([8.0, 8.0, 8.0, 8.0]), 32, 0.0, 0.0).as_dict()
    assert healthy["balance"] == pytest.approx(1.0)
    assert healthy["dead"] == 0


def test_a_zero_router_sends_every_token_to_the_same_experts():
    """Worth knowing, and the reason the gate is randomly initialised from scratch.

    A zero gate gives a uniform *probability* over experts — and `topk` has to break the
    resulting exact ties somehow, so every token picks the same k experts and the rest never
    train. (Which k is an implementation detail of `topk`, so this asserts the shape of the
    failure rather than the indices.) It is safe in exactly one place: sparse upcycling,
    where every expert is a copy of the same trained FFN, so which two are chosen does not
    matter — and the ties break as soon as the router takes a gradient step.
    """
    torch.manual_seed(0)
    model = Transformer(cfg(n_experts=4, moe_top_k=2))
    with torch.no_grad():
        for block in model.blocks:
            block.ffn.router.gate.weight.zero_()
    model(torch.randint(0, 64, (4, 16)))
    stats = model.routing()
    assert sorted(stats["shares"]) == [0.0, 0.0, 0.5, 0.5]
    assert stats["dead"] == 2


def test_healthy_routing_reports_a_balance_near_one():
    """A freshly initialised router on real data spreads tokens reasonably evenly. Not
    perfectly — that would need the balancing loss to have done some work — but nothing
    should be dead at step 0."""
    torch.manual_seed(0)
    model = Transformer(cfg(n_experts=4, moe_top_k=2))
    model(torch.randint(0, 64, (8, 16)))
    stats = model.routing()
    assert stats["dead"] == 0
    assert stats["balance"] > 0.3
    assert sum(stats["shares"]) == pytest.approx(1.0)


def test_stats_are_detached_from_the_graph():
    model = Transformer(cfg(n_experts=4))
    model(torch.randint(0, 64, (2, 8)))
    for block in model.blocks:
        assert not block.ffn.stats.counts.requires_grad


def test_moe_stats_is_none_for_a_dense_model(dense):
    dense(torch.randint(0, 64, (2, 8)))
    assert moe_stats(dense) is None
    assert dense.moe_aux_loss() is None


# ---- the loss ---------------------------------------------------------------------------

def test_aux_loss_is_added_while_training_and_not_while_evaluating():
    """The one that protects the experiment: val loss must stay a plain cross-entropy so it
    is comparable with the dense baseline's 1.472."""
    torch.manual_seed(0)
    model = Transformer(cfg(n_experts=4, moe_aux_alpha=0.5))
    idx = torch.randint(0, 64, (2, 8))
    tgt = torch.randint(0, 64, (2, 8))

    model.train()
    _, train_loss = model(idx, tgt)
    assert float(train_loss) > float(model.last_ce), "aux should be inside the training loss"

    model.eval()
    _, val_loss = model(idx, tgt)
    assert float(val_loss) == pytest.approx(float(model.last_ce))


def test_the_aux_loss_reaches_the_router_and_nothing_else():
    torch.manual_seed(0)
    model = Transformer(cfg(n_experts=4, moe_aux_alpha=1.0))
    model.train()
    _, loss = model(torch.randint(0, 64, (2, 8)), torch.randint(0, 64, (2, 8)))
    loss.backward()
    gate = model.blocks[0].ffn.router.gate.weight
    assert gate.grad is not None and gate.grad.abs().sum() > 0


def test_turning_the_aux_loss_off_is_possible_and_says_nothing_about_wisdom():
    model = Transformer(cfg(n_experts=4, moe_aux_alpha=0.0, moe_z_alpha=0.0))
    model.train()
    _, loss = model(torch.randint(0, 64, (2, 8)), torch.randint(0, 64, (2, 8)))
    assert float(loss) == pytest.approx(float(model.last_ce))


# ---- sparse upcycling -------------------------------------------------------------------

def test_upcycled_model_is_exactly_the_dense_model_at_init(dense):
    """The whole argument for upcycling. Every expert is a copy, the router is zeros, the
    top-k weights renormalise to 1 — so step 0 reproduces the trained model exactly, the
    same way LoRA's `B = 0` does (docs/11)."""
    up = Transformer(cfg(n_experts=4, moe_top_k=2, moe_expert_d_ff=48)).eval()
    up.load_state_dict(upcycle_state_dict(dense.state_dict(), n_experts=4), strict=True)

    idx = torch.randint(0, 64, (2, 8))
    with torch.no_grad():
        a, _ = dense(idx, full_logits=True)
        b, _ = up(idx, full_logits=True)
    assert torch.equal(a, b), "upcycling must be an identity, not an approximation"


def test_upcycling_holds_for_any_top_k(dense):
    for k in (1, 2, 4):
        up = Transformer(cfg(n_experts=4, moe_top_k=k, moe_expert_d_ff=48)).eval()
        up.load_state_dict(upcycle_state_dict(dense.state_dict(), n_experts=4), strict=True)
        idx = torch.randint(0, 64, (2, 8))
        with torch.no_grad():
            a, _ = dense(idx, full_logits=True)
            b, _ = up(idx, full_logits=True)
        assert torch.allclose(a, b, atol=1e-6), f"top_k={k}"


def test_jitter_breaks_the_identity_on_purpose(dense):
    up = Transformer(cfg(n_experts=4, moe_top_k=2, moe_expert_d_ff=48)).eval()
    gen = torch.Generator().manual_seed(0)
    up.load_state_dict(upcycle_state_dict(dense.state_dict(), 4, jitter=0.05, generator=gen),
                       strict=True)
    idx = torch.randint(0, 64, (2, 8))
    with torch.no_grad():
        a, _ = dense(idx, full_logits=True)
        b, _ = up(idx, full_logits=True)
    assert not torch.equal(a, b)


def test_upcycled_experts_start_identical_to_each_other(dense):
    sd = upcycle_state_dict(dense.state_dict(), n_experts=4)
    w1 = sd["blocks.0.ffn.w1"]
    assert w1.shape[0] == 4
    for e in range(1, 4):
        assert torch.equal(w1[0], w1[e])


# ---- the model still works -------------------------------------------------------------

def test_generation_works_through_the_kv_cache():
    """Routing is per token, so nothing about the KV cache changes — but a shape error in
    the dispatch would only show up at T=1, which is exactly the decode path."""
    from aksharallm.model.transformer import KVCache

    torch.manual_seed(0)
    c = cfg(n_experts=4)
    model = Transformer(c).eval()
    caches = [KVCache(1, c.n_kv_heads, c.max_seq_len, c.head_dim, torch.float32, "cpu")
              for _ in range(c.n_layers)]
    with torch.no_grad():
        logits, _ = model(torch.randint(0, 64, (1, 4)), caches=caches)
        assert logits.shape == (1, 1, 64)
        logits, _ = model(torch.randint(0, 64, (1, 1)), caches=caches)   # decode step
        assert logits.shape == (1, 1, 64)


def test_an_moe_checkpoint_round_trips(tmp_path):
    torch.manual_seed(0)
    c = cfg(n_experts=4)
    model = Transformer(c)
    torch.save({"model": model.state_dict(), "model_config": c.__dict__}, tmp_path / "m.pt")
    loaded = Transformer(ModelConfig(**{k: v for k, v in c.__dict__.items()
                                        if k in ModelConfig.__dataclass_fields__}))
    loaded.load_state_dict(torch.load(tmp_path / "m.pt")["model"], strict=True)
    idx = torch.randint(0, 64, (2, 8))
    model.eval(), loaded.eval()
    with torch.no_grad():
        assert torch.equal(model(idx, full_logits=True)[0], loaded(idx, full_logits=True)[0])


def test_routing_is_deterministic_for_the_same_input():
    torch.manual_seed(0)
    model = Transformer(cfg(n_experts=4)).eval()
    idx = torch.randint(0, 64, (2, 8))
    with torch.no_grad():
        model(idx)
        first = model.routing()["shares"]
        model(idx)
    assert model.routing()["shares"] == first


def test_dispatch_matches_a_naive_masked_implementation():
    """The sorted dispatch is an optimisation. This is the slow, obviously-correct version
    it has to agree with — the only real check that the sort, the gather and the weighted
    scatter-add line up."""
    torch.manual_seed(0)
    c = cfg(n_experts=4, moe_top_k=2)
    moe = MoEFeedForward(c).eval()
    x = torch.randn(2, 5, c.d_model)

    fast = moe(x)

    flat = x.reshape(-1, c.d_model)
    w, idx, _, _ = moe.router(flat)
    slow = torch.zeros_like(flat)
    for token in range(flat.shape[0]):
        for slot in range(c.moe_top_k):
            e = int(idx[token, slot])
            xe = flat[token]
            h = torch.nn.functional.silu(xe @ moe.w1[e]) * (xe @ moe.w3[e])
            slow[token] += w[token, slot] * (h @ moe.w2[e])
    assert torch.allclose(fast.reshape(-1, c.d_model), slow, atol=1e-5)


# ---- what MoE breaks elsewhere ----------------------------------------------------------

def test_quantizing_an_moe_model_is_refused_not_half_done():
    """It would leave every expert in float (68% of the 300M) and quantize the router, which
    must never be quantized: a wrong route is a different expert, not a smaller number."""
    from aksharallm.quant.convert import quantize_model
    from aksharallm.quant.qtensor import QuantScheme

    model = Transformer(cfg(n_experts=4))
    with pytest.raises(ValueError, match="mixture of experts"):
        quantize_model(model, QuantScheme(bits=4, group_size=16))


def test_lora_reports_the_experts_it_cannot_reach():
    """Adapting attention only is a legitimate choice; doing it silently is not."""
    from aksharallm.lora.inject import LoRAConfig, apply_lora

    model = Transformer(cfg(n_experts=4))
    report = apply_lora(model, LoRAConfig(r=4, targets="all-linear"))
    skipped = " ".join(f"{n} {why}" for n, why in report.skipped)
    assert "experts" in skipped and "router" in skipped
    assert report.adapted, "attention should still be adapted"


# ---- the final step gets a log line ------------------------------------------------------

def test_a_completed_run_logs_its_final_step(tmp_path):
    """A run of N steps used to end its log at the last multiple of `log_every` — 7,980 of
    8,000 — and read on a dashboard as though it had stopped 20 steps early. A bounded stop
    always logged its final step; a natural finish now does too.

    This drives the real trainer end to end, because the rule lives in one condition inside
    the training loop and asserting it any other way would be asserting a copy of it.
    """
    import json
    import subprocess
    import sys

    import numpy as np

    from aksharallm.tokenizer.tokenizer import train_bpe

    corpus = ["the quick brown fox jumps over the lazy dog and runs away home. "] * 200
    tok_path = tmp_path / "tok.json"
    train_bpe(iter(corpus), vocab_size=300, out_path=tok_path, min_frequency=1)

    rng = np.random.default_rng(0)
    rng.integers(0, 300, 20000, dtype=np.uint16).tofile(tmp_path / "train.bin")
    rng.integers(0, 300, 4000, dtype=np.uint16).tofile(tmp_path / "val.bin")

    cfg = tmp_path / "t.yaml"
    cfg.write_text(f"""
name: t
model:
  vocab_size: 300
  d_model: 32
  n_layers: 2
  n_heads: 4
  max_seq_len: 32
data:
  train_bin: {tmp_path}/train.bin
  val_bin: {tmp_path}/val.bin
  tokenizer: {tok_path}
train:
  out_dir: {tmp_path}/out
  batch_size: 2
  grad_accum: 1
  seq_len: 16
  max_steps: 7
  log_every: 5
  eval_every: 100
  eval_batches: 2
  ckpt_every: 100
  sample_every: 0
  compile: false
""")
    done = subprocess.run([sys.executable, "-m", "aksharallm.train.pretrain", str(cfg)],
                          capture_output=True, text=True)
    assert done.returncode == 0, done.stderr[-2000:]

    lines = (tmp_path / "out" / "train_log.jsonl").read_text().splitlines()
    steps = [json.loads(l)["step"] for l in lines if '"loss"' in l and '"val_loss"' not in l]
    # max_steps=7 means steps 0..6, and log_every=5 would otherwise stop the log at 5.
    assert steps[-1] == 6, f"the final step must be logged, got {steps}"

    ends = [json.loads(l) for l in lines if '"session_end"' in l]
    assert ends and ends[-1]["last_step"] == 6


def test_restarting_a_finished_run_does_nothing_and_says_so(tmp_path):
    """Pressing Start on a run that has spent its budget used to produce a full pre-flight
    followed by "ran 0 steps", which reads as a launch that failed."""
    import subprocess
    import sys

    import numpy as np

    from aksharallm.tokenizer.tokenizer import train_bpe

    corpus = ["the quick brown fox jumps over the lazy dog and runs away home. "] * 200
    train_bpe(iter(corpus), vocab_size=300, out_path=tmp_path / "tok.json", min_frequency=1)
    rng = np.random.default_rng(0)
    rng.integers(0, 300, 20000, dtype=np.uint16).tofile(tmp_path / "train.bin")
    rng.integers(0, 300, 4000, dtype=np.uint16).tofile(tmp_path / "val.bin")

    cfg = tmp_path / "t.yaml"
    cfg.write_text(f"""
name: t
model: {{vocab_size: 300, d_model: 32, n_layers: 2, n_heads: 4, max_seq_len: 32}}
data:
  train_bin: {tmp_path}/train.bin
  val_bin: {tmp_path}/val.bin
  tokenizer: {tmp_path}/tok.json
train:
  out_dir: {tmp_path}/out
  batch_size: 2
  grad_accum: 1
  seq_len: 16
  max_steps: 4
  log_every: 2
  eval_every: 100
  eval_batches: 2
  ckpt_every: 2
  sample_every: 0
  compile: false
  resume: auto
""")
    run = [sys.executable, "-m", "aksharallm.train.pretrain", str(cfg)]
    assert subprocess.run(run, capture_output=True, text=True).returncode == 0
    again = subprocess.run(run, capture_output=True, text=True)
    assert again.returncode == 0
    assert "nothing to do" in again.stdout
    assert "raise train.max_steps" in again.stdout
