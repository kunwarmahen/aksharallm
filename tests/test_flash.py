"""Tests for the from-scratch FlashAttention kernel.

The kernel-numerics tests skip without a GPU so the suite stays runnable on a laptop; the
routing tests (which shapes go to Triton and which fall back to SDPA) run everywhere,
because *that* is the part a change to the model would break silently.

The bar throughout is equality with `reference_attention` -- the (T, S)-matrix definition
written out in fp32. Not "close enough": FlashAttention is an exact reordering of the same
sum, so in fp32 the two agree to ~1e-6 and anything looser means a real mistake. bf16 is
tested against the same fp32 reference at bf16's own resolution.

Two of these exist because of specific ways this could be wrong while looking right:

  * `test_causal_is_bottom_right_aligned` -- the trap already documented in
    `Attention.forward`. A top-left triangle over a warm cache trains fine and generates
    fluent nonsense; nothing but an explicit test notices.
  * `test_grouped_query_gradients_are_summed_over_the_group` -- with GQA each key/value
    head is read by `n_rep` query heads, so its gradient is a sum of `n_rep` terms. Drop
    the sum and the model still trains, just with KV gradients n_rep times too small.
"""

import math

import pytest
import torch

from aksharallm.config import ModelConfig
from aksharallm.model import flash
from aksharallm.model.transformer import Transformer

triton_only = pytest.mark.skipif(
    not flash.available(), reason="needs triton on a CUDA device")


def qkv(B=2, H=4, Hk=None, T=64, S=None, D=32, dtype=torch.float32, grad=False, seed=0):
    torch.manual_seed(seed)
    Hk, S = Hk or H, S or T
    make = lambda n, t: torch.randn(  # noqa: E731
        B, n, t, D, device="cuda", dtype=dtype, requires_grad=grad)
    return make(H, T), make(Hk, S), make(Hk, S)


# ---- routing: which calls the kernel accepts, and what happens to the rest -------------

def test_usable_is_false_without_a_gpu_kernel():
    """On CPU there is no Triton path at all, and `usable` says so rather than raising."""
    q = torch.randn(1, 2, 64, 32)
    assert flash.usable(q, q) is False


@triton_only
@pytest.mark.parametrize("kwargs, why", [
    ({"dropout_p": 0.1}, "dropout inside attention is not implemented"),
    ({"attn_mask": torch.ones(1, 1)}, "only the causal mask shape is built in"),
])
def test_usable_refuses_what_the_kernel_does_not_handle(kwargs, why):
    q, k, _ = qkv()
    assert flash.usable(q, k) is True, "the plain case must be accepted"
    assert flash.usable(q, k, **kwargs) is False, why


@triton_only
def test_a_single_decode_row_falls_back():
    """T == 1 is a matrix-vector product. Tiling it is pointless and `tl.dot` cannot even
    express it, so it belongs on SDPA -- and the wrapper has to say so rather than crash."""
    q, k, _ = qkv(T=1, S=128)
    assert flash.usable(q, k) is False


@triton_only
@pytest.mark.parametrize("head_dim", [8, 48, 96])
def test_an_unsupported_head_dim_falls_back(head_dim):
    q, k, _ = qkv(D=head_dim)
    assert flash.usable(q, k) is False


# ---- numerics: the kernel against the definition ---------------------------------------

@triton_only
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("T", [64, 128, 200])  # 200 is not a multiple of any block size
def test_forward_matches_the_matrix_definition(causal, T):
    q, k, v = qkv(T=T)
    out = flash.flash_attention(q, k, v, causal=causal)
    ref = flash.reference_attention(q, k, v, causal=causal)
    assert torch.allclose(out, ref, atol=1e-5), (out - ref).abs().max().item()


@triton_only
@pytest.mark.parametrize("T", [64, 200])
def test_backward_matches_the_matrix_definition(T):
    """dQ, dK and dV, against autograd through the naive version. This is the pass that
    recomputes the scores from the saved log-sum-exp, so it also proves the forward saved
    the right thing."""
    q, k, v = qkv(T=T, grad=True)
    g = torch.randn_like(q)
    ours = torch.autograd.grad(flash.flash_attention(q, k, v), (q, k, v), g)
    ref_in = [x.detach().clone().requires_grad_(True) for x in (q, k, v)]
    ref = torch.autograd.grad(flash.reference_attention(*ref_in), ref_in, g)
    for name, a, b in zip("qkv", ours, ref):
        assert torch.allclose(a, b, atol=1e-4), f"d{name}: {(a - b).abs().max().item()}"


@triton_only
def test_bf16_matches_the_fp32_definition_at_bf16_resolution():
    """The dtype a run actually uses. Compared against the *fp32* reference, because
    comparing two bf16 computations hides an error the size of bf16's own resolution."""
    q, k, v = qkv(T=256, D=64, dtype=torch.bfloat16)
    out = flash.flash_attention(q, k, v).float()
    ref = flash.reference_attention(q.float(), k.float(), v.float())
    assert (out - ref).abs().max().item() < 0.05


@triton_only
@pytest.mark.parametrize("Hk", [1, 2, 8])
def test_grouped_query_attention(Hk):
    q, k, v = qkv(H=8, Hk=Hk, T=128)
    ref = flash.reference_attention(q, k, v)
    assert torch.allclose(flash.flash_attention(q, k, v), ref, atol=1e-5)


@triton_only
def test_grouped_query_gradients_are_summed_over_the_group():
    """Four query heads read one KV head, so dK for that head is the sum of four terms.
    Comparing against MHA with the key/value tensors repeated by hand is the check: if the
    kernel wrote one group member's gradient instead of the sum, this is 4x too small and
    the model would still train, slightly wrong, forever."""
    q, k, v = qkv(H=8, Hk=2, T=128, grad=True)
    g = torch.randn_like(q)
    dq, dk, dv = torch.autograd.grad(flash.flash_attention(q, k, v), (q, k, v), g)

    qe = q.detach().clone().requires_grad_(True)
    ke, ve = (x.detach().repeat_interleave(4, dim=1).requires_grad_(True) for x in (k, v))
    edq, edk, edv = torch.autograd.grad(flash.reference_attention(qe, ke, ve), (qe, ke, ve), g)
    assert torch.allclose(dq, edq, atol=1e-4)
    for ours, expanded in ((dk, edk), (dv, edv)):
        assert torch.allclose(ours, expanded.view(2, 2, 4, 128, 32).sum(2), atol=1e-4)


@triton_only
@pytest.mark.parametrize("T", [16, 64])
def test_causal_is_bottom_right_aligned(T):
    """Queries sit at the END of the keys. Query j of T may see key j + (S - T), i.e. the
    whole cached prefix plus the part of this block before it -- NOT keys 0..j.

    The distinction only exists when T != S, which is exactly the speculative-decoding
    shape, and getting it wrong hides most of the prompt from every query."""
    S = 256
    q, k, v = qkv(T=T, S=S)
    out = flash.flash_attention(q, k, v, causal=True)

    # The same T queries, scored one at a time against everything up to their own position.
    for j in (0, T // 2, T - 1):
        end = S - T + j + 1
        one = flash.reference_attention(q[:, :, j:j + 1], k[:, :, :end], v[:, :, :end],
                                        causal=True)
        assert torch.allclose(out[:, :, j:j + 1], one, atol=1e-5), f"query {j}"


@triton_only
def test_a_masked_out_tail_does_not_poison_the_row():
    """The last block of queries is partly past the end of the sequence and the last block
    of keys is partly past the end of the cache. Both are handled by filling with -inf,
    which is one `exp(-inf) - exp(-inf)` away from nan. A length that is 1 past a block
    boundary is where that shows up."""
    q, k, v = qkv(T=65, D=64)
    out = flash.flash_attention(q, k, v)
    assert torch.isfinite(out).all()
    assert torch.allclose(out, flash.reference_attention(q, k, v), atol=1e-5)


@triton_only
def test_the_scale_is_honoured():
    q, k, v = qkv(T=64)
    scale = 0.37
    ref = flash.reference_attention(q, k, v, sm_scale=scale)
    assert torch.allclose(flash.flash_attention(q, k, v, sm_scale=scale), ref, atol=1e-5)
    # ...and the default really is 1/sqrt(D), not something that happens to be close.
    default = flash.flash_attention(q, k, v)
    expect = flash.reference_attention(q, k, v, sm_scale=1 / math.sqrt(q.shape[-1]))
    assert torch.allclose(default, expect, atol=1e-5)


# ---- integration with the model ---------------------------------------------------------

def test_config_rejects_an_unknown_impl():
    with pytest.raises(ValueError, match="attn_impl"):
        ModelConfig(attn_impl="flashattention3")


def test_sdpa_is_still_the_default():
    """Changing this default changes every training run in the repo. It should take a
    deliberate edit and a failing test, not a drive-by."""
    assert ModelConfig().attn_impl == "sdpa"


def test_the_model_accepts_the_flash_impl_on_any_device():
    """Building a `flash` model on a CPU-only box must work -- every layer routes back to
    SDPA through `usable`. Otherwise the config is un-testable anywhere but the 3090."""
    cfg = ModelConfig(vocab_size=64, d_model=64, n_layers=2, n_heads=4, max_seq_len=32,
                      attn_impl="flash")
    m = Transformer(cfg).eval()
    idx = torch.randint(0, 64, (2, 16))
    with torch.no_grad():
        logits, _ = m(idx, full_logits=True)
    assert logits.shape == (2, 16, 64)


@triton_only
def test_the_model_gives_the_same_answer_either_way():
    """The one that matters: same weights, same input, two kernels, one answer. In fp32,
    because in bf16 a difference of the wrong kind hides under the rounding."""
    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=128, d_model=128, n_layers=2, n_heads=4, n_kv_heads=2,
                      max_seq_len=128)
    m = Transformer(cfg).cuda().eval()
    idx = torch.randint(0, 128, (2, 96), device="cuda")
    with torch.no_grad():
        sdpa, _ = m(idx, full_logits=True)
        for block in m.blocks:
            block.attn.attn_impl = "flash"
        ours, _ = m(idx, full_logits=True)
    assert torch.allclose(sdpa, ours, atol=2e-4), (sdpa - ours).abs().max().item()


@triton_only
def test_a_training_step_produces_the_same_gradients():
    """Forward equality is not enough: a wrong backward trains a slightly wrong model and
    the loss curve looks completely normal."""
    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=128, d_model=128, n_layers=2, n_heads=4, n_kv_heads=2,
                      max_seq_len=128)
    m = Transformer(cfg).cuda()
    idx = torch.randint(0, 128, (2, 96), device="cuda")
    tgt = torch.randint(0, 128, (2, 96), device="cuda")

    def grads():
        m.zero_grad(set_to_none=True)
        m(idx, targets=tgt)[1].backward()
        return {n: p.grad.detach().clone() for n, p in m.named_parameters()}

    ref = grads()
    for block in m.blocks:
        block.attn.attn_impl = "flash"
    for name, g in grads().items():
        assert torch.allclose(g, ref[name], atol=2e-4), name
