"""Tests for the fused Triton kernel.

Every test here skips without a GPU, so the suite stays runnable on a laptop.

The kernel's whole job is to produce the same numbers as the torch path while reading a
quarter of the bytes. So that is what is tested: equality with the reference, across every
scheme and every awkward shape, and specifically the shapes where an indexing mistake
would still produce plausible-looking output.
"""

import pytest
import torch
import torch.nn as nn

from aksharallm.quant import kernels
from aksharallm.quant.qlinear import QuantLinear
from aksharallm.quant.qtensor import QuantScheme

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
triton_only = pytest.mark.skipif(
    not (torch.cuda.is_available() and kernels.available(torch.device("cuda"))),
    reason="needs triton on a CUDA device")

SCHEMES = [
    QuantScheme(bits=4, group_size=64, sym=False),
    QuantScheme(bits=4, group_size=128, sym=False),
    QuantScheme(bits=4, group_size=64, sym=True),
    QuantScheme(bits=4, group_size=-1, sym=False),
    QuantScheme(bits=8, group_size=64, sym=True),
    QuantScheme(bits=8, group_size=64, sym=False),
]


@pytest.fixture(autouse=True)
def restore_backend():
    old = QuantLinear.backend
    yield
    QuantLinear.backend = old


def _compare(scheme, K, N, M, tol=0.02):
    torch.manual_seed(0)
    lin = nn.Linear(K, N, bias=False).cuda().to(torch.bfloat16)
    q = QuantLinear.from_linear(lin, scheme)
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    QuantLinear.backend = "torch"
    ref = q(x)
    QuantLinear.backend = "triton"
    out = q(x)
    rel = (out.float() - ref.float()).abs().max().item() / ref.float().abs().max().item()
    assert rel < tol, f"{scheme.label()} K={K} N={N} M={M}: relative error {rel:.4f}"


@triton_only
@pytest.mark.parametrize("scheme", SCHEMES, ids=lambda s: s.label())
def test_fused_matches_the_torch_path(scheme):
    _compare(scheme, K=512, N=256, M=1)


@triton_only
@pytest.mark.parametrize("M", [1, 2, 4, 5, 16, 33])
def test_every_row_of_the_output_is_written(M):
    """The kernel caps BLOCK_M and grids over M. Get that wrong and rows past BLOCK_M are
    never stored -- they come back as whatever `torch.empty` had, which for a small M is
    often plausible-looking garbage rather than an obvious NaN."""
    _compare(QuantScheme(bits=4, group_size=64, sym=False), K=512, N=256, M=M)


@triton_only
@pytest.mark.parametrize("K", [128, 512, 1024, 2752])
def test_reduction_lengths_including_the_awkward_one(K):
    """2752 is d_ff on the real 300M config: not a power of two, and not divisible by 128.
    The K loop has to mask its tail correctly."""
    _compare(QuantScheme(bits=4, group_size=64, sym=False), K=K, N=256, M=1)


@triton_only
@pytest.mark.parametrize("N", [16, 32, 64, 100, 1024])
def test_output_widths_including_a_partial_tile(N):
    """N=100 leaves a partial BLOCK_N tile, which must be masked rather than written."""
    _compare(QuantScheme(bits=4, group_size=64, sym=False), K=512, N=N, M=1)


@triton_only
def test_per_channel_takes_the_hoisted_path():
    """group_size=-1 means one scale per row, loaded once outside the K loop. It is a
    separate code path, so it needs its own coverage."""
    _compare(QuantScheme(bits=4, group_size=-1, sym=False), K=2752, N=512, M=1)


@triton_only
def test_symmetric_four_bit_undoes_the_pack_shift():
    """Symmetric 4-bit codes are stored shifted by +8 so the nibble is unsigned. The
    kernel subtracts it back. Forget that and every weight is off by 8 scale steps --
    the model still runs and produces fluent nonsense."""
    _compare(QuantScheme(bits=4, group_size=64, sym=True), K=512, N=256, M=1)


@triton_only
def test_3d_input_keeps_its_shape():
    """Real calls are (batch, time, features), not 2-D."""
    torch.manual_seed(1)
    scheme = QuantScheme(bits=4, group_size=64, sym=False)
    lin = nn.Linear(256, 128, bias=False).cuda().to(torch.bfloat16)
    q = QuantLinear.from_linear(lin, scheme)
    x = torch.randn(2, 3, 256, device="cuda", dtype=torch.bfloat16)
    QuantLinear.backend = "torch"
    ref = q(x)
    QuantLinear.backend = "triton"
    out = q(x)
    assert out.shape == ref.shape == (2, 3, 128)
    assert (out.float() - ref.float()).abs().max() / ref.float().abs().max() < 0.02


@triton_only
def test_auto_sends_decode_to_triton_and_prefill_to_torch():
    """The `auto` policy is the whole reason both paths exist: the fused kernel wins at
    one row and loses badly on a big prefill matmul, where cuBLAS on a dequantized weight
    is far better."""
    scheme = QuantScheme(bits=4, group_size=64, sym=False)
    lin = nn.Linear(256, 128, bias=False).cuda().to(torch.bfloat16)
    q = QuantLinear.from_linear(lin, scheme)
    QuantLinear.backend = "auto"
    assert q._use_triton(torch.randn(1, 256, device="cuda"))
    assert not q._use_triton(torch.randn(512, 256, device="cuda"))


@cuda
def test_torch_backend_never_uses_triton():
    scheme = QuantScheme(bits=4, group_size=64, sym=False)
    lin = nn.Linear(256, 128, bias=False).cuda().to(torch.bfloat16)
    q = QuantLinear.from_linear(lin, scheme)
    QuantLinear.backend = "torch"
    assert not q._use_triton(torch.randn(1, 256, device="cuda"))


def test_cpu_falls_back_without_triton():
    """A CPU tensor must take the torch path whatever the backend is set to, rather than
    trying to launch a CUDA kernel."""
    scheme = QuantScheme(bits=4, group_size=64, sym=False)
    q = QuantLinear.from_linear(nn.Linear(256, 128, bias=False), scheme)
    QuantLinear.backend = "triton"
    x = torch.randn(2, 256)
    assert not q._use_triton(x)
    assert torch.isfinite(q(x)).all()


def test_available_is_false_on_cpu():
    assert not kernels.available(torch.device("cpu"))
