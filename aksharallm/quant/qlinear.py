"""QuantLinear: a drop-in replacement for `nn.Linear(bias=False)` that stores bytes.

Our transformer's every matmul is a bias-free Linear -- `wq/wk/wv/wo` in attention,
`w1/w2/w3` in the SwiGLU, and `lm_head`. So the entire quantization story at the model
level is "swap those modules out", and this is the module they are swapped for.

What it holds
-------------
    qweight   uint8/int8, (out_features, in_features)   -- or in_features/2 at 4 bits
    scales    fp16,       (out_features, n_groups)
    qzeros    uint8,      (out_features, n_groups)      -- asymmetric only

and nothing else. There is no float copy of the weight anywhere; that is the point.

How forward works
-----------------
Two paths, chosen by `QuantLinear.backend`:

  "torch"  -- unpack + dequantize the whole weight to bf16, then `F.linear`. Simple,
              runs anywhere, and correct. But it *materialises the full weight on every
              call*, so it saves memory only where memory is spent at rest (the weights
              in the checkpoint and in VRAM between calls), not at peak during a matmul.
              It is also slower than plain bf16, because it does everything bf16 does
              plus the unpacking.

  "triton" -- a fused kernel that dequantizes inside the matmul, so the full weight is
              never built. This is where the memory bandwidth saving turns into actual
              speed. See kernels.py.

The honest framing, which the docs repeat: weight-only quantization is a *memory* win
first. It becomes a *speed* win only once the dequantization is fused into the matmul,
because single-token decoding is bandwidth-bound -- the GPU spends its time reading
weights, so quartering the bytes read is the whole game.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .qtensor import (
    QuantScheme,
    dequantize,
    pack,
    packed_shape,
    quantize_group,
    resolve_group_size,
    unpack,
)


class QuantLinear(nn.Module):
    """Quantized `y = x @ W.T`, with W stored in `scheme.bits` bits."""

    # Class-level so a whole model switches at once: "torch" | "triton" | "auto".
    backend: str = "auto"

    def __init__(
        self,
        in_features: int,
        out_features: int,
        scheme: QuantScheme,
        group_size: int | None = None,
        device=None,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scheme = scheme
        # The *effective* group size for this layer, which may be coarser than requested
        # if the requested one did not divide in_features (see resolve_group_size).
        self.group_size = (
            resolve_group_size(scheme.group_size, in_features)
            if group_size is None
            else group_size
        )
        g = in_features if self.group_size == -1 else self.group_size
        n_groups = in_features // g
        self.n_groups = n_groups
        self.out_dtype = dtype

        qw_shape = packed_shape(out_features, in_features, scheme)
        qw_dtype = torch.uint8 if (scheme.packed or not scheme.sym) else torch.int8
        # Buffers, not Parameters: these carry no gradient and must not be picked up by
        # the optimiser or weight decay if a quantized model is ever fine-tuned.
        self.register_buffer("qweight", torch.zeros(qw_shape, dtype=qw_dtype, device=device))
        self.register_buffer(
            "scales", torch.zeros((out_features, n_groups), dtype=torch.float16, device=device)
        )
        if scheme.sym:
            self.register_buffer("qzeros", None)
        else:
            self.register_buffer(
                "qzeros", torch.zeros((out_features, n_groups), dtype=torch.uint8, device=device)
            )

    # ---- construction ------------------------------------------------------------

    @classmethod
    def from_linear(
        cls,
        lin: nn.Linear,
        scheme: QuantScheme,
        qweight: torch.Tensor | None = None,
        scales: torch.Tensor | None = None,
        zeros: torch.Tensor | None = None,
    ) -> "QuantLinear":
        """Build from a float Linear.

        With no `qweight`/`scales` supplied it quantizes round-to-nearest. GPTQ and AWQ
        do their own (cleverer) search and hand the results in, which is why those
        arguments exist -- the storage format is shared by every method.
        """
        if lin.bias is not None:
            raise ValueError("QuantLinear supports bias=False layers only (ours all are)")
        out_f, in_f = lin.weight.shape
        q = cls(in_f, out_f, scheme, device=lin.weight.device, dtype=lin.weight.dtype)
        if qweight is None:
            codes, scales, zeros = quantize_group(
                lin.weight.data, scheme, group_size=q.group_size
            )
            qweight = pack(codes, scheme)
        q.load_quantized(qweight, scales, zeros)
        return q

    def load_quantized(self, qweight, scales, zeros):
        self.qweight.copy_(qweight.to(self.qweight.dtype))
        self.scales.copy_(scales.to(self.scales.dtype))
        if self.qzeros is not None:
            if zeros is None:
                raise ValueError("asymmetric scheme needs zero-points")
            self.qzeros.copy_(zeros.to(self.qzeros.dtype))

    # ---- use ---------------------------------------------------------------------

    def dequantize_weight(self, dtype: torch.dtype | None = None) -> torch.Tensor:
        """Rebuild the (out_features, in_features) float weight. Used by the torch
        backend, by the tests, and by anything that wants to measure the error."""
        codes = unpack(self.qweight, self.scheme)
        zeros = None if self.qzeros is None else self.qzeros.to(torch.int16)
        g = self.in_features if self.group_size == -1 else self.group_size
        return dequantize(codes, self.scales, zeros, g, dtype=dtype or self.out_dtype)

    def _use_triton(self, x: torch.Tensor) -> bool:
        if self.backend == "torch":
            return False
        from . import kernels

        if not kernels.available(x.device):
            return False
        if self.backend == "triton":
            return True
        # "auto": the fused kernel is a decode-time win (few rows, weight-bandwidth
        # bound). For a big prefill matmul, dequantizing once and letting cuBLAS have
        # the whole tile is faster, so fall back to torch there.
        rows = x.numel() // x.shape[-1]
        return rows <= kernels.AUTO_MAX_ROWS

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._use_triton(x):
            from . import kernels

            return kernels.qlinear_forward(x, self)
        return F.linear(x, self.dequantize_weight(x.dtype))

    # ---- reporting ---------------------------------------------------------------

    def nbytes(self) -> int:
        n = self.qweight.numel() * self.qweight.element_size()
        n += self.scales.numel() * self.scales.element_size()
        if self.qzeros is not None:
            n += self.qzeros.numel() * self.qzeros.element_size()
        return n

    def float_nbytes(self) -> int:
        """What the same weight would cost in bf16 -- the denominator of the win."""
        return self.in_features * self.out_features * 2

    def extra_repr(self) -> str:
        g = "chan" if self.group_size == -1 else self.group_size
        return (
            f"in={self.in_features}, out={self.out_features}, bits={self.scheme.bits}, "
            f"group={g}, {'sym' if self.scheme.sym else 'asym'}, "
            f"{self.nbytes() / self.float_nbytes():.2f}x of bf16"
        )
