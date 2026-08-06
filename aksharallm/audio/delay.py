"""The delay pattern — how eight codebooks become one autoregressive stream.

The codec hands the language model **eight integers per frame**, not one, and they are not
independent: codebook 2 quantizes the error codebook 1 left behind, so predicting them as if
they were unrelated throws away most of what the residual structure was for. Three ways to
handle that, and the third is the one worth building.

```
frame:            0    1    2    3
                ┌────┬────┬────┬────┐
flatten     →   │0₀1₀2₀3₀│0₁1₁2₁3₁│…  8x the positions. Correct, and 10 s of audio
                └────┴────┴────┴────┘  becomes 4,000 tokens instead of 500.

parallel    →   predict all 8 of frame t at once from frames < t.
                Fast, and WRONG: it assumes the eight are independent given the past,
                and the whole point of a residual is that they are not.

delay       →   codebook k is shifted right by k frames:
                book 0:  c⁰₀  c⁰₁  c⁰₂  c⁰₃
                book 1:   ·   c¹₀  c¹₁  c¹₂
                book 2:   ·    ·   c²₀  c²₁
                book 3:   ·    ·    ·   c³₀
                Now everything in one column can be predicted at once, because by the
                time c¹₀ is predicted, c⁰₀ is already in the past and visible.
```

That is the whole trick (MusicGen, Copet et al. 2023). The sequence grows from `T` to
`T + N − 1` — eight extra positions on five hundred — instead of to `8T`, and the dependency
chain that makes residual codes meaningful is preserved rather than assumed away.

**Trap 4 of the phase lives here.** An off-by-one in either direction is gotcha #2's family:
it trains perfectly well and generates garbage, because the decoder is handed codebook 1 of
frame 5 alongside codebook 0 of frame 4 and reconstructs an interleaving of two different
moments. The defence is that `undelay(delay(x)) == x` is asserted exactly, and that the
padding is a distinct token id rather than a zero that could be mistaken for code 0.

Read with: docs/20-audio.md -- the chapter this implements; it ends with the order to read
these files in.
"""

from __future__ import annotations

import torch


def special_ids(codebook_size: int) -> tuple[int, int]:
    """`(pad_id, bos_id)` — the two ids past the end of a codebook's real range.

    A distinct id, never 0. Zero is a perfectly good code that the codec emits constantly,
    so filling the delay triangle with zeros would train the model to predict silence-shaped
    nonsense at the edges and give the loss nothing to ignore.
    """
    return codebook_size, codebook_size + 1


def delay(codes: torch.Tensor, pad_id: int) -> torch.Tensor:
    """`(B, N, T)` -> `(B, N, T + N − 1)`, with codebook k shifted right by k frames.

    Position `s` of codebook `k` holds original frame `s − k`; everything outside `0 ≤ s − k
    < T` is `pad_id`. The first `N − 1` columns and the last `N − 1` are therefore partly
    padding — the price of the pattern, and it is `N − 1` frames of a sequence hundreds long.
    """
    b, n, t = codes.shape
    out = codes.new_full((b, n, t + n - 1), pad_id)
    for k in range(n):
        out[:, k, k : k + t] = codes[:, k]
    return out


def undelay(delayed: torch.Tensor, n_frames: int | None = None) -> torch.Tensor:
    """The exact inverse of `delay`. `(B, N, T + N − 1)` -> `(B, N, T)`.

    `n_frames` defaults to the `T` that `delay` would have produced this width from. Passing
    it explicitly matters during generation, where the tail is still being filled in and the
    caller knows how many frames are actually complete.
    """
    b, n, s = delayed.shape
    t = s - n + 1 if n_frames is None else n_frames
    out = delayed.new_empty((b, n, t))
    for k in range(n):
        out[:, k] = delayed[:, k, k : k + t]
    return out


def valid_mask(n_codebooks: int, n_frames: int, device=None) -> torch.Tensor:
    """`(N, T + N − 1)` — True where a delayed position holds a real code.

    This is what becomes `-100` in the targets. Training on the padding would teach the model
    to predict a token that carries no information, at the two places where it has the least
    context to do it from.
    """
    s = n_frames + n_codebooks - 1
    k = torch.arange(n_codebooks, device=device).unsqueeze(1)
    pos = torch.arange(s, device=device).unsqueeze(0)
    return (pos >= k) & (pos < k + n_frames)
