"""G3's corpus: every picture as the tokenizer's continuous latent, normalised.

G3 (docs/26 § G3) is G2's flow matching moved from pixels into G0's latent space: instead of a
64×64×3 picture (12,288 numbers, 256 patch tokens), the flow model works on an 8×8×64 grid
(4,096 numbers, 64 tokens), and the tokenizer's decoder turns the result into pixels. Every real
image model (Stable Diffusion, Flux, Veo) works this way, because almost all of a picture's
pixels are predictable from their neighbours, and the decoder can supply them.

**Which latent.** The encoder's output `z` is not what the decoder was trained on; the decoder
only ever saw `q`, the sum of the chosen codebook vectors. So the corpus is `q` at all of the
tokenizer's codebooks: continuous numbers the decoder knows how to paint. A drawn latent can be
decoded as it is, or **snapped** back onto the codebooks first (`snap`), which removes small
errors the decoder has never seen; both are measured.

**Normalised per channel**, as Stable Diffusion scales its latents: flow matching starts from
unit-variance noise, so the data it flows to should have unit variance too, or the straight
line from noise to data is mostly a change of scale. The mean and standard deviation of every
channel over the training pictures are saved beside the latents and inside every G3 checkpoint,
because a latent drawn in normalised units decodes to nonsense if it is not mapped back.

    latents.npy   (N, 64, 8, 8) float16, normalised
    meta.json     tokenizer, its step, books, the per-channel mean and std, the corpus

Read with: docs/26-pictures.md -- the chapter this implements; it ends with the order to read
these files in.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from ..vision.image import ImageCaptions, ImageManifest, to_tensor


@torch.no_grad()
def picture_latent(tokenizer, x: torch.Tensor, books: int | None = None) -> torch.Tensor:
    """Pictures `(B, 3, H, W)` in [-1, 1] -> the quantized latent `q` `(B, D, h, w)`."""
    q, _, _, _ = tokenizer._quantize(tokenizer.encoder(x), books or tokenizer.cfg.n_codebooks)
    return q


@torch.no_grad()
def decode_latent(tokenizer, q: torch.Tensor, *, snap: bool = False,
                  books: int | None = None) -> torch.Tensor:
    """A latent `(B, D, h, w)` -> pictures `(B, 3, H, W)`. `snap` re-quantizes it first, so the
    decoder sees only sums of codebook vectors, as it did in training."""
    if snap:
        q, _, _, _ = tokenizer._quantize(q, books or tokenizer.cfg.n_codebooks)
    return tokenizer.decoder(q)


def normalise(q, mean, std):
    return (q - mean) / std


def denormalise(z, mean, std):
    return z * std + mean


def stats_tensors(mean: list[float], std: list[float], device="cpu"):
    m = torch.tensor(mean, dtype=torch.float32, device=device)[None, :, None, None]
    s = torch.tensor(std, dtype=torch.float32, device=device)[None, :, None, None]
    return m, s


@torch.no_grad()
def encode_latents(tokenizer_path: str | Path, corpus: str | Path, out_dir: str | Path, *,
                   books: int | None = None, device: str = "cpu", batch: int = 256,
                   val_frac: float = 0.05, progress=print) -> dict:
    """Encode every picture of `corpus` to its normalised latent. Statistics come from the
    TRAINING split only (the last `val_frac` are val, as everywhere else in this package)."""
    from .train import load_tokenizer
    model, blob = load_tokenizer(tokenizer_path, device)
    books = books or model.cfg.n_codebooks
    ds = ImageCaptions(corpus, split="val", val_frac=1.0)           # every picture, in order
    n = ds.manifest.n_images
    parts = []
    for i in range(0, n, batch):
        x = torch.from_numpy(to_tensor(np.asarray(ds.images[i:i + batch]))).to(device)
        parts.append(picture_latent(model, x, books).float().cpu())
        if progress and (i // batch) % 40 == 0:
            progress(f"  {min(i + batch, n):,}/{n:,} pictures")
    q = torch.cat(parts)                                              # (N, D, h, w)
    n_train = n - max(1, int(n * val_frac))
    mean = q[:n_train].mean(dim=(0, 2, 3))
    std = q[:n_train].std(dim=(0, 2, 3)).clamp_min(1e-4)
    z = normalise(q, mean[None, :, None, None], std[None, :, None, None])
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "latents.npy", z.numpy().astype(np.float16))
    meta = {"tokenizer": str(tokenizer_path), "tokenizer_step": int(blob.get("step", -1)),
            "books": books, "shape": list(q.shape[1:]), "mean": mean.tolist(),
            "std": std.tolist(), "corpus": str(corpus), "n": n,
            "built": time.strftime("%Y-%m-%d %H:%M:%S")}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    if progress:
        progress(f"{n:,} latents {tuple(q.shape[1:])} -> {out_dir}/latents.npy "
                 f"({(out_dir / 'latents.npy').stat().st_size / 1e6:.0f} MB; "
                 f"{q[0].numel():,} numbers a picture against {3 * 64 * 64:,} pixels)")
    return meta


class LatentDataset:
    """Normalised latents and their captions, split by index as `ImageCaptions` splits the
    pictures, so G3's val pictures are G2's val pictures."""

    def __init__(self, path: str | Path, *, split: str = "train", val_frac: float = 0.05,
                 seed: int | None = None):
        d = Path(path)
        self.meta = json.loads((d / "meta.json").read_text())
        self.z = np.load(d / "latents.npy", mmap_mode="r")
        man = ImageManifest.load(Path(self.meta["corpus"]) / "manifest.json")
        if len(man.captions) != len(self.z):
            raise ValueError(f"{d}: {len(self.z)} latents for {len(man.captions)} captions; "
                             "re-run `pictures latents`")
        self.manifest = man
        n_val = max(1, int(len(self.z) * val_frac))
        self.index = (list(range(len(self.z) - n_val, len(self.z))) if split == "val"
                      else list(range(len(self.z) - n_val)))
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.index)

    def rows(self, idx) -> tuple[np.ndarray, list[str]]:
        idx = [int(i) for i in idx]
        return (np.asarray(self.z[idx], dtype=np.float32),
                [self.manifest.captions[int(i)] for i in idx])

    def batch(self, batch_size: int) -> tuple[np.ndarray, list[str]]:
        picks = self.rng.choice(self.index, size=batch_size)
        return (np.asarray(self.z[picks], dtype=np.float32),
                [self.manifest.captions[int(i)] for i in picks])
