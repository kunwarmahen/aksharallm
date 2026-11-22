"""The shapes corpus as codes: what G1 trains on, and the vocabulary that joins words to codes.

G1 (docs/26 § "From a sentence to a picture") is a transformer that reads a caption and
writes a picture's codes. It never sees a pixel: every picture is encoded once, here, by the
trained tokenizer (G0), and the generated codes go back through the same tokenizer's decoder
to be judged. So the corpus for G1 is a small array of integers plus the captions it already
had, written to `data/pictures/<name>/`:

    codes.npy    (N, books, grid*grid) int16, codebook-major as the tokenizer returns them
    meta.json    the tokenizer checkpoint and its step, books, grid, codebook size, corpus

**One vocabulary for words and codes.** A sequence is

    [three] [red] [circles] [IMG] c(0,0) c(0,1) c(1,0) c(1,1) ... c(63,1)

the three caption words, a start-of-picture marker, then the codes cell by cell in raster
order with each cell's codebooks together (position-major). A code from codebook `b` is token
`CODE0 + b * K + c`, so the two codebooks never share an id and the model always knows which
one a slot holds. The word list is built from the corpus's own constants
(`vision.image.NUMBERS`, `COLOURS`, `SHAPES`) rather than from the captions it happens to
contain, so "purple" and "triangles" are words even though no picture shows both: that is the
held-out combination, and a vocabulary read off the training captions would still contain
both words, but only by luck.

`NULL` is the empty caption. Classifier-free guidance (gen.py) needs the same model to draw
with no caption at all, so a share of training examples have their words replaced by it.

Read with: docs/26-pictures.md -- the chapter this implements; it ends with the order to read
these files in.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from ..vision.image import COLOURS, NUMBERS, SHAPES, ImageCaptions, ImageManifest, to_tensor

PAD, NULL, IMG = 0, 1, 2
WORDS: tuple[str, ...] = (tuple(NUMBERS) + tuple(COLOURS)
                          + tuple(w for s in SHAPES for w in (s, s + "s")))
WORD0 = 3
CODE0 = WORD0 + len(WORDS)
CAPTION_LEN = 3            # "<count> <colour> <shape(s)>", always three words


def word_ids(caption: str) -> list[int]:
    """`"two red circles"` -> three token ids. An unknown word is an error that lists the
    words there are, because the portal's text box sends whatever was typed."""
    words = caption.lower().split()
    bad = [w for w in words if w not in WORDS]
    if bad:
        raise ValueError(f"unknown word(s) {bad}; this model knows only: {' '.join(WORDS)}")
    if len(words) != CAPTION_LEN:
        raise ValueError(f"a caption is {CAPTION_LEN} words (count colour shape), "
                         f"e.g. 'two red circles'; got {len(words)}")
    return [WORD0 + WORDS.index(w) for w in words]


def fact_of(caption: str) -> dict:
    """The `{count, colour, shape}` a caption asks for, which is what the reader checks."""
    n, colour, shape = caption.lower().split()
    return {"count": NUMBERS.index(n) + 1, "colour": colour, "shape": shape.rstrip("s")}


def all_captions() -> list[str]:
    """Every caption the vocabulary can say: 4 counts x 5 colours x 3 shapes = 60."""
    from ..vision.image import caption_of
    return [caption_of({"count": n, "colour": c, "shape": s})
            for s in SHAPES for c in COLOURS for n in range(1, len(NUMBERS) + 1)]


def vocab_size(books: int, codebook_size: int) -> int:
    return CODE0 + books * codebook_size


def flatten(codes: np.ndarray | torch.Tensor, books: int, codebook_size: int):
    """`(B, N, P)` codebook-major codes -> `(B, P * books)` token ids, position-major."""
    c = codes[:, :books]                                   # (B, books, P)
    offs = (np.arange(books) * codebook_size)[None, :, None]
    if isinstance(c, torch.Tensor):
        offs = torch.as_tensor(offs, device=c.device)
    t = c + offs + CODE0
    t = t.transpose(1, 2) if isinstance(t, torch.Tensor) else t.transpose(0, 2, 1)
    return t.reshape(t.shape[0], -1)


def unflatten(tokens: torch.Tensor, books: int, codebook_size: int, grid: int) -> torch.Tensor:
    """`(B, P * books)` token ids -> `(B, books, grid, grid)` codes for the tokenizer."""
    B = tokens.shape[0]
    t = tokens.reshape(B, grid * grid, books).transpose(1, 2)
    offs = (torch.arange(books, device=t.device) * codebook_size)[None, :, None]
    return (t - CODE0 - offs).reshape(B, books, grid, grid)


@torch.no_grad()
def encode_corpus(tokenizer_path: str | Path, corpus: str | Path, out_dir: str | Path, *,
                  books: int | None = None, device: str = "cpu", batch: int = 256,
                  progress=print) -> dict:
    """Encode every picture of `corpus` with a trained tokenizer. Seconds on the CPU."""
    from .train import load_tokenizer
    model, blob = load_tokenizer(tokenizer_path, device)
    pc = model.cfg
    books = books or pc.n_codebooks
    if not 1 <= books <= pc.n_codebooks:
        raise ValueError(f"books {books}: the tokenizer has {pc.n_codebooks} codebooks")
    ds = ImageCaptions(corpus, split="val", val_frac=1.0)          # every picture, in order
    n = ds.manifest.n_images
    out = np.zeros((n, books, pc.grid * pc.grid), dtype=np.int16)
    for i in range(0, n, batch):
        x = torch.from_numpy(to_tensor(np.asarray(ds.images[i:i + batch]))).to(device)
        idx = model.encode(x, books)[:, :books]
        out[i:i + len(x)] = idx.reshape(len(x), books, -1).cpu().numpy()
        if progress and (i // batch) % 10 == 0:
            progress(f"  {min(i + batch, n):,}/{n:,} pictures")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "codes.npy", out)
    meta = {"tokenizer": str(tokenizer_path), "tokenizer_step": int(blob.get("step", -1)),
            "books": books, "grid": pc.grid, "codebook_size": pc.codebook_size,
            "corpus": str(corpus), "n": n, "built": time.strftime("%Y-%m-%d %H:%M:%S")}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    if progress:
        progress(f"{n:,} pictures -> {out_dir}/codes.npy ({books} codebooks x "
                 f"{pc.grid}x{pc.grid} = {books * pc.grid ** 2} codes each)")
    return meta


class CodesDataset:
    """Batches of `(input ids, targets)` for G1, split by index exactly as `ImageCaptions`
    splits the pictures (the last `val_frac` are val), so G1's val captions are pictures the
    tokenizer also never trained on."""

    def __init__(self, path: str | Path, *, books: int, split: str = "train",
                 val_frac: float = 0.05, seed: int | None = None):
        d = Path(path)
        self.meta = json.loads((d / "meta.json").read_text())
        if books > self.meta["books"]:
            raise ValueError(f"{d} holds {self.meta['books']} codebooks, config asks for {books}")
        codes = np.load(d / "codes.npy")
        man = ImageManifest.load(Path(self.meta["corpus"]) / "manifest.json")
        if len(man.captions) != len(codes):
            raise ValueError(f"{d}: {len(codes)} codes for {len(man.captions)} captions; "
                             "re-run `pictures codes`")
        self.books, self.K, self.grid = books, self.meta["codebook_size"], self.meta["grid"]
        self.captions = man.captions
        self.tokens = flatten(codes.astype(np.int64), books, self.K)   # (N, P * books)
        self.words = np.asarray([word_ids(c) for c in man.captions], dtype=np.int64)
        n_val = max(1, int(len(codes) * val_frac))
        self.index = (np.arange(len(codes) - n_val, len(codes)) if split == "val"
                      else np.arange(len(codes) - n_val))
        self.rng = np.random.default_rng(seed)

    @property
    def seq_len(self) -> int:
        return CAPTION_LEN + 1 + self.tokens.shape[1]

    def __len__(self) -> int:
        return len(self.index)

    def rows(self, picks: np.ndarray, drop: float = 0.0) -> tuple[torch.Tensor, torch.Tensor]:
        words = self.words[picks].copy()
        if drop > 0:
            words[self.rng.random(len(picks)) < drop] = NULL      # the unconditional rows
        seq = np.concatenate([words, np.full((len(picks), 1), IMG), self.tokens[picks]], axis=1)
        x, y = seq[:, :-1].copy(), seq[:, 1:].copy()
        # Only the codes are the task: the caption's own words and the IMG marker after them
        # carry no loss, so the val loss is exactly the cross-entropy per code.
        y[:, :CAPTION_LEN] = -100
        return torch.from_numpy(x), torch.from_numpy(y)

    def batch(self, batch_size: int, drop: float = 0.0):
        return self.rows(self.rng.choice(self.index, size=batch_size), drop)
