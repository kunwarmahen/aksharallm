"""The Vision tab's back end: look at the pictures, and score what the model says about them.

The reason this is a tab rather than a chart is that a caption is only interesting **beside
the image it describes**. Every other measurement in this portal is a number you read; this
one is a picture and a sentence, and the judgement is immediate.

It also carries the one number the corpus was designed to produce: three attributes scored
*separately*. A model that names the colour and the shape and never counts is a specific,
diagnosable failure, and a single accuracy would average it away.

Runs inline, like `interp`, `diffusion` and `audio` — captioning sixteen 64×64 images is a
handful of forward passes over a 14M-parameter model, so a job runner with a pid file would
be more machinery than the work. Device policy is the repo's standing one: the CPU while a
training run holds the card.

Read with: docs/22-vision.md -- the chapter this implements; it ends with the order to read
these files in.
"""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import numpy as np
import torch

from ..vision.image import COLOURS, SHAPES, ImageCaptions, ImageManifest, from_tensor

#: A browser must not be able to ask for a hundred greedy decodes in one request.
MAX_IMAGES = 32


class VisionError(RuntimeError):
    """Something to show as a message rather than a stack trace."""


def png_data_uri(image: np.ndarray, scale: int = 3) -> str:
    """A `data:` URI for one image, nearest-neighbour upscaled so 64px is visible.

    Inlined rather than served from a route for the same reason the Audio tab inlines its
    audio: a rendered image is derived from a corpus and an index and has no stable identity,
    so a URL for it would be a cache-invalidation problem in exchange for nothing.

    **Nearest neighbour, not smooth.** These are hard-edged shapes on a flat background, and
    a bilinear upscale would blur exactly the boundary the model is being asked to classify.
    """
    from PIL import Image

    img = np.asarray(image, dtype=np.uint8)
    im = Image.fromarray(img).resize(
        (img.shape[1] * scale, img.shape[0] * scale), Image.NEAREST
    )
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


class Vision:
    """Everything the Vision tab can ask for."""

    def __init__(self, root: Path | None = None, device_for=None):
        self.root = Path(root) if root else Path.cwd()
        self._device_for = device_for
        self._cache: dict = {}

    # ---- discovery --------------------------------------------------------------------

    def device(self) -> tuple[str, str]:
        if self._device_for is not None:
            return self._device_for()
        if torch.cuda.is_available():
            return "cuda", "the card is free"
        return "cpu", "no CUDA device"

    def checkpoints(self) -> list[dict]:
        """Vision checkpoints, identified by `stage` rather than by a filename convention."""
        out = []
        for path in sorted(self.root.glob("checkpoints/*/ckpt_*.pt")):
            try:
                blob = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
            except Exception:
                continue
            if blob.get("stage") != "vision":
                continue
            v = blob.get("vision") or {}
            grid = (v.get("image_size", 0) // v.get("patch", 1)) if v.get("patch") else 0
            out.append({
                "rel": str(path.relative_to(self.root)),
                "step": blob.get("step"),
                "best_val": blob.get("best_val"),
                "base": blob.get("base"),
                "image_size": v.get("image_size"),
                "patch": v.get("patch"),
                "patches": grid * grid,
                "image_tokens": v.get("n_tokens") or grid * grid,
                "mtime": path.stat().st_mtime,
            })
        return sorted(out, key=lambda r: r["mtime"], reverse=True)

    def corpora(self) -> list[dict]:
        out = []
        for man_path in sorted(self.root.glob("data/vision/*/manifest.json")):
            if not (man_path.parent / "images.bin").is_file():
                continue
            try:
                man = ImageManifest.load(man_path)
            except Exception:
                continue
            pairs = {(f.get("colour"), f.get("shape")) for f in man.facts}
            out.append({
                "rel": str(man_path.parent.relative_to(self.root)),
                "images": man.n_images,
                "size": man.size,
                # Which (colour, shape) pairs never occur — the compositional test, computed
                # from the corpus rather than trusted from a config that could have drifted.
                "missing_pairs": sorted(
                    f"{c} {s}" for c in COLOURS for s in SHAPES if (c, s) not in pairs
                ),
            })
        return out

    def overview(self) -> dict:
        device, why = self.device()
        return {"checkpoints": self.checkpoints(), "corpora": self.corpora(),
                "device": device, "device_reason": why, "max_images": MAX_IMAGES,
                "colours": list(COLOURS), "shapes": list(SHAPES)}

    # ---- the corpus --------------------------------------------------------------------

    def _dataset(self, corpus: str, split: str) -> ImageCaptions:
        path = (self.root / corpus).resolve()
        if not str(path).startswith(str(self.root.resolve())):
            raise VisionError("corpus path escapes the repository")
        try:
            return ImageCaptions(path, split=split, seed=0)
        except Exception as e:
            raise VisionError(f"cannot read {corpus}: {e}") from e

    def samples(self, corpus: str, n: int = 12, split: str = "val") -> dict:
        """Images and their true captions — what the model is being asked about."""
        ds = self._dataset(corpus, split)
        n = min(n, MAX_IMAGES, len(ds))
        rows = []
        for i in range(n):
            image, truth, fact = ds.item(i)
            rows.append({"index": i, "truth": truth, "fact": fact,
                         "png": png_data_uri(from_tensor(image))})
        return {"corpus": corpus, "split": split, "n": len(rows), "rows": rows,
                "total": len(ds)}

    # ---- captioning ---------------------------------------------------------------------

    def _model(self, rel: str):
        from ..config import ModelConfig
        from ..model.transformer import Transformer
        from ..tokenizer.tokenizer import Tokenizer
        from ..vision.encoder import VisionConfig
        from ..vision.lm import VisionLanguageModel

        device, _ = self.device()
        key = f"{rel}@{device}"
        if self._cache.get("key") != key:
            path = (self.root / rel).resolve()
            if not str(path).startswith(str(self.root.resolve())):
                raise VisionError("checkpoint path escapes the repository")
            if not path.is_file():
                raise VisionError(f"no such checkpoint: {rel}")
            blob = torch.load(path, map_location=device, weights_only=False)
            if blob.get("stage") != "vision":
                raise VisionError(
                    f"{rel} is not a vision checkpoint (stage={blob.get('stage')!r}). "
                    "Vision, audio and text checkpoints are separate families."
                )
            base_path = self.root / blob["base"]
            if not base_path.is_file():
                raise VisionError(
                    f"the language model this was trained against is missing: {blob['base']}. "
                    "A vision checkpoint is only the tower — it is useless without its base."
                )
            base = torch.load(base_path, map_location=device, weights_only=False)
            lm = Transformer(ModelConfig(**base["model_config"])).to(device)
            lm.load_state_dict(base["model"])
            model = VisionLanguageModel(lm, VisionConfig(**blob["vision"])).to(device)
            model.tower.load_state_dict(blob["tower"])
            model.eval()
            self._cache = {"key": key, "model": model,
                           "tok": Tokenizer(self.root / blob["tokenizer"])}
        return self._cache["model"], self._cache["tok"]

    def caption(self, checkpoint: str, corpus: str, n: int = 12,
                split: str = "val") -> dict:
        """Caption `n` images and score each against the description it was rendered from."""
        from ..vision.lm import caption as caption_one
        from ..vision.lm import score_batch, score_caption

        model, tok = self._model(checkpoint)
        ds = self._dataset(corpus, split)
        device, why = self.device()
        n = min(n, MAX_IMAGES, len(ds))

        rows, pairs = [], []
        for i in range(n):
            image, truth, fact = ds.item(i)
            said = caption_one(model, torch.from_numpy(image), tok, device=device)
            marks = score_caption(fact, said)
            pairs.append((fact, said))
            rows.append({"index": i, "truth": truth, "said": said, "fact": fact,
                         "marks": marks, "all_three": all(marks.values()),
                         "png": png_data_uri(from_tensor(image))})

        score = score_batch(pairs)
        return {
            "checkpoint": checkpoint, "corpus": corpus, "split": split,
            "rows": rows, "score": score, "device": device, "device_reason": why,
            "image_tokens": model.n_image_tokens,
            "params": model.n_params(),
            # Repeated on every response so the caveat travels with the numbers.
            "note": (
                "Three attributes scored SEPARATELY: a model that names the colour and the "
                "shape and never counts is a specific failure that one accuracy would "
                "average away. Counting is normally the last of the three to be learned."
            ),
        }

    def runs(self) -> list[dict]:
        """Vision runs and where they have got to, for the tab's status strip."""
        out = []
        for cfg_path in sorted(self.root.glob("configs/*.yaml")):
            if "\nvision:" not in cfg_path.read_text():
                continue
            name = cfg_path.stem
            run_dir = self.root / "checkpoints" / name
            step, score = None, None
            log = run_dir / "train_log.jsonl"
            if log.is_file():
                for line in reversed(log.read_text().splitlines()[-400:]):
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if score is None and "score" in rec:
                        score = rec["score"]
                    if step is None and "step" in rec:
                        step = rec["step"]
                    if step is not None and score is not None:
                        break
            out.append({"name": name, "training": (run_dir / "train.pid").is_file(),
                        "step": step, "score": score})
        return out
