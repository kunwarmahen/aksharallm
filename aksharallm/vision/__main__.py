"""`python -m aksharallm.vision` — make the corpus, look at it, and caption with a checkpoint.

    python -m aksharallm.vision corpus --out data/vision/shapes --images 8000
    python -m aksharallm.vision show data/vision/shapes --out /tmp/grid.png
    python -m aksharallm.vision caption checkpoints/vision-shapes/ckpt_best.pt

Read with: docs/21-vision.md -- the chapter this implements; it ends with the order to read
these files in.
"""

from __future__ import annotations

import argparse

import numpy as np


def cmd_corpus(args) -> int:
    from .image import holdout_set, synth_corpus

    man = synth_corpus(args.out, n_images=args.images, size=args.size, seed=args.seed,
                       hold_out=tuple(args.hold_out.split(",")) if args.hold_out else None)
    print(f"{man.n_images:,} images -> {args.out}")
    if args.hold_out:
        combo = tuple(args.hold_out.split(","))
        h = holdout_set(f"{args.out}-holdout", combo, n=args.holdout_images, size=args.size)
        print(f"{h.n_images} held-out {combo[0]} {combo[1]}s -> {args.out}-holdout")
        print("  the compositional test: both attributes are common, the PAIR was never seen")
    print("\ntrain it:  .venv/bin/python -m aksharallm.vision.train configs/vision-shapes.yaml")
    return 0


def cmd_show(args) -> int:
    """A grid of the corpus as one PNG, because a corpus you have not looked at is a guess."""
    from .image import ImageCaptions, write_png

    ds = ImageCaptions(args.corpus, split="train", seed=0)
    n = args.rows * args.cols
    tiles = [ds.images[ds.index[i % len(ds)]] for i in range(n)]
    size = ds.manifest.size
    grid = np.zeros((args.rows * size, args.cols * size, 3), dtype=np.uint8)
    for i, tile in enumerate(tiles):
        r, c = divmod(i, args.cols)
        grid[r * size : (r + 1) * size, c * size : (c + 1) * size] = tile
    write_png(args.out, grid)
    print(f"{n} images -> {args.out}")
    for i in range(min(n, args.cols)):
        print(f"  {ds.manifest.captions[ds.index[i]]}")
    return 0


def cmd_caption(args) -> int:
    import torch

    from ..config import ModelConfig
    from ..model.transformer import Transformer
    from ..tokenizer.tokenizer import Tokenizer
    from .encoder import VisionConfig
    from .image import ImageCaptions
    from .lm import VisionLanguageModel, caption, score_batch

    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    blob = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if blob.get("stage") != "vision":
        raise SystemExit(f"{args.checkpoint} is not a vision checkpoint "
                         f"(stage={blob.get('stage')!r})")
    base = torch.load(blob["base"], map_location=device, weights_only=False)
    lm = Transformer(ModelConfig(**base["model_config"])).to(device)
    lm.load_state_dict(base["model"])
    model = VisionLanguageModel(lm, VisionConfig(**blob["vision"])).to(device)
    model.tower.load_state_dict(blob["tower"])
    tok = Tokenizer(blob["tokenizer"])

    ds = ImageCaptions(args.corpus, split="val", seed=0)
    pairs = []
    for i in range(min(args.n, len(ds))):
        image, truth, fact = ds.item(i)
        said = caption(model, torch.from_numpy(image), tok, device=device)
        pairs.append((fact, said))
        if i < args.show:
            mark = "ok " if all(__import__("aksharallm.vision.lm", fromlist=["x"])
                                .score_caption(fact, said).values()) else "   "
            print(f"  {mark}{truth:<28} -> {said!r}")
    s = score_batch(pairs)
    print(f"\n  n={s['n']}  count {s['count'] * 100:.0f}%  colour {s['colour'] * 100:.0f}%  "
          f"shape {s['shape'] * 100:.0f}%  ALL {s['all_three'] * 100:.0f}%")
    print("  Three attributes reported separately: a model that never counts is a specific")
    print("  failure that a single accuracy would average away.")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m aksharallm.vision", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("corpus", help="render the shapes corpus (no download)")
    s.add_argument("--out", default="data/vision/shapes")
    s.add_argument("--images", type=int, default=8000)
    s.add_argument("--size", type=int, default=64)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--hold-out", default="purple,triangle",
                   help="a colour,shape PAIR to keep out of training (empty to keep all)")
    s.add_argument("--holdout-images", type=int, default=64)
    s.set_defaults(fn=cmd_corpus)

    s = sub.add_parser("show", help="a grid of the corpus as one PNG")
    s.add_argument("corpus")
    s.add_argument("--out", default="logs/vision/corpus.png")
    s.add_argument("--rows", type=int, default=4)
    s.add_argument("--cols", type=int, default=8)
    s.set_defaults(fn=cmd_show)

    s = sub.add_parser("caption", help="caption held-out images and score them")
    s.add_argument("checkpoint")
    s.add_argument("--corpus", default="data/vision/shapes")
    s.add_argument("--n", type=int, default=64)
    s.add_argument("--show", type=int, default=10)
    s.add_argument("--cpu", action="store_true")
    s.set_defaults(fn=cmd_caption)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
