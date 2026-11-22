"""The picture line from a terminal (docs/26). Every command has a portal control on the
Pictures tab.

    python -m aksharallm.pictures read                       # the judge on the corpus: must be 100%
    python -m aksharallm.pictures report imgtok-shapes       # ladder + held-out + gallery
    python -m aksharallm.pictures compare A.json B.json      # two reports, paired per picture
    python -m aksharallm.pictures reports                    # every report so far

Training is `scripts/audio.sh imgtok-shapes` (or `python -m aksharallm.pictures.train
configs/imgtok-shapes.yaml`).

Read with: docs/26-pictures.md -- the chapter this implements; it ends with the order to read
these files in.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

RESULTS = Path("logs/pictures")


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


def find_checkpoint(name: str, root: Path | None = None) -> Path:
    root = root or _root()
    p = Path(name)
    for cand in (p, root / p, root / "checkpoints" / name / "ckpt_best.pt",
                 root / "checkpoints" / name / "ckpt_last.pt"):
        if cand.is_file():
            return cand
    raise FileNotFoundError(f"no checkpoint {name!r} (a .pt path, or a run in checkpoints/)")


def cmd_read(args) -> int:
    import numpy as np

    from ..vision.image import ImageCaptions
    from .reader import judge
    ds = ImageCaptions(args.corpus, split="val", val_frac=1.0)   # every picture
    n = min(args.n, len(ds)) if args.n else len(ds)
    wrong = [i for i in range(n) if not judge(np.asarray(ds.images[i]), ds.manifest.facts[i])["all"]]
    print(f"{args.corpus}: the reader is right on {n - len(wrong):,} of {n:,} pictures")
    for i in wrong[:10]:
        print(f"  #{i}: drawn {ds.manifest.facts[i]}, read "
              f"{judge(np.asarray(ds.images[i]), ds.manifest.facts[i])['read']}")
    return 0 if not wrong else 1


def cmd_report(args) -> int:
    import torch

    from ..vision.image import ImageCaptions
    from .measure import gallery, ladder, public
    from .train import load_tokenizer
    root = _root()
    path = find_checkpoint(args.checkpoint, root)
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model, blob = load_tokenizer(path, device)
    pc = model.cfg
    data = (blob.get("config") or {}).get("data") or {}
    corpus = args.corpus or data.get("corpus", "data/vision/shapes")
    holdout = data.get("holdout", "data/vision/shapes-holdout")
    t = time.time()
    ds = ImageCaptions(root / corpus, split="val")
    r = ladder(model, ds, n=args.n, device=device)
    run = path.parent.name
    stem = f"report-{run}-step{blob.get('step')}"
    out_dir = root / RESULTS / stem
    pics = gallery(r, out_dir, k=args.gallery)
    print(f"{path}  ({pc.grid}x{pc.grid} grid x {pc.n_codebooks} codebooks of "
          f"{pc.codebook_size}, step {blob.get('step')})")
    print(f"{r['n']} val pictures from {corpus}; the reader on the originals: "
          f"all {r['originals']['all'] * 100:.1f}%  (must be 100: the positive control)\n")
    print("  books     bits     PSNR    count  colour   shape     all")
    for row in r["ladder"]:
        print(f"  {row['books']:>5} {row['bits']:>8.0f} {row['psnr']:>7.2f}dB "
              f"{row['count'] * 100:>6.1f}% {row['colour'] * 100:>6.1f}% "
              f"{row['shape'] * 100:>6.1f}% {row['all'] * 100:>6.1f}%")
    print("\n  codebook use (all books, val pictures):  "
          + "  ".join(f"#{u['book']} {u['used']} used / ppl {u['perplexity']:.0f}"
                      for u in r["usage"]))
    out = public(r)
    held = root / holdout
    if (held / "manifest.json").is_file():
        hd = ImageCaptions(held, split="val", val_frac=1.0)   # all of them (val_frac 0 keeps one back)
        h = ladder(model, hd, n=args.n, device=device)
        out["holdout"] = public(h) | {"corpus": holdout}
        print(f"  held-out combination ({holdout}, never drawn in training): "
              + "  ".join(f"{row['books']}b all {row['all'] * 100:.1f}%" for row in h["ladder"]))
    out |= {"checkpoint": str(path), "run": run, "step": blob.get("step"), "corpus": corpus,
            "grid": pc.grid, "n_codebooks": pc.n_codebooks, "codebook_size": pc.codebook_size,
            "gallery": [f"{stem}/{p}" for p in pics], "seconds": round(time.time() - t, 1)}
    f = root / RESULTS / f"{stem}.json"
    f.write_text(json.dumps(out, indent=1))
    print(f"\nwrote {f}  (gallery: {out_dir}/, original then each codebook count)")
    return 0


def cmd_compare(args) -> int:
    from .measure import compare
    a, b = (json.loads(Path(f).read_text()) for f in (args.a, args.b))
    for label, key in (("val pictures", None), ("held-out combination", "holdout")):
        x, y = (a, b) if key is None else (a.get(key), b.get(key))
        if not x or not y:
            continue
        r = compare(x, y, args.books_a, args.books_b)
        p = "–" if r["p"] is None else f"{r['p']:.3g}"
        print(f"{label:<22} n {r['n']}   a {r['a'] * 100:5.1f}%   b {r['b'] * 100:5.1f}%   "
              f"b only {r['b_only']:>3}   a only {r['a_only']:>3}   p {p}")
    print(f"\na = {args.a}\nb = {args.b}")
    return 0


def cmd_reports(args) -> int:
    d = _root() / RESULTS
    files = sorted(d.glob("report-*.json")) if d.is_dir() else []
    if not files:
        print("no picture reports yet: python -m aksharallm.pictures report <run>")
        return 0
    print(f"{'report':<44} {'grid':>5} {'books':>5} {'PSNR':>7} {'all':>7} {'held out':>9}")
    for f in files:
        r = json.loads(f.read_text())
        top = r["ladder"][-1]
        h = (r.get("holdout") or {}).get("ladder", [{}])[-1].get("all")
        print(f"{f.stem:<44} {r['grid']:>3}x{r['grid']:<1} {top['books']:>5} {top['psnr']:>6.1f}dB "
              f"{top['all'] * 100:>6.1f}% " + ("–" if h is None else f"{h * 100:>8.1f}%"))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m aksharallm.pictures",
                                 description="The picture line: tokenizer reports (docs/26).")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("read", help="the rule-based reader on a corpus (the positive control)")
    s.add_argument("--corpus", default="data/vision/shapes")
    s.add_argument("--n", type=int, default=0, help="first N pictures (0 = all)")
    s.set_defaults(fn=cmd_read)
    s = sub.add_parser("report", help="reconstruct val pictures at each codebook count and judge them")
    s.add_argument("checkpoint", help="a .pt, or a run name (checkpoints/<run>/ckpt_best.pt)")
    s.add_argument("--n", type=int, default=256, help="pictures to judge")
    s.add_argument("--gallery", type=int, default=8, help="pictures to save as PNG rows")
    s.add_argument("--corpus", default=None, help="default: the corpus the run trained on")
    s.add_argument("--device", default="cpu", help="cpu (default), cuda or auto")
    s.set_defaults(fn=cmd_report)
    s = sub.add_parser("compare", help="two reports, paired picture by picture")
    s.add_argument("a")
    s.add_argument("b")
    s.add_argument("--books-a", type=int, default=None, help="codebook count of a (default: all)")
    s.add_argument("--books-b", type=int, default=None, help="codebook count of b (default: all)")
    s.set_defaults(fn=cmd_compare)
    s = sub.add_parser("reports", help="every picture report, one line each")
    s.set_defaults(fn=cmd_reports)
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
