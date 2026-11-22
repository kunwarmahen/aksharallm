"""The picture line from a terminal (docs/26). Every command has a portal control on the
Pictures tab.

    python -m aksharallm.pictures read                       # the judge on the corpus: must be 100%
    python -m aksharallm.pictures report imgtok-shapes       # ladder + held-out + gallery
    python -m aksharallm.pictures compare A.json B.json      # two reports, paired per picture
    python -m aksharallm.pictures reports                    # every report so far

G1, a sentence to a picture:

    python -m aksharallm.pictures codes imgtok-shapes        # encode the corpus -> data/pictures/
    python -m aksharallm.pictures draw imggen-shapes "two red circles" --guidance 3
    python -m aksharallm.pictures gen-report imggen-shapes --guidance 3   # all 60 captions, judged
    python -m aksharallm.pictures gen-compare A.json B.json  # two G1 reports, paired

Training is `scripts/audio.sh imgtok-shapes` (or `python -m aksharallm.pictures.train
configs/imgtok-shapes.yaml`), and `scripts/audio.sh imggen-shapes` for G1.

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
    gens = sorted(d.glob("gen-*.json"))
    if gens:
        print(f"\n{'G1 report (drawn from captions)':<44} {'k':>3} {'seen':>7} {'held out':>9} "
              f"{'copies':>7} {'variety':>8}")
        for f in gens:
            r = json.loads(f.read_text())
            print(f"{f.stem:<44} {r['k']:>3} {r['seen']['all'] * 100:>6.1f}% "
                  f"{r['held_out'].get('all', 0) * 100:>8.1f}% {r['seen']['copies'] * 100:>6.1f}% "
                  f"{r['variety'] * 100:>7.0f}%")
    return 0


def _device(name: str) -> str:
    import torch
    return ("cuda" if torch.cuda.is_available() else "cpu") if name == "auto" else name


def cmd_codes(args) -> int:
    from .codes import encode_corpus
    root = _root()
    path = find_checkpoint(args.tokenizer, root)
    out = args.out or f"data/pictures/codes-{path.parent.name}"
    try:
        rel = path.resolve().relative_to(root)
    except ValueError:
        rel = path
    encode_corpus(rel, args.corpus, root / out, books=args.books, device=_device(args.device))
    return 0


def _load_g1(name: str, device: str):
    from .gen import load_generator
    from .train import load_tokenizer
    root = _root()
    path = find_checkpoint(name, root)
    model, pg, blob = load_generator(path, device)
    tok = Path(pg.tokenizer)
    tokenizer, _ = load_tokenizer(tok if tok.is_absolute() else root / tok, device)
    return path, model, pg, blob, tokenizer


def cmd_draw(args) -> int:
    from ..vision.image import write_png
    from .codes import fact_of
    from .gen import draw
    from .reader import judge
    device = _device(args.device)
    path, model, pg, blob, tokenizer = _load_g1(args.checkpoint, device)
    pics, _ = draw(model, tokenizer, [args.caption] * args.n, pg=pg, guidance=args.guidance,
                   temperature=args.temperature, top_k=args.top_k, seed=args.seed, device=device)
    want = fact_of(args.caption)
    verdicts = [judge(p, want) for p in pics]
    ok = sum(v["all"] for v in verdicts)
    print(f"{path} (step {blob.get('step')}): {args.n} x {args.caption!r} at guidance "
          f"{args.guidance:g}: the reader agrees on {ok} of {args.n}")
    for i, v in enumerate(verdicts):
        r = v["read"]
        print(f"  #{i}: {'ok   ' if v['all'] else 'WRONG'} read {r['count']} {r['colour']} {r['shape']}")
    import numpy as np
    gap = np.full((64, 3, 3), 255, dtype=np.uint8)
    row = np.concatenate(sum(([t, gap] for t in pics), [])[:-1], axis=1)
    out = Path(args.out or _root() / RESULTS / "draw" / (args.caption.replace(" ", "-") + ".png"))
    out.parent.mkdir(parents=True, exist_ok=True)
    write_png(out, row.repeat(3, axis=0).repeat(3, axis=1))
    print(f"wrote {out}")
    return 0


def cmd_gen_report(args) -> int:
    from .codes import CodesDataset
    from .score import gallery, public, report
    root = _root()
    device = _device(args.device)
    path, model, pg, blob, tokenizer = _load_g1(args.checkpoint, device)
    codes = (blob.get("config") or {}).get("data", {}).get("codes")
    train_ds = CodesDataset(root / codes, books=pg.books, split="train")
    r = report(model, tokenizer, pg, train_ds=train_ds, k=args.k, guidance=args.guidance,
               temperature=args.temperature, top_k=args.top_k, seed=args.seed, device=device)
    run = path.parent.name
    stem = f"gen-{run}-step{blob.get('step')}-g{args.guidance:g}"
    if args.temperature != 1.0 or args.top_k or args.seed:
        stem += f"-t{args.temperature:g}-k{args.top_k or 0}-s{args.seed}"
    names = gallery(r, root / RESULTS / stem)
    print(f"{path} (step {blob.get('step')}): {args.k} pictures for each of "
          f"{len(r['captions'])} captions, guidance {args.guidance:g}, {r['seconds']}s\n")
    print("               n    count  colour   shape     all  copies")
    for label, key in (("seen", "seen"), ("held out", "held_out")):
        g = r[key]
        if g.get("n"):
            print(f"  {label:<10} {g['n']:>4} " + " ".join(
                f"{g[c] * 100:>6.1f}%" for c in ("count", "colour", "shape", "all"))
                  + f" {g['copies'] * 100:>6.1f}%")
    print(f"\n  variety: {r['variety'] * 100:.0f}% of the {args.k} pictures per caption are distinct")
    print("  held-out captions: " + "  ".join(
        f"{p['caption']} {p['all'] * 100:.0f}%" for p in r["per_caption"] if p["held_out"]))
    out = public(r) | {"checkpoint": str(path), "run": run, "step": blob.get("step"),
                       "books": pg.books, "gallery": [f"{stem}/{n}" for n in names]}
    f = root / RESULTS / f"{stem}.json"
    f.write_text(json.dumps(out, indent=1))
    print(f"\nwrote {f}  (gallery: one PNG per caption)")
    return 0


def cmd_gen_compare(args) -> int:
    from .score import compare
    a, b = (json.loads(Path(f).read_text()) for f in (args.a, args.b))
    for part, r in compare(a, b).items():
        p = "–" if r["p"] is None else f"{r['p']:.3g}"
        print(f"{part:<10} n {r['n']:>4}   a {r['a'] * 100:5.1f}%   b {r['b'] * 100:5.1f}%   "
              f"b only {r['b_only']:>3}   a only {r['a_only']:>3}   p {p}")
    print(f"\na = {args.a}\nb = {args.b}")
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
    s = sub.add_parser("codes", help="G1: encode a picture corpus with a trained tokenizer")
    s.add_argument("tokenizer", help="a .pt, or a tokenizer run name")
    s.add_argument("--corpus", default="data/vision/shapes")
    s.add_argument("--books", type=int, default=None, help="codebooks to keep (default: all)")
    s.add_argument("--out", default=None, help="default: data/pictures/codes-<run>")
    s.add_argument("--device", default="cpu")
    s.set_defaults(fn=cmd_codes)
    for name, fn, hlp in (("draw", cmd_draw, "G1: draw pictures for one caption"),
                          ("gen-report", cmd_gen_report, "G1: draw every caption and judge it")):
        s = sub.add_parser(name, help=hlp)
        s.add_argument("checkpoint", help="a .pt, or a G1 run name")
        if name == "draw":
            s.add_argument("caption", help="three words: count colour shape, e.g. 'two red circles'")
            s.add_argument("--n", type=int, default=8)
            s.add_argument("--out", default=None, help="PNG path (default logs/pictures/draw/)")
        else:
            s.add_argument("--k", type=int, default=8, help="pictures per caption")
        s.add_argument("--guidance", type=float, default=3.0, help="1 = no guidance")
        s.add_argument("--temperature", type=float, default=1.0)
        s.add_argument("--top-k", type=int, default=None)
        s.add_argument("--seed", type=int, default=0)
        s.add_argument("--device", default="cpu", help="cpu (default), cuda or auto")
        s.set_defaults(fn=fn)
    s = sub.add_parser("gen-compare", help="G1: two gen-reports, paired per (caption, sample)")
    s.add_argument("a")
    s.add_argument("b")
    s.set_defaults(fn=cmd_gen_compare)
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
