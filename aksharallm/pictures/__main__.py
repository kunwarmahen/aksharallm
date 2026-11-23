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

G4b, our own contrastive model as a second judge:

    python -m aksharallm.pictures scenes                     # the scenes corpus: size and place too
    python -m aksharallm.pictures clip-report clip-shapes    # beside the reader, on known pictures
    python -m aksharallm.pictures gen-report <model> --clip clip-shapes

G2, flow matching in pixels: `draw` and `gen-report` take a G2 run the same way
(`flow-shapes`), and `--steps` sets its number of Euler steps. G3 is the same in the
tokenizer's latent: `pictures latents imgtok-shapes` encodes the corpus, `flow-latent-shapes`
trains on it, and `--snap` re-quantizes a drawn latent before the decoder paints it.

Training is `scripts/audio.sh imgtok-shapes` (or `python -m aksharallm.pictures.train
configs/imgtok-shapes.yaml`), `scripts/audio.sh imggen-shapes` for G1, and
`scripts/audio.sh flow-shapes` for G2.

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


def cmd_scenes(args) -> int:
    from .scenes import holdout_corpus, scenes_corpus
    out = Path(args.out)
    scenes_corpus(out, n_images=args.n, seed=args.seed)
    held = Path(str(out) + "-holdout")
    holdout_corpus(held)
    print(f"held-out combinations, 64 of each -> {held}/")
    return 0


def cmd_coco(args) -> int:
    """G4c: pack COCO 2017 (train -> OUT, val -> OUT-val) as 64x64 picture corpora."""
    from .coco import pack
    for split in args.splits.split(","):
        out = args.out if split == "train" else f"{args.out}-{split}"
        pack(args.raw, out, split, limit=args.limit or None, workers=args.workers)
    return 0


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


def _load_drawer(name: str, device: str, snap: bool = False):
    from .drawer import load_drawer
    root = _root()
    d = load_drawer(find_checkpoint(name, root), root, device)
    if snap:
        d.snap = True
    return d


def cmd_latents(args) -> int:
    from .latents import encode_latents
    root = _root()
    path = find_checkpoint(args.tokenizer, root)
    out = args.out or f"data/pictures/latents-{path.parent.name}"
    try:
        rel = path.resolve().relative_to(root)
    except ValueError:
        rel = path
    encode_latents(rel, args.corpus, root / out, books=args.books, device=_device(args.device))
    return 0


def cmd_draw(args) -> int:
    import numpy as np

    from ..vision.image import write_png
    from .codes import fact_of
    from .reader import caption, judge
    d = _load_drawer(args.checkpoint, _device(args.device), args.snap)
    pics = d.draw([args.caption] * args.n, guidance=args.guidance, temperature=args.temperature,
                  top_k=args.top_k, seed=args.seed, steps=args.steps)
    try:
        want = fact_of(args.caption)
    except ValueError:                       # a G4 sentence: judged if the words are there
        from .phrases import fact_from_text
        want = fact_from_text(args.caption)
    if want is None:
        raise SystemExit("no count, colour and shape found in the sentence; nothing to judge")
    if d.corpus_kind() == "shapes":       # a shapes model never drew size or place
        want = {k: want[k] for k in ("count", "colour", "shape")}
    verdicts = [judge(p, want) for p in pics]
    ok = sum(v["all"] for v in verdicts)
    print(f"{d.path} ({d.kind}, step {d.step}): {args.n} x {args.caption!r} at guidance "
          f"{args.guidance:g}: the reader agrees on {ok} of {args.n}")
    for i, v in enumerate(verdicts):
        r = v["read"]
        print(f"  #{i}: {'ok   ' if v['all'] else 'WRONG'} read {caption(r)}")
    gap = np.full((64, 3, 3), 255, dtype=np.uint8)
    row = np.concatenate(sum(([t, gap] for t in pics), [])[:-1], axis=1)
    out = Path(args.out or _root() / RESULTS / "draw" / (args.caption.replace(" ", "-") + ".png"))
    out.parent.mkdir(parents=True, exist_ok=True)
    write_png(out, row.repeat(3, axis=0).repeat(3, axis=1))
    print(f"wrote {out}")
    return 0


def cmd_gen_report(args) -> int:
    from .score import gallery, public
    root = _root()
    d = _load_drawer(args.checkpoint, _device(args.device), args.snap)
    clip = None
    if args.clip:
        from .clip import load_clip
        cp = find_checkpoint(args.clip, root)
        clip = (load_clip(cp, d.device, root)[0], cp)
    r = d.report(k=args.k, guidance=args.guidance, temperature=args.temperature,
                 top_k=args.top_k, seed=args.seed, steps=args.steps, phrasing=args.phrasing,
                 clip=clip)
    run = d.path.parent.name
    stem = f"gen-{run}-step{d.step}-g{args.guidance:g}"
    if args.k != 8:                       # never overwrite the default k=8 report
        stem += f"-k{args.k}"
    if args.temperature != 1.0 or args.top_k or args.seed:
        stem += f"-t{args.temperature:g}-k{args.top_k or 0}-s{args.seed}"
    if args.steps:
        stem += f"-n{args.steps}"
    if args.snap:
        stem += "-snap"
    if args.phrasing:
        stem += f"-p{args.phrasing}"
    if args.clip:
        stem += "-clip"
    names = gallery(r, root / RESULTS / stem)
    print(f"{d.path} ({d.kind}, step {d.step}): {args.k} pictures for each of "
          f"{len(r['captions'])} captions, guidance {args.guidance:g}, {r['seconds']}s\n")
    attrs = r.get("attributes") or ["count", "colour", "shape"]
    print("                    n " + "".join(f"{a:>8}" for a in attrs) + "     all  copies")
    groups = [("seen", r["seen"]), ("held out", r["held_out"])]
    groups += [(f"  {name}", g) for name, g in (r.get("held_out_by") or {}).items()]
    for label, g in groups:
        if g.get("n"):
            print(f"  {label:<16} {g['n']:>5} " + " ".join(
                f"{g[c] * 100:>6.1f}%" for c in [*attrs, "all"]) + f" {g['copies'] * 100:>6.1f}%")
    oc = r.get("own_clip")
    if oc:
        fl = "" if oc.get("floor") is None else f" and fits above the floor {oc['floor']:.3f}"
        print(f"\n  our contrastive model ({oc['checkpoint']}): the asked caption is the closest of "
              f"{oc['candidates']}{fl}")
        for label, key in (("seen", "seen"), ("held out", "held_out")):
            g = oc[key]
            if g.get("n"):
                a = g["agreement"]
                kappa = "–" if a["kappa"] is None else f"{a['kappa']:.2f}"
                print(f"  {label:<10} top-1 {g['top1'] * 100:5.1f}%  agrees with the reader on "
                      f"{a['agree'] * 100:5.1f}% (kappa {kappa}; model only {a['model_only']}, "
                      f"reader only {a['reader_only']})")
        if oc.get("closed"):
            c = oc["closed"]
            ck = "–" if c["agreement"]["kappa"] is None else f"{c['agreement']['kappa']:.2f}"
            print(f"  without the floor (closed set): top-1 {c['top1'] * 100:.1f}%, agrees "
                  f"{c['agreement']['agree'] * 100:.1f}% (kappa {ck}); {c['none_fits']} pictures "
                  "judged 'none fits'")
    print(f"\n  variety: {r['variety'] * 100:.0f}% of the {args.k} pictures per caption are distinct")
    print("  held-out captions: " + "  ".join(
        f"{p['caption']} {p['all'] * 100:.0f}%" for p in r["per_caption"] if p["held_out"]))
    out = public(r) | {"checkpoint": str(d.path), "run": run, "step": d.step,
                       "books": d.books, "gallery": [f"{stem}/{n}" for n in names]}
    f = root / RESULTS / f"{stem}.json"
    f.write_text(json.dumps(out, indent=1))
    print(f"\nwrote {f}  (gallery: one PNG per caption)")
    return 0


def cmd_clip_calibrate(args) -> int:
    """G4b: measure the none-fits floor on clean originals and write it beside the checkpoint."""
    import numpy as np

    from ..vision.image import ImageCaptions
    from .clip import NONE_FITS, calibrate, load_clip
    root = _root()
    path = find_checkpoint(args.checkpoint, root)
    model, blob = load_clip(path, _device(args.device), root)
    sets = []
    for corpus in ((blob.get("config") or {}).get("data") or {}).get("corpora") or []:
        ds = ImageCaptions(root / corpus, split="val")
        # the LAST val pictures: clip-report reads the first ones, so the floor is never
        # measured on the pictures it is then reported on
        idx = ds.index[-args.n:]
        sets.append((getattr(ds.manifest, "kind", "shapes"), np.asarray(ds.images[idx]),
                     [ds.manifest.facts[int(i)] for i in idx]))
    floors = calibrate(model, sets, q=args.q, device=_device(args.device))
    f = path.parent / NONE_FITS
    d = json.loads(f.read_text()) if f.is_file() else {}
    d[path.name] = {"step": blob.get("step"), "q": args.q, "n": args.n, "floors": floors,
                    "built": time.strftime("%Y-%m-%d %H:%M:%S")}
    f.write_text(json.dumps(d, indent=1) + "\n")
    print(f"{path} (step {blob.get('step')}): a picture whose best caption fits worse than this "
          f"is judged 'none fits' ({args.q:.0%} of clean originals would be)")
    for kind, v in floors.items():
        print(f"  {kind:<8} cosine floor {v:.4f}")
    print(f"wrote {f}")
    return 0


def cmd_clip_report(args) -> int:
    """G4b: our contrastive model on pictures whose facts are known, beside the reader."""
    import numpy as np

    from ..vision.image import ImageCaptions
    from .clip import agreement, load_clip, rank
    from .phrases import all_facts
    from .reader import ATTRS, judge
    root = _root()
    device = _device(args.device)
    path = find_checkpoint(args.checkpoint, root)
    model, blob = load_clip(path, device, root)
    data = (blob.get("config") or {}).get("data") or {}
    coder = None
    if args.through:
        from .train import load_coder
        from .latents import decode_latent, picture_latent
        coder, _ = load_coder(find_checkpoint(args.through, root), device)
    sets = [(c, "val", 0.05) for c in data.get("corpora") or []]
    sets += [(h, "held out", 1.0) for h in data.get("holdouts") or []
             if (root / h / "manifest.json").is_file()]
    run = path.parent.name
    out = {"checkpoint": str(path), "run": run, "step": blob.get("step"),
           "through": args.through, "sets": []}
    print(f"{path} (step {blob.get('step')}): is each picture's own caption the closest of every "
          "caption its corpus can say?" + (f"  Pictures passed through {args.through} first."
                                           if coder else ""))
    floors = getattr(model, "none_fits", None) or {}
    print("  none-fits floor: " + (", ".join(f"{k} {v:.4f}" for k, v in floors.items()) if floors
                                   else "none calibrated (pictures clip-calibrate): closed set only"))
    print(f"\n  {'pictures':<36} {'n':>5} {'reader':>7} {'model':>7} {'closed':>7} {'show_me':>8} "
          f"{'agree':>7} {'kappa':>6}  model's misses differ in")
    for corpus, split, frac in sets:
        ds = ImageCaptions(root / corpus, split="val", val_frac=frac)
        if getattr(ds.manifest, "kind", "shapes") == "coco":
            print(f"  {corpus}: photos have no reader to agree with; see pictures coco-report")
            continue
        idx = ds.index[: args.n]
        pics = np.asarray(ds.images[idx])
        if coder is not None:
            import torch

            from ..vision.image import from_tensor, to_tensor
            x = torch.from_numpy(to_tensor(pics)).to(device)
            with torch.no_grad():
                pics = from_tensor(decode_latent(coder, picture_latent(coder, x)).cpu().numpy())
        facts = [ds.manifest.facts[int(i)] for i in idx]
        kind = getattr(ds.manifest, "kind", "shapes")
        cands = all_facts(kind)
        r = rank(model, pics, facts, cands, device=device, floor=floors.get(kind))
        # the floor is a cosine measured against CANONICAL captions; another phrasing sits on
        # its own scale, so the held-out phrasing is read closed-set
        rs = rank(model, pics, facts, cands, phrasing="show_me", device=device)
        closed = r.get("closed_top1", r["top1"])
        reader_ok = [judge(p, f)["all"] for p, f in zip(pics, facts, strict=True)]
        a = agreement(r["top1"], reader_ok)
        miss = {}
        for ok, b, f in zip(closed, r["best"], facts, strict=True):
            if not ok:
                for attr in ATTRS:
                    if attr in f and cands[b][attr] != f[attr]:
                        miss[attr] = miss.get(attr, 0) + 1
        row = {"pictures": corpus, "split": split, "kind": kind, "n": len(idx),
               "reader": float(np.mean(reader_ok)), "top1": float(np.mean(r["top1"])),
               "show_me": float(np.mean(rs["top1"])), "cos": float(np.mean(r["cos"])),
               "closed_top1": float(np.mean(closed)), "floor": floors.get(kind),
               "none_fits": int(sum(not f for f in r.get("fits", []))),
               "closed_agreement": agreement(closed, reader_ok),
               "agreement": a, "misses_by_attribute": miss,
               "correct": "".join("1" if t else "0" for t in r["top1"]),
               "reader_correct": "".join("1" if t else "0" for t in reader_ok)}
        out["sets"].append(row)
        kappa = "–" if a["kappa"] is None else f"{a['kappa']:.2f}"
        print(f"  {corpus + ' (' + split + ')':<36} {len(idx):>5} {row['reader'] * 100:>6.1f}% "
              f"{row['top1'] * 100:>6.1f}% {row['closed_top1'] * 100:>6.1f}% "
              f"{row['show_me'] * 100:>7.1f}% {a['agree'] * 100:>6.1f}% "
              f"{kappa:>6}  " + (", ".join(f"{k} {v}" for k, v in miss.items()) or "–"))
    stem = f"clip-{run}-step{blob.get('step')}" + (f"-through-{Path(args.through).stem if args.through.endswith('.pt') else args.through}"
                                                  if args.through else "")
    f = root / RESULTS / f"{stem}.json"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(out, indent=1))
    print(f"\nwrote {f}\n  reader = the rule-based reader on the same pictures (100% on originals: "
          "the positive control);\n  model = our contrastive model's closest caption is the right "
          "one\n  (and fits above the floor: below it, 'none fits'); closed = the same without the "
          "floor; agree/kappa = the two verdicts per picture.")
    return 0


def cmd_coco_report(args) -> int:
    """G4c: draw a photo for each of the first N COCO val captions, scored by clip-coco."""
    from .clip import load_clip
    from .photos import gallery, public, report
    root = _root()
    device = _device(args.device)
    d = _load_drawer(args.checkpoint, device)
    cp = find_checkpoint(args.clip, root)
    clip, _ = load_clip(cp, device, root)
    r = report(d, clip, root / args.corpus, n=args.n, guidance=args.guidance, seed=args.seed,
               steps=args.steps, device=device, clip_path=str(cp.relative_to(root)
                                                             if cp.is_relative_to(root) else cp))
    run = d.path.parent.name
    stem = f"coco-{run}-step{d.step}-g{args.guidance:g}-n{args.n}"
    gallery(r, root / RESULTS / f"{stem}.png")
    print(f"{d.path} (step {d.step}): {args.n} COCO val captions, one drawing each, guidance "
          f"{args.guidance:g}, {r['seconds']}s; judged by {cp}")
    print(f"\n  {'':<16} {'match':>7} {'mismatch':>9} {'top-1':>7} {'top-5':>7} {'top-10':>7}")
    for name in ("real", "drawn"):
        x = r[name]
        print(f"  {name + (' photo' if name == 'real' else ''):<16} {x['match']:>7.3f} "
              f"{x['mismatch']:>9.3f} {x['top1'] * 100:>6.1f}% {x['top5'] * 100:>6.1f}% "
              f"{x['top10'] * 100:>6.1f}%")
    c = r["chance"]
    print(f"  {'chance':<16} {'':>7} {'':>9} {c['top1'] * 100:>6.1f}% {c['top5'] * 100:>6.1f}% "
          f"{c['top10'] * 100:>6.1f}%")
    dist = r["distance"]
    print(f"\n  distance to real photos in our features (Frechet, NOT FID): drawings "
          f"{dist['drawn_vs_real']:.4f}; two halves of the real photos {dist['real_half_vs_half']:.4f}")
    out = public(r) | {"checkpoint": str(d.path), "run": run, "step": d.step,
                       "gallery": [f"{stem}.png"]}
    f = root / RESULTS / f"{stem}.json"
    f.write_text(json.dumps(out, indent=1))
    print(f"\nwrote {f}  (gallery {stem}.png: real photos above, drawings below)")
    return 0


def cmd_coco_compare(args) -> int:
    from .photos import compare
    a, b = (json.loads(Path(f).read_text()) for f in (args.a, args.b))
    r = compare(a, b)
    p = "–" if r["p"] is None else f"{r['p']:.3g}"
    print(f"top-5, n {r['n']}   a {r['a'] * 100:5.1f}%   b {r['b'] * 100:5.1f}%   "
          f"b only {r['b_only']}   a only {r['a_only']}   p {p}\n\na = {args.a}\nb = {args.b}")
    return 0


def cmd_gen_compare(args) -> int:
    from .score import compare
    a, b = (json.loads(Path(f).read_text()) for f in (args.a, args.b))
    for part, r in compare(a, b, by_fact=args.by_fact).items():
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
    s = sub.add_parser("scenes", help="G4b: render the scenes corpus (size and place as well)")
    s.add_argument("--out", default="data/vision/scenes64k")
    s.add_argument("--n", type=int, default=64000)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(fn=cmd_scenes)
    s = sub.add_parser("coco", help="G4c: pack COCO 2017 photos + captions at 64x64")
    s.add_argument("--raw", default="data/coco/raw", help="folder with the three downloaded zips")
    s.add_argument("--out", default="data/vision/coco64")
    s.add_argument("--splits", default="val,train")
    s.add_argument("--limit", type=int, default=0, help="first N photos per split (0 = all)")
    s.add_argument("--workers", type=int, default=12)
    s.set_defaults(fn=cmd_coco)
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
        s.add_argument("checkpoint", help="a .pt, or a G1 / G2 run name")
        if name == "draw":
            s.add_argument("caption", help="three words: count colour shape, e.g. 'two red circles'")
            s.add_argument("--n", type=int, default=8)
            s.add_argument("--out", default=None, help="PNG path (default logs/pictures/draw/)")
        else:
            s.add_argument("--k", type=int, default=8, help="pictures per caption")
            s.add_argument("--phrasing", default=None,
                           help="G4a: say the 60 captions in this phrasing (pictures/phrases.py: "
                                "a training one such as picture_of, or held-out show_me / "
                                "in_colour); default: the three-word captions")
        s.add_argument("--guidance", type=float, default=3.0, help="1 = no guidance")
        s.add_argument("--temperature", type=float, default=1.0)
        s.add_argument("--top-k", type=int, default=None)
        s.add_argument("--steps", type=int, default=None, help="G2/G3: Euler steps (default: the config's)")
        s.add_argument("--snap", action="store_true",
                       help="G3: re-quantize each drawn latent onto the codebooks before decoding")
        s.add_argument("--seed", type=int, default=0)
        s.add_argument("--device", default="cpu", help="cpu (default), cuda or auto")
        if name == "gen-report":
            s.add_argument("--clip", default=None,
                           help="G4b: also score every picture with our contrastive model "
                                "(a .pt or run name, e.g. clip-shapes), beside the reader")
        s.set_defaults(fn=fn)
    s = sub.add_parser("coco-report", help="G4c: draw from COCO val captions, judged by clip-coco")
    s.add_argument("checkpoint", help="a G4c flow model (.pt or run name, e.g. flow-coco)")
    s.add_argument("--clip", default="clip-coco", help="the contrastive model that judges")
    s.add_argument("--corpus", default="data/vision/coco64-val")
    s.add_argument("--n", type=int, default=1000, help="captions (one drawing each)")
    s.add_argument("--guidance", type=float, default=3.0)
    s.add_argument("--steps", type=int, default=None)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--device", default="cpu")
    s.set_defaults(fn=cmd_coco_report)
    s = sub.add_parser("coco-compare", help="G4c: two coco-reports on the same captions, paired")
    s.add_argument("a")
    s.add_argument("b")
    s.set_defaults(fn=cmd_coco_compare)
    s = sub.add_parser("clip-calibrate", help="G4b: measure the none-fits floor on clean originals")
    s.add_argument("checkpoint", help="a .pt, or a run name (clip-shapes)")
    s.add_argument("--q", type=float, default=0.01,
                   help="share of clean originals the floor refuses (default 0.01)")
    s.add_argument("--n", type=int, default=2048, help="val originals per corpus (the last ones)")
    s.add_argument("--device", default="cpu")
    s.set_defaults(fn=cmd_clip_calibrate)
    s = sub.add_parser("clip-report", help="G4b: our contrastive model on known pictures, beside the reader")
    s.add_argument("checkpoint", help="a .pt, or a run name (clip-shapes)")
    s.add_argument("--n", type=int, default=512, help="pictures per set")
    s.add_argument("--through", default=None,
                   help="pass every picture through this coder first (e.g. ae-shapes): damaged "
                        "pictures, where the reader is not 100%%")
    s.add_argument("--device", default="cpu")
    s.set_defaults(fn=cmd_clip_report)
    s = sub.add_parser("latents", help="G3: encode a picture corpus to normalised tokenizer latents")
    s.add_argument("tokenizer", help="a .pt, or a tokenizer run name")
    s.add_argument("--corpus", default="data/vision/shapes64k")
    s.add_argument("--books", type=int, default=None, help="codebooks summed into the latent (default: all)")
    s.add_argument("--out", default=None, help="default: data/pictures/latents-<run>")
    s.add_argument("--device", default="cpu")
    s.set_defaults(fn=cmd_latents)
    s = sub.add_parser("gen-compare", help="two gen-reports, paired per (caption, sample)")
    s.add_argument("a")
    s.add_argument("b")
    s.add_argument("--by-fact", action="store_true",
                   help="G4a: pair two phrasings of the same facts (the captions differ)")
    s.set_defaults(fn=cmd_gen_compare)
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
