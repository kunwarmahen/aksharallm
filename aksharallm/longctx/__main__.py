"""`python -m aksharallm.longctx` — make a model read further, and check whether it worked.

    # what does the context actually look like right now?
    python -m aksharallm.longctx curve small-code --len 2048

    # compare every scaling method at 4x, on the same windows, in one table
    python -m aksharallm.longctx sweep small-code --factor 4 --methods none linear ntk yarn

    # write an extended checkpoint (weights untouched)
    python -m aksharallm.longctx extend small-code --method yarn --factor 4 \
        --out checkpoints/small-code/ckpt_yarn4.pt

    # can it still find something back there?
    python -m aksharallm.longctx needle small-code --lengths 512 1024 2048

The portal's **Context** tab drives the same functions. This exists so the numbers are
available over ssh, and because `sweep` is the one command that answers the question the
whole chapter is about.

Read with: docs/18-long-context.md -- the chapter this implements; it ends with the order to
read these files in.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from ..infer.checkpoints import CheckpointStore, InferError, repo_root
from ..model.rope import METHODS, RopeScaling
from ..tokenizer.tokenizer import Tokenizer
from .curve import cliff, position_curve
from .extend import default_out_name, describe, extend, plan_extension
from .haystack import DEFAULT_DEPTHS, run as haystack_run

#: Where a sweep leaves its JSON, next to every other measurement this repo keeps.
RESULTS = "logs/longctx"

#: Room reserved past the requested context for the answer being scored. Four-digit codes
#: are a few tokens; 32 is slack, and slack is free here.
CANDIDATE_MARGIN = 32


def _load(args, method: str | None = None, factor: float = 1.0,
          max_seq_len: int | None = None):
    """The checkpoint, optionally re-configured for a different context on the way in.

    Re-configuring at load time is what makes `sweep` cheap: the weights are read once and
    only the config differs between methods, which is the honest way to compare them
    anyway — same model, same windows, one variable.

    `original_max_seq_len` is set from the checkpoint rather than derived from
    `max_seq_len / factor`. Those agree only when you happen to measure at exactly
    `trained x factor`, and every off-by-one there silently changes which method is being
    measured while the table still prints.
    """
    from ..config import ModelConfig
    from ..infer.cli import resolve_tokenizer
    from ..model.transformer import Transformer

    store = CheckpointStore(args.root)
    path = store.resolve(*store.identify(args.checkpoint).split("/"))
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    mcfg = dict(ckpt["model_config"])
    trained = _trained_window(ckpt)
    if method:
        mcfg["rope_scaling"] = RopeScaling(type=method, factor=factor,
                                           original_max_seq_len=trained).__dict__
    if max_seq_len is not None:
        mcfg["max_seq_len"] = int(max_seq_len)
    if getattr(args, "window", None):
        mcfg["attn_window"] = int(args.window)
        mcfg["attn_sinks"] = int(args.sinks)

    model = Transformer(ModelConfig(**mcfg))
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval().to(args.device)
    tok = Tokenizer(resolve_tokenizer(ckpt, args.tokenizer))
    return model, tok, ckpt, path


def _val_bin(args, ckpt) -> str:
    if args.val_bin:
        return args.val_bin
    path = (ckpt.get("config") or {}).get("data", {}).get("val_bin")
    if not path:
        raise InferError("this checkpoint does not record its val_bin; pass --val-bin")
    return path


def _trained_window(ckpt) -> int:
    mcfg = ckpt["model_config"]
    sc = mcfg.get("rope_scaling") or {}
    return int(sc.get("original_max_seq_len") or mcfg["max_seq_len"])


def _check_device(args) -> None:
    """Refuse `--device cuda` while a training run owns the card, unless told twice.

    Every measurement here is a *long* forward pass — 4,096 tokens of logits is half a
    gigabyte before the loss is computed — and the run holding the card has ~3 GB spare.
    This is the same policy `infer/engine.py` applies to the Playground, with the same
    reasoning: a measurement is never worth killing a six-day run at 3am. `--force-gpu`
    is there because the user is the one who decides, and sometimes the run has just been
    stopped and the pid file has not caught up.
    """
    if not str(args.device).startswith("cuda") or getattr(args, "force_gpu", False):
        return
    root = Path(args.root) if args.root else repo_root()
    live = [p.parent.name for p in (root / "checkpoints").glob("*/train.pid")
            if _alive(p)]
    if live:
        raise InferError(
            f"{', '.join(live)} is training and holding the GPU. Re-run with "
            f"--device cpu (slower, always safe), or --force-gpu if you know there is room.")


def _alive(pid_file: Path) -> bool:
    try:
        import os
        os.kill(int(pid_file.read_text().strip()), 0)
        return True
    except (ValueError, OSError):
        return False


def _progress(quiet: bool):
    if quiet:
        return None
    def show(done, total, label):
        print(f"\r  {label}: {done}/{total}", end="", flush=True)
        if done >= total:
            print()
    return show


# ---- commands ---------------------------------------------------------------------------

def cmd_curve(args) -> int:
    model, tok, ckpt, _ = _load(args, args.method, args.factor, args.len)
    scaling = model.cfg.rope_scaling
    curve = position_curve(model, _val_bin(args, ckpt), args.len, bucket=args.bucket,
                           n_windows=args.windows, batch_size=args.batch,
                           device=args.device, progress=_progress(args.quiet))
    trained = _trained_window(ckpt)
    print(f"\n{args.checkpoint}  window {args.len}  trained on {trained}  "
          f"scaling {scaling.describe(args.len)}")
    print(f"{'positions':>14} {'loss':>8} {'ppl':>10}")
    for b in curve["buckets"]:
        edge = " <- trained window ends here" if b["start"] == trained else ""
        print(f"{b['start']:>6}-{b['end']:<7} {b['loss']:>8.3f} {b['perplexity']:>10.1f}{edge}")
    broke = cliff(curve, trained)
    print(f"\noverall loss {curve['loss']:.4f}  ppl {curve['perplexity']:.2f}")
    print(f"cliff: at position {broke['position']} (+{broke['excess']:.2f} nats over the "
          f"in-window baseline {broke['baseline']:.2f})" if broke
          else "cliff: none — the curve never breaks")
    return 0


def cmd_sweep(args) -> int:
    """Every method, same windows, one table. The command this module exists for."""
    trained = None
    rows = []
    for method in args.methods:
        factor = 1.0 if method == "none" else args.factor
        model, tok, ckpt, _ = _load(args, None if method == "none" else method,
                                    factor, args.len)
        trained = trained or _trained_window(ckpt)
        curve = position_curve(model, _val_bin(args, ckpt), args.len, bucket=args.bucket,
                               n_windows=args.windows, batch_size=args.batch,
                               device=args.device, progress=_progress(args.quiet))
        broke = cliff(curve, trained)
        rows.append({"method": method, "factor": factor, "curve": curve,
                     "cliff": broke})
        del model

    print(f"\n{args.checkpoint}: trained on {trained}, measured at {args.len} "
          f"({args.windows} windows)\n")
    inside = [b for b in rows[0]["curve"]["buckets"] if b["end"] < trained]
    print(f"{'method':>8} {'overall':>9} {'in-window':>11} {'past it':>9} {'cliff at':>10}")
    for r in rows:
        bs = r["curve"]["buckets"]
        ins = [b["loss"] for b in bs if b["end"] < trained]
        out = [b["loss"] for b in bs if b["start"] >= trained]
        print(f"{r['method']:>8} {r['curve']['loss']:>9.3f} "
              f"{(sum(ins) / len(ins)) if ins else float('nan'):>11.3f} "
              f"{(sum(out) / len(out)) if out else float('nan'):>9.3f} "
              f"{(r['cliff']['position'] if r['cliff'] else '—'):>10}")
    print(f"\n({len(inside)} buckets inside the trained window, "
          f"{len(rows[0]['curve']['buckets']) - len(inside)} past it)")

    out = _write(args, "sweep", {"checkpoint": args.checkpoint, "trained": trained,
                                 "seq_len": args.len, "rows": rows})
    print(f"written to {out}")
    return 0


def cmd_extend(args) -> int:
    store = CheckpointStore(args.root)
    src = store.resolve(*store.identify(args.checkpoint).split("/"))
    if args.dry_run:
        ckpt = torch.load(src, map_location="cpu", weights_only=False)
        after = plan_extension(ckpt["model_config"], args.method, args.factor,
                               args.original)
        for line in describe(ckpt["model_config"], after):
            print(f"  {line}")
        print("\n(dry run — nothing written)")
        return 0
    out = args.out or str(default_out_name(src, args.method, args.factor))
    result = extend(src, out, args.method, args.factor, args.original,
                    window=args.window, sinks=args.sinks)
    for line in result["changes"]:
        print(f"  {line}")
    print(f"\nwrote {result['out']} — same weights, {result['trained_window']} -> "
          f"{result['addressable']} addressable positions")
    return 0


def cmd_needle(args) -> int:
    # The probe's candidate tokens are appended *after* the haystack, so the model has to
    # be able to address a little past the longest context asked for. Without this margin
    # the sweep dies on its last cell with an off-by-five.
    longest = max(args.lengths)
    model, tok, ckpt, _ = _load(args, args.method, args.factor,
                                longest + CANDIDATE_MARGIN)
    grid = haystack_run(model, tok, _val_bin(args, ckpt), args.lengths,
                        depths=args.depths, trials=args.trials,
                        n_candidates=args.candidates, device=args.device,
                        progress=_progress(args.quiet))
    print(f"\n{args.checkpoint}: needle in a haystack, {args.trials} trials/cell, "
          f"{args.candidates}-way choice (chance {grid['chance']:.0%})\n")
    head = "  depth  " + "".join(f"{n:>9}" for n in args.lengths)
    print(head)
    for row in grid["grid"]:
        cells = "".join(
            f"{c['accuracy']:>8.0%} " if c["accuracy"] is not None else f"{'—':>9}"
            for c in row)
        print(f"  {row[0]['depth']:>5.0%}  {cells}")
    acc, se = grid["accuracy"], grid["stderr"]
    print(f"\noverall {acc:.1%}" + (f" ± {se:.1%}" if se else "")
          + f"  (chance {grid['chance']:.0%})")
    if acc is not None and se and acc - 2 * se <= grid["chance"]:
        print("NOT distinguishable from chance — see docs/18 on why that is still a result.")
    print(f"written to {_write(args, 'needle', grid)}")
    return 0


def _write(args, kind: str, payload: dict) -> Path:
    root = Path(args.root) if args.root else repo_root()
    out = root / RESULTS
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{args.checkpoint.replace('/', '-')}-{kind}-{int(time.time())}.json"
    path.write_text(json.dumps(payload, indent=2, default=str))
    return path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m aksharallm.longctx",
                                description=__doc__.split("\n")[0])
    p.add_argument("--root", default=None, help="repo root (defaults to this one)")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp, with_method=True):
        sp.add_argument("checkpoint")
        sp.add_argument("--device", default="cpu")
        sp.add_argument("--tokenizer", default=None)
        sp.add_argument("--val-bin", default=None)
        sp.add_argument("--window", type=int, default=None,
                        help="sliding-window attention, in tokens")
        sp.add_argument("--sinks", type=int, default=4,
                        help="attention sinks kept visible (only with --window)")
        sp.add_argument("--quiet", action="store_true")
        sp.add_argument("--force-gpu", action="store_true",
                        help="use the GPU even while a run is training")
        if with_method:
            sp.add_argument("--method", choices=[m for m in METHODS if m != "none"],
                            default=None, help="apply a RoPE scaling at load time")
            sp.add_argument("--factor", type=float, default=4.0)

    c = sub.add_parser("curve", help="loss by position over one long window")
    common(c)
    c.add_argument("--len", type=int, default=2048)
    c.add_argument("--bucket", type=int, default=128)
    c.add_argument("--windows", type=int, default=16)
    c.add_argument("--batch", type=int, default=1)
    c.set_defaults(fn=cmd_curve)

    s = sub.add_parser("sweep", help="compare every scaling method on the same windows")
    common(s, with_method=False)
    s.add_argument("--methods", nargs="+", default=["none", "linear", "ntk", "yarn"],
                   choices=list(METHODS))
    s.add_argument("--factor", type=float, default=4.0)
    s.add_argument("--len", type=int, default=4096)
    s.add_argument("--bucket", type=int, default=256)
    s.add_argument("--windows", type=int, default=8)
    s.add_argument("--batch", type=int, default=1)
    s.set_defaults(fn=cmd_sweep)

    e = sub.add_parser("extend", help="write an extended checkpoint (weights untouched)")
    common(e, with_method=False)
    e.add_argument("--method", choices=list(METHODS), default="yarn")
    e.add_argument("--factor", type=float, default=4.0)
    e.add_argument("--original", type=int, default=None,
                   help="the window the weights were trained on, if not max_seq_len")
    e.add_argument("--out", default=None)
    e.add_argument("--dry-run", action="store_true")
    e.set_defaults(fn=cmd_extend)

    n = sub.add_parser("needle", help="needle in a haystack, length x depth")
    common(n)
    n.add_argument("--lengths", type=int, nargs="+", default=[512, 1024, 2048])
    n.add_argument("--depths", type=float, nargs="+", default=list(DEFAULT_DEPTHS))
    n.add_argument("--trials", type=int, default=3)
    n.add_argument("--candidates", type=int, default=4)
    n.set_defaults(fn=cmd_needle)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        _check_device(args)
        return args.fn(args)
    except (InferError, ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}")
        return 1


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())
