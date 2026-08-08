"""`python -m aksharallm.diffusion` — generate, infill and measure, from a terminal.

The Diffusion tab in the portal calls exactly these functions, so anything you can see in
a browser you can reproduce here and paste into a note::

    python -m aksharallm.diffusion tiny-diffusion/ckpt_best.pt generate \\
        --prompt "Once upon a time" --length 64 --steps 32 --show-trace
    python -m aksharallm.diffusion tiny-diffusion/ckpt_best.pt infill \\
        --prefix "The cat sat" --suffix "and fell asleep." --length 12
    python -m aksharallm.diffusion tiny-diffusion/ckpt_best.pt elbo --batches 20
    python -m aksharallm.diffusion tiny-diffusion/ckpt_best.pt by-t

Read with: docs/20-diffusion.md -- the chapter this implements; it ends with the order to read
these files in.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

from ..config import ModelConfig
from ..data.loader import TokenDataset
from ..infer.checkpoints import CheckpointStore, InferError
from ..model.transformer import Transformer
from ..tokenizer.tokenizer import Tokenizer
from .evaluate import elbo, loss_by_t
from .generate import DiffusionError, decode_with_masks, diffusion_generate, infill

#: Where the JSON measurements land, so the tab and the terminal read one set of files.
RESULTS = Path("logs/diffusion")


def load(ckpt_id: str, device: str):
    """Load a checkpoint and its tokenizer, refusing anything that is not a diffusion model."""
    try:
        store = CheckpointStore(Path.cwd())
        path = store.get(store.identify(ckpt_id)).path
    except InferError:
        # Not under checkpoints/ — take it as a plain path, which is what a smoke test
        # writing to /tmp produces.
        path = Path(ckpt_id)
        if not path.is_file():
            raise
    blob = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ModelConfig(**blob["model_config"])
    if not cfg.is_diffusion:
        raise DiffusionError(
            f"{path} is an autoregressive checkpoint (causal attention, no mask token). "
            "This CLI only drives masked diffusion models — see docs/20.")
    model = Transformer(cfg)
    model.load_state_dict(blob["model"])
    model = model.to(device=device, dtype=torch.bfloat16 if device == "cuda"
                     else torch.float32).eval()
    tok_path = ((blob.get("config") or {}).get("data") or {}).get("tokenizer")
    if not tok_path:
        raise DiffusionError(f"{path} does not record its tokenizer, so it cannot be decoded.")
    return model, Tokenizer(tok_path), blob


def _device(arg: str) -> str:
    if arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return arg


def _write(name: str, blob: dict) -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / f"{name}-{time.strftime('%Y%m%d-%H%M%S')}.json"
    path.write_text(json.dumps(blob, indent=2))
    return path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m aksharallm.diffusion",
                                 description="Generate from and measure a masked diffusion "
                                             "language model.")
    ap.add_argument("checkpoint", help="path to a diffusion .pt (or checkpoints/-relative id)")
    ap.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="denoise a fresh sequence into text")
    g.add_argument("--prompt", default="", help="a prefix the model keeps (may be empty)")
    g.add_argument("--length", type=int, default=64)
    g.add_argument("--steps", type=int, default=32)
    g.add_argument("--temperature", type=float, default=0.8)
    g.add_argument("--top-k", type=int, default=50)
    g.add_argument("--top-p", type=float, default=0.95)
    g.add_argument("--remask", default="low_confidence",
                   choices=("low_confidence", "random"))
    g.add_argument("--seed", type=int, default=None)
    g.add_argument("--show-trace", action="store_true",
                   help="print the sequence after every denoising step")

    f = sub.add_parser("infill", help="write the middle, given both ends")
    f.add_argument("--prefix", required=True)
    f.add_argument("--suffix", required=True)
    f.add_argument("--length", type=int, default=16)
    f.add_argument("--steps", type=int, default=16)
    f.add_argument("--temperature", type=float, default=0.8)
    f.add_argument("--seed", type=int, default=None)

    e = sub.add_parser("elbo", help="the validation bound, on the run's own val split")
    e.add_argument("--batches", type=int, default=20)
    e.add_argument("--batch-size", type=int, default=8)
    e.add_argument("--repeats", type=int, default=1)
    e.add_argument("--val-bin", default=None)

    b = sub.add_parser("by-t", help="cross-entropy against how much was masked")
    b.add_argument("--batches", type=int, default=4)
    b.add_argument("--batch-size", type=int, default=8)
    b.add_argument("--buckets", type=int, default=10)
    b.add_argument("--val-bin", default=None)

    args = ap.parse_args(argv)
    device = _device(args.device)
    try:
        model, tok, blob = load(args.checkpoint, device)
    except (DiffusionError, InferError, OSError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    mask_id = int(model.cfg.mask_token_id)

    if args.cmd == "generate":
        prefix = tok.encode(args.prompt, bos=True) if args.prompt else [tok.bos_id]
        t0 = time.monotonic()
        ids, trace = diffusion_generate(
            model, length=args.length, steps=args.steps, prefix=prefix,
            temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
            remask=args.remask, seed=args.seed, device=device, trace=args.show_trace)
        elapsed = time.monotonic() - t0
        if args.show_trace:
            for st in trace:
                print(f"[{st.step:>3}] {decode_with_masks(tok, st.ids, mask_id)!r}")
            print()
        print(decode_with_masks(tok, ids, mask_id))
        print(f"\n{len(ids)} tokens in {args.steps} forward passes, {elapsed:.2f}s "
              f"({len(ids) / max(args.steps, 1):.1f} tokens per pass)")
        return 0

    if args.cmd == "infill":
        prefix = tok.encode(args.prefix, bos=True)
        suffix = tok.encode(args.suffix)
        middle, _ = infill(model, prefix, suffix, length=args.length, steps=args.steps,
                           temperature=args.temperature, seed=args.seed, device=device)
        print(f"{args.prefix} [{decode_with_masks(tok, middle, mask_id)}] {args.suffix}")
        return 0

    val_bin = args.val_bin or ((blob.get("config") or {}).get("data") or {}).get("val_bin")
    if not val_bin or not Path(val_bin).is_file():
        print(f"error: no validation data ({val_bin!r}); pass --val-bin", file=sys.stderr)
        return 2
    seq_len = ((blob.get("config") or {}).get("train") or {}).get(
        "seq_len", model.cfg.max_seq_len)
    ds = TokenDataset(val_bin, seq_len, device)

    if args.cmd == "elbo":
        out = elbo(model, ds, args.batch_size, args.batches, repeats=args.repeats)
        out["checkpoint"] = str(args.checkpoint)
        print(f"NELBO      {out['nelbo']:.4f} nats/token   (upper bound on the true NLL)")
        print(f"ppl bound  {out['ppl_upper_bound']:.2f}         (NOT an AR perplexity)")
        print(f"ce masked  {out['ce_masked']:.4f}         (unweighted, on masked positions)")
        print(f"\nwritten to {_write('elbo', out)}")
        return 0

    rows = loss_by_t(model, ds, args.batch_size, args.batches, buckets=args.buckets)
    print(f"{'mask rate':>10}  {'ce (nats)':>10}")
    for r in rows:
        bar = "#" * int(r["ce_masked"] * 6)
        print(f"{r['t'] * 100:>9.0f}%  {r['ce_masked']:>10.4f}  {bar}")
    out = {"checkpoint": str(args.checkpoint), "rows": rows}
    print(f"\nwritten to {_write('by-t', out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
