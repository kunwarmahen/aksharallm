"""`python -m aksharallm.interp` — look inside a checkpoint from a terminal.

    python -m aksharallm.interp lens small-code --prompt "The capital of France is"
    python -m aksharallm.interp attn small-code --prompt "def add(a, b):" --layer 12
    python -m aksharallm.interp patch small-code \
        --clean "The capital of France is" --corrupt "The capital of Italy is" \
        --answer " Paris" --other " Rome"
    python -m aksharallm.interp sae small-code --layer 12 --steps 2000
    python -m aksharallm.interp features small-code --layer 12 --feature 137

The portal's Interp tab drives the same functions; this exists so the answers are available
over ssh, and because a grid of numbers is often easier to read in a terminal than in a
browser.

Read with: docs/17-interpretability.md -- the chapter this implements; it ends with the order
to read these files in.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from ..infer.checkpoints import CheckpointStore, InferError, repo_root
from ..tokenizer.tokenizer import Tokenizer
from .capture import attention_maps, attention_summary, run
from .lens import layer_contributions, lens_story, logit_lens
from .patch import PatchError, patch_grid, summarise
from .sae import SAEConfig, collect_activations, feature_report, save, top_activating, train_sae


def load(args) -> tuple:
    from ..infer.cli import load_model, resolve_tokenizer

    store = CheckpointStore(args.root)
    path = store.resolve(*store.identify(args.checkpoint).split("/"))
    model, ckpt = load_model(str(path), device=args.device)
    return model, Tokenizer(resolve_tokenizer(ckpt, args.tokenizer)), path


def sae_path(root: Path | None, run: str, layer: int) -> Path:
    return (Path(root) if root else repo_root()) / "logs" / "interp" / f"{run}-layer{layer}.pt"


def cmd_lens(args) -> int:
    model, tok, _ = load(args)
    ids = tok.encode(args.prompt, bos=True)
    cap = run(model, ids, device=args.device)
    rows = logit_lens(model, cap, top=args.top)
    story = lens_story(rows, tok.decode)
    print(f"prompt   {args.prompt!r}  ({len(ids)} tokens)")
    print(f"answer   {story['answer_text']!r}")
    print(f"settled  {story['settled_label']}  ({story['flips']} changes of mind on the way)\n")
    for row in rows:
        top = "  ".join(f"{tok.decode([t['id']])!r} {t['prob']:.2f}" for t in row["top"])
        print(f"  {row['label']:>10}  H={row['entropy']:5.2f}  {top}")
    if args.contributions:
        print("\n  block   ||delta||   effect on the answer's logit")
        for c in layer_contributions(model, cap):
            print(f"  {c['layer']:>5}   {c['norm_delta']:8.2f}   {c['answer_delta']:+8.3f}")
    return 0


def cmd_attn(args) -> int:
    model, tok, _ = load(args)
    ids = tok.encode(args.prompt, bos=True)
    cap = run(model, ids, device=args.device)
    tokens = [tok.decode([i]) for i in ids]
    weights = attention_maps(model, cap, args.layer)
    print(f"layer {args.layer}, {weights.shape[0]} heads, {len(tokens)} tokens\n")
    for head in attention_summary(weights, tokens):
        where = ", ".join(f"{a['token']!r} {a['weight']:.2f}" for a in head["attends_to"])
        print(f"  head {head['head']:>2}  looks back {head['distance']:5.2f}  "
              f"self {head['self_weight']:.2f}  last token -> {where}")
    if args.head is not None:
        print(f"\n  head {args.head}, row by row (what each position attended to):")
        w = weights[args.head]
        for i, token in enumerate(tokens):
            row = " ".join(f"{v:4.2f}" for v in w[i][: i + 1].tolist())
            print(f"   {i:>3} {token!r:>14}  {row}")
    return 0


def cmd_patch(args) -> int:
    model, tok, _ = load(args)
    clean = tok.encode(args.clean, bos=True)
    corrupt = tok.encode(args.corrupt, bos=True)
    answer = tok.encode(args.answer)[0]
    other = tok.encode(args.other)[0]
    try:
        result = patch_grid(model, clean, corrupt, answer, other, device=args.device)
    except PatchError as exc:
        print(f"error: {exc}")
        return 2
    tokens = [tok.decode([i]) for i in clean]
    print(f"clean     {args.clean!r} -> {args.answer!r}")
    print(f"corrupt   {args.corrupt!r} -> {args.other!r}")
    print(f"logit diff: {result['corrupt_diff']:+.2f} corrupted, "
          f"{result['clean_diff']:+.2f} clean\n")
    header = "  block  " + " ".join(f"{t!r:>10}" for t in tokens)
    print(header)
    for li, row in enumerate(result["grid"]):
        cells = " ".join(f"{v:10.2f}" for v in row)
        print(f"  {li:>5}  {cells}")
    print(f"\n{summarise(result, tokens)}")
    return 0


def cmd_sae(args) -> int:
    from ..config import load_config
    from ..data.loader import TokenDataset

    model, tok, path = load(args)
    run_name = path.parent.name
    cfg_path = (Path(args.root) if args.root else repo_root()) / "configs" / f"{run_name}.yaml"
    if args.data:
        bin_path = args.data
    elif cfg_path.exists():
        cfg = load_config(str(cfg_path))
        bin_path = (cfg.data.train_sources[0]["bin"] if cfg.data.train_sources
                    else cfg.data.train_bin)
    else:
        print("error: pass --data <tokens.bin>; this checkpoint has no config to read one from")
        return 2

    print(f"activations from {bin_path} at layer {args.layer}")
    ds = TokenDataset(bin_path, min(args.seq_len, model.cfg.max_seq_len), args.device)
    batches = (ds.get_batch(args.batch_tokens // args.seq_len)[0]
               for _ in range(10_000))
    acts = collect_activations(model, batches, args.layer, device=args.device,
                               limit=args.n_acts)
    print(f"{acts.shape[0]:,} activations x {acts.shape[1]} dims "
          f"({acts.numel() * 4 / 1e9:.2f} GB)")

    cfg = SAEConfig(d_model=model.cfg.d_model, n_features=args.features, layer=args.layer,
                    alpha=args.alpha, lr=args.lr, steps=args.steps, batch=args.batch,
                    run=run_name)
    sae, history = train_sae(acts, cfg, device=args.device)
    report = feature_report(sae, acts, device=args.device)
    out = save(sae, sae_path(args.root, run_name, args.layer), history, report)
    print(f"\ndead features {report['dead']:,} of {report['n_features']:,} "
          f"({report['dead_fraction'] * 100:.0f}%)")
    print(f"saved {out}")
    print(f"look at one:  python -m aksharallm.interp features {args.checkpoint} "
          f"--layer {args.layer} --feature {report['features'][0]['id'] if report['features'] else 0}")
    return 0


def cmd_features(args) -> int:
    from ..config import load_config
    from ..data.loader import TokenDataset
    from .sae import load as load_sae

    model, tok, path = load(args)
    run_name = path.parent.name
    saved = sae_path(args.root, run_name, args.layer)
    if not saved.exists():
        print(f"no SAE at {saved} — train one first:\n"
              f"  python -m aksharallm.interp sae {args.checkpoint} --layer {args.layer}")
        return 2
    sae = load_sae(saved, device=args.device)

    cfg_path = (Path(args.root) if args.root else repo_root()) / "configs" / f"{run_name}.yaml"
    cfg = load_config(str(cfg_path))
    bin_path = args.data or (cfg.data.train_sources[0]["bin"] if cfg.data.train_sources
                             else cfg.data.train_bin)
    ds = TokenDataset(bin_path, min(256, model.cfg.max_seq_len), args.device)
    batches = [ds.get_batch(1)[0] for _ in range(args.samples)]
    rows = top_activating(model, sae, args.feature, batches, args.layer, tok.decode,
                          top=args.top, device=args.device)
    print(f"feature {args.feature} of layer {args.layer}, top {len(rows)} activations:\n")
    for row in rows:
        print(f"  {row['activation']:6.2f}  ...{row['context']!r} [{row['token']!r}] "
              f"{row['after']!r}...")
    if not rows:
        print("  it never fired on this sample — try more --samples, or a busier feature")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m aksharallm.interp",
        description="Look inside a trained model: what each layer believed, what attention "
                    "attended to, which activation carries a fact, and what a sparse "
                    "dictionary finds in the residual stream.")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--root", default=None)
    ap.add_argument("--tokenizer", default=None)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("checkpoint", help="run name, id or path")
        return p

    lens = common(sub.add_parser("lens", help="what each layer would have predicted"))
    lens.add_argument("--prompt", default="The capital of France is")
    lens.add_argument("--top", type=int, default=5)
    lens.add_argument("--contributions", action="store_true",
                      help="also show how much each block moved the residual stream")
    lens.set_defaults(fn=cmd_lens)

    attn = common(sub.add_parser("attn", help="what attention attended to"))
    attn.add_argument("--prompt", default="The capital of France is")
    attn.add_argument("--layer", type=int, default=0)
    attn.add_argument("--head", type=int, default=None, help="print this head's full matrix")
    attn.set_defaults(fn=cmd_attn)

    patch = common(sub.add_parser("patch", help="which activation carries the difference"))
    patch.add_argument("--clean", required=True)
    patch.add_argument("--corrupt", required=True)
    patch.add_argument("--answer", required=True, help="the clean prompt's answer, e.g. ' Paris'")
    patch.add_argument("--other", required=True, help="the corrupted prompt's answer")
    patch.set_defaults(fn=cmd_patch)

    sae = common(sub.add_parser("sae", help="train a sparse autoencoder on one layer"))
    sae.add_argument("--layer", type=int, default=12)
    sae.add_argument("--features", type=int, default=None)
    sae.add_argument("--alpha", type=float, default=3e-3)
    sae.add_argument("--lr", type=float, default=1e-3)
    sae.add_argument("--steps", type=int, default=2000)
    sae.add_argument("--batch", type=int, default=4096)
    sae.add_argument("--n-acts", type=int, default=500_000, help="activations to collect")
    sae.add_argument("--seq-len", type=int, default=256)
    sae.add_argument("--batch-tokens", type=int, default=8192)
    sae.add_argument("--data", default=None, help="a .bin of tokens (default: the run's own)")
    sae.set_defaults(fn=cmd_sae)

    feats = common(sub.add_parser("features", help="what one SAE feature fires on"))
    feats.add_argument("--layer", type=int, default=12)
    feats.add_argument("--feature", type=int, required=True)
    feats.add_argument("--samples", type=int, default=200)
    feats.add_argument("--top", type=int, default=10)
    feats.add_argument("--data", default=None)
    feats.set_defaults(fn=cmd_features)

    args = ap.parse_args(argv)
    if args.cmd == "sae" and args.features is None:
        args.features = 0            # filled in once the model is loaded (8x d_model)
    try:
        if args.cmd == "sae":
            from ..infer.cli import load_model
            store = CheckpointStore(args.root)
            path = store.resolve(*store.identify(args.checkpoint).split("/"))
            peek = torch.load(path, map_location="meta", weights_only=False)["model_config"]
            args.features = args.features or peek["d_model"] * 8
        return args.fn(args)
    except InferError as exc:
        print(f"error: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
