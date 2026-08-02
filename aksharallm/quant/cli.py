"""Command line for quantization.

    python -m aksharallm.quant tiny/ckpt_best.pt --bits 4
    python -m aksharallm.quant tiny/ckpt_best.pt --method gptq --bench
    python -m aksharallm.quant tiny/ckpt_best.pt --compare      # every method, one table

`--compare` is the one to reach for first. A single scheme's numbers in isolation say
almost nothing: "perplexity 4.43" is meaningless without the bf16 number beside it, and
"int4 works fine" is a claim about a particular group size and a particular method. The
table puts all of them next to each other on the same evaluation batches.

Read with: docs/10-quantization.md -- the chapter this implements; it ends with the order to
read these files in.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

from ..config import ModelConfig
from ..infer.checkpoints import CheckpointStore, stage_for
from ..model.transformer import Transformer
from ..train import stopfile
from . import bench as bench_mod
from .awq import apply_awq
from .calib import collect
from .convert import (
    is_quantized_checkpoint,
    quantize_model,
    save_quantized,
)
from .gptq import make_gptq_quantizer
from .qat import convert_qat, prepare_qat, train_qat
from .qlinear import QuantLinear
from .qtensor import QuantScheme

#: What --compare runs. Chosen so each row isolates one variable against the row above:
#: bits, then group size, then method.
COMPARE = (
    ("rtn", QuantScheme(bits=8, group_size=64, sym=True, method="rtn")),
    ("rtn", QuantScheme(bits=4, group_size=64, sym=False, method="rtn")),
    ("rtn", QuantScheme(bits=4, group_size=128, sym=False, method="rtn")),
    ("rtn", QuantScheme(bits=4, group_size=-1, sym=False, method="rtn")),
    # Same bits, same group, same method as the second row -- only the *grid* differs.
    # That pairing is the point: it isolates what the NF4 levels are worth on their own.
    ("rtn", QuantScheme(bits=4, group_size=64, dtype="nf4", method="rtn")),
    ("rtn", QuantScheme(bits=4, group_size=64, dtype="nf4", double_quant=True, method="rtn")),
    ("awq", QuantScheme(bits=4, group_size=64, sym=False, method="awq")),
    ("gptq", QuantScheme(bits=4, group_size=64, sym=False, method="gptq")),
    ("gptq", QuantScheme(bits=4, group_size=64, dtype="nf4", method="gptq")),
    ("gptq", QuantScheme(bits=4, group_size=-1, sym=False, method="gptq")),
)


def _resolve(ref: str) -> Path:
    p = Path(ref)
    if p.is_file():
        return p
    try:
        store = CheckpointStore()
        return store.get(store.identify(ref)).path
    except Exception as e:  # noqa: BLE001
        raise SystemExit(f"cannot find checkpoint {ref!r}: {e}")


def _load_float(path: Path, device: str) -> tuple[Transformer, dict]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if is_quantized_checkpoint(ckpt):
        raise SystemExit(
            f"{path} is already quantized ({ckpt['quant'].get('label')}). Quantizing a "
            "quantized model compounds the error — start from the float checkpoint.")
    model = Transformer(ModelConfig(**ckpt["model_config"]))
    model.load_state_dict(ckpt["model"])
    model = model.to(device=device, dtype=torch.bfloat16 if device == "cuda" else torch.float32)
    model.eval()
    return model, ckpt


def _val_bin(ckpt: dict, override: str | None) -> str | None:
    if override:
        return override
    v = ((ckpt.get("config") or {}).get("data") or {}).get("val_bin")
    return v if v and Path(v).is_file() else None


def _train_bin(ckpt: dict, src: Path, args) -> str:
    """Data for QAT. Prefers an explicit --train-bin, then the run's own train split,
    then the val split (with a warning -- fine-tuning on val makes the val number a lie).

    The refusal below is the important part. Quantization happens at the *end* of the
    pipeline -- base -> SFT -> DPO -> quantize -- so the checkpoint being quantized is
    usually a chat model. But `sft.py` carries the *base run's* data config forward (so
    downstream tools can still find the tokenizer), which means an SFT checkpoint's
    `train_bin` points at raw pretraining text.

    Left alone, `--method qat` on `sft_best.pt` would run several hundred steps of
    next-token prediction on FineWeb against a model that was fine-tuned to answer
    questions -- quietly undoing the SFT. It would not error. It would not even look
    wrong: perplexity is measured on that same pretraining split, so the number would
    *improve* while the model forgot how to hold a conversation.

    So for anything past the base stage, QAT requires the data to be named explicitly.
    """
    if args.train_bin:
        return args.train_bin

    stage = stage_for(src.name)
    if stage != "base":
        raise SystemExit(
            f"{src.name} is a {stage.upper()} checkpoint, and it records the *base run's*\n"
            f"training data — raw pretraining text. Running QAT on that would fine-tune a\n"
            f"chat model back towards next-token prediction and undo the {stage.upper()},\n"
            f"silently, while the perplexity number improved.\n\n"
            f"Pass the data this model was actually trained on:\n"
            f"    --train-bin data/sft/train.bin\n\n"
            f"Or quantize it without training: --method gptq is the strongest option that\n"
            f"never touches the weights' meaning.")

    data = (ckpt.get("config") or {}).get("data") or {}
    for key in ("train_bin",):
        v = data.get(key)
        if v and Path(v).is_file():
            return v
    srcs = data.get("train_sources") or []
    for source in srcs:          # not `src` — that is the checkpoint path parameter
        if Path(source.get("bin", "")).is_file():
            return source["bin"]
    v = data.get("val_bin")
    if v and Path(v).is_file():
        print("  warning: no training split on disk, fine-tuning on the VALIDATION split."
              "\n  The reported perplexity is then measured on data QAT has seen.",
              file=sys.stderr)
        return v
    raise SystemExit("QAT needs training data; pass --train-bin")


def _seq_len(ckpt: dict) -> int:
    return ((ckpt.get("config") or {}).get("train") or {}).get("seq_len", 512)


def _out_path(src: Path, scheme: QuantScheme, out: str | None) -> Path:
    if out:
        return Path(out)
    # Original stem first, so the stage prefix (ckpt_/sft_/dpo_) still parses: quantizing
    # does not change what a model has been trained to do.
    return src.with_name(f"{src.stem}-{scheme.label()}.pt")


def _calibrate(model, ckpt: dict, args, want_hessian: bool, stage: str = "base"):
    """Run calibration text through the model, or explain why we cannot."""
    val_bin = _val_bin(ckpt, args.calib_bin or args.val_bin)
    if not val_bin:
        raise SystemExit(
            "this method needs calibration data, and the checkpoint's val split is not on "
            "disk. Pass --calib-bin with a tokenized .bin from the same tokenizer.")
    # GPTQ and AWQ measure what each layer's inputs actually look like, so the
    # calibration set has to resemble the model's real traffic. A chat model sees ChatML
    # turns at inference; calibrating it on the pretraining prose its config still points
    # at measures the wrong activations. Not destructive like the QAT case above -- the
    # weights keep their meaning -- but it leaves quality on the table.
    if stage != "base" and not (args.calib_bin or args.val_bin):
        print(f"  note: {stage.upper()} checkpoint calibrating on {val_bin}, which is the "
              f"base run's\n  pretraining split. This model sees chat-formatted text at "
              f"inference, so\n  --calib-bin with SFT-format data will fit its activations "
              f"better.", file=sys.stderr)
    print(f"  calibrating on {args.calib_seqs} sequences from {val_bin}"
          f"{' (+Hessians)' if want_hessian else ''}...", flush=True)
    return collect(model, val_bin, _seq_len(ckpt), n_sequences=args.calib_seqs,
                   batch_size=args.calib_batch, device=args.device,
                   want_hessian=want_hessian)


def _qat_report(model, scheme: QuantScheme, seconds: float):
    """Rebuild a QuantReport after QAT.

    QAT converts its own layers, so it does not go through `quantize_model` and has to
    account for itself. It must count the layers it *left in float* too -- omitting the
    tied lm_head made this path report 3.95x on the same model the RTN path reported
    2.34x for, which is the kind of inconsistency that makes a whole results table
    untrustworthy.
    """
    from .convert import LayerReport, QuantReport, _other_bytes, linear_layers

    report = QuantReport(scheme=scheme, seconds=seconds)
    for nm, m in model.named_modules():
        if isinstance(m, QuantLinear):
            report.layers.append(LayerReport(
                name=nm, in_features=m.in_features, out_features=m.out_features,
                float_bytes=m.float_nbytes(), quant_bytes=m.nbytes(),
                group_size=m.group_size, requested_group=scheme.group_size))
    for nm, lin in linear_layers(model).items():
        nbytes = lin.in_features * lin.out_features * 2
        report.layers.append(LayerReport(
            name=nm, in_features=lin.in_features, out_features=lin.out_features,
            float_bytes=nbytes, quant_bytes=nbytes, group_size=-1,
            requested_group=scheme.group_size,
            skipped="left in float (tied head, or excluded)"))
    report.other_bytes = _other_bytes(model)
    return report


def _build(src: Path, method: str, scheme: QuantScheme, args):
    """Load, optionally calibrate, optionally AWQ-prescale, then quantize.

    Returns (model, ckpt, report, extra) with `extra` carrying whatever the method wants
    reported -- the AWQ site table, or GPTQ's fallback list.
    """
    model, ckpt = _load_float(src, args.device)
    skip = tuple(s.strip() for s in args.skip.split(",") if s.strip())
    extra: dict = {}
    quantizer = None

    if method == "awq":
        calib = _calibrate(model, ckpt, args, want_hessian=False, stage=stage_for(src.name))
        print("  searching AWQ scales...", flush=True)
        extra["awq"] = apply_awq(model, calib, scheme)
        print(f"  scaled {extra['awq']['n_sites']} sites, "
              f"mean alpha {extra['awq']['mean_alpha']:.2f}, "
              f"predicted error x{extra['awq']['mean_gain']:.2f} better")
    elif method == "gptq":
        calib = _calibrate(model, ckpt, args, want_hessian=True, stage=stage_for(src.name))
        print("  running GPTQ...", flush=True)
        quantizer = make_gptq_quantizer(calib, damp=args.damp)
    elif method == "qat":
        train_bin = _train_bin(ckpt, src, args)
        wrapped = prepare_qat(model, scheme, quantize_head=args.quantize_head, skip=skip)
        print(f"  fake-quantizing {len(wrapped)} layers, fine-tuning "
              f"{args.qat_steps} steps on {train_bin}...", flush=True)
        res = train_qat(model, train_bin, _seq_len(ckpt), scheme, steps=args.qat_steps,
                        batch_size=args.qat_batch, lr=args.qat_lr, device=args.device,
                        stop_file=Path(args.stop_file) if args.stop_file else None,
                        stop_by=(time.time() + stopfile.parse_duration(args.stop_in)
                                 if args.stop_in else None))
        if res.steps < args.qat_steps:
            extra["qat_stopped_early"] = res.steps
        print(f"  loss {res.loss_start:.4f} -> {res.loss_end:.4f} in {res.seconds:.0f}s")
        extra["qat"] = res.as_dict()
        convert_qat(model, scheme)
        report = _qat_report(model, scheme, res.seconds)
        return model, ckpt, report, extra

    report = quantize_model(model, scheme, quantizer=quantizer,
                           quantize_head=args.quantize_head, skip=skip)
    if quantizer is not None and getattr(quantizer, "fell_back", None):
        extra["gptq_fell_back"] = quantizer.fell_back
        print(f"  note: {len(quantizer.fell_back)} layers had no Hessian and used RTN")
    return model, ckpt, report, extra


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m aksharallm.quant",
        description="Quantize a checkpoint to 8 or 4 bits, and measure what it cost.")
    ap.add_argument("checkpoint", help="run/name.pt, or a path")
    ap.add_argument("--bits", type=int, default=4, choices=(4, 8))
    ap.add_argument("--group", type=int, default=64,
                    help="weights per scale along in_features; -1 for per-channel. Must "
                         "divide in_features — 128 does not divide d_ff=2752 on "
                         "small-code, so that layer falls back to 64 and says so.")
    ap.add_argument("--sym", action="store_true", help="symmetric (scale only, no zero-point)")
    ap.add_argument("--dtype", default="int", choices=("int", "nf4"),
                    help="'int' spaces the 16 levels evenly; 'nf4' puts them at the "
                         "quantiles of a normal distribution, which is what trained "
                         "weights actually look like. nf4 implies 4 bits and no "
                         "zero-point. This is the datatype QLoRA fine-tunes on top of.")
    ap.add_argument("--double-quant", action="store_true",
                    help="also quantize the scales (int8 + one fp32 scale and mean per "
                         "256). Saves ~0.12 bits/weight at group 64 — real, and small.")
    ap.add_argument("--method", default="rtn", choices=("rtn", "gptq", "awq", "qat"))
    ap.add_argument("--qat-steps", type=int, default=200,
                    help="quantization-aware fine-tuning steps (method=qat)")
    ap.add_argument("--qat-lr", type=float, default=5e-5,
                    help="measured optimum on the tiny model: 1e-5 recovers nothing, "
                         "5e-5 beats GPTQ, 2e-4 starts undoing the pretraining")
    ap.add_argument("--qat-batch", type=int, default=4)
    # Bounded stops for the one part of quantizing that is a training loop. RTN, GPTQ and
    # AWQ are single passes over the weights with nothing to stop early -- for those, the
    # only honest answer to "stop in 10 minutes" is "it will be finished by then".
    ap.add_argument("--stop-file", default=None,
                    help="QAT only: poll this file for a stop request. Empty = stop now, "
                         "a number = stop at that step, @<epoch> = stop at that time.")
    ap.add_argument("--stop-in", default=None, metavar="DURATION",
                    help="QAT only: fine-tune for this long, then export: 30m / 90s / 2h.")
    ap.add_argument("--train-bin", default=None,
                    help="training data for QAT; defaults to the run's own train split")
    ap.add_argument("--quantize-head", action="store_true",
                    help="also quantize lm_head. With tied embeddings this saves nothing "
                         "and costs accuracy — off by default for that reason")
    ap.add_argument("--skip", default="", help="comma-separated substrings of layer names")
    ap.add_argument("--calib-seqs", type=int, default=128,
                    help="calibration sequences for gptq/awq (128 is plenty)")
    ap.add_argument("--calib-batch", type=int, default=4)
    ap.add_argument("--calib-bin", default=None, help="tokenized .bin to calibrate on")
    ap.add_argument("--damp", type=float, default=0.01,
                    help="GPTQ Hessian ridge, as a fraction of the mean diagonal")
    ap.add_argument("-o", "--out", default=None, help="output .pt (default: alongside)")
    ap.add_argument("--no-save", action="store_true", help="measure only, write nothing")
    ap.add_argument("--bench", action="store_true", help="perplexity + speed after quantizing")
    ap.add_argument("--compare", action="store_true", help="every method, one table")
    ap.add_argument("--val-bin", default=None, help="defaults to the run's own val split")
    ap.add_argument("--n-batches", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--new-tokens", type=int, default=128, help="decode length for tok/s")
    ap.add_argument("--no-speed", action="store_true")
    ap.add_argument("--backend", default="auto", choices=("auto", "torch", "triton"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--json", default=None, help="write results as JSON here")
    args = ap.parse_args(argv)

    QuantLinear.backend = args.backend
    src = _resolve(args.checkpoint)
    payload = {"checkpoint": str(src), "device": args.device}

    if args.compare:
        return _compare(args, src, payload)

    try:
        scheme = QuantScheme(bits=args.bits, group_size=args.group, sym=args.sym,
                             method=args.method, dtype=args.dtype,
                             double_quant=args.double_quant)
    except ValueError as e:
        raise SystemExit(f"bad scheme: {e}")
    results, base = [], None

    if args.bench:
        model, ckpt = _load_float(src, args.device)
        print("measuring the float baseline first...", flush=True)
        base = bench_mod.measure(model, "bf16 (baseline)", val_bin=_val_bin(ckpt, args.val_bin),
                                 seq_len=_seq_len(ckpt), device=args.device,
                                 n_batches=args.n_batches, batch_size=args.batch_size,
                                 speed=not args.no_speed, new_tokens=args.new_tokens)
        results.append(base)
        del model
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    print(f"\n{scheme.label()}:")
    model, ckpt, report, extra = _build(src, args.method, scheme, args)
    print()
    print(report.summary())
    payload["report"] = report.as_dict()
    payload.update(extra)

    if args.bench:
        print("\nmeasuring the quantized model...", flush=True)
        results.append(bench_mod.measure(
            model, scheme.label(), val_bin=_val_bin(ckpt, args.val_bin),
            seq_len=_seq_len(ckpt), device=args.device, n_batches=args.n_batches,
            batch_size=args.batch_size, speed=not args.no_speed,
            new_tokens=args.new_tokens))
        print()
        print(bench_mod.format_table(results, baseline=base))
        payload["bench"] = [x.as_dict() for x in results]

    if not args.no_save:
        out = _out_path(src, scheme, args.out)
        save_quantized(out, model, scheme, report, ckpt, source_path=str(src))
        print(f"\nwrote {out}  ({out.stat().st_size / 1e6:.1f} MB on disk)")
        payload["out"] = str(out)

    if args.json:
        Path(args.json).write_text(json.dumps(payload, indent=2))
        print(f"wrote {args.json}")
    return 0


def _compare(args, src: Path, payload: dict) -> int:
    model, ckpt = _load_float(src, args.device)
    val_bin = _val_bin(ckpt, args.val_bin)
    seq_len = _seq_len(ckpt)
    if not val_bin:
        print("note: no validation split on disk, so no perplexity column", file=sys.stderr)

    measure = lambda m, label: bench_mod.measure(  # noqa: E731
        m, label, val_bin=val_bin, seq_len=seq_len, device=args.device,
        n_batches=args.n_batches, batch_size=args.batch_size,
        speed=not args.no_speed, new_tokens=args.new_tokens)

    print("bf16 baseline...", flush=True)
    base = measure(model, "bf16 (baseline)")
    results, reports = [base], {}
    del model
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()

    for method, scheme in COMPARE:
        print(f"\n{scheme.label()}:", flush=True)
        try:
            m, _ck, rep, extra = _build(src, method, scheme, args)
        except SystemExit as e:
            print(f"  skipped: {e}", file=sys.stderr)
            continue
        reports[scheme.label()] = {**rep.as_dict(), **extra}
        results.append(measure(m, scheme.label()))
        del m
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    print()
    print(bench_mod.format_table(results, baseline=base))
    print("\nsize is the whole model in memory including the embedding table, which is"
          "\nnever quantized — so the ratio is always below the nominal 2x / 4x.")
    payload["bench"] = [r.as_dict() for r in results]
    payload["reports"] = reports
    if args.json:
        Path(args.json).write_text(json.dumps(payload, indent=2))
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
