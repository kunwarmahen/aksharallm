"""`python -m aksharallm.asr` — get speech, measure a recogniser, transcribe a file.

    python -m aksharallm.asr fetch test-clean                  # LibriSpeech split (0.35 GB)
    python -m aksharallm.asr pack  test-clean                  # -> data/asr/test-clean
    python -m aksharallm.asr eval  asr-libri100 --corpus data/asr/test-clean
    python -m aksharallm.asr silence asr-libri100              # day-two problem 1
    python -m aksharallm.asr transcribe asr-libri100 me.wav

A checkpoint argument is a path to a `.pt`, or a run name (`asr-libri100` means
`checkpoints/asr-libri100/ckpt_best.pt`).

`eval` writes one JSON per run into `logs/asr/` and never into `logs/eval/`: that folder
holds the language-model harness's results, read by shape (gotcha 18), and a second shape
there is the exact bug that once emptied the whole Eval tab.

Read with: docs/23-speech-recognition.md -- the chapter this implements; it ends with the
order to read these files in.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from ..audio.io import load_audio
from .data import LIBRISPEECH_SIZES_GB, Utterances, decode_flac, fetch_librispeech, pack_librispeech
from .measure import score, silence_check, transcribe, wer_interval
from .train import feasibility, load_recognizer

RESULTS = Path("logs/asr")


def resolve(ckpt: str) -> Path:
    p = Path(ckpt)
    if p.suffix == ".pt":
        return p
    for name in ("ckpt_best.pt", "ckpt_last.pt"):
        q = Path("checkpoints") / ckpt / name
        if q.is_file():
            return q
    raise SystemExit(f"no checkpoint for {ckpt!r} (looked for checkpoints/{ckpt}/ckpt_best.pt)")


def _device(arg: str | None) -> str:
    return arg or ("cuda" if torch.cuda.is_available() else "cpu")


def cmd_fetch(args) -> int:
    fetch_librispeech(args.split, args.dest)
    return 0


def cmd_pack(args) -> int:
    src = Path(args.dest) / "LibriSpeech" / args.split
    pack_librispeech(src, Path(args.out or f"data/asr/{args.split}"), workers=args.workers)
    return 0


def cmd_eval(args) -> int:
    path = resolve(args.checkpoint)
    device = _device(args.device)
    model, blob = load_recognizer(path, device)
    if args.split == "val" and not args.val_clips:
        raise SystemExit("--split val needs --val-clips N: the run held out its last N clips")
    corpus = Utterances(args.corpus, split=args.split, val_clips=args.val_clips or 0,
                        max_seconds=args.max_seconds,
                        feasible=feasibility(model.cfg, model.cfg.sample_rate), limit=args.limit)
    t0 = time.time()
    hyps = []
    for i in range(0, len(corpus), args.batch):
        chunk = corpus.utts[i : i + args.batch]
        hyps += transcribe(model, [corpus.wave(u) for u in chunk], device, batch=args.batch)
    dt = time.time() - t0
    s = score([(u.speaker, u.text, h) for u, h in zip(corpus.utts, hyps, strict=True)])
    sil = silence_check(model, device)
    pm = wer_interval(s["wer"], s["words"])
    print(f"checkpoint   {path}  (step {blob.get('step')})")
    print(f"corpus       {args.corpus}: {s['utts']:,} utterances, {s['words']:,} words, "
          f"{corpus.seconds / 3600:.2f} h" + (f"  (dropped {dict(corpus.dropped)})" if corpus.dropped else ""))
    print(f"WER          {s['wer'] * 100:.2f}%  ± {pm * 100:.2f}   (greedy, no language model)")
    print(f"CER          {s['cer'] * 100:.2f}%")
    print(f"speed        {corpus.seconds / max(dt, 1e-9):.0f}x real time on {device}")
    print(f"silence      {sil['chars']} characters written on {sil['clips']} clips with no speech"
          + ("  <- should be 0" if sil["chars"] else ""))
    if len(s["speakers"]) > 1:
        rates = sorted(v["wer"] for v in s["speakers"].values())
        print(f"speakers     {len(rates)}: best {rates[0] * 100:.1f}%  median "
              f"{rates[len(rates) // 2] * 100:.1f}%  worst {rates[-1] * 100:.1f}%")
        for w in s["worst_speakers"][:3]:
            print(f"               speaker {w['speaker']:>6}: {w['wer'] * 100:5.1f}% over {w['words']} words")
    for u, h in list(zip(corpus.utts, hyps, strict=True))[: args.show]:
        print(f"  ref  {u.text[:100]}")
        print(f"  hyp  {h[:100] or '(nothing)'}")
    if not args.no_write:
        RESULTS.mkdir(parents=True, exist_ok=True)
        run = path.parent.name
        tag = Path(args.corpus).name + ("-val" if args.split == "val" else "")
        out = RESULTS / f"{run}-step{blob.get('step')}-{tag}.json"
        out.write_text(json.dumps({
            "kind": "asr_eval", "checkpoint": str(path), "run": run, "step": blob.get("step"),
            "corpus": str(args.corpus), "split": args.split, "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "decoder": "greedy", "seconds": corpus.seconds, "dropped": dict(corpus.dropped),
            "wer": s["wer"], "wer_pm": pm, "cer": s["cer"], "words": s["words"], "utts": s["utts"],
            "speakers": s["speakers"], "worst_speakers": s["worst_speakers"], "silence": sil,
            "examples": [{"ref": u.text, "hyp": h} for u, h in list(zip(corpus.utts, hyps, strict=True))[:20]],
        }, indent=1))
        print(f"written      {out}")
    return 0


def cmd_silence(args) -> int:
    model, _ = load_recognizer(resolve(args.checkpoint), _device(args.device))
    r = silence_check(model, _device(args.device))
    for name, text in r["outputs"].items():
        print(f"  {name:24s} -> {text!r}" if text.strip() else f"  {name:24s} -> (nothing)  ok")
    print(f"{r['chars']} characters on {r['clips']} clips with no speech in them. The pass mark is 0.")
    return 0 if r["chars"] == 0 else 1


def cmd_transcribe(args) -> int:
    device = _device(args.device)
    model, _ = load_recognizer(resolve(args.checkpoint), device)
    waves = []
    for f in args.files:
        if f.endswith(".flac"):
            waves.append(decode_flac(f).astype(np.float32) / 32768.0)
        else:
            clip = load_audio(f, sample_rate=model.cfg.sample_rate)
            waves.append(clip.samples)
    for f, text in zip(args.files, transcribe(model, waves, device), strict=True):
        print(f"{f}: {text}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m aksharallm.asr",
                                description="Speech recognition from scratch (docs/23).")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("fetch", help="download a LibriSpeech split")
    s.add_argument("split", choices=sorted(LIBRISPEECH_SIZES_GB))
    s.add_argument("--dest", default="data/asr/librispeech")
    s.set_defaults(fn=cmd_fetch)

    s = sub.add_parser("pack", help="pack a downloaded split into audio.bin + transcripts")
    s.add_argument("split", choices=sorted(LIBRISPEECH_SIZES_GB))
    s.add_argument("--dest", default="data/asr/librispeech")
    s.add_argument("--out", default=None, help="default data/asr/<split>")
    s.add_argument("--workers", type=int, default=8)
    s.set_defaults(fn=cmd_pack)

    s = sub.add_parser("eval", help="WER/CER on a packed corpus, per speaker, plus the silence check")
    s.add_argument("checkpoint", help="a .pt, or a run name")
    s.add_argument("--corpus", default="data/asr/test-clean")
    s.add_argument("--split", choices=["all", "val"], default="all",
                   help="'val' = only the last --val-clips clips: what a run trained without "
                        "a separate val corpus held out. Scoring 'all' of a training corpus "
                        "measures memory, not recognition.")
    s.add_argument("--val-clips", type=int, default=None)
    s.add_argument("--limit", type=int, default=None, help="only the first N utterances")
    s.add_argument("--max-seconds", type=float, default=40.0)
    s.add_argument("--batch", type=int, default=16)
    s.add_argument("--show", type=int, default=3, help="print this many ref/hyp pairs")
    s.add_argument("--device", default=None)
    s.add_argument("--no-write", action="store_true", help="do not write logs/asr/*.json")
    s.set_defaults(fn=cmd_eval)

    s = sub.add_parser("silence", help="what it writes on audio with no speech (should be nothing)")
    s.add_argument("checkpoint")
    s.add_argument("--device", default=None)
    s.set_defaults(fn=cmd_silence)

    s = sub.add_parser("transcribe", help="transcribe WAV or FLAC files")
    s.add_argument("checkpoint")
    s.add_argument("files", nargs="+")
    s.add_argument("--device", default=None)
    s.set_defaults(fn=cmd_transcribe)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
