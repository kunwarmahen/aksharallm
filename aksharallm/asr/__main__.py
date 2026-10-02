"""`python -m aksharallm.asr` — get speech, measure a recogniser, transcribe a file.

    python -m aksharallm.asr fetch test-clean                  # LibriSpeech split (0.35 GB)
    python -m aksharallm.asr pack  test-clean                  # -> data/asr/test-clean
    python -m aksharallm.asr eval  asr-libri100 --corpus data/asr/test-clean
    python -m aksharallm.asr silence asr-libri100              # day-two problem 1
    python -m aksharallm.asr transcribe asr-libri100 me.wav
    python -m aksharallm.asr lm fetch                          # its text, 1.5 GB (OpenSLR 11)
    python -m aksharallm.asr lm build                          # word trigram LM (piece 5)
    python -m aksharallm.asr tune asr-libri100                 # alpha/beta on dev-clean only
    python -m aksharallm.asr eval asr-libri100 --decoder beam --lm data/asr/lm/trigram.npz
    python -m aksharallm.asr daytwo asr-libri100               # the five day-two checks at once

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


def _decoder(args, lm, dictionary=()):
    from .decode import BeamDecoder
    return BeamDecoder(lm=lm, alpha=args.alpha, beta=args.beta, beam=args.beam,
                       unk_penalty=args.unk_penalty, dictionary=set(dictionary))


def _decode(lps, kind: str, dec=None, workers: int = 8) -> list[str]:
    from .ctc import greedy_decode
    from .decode import decode_many
    from . import vocab
    if kind == "greedy":
        return [" ".join(vocab.decode(greedy_decode(torch.from_numpy(x)[None],
                                                    torch.tensor([len(x)]))[0]).split())
                for x in lps]
    return decode_many(dec, lps, workers)


def _load_lm(path):
    from .ngram import TrigramLM
    if not path:
        return None
    lm = TrigramLM.load(path)
    print(f"lm           {path}: {lm.describe()}")
    return lm


def cmd_eval(args) -> int:
    from .measure import log_probs, name_recall
    path = resolve(args.checkpoint)
    device = _device(args.device)
    model, blob = load_recognizer(path, device)
    if args.split == "val" and not args.val_clips:
        raise SystemExit("--split val needs --val-clips N: the run held out its last N clips")
    corpus = Utterances(args.corpus, split=args.split, val_clips=args.val_clips or 0,
                        max_seconds=args.max_seconds,
                        feasible=feasibility(model.cfg, model.cfg.sample_rate), limit=args.limit)
    lm = _load_lm(args.lm) if args.decoder == "beam" else None
    dictionary = set()
    if args.dict:
        dictionary = {w.strip().lower() for w in Path(args.dict).read_text().split() if w.strip()}
    t0 = time.time()
    lps = log_probs(model, [corpus.wave(u) for u in corpus.utts], device, batch=args.batch)
    t_enc = time.time() - t0
    dec = _decoder(args, lm, dictionary) if args.decoder == "beam" else None
    t0 = time.time()
    hyps = _decode(lps, args.decoder, dec, args.workers)
    t_dec = time.time() - t0
    dt = t_enc + t_dec
    s = score([(u.speaker, u.text, h) for u, h in zip(corpus.utts, hyps, strict=True)])
    sil = silence_check(model, device)
    pm = wer_interval(s["wer"], s["words"])
    how = "greedy, no language model" if args.decoder == "greedy" else (
        f"beam {args.beam}" + (f", LM a={args.alpha} b={args.beta} unk={args.unk_penalty}" if lm else ", no LM")
        + (f", dictionary of {len(dictionary)}" if dictionary else ""))
    print(f"checkpoint   {path}  (step {blob.get('step')})")
    print(f"corpus       {args.corpus}: {s['utts']:,} utterances, {s['words']:,} words, "
          f"{corpus.seconds / 3600:.2f} h" + (f"  (dropped {dict(corpus.dropped)})" if corpus.dropped else ""))
    print(f"WER          {s['wer'] * 100:.2f}%  ± {pm * 100:.2f}   ({how})")
    print(f"CER          {s['cer'] * 100:.2f}%")
    print(f"speed        {corpus.seconds / max(dt, 1e-9):.0f}x real time on {device} "
          f"(encoder {t_enc:.1f}s, decoder {t_dec:.1f}s)")
    print(f"silence      {sil['chars']} characters written on {sil['clips']} clips with no speech"
          + ("  <- should be 0" if sil["chars"] else ""))
    names = None
    if lm is not None:
        targets = {w for u in corpus.utts for w in u.text.split() if w not in lm.index}
        names = name_recall([u.text for u in corpus.utts], hyps, targets)
        print(f"names        {names['recalled']}/{names['said']} words the LM has never seen came out "
              f"right ({names['recall']:.0%}); {names['false_alarms']} written where not said")
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
        dtag = "" if args.decoder == "greedy" else ("-beam-lm" if lm else "-beam") + ("-dict" if dictionary else "")
        out = RESULTS / f"{run}-step{blob.get('step')}-{tag}{dtag}.json"
        out.write_text(json.dumps({
            "kind": "asr_eval", "checkpoint": str(path), "run": run, "step": blob.get("step"),
            "corpus": str(args.corpus), "split": args.split, "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "decoder": args.decoder, "decoder_desc": how, "lm": args.lm if lm else None,
            "alpha": args.alpha if lm else None, "beta": args.beta if lm else None,
            "unk_penalty": args.unk_penalty if lm else None, "beam": args.beam if dec else None,
            "dictionary_size": len(dictionary), "names": names,
            "seconds": corpus.seconds, "dropped": dict(corpus.dropped),
            "wer": s["wer"], "wer_pm": pm, "cer": s["cer"], "words": s["words"], "utts": s["utts"],
            "speakers": s["speakers"], "worst_speakers": s["worst_speakers"], "silence": sil,
            "examples": [{"ref": u.text, "hyp": h} for u, h in list(zip(corpus.utts, hyps, strict=True))[:20]],
        }, indent=1))
        print(f"written      {out}")
    return 0


def cmd_lm_fetch(args) -> int:
    from .data import fetch_lm_text
    fetch_lm_text(args.dest)
    return 0


def cmd_lm_build(args) -> int:
    from .ngram import TrigramLM, verbatim_overlap
    lm = TrigramLM.build(args.corpus, vocab_size=args.vocab_size, max_words=args.max_words,
                         every=args.every)
    out = lm.save(args.out)
    print(f"built        {lm.describe()}  in {lm.meta['seconds']:.0f}s")
    print(f"saved        {out}")
    for c in args.check:
        u = Utterances(c, max_seconds=1e9)
        ppl = lm.perplexity([x.text for x in u.utts])
        lm.meta.setdefault("perplexity", {})[Path(c).name] = ppl
        print(f"perplexity   {Path(c).name}: {ppl['perplexity']:.1f}  (OOV {ppl['oov_rate']:.2%} of "
              f"{ppl['words']:,} words)")
        if args.overlap:
            ov = verbatim_overlap(args.corpus, [x.text for x in u.utts], args.max_words)
            lm.meta.setdefault("overlap", {})[Path(c).name] = ov
            print(f"overlap      {ov['found']}/{ov['checked']} of {Path(c).name}'s sentences (5+ words) "
                  f"appear verbatim in the LM text ({ov['rate']:.2%})")
    lm.save(args.out)
    return 0


def cmd_tune(args) -> int:
    """Grid over alpha, beta and the <unk> penalty on a VALIDATION corpus. The encoder runs
    once; each grid point is only a beam search."""
    from .measure import log_probs
    if "test" in Path(args.corpus).name:
        raise SystemExit("tune on dev, never on test: the test number has to be one the weights "
                         "were not chosen on")
    path = resolve(args.checkpoint)
    device = _device(args.device)
    model, blob = load_recognizer(path, device)
    lm = _load_lm(args.lm)
    corpus = Utterances(args.corpus, max_seconds=40.0, limit=args.limit,
                        feasible=feasibility(model.cfg, model.cfg.sample_rate))
    lps = log_probs(model, [corpus.wave(u) for u in corpus.utts], device)
    refs = [(u.speaker, u.text) for u in corpus.utts]
    base = score([(sp, r, h) for (sp, r), h in zip(refs, _decode(lps, "greedy"), strict=True)])["wer"]
    print(f"greedy       {base * 100:.2f}% on {len(refs)} utterances of {args.corpus}")
    rows = []
    from .decode import BeamDecoder
    for a in args.alphas:
        for b in args.betas:
            for k in args.unk_penalties:
                dec = BeamDecoder(lm=lm, alpha=a, beta=b, beam=args.beam, unk_penalty=k)
                wer = score([(sp, r, h) for (sp, r), h in
                             zip(refs, _decode(lps, "beam", dec, args.workers), strict=True)])["wer"]
                rows.append({"alpha": a, "beta": b, "unk_penalty": k, "wer": wer})
                print(f"  alpha {a:<4} beta {b:<4} unk {k:<5}  WER {wer * 100:.2f}%")
    best = min(rows, key=lambda r: r["wer"])
    print(f"best         alpha {best['alpha']} beta {best['beta']} unk {best['unk_penalty']}: "
          f"{best['wer'] * 100:.2f}%  (greedy {base * 100:.2f}%)")
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / f"tune-{path.parent.name}-step{blob.get('step')}-{Path(args.corpus).name}.json"
    out.write_text(json.dumps({"kind": "asr_tune", "checkpoint": str(path), "lm": args.lm,
                               "corpus": args.corpus, "utts": len(refs), "beam": args.beam,
                               "greedy_wer": base, "grid": rows, "best": best}, indent=1))
    print(f"written      {out}")
    return 0


def cmd_daytwo(args) -> int:
    """The day-two suite (asr/daytwo.py): one corpus, one pass, one JSON."""
    from ..dictate.pipeline import best_tuning
    from .daytwo import run
    path = resolve(args.checkpoint)
    device = _device(args.device)
    model, blob = load_recognizer(path, device)
    lm = _load_lm(args.lm)
    if lm is None:
        raise SystemExit("daytwo needs the word LM (names and corrections are LM questions): "
                         "python -m aksharallm.asr lm build")
    t = best_tuning(Path.cwd(), args.lm) or {}
    tuning = {"alpha": args.alpha if args.alpha is not None else t.get("alpha", 0.8),
              "beta": args.beta if args.beta is not None else t.get("beta", 2.0),
              "unk_penalty": args.unk_penalty if args.unk_penalty is not None else t.get("unk_penalty", -24.0),
              "beam": t.get("beam", 16)}
    punctuator = None
    if args.punctuator and Path(args.punctuator).is_file():
        from ..dictate.tagger import Punctuator
        punctuator = Punctuator.load(args.punctuator, "cpu")
    corpus = Utterances(args.corpus, max_seconds=40.0, limit=args.limit,
                        feasible=feasibility(model.cfg, model.cfg.sample_rate))
    print(f"checkpoint   {path}  (step {blob.get('step')})")
    print(f"corpus       {args.corpus}: {len(corpus.utts):,} utterances; beam a={tuning['alpha']} "
          f"b={tuning['beta']} unk={tuning['unk_penalty']}" + (" (tuned on dev)" if t else ""))
    t0 = time.time()
    r = run(model, device, corpus, lm, tuning, punctuator, workers=args.workers,
            min_occurrences=args.min_occurrences, score_limit=args.score_limit)
    sil, sp, nm, co, cl = r["silence"], r["speakers"], r["names"], r["corrections"], r["cleanup"]
    mark = lambda ok: "pass" if ok else "FAIL"   # noqa: E731
    print(f"1 silence    {sil['chars']} characters on {sil['clips']} no-speech clips   {mark(sil['pass'])}")
    print(f"2 speakers   WER {sp['wer'] * 100:.2f}% over {sp['speakers']} speakers: best "
          f"{sp['best'] * 100:.1f}%  median {sp['median'] * 100:.1f}%  worst {sp['worst'] * 100:.1f}%")
    print(f"3 languages  {r['languages']['status']} — {r['languages']['why']}")
    print(f"4 names      {nm['words']} unseen words: recall {nm['without']['recall']:.0%} -> "
          f"{nm['with_dictionary']['recall']:.0%} with them in a dictionary "
          f"({nm['with_dictionary']['false_alarms']} written where not said)")
    curve = "  ".join(f"#{c['occurrence']} {c['recall']:.0%}" for c in co["curve"])
    print(f"5 learning   {co['targets']} words said {co['min_occurrences']}+ times, corrected after "
          f"each: {curve}   {mark(co['learns'])}")
    print(f"             {co['false_insertions']} false insertions of {co['learned_words']} learned "
          f"words across {co['false_insertion_utts']} utterances that never say them")
    print(f"+ cleanup    {cl['invented']} words invented, {cl['dropped']} dropped, over "
          f"{cl['words']:,} words ({cl['punctuator']})   {mark(cl['pass'])}")
    print(f"took         {time.time() - t0:.0f}s")
    if not args.no_write:
        RESULTS.mkdir(parents=True, exist_ok=True)
        out = RESULTS / f"daytwo-{path.parent.name}-step{blob.get('step')}-{Path(args.corpus).name}.json"
        out.write_text(json.dumps({"kind": "asr_daytwo", "checkpoint": str(path),
                                   "run": path.parent.name, "step": blob.get("step"),
                                   "corpus": str(args.corpus), "lm": args.lm, "tuning": tuning,
                                   "punctuator": args.punctuator if punctuator else None,
                                   "time": time.strftime("%Y-%m-%d %H:%M:%S"), **r}, indent=1))
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
    s.add_argument("--decoder", choices=["greedy", "beam"], default="greedy")
    s.add_argument("--lm", default=None, help="a word LM from `asr lm build` (beam only)")
    s.add_argument("--alpha", type=float, default=0.5, help="LM weight (tune on dev)")
    s.add_argument("--beta", type=float, default=1.0, help="per-word bonus (tune on dev)")
    s.add_argument("--unk-penalty", type=float, default=-6.0)
    s.add_argument("--beam", type=int, default=16)
    s.add_argument("--dict", default=None, help="a personal dictionary: one word per line")
    s.add_argument("--workers", type=int, default=8)
    s.set_defaults(fn=cmd_eval)

    s = sub.add_parser("lm", help="build the word language model the beam search uses")
    lsub = s.add_subparsers(dest="lm_cmd", required=True)
    f = lsub.add_parser("fetch", help="download the LM text corpus (OpenSLR 11, 1.5 GB)")
    f.add_argument("--dest", default="data/asr/lm")
    f.set_defaults(fn=cmd_lm_fetch)
    b = lsub.add_parser("build", help="count a trigram Kneser-Ney LM from a text corpus")
    b.add_argument("--corpus", default="data/asr/lm/librispeech-lm-norm.txt.gz")
    b.add_argument("--out", default="data/asr/lm/trigram.npz")
    b.add_argument("--vocab-size", type=int, default=200_000)
    b.add_argument("--every", type=int, default=1,
                   help="keep one line in N, evenly across the file. Use this to subsample: "
                        "the LibriSpeech LM text is sorted alphabetically")
    b.add_argument("--max-words", type=int, default=None,
                   help="stop after N words -- a PREFIX, so biased on a sorted corpus")
    b.add_argument("--check", nargs="*", default=["data/asr/dev-clean"],
                   help="packed corpora to report perplexity on")
    b.add_argument("--overlap", action="store_true",
                   help="also count check sentences found verbatim in the LM text (another pass)")
    b.set_defaults(fn=cmd_lm_build)

    s = sub.add_parser("tune", help="choose alpha / beta / unk penalty on a dev corpus")
    s.add_argument("checkpoint")
    s.add_argument("--lm", default="data/asr/lm/trigram.npz")
    s.add_argument("--corpus", default="data/asr/dev-clean")
    s.add_argument("--limit", type=int, default=800)
    s.add_argument("--alphas", type=float, nargs="+", default=[0.3, 0.5, 0.7, 1.0])
    s.add_argument("--betas", type=float, nargs="+", default=[0.0, 1.0, 2.0])
    s.add_argument("--unk-penalties", type=float, nargs="+", default=[-4.0, -8.0])
    s.add_argument("--beam", type=int, default=16)
    s.add_argument("--workers", type=int, default=12)
    s.add_argument("--device", default=None)
    s.set_defaults(fn=cmd_tune)

    s = sub.add_parser("daytwo", help="the day-two checks: silence, speakers, names, "
                                      "learning from corrections, cleanup")
    s.add_argument("checkpoint")
    s.add_argument("--corpus", default="data/asr/test-clean")
    s.add_argument("--lm", default="data/asr/lm/trigram.npz")
    s.add_argument("--punctuator", default="checkpoints/punct/ckpt_best.pt",
                   help="the cleanup tagger; missing = rules only")
    s.add_argument("--limit", type=int, default=None, help="only the first N utterances")
    s.add_argument("--score-limit", type=int, default=None,
                   help="score speakers/names on the first N only (the replay uses all)")
    s.add_argument("--min-occurrences", type=int, default=4)
    s.add_argument("--alpha", type=float, default=None)
    s.add_argument("--beta", type=float, default=None)
    s.add_argument("--unk-penalty", type=float, default=None)
    s.add_argument("--workers", type=int, default=8)
    s.add_argument("--device", default=None)
    s.add_argument("--no-write", action="store_true")
    s.set_defaults(fn=cmd_daytwo)

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
