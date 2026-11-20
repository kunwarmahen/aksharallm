"""`python -m aksharallm.audio` — look at a sound file, and listen to what the model sees.

The front end of the audio phase has no model in it, so everything here runs on a laptop in
under a second and needs no checkpoint::

    python -m aksharallm.audio info      data/audio/LJSpeech/wavs/LJ001-0001.wav
    python -m aksharallm.audio spec      data/audio/LJSpeech/wavs/LJ001-0001.wav
    python -m aksharallm.audio roundtrip data/audio/LJSpeech/wavs/LJ001-0001.wav --mel
    python -m aksharallm.audio resample  in.wav --to 16000 --out out.wav
    python -m aksharallm.audio tone --out beep.wav          # no dataset needed

`roundtrip --mel` is the one to run first. It takes the 80-band log-mel the model would be
given, throws the phase and 433 of the 513 frequency bins away, reconstructs a waveform from
what is left, and writes it beside the original. Play the two files back to back and the
question "what does the front end cost?" stops being theoretical.

Read with: docs/21-audio.md -- the chapter this implements; it ends with the order to read
these files in.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch

from .features import (
    MelConfig,
    griffin_lim,
    hz_to_mel,
    log_mel,
    magnitude,
    mel_to_hz,
    mel_to_magnitude,
    spectral_convergence,
)
from .io import TARGET_SR, load_audio, read_wav, resample, write_wav

#: Darkest to lightest. A terminal heat map beats no picture at all, and unlike a PNG it
#: survives being pasted into a note — the same reasoning as `scripts/gpu.sh`'s sparklines.
RAMP = " .:-=+*#%@"


def heatmap(m: np.ndarray, width: int = 100, height: int = 24) -> list[str]:
    """Render a `(bands, frames)` array as text, low frequencies at the bottom."""
    bands, frames = m.shape
    # Nearest-neighbour resize. Averaging would be prettier and would also hide exactly the
    # kind of narrow horizontal stripe (a single dead mel band) worth spotting.
    ys = np.linspace(0, bands - 1, min(height, bands)).round().astype(int)
    xs = np.linspace(0, frames - 1, min(width, frames)).round().astype(int)
    small = m[np.ix_(ys, xs)]
    lo, hi = float(small.min()), float(small.max())
    scaled = (small - lo) / max(hi - lo, 1e-9)
    idx = np.clip((scaled * (len(RAMP) - 1)).round().astype(int), 0, len(RAMP) - 1)
    return ["".join(RAMP[i] for i in row) for row in idx[::-1]]


def _mel_config(args) -> MelConfig:
    return MelConfig(
        sample_rate=args.sr, n_fft=args.n_fft, hop=args.hop, n_mels=args.n_mels
    )


def cmd_info(args) -> int:
    raw, sr = read_wav(args.wav)
    clip = load_audio(args.wav, sample_rate=args.sr)
    cfg = _mel_config(args)
    print(f"file        {args.wav}")
    print(f"on disk     {sr} Hz, {raw.shape[1]} channel(s), {raw.shape[0]} frames "
          f"({raw.shape[0] / sr:.2f}s)")
    print(f"loaded      {clip.describe()}")
    rms = float(np.sqrt((clip.samples**2).mean())) if clip.samples.size else 0.0
    print(f"level       peak {clip.peak:.3f}  rms {rms:.4f}  "
          f"({20 * math.log10(max(rms, 1e-9)):.1f} dBFS)")
    if clip.peak >= 0.999:
        print("            WARNING: the file is already clipped at full scale")
    frames = 1 + len(clip.samples) // cfg.hop
    print(f"log-mel     {cfg.n_mels} x {frames} at {cfg.frames_per_second:.1f} frames/s "
          f"(n_fft {cfg.n_fft}, hop {cfg.hop})")
    return 0


def cmd_spec(args) -> int:
    clip = load_audio(args.wav, sample_rate=args.sr)
    cfg = _mel_config(args)
    m = log_mel(torch.from_numpy(clip.samples), cfg).numpy()
    print(f"{args.wav} — log-mel, {m.shape[0]} bands x {m.shape[1]} frames, "
          f"{clip.seconds:.2f}s")
    # The axis is MEL, not Hz, so the midpoint is not half the top frequency — it is about
    # 1.8 kHz of an 8 kHz range. Labelling it 4 kHz would misrepresent the whole picture.
    rows = heatmap(m, args.width, args.height)
    mid_hz = float(mel_to_hz((hz_to_mel(cfg.fmin) + hz_to_mel(cfg.top_hz)) / 2))
    for i, row in enumerate(rows):
        mark = ""
        if i == 0:
            mark = f" {cfg.top_hz / 1000:.1f}k"
        elif i == len(rows) // 2:
            mark = f" {mid_hz / 1000:.1f}k (mel mid)"
        elif i == len(rows) - 1:
            mark = f" {cfg.fmin / 1000:.0f}"
        print(row + mark)
    print(f"0{' ' * (args.width - 10)}{clip.seconds:.2f}s")
    return 0


def cmd_roundtrip(args) -> int:
    clip = load_audio(args.wav, sample_rate=args.sr)
    cfg = _mel_config(args)
    x = torch.from_numpy(clip.samples)
    target = magnitude(x, cfg)

    if args.mel:
        # The lossy path the model actually uses: 513 bins -> 80 -> log -> back.
        source = mel_to_magnitude(log_mel(x, cfg), cfg)
        label = f"{cfg.n_mels}-band log-mel"
        print(f"mel round trip alone (before any phase guessing): "
              f"{spectral_convergence(target, source):.4f}")
    else:
        source = target
        label = "linear magnitude"

    y = griffin_lim(source, cfg, n_iter=args.iters, momentum=args.momentum,
                    length=x.numel())
    conv = spectral_convergence(target, magnitude(y, cfg))
    print(f"{label} + Griffin-Lim x{args.iters}: spectral convergence {conv:.4f}")

    out = Path(args.out) if args.out else Path(args.wav).with_suffix(".rebuilt.wav")
    write_wav(out, y.numpy(), cfg.sample_rate)
    print(f"written to {out}   (play it against the original)")
    return 0


def cmd_resample(args) -> int:
    data, sr = read_wav(args.wav)
    mono = data.mean(axis=1)
    out = resample(mono, sr, args.to)
    dest = Path(args.out) if args.out else Path(args.wav).with_suffix(f".{args.to}.wav")
    write_wav(dest, out, args.to)
    print(f"{sr} Hz x {len(mono)} -> {args.to} Hz x {len(out)}   written to {dest}")
    return 0


def cmd_tone(args) -> int:
    """A synthetic vowel, so every command here is runnable before any dataset exists."""
    sr = args.sr
    t = np.arange(int(sr * args.seconds)) / sr
    # A buzzy harmonic stack under a slow tremolo — closer to speech than a sine, and it
    # makes the front end's failure modes audible (a broken filterbank kills the harmonics).
    x = sum((0.6 / k) * np.sin(2 * math.pi * args.f0 * k * t) for k in range(1, 12))
    x = x * (0.5 + 0.5 * np.sin(2 * math.pi * 3 * t))
    x = (x / np.abs(x).max() * 0.7).astype(np.float32)
    write_wav(args.out, x, sr)
    print(f"{args.seconds}s at {sr} Hz, f0 {args.f0} Hz -> {args.out}")
    return 0


def cmd_corpus(args) -> int:
    from .dataset import synth_corpus

    man = synth_corpus(args.out, n_clips=args.clips, seconds=args.seconds,
                       sample_rate=args.sr, seed=args.seed)
    print(f"{man.n_clips} clips, {man.seconds / 60:.1f} minutes -> {args.out}")
    print("train it:  scripts/audio.sh codec-synth")
    return 0


def cmd_pack(args) -> int:
    from .dataset import find_wavs, pack

    wavs = find_wavs(args.dir)
    if not wavs:
        print(f"no .wav files under {args.dir}", file=__import__("sys").stderr)
        return 1
    print(f"{len(wavs)} wav files under {args.dir}")
    pack(wavs, args.out, sample_rate=args.sr)
    return 0


def cmd_fetch(args) -> int:
    from .dataset import fetch_ljspeech

    wavs = fetch_ljspeech(args.dest)
    print(f"\nnow pack it:\n    python -m aksharallm.audio pack {wavs} --out data/audio/lj")
    return 0


def cmd_encode(args) -> int:
    from .codec import load_codec
    from .dataset import encode_corpus

    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    codec = load_codec(args.checkpoint, device)
    encode_corpus(codec, args.corpus, args.out)
    # Name the config that reads these codes rather than always pointing at the smoke one.
    import yaml
    users = [c.stem for c in sorted(Path("configs").glob("audiolm-*.yaml"))
             if ((yaml.safe_load(c.read_text()) or {}).get("data") or {}).get("codes") == args.out]
    print("\nnow train the audio LM:  " + (" or ".join(f"scripts/audio.sh {u}" for u in users)
                                         or f"a configs/audiolm-*.yaml with data.codes: {args.out}"))
    return 0


def cmd_encode_many(args) -> int:
    """Several packed corpora -> one shuffled codes directory, with speakers (multi-speaker TTS)."""
    from .codec import load_codec
    from .dataset import encode_many

    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    codec = load_codec(args.checkpoint, device)
    spk = None
    if args.speaker:
        from .speaker import load_speaker
        spk = load_speaker(args.speaker, device).eval()
    import functools
    encode_many(codec, args.corpus, args.out, speaker_model=spk, seed=args.seed,
                progress=functools.partial(print, flush=True))
    return 0


def _codec_clips(args, sr: int):
    """Validation clips to measure on: the corpus's held-out split, or the files given."""
    import torch as _t

    if args.wav:
        return [_t.from_numpy(load_audio(w, sample_rate=sr).samples) for w in args.wav]
    from .dataset import AudioDataset

    ds = AudioDataset(args.corpus, 16_000, split="val", val_clips=args.val_clips)
    return [ds.clip(i) for i in range(min(args.clips, len(ds.clips)))]


def cmd_codec_report(args) -> int:
    from .codec import load_codec
    from .measure import bitrate_ladder, codebook_usage

    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    model = load_codec(args.checkpoint, device)
    cfg = model.cfg
    counts = model.n_params()
    print(f"checkpoint  {args.checkpoint}  (step {model.step}, best val {model.best_val:.4f})")
    print(f"codec       {cfg.describe()}")
    print(f"params      {counts['total'] / 1e6:.2f}M trainable "
          f"+ {counts['codebooks'] / 1e6:.2f}M of codebook")
    print(f"device      {device}\n")

    clips = _codec_clips(args, cfg.sample_rate)
    print(f"measured on {len(clips)} held-out clips\n")

    print(f"{'books':>6} {'kbps':>7} {'vs PCM':>8} {'converge':>9} {'MCD dB':>8}")
    for row in bitrate_ladder(model, clips):
        print(f"{row['codebooks']:>6} {row['kbps']:>7.2f} {row['compression']:>7.0f}x "
              f"{row['convergence']:>9.4f} {row['mcd_db']:>8.2f}")
    print("\n  MCD: under ~4 dB is very close, 6-8 dB recognisably degraded. Ours is")
    print("  calibrated on constructed distortions, not against a paper's toolchain —")
    print("  compare it across our own checkpoints. Both columns are proxies; the table")
    print("  at the top of aksharallm/audio/measure.py says what each one cannot see.\n")

    print(f"{'codebook':>9} {'used':>6} {'perplexity':>11} {'usage':>7}")
    for row in codebook_usage(model, clips):
        print(f"{row['codebook']:>9} {row['used']:>4}/{row['size']} "
              f"{row['perplexity']:>11.1f} {row['usage'] * 100:>6.1f}%")
    print("\n  A RISING curve is normal: later stages quantize a residual closer to noise,")
    print("  and noise spreads evenly over a codebook. Collapse is a perplexity of a few")
    print("  dozen out of a thousand ANYWHERE — read the numbers, not the shape.")
    return 0


def transfer_suffix(args) -> str:
    """A smaller or different sample gets its own file name, so a quick check can never
    overwrite the full report the docs and the portal quote (it nearly did, 2026-11-19)."""
    default = (args.home_clips == 64 and args.per_speaker == 3
               and Path(args.home).name == "lj" and Path(args.other).name == "test-clean")
    judged = getattr(args, "judged", "4,8")
    tail = "" if judged == "4,8" else "-judged" + judged.replace(",", "-")
    return tail if default else (f"-{Path(args.home).name}{args.home_clips}"
                                 f"-{Path(args.other).name}x{args.per_speaker}{tail}")


def cmd_codec_transfer(args) -> int:
    """`transfer.measure_corpus` on the codec's own voice and on voices it never heard."""
    import json as _json
    from ..asr.__main__ import resolve as resolve_asr
    from ..asr.measure import transcribe as asr_transcribe
    from ..asr.train import load_recognizer
    from .codec import load_codec
    from .transfer import measure_corpus, pick_clips

    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    codec = load_codec(args.checkpoint, device)
    asr_model, _ = load_recognizer(resolve_asr(args.asr), device)
    embed_fn = None
    if args.speaker:
        from .speaker import embed, load_speaker
        spk_model = load_speaker(args.speaker, device)
        embed_fn = lambda ws: embed(spk_model, ws, device)  # noqa: E731
    run = Path(args.checkpoint).parent.name
    out_dir = Path(args.out_dir or f"logs/audio/codec-transfer-{run}")
    sr = codec.cfg.sample_rate
    print(f"codec       {args.checkpoint} (step {codec.step}), {codec.cfg.describe()}")
    print(f"judge       {args.asr}, greedy\ndevice      {device}\n")
    corpora = {
        "home": pick_clips(args.home, last=args.home_clips, sample_rate=sr),
        "other": pick_clips(args.other, per_speaker=args.per_speaker, sample_rate=sr),
    }
    result = {"checkpoint": str(args.checkpoint), "step": codec.step, "judge": args.asr,
              "corpora": {"home": str(args.home), "other": str(args.other)}}
    for key, clips in corpora.items():
        d = out_dir / key
        d.mkdir(parents=True, exist_ok=True)
        keep = (lambda j, label, w, d=d: write_wav(d / f"{j:02d}-{label}.wav", w, sr)
                if j < args.keep else None)
        print(f"{key:<6} {len(clips)} clips from {len({c['speaker'] for c in clips})} "
              f"speaker(s) ...", flush=True)
        result[key] = measure_corpus(codec, clips,
                                     lambda ws: asr_transcribe(asr_model, ws, device),
                                     keep=keep, embed_fn=embed_fn,
                                     judged=tuple(int(x) for x in args.judged.split(",")))
    if args.speaker:
        result["speaker_model"] = str(args.speaker)
    print()
    rungs = list(result["home"]["mcd"])
    print(f"{'':>8} " + " ".join(f"{'MCD@' + str(r):>7}" for r in rungs)
          + f" {'WER orig':>9} " + " ".join(f"{'@' + k[5:]:>7}" for k in
                                             result["home"]["paired_wer"]))
    for key in ("home", "other"):
        r = result[key]
        print(f"{key:>8} " + " ".join(f"{r['mcd'][n]['mean']:>7.2f}" for n in rungs)
              + f" {r['wer']['original'] * 100:>8.1f}% "
              + " ".join(f"{r['wer'][k] * 100:>6.1f}%" for k in r["paired_wer"]))
    print("\n  paired, per clip: did the codec make the recogniser do worse on THIS clip?")
    for key in ("home", "other"):
        for k, pw in result[key]["paired_wer"].items():
            p = "-" if pw["p"] is None else f"{pw['p']:.3g}"
            print(f"  {key:>6} {k:>7}: worse {pw['worse']:>3}  better {pw['better']:>3}  "
                  f"same {pw['same']:>3}   sign test p {p}")
    if args.speaker:
        print("\n  voice identity (speaker model, cosine to the clip's own original):")
        for key in ("home", "other"):
            r = result[key]
            ref = r["voice_ref"]
            fmt = lambda v: "   -" if v is None else f"{v:.3f}"  # noqa: E731
            print(f"  {key:>6} " + "  ".join(f"{k}: {v:.3f}" for k, v in r["voice"].items())
                  + f"   | same speaker, other clip {fmt(ref['same_speaker'])}"
                  f"   different speakers {fmt(ref['different_speaker'])}")
    worst = result["other"]["per_speaker"][:5]
    print("\n  other voices with the highest MCD@%d:" % rungs[-1])
    for s in worst:
        print(f"    speaker {s['speaker']:>5}  MCD {s['mcd']:.2f}  "
              f"WER {s['wer_original'] * 100:.0f}% -> {s['wer_codec'] * 100:.0f}%")
    print(f"\n  MCD sees the envelope, not identity: listen to {out_dir}/other/NN-codec8.wav")
    print("  beside NN-original.wav to hear whether a voice still sounds like itself.")
    path = Path("logs/audio") / f"codec-transfer-{run}-step{codec.step}{transfer_suffix(args)}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_json.dumps(result, indent=1))
    print(f"wrote {path}")
    return 0


def cmd_speaker_eval(args) -> int:
    """EER over every pair of clips from unseen speakers; JSON in logs/audio."""
    import json as _json
    from .speaker import load_speaker, verification
    from .transfer import pick_clips

    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    model = load_speaker(args.checkpoint, device)
    blob = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    clips = pick_clips(args.corpus, per_speaker=args.per_speaker,
                       sample_rate=model.cfg.encoder.sample_rate)
    r = verification(model, clips, device)
    print(f"speaker model  {args.checkpoint} (step {blob.get('step')})")
    print(f"corpus         {args.corpus}: {r['speakers']} speakers, {r['clips']} clips, "
          f"{r['pairs']} pairs")
    print(f"EER            {r['eer'] * 100:.2f}%   (chance 50%)")
    print(f"similarity     same speaker {r['same_speaker']:.3f}   "
          f"different speakers {r['different_speaker']:.3f}")
    run = Path(args.checkpoint).parent.name
    path = Path("logs/audio") / f"speaker-eval-{run}-step{blob.get('step')}-{Path(args.corpus).name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_json.dumps({"checkpoint": str(args.checkpoint), "step": blob.get("step"),
                                 "corpus": str(args.corpus), **r}, indent=1))
    print(f"wrote {path}")
    return 0


def cmd_tts_report(args) -> int:
    """`tts_ladder` over held-out LJSpeech-style sentences, WAVs written, JSON in logs/audio."""
    import json as _json
    from ..asr.__main__ import resolve as resolve_asr
    from ..asr.measure import transcribe as asr_transcribe
    from ..asr.train import load_recognizer
    from .codec import load_codec
    from .dataset import Manifest, UtteranceDataset
    from .speech import load_tts, tts_ladder
    from .text import load_transcripts

    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    model, tok, blob = load_tts(args.checkpoint, device)
    codec = load_codec(blob["codec"], device)
    data = blob["config"]["data"]
    corpus = Path(data.get("corpus") or data["codes"])
    val = UtteranceDataset(data["codes"], load_transcripts(corpus), tok, split="val",
                           n_codebooks=model.cfg.n_codebooks, max_frames=model.cfg.max_frames,
                           max_text=model.cfg.max_text, val_clips=data["val_clips"])
    # The original recordings: the packed corpus, or (a multi-speaker codes directory from
    # `encode-many`) every corpus it was made from, found by clip name.
    if (corpus / "audio.bin").is_file():
        sources = [corpus]
    else:
        sources = [Path(c) for c in Manifest.load(Path(data["codes"]) / "manifest.json").sources[0]["corpora"]]
    where = {}
    for c in sources:
        m = Manifest.load(c / "manifest.json")
        mm = np.memmap(c / "audio.bin", dtype="<i2", mode="r")
        for i, nm in enumerate(m.names):
            where[nm] = (mm, m.offsets[i], m.offsets[i + 1])
    names = val.base.manifest.names
    n = min(args.n, len(val))
    utts, origs = [], []
    for j in range(n):
        mm, a, b = where[names[val.items[j][0]]]
        utts.append(val.get(j))
        origs.append(mm[a:b].astype(np.float32) / 32768.0)
    speakers = None
    if model.spk_proj is not None:
        # Each sentence spoken in its own speaker's voice, from a DIFFERENT clip by them.
        val = UtteranceDataset(data["codes"], load_transcripts(corpus), tok, split="val",
                               n_codebooks=model.cfg.n_codebooks, max_frames=model.cfg.max_frames,
                               max_text=model.cfg.max_text, val_clips=data["val_clips"],
                               speakers=True)
        rng = np.random.default_rng(args.seed)
        speakers = [val.reference(j, rng) for j in range(n)]
        who = [val.speaker[val.items[j][0]] for j in range(n)]

    run = Path(args.checkpoint).parent.name
    out_dir = Path(args.out_dir or f"logs/audio/tts-{run}-{args.duration}")
    out_dir.mkdir(parents=True, exist_ok=True)
    asr_model, _ = load_recognizer(resolve_asr(args.asr), device)
    print(f"tts         {args.checkpoint} (step {blob.get('step')}), {n} held-out sentences")
    print(f"judge       {args.asr}, greedy\ndevice      {device}\n")
    fpc = blob.get("frames_per_char") or _frames_per_char(blob, tok, model)
    kept = {}

    def keep(j, kind, w):
        kept[(j, kind)] = w
        write_wav(out_dir / f"{j:02d}-{kind}.wav", w, codec.cfg.sample_rate)

    rep = tts_ladder(model, tok, codec, utts, origs,
                     lambda ws: asr_transcribe(asr_model, [_min_len(w) for w in ws], device),
                     temperature=args.temperature, seed=args.seed, device=device,
                     keep=keep, duration=args.duration, frames_per_char=fpc,
                     speakers=speakers)
    if speakers is not None:
        rep["voice"] = _tts_voice(data["speaker_model"], kept, who, n, device)
        v = rep["voice"]
        print(f"voice: cosine of the TTS to its own speaker's recording {v['tts_own']:.3f}, "
              f"to a different speaker's {v['tts_other']:.3f}; paired, own closer in "
              f"{v['own_closer']}/{v['pairs']} (sign test p {v['p']:.3g}). Reference: two "
              f"recordings by one speaker {v['same_speaker']:.3f}, codec {v['codec_own']:.3f}")
    print(f"{'row':>9} {'WER':>7} {'CER':>7} {'runaway':>8}")
    for kind, row in rep["rows"].items():
        print(f"{kind:>9} {row['wer'] * 100:>6.1f}% {row['cer'] * 100:>6.1f}% {row['runaway']:>8}")
    ex = rep["rows"]["tts"]["items"][0]
    print(f"\n  e.g. said  {ex['ref']}\n       heard {ex['hyp']}")
    print(f"  TTS duration: {rep['duration']}; spoken/real length median "
          f"{rep['length_ratio']['median']:.2f}. Intelligibility is not quality:")
    print(f"  listen to {out_dir}/NN-tts.wav beside NN-original.wav.")
    path = Path("logs/audio") / f"tts-{run}-step{blob.get('step')}-{args.duration}.json"
    path.write_text(_json.dumps({"checkpoint": str(args.checkpoint), "step": blob.get("step"),
                                 "judge": args.asr, **rep}, indent=1))
    print(f"wrote {path}")
    return 0


def _min_len(w, sr: int = 16_000):
    """A model that stops after a frame or two (early in training) makes a clip shorter than
    the judges' convolutions can read, and one such clip would crash the whole report.
    Padded with silence to 0.25 s for judging only; the length ratio still records it."""
    return np.pad(w, (0, max(0, sr // 4 - len(w))))


def _tts_voice(speaker_ckpt, kept, who, n, device) -> dict:
    """Does a multi-speaker TTS speak in the RIGHT voice? speaker-libri cosines, paired.

    For each sentence: its TTS output against its own speaker's original recording, and
    against the original of the next sentence by a DIFFERENT speaker. If conditioning works,
    own > other on most sentences. The codec row is the ceiling the codes allow."""
    from math import comb

    from .speaker import embed, load_speaker
    spk = load_speaker(speaker_ckpt, device)
    e = {k: v for k, v in zip(kept, embed(spk, [_min_len(w) for w in kept.values()], device))}
    own, other, codec, wins, pairs = [], [], [], 0, 0
    for j in range(n):
        o = next((k for k in list(range(j + 1, n)) + list(range(j)) if who[k] != who[j]), None)
        t = e[(j, "tts")]
        own.append(float(t @ e[(j, "original")]))
        codec.append(float(e[(j, "codec")] @ e[(j, "original")]))
        if o is not None:
            other.append(float(t @ e[(o, "original")]))
            pairs += 1
            wins += own[-1] > other[-1]
    k = min(wins, pairs - wins)
    p = min(1.0, 2 * sum(comb(pairs, i) for i in range(k + 1)) / 2 ** pairs) if pairs else 1.0
    same = [float(e[(a, "original")] @ e[(b, "original")]) for a in range(n) for b in range(a + 1, n)
            if who[a] == who[b]]
    mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")  # noqa: E731
    return {"tts_own": mean(own), "tts_other": mean(other), "codec_own": mean(codec),
            "same_speaker": mean(same), "own_closer": wins, "pairs": pairs, "p": p,
            "per_item": [{"speaker": who[j], "tts_own": own[j]} for j in range(n)]}


def _frames_per_char(blob, tok, model) -> float:
    """For checkpoints saved before `frames_per_char` was: measure it from the corpus."""
    from .dataset import UtteranceDataset
    from .text import load_transcripts
    data = blob["config"]["data"]
    ds = UtteranceDataset(data["codes"], load_transcripts(data.get("corpus") or data["codes"]),
                          tok, n_codebooks=model.cfg.n_codebooks, max_frames=model.cfg.max_frames,
                          max_text=model.cfg.max_text, val_clips=data["val_clips"])
    return ds.frames_per_char()


def cmd_speak(args) -> int:
    """Any text, in the model's voice, to a WAV."""
    from .codec import load_codec
    from .dataset import normalise_text
    from .speech import load_tts, speak

    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    model, tok, blob = load_tts(args.checkpoint, device)
    text = normalise_text(args.text)
    ids = tok.encode(text)
    lost = sorted({c for c in text if c not in tok.stoi})
    if lost:
        print(f"note        these characters are not in its alphabet and are skipped: {''.join(lost)!r}")
    if len(ids) > model.cfg.max_text:
        raise SystemExit(f"{len(ids)} characters; this model reads at most {model.cfg.max_text}. "
                         "Split the text into sentences.")
    how = (f"{args.frames} frames (given)" if args.frames else
           "its own stop head" if model.stop_head is not None else None)
    fpc = None
    if how is None:
        fpc = blob.get("frames_per_char") or _frames_per_char(blob, tok, model)
        how = f"{fpc:.2f} frames per character (it has no stop head)"
    codec = load_codec(blob["codec"], device)
    voice = None
    if model.spk_proj is not None:
        voice, said = _voice(args, blob, device)
        print(f"voice       {said}")
    gen = torch.Generator(device=device).manual_seed(args.seed)
    codes = speak(model, ids, args.frames, temperature=args.temperature, device=device,
                  generator=gen, frames_per_char=fpc, speaker=voice)
    wave = codec.decode(codes.to(device))[0].float().cpu().numpy()
    out = Path(args.out or f"logs/audio/speak/{'-'.join(text.split()[:6]) or 'speech'}.wav")
    write_wav(out, wave, codec.cfg.sample_rate)
    sec = codes.shape[-1] / codec.cfg.frames_per_second
    print(f"spoke       {text!r}\nlength      {codes.shape[-1]} frames = {sec:.1f} s, from {how}")
    print(f"wrote       {out}")
    return 0


def _voice(args, blob, device):
    """The speaker embedding a multi-speaker TTS speaks with: `--voice clip.wav` (anyone's,
    embedded by the run's speaker model), else the mean of `--speaker-id`'s training clips
    (default LJ, the voice every earlier TTS here used)."""
    import json as _json
    data = blob["config"]["data"]
    if args.voice:
        from .speaker import embed, load_speaker
        spk = load_speaker(data["speaker_model"], device)
        w = load_audio(args.voice, sample_rate=16_000).samples
        return embed(spk, [w], device)[0], f"from {args.voice}"
    d = Path(data["codes"])
    emb = np.load(d / "speaker_emb.npy").astype(np.float32)
    who = _json.loads((d / "speakers.json").read_text())
    names = _json.loads((d / "manifest.json").read_text())["names"]
    rows = [i for i, nm in enumerate(names) if who[Path(nm).name] == args.speaker_id]
    if not rows:
        raise SystemExit(f"no clips by speaker {args.speaker_id!r} in {d}")
    m = torch.from_numpy(emb[rows].mean(0))
    return m / m.norm(), f"speaker {args.speaker_id} (mean of {len(rows)} clips)"


def cmd_codec_reconstruct(args) -> int:
    from .codec import load_codec
    from .features import MelConfig, magnitude
    from .measure import mcd, reconstruct

    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    model = load_codec(args.checkpoint, device)
    sr = model.cfg.sample_rate
    clip = load_audio(args.wav, sample_rate=sr)
    x = torch.from_numpy(clip.samples)
    mel = MelConfig(sample_rate=sr)

    out_dir = Path(args.out_dir)
    write_wav(out_dir / "original.wav", clip.samples, sr)
    rungs = [int(n) for n in args.codebooks.split(",")]
    print(f"{args.wav}  {clip.seconds:.2f}s at {sr} Hz -> {out_dir}/\n")
    print(f"{'books':>6} {'kbps':>7} {'converge':>9} {'MCD dB':>8}  file")
    for n in rungs:
        y = reconstruct(model, x, n_codebooks=n).cpu()
        path = out_dir / f"codebooks{n}.wav"
        write_wav(path, y.numpy(), sr)
        kbps = model.cfg.frames_per_second * n * math.log2(model.cfg.codebook_size) / 1000
        conv = spectral_convergence(magnitude(x[: y.numel()], mel), magnitude(y, mel))
        print(f"{n:>6} {kbps:>7.2f} {conv:>9.4f} {mcd(x, y, mel):>8.2f}  {path.name}")
    print("\nPlay them in order against original.wav. The bitrate/quality trade is the")
    print("same one docs/11-quantization.md makes silently in the weights — here you hear it.")
    return 0


def cmd_codec_tokens(args) -> int:
    from .codec import load_codec

    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    model = load_codec(args.checkpoint, device)
    sr = model.cfg.sample_rate
    clip = load_audio(args.wav, sample_rate=sr)
    codes = model.encode(torch.from_numpy(clip.samples).unsqueeze(0).to(device))[0].cpu()
    n_books, frames = codes.shape
    print(f"{args.wav}: {clip.seconds:.2f}s -> {frames} frames x {n_books} codebooks "
          f"= {frames * n_books} tokens at {model.cfg.frames_per_second:.0f} frames/s")
    print(f"a language model over this pays {frames * n_books} positions for "
          f"{clip.seconds:.1f} seconds of audio.\n")
    show = min(frames, args.frames)
    for i in range(n_books):
        row = " ".join(f"{int(v):>4}" for v in codes[i, :show])
        print(f"  book {i}: {row}{' ...' if show < frames else ''}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m aksharallm.audio", description=__doc__)
    p.add_argument("--sr", type=int, default=TARGET_SR, help="resample everything to this")
    p.add_argument("--n-fft", type=int, default=1024)
    p.add_argument("--hop", type=int, default=256)
    p.add_argument("--n-mels", type=int, default=80)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("info", help="what is in this file")
    s.add_argument("wav")
    s.set_defaults(fn=cmd_info)

    s = sub.add_parser("spec", help="print the log-mel as a terminal heat map")
    s.add_argument("wav")
    s.add_argument("--width", type=int, default=100)
    s.add_argument("--height", type=int, default=24)
    s.set_defaults(fn=cmd_spec)

    s = sub.add_parser("roundtrip", help="spectrogram -> audio, and what it cost")
    s.add_argument("wav")
    s.add_argument("--mel", action="store_true", help="go through the 80-band log-mel too")
    s.add_argument("--iters", type=int, default=60)
    s.add_argument("--momentum", type=float, default=0.99)
    s.add_argument("--out")
    s.set_defaults(fn=cmd_roundtrip)

    s = sub.add_parser("resample", help="change a file's sample rate")
    s.add_argument("wav")
    s.add_argument("--to", type=int, default=TARGET_SR)
    s.add_argument("--out")
    s.set_defaults(fn=cmd_resample)

    s = sub.add_parser("tone", help="write a synthetic vowel to test the front end")
    s.add_argument("--out", default="tone.wav")
    s.add_argument("--seconds", type=float, default=2.0)
    s.add_argument("--f0", type=float, default=140.0)
    s.set_defaults(fn=cmd_tone)

    s = sub.add_parser("corpus", help="build the synthetic babble corpus (no download)")
    s.add_argument("--out", default="data/audio/synth")
    s.add_argument("--clips", type=int, default=400)
    s.add_argument("--seconds", type=float, default=2.0)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(fn=cmd_corpus)

    s = sub.add_parser("pack", help="pack a folder of WAVs into audio.bin + manifest.json")
    s.add_argument("dir")
    s.add_argument("--out", required=True)
    s.set_defaults(fn=cmd_pack)

    s = sub.add_parser("fetch", help="download a speech corpus (LJSpeech, 2.6 GB)")
    s.add_argument("corpus", choices=["ljspeech"])
    s.add_argument("--dest", default="data/audio/ljspeech")
    s.set_defaults(fn=cmd_fetch)

    def codec_args(sp):
        sp.add_argument("checkpoint")
        sp.add_argument("--cpu", action="store_true", help="force the CPU (a run has the card)")
        return sp

    s = codec_args(sub.add_parser("report", help="what a trained codec costs and preserves"))
    s.add_argument("--corpus", default="data/audio/synth")
    s.add_argument("--clips", type=int, default=8)
    s.add_argument("--val-clips", type=int, default=16)
    s.add_argument("--wav", nargs="*", default=None, help="measure these files instead")
    s.set_defaults(fn=cmd_codec_report)

    s = codec_args(sub.add_parser("reconstruct", help="the same clip at 1, 2, 4 and 8 codebooks"))
    s.add_argument("wav")
    s.add_argument("--codebooks", default="1,2,4,8")
    s.add_argument("--out-dir", default="logs/audio/reconstruct")
    s.set_defaults(fn=cmd_codec_reconstruct)

    s = codec_args(sub.add_parser("encode-many", help="several corpora -> one shuffled codes "
                                  "directory with speakers + speaker embeddings (multi-speaker TTS)"))
    s.add_argument("--corpus", action="append", required=True, help="repeat for each corpus")
    s.add_argument("--out", required=True)
    s.add_argument("--speaker", default=None, metavar="PT", help="speaker-libri checkpoint: "
                   "also write one embedding per clip (speaker_emb.npy)")
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(fn=cmd_encode_many)

    s = codec_args(sub.add_parser("encode", help="turn a packed corpus into codec tokens"))
    s.add_argument("--corpus", default="data/audio/synth")
    s.add_argument("--out", default="data/audio/synth-codes")
    s.set_defaults(fn=cmd_encode)

    s = codec_args(sub.add_parser("tokens", help="print the integers a clip becomes"))
    s.add_argument("wav")
    s.add_argument("--frames", type=int, default=16)
    s.set_defaults(fn=cmd_codec_tokens)

    s = codec_args(sub.add_parser("codec-transfer", help="does the codec carry voices it never "
                                                    "heard? MCD + recogniser WER, paired"))
    s.add_argument("--home", default="data/audio/lj", help="the corpus the codec trained on")
    s.add_argument("--home-clips", type=int, default=64,
                   help="its last N clips: the codec's own held-out split (val_clips)")
    s.add_argument("--other", default="data/asr/test-clean", help="voices it never heard")
    s.add_argument("--per-speaker", type=int, default=3)
    s.add_argument("--asr", default="asr-libri460", help="the recogniser that judges")
    s.add_argument("--keep", type=int, default=6, help="clips per corpus written as WAVs")
    s.add_argument("--judged", default="4,8", help="codebook counts the recogniser judges; "
                   "'1,4,8' adds codebook 0 alone, the test of a content codebook (own JSON name)")
    s.add_argument("--out-dir", default=None)
    s.add_argument("--speaker", default=None, metavar="PT",
                   help="a speaker model (train_speaker) -> a voice-identity column")
    s.set_defaults(fn=cmd_codec_transfer)

    s = sub.add_parser("speaker-eval", help="a speaker model's EER on speakers it never "
                                            "trained on, and its two reference similarities")
    s.add_argument("checkpoint")
    s.add_argument("--corpus", default="data/asr/test-clean")
    s.add_argument("--per-speaker", type=int, default=4)
    s.add_argument("--cpu", action="store_true")
    s.set_defaults(fn=cmd_speaker_eval)

    s = sub.add_parser("tts-report", help="TTS, judged by the dictation recogniser: "
                                          "original vs codec vs spoken, on held-out sentences")
    s.add_argument("checkpoint", help="an audio LM trained with data.task: tts")
    s.add_argument("--asr", default="asr-libri460", help="the recogniser that judges (run or .pt)")
    s.add_argument("--n", type=int, default=20, help="held-out sentences to speak")
    s.add_argument("--temperature", type=float, default=0.8)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--out-dir", default=None, help="WAVs to listen to (default logs/audio/tts-<run>)")
    s.add_argument("--duration", choices=["real", "own"], default="real",
                   help="real = given each sentence's recorded length; own = the model decides")
    s.add_argument("--cpu", action="store_true", help="force the CPU (a run has the card)")
    s.set_defaults(fn=cmd_tts_report)

    s = sub.add_parser("speak", help="say any text in a TTS model's voice, to a WAV")
    s.add_argument("checkpoint", help="an audio LM trained with data.task: tts")
    s.add_argument("text")
    s.add_argument("--out", default=None, help="default logs/audio/speak/<first words>.wav")
    s.add_argument("--voice", default=None, metavar="WAV", help="multi-speaker models: speak "
                   "in this recording's voice")
    s.add_argument("--speaker-id", default="LJ", help="multi-speaker models without --voice: "
                   "a speaker in the training set (LibriSpeech id, or LJ)")
    s.add_argument("--frames", type=int, default=None, help="force a length (50 per second)")
    s.add_argument("--temperature", type=float, default=0.8)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--cpu", action="store_true", help="force the CPU (a run has the card)")
    s.set_defaults(fn=cmd_speak)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
