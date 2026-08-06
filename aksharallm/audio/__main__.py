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

Read with: docs/20-audio.md -- the chapter this implements; it ends with the order to read
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
    print("\nnow train the audio LM:  scripts/audio.sh audiolm-synth")
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
    print("same one docs/10-quantization.md makes silently in the weights — here you hear it.")
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

    s = codec_args(sub.add_parser("encode", help="turn a packed corpus into codec tokens"))
    s.add_argument("--corpus", default="data/audio/synth")
    s.add_argument("--out", default="data/audio/synth-codes")
    s.set_defaults(fn=cmd_encode)

    s = codec_args(sub.add_parser("tokens", help="print the integers a clip becomes"))
    s.add_argument("wav")
    s.add_argument("--frames", type=int, default=16)
    s.set_defaults(fn=cmd_codec_tokens)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
