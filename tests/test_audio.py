"""The audio front end, pinned to things that are true rather than things that look right.

A spectrogram is a picture, and a picture that is wrong is still a picture — the same rule
`tests/test_interp.py` is built on. Every check here is against something unarguable:

* `istft(stft(x)) == x` to float32 precision, so the reconstruction loss in the codec is
  measuring the model and not the window;
* the mel filterbank passes white noise as white noise, so the network does not spend its
  first epochs unlearning a tilt we built in;
* a resampled sine keeps its frequency and its amplitude, and a tone above the new Nyquist
  rate is *attenuated* rather than folded down to a frequency that was never there;
* Griffin-Lim's spectral convergence goes down, and goes down faster with momentum.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from aksharallm.audio.features import (
    MelConfig,
    frame,
    griffin_lim,
    hann,
    hz_to_mel,
    istft,
    log_mel,
    magnitude,
    mel_filterbank,
    mel_to_hz,
    mel_to_magnitude,
    spectral_convergence,
    stft,
)
from aksharallm.audio.io import Clip, load_audio, read_wav, resample, to_mono, write_wav

CFG = MelConfig()


def vowel(seconds: float = 1.0, sr: int = 16_000, f0: float = 140.0) -> torch.Tensor:
    """A buzzy harmonic stack under a slow tremolo — a stand-in for speech.

    White noise would pass most of these tests while hiding the ones that matter: a broken
    filterbank still looks like noise, and Griffin-Lim converges trivially on noise because
    any phase is as good as any other.
    """
    t = torch.arange(int(sr * seconds), dtype=torch.float32) / sr
    x = sum((0.6 / k) * torch.sin(2 * math.pi * f0 * k * t) for k in range(1, 12))
    x = x * (0.5 + 0.5 * torch.sin(2 * math.pi * 3 * t))
    return (x / x.abs().max() * 0.7).float()


# ---------------------------------------------------------------------------------------
# windows and framing
# ---------------------------------------------------------------------------------------


def test_hann_is_periodic_not_symmetric():
    """The periodic window is the one that satisfies COLA. The symmetric variant differs by
    one sample of stretch and leaves a slow ripple in every reconstruction."""
    w = hann(8)
    assert w[0] == pytest.approx(0.0)
    assert w[4] == pytest.approx(1.0)  # peak lands exactly on n/2
    assert w[-1] != pytest.approx(0.0)  # symmetric would end at 0; periodic does not


def test_overlapping_hann_squares_sum_to_a_constant():
    """The COLA condition at hop = n_fft/4, which is why the reconstruction below is exact."""
    n, hop = 64, 16
    w = hann(n) ** 2
    acc = torch.zeros(n * 8)
    for i in range(0, len(acc) - n, hop):
        acc[i : i + n] += w
    interior = acc[n:-n]
    assert interior.std() < 1e-6, "hop is not COLA for this window"


def test_frames_overlap_and_share_memory():
    x = torch.arange(20, dtype=torch.float32)
    f = frame(x, 8, 4)
    assert f.shape == (4, 8)
    assert torch.equal(f[1], x[4:12])
    assert f.untyped_storage().data_ptr() == x.untyped_storage().data_ptr()


# ---------------------------------------------------------------------------------------
# the transform
# ---------------------------------------------------------------------------------------


def test_istft_inverts_stft_exactly():
    """The load-bearing test of the file. If this drifts, every reconstruction number in the
    codec is partly a measurement of the window."""
    x = vowel(0.5)
    y = istft(stft(x, CFG), CFG, length=x.numel())
    assert (y - x).abs().max() < 1e-5


def test_the_transform_keeps_leading_dimensions():
    x = torch.randn(3, 2, 4096)
    spec = stft(x, CFG)
    assert spec.shape[:2] == (3, 2)
    assert spec.shape[2] == CFG.n_freqs
    assert istft(spec, CFG, length=4096).shape == (3, 2, 4096)


def test_a_batch_transforms_identically_to_one_at_a_time():
    """Same argument as the eval harness's mixed-length batch test: a shape bug that only
    appears with a batch dimension is invisible in every single-clip experiment."""
    xs = torch.stack([vowel(0.3, f0=f) for f in (110.0, 150.0, 190.0)])
    batched = stft(xs, CFG)
    for i in range(len(xs)):
        assert torch.allclose(batched[i], stft(xs[i], CFG), atol=1e-5)


def test_a_pure_tone_lands_in_the_right_bin():
    sr = CFG.sample_rate
    t = torch.arange(sr, dtype=torch.float32) / sr
    for hz in (200.0, 1000.0, 4000.0):
        mag = magnitude(torch.sin(2 * math.pi * hz * t), CFG)
        peak = int(mag.mean(-1).argmax())
        assert abs(peak * sr / CFG.n_fft - hz) < sr / CFG.n_fft, f"{hz} Hz landed at bin {peak}"


def test_the_first_frame_describes_the_beginning():
    """`center=True` means frame 0 is centred on sample 0. Without it the first 32 ms of
    every clip is described by no frame at all, which for a 1-second utterance is 3%."""
    x = vowel(0.5)
    frames = stft(x, CFG).shape[-1]
    assert frames == 1 + x.numel() // CFG.hop


# ---------------------------------------------------------------------------------------
# the mel scale
# ---------------------------------------------------------------------------------------


def test_mel_and_hz_invert_each_other():
    hz = np.array([0.0, 100.0, 700.0, 1000.0, 4000.0, 8000.0])
    assert np.allclose(mel_to_hz(hz_to_mel(hz)), hz, atol=1e-6)


def test_the_mel_scale_is_compressive_in_hertz():
    """The property that makes it worth using: the same *gap in Hz* is worth fewer mels the
    higher up it sits, so the filterbank spends its bands where hearing has resolution.

    Stated in octaves it goes the other way — an octave costs a constant ~780 mel in the log
    regime and less than that down in the linear one — which is why this is written in Hz.
    """
    low = hz_to_mel(400.0) - hz_to_mel(200.0)
    high = hz_to_mel(6400.0) - hz_to_mel(6200.0)
    assert high < low / 5


def test_filterbank_shape_and_support():
    fb = mel_filterbank(CFG)
    assert fb.shape == (CFG.n_mels, CFG.n_freqs)
    assert (fb >= 0).all(), "a triangular filter is never negative"
    assert (fb.sum(dim=1) > 0).all(), "an empty mel band means n_mels is too high for n_fft"
    # Each band peaks higher than the last: the triangles march up the spectrum in order.
    peaks = fb.argmax(dim=1)
    assert (peaks[1:] >= peaks[:-1]).all()


def test_white_noise_stays_white_through_the_filterbank():
    """Unit-area normalisation, asserted. Without it the wide high bands pass more energy
    than the narrow low ones and a flat spectrum comes out as a rising ramp."""
    torch.manual_seed(0)
    fb = mel_filterbank(CFG)
    mel = (fb @ magnitude(torch.randn(CFG.sample_rate * 4) * 0.1, CFG)).mean(-1)
    # Skip the first band: it starts at 0 Hz where there is no room for a full triangle.
    body = mel[1:]
    assert float(body.max() / body.min()) < 1.5


def test_log_mel_has_a_floor():
    """Silence must land on the floor rather than at -inf, or one silent frame is the whole
    batch's loss."""
    silence = torch.zeros(CFG.sample_rate // 2)
    m = log_mel(silence, CFG)
    assert torch.isfinite(m).all()
    assert m.min() == pytest.approx(math.log(CFG.log_eps), abs=1e-5)


def test_the_mel_front_end_is_lossy_and_says_so():
    """80 bands cannot reconstruct 513 bins. The pseudo-inverse is a best guess, and this
    test exists so that number is *recorded* rather than assumed to be small."""
    x = vowel(0.5)
    target = magnitude(x, CFG)
    approx = mel_to_magnitude(log_mel(x, CFG), CFG)
    conv = spectral_convergence(target, approx)
    assert 0.05 < conv < 0.6, f"mel round trip = {conv:.3f}; if this collapsed, so did a band"


# ---------------------------------------------------------------------------------------
# Griffin-Lim
# ---------------------------------------------------------------------------------------


def test_griffin_lim_converges():
    x = vowel(0.5)
    mag = magnitude(x, CFG)
    errs = [
        spectral_convergence(mag, magnitude(griffin_lim(mag, CFG, n_iter=n, length=x.numel()), CFG))
        for n in (1, 10, 40)
    ]
    assert errs[0] > errs[1] > errs[2], errs
    assert errs[-1] < 0.15, f"40 iterations should be clearly recognisable, got {errs[-1]:.3f}"


def test_momentum_helps():
    """Fast Griffin-Lim earns its name, and the `momentum/(1+momentum)` denominator is why.
    Feeding the raw momentum in overshoots and converges *worse* than no momentum at all —
    a mistake made here once and caught by exactly this comparison."""
    x = vowel(0.5)
    mag = magnitude(x, CFG)
    plain = griffin_lim(mag, CFG, n_iter=30, momentum=0.0, length=x.numel())
    fast = griffin_lim(mag, CFG, n_iter=30, momentum=0.99, length=x.numel())
    assert spectral_convergence(mag, magnitude(fast, CFG)) < spectral_convergence(
        mag, magnitude(plain, CFG)
    )


def test_griffin_lim_is_reproducible():
    mag = magnitude(vowel(0.2), CFG)
    a = griffin_lim(mag, CFG, n_iter=5, seed=3)
    b = griffin_lim(mag, CFG, n_iter=5, seed=3)
    assert torch.equal(a, b)


# ---------------------------------------------------------------------------------------
# reading, writing, resampling
# ---------------------------------------------------------------------------------------


def test_wav_round_trip(tmp_path):
    x = vowel(0.25).numpy()
    p = tmp_path / "a.wav"
    write_wav(p, x, 16_000)
    back, sr = read_wav(p)
    assert sr == 16_000 and back.shape == (len(x), 1)
    # 16-bit quantisation is the only loss, and it is one part in 32768.
    assert np.abs(back[:, 0] - x).max() < 2e-4


def test_writing_clips_rather_than_wrapping(tmp_path):
    """A sample of 1.5 must come back as full scale, not as a wrap-around to -0.5. Wrapping
    is the loud, obvious version of this bug and clipping is the quiet one; both are wrong,
    but only one of them destroys the file."""
    p = tmp_path / "loud.wav"
    write_wav(p, np.array([1.5, -1.5, 0.0], dtype=np.float32), 16_000)
    back, _ = read_wav(p)
    assert back[0, 0] > 0.99 and back[1, 0] < -0.99


def test_stereo_is_averaged_not_summed():
    """Summing makes every stereo clip in a corpus 6 dB louder than every mono one."""
    both = np.stack([np.full(10, 0.5), np.full(10, 0.5)], axis=1).astype(np.float32)
    assert to_mono(both).max() == pytest.approx(0.5)


def test_load_audio_reports_what_it_changed(tmp_path):
    """The assertion rule of the phase: conversions are recorded, never silent."""
    p = tmp_path / "b.wav"
    write_wav(p, vowel(0.2, sr=22_050).numpy(), 22_050)
    clip = load_audio(p, sample_rate=16_000)
    assert isinstance(clip, Clip)
    assert clip.orig_sr == 22_050 and clip.sample_rate == 16_000
    assert clip.orig_channels == 1
    assert "resampled from 22050" in clip.describe()
    assert abs(clip.seconds - 0.2) < 0.01


def test_load_audio_does_not_normalise_by_default(tmp_path):
    """Per-clip normalisation destroys the relative loudness of a corpus, so it is opt-in."""
    p = tmp_path / "quiet.wav"
    write_wav(p, (vowel(0.2) * 0.1).numpy(), 16_000)
    assert np.abs(load_audio(p).samples).max() < 0.15
    assert np.abs(load_audio(p, normalize=True).samples).max() == pytest.approx(0.95, abs=0.01)


@pytest.mark.parametrize("hz", [100.0, 440.0, 2000.0, 6000.0])
def test_resampling_preserves_frequency_and_amplitude(hz):
    sr = 16_000
    t = np.arange(sr, dtype=np.float64) / sr
    x = (0.5 * np.sin(2 * math.pi * hz * t)).astype(np.float32)
    y = resample(x, sr, 22_050)
    assert len(y) == pytest.approx(sr * 22_050 / 16_000, rel=1e-3)
    assert y.std() == pytest.approx(x.std(), rel=0.02)
    spec = np.abs(np.fft.rfft(y * np.hanning(len(y))))
    peak = np.fft.rfftfreq(len(y), 1 / 22_050)[spec.argmax()]
    assert abs(peak - hz) < 3.0


def test_resampling_round_trips():
    x = vowel(0.5).numpy()
    y = resample(resample(x, 16_000, 22_050), 22_050, 16_000)
    n = min(len(x), len(y))
    # Trim the edges: the kernel needs a full window either side and the corpus builder
    # never cares about the first millisecond.
    assert np.abs(y[:n] - x[:n])[200:-200].max() < 1e-3


def test_downsampling_attenuates_rather_than_aliases():
    """The reason the sinc is stretched when going down. A 10 kHz tone cannot exist at
    16 kHz; the only question is whether it disappears or comes back as 6 kHz."""
    sr = 22_050
    t = np.arange(sr, dtype=np.float64) / sr
    loud = (0.5 * np.sin(2 * math.pi * 10_000 * t)).astype(np.float32)
    quiet = resample(loud, sr, 16_000)
    assert quiet.std() < 0.01 * loud.std(), "the tone folded down instead of being filtered"


def test_a_resampler_that_does_nothing_copies():
    x = np.arange(10, dtype=np.float32)
    y = resample(x, 16_000, 16_000)
    assert np.array_equal(x, y) and y is not x
