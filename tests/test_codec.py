"""The codec: the quantizer, the conv stack, and the numbers we report about them.

Three of these tests exist because the thing they check went wrong during the build, and
each failure was silent:

* the codebook ran in **bf16** under autocast, where an EMA of 1% per step is below the
  dtype's own resolution — so it would have stopped learning with nothing to show for it;
* `spectral_convergence` divides by its **first** argument, and passing the reconstruction
  first made a half-trained codec look worse than it was (0.72 read as 1.10);
* MCD with an **absolute** log floor measured inaudible bands 80 dB down, and scored a clip
  against itself-plus-inaudible-noise at 86 dB.

The rest pin the things that would be invisible if wrong: the straight-through estimator's
direction, the residual prefix property that the whole bitrate demo rests on, and the
encoder/decoder agreeing about length when a stride is odd.
"""

from __future__ import annotations

import math

import pytest
import torch

from aksharallm.audio.codec import (
    Codec,
    CodecConfig,
    Down,
    ReconstructionLoss,
    Up,
    codebook_report,
    load_codec,
)
from aksharallm.audio.features import MelConfig, magnitude, spectral_convergence
from aksharallm.audio.measure import bitrate_ladder, cepstrum, codebook_usage, mcd, reconstruct
from aksharallm.audio.vq import ResidualVQ, VectorQuantizer

#: Small enough that the whole file runs in a few seconds on a CPU.
TINY = CodecConfig(channels=8, strides=(2, 4, 5), dim=16, n_codebooks=4, codebook_size=32)


def vowel(n: int = 8000, f0: float = 140.0) -> torch.Tensor:
    t = torch.arange(n, dtype=torch.float32) / 16_000
    x = sum((0.6 / k) * torch.sin(2 * math.pi * f0 * k * t) for k in range(1, 10))
    return (x / x.abs().max() * 0.7).float()


# ---------------------------------------------------------------------------------------
# the quantizer
# ---------------------------------------------------------------------------------------


def test_the_forward_value_is_a_codebook_entry():
    """Whatever the gradient does, the *value* leaving the quantizer must be one of the
    vectors in the table — otherwise the integers do not describe the signal.

    In `eval()`: during training the EMA moves the codebook *after* the lookup, so a
    re-lookup afterwards legitimately disagrees. That is the update working, not a bug.
    """
    q = VectorQuantizer(4, 8).eval()
    z = torch.randn(2, 4, 6)
    out, idx, _, _ = q(z)
    assert torch.allclose(out, q.decode(idx), atol=1e-6)


def test_the_straight_through_gradient_reaches_the_encoder():
    """`z + (q − z).detach()` differentiates as the identity. Detaching the other side is
    the classic mistake, and it trains smoothly while the encoder learns nothing."""
    q = VectorQuantizer(4, 8)
    z = torch.randn(2, 4, 6, requires_grad=True)
    out, _, _, _ = q(z)
    out.sum().backward()
    assert z.grad is not None and z.grad.abs().sum() > 0
    # It really is the identity: d(out)/d(z) = 1 everywhere.
    assert torch.allclose(z.grad, torch.ones_like(z))


def test_the_codebook_stays_float32_under_autocast():
    """bf16 has eight bits of mantissa. An EMA that adds 1% of a centroid per step adds
    less than one ULP, so in bf16 the codebook silently stops moving."""
    if not torch.cuda.is_available():
        pytest.skip("autocast dtype behaviour is the point; needs a CUDA device")
    q = VectorQuantizer(8, 16).cuda()
    z = torch.randn(2, 8, 10, device="cuda")
    before = q.codebook.clone()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out, _, _, _ = q(z.to(torch.bfloat16))
    assert q.codebook.dtype == torch.float32
    assert not torch.equal(before, q.codebook), "the codebook did not move at all"
    assert out.dtype == torch.bfloat16, "the decoder should still get its fast dtype"


def test_the_codebook_is_seeded_from_real_data():
    """A `randn·0.02` codebook against an encoder whose outputs have scale 3 is a codebook
    where one entry is nearest to everything: perplexity 1.0 from step one."""
    q = VectorQuantizer(4, 16)
    z = torch.randn(2, 4, 40) * 5.0 + 10.0  # nowhere near the initialisation
    _, _, _, stats = q(z)
    assert bool(q.seeded)
    assert stats.perplexity > 4.0, "seeding failed: everything collapsed onto one entry"


def test_the_ema_pulls_an_entry_towards_the_vectors_that_chose_it():
    q = VectorQuantizer(2, 4, decay=0.5, restart_after=0)
    z = torch.full((1, 2, 20), 3.0)
    for _ in range(20):
        q(z)
    chosen = q.encode(z)[0, 0]
    assert torch.allclose(q.codebook[chosen], torch.full((2,), 3.0), atol=0.05)


def test_dead_entries_are_restarted_from_real_vectors():
    """Codebook collapse is router collapse. Without restarts a 32-entry table quietly
    becomes a 3-entry one and the loss curve does not say so."""
    q = VectorQuantizer(2, 32, restart_after=2)
    z = torch.randn(1, 2, 8)
    total = 0
    for _ in range(6):
        _, _, _, stats = q(z)
        total += stats.dead
    assert total > 0, "nothing was ever restarted"
    assert torch.isfinite(q.codebook).all()


def test_restarts_can_be_turned_off():
    """`restart_after: 0` exists so the collapse can be watched happening. That is a lesson,
    not a bug, and it has to keep working."""
    q = VectorQuantizer(2, 32, restart_after=0)
    for _ in range(10):
        _, _, _, stats = q(torch.randn(1, 2, 4))
        assert stats.dead == 0


def test_eval_mode_does_not_move_the_codebook():
    q = VectorQuantizer(4, 8)
    q(torch.randn(1, 4, 20))  # seed it
    q.eval()
    before = q.codebook.clone()
    q(torch.randn(1, 4, 20))
    assert torch.equal(before, q.codebook)


# ---------------------------------------------------------------------------------------
# residual VQ
# ---------------------------------------------------------------------------------------


def test_each_stage_shrinks_the_residual():
    """The whole idea: stage k+1 quantizes what stage k got wrong, so the error falls."""
    rvq = ResidualVQ(8, n_codebooks=4, size=64, dropout=False)
    z = torch.randn(4, 8, 50)
    for _ in range(30):  # let the codebooks settle
        rvq(z)
    rvq.eval()
    _, idx, _, _ = rvq(z)
    errs = [float((z - rvq.decode(idx, n_active=n)).pow(2).mean()) for n in (1, 2, 3, 4)]
    assert errs == sorted(errs, reverse=True), errs


def test_a_prefix_of_the_code_is_a_valid_code():
    """The property the entire bitrate demo rests on: decoding 1 of 4 codebooks is a
    coarser reconstruction, not a broken one."""
    rvq = ResidualVQ(8, n_codebooks=4, size=64, dropout=False)
    z = torch.randn(2, 8, 20)
    _, idx, _, _ = rvq(z)
    for n in (1, 2, 3, 4):
        out = rvq.decode(idx, n_active=n)
        assert out.shape == z.shape and torch.isfinite(out).all()


def test_quantizer_dropout_is_training_only():
    """Evaluating under dropout would make the val curve partly a record of which bitrates
    were drawn — the same argument as the diffusion trainer's fixed eval mask."""
    rvq = ResidualVQ(8, n_codebooks=8, size=32, dropout=True)
    rvq.eval()
    for _ in range(10):
        _, _, _, stats = rvq(torch.randn(1, 8, 10))
        assert len(stats) == 8


def test_inactive_codebooks_leave_their_indices_at_zero():
    rvq = ResidualVQ(8, n_codebooks=4, size=32, dropout=False)
    _, idx, _, stats = rvq(torch.randn(1, 8, 10), n_active=2)
    assert len(stats) == 2
    assert idx.shape[1] == 4
    assert (idx[:, 2:] == 0).all()


# ---------------------------------------------------------------------------------------
# the conv stack
# ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("stride", [2, 3, 4, 5, 8])
def test_a_stage_downsamples_by_exactly_its_stride(stride):
    """Odd strides are where symmetric padding is off by a frame, and the encoder and
    decoder then disagree about how long the signal is — invisible on a short clip, a
    growing offset on a long one."""
    d = Down(2, 3, stride, (1,))
    x = torch.randn(1, 2, stride * 7)
    assert d(x).shape[-1] == 7


@pytest.mark.parametrize("stride", [2, 3, 4, 5, 8])
def test_a_stage_and_its_mirror_round_trip_the_length(stride):
    d, u = Down(2, 3, stride, (1,)), Up(3, 2, stride, (1,))
    x = torch.randn(1, 2, stride * 7)
    assert u(d(x)).shape == x.shape


def test_the_codec_returns_exactly_the_length_it_was_given():
    m = Codec(TINY)
    for n in (4321, 8000, 40, 12345):
        y, _, _, _ = m(torch.randn(1, n) * 0.1)
        assert y.shape == (1, n), n


def test_the_frame_rate_is_the_product_of_the_strides():
    cfg = CodecConfig(strides=(2, 4, 5, 8), sample_rate=16_000)
    assert cfg.hop == 320
    assert cfg.frames_per_second == 50.0
    assert cfg.bits_per_second == pytest.approx(50 * 8 * 10)


def test_encode_gives_one_integer_per_codebook_per_frame():
    m = Codec(TINY).eval()
    n = TINY.hop * 25
    codes = m.encode(torch.randn(2, n) * 0.1)
    assert codes.shape == (2, TINY.n_codebooks, 25)
    assert codes.min() >= 0 and codes.max() < TINY.codebook_size


def test_the_output_is_bounded():
    """`tanh` on the last layer. Without it, early training happily emits samples of ±40 and
    the STFT loss chases them instead of the signal."""
    m = Codec(TINY)
    y, _, _, _ = m(torch.randn(1, 4000) * 5.0)
    assert y.abs().max() <= 1.0


def test_the_codebooks_are_not_optimizer_parameters():
    """They are updated by an EMA. As Parameters, weight decay would shrink every entry
    towards the origin between updates."""
    m = Codec(TINY)
    names = {n for n, _ in m.named_parameters()}
    assert not any("codebook" in n for n in names)
    assert m.n_params()["codebooks"] == TINY.n_codebooks * TINY.codebook_size * TINY.dim


def test_a_text_checkpoint_is_refused(tmp_path):
    """Audio tokens are a different vocabulary — a separate checkpoint family. Loading one
    into the other must fail loudly rather than produce plausible nonsense."""
    p = tmp_path / "text.pt"
    torch.save({"model": {}, "stage": "base", "step": 5}, p)
    with pytest.raises(ValueError, match="not a codec checkpoint"):
        load_codec(p)


# ---------------------------------------------------------------------------------------
# the loss
# ---------------------------------------------------------------------------------------


def test_the_reconstruction_loss_is_zero_for_a_perfect_copy():
    x = vowel()
    loss, parts = ReconstructionLoss()(x, x.clone())
    assert float(loss) == pytest.approx(0.0, abs=1e-5)
    assert all(v == pytest.approx(0.0, abs=1e-5) for v in parts.values())


def test_the_time_domain_calls_an_inaudible_shift_a_total_mismatch():
    """Half of why the loss is not an L2 on the waveform, and the half that can be asserted:
    eight samples of shift at 16 kHz is half a millisecond, which nobody hears, and it
    already moves a *relative* L2 most of the way to "these are unrelated signals".

    The other half — that the spectral loss is correspondingly *un*bothered — is deliberately
    not asserted here. On a stack of nine harmonics 140 Hz apart the leakage between adjacent
    partials genuinely does interfere differently after a shift, so the magnitudes really do
    change; measured, 0.59 against an L2 of 0.72. The claim holds for speech and not for this
    signal, and a test that passed by luck on a toy would be worse than none.
    """
    x = vowel()
    l2 = float(torch.nn.functional.mse_loss(torch.roll(x, 8), x) / x.pow(2).mean())
    assert l2 > 0.5


def test_the_log_term_lives_on_mel_bands_not_fft_bins():
    """Regression. On linear bins the log term is mostly FFT numerical noise from bins with
    no energy in them: a one-sample circular shift — 60 microseconds, inaudible — scored
    0.66. A mel band sums dozens of bins and is never at the numerical floor."""
    x = vowel()
    log_only = ReconstructionLoss(wave_weight=0.0, convergence_weight=0.0)
    assert float(log_only(x, torch.roll(x, 1))[0]) < 0.1


def test_scales_shorter_than_the_clip_are_skipped_not_crashed():
    loss, parts = ReconstructionLoss()(vowel(300), vowel(300, f0=150))
    assert torch.isfinite(loss)
    assert "stft2048" not in parts and "stft128" in parts


def test_the_codebook_report_says_when_nothing_ran():
    assert codebook_report([], 1024)["used"] == 0


# ---------------------------------------------------------------------------------------
# the metrics
# ---------------------------------------------------------------------------------------


def test_spectral_convergence_divides_by_its_first_argument():
    """Regression. The reconstruction passed first divides by the wrong norm, and because a
    half-trained decoder is quieter than its input the mistake makes the number *worse* than
    the truth — the direction nobody double-checks."""
    cfg = MelConfig()
    target = magnitude(vowel(), cfg)
    quiet = target * 0.5
    assert spectral_convergence(target, quiet) == pytest.approx(0.5, abs=1e-4)
    assert spectral_convergence(quiet, target) == pytest.approx(1.0, abs=1e-4)


def test_mcd_is_zero_for_a_perfect_copy_and_ignores_gain():
    """MCD compares the *shape* of the spectrum. A reconstruction 6 dB quiet is not a
    reconstruction that says something different, which is why coefficient 0 is dropped."""
    x = vowel()
    assert mcd(x, x.clone()) == pytest.approx(0.0, abs=1e-4)
    assert mcd(x, x * 0.5) == pytest.approx(0.0, abs=0.05)


def test_mcd_ignores_inaudible_noise():
    """Regression for the absolute log floor. Noise at amplitude 1e-4 against a peak of 0.7
    is 77 dB down — below any recording's own floor — and once scored 86 dB here."""
    x = vowel()
    torch.manual_seed(0)
    assert mcd(x, x + 1e-4 * torch.randn_like(x)) < 1.0


def test_mcd_rises_with_distortion():
    x = vowel()
    torch.manual_seed(0)
    scores = [mcd(x, x + a * torch.randn_like(x)) for a in (1e-3, 1e-2, 1e-1)]
    assert scores == sorted(scores), scores


def test_the_cepstrum_keeps_thirteen_coefficients_starting_at_one():
    c = cepstrum(vowel())
    assert c.shape[0] == 13


def test_the_bitrate_ladder_reports_every_rung_it_can():
    m = Codec(TINY).eval()
    rows = bitrate_ladder(m, [vowel(TINY.hop * 30)], rungs=(1, 2, 4, 8))
    # 8 is past this codec's 4 codebooks and must be dropped, not faked.
    assert [r["codebooks"] for r in rows] == [1, 2, 4]
    assert all(r["kbps"] > 0 and r["compression"] > 1 for r in rows)
    assert rows[0]["kbps"] < rows[-1]["kbps"]


def test_reconstruct_preserves_length_and_shape():
    m = Codec(TINY).eval()
    x = vowel(7777)
    assert reconstruct(m, x).shape == x.shape
    assert reconstruct(m, x.unsqueeze(0).repeat(3, 1)).shape == (3, 7777)


def test_codebook_usage_reports_one_row_per_stage():
    m = Codec(TINY).eval()
    rows = codebook_usage(m, [vowel(TINY.hop * 20)])
    assert len(rows) == TINY.n_codebooks
    assert all(0 <= r["usage"] <= 1 for r in rows)
