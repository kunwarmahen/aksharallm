"""The recogniser, pinned to things that are true rather than things that look right.

A CTC model that is subtly wrong still trains: the loss falls, the transcripts get closer, and
the WER is merely worse than it should be with nothing to say why. So every check here is
against something unarguable:

* the hand-written CTC equals `F.ctc_loss` — value **and** gradient — at LibriSpeech lengths,
  with mixed lengths, empty targets and impossible alignments in one batch (trap 3: log-space
  bugs only show up long);
* an utterance gives the same output alone and padded into a batch beside a longer one (trap
  1: padding must not reach a real frame — through normalisation, the convolution, or the
  STFT's own edge);
* corpus WER is total edits over total words, not a mean of per-utterance rates;
* a model trained on no-speech clips is told about them with an *empty* transcript, which CTC
  scores as all-blank and which must be finite.
"""

from __future__ import annotations

import json
import math
import struct
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from aksharallm.asr import vocab
from aksharallm.asr.ctc import ctc_loss, greedy_decode
from aksharallm.asr.data import Utterances, flac_info, speaker_of
from aksharallm.asr.measure import edits, no_speech_clips, score
from aksharallm.asr.model import AsrModelConfig, Recognizer, normalise, seconds_to_frames
from aksharallm.asr.noise import no_speech, replace_with_no_speech

TINY = dict(d_model=48, n_layers=2, n_heads=4, conv_kernel=7, subsample_channels=16, dropout=0.0)


# ---------------------------------------------------------------------------------------
# CTC
# ---------------------------------------------------------------------------------------


def _batch(T, B, C, L, seed=0):
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(T, B, C, dtype=torch.float64, generator=g, requires_grad=True)
    tl = torch.randint(1, L + 1, (B,), generator=g)
    tl[0] = L
    il = torch.randint(max(1, T // 2), T + 1, (B,), generator=g)
    il[0] = T
    tgt = torch.randint(1, C, (B, L), generator=g)
    return logits, tgt, il, tl


@pytest.mark.parametrize("T,B,C,L", [(12, 3, 5, 4), (400, 6, 29, 150)])
@pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
def test_ctc_matches_torch(T, B, C, L, reduction):
    logits, tgt, il, tl = _batch(T, B, C, L)
    ours = ctc_loss(logits.log_softmax(-1), tgt, il, tl, reduction=reduction)
    ref = F.ctc_loss(logits.log_softmax(-1), tgt, il, tl, reduction=reduction)
    torch.testing.assert_close(ours, ref, rtol=1e-9, atol=1e-9)


def test_ctc_gradient_matches_torch_at_librispeech_lengths():
    """The backward is alpha-beta, not autograd, so it needs its own check — at length."""
    logits, tgt, il, tl = _batch(500, 4, 29, 200, seed=3)
    g1, = torch.autograd.grad(ctc_loss(logits.log_softmax(-1), tgt, il, tl), logits)
    g2, = torch.autograd.grad(F.ctc_loss(logits.log_softmax(-1), tgt, il, tl), logits)
    torch.testing.assert_close(g1, g2, rtol=1e-6, atol=1e-9)


def test_ctc_gradcheck():
    logits, tgt, il, tl = _batch(7, 2, 4, 3, seed=5)
    il = torch.tensor([7, 6])  # feasible: an impossible sample has an infinite loss and no gradient
    assert torch.autograd.gradcheck(
        lambda x: ctc_loss(x.log_softmax(-1), tgt, il, tl, reduction="sum"), (logits,))


def test_ctc_empty_target_is_all_blank_and_finite():
    """A no-speech clip's transcript is empty. Its only path is blank on every frame."""
    lp = torch.randn(20, 1, 6, dtype=torch.float64).log_softmax(-1)
    loss = ctc_loss(lp, torch.zeros(1, 1, dtype=torch.long), torch.tensor([20]),
                    torch.tensor([0]), reduction="none")
    torch.testing.assert_close(loss, -lp[:, 0, 0].sum().reshape(1))


def test_ctc_impossible_alignment_is_inf_like_torch_and_has_zero_gradient():
    """'aaa' needs five frames (a _ a _ a). Three is impossible; the trainer drops and counts it."""
    x = torch.randn(3, 2, 5, dtype=torch.float64, requires_grad=True)
    tgt = torch.tensor([[1, 1, 1], [2, 3, 0]])
    tl, il = torch.tensor([3, 2]), torch.tensor([3, 3])
    ours = ctc_loss(x.log_softmax(-1), tgt, il, tl, reduction="none")
    ref = F.ctc_loss(x.log_softmax(-1), tgt, il, tl, reduction="none")
    assert torch.isinf(ours[0]) and torch.isinf(ref[0])
    torch.testing.assert_close(ours[1], ref[1])
    (g,) = torch.autograd.grad(ours[1] + 0 * ours[0].clamp(max=0), x)
    assert torch.isfinite(g).all()


def test_ctc_promotes_bf16_to_float32():
    lp = torch.randn(50, 2, 8).log_softmax(-1)
    tgt, il, tl = torch.randint(1, 8, (2, 10)), torch.tensor([50, 40]), torch.tensor([10, 8])
    a = ctc_loss(lp.bfloat16(), tgt, il, tl, reduction="none")
    assert a.dtype == torch.float32


def test_greedy_decode_merges_repeats_and_keeps_doubles_split_by_blank():
    # frames: a a _ a b b _   ->  "a", "a", "b": the blank is what keeps a double letter.
    seq = [1, 1, 0, 1, 2, 2, 0]
    lp = torch.full((1, len(seq), 3), -10.0)
    for t, k in enumerate(seq):
        lp[0, t, k] = 0.0
    assert greedy_decode(lp, torch.tensor([len(seq)])) == [[1, 1, 2]]


# ---------------------------------------------------------------------------------------
# the model and its padding
# ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["global", "utterance"])
def test_an_utterance_is_the_same_alone_and_padded_into_a_batch(mode):
    torch.manual_seed(0)
    m = Recognizer(AsrModelConfig(**TINY, normalise=mode)).eval()
    a, b = torch.randn(16_000) * 0.1, torch.randn(40_000) * 0.1
    alone, la = m(a[None], torch.tensor([16_000]))
    w = torch.zeros(2, 40_000)
    w[0, :16_000], w[1] = a, b
    both, lb = m(w, torch.tensor([16_000, 40_000]))
    assert int(la[0]) == int(lb[0])
    torch.testing.assert_close(both[0, : la[0]], alone[0], atol=1e-4, rtol=1e-4)


def test_normalise_ignores_padding():
    x = torch.randn(1, 4, 30)
    padded = torch.cat([x, torch.full((1, 4, 70), 123.0)], dim=-1)
    a = normalise(x, torch.tensor([30]))
    b = normalise(padded, torch.tensor([30]))
    torch.testing.assert_close(b[..., :30], a)
    assert (b[..., 30:] == 0).all()


def test_global_normalisation_keeps_loudness_visible():
    """The measured reason `global` is the default: per-utterance scaling makes faint hiss and
    loud hiss the same picture, so the model cannot tell nearly-silent from speech."""
    g = np.random.default_rng(0).standard_normal(16_000).astype(np.float32)
    quiet, loud = torch.from_numpy(g * 0.001)[None], torch.from_numpy(g * 0.3)[None]
    n = torch.tensor([16_000])
    per = Recognizer(AsrModelConfig(**TINY, normalise="utterance"))
    glob = Recognizer(AsrModelConfig(**TINY, normalise="global"))
    fq, _ = per.features(quiet, n)
    fl, _ = per.features(loud, n)
    # indistinguishable, bar the bands where the faint copy hits the log floor
    assert (fq - fl).abs().mean() < 0.05
    gq, _ = glob.features(quiet, n)
    gl, _ = glob.features(loud, n)
    assert (gl.mean() - gq.mean()) > 5.0           # ~100x apart in level, visibly


def test_feature_stats_are_saved_with_the_weights():
    m = Recognizer(AsrModelConfig(**TINY))
    m.set_feature_stats([torch.randn(8000) * 0.2])
    sd = m.state_dict()
    assert "feat_mean" in sd and "feat_std" in sd
    assert not torch.allclose(sd["feat_mean"], torch.zeros_like(sd["feat_mean"]))


def test_ctc_frame_rate_is_what_the_feasibility_filter_assumes():
    """The data filter and the encoder must agree on frames per second, or 'skipped' is not 0."""
    cfg = AsrModelConfig(**TINY)
    m = Recognizer(cfg).eval()
    for seconds in (0.6, 1.0, 3.7):
        n = int(seconds * 16_000)
        _, out = m(torch.zeros(1, n), torch.tensor([n]))
        assert int(out[0]) == seconds_to_frames(n / 16_000, cfg)


# ---------------------------------------------------------------------------------------
# text, data, measurement
# ---------------------------------------------------------------------------------------


def test_normalise_text():
    assert vocab.normalise("Well-known, Mr. O’Brien's “café”!") == "well known mr o'brien's cafe"
    assert vocab.normalise("'Quoted' words") == "quoted words"
    with pytest.raises(ValueError):
        vocab.encode("Café")


def test_corpus_wer_is_total_edits_over_total_words_not_a_mean_of_rates():
    short = ("s", "a b", "a x")                      # 1 error / 2 words = 50%
    long_ = ("s", " ".join("w" * 40), " ".join("w" * 39) + " x")  # 1 / 40 = 2.5%
    s = score([short, long_])
    assert s["wer"] == pytest.approx(2 / 42)         # not (0.5 + 0.025) / 2
    assert edits(list("kitten"), list("sitting")) == 3


def test_score_reports_the_worst_speaker():
    s = score([("good", "a b c d", "a b c d"), ("bad", "a b c d", "x y c d")])
    assert s["worst_speakers"][0]["speaker"] == "bad"
    assert s["speakers"]["bad"]["wer"] == 0.5


def test_speaker_of():
    assert speaker_of("103-1240-0000.flac") == "103"
    assert speaker_of("LJ001-0001.wav") == "LJ"
    assert speaker_of("synth-0003.wav") == "synth"


def _flac_header(sr, ch, bits, n):
    packed = (sr << 44) | ((ch - 1) << 41) | ((bits - 1) << 36) | n
    info = b"\x00" * 10 + struct.pack(">Q", packed) + b"\x00" * 16
    return b"fLaC" + bytes([0x80, 0, 0, 34]) + info


def test_flac_header_is_read_not_trusted_to_ffmpeg(tmp_path):
    p = tmp_path / "x.flac"
    p.write_bytes(_flac_header(44_100, 2, 24, 123_456))
    info = flac_info(p)
    assert (info.sample_rate, info.channels, info.bits, info.samples) == (44_100, 2, 24, 123_456)
    from aksharallm.asr.data import decode_flac
    with pytest.raises(ValueError, match="expected 16000 Hz mono"):
        decode_flac(p)


def _corpus(tmp_path, texts, seconds):
    from aksharallm.audio.dataset import Manifest
    sr = 16_000
    rng = np.random.default_rng(0)
    off, names = [0], []
    with open(tmp_path / "audio.bin", "wb") as f:
        for i, s in enumerate(seconds):
            x = (rng.standard_normal(int(s * sr)) * 3000).astype("<i2")
            f.write(x.tobytes())
            off.append(off[-1] + len(x))
            names.append(f"c{i}.wav")
    Manifest(sample_rate=sr, offsets=off, names=names, sources=[{}] * len(names),
             seconds=off[-1] / sr, built="").save(tmp_path / "manifest.json")
    (tmp_path / "transcripts.json").write_text(json.dumps(dict(zip(names, texts, strict=True))))
    return tmp_path


def test_utterances_count_every_drop_by_reason(tmp_path):
    from aksharallm.asr.train import feasibility
    d = _corpus(tmp_path, ["hello there", "!!!", "x" * 200, "ok", "fine"], [2.0, 2.0, 1.0, 0.1, 30.0])
    u = Utterances(d, max_seconds=20, feasible=feasibility(AsrModelConfig(**TINY), 16_000))
    assert [x.text for x in u.utts] == ["hello there"]
    assert u.dropped == {"empty transcript": 1, "more letters than frames": 1,
                         "shorter than 0.5s": 1, "longer than 20s": 1}


def test_buckets_respect_the_padded_budget(tmp_path):
    d = _corpus(tmp_path, ["a b"] * 12, [1, 1.5, 2, 2, 3, 3, 4, 5, 6, 7, 8, 9])
    u = Utterances(d)
    for b in u.buckets(max_batch_seconds=12):
        assert max(u.utts[i].n for i in b) * len(b) <= 12 * 16_000 or len(b) == 1


def test_no_speech_rows_have_empty_targets_and_keep_their_length():
    rng = np.random.default_rng(0)
    batch = {"wave": torch.randn(8, 4000), "n_samples": torch.tensor([4000, 3000] * 4),
             "targets": torch.ones(8, 5, dtype=torch.long), "target_lengths": torch.full((8,), 5)}
    k = replace_with_no_speech(batch, 0.5, rng)
    assert k > 0
    empty = (batch["target_lengths"] == 0).nonzero().flatten()
    assert len(empty) == k
    for r in empty:
        n = int(batch["n_samples"][r])
        assert (batch["wave"][r, n:] == 0).all()
        assert batch["wave"][r, :n].abs().max() < 0.2  # at most -15 dBFS


def test_training_noise_never_reproduces_a_check_clip():
    """The check would then measure memory. The families overlap; the clips must not."""
    rng = np.random.default_rng(1)
    check = [c.astype(np.float32) for c in no_speech_clips(seconds=1.0).values() if c.any()]
    for _ in range(50):
        x = no_speech(rng, 16_000)
        for c in check:
            assert not np.allclose(x, c, atol=1e-4)


# ---------------------------------------------------------------------------------------
# the trainer, end to end
# ---------------------------------------------------------------------------------------


def test_trainer_runs_resumes_and_publishes_the_shared_contract(tmp_path):
    from aksharallm.asr.train import load_recognizer, main
    (tmp_path / "c").mkdir()
    d = _corpus(tmp_path / "c", ["aa ee"] * 30, [1.0] * 30)
    cfg = tmp_path / "run.yaml"
    out = tmp_path / "out"
    cfg.write_text(f"""
name: t
asr: {json.dumps(TINY)}
data: {{train: [{d}], val_clips: 4, eval_utts: 4, max_batch_seconds: 8.0, no_speech_ratio: 0.2}}
optim: {{lr: 1.0e-3, warmup_steps: 2}}
train: {{out_dir: {out}, max_steps: 6, eval_every: 3, ckpt_every: 3, log_every: 1}}
""")
    assert main([str(cfg), "--device", "cpu", "-o", "train.stop_after=4"]) == 0
    import os
    # claimed by this process (released at interpreter exit, which a test never reaches)
    assert (out / "train.pid").read_text().strip() == str(os.getpid())
    recs = [json.loads(x) for x in (out / "train_log.jsonl").read_text().splitlines()]
    assert recs[0]["event"] == "session_start" and recs[0]["metric"] == "wer"
    assert recs[-1]["event"] == "session_end" and recs[-1]["last_step"] == 3
    assert any("val_wer" in r and "silence_chars" in r for r in recs)
    assert main([str(cfg), "--device", "cpu"]) == 0   # resumes at 4, finishes at 5
    recs = [json.loads(x) for x in (out / "train_log.jsonl").read_text().splitlines()]
    starts = [r for r in recs if r.get("event") == "session_start"]
    assert starts[-1]["start_step"] == 4 and recs[-1]["last_step"] == 5
    model, blob = load_recognizer(out / "ckpt_last.pt")
    assert blob["stage"] == "asr" and blob["best_metric"] == "wer"
    assert not torch.allclose(model.feat_mean, torch.zeros_like(model.feat_mean))


def test_a_text_checkpoint_is_refused(tmp_path):
    from aksharallm.asr.train import load_recognizer
    p = tmp_path / "lm.pt"
    torch.save({"model": {}, "stage": "sft"}, p)
    with pytest.raises(ValueError, match="not a recogniser"):
        load_recognizer(p)


def test_an_empty_no_speech_row_does_not_dominate_the_batch_loss():
    """Divided by a clamped length of 1, a 16 s silent row is ~400 frames x 3.4 nats in one
    term: measured as a step-0 loss of 621 on LibriSpeech, against ~5 for the speech rows."""
    from aksharallm.asr.train import compute_loss
    T, V = 400, 29
    lp = torch.randn(2, T, V).log_softmax(-1)
    speech = {"targets": torch.randint(1, V, (1, 150)), "target_lengths": torch.tensor([150])}
    alone, _ = compute_loss(lp[:1], torch.tensor([T]), speech, "torch")
    both = {"targets": torch.cat([speech["targets"], torch.zeros(1, 150, dtype=torch.long)]),
            "target_lengths": torch.tensor([150, 0])}
    mixed, _ = compute_loss(lp, torch.tensor([T, T]), both, "torch")
    assert float(mixed) < 3 * float(alone)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_a_gpu_run_resumes(tmp_path):
    """Every resume of the first LibriSpeech run died on start: the checkpoint is loaded with
    map_location='cuda', so the augment generator's state came back as a CUDA tensor and a CPU
    generator refused it. The CPU resume test above cannot see that, so this one exists."""
    from aksharallm.asr.train import main
    (tmp_path / "c").mkdir()
    d = _corpus(tmp_path / "c", ["aa ee"] * 30, [1.0] * 30)
    cfg = tmp_path / "run.yaml"
    out = tmp_path / "out"
    cfg.write_text(f"""
name: t
asr: {json.dumps(TINY)}
data: {{train: [{d}], val_clips: 4, eval_utts: 4, max_batch_seconds: 8.0}}
optim: {{lr: 1.0e-3, warmup_steps: 2}}
train: {{out_dir: {out}, max_steps: 4, eval_every: 0, ckpt_every: 0, log_every: 1}}
""")
    assert main([str(cfg), "--device", "cuda", "-o", "train.stop_after=2"]) == 0
    assert main([str(cfg), "--device", "cuda"]) == 0
    recs = [json.loads(x) for x in (out / "train_log.jsonl").read_text().splitlines()]
    assert [r["start_step"] for r in recs if r.get("event") == "session_start"] == [0, 2]


def test_the_default_layout_is_pre_norm_and_old_checkpoints_still_load(tmp_path):
    """The paper's per-block LayerNorm pinned the residual stream at |x| ~ 16 while sub-layers
    emitted time-constant vectors of norm 100-500: on LibriSpeech every utterance got the same
    transcript for 11,737 steps. Pre-norm is the default; the old layout loads only for the
    checkpoints trained with it."""
    from aksharallm.asr.model import Recognizer as R
    from aksharallm.asr.train import load_recognizer
    m = R(AsrModelConfig(**TINY))
    assert all(isinstance(b.norm, torch.nn.Identity) for b in m.blocks)
    assert isinstance(m.in_norm, torch.nn.LayerNorm) and isinstance(m.out_norm, torch.nn.LayerNorm)
    old_cfg = AsrModelConfig(**TINY, block_norm=True)
    old = R(old_cfg)
    shape = {k: v for k, v in vars(old_cfg).items() if k != "block_norm"}   # as saved before the fix
    p = tmp_path / "old.pt"
    torch.save({"model": old.state_dict(), "asr": shape, "alphabet": vocab.ALPHABET, "stage": "asr"}, p)
    loaded, _ = load_recognizer(p)
    assert loaded.cfg.block_norm and isinstance(loaded.blocks[0].norm, torch.nn.LayerNorm)


# ---------------------------------------------------------------------------------------
# the word language model and the beam search (piece 5)
# ---------------------------------------------------------------------------------------


def _lm(tmp_path, lines):
    from aksharallm.asr.ngram import TrigramLM
    p = tmp_path / "lm.txt"
    p.write_text("\n".join(lines) + "\n")
    return TrigramLM.build(p, vocab_size=1000, chunk_words=7, progress=None)


CORPUS = ["the cat sat on the mat", "the dog sat on the log", "a cat ate the fish",
          "the cat ate", "san francisco is a city", "the city of san francisco",
          "i ate turnips and carrots", "turnips and carrots and stew"]


def test_kneser_ney_is_a_probability_distribution_in_every_context(tmp_path):
    """A smoothing bug still yields a plausible-looking number; summing to one does not lie."""
    lm = _lm(tmp_path, CORPUS)
    from aksharallm.asr.ngram import BOS
    for ctx in [("the", "cat"), ("<s>", "<s>"), ("zzz", "the"), ("and", "carrots"), ("san", "zzz")]:
        u, v = (BOS if c == "<s>" else lm.id(c) for c in ctx)
        total = sum(math.exp(lm.logprob(u, v, w)) for w in range(lm.V) if w != BOS)
        assert total == pytest.approx(1.0, abs=1e-9), ctx


def test_chunked_counting_equals_one_pass(tmp_path):
    from aksharallm.asr.ngram import TrigramLM
    p = tmp_path / "lm.txt"
    p.write_text("\n".join(CORPUS) + "\n")
    a = TrigramLM.build(p, vocab_size=1000, chunk_words=3, progress=None)
    b = TrigramLM.build(p, vocab_size=1000, chunk_words=10**9, progress=None)
    assert np.array_equal(a.tri_keys, b.tri_keys) and np.array_equal(a.tri_counts, b.tri_counts)


def test_continuation_counts_are_what_kneser_ney_is_for(tmp_path):
    """"francisco" occurs as often as "cat" here, but only ever after "san": its unigram
    (continuation) estimate must be far smaller — the reason KN beats raw frequency."""
    lm = _lm(tmp_path, CORPUS + ["san francisco"] * 3)
    assert lm.uni_cont[lm.id("francisco")] == 1
    assert lm.uni_cont[lm.id("cat")] >= 2


def test_a_saved_lm_scores_identically(tmp_path):
    from aksharallm.asr.ngram import TrigramLM
    lm = _lm(tmp_path, CORPUS)
    lm.save(tmp_path / "x.npz")
    again = TrigramLM.load(tmp_path / "x.npz")
    s = "the cat ate turnips and carrots"
    assert again.sentence_logprob(s.split()) == lm.sentence_logprob(s.split())


def _frames(text: str, noise_letter: dict | None = None) -> np.ndarray:
    """Log-probs that spell `text` one letter per frame with a blank between, optionally with
    a competing letter at given positions — a hand-built encoder output."""
    rows = []
    for i, ch in enumerate(text):
        r = np.full(vocab.VOCAB_SIZE, -20.0)
        r[vocab.STOI[ch]] = math.log(0.6 if noise_letter and i in noise_letter else 0.98)
        if noise_letter and i in noise_letter:
            r[vocab.STOI[noise_letter[i]]] = math.log(0.38)
        rows.append(r)
        b = np.full(vocab.VOCAB_SIZE, -20.0)
        b[vocab.BLANK] = 0.0
        rows.append(b)
    return np.stack(rows)


def test_beam_search_without_an_lm_reads_what_the_frames_say():
    from aksharallm.asr.decode import BeamDecoder
    assert BeamDecoder(beam=8, beta=0.0).decode(_frames("the cat sat")) == "the cat sat"
    # a doubled letter needs its blank, and gets it from the frames
    assert BeamDecoder(beam=8, beta=0.0).decode(_frames("all")) == "all"


def test_the_lm_turns_a_phonetic_spelling_into_the_word(tmp_path):
    """The real error, built by hand: the frames prefer 'k' for the 'c' of carrots."""
    from aksharallm.asr.decode import BeamDecoder
    lm = _lm(tmp_path, CORPUS)
    lp = _frames("turnips and carrots", noise_letter={12: "k"})   # 'c' 0.6 vs 'k' 0.38...
    lp[24][vocab.STOI["c"]], lp[24][vocab.STOI["k"]] = math.log(0.38), math.log(0.6)  # ...flipped
    assert BeamDecoder(beam=8, beta=0.0).decode(lp) == "turnips and karrots"
    assert BeamDecoder(lm=lm, alpha=1.0, beta=0.5, beam=16).decode(lp) == "turnips and carrots"


def test_a_dictionary_word_wins_and_its_prefix_bonus_is_refunded_when_unused(tmp_path):
    """Day-two problem 4. 'shaun' is not in the LM, and its 'h' is heard weakly (the blank is
    likelier at that frame), so without help the LM writes a word it knows. In the dictionary,
    'shaun'. And a word that merely *starts* like a dictionary word keeps no bonus."""
    from aksharallm.asr.decode import BeamDecoder, _State
    # "met" must be a known word: two unknown words side by side are cheaper fused into one
    # ("metsaun", one <unk> penalty instead of two), which is what beta exists to resist.
    lm = _lm(tmp_path, CORPUS + ["i met him", "we met them"])
    lp = _frames("i met shaun")
    h = 2 * "i met shaun".index("h")
    lp[h][vocab.STOI["h"]], lp[h][vocab.BLANK] = math.log(0.4), math.log(0.58)
    plain = BeamDecoder(lm=lm, alpha=0.8, beta=0.5, beam=16)
    biased = BeamDecoder(lm=lm, alpha=0.8, beta=0.5, beam=16, dictionary={"Shaun"})
    # Without the dictionary the LM does what day-two problem 4 describes: the name becomes
    # the nearest word it knows ("san", from "san francisco").
    assert plain.decode(lp) == "i met san"
    assert biased.decode(lp) == "i met shaun"
    # Two ways to get a prefix bonus and not deserve it: wander off mid-word ("shat"), or END
    # on a dictionary prefix that is not itself a dictionary word ("sha"). Both leave nothing.
    for spelt in ("shat ", "sha "):
        st, st_plain = _State(), _State()
        for ch in spelt:
            st, st_plain = biased._extend(st, ch), plain._extend(st_plain, ch)
        assert st.extra == pytest.approx(st_plain.extra), spelt


def test_the_early_unk_penalty_is_charged_exactly_once(tmp_path):
    """Charged when the letters leave the vocabulary, not again when the word ends — and
    handed back if the word turns out to be in the dictionary."""
    from aksharallm.asr.decode import BeamDecoder, _State
    lm = _lm(tmp_path, CORPUS)
    early = BeamDecoder(lm=lm, alpha=1.0, beta=0.0, unk_penalty=-5.0, prefix_check=True)
    late = BeamDecoder(lm=lm, alpha=1.0, beta=0.0, unk_penalty=-5.0, prefix_check=False)
    for dec in (early, late):
        st = _State()
        for ch in "karrots ":
            st = dec._extend(st, ch)
        dec.total = st.extra
    assert early.total == pytest.approx(late.total)
    # A dictionary word owes no <unk> penalty: charging early and refunding must land exactly
    # where never charging lands.
    totals = []
    for check in (True, False):
        d = BeamDecoder(lm=lm, alpha=1.0, beta=0.0, unk_penalty=-5.0, dictionary={"karrots"},
                        prefix_check=check)
        st = _State()
        for ch in "karrots ":
            st = d._extend(st, ch)
        totals.append(st.extra)
    assert totals[0] == pytest.approx(totals[1])
    assert totals[0] > early.total


def test_a_dictionary_word_is_never_charged_while_it_is_being_spelt(tmp_path):
    """Charged-then-refunded is right on paper and fatal in a beam: the charged prefix falls
    out before the word ends. So no partial of a dictionary word may ever cost <unk>."""
    from aksharallm.asr.decode import BeamDecoder, _State
    lm = _lm(tmp_path, CORPUS)
    d = BeamDecoder(lm=lm, alpha=1.0, beta=0.0, unk_penalty=-20.0, dictionary={"ambrosch"})
    st, worst = _State(), 0.0
    for ch in "ambrosch":
        st = d._extend(st, ch)
        worst = min(worst, st.extra)
    assert worst >= 0.0


# ---------------------------------------------------------------------------------------
# the portal runs the same CLI
# ---------------------------------------------------------------------------------------


def _jobs(tmp_path):
    from aksharallm.portal.dictate import AsrJobs, Dictation
    d = Dictation(tmp_path, device_for=lambda: ("cpu", "test"))
    return d, AsrJobs(d)


def test_a_portal_job_refuses_anything_it_cannot_name(tmp_path):
    from aksharallm.portal.dictate import DictationError
    _, jobs = _jobs(tmp_path)
    for spec in [{"kind": "rm"}, {"kind": "fetch", "split": "../../etc"},
                 {"kind": "eval", "checkpoint": "x; rm -rf /"},
                 {"kind": "pack", "split": "dev-clean"},          # not downloaded
                 {"kind": "lm"}]:                                  # no LM text
        with pytest.raises(DictationError):
            jobs.command(spec)
    assert jobs.command({"kind": "fetch", "split": "dev-clean"})[0] == ["fetch", "dev-clean"]


@pytest.mark.parametrize("argv,state", [(["--help"], "done"), (["no-such-command"], "failed")])
def test_a_portal_job_is_judged_by_the_clis_exit_code(tmp_path, monkeypatch, argv, state):
    """Not by "an output file appeared" -- the check that called every Eval audit failed on
    success (gotcha 20). One exit code covers every kind of job."""
    import time as _t
    d, jobs = _jobs(tmp_path)
    monkeypatch.setattr(jobs, "command", lambda spec: (argv, "test"))
    cur = jobs.start({"kind": "fetch"})
    assert cur["command"].startswith("python -m aksharallm.asr ")
    # Up to 60 s: the job is a real `python -m aksharallm.asr` subprocess, and importing
    # torch under a loaded machine (the launch gate runs this suite) took longer than the
    # 10 s this used to allow -- a timing flake that could cancel a night's training run.
    for _ in range(600):
        st = jobs.status()
        if not st["running"]:
            break
        _t.sleep(0.1)
    assert st["current"]["state"] == state


def test_init_starts_from_another_runs_weights_and_keeps_its_feature_stats(tmp_path):
    """Adaptation (configs/asr-me.yaml): a fresh run whose weights AND feature statistics are
    the base's. Re-measuring the stats on the new corpus would rescale every input under
    weights that never saw that scaling."""
    from aksharallm.asr.train import load_recognizer, main
    (tmp_path / "c").mkdir()
    (tmp_path / "d").mkdir()
    d = _corpus(tmp_path / "c", ["aa ee"] * 30, [1.0] * 30)
    d2 = _corpus(tmp_path / "d", ["ee aa"] * 30, [1.0] * 30)
    # A much quieter second corpus: re-measured stats would differ, which is the point.
    a = np.fromfile(Path(d2) / "audio.bin", dtype="<i2")
    (a // 20).astype("<i2").tofile(Path(d2) / "audio.bin")

    def run(name, corpus, extra=""):
        cfg = tmp_path / f"{name}.yaml"
        cfg.write_text(f"""
name: {name}
asr: {json.dumps(TINY)}
data: {{train: [{corpus}], val_clips: 4, eval_utts: 4, max_batch_seconds: 8.0}}
optim: {{lr: 1.0e-3, warmup_steps: 2}}
train: {{out_dir: {tmp_path / name}, max_steps: 3, eval_every: 3, ckpt_every: 3, log_every: 1{extra}}}
""")
        assert main([str(cfg), "--device", "cpu"]) == 0
        return load_recognizer(tmp_path / name / "ckpt_last.pt")[0]

    base = run("base", d)
    adapted = run("adapted", d2, f", init: {tmp_path / 'base' / 'ckpt_last.pt'}, max_steps: 1")
    assert torch.equal(adapted.feat_mean, base.feat_mean)
    assert torch.equal(adapted.feat_std, base.feat_std)
    log = [json.loads(x) for x in (tmp_path / "adapted" / "train_log.jsonl").read_text().splitlines()]
    assert [r for r in log if r.get("event") == "session_start"][0]["start_step"] == 0
