"""The audio language model: the delay pattern, the eight heads, and TTS/ASR as one model.

The load-bearing test here is `undelay(delay(x)) == x`. Trap 4 of the audio phase is an
off-by-one in the delay pattern, and it belongs to gotcha #2's family — it trains perfectly
well and generates garbage, because the decoder is handed codebook 1 of frame 5 alongside
codebook 0 of frame 4 and reconstructs an interleaving of two different moments. Nothing
about the loss curve would say so.

Second is `ln(codebook_size)` at initialisation. A model that knows nothing scores exactly
that, so seeing it is the cheapest check that the delay pattern, the target masking and the
eight heads are all wired together correctly — the same argument as the DPO loop's `ln 2`.
"""

from __future__ import annotations

import math

import pytest
import torch

from aksharallm.audio.delay import delay, special_ids, undelay, valid_mask
from aksharallm.audio.lm import AudioLM, AudioLMConfig, generate, make_targets
from aksharallm.audio.speech import asr_batch, asr_report, pad_text, speak, transcribe, tts_batch
from aksharallm.audio.text import (
    BOS,
    EOS,
    CharTokenizer,
    character_error_rate,
    word_error_rate,
)
from aksharallm.config import ModelConfig

SIZE, BOOKS = 32, 4
PAD_ID, BOS_ID = special_ids(SIZE)


def tiny(text_vocab: int = 0, max_text: int = 0) -> AudioLMConfig:
    return AudioLMConfig(
        codebook_size=SIZE, n_codebooks=BOOKS, max_frames=12,
        max_text=max_text, text_vocab_size=text_vocab,
        model=ModelConfig(d_model=32, n_layers=2, n_heads=4, n_kv_heads=2, max_seq_len=64),
    )


# ---------------------------------------------------------------------------------------
# the delay pattern
# ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("n_books,n_frames", [(1, 5), (2, 5), (4, 10), (8, 3)])
def test_undelay_inverts_delay_exactly(n_books, n_frames):
    """Trap 4. Asserted for a codebook count LARGER than the frame count too, because that
    is where the triangle overlaps itself and an off-by-one stops being obvious."""
    codes = torch.randint(0, SIZE, (2, n_books, n_frames))
    assert torch.equal(undelay(delay(codes, PAD_ID)), codes)


def test_delay_shifts_each_codebook_by_its_index():
    codes = torch.arange(12).reshape(1, 3, 4)
    d = delay(codes, PAD_ID)
    assert d.shape == (1, 3, 6)
    for k in range(3):
        assert torch.equal(d[0, k, k : k + 4], codes[0, k])
        assert (d[0, k, :k] == PAD_ID).all()
        assert (d[0, k, k + 4 :] == PAD_ID).all()


def test_the_pad_id_is_not_a_real_code():
    """Zero is a perfectly good code the codec emits constantly, so filling the triangle
    with zeros would train the model to predict silence-shaped nonsense at the edges."""
    pad, bos = special_ids(SIZE)
    assert pad >= SIZE and bos > pad


def test_the_sequence_grows_by_n_minus_one_not_by_a_factor_of_n():
    """The entire argument for the pattern over flattening."""
    codes = torch.zeros(1, 8, 500, dtype=torch.long)
    assert delay(codes, PAD_ID).shape[-1] == 507  # not 4,000


def test_valid_mask_marks_exactly_the_real_cells():
    mask = valid_mask(3, 4)
    assert mask.shape == (3, 6)
    assert mask.sum(dim=1).tolist() == [4, 4, 4]
    assert not mask[2, 0] and mask[2, 2]


def test_targets_ignore_the_padding_triangle():
    """Training on the padding teaches the model to predict a token carrying no information
    at exactly the two places where it has the least context to do it from."""
    codes = torch.randint(0, SIZE, (1, BOOKS, 6))
    tgt = make_targets(delay(codes, PAD_ID), 6, PAD_ID)
    assert (tgt == -100).sum() == BOOKS * (BOOKS - 1)  # two triangles of (N-1)N/2 each
    assert tgt.max() < SIZE


# ---------------------------------------------------------------------------------------
# the model
# ---------------------------------------------------------------------------------------


def test_an_untrained_model_scores_exactly_uniform():
    """`ln(codebook_size)`. The cheapest possible check that everything is wired together."""
    torch.manual_seed(0)
    model = AudioLM(tiny())
    codes = torch.randint(0, SIZE, (4, BOOKS, 12))
    d = delay(codes, PAD_ID)
    _, loss = model(d, targets=make_targets(d, 12, PAD_ID))
    assert float(loss.detach()) == pytest.approx(math.log(SIZE), abs=0.1)


def test_the_logits_have_a_codebook_axis():
    """One head per codebook, not one head over N x V. The eight vocabularies are unrelated,
    and a shared head could put mass on a book-3 code while predicting book 0."""
    model = AudioLM(tiny())
    codes = torch.randint(0, SIZE, (2, BOOKS, 12))
    logits, _ = model(delay(codes, PAD_ID))
    assert logits.shape == (2, 12 + BOOKS - 1, BOOKS, SIZE)


def test_the_embedding_tables_are_separate():
    """Summing works because code 5 of book 0 and code 5 of book 3 are different vectors.
    Sharing one table would make the sum a collision rather than a composition."""
    model = AudioLM(tiny())
    assert not torch.equal(model.embeds[0].weight, model.embeds[1].weight)


def test_a_context_that_cannot_hold_the_sequence_is_refused():
    cfg = tiny()
    cfg.max_frames = 200  # + 3 of delay = 203 > max_seq_len 64
    with pytest.raises(ValueError, match="cannot hold"):
        AudioLM(cfg)


def test_generation_never_emits_a_special_token():
    """The delay triangle is PLACED, not predicted — the heads emit real codes only, and the
    loop writes `pad_id` where the pattern says it belongs."""
    torch.manual_seed(0)
    model = AudioLM(tiny())
    out = generate(model, 10, temperature=1.0, top_k=0)
    assert out.shape == (1, BOOKS, 10)
    assert int(out.max()) < SIZE, "sampled a [PAD] or [BOS] as if it were audio"


def test_generation_is_reproducible():
    torch.manual_seed(0)
    model = AudioLM(tiny())
    g1 = torch.Generator().manual_seed(7)
    g2 = torch.Generator().manual_seed(7)
    assert torch.equal(generate(model, 6, generator=g1), generate(model, 6, generator=g2))


# ---------------------------------------------------------------------------------------
# the seam in Transformer
# ---------------------------------------------------------------------------------------


def test_supplying_embeddings_matches_supplying_ids():
    """The audio path must be an exact no-op on the text path. If these ever diverge, every
    language model in the repo is running through a subtly different forward pass."""
    from aksharallm.model.transformer import Transformer

    torch.manual_seed(0)
    body = Transformer(ModelConfig(vocab_size=50, d_model=32, n_layers=2, n_heads=4,
                                   max_seq_len=32)).eval()
    idx = torch.randint(0, 50, (2, 8))
    with torch.no_grad():
        by_ids, _ = body(idx, full_logits=True)
        hidden, _ = body(None, inputs_embeds=body.tok_emb(idx), return_hidden=True)
    assert torch.allclose(by_ids, body.lm_head(hidden), atol=1e-5)


def test_passing_both_ids_and_embeddings_is_refused():
    from aksharallm.model.transformer import Transformer

    body = Transformer(ModelConfig(vocab_size=50, d_model=32, n_layers=2, n_heads=4,
                                   max_seq_len=32))
    with pytest.raises(ValueError, match="either idx or inputs_embeds"):
        body(torch.zeros(1, 4, dtype=torch.long), inputs_embeds=torch.zeros(1, 4, 32))


# ---------------------------------------------------------------------------------------
# text
# ---------------------------------------------------------------------------------------


def test_the_alphabet_is_sorted_so_ids_are_stable():
    """Built from a corpus, so it must not depend on the order the corpus was walked in —
    otherwise no two machines agree what id 7 means."""
    a = CharTokenizer.from_texts(["cab", "bad"])
    b = CharTokenizer.from_texts(["dab", "cab"])
    assert a.alphabet == b.alphabet == "abcd"


def test_encode_decode_round_trips():
    tok = CharTokenizer.from_texts(["hello world"])
    ids = tok.encode("hello")
    assert ids[0] == BOS and ids[-1] == EOS
    assert tok.decode(ids[1:-1]) == "hello"


def test_unknown_characters_are_dropped_not_mapped_to_one_id():
    """A single id standing for "some symbol" is a sound the model would have to invent."""
    tok = CharTokenizer.from_texts(["abc"])
    assert tok.decode(tok.encode("axbxc", bos=False, eos=False)) == "abc"


def test_word_error_rate_counts_the_three_edits():
    assert word_error_rate("a b c", "a b c") == 0.0
    assert word_error_rate("a b c", "a x c") == pytest.approx(1 / 3)  # substitution
    assert word_error_rate("a b c", "a b") == pytest.approx(1 / 3)  # deletion
    assert word_error_rate("a b c", "a b c d") == pytest.approx(1 / 3)  # insertion


def test_word_error_rate_can_exceed_one():
    """A model that will not stop produces a WER over 1. Clamping it would hide exactly that
    failure, which is a different problem from getting words wrong."""
    assert word_error_rate("a", "a b c d") > 1.0


def test_character_error_rate_is_gentler_than_word_error_rate():
    """A character-level model that misses one letter loses a whole word to WER, so quoting
    only WER overstates how bad it is by roughly the word length."""
    assert character_error_rate("hello", "hallo") < word_error_rate("hello", "hallo")


def test_the_asr_report_separates_runaway_from_wrong():
    report = asr_report([("a b c", "a b c"), ("a", "a b c d e")])
    assert report["n"] == 2 and report["runaway"] == 1


# ---------------------------------------------------------------------------------------
# TTS and ASR
# ---------------------------------------------------------------------------------------


def test_padding_text_marks_the_pad_as_ignored():
    ids, tgt = pad_text([[1, 2, 3], [1, 2]])
    assert ids.shape == (2, 3)
    assert tgt[1, 2] == -100 and tgt[1, 1] == 2


def test_tts_and_asr_are_one_model_one_flag_apart():
    """The whole of part 4: put two things in one sequence and take the loss on one of them.
    Both directions must run through the same module without a second forward pass."""
    torch.manual_seed(0)
    model = AudioLM(tiny(text_vocab=20, max_text=8))
    codes = torch.randint(0, SIZE, (2, BOOKS, 10))
    texts = [[BOS, 5, 6, EOS], [BOS, 7, EOS]]

    _, tts_loss = tts_batch(model, codes, texts)
    _, asr_loss = asr_batch(model, codes, texts)
    assert torch.isfinite(tts_loss) and torch.isfinite(asr_loss)
    # TTS supervises audio (ln 32); ASR supervises text (ln 20). Different vocabularies,
    # so the two losses start at different values — which is the check that the loss is
    # really being taken on the side it claims.
    assert float(tts_loss.detach()) == pytest.approx(math.log(SIZE), abs=0.3)
    assert float(asr_loss.detach()) == pytest.approx(math.log(20), abs=0.5)


def test_a_model_without_a_text_path_refuses_text():
    model = AudioLM(tiny())
    with pytest.raises(ValueError, match="no text path"):
        model.embed_text(torch.zeros(1, 2, dtype=torch.long))


def test_speak_returns_undelayed_codes():
    torch.manual_seed(0)
    model = AudioLM(tiny(text_vocab=20, max_text=8))
    out = speak(model, [BOS, 5, EOS], 8, temperature=1.0, top_k=0)
    assert out.shape == (1, BOOKS, 8)
    assert int(out.max()) < SIZE


def test_transcribe_stops_at_eos_and_returns_a_string():
    torch.manual_seed(0)
    model = AudioLM(tiny(text_vocab=20, max_text=8))
    tok = CharTokenizer("abcdefghij" + "klmnopq")
    codes = torch.randint(0, SIZE, (1, BOOKS, 8))
    out = transcribe(model, codes, tok, max_chars=12)
    assert isinstance(out, str) and len(out) <= 12
