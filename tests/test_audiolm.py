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


# ---------------------------------------------------------------------------------------
# training sees the layout generation uses
# ---------------------------------------------------------------------------------------


def _memorise(model, inputs_fn, steps=300, lr=3e-3):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    model.train()
    for _ in range(steps):
        loss = inputs_fn()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return float(loss)


def test_greedy_generation_reproduces_a_memorised_clip():
    """`generate` starts from a BOS column and predicts every column after it. Training on
    `delay(codes)` alone never made the first real column a target and never trained the BOS
    embedding, so a model that had memorised a clip perfectly still could not say its first
    frame. `training_pair` prepends the column; a memorised clip must then come back exactly."""
    from aksharallm.audio.lm import training_pair
    torch.manual_seed(0)
    model = AudioLM(tiny())
    codes = torch.randint(0, SIZE, (1, BOOKS, 8))

    def step():
        inputs, targets = training_pair(codes, PAD_ID, BOS_ID)
        return model(inputs, targets=targets)[1]

    assert _memorise(model, step) < 0.05
    out = generate(model, 8, top_k=1, generator=torch.Generator().manual_seed(0))
    assert torch.equal(out, codes)


def test_training_pair_masks_each_row_to_its_own_length():
    from aksharallm.audio.lm import training_pair
    codes = torch.randint(0, SIZE, (2, BOOKS, 6))
    codes[1, :, 4:] = PAD_ID                                 # row 1 is 4 frames, padded
    inputs, targets = training_pair(codes, PAD_ID, BOS_ID, n_frames=[6, 4])
    assert inputs.shape == targets.shape == (2, BOOKS, 6 + BOOKS)
    assert (inputs[:, :, 0] == BOS_ID).all() and (targets[:, :, 0] == -100).all()
    for i, t in enumerate([6, 4]):
        assert int((targets[i] != -100).sum()) == BOOKS * t   # every real code, once


def _two_utterances():
    tok = CharTokenizer.from_texts(["ab", "cab d"])
    texts = [tok.encode("ab"), tok.encode("cab d")]
    g = torch.Generator().manual_seed(1)
    clips = [torch.randint(0, SIZE, (BOOKS, 9), generator=g),
             torch.randint(0, SIZE, (BOOKS, 5), generator=g)]
    return tok, texts, clips


@pytest.mark.parametrize("text_first", [True, False])
def test_a_ragged_batch_scores_what_its_rows_score_alone(text_first):
    """A short sentence must not have pads between its text and its audio, which is what
    padding the two blocks separately does. Token-weighted, because the loss is a mean over
    every supervised cell in the batch."""
    from aksharallm.audio.speech import utterance_loss
    torch.manual_seed(0)
    tok, texts, clips = _two_utterances()
    model = AudioLM(tiny(text_vocab=tok.vocab_size, max_text=8)).eval()
    together = utterance_loss(model, clips, texts, text_first)
    n = [BOOKS * c.shape[-1] if text_first else len(t) - 1 for c, t in zip(clips, texts)]
    alone = sum(float(utterance_loss(model, [c], [t], text_first)) * k
                for c, t, k in zip(clips, texts, n)) / sum(n)
    assert float(together) == pytest.approx(alone, abs=1e-5)


def test_a_memorised_utterance_is_spoken_from_its_own_text_and_transcribed_back():
    """End to end through `speak` and `transcribe`, which build the layout their own way: if
    training laid the rows out differently, a perfectly memorised pair would not come back."""
    from aksharallm.audio.speech import utterance_loss
    torch.manual_seed(0)
    tok, texts, clips = _two_utterances()
    model = AudioLM(tiny(text_vocab=tok.vocab_size, max_text=8))
    _memorise(model, lambda: utterance_loss(model, clips, texts, True)
              + utterance_loss(model, clips, texts, False), steps=400)
    for clip, ids, text in zip(clips, texts, ["ab", "cab d"]):
        spoken = speak(model, ids, clip.shape[-1], top_k=1)
        assert torch.equal(spoken[0], clip)
        assert transcribe(model, clip.unsqueeze(0), tok) == text


# ---------------------------------------------------------------------------------------
# the trainer, end to end, as TTS
# ---------------------------------------------------------------------------------------


def _tts_corpus(tmp_path, transcripts):
    """A tiny codec checkpoint plus a codes directory with transcripts beside it."""
    import json as _json
    from dataclasses import asdict
    import numpy as np
    from aksharallm.audio.codec import Codec, CodecConfig
    from aksharallm.audio.dataset import Manifest

    ccfg = CodecConfig(channels=8, strides=(2, 4, 5), dim=16, n_codebooks=BOOKS,
                       codebook_size=SIZE)
    torch.manual_seed(0)
    torch.save({"codec": asdict(ccfg), "model": Codec(ccfg).state_dict(), "stage": "codec"},
               tmp_path / "codec.pt")
    codes_dir = tmp_path / "codes"
    codes_dir.mkdir()
    rng = np.random.default_rng(0)
    lens = [6, 9, 5, 8, 7, 6]
    names = [f"c{i}.wav" for i in range(len(lens))]
    with open(codes_dir / "codes.bin", "wb") as f:
        for t in lens:
            f.write(rng.integers(0, SIZE, (BOOKS, t)).astype("<i2").tobytes())
    Manifest(sample_rate=16000, offsets=[int(x) for x in np.cumsum([0] + lens)], names=names,
             sources=[{}] * len(lens), seconds=0.0, built="test").save(codes_dir / "manifest.json")
    (codes_dir / "transcripts.json").write_text(_json.dumps(dict(zip(names, transcripts))))
    return codes_dir


def test_the_trainer_runs_tts_end_to_end_and_a_resume_keeps_its_tokenizer(tmp_path):
    """Through `train_lm.main()`. The tokenizer is the text embedding's index (gotcha 3): a
    resume must use the one it trained with, even when the corpus has since gained a
    character, or every id after that character silently means a different letter."""
    import yaml
    from aksharallm.audio import train_lm

    codes_dir = _tts_corpus(tmp_path, ["ab", "ba ab", "a", "abba", "b a", "bab"])
    cfg = {"name": "tts-test",
           "audiolm": {"codebook_size": SIZE, "n_codebooks": BOOKS, "max_frames": 12,
                       "max_text": 10, "model": {"d_model": 32, "n_layers": 2, "n_heads": 4,
                                                 "n_kv_heads": 2, "max_seq_len": 32}},
           "data": {"codes": str(codes_dir), "codec": str(tmp_path / "codec.pt"),
                    "task": "tts", "val_clips": 2},
           "train": {"out_dir": str(tmp_path / "out"), "batch_size": 3, "max_steps": 3,
                     "eval_every": 2, "eval_batches": 1, "sample_every": 2, "ckpt_every": 0,
                     "log_every": 1}}
    (tmp_path / "tts.yaml").write_text(yaml.safe_dump(cfg))
    assert train_lm.main([str(tmp_path / "tts.yaml")]) == 0

    blob = torch.load(tmp_path / "out" / "ckpt_last.pt", weights_only=False)
    assert blob["task"] == "tts" and blob["tokenizer"]["alphabet"] == " ab"
    assert (tmp_path / "out" / "samples" / "step000002.wav").exists()
    assert (tmp_path / "out" / "samples" / "step000002.txt").read_text().strip() in {"b a", "bab"}

    # The corpus gains a "c". A rebuilt tokenizer would shift every id; the resume must not.
    (codes_dir / "transcripts.json").write_text(
        (codes_dir / "transcripts.json").read_text().replace('"a"', '"ca"'))
    cfg["train"]["max_steps"] = 4
    (tmp_path / "tts.yaml").write_text(yaml.safe_dump(cfg))
    assert train_lm.main([str(tmp_path / "tts.yaml")]) == 0
    blob = torch.load(tmp_path / "out" / "ckpt_last.pt", weights_only=False)
    assert blob["step"] == 3 and blob["tokenizer"]["alphabet"] == " ab"


def test_the_tts_ladder_judges_each_row_against_the_plain_words_of_the_text():
    """original / codec / tts, one judge, one reference: LJSpeech text with its punctuation
    stripped to what a recogniser writes. Each row's WER is its own; the TTS is spoken at
    the clip's real length."""
    from aksharallm.audio.speech import plain_words, tts_ladder
    assert plain_words("Mr. Smith's, 'dog' -- ran!") == "mr smith's 'dog' ran"

    tok = CharTokenizer.from_texts(["the cat.", "a dog!"])
    model = AudioLM(tiny(text_vocab=tok.vocab_size, max_text=12))
    clips = [torch.randint(0, SIZE, (BOOKS, 6)), torch.randint(0, SIZE, (BOOKS, 4))]
    utts = [(clips[0], tok.encode("the cat.")), (clips[1], tok.encode("a dog!"))]

    class Codec:
        def decode(self, codes):                 # one sample per frame, tagged by length
            return torch.zeros(1, codes.shape[-1])

    heard = {"original": ["the cat", "a dog"], "codec": ["the cat", "a fog"],
             "tts": ["bat", "a dog dog dog dog"]}
    calls = iter(heard)
    lengths = []
    rep = tts_ladder(model, tok, Codec(), utts, [torch.zeros(9).numpy()] * 2,
                     lambda ws: (lengths.append([len(w) for w in ws]), heard[next(calls)])[1])
    assert lengths[2] == [6, 4]                  # the TTS row was spoken at the real lengths
    assert rep["rows"]["original"]["wer"] == 0.0
    assert rep["rows"]["codec"]["wer"] == pytest.approx(0.25)     # 0/2 and 1/2 words
    assert rep["rows"]["tts"]["items"][0]["ref"] == "the cat"
    assert rep["rows"]["tts"]["runaway"] == 1    # 3 insertions over 2 words: WER 1.5


# ---------------------------------------------------------------------------------------
# knowing when to stop
# ---------------------------------------------------------------------------------------


def test_stop_targets_mark_where_codebook_0_has_ended():
    from aksharallm.audio.speech import stop_targets
    t = stop_targets([3, 1], n_codebooks=2, width=5)
    # row 0: T=3, columns 0..3 exist (T + N - 1 = 4); codebook 0 ends at column 3
    assert t[0].tolist() == [0.0, 0.0, 0.0, 1.0, -1.0]
    assert t[1].tolist() == [0.0, 1.0, -1.0, -1.0, -1.0]


def test_a_memorised_utterance_is_spoken_at_its_own_length_with_no_length_given():
    """The stop head is what lets `speak` be given text alone. Two memorised utterances of
    9 and 5 frames must each come back whole and stop where they stopped."""
    from aksharallm.audio.speech import utterance_loss
    torch.manual_seed(0)
    tok, texts, clips = _two_utterances()
    cfg = tiny(text_vocab=tok.vocab_size, max_text=8)
    cfg.predict_stop = True
    model = AudioLM(cfg)
    _memorise(model, lambda: utterance_loss(model, clips, texts, True), steps=400)
    for clip, ids in zip(clips, texts):
        spoken = speak(model, ids, top_k=1)
        assert spoken.shape[-1] == clip.shape[-1]
        assert torch.equal(spoken[0], clip)


def test_without_a_stop_head_speak_needs_a_length_or_a_ratio():
    model = AudioLM(tiny(text_vocab=20, max_text=8))
    with pytest.raises(ValueError, match="no stop head"):
        speak(model, [BOS, 5, 6, EOS])
    assert speak(model, [BOS, 5, 6, EOS], frames_per_char=3.0).shape[-1] == 6   # 2 chars x 3


def test_the_trainer_saves_a_stop_head_and_its_ratio_and_keeps_val_on_the_audio(tmp_path):
    """A run with `predict_stop` must save the head (or `speak` cannot stop) and the corpus's
    frames-per-character (the fallback). Its val number must be the audio alone, or the two
    LJSpeech runs could not be compared."""
    import json as _json
    import yaml
    from aksharallm.audio import train_lm
    from aksharallm.audio.dataset import UtteranceDataset
    from aksharallm.audio.speech import load_tts, utterance_loss

    codes_dir = _tts_corpus(tmp_path, ["ab", "ba ab", "a", "abba", "b a", "bab"])
    cfg = {"name": "stop-test",
           "audiolm": {"codebook_size": SIZE, "n_codebooks": BOOKS, "max_frames": 12,
                       "max_text": 10, "predict_stop": True,
                       "model": {"d_model": 32, "n_layers": 2, "n_heads": 4, "n_kv_heads": 2,
                                 "max_seq_len": 32}},
           "data": {"codes": str(codes_dir), "codec": str(tmp_path / "codec.pt"),
                    "task": "tts", "val_clips": 2},
           "train": {"out_dir": str(tmp_path / "out"), "batch_size": 3, "max_steps": 3,
                     "eval_every": 2, "eval_batches": 1, "sample_every": 0, "ckpt_every": 0,
                     "log_every": 1}}
    (tmp_path / "s.yaml").write_text(yaml.safe_dump(cfg))
    assert train_lm.main([str(tmp_path / "s.yaml")]) == 0

    model, tok, blob = load_tts(tmp_path / "out" / "ckpt_last.pt")
    assert model.stop_head is not None and blob["frames_per_char"] > 0
    val = [_json.loads(l) for l in (tmp_path / "out" / "train_log.jsonl").read_text()
           .splitlines() if '"val_loss"' in l][0]["val_loss"]
    best = torch.load(tmp_path / "out" / "ckpt_best.pt", weights_only=False)
    model.load_state_dict(best["model"])
    ds = UtteranceDataset(codes_dir, _json.loads((codes_dir / "transcripts.json").read_text()),
                          tok, n_codebooks=BOOKS, max_frames=12, max_text=10, split="val",
                          val_clips=2)
    import numpy as np
    ds.rng = np.random.default_rng(1337 + 1)          # the trainer's val seed
    clips, texts = ds.batch(3)
    audio_only = float(utterance_loss(model.eval(), clips, texts, True, with_stop=False))
    assert val == pytest.approx(audio_only, abs=1e-4)
