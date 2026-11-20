"""Multi-speaker TTS (docs/21 § "Many voices, one TTS"): one codes directory over several
corpora (`encode_many`), a reference clip's speaker embedding in front of the text, and the
voice it asks for being the voice that comes out."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from aksharallm.audio.delay import special_ids
from aksharallm.audio.lm import AudioLM, AudioLMConfig
from aksharallm.audio.speech import speak, utterance_loss
from aksharallm.audio.text import CharTokenizer
from aksharallm.config import ModelConfig

SIZE, BOOKS, DIM = 32, 4, 6
PAD_ID, BOS_ID = special_ids(SIZE)


def tiny(text_vocab: int, speaker_dim: int = DIM) -> AudioLMConfig:
    return AudioLMConfig(
        codebook_size=SIZE, n_codebooks=BOOKS, max_frames=12, max_text=8,
        text_vocab_size=text_vocab, speaker_dim=speaker_dim,
        model=ModelConfig(d_model=32, n_layers=2, n_heads=4, n_kv_heads=2, max_seq_len=64),
    )


def _voices():
    """One sentence said by two speakers: same text, different codes, different voices."""
    tok = CharTokenizer.from_texts(["ab"])
    ids = tok.encode("ab")
    g = torch.Generator().manual_seed(3)
    clips = [torch.randint(0, SIZE, (BOOKS, 7), generator=g),
             torch.randint(0, SIZE, (BOOKS, 7), generator=g)]
    voices = torch.nn.functional.normalize(torch.randn(2, DIM, generator=g), dim=-1)
    return tok, ids, clips, voices


def test_the_same_text_comes_out_in_the_voice_it_is_asked_for():
    """The only thing telling the two rows apart is the speaker position. If training or
    `speak` dropped it (or put it somewhere the other does not), both voices would come back
    as one average and neither clip would be reproduced."""
    torch.manual_seed(0)
    tok, ids, clips, voices = _voices()
    model = AudioLM(tiny(tok.vocab_size))
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.0)
    for _ in range(400):
        loss = utterance_loss(model, clips, [ids, ids], True, speaker=voices)
        opt.zero_grad()
        loss.backward()
        opt.step()
    for clip, v in zip(clips, voices):
        assert torch.equal(speak(model, ids, clip.shape[-1], top_k=1, speaker=v)[0], clip)


def test_a_ragged_multispeaker_batch_scores_what_its_rows_score_alone():
    torch.manual_seed(0)
    tok = CharTokenizer.from_texts(["ab", "cab d"])
    texts = [tok.encode("ab"), tok.encode("cab d")]
    g = torch.Generator().manual_seed(1)
    clips = [torch.randint(0, SIZE, (BOOKS, 9), generator=g),
             torch.randint(0, SIZE, (BOOKS, 5), generator=g)]
    voices = torch.randn(2, DIM, generator=g)
    model = AudioLM(tiny(tok.vocab_size)).eval()
    together = float(utterance_loss(model, clips, texts, True, speaker=voices))
    n = [BOOKS * c.shape[-1] for c in clips]
    alone = sum(float(utterance_loss(model, [c], [t], True, speaker=v[None])) * k
                for c, t, v, k in zip(clips, texts, voices, n)) / sum(n)
    assert together == pytest.approx(alone, abs=1e-5)


def test_training_lays_a_row_out_exactly_as_speak_does():
    """Score one clip by hand with `speak`'s layout, `[speaker][text][BOS + delay(audio)]`,
    and require the training loss to equal it. A memorised model can find the speaker vector
    wherever it sits, so only a direct comparison pins the position."""
    import torch.nn.functional as F

    from aksharallm.audio.delay import delay, valid_mask
    torch.manual_seed(0)
    tok, ids, clips, voices = _voices()
    model = AudioLM(tiny(tok.vocab_size)).eval()
    clip, v = clips[0], voices[0]
    text_x = torch.cat([model.embed_speaker(v[None]), model.embed_text(torch.tensor([ids]))], 1)
    d = torch.cat([torch.full((1, BOOKS, 1), BOS_ID), delay(clip[None], PAD_ID)], dim=-1)
    hidden, _ = model.body(None, inputs_embeds=torch.cat([text_x, model.embed_audio(d)], 1),
                           return_hidden=True)
    h = hidden[:, text_x.shape[1]:]
    logits = torch.stack([head(h) for head in model.heads], dim=2)[:, :-1]   # (1, S-1, N, V)
    gold = torch.where(valid_mask(BOOKS, clip.shape[-1]), d[0, :, 1:], torch.full_like(d[0, :, 1:], -100))
    by_hand = F.cross_entropy(logits.reshape(-1, SIZE), gold.T.reshape(-1), ignore_index=-100)
    got = utterance_loss(model, [clip], [ids], True, with_stop=False, speaker=v[None])
    assert float(got) == pytest.approx(float(by_hand), abs=1e-5)


def test_a_multispeaker_model_refuses_to_train_or_speak_without_a_voice():
    tok, ids, clips, _ = _voices()
    model = AudioLM(tiny(tok.vocab_size))
    with pytest.raises(ValueError, match="speaker"):
        utterance_loss(model, clips, [ids, ids], True)
    with pytest.raises(ValueError, match="speaker"):
        speak(model, ids, 3)


def test_the_context_check_counts_the_speaker_position():
    cfg = tiny(10)
    cfg.model.max_seq_len = cfg.audio_positions + cfg.max_text   # one short
    with pytest.raises(ValueError, match="1 speaker"):
        AudioLM(cfg)


# ---------------------------------------------------------------------------------------
# encode_many + the dataset's references
# ---------------------------------------------------------------------------------------


def _corpus(d, names, lens, transcripts, rng):
    from aksharallm.audio.dataset import Manifest
    d.mkdir(parents=True)
    off = [0]
    with open(d / "audio.bin", "wb") as f:
        for n in lens:
            f.write((rng.standard_normal(n) * 3000).astype("<i2").tobytes())
            off.append(off[-1] + n)
    Manifest(sample_rate=16000, offsets=off, names=names, sources=[{}] * len(names),
             seconds=off[-1] / 16000, built="test").save(d / "manifest.json")
    (d / "transcripts.json").write_text(json.dumps(dict(zip(names, transcripts))))


@pytest.fixture
def two_corpora(tmp_path):
    from aksharallm.audio.codec import Codec, CodecConfig
    from aksharallm.audio.speaker import SpeakerEncoder, SpeakerModelConfig
    from aksharallm.asr.model import AsrModelConfig

    rng = np.random.default_rng(0)
    lj = [f"LJ001-{i:04d}.wav" for i in range(5)]
    ls = [f"{s}-10-{i:04d}.flac" for s in (11, 22) for i in range(4)]
    _corpus(tmp_path / "lj", lj, rng.integers(1600, 4000, 5).tolist(), [f"lj {i}" for i in range(5)], rng)
    _corpus(tmp_path / "ls", ls, rng.integers(1600, 4000, 8).tolist(), [f"ls {i}" for i in range(8)], rng)
    torch.manual_seed(0)
    codec = Codec(CodecConfig(channels=8, strides=(2, 4, 5), dim=16, n_codebooks=BOOKS,
                              codebook_size=SIZE)).eval()
    spk = SpeakerEncoder(SpeakerModelConfig(encoder=AsrModelConfig(
        d_model=16, n_layers=1, n_heads=2, conv_kernel=3, subsample_channels=8,
        max_frames=256, normalise="utterance"), emb_dim=DIM)).eval()
    return tmp_path, codec, spk, lj + ls


def test_encode_many_writes_one_shuffled_directory_whose_files_agree(two_corpora):
    from aksharallm.audio.dataset import CodeDataset, Manifest, encode_many
    root, codec, spk, names = two_corpora
    out = root / "codes"
    encode_many(codec, [root / "lj", root / "ls"], out, speaker_model=spk, seed=0,
                max_batch=3, progress=None)
    man = Manifest.load(out / "manifest.json")
    assert sorted(man.names) == sorted(names) and man.names != names     # all, shuffled
    who = json.loads((out / "speakers.json").read_text())
    assert {who[n] for n in names} == {"LJ", "11", "22"}
    assert set(json.loads((out / "transcripts.json").read_text())) == set(names)
    assert np.load(out / "speaker_emb.npy").shape == (len(names), DIM)
    # every clip's codes are the frames encoding it ALONE would give, bar its edge frames
    ds = CodeDataset(out, 1, n_codebooks=BOOKS, val_clips=1)
    hop = codec.cfg.hop
    src = {n: (root / ("lj" if n.startswith("LJ") else "ls")) for n in names}
    for i, n in enumerate(man.names):
        m = Manifest.load(src[n] / "manifest.json")
        j = m.names.index(n)
        a, b = m.offsets[j], m.offsets[j + 1]
        wave = np.fromfile(src[n] / "audio.bin", dtype="<i2")[a:b].astype(np.float32) / 32768
        alone = codec.encode(torch.from_numpy(wave)[None])[0]
        got = ds.clip(i)
        assert got.shape == alone.shape == (BOOKS, -(-(b - a) // hop))
        assert torch.equal(got[:, :-4], alone[:, :-4])


def test_a_reference_is_another_training_clip_by_the_same_speaker(two_corpora):
    from aksharallm.audio.dataset import UtteranceDataset, encode_many
    root, codec, spk, names = two_corpora
    out = root / "codes"
    encode_many(codec, [root / "lj", root / "ls"], out, speaker_model=spk, seed=0, progress=None)
    tok = CharTokenizer.from_texts([f"lj {i}" for i in range(5)] + [f"ls {i}" for i in range(8)])
    tr = json.loads((out / "transcripts.json").read_text())
    ds = UtteranceDataset(out, tr, tok, n_codebooks=BOOKS, max_frames=999, max_text=99,
                          val_clips=1, speakers=True, seed=0)
    emb = np.load(out / "speaker_emb.npy").astype(np.float32)
    rng = np.random.default_rng(0)
    for j in range(len(ds)):
        i = ds.items[j][0]
        for _ in range(5):
            r = ds.reference(j, rng).numpy()
            hit = [k for k in range(len(emb)) if np.allclose(emb[k], r)]
            assert hit and all(ds.speaker[k] == ds.speaker[i] for k in hit)
            assert i not in hit                     # never the clip itself
    clips, texts = ds.batch(4)
    assert ds.speaker_batch().shape == (4, DIM)


def test_a_judged_codebook_list_gets_its_own_report_name():
    from types import SimpleNamespace

    from aksharallm.audio.__main__ import transfer_suffix
    base = dict(home_clips=64, per_speaker=3, home="data/audio/lj", other="data/asr/test-clean")
    assert transfer_suffix(SimpleNamespace(**base, judged="4,8")) == ""
    assert transfer_suffix(SimpleNamespace(**base, judged="1,4,8")) == "-judged1-4-8"
