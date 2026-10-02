"""Dictation: the punctuation tagger, cleanup, learning from corrections, the day-two suite.

The property the whole layer is built around is that **nothing appears in your text that you
did not say**. Several tests below are that property from different sides: `render` cannot
change a word, cleanup removes only what it lists, and the day-two `cleanup_check` counts it.
"""

from __future__ import annotations

import math
import random

import numpy as np
import pytest
import torch

from aksharallm.asr import vocab
from aksharallm.dictate import punct
from aksharallm.dictate.cleanup import clean
from aksharallm.dictate.personal import Personal
from aksharallm.dictate.punct import (CAP, COMMA, LOWER, NONE, PERIOD, QUESTION, UPPER, label,
                                      render, split_label, words_of)

# ---------------------------------------------------------------------------------------
# text -> training pairs
# ---------------------------------------------------------------------------------------


def _pairs(text, drop_first=False):
    return [(w.text, split_label(w.label)) for w in punct.examples_from_text(text, drop_first)]


def test_text_becomes_the_recognisers_words_and_the_labels_to_restore():
    got = _pairs("Is that alright? I think so, NASA said.")
    assert [w for w, _ in got] == ["is", "that", "alright", "i", "think", "so", "nasa", "said"]
    assert got[0][1] == (NONE, CAP)
    assert got[2][1] == (QUESTION, LOWER)
    assert got[3][1] == (NONE, CAP)        # "I"
    assert got[5][1] == (COMMA, LOWER)
    assert got[6][1] == (NONE, UPPER)
    assert got[7][1] == (PERIOD, LOWER)


def test_a_sentence_the_ear_could_not_have_heard_is_dropped_whole_not_cleaned():
    """Deleting "1984" from "In 1984, he left." would teach a comma after "in"."""
    got = _pairs("In 1984, he left. She stayed.")
    assert [w for w, _ in got] == ["she", "stayed"]


def test_headings_and_the_window_s_first_sentence_are_not_training_data():
    text = "the middle of a sentence. Real one.\nA Heading Without A Stop\nSecond real one!"
    got = _pairs(text, drop_first=True)
    assert [w for w, _ in got] == ["real", "one", "second", "real", "one"]
    assert got[-1][1] == (PERIOD, LOWER)   # "!" is taught as "."


def test_dashes_become_commas_and_hyphens_split_words():
    got = _pairs("It was well-known—everyone agreed.")
    assert [w for w, _ in got] == ["it", "was", "well", "known", "everyone", "agreed"]
    assert got[3][1][0] == COMMA


# ---------------------------------------------------------------------------------------
# labels -> text: decoration only
# ---------------------------------------------------------------------------------------


def test_render_never_changes_a_word():
    rng = random.Random(0)
    vocab_words = ["the", "i", "i'm", "shaun", "nasa", "is", "it", "ok", "a"]
    for _ in range(300):
        words = [rng.choice(vocab_words) for _ in range(rng.randint(1, 12))]
        labels = [rng.randrange(punct.N_LABELS) for _ in words]
        assert words_of(render(words, labels)) == words


def test_render_rules_sit_on_top_of_the_model():
    lab = label(NONE, LOWER)
    assert render(["so", "i", "think"], [lab] * 3) == "So I think."
    assert render(["hello", "there"], [label(PERIOD, LOWER), lab]) == "Hello. There."
    assert render(["met", "shaun"], [lab, lab], spellings={"shaun": "Shaun"}) == "Met Shaun."


def test_a_personal_spelling_still_starts_a_sentence_with_a_capital():
    assert render(["iphone", "works"], [label(NONE, LOWER)] * 2,
                  spellings={"iphone": "iPhone"}) == "IPhone works."


# ---------------------------------------------------------------------------------------
# the tagger model
# ---------------------------------------------------------------------------------------


class _Tok:
    """A word-level stand-in for the BPE tokenizer: two pieces for long words, so the
    word-end bookkeeping is exercised."""

    eos_id = 0
    pad_id = 1
    vocab_size = 64

    def encode(self, text):
        w = text.strip()
        h = 2 + (sum(map(ord, w)) % 60)
        return [h, 2 + (h + 7) % 60] if len(w) > 4 else [h]


def _tiny_tagger(max_seq_len=32):
    from aksharallm.config import ModelConfig
    from aksharallm.model.transformer import Transformer
    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=_Tok.vocab_size + punct.N_LABELS, tag_classes=punct.N_LABELS,
                      causal=False, d_model=32, n_layers=2, n_heads=2, n_kv_heads=2,
                      max_seq_len=max_seq_len)
    return Transformer(cfg).eval()


def test_a_tagger_is_bidirectional_and_never_a_diffusion_model():
    from aksharallm.config import ModelConfig
    with pytest.raises(ValueError, match="causal: false"):
        ModelConfig(vocab_size=76, tag_classes=12, causal=True)
    with pytest.raises(ValueError, match="not both"):
        ModelConfig(vocab_size=76, tag_classes=12, causal=False, mask_token_id=75)
    c = ModelConfig(vocab_size=76, tag_classes=12, causal=False)
    assert c.is_tagger and not c.is_diffusion and c.tag_base == 64


def test_the_word_encoder_marks_the_last_piece_of_every_word():
    from aksharallm.dictate.tagger import WordEncoder
    ids, ends = WordEncoder(_Tok()).encode(["hello", "a", "carrots"])
    assert len(ids) == 5 and ends == [1, 2, 4]


def test_tag_logits_come_from_the_label_rows_of_the_embedding():
    from aksharallm.dictate.tagger import tag_logits
    m = _tiny_tagger()
    x = torch.randint(2, 60, (2, 7))
    got = tag_logits(m, x)
    full, _ = m(x, full_logits=True)
    assert got.shape == (2, 7, punct.N_LABELS)
    assert torch.allclose(got, full[..., m.cfg.tag_base:], atol=1e-5)


def test_a_long_dictation_is_windowed_and_every_word_gets_a_label():
    from aksharallm.dictate.tagger import Punctuator
    p = Punctuator(_tiny_tagger(max_seq_len=16), _Tok())
    words = ["word" + str(i % 7) for i in range(50)]
    assert len(p.labels(words)) == 50
    short = words[:6]
    direct = Punctuator(_tiny_tagger(max_seq_len=64), _Tok()).labels(short)
    assert p.labels(short) == direct   # same weights (seeded), fits one window either way


def test_the_objective_labels_word_ends_only_and_trains(tmp_path):
    from aksharallm.config import Config
    from aksharallm.dictate.tagger import TaggerObjective

    class _DS:
        device = "cpu"
        path = None

    cfg = Config()
    cfg.model = _tiny_tagger().cfg
    cfg.train.seq_len = 32
    obj = TaggerObjective(cfg)
    obj.tok = _Tok()
    from aksharallm.dictate.tagger import WordEncoder
    obj.enc = WordEncoder(obj.tok)
    words = punct.examples_from_text("Real text here. Another sentence, with a comma? Yes.", False)
    obj._windows = lambda ds, k, g=None: [np.zeros(4, dtype=np.int64)] * k
    obj._workers = False
    import aksharallm.dictate.tagger as T
    orig = T._window_words
    T._window_words = lambda tok, w: words
    try:
        ids, lab = obj._rows(_DS(), 3)
    finally:
        T._window_words = orig
    assert ids.shape == (3, 32)
    # Every label sits on the LAST piece of its word, in order, words repeating as the row is
    # filled from window after window; every other position is ignored.
    pos, want = 0, {}
    k = 0
    while True:
        w = words[k % len(words)]
        n = len(obj.enc.word(w.text))
        if pos + n > 32:
            break
        want[pos + n - 1] = w.label
        pos += n
        k += 1
    got = {i: int(lab[0, i]) for i in range(32) if lab[0, i] != punct.IGNORE}
    assert got == want
    m = _tiny_tagger()
    loss = obj.loss(m, (torch.from_numpy(ids), torch.from_numpy(lab)))
    assert math.isfinite(float(loss.detach())) and abs(float(loss.detach()) - math.log(12)) < 1.5
    loss.backward()


def test_the_pretraining_loop_picks_the_tagger_objective():
    from aksharallm.config import load_config
    from aksharallm.dictate.tagger import TaggerObjective
    from aksharallm.train.pretrain import objective_for
    assert isinstance(objective_for(load_config("configs/punct.yaml")), TaggerObjective)


def test_score_is_per_class_and_never_accuracy():
    from aksharallm.dictate.tagger import rules_only, score_labels
    gold = [label(COMMA, CAP)] + [label(NONE, LOWER)] * 8 + [label(PERIOD, LOWER)]
    s = score_labels(gold, rules_only(["a"] * 10))
    assert s["exact"] > 0.8                    # a do-nothing tagger looks good on accuracy...
    assert s["comma"]["f1"] == 0.0             # ...and scores zero where it matters
    assert s["period"]["f1"] == 1.0


# ---------------------------------------------------------------------------------------
# cleanup
# ---------------------------------------------------------------------------------------


class _Fake:
    """A punctuator that puts a full stop after the words it is told to."""

    def __init__(self, stops=()):
        self.stops = set(stops)

    def labels(self, words):
        return [label(PERIOD if w in self.stops else NONE, LOWER) for w in words]


def test_cleanup_removes_fillers_and_obeys_the_three_commands():
    c = clean("um so i met shaun uh at the station new paragraph we meet at five "
              "scratch that call me", _Fake(stops={"station"}))
    assert c.text == "So I met shaun at the station.\n\nCall me."
    kinds = [s["step"] for s in c.steps]
    assert kinds == ["fillers", "new paragraph", "scratch that"]
    assert c.steps[-1]["removed"] == "we meet at five"


def test_scratch_that_keeps_the_finished_sentence_before_it():
    c = clean("i am here we go at five scratch that at six", _Fake(stops={"here"}))
    assert c.text == "I am here. At six."


def test_cleanup_never_adds_a_word():
    rng = random.Random(1)
    pool = ["the", "um", "new", "line", "scratch", "that", "i", "go", "uh", "paragraph", "so"]
    for _ in range(300):
        words = [rng.choice(pool) for _ in range(rng.randint(0, 15))]
        out = words_of(clean(" ".join(words), _Fake(stops={"go"})).text.replace("\n", " "))
        from collections import Counter
        assert not (Counter(w for w in out if w) - Counter(words))


# ---------------------------------------------------------------------------------------
# learning from corrections
# ---------------------------------------------------------------------------------------


def test_one_correction_teaches_a_word_and_a_spelling_and_two_make_a_rule(tmp_path):
    p = Personal(tmp_path, known=lambda w: w in {"i", "met", "at", "the", "office", "today"})
    L = p.learn("I met Sean at the get hub office.", "I met Shaun at the GitHub office.")
    assert set(L["words"]) == {"shaun", "github"} and set(L["spellings"]) == {"Shaun", "GitHub"}
    assert p.active_replacements() == {}
    p.learn("I met Sean today.", "I met Shaun today.")
    assert p.active_replacements() == {("sean",): ["shaun"]}
    assert p.rewrite("i saw sean".split())[0] == ["i", "saw", "shaun"]
    again = Personal(tmp_path)
    assert again.spellings() == {"shaun": "Shaun", "github": "GitHub"}


def test_a_fix_between_two_ordinary_words_teaches_nothing(tmp_path):
    p = Personal(tmp_path, known=lambda w: True)
    L = p.learn("their car", "there car")
    assert L["ignored"] and not L["words"] and not p.words and not p.replacements


def test_a_rewrite_is_not_a_correction(tmp_path):
    p = Personal(tmp_path)
    L = p.learn("meet at five", "let us have lunch tomorrow instead please")
    assert L["ignored"] and not p.words


def test_a_capital_at_a_sentence_start_is_not_a_spelling(tmp_path):
    p = Personal(tmp_path)
    p.learn("ok. maybe later", "ok. Maybe later")
    assert p.spellings() == {}


def test_correcting_a_word_to_something_else_resets_its_count(tmp_path):
    p = Personal(tmp_path)
    p.learn("sean", "shaun")
    p.learn("sean", "shawn")
    assert p.replacements["sean"] == {**p.replacements["sean"], "to": "shawn", "count": 1}


# ---------------------------------------------------------------------------------------
# the day-two replay
# ---------------------------------------------------------------------------------------


def _frames(text, competitor=None):
    """One letter per frame with a blank between; optionally a likelier wrong letter."""
    rows = []
    for i, ch in enumerate(text):
        r = np.full(vocab.VOCAB_SIZE, -20.0)
        r[vocab.STOI[ch]] = math.log(0.98)
        if competitor and i in competitor:
            r[vocab.STOI[ch]] = math.log(0.4)
            r[vocab.STOI[competitor[i]]] = math.log(0.58)
        rows.append(r)
        b = np.full(vocab.VOCAB_SIZE, -20.0)
        b[vocab.BLANK] = 0.0
        rows.append(b)
    return np.stack(rows)


def test_the_replay_shows_a_corrected_name_coming_out_right_next_time(tmp_path):
    """The ear hears 'zorc' every time; after one correction the dictionary carries 'zork'.
    A curve that does not rise here would mean the correction loop is decoration."""
    from aksharallm.asr.daytwo import correction_replay
    from aksharallm.asr.ngram import TrigramLM
    p = tmp_path / "lm.txt"
    p.write_text("\n".join(["the cat sat", "we met the man", "the man sat"] * 5) + "\n")
    lm = TrigramLM.build(p, vocab_size=100, chunk_words=50, progress=None)
    texts = ["we met zork", "the man met zork", "zork sat", "the cat met zork", "the cat sat"]
    lps = [_frames(t, {t.index("zork") + 3: "c"} if "zork" in t else None) for t in texts]
    r = correction_replay(texts, lps, lm, {"alpha": 0.5, "beta": 1.0, "unk_penalty": -4.0},
                          min_occurrences=4, false_alarm_utts=5)
    assert r["targets"] == 1
    assert [c["recall"] for c in r["curve"]] == [0.0, 1.0, 1.0, 1.0]
    assert r["false_insertions"] == 0


def test_the_cleanup_check_counts_invented_words():
    from aksharallm.asr.daytwo import cleanup_check
    r = cleanup_check(["um i went home", "we left early"], None)
    assert r["invented"] == 0 and r["dropped"] == 0 and r["words"] == 6


# ---------------------------------------------------------------------------------------
# the pipeline and the desktop pieces
# ---------------------------------------------------------------------------------------


def test_long_audio_is_cut_in_its_quietest_moments():
    from aksharallm.dictate.pipeline import split_points
    sr = 16_000
    x = np.random.default_rng(0).standard_normal(sr * 60).astype(np.float32) * 0.3
    x[int(sr * 22.0): int(sr * 22.4)] = 0.0            # a pause before the first boundary
    cuts = split_points(x, sr)
    assert cuts and 22.0 * sr <= cuts[0] <= 22.4 * sr
    assert split_points(x[: sr * 20], sr) == []


def test_settings_refuse_a_key_they_do_not_know(tmp_path):
    from aksharallm.dictate.pipeline import Settings
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/portal.yaml").write_text("dictate:\n  outptu: type\n")
    with pytest.raises(ValueError, match="outptu"):
        Settings.load(tmp_path)
    (tmp_path / "configs/portal.yaml").write_text("dictate:\n  output: paste\n")
    assert Settings.load(tmp_path, device="cuda").output == "paste"


def test_the_toggle_says_how_to_start_the_daemon_when_none_is_running(tmp_path):
    from aksharallm.dictate import daemon as dm
    from aksharallm.dictate.pipeline import Settings
    r = dm.send(tmp_path, Settings(personal_dir="state"), "toggle", timeout=1)
    assert not r["ok"] and "daemon --bg" in r["error"]


def test_with_no_desktop_tools_delivery_says_so_instead_of_pretending(monkeypatch):
    from aksharallm.dictate import daemon as dm
    monkeypatch.setattr(dm.shutil, "which", lambda name: None)
    assert dm.deliver("hello", "type").startswith("none")
    assert dm.deliver("", "type") == "none"


def test_portal_jobs_cover_the_new_commands(tmp_path):
    """The parity rule: an `asr`/`dictate` command the tab can run gets a job kind."""
    from aksharallm.portal.dictate import AsrJobs, Dictation, DictationError
    d = Dictation(tmp_path, device_for=lambda: ("cpu", "test"))
    jobs = AsrJobs(d)
    assert {"daytwo", "punct_eval"} <= set(jobs.KINDS)
    with pytest.raises(DictationError, match="tagger"):
        jobs.command({"kind": "punct_eval"})
    with pytest.raises(DictationError):
        jobs.command({"kind": "daytwo", "checkpoint": "nope"})


def test_a_shortcut_gnome_did_not_keep_is_reported_not_claimed(monkeypatch):
    """Conda's own gsettings has no dconf backend: every `set` succeeds and is forgotten. The
    installer once said "installed" for a shortcut GNOME never saw. It reads back now."""
    from aksharallm.dictate import daemon as dm
    calls = []

    def forgetful(*args):          # accepts every write, remembers nothing
        calls.append(args)
        return "@as []" if args[0] == "get" else ""
    monkeypatch.setattr(dm, "_gsettings", forgetful)
    with pytest.raises(RuntimeError, match="did not keep"):
        dm.install_shortcut("cmd", "<Super><Alt>d")


def test_the_system_gsettings_is_preferred_over_whatever_is_first_on_path(monkeypatch):
    from aksharallm.dictate import daemon as dm
    monkeypatch.setattr(dm.shutil, "which", lambda name: "/opt/anaconda3/bin/gsettings")
    import pathlib
    if pathlib.Path("/usr/bin/gsettings").is_file():
        assert dm._gsettings_bin() == "/usr/bin/gsettings"


def test_the_portal_cleans_typed_words_with_the_same_pipeline(tmp_path):
    """The tab's "try the cleanup without speaking" box: `dictate clean`, from the browser."""
    from aksharallm.portal.dictate import Dictation, DictationError
    d = Dictation(tmp_path, device_for=lambda: ("cpu", "test"))
    r = d.clean("um so i met shaun new line see you")
    assert r["text"] == "So I met shaun.\nSee you."
    assert r["punctuator"] == "rules" and r["steps"][0] == {"step": "fillers", "removed": "um"}
    with pytest.raises(DictationError):
        d.clean("x" * 20_001)
