"""The toy world model (docs/25): the game's rules, its tokens, and a scorer that is right.

The scorer is tested against a model whose answers are known: one that copies the previous
frame. It must score exactly 100% where the world does not change (`blocked`, `stayed`) and
exactly 0% where it does, which pins every slice the scorer takes (one token off and the
copy lands on the wrong cell)."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import torch

from aksharallm.world import game
from aksharallm.world.data import FRAME, VOCAB, episode_tokens, prepare
from aksharallm.world.evaluate import next_frame, rollout


def _level(rows):
    return game.parse([ch for row in rows for ch in row.split()])


def test_a_push_moves_the_box_and_a_wall_or_a_second_box_blocks_it():
    s = _level(["# # # # # # # #",
                "# . . . . . . #",
                "# . @ $ . $ $ #",
                "# . . . . . . #",
                "# . . . . . . #",
                "# . . . . o o #",
                "# . . . . . o #",
                "# # # # # # # #"])
    n, why = game.step(s, "right")
    assert why == "pushed" and n.agent == (2, 3) and n.boxes[2, 4] and not n.boxes[2, 3]
    n2, why = game.step(n, "right")                 # box at (2,4) then box at (2,5): blocked
    assert why == "blocked" and n2.agent == n.agent and (n2.boxes == n.boxes).all()
    n3, why = game.step(s, "up")
    assert why == "moved" and n3.agent == (1, 2)
    n4, why = game.step(n3, "up")                   # into the border
    assert why == "blocked" and n4.agent == (1, 2)
    assert game.step(s, "stay")[1] == "stayed"


def test_render_and_parse_round_trip_and_parse_refuses_impossible_frames():
    rng = np.random.default_rng(0)
    for _ in range(50):
        s = game.random_level(rng)
        f = game.render(s)
        assert game.render(game.parse(f)) == f
    bad = game.render(game.random_level(rng))
    floor = bad.index(".")
    two_agents = list(bad)
    two_agents[floor] = "@"
    assert game.parse(two_agents) is None
    holed = list(bad)
    holed[0] = "."
    assert game.parse(holed) is None


def test_an_episode_is_written_frame_action_frame(tmp_path):
    frames, actions, _ = game.episode(np.random.default_rng(1), 3)
    toks = episode_tokens(frames, actions)
    assert len(toks) == 1 + 4 * FRAME + 3
    assert [VOCAB[i] for i in toks[1 + FRAME : 2 + FRAME]] == [actions[0]]
    assert [VOCAB[i] for i in toks[2 + FRAME : 2 + 2 * FRAME]] == frames[1]
    meta = prepare(tmp_path, episodes=4, val_episodes=2, steps=3, progress=None)
    assert meta["tokens"]["train"] == 4 * len(toks)
    assert np.fromfile(tmp_path / "train.bin", dtype=np.uint16).max() < len(VOCAB)
    from aksharallm.tokenizer.tokenizer import Tokenizer
    tok = Tokenizer(tmp_path / "tokenizer.json")
    assert tok.vocab_size == len(VOCAB) and tok.encode("# @ up") == [VOCAB.index(w) for w in "# @ up".split()]
    assert json.loads((tmp_path / "meta.json").read_text())["vocab"] == list(VOCAB)


class CopyLastFrame(torch.nn.Module):
    """Predicts that the cell 65 tokens back (the same cell, one frame earlier) comes next."""

    cfg = SimpleNamespace(max_seq_len=512)

    def forward(self, x, full_logits=True):
        prev = torch.roll(x, FRAME, dims=1)          # position t sees x[t - 64]
        return torch.nn.functional.one_hot(prev, len(VOCAB)).float(), None


def test_the_scorer_gives_a_copying_model_exactly_its_due():
    r = next_frame(CopyLastFrame(), episodes=20, steps=12, history=2)
    by = r["by_outcome"]
    for w in ("blocked", "stayed"):
        if w in by:
            assert by[w]["exact"] == 1.0
    for w in ("moved", "pushed"):
        assert by[w]["exact"] == 0.0 and by[w]["cells"] > 0.9   # 2-3 cells change, not 64
    assert abs(r["exact"] - r["copy_baseline"]) < 1e-12


def test_a_copying_model_rolls_out_true_frames_only_while_nothing_moves():
    r = rollout(CopyLastFrame(), episodes=10, horizon=4)
    assert r["impossible_by_step"] == [0.0] * 4      # a copied frame is always a real one
    assert all(0.0 <= x <= 1.0 for x in r["exact_by_step"])


# ---- the harder world: keys, doors and a slippery floor (world/keys.py, world/calibrate.py) ----

from math import log

from aksharallm.world import keys
from aksharallm.world.calibrate import score, transitions
from aksharallm.world.data import game_for_vocab, vocab_of


def _keys_level(rows, held=0):
    s = keys.parse([ch for row in rows for ch in row.split()])
    s.held = held
    return s


ROOM = ["# # # # # # # #",
        "# . . . . . . #",
        "# . k @ D . . #",
        "# . . . . . . #",
        "# . . . . . . #",
        "# . . . . . . #",
        "# . . . . . . #",
        "# # # # # # # #"]


def test_a_door_opens_only_for_a_key_the_frame_does_not_show():
    s = _keys_level(ROOM)
    shut, why = keys.apply(s, "right")
    assert why == "locked" and shut.agent == s.agent and shut.doors[2, 4]
    took, why = keys.apply(s, "left")
    assert why == "took_key" and took.held == 1 and not took.keys[2, 2]
    assert keys.render(took) == keys.render(keys.parse(keys.render(took)))   # held is invisible
    back, _ = keys.apply(took, "right")
    opened, why = keys.apply(back, "right")
    assert why == "unlocked" and opened.held == 0 and not opened.doors[2, 4]
    # the same frame, with and without a key in hand, goes two different ways
    assert keys.render(_keys_level(ROOM, held=1)) == keys.render(s)
    assert keys.apply(_keys_level(ROOM, held=1), "right")[1] == "unlocked"


def test_the_world_distribution_is_exact():
    s = _keys_level(ROOM)
    br = keys.transition(s, "up")
    assert abs(sum(p for p, *_ in br) - 1) < 1e-12 and len(br) == 4
    assert br[0][0] == 1 - keys.SLIP and not br[0][3] and all(b[3] for b in br[1:])
    assert [b[2] for b in br] == ["moved", "moved", "took_key", "locked"]   # up; down left right
    (p, _, why, slipped), = keys.transition(s, "stay")       # stay never slips
    assert (p, why, slipped) == (1.0, "stayed", False)


def test_each_game_is_told_apart_by_its_vocabulary_alone():
    assert game_for_vocab(len(vocab_of("sokoban"))) == "sokoban" == game_for_vocab(len(VOCAB))
    assert game_for_vocab(len(vocab_of("keys"))) == "keys"
    assert game_for_vocab(32768) is None


class _Uniform(torch.nn.Module):
    cfg = SimpleNamespace(max_seq_len=512)

    def forward(self, x, full_logits=True):
        return torch.zeros(*x.shape, len(vocab_of("keys"))), None


class _CopyKeys(torch.nn.Module):
    """Certain that each cell is what it was one frame earlier."""
    cfg = SimpleNamespace(max_seq_len=512)

    def forward(self, x, full_logits=True):
        prev = torch.roll(x, FRAME, dims=1)
        return 50.0 * torch.nn.functional.one_hot(prev, len(vocab_of("keys"))).float(), None


def test_kl_is_exact_against_a_model_whose_probabilities_are_known():
    """A uniform model gives every frame 17^-64, so KL(true || model) for one transition is
    64 log 17 - H(true), whatever the frames: the scorer must reproduce that to rounding."""
    recs = transitions(6, 12, 2, seed=1)
    want = sum(64 * log(17) + sum(p * log(p) for p in r["p"]) for r in recs) / len(recs)
    got = score(_Uniform(), episodes=6, steps=12, history=2, seed=1)
    assert abs(got["calibration"]["kl"] - want) < 1e-6 * want     # float32 sums of 64 logs
    assert all(r["belief"] == 0.5 for r in got["calibration"]["doors"].values() if r["n"])


def test_the_door_split_reads_a_model_that_never_believes_in_keys():
    """A copying model predicts the shut door every time (shut = unchanged), so its belief is
    ~0: right on every 'not held' door and wrong on every held one, in or before the context.
    And its most-likely-frame score must equal the copy baseline exactly."""
    r = score(_CopyKeys(), episodes=120, steps=30, history=5, seed=3)
    d = r["calibration"]["doors"]
    assert all(d[w]["n"] > 0 for w in d)
    assert d["not held"]["right"] == 1.0
    assert d["held, seen in context"]["right"] == 0.0 == d["held, before the context"]["right"]
    nf = r["next_frame"]
    assert abs(nf["exact"] - nf["copy_baseline"]) < 1e-12
    # the mode is the most likely frame: the intended move alone carries 1 - SLIP
    assert all(b["p_mode_true"] >= 1 - keys.SLIP - 1e-12 for b in nf["by_outcome"].values())
    json.dumps(r, allow_nan=False)        # the portal reads this file; one NaN empties the tab


def test_a_keys_dataset_is_written_with_its_own_vocabulary(tmp_path):
    meta = prepare(tmp_path, episodes=3, val_episodes=1, steps=4, game="keys", progress=None)
    assert meta["game"] == "keys" and meta["vocab"] == list(vocab_of("keys"))
    assert np.fromfile(tmp_path / "train.bin", dtype=np.uint16).max() < len(vocab_of("keys"))


def test_a_keys_rollout_follows_the_game_with_no_slips():
    from aksharallm.world.evaluate import play
    steps = play(_CopyKeys(), horizon=6, seed=0, game="keys")
    assert all(st["outcome"] in keys.OUTCOMES for st in steps[1:])
    assert all(not st["impossible"] for st in steps)


def test_seen_in_context_means_the_context_shows_a_key_vanishing():
    """The split is only meaningful if 'seen in context' is exactly 'the window contains a
    frame pair where a key disappears' (only a pickup removes a key)."""
    K = vocab_of("keys").index(keys.KEY)
    for r in transitions(300, 30, 5, seed=7):
        if not r["door"] or not r["door"]["held"]:
            continue
        frames = [r["ctx"][i:i + FRAME] for i in range(0, len(r["ctx"]) - 1, FRAME + 1)]
        counts = [f.count(K) for f in frames]
        vanished = any(b < a for a, b in zip(counts, counts[1:]))
        assert vanished == (r["door"]["where"] == "held, seen in context"), r["door"]["where"]
