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
