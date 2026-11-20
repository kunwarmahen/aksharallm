"""From games to tokens: a word-level tokenizer and the `train.bin` / `val.bin` the
pretraining loop already reads.

There is no codec here, and none is needed: a frame of this game is already 64 symbols from
an alphabet of seven, so each cell IS a token. (Vision needed patches and audio needed a
learned codec only because pixels and samples are not small alphabets.) One episode is

    <|endoftext|>  F0  a0  F1  a1  F2 ...      F = 64 cell tokens, a = one action token

written back to back, so the trainer's random 512-token windows start anywhere, mid-frame
included, exactly as a text model's windows start mid-sentence. The tokenizer is a real
`tokenizers` WordLevel model saved as `tokenizer.json`, so `Tokenizer`, the trainer's vocab
check and the checkpoint's tokenizer path all work unchanged.

Read with: docs/25-world-model.md -- the chapter this implements; it ends with the order to
read these files in.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .game import ACTIONS, CELLS, SIZE, episode

SPECIALS = ("<|endoftext|>", "<|pad|>", "<unk>")
VOCAB = SPECIALS + CELLS + ACTIONS
FRAME = SIZE * SIZE


def write_tokenizer(path: Path) -> None:
    """A WordLevel tokenizer over `VOCAB`, split on whitespace."""
    from tokenizers import Tokenizer, models, pre_tokenizers
    tok = Tokenizer(models.WordLevel({w: i for i, w in enumerate(VOCAB)}, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tok.add_special_tokens(list(SPECIALS))
    tok.save(str(path))


def ids_of(words: list[str]) -> list[int]:
    index = {w: i for i, w in enumerate(VOCAB)}
    return [index[w] for w in words]


def episode_tokens(frames, actions) -> list[int]:
    words = ["<|endoftext|>"]
    for f, a in zip(frames, actions + [None]):
        words += f
        if a is not None:
            words.append(a)
    return ids_of(words)


def prepare(out: str | Path, *, episodes: int = 60_000, val_episodes: int = 2_000,
            steps: int = 30, seed: int = 0, progress=print) -> dict:
    """Write `train.bin`, `val.bin`, `tokenizer.json` and `meta.json`; returns the meta.

    The val episodes come from a separate seed stream, and the evaluation draws its own
    episodes from a third (`evaluate.py`), so nothing scored was trained on. `steps` 30 makes
    an episode 31 x 64 + 30 + 1 = 2,015 tokens: a 512-token window holds about 7.8 steps."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    write_tokenizer(out / "tokenizer.json")
    meta = {"vocab": list(VOCAB), "frame_tokens": FRAME, "steps": steps, "seed": seed,
            "episodes": {}, "tokens": {}, "outcomes": {}}
    for split, n, s in (("train", episodes, seed), ("val", val_episodes, seed + 10_000)):
        rng = np.random.default_rng(s)
        buf, counts = [], {}
        for e in range(n):
            frames, actions, outcomes = episode(rng, steps)
            buf.extend(episode_tokens(frames, actions))
            for o in outcomes:
                counts[o] = counts.get(o, 0) + 1
            if progress and (e + 1) % 10_000 == 0:
                progress(f"[world] {split} {e + 1}/{n} episodes")
        arr = np.array(buf, dtype=np.uint16)
        arr.tofile(out / f"{split}.bin")
        meta["episodes"][split], meta["tokens"][split] = n, int(arr.size)
        meta["outcomes"][split] = counts
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    return meta
