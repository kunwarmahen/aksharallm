"""Continued pretraining: `train.init` starts a run from another run's weights.

The Python specialist (PLAN.md Phase 4) is the first run in this repo that does not start
from random weights. `resume` cannot do it: it restores the old run's step, optimizer and
best-val, so pointing it at the base's ckpt_best.pt starts the new run at step 39,999 of a
schedule that has already decayed. `init` is weights only, and these tests pin the three
things that make it safe:

* it really is the old weights, at step 0;
* a resume wins over it, so night two continues instead of starting over from the base;
* a checkpoint trained with a different tokenizer is refused (gotcha 3).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
TOK = ROOT / "data" / "tinystories" / "tokenizer.json"

pytestmark = pytest.mark.skipif(not TOK.exists(), reason="needs a tokenizer on disk")


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    d = tmp_path_factory.mktemp("cpt")
    rng = np.random.default_rng(0)
    for name in ("train.bin", "val.bin"):
        rng.integers(0, 8192, size=40_000, dtype=np.uint16).tofile(d / name)
    return d


def config(path: Path, out_dir: Path, corpus: Path, max_steps: int, init: str = "null",
           lr: float = 1e-3) -> Path:
    path.write_text(f"""
name: cpt
model: {{vocab_size: 8192, d_model: 32, n_layers: 2, n_heads: 4, max_seq_len: 32}}
data:
  train_bin: {corpus / 'train.bin'}
  val_bin: {corpus / 'val.bin'}
  tokenizer: {TOK}
optim: {{lr: {lr}, warmup_steps: 2}}
train:
  out_dir: {out_dir}
  batch_size: 2
  grad_accum: 1
  seq_len: 32
  max_steps: {max_steps}
  eval_every: 0
  sample_every: 0
  ckpt_every: 0
  log_every: 0
  compile: false
  seed: 7
  resume: auto
  init: {init}
""")
    return path


def train(cfg: Path) -> None:
    proc = subprocess.run([sys.executable, "-m", "aksharallm.train.pretrain", str(cfg)],
                          cwd=ROOT, capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]


def sessions(out_dir: Path) -> list[dict]:
    recs = [json.loads(l) for l in (out_dir / "train_log.jsonl").read_text().splitlines()]
    return [r for r in recs if r.get("event") == "session_start"]


def test_init_loads_the_weights_and_starts_at_step_zero(tmp_path, corpus):
    """With lr 0 nothing moves, so the new run's weights must be the base's exactly -- and
    the run must be at ITS step 0, not the base's last step."""
    base = tmp_path / "base"
    train(config(tmp_path / "base.yaml", base, corpus, 4))
    src = torch.load(base / "ckpt_last.pt", map_location="cpu", weights_only=False)

    child = tmp_path / "child"
    train(config(tmp_path / "child.yaml", child, corpus, 1,
                 init=str(base / "ckpt_last.pt"), lr=0.0))
    got = torch.load(child / "ckpt_last.pt", map_location="cpu", weights_only=False)
    assert got["step"] == 0
    for k, v in src["model"].items():
        assert torch.equal(v, got["model"][k]), k
    s = sessions(child)[0]
    assert s["start_step"] == 0
    assert s["init"] == str(base / "ckpt_last.pt") and s["init_step"] == 3
    # A fresh optimizer: the base's AdamW moments describe a schedule that has ended.
    assert got["optimizer"]["state"] != {} and src["optimizer"]["state"] != {}


def test_a_resume_wins_over_init(tmp_path, corpus):
    """Night two must carry on from the child's own ckpt_last.pt. If `init` won, every
    night would restart from the base and the run would never get past its first window."""
    base = tmp_path / "base"
    train(config(tmp_path / "base.yaml", base, corpus, 2))
    child = tmp_path / "child"
    cfg = config(tmp_path / "child.yaml", child, corpus, 2, init=str(base / "ckpt_last.pt"))
    train(cfg)
    first = torch.load(child / "ckpt_last.pt", map_location="cpu", weights_only=False)
    cfg = config(tmp_path / "child.yaml", child, corpus, 4, init=str(base / "ckpt_last.pt"))
    train(cfg)
    s = sessions(child)
    assert [r["start_step"] for r in s] == [0, 2]
    assert s[1]["init"] is None
    assert first["step"] == 1


def test_a_checkpoint_from_another_tokenizer_is_refused(tmp_path):
    from aksharallm.config import load_config
    from aksharallm.model.transformer import Transformer
    from aksharallm.train.pretrain import init_weights

    cfg_path = config(tmp_path / "c.yaml", tmp_path / "o", tmp_path, 1)
    cfg = load_config(str(cfg_path))
    model = Transformer(cfg.model)
    ck = tmp_path / "other.pt"
    torch.save({"model": model.state_dict(), "step": 5,
                "config": {"data": {"tokenizer": str(tmp_path / "different.json")}}}, ck)
    with pytest.raises(ValueError, match="tokenizer"):
        init_weights(ck, model, cfg, "cpu")
    # ...and the same checkpoint with the matching tokenizer loads.
    torch.save({"model": model.state_dict(), "step": 5,
                "config": {"data": {"tokenizer": str(TOK)}}}, ck)
    assert init_weights(ck, model, cfg, "cpu") == (str(ck), 5)
