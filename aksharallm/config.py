"""Config objects. Everything that differs between runs lives in a YAML file."""

from __future__ import annotations

from dataclasses import MISSING, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    vocab_size: int = 8192
    d_model: int = 384
    n_layers: int = 6
    n_heads: int = 6
    n_kv_heads: int | None = None  # None -> = n_heads (plain MHA). Set lower for GQA.
    d_ff: int | None = None  # None -> auto: ~8/3 * d_model, rounded to multiple_of
    multiple_of: int = 64  # keeps d_ff a nice size for tensor cores
    max_seq_len: int = 512
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    tie_embeddings: bool = True
    dropout: float = 0.0

    def __post_init__(self):
        if self.n_kv_heads is None:
            self.n_kv_heads = self.n_heads
        assert self.d_model % self.n_heads == 0, "d_model must divide evenly by n_heads"
        assert self.n_heads % self.n_kv_heads == 0, "n_heads must be a multiple of n_kv_heads"
        if self.d_ff is None:
            hidden = int(8 * self.d_model / 3)
            self.d_ff = self.multiple_of * ((hidden + self.multiple_of - 1) // self.multiple_of)

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads


@dataclass
class DataConfig:
    train_bin: str = "data/tinystories/train.bin"
    val_bin: str = "data/tinystories/val.bin"
    tokenizer: str = "data/tinystories/tokenizer.json"
    # Optional blended training: a list of {bin, weight} dicts. When set, training samples
    # each batch from these files by weight (via MixedTokenDataset) and `train_bin` is
    # ignored. `val_bin` is still a single file. Example:
    #   train_sources:
    #     - {bin: data/blend/fineweb.bin, weight: 0.85}
    #     - {bin: data/blend/code.bin,    weight: 0.15}
    train_sources: list | None = None


@dataclass
class OptimConfig:
    lr: float = 6e-4
    min_lr_ratio: float = 0.1  # final LR = lr * min_lr_ratio
    warmup_steps: int = 200
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    schedule: str = "cosine"  # cosine | wsd | constant


@dataclass
class TrainConfig:
    out_dir: str = "checkpoints/tiny"
    # Global batch size in tokens = batch_size * grad_accum * seq_len
    batch_size: int = 32  # micro-batch, per step, per device
    grad_accum: int = 4
    seq_len: int = 512  # must be <= model.max_seq_len
    max_steps: int = 6000
    eval_every: int = 250
    eval_batches: int = 40
    sample_every: int = 500
    ckpt_every: int = 1000
    keep_last_n: int = 2
    log_every: int = 10
    compile: bool = True
    seed: int = 1337
    wandb_project: str | None = None
    wandb_run: str | None = None
    resume: str | None = None  # path to ckpt, or "auto" to pick up latest in out_dir
    # Bounded stops. Neither ends the run: both save ckpt_last.pt and exit cleanly, so
    # re-running with resume:auto continues with no loss spike. Use them to train in
    # chunks ("give me 500 more steps tonight") instead of babysitting a kill.
    # Both are INCLUSIVE -- the step you name is trained, logged and checkpointed, and the
    # resume picks up the one after it.
    stop_after: int | None = None  # do N steps in this invocation, then stop
    stop_at: int | None = None  # finish this absolute step, then stop
    # The same idea measured in wall-clock rather than steps: "train for half an hour".
    # Counted from the first training step, so pre-flight and torch.compile don't eat it.
    stop_after_s: int | None = None  # train for N seconds this invocation, then stop


@dataclass
class Config:
    name: str = "tiny"
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


def _build(cls, raw: dict[str, Any]):
    """Recursively instantiate nested dataclasses, erroring on unknown keys."""
    kwargs = {}
    known = {f.name: f for f in fields(cls)}
    for key, value in raw.items():
        if key not in known:
            raise ValueError(f"unknown config key '{key}' for {cls.__name__}")
        # Nested dataclass sections (model:, data:, ...) are declared with
        # default_factory; the "not set" sentinel is dataclasses.MISSING, not None.
        factory = known[key].default_factory
        default = factory() if factory is not MISSING else None
        if is_dataclass(default) and isinstance(value, dict):
            kwargs[key] = _build(type(default), value)
        else:
            kwargs[key] = value
    return cls(**kwargs)


def load_config(path: str | Path, overrides: list[str] | None = None) -> Config:
    """Load a YAML config. `overrides` are `dotted.key=value` strings from the CLI."""
    raw = yaml.safe_load(Path(path).read_text()) or {}
    for ov in overrides or []:
        key, _, val = ov.partition("=")
        node = raw
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(val)  # parses ints/floats/bools/null
    cfg = _build(Config, raw)
    assert cfg.train.seq_len <= cfg.model.max_seq_len, "train.seq_len exceeds model.max_seq_len"
    return cfg


def config_to_dict(obj) -> dict:
    if is_dataclass(obj):
        return {f.name: config_to_dict(getattr(obj, f.name)) for f in fields(obj)}
    return obj
