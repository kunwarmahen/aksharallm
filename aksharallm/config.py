"""Config objects. Everything that differs between runs lives in a YAML file.

Read with: docs/04-pretraining.md -- the chapter this implements; it ends with the order to
read these files in. See also docs/03-model.md.
"""

from __future__ import annotations

from dataclasses import MISSING, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml

from .model.rope import RopeScaling


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

    #: Which attention kernel runs the softmax-weighted sum.
    #:   "sdpa"  — `F.scaled_dot_product_attention`, i.e. someone else's FlashAttention-2.
    #:   "flash" — ours, `model/flash.py`, written in Triton.
    #: The default is "sdpa" and should stay that way for a real run: our kernel matches it
    #: on the forward and is ~20% behind on the backward, so choosing it costs real hours
    #: over a six-day run. It is here to be *read*, benchmarked and mutated, and it silently
    #: falls back to SDPA for any shape it does not handle (see `flash.usable`).
    attn_impl: str = "sdpa"

    # ---- which direction attention runs (see docs/19) ----------------------------------
    #: `True` is a decoder-only language model: token *n* may look at 1..n and no further,
    #: which is what makes next-token prediction a valid objective at every position at once.
    #: `False` lets every position see every other one — the setting a **masked diffusion**
    #: model needs, because it denoises a whole corrupted sequence rather than extending a
    #: prefix. It is not a knob to try on an autoregressive run: with it off, predicting the
    #: next token is trivially solved by reading it, and the model learns nothing.
    causal: bool = True
    #: The id of the `[MASK]` token, for a diffusion model. Conventionally the LAST id in the
    #: vocabulary: the tokenizer keeps the ids it always had and the model gets one more row
    #: in its embedding, so `vocab_size = tokenizer.vocab_size + 1`. It lives in the *model*
    #: config, not the data config, because a checkpoint has to be able to say what its own
    #: mask id was — decoding it as an ordinary token would print a random word.
    mask_token_id: int | None = None

    # ---- long context (see docs/18) ----------------------------------------------------
    #: How to stretch RoPE past the window the weights were trained on. `type: none` is the
    #: identity, so a model that has never heard of this is unaffected. Written as a nested
    #: block in YAML:
    #:     rope_scaling: {type: yarn, factor: 4.0, original_max_seq_len: 1024}
    rope_scaling: RopeScaling = field(default_factory=RopeScaling)
    #: Sliding-window attention: each token sees at most this many keys back (None = all).
    #: Turns attention from O(T²) into O(T·w) and bounds the KV cache — but on its own it
    #: makes the model blind past `attn_window`, which is what `attn_sinks` repairs.
    attn_window: int | None = None
    #: "Attention sinks" — the first N tokens stay visible no matter how far the window has
    #: slid. Costs four keys and is the difference between a sliding window that works and
    #: one whose perplexity explodes; see docs/18 for why the model needs somewhere to park
    #: attention it does not want to spend.
    attn_sinks: int = 0

    # ---- mixture of experts (0 = dense; everything below is ignored) -------------------
    #: How many experts replace the single SwiGLU in each MoE block.
    n_experts: int = 0
    #: How many of them each token is routed to.
    moe_top_k: int = 2
    #: Width of ONE expert. Left unset it is `d_ff // moe_top_k`, which holds *active*
    #: parameters equal to the dense model's — identical FLOPs per token, more capacity,
    #: which is the claim MoE actually makes and therefore the honest thing to compare.
    #: Set it to `d_ff` for sparse upcycling, where each expert is a copy of a trained FFN.
    moe_expert_d_ff: int | None = None
    #: Weight on the load-balancing loss. Without it a few experts take everything and the
    #: rest never train — and the loss curve does not show it happening.
    moe_aux_alpha: float = 0.01
    #: Weight on the router z-loss, which stops the gate's logits drifting large.
    moe_z_alpha: float = 1e-3
    #: Put an MoE block every Nth layer (1 = all of them). 2 is the common published choice:
    #: it halves the parameter growth for most of the benefit, because neighbouring layers
    #: learn similar things and one of the pair can stay dense.
    moe_every: int = 1

    def __post_init__(self):
        if self.n_kv_heads is None:
            self.n_kv_heads = self.n_heads
        assert self.d_model % self.n_heads == 0, "d_model must divide evenly by n_heads"
        assert self.n_heads % self.n_kv_heads == 0, "n_heads must be a multiple of n_kv_heads"
        if self.attn_impl not in ("sdpa", "flash"):
            raise ValueError(f"attn_impl must be 'sdpa' or 'flash', got {self.attn_impl!r}")
        # Ten call sites rebuild a model with `ModelConfig(**ckpt["model_config"])`, where
        # the nested section is a plain dict because that is what `config_to_dict` wrote.
        # Coercing here is what makes every one of them work without ten edits -- and what
        # stops a `.type` lookup silently reading a dict attribute that does not exist.
        if isinstance(self.rope_scaling, dict):
            self.rope_scaling = RopeScaling(**self.rope_scaling)
        if self.attn_window is not None and self.attn_window < 1:
            raise ValueError(f"attn_window must be >= 1 or null, got {self.attn_window}")
        if self.attn_sinks < 0:
            raise ValueError(f"attn_sinks must be >= 0, got {self.attn_sinks}")
        if self.attn_sinks and self.attn_window is None:
            raise ValueError("attn_sinks only means something with attn_window set")
        if not self.causal:
            # Both of these are causal-shaped ideas. A sliding window is defined here as
            # "the last w keys" and attention sinks exist because a causal model dumps
            # leftover attention on the first token it can always see; neither has a
            # meaning once every position sees every other one. Refuse rather than compute
            # something that looks like an answer.
            if self.attn_window is not None:
                raise ValueError("attn_window is a causal idea; it does not apply with "
                                 "causal: false (see docs/19)")
            if self.mask_token_id is None:
                raise ValueError("causal: false with no mask_token_id — a bidirectional "
                                 "model has no objective to train on. Set "
                                 "mask_token_id: <vocab_size - 1> for masked diffusion.")
        if self.mask_token_id is not None and not 0 <= self.mask_token_id < self.vocab_size:
            raise ValueError(f"mask_token_id {self.mask_token_id} is outside the vocabulary "
                             f"(0..{self.vocab_size - 1})")
        if self.d_ff is None:
            hidden = int(8 * self.d_model / 3)
            self.d_ff = self.multiple_of * ((hidden + self.multiple_of - 1) // self.multiple_of)
        if self.n_experts:
            if not 1 <= self.moe_top_k <= self.n_experts:
                raise ValueError(f"moe_top_k must be 1..{self.n_experts}, "
                                 f"got {self.moe_top_k}")
            if self.moe_expert_d_ff is None:
                # Matched active parameters. Integer division can lose a little width when
                # d_ff does not divide by k; that is a real (small) difference and it is
                # better to be slightly *under* the dense budget than over it, because a
                # comparison that quietly favours the new thing is worth nothing.
                self.moe_expert_d_ff = max(1, self.d_ff // self.moe_top_k)
            if self.moe_every < 1:
                raise ValueError("moe_every must be >= 1")

    @property
    def is_moe(self) -> bool:
        return bool(self.n_experts)

    @property
    def is_diffusion(self) -> bool:
        """A masked diffusion model: bidirectional, with a `[MASK]` id to corrupt with.

        Every loader in the project rebuilds a model with `ModelConfig(**ckpt["model_config"])`,
        so this property makes a checkpoint **self-describing**. Nothing has to be told which
        paradigm a `.pt` came from — which matters, because an autoregressive sampler run
        against a diffusion checkpoint produces fluent-looking nonsense rather than an error.
        """
        return not self.causal and self.mask_token_id is not None

    def moe_layer(self, i: int) -> bool:
        """Whether layer `i` is a mixture-of-experts block."""
        return self.is_moe and (i % self.moe_every == 0)

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
class DiffusionConfig:
    """Knobs for the masked-diffusion objective. Ignored unless `model.causal: false`.

    See docs/19. The defaults are the MDLM/LLaDA formulation with nothing tuned, which is
    the point: there is no noise schedule to get right — "noise" for discrete tokens just
    means "replaced by [MASK]", and the mask rate is drawn uniformly.
    """

    #: Smallest mask rate drawn. The loss is weighted by 1/t, so a t of 1e-9 would multiply
    #: one unlucky token's cross-entropy by a billion and blow the gradient clip. Clamping
    #: the *draw* rather than the weight keeps the estimator honest: it is an unbiased ELBO
    #: for t ~ U(t_min, 1), and t_min is small enough that the missing slice is negligible.
    t_min: float = 1e-3
    #: The seed the validation masking uses. Fixed on purpose — with a fresh mask every
    #: evaluation, "best val" would partly be a record of which draw was kindest, and the
    #: 13.8M comparison against the dense baseline's 1.472 would be reading noise.
    eval_seed: int = 20260806
    #: Denoising steps used when the trainer samples mid-run. More steps = better text and
    #: linearly more compute; this is the one number that has no equivalent in an AR model.
    sample_steps: int = 32


@dataclass
class Config:
    name: str = "tiny"
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)


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
