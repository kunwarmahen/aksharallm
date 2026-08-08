"""The masked-diffusion training objective, as a drop-in for the autoregressive one.

There is no `train/diffusion.py` in this project, and that is the point of this file.
Gradient accumulation, the LR schedule, mixed precision, checkpoint/resume, the stop file,
the session log, throughput and the end-of-run report are **not** properties of
next-token prediction — they are properties of training anything for six days on one card.
So `train/pretrain.py` keeps all of it and asks an *objective* for the only three things
that differ between the two paradigms:

    batch(dataset, n)      what a micro-batch is
    loss(model, batch)     what the model is scored on
    evaluate(...)          what "val loss" means
    sample(...)            what "show me what it writes" means

`pretrain.ARObjective` is the one this repo has always had, extracted unchanged. This is
the second one. Swapping between them is a config key — `model.causal` — and the loop
around them never learns which is which.

Read with: docs/20-diffusion.md -- the chapter this implements; it ends with the order to read
these files in. See also docs/05-pretraining.md.
"""

from __future__ import annotations

import torch

from .corrupt import corrupt, diffusion_loss, sample_t
from .evaluate import elbo
from .generate import decode_with_masks, diffusion_generate


class DiffusionObjective:
    """Train a bidirectional transformer to un-mask what was hidden from it."""

    name = "masked diffusion"
    #: What the val number is. Printed beside it everywhere, because "loss 1.9" from this
    #: objective and "loss 1.9" from the autoregressive one are different quantities.
    metric = "nelbo"
    comparable_to_ar = False

    def __init__(self, cfg):
        self.cfg = cfg
        self.mask_id = int(cfg.model.mask_token_id)
        self.t_min = float(cfg.diffusion.t_min)
        self.eval_seed = int(cfg.diffusion.eval_seed)
        self.sample_steps = int(cfg.diffusion.sample_steps)
        self._stats: dict[str, float] = {}

    # ---- pre-flight ---------------------------------------------------------------------

    def check(self, tok) -> None:
        """The vocabulary rule, which is the one thing about this that breaks compatibility.

        The tokenizer is untouched — the same `tokenizer.json` the dense baseline used, so
        the two runs see identical text. The model gets **one extra embedding row** for
        `[MASK]`, at an id the tokenizer can never emit. That is why `vocab_size` here is
        the tokenizer's plus one (or more, if padded for the tensor cores) and why a
        diffusion checkpoint cannot be loaded into an autoregressive run or the reverse:
        the embedding matrices are different shapes.
        """
        v_model, v_tok = self.cfg.model.vocab_size, tok.vocab_size
        if v_model <= v_tok:
            raise ValueError(
                f"a diffusion run needs room for [MASK]: model.vocab_size is {v_model} but "
                f"the tokenizer already uses {v_tok} ids. Set vocab_size to at least "
                f"{v_tok + 1} and mask_token_id to {v_tok}.")
        if self.mask_id < v_tok:
            raise ValueError(
                f"mask_token_id {self.mask_id} is inside the tokenizer's range (0..{v_tok - 1}), "
                f"so it collides with a real token. Use {v_tok} or above.")

    def describe(self) -> str:
        return (f"masked diffusion — t ~ U({self.t_min}, 1), bidirectional attention, "
                f"[MASK] = id {self.mask_id}. Val loss is an ELBO upper bound and is NOT "
                "comparable with an autoregressive run's cross-entropy.")

    # ---- the three things that differ ----------------------------------------------------

    def batch(self, dataset, batch_size: int):
        """Only `x` is needed. The targets *are* `x` — that is what denoising means."""
        x, _y = dataset.get_batch(batch_size)
        return (x,)

    def loss(self, model, batch):
        (x,) = batch
        t = sample_t(x.shape[0], self.t_min, x.device)
        c = corrupt(x, self.mask_id, t)
        logits, _ = model(c.x_t, full_logits=True)
        loss, stats = diffusion_loss(logits, x, c)
        self._stats = {k: float(v) for k, v in stats.items()}
        return loss

    def evaluate(self, model, dataset, batch_size: int, n_batches: int, ctx) -> float:
        return elbo(model, dataset, batch_size, n_batches,
                    t_min=self.t_min, seed=self.eval_seed, ctx=ctx)["nelbo"]

    def sample(self, model, tok, prompt: str, device: str) -> str:
        """A short unconditional sample, denoised from all-masks.

        The prompt is used as a prefix so the mid-run samples of this run and of the dense
        baseline start from the same words and can be read side by side.
        """
        raw = model._orig_mod if hasattr(model, "_orig_mod") else model
        was_training = raw.training
        raw.eval()
        prefix = tok.encode(prompt, bos=True) if prompt else [tok.bos_id]
        ids, _ = diffusion_generate(raw, length=64, steps=self.sample_steps, prefix=prefix,
                                    temperature=0.8, top_k=50, device=device)
        if was_training:
            raw.train()
        return decode_with_masks(tok, ids, self.mask_id)

    # ---- extras for the step line ---------------------------------------------------------

    def stats(self) -> dict:
        """Numbers from the last micro-batch, for the log.

        `ce` — plain cross-entropy on the masked positions — is here because the loss itself
        is `1/t`-weighted and therefore has no intuitive scale. `ce` does: it is "how
        surprised was the model by a token it could not see", in nats, and it is the number
        to watch when wondering whether anything is being learned at all.
        """
        if not self._stats:
            return {}
        return {"ce": self._stats.get("ce_masked"), "mask": self._stats.get("mask_rate")}

    def log_suffix(self) -> str:
        s = self.stats()
        if not s:
            return ""
        return f" | ce {s['ce']:.3f} | masked {s['mask'] * 100:.0f}%"
