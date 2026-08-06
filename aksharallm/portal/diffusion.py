"""The Diffusion tab's back end: watch a sentence resolve out of noise.

Shape of this one, and why
--------------------------
Like `interp`, everything here runs **inline** on the Playground's resident model rather
than as a subprocess job. A denoising run at Phase-1 scale is `steps` forward passes over a
few hundred tokens — well under a second on the card and a few seconds on the CPU — so a job
runner with a pid file and a log tail would be more machinery than the work. Reusing
`Playground.engine` also means this tab inherits the device policy, the idle unload and the
generation lock, so opening it can never cost somebody a training run.

The one thing here that touches no model at all is `corrupt_preview`, and it is the panel a
newcomer should meet first: it applies the *forward* process to text they typed and shows
what the model is asked to reconstruct. The whole objective is legible before anything is
generated, which is the same argument the Finetune tab's memory-budget table makes.

Read with: docs/19-diffusion.md -- the chapter this implements; it ends with the order to read
these files in.
"""

from __future__ import annotations

import time
from pathlib import Path

import torch

from ..diffusion.corrupt import corrupt, sample_t
from ..diffusion.evaluate import elbo, loss_by_t
from ..diffusion.generate import DiffusionError, diffusion_generate, infill
from ..infer.checkpoints import InferError

#: Ceilings, so a browser cannot ask for a denoising run that holds the model for a minute.
#: Generation is `steps` passes over the WHOLE sequence — there is no cache to make a longer
#: one cheap — so both numbers matter and neither is generous.
MAX_LENGTH = 256
MAX_STEPS = 128
#: Measurement is bounded the same way. The full curve is a CLI job.
MAX_BATCHES = 20


class Diffusion:
    """Everything the Diffusion tab can ask for, against the resident model."""

    def __init__(self, playground, root: Path | None = None):
        self.playground = playground
        self.root = Path(root) if root else Path.cwd()

    # ---- shared -----------------------------------------------------------------------

    def _loaded(self, ckpt_id: str):
        loaded = self.playground.engine.load(ckpt_id)
        if not loaded.model.cfg.is_diffusion:
            raise InferError(
                f"{loaded.info.rel} is an autoregressive checkpoint — causal attention and no "
                "[MASK] token, so it has never seen a masked sequence and unmasking it would "
                "return noise. Train configs/tiny-diffusion.yaml (the Dashboard's Start "
                "button) and come back. See docs/19.")
        return loaded

    def _cells(self, loaded, ids, mask_id: int) -> list:
        """Token ids -> one display string per position, or None where still masked."""
        out = []
        for i in ids:
            out.append(None if i == mask_id else loaded.tokenizer.decode([int(i)]))
        return out

    def overview(self, ckpt_id: str | None = None) -> dict:
        """What the tab needs before anything runs.

        Diffusion checkpoints are flagged here rather than filtered out: a picker that
        silently hides every checkpoint on a machine that has not trained one yet looks
        broken. Showing them all, greyed with a reason, is the honest empty state.
        """
        info = self.playground.overview()
        cks = []
        for c in info.get("checkpoints", []):
            cks.append({**c, "diffusion": bool(c.get("diffusion"))})
        out = {
            "checkpoints": cks,
            "any": any(c["diffusion"] for c in cks),
            "loaded": info.get("loaded"),
            "device": info.get("device"),
            "plan": info.get("plan"),
            "max_length": MAX_LENGTH, "max_steps": MAX_STEPS,
            "results": self.results(),
        }
        if ckpt_id:
            try:
                loaded = self._loaded(ckpt_id)
                out["current"] = {
                    "checkpoint": loaded.info.rel,
                    "mask_token_id": int(loaded.model.cfg.mask_token_id),
                    "vocab_size": loaded.model.cfg.vocab_size,
                    "max_seq_len": loaded.model.cfg.max_seq_len,
                    "device": loaded.device,
                }
            except (InferError, DiffusionError) as exc:
                out["current"] = {"error": str(exc)}
        return out

    # ---- the forward process, with no model in it ---------------------------------------

    def corrupt_preview(self, ckpt_id: str, text: str, t: float, seed: int = 0) -> dict:
        """Show what the model is *given* at mask rate `t`, and what it has to put back.

        No forward pass. This is the training objective's first half, applied to a sentence
        the reader typed — which is the cheapest possible way to understand what "noise"
        means for discrete tokens.
        """
        loaded = self._loaded(ckpt_id)
        mask_id = int(loaded.model.cfg.mask_token_id)
        ids = loaded.tokenizer.encode(text or "", bos=False)
        if not ids:
            raise InferError("type something to corrupt")
        ids = ids[:MAX_LENGTH]
        x = torch.tensor(ids, dtype=torch.long)[None, :]
        gen = torch.Generator().manual_seed(int(seed))
        rate = max(0.0, min(float(t), 1.0))
        c = corrupt(x, mask_id, torch.full((1,), rate), generator=gen)
        return {
            "t": rate,
            "realised": c.rate,
            "tokens": [loaded.tokenizer.decode([i]) for i in ids],
            "masked": c.masked[0].tolist(),
            # The 1/t weight, spelled out. At t = 0.05 one masked token counts for twenty;
            # seeing that number move as the slider does is the point of the panel.
            "weight": round(1.0 / max(rate, 1e-3), 1),
            "n_masked": int(c.masked.sum()),
            "n_tokens": len(ids),
        }

    # ---- generation ----------------------------------------------------------------------

    def generate(self, ckpt_id: str, *, prompt: str = "", length: int = 48,
                 steps: int = 16, temperature: float = 0.8, top_k: int = 50,
                 top_p: float = 0.95, remask: str = "low_confidence",
                 seed: int | None = None) -> dict:
        """Denoise a fresh sequence, and return every intermediate state.

        The trace is the whole reason this tab exists. The finished string says nothing
        about the order the positions were decided in, and that order is the model's
        reasoning made visible — the one thing about this paradigm that cannot be explained
        as well in prose as it can be watched.
        """
        loaded = self._loaded(ckpt_id)
        mask_id = int(loaded.model.cfg.mask_token_id)
        length = max(1, min(int(length), MAX_LENGTH))
        steps = max(1, min(int(steps), MAX_STEPS))
        prefix = (loaded.tokenizer.encode(prompt, bos=True) if prompt.strip()
                  else [loaded.tokenizer.bos_id])

        t0 = time.monotonic()
        ids, trace = diffusion_generate(
            loaded.model, length=length, steps=steps, prefix=prefix,
            temperature=float(temperature), top_k=int(top_k) or None, top_p=float(top_p),
            remask=remask, seed=seed, device=loaded.device, trace=True)
        elapsed = time.monotonic() - t0

        return {
            "checkpoint": loaded.info.rel,
            "device": loaded.device,
            "prefix_len": len(prefix),
            "text": loaded.tokenizer.decode([i for i in ids if i != mask_id]),
            "steps": [
                {"step": s.step, "cells": self._cells(loaded, s.ids, mask_id),
                 "committed": s.committed,
                 "confidence": [round(c, 3) for c in s.confidence],
                 "remaining": s.remaining}
                for s in trace
            ],
            "elapsed_s": round(elapsed, 3),
            "passes": len(trace) - 1,
            # The headline arithmetic of the paradigm: an autoregressive model would have
            # needed one forward pass per token. Shown as a ratio because that ratio is the
            # only thing diffusion is unambiguously better at.
            "tokens_per_pass": round(length / max(len(trace) - 1, 1), 2),
        }

    def infill(self, ckpt_id: str, prefix: str, suffix: str, length: int = 12,
               steps: int = 12, temperature: float = 0.8,
               seed: int | None = None) -> dict:
        """Write the middle, given both ends — the capability an AR model does not have."""
        loaded = self._loaded(ckpt_id)
        mask_id = int(loaded.model.cfg.mask_token_id)
        if not prefix.strip() and not suffix.strip():
            raise InferError("give it at least one end to work from")
        pre = loaded.tokenizer.encode(prefix, bos=True) if prefix.strip() else [
            loaded.tokenizer.bos_id]
        suf = loaded.tokenizer.encode(suffix) if suffix.strip() else []
        length = max(1, min(int(length), MAX_LENGTH))
        steps = max(1, min(int(steps), MAX_STEPS))
        middle, trace = infill(loaded.model, pre, suf, length=length, steps=steps,
                               temperature=float(temperature), seed=seed,
                               device=loaded.device, trace=True)
        return {
            "checkpoint": loaded.info.rel,
            "device": loaded.device,
            "prefix": prefix, "suffix": suffix,
            "middle": loaded.tokenizer.decode([i for i in middle if i != mask_id]),
            "steps": [
                {"step": s.step, "cells": self._cells(loaded, s.ids, mask_id),
                 "committed": s.committed, "remaining": s.remaining}
                for s in trace
            ],
            "prefix_len": len(pre), "suffix_len": len(suf),
        }

    # ---- measurement -----------------------------------------------------------------------

    def measure(self, ckpt_id: str, kind: str = "elbo", batches: int = 4,
                batch_size: int = 4, buckets: int = 8) -> dict:
        """The validation bound, or cross-entropy against mask rate.

        Bounded much harder than the CLI's defaults: this is a click, and a click that takes
        thirty seconds reads as a broken page. The numbers to put in a write-up come from
        `python -m aksharallm.diffusion <ckpt> elbo`, which the panel says.
        """
        from ..data.loader import TokenDataset

        loaded = self._loaded(ckpt_id)
        val_bin = self._val_bin(loaded)
        batches = max(1, min(int(batches), MAX_BATCHES))
        ds = TokenDataset(str(val_bin), loaded.model.cfg.max_seq_len, loaded.device)

        if kind == "by-t":
            rows = loss_by_t(loaded.model, ds, int(batch_size), batches,
                             buckets=max(2, min(int(buckets), 20)))
            return {"kind": "by-t", "rows": rows, "checkpoint": loaded.info.rel,
                    "device": loaded.device}
        out = elbo(loaded.model, ds, int(batch_size), batches)
        out.update({"kind": "elbo", "checkpoint": loaded.info.rel,
                    "device": loaded.device})
        return out

    def _val_bin(self, loaded) -> Path:
        rel = loaded.info.val_bin
        if not rel:
            raise InferError(
                f"{loaded.info.rel} does not record a validation split, so there is nothing "
                "to measure against.")
        path = Path(rel)
        if not path.is_absolute():
            path = self.root / rel
        if not path.is_file():
            raise InferError(f"the run's validation data ({rel}) is not on disk.")
        return path

    def results(self, limit: int = 20) -> list[dict]:
        """Measurements the CLI has written into `logs/diffusion/`, newest first."""
        import json

        folder = self.root / "logs" / "diffusion"
        if not folder.is_dir():
            return []
        rows = []
        for path in sorted(folder.glob("*.json"), key=lambda p: p.stat().st_mtime,
                           reverse=True)[:limit]:
            try:
                blob = json.loads(path.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            rows.append({"name": path.name, "when": int(path.stat().st_mtime),
                         "checkpoint": blob.get("checkpoint"),
                         "nelbo": blob.get("nelbo"),
                         "kind": "by-t" if "rows" in blob else "elbo"})
        return rows


def sample_rate(batch: int = 1, t_min: float = 1e-3) -> float:
    """One draw of the training mask rate — used by the tab's "surprise me" button."""
    return float(sample_t(batch, t_min, "cpu")[0])
