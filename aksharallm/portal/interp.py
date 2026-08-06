"""The Interp tab's back end: one loaded model, four ways of looking inside it.

Every other job runner in this package (`evals`, `quantize`, `finetune`, `synth`) starts a
*subprocess* of a CLI and streams its log, because those jobs take minutes to hours and must
survive the portal restarting. Interpretability is the opposite shape: a logit lens is one
forward pass, an attention map is one more, and the answer is a small table rather than a
file. So this one runs **inline**, on the Playground's already-resident model — the same
decision `learn.py` made, and for the same reason.

Reusing `Playground.engine` matters beyond convenience: it means the Interp tab inherits the
device policy (a live training run keeps the card, the model loads on the CPU and says so),
the idle unload, and the generation lock, so opening this tab can never cost somebody a run.

The one thing that *is* slow is activation patching — layers × positions forward passes — so
it is bounded here rather than trusted: a long prompt against a 24-layer model would otherwise
hold the lock for a minute and look like a hung page.

Read with: docs/17-interpretability.md -- the chapter this implements; it ends with the order
to read these files in.
"""

from __future__ import annotations

from pathlib import Path

from ..infer.checkpoints import InferError
from ..interp.capture import attention_maps, attention_summary, run as capture_run
from ..interp.lens import layer_contributions, lens_story, logit_lens
from ..interp.patch import PatchError, patch_grid, summarise
from ..interp.sae import load as load_sae

#: Longest prompt the tab will accept, in tokens. Patching is quadratic in attention *and*
#: linear in positions, so the ceiling is what keeps a click from holding the model for a
#: minute. Generous enough for the prompts these tools are actually used on.
MAX_TOKENS = 64
MAX_PATCH_TOKENS = 24


class Interp:
    """Everything the Interp tab can ask for, against the resident model."""

    def __init__(self, playground, root: Path | None = None):
        self.playground = playground
        self.root = Path(root) if root else None

    # ---- shared ----------------------------------------------------------------------
    def _loaded(self, ckpt_id: str):
        loaded = self.playground.engine.load(ckpt_id)
        return loaded

    def _ids(self, loaded, text: str, limit: int = MAX_TOKENS) -> list[int]:
        ids = loaded.tokenizer.encode(text or "", bos=True)
        if not ids:
            raise InferError("the prompt is empty")
        if len(ids) > limit:
            raise InferError(
                f"that prompt is {len(ids)} tokens; this tab stops at {limit}. Looking inside "
                f"a model is one forward pass per layer per position — the limit is what keeps "
                f"a click from holding the model for a minute.")
        return ids

    def _tokens(self, loaded, ids: list[int]) -> list[str]:
        return [loaded.tokenizer.decode([i]) for i in ids]

    def overview(self, ckpt_id: str | None = None) -> dict:
        """What the tab needs before anything is run: which checkpoints exist, and whether a
        sparse autoencoder has been trained for any layer of the current one."""
        info = self.playground.overview()
        out = {"checkpoints": info.get("checkpoints", []),
               "loaded": info.get("loaded"), "device": info.get("device"),
               "max_tokens": MAX_TOKENS, "max_patch_tokens": MAX_PATCH_TOKENS}
        if ckpt_id:
            run = ckpt_id.split("/")[0]
            folder = (self.root or Path.cwd()) / "logs" / "interp"
            out["saes"] = sorted(
                {"layer": int(p.stem.split("layer")[-1]), "path": str(p.name)}
                for p in folder.glob(f"{run}-layer*.pt")) if folder.is_dir() else []
        return out

    # ---- the four views -------------------------------------------------------------------
    def lens(self, ckpt_id: str, prompt: str, top: int = 5) -> dict:
        loaded = self._loaded(ckpt_id)
        ids = self._ids(loaded, prompt)
        cap = capture_run(loaded.model, ids, device=loaded.device)
        rows = logit_lens(loaded.model, cap, top=top)
        story = lens_story(rows, loaded.tokenizer.decode)
        # Token *text* is added here rather than in `interp/lens.py`: that module never needs
        # a tokenizer, which is what lets a test run it without one.
        for row in story["rows"]:
            for entry in row["top"]:
                entry["text"] = loaded.tokenizer.decode([entry["id"]])
        story["tokens"] = self._tokens(loaded, ids)
        story["contributions"] = layer_contributions(loaded.model, cap)
        story["checkpoint"] = loaded.info.rel
        story["device"] = loaded.device
        return story

    def attention(self, ckpt_id: str, prompt: str, layer: int, head: int | None = None) -> dict:
        loaded = self._loaded(ckpt_id)
        n_layers = loaded.model.cfg.n_layers
        if not 0 <= layer < n_layers:
            raise InferError(f"layer {layer} is out of range (0–{n_layers - 1})")
        ids = self._ids(loaded, prompt)
        cap = capture_run(loaded.model, ids, device=loaded.device)
        weights = attention_maps(loaded.model, cap, layer)
        tokens = self._tokens(loaded, ids)
        out = {
            "checkpoint": loaded.info.rel, "layer": layer, "layers": n_layers,
            "heads": int(weights.shape[0]), "tokens": tokens,
            "summary": attention_summary(weights, tokens),
        }
        if head is not None:
            if not 0 <= head < weights.shape[0]:
                raise InferError(f"head {head} is out of range (0–{weights.shape[0] - 1})")
            out["head"] = head
            out["matrix"] = [[round(v, 4) for v in row]
                             for row in weights[head].float().tolist()]
        return out

    def patch(self, ckpt_id: str, clean: str, corrupt: str, answer: str,
              other: str) -> dict:
        loaded = self._loaded(ckpt_id)
        clean_ids = self._ids(loaded, clean, MAX_PATCH_TOKENS)
        corrupt_ids = self._ids(loaded, corrupt, MAX_PATCH_TOKENS)
        answer_ids = loaded.tokenizer.encode(answer)
        other_ids = loaded.tokenizer.encode(other)
        if not answer_ids or not other_ids:
            raise InferError("both answers must be non-empty")
        try:
            result = patch_grid(loaded.model, clean_ids, corrupt_ids, answer_ids[0],
                                other_ids[0], device=loaded.device)
        except PatchError as exc:
            raise InferError(str(exc)) from exc
        tokens = self._tokens(loaded, clean_ids)
        result.update({
            "tokens": tokens,
            "corrupt_tokens": self._tokens(loaded, corrupt_ids),
            "answer": answer, "other": other,
            "summary": summarise(result, tokens),
            "checkpoint": loaded.info.rel,
        })
        return result

    def features(self, ckpt_id: str, layer: int, limit: int = 24) -> dict:
        """The trained dictionary for one layer, if there is one.

        Only the *report* — firing rates and strengths — because finding what a feature means
        needs a corpus pass, which is a CLI job (`python -m aksharallm.interp features`)
        rather than something to do inside a click.
        """
        run = ckpt_id.split("/")[0]
        path = (self.root or Path.cwd()) / "logs" / "interp" / f"{run}-layer{layer}.pt"
        meta = path.with_suffix(".json")
        if not path.exists():
            return {"trained": False, "layer": layer, "hint":
                    f"no sparse autoencoder for layer {layer} yet — train one with "
                    f"`python -m aksharallm.interp sae {run} --layer {layer}`"}
        import json
        blob = json.loads(meta.read_text()) if meta.exists() else {}
        report = blob.get("report") or {}
        return {
            "trained": True, "layer": layer, "config": blob.get("config"),
            "history": blob.get("history", [])[-40:],
            "report": {**report, "features": (report.get("features") or [])[:limit]},
            "path": str(path.name),
        }
