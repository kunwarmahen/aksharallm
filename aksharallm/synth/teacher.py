"""The model that writes the data, and how much of the machine it is allowed to have.

One Ollama client serves three features now — the Code tab's explainer, the eval harness's
judge, and this. They differ only in which section of `configs/portal.yaml` they read, so
:class:`SynthConfig` is :class:`~aksharallm.portal.explain.ExplainConfig` with
``SECTION = "synth"``. That is not a saving of forty lines; it means a fix to the "is Ollama
even running" error message is a fix everywhere, and there is exactly one place where the
`think: false` trap is handled.

Two things here are specific to generating data rather than reading it.

**The teacher is per recipe.** `starcoder2:3b` writes a plausible Python function in two
seconds and cannot hold a conversation; `gemma4:31b` writes good instruction data and takes
half a minute a sample. Quality-per-hour differs between them by an order of magnitude *in
opposite directions depending on the recipe*, so a single global `model:` would be wrong for
at least one recipe at all times. `synth.recipes.<name>.model` overrides `synth.model`.

**Temperature is high, and that is deliberate.** Everywhere else in this project sampling is
conservative (the judge is at 0.0, the explainer at 0.2) because the answer should be the
same each time. Here the *opposite* is wanted: 5,000 samples that are all the teacher's
single most likely answer is a dataset with 5,000 rows and almost no information in it. The
real diversity comes from the seed grid in :mod:`~aksharallm.synth.prompts` — temperature
alone mostly produces the same answer with different adjectives — but a low temperature
would undo the grid's work.
"""

from __future__ import annotations

import copy
import os
import time
from dataclasses import dataclass
from pathlib import Path

import yaml

from ..portal.explain import ExplainConfig, Ollama
from ..portal.runs import RunError

#: What each recipe wants, when nothing is configured. A code model for code, the biggest
#: model on the machine for anything a person would read.
DEFAULT_MODELS = {
    "python": "qwen2.5:14b",
    "chat": "gemma4:31b",
    "preference": "gemma4:31b",
}

#: Roughly what each teacher parks on the card, for the contention warning. Ollama reports
#: the real number in `ollama ps`; this is only used to say "this will not fit beside a
#: training run", which does not need to be exact.
MODEL_VRAM_GB = {"gemma4:31b": 19.0, "gemma4:26b": 17.0, "qwen3.5:27b": 17.0,
                 "gemma4:12b": 8.0, "gemma4:e4b": 9.6, "qwen2.5:14b": 9.0,
                 "starcoder2:3b": 1.7}


class SynthConfig(ExplainConfig):
    """`synth:` in `configs/portal.yaml`, plus the per-recipe overrides.

    Everything :class:`ExplainConfig` reads is read the same way here. What this adds is the
    `recipes:` sub-mapping, which the parent knows nothing about, so :meth:`reload` re-opens
    the file for it rather than teaching the parent about a shape only this caller has.
    """

    SECTION = "synth"
    ENV_PREFIX = "AKSHARALLM_SYNTH"

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.model = "qwen2.5:14b"
        # High, on purpose — see the module docstring.
        self.temperature = 0.9
        self.num_predict = 1200
        self.num_ctx = 8192
        self.timeout_s = 600.0
        self.think = False
        self.keep_alive = "10m"
        #: recipe -> {"model": ..., "temperature": ...}
        self.recipes: dict[str, dict] = {}
        #: How many times a single seed may be re-asked before it is abandoned. A teacher
        #: that ignores the output format twice will ignore it a third time; the budget is
        #: better spent on the next seed.
        self.max_attempts = 2

    def reload(self) -> "SynthConfig":
        super().reload()
        data: dict = {}
        if self.path and self.path.is_file():
            try:
                loaded = yaml.safe_load(self.path.read_text()) or {}
                data = (loaded.get(self.SECTION) or {}) if isinstance(loaded, dict) else {}
            except (OSError, yaml.YAMLError):
                data = {}                      # the parent already recorded the note
        recipes = data.get("recipes")
        self.recipes = recipes if isinstance(recipes, dict) else {}
        # Whether `model:` was actually written down, rather than left at the class default.
        # It decides one precedence question: a configured section model beats the built-in
        # per-recipe default, and an unconfigured one does not (see `model_for`).
        self.model_explicit = bool(data.get("model")) or \
            bool(os.environ.get(f"{self.ENV_PREFIX}_MODEL"))
        if data.get("max_attempts") is not None:
            try:
                self.max_attempts = max(1, int(data["max_attempts"]))
            except (TypeError, ValueError):
                pass
        return self

    # ---- per-recipe view ----------------------------------------------------------------
    def model_for(self, recipe: str) -> str:
        """Which model writes this recipe's data.

        Precedence, most specific first:

        1. `AKSHARALLM_SYNTH_MODEL_PYTHON` — one recipe, one session, no file edit;
        2. `synth.recipes.<recipe>.model` — the per-recipe choice in the config;
        3. `synth.model` / `AKSHARALLM_SYNTH_MODEL`, **if it was written down**;
        4. the built-in default for this recipe.

        Order 3 before 4 is the whole point of tracking `model_explicit`: somebody who sets
        one model for the section means it for every recipe, and silently overriding them
        with "but chat wants a bigger model" would be the config lying. Left unset, though,
        the built-in per-recipe defaults must win over the class default — "a good model for
        code" and "a good model for chat" are different models.
        """
        env = os.environ.get(f"{self.ENV_PREFIX}_MODEL_{recipe.upper()}")
        if env:
            return env
        entry = self.recipes.get(recipe) or {}
        if isinstance(entry, dict) and entry.get("model"):
            return str(entry["model"])
        if isinstance(entry, str):
            return entry
        if getattr(self, "model_explicit", False):
            return self.model
        return DEFAULT_MODELS.get(recipe, self.model)

    def temperature_for(self, recipe: str) -> float:
        entry = self.recipes.get(recipe) or {}
        if isinstance(entry, dict) and entry.get("temperature") is not None:
            try:
                return float(entry["temperature"])
            except (TypeError, ValueError):
                pass
        return self.temperature

    @classmethod
    def load(cls, root: Path | None = None) -> "SynthConfig":
        return super().load(root)          # typed for the caller; behaviour is the parent's


@dataclass
class Reply:
    """One answer from the teacher, with what it cost."""

    text: str
    thinking: str = ""
    duration_s: float = 0.0
    model: str = ""


class Teacher:
    """A model that answers a prompt with a whole string.

    The shared client streams, because the Code tab renders tokens as they arrive. Nothing
    here watches: a sample is worthless until it is complete, and half a parsed function is
    not half a sample. So this collects the stream and hands back the text.

    Recipes only ever call :meth:`ask`, which is why the tests can pass a scripted stand-in
    and exercise every filter, every parser and the whole loop without Ollama running.
    """

    def __init__(self, cfg: SynthConfig, model: str | None = None,
                 temperature: float | None = None):
        self.cfg = cfg
        self.name = model or cfg.model
        self.temperature = cfg.temperature if temperature is None else temperature

    def ask(self, messages: list[dict]) -> Reply:
        # A copy, because two recipes with different temperatures may share one config
        # object and the client reads `cfg.temperature` at request time.
        cfg = copy.copy(self.cfg)
        cfg.temperature = self.temperature
        client = Ollama(cfg)
        t0 = time.monotonic()
        text: list[str] = []
        thinking: list[str] = []
        stream = client.chat(messages, model=self.name)
        try:
            for kind, piece in stream:
                (thinking if kind == "thinking" else text).append(piece)
        finally:
            stream.close()
        return Reply(text="".join(text), thinking="".join(thinking),
                     duration_s=time.monotonic() - t0, model=self.name)

    # ---- pre-flight ---------------------------------------------------------------------
    def available(self) -> tuple[bool, str]:
        """Is Ollama up and is this model pulled?

        Asked *before* the loop starts. A generation run is minutes to hours; finding out at
        the end that the model name had a typo in it is the kind of mistake that only has to
        happen once to be worth eight lines.
        """
        try:
            models = Ollama(self.cfg).models()
        except RunError as exc:
            return False, str(exc)
        names = {m["name"] for m in models}
        if self.name in names:
            return True, f"generating with {self.name} at {self.cfg.host}"
        # `qwen2.5:14b` vs `qwen2.5:14b-instruct-q4_K_M`: same model, tagged differently.
        stem = self.name.split(":")[0]
        if stem in {n.split(":")[0] for n in names}:
            return True, (f"'{self.name}' is not pulled exactly, but a {stem} tag is — "
                          "using the configured name; Ollama will resolve or fail loudly.")
        return False, (f"the teacher '{self.name}' is not pulled. Available: "
                       f"{', '.join(sorted(names)) or 'none'}.  "
                       f"Pull it with:  ollama pull {self.name}")


def contention(root: Path | None = None, model: str | None = None) -> dict:
    """Whether generating right now would take the card away from a training run.

    This one cannot be solved the way the Playground and the quantize panel solve it. Those
    load *our* model and can simply choose the CPU; here the model is loaded by Ollama, in
    another process, and the only lever is which model is asked for and `num_gpu`. So this
    reports rather than decides — the caller (the CLI's warning line, the portal's panel)
    states the trade and the human picks a smaller teacher or waits.

    A 31B teacher against a Phase-2 run's ~21 GB of 24 is not a slow tab, it is the run
    dying overnight; `starcoder2:3b` at 1.7 GB genuinely fits beside it.
    """
    from ..infer.checkpoints import CheckpointStore
    from ..portal.runs import _alive, _read_int, repo_root

    root = Path(root) if root else repo_root()
    training = []
    for run in CheckpointStore(root).dirs():
        pid = _read_int(run / "train.pid")
        if pid and _alive(pid):
            training.append(run.name)

    need = MODEL_VRAM_GB.get(model or "", None)
    if not training:
        return {"training": [], "model": model, "vram_gb": need, "safe": True,
                "reason": "nothing is training — the card is free for the teacher."}
    who = ", ".join(training)
    if need is not None and need <= 3.0:
        return {"training": training, "model": model, "vram_gb": need, "safe": True,
                "reason": f"{who} is training, but {model} is about {need:.1f} GB and fits "
                          "in what a run leaves free. Generation will be slower than usual."}
    size = f"about {need:.0f} GB" if need else "several GB"
    return {"training": training, "model": model, "vram_gb": need, "safe": False,
            "reason": f"{who} is training and {model or 'this teacher'} wants {size} of "
                      "VRAM. A run holds ~21 GB of the 24 — loading this teacher can kill "
                      "it. Use starcoder2:3b, set synth.num_gpu: 0 to keep the teacher on "
                      "the CPU, or generate when the run is stopped."}
