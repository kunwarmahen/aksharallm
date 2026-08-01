"""Making training data with a bigger model — and checking it before believing it.

Every dataset this project has trained on so far was written by people and downloaded:
FineWeb-Edu, Python from The Stack, SmolTalk. This package is the other way of getting
data — **ask a model on this machine to write it** — and it exists for the two cases where
downloading does not work: a task nobody published a dataset for, and a base model that has
never seen an instruction and needs one to bootstrap from.

The honest name for it
----------------------
"Distilling a 31B into our 300M" is what this is usually called, and for us it would be
wrong. Classic distillation matches the teacher's *logit distribution*, which requires both
models to share a vocabulary; every local teacher (gemma4:31b, qwen3.5:27b, qwen2.5:14b,
starcoder2:3b) has its own tokenizer and ours is a 32k BPE trained on the blend. Matching
probability mass across two different tokenizations is a research problem, not a build. So
what happens here is **sequence-level distillation**: the teacher writes *text*, we tokenize
it our way, and train on it normally — which is exactly what a synthetic-data pipeline is.
True logit KD is honest only between two of our own models, and that lives in
`aksharallm/train/distil.py`.

Why this package is mostly filters
----------------------------------
Generating text is four lines. The reason this is a package is that **synthetic data is the
easiest way to make a model worse while its training loss improves**: duplicate-heavy,
low-diversity or subtly wrong data trains beautifully and produces a model that is fluent
and useless. Nothing in the loss curve can see it. So every recipe here ends in a filter,
and the Python recipe ends in the strongest filter available anywhere in this repo — the
sandbox actually *runs* the tests.

    seed grid ──▶ teacher ──▶ parse ──▶ filters ──▶ verify ──▶ dedup ──▶ samples.jsonl
                                 │         │           │          │
                              rejected, with the reason, into rejects.jsonl

`data/synth/<name>/meta.json` records the teacher, the prompt template and its version, the
sampling parameters and the pass rate at every stage, because "where did this data come
from" is a question that gets asked after the model is strange, not before.

The pieces
----------
* :mod:`~aksharallm.synth.teacher`  — the Ollama client (shared with the Code tab and the
  eval harness's judge) and the per-recipe model choice.
* :mod:`~aksharallm.synth.prompts`  — the seed grid. Diversity comes from *structurally
  different prompts*, not from a higher temperature.
* :mod:`~aksharallm.synth.recipes`  — the three recipes: python, chat, preference.
* :mod:`~aksharallm.synth.filters`  — validity and near-duplicate detection.
* :mod:`~aksharallm.synth.verify`   — run the tests, then run them again against a stubbed
  solution to prove they were not vacuous.
* :mod:`~aksharallm.synth.dataset`  — the writer, the provenance record, and the exports
  that `prepare_sft` / `prepare_dpo` consume.
* :mod:`~aksharallm.synth.run`      — the generation loop, its budget and its stop file.
"""

from .dataset import Dataset, SynthError, list_datasets
from .recipes import RECIPES, get_recipe
from .run import GenerateOptions, Stats, generate
from .teacher import SynthConfig, Teacher, contention

__all__ = ["Dataset", "GenerateOptions", "RECIPES", "Stats", "SynthConfig", "SynthError",
           "Teacher", "contention", "generate", "get_recipe", "list_datasets"]
