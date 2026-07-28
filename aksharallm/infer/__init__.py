"""Talking to a trained checkpoint.

`generate` is the KV-cached sampling loop. Everything else here exists to answer the
question you have the moment a run has produced its first checkpoint: *is this thing
learning anything?*

    checkpoints  which trained models exist, and what each one is (step, loss, stage)
    engine       one of them loaded and kept warm, on a device chosen not to kill the run
    tasks        the fixed things to ask: prose probes, chat prompts, Python tasks
    sandbox      running the Python it wrote, under limits, to see if it actually works
    history      every generation, stamped with the training state of the model that made it
    playground   all of the above in the order the CLI and the portal both use
    cli          `python -m aksharallm.infer.cli` — the terminal front end

Imports are deliberately lazy-ish: `generate` and `checkpoints` are the only things
`aksharallm.eval` needs, and nothing here should pull in the portal.
"""

from .checkpoints import Checkpoint, CheckpointStore, InferError
from .engine import Engine, InferConfig, SamplingParams, plan_device, training_runs
from .generate import IncrementalDecoder, generate, stream_generate
from .history import History
from .playground import Playground

__all__ = ["Checkpoint", "CheckpointStore", "Engine", "History", "InferConfig",
           "InferError", "IncrementalDecoder", "Playground", "SamplingParams",
           "generate", "plan_device", "stream_generate", "training_runs"]
