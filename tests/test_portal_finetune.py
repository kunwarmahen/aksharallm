"""Tests for the portal's Finetune panel and the adapter store behind it.

The panel starts a subprocess driven by whatever the browser posts, so most of these are
about the *refusals* — a missing dataset, a second concurrent job, an unknown preset — and
about the one with real consequences: quietly moving to the CPU when a pretraining run
owns the card.

Nothing here launches a real job. `start()` runs with a stubbed Popen so the command line
it would have run is asserted directly, because that command line is the entire contract
between the panel and `train/sft.py`.
"""

import json
import time
from pathlib import Path

import numpy as np
import pytest
import torch

from aksharallm.config import ModelConfig
from aksharallm.infer.checkpoints import AdapterStore, CheckpointStore
from aksharallm.lora.adapter import base_identity, save_adapter
from aksharallm.lora.inject import LoRAConfig, apply_lora
from aksharallm.model.transformer import Transformer
from aksharallm.portal.finetune import METHODS, FinetuneJobs
from aksharallm.portal.runs import RunError

CFG = ModelConfig(vocab_size=64, d_model=32, n_layers=2, n_heads=4, max_seq_len=16)


@pytest.fixture
def repo(tmp_path):
    (tmp_path / "checkpoints" / "tiny").mkdir(parents=True)
    (tmp_path / "logs").mkdir()
    (tmp_path / "data").mkdir()
    return tmp_path


def _ckpt(repo: Path, run="tiny", name="ckpt_best.pt", step=100):
    d = repo / "checkpoints" / run
    d.mkdir(parents=True, exist_ok=True)
    path = d / name
    model = Transformer(CFG)
    torch.save({
        "model": model.state_dict(),
        "model_config": dict(vars(CFG)),
        "config": {"data": {"tokenizer": "t.json"}, "train": {"seq_len": 16}},
        "step": step, "best_val": 1.5,
    }, path)
    (repo / "t.json").write_text("{}")
    return path


def _adapter(repo: Path, run="tiny", name="sft_best.lora.pt", r=8, stage="sft"):
    d = repo / "checkpoints" / run
    d.mkdir(parents=True, exist_ok=True)
    model = Transformer(CFG)
    cfg = LoRAConfig(r=r, targets="all-linear")
    apply_lora(model, cfg)
    ckpt = {"model_config": dict(vars(CFG)), "step": 100, "best_val": 1.5,
            "config": {"data": {"tokenizer": "t.json"}}}
    return save_adapter(d / name, model, cfg, base_identity(ckpt, "base.pt"),
                        extra={"training": {"stage": stage, "val_loss": 1.23}})


def _dataset(repo: Path, name="sft-synthetic", blocks=8, seq_len=16):
    d = repo / "data" / name
    d.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val"):
        n = blocks if split == "train" else 2
        np.save(d / f"{split}_tokens.npy", np.zeros((n, seq_len), dtype=np.uint16))
        np.save(d / f"{split}_mask.npy", np.zeros((n, seq_len), dtype=np.uint8))
    return d


class FakeProc:
    def __init__(self, pid=515151):
        self.pid = pid


@pytest.fixture
def spy_popen(monkeypatch):
    seen = {}

    def fake(cmd, **kw):
        seen["cmd"] = cmd
        seen["kw"] = kw
        return FakeProc()

    monkeypatch.setattr("aksharallm.portal.finetune.subprocess.Popen", fake)
    return seen


# ---- listing ---------------------------------------------------------------------------


def test_lists_checkpoints_including_quantized_ones(repo):
    """Unlike the Quantize panel, an already-quantized checkpoint is a perfectly good QLoRA
    base — better, even, since it was quantized with a real method rather than on the fly."""
    _ckpt(repo)
    _ckpt(repo, name="ckpt_best-gptq-nf4-g64.pt")
    f = FinetuneJobs(repo)
    names = {c["name"]: c for c in f.checkpoints()}
    assert len(names) == 2
    assert names["ckpt_best-gptq-nf4-g64.pt"]["quantized"]
    assert not names["ckpt_best.pt"]["quantized"]


def test_finds_prepared_datasets_by_their_file_layout(repo):
    _dataset(repo, "sft-synthetic", blocks=12, seq_len=16)
    (repo / "data" / "not-a-dataset").mkdir()
    f = FinetuneJobs(repo)
    sets = f.datasets()
    assert [s["name"] for s in sets] == ["sft-synthetic"]
    assert sets[0]["blocks"] == 12 and sets[0]["seq_len"] == 16


def test_status_carries_the_explanations_the_tab_renders(repo):
    f = FinetuneJobs(repo)
    st = f.status()
    assert "adapter" in st["why"]
    assert {m["id"] for m in st["methods"]} == set(METHODS)
    assert all(m["blurb"] for m in st["methods"])
    assert all(t["blurb"] for t in st["targets"])
    assert st["ranks"]


# ---- the budget ---------------------------------------------------------------------------


def test_budget_ranks_qlora_cheapest_and_full_dearest(repo):
    _ckpt(repo)
    b = FinetuneJobs(repo).budget("tiny/ckpt_best.pt", ranks=(4, 8))
    strategies = [r["strategy"] for r in b["rows"]]
    assert strategies[0] == "full"
    totals = {r["label"]: r["total_bytes"] for r in b["rows"]}
    assert totals["Full fine-tune"] == max(totals.values())
    assert min(totals, key=totals.get).startswith("QLoRA")
    # LoRA and QLoRA train the same number of parameters at the same rank; only the frozen
    # base differs. If that ever stopped being true the headline would be wrong.
    by = {r["label"]: r for r in b["rows"]}
    assert by["LoRA r=8"]["trainable_params"] == by["QLoRA r=8"]["trainable_params"]
    assert by["QLoRA r=8"]["weight_bytes"] < by["LoRA r=8"]["weight_bytes"]
    assert "less" in b["headline"]


def test_budget_says_activations_are_not_included(repo):
    """The one honest caveat: LoRA does not shrink activations. Leaving it out would make
    the table read as a promise it cannot keep."""
    _ckpt(repo)
    b = FinetuneJobs(repo).budget("tiny/ckpt_best.pt", ranks=(4,))
    assert "ctivations" in b["note"]


# ---- device policy -------------------------------------------------------------------------


def test_defaults_to_the_gpu_when_nothing_is_training(repo):
    plan = FinetuneJobs(repo).plan_device()
    assert plan["device"] == "cuda" and not plan["forced"]


def test_drops_to_the_cpu_while_a_run_owns_the_card(repo, monkeypatch):
    monkeypatch.setattr(FinetuneJobs, "training", lambda self: ["small-code"])
    plan = FinetuneJobs(repo).plan_device()
    assert plan["device"] == "cpu"
    assert "small-code" in plan["reason"]


def test_cuda_can_be_forced_and_the_warning_still_appears(repo, monkeypatch):
    monkeypatch.setattr(FinetuneJobs, "training", lambda self: ["small-code"])
    plan = FinetuneJobs(repo).plan_device("cuda")
    assert plan["device"] == "cuda" and plan["forced"]
    assert "small-code" in plan["reason"]


# ---- starting a job -------------------------------------------------------------------------


def test_start_builds_the_command_line_a_person_would_type(repo, spy_popen):
    _ckpt(repo)
    _dataset(repo)
    f = FinetuneJobs(repo)
    res = f.start({"checkpoint": "tiny/ckpt_best.pt", "data_dir": "data/sft-synthetic",
                   "method": "qlora", "r": 16, "targets": "attn", "epochs": 3,
                   "device": "cpu"})
    cmd = " ".join(spy_popen["cmd"])
    assert "-m aksharallm.train.sft" in cmd
    assert "--qlora" in cmd and "--qlora-double-quant" in cmd
    assert "--lora-r 16" in cmd
    assert "--lora-targets attn" in cmd
    assert "--epochs 3" in cmd
    assert "--device cpu" in cmd
    # The adapter must land beside its base so the Playground and --list-adapters find it.
    assert res["out"] == "checkpoints/tiny/sft_best.lora.pt"


def test_a_full_fine_tune_passes_no_lora_flags(repo, spy_popen):
    _ckpt(repo)
    _dataset(repo)
    res = FinetuneJobs(repo).start({
        "checkpoint": "tiny/ckpt_best.pt", "data_dir": "data/sft-synthetic",
        "method": "full"})
    cmd = " ".join(spy_popen["cmd"])
    assert "--lora" not in cmd and "--qlora" not in cmd
    assert res["out"].endswith("sft_best.pt")
    assert not res["out"].endswith(".lora.pt")


def test_plain_lora_does_not_quantize_the_base(repo, spy_popen):
    _ckpt(repo)
    _dataset(repo)
    FinetuneJobs(repo).start({"checkpoint": "tiny/ckpt_best.pt",
                              "data_dir": "data/sft-synthetic", "method": "lora"})
    cmd = " ".join(spy_popen["cmd"])
    assert "--lora" in cmd and "--qlora" not in cmd


def test_refuses_a_dataset_that_is_not_prepared(repo, spy_popen):
    _ckpt(repo)
    with pytest.raises(RunError, match="not a prepared SFT dataset"):
        FinetuneJobs(repo).start({"checkpoint": "tiny/ckpt_best.pt",
                                  "data_dir": "data/nope", "method": "qlora"})


def test_refuses_with_no_dataset_chosen(repo):
    _ckpt(repo)
    with pytest.raises(RunError, match="prepared SFT dataset"):
        FinetuneJobs(repo).start({"checkpoint": "tiny/ckpt_best.pt", "method": "qlora"})


def test_refuses_an_unknown_method_or_preset(repo):
    _ckpt(repo)
    _dataset(repo)
    f = FinetuneJobs(repo)
    base = {"checkpoint": "tiny/ckpt_best.pt", "data_dir": "data/sft-synthetic"}
    with pytest.raises(RunError, match="unknown method"):
        f.start({**base, "method": "magic"})
    with pytest.raises(RunError, match="target preset"):
        f.start({**base, "method": "qlora", "targets": "everything"})


def test_refuses_a_second_job_while_one_is_running(repo, spy_popen, monkeypatch):
    _ckpt(repo)
    _dataset(repo)
    f = FinetuneJobs(repo)
    spec = {"checkpoint": "tiny/ckpt_best.pt", "data_dir": "data/sft-synthetic",
            "method": "qlora"}
    f.start(spec)
    monkeypatch.setattr(FinetuneJobs, "_pid", lambda self: 515151)
    with pytest.raises(RunError, match="already running"):
        f.start(spec)


def test_refuses_a_checkpoint_with_no_tokenizer(repo, spy_popen):
    """An adapter trained on a checkpoint whose tokenizer is unknown could never be decoded
    safely, so the refusal belongs here rather than three steps later."""
    _dataset(repo)
    d = repo / "checkpoints" / "tiny"
    d.mkdir(parents=True, exist_ok=True)
    torch.save({"model": Transformer(CFG).state_dict(),
                "model_config": dict(vars(CFG)), "config": {}, "step": 1},
               d / "ckpt_best.pt")
    with pytest.raises(RunError, match="tokenizer"):
        FinetuneJobs(repo).start({"checkpoint": "tiny/ckpt_best.pt",
                                  "data_dir": "data/sft-synthetic", "method": "qlora"})


def test_epochs_are_clamped(repo, spy_popen):
    _ckpt(repo)
    _dataset(repo)
    FinetuneJobs(repo).start({"checkpoint": "tiny/ckpt_best.pt",
                              "data_dir": "data/sft-synthetic", "method": "qlora",
                              "epochs": 9999})
    assert "--epochs 20" in " ".join(spy_popen["cmd"])


def test_stop_refuses_when_nothing_is_running(repo):
    with pytest.raises(RunError, match="no fine-tuning job"):
        FinetuneJobs(repo).stop()


def test_the_stop_file_is_never_the_one_a_pretraining_run_reads(repo, spy_popen):
    """A fine-tune writes its adapter into the base model's run directory. A file called
    STOP in there is the *pretrainer's* — one fine-tune ending a six-day run is not a
    mistake worth leaving available, so the job's stop file lives under logs/finetune/."""
    _ckpt(repo)
    _dataset(repo)
    f = FinetuneJobs(repo)
    f.start({"checkpoint": "tiny/ckpt_best.pt", "data_dir": "data/sft-synthetic"})
    cmd = " ".join(spy_popen["cmd"])
    assert "--stop-file" in cmd
    assert str(f.stop_file) in cmd
    assert f.stop_file.parent != repo / "checkpoints" / "tiny"


def test_a_bounded_stop_writes_the_file_the_trainer_polls(repo, monkeypatch):
    f = FinetuneJobs(repo)
    f.dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(f, "_pid", lambda: 424242)

    f.stop("at", steps=300)
    assert f.stop_file.read_text() == "300"
    assert f.stop_request()["target"] == 300

    f.stop("in", seconds=600)
    assert f.stop_file.read_text().startswith("@")
    assert abs(f.stop_request()["deadline"] - (time.time() + 600)) < 2

    f.stop("cancel")
    assert not f.stop_file.exists()
    with pytest.raises(RunError, match="no stop is queued"):
        f.stop("cancel")


def test_a_stale_stop_file_cannot_end_the_next_fine_tune_at_step_zero(repo, spy_popen):
    _ckpt(repo)
    _dataset(repo)
    f = FinetuneJobs(repo)
    f.dir.mkdir(parents=True, exist_ok=True)
    f.stop_file.write_text("50")
    f.start({"checkpoint": "tiny/ckpt_best.pt", "data_dir": "data/sft-synthetic"})
    assert not f.stop_file.exists()


# ---- the adapter store ----------------------------------------------------------------------


def test_adapters_are_not_listed_as_checkpoints(repo):
    """They live in the same directory and also end in `.pt`. Listing one as a model would
    put an entry in every picker that cannot load."""
    _ckpt(repo)
    _adapter(repo)
    assert [c.name for c in CheckpointStore(repo).list()] == ["ckpt_best.pt"]
    assert [a.name for a in AdapterStore(repo).list()] == ["sft_best.lora.pt"]


def test_an_adapter_is_described_without_loading_a_model(repo):
    _adapter(repo, r=16, stage="dpo")
    a = AdapterStore(repo).list()[0]
    assert a.r == 16
    assert a.targets == "all-linear"
    assert a.stage == "dpo"        # what it was *trained* for, not the filename prefix
    assert a.val_loss == 1.23
    assert a.arch and "d=32" in a.arch
    assert a.error is None


def test_the_adapter_arch_string_matches_its_checkpoint(repo):
    """The Playground filters adapters by this string, so the two must be built the same
    way — otherwise every adapter is hidden, or every incompatible one is offered."""
    _ckpt(repo)
    _adapter(repo)
    ck = CheckpointStore(repo).list()[0]
    ad = AdapterStore(repo).list()[0]
    assert ad.arch == ck.arch


def test_a_corrupt_adapter_is_listed_with_its_reason(repo):
    d = repo / "checkpoints" / "tiny"
    d.mkdir(parents=True, exist_ok=True)
    (d / "broken.lora.pt").write_bytes(b"not a torch file")
    a = AdapterStore(repo).list()[0]
    assert a.error and a.r is None


def test_resolving_rejects_a_name_that_is_not_an_adapter(repo):
    from aksharallm.infer.checkpoints import InferError

    _ckpt(repo)
    with pytest.raises(InferError, match="not an adapter name"):
        AdapterStore(repo).resolve("tiny/ckpt_best.pt")


def test_resolving_cannot_escape_the_checkpoints_directory(repo):
    from aksharallm.infer.checkpoints import InferError

    with pytest.raises(InferError):
        AdapterStore(repo).resolve("../../etc/passwd.lora.pt")
