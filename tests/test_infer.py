"""Tests for the playground: checkpoint discovery, the device policy, the code sandbox,
the history log, and the portal routes in front of them.

Nothing here needs a GPU or a real 300M checkpoint. A two-layer model with a 64-token
vocabulary is saved into a temporary repo in exactly the shape `save_checkpoint` writes,
which is enough to exercise everything that matters — and the parts that *can't* be faked
(is a base model refused a chat turn? does a `while True:` get killed?) are the parts these
tests care most about.

The sandbox tests really do execute Python in a subprocess. That is the point of them.
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request

import pytest
import torch

from aksharallm.config import ModelConfig
from aksharallm.infer import sandbox, tasks
from aksharallm.infer.checkpoints import CheckpointStore, InferError, stage_for
from aksharallm.infer.engine import Engine, InferConfig, SamplingParams, plan_device
from aksharallm.infer.generate import IncrementalDecoder, fit_prompt
from aksharallm.infer.history import History, record_from
from aksharallm.infer.playground import Playground
from aksharallm.model.transformer import Transformer
from aksharallm.portal.server import serve
from aksharallm.tokenizer.tokenizer import train_bpe

MODEL_CFG = dict(vocab_size=300, d_model=32, n_layers=2, n_heads=4, n_kv_heads=2,
                 max_seq_len=64, tie_embeddings=True)


@pytest.fixture
def repo(tmp_path):
    """A miniature repo: a real tokenizer, and checkpoints in the trainer's exact format."""
    tok_path = tmp_path / "data" / "tok.json"
    tok_path.parent.mkdir(parents=True)
    train_bpe(iter(["hello world this is a small corpus for a small tokenizer. "
                    "def add(a, b): return a + b\n"] * 40),
              vocab_size=300, out_path=tok_path)

    model = Transformer(ModelConfig(**MODEL_CFG))
    payload = {
        "model": model.state_dict(),
        "optimizer": {},
        "model_config": MODEL_CFG,
        "config": {"name": "demo",
                   "data": {"tokenizer": "data/tok.json"},
                   "train": {"batch_size": 2, "grad_accum": 3, "seq_len": 64,
                             "max_steps": 1000, "out_dir": "checkpoints/demo"}},
        "step": 500,
        "best_val": 3.25,
    }
    ckpt_dir = tmp_path / "checkpoints" / "demo"
    ckpt_dir.mkdir(parents=True)
    torch.save(payload, ckpt_dir / "ckpt_last.pt")
    # An SFT checkpoint too, so the chat/base distinction can be tested.
    torch.save({**payload, "step": 40, "best_val": 2.1}, ckpt_dir / "sft_last.pt")
    (ckpt_dir / "train_log.jsonl").write_text(
        '{"step": 100, "loss": 4.0, "ema": 4.1}\n'
        '{"step": 500, "loss": 3.3, "ema": 3.4}\n'
        '{"step": 900, "loss": 3.0, "ema": 3.05}\n')   # after the checkpoint's step
    return tmp_path


@pytest.fixture
def pg(repo):
    p = Playground(repo, busy_cb=lambda: [])
    yield p
    p.close()


# ---------------------------------------------------------------- discovery -------------

def test_stage_is_read_from_the_filename():
    assert stage_for("ckpt_best.pt") == "base"
    assert stage_for("sft_last.pt") == "sft"
    assert stage_for("dpo_best.pt") == "dpo"
    assert stage_for("something.pt") == "unknown"


def test_checkpoints_are_described_without_loading_weights(repo):
    store = CheckpointStore(repo)
    found = {c.name: c for c in store.list()}
    assert set(found) == {"ckpt_last.pt", "sft_last.pt"}
    last = found["ckpt_last.pt"]
    assert last.step == 500
    assert last.best_val == pytest.approx(3.25)
    assert last.max_steps == 1000
    assert last.tokens_per_step == 2 * 3 * 64
    assert last.tokens_seen == 2 * 3 * 64 * 501
    assert last.params > 0
    assert last.tokenizer_ok is True
    assert last.error is None


def test_train_loss_is_the_ema_at_this_step_not_the_latest(repo):
    """`best_val` is the run's all-time best; the ema at the checkpoint's own step is what
    actually describes the file. A record from step 900 must not be attributed to it."""
    last = CheckpointStore(repo).get("demo/ckpt_last.pt")
    assert last.train_loss == pytest.approx(3.4)


def test_a_corrupt_checkpoint_is_listed_with_its_reason(repo):
    (repo / "checkpoints" / "demo" / "ckpt_broken.pt").write_bytes(b"not a checkpoint")
    broken = [c for c in CheckpointStore(repo).list() if c.name == "ckpt_broken.pt"]
    assert broken and broken[0].error
    # …and is not offered as the default.
    assert CheckpointStore(repo).default().name != "ckpt_broken.pt"


def test_the_cache_notices_a_rewritten_checkpoint(repo):
    """`ckpt_last.pt` is overwritten every few hundred steps by a live run. Testing during
    a run is the whole point, so a stale cached step would defeat the feature."""
    store = CheckpointStore(repo)
    assert store.get("demo/ckpt_last.pt").step == 500
    path = repo / "checkpoints" / "demo" / "ckpt_last.pt"
    payload = torch.load(path, weights_only=False)
    payload["step"] = 900
    torch.save(payload, path)
    assert store.get("demo/ckpt_last.pt").step == 900


@pytest.mark.parametrize("bad", ["../../etc/passwd", "demo/../../x.pt", "demo/pas swd.pt",
                                 "/etc/shadow", "demo/nope.txt"])
def test_paths_from_the_browser_cannot_escape_checkpoints(repo, bad):
    store = CheckpointStore(repo)
    run, _, name = bad.partition("/")
    with pytest.raises(InferError):
        store.resolve(run, name)


def test_identify_accepts_a_run_a_path_or_an_id(repo):
    store = CheckpointStore(repo)
    assert store.identify("demo") == "demo/ckpt_last.pt"        # no ckpt_best here
    assert store.identify("demo/sft_last.pt") == "demo/sft_last.pt"
    full = repo / "checkpoints" / "demo" / "ckpt_last.pt"
    assert store.identify(str(full)) == "demo/ckpt_last.pt"
    with pytest.raises(InferError):
        store.identify("nosuchrun")


# ---------------------------------------------------------------- device policy ---------

def test_auto_moves_to_the_cpu_while_a_run_is_training():
    """The decision that protects a six-day run. It must not depend on how much VRAM
    happens to be free at the moment the tab is opened."""
    plan = plan_device(InferConfig(device="auto"), training=["small-code"])
    assert plan.device == "cpu"
    assert "small-code" in plan.reason
    assert plan.slow is True


def test_cuda_can_be_forced_but_is_still_told_about_the_run():
    plan = plan_device(InferConfig(device="cuda"), training=["small-code"])
    assert plan.forced is True
    assert plan.training == ["small-code"]
    # Whether it actually gets CUDA depends on the machine; the *reason* must say why.
    assert "cuda" in plan.reason.lower() or "no CUDA" in plan.reason


def test_cpu_is_honoured_absolutely():
    plan = plan_device(InferConfig(device="cpu"), training=[])
    assert plan.device == "cpu" and plan.forced is True


def test_an_unknown_device_falls_back_to_auto_with_a_note(tmp_path):
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "portal.yaml").write_text("infer:\n  device: tpu\n")
    cfg = InferConfig.load(tmp_path)
    assert cfg.device == "auto" and "tpu" in cfg.note


def test_config_reloads_when_the_file_changes(tmp_path):
    (tmp_path / "configs").mkdir()
    path = tmp_path / "configs" / "portal.yaml"
    path.write_text("infer:\n  device: cpu\n  sandbox_timeout_s: 3\n")
    cfg = InferConfig.load(tmp_path)
    assert cfg.device == "cpu" and cfg.sandbox_timeout_s == 3
    path.write_text("infer:\n  device: auto\n  sandbox_timeout_s: 9\n")
    assert cfg.reload_if_changed().device == "auto"
    assert cfg.sandbox_timeout_s == 9


def test_a_broken_config_degrades_to_defaults_rather_than_failing(tmp_path):
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "portal.yaml").write_text("infer: [this is not a mapping\n")
    cfg = InferConfig.load(tmp_path)
    assert cfg.device == "auto"
    assert cfg.note and "could not be read" in cfg.note


# ---------------------------------------------------------------- generation ------------

def test_generation_streams_and_records(pg):
    stats = pg.generate(ckpt_id="demo/ckpt_last.pt", mode="complete", prompt="hello",
                        probe="fluency",
                        params=SamplingParams(max_new_tokens=6, temperature=0.0))
    assert stats["tokens"] <= 6
    assert stats["device"] == "cpu" or stats["device"] == "cuda"
    assert stats["provenance"]["step"] == 500
    assert stats["record_id"]
    row = pg.history.get(stats["record_id"])
    assert row["probe"] == "fluency" and row["step"] == 500
    assert row["best_val"] == pytest.approx(3.25)


def test_chat_is_refused_on_a_base_model_before_the_stream_starts(pg):
    """The refusal must be raised by `stream()` itself, not from inside the generator —
    the portal turns it into a 409 the page can show in place of an answer."""
    with pytest.raises(InferError, match="base model"):
        pg.stream(ckpt_id="demo/ckpt_last.pt", mode="chat", prompt="hi")


def test_chat_is_allowed_on_an_sft_checkpoint(pg):
    stats = pg.generate(ckpt_id="demo/sft_last.pt", mode="chat", prompt="hi",
                        params=SamplingParams(max_new_tokens=4, temperature=0.0))
    assert "<|im_start|>user" in stats["rendered"] if "rendered" in stats else True
    assert stats["tokens"] <= 4


def test_a_prompt_longer_than_the_context_is_trimmed_not_rejected(pg):
    stats = pg.generate(ckpt_id="demo/ckpt_last.pt", mode="complete",
                        prompt="word " * 400,
                        params=SamplingParams(max_new_tokens=4, temperature=0.0))
    assert stats["truncated_tokens"] > 0
    assert stats["prompt_tokens"] <= MODEL_CFG["max_seq_len"]


def test_switching_checkpoints_swaps_the_resident_model(pg):
    pg.engine.load("demo/ckpt_last.pt")
    assert pg.engine.status()["loaded"]["name"] == "ckpt_last.pt"
    pg.engine.load("demo/sft_last.pt")
    assert pg.engine.status()["loaded"]["name"] == "sft_last.pt"
    assert pg.engine.unload() is True
    assert pg.engine.status()["loaded"] is None


def test_a_missing_tokenizer_refuses_rather_than_guessing(repo):
    """Decoding with the wrong tokenizer produces fluent nonsense and no error, which is a
    genuinely horrible thing to debug. Refusing is the only safe answer."""
    (repo / "data" / "tok.json").unlink()
    engine = Engine(repo, busy_cb=lambda: [])
    with pytest.raises(InferError, match="tokenizer"):
        engine.load("demo/ckpt_last.pt")
    engine.close()


def test_fit_prompt_keeps_the_end_and_leaves_room(pg):
    idx = fit_prompt(list(range(100)), 64, device="cpu")
    assert idx.size(1) == 63           # room for at least one new token
    assert idx[0, -1].item() == 99     # the newest context survives


def test_incremental_decoder_holds_back_partial_characters(repo):
    from aksharallm.tokenizer.tokenizer import Tokenizer
    tok = Tokenizer(repo / "data" / "tok.json")
    ids = tok.encode("hello world")
    dec = IncrementalDecoder(tok)
    out = "".join(dec.push(i) for i in ids) + dec.flush()
    assert out == tok.decode(ids)


# ---------------------------------------------------------------- the sandbox -----------

def test_correct_code_passes():
    task = tasks.TASKS_BY_ID["add"]
    r = sandbox.run_task(task, "    return a + b\n", timeout_s=5)
    assert r.ok and r.status == "pass"


def test_wrong_code_fails_with_the_assertion():
    r = sandbox.run_task(tasks.TASKS_BY_ID["add"], "    return a - b\n", timeout_s=5)
    assert not r.ok and r.status == "fail"


def test_unparseable_code_is_caught_before_it_is_run():
    r = sandbox.run_task(tasks.TASKS_BY_ID["add"], "    return a +\n", timeout_s=5)
    assert r.status == "syntax"


def test_an_infinite_loop_is_killed():
    r = sandbox.run_task(tasks.TASKS_BY_ID["add"], "    while True:\n        pass\n",
                         timeout_s=3)
    assert not r.ok and r.status == "timeout"


def test_a_memory_bomb_hits_its_limit_rather_than_the_machine():
    r = sandbox.run_task(tasks.TASKS_BY_ID["add"],
                         "    x = [0] * (10 ** 10)\n    return a + b\n",
                         timeout_s=5, memory_mb=256)
    assert not r.ok and r.status in ("error", "timeout")


def test_the_sandbox_cannot_import_this_project():
    """`-I` isolated mode: no PYTHONPATH, no cwd on sys.path. Generated code has nothing of
    ours within reach."""
    r = sandbox.run_task(tasks.TASKS_BY_ID["add"],
                         "    import aksharallm\n    return a + b\n", timeout_s=5)
    assert not r.ok
    assert "No module named" in (r.stderr + r.detail)


def test_execution_can_be_turned_off_entirely():
    r = sandbox.run_task(tasks.TASKS_BY_ID["add"], "    return a + b\n", enabled=False)
    assert r.status == "disabled" and not r.ok


# ---------------------------------------------------------------- extraction ------------

def test_a_base_completion_is_trimmed_at_the_next_top_level_statement():
    """A base model continues the *file*, not the function. Everything from the next line
    at column zero onwards belongs to whatever it wrote next."""
    body = tasks.extract_code("    return a + b\n\ndef something_else(x):\n    return x",
                              prompt="def add(a, b):\n", entry_point="add")
    assert "something_else" not in body
    assert "return a + b" in body


def test_a_fenced_block_from_a_chat_model_wins():
    code = tasks.extract_code("Sure!\n\n```python\ndef add(a, b):\n    return a + b\n```\n")
    assert code.startswith("def add")
    assert "Sure!" not in code


def test_assemble_prepends_the_signature_for_a_bare_body():
    program = tasks.assemble(tasks.TASKS_BY_ID["add"], "    return a + b\n")
    assert program.startswith("def add(a, b):")
    assert "assert add(1, 2) == 3" in program


def test_assemble_does_not_double_the_signature_when_the_model_restated_it():
    program = tasks.assemble(tasks.TASKS_BY_ID["add"],
                             "def add(a, b):\n    return a + b\n")
    assert program.count("def add(") == 1


# ---------------------------------------------------------------- history ---------------

def test_history_round_trips_and_keeps_provenance(tmp_path):
    h = History(tmp_path, max_records=100)
    stats = {"tokens": 5, "device": "cpu", "params": {}, "text": "out",
             "provenance": {"run": "demo", "checkpoint": "ckpt_last.pt", "step": 500,
                            "best_val": 3.25, "train_loss": 3.4, "tokens_seen": 1000}}
    rec = h.append(record_from(stats, mode="complete", prompt="p", output="out",
                               probe="fluency"))
    back = h.get(rec["id"])
    assert back["step"] == 500 and back["best_val"] == 3.25 and back["probe"] == "fluency"


def test_compare_orders_by_step_so_progress_reads_top_to_bottom(tmp_path):
    h = History(tmp_path, max_records=100)
    for step in (9000, 1000, 5000):
        h.append({"probe": "fluency", "run": "demo", "step": step, "output": f"at {step}"})
    rows = h.compare("fluency")["rows"]
    assert [r["step"] for r in rows] == [1000, 5000, 9000]


def test_history_is_trimmed_so_it_cannot_grow_without_limit(tmp_path):
    h = History(tmp_path, max_records=50)
    for i in range(200):
        h.append({"probe": "p", "step": i, "output": "x"})
    assert len(h.load()) <= 63          # max_records * 1.25
    assert h.load()[-1]["step"] == 199  # the newest survive, not the oldest


def test_a_truncated_final_line_is_skipped(tmp_path):
    h = History(tmp_path, max_records=10)
    h.append({"probe": "p", "step": 1, "output": "ok"})
    with open(h.path, "a") as fh:
        fh.write('{"probe": "p", "step":')     # a kill -9 mid-write
    assert len(h.load()) == 1


# ---------------------------------------------------------------- portal routes ---------

@pytest.fixture
def server(repo):
    httpd = serve(repo, port=0)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_port}"
    httpd.shutdown()
    httpd.playground.close()
    httpd.server_close()


def get(base, path):
    with urllib.request.urlopen(base + path) as resp:
        return json.loads(resp.read())


def post(base, path, body, guard=True):
    headers = {"Content-Type": "application/json"}
    if guard:
        headers["X-Portal"] = "1"
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(),
                                 headers=headers)
    return urllib.request.urlopen(req)


def test_overview_lists_checkpoints_probes_and_tasks(server):
    data = get(server, "/api/infer")
    assert {c["rel"] for c in data["checkpoints"]} == {"demo/ckpt_last.pt",
                                                      "demo/sft_last.pt"}
    assert data["probes"] and data["tasks"]
    assert data["sandbox"]["available"] is True


def test_generate_streams_server_sent_events(server):
    resp = post(server, "/api/infer/generate",
                {"checkpoint": "demo/ckpt_last.pt", "mode": "complete", "prompt": "hi",
                 "sampling": {"max_new_tokens": 4, "temperature": 0}})
    kinds = []
    for raw in resp:
        line = raw.decode().strip()
        if line.startswith("data: "):
            kinds.append(next(iter(json.loads(line[6:]))))
    assert "start" in kinds and "done" in kinds


def test_the_start_event_carries_the_training_state(server):
    resp = post(server, "/api/infer/generate",
                {"checkpoint": "demo/ckpt_last.pt", "mode": "complete", "prompt": "hi",
                 "sampling": {"max_new_tokens": 2, "temperature": 0}})
    start = None
    for raw in resp:
        line = raw.decode().strip()
        if line.startswith("data: "):
            evt = json.loads(line[6:])
            if "start" in evt:
                start = evt["start"]
                break
    assert start["checkpoint"]["step"] == 500
    assert start["checkpoint"]["best_val"] == pytest.approx(3.25)


def test_a_refusal_arrives_as_a_4xx_not_inside_the_stream(server):
    """Checked before the SSE response is committed, so the page can show it in place."""
    with pytest.raises(urllib.error.HTTPError) as exc:
        post(server, "/api/infer/generate",
             {"checkpoint": "demo/ckpt_last.pt", "mode": "chat", "prompt": "hi"})
    assert exc.value.code == 409
    assert "base model" in json.loads(exc.value.read())["error"]


def test_generating_needs_the_guard_header(server):
    with pytest.raises(urllib.error.HTTPError) as exc:
        post(server, "/api/infer/generate",
             {"checkpoint": "demo/ckpt_last.pt", "mode": "complete", "prompt": "hi"},
             guard=False)
    assert exc.value.code == 403


def test_an_unknown_checkpoint_is_a_4xx_with_the_list(server):
    with pytest.raises(urllib.error.HTTPError) as exc:
        post(server, "/api/infer/generate",
             {"checkpoint": "demo/nope.pt", "mode": "complete", "prompt": "hi"})
    assert "demo/ckpt_last.pt" in json.loads(exc.value.read())["error"]


def test_history_and_compare_routes(server):
    post(server, "/api/infer/generate",
         {"checkpoint": "demo/ckpt_last.pt", "mode": "complete", "prompt": "hi",
          "probe": "fluency", "sampling": {"max_new_tokens": 2, "temperature": 0}}).read()
    hist = get(server, "/api/infer/history?limit=5")
    assert hist["rows"] and hist["rows"][0]["probe"] == "fluency"
    cmp = get(server, "/api/infer/compare?probe=fluency")
    assert cmp["count"] >= 1 and cmp["rows"][0]["step"] == 500


def test_unload_frees_the_model(server):
    post(server, "/api/infer/generate",
         {"checkpoint": "demo/ckpt_last.pt", "mode": "complete", "prompt": "hi",
          "sampling": {"max_new_tokens": 2, "temperature": 0}}).read()
    assert get(server, "/api/infer/status")["loaded"] is not None
    with post(server, "/api/infer/unload", {}) as resp:
        assert json.loads(resp.read())["freed"] is True
    assert get(server, "/api/infer/status")["loaded"] is None


def test_portal_yaml_is_still_not_mistaken_for_a_run():
    """`configs/portal.yaml` gained an `infer:` section, which mentions models a lot. It
    must still fail the "declares a top-level `model:`" test that keeps it out of the run
    picker — a phantom run there is exactly the kind of thing you waste an evening on."""
    from aksharallm.infer.checkpoints import repo_root
    from aksharallm.portal.runs import _is_run_config
    portal_yaml = repo_root() / "configs" / "portal.yaml"
    assert portal_yaml.is_file(), "the repo's own portal.yaml should exist"
    assert _is_run_config(portal_yaml) is False


# --- the model's sloppiness must not read as ours ------------------------------------------
# `run_program` compiles the generated source in-process before dispatching it to the
# subprocess, because "SyntaxError, line 4" beats a traceback from a child. Compiling only
# parses -- it never runs -- which is what makes that safe. But since Python 3.12 an invalid
# escape sequence in a string literal is a *compile-time* SyntaxWarning, so a model writing
# `re.findall("\w+", s)` instead of `r"\w+"` printed
#     <model>:6: SyntaxWarning: invalid escape sequence '\w'
# straight into the GRPO training log, between two step lines, with no step number and
# nothing to say it came from generated code rather than from the trainer.

def test_a_warning_from_generated_code_does_not_escape_into_our_output(recwarn):
    """The model's regex is the model's business; the reward is unaffected either way."""
    program = ('import re\n'
               'def words(s):\n'
               '    return re.findall("\\w+", s)\n'          # not a raw string, on purpose
               'assert words("a b") == ["a", "b"]\n')
    res = sandbox.run_program(program)
    assert res.ok and res.status == "pass", f"the program itself is fine: {res.detail}"
    assert not [w for w in recwarn if issubclass(w.category, SyntaxWarning)], (
        "a SyntaxWarning from the generated source reached our own warning stream; it lands "
        "in the trainer's log looking like the trainer's problem")


def test_suppressing_warnings_did_not_suppress_real_syntax_errors():
    """The compile check earns its place by catching these — the single most common outcome
    for a base model. Silencing warnings must not silence errors."""
    res = sandbox.run_program("def f(:\n    pass\n")
    assert not res.ok and res.status == "syntax"
    assert "not valid Python" in res.detail
