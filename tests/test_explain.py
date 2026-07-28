"""Tests for the portal's code explainer.

No model is loaded and no GPU is touched: a fake Ollama — twenty lines of `http.server`
speaking the same NDJSON that the real one does — stands in for it, so the whole path is
exercised end to end, from "the browser posts a line range" to "tokens arrive as
server-sent events".

What these guard hardest is the reading boundary. The Code tab browses the tree the portal
was started in, and it is attached to a server that can be put on a LAN — so the tests spend
most of their effort trying to make it read something outside that tree: a `..` path, an
absolute path, a symlink pointing out, or a 20 GB `train.bin`.
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from aksharallm.portal.explain import (
    DEFAULT_ASK,
    ExplainConfig,
    Ollama,
    SourceTree,
    build_messages,
    doc_for,
    number_lines,
    window_file,
)
from aksharallm.portal.runs import RunError, RunStore
from aksharallm.portal.server import serve

# --------------------------------------------------------------------------------------
# a repo to read
# --------------------------------------------------------------------------------------

SAMPLE = '''\
"""A module docstring
that runs over two lines."""

import math


def area(r):
    # the circle, not the disc
    return math.pi * r ** 2
'''


@pytest.fixture
def repo(tmp_path):
    """A miniature checkout: two source dirs, one artifact dir, one environment dir."""
    (tmp_path / "aksharallm" / "model").mkdir(parents=True)
    (tmp_path / "aksharallm" / "model" / "geom.py").write_text(SAMPLE)
    (tmp_path / "aksharallm" / "__init__.py").write_text("")
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "tiny.yaml").write_text("name: tiny\nmodel:\n  d_model: 8\n")
    (tmp_path / "configs" / "portal.yaml").write_text(
        "explain:\n  model: test-model\n  num_ctx: 512\n  max_file_chars: 4000\n")
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "03-model.md").write_text("# the model\n")
    (tmp_path / "README.md").write_text("# demo\n")
    # Things that must never be reachable.
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "train.bin").write_bytes(b"\x00" * 64)
    (tmp_path / "data" / "notes.md").write_text("secret\n")
    (tmp_path / ".venv" / "lib").mkdir(parents=True)
    (tmp_path / ".venv" / "lib" / "evil.py").write_text("import os\n")
    (tmp_path / "aksharallm" / "model" / "__pycache__").mkdir()
    (tmp_path / "aksharallm" / "model" / "__pycache__" / "geom.pyc").write_text("x")
    (tmp_path / "outside.py").write_text("nope\n")
    return tmp_path


# --------------------------------------------------------------------------------------
# what is readable
# --------------------------------------------------------------------------------------

def test_tree_walks_the_whole_root(repo):
    """The browser is rooted where the portal runs and goes up and down inside it, so a
    text file anywhere under the root is readable — including one the author added today in
    a directory nobody thought to allowlist."""
    tree = SourceTree(repo).files()
    paths = {f["path"] for f in tree["files"]}
    assert {"aksharallm/model/geom.py", "configs/tiny.yaml", "docs/03-model.md",
            "README.md", "outside.py"} <= paths
    assert tree["root"] == str(repo)


def test_tree_prunes_environments_binaries_and_empty_folders(repo):
    tree = SourceTree(repo).files()
    paths = {f["path"] for f in tree["files"]}
    assert not any(p.startswith(".venv/") for p in paths)      # a dot directory
    assert not any("__pycache__" in p for p in paths)
    assert "data/train.bin" not in paths                       # not a text suffix
    # `data/` holds one readable file, so it is navigable; a folder with none is not listed.
    assert "data" in tree["dirs"]
    assert "aksharallm/model" in tree["dirs"] and "aksharallm" in tree["dirs"]
    assert not any(d.startswith(".venv") or "__pycache__" in d for d in tree["dirs"])


def test_tree_lists_every_ancestor_so_navigation_never_dead_ends(repo):
    deep = repo / "a" / "b" / "c"
    deep.mkdir(parents=True)
    (deep / "note.md").write_text("hi\n")
    dirs = SourceTree(repo).files()["dirs"]
    assert {"a", "a/b", "a/b/c"} <= set(dirs)


@pytest.mark.parametrize("bad", [
    "../../../etc/passwd",
    "/etc/passwd",
    "data/train.bin",
    ".venv/lib/evil.py",
    "aksharallm/model/__pycache__/geom.pyc",
    "aksharallm/../../elsewhere.py",
    "",
])
def test_resolve_refuses(repo, bad):
    with pytest.raises(RunError):
        SourceTree(repo).resolve(bad)


def test_resolve_allows_walking_up_and_down_inside_the_root(repo):
    tree = SourceTree(repo)
    assert tree.resolve("aksharallm/../configs/tiny.yaml").name == "tiny.yaml"


def test_resolve_refuses_a_symlink_out_of_the_repo(repo, tmp_path):
    secret = tmp_path.parent / "secret.py"
    secret.write_text("token = 'hunter2'\n")
    link = repo / "aksharallm" / "leak.py"
    link.symlink_to(secret)
    with pytest.raises(RunError):
        SourceTree(repo).resolve("aksharallm/leak.py")
    # and it is not offered in the first place
    assert "aksharallm/leak.py" not in {f["path"] for f in SourceTree(repo).files()["files"]}


def test_read_returns_text_and_the_matching_doc(repo):
    got = SourceTree(repo).read("aksharallm/model/geom.py")
    assert got["path"] == "aksharallm/model/geom.py"
    assert got["lang"] == "python"
    assert got["lines"] == 9
    assert got["doc"] == "docs/03-model.md"
    assert "math.pi" in got["text"]


def test_doc_hints_prefer_the_longest_prefix():
    assert doc_for("aksharallm/train/sft.py") == "docs/05-posttraining.md"
    assert doc_for("aksharallm/train/pretrain.py") == "docs/04-pretraining.md"
    assert doc_for("aksharallm/portal/server.py") == "docs/07-scaling.md"


def test_portal_yaml_is_not_mistaken_for_a_run(repo):
    """`configs/portal.yaml` configures the explainer; it must not appear as a training run
    with no launcher, no log and no way to start it."""
    assert RunStore(repo).runs() == ["tiny"]


# --------------------------------------------------------------------------------------
# the prompt
# --------------------------------------------------------------------------------------

def test_number_lines_is_1_based_and_aligned():
    out = number_lines("a\nb").splitlines()
    assert out[0].endswith("| a") and out[0].strip().startswith("1")
    assert out[1].strip().startswith("2")


def test_window_keeps_the_selection_and_says_what_it_dropped():
    text = "\n".join(f"line {i}" for i in range(1, 501))
    listing, truncated = window_file(text, 250, 252, max_chars=600)
    assert truncated
    assert "line 250" in listing and "line 252" in listing
    assert "not shown" in listing            # the model is told it is looking at a fragment
    assert len(listing) < 1600
    # A file that fits is never windowed.
    small, cut = window_file("a\nb\nc", 1, 1, max_chars=1000)
    assert not cut and "not shown" not in small


def test_build_messages_carries_file_selection_and_primer(repo):
    cfg = ExplainConfig.load(repo)
    src = SourceTree(repo).read("aksharallm/model/geom.py")
    msgs = build_messages(cfg, path=src["path"], text=src["text"], start=7, end=9,
                          doc=src["doc"])
    assert [m["role"] for m in msgs] == ["system", "user"]
    user = msgs[1]["content"]
    assert "aksharallm" in user                       # the primer
    assert "`aksharallm/model/geom.py`" in user       # which file
    assert "docs/03-model.md" in user                 # where the human version lives
    assert "lines 7–9" in user
    assert "return math.pi * r ** 2" in user          # the selection, verbatim
    assert "   7 | def area(r):" in user              # and the numbered listing
    assert DEFAULT_ASK in user


def test_build_messages_includes_a_partial_line_selection(repo):
    cfg = ExplainConfig.load(repo)
    src = SourceTree(repo).read("aksharallm/model/geom.py")
    msgs = build_messages(cfg, path=src["path"], text=src["text"], start=9, end=9,
                          snippet="r ** 2", question="what is this exponent?")
    user = msgs[1]["content"]
    assert "highlighted exactly this text" in user and "r ** 2" in user
    assert "what is this exponent?" in user


def test_history_is_appended_and_capped(repo):
    cfg = ExplainConfig.load(repo)
    cfg.max_history = 2
    src = SourceTree(repo).read("aksharallm/model/geom.py")
    history = [{"role": "user", "content": f"q{i}"} for i in range(5)]
    history.append({"role": "nonsense", "content": "ignore me"})
    msgs = build_messages(cfg, path=src["path"], text=src["text"], start=1, end=1,
                          history=history)
    assert [m["content"] for m in msgs[2:]] == ["q4"]   # last 2, minus the bad role


# --------------------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------------------

def test_config_file_then_env_then_defaults(repo, monkeypatch):
    monkeypatch.delenv("AKSHARALLM_OLLAMA_HOST", raising=False)
    monkeypatch.delenv("AKSHARALLM_EXPLAIN_MODEL", raising=False)
    cfg = ExplainConfig.load(repo)
    assert cfg.model == "test-model"          # from configs/portal.yaml
    assert cfg.num_ctx == 512
    assert cfg.keep_alive == "5m"             # untouched by the file -> the default

    monkeypatch.setenv("AKSHARALLM_EXPLAIN_MODEL", "other:7b")
    monkeypatch.setenv("AKSHARALLM_OLLAMA_HOST", "http://box:11434/")
    cfg = ExplainConfig.load(repo)
    assert cfg.model == "other:7b"
    assert cfg.host == "http://box:11434"     # trailing slash trimmed


def test_config_survives_a_broken_yaml(repo):
    (repo / "configs" / "portal.yaml").write_text("explain: [this is not: a mapping\n")
    cfg = ExplainConfig.load(repo)
    assert cfg.model == "gemma4:12b"          # the default, not a crash
    assert cfg.note and "portal.yaml" in cfg.note


def test_config_reloads_when_the_file_changes(repo):
    cfg = ExplainConfig.load(repo)
    assert cfg.model == "test-model"
    (repo / "configs" / "portal.yaml").write_text("explain:\n  model: changed:1b\n")
    # mtime resolution is coarse enough on some filesystems to need a nudge
    cfg._mtime = -1
    assert cfg.reload_if_changed().model == "changed:1b"


# --------------------------------------------------------------------------------------
# a fake Ollama, and the wire
# --------------------------------------------------------------------------------------

class FakeOllama(BaseHTTPRequestHandler):
    """Speaks the two endpoints the client uses. `prompts` records what it was asked."""

    prompts: list = []
    reply = ["Line 9 ", "squares the radius."]
    thinking: list = []
    fail: str | None = None
    reject_think = False

    def log_message(self, *a):
        pass

    def do_GET(self):
        body = json.dumps({"models": [
            {"name": "test-model", "size": 42, "details": {"family": "fake"}},
            {"name": "other:7b", "size": 43, "details": {"family": "fake"}},
        ]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        payload = json.loads(self.rfile.read(n) or b"{}")
        FakeOllama.prompts.append(payload)
        if FakeOllama.reject_think and "think" in payload:
            body = json.dumps({"error": "\"think\" is not supported by this model"}).encode()
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.end_headers()
        if FakeOllama.fail:
            self.wfile.write(json.dumps({"error": FakeOllama.fail}).encode() + b"\n")
            return
        for piece in FakeOllama.thinking:
            self.wfile.write(
                json.dumps({"message": {"thinking": piece}, "done": False}).encode() + b"\n")
            self.wfile.flush()
        for piece in FakeOllama.reply:
            self.wfile.write(
                json.dumps({"message": {"content": piece}, "done": False}).encode() + b"\n")
            self.wfile.flush()
        self.wfile.write(json.dumps({"message": {"content": ""}, "done": True}).encode()
                         + b"\n")


@pytest.fixture
def fake_ollama():
    FakeOllama.prompts = []
    FakeOllama.fail = None
    FakeOllama.thinking = []
    FakeOllama.reject_think = False
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), FakeOllama)
    httpd.daemon_threads = True
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_port}"
    httpd.shutdown()
    httpd.server_close()


def test_client_lists_models(fake_ollama):
    cfg = ExplainConfig(host=fake_ollama)
    assert [m["name"] for m in Ollama(cfg).models()] == ["test-model", "other:7b"]


def test_client_streams_and_stops_on_done(fake_ollama):
    cfg = ExplainConfig(host=fake_ollama, model="test-model")
    got = list(Ollama(cfg).chat([{"role": "user", "content": "hi"}]))
    assert "".join(t for k, t in got if k == "delta") == "Line 9 squares the radius."
    sent = FakeOllama.prompts[-1]
    assert sent["stream"] is True and sent["model"] == "test-model"
    assert sent["options"]["num_ctx"] == cfg.num_ctx


def test_thinking_is_kept_separate_from_the_answer(fake_ollama):
    """A reasoning model narrates before it answers. Mixing the two would bury the sentence
    the reader asked for — and an answer that is *only* thinking is the failure mode that
    looks exactly like a broken portal."""
    FakeOllama.thinking = ["let me look ", "at line 9."]
    cfg = ExplainConfig(host=fake_ollama)
    got = list(Ollama(cfg).chat([{"role": "user", "content": "hi"}]))
    assert "".join(t for k, t in got if k == "thinking") == "let me look at line 9."
    assert "".join(t for k, t in got if k == "delta") == "Line 9 squares the radius."


def test_think_is_sent_and_dropped_if_the_model_rejects_it(fake_ollama):
    """`think: false` is what stops a reasoning model spending its whole budget thinking.
    Models that have never heard of the field must not break because of it."""
    cfg = ExplainConfig(host=fake_ollama)
    list(Ollama(cfg).chat([{"role": "user", "content": "hi"}]))
    assert FakeOllama.prompts[-1]["think"] is False

    FakeOllama.reject_think = True
    got = list(Ollama(cfg).chat([{"role": "user", "content": "hi"}]))
    assert "".join(t for k, t in got if k == "delta") == "Line 9 squares the radius."
    assert "think" not in FakeOllama.prompts[-1]      # retried without it

    cfg.think = None
    FakeOllama.reject_think = False
    list(Ollama(cfg).chat([{"role": "user", "content": "hi"}]))
    assert "think" not in FakeOllama.prompts[-1]      # never sent at all


def test_num_gpu_is_only_sent_when_it_is_set(fake_ollama):
    """Unset means "Ollama decides"; 0 means "stay off the card the trainer is using". The
    difference must survive to the wire, because it is the difference between a slow answer
    and a dead training run."""
    cfg = ExplainConfig(host=fake_ollama)
    list(Ollama(cfg).chat([{"role": "user", "content": "hi"}]))
    assert "num_gpu" not in FakeOllama.prompts[-1]["options"]

    cfg.num_gpu = 0
    list(Ollama(cfg).chat([{"role": "user", "content": "hi"}]))
    assert FakeOllama.prompts[-1]["options"]["num_gpu"] == 0


def test_client_raises_a_useful_error_when_ollama_is_down():
    cfg = ExplainConfig(host="http://127.0.0.1:1")     # nothing listens on port 1
    with pytest.raises(RunError, match="ollama serve"):
        Ollama(cfg).models()


def test_client_surfaces_a_model_error(fake_ollama):
    FakeOllama.fail = "model 'ghost' not found"
    cfg = ExplainConfig(host=fake_ollama)
    with pytest.raises(RunError, match="ghost"):
        list(Ollama(cfg).chat([{"role": "user", "content": "hi"}]))


# --------------------------------------------------------------------------------------
# the HTTP endpoints
# --------------------------------------------------------------------------------------

@pytest.fixture
def portal(repo, fake_ollama, monkeypatch):
    monkeypatch.setenv("AKSHARALLM_OLLAMA_HOST", fake_ollama)
    monkeypatch.delenv("AKSHARALLM_EXPLAIN_MODEL", raising=False)
    httpd = serve(repo, "127.0.0.1", 0)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_port}"
    httpd.shutdown()
    httpd.server_close()


def get(base, path):
    with urllib.request.urlopen(base + path, timeout=10) as resp:
        return json.loads(resp.read())


def post(base, path, body, guard=True):
    headers = {"Content-Type": "application/json"}
    if guard:
        headers["X-Portal"] = "1"
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(),
                                 headers=headers)
    return urllib.request.urlopen(req, timeout=20)


def test_source_endpoints(portal, repo):
    tree = get(portal, "/api/source")
    assert any(f["path"] == "aksharallm/model/geom.py" for f in tree["files"])
    assert "aksharallm/model" in tree["dirs"]
    assert tree["root"] == str(repo)          # the tree is rooted where the server runs
    got = get(portal, "/api/source/file?path=aksharallm/model/geom.py")
    assert got["lang"] == "python" and "math.pi" in got["text"]


def test_source_file_refuses_a_binary_artifact(portal):
    with pytest.raises(urllib.error.HTTPError) as exc:
        get(portal, "/api/source/file?path=data/train.bin")
    assert exc.value.code == 400
    assert "not a text file" in json.loads(exc.value.read())["error"]


def test_models_endpoint_reports_the_configured_model(portal):
    info = get(portal, "/api/explain/models")
    assert info["available"] is True
    assert info["model"] == "test-model"
    assert [m["name"] for m in info["models"]] == ["test-model", "other:7b"]


def test_models_endpoint_names_any_run_that_is_training(portal, repo):
    """The page needs this to warn that the explainer would land on a card the trainer is
    already 21 GB into."""
    info = get(portal, "/api/explain/models")
    assert info["training"] == [] and info["on_cpu"] is False

    ckpt = repo / "checkpoints" / "tiny"
    ckpt.mkdir(parents=True)
    (ckpt / "train_log.jsonl").write_text('{"step": 1, "loss": 5.0}\n')
    # A live process whose command line the store will accept as the trainer's.
    trainer = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)",
         "-m", "aksharallm.train.pretrain", "configs/tiny.yaml"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        (ckpt / "train.pid").write_text(f"{trainer.pid}\n")
        assert "tiny" in get(portal, "/api/explain/models")["training"]
    finally:
        trainer.kill()
        trainer.wait()


def test_models_endpoint_explains_a_missing_ollama(repo, monkeypatch):
    monkeypatch.setenv("AKSHARALLM_OLLAMA_HOST", "http://127.0.0.1:1")
    httpd = serve(repo, "127.0.0.1", 0)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        info = get(f"http://127.0.0.1:{httpd.server_port}", "/api/explain/models")
        assert info["available"] is False and "ollama serve" in info["error"]
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_explain_streams_server_sent_events(portal):
    resp = post(portal, "/api/explain", {"path": "aksharallm/model/geom.py",
                                         "start": 7, "end": 9})
    frames = [json.loads(line[len("data: "):])
              for line in resp.read().decode().split("\n\n") if line.strip()]
    assert frames[0]["start"] is True
    assert frames[0]["lines"] == [7, 9]
    assert frames[0]["doc"] == "docs/03-model.md"
    assert "".join(f.get("delta", "") for f in frames) == "Line 9 squares the radius."
    assert frames[-1]["done"] is True
    # and the model was given the file, not just the three lines
    asked = FakeOllama.prompts[-1]["messages"][-1]["content"]
    assert "import math" in asked and "def area(r):" in asked


def test_explain_clamps_a_line_range_it_cannot_honour(portal):
    resp = post(portal, "/api/explain", {"path": "aksharallm/model/geom.py",
                                         "start": 900, "end": 4000})
    first = json.loads(resp.read().decode().split("\n\n")[0][len("data: "):])
    assert first["lines"] == [9, 9]        # the last line, not an empty selection


def test_explain_reports_a_model_error_inside_the_stream(portal):
    FakeOllama.fail = "model 'ghost' not found"
    resp = post(portal, "/api/explain", {"path": "aksharallm/model/geom.py",
                                         "start": 1, "end": 1})
    frames = [json.loads(line[len("data: "):])
              for line in resp.read().decode().split("\n\n") if line.strip()]
    assert any("ghost" in (f.get("error") or "") for f in frames)


def test_explain_rejects_an_unknown_file_before_streaming(portal):
    with pytest.raises(urllib.error.HTTPError) as exc:
        post(portal, "/api/explain", {"path": "../../etc/passwd", "start": 1, "end": 1})
    assert "outside" in json.loads(exc.value.read())["error"]


def test_explain_needs_the_guard_header(portal):
    with pytest.raises(urllib.error.HTTPError) as exc:
        post(portal, "/api/explain", {"path": "aksharallm/model/geom.py",
                                      "start": 1, "end": 1}, guard=False)
    assert exc.value.code == 403
