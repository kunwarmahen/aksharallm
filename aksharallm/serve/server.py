"""An HTTP server for the model, shaped like the API everything already speaks.

    python -m aksharallm.serve small-code            # then http://127.0.0.1:8770/v1

Three endpoints — `/v1/models`, `/v1/completions`, `/v1/chat/completions` — with the request
and response fields of the OpenAI API, because that is the shape every client library,
editor plugin and script already knows. Nothing here is an endorsement of that API; it is the
cheapest way to make a model you trained yourself usable by tools you did not write.

What makes this a *server* rather than a loop with sockets attached:

* **One engine, one queue, many connections.** Every request joins the same
  `BatchEngine` and is advanced by the same pass over the weights, so thirty clients cost
  barely more per token than one. A worker thread owns the model; the HTTP threads only put
  requests in and take tokens out.
* **The training run still owns the card.** Device selection goes through the same
  `plan_device` policy as the playground: if a run is training, the server loads on the CPU
  and says so in `/health`. A serving process must never be the reason a six-day run dies.
* **Loopback unless told otherwise.** Same rule as the portal — this generates text on your
  machine, and `--host 0.0.0.0` is a decision, not a default.
* **Backpressure is honest.** When the KV pool is full, new requests wait in the queue and
  `/health` says how many; the server never evicts a sequence someone is already reading.

Read with: docs/16-serving.md -- the chapter this implements; it ends with the order to read
these files in.
"""

from __future__ import annotations

import argparse
import json
import queue
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import torch

from ..infer.checkpoints import CheckpointStore, InferError
from ..infer.engine import InferConfig, plan_device, training_runs
from ..tokenizer.tokenizer import Tokenizer
from .batch import BatchEngine, Request
from .paged import BLOCK_SIZE, BlockPool

MAX_BODY = 1 << 20
DEFAULT_PORT = 8770


class Job:
    """One HTTP request's view of a sequence in the batch: a queue of tokens and an ending."""

    def __init__(self, req: Request):
        self.req = req
        self.tokens: queue.Queue = queue.Queue()
        self.done = threading.Event()
        self.finish_reason = "length"
        self.started = time.time()


class ModelServer:
    """The model, the batch engine, and the worker thread that steps it.

    Deliberately separate from the HTTP handler: a benchmark or a test drives this class
    directly, and nothing about batching depends on there being a socket.
    """

    def __init__(self, ckpt_id: str, root: Path | None = None, device: str | None = None,
                 max_batch: int = 32, pool_blocks: int | None = None,
                 tokenizer: str | None = None):
        from ..infer.cli import load_model, resolve_tokenizer

        self.store = CheckpointStore(root)
        self.ckpt_id = self.store.identify(ckpt_id)
        path = self.store.resolve(*self.ckpt_id.split("/"))

        cfg = InferConfig.load(root)
        if device:
            cfg.device = device
        self.plan = plan_device(cfg, training_runs(root))
        self.device = self.plan.device

        self.model, ckpt = load_model(str(path), device=self.device)
        self.tokenizer = Tokenizer(resolve_tokenizer(ckpt, tokenizer))

        # How much memory the KV pool is allowed. A block is
        # `2 * n_layers * n_kv_heads * block_size * head_dim` elements; with GQA that is small
        # enough that a few thousand blocks is a fraction of a gigabyte — which is the point
        # of paging rather than reserving a window per client.
        mcfg = self.model.cfg
        blocks = pool_blocks or max(64, max_batch * (mcfg.max_seq_len // BLOCK_SIZE + 1))
        self.pool = BlockPool(
            n_layers=mcfg.n_layers, n_blocks=blocks, n_kv_heads=mcfg.n_kv_heads,
            head_dim=mcfg.head_dim, block_size=BLOCK_SIZE,
            dtype=torch.bfloat16 if self.device.startswith("cuda") else torch.float32,
            device=self.device)
        self.engine = BatchEngine(self.model, self.pool, max_batch=max_batch,
                                  device=self.device)

        self.jobs: dict[int, Job] = {}
        self.lock = threading.Lock()
        self.wake = threading.Event()
        self.stop = threading.Event()
        self.worker = threading.Thread(target=self._loop, name="aksharallm-serve",
                                       daemon=True)
        self.worker.start()

    # ---- the loop ---------------------------------------------------------------------
    def _loop(self) -> None:
        """Step the engine whenever there is anything to step, sleep when there is not.

        An idle server must not spin a core: it waits on an event that `submit` sets. There
        is exactly one of these threads, which is what makes the engine's lack of locking
        safe — every mutation of a sequence happens here.
        """
        while not self.stop.is_set():
            if not self.engine.busy:
                self.wake.wait(timeout=0.5)
                self.wake.clear()
                continue
            with self.lock:
                produced = self.engine.step()
                finished = self.engine.take_finished()
            for seq_id, token in produced:
                job = self.jobs.get(seq_id)
                if job is not None:
                    job.tokens.put(token)
            for seq in finished:
                job = self.jobs.pop(seq.id, None)
                if job is not None:
                    job.finish_reason = seq.finish_reason or "length"
                    job.done.set()

    def submit(self, prompt_ids: list[int], **kw) -> Job:
        req = Request(prompt_ids=prompt_ids, eos_id=self.tokenizer.eos_id, **kw)
        with self.lock:
            self.engine.submit(req)
        job = Job(req)
        self.jobs[req.id] = job
        self.wake.set()
        return job

    def stream(self, job: Job, timeout: float = 300.0):
        """Yield token ids for one job until it finishes. The HTTP layer turns these into
        text; a benchmark counts them."""
        deadline = time.time() + timeout
        while True:
            try:
                yield job.tokens.get(timeout=0.25)
            except queue.Empty:
                if job.done.is_set() and job.tokens.empty():
                    return
                if time.time() > deadline:
                    job.finish_reason = "timeout"
                    return

    def shutdown(self) -> None:
        self.stop.set()
        self.wake.set()
        self.worker.join(timeout=2.0)

    # ---- what the endpoints report -------------------------------------------------------
    def health(self) -> dict:
        stats = self.engine.stats
        return {
            "ok": True,
            "model": self.ckpt_id,
            "device": self.device,
            "device_reason": self.plan.reason,
            "training": self.plan.training,
            "running": len(self.engine.running),
            "waiting": len(self.engine.waiting),
            "max_batch": self.engine.max_batch,
            "kv_blocks": {"total": self.pool.n_blocks, "used": self.pool.used_blocks,
                          "free": self.pool.free_blocks, "block_size": self.pool.block_size,
                          "bytes": self.pool.bytes()},
            "stats": stats.as_dict(),
        }


class Handler(BaseHTTPRequestHandler):
    """The OpenAI-shaped surface. `server_ref` is injected by `serve()`."""

    server_version = "aksharallm"
    protocol_version = "HTTP/1.1"
    quiet = True

    def log_message(self, fmt, *args):  # noqa: D102 - stdlib hook
        if not self.quiet:
            super().log_message(fmt, *args)

    # ---- plumbing ------------------------------------------------------------------------
    def _send(self, code: int, body: bytes, ctype: str = "application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code: int = 200):
        self._send(code, json.dumps(obj).encode())

    def _error(self, code: int, msg: str):
        # The error shape clients expect, so a library surfaces the message rather than
        # "unknown error".
        self._json({"error": {"message": msg, "type": "invalid_request_error"}}, code)

    def _body(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        if n > MAX_BODY:
            raise ValueError("request body too large")
        if not n:
            return {}
        data = json.loads(self.rfile.read(n) or b"{}")
        if not isinstance(data, dict):
            raise ValueError("body must be a JSON object")
        return data

    # ---- routes ---------------------------------------------------------------------------
    def do_GET(self):
        path = urlparse(self.path).path
        srv = self.server_ref
        if path in ("/health", "/v1/health"):
            return self._json(srv.health())
        if path == "/v1/models":
            return self._json({"object": "list", "data": [
                {"id": srv.ckpt_id, "object": "model", "owned_by": "aksharallm",
                 "created": int(time.time())}]})
        return self._error(404, f"no such path: {path}")

    def do_POST(self):
        path = urlparse(self.path).path
        try:
            data = self._body()
        except (ValueError, json.JSONDecodeError) as exc:
            return self._error(400, str(exc))
        if path == "/v1/completions":
            return self._generate(data, chat=False)
        if path == "/v1/chat/completions":
            return self._generate(data, chat=True)
        return self._error(404, f"no such path: {path}")

    # ---- generation ------------------------------------------------------------------------
    def _prompt_ids(self, data: dict, chat: bool) -> list[int]:
        srv = self.server_ref
        if not chat:
            prompt = data.get("prompt")
            if isinstance(prompt, list):
                prompt = "".join(str(p) for p in prompt)
            return srv.tokenizer.encode(str(prompt or ""), bos=True)
        messages = [m for m in (data.get("messages") or [])
                    if isinstance(m, dict) and m.get("content")]
        if not messages:
            raise ValueError("messages must be a non-empty list of {role, content}")
        ids, _ = srv.tokenizer.render_chat(messages, add_generation_prompt=True)
        return ids

    def _generate(self, data: dict, chat: bool):
        srv = self.server_ref
        try:
            ids = self._prompt_ids(data, chat)
        except ValueError as exc:
            return self._error(400, str(exc))

        job = srv.submit(
            ids,
            max_new_tokens=int(data.get("max_tokens") or 128),
            temperature=float(data.get("temperature", 0.8)),
            top_p=float(data.get("top_p", 0.95)),
            top_k=int(data.get("top_k") or 50) or None,
        )
        stamp = int(time.time())
        rid = f"cmpl-{uuid.uuid4().hex[:16]}"

        if data.get("stream"):
            return self._stream(job, rid, stamp, chat)

        out_ids = list(srv.stream(job))
        produced = len(out_ids)
        text = srv.tokenizer.decode(out_ids)
        payload = {
            "id": rid, "object": "chat.completion" if chat else "text_completion",
            "created": stamp, "model": srv.ckpt_id,
            "choices": [{
                "index": 0, "finish_reason": job.finish_reason,
                **({"message": {"role": "assistant", "content": text}} if chat
                   else {"text": text}),
            }],
            "usage": {"prompt_tokens": len(ids), "completion_tokens": produced,
                      "total_tokens": len(ids) + produced},
        }
        return self._json(payload)

    def _stream(self, job, rid: str, stamp: int, chat: bool):
        """Server-sent events, in the chunk shape clients expect, ending with `[DONE]`."""
        from ..infer.generate import IncrementalDecoder

        srv = self.server_ref
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.end_headers()
        decoder = IncrementalDecoder(srv.tokenizer,
                                     skip_ids={srv.tokenizer.eos_id}
                                     if srv.tokenizer.eos_id is not None else set())

        def chunk(delta: str, finish=None) -> bytes:
            body = {"id": rid, "object": "chat.completion.chunk" if chat
                    else "text_completion", "created": stamp, "model": srv.ckpt_id,
                    "choices": [{"index": 0, "finish_reason": finish,
                                 **({"delta": {"content": delta}} if chat
                                    else {"text": delta})}]}
            return f"data: {json.dumps(body)}\n\n".encode()

        try:
            for token in srv.stream(job):
                piece = decoder.push(token)
                if piece:
                    self.wfile.write(chunk(piece))
                    self.wfile.flush()
            tail = decoder.flush()
            if tail:
                self.wfile.write(chunk(tail))
            self.wfile.write(chunk("", finish=job.finish_reason))
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            # The client hung up mid-answer. The sequence keeps going until it finishes —
            # cancelling it mid-batch is a future refinement, and a wrong one to rush.
            pass


def serve(ckpt: str, host: str = "127.0.0.1", port: int = DEFAULT_PORT,
          root: Path | None = None, device: str | None = None, max_batch: int = 32,
          pool_blocks: int | None = None, quiet: bool = True):
    """Start the server and block. Returns only on Ctrl-C."""
    model_server = ModelServer(ckpt, root=root, device=device, max_batch=max_batch,
                               pool_blocks=pool_blocks)
    handler = type("BoundHandler", (Handler,),
                   {"server_ref": model_server, "quiet": quiet})
    httpd = ThreadingHTTPServer((host, port), handler)
    health = model_server.health()
    print(f"serving {model_server.ckpt_id} on http://{host}:{port}/v1  "
          f"({health['device']}: {health['device_reason']})")
    print(f"kv pool {health['kv_blocks']['total']} blocks "
          f"({health['kv_blocks']['bytes'] / 1e6:.0f} MB), max batch {max_batch}")
    print(f"try:  curl -s http://{host}:{port}/v1/completions -d "
          f"'{{\"prompt\": \"def quicksort(arr):\", \"max_tokens\": 64}}'")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        httpd.server_close()
        model_server.shutdown()
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m aksharallm.serve",
        description="Serve a checkpoint over HTTP, with continuous batching and a paged KV "
                    "cache. The API is OpenAI-shaped so existing clients work unmodified.")
    ap.add_argument("checkpoint", help="run name, id, or path")
    ap.add_argument("--host", default="127.0.0.1",
                    help="0.0.0.0 exposes it to your network — a decision, not a default")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--max-batch", type=int, default=32,
                    help="sequences advanced per pass over the weights")
    ap.add_argument("--pool-blocks", type=int, default=None,
                    help=f"KV blocks of {BLOCK_SIZE} tokens each (default: sized for "
                         f"max-batch full windows)")
    ap.add_argument("--device", default=None, help="auto | cuda | cpu")
    ap.add_argument("--root", default=None)
    ap.add_argument("--verbose", action="store_true", help="log every request")
    args = ap.parse_args(argv)

    try:
        return serve(args.checkpoint, host=args.host, port=args.port,
                     root=Path(args.root) if args.root else None, device=args.device,
                     max_batch=args.max_batch, pool_blocks=args.pool_blocks,
                     quiet=not args.verbose)
    except InferError as exc:
        print(f"error: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
