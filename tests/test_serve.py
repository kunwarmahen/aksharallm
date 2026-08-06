"""Tests for the serving path: paged KV blocks and continuous batching.

The claim being defended is the same shape as speculative decoding's: **batching must not
change the answer.** A server that gives a slightly different reply depending on who else was
in the batch is unusable, and the difference would never be noticed by looking — it is one
token in thirty, and the text stays fluent.

So the load-bearing test is `test_a_batch_gives_each_sequence_what_it_would_have_got_alone`.
Everything else is the bookkeeping that makes it possible: blocks are handed out and given
back, a shared prefix is not written into, and a sequence that finishes frees its memory on
the step it finishes rather than at the end of the batch.
"""

from __future__ import annotations

import functools

import pytest
import torch

from aksharallm.config import ModelConfig
from aksharallm.infer.generate import generate
from aksharallm.model.transformer import Transformer
from aksharallm.serve.batch import BatchEngine, Request
from aksharallm.serve.paged import BLOCK_SIZE, BlockPool, OutOfBlocks, PagedCache, Sequence


@functools.lru_cache(maxsize=4)
def tiny(seed: int = 0, vocab: int = 64, max_seq_len: int = 128) -> Transformer:
    """A **briefly trained** model, and the training is the point.

    An untrained transformer predicts almost the same token whatever it is shown, so every
    test below passes even with the positions scrambled or the mask inverted — mutating those
    lines changed nothing about the greedy output, which is how three broken versions of this
    module first went green. Two seconds of training on sequences that repeat a random
    seven-token pattern makes the prediction genuinely depend on *what came before and how far
    back*, which is exactly what a serving bug corrupts.
    """
    torch.manual_seed(seed)
    cfg = ModelConfig(vocab_size=vocab, d_model=32, n_layers=2, n_heads=4, n_kv_heads=2,
                      max_seq_len=max_seq_len, dropout=0.0)
    model = Transformer(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    gen = torch.Generator().manual_seed(seed + 1)
    for _ in range(300):
        pattern = torch.randint(0, vocab, (8, 7), generator=gen)
        x = pattern.repeat(1, 16)[:, : min(64, max_seq_len)]
        _, loss = model(x[:, :-1], targets=x[:, 1:])
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model.eval()


def pool_for(model: Transformer, n_blocks: int = 64) -> BlockPool:
    return BlockPool(n_layers=len(model.blocks), n_blocks=n_blocks,
                     n_kv_heads=model.cfg.n_kv_heads, head_dim=model.cfg.head_dim,
                     dtype=torch.float32, device="cpu")


# ---- the claim ---------------------------------------------------------------------------

def test_a_batch_gives_each_sequence_what_it_would_have_got_alone():
    """Three prompts of different lengths, generated together, must match what each would
    have produced on its own.

    This is where every serving bug shows up at once: a wrong RoPE position, a mask that
    lets one row see another's keys, a block table off by one, padding that is attended to.
    All of them produce fluent, plausible, different text.
    """
    model = tiny()
    prompts = [[5, 9, 2, 5, 9, 2, 5], [7],
               [3, 11, 40, 8, 1, 6, 22, 3, 11, 40, 8, 1, 6, 22, 3, 11, 40]]
    alone = [generate(model, p, max_new_tokens=12, temperature=0.0, device="cpu")[len(p):]
             for p in prompts]

    engine = BatchEngine(model, pool_for(model), max_batch=8, device="cpu")
    reqs = [Request(prompt_ids=p, max_new_tokens=12, temperature=0.0) for p in prompts]
    out = engine.collect(reqs)

    for req, expected in zip(reqs, alone):
        assert out[req.id] == expected


def test_a_sequence_admitted_mid_flight_is_unaffected_by_its_neighbours():
    """Continuous batching means arriving while others are mid-answer. The late arrival must
    get exactly what it would have got in an empty server."""
    model = tiny()
    late_prompt = [11, 4, 9, 11, 4, 9, 11]
    alone = generate(model, late_prompt, max_new_tokens=10, temperature=0.0,
                     device="cpu")[len(late_prompt):]

    engine = BatchEngine(model, pool_for(model), max_batch=8, device="cpu")
    engine.submit(Request(prompt_ids=[2, 8, 1, 5, 2, 8, 1, 5], max_new_tokens=20,
                          temperature=0.0))
    for _ in range(4):
        engine.step()                       # the first request is already several tokens in
    late = engine.submit(Request(prompt_ids=late_prompt, max_new_tokens=10, temperature=0.0))

    got: list[int] = []
    for seq_id, token in engine.run():
        if seq_id == late.id:
            got.append(token)
    assert got == alone


def test_the_paged_path_produces_the_same_logits_as_a_contiguous_cache():
    """Logits, not sampled tokens — the sensitive version of the test above.

    A sampled token only changes when an error crosses the gap to the next-best candidate, so
    an address that is wrong by one block can leave the greedy answer untouched on a small
    model. Comparing the numbers themselves catches it. The sequence is deliberately longer
    than three blocks, because a bug in the block table is invisible in a sequence that fits
    in one.
    """
    model = tiny()
    tokens = [(i * 7 + 3) % 60 for i in range(40)]          # 40 tokens = 3 blocks
    engine = BatchEngine(model, pool_for(model), max_batch=2, device="cpu")
    req = Request(prompt_ids=tokens, max_new_tokens=1, temperature=0.0)
    seq = engine.submit(req)
    engine._admit()
    paged = engine._forward([seq], [0], [len(tokens)])[0, -1]

    caches = model.init_caches(1, model.cfg.max_seq_len, dtype=torch.float32, device="cpu")
    with torch.no_grad():
        contiguous, _ = model(torch.tensor([tokens]), caches=caches, full_logits=True)
    assert torch.allclose(paged, contiguous[0, -1], atol=1e-4), \
        (paged - contiguous[0, -1]).abs().max()


# ---- the blocks ---------------------------------------------------------------------------

def test_blocks_are_handed_out_and_given_back():
    model = tiny()
    pool = pool_for(model, n_blocks=8)
    cache = PagedCache(pool)
    seq = Sequence(id=1, tokens=list(range(BLOCK_SIZE + 1)))

    cache.reserve(seq)
    assert len(seq.blocks) == 2                      # 17 tokens needs two 16-token blocks
    assert pool.used_blocks == 2 and pool.free_blocks == 6
    cache.free(seq)
    assert pool.used_blocks == 0 and seq.blocks == []


def test_a_sequence_that_exactly_fills_its_blocks_is_addressable_to_the_last_token():
    """The boundary case, and the one an off-by-one in the block table hides in.

    Any addressing scheme is self-consistent as long as it writes and reads through the same
    (wrong) mapping — the cache simply uses different slots and nothing looks broken. What it
    cannot survive is the *last* token of a sequence whose length is an exact multiple of the
    block size, which is where an index that rounds the wrong way runs off the end of the
    block table.
    """
    model = tiny()
    cache = PagedCache(pool_for(model))
    n = 3 * BLOCK_SIZE
    seq = Sequence(id=1, tokens=list(range(n)))
    cache.reserve(seq)
    assert len(seq.blocks) == 3

    k = torch.arange(2 * n * 8, dtype=torch.float32).reshape(2, n, 8)
    cache.write(0, seq, 0, k, k)
    got_k, _ = cache.gather(0, [seq], [n])
    assert torch.equal(got_k[0], k)          # every token back, in order, including the last


def test_a_full_pool_refuses_rather_than_half_allocating():
    """All-or-nothing: a half-allocated sequence would have to be unwound by every caller,
    and one of them would forget."""
    pool = BlockPool(n_layers=1, n_blocks=3, n_kv_heads=2, head_dim=8,
                     dtype=torch.float32, device="cpu")
    with pytest.raises(OutOfBlocks):
        pool.allocate(4)
    assert pool.free_blocks == 3                     # nothing was taken


def test_memory_is_bounded_by_what_is_used_not_by_the_context_window():
    """The reason paging exists. Thirty short sequences must not cost thirty context
    windows."""
    model = tiny(max_seq_len=1024)
    pool = pool_for(model, n_blocks=64)
    cache = PagedCache(pool)
    seqs = [Sequence(id=i, tokens=list(range(8))) for i in range(30)]
    for s in seqs:
        cache.reserve(s)
    # One block each, not 1024 tokens each: 30 blocks of 16 against 30 windows of 1024.
    assert pool.used_blocks == 30
    assert pool.used_blocks * BLOCK_SIZE < 30 * model.cfg.max_seq_len / 10


def test_a_shared_prefix_is_counted_and_freed_once_per_holder():
    model = tiny()
    pool = pool_for(model, n_blocks=16)
    cache = PagedCache(pool)
    donor = Sequence(id=1, tokens=list(range(40)))
    cache.reserve(donor)
    donor.cached = 40
    before = pool.used_blocks

    borrower = Sequence(id=2, tokens=list(range(40)))
    shared = cache.share_prefix(borrower, donor, 40)
    assert shared == 32                     # rounded DOWN to whole blocks: 2 x 16
    assert pool.used_blocks == before       # sharing allocated nothing
    assert pool.refs[borrower.blocks[0]] == 2

    cache.free(borrower)
    assert pool.used_blocks == before       # the donor still holds them
    cache.free(donor)
    assert pool.used_blocks == 0


def test_only_whole_blocks_are_shared():
    """A partly-filled block still has to be written into. Sharing one would mean two
    conversations writing to the same memory — which reads as one quoting the other."""
    model = tiny()
    cache = PagedCache(pool_for(model))
    donor = Sequence(id=1, tokens=list(range(BLOCK_SIZE + 5)))
    cache.reserve(donor)
    donor.cached = BLOCK_SIZE + 5
    borrower = Sequence(id=2, tokens=list(range(BLOCK_SIZE + 5)))
    assert cache.share_prefix(borrower, donor, BLOCK_SIZE + 5) == BLOCK_SIZE

    # ...and a prefix shorter than one block shares *nothing*. Rounding up here would hand
    # over a block whose second half the donor is still writing into.
    short = Sequence(id=3, tokens=list(range(BLOCK_SIZE - 6)))
    cache.reserve(short)
    short.cached = BLOCK_SIZE - 6
    taker = Sequence(id=4, tokens=list(range(BLOCK_SIZE - 6)))
    assert cache.share_prefix(taker, short, BLOCK_SIZE - 6) == 0
    assert taker.blocks == []


# ---- the scheduler ---------------------------------------------------------------------

def test_a_finished_sequence_frees_its_blocks_on_the_step_it_finishes():
    """Not at the end of the batch. The whole point of continuous batching is that the
    memory and the slot come back immediately, so the next request can start."""
    model = tiny()
    engine = BatchEngine(model, pool_for(model, n_blocks=32), max_batch=4, device="cpu")
    engine.submit(Request(prompt_ids=[1, 2, 3], max_new_tokens=2, temperature=0.0))
    engine.submit(Request(prompt_ids=[4, 5, 6], max_new_tokens=20, temperature=0.0))
    for _ in range(3):
        engine.step()
    assert engine.stats.finished == 1
    assert len(engine.running) == 1                    # the long one is still going
    assert engine.cache.pool.used_blocks == 1          # only its blocks remain


def test_admission_waits_for_room_instead_of_evicting_someone_mid_answer():
    """A server that admits optimistically has to throw out work a person is already
    reading. This one leaves the request in the queue and says so in the stats."""
    model = tiny()
    engine = BatchEngine(model, pool_for(model, n_blocks=2), max_batch=8, device="cpu")
    engine.submit(Request(prompt_ids=list(range(20)), max_new_tokens=4, temperature=0.0))
    engine.submit(Request(prompt_ids=list(range(20)), max_new_tokens=4, temperature=0.0))
    engine.step()
    assert len(engine.running) == 1 and len(engine.waiting) == 1
    assert engine.stats.rejected >= 1
    # ...and the queued one still runs, once the first finishes and gives its blocks back.
    list(engine.run())
    assert engine.stats.finished == 2


def test_the_batch_is_the_point_and_the_stats_say_so():
    model = tiny()
    engine = BatchEngine(model, pool_for(model), max_batch=8, device="cpu")
    reqs = [Request(prompt_ids=[i + 1, 3], max_new_tokens=6, temperature=0.0)
            for i in range(6)]
    engine.collect(reqs)
    # Six sequences advanced by each pass over the weights, not one.
    assert engine.stats.tokens_per_step > 4
    assert engine.stats.mean_batch > 4
    assert engine.stats.finished == 6


def test_a_prompt_longer_than_the_window_is_trimmed_not_refused():
    model = tiny(max_seq_len=64)
    engine = BatchEngine(model, pool_for(model), max_batch=2, device="cpu")
    seq = engine.submit(Request(prompt_ids=[i % 60 for i in range(200)], max_new_tokens=4,
                                temperature=0.0))
    assert seq.length < model.cfg.max_seq_len
    list(engine.run())
    assert seq.finished


def test_per_request_sampling_is_per_request():
    """Two clients of one server have no reason to agree about temperature, and one asking
    for randomness must not make the other's greedy answer random."""
    model = tiny()
    greedy = generate(model, [5, 9], max_new_tokens=6, temperature=0.0,
                      device="cpu")[2:]
    engine = BatchEngine(model, pool_for(model), max_batch=4, device="cpu")
    torch.manual_seed(0)
    reqs = [Request(prompt_ids=[5, 9], max_new_tokens=6, temperature=0.0),
            Request(prompt_ids=[5, 9], max_new_tokens=6, temperature=2.0)]
    out = engine.collect(reqs)
    assert out[reqs[0].id] == greedy


# ---- the HTTP surface --------------------------------------------------------------------

def test_the_server_speaks_the_api_clients_already_know(tmp_path, monkeypatch):
    """The endpoints exist, in the shape a client library expects, and generation works
    through them end to end.

    A real socket on a real port: the value of this test is that it exercises the worker
    thread, the per-request queues and the JSON shape at once — the three things a unit test
    of `BatchEngine` cannot see.
    """
    import json as _json
    import threading
    import urllib.request
    from http.server import ThreadingHTTPServer

    from aksharallm.serve import server as srv_mod

    model = tiny()
    fake = object.__new__(srv_mod.ModelServer)      # skip loading a checkpoint from disk
    fake.store = None
    fake.ckpt_id = "test/ckpt.pt"
    fake.device = "cpu"
    fake.plan = type("P", (), {"device": "cpu", "reason": "test", "training": []})()
    fake.model = model
    fake.tokenizer = _ToyTokenizer(model.cfg.vocab_size)
    fake.pool = pool_for(model, n_blocks=64)
    fake.engine = BatchEngine(model, fake.pool, max_batch=4, device="cpu")
    fake.jobs, fake.lock = {}, threading.Lock()
    fake.wake, fake.stop = threading.Event(), threading.Event()
    fake.worker = threading.Thread(target=fake._loop, daemon=True)
    fake.worker.start()

    handler = type("H", (srv_mod.Handler,), {"server_ref": fake, "quiet": True})
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"

    try:
        health = _json.loads(urllib.request.urlopen(f"{base}/health").read())
        assert health["ok"] and health["kv_blocks"]["total"] == 64

        models = _json.loads(urllib.request.urlopen(f"{base}/v1/models").read())
        assert models["data"][0]["id"] == "test/ckpt.pt"

        req = urllib.request.Request(
            f"{base}/v1/completions",
            data=_json.dumps({"prompt": "abc", "max_tokens": 5, "temperature": 0}).encode(),
            headers={"Content-Type": "application/json"})
        body = _json.loads(urllib.request.urlopen(req).read())
        assert body["object"] == "text_completion"
        assert body["choices"][0]["finish_reason"] in ("length", "stop")
        assert body["usage"]["completion_tokens"] == 5

        # an unknown path is a 404 in the error shape a client can read
        try:
            urllib.request.urlopen(f"{base}/v1/nope")
            raise AssertionError("expected a 404")
        except urllib.error.HTTPError as exc:
            assert exc.code == 404
            assert "message" in _json.loads(exc.read())["error"]
    finally:
        httpd.shutdown()
        fake.shutdown()


class _ToyTokenizer:
    """Enough tokenizer for the HTTP test: one byte per token, no files on disk."""

    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size
        self.eos_id = None

    def encode(self, text: str, bos: bool = False) -> list[int]:
        return [ord(c) % self.vocab_size for c in text] or [1]

    def decode(self, ids, skip_special: bool = True) -> str:
        return "".join(chr(65 + (i % 26)) for i in ids)

    def render_chat(self, messages, add_generation_prompt=False):
        return self.encode(" ".join(m["content"] for m in messages)), []


# ---- speculation inside the batch ------------------------------------------------------

def test_drafting_inside_the_batch_changes_nothing_but_the_step_count():
    """The two speedups compose, and neither may change a token.

    Batching gets more *sequences* out of one pass over the weights; drafting gets more
    *tokens* out of one pass per sequence. Together they must still produce exactly what each
    request would have got alone — so this compares against the non-speculative batch, which
    `test_a_batch_gives_each_sequence_what_it_would_have_got_alone` already ties to
    single-sequence generation.
    """
    model = tiny()
    prompts = [[2, 3, 4, 5] * 4, [7, 1, 7, 1, 7, 1, 7], [9, 9, 9, 9, 9, 9]]

    plain = BatchEngine(model, pool_for(model), max_batch=8, device="cpu")
    plain_reqs = [Request(prompt_ids=p, max_new_tokens=16, temperature=0.0) for p in prompts]
    expected = plain.collect(plain_reqs)

    fast = BatchEngine(model, pool_for(model), max_batch=8, device="cpu", speculate=4)
    fast_reqs = [Request(prompt_ids=p, max_new_tokens=16, temperature=0.0) for p in prompts]
    got = fast.collect(fast_reqs)

    for a, b in zip(plain_reqs, fast_reqs):
        assert got[b.id] == expected[a.id]
    # ...and it actually did something: fewer passes over the weights for the same tokens.
    assert fast.stats.drafted > 0
    assert fast.stats.accepted > 0
    assert fast.stats.steps < plain.stats.steps
    assert fast.stats.tokens_per_step > plain.stats.tokens_per_step


def test_a_cancelled_sequence_stops_and_gives_its_blocks_back():
    """A client that hung up is not owed the rest of its answer, and every token it is still
    sent costs a slot in the batch and blocks a waiting request could have had."""
    model = tiny()
    engine = BatchEngine(model, pool_for(model, n_blocks=32), max_batch=4, device="cpu")
    a = engine.submit(Request(prompt_ids=[1, 2, 3], max_new_tokens=50, temperature=0.0))
    engine.submit(Request(prompt_ids=[4, 5, 6], max_new_tokens=4, temperature=0.0))
    engine.step()
    used = engine.cache.pool.used_blocks

    assert engine.cancel(a.id) is True
    assert a.finished and a.finish_reason == "cancelled"
    assert engine.cache.pool.used_blocks < used
    assert all(s.id != a.id for s in engine.running)
    assert engine.stats.cancelled == 1

    # It is gone from the loop entirely: no further tokens carry its id.
    assert all(seq_id != a.id for seq_id, _ in engine.run())
    assert engine.cancel(a.id) is False          # cancelling twice is a no-op, not a crash


def test_cancelling_a_queued_request_stops_it_being_admitted():
    """The queue is the other half: a request abandoned while waiting should never start."""
    model = tiny()
    engine = BatchEngine(model, pool_for(model, n_blocks=32), max_batch=1, device="cpu")
    engine.submit(Request(prompt_ids=[1, 2, 3], max_new_tokens=6, temperature=0.0))
    queued = engine.submit(Request(prompt_ids=[4, 5, 6], max_new_tokens=6, temperature=0.0))
    engine.step()
    assert len(engine.waiting) == 1

    assert engine.cancel(queued.id) is True
    assert not engine.waiting
    assert all(seq_id != queued.id for seq_id, _ in engine.run())
