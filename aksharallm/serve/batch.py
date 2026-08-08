"""Continuous batching: many conversations sharing one pass over the weights.

Decoding is memory-bound — a forward pass spends its time reading 300M parameters, not
multiplying by them. Reading them once to advance **one** sequence and reading them once to
advance **thirty** costs almost the same. That is the whole argument for batching, and it is
why a server's throughput can be twenty times a terminal's while each individual reply is no
faster at all.

*Continuous* batching is the part that matters in practice. The naive version runs a batch to
completion: thirty requests start together, twenty-nine finish, and the batch keeps stepping
one sequence while twenty-nine slots sit idle until the longest reply ends. Here, a finished
sequence leaves the batch on the step it finishes and a waiting one joins on the next — so
the batch is refilled continuously and a short request behind a long one waits for a slot,
not for the long one to end.

```
step 1   A B C            step 4   A   C D      (B finished, D admitted)
step 2   A B C            step 5   A   C D
step 3   A B C            step 6   A     D      (C finished)
```

Two rules the scheduler follows, both of them about *not* disturbing work in flight:

* **Admission is checked against free blocks, not hope.** A sequence is only admitted if the
  pool can hold its prompt now; otherwise it waits in the queue. A server that admits
  optimistically has to evict something mid-answer, and the sequence it evicts is one a
  person is reading.
* **Prefill and decode share a step.** A newly admitted sequence has a whole prompt to
  process while everyone else needs one token, so a step is ragged by construction: each row
  contributes however many tokens it owes, padded to the widest, with a mask that hides the
  padding. Separating the two phases would be simpler and would leave the card idle during
  every prefill.

Read with: docs/17-serving.md -- the chapter this implements; it ends with the order to read
these files in.
"""

from __future__ import annotations

import itertools
from contextlib import nullcontext
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from ..infer.generate import _filter_logits
from ..infer.speculative import NgramDrafter, accept_or_correct, next_distribution
from .paged import BlockPool, LayerView, OutOfBlocks, PagedCache, Sequence


@dataclass
class Request:
    """What a caller asked for. Sampling is per request, because two clients of one server
    have no reason to agree about temperature."""

    prompt_ids: list[int]
    max_new_tokens: int = 128
    temperature: float = 0.8
    top_k: int | None = 50
    top_p: float | None = 0.95
    eos_id: int | None = None
    #: Set by the engine; the caller reads it to match streamed tokens to its request.
    id: int = 0


@dataclass
class BatchStats:
    steps: int = 0
    tokens: int = 0                 #: generated tokens, across every sequence
    prefill_tokens: int = 0
    admitted: int = 0
    finished: int = 0
    rejected: int = 0               #: admissions deferred because the pool was full
    drafted: int = 0                #: tokens guessed ahead (0 unless `speculate` is on)
    accepted: int = 0               #: ...of which the model agreed with
    cancelled: int = 0              #: clients that hung up before their answer finished
    batch_sizes: list[int] = field(default_factory=list)

    @property
    def accept_rate(self) -> float:
        return self.accepted / self.drafted if self.drafted else 0.0

    @property
    def mean_batch(self) -> float:
        return sum(self.batch_sizes) / len(self.batch_sizes) if self.batch_sizes else 0.0

    @property
    def tokens_per_step(self) -> float:
        """The number the whole design exists to raise: generated tokens per pass over the
        weights. One is what a terminal gets."""
        return self.tokens / self.steps if self.steps else 0.0

    def as_dict(self) -> dict:
        return {"steps": self.steps, "tokens": self.tokens,
                "prefill_tokens": self.prefill_tokens, "admitted": self.admitted,
                "finished": self.finished, "rejected": self.rejected,
                "cancelled": self.cancelled, "drafted": self.drafted,
                "accepted": self.accepted, "accept_rate": self.accept_rate,
                "mean_batch": self.mean_batch, "tokens_per_step": self.tokens_per_step}


def sample_row(logits: torch.Tensor, temperature: float, top_k: int | None,
               top_p: float | None) -> int:
    """One row of logits → one token id, by the same rules as single-sequence generation.

    Deliberately the same filtering function `infer/generate.py` uses. A server that sampled
    even slightly differently would give different answers from the CLI for the same prompt
    and seed, and the difference would be blamed on the model.
    """
    logits = logits.float()
    if temperature <= 0.0:
        return int(logits.argmax())
    filtered = _filter_logits(logits[None, :] / temperature, top_k, top_p)
    return int(torch.multinomial(F.softmax(filtered[0], dim=-1), num_samples=1))


class BatchEngine:
    """A model, a pool of KV blocks, and the loop that advances every live sequence by a step.

    Holds no HTTP, no threads and no tokenizer: it takes token ids and yields token ids, so
    it can be driven by a server, a benchmark or a test with nothing mocked.
    """

    def __init__(self, model, pool: BlockPool, max_batch: int = 32,
                 device: str = "cuda", speculate: int = 0, ngram: int = 3):
        self.model = model
        self.model.eval()
        self.cache = PagedCache(pool)
        self.max_batch = max_batch
        self.device = device
        #: Tokens to guess per sequence per step, by looking the continuation up in that
        #: sequence's own text (`infer/speculative.py`). 0 is off. It composes with batching
        #: rather than competing with it: batching gets more *sequences* out of one pass over
        #: the weights, drafting gets more *tokens* out of one pass per sequence, and the two
        #: multiply. The ragged step already handles rows that owe different numbers of
        #: tokens, which is why this costs so little here.
        self.speculate = speculate
        self.drafters: dict[int, NgramDrafter] = {}
        self.ngram = ngram
        self.max_seq_len = model.cfg.max_seq_len
        self.running: list[Sequence] = []
        self.waiting: list[tuple[Sequence, Request]] = []
        self.params: dict[int, Request] = {}
        self.stats = BatchStats()
        self._ids = itertools.count(1)
        #: Sequences that ended since somebody last asked. A server needs to know *which*
        #: request finished and why; the engine keeps a list rather than taking a callback,
        #: so nothing about HTTP leaks into the loop.
        self._just_finished: list[Sequence] = []

    # ---- the queue -----------------------------------------------------------------------
    def submit(self, req: Request) -> Sequence:
        """Accept a request into the *queue*. Admission to the batch happens in `step`, when
        the pool's state is known."""
        req.id = req.id or next(self._ids)
        # The window is the model's, and a prompt that fills it leaves nowhere to answer.
        prompt = list(req.prompt_ids)[-(self.max_seq_len - 1):]
        seq = Sequence(id=req.id, tokens=prompt,
                       max_new_tokens=min(req.max_new_tokens,
                                          self.max_seq_len - len(prompt)))
        self.params[seq.id] = req
        self.waiting.append((seq, req))
        return seq

    def _admit(self) -> None:
        """Move as many waiting sequences into the batch as the pool can actually hold.

        Order is FIFO and the loop stops at the first sequence that does not fit, rather than
        skipping it to admit a smaller one behind it. Fair beats clever here: a long prompt
        that keeps being overtaken never runs at all.
        """
        while self.waiting and len(self.running) < self.max_batch:
            seq, _ = self.waiting[0]
            # One extra block of headroom: the sequence is about to generate into it, and
            # discovering that mid-step would mean unwinding a forward pass.
            need = self.cache.blocks_needed(seq, extra=1)
            if need > self.cache.pool.free_blocks:
                self.stats.rejected += 1
                return
            self._share_prefix(seq)
            self.cache.reserve(seq, extra=1)
            self.waiting.pop(0)
            self.running.append(seq)
            self.stats.admitted += 1

    def _share_prefix(self, new: Sequence) -> None:
        """Point a new sequence at an existing one's blocks where their prompts agree.

        The common case in a real server is not a coincidence: every request carries the same
        system prompt, so the first few hundred tokens of every conversation are identical and
        would otherwise be recomputed and stored per conversation.
        """
        best, best_n = None, 0
        for other in self.running:
            n = 0
            for a, b in zip(new.tokens, other.tokens[:other.cached]):
                if a != b:
                    break
                n += 1
            if n > best_n:
                best, best_n = other, n
        if best is not None and best_n >= self.cache.block_size:
            self.cache.share_prefix(new, best, best_n)

    # ---- one pass over the weights --------------------------------------------------------
    def step(self) -> list[tuple[int, int]]:
        """Advance every running sequence by one token. Returns `(sequence id, token)` pairs.

        Ragged by construction: a freshly admitted sequence owes its whole prompt while the
        others owe one token each, so rows are padded to the widest and the mask hides it.
        """
        self._admit()
        seqs = [s for s in self.running if not s.finished]
        if not seqs:
            return []

        starts = [s.cached for s in seqs]
        drafts = self._draft(seqs) if self.speculate else [([], []) for _ in seqs]
        lengths = [s.length for s in seqs]
        logits = self._forward(seqs, starts, lengths)
        out: list[tuple[int, int]] = []
        for i, seq in enumerate(seqs):
            req = self.params[seq.id]
            guesses, probs = drafts[i]
            # The row's last *non-drafted* query: with no drafts this is the final row, and
            # with g drafts it is g rows earlier — the position that predicts the first guess.
            last = lengths[i] - starts[i] - 1 - len(guesses)
            self.stats.prefill_tokens += last
            base = len(seq.tokens) - len(guesses)    # the sequence without its guesses

            accepted: list[int] = []
            emitted: int | None = None
            for j, guess in enumerate(guesses):
                p = self._dist(logits[i, last + j], req)
                ok, replacement = accept_or_correct(p, probs[j].to(p.device), guess)
                if ok:
                    accepted.append(guess)
                    continue
                emitted = replacement
                break
            if emitted is None:
                # Every guess survived (or there were none), so the row after the last one is
                # already computed and its token is free.
                emitted = sample_row(logits[i, last + len(guesses)], req.temperature,
                                     req.top_k, req.top_p)
            self.stats.drafted += len(guesses)
            self.stats.accepted += len(accepted)

            # A round can emit several tokens at once, so both endings have to be honoured
            # *inside* it: everything after an EOS is discarded, and so is anything past the
            # caller's budget. Without this a request for 16 tokens gets 17 whenever the last
            # round happened to accept two — output that is correct token for token and one
            # token too long, which is the sort of bug a diff finds and a reader does not.
            produced = accepted + [emitted]
            if req.eos_id is not None and req.eos_id in produced:
                produced = produced[:produced.index(req.eos_id) + 1]
            room = max(0, seq.max_new_tokens - seq.generated)
            produced = produced[:room]

            # Keep what survived, drop the rest. The accepted guesses' keys and values were
            # written during this very forward and are correct *because* the prefix was; the
            # rejected ones sit past `cached`, where `gather` never looks, and are overwritten
            # by the next step. That is paging paying off: no rewind, no copy, one integer.
            kept_cached = min(base + len(accepted), base + len(produced))
            seq.tokens = seq.tokens[:base] + produced
            seq.cached = kept_cached
            seq.generated += len(produced)
            self.stats.tokens += len(produced)
            for token in produced:
                out.append((seq.id, token))

            eos = req.eos_id is not None and req.eos_id in produced
            if eos:
                self._finish(seq, "stop")
            elif seq.generated >= seq.max_new_tokens or seq.length >= self.max_seq_len:
                self._finish(seq, "length")
            else:
                try:
                    self.cache.reserve(seq, extra=1 + self.speculate)
                except OutOfBlocks:
                    # The pool filled while this sequence was mid-answer. Ending it cleanly
                    # and saying so beats corrupting it or dying: the caller gets its tokens
                    # so far and a reason.
                    self._finish(seq, "out_of_memory")

        self.stats.steps += 1
        self.stats.batch_sizes.append(len(seqs))
        return out

    def _dist(self, row: torch.Tensor, req: Request) -> torch.Tensor:
        """The distribution this request would sample from — the same one `sample_row` uses.

        Acceptance compares the target's *sampled* distribution with the draft's, so it has to
        be built the same way here or the output stops being the model's.
        """
        return next_distribution(row, set(), req.temperature, req.top_k, req.top_p, 1.0)

    def _draft(self, seqs: list[Sequence]) -> list[tuple[list[int], list[torch.Tensor]]]:
        """Guess the next few tokens of every sequence, and append them to be verified.

        Per sequence, because a batch is unrelated conversations and a lookup in one of them
        says nothing about another. The guesses go straight onto `seq.tokens`, which is what
        makes the existing ragged step verify them with no special case: a row that owes four
        guesses looks exactly like a row that owes four prompt tokens.
        """
        out = []
        for seq in seqs:
            drafter = self.drafters.get(seq.id)
            if drafter is None:
                drafter = self.drafters[seq.id] = NgramDrafter(
                    self.model.cfg.vocab_size, n=self.ngram, device=self.device)
            room = min(self.speculate, self.max_seq_len - seq.length - 1)
            guesses, probs = drafter.propose(seq.tokens, max(0, room), None) if room > 0 \
                else ([], [])
            seq.tokens.extend(guesses)
            out.append((guesses, probs))
        return out

    def _forward(self, seqs: list[Sequence], starts: list[int],
                 lengths: list[int]) -> torch.Tensor:
        """One pass over the weights for a ragged batch. Returns `(B, widest, vocab)`.

        Separate from `step` because this is the half that is *arithmetic* — positions, the
        mask, the paged reads — while `step` is bookkeeping. A test can call it and compare
        the numbers against a contiguous cache, which is far more sensitive than comparing
        sampled tokens: an address wrong by one block often leaves the greedy answer alone.
        """
        widest = max(l - c for l, c in zip(lengths, starts))
        longest = max(lengths)

        idx = torch.zeros(len(seqs), widest, dtype=torch.long, device=self.device)
        for i, seq in enumerate(seqs):
            owed = seq.tokens[starts[i]:]
            idx[i, :len(owed)] = torch.tensor(owed, dtype=torch.long, device=self.device)

        q_pos = (torch.tensor(starts, device=self.device)[:, None]
                 + torch.arange(widest, device=self.device)[None, :])
        k_pos = torch.arange(longest, device=self.device)
        len_t = torch.tensor(lengths, device=self.device)[:, None, None]
        # A query may see every key up to its own position, and nothing past the end of its
        # own sequence. Padding columns are masked out; padding *rows* are given key 0 so the
        # softmax has something to normalise — an all-False row is a NaN, and a NaN in one
        # padded row poisons the whole batch through the shared weights.
        mask = (k_pos[None, None, :] <= q_pos[:, :, None]) & (k_pos[None, None, :] < len_t)
        real = torch.tensor([l - c for l, c in zip(lengths, starts)], device=self.device)
        padded_rows = torch.arange(widest, device=self.device)[None, :] >= real[:, None]
        mask = mask.clone()
        mask[padded_rows] = False
        mask[padded_rows.unsqueeze(-1).expand_as(mask) & (k_pos[None, None, :] == 0)] = True

        views = [LayerView(self.cache, layer, seqs, starts, lengths)
                 for layer in range(len(self.model.blocks))]
        # The same autocast the single-sequence path uses. It is not an optimisation here but
        # a *correctness* requirement: the KV pool is bf16 on the card, so a query computed in
        # fp32 cannot be multiplied by it — attention refuses to mix dtypes, which is a much
        # better failure than silently upcasting the whole cache.
        ctx = (torch.autocast("cuda", dtype=torch.bfloat16)
               if self.device.startswith("cuda") else nullcontext())
        with torch.no_grad(), ctx:
            logits, _ = self.model(idx, caches=views, full_logits=True,
                                   positions=q_pos.clamp(max=self.max_seq_len - 1),
                                   attn_mask=mask[:, None, :, :])
        return logits

    def _finish(self, seq: Sequence, reason: str) -> None:
        seq.finished = True
        seq.finish_reason = reason
        self.cache.free(seq)
        self.running = [s for s in self.running if s is not seq]
        self.stats.finished += 1
        if reason == "cancelled":
            self.stats.cancelled += 1
        self._just_finished.append(seq)
        self.drafters.pop(seq.id, None)

    def take_finished(self) -> list[Sequence]:
        """Sequences that ended since the last call, and clear the list."""
        out, self._just_finished = self._just_finished, []
        return out

    def cancel(self, seq_id: int) -> bool:
        """Stop a sequence and hand its memory back now. Returns whether it found one.

        A client that closes its connection mid-answer has stopped caring about the rest, and
        every further token it is sent costs a slot in the batch and blocks in the pool that a
        waiting request could have had. Before this existed such a sequence ran to its full
        `max_tokens` into a socket nobody was reading — which is *safe* and quietly halves the
        throughput of a busy server.

        Cancelling from the queue is just as important as cancelling from the batch: a request
        that waited for a slot and was abandoned while waiting should never be admitted at all.
        """
        for seq in self.running:
            if seq.id == seq_id:
                self._finish(seq, "cancelled")
                return True
        for i, (seq, _req) in enumerate(self.waiting):
            if seq.id == seq_id:
                self.waiting.pop(i)
                seq.finished = True
                seq.finish_reason = "cancelled"
                self._just_finished.append(seq)
                self.stats.cancelled += 1
                return True
        return False

    # ---- driving it -------------------------------------------------------------------------
    @property
    def busy(self) -> bool:
        return bool(self.running or self.waiting)

    def run(self, max_steps: int = 10_000):
        """Step until everything submitted has finished, yielding `(id, token)` as they come.

        A generator rather than a callback: the server streams what it yields straight to the
        client, and a test can simply collect it.
        """
        for _ in range(max_steps):
            if not self.busy:
                return
            for pair in self.step():
                yield pair

    def collect(self, requests: list[Request]) -> dict[int, list[int]]:
        """Every request's generated tokens, keyed by request id — the batch equivalent of
        `generate()`."""
        for req in requests:
            self.submit(req)
        out: dict[int, list[int]] = {r.id: [] for r in requests}
        for seq_id, token in self.run():
            out.setdefault(seq_id, []).append(token)
        return out
