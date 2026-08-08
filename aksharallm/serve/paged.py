"""A paged KV cache: memory for many conversations at once, without reserving for the worst.

`infer/generate.py` gives every sequence a contiguous cache sized for the **whole context
window**, because there is only ever one sequence and the allocation happens once. Serving
breaks that immediately. A server holding 32 conversations would reserve 32 full windows, so
the memory bill is set by the longest reply anyone *might* write rather than what they
actually wrote — and 30 of those 32 are eight tokens into a question.

The fix is the one operating systems made decades ago: **pages**. Keys and values live in
fixed-size blocks drawn from one pool, and a sequence holds a *block table* — a list of block
ids, in order. Nothing is contiguous, nothing is reserved, and a sequence grows by taking one
more block from the free list every `block_size` tokens.

    pool     [ b0 ][ b1 ][ b2 ][ b3 ][ b4 ][ b5 ]  ...   one flat tensor per layer
    seq A    [ b0 ][ b3 ][ b4 ]                          40 tokens, 3 blocks
    seq B    [ b1 ]                                      9 tokens, 1 block
    free     [ b2 ][ b5 ] ...

Two things fall out of it, and both are the reason to build it rather than pad a batch:

* **The waste is bounded by one block per sequence** (16 tokens here) instead of by the
  context window. At 32 sequences and a 1024-token window that is the difference between
  reserving 32 windows and reserving what is used.
* **Two sequences can share a block.** A system prompt tokenizes to the same ids every time,
  so the blocks holding it can be pointed at by every conversation that starts with it and
  computed once. `share_prefix` does that, with a reference count so the last user to let go
  is the one that frees it.

The honest caveat, stated where it cannot be missed: gathering scattered blocks into the
dense `(B, H, S, D)` tensor that attention wants is a **copy**, done here in PyTorch. Real
serving stacks write a custom kernel that reads the blocks in place. This implementation is
about being correct and legible about the bookkeeping — see `docs/17-serving.md` for what
that costs.

Read with: docs/17-serving.md -- the chapter this implements; it ends with the order to read
these files in.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

#: Tokens per block. Small enough that the last, partly-filled block of a sequence wastes
#: little; large enough that a 1,000-token sequence is 60-odd table entries rather than a
#: thousand. 16 is what the paged-attention literature settled on for the same reasons.
BLOCK_SIZE = 16


class OutOfBlocks(Exception):
    """The pool is full. A server turns this into "come back later", never a crash: the
    sequences already running are mid-answer and must not be disturbed by a new arrival."""


class BlockPool:
    """Every block of key/value memory the server owns, for every layer, allocated once.

    One tensor per layer, shaped `(n_blocks, n_kv_heads, block_size, head_dim)`. The whole
    pool is allocated up front and never grows: that is the point of a server — you decide
    what memory you are willing to spend, and admission control does the rest. Running out is
    a scheduling decision, not an out-of-memory error four days into a run.
    """

    def __init__(self, n_layers: int, n_blocks: int, n_kv_heads: int, head_dim: int,
                 block_size: int = BLOCK_SIZE, dtype=torch.bfloat16, device: str = "cuda"):
        self.n_layers = n_layers
        self.n_blocks = n_blocks
        self.block_size = block_size
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.device = device
        # One flat slot axis per layer — `(n_kv_heads, n_blocks * block_size, head_dim)` —
        # rather than a block dimension of its own. Blocks are then purely an *addressing*
        # convention: block `b` owns slots `[b*block_size, (b+1)*block_size)`, and reading or
        # writing any set of tokens is one indexing operation into a contiguous tensor.
        #
        # The shape matters more than it looks. With a `(n_blocks, n_kv_heads, ...)` pool the
        # natural view is `pool.transpose(0, 1).reshape(...)` — and **reshape on a transposed
        # tensor returns a copy**, so every write landed in a temporary and the pool stayed
        # full of zeros. The model then attended to nothing but the token it had just been
        # given and repeated it forever, which reads as a bad model rather than as a bug.
        shape = (n_kv_heads, n_blocks * block_size, head_dim)
        self.k = [torch.zeros(shape, dtype=dtype, device=device) for _ in range(n_layers)]
        self.v = [torch.zeros(shape, dtype=dtype, device=device) for _ in range(n_layers)]
        self._free = list(range(n_blocks))
        #: How many sequences point at each block. Zero means it is on the free list; one is
        #: the ordinary case; more than one is a shared prefix.
        self.refs = [0] * n_blocks

    # ---- accounting -------------------------------------------------------------------
    @property
    def free_blocks(self) -> int:
        return len(self._free)

    @property
    def used_blocks(self) -> int:
        return self.n_blocks - len(self._free)

    def bytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self.k + self.v)

    def allocate(self, n: int = 1) -> list[int]:
        """Take `n` blocks, or take none at all.

        All-or-nothing on purpose: a half-allocated sequence would have to be unwound by the
        caller, and every caller would have to remember to.
        """
        if n > len(self._free):
            raise OutOfBlocks(
                f"asked for {n} block(s) with {len(self._free)} free of {self.n_blocks}. "
                f"Wait for a sequence to finish, or start the server with a bigger pool.")
        taken = [self._free.pop() for _ in range(n)]
        for b in taken:
            self.refs[b] = 1
        return taken

    def release(self, blocks: list[int]) -> None:
        """Give blocks back, one reference at a time. A shared block only returns to the free
        list when its last holder lets go — which is what makes prefix sharing safe to use
        without tracking who copied what."""
        for b in blocks:
            if self.refs[b] <= 0:
                continue                      # already free: releasing twice is not fatal
            self.refs[b] -= 1
            if self.refs[b] == 0:
                self._free.append(b)

    def share(self, blocks: list[int]) -> list[int]:
        """Take another reference to blocks somebody else already holds."""
        for b in blocks:
            self.refs[b] += 1
        return list(blocks)


@dataclass
class Sequence:
    """One request in flight: its tokens, the blocks holding their keys and values, and how
    much of it the model has actually seen."""

    id: int
    tokens: list[int]
    #: Blocks in order. Token `i` of the sequence lives in `blocks[i // block_size]` at
    #: offset `i % block_size` — the whole address translation, in one line.
    blocks: list[int] = field(default_factory=list)
    #: Tokens whose keys and values are in the pool. Always <= len(tokens): the difference is
    #: what the next forward pass has to compute.
    cached: int = 0
    max_new_tokens: int = 128
    generated: int = 0
    finished: bool = False
    finish_reason: str | None = None
    #: Blocks that came from another sequence's prefix and must not be written into.
    shared: int = 0

    @property
    def length(self) -> int:
        return len(self.tokens)

    @property
    def pending(self) -> int:
        """Tokens the model has not processed yet — the whole prompt at first, one token per
        step after that."""
        return len(self.tokens) - self.cached


class PagedCache:
    """The block pool, plus the address translation that makes it look like a KV cache.

    This is the piece the model talks to. It offers exactly two operations per layer —
    *write these new keys and values into the right blocks*, and *gather everything this
    batch may attend to into one dense tensor* — and hides the fact that the second one is
    stitching scattered pages back together.
    """

    def __init__(self, pool: BlockPool):
        self.pool = pool
        self.block_size = pool.block_size

    # ---- growing a sequence -------------------------------------------------------------
    def blocks_needed(self, seq: Sequence, extra: int = 0) -> int:
        total = len(seq.tokens) + extra
        want = (total + self.block_size - 1) // self.block_size
        return max(0, want - len(seq.blocks))

    def reserve(self, seq: Sequence, extra: int = 0) -> None:
        """Make sure the sequence has room for its tokens (plus `extra` about to arrive)."""
        need = self.blocks_needed(seq, extra)
        if need:
            seq.blocks.extend(self.pool.allocate(need))

    def free(self, seq: Sequence) -> None:
        self.pool.release(seq.blocks)
        seq.blocks = []
        seq.cached = 0

    def share_prefix(self, seq: Sequence, donor: Sequence, n_tokens: int) -> int:
        """Point `seq` at `donor`'s first `n_tokens` of keys and values instead of computing
        them again.

        Only *whole* blocks can be shared — a partly-filled block still has to be written
        into, and writing into a block someone else is reading is how one conversation ends
        up quoting another. So the shared prefix is rounded **down** to a block boundary, and
        the tokens in the leftover are simply recomputed. Returns how many were shared.
        """
        usable = min(n_tokens, donor.cached) // self.block_size
        if usable <= 0 or seq.blocks:
            return 0
        seq.blocks = self.pool.share(donor.blocks[:usable])
        seq.shared = usable
        seq.cached = usable * self.block_size
        return seq.cached

    # ---- the address translation ---------------------------------------------------------
    def _slots(self, seq: Sequence, start: int, count: int) -> torch.Tensor:
        """Flat positions in the pool for `count` tokens of `seq` starting at `start`.

        The pool is viewed as one long axis of `n_blocks * block_size` slots, so writing and
        gathering are both a single `index_select` rather than a loop over blocks. This is
        the whole of paging, and it is four lines.
        """
        idx = torch.arange(start, start + count, device=self.pool.device)
        block = torch.tensor([seq.blocks[i // self.block_size]
                              for i in range(start, start + count)],
                             device=self.pool.device, dtype=torch.long)
        return block * self.block_size + (idx % self.block_size)

    def write(self, layer: int, seq: Sequence, start: int,
              k: torch.Tensor, v: torch.Tensor) -> None:
        """Store `k`/`v` — shaped `(n_kv_heads, count, head_dim)` — for a run of tokens."""
        count = k.shape[1]
        if count == 0:
            return
        slots = self._slots(seq, start, count)
        pool_k, pool_v = self.pool.k[layer], self.pool.v[layer]
        pool_k[:, slots] = k.to(pool_k.dtype)
        pool_v[:, slots] = v.to(pool_v.dtype)

    def gather(self, layer: int, seqs: list[Sequence],
               lengths: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        """Every sequence's keys and values, padded to the longest, as `(B, H, S, D)`.

        The copy the module docstring warns about. It is honest work — the alternative is a
        custom attention kernel that walks block tables, which is a different project — and
        it is done once per layer per step rather than once per sequence.
        """
        longest = max(lengths) if lengths else 0
        pool_k, pool_v = self.pool.k[layer], self.pool.v[layer]
        # One indexing operation for the whole batch, not one per sequence. With 24 layers and
        # 32 sequences the per-sequence version issued 768 gathers per step and spent more
        # time in Python than the model spent in matmuls — batch 8 was barely faster than no
        # batching at all. The address table is built once by `batch_slots` and reused for
        # every layer.
        slots = self.batch_slots(seqs, lengths, longest)
        flat = slots.reshape(-1)
        heads, dim = self.pool.n_kv_heads, self.pool.head_dim
        k = pool_k[:, flat].reshape(heads, len(seqs), longest, dim).permute(1, 0, 2, 3)
        v = pool_v[:, flat].reshape(heads, len(seqs), longest, dim).permute(1, 0, 2, 3)
        return k, v

    def batch_slots(self, seqs: list[Sequence], lengths: list[int],
                    longest: int) -> torch.Tensor:
        """`(B, longest)` pool slots, padded with slot 0.

        Padding with a real slot rather than a sentinel keeps the gather branch-free; the
        attention mask is what makes those columns invisible, and it is built from the same
        `lengths`.
        """
        rows = []
        bs = self.block_size
        for seq, n in zip(seqs, lengths):
            row = [seq.blocks[i // bs] * bs + (i % bs) for i in range(n)]
            row.extend([0] * (longest - n))
            rows.append(row)
        return torch.tensor(rows, dtype=torch.long, device=self.pool.device)


class LayerView:
    """What one layer of the model sees: a `KVCache`-shaped object backed by the pool.

    `Transformer` asks a cache for `update(k, v) -> (all_k, all_v)` and reads `.pos`. Meeting
    that interface rather than changing the model is deliberate — the architecture should not
    have to know whether its memory is one slab or sixty scattered pages, and the same weights
    then serve a batch of thirty and a single terminal session.
    """

    def __init__(self, cache: PagedCache, layer: int, seqs: list[Sequence],
                 starts: list[int], lengths: list[int]):
        self.cache = cache
        self.layer = layer
        self.seqs = seqs
        self.starts = starts
        self.lengths = lengths
        #: The model reads this to decide RoPE offsets and masking when it is *not* given
        #: explicit positions. The serving path always gives them, so it is only ever the
        #: assertion's business — but it must not lie, so it is the longest row.
        self.pos = max(lengths) if lengths else 0

    def update(self, k: torch.Tensor, v: torch.Tensor):
        for i, seq in enumerate(self.seqs):
            n = self.lengths[i] - self.starts[i]
            if n > 0:
                self.cache.write(self.layer, seq, self.starts[i], k[i, :, :n], v[i, :, :n])
        return self.cache.gather(self.layer, self.seqs, self.lengths)
