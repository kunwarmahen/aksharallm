"""Near-duplicate detection over the training corpus — MinHash and LSH, from scratch.

A web crawl is full of the same text twice. Not byte-identical twice — that would be easy —
but the same article behind three different templates, the same licence header on ten
thousand files, the same Stack Overflow answer quoted in four blog posts. Published
deduplication work finds it worth a measurable amount of loss, and the reason is not subtle:
a duplicated document is trained on twice, which is a silent, unrequested extra epoch on
whatever happened to be popular.

Exact matching cannot see any of it. Comparing every pair cannot either — 8 million
documents is 32 trillion pairs. What works is two ideas stacked:

```mermaid
flowchart LR
    D["a document"] --> S["k-token shingles<br/>hashed to 64 bits"]
    S --> M["MinHash: for each of P<br/>hash functions, keep the MINIMUM"]
    M --> G["signature<br/>P integers"]
    G --> B["LSH: split into B bands<br/>of R rows"]
    B --> C["two docs are CANDIDATES if<br/>any band matches exactly"]
    C --> V["verify the candidates"]
```

**MinHash.** Take a document's set of shingles, hash them all with one function, keep the
minimum. Do that for P different functions. The probability that two documents share a
minimum, for a random hash function, is *exactly* their Jaccard similarity — so the fraction
of the P positions where two signatures agree is an unbiased estimate of it. A document of
any length collapses to P integers, and similarity becomes a comparison of P numbers.

**LSH.** Even P-integer comparisons are quadratic. So split each signature into `B` bands of
`R` rows and index each band. Two documents are *candidates* if any one band matches exactly.
The probability of that is `1 - (1 - t^R)^B`, an S-curve in the true similarity `t`: it is
near zero below the knee and near one above it, and the knee sits at roughly `(1/B)^(1/R)`.
Choosing B and R is choosing where the curve turns.

Four things that make this lie, all reported rather than hidden
----------------------------------------------------------------
1. **MinHash estimates Jaccard; it does not compute it.** The standard error is
   `sqrt(t(1-t)/P)` — at P=128 and t=0.8 that is ±3.5 points. `--verify` recomputes exact
   Jaccard on the candidate pairs, which is affordable *because* LSH already threw most of
   them away, and is the same argument `eval/contamination.py` makes for its own verifier.
2. **LSH has false negatives and they are invisible.** A genuinely similar pair that happens
   to share no band is never compared, so it does not appear as a near-miss — it does not
   appear at all. `detection_probability()` gives the S-curve so the miss rate at your
   threshold is a number you can quote instead of a hope.
3. **A duplicate cluster is not a duplicate count.** Ten copies of one document are one
   cluster of ten, and "how many documents are duplicates" is nine, not ten — one of them is
   the original and keeping it is the point. The report says both.
4. **Documents come from EOS boundaries.** `prepare.py` writes an EOS after every document,
   so the token stream can be split back. A corpus without them would look like one enormous
   document, so that is checked rather than assumed.

Read with: docs/02-data.md -- the chapter this implements; it ends with the order to read
these files in.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

#: A large prime under 2^61, for the universal hash family `h(x) = (a·x + b) mod p`. Under
#: 2^61 so that `a·x + b` cannot overflow int64 once the operands are reduced.
PRIME = (1 << 61) - 1

#: Multiplier for the shingle rolling hash. Same Horner trick as `eval/contamination.py`:
#: the loop runs `k` times, not once per token, which is what makes this cross a corpus.
_MULT = np.uint64(1_000_003)

#: Shingle width, in tokens. Wide enough that a shared shingle means a shared *phrase* and
#: not a shared word; narrow enough that a paraphrase still shares some. 8 tokens is roughly
#: five or six words of prose.
SHINGLE = 8


# ---------------------------------------------------------------------------------------
# the parameters, and what they mean
# ---------------------------------------------------------------------------------------


@dataclass(frozen=True)
class LSHParams:
    """`bands × rows` permutations, and the threshold that implies."""

    bands: int = 16
    rows: int = 8

    @property
    def permutations(self) -> int:
        return self.bands * self.rows

    @property
    def threshold(self) -> float:
        """The knee of the S-curve, `(1/B)^(1/R)` — the usual rule of thumb.

        It is a rule of thumb and not a guarantee: the curve is smooth, so pairs a little
        below the knee are sometimes found and pairs a little above are sometimes missed.
        `detection_probability` is the honest version.
        """
        return (1.0 / self.bands) ** (1.0 / self.rows)

    def detection_probability(self, similarity: float) -> float:
        """`1 - (1 - t^R)^B` — the chance a pair this similar becomes a candidate at all."""
        return 1.0 - (1.0 - similarity**self.rows) ** self.bands

    def curve(self, points=(0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95)) -> list[dict]:
        return [{"similarity": s, "detected": self.detection_probability(s)} for s in points]

    def standard_error(self, similarity: float) -> float:
        """How far a signature's estimate of Jaccard typically is from the truth."""
        return math.sqrt(max(similarity * (1 - similarity), 0.0) / self.permutations)


# ---------------------------------------------------------------------------------------
# shingles and signatures
# ---------------------------------------------------------------------------------------


def shingle_hashes(tokens: np.ndarray, k: int = SHINGLE) -> np.ndarray:
    """Every k-token window of `tokens`, as a 64-bit hash. `(n - k + 1,)` uint64.

    Horner over the whole array at once — the Python loop runs `k` times, not once per
    token. Duplicates are *kept* here and removed by the caller, because `np.unique` on a
    short document is more expensive than the shingling was.
    """
    tokens = np.asarray(tokens, dtype=np.uint64)
    if tokens.size < k:
        return np.empty(0, dtype=np.uint64)
    out = np.zeros(tokens.size - k + 1, dtype=np.uint64)
    for i in range(k):
        out *= _MULT
        out += tokens[i : tokens.size - k + 1 + i]
    return out


def hash_family(n: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """`n` pairs `(a, b)` for the universal family `h(x) = (a·x + b) mod PRIME`.

    Seeded, so two runs over one corpus produce the same signatures and therefore the same
    clusters. A dedup report that changes when you rerun it is not a measurement.
    """
    rng = np.random.default_rng(seed)
    a = rng.integers(1, PRIME, size=n, dtype=np.int64)  # a != 0, or h is constant
    b = rng.integers(0, PRIME, size=n, dtype=np.int64)
    return a, b


def signature(hashes: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """MinHash signature: the minimum of each hash function over the shingle set.

    Computed as one `(P, n_shingles)` broadcast rather than a loop over P — which is the
    difference between a document taking a millisecond and taking a tenth of a second, and
    at eight million documents that is the difference between hours and days.

    Shingles are reduced mod PRIME *before* the multiply so `a·x + b` cannot overflow.
    """
    if hashes.size == 0:
        # An empty document matches nothing rather than everything: PRIME is larger than any
        # real minimum, so no band of it can collide with a real document's.
        return np.full(a.size, PRIME, dtype=np.int64)
    x = (hashes % np.uint64(PRIME)).astype(np.int64)
    return ((a[:, None] * x[None, :] + b[:, None]) % PRIME).min(axis=1)


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    """Exact Jaccard between two shingle-hash sets — the verifier, not the estimator."""
    sa, sb = np.unique(a), np.unique(b)
    if sa.size == 0 and sb.size == 0:
        return 1.0
    inter = np.intersect1d(sa, sb, assume_unique=True).size
    union = sa.size + sb.size - inter
    return inter / union if union else 0.0


def estimated_jaccard(sig_a: np.ndarray, sig_b: np.ndarray) -> float:
    """The fraction of signature positions that agree — an unbiased estimate of Jaccard."""
    return float((sig_a == sig_b).mean())


# ---------------------------------------------------------------------------------------
# splitting a token stream into documents
# ---------------------------------------------------------------------------------------


def document_spans(tokens: np.ndarray, eos_id: int, *, min_tokens: int = 32,
                   max_tokens: int | None = None) -> list[tuple[int, int]]:
    """`[(start, end)]` for each document in an EOS-separated stream.

    `min_tokens` drops fragments: a 6-token "document" is a stub, it shares shingles with
    everything, and it would dominate a duplicate report with noise. `max_tokens` truncates
    long ones — a documented approximation that bounds the work per document, and which
    biases *towards* finding duplicates (two long documents that differ only after the
    truncation point will look identical), so it is reported rather than silent.
    """
    ends = np.flatnonzero(tokens == eos_id)
    if ends.size == 0:
        raise ValueError(
            "no EOS token in this stream, so it has no document boundaries. "
            "`prepare.py` writes one after every document; a corpus without them would be "
            "treated as a single enormous document and every number here would be nonsense."
        )
    spans, start = [], 0
    for e in ends:
        end = int(e)
        if end - start >= min_tokens:
            spans.append((start, min(end, start + max_tokens) if max_tokens else end))
        start = end + 1
    return spans


# ---------------------------------------------------------------------------------------
# the index
# ---------------------------------------------------------------------------------------


@dataclass
class MinHashIndex:
    """LSH over MinHash signatures: add documents, then ask which are near-duplicates."""

    params: LSHParams = field(default_factory=LSHParams)
    seed: int = 0
    _a: np.ndarray = field(init=False, repr=False)
    _b: np.ndarray = field(init=False, repr=False)
    #: band index -> bucket key -> [document ids]. A plain dict of dicts: the keys are
    #: tuples of R int64s, and Python's hashing of those is fast enough that a custom
    #: 64-bit fold would be optimising the wrong line.
    _buckets: list[dict] = field(init=False, repr=False)
    signatures: list[np.ndarray] = field(default_factory=list, repr=False)
    lengths: list[int] = field(default_factory=list, repr=False)

    def __post_init__(self):
        self._a, self._b = hash_family(self.params.permutations, self.seed)
        self._buckets = [{} for _ in range(self.params.bands)]

    def add(self, tokens: np.ndarray, k: int = SHINGLE) -> int:
        """Index one document. Returns its id."""
        sig = signature(shingle_hashes(tokens, k), self._a, self._b)
        doc = len(self.signatures)
        self.signatures.append(sig)
        self.lengths.append(int(np.asarray(tokens).size))
        rows = self.params.rows
        for band in range(self.params.bands):
            key = tuple(sig[band * rows : (band + 1) * rows].tolist())
            self._buckets[band].setdefault(key, []).append(doc)
        return doc

    def candidate_pairs(self) -> set[tuple[int, int]]:
        """Every pair sharing at least one band. The set LSH exists to make small."""
        pairs: set[tuple[int, int]] = set()
        for band in self._buckets:
            for docs in band.values():
                if len(docs) < 2:
                    continue
                # A bucket of 5,000 identical licence headers is 12 million pairs. Cap it:
                # everything in one bucket is already known to be near-identical, so linking
                # each to the first is enough to put them all in one cluster.
                if len(docs) > 64:
                    first = docs[0]
                    pairs.update((first, d) for d in docs[1:])
                    continue
                for i, x in enumerate(docs):
                    for y in docs[i + 1 :]:
                        pairs.add((x, y) if x < y else (y, x))
        return pairs

    def duplicates(self, threshold: float | None = None,
                   verify: bool = False, docs=None) -> list[tuple[int, int, float]]:
        """`[(a, b, similarity)]` for candidate pairs at or above the threshold.

        `verify=True` recomputes **exact** Jaccard from the shingle sets in `docs`, which is
        affordable only because LSH already discarded almost every pair — the same argument
        `eval/contamination.py` makes for its own verifier.
        """
        threshold = self.params.threshold if threshold is None else threshold
        out = []
        for x, y in sorted(self.candidate_pairs()):
            sim = estimated_jaccard(self.signatures[x], self.signatures[y])
            if verify and docs is not None:
                sim = jaccard(shingle_hashes(docs[x]), shingle_hashes(docs[y]))
            if sim >= threshold:
                out.append((x, y, sim))
        return out


# ---------------------------------------------------------------------------------------
# clustering and reporting
# ---------------------------------------------------------------------------------------


def clusters(pairs, n_docs: int) -> list[list[int]]:
    """Connected components over the duplicate pairs, by union-find.

    Transitivity is an *assumption* and worth naming: A~B and B~C does not imply A~C at the
    same threshold. Clustering anyway is what every deduplication pipeline does, because the
    alternative — keeping one representative per *pair* — removes far more than it should.
    """
    parent = list(range(n_docs))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]  # path halving
            x = parent[x]
        return x

    for a, b, *_ in pairs:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    groups: dict[int, list[int]] = {}
    for i in range(n_docs):
        groups.setdefault(find(i), []).append(i)
    return [g for g in groups.values() if len(g) > 1]


def report(index: MinHashIndex, pairs, n_docs: int, total_tokens: int) -> dict:
    """What fraction of the corpus is a repeat — in documents and, more usefully, in tokens.

    **Tokens is the number that matters.** Documents are not equal: removing 200,000 stub
    pages that are 40 tokens each changes the corpus by almost nothing, while removing 3,000
    duplicated long articles changes it measurably. A deduplication report quoted in
    documents can be an order of magnitude out on how much data it would actually drop.
    """
    groups = clusters(pairs, n_docs)
    # Keep one of each cluster: the removable count is the size minus one.
    removable = sum(len(g) - 1 for g in groups)
    removable_tokens = sum(
        sum(index.lengths[d] for d in sorted(g, key=lambda d: -index.lengths[d])[1:])
        for g in groups
    )
    sizes = sorted((len(g) for g in groups), reverse=True)
    return {
        "documents": n_docs,
        "tokens": total_tokens,
        "clusters": len(groups),
        "duplicate_documents": removable,
        "duplicate_token_share": removable_tokens / total_tokens if total_tokens else 0.0,
        "duplicate_document_share": removable / n_docs if n_docs else 0.0,
        "duplicate_tokens": removable_tokens,
        "largest_clusters": sizes[:10],
        "pairs_checked": len(pairs),
        "threshold": index.params.threshold,
        "permutations": index.params.permutations,
        "bands": index.params.bands,
        "rows": index.params.rows,
        "standard_error": index.params.standard_error(index.params.threshold),
        "curve": index.params.curve(),
        "caveat": (
            "MinHash ESTIMATES Jaccard; the standard error above is how far a signature's "
            "estimate typically is from the truth. LSH also has false negatives that are "
            "invisible — a similar pair sharing no band is never compared at all — so read "
            "`curve` for the detection probability at each similarity."
        ),
    }


def scan_bin(path: str | Path, eos_id: int, *, params: LSHParams | None = None,
             limit: int | None = None, max_doc_tokens: int = 4096,
             min_doc_tokens: int = 32, seed: int = 0, chunk: int = 50_000_000,
             start_token: int = 0, progress=None) -> dict:
    """Run the whole thing over a `.bin` token stream, streaming it in chunks.

    `limit` caps the number of documents. **A sample is the default way to use this**, and it
    is not a compromise: duplication is a property of the corpus, so a random-enough sample
    of 200,000 documents estimates it to well within the error MinHash already has. A full
    pass over 10 B tokens is hours, and the answer moves by less than the caveat.
    """
    params = params or LSHParams()
    index = MinHashIndex(params, seed=seed)
    tokens = np.memmap(Path(path), dtype=np.uint16, mode="r")
    docs: list[np.ndarray] = []
    total = 0
    # `start_token` exists because a sample taken from the front of a file is a sample of
    # however that file happened to be ordered — and a corpus written repo by repo is very
    # much ordered. Scanning from two different offsets and comparing is the cheapest check
    # that the number is a property of the corpus rather than of its first hundred megabytes.
    pos = int(start_token)

    while pos < tokens.size and (limit is None or len(docs) < limit):
        block = np.asarray(tokens[pos : pos + chunk])
        try:
            spans = document_spans(block, eos_id, min_tokens=min_doc_tokens,
                                   max_tokens=max_doc_tokens)
        except ValueError:
            # A chunk with no EOS at all is one very long document; skip it rather than
            # failing the scan, but only after the first chunk has proved the corpus has any.
            if pos == 0:
                raise
            pos += chunk
            continue
        for a, b in spans:
            if limit is not None and len(docs) >= limit:
                break
            doc = block[a:b]
            docs.append(doc)
            index.add(doc)
            total += int(doc.size)
            # Tagged and counted, in the one shape every job in this repo reports progress
            # in — `[tag] label done/total (pct%)`. The portal parses that line and nothing
            # else, so a scan printing its own format had no progress bar at all: 90 seconds
            # of a blank panel, which reads as a button that did nothing. Reported every
            # 5,000 rather than every 20,000 documents for the same reason.
            if progress and len(docs) % 5_000 == 0:
                # A full pass is measured in tokens read, and both ends of that are counted
                # from `start_token` — the scan begins there, so a percentage measured from
                # zero would start partway along and never reach 100.
                done, whole = ((len(docs), limit) if limit
                               else (pos + b - int(start_token),
                                     int(tokens.size) - int(start_token)))
                progress(f"[dedup] {'documents' if limit else 'tokens'} "
                         f"{done:,}/{whole:,} ({done / max(1, whole) * 100:.0f}%)")
        pos += chunk

    pairs = index.duplicates()
    out = report(index, pairs, len(docs), total)
    out["source"] = str(path)
    out["sampled"] = limit is not None
    out["start_token"] = int(start_token)
    out["max_doc_tokens"] = max_doc_tokens
    return out


# ---------------------------------------------------------------------------------------
# filtering: write a deduplicated copy
# ---------------------------------------------------------------------------------------
# `scan_bin` measures; this removes. It differs from the scan in three ways that only matter
# once documents are actually being dropped:
#   * documents are found over the WHOLE file at once, so one straddling a chunk edge is
#     neither split in two nor lost (the scan tolerates that; a filter would write halves);
#   * banding is a sort per band rather than a dict of tuples -- over a million documents
#     the dicts alone are gigabytes;
#   * it can deduplicate AGAINST other bins. The Python specialist's fresh download is
#     filtered against the bin Phase 2 already trained on, which removes repeats of code the
#     model has seen *and* doubles as the check that the fresh stream really is fresh: a
#     shifted stream shows up as most of it matching the old bin.


def all_document_spans(tokens: np.ndarray, eos_id: int,
                       chunk: int = 200_000_000) -> np.ndarray:
    """`(n, 2)` int64 `[start, end)` of every EOS-terminated document in the whole stream.

    Anything after the last EOS is dropped: `prepare.py` stops at a token budget, so the
    file's tail is a document cut off mid-way, and copying half a file into a corpus is
    exactly what a filter should not do."""
    ends = [np.flatnonzero(np.asarray(tokens[p : p + chunk]) == eos_id) + p
            for p in range(0, tokens.size, chunk)]
    ends = np.concatenate(ends) if ends else np.empty(0, dtype=np.int64)
    if ends.size == 0:
        raise ValueError("no EOS token in this stream, so it has no document boundaries")
    starts = np.concatenate([[0], ends[:-1] + 1])
    return np.stack([starts, ends], axis=1).astype(np.int64)


_w_tokens: dict = {}
_w_hash: tuple | None = None


def _sig_init(seed: int, n_perm: int):
    global _w_hash
    _w_hash = hash_family(n_perm, seed)


def _sig_batch(job) -> np.ndarray:
    """Signatures for one batch of spans of one file. Workers open the memmap themselves, so
    only (path, spans) crosses the process boundary, never the tokens."""
    path, spans, max_doc_tokens = job
    if path not in _w_tokens:
        _w_tokens[path] = np.memmap(path, dtype=np.uint16, mode="r")
    toks = _w_tokens[path]
    a, b = _w_hash
    out = np.empty((len(spans), a.size), dtype=np.int64)
    for i, (s, e) in enumerate(spans):
        out[i] = signature(shingle_hashes(np.asarray(toks[s : min(e, s + max_doc_tokens)])), a, b)
    return out


def signatures_for(path: str, spans: np.ndarray, *, params: LSHParams, seed: int = 0,
                   max_doc_tokens: int = 4096, n_proc: int = 8, batch: int = 2_000,
                   progress=None, tag: str = "") -> np.ndarray:
    """MinHash signatures for `spans` of `path`, `(n, P)` int64, computed in a process pool."""
    import multiprocessing as mp
    jobs = [(str(path), spans[i : i + batch], max_doc_tokens)
            for i in range(0, len(spans), batch)]
    out = np.empty((len(spans), params.permutations), dtype=np.int64)
    with mp.get_context("fork").Pool(n_proc, initializer=_sig_init,
                                      initargs=(seed, params.permutations)) as pool:
        for j, sig in enumerate(pool.imap(_sig_batch, jobs)):
            out[j * batch : j * batch + len(sig)] = sig
            if progress and (j % 25 == 0 or j == len(jobs) - 1):
                done = min((j + 1) * batch, len(spans))
                progress(f"[dedup] {tag} {done:,}/{len(spans):,} "
                         f"({done / max(1, len(spans)) * 100:.0f}%)")
    return out


def duplicate_pairs(sigs: np.ndarray, params: LSHParams,
                    threshold: float | None = None) -> np.ndarray:
    """`(m, 2)` index pairs that share a band AND whose estimated Jaccard clears the
    threshold. Banding by sorting: equal band keys end up adjacent, and each run of equal
    keys is a bucket. Inside a bucket every member is compared with the bucket's first --
    enough to put a bucket into one cluster, linear rather than quadratic in its size."""
    threshold = params.threshold if threshold is None else threshold
    rows = params.rows
    rng = np.random.default_rng(12345)
    found = []
    for band in range(params.bands):
        block = sigs[:, band * rows : (band + 1) * rows].astype(np.uint64)
        # Fold R int64s into one uint64 key; a collision here only creates a candidate,
        # and every candidate is checked against the signature estimate below.
        mult = rng.integers(1, 1 << 62, size=rows, dtype=np.int64).astype(np.uint64)
        key = (block * mult).sum(axis=1)
        order = np.argsort(key, kind="stable")
        k = key[order]
        new_run = np.concatenate([[True], k[1:] != k[:-1]])
        run_first = order[np.maximum.accumulate(np.where(new_run, np.arange(k.size), 0))]
        member = ~new_run
        a, b = run_first[member], order[member]
        if a.size:
            est = (sigs[a] == sigs[b]).mean(axis=1)
            keep = est >= threshold
            found.append(np.stack([a[keep], b[keep]], axis=1))
    return np.concatenate(found) if found else np.empty((0, 2), dtype=np.int64)


def filter_bin(src: str | Path, out: str | Path, eos_id: int = 0, *, against=(),
               params: LSHParams | None = None, seed: int = 0, max_doc_tokens: int = 4096,
               min_doc_tokens: int = 32, n_proc: int = 8, progress=None) -> dict:
    """Write `out`: `src` with near-duplicates removed, one document kept per cluster.

    The rule for which copy survives: if a cluster contains a document from an `against`
    bin, **no** `src` member survives (the model has already trained on that code). Otherwise
    the longest member is kept. Documents under `min_doc_tokens` are dropped too -- a
    20-token Python file is an empty `__init__.py` or a coding line, there are thousands of
    identical ones, and they are too short to shingle reliably. Every count is reported."""
    params = params or LSHParams()
    src, out = Path(src), Path(out)
    toks = np.memmap(src, dtype=np.uint16, mode="r")
    spans = all_document_spans(toks, eos_id)
    lens = spans[:, 1] - spans[:, 0]
    short = lens < min_doc_tokens
    live = np.flatnonzero(~short)

    ref_spans, ref_paths = [], []
    for ref in against:
        rs = all_document_spans(np.memmap(ref, dtype=np.uint16, mode="r"), eos_id)
        rs = rs[(rs[:, 1] - rs[:, 0]) >= min_doc_tokens]
        ref_spans.append(rs)
        ref_paths.append(str(ref))

    sig_parts = [signatures_for(str(src), spans[live], params=params, seed=seed,
                                max_doc_tokens=max_doc_tokens, n_proc=n_proc,
                                progress=progress, tag=f"{src.name}")]
    for rp, rs in zip(ref_paths, ref_spans):
        sig_parts.append(signatures_for(rp, rs, params=params, seed=seed,
                                        max_doc_tokens=max_doc_tokens, n_proc=n_proc,
                                        progress=progress, tag=f"{Path(rp).name} (reference)"))
    sigs = np.concatenate(sig_parts)
    n_src = live.size
    n_all = sigs.shape[0]
    pairs = duplicate_pairs(sigs, params)

    # Union-find over every index, src first, references after.
    parent = np.arange(n_all)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in pairs.tolist():
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
    roots = np.fromiter((find(i) for i in range(n_all)), dtype=np.int64, count=n_all)
    src_len = lens[live]
    drop = np.zeros(n_src, dtype=bool)
    # Clusters that reach a reference document: drop every src member.
    ref_roots = set(roots[n_src:].tolist())
    seen_ref = np.isin(roots[:n_src], np.fromiter(ref_roots, dtype=np.int64,
                                                  count=len(ref_roots)))
    drop |= seen_ref
    # Clusters purely inside src: keep the longest member.
    best: dict[int, int] = {}
    for i in np.flatnonzero(~seen_ref).tolist():
        r = int(roots[i])
        j = best.get(r)
        if j is None or src_len[i] > src_len[j]:
            if j is not None:
                drop[j] = True
            best[r] = i
        else:
            drop[i] = True
    # Exactly identical signatures against the reference: the "is this stream fresh" check.
    exact_ref = 0
    if n_all > n_src:
        ref_keys = {s.tobytes() for s in sigs[n_src:]}
        exact_ref = int(sum(sigs[i].tobytes() in ref_keys for i in range(n_src)))

    keep_idx = live[~drop]
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    written = 0
    eos = np.array([eos_id], dtype=np.uint16)
    with open(tmp, "wb") as f:
        for i in keep_idx.tolist():
            s, e = spans[i]
            np.asarray(toks[s:e]).tofile(f)
            eos.tofile(f)
            written += int(e - s) + 1
        f.flush()
    tmp.replace(out)

    total = int(spans[-1, 1]) + 1
    return {
        "source": str(src), "out": str(out), "against": ref_paths,
        "documents": int(len(spans)), "tokens": total,
        "dropped_short": int(short.sum()), "dropped_short_tokens": int(lens[short].sum()),
        "dropped_duplicate_within": int((drop & ~seen_ref).sum()),
        "dropped_seen_in_reference": int(seen_ref.sum()),
        "exact_matches_in_reference": exact_ref,
        "kept_documents": int(keep_idx.size), "kept_tokens": written,
        "kept_token_share": written / total if total else 0.0,
        "threshold": params.threshold, "permutations": params.permutations,
        "max_doc_tokens": max_doc_tokens, "min_doc_tokens": min_doc_tokens,
    }


# ---------------------------------------------------------------------------------------
# the CLI
# ---------------------------------------------------------------------------------------


def main(argv=None) -> int:
    """`python -m aksharallm.data.dedup data/blend/fineweb-edu-10bt.bin`

    Defaults to a **sample**, and says so in the output. A full pass over 10 B tokens is
    hours and moves the answer by less than MinHash's own error.
    """
    import argparse
    import json
    import time

    # Imported here rather than at module scope: this module is a leaf of the data package
    # and nothing but its CLI needs to know where the portal reads reports from.
    from ..eval.report import results_dir

    ap = argparse.ArgumentParser(
        prog="python -m aksharallm.data.dedup",
        description="How much of this corpus is a near-duplicate of the rest of it?")
    ap.add_argument("bin", help="a tokenized .bin written by prepare.py / prepare_blend.py")
    ap.add_argument("--eos", type=int, default=0, help="the document separator token id")
    ap.add_argument("--limit", type=int, default=60_000,
                    help="documents to sample (0 = the whole file, which is hours)")
    ap.add_argument("--bands", type=int, default=16)
    ap.add_argument("--rows", type=int, default=8)
    ap.add_argument("--max-doc-tokens", type=int, default=4096)
    ap.add_argument("--start-token", type=int, default=0,
                    help="begin the sample here rather than at the front of the file — "
                         "run it twice at different offsets to check the number is stable")
    ap.add_argument("--out", default=None,
                    help="write the report here instead of the default "
                         "logs/eval/dedup-<corpus>-<when>.json")
    ap.add_argument("--no-write", action="store_true",
                    help="print only; do not keep the report")
    ap.add_argument("--write-kept", default=None, metavar="OUT.bin",
                    help="FILTER instead of sampling: a whole-file pass that writes a "
                         "deduplicated copy here (one document kept per cluster)")
    ap.add_argument("--against", action="append", default=[], metavar="BIN",
                    help="with --write-kept: also drop documents near-duplicating one in "
                         "this bin (e.g. the corpus a model already trained on). Repeatable")
    ap.add_argument("--n-proc", type=int, default=8)
    args = ap.parse_args(argv)

    params = LSHParams(bands=args.bands, rows=args.rows)
    if args.write_kept:
        t0 = time.time()
        rep = filter_bin(args.bin, args.write_kept, args.eos, against=args.against,
                         params=params, max_doc_tokens=args.max_doc_tokens,
                         n_proc=args.n_proc, progress=print)
        rep["seconds"] = round(time.time() - t0, 1)
        print(f"\n  {rep['documents']:,} documents, {rep['tokens']:,} tokens in {rep['source']}")
        print(f"  dropped {rep['dropped_short']:,} under {rep['min_doc_tokens']} tokens, "
              f"{rep['dropped_duplicate_within']:,} duplicates within, "
              f"{rep['dropped_seen_in_reference']:,} already in the reference "
              f"({rep['exact_matches_in_reference']:,} of those exactly)")
        print(f"  kept {rep['kept_documents']:,} documents, {rep['kept_tokens']:,} tokens "
              f"({rep['kept_token_share'] * 100:.1f}%)  ->  {rep['out']}   "
              f"[{rep['seconds']:.0f}s]")
        Path(args.write_kept).with_suffix(".dedup.json").write_text(json.dumps(rep, indent=1))
        return 0
    print(f"{args.bin}")
    print(f"  {params.permutations} permutations in {params.bands} bands of {params.rows}"
          f"  ->  threshold ~{params.threshold:.3f}, "
          f"standard error +-{params.standard_error(params.threshold):.3f}")
    t0 = time.time()
    rep = scan_bin(args.bin, args.eos, params=params,
                   limit=args.limit or None, max_doc_tokens=args.max_doc_tokens,
                   start_token=args.start_token, progress=print)
    print(f"\n  scanned {rep['documents']:,} documents "
          f"({rep['tokens'] / 1e6:.1f}M tokens) in {time.time() - t0:.0f}s"
          f"{'  [SAMPLE]' if rep['sampled'] else '  [full pass]'}")
    print(f"  {rep['clusters']:,} duplicate clusters, largest {rep['largest_clusters'][:5]}")
    print(f"  {rep['duplicate_documents']:,} removable documents "
          f"({rep['duplicate_document_share'] * 100:.2f}% of documents)")
    print(f"  {rep['duplicate_tokens']:,} removable tokens "
          f"({rep['duplicate_token_share'] * 100:.2f}% of tokens)   <- the one that matters")
    print(f"\n  {'similarity':>10} {'chance LSH sees it':>20}")
    for row in rep["curve"]:
        print(f"  {row['similarity']:>10.2f} {row['detected'] * 100:>19.1f}%")
    print(f"\n  {rep['caveat']}")
    # Written by default, not only when asked. A dedup number is quoted *per offset* and is
    # only honest read beside another one taken elsewhere in the file — which is impossible
    # if the first one scrolled out of a terminal. The portal was passing `--out` and so had
    # a card; the same command typed by hand left no trace and showed up nowhere.
    if not args.no_write:
        dest = (Path(args.out) if args.out else
                results_dir() / f"dedup-{Path(args.bin).stem}-"
                                f"{time.strftime('%Y%m%d-%H%M%S')}.json")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps({**rep, "source": str(args.bin),
                                    "start_token": args.start_token,
                                    "limit": args.limit}, indent=1))
        print(f"\n  written to {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
