"""CTC prefix beam search, with a word language model and a personal dictionary.

The greedy decoder takes the likeliest *character* at every frame. That is not the likeliest
*transcript*: CTC gives a transcript the summed probability of every frame path that collapses
to it, and a text whose probability is spread over many paths can beat one whose best single
path is likelier. Beam search keeps the best `beam` transcripts so far — **prefixes** — and adds
up the paths into each one (Hannun et al., 2014):

```mermaid
flowchart LR
    P["prefix 'the ca'<br/>p(ends in blank), p(ends in 'a')"] -->|blank| S["'the ca'"]
    P -->|"'a' again"| S
    P -->|"'r'"| N["'the car'"]
    N -->|"' ' : a word ended"| W["ask the LM:<br/>log P(car | <s>, the)"]
```

Two probabilities per prefix, because CTC's merge rule needs them: a prefix whose path ended in
a **blank** can take the same letter again as a *new* letter ("l _ l" → "ll"); one whose path
ended in that letter cannot (it merges).

**Where the language model comes in: at the end of each word.** The moment a space follows a
word, the prefix's score gains `α · log P(word | previous two)` from the trigram LM
(`asr/ngram.py`) and a `β` bonus per word. Without β an LM-weighted search prefers fewer, longer
words, because every word costs probability. α and β are **tuned on dev-clean only**;
test-clean is scored once, with what dev chose (`python -m aksharallm.asr tune`).

**Words the LM has never seen** score as `<unk>` plus `unk_penalty`. That is the dial between
"karots" (should lose to "carrots") and "quilter" (a real name the LM has never seen, which
should still be writable).

**The personal dictionary (day-two problem 4)** rides on the same mechanism. A dictionary word
gets `word_bonus` when it completes, and no `<unk>` penalty. It also gets `prefix_bonus` *per
letter* while it is being spelt — "s", "sh", "sha"… — because a word only scores when it ends,
and a name nobody expects can fall out of the beam before it gets there. If the letters turn out
to spell something else, the prefix bonus is **refunded**, so the dictionary helps the words in
it without nudging every word that happens to start the same way.

All of this is plain Python over a `(frames, 29)` array of log-probabilities: the encoder runs
once and the decoder can then be re-run under different weights for the cost of the search,
which is what makes tuning cheap.

Read with: docs/23-speech-recognition.md -- the chapter this implements; it ends with the
order to read these files in.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import vocab
from .ngram import BOS, EOS, UNK, TrigramLM

NEG = -1e30
SPACE = vocab.STOI[" "]


def _lse(a: float, b: float) -> float:
    if a < b:
        a, b = b, a
    if b <= NEG:
        return a
    return a + math.log1p(math.exp(b - a))


@dataclass
class _State:
    """Everything about a prefix that is not CTC probability: its LM history and bonuses."""

    extra: float = 0.0            # alpha*LM + beta*words + dictionary bonuses so far
    u: int = BOS
    v: int = BOS
    partial: str = ""             # the word being spelt
    pending: float = 0.0          # prefix bonus granted to `partial`, refundable
    oov: bool = False             # `partial` has left the LM's vocabulary; penalty already paid


@dataclass
class BeamDecoder:
    lm: TrigramLM | None = None
    alpha: float = 0.5
    beta: float = 1.0
    beam: int = 16
    unk_penalty: float = -6.0
    dictionary: set[str] = field(default_factory=set)
    word_bonus: float = 3.0
    prefix_bonus: float = 0.4
    #: A dictionary word the LM has never seen is scored as `<unk>` — whose probability depends
    #: entirely on how much out-of-vocabulary text the LM's corpus happened to have (none, in a
    #: small one: log P ~ -50, and no bonus can recover that). It is floored here at the score
    #: of a rare word instead, so a name in the dictionary competes on its letters.
    dict_floor: float = -12.0
    #: Charge the `<unk>` penalty the moment the letters stop being the start of ANY word the
    #: LM knows, instead of when the word ends. The LM only speaks at word boundaries, so an
    #: unfinished misspelling otherwise looks free beside finished words that have paid their
    #: LM cost — and a strong alpha then fills the beam with them (measured: alpha 1.0 took
    #: dev WER from 14% to 46% without this).
    prefix_check: bool = True
    #: Characters below this log-probability at a frame are not considered there. CTC's
    #: per-frame distributions are very peaked once trained, so this keeps a frame to the one
    #: to three letters that could matter instead of all 28.
    char_prune: float = -8.0

    def __post_init__(self):
        self.dictionary = {w.strip().lower() for w in self.dictionary if w.strip()}
        self.prefixes = {w[:i] for w in self.dictionary for i in range(1, len(w) + 1)}
        self.lm_prefixes: set[str] = set()
        if self.lm is not None and self.prefix_check:
            # Cached on the LM: building it is a pass over 200k words, and one LM serves many
            # decoders (every grid point of a tune).
            cached = getattr(self.lm, "_prefix_set", None)
            if cached is None:
                cached = {w[:i] for w in self.lm.words[3:] for i in range(1, len(w) + 1)}
                self.lm._prefix_set = cached
            self.lm_prefixes = cached

    # ---- the word-level score ---------------------------------------------------------------

    def _word(self, st: _State, word: str) -> _State:
        """Close `word`: add its LM score, β, and the dictionary's verdict."""
        extra = st.extra
        in_dict = word in self.dictionary
        if self.lm is not None:
            wid = self.lm.id(word)
            lp = self.lm.logprob(st.u, st.v, wid)
            if wid == UNK:
                if in_dict:
                    lp = max(lp, self.dict_floor)
                    if st.oov:
                        lp -= self.unk_penalty   # it was charged early; a dictionary word is owed it back
                elif not st.oov:
                    lp += self.unk_penalty       # not charged yet (prefix_check off, or a word
                                                 # whose letters are all a prefix of real words)
            extra += self.alpha * lp
            u, v = st.v, wid
        else:
            u, v = st.v, UNK
        extra += self.beta
        if in_dict:
            extra += self.word_bonus
        else:
            extra -= st.pending   # refund: it spelt something else
        return _State(extra, u, v, "", 0.0)

    def _extend(self, st: _State, ch: str) -> _State:
        if ch == " ":
            return self._word(st, st.partial)
        partial = st.partial + ch
        extra, pending, oov = st.extra, st.pending, st.oov
        # Not while the letters are still spelling a DICTIONARY word: charging there and
        # refunding at the word's end is correct on paper and fatal in the search — "ambrosch"
        # paid alpha x unk (~-19) the moment it left the LM's vocabulary, fell out of the beam,
        # and never reached the end where the refund was. Measured: dictionary recall 20-31%
        # on dev-clean with that order; see docs/23 for the figure without it.
        if (self.lm_prefixes and not oov and partial not in self.lm_prefixes
                and partial not in self.prefixes):
            extra += self.alpha * self.unk_penalty
            oov = True
        if self.dictionary:
            if partial in self.prefixes:
                extra += self.prefix_bonus
                pending += self.prefix_bonus
            elif pending:
                extra -= pending
                pending = 0.0
        return _State(extra, st.u, st.v, partial, pending, oov)

    def _final(self, st: _State) -> float:
        if st.partial:
            st = self._word(st, st.partial)
        if self.lm is not None:
            return st.extra + self.alpha * self.lm.logprob(st.u, st.v, EOS)
        return st.extra

    # ---- the search ---------------------------------------------------------------------------

    def decode(self, lp: np.ndarray) -> str:
        """`lp` is `(frames, 29)` log-probabilities for one utterance (already trimmed)."""
        beams: dict[str, list[float]] = {"": [0.0, NEG]}   # prefix -> [p_blank, p_nonblank]
        states: dict[str, _State] = {"": _State()}
        for row in lp:
            row = row.tolist()
            cands = [c for c in range(1, len(row)) if row[c] > self.char_prune]
            nxt: dict[str, list[float]] = {}

            def add(prefix, which, value):
                e = nxt.get(prefix)
                if e is None:
                    e = nxt[prefix] = [NEG, NEG]
                e[which] = _lse(e[which], value)

            for prefix, (pb, pnb) in beams.items():
                tot = _lse(pb, pnb)
                add(prefix, 0, tot + row[0])
                last = prefix[-1] if prefix else ""
                for c in cands:
                    ch = vocab.ITOS[c]
                    p = row[c]
                    if ch == " " and (not prefix or last == " "):
                        # A leading or doubled space is not text; it behaves like a blank.
                        add(prefix, 0, tot + p)
                        continue
                    if ch == last:
                        add(prefix, 1, pnb + p)        # the same letter, merged
                        new = prefix + ch
                        if new not in states:
                            states[new] = self._extend(states[prefix], ch)
                        add(new, 1, pb + p)            # a new letter after a blank
                    else:
                        new = prefix + ch
                        if new not in states:
                            states[new] = self._extend(states[prefix], ch)
                        add(new, 1, tot + p)
            ranked = sorted(nxt.items(), key=lambda kv: _lse(*kv[1]) + states[kv[0]].extra,
                            reverse=True)[: self.beam]
            beams = dict(ranked)
            # Forget states nothing in the beam can reach any more, or the dict grows per frame.
            if len(states) > 50 * self.beam:
                states = {p: states[p] for p in beams}
        best = max(beams.items(), key=lambda kv: _lse(*kv[1]) + self._final(states[kv[0]]))
        return " ".join(best[0].split())


# ---------------------------------------------------------------------------------------
# many utterances at once
# ---------------------------------------------------------------------------------------

_WORKER: BeamDecoder | None = None


def _init(dec: BeamDecoder) -> None:
    global _WORKER
    _WORKER = dec


def _one(lp: np.ndarray) -> str:
    return _WORKER.decode(lp)


def decode_many(dec: BeamDecoder, lps: list[np.ndarray], workers: int = 8) -> list[str]:
    """Beam-decode a list of utterances across processes. The decoder (LM arrays included) is
    inherited by fork, so a 1 GB language model is shared, not copied per task."""
    if workers <= 1 or len(lps) < 4:
        return [dec.decode(x) for x in lps]
    import multiprocessing as mp

    ctx = mp.get_context("fork")
    _init(dec)
    with ctx.Pool(workers) as pool:
        return pool.map(_one, lps, chunksize=max(1, len(lps) // (workers * 8)))
