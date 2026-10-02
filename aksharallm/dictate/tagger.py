"""The punctuation tagger: training objective, inference, and the score it is judged by.

**There is no `train/punct.py`.** `train/pretrain.py` already owns everything about training
something for an hour on one card — accumulation, the schedule, autocast, checkpoint/resume,
the stop file, the session log, the report — and asks an *objective* for the four things that
differ (docs/20 made that seam for masked diffusion). This is the third objective:

    batch     read FineWeb-Edu windows, strip what the tagger restores (dictate/punct.py)
    loss      cross-entropy over 12 labels at the LAST token of each word
    evaluate  the same on val.bin windows drawn with a fixed seed
    sample    punctuate a fixed lower-case paragraph, so the log shows it learning

**The classifier is the embedding matrix.** A tagger needs a 12-way head, and the model has
a tied `vocab_size`-way one. `model.tag_classes: 12` gives the vocabulary twelve extra rows
that the tokenizer can never emit; the logit for label *k* is the hidden state dotted with row
`tag_base + k`. Nothing about the transformer changes — the same trick masked diffusion uses
for `[MASK]` — and the forward pass asks for hidden states (`return_hidden`) so the other
32,768 logits are never computed.

**Why the last sub-token of a word.** "alright" may be two BPE pieces; the label belongs to the
word, so it is read where the word ends — the position that has seen all of it even in a
causal model, and in this bidirectional one has seen the next word too, which is what a comma
decision needs.

**How it is judged.** Per-class precision / recall / F1 for `,` `.` `?` and capitalisation —
never accuracy, because ~85% of words take *no* punctuation and a tagger that never punctuates
scores 85%. A rules-only baseline (capital first, full stop last) is scored beside it, so the
number says what the model adds.

Read with: docs/23-speech-recognition.md -- the chapter this implements; it ends with the
order to read these files in. See also docs/20-diffusion.md (the objective seam).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from . import punct
from .punct import IGNORE, N_LABELS

#: Shown in every mid-run sample: questions, a name, commas in a list, and two sentences.
SAMPLE = ("hello mary how was your trip to paris did you see the eiffel tower i hope the "
          "weather was good we had rain snow and wind all week but tomorrow should be better")


class WordEncoder:
    """Words -> BPE ids, one word at a time (`" " + word`), cached.

    Encoding word by word rather than the joined string keeps every word's pieces its own —
    BPE never merges across the space — so "where does word *i* end" is a count, not a search.
    It is the same at training and inference by construction: both call this.
    """

    def __init__(self, tok):
        self.tok = tok
        self._cache: dict[str, list[int]] = {}

    def word(self, w: str) -> list[int]:
        ids = self._cache.get(w)
        if ids is None:
            ids = self.tok.encode(" " + w)
            if len(self._cache) < 2_000_000:
                self._cache[w] = ids
        return ids

    def encode(self, words: list[str]) -> tuple[list[int], list[int]]:
        """`(ids, ends)` where `ends[i]` is the index of word i's last token."""
        ids: list[int] = []
        ends: list[int] = []
        for w in words:
            ids.extend(self.word(w))
            ends.append(len(ids) - 1)
        return ids, ends


def tag_logits(model, idx: torch.Tensor) -> torch.Tensor:
    """(B, T) ids -> (B, T, n_labels) logits, through the model's own embedding rows."""
    raw = getattr(model, "_orig_mod", model)
    cfg = raw.cfg
    h, _ = model(idx, return_hidden=True)
    w = raw.lm_head.weight[cfg.tag_base: cfg.tag_base + cfg.tag_classes]
    return h @ w.t().to(h.dtype)


# ---------------------------------------------------------------------------------------
# training: the objective pretrain.py drives
# ---------------------------------------------------------------------------------------

_TOK = None


def _window_words(tok, window: np.ndarray) -> list[punct.Word]:
    """One token window of web text -> normalised words with labels. Documents are split at
    EOS first, so two documents never run together into one sentence."""
    eos = tok.eos_id
    parts, cur = [], []
    for i in window.tolist():
        if i == eos:
            parts.append(cur)
            cur = []
        else:
            cur.append(i)
    parts.append(cur)
    out: list[punct.Word] = []
    for text in tok._tok.decode_batch(parts):
        out.extend(punct.examples_from_text(text))
    return out


def _worker_init() -> None:
    """Forked workers inherit the trainer's SIGTERM/SIGINT handlers, which print "[stop]
    signal received" and wait for a step that a worker never takes. A worker should just die
    when its pool is shut down, and leave Ctrl-C to the trainer."""
    import signal
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def _window_words_worker(window: np.ndarray) -> list[punct.Word]:
    return _window_words(_TOK, window)


class TaggerObjective:
    """Restore punctuation and capitals to normalised text. See the module docstring."""

    name = "punctuation tagger"
    metric = "tag cross-entropy"
    comparable_to_ar = False
    #: exp(CE) over twelve labels: "as unsure as a choice between this many labels". Not a
    #: language model's perplexity, and not a bound on one either (gotcha 12).
    ppl_label = "label-ppl"

    def __init__(self, cfg):
        self.cfg = cfg
        self.T = int(cfg.train.seq_len)
        self.eval_seed = 20261002
        self.enc: WordEncoder | None = None
        self._stats: dict[str, float] = {}

    def check(self, tok) -> None:
        m = self.cfg.model
        if m.tag_classes != N_LABELS:
            raise ValueError(f"the punctuation tagger has {N_LABELS} labels; "
                             f"model.tag_classes is {m.tag_classes}")
        if m.vocab_size != tok.vocab_size + N_LABELS:
            raise ValueError(f"model.vocab_size must be the tokenizer's {tok.vocab_size} + "
                             f"{N_LABELS} label rows = {tok.vocab_size + N_LABELS}; "
                             f"it is {m.vocab_size}")
        self.tok = tok
        self.enc = WordEncoder(tok)

    def describe(self) -> str:
        return (f"punctuation + capitals as {N_LABELS} labels per word, bidirectional; "
                "inputs are web text normalised to the recogniser's alphabet")

    # ---- data ---------------------------------------------------------------------------

    def _words(self, window: np.ndarray) -> list[punct.Word]:
        return _window_words(self.tok, window)

    def _pool(self):
        """Worker processes for the text work, which is the bottleneck: decoding, normalising
        and re-encoding cost ~4 ms per window against a ~20 ms GPU step for 64 rows. Windows
        are still DRAWN here, by the dataset's own generator, so a resumed run reads the same
        stream; the workers only compute a pure function of them."""
        if getattr(self, "_workers", None) is None:
            import multiprocessing as mp
            global _TOK
            _TOK = self.tok
            n = int(getattr(self.cfg, "punct_workers", 0) or min(8, max(1, (os.cpu_count() or 2) - 2)))
            self._workers = (mp.get_context("fork").Pool(n, initializer=_worker_init)
                             if n > 1 else False)
            if self._workers:
                import atexit
                atexit.register(self._workers.terminate)
        return self._workers

    #: Source window length. One read of 1,024 tokens yields ~400 tokens of training text,
    #: a row's worth; four reads of 256 cost four random seeks into a 17 GB file, which
    #: measured as half of all the batch time.
    SOURCE = 1024

    def _windows(self, dataset, k: int, generator=None) -> list[np.ndarray]:
        """`k` raw token windows, read on the CPU from the dataset's own file and drawn with
        the dataset's own generator — so the stream position is still the one the checkpoint
        saves (`dataset.rng_state`) and a resume reads what an unbroken run would have."""
        from ..data.loader import TokenDataset
        src = getattr(self, "_src", None)
        if src is None or src.path != dataset.path:
            src = self._src = TokenDataset(dataset.path, self.SOURCE, "cpu")
        x, _ = src.get_batch(k, generator=generator if generator is not None else dataset.rng)
        return list(x.numpy())

    def _rows(self, dataset, n: int, generator=None) -> tuple[np.ndarray, np.ndarray]:
        """`n` rows of `(ids, labels)`, each filled from as many source windows as it takes.
        A source window yields ~40% of its words (the rest are dropped sentences), so a row
        is usually three or four of them. Padding is the last resort, labelled IGNORE."""
        ids = np.full((n, self.T), self.tok.pad_id, dtype=np.int64)
        lab = np.full((n, self.T), IGNORE, dtype=np.int64)
        pool: list[list[punct.Word]] = []

        def refill(k: int):
            wins = self._windows(dataset, k, generator)
            workers = self._pool()
            done = workers.map(_window_words_worker, wins) if workers else [self._words(w) for w in wins]
            pool.extend(reversed(done))

        refill(n + n // 4)
        for r in range(n):
            got: list[int] = []
            labs: list[int] = []
            tries = 0
            while len(got) < self.T and tries < 8:
                if not pool:
                    refill(max(4, n // 4))
                tries += 1
                for w in pool.pop():
                    p = self.enc.word(w.text)
                    if len(got) + len(p) > self.T:
                        break
                    got.extend(p)
                    labs.extend([IGNORE] * (len(p) - 1) + [w.label])
            ids[r, : len(got)] = got
            lab[r, : len(labs)] = labs
        return ids, lab

    def batch(self, dataset, batch_size: int):
        ids, lab = self._rows(dataset, batch_size)
        dev = dataset.device
        return (torch.from_numpy(ids).to(dev, non_blocking=True),
                torch.from_numpy(lab).to(dev, non_blocking=True))

    # ---- loss ---------------------------------------------------------------------------

    def loss(self, model, batch):
        idx, lab = batch
        logits = tag_logits(model, idx).float()
        loss = F.cross_entropy(logits.view(-1, N_LABELS), lab.view(-1), ignore_index=IGNORE)
        with torch.no_grad():
            m = lab != IGNORE
            pred = logits.argmax(-1)
            self._stats = {"acc": float((pred[m] == lab[m]).float().mean()),
                           "punct": float(((lab[m] // punct.N_CASE) != punct.NONE).float().mean())}
        return loss

    @torch.no_grad()
    def evaluate(self, model, dataset, batch_size: int, n_batches: int, ctx) -> float:
        """Fixed seed: every evaluation reads the same val windows, so the curve is the model."""
        was = model.training
        model.eval()
        rng = np.random.default_rng(self.eval_seed)
        tot = cnt = 0.0
        for _ in range(n_batches):
            ids, lab = self._rows(dataset, batch_size, generator=rng)
            idx = torch.from_numpy(ids).to(dataset.device)
            y = torch.from_numpy(lab).to(dataset.device)
            with ctx:
                logits = tag_logits(model, idx).float()
            tot += float(F.cross_entropy(logits.view(-1, N_LABELS), y.view(-1),
                                         ignore_index=IGNORE, reduction="sum"))
            cnt += float((y != IGNORE).sum())
        model.train(was)
        return tot / max(cnt, 1)

    @torch.no_grad()
    def sample(self, model, tok, prompt: str, device: str) -> str:
        raw = getattr(model, "_orig_mod", model)
        return Punctuator(raw, tok, device).punctuate(SAMPLE.split())

    def stats(self) -> dict:
        return dict(self._stats)

    def log_suffix(self) -> str:
        s = self._stats
        return f" | acc {s['acc'] * 100:.1f}%" if s else ""


# ---------------------------------------------------------------------------------------
# inference
# ---------------------------------------------------------------------------------------

class Punctuator:
    """A loaded tagger. `labels(words)` and `punctuate(words)`; long input is windowed."""

    def __init__(self, model, tok, device: str = "cpu"):
        self.model = model
        self.tok = tok
        self.device = device
        self.enc = WordEncoder(tok)
        self.T = model.cfg.max_seq_len

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> "Punctuator":
        from ..config import ModelConfig
        from ..model.transformer import Transformer
        from ..tokenizer.tokenizer import Tokenizer

        blob = torch.load(path, map_location="cpu", weights_only=False)
        mc = ModelConfig(**blob["model_config"])
        if not mc.is_tagger:
            raise ValueError(f"{path} is not a tagger (model.tag_classes is 0)")
        model = Transformer(mc)
        state = {k.removeprefix("_orig_mod."): v for k, v in blob["model"].items()}
        model.load_state_dict(state)
        model.to(device).eval()
        tok_path = (blob.get("config") or {}).get("data", {}).get("tokenizer", "data/blend/tokenizer.json")
        p = cls(model, Tokenizer(tok_path), device)
        p.step = blob.get("step")
        p.path = str(path)
        return p

    @torch.no_grad()
    def labels(self, words: list[str]) -> list[int]:
        if not words:
            return []
        ids, ends = self.enc.encode(words)
        T = self.T
        if len(ids) <= T:
            spans = [(0, len(ids))]
        else:
            # Overlapping windows; each token takes its label from the window where it sits
            # furthest from an edge, because a tagger near an edge is missing half its context.
            step = T // 2
            spans = [(s, min(s + T, len(ids))) for s in range(0, len(ids) - T + step, step)]
        best = np.full(len(ids), -1, dtype=np.int64)
        margin = np.full(len(ids), -1, dtype=np.int64)
        for s, e in spans:
            x = torch.tensor(ids[s:e], device=self.device)[None]
            pred = tag_logits(self.model, x)[0].argmax(-1).cpu().numpy()
            pos = np.arange(s, e)
            m = np.minimum(pos - s, e - 1 - pos)
            take = m > margin[s:e]
            best[s:e][take] = pred[take]
            margin[s:e][take] = m[take]
        return [int(best[i]) for i in ends]

    def punctuate(self, words: list[str], spellings: dict[str, str] | None = None) -> str:
        return punct.render(words, self.labels(words), spellings=spellings)


# ---------------------------------------------------------------------------------------
# how good is it
# ---------------------------------------------------------------------------------------

def _prf(tp: int, fp: int, fn: int) -> dict:
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    return {"precision": p, "recall": r, "f1": 2 * p * r / max(p + r, 1e-9), "support": tp + fn}


def score_labels(gold: list[int], pred: list[int]) -> dict:
    """Per-class P/R/F1 for `,` `.` `?` and for capitalised words."""
    out = {}
    g = np.array(gold)
    p = np.array(pred)
    gp, pp = g // punct.N_CASE, p // punct.N_CASE
    gc, pc = g % punct.N_CASE, p % punct.N_CASE
    for name, k in (("comma", punct.COMMA), ("period", punct.PERIOD), ("question", punct.QUESTION)):
        out[name] = _prf(int(((gp == k) & (pp == k)).sum()), int(((gp != k) & (pp == k)).sum()),
                         int(((gp == k) & (pp != k)).sum()))
    gcap, pcap = gc != punct.LOWER, pc != punct.LOWER
    out["capital"] = _prf(int((gcap & pcap).sum()), int((~gcap & pcap).sum()),
                          int((gcap & ~pcap).sum()))
    out["words"] = len(gold)
    out["exact"] = float((g == p).mean()) if len(g) else 0.0
    return out


def rules_only(words: list[str]) -> list[int]:
    """The baseline: capital first word and "i", a full stop at the very end, nothing else."""
    labs = [punct.label(punct.NONE, punct.CAP if (k == 0 or w == "i" or w.startswith("i'"))
                        else punct.LOWER) for k, w in enumerate(words)]
    if labs:
        labs[-1] = punct.label(punct.PERIOD, labs[-1] % punct.N_CASE)
    return labs


def heldout_passages(tok, val_bin: str, n: int, words_per: int = 120, seed: int = 7) -> list[list[punct.Word]]:
    """`n` passages of held-out text from `val_bin`, normalised the way training data is."""
    from ..data.loader import TokenDataset
    ds = TokenDataset(val_bin, 512, "cpu", seed=seed)
    obj = TaggerObjective.__new__(TaggerObjective)
    obj.tok = tok
    out = []
    tries = 0
    while len(out) < n and tries < n * 20:
        tries += 1
        x, _ = ds.get_batch(1)
        ws = obj._words(x[0].numpy())
        if len(ws) >= 20:
            out.append(ws[:words_per])
    return out


def evaluate_tagger(p: Punctuator, passages: list[list[punct.Word]]) -> dict:
    """Score the tagger and the rules-only baseline on the same passages."""
    gold, model, rules = [], [], []
    for ws in passages:
        words = [w.text for w in ws]
        gold += [w.label for w in ws]
        model += p.labels(words)
        rules += rules_only(words)
    return {"tagger": score_labels(gold, model), "rules_only": score_labels(gold, rules),
            "passages": len(passages)}
