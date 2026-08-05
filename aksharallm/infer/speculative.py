"""Speculative decoding: a small model guesses, the big one checks, the output is unchanged.

Generation is memory-bound, not compute-bound. Reading 300M parameters out of VRAM to
produce **one** token wastes almost all of a 3090's arithmetic — the card is idle waiting on
memory. But reading those same weights once and scoring *five* candidate tokens costs barely
more than scoring one. That is the entire opportunity, and speculative decoding is how you
take it:

    1. a small draft model writes the next `gamma` tokens, cheaply and autoregressively
    2. the big target model reads all of them in ONE forward pass
    3. each guess is accepted or rejected by a rule that leaves the output distribution
       *exactly* the target's
    4. rejected guesses are thrown away by rewinding the KV cache

The third point is what makes this worth building rather than a quality trade. This is not
"a smaller model answers when it is confident". The text you get is **the text the target
model would have produced anyway** — with greedy decoding, token for token; with sampling,
draw for draw from the same distribution. The draft model can be bad. A bad draft is slow,
never wrong.

**The acceptance rule** (Leviathan et al. / Chen et al.), for a draft distribution `q` and a
target distribution `p` at the same position:

    accept a drafted token x with probability   min(1, p(x) / q(x))
    on rejection, emit a sample from            norm(max(p - q, 0))

and the two paths together emit exactly `p`:

    P(emit x) = q(x)·min(1, p(x)/q(x)) + P(reject)·norm(max(p-q,0))(x)
              = min(q(x), p(x)) + max(p(x)-q(x), 0)
              = p(x)

`test_the_acceptance_rule_emits_exactly_the_targets_distribution` asserts that identity on
random distributions, because it is the whole claim.

Greedy decoding needs no special case: at temperature 0 both distributions are one-hot, so
the rule reduces to "accept while the draft agrees with the target's argmax, otherwise take
the target's argmax". That falls out of the same four lines.

**What it costs.** Each round runs the draft `gamma` times and the target once. If the draft
is ~20x cheaper and accepts ~70% of its guesses, you emit ~3 tokens per target forward
instead of 1. If it accepts nothing you have paid one extra target forward's worth of draft
time per token, so a hopeless draft model is a slowdown of a few percent — the acceptance
rate in the stats is there to tell you which of the two you have.

Read with: docs/06-inference.md -- the chapter this implements; it ends with the order to read
these files in.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

import torch
import torch.nn.functional as F

from .generate import _filter_logits, fit_prompt


@dataclass
class SpecStats:
    """What the round actually bought. Read `accept_rate` first: it is the only number that
    says whether this draft model is worth its own forward passes."""

    rounds: int = 0                 #: draft-then-verify cycles
    drafted: int = 0                #: tokens the draft model proposed
    accepted: int = 0               #: ...of which the target agreed with
    corrections: int = 0            #: rejections, each costing the round's remaining drafts
    bonus: int = 0                  #: rounds where every draft was accepted, earning a free token
    emitted: int = 0                #: tokens actually returned
    target_forwards: int = 0        #: what plain decoding would have needed per token
    per_round: list[int] = field(default_factory=list)   #: accepted count, round by round

    @property
    def accept_rate(self) -> float:
        return self.accepted / self.drafted if self.drafted else 0.0

    @property
    def tokens_per_forward(self) -> float:
        """Emitted tokens per target forward pass — the speedup, before overheads. Plain
        decoding is 1.0 by definition, so anything above it is what this bought."""
        return self.emitted / self.target_forwards if self.target_forwards else 0.0

    def as_dict(self) -> dict:
        return {"rounds": self.rounds, "drafted": self.drafted, "accepted": self.accepted,
                "corrections": self.corrections, "bonus": self.bonus,
                "emitted": self.emitted, "target_forwards": self.target_forwards,
                "accept_rate": self.accept_rate,
                "tokens_per_forward": self.tokens_per_forward,
                "per_round": self.per_round}

    def summary(self) -> str:
        return (f"{self.emitted} tokens in {self.rounds} rounds · "
                f"accepted {self.accepted}/{self.drafted} ({self.accept_rate * 100:.0f}%) · "
                f"{self.tokens_per_forward:.2f} tokens per target forward")


class SpeculativeError(Exception):
    """The two models cannot be used together. Always a refusal, never a warning — see
    :func:`check_pair`."""


def check_pair(target, draft) -> None:
    """Refuse a pairing that would produce plausible nonsense.

    A shared vocabulary is not a nicety here: token id 8,412 has to mean the same string to
    both models or the draft's guesses are noise *and the acceptance rule cannot tell*, since
    it only ever compares probabilities of an id. This is the same failure mode that makes
    cross-tokenizer distillation a research problem rather than a build, and it must be a
    hard refusal for the same reason: everything downstream still runs.
    """
    if draft is target:
        raise SpeculativeError(
            "the draft and target are the same model — that is plain decoding with extra "
            "steps. Pass a smaller checkpoint as the draft.")
    if target.cfg.vocab_size != draft.cfg.vocab_size:
        raise SpeculativeError(
            f"vocabulary mismatch: the target has {target.cfg.vocab_size:,} tokens and the "
            f"draft {draft.cfg.vocab_size:,}. Both models must share the tokenizer — a token "
            f"id has to mean the same string to each of them.")


def next_distribution(logits: torch.Tensor, used: set[int], temperature: float,
                      top_k: int | None, top_p: float | None,
                      repetition_penalty: float) -> torch.Tensor:
    """One row of logits → the probability vector the model would actually sample from.

    Both models go through this, and that is load-bearing. The acceptance rule compares `p`
    with `q`, so `p` must be the *sampled* distribution — after temperature, top-k, top-p and
    any repetition penalty — not the raw softmax. Compare against the raw one and the output
    distribution silently stops being the target's, which is the only thing this module
    promises.
    """
    logits = logits.float().clone()
    if repetition_penalty != 1.0:
        for t in used:
            if logits[t] > 0:
                logits[t] /= repetition_penalty
            else:
                logits[t] *= repetition_penalty
    if temperature <= 0.0:
        # Greedy as a distribution, so the acceptance rule needs no special case: p and q
        # are one-hot, `min(1, p/q)` is 1 exactly when the draft picked the target's argmax,
        # and the residual is the target's argmax.
        probs = torch.zeros_like(logits)
        probs[int(logits.argmax())] = 1.0
        return probs
    logits = _filter_logits(logits[None, :] / temperature, top_k, top_p)[0]
    return F.softmax(logits, dim=-1)


def residual_distribution(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """`norm(max(p - q, 0))` — what to emit when a guess is rejected.

    Rejecting is not "fall back to the target": that would double-count the mass the draft
    already got right and bias the output towards tokens both models like. Subtracting the
    draft's distribution first is what makes the two paths sum to exactly `p`.
    """
    diff = torch.clamp(p - q, min=0.0)
    total = float(diff.sum())
    # p == q happens (identical models, or a one-hot agreement), and then rejection has
    # probability zero and this branch is unreachable — but a zero vector handed to
    # `multinomial` is a crash, and an unreachable crash is still a crash.
    return p if total <= 0 else diff / total


# --------------------------------------------------------------------------------------
# who does the guessing
# --------------------------------------------------------------------------------------

class Drafter:
    """Something that proposes the next few tokens. It does not have to be a neural network.

    The verify-and-accept half of this module only ever asks two things of a draft: *which
    tokens* and *with what probability did you pick them*. A small model is one way to answer
    (:class:`ModelDrafter`); looking the answer up in the text so far is another
    (:class:`NgramDrafter`), and it needs no training, no weights and no tokenizer agreement.
    """

    name = "drafter"

    def propose(self, tokens: list[int], g: int,
                dist) -> tuple[list[int], list[torch.Tensor]]:
        """`g` guesses for what follows `tokens`, and the distribution each came from."""
        raise NotImplementedError

    def rollback(self, keep: int) -> None:
        """Told how many tokens survived, so any cached state can drop the rest."""


class ModelDrafter(Drafter):
    """The usual thing: a smaller model of the same family, run autoregressively.

    It keeps its own KV cache and rewinds it exactly like the target's, because a rejected
    guess must leave no trace in either model — the draft would otherwise keep predicting
    from a sequence that never happened.
    """

    def __init__(self, model, target, device: str = "cuda", max_len: int | None = None):
        check_pair(target, model)
        self.model = model
        self.device = device
        self.max_len = max_len or min(model.cfg.max_seq_len, target.cfg.max_seq_len)
        dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
        self.caches = model.init_caches(1, self.max_len, dtype=dtype, device=device)
        self.consumed = 0
        # Anything with the model interface may be a draft — a Transformer, a quantized one,
        # a test double — so the label is best-effort rather than an assumption.
        count = getattr(model, "num_params", None)
        self.name = f"model({count() / 1e6:.0f}M)" if callable(count) else "model"

    def propose(self, tokens, g, dist):
        drafted: list[int] = []
        probs: list[torch.Tensor] = []
        for _ in range(g):
            block = (tokens + drafted)[self.consumed:]
            rows, _ = self.model(torch.tensor([block], dtype=torch.long, device=self.device),
                                 caches=self.caches, full_logits=True)
            self.consumed = len(tokens) + len(drafted)
            q = dist(rows[0][-1], set(tokens) | set(drafted))
            token = int(torch.multinomial(q, num_samples=1))
            drafted.append(token)
            probs.append(q)
        return drafted, probs

    def rollback(self, keep: int) -> None:
        for c in self.caches:
            c.rewind(keep)
        self.consumed = keep


class NgramDrafter(Drafter):
    """Draft by copying: find where the last `n` tokens occurred before, and guess that what
    followed them then follows them now.

    No model, no weights, no training, no shared tokenizer to arrange — and for the text this
    project actually generates it is not a toy. Code repeats itself constantly (a variable
    name, `for i in range(`, a docstring echoing the signature), and a chat model quoting the
    question back is copying too. Where the text is genuinely novel it finds nothing, proposes
    nothing, and the round costs exactly one target forward: plain decoding.

    Its guesses are stated as certainties (a one-hot `q`), which is what a lookup is. The
    acceptance rule then reduces to "accept with probability `p(x)`" — so a wrong guess is
    thrown out in proportion to how wrong the target thinks it is, and greedy decoding
    accepts exactly when the copy matches the argmax. Nothing special-cased.
    """

    name = "ngram"

    def __init__(self, vocab_size: int, n: int = 3, min_n: int = 1, device: str = "cpu"):
        self.vocab_size = vocab_size
        self.n = n
        self.min_n = min_n
        self.device = device

    def propose(self, tokens, g, dist):
        # Longest match first: a 3-token context that has occurred before is a much better
        # predictor than a 1-token one, and falling back only when it fails costs nothing.
        for n in range(self.n, self.min_n - 1, -1):
            if len(tokens) <= n:
                continue
            pattern = tokens[-n:]
            # Search backwards: the most recent occurrence is the most relevant one.
            for i in range(len(tokens) - n - 1, -1, -1):
                if tokens[i:i + n] == pattern:
                    guess = tokens[i + n:i + n + g]
                    if guess:
                        probs = []
                        for token in guess:
                            one_hot = torch.zeros(self.vocab_size, device=self.device)
                            one_hot[token] = 1.0
                            probs.append(one_hot)
                        return list(guess), probs
        return [], []


def accept_or_correct(p: torch.Tensor, q: torch.Tensor,
                      token: int) -> tuple[bool, int | None]:
    """The rule itself, in one place: accept the drafted `token`, or replace it.

    Kept as its own function so it can be tested against a hand-built pair of distributions
    where the right answer is known exactly — an end-to-end test can only observe samples,
    and would happily pass with `p` in place of the residual, since the two agree whenever
    the draft is greedy.
    """
    # A drafter need not live where the target does — an n-gram lookup is CPU work even when
    # the big model is on the card — so the two distributions are reconciled here rather than
    # by a rule every drafter has to remember.
    q = q.to(p.device)
    qx = float(q[token])
    ratio = float(p[token]) / qx if qx > 0 else 0.0
    if float(torch.rand(())) < min(1.0, ratio):
        return True, None
    return False, int(torch.multinomial(residual_distribution(p, q), num_samples=1))


def speculative_generate(
    target,
    draft,
    prompt_ids: list[int] | torch.Tensor,
    max_new_tokens: int = 256,
    gamma: int = 4,
    temperature: float = 0.8,
    top_k: int | None = 50,
    top_p: float | None = 0.95,
    repetition_penalty: float = 1.0,
    eos_id: int | None = None,
    device: str = "cuda",
    stats: SpecStats | None = None,
) -> Iterator[int]:
    """Yield token ids, drafted by `draft` and checked by `target`.

    The signature deliberately mirrors :func:`~aksharallm.infer.generate.stream_generate`,
    plus `gamma` (how many tokens to guess per round) and `stats`, which is filled in as the
    generator runs so a caller can read the acceptance rate without waiting for the end.
    """
    stats = stats if stats is not None else SpecStats()
    target.eval()
    drafter = draft if isinstance(draft, Drafter) else ModelDrafter(draft, target, device)
    if hasattr(drafter, "model"):
        drafter.model.eval()

    max_len = getattr(drafter, "max_len", None) or target.cfg.max_seq_len
    idx = fit_prompt(prompt_ids, max_len, device=device)
    tokens: list[int] = idx[0].tolist()
    used = set(tokens)

    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    ctx = (torch.autocast("cuda", dtype=torch.bfloat16)
           if device.startswith("cuda") else torch.autocast("cpu", enabled=False))
    t_cache = target.init_caches(1, max_len, dtype=dtype, device=device)

    def feed(consumed: int, upto: list[int]) -> tuple[torch.Tensor, int]:
        """Bring the target's cache up to `upto`, and return every row of logits.

        Everything unconsumed goes in a *single* forward — the whole point of the exercise,
        and the reason `Transformer.forward` had to learn to mask a block of tokens against a
        warm cache correctly.
        """
        block = upto[consumed:]
        out, _ = target(torch.tensor([block], dtype=torch.long, device=device),
                        caches=t_cache, full_logits=True)
        return out[0], len(upto)

    def dist(row, seen):
        return next_distribution(row, seen, temperature, top_k, top_p, repetition_penalty)

    budget = min(max_new_tokens, max_len - len(tokens))
    if budget <= 0:
        return

    with torch.no_grad(), ctx:
        # Prefill the target, deliberately leaving the last prompt token unconsumed. Every
        # round then has the same shape — one block, one set of rows — instead of a special
        # first iteration whose distributions come from somewhere else.
        _, t_consumed = feed(0, tokens[:-1])
        stats.target_forwards += 1

        while stats.emitted < budget:
            room = max_len - len(tokens) - 1
            if room <= 0:
                break
            g = max(1, min(gamma, room))

            # ---- 1. the drafter guesses (cheaply, and possibly not at all) ----------------
            drafted, q_list = drafter.propose(tokens, g, dist)
            stats.drafted += len(drafted)

            # ---- 2. the target reads all of them in ONE forward --------------------------
            rows, _ = feed(t_consumed, tokens + drafted)
            stats.target_forwards += 1
            # Row j is the distribution *after* the j-th token of the block. The block starts
            # with the one prompt/emitted token the target had not seen, so the row that
            # predicts drafted[0] is the one before the drafted tokens begin.
            base = len(tokens) - t_consumed - 1

            # ---- 3. accept, or correct once and stop --------------------------------------
            accepted: list[int] = []
            emitted_token: int | None = None
            for i, token in enumerate(drafted):
                p = dist(rows[base + i], used | set(accepted))
                ok, replacement = accept_or_correct(p, q_list[i], token)
                if ok:
                    accepted.append(token)
                    continue
                emitted_token = replacement
                stats.corrections += 1
                break
            else:
                # Every guess survived, so the target's own next distribution is already
                # computed and one extra token is free. This is why gamma drafts can yield
                # gamma+1 tokens, why the speedup is not capped at gamma, and why a round
                # where the drafter proposed *nothing* still emits a token: it is then plain
                # decoding, at the cost of exactly one target forward.
                p = dist(rows[base + len(drafted)], used | set(accepted))
                emitted_token = int(torch.multinomial(p, num_samples=1))
                stats.bonus += 1

            stats.accepted += len(accepted)
            stats.per_round.append(len(accepted))
            stats.rounds += 1

            # ---- 4. throw away what was rejected -----------------------------------------
            # Both caches hold `tokens + drafted`; everything past the accepted prefix is a
            # position the target did not choose, so it is rewound away rather than trusted.
            for token in accepted + [emitted_token]:
                tokens.append(token)
                used.add(token)
                stats.emitted += 1
                yield token
                if (eos_id is not None and token == eos_id) or stats.emitted >= budget:
                    return
            keep = len(tokens) - 1        # the last token stays unconsumed, as at prefill
            for c in t_cache:
                c.rewind(keep)
            drafter.rollback(keep)
            t_consumed = keep


def speculative_collect(target, draft, prompt_ids, **kw) -> tuple[list[int], SpecStats]:
    """The whole continuation plus its statistics — the batteries-included form."""
    stats = kw.pop("stats", None) or SpecStats()
    device = kw.get("device", "cuda")
    # A drafter need not be a model, so the window is the target's unless a draft model has
    # a shorter one of its own.
    max_len = target.cfg.max_seq_len
    if hasattr(draft, "cfg"):
        max_len = min(max_len, draft.cfg.max_seq_len)
    out = fit_prompt(prompt_ids, max_len, device=device)[0].tolist()
    out += list(speculative_generate(target, draft, prompt_ids, stats=stats, **kw))
    return out, stats


# --------------------------------------------------------------------------------------
# the terminal: run it, and prove the claim on your own checkpoints
# --------------------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    """`python -m aksharallm.infer.speculative <target> --draft <ckpt> [--compare]`

    `--compare` is the point of having a CLI at all: it decodes the same prompt twice, once
    with the target alone and once with the draft, and **checks that the two agree token for
    token** before reporting the speedup. A speedup nobody checked the output of is not a
    result.
    """
    import argparse
    import time

    from .checkpoints import CheckpointStore, InferError
    from .cli import load_model, resolve_tokenizer
    from .generate import generate
    from ..tokenizer.tokenizer import Tokenizer

    ap = argparse.ArgumentParser(
        prog="python -m aksharallm.infer.speculative",
        description="Decode with a small draft model checking against a large one. The "
                    "output is the large model's, exactly; only the speed changes.")
    ap.add_argument("target", help="the model whose output you want (run name or path)")
    ap.add_argument("--draft", default=None,
                    help="a smaller checkpoint sharing its tokenizer; omit for --ngram")
    ap.add_argument("--ngram", type=int, nargs="?", const=3, default=None, metavar="N",
                    help="draft by looking the continuation up in the text so far, with no "
                         "draft model at all (default context: 3 tokens)")
    ap.add_argument("--prompt", default="Once upon a time")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--gamma", type=int, default=4, help="tokens to guess per round")
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="0 = greedy, which is the mode --compare can check exactly")
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--compare", action="store_true",
                    help="also decode with the target alone, assert the outputs match, and "
                         "report the speedup")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--root", default=None)
    args = ap.parse_args(argv)

    if not args.draft and args.ngram is None:
        print("error: pass --draft <checkpoint>, or --ngram to draft with no model at all")
        return 2

    store = CheckpointStore(args.root)
    try:
        target_path = store.resolve(*store.identify(args.target).split("/"))
        target, t_ckpt = load_model(str(target_path), device=args.device)
        if args.draft:
            draft_path = store.resolve(*store.identify(args.draft).split("/"))
            draft_model, _ = load_model(str(draft_path), device=args.device)
            drafter = ModelDrafter(draft_model, target, device=args.device)
            label = (f"{draft_path.parent.name}/{draft_path.name}  "
                     f"{draft_model.num_params() / 1e6:.1f}M params "
                     f"({target.num_params() / max(draft_model.num_params(), 1):.0f}x smaller)")
        else:
            drafter = NgramDrafter(target.cfg.vocab_size, n=args.ngram,
                                   device=args.device)
            label = (f"{args.ngram}-token lookup in the text so far — no draft model, no "
                     f"weights, nothing to train")
    except (InferError, SpeculativeError) as exc:
        print(f"error: {exc}")
        return 2

    tok = Tokenizer(resolve_tokenizer(t_ckpt, args.tokenizer))
    ids = tok.encode(args.prompt, bos=True)

    print(f"target {target_path.parent.name}/{target_path.name}  "
          f"{target.num_params() / 1e6:.1f}M params")
    print(f"draft  {label}")
    print(f"gamma  {args.gamma}   temperature {args.temperature}   device {args.device}\n")

    kw = dict(max_new_tokens=args.max_new_tokens, temperature=args.temperature,
              top_k=args.top_k, top_p=args.top_p, eos_id=tok.eos_id, device=args.device)
    t0 = time.time()
    out, stats = speculative_collect(target, drafter, ids, gamma=args.gamma, **kw)
    spec_s = time.time() - t0
    new = len(out) - len(ids)
    print(tok.decode(out))
    print(f"\n{stats.summary()}")
    print(f"speculative: {new} tokens in {spec_s:.2f}s = {new / spec_s:.1f} tok/s")

    if args.compare:
        t0 = time.time()
        alone = generate(target, ids, **kw)
        alone_s = time.time() - t0
        n_alone = len(alone) - len(ids)
        print(f"target only: {n_alone} tokens in {alone_s:.2f}s = {n_alone / alone_s:.1f} tok/s")
        if args.temperature <= 0.0:
            # The whole claim, checked rather than asserted in a docstring. Sampling cannot
            # be compared this way (two draws differ legitimately), so say so instead of
            # printing a green tick that means nothing.
            same = out == alone
            print(f"identical output: {'yes' if same else 'NO — this is a bug'}"
                  f" ({len(out)} vs {len(alone)} tokens)")
            if not same:
                return 1
        else:
            print("identical output: not checkable while sampling — rerun with "
                  "--temperature 0 for the exact comparison")
        print(f"speedup: {alone_s / spec_s:.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
