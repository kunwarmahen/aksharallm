"""Problems a 300M model cannot do in its head, and the conversations that teach a tool.

Each problem has an exact answer computed here, so a reply is right or wrong with no judge.
They are the questions small models are famously bad at: multiplying five-digit numbers,
adding a list, counting letters, the days between two dates, compound interest, the n-th
prime. Every one is a line of Python.

A teaching conversation is three turns, in ChatML (`Tokenizer.render_chat`):

    user        What is 48213 times 9071?
    assistant   <tool>
                print(48213 * 9071)
                </tool>
    tool        437340123                       (the sandbox's stdout, never trained on)
    assistant   The answer is 437340123.

Only the assistant turns carry loss, so the model learns to WRITE the call and to READ the
result, and is never trained to predict what the tool says. That last part is what makes it
use the tool: a model that could predict the output would not need it.

One kind is held out of training entirely (`HELD_OUT_KIND`, the remainder questions), so the
evaluation can ask whether the model learned "use the tool for arithmetic" or only "use the
tool for these five templates".

Read with: docs/06-posttraining.md -- the chapter this implements; it ends with the order to
read these files in.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np

HELD_OUT_KIND = "remainder"
TOOL_OPEN, TOOL_CLOSE = "<tool>", "</tool>"


@dataclass
class Problem:
    kind: str
    question: str
    code: str        # the call a good answer makes
    answer: str      # exact, as the final line should state it


def _primes(n: int) -> list[int]:
    out, k = [], 2
    while len(out) < n:
        if all(k % p for p in out if p * p <= k):
            out.append(k)
        k += 1
    return out


_PRIMES = _primes(400)
WORDS = ("strawberry", "mississippi", "bookkeeper", "banana", "assessment", "committee",
         "parallel", "occurrence", "nevertheless", "abracadabra", "tennessee", "possession",
         "referee", "successful", "programming", "independence", "unbelievable", "coffee")


def make(kind: str, rng: np.random.Generator) -> Problem:
    r = lambda a, b: int(rng.integers(a, b))  # noqa: E731
    pick = lambda xs: xs[int(rng.integers(len(xs)))]  # noqa: E731
    if kind == "multiply":
        a, b = r(1000, 99999), r(100, 9999)
        q = pick([f"What is {a} times {b}?", f"Multiply {a} by {b}.", f"Compute {a} * {b}."])
        return Problem(kind, q, f"print({a} * {b})", str(a * b))
    if kind == "sum":
        xs = [r(10, 9999) for _ in range(r(6, 11))]
        lst = ", ".join(map(str, xs))
        q = pick([f"What is the sum of {lst}?", f"Add these numbers: {lst}.",
                  f"Total of {lst}?"])
        return Problem(kind, q, f"print(sum([{lst}]))", str(sum(xs)))
    if kind == "letters":
        w, ch = pick(WORDS), None
        ch = pick(sorted(set(w)))
        q = pick([f'How many times does the letter "{ch}" appear in "{w}"?',
                  f'Count the "{ch}"s in the word {w}.'])
        return Problem(kind, q, f'print("{w}".count("{ch}"))', str(w.count(ch)))
    if kind == "days":
        d1 = dt.date(2000, 1, 1) + dt.timedelta(days=r(0, 9000))
        d2 = d1 + dt.timedelta(days=r(1, 3000))
        q = pick([f"How many days are there from {d1.isoformat()} to {d2.isoformat()}?",
                  f"Count the days between {d1.isoformat()} and {d2.isoformat()}."])
        code = (f"import datetime as dt\nprint((dt.date.fromisoformat('{d2}') - "
                f"dt.date.fromisoformat('{d1}')).days)")
        return Problem(kind, q, code, str((d2 - d1).days))
    if kind == "interest":
        p, rate, n = r(100, 20000), r(1, 12), r(2, 30)
        q = pick([f"If you invest {p} dollars at {rate}% interest compounded yearly, how much "
                  f"do you have after {n} years, rounded to the cent?",
                  f"{p} dollars grows at {rate}% a year for {n} years. What is it worth, to "
                  f"the cent?"])
        return Problem(kind, q, f"print(round({p} * (1 + {rate} / 100) ** {n}, 2))",
                       str(round(p * (1 + rate / 100) ** n, 2)))
    if kind == "prime":
        n = r(20, 400)
        q = pick([f"What is the {n}th prime number?", f"Which prime is number {n} in the list "
                  f"of primes?"])
        code = ("ps = []\nk = 2\nwhile len(ps) < %d:\n    if all(k %% p for p in ps if p * p "
                "<= k):\n        ps.append(k)\n    k += 1\nprint(ps[-1])" % n)
        return Problem(kind, q, code, str(_PRIMES[n - 1]))
    if kind == "remainder":
        a, b = r(10_000, 9_999_999), r(7, 997)
        q = pick([f"What is the remainder when {a} is divided by {b}?", f"Compute {a} mod {b}."])
        return Problem(kind, q, f"print({a} % {b})", str(a % b))
    raise ValueError(kind)


KINDS = ("multiply", "sum", "letters", "days", "interest", "prime", "remainder")
TRAIN_KINDS = tuple(k for k in KINDS if k != HELD_OUT_KIND)


def conversation(p: Problem, tool_output: str | None = None) -> list[dict]:
    """The three-turn teaching conversation. `tool_output` defaults to the true answer,
    which is what the sandbox prints for every `code` here (a test runs them all)."""
    out = p.answer if tool_output is None else tool_output
    return [{"role": "user", "content": p.question},
            {"role": "assistant", "content": f"{TOOL_OPEN}\n{p.code}\n{TOOL_CLOSE}"},
            {"role": "tool", "content": out},
            {"role": "assistant", "content": f"The answer is {p.answer}."}]


def problems(n: int, seed: int, kinds=TRAIN_KINDS) -> list[Problem]:
    rng = np.random.default_rng(seed)
    return [make(kinds[i % len(kinds)], rng) for i in range(n)]
