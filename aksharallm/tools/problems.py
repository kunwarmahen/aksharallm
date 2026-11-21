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
tool for these six templates".

Two problem SETS (`SETS`). `v1` trains six kinds and holds out one; the first run on it
called the tool on every held-out question and wrote the wrong operation every time. `v2`
trains twenty kinds and holds out four (remainder again, gcd, digit sum, day of the year),
at the same 30,000 conversations, to ask whether breadth is what turns templates into a
habit. Each held-out kind sits near a trained one (lcm, days, letters), so a model that
learned the idea has something to stand on, and one that learned templates does not.

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
    return _make_v2(kind, rng, r, pick)


def _make_v2(kind: str, rng, r, pick) -> Problem:
    """The kinds `v2` adds: fourteen to train on, three more to hold out."""
    if kind == "subtract":
        a, b = r(10_000, 9_999_999), r(1000, 999_999)
        q = pick([f"What is {a} minus {b}?", f"Subtract {b} from {a}.", f"Compute {a} - {b}."])
        return Problem(kind, q, f"print({a} - {b})", str(a - b))
    if kind == "divide":
        a, b = r(1000, 999_999), r(7, 997)
        q = pick([f"What is {a} divided by {b}, to two decimal places?",
                  f"Divide {a} by {b} and round to 2 decimals."])
        return Problem(kind, q, f"print(round({a} / {b}, 2))", str(round(a / b, 2)))
    if kind == "power":
        a, b = r(2, 30), r(3, 12)
        q = pick([f"What is {a} to the power of {b}?", f"Compute {a}^{b}.",
                  f"What is {a} raised to the {b}th power?"])
        return Problem(kind, q, f"print({a} ** {b})", str(a ** b))
    if kind == "sqrt":
        a = r(2, 999_999)
        q = pick([f"What is the square root of {a}, to three decimal places?",
                  f"Square root of {a}, rounded to 3 decimals?"])
        return Problem(kind, q, f"print(round({a} ** 0.5, 3))", str(round(a ** 0.5, 3)))
    if kind == "mean":
        xs = [r(1, 999) for _ in range(r(5, 10))]
        lst = ", ".join(map(str, xs))
        q = pick([f"What is the average of {lst}, to two decimal places?",
                  f"Mean of {lst}, rounded to 2 decimals?"])
        return Problem(kind, q, f"print(round(sum([{lst}]) / {len(xs)}, 2))",
                       str(round(sum(xs) / len(xs), 2)))
    if kind in ("maximum", "minimum"):
        xs = [r(-9999, 99_999) for _ in range(r(8, 15))]
        lst = ", ".join(map(str, xs))
        fn, word = ("max", "largest") if kind == "maximum" else ("min", "smallest")
        q = pick([f"Which is the {word} of {lst}?", f"Find the {word} number in: {lst}."])
        return Problem(kind, q, f"print({fn}([{lst}]))", str(max(xs) if fn == "max" else min(xs)))
    if kind == "factorial":
        n = r(8, 25)
        q = pick([f"What is {n} factorial?", f"Compute {n}!."])
        import math
        return Problem(kind, q, f"import math\nprint(math.factorial({n}))", str(math.factorial(n)))
    if kind == "lcm":
        a, b = r(6, 9999), r(6, 9999)
        q = pick([f"What is the least common multiple of {a} and {b}?", f"LCM of {a} and {b}?"])
        import math
        return Problem(kind, q, f"import math\nprint(math.lcm({a}, {b}))", str(math.lcm(a, b)))
    if kind == "vowels":
        w = pick(WORDS)
        q = pick([f'How many vowels are in the word "{w}"?', f"Count the vowels in {w}."])
        return Problem(kind, q, f'print(sum(ch in "aeiou" for ch in "{w}"))',
                       str(sum(ch in "aeiou" for ch in w)))
    if kind == "length":
        w = "".join(pick(WORDS) for _ in range(r(2, 4)))
        q = pick([f'How many letters long is "{w}"?', f'What is the length of the string "{w}"?'])
        return Problem(kind, q, f'print(len("{w}"))', str(len(w)))
    if kind == "percent":
        p, x = r(1, 99), r(100, 999_999)
        q = pick([f"What is {p}% of {x}?", f"Compute {p} percent of {x}."])
        return Problem(kind, q, f"print({p} * {x} / 100)", str(p * x / 100))
    if kind == "seconds":
        h, m = r(1, 99), r(0, 59)
        q = pick([f"How many seconds are in {h} hours and {m} minutes?",
                  f"Convert {h} h {m} min to seconds."])
        return Problem(kind, q, f"print({h} * 3600 + {m} * 60)", str(h * 3600 + m * 60))
    if kind == "fibonacci":
        n = r(15, 90)
        a, b = 0, 1
        for _ in range(n):
            a, b = b, a + b
        q = pick([f"What is the {n}th Fibonacci number?", f"Fibonacci number {n}?"])
        return Problem(kind, q, f"a, b = 0, 1\nfor _ in range({n}):\n    a, b = b, a + b\nprint(a)",
                       str(a))
    # ---- held out of v2's training ----
    if kind == "gcd":
        g = r(2, 99)
        a, b = g * r(10, 999), g * r(10, 999)
        q = pick([f"What is the greatest common divisor of {a} and {b}?", f"GCD of {a} and {b}?"])
        import math
        return Problem(kind, q, f"import math\nprint(math.gcd({a}, {b}))", str(math.gcd(a, b)))
    if kind == "digitsum":
        n = r(10**8, 10**15)
        q = pick([f"What is the sum of the digits of {n}?", f"Add up the digits of {n}."])
        return Problem(kind, q, f"print(sum(int(d) for d in str({n})))",
                       str(sum(int(d) for d in str(n))))
    if kind == "dayofyear":
        d = dt.date(1990, 1, 1) + dt.timedelta(days=r(0, 15000))
        q = pick([f"What day of the year is {d.isoformat()}?",
                  f"{d.isoformat()} is which day of its year, counting January 1 as day 1?"])
        code = f"import datetime as dt\nprint(dt.date.fromisoformat('{d}').timetuple().tm_yday)"
        return Problem(kind, q, code, str(d.timetuple().tm_yday))
    raise ValueError(kind)


KINDS = ("multiply", "sum", "letters", "days", "interest", "prime", "remainder")
TRAIN_KINDS = tuple(k for k in KINDS if k != HELD_OUT_KIND)
MORE_KINDS = ("subtract", "divide", "power", "sqrt", "mean", "maximum", "minimum", "factorial",
              "lcm", "vowels", "length", "percent", "seconds", "fibonacci")

#: Which kinds a set trains on and which it holds out. The evaluation asks every kind of the
#: set, trained and held out, so two models compared on one set answer the same problems.
SETS = {
    "v1": {"train": TRAIN_KINDS, "held_out": (HELD_OUT_KIND,)},
    "v2": {"train": TRAIN_KINDS + MORE_KINDS,
           "held_out": ("remainder", "gcd", "digitsum", "dayofyear")},
}


def eval_kinds(name: str) -> tuple[str, ...]:
    """The kinds an evaluation on set `name` asks, trained first then held out. For v1 this is
    `KINDS` in its original order, so a v1 report means what it always meant."""
    if name == "v1":
        return KINDS
    return SETS[name]["train"] + SETS[name]["held_out"]


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
