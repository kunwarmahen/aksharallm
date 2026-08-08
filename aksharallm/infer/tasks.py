"""What to ask a model, and how to tell whether it answered.

Testing a half-trained model by typing whatever comes to mind is a bad way to notice
progress: you change the prompt every time, so you cannot tell an improving model from a
lucky sample. This module is the fixed set of things to ask — the same prompts at step
5,000 and at step 40,000 — plus the Python tasks that have an actual right answer.

Three kinds:

* **probes** (`complete`) — short prompts that expose specific abilities of a *base* model.
  Each one says what "good" looks like at this scale, because a 300M model trained on 10B
  tokens is not going to know who wrote *Hamlet*, and being disappointed by that is a
  misunderstanding rather than a bug.
* **chat prompts** — for a model that has been through SFT. Inert until Phase 3.
* **code tasks** — a function signature and a docstring, plus asserts. The model writes the
  body; `sandbox.run_tests` runs the asserts. This is the only place in the project where a
  generation is graded automatically rather than read.

The code tasks are deliberately easier than HumanEval's. `aksharallm.eval.evaluate` has the
real benchmark; these exist so that a model 15% of the way through pretraining shows a
signal other than zero, which is what makes them useful for watching a run.

Read with: docs/07-inference.md -- the chapter this implements; it ends with the order to read
these files in.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class Probe:
    """One fixed prompt, and what a good answer would look like."""

    id: str
    group: str
    prompt: str
    expect: str

    def as_dict(self) -> dict:
        return {"id": self.id, "group": self.group, "prompt": self.prompt,
                "expect": self.expect}


#: Base-model probes. Grouped by the ability they test, because "the model is bad" is never
#: as useful as "the model has grammar and no facts", which is the expected state of a
#: 300M model a fifth of the way through its budget.
PROBES = [
    Probe("fluency", "prose",
          "The city of Venice is built on a lagoon, and",
          "Grammatical, on-topic English that keeps its subject for a few sentences. This "
          "is the first thing to appear in training and the last thing to break."),
    Probe("facts", "prose",
          "The three states of matter are",
          "'solid, liquid and gas'. Simple, high-frequency facts arrive early; anything "
          "specific (dates, names, numbers) will be confidently wrong for a long time."),
    Probe("definition", "prose",
          "In machine learning, a loss function is",
          "A definition rather than a change of subject. FineWeb-Edu is educational text, "
          "so explanatory register is exactly what this corpus should teach."),
    Probe("list", "structure",
          "Here are five things to pack for a camping trip:\n1.",
          "Five distinct numbered items, numbered in order. Losing count, or looping on "
          "one item, is the classic undertrained-model failure."),
    Probe("arithmetic", "reasoning",
          "If a train leaves at 3pm and takes 2 hours and 30 minutes, it arrives at",
          "'5:30pm'. Expect this to be wrong at 300M — the tokenizer splits numbers at "
          "three digits precisely so the model cannot memorise its way out of arithmetic."),
    Probe("code-switch", "code",
          "Here is a Python function that reverses a string:\n\ndef",
          "A syntactically valid function. This is the one to watch: it tells you whether "
          "the 15% code in the blend is reaching the model at all."),
    Probe("repetition", "health",
          "The quick brown fox jumps over the lazy dog. The quick brown fox",
          "It should continue and then *move on*. A model that repeats this line forever "
          "is showing you degenerate looping — try a repetition penalty above 1.0."),
]

#: Chat prompts. Every one of these returns noise from a base model, which is why the tab
#: refuses to send them to one.
CHAT_PROMPTS = [
    Probe("greeting", "chat", "Hello! Who are you?",
          "A short, coherent self-introduction that stays in the assistant role and stops."),
    Probe("explain", "chat", "Explain what a neural network is, in two sentences.",
          "Two sentences. Following a length instruction is a good early sign that SFT "
          "took."),
    Probe("refusal", "chat", "What is the population of Mars?",
          "Ideally 'nobody lives on Mars' rather than an invented number — but a small "
          "model will usually invent the number."),
    Probe("format", "chat", "List three prime numbers as a bulleted list.",
          "Three bullets, three primes. Tests instruction-following and format at once."),
    Probe("code", "chat", "Write a Python function that returns the largest item in a list.",
          "A correct function in a code block. This is the chat-side equivalent of the "
          "code tasks below."),
]


@dataclass
class CodeTask:
    """A function to write, and asserts that say whether it was written correctly.

    `prompt` ends mid-function on purpose — after the docstring, with the body missing.
    That is the shape a *base* model can answer: it is text continuation, not an
    instruction. The same task works for a chat model through the `instruction` form.
    """

    id: str
    title: str
    prompt: str
    tests: str
    entry_point: str
    difficulty: str = "easy"

    @property
    def instruction(self) -> str:
        """The same task phrased for a chat model."""
        return (f"Write a Python function `{self.entry_point}` that does the following. "
                f"Reply with the function only.\n\n```python\n{self.prompt.rstrip()}\n```")

    def as_dict(self) -> dict:
        return {"id": self.id, "title": self.title, "prompt": self.prompt,
                "tests": self.tests, "entry_point": self.entry_point,
                "difficulty": self.difficulty, "instruction": self.instruction}


CODE_TASKS = [
    CodeTask(
        "add", "Add two numbers", difficulty="trivial",
        entry_point="add",
        prompt='def add(a, b):\n    """Return the sum of a and b."""\n',
        tests="assert add(1, 2) == 3\nassert add(-5, 5) == 0\nassert add(0.5, 0.25) == 0.75\n",
    ),
    CodeTask(
        "reverse_string", "Reverse a string", difficulty="trivial",
        entry_point="reverse_string",
        prompt='def reverse_string(s):\n    """Return s reversed."""\n',
        tests="assert reverse_string('abc') == 'cba'\n"
              "assert reverse_string('') == ''\n"
              "assert reverse_string('a') == 'a'\n",
    ),
    CodeTask(
        "is_even", "Test for even numbers", difficulty="trivial",
        entry_point="is_even",
        prompt='def is_even(n):\n    """Return True if n is even, False otherwise."""\n',
        tests="assert is_even(2) is True\nassert is_even(3) is False\n"
              "assert is_even(0) is True\nassert is_even(-4) is True\n",
    ),
    CodeTask(
        "max_in_list", "Largest item in a list", difficulty="easy",
        entry_point="max_in_list",
        prompt='def max_in_list(items):\n'
               '    """Return the largest number in the list `items`.\n\n'
               '    The list is never empty.\n    """\n',
        tests="assert max_in_list([1, 5, 3]) == 5\n"
              "assert max_in_list([-2, -7]) == -2\n"
              "assert max_in_list([4]) == 4\n",
    ),
    CodeTask(
        "count_vowels", "Count the vowels", difficulty="easy",
        entry_point="count_vowels",
        prompt='def count_vowels(text):\n'
               '    """Return how many vowels (a, e, i, o, u) appear in text.\n\n'
               '    Upper and lower case both count.\n    """\n',
        tests="assert count_vowels('hello') == 2\n"
              "assert count_vowels('AEIOU') == 5\n"
              "assert count_vowels('xyz') == 0\n",
    ),
    CodeTask(
        "is_palindrome", "Palindrome check", difficulty="easy",
        entry_point="is_palindrome",
        prompt='def is_palindrome(s):\n'
               '    """Return True if s reads the same forwards and backwards.\n\n'
               '    Case is ignored; spaces and punctuation are not removed.\n    """\n',
        tests="assert is_palindrome('racecar') is True\n"
              "assert is_palindrome('Racecar') is True\n"
              "assert is_palindrome('hello') is False\n",
    ),
    CodeTask(
        "fizzbuzz", "FizzBuzz for one number", difficulty="easy",
        entry_point="fizzbuzz",
        prompt='def fizzbuzz(n):\n'
               '    """Return "Fizz" if n divides by 3, "Buzz" if by 5, "FizzBuzz" if by\n'
               '    both, and str(n) otherwise.\n    """\n',
        tests="assert fizzbuzz(3) == 'Fizz'\nassert fizzbuzz(5) == 'Buzz'\n"
              "assert fizzbuzz(15) == 'FizzBuzz'\nassert fizzbuzz(7) == '7'\n",
    ),
    CodeTask(
        "unique", "Unique items, order kept", difficulty="medium",
        entry_point="unique",
        prompt='def unique(items):\n'
               '    """Return a list of the items with duplicates removed.\n\n'
               '    The original order of first appearance is preserved.\n    """\n',
        tests="assert unique([1, 2, 1, 3, 2]) == [1, 2, 3]\n"
              "assert unique([]) == []\n"
              "assert unique(['a', 'a']) == ['a']\n",
    ),
    CodeTask(
        "fibonacci", "Nth Fibonacci number", difficulty="medium",
        entry_point="fibonacci",
        prompt='def fibonacci(n):\n'
               '    """Return the nth Fibonacci number, where fibonacci(0) == 0 and\n'
               '    fibonacci(1) == 1.\n    """\n',
        tests="assert fibonacci(0) == 0\nassert fibonacci(1) == 1\n"
              "assert fibonacci(10) == 55\n",
    ),
    CodeTask(
        "word_count", "Count words", difficulty="medium",
        entry_point="word_count",
        prompt='def word_count(text):\n'
               '    """Return a dict mapping each whitespace-separated word in text to\n'
               '    the number of times it appears.\n    """\n',
        tests="assert word_count('a b a') == {'a': 2, 'b': 1}\n"
              "assert word_count('') == {}\n",
    ),
]

TASKS_BY_ID = {t.id: t for t in CODE_TASKS}

#: A fenced code block, for pulling the function out of a chat model's prose answer.
FENCE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)(?:\n```|\Z)", re.DOTALL)


def extract_code(completion: str, prompt: str = "", entry_point: str = "") -> str:
    """The Python worth executing, out of whatever the model produced.

    Two shapes have to be handled, because the same task is asked of two kinds of model:

    * A **base** model continues the prompt, so the completion is a bare function body and
      then — invariably — whatever it felt like writing next: a second function, a chunk of
      prose, a fresh `import`. Everything from the first line at column zero onwards is
      dropped. That is the standard HumanEval post-processing, and it is not cheating: the
      model was asked to continue a file, and a file does not stop.
    * A **chat** model answers in prose with a fenced code block, so the fence wins if there
      is one.

    Returns the body only for the base form (the caller re-joins it to the prompt), and a
    whole function for the fenced form.
    """
    text = completion.replace("\r\n", "\n")

    fenced = FENCE_RE.search(text)
    if fenced and (not prompt or "def " in fenced.group(1)):
        return fenced.group(1).rstrip()

    # Base-model continuation: keep lines until one starts at column zero. Blank lines and
    # comment lines are kept, so a body with paragraph breaks in it survives.
    kept: list[str] = []
    for line in text.split("\n"):
        if line.strip() and not line[0].isspace():
            break
        kept.append(line)
    body = "\n".join(kept).rstrip()
    if not body.strip() and entry_point and f"def {entry_point}" in text:
        # It restated the signature instead of continuing. Take the whole thing and let the
        # sandbox judge it.
        return text.rstrip()
    return body


def assemble(task: CodeTask, completion: str, chat: bool = False) -> str:
    """The full program to run: the model's function, then the asserts.

    For a base completion the prompt is prepended, because the signature and docstring the
    model was *shown* are part of the function and were never generated.
    """
    code = extract_code(completion, task.prompt, task.entry_point)
    if chat or code.lstrip().startswith("def "):
        program = code
    else:
        program = task.prompt.rstrip("\n") + "\n" + code
    return f"{program}\n\n{task.tests}"


def catalogue() -> dict:
    """Everything the playground offers, in one response."""
    return {
        "probes": [p.as_dict() for p in PROBES],
        "chat": [p.as_dict() for p in CHAT_PROMPTS],
        "tasks": [t.as_dict() for t in CODE_TASKS],
    }
