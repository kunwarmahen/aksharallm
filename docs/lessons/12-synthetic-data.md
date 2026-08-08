---
id: synthetic-data
title: Making the training set, and checking it twice
doc: docs/14-synthetic-data.md
files:
  - aksharallm/synth/verify.py
  - aksharallm/synth/filters.py
verify: tests/test_synth.py::test_tests_that_never_call_the_function_are_caught_by_the_stub_run
prereqs: [eval]
minutes: 30
summary: When a bigger model writes your data, generating is four lines and deciding whether to believe it is the whole job.
---

# 12. Making the training set, and checking it twice

Some datasets do not exist to download. "Python exercises with tests that actually pass, at
the level a 300M model can learn from" is not a file on the Hub — but a bigger model already
running on this machine can write them.

Generating is four lines. Everything else exists because of one sentence:

> **Synthetic data is the easiest way to make a model worse while its training loss
> improves.**

Duplicate-heavy, low-diversity or subtly wrong data trains *beautifully*. The loss falls, the
curve is smooth, and the model that comes out is fluent and useless. Nothing in the training
run can see it happen — which is why the eval harness came first, and why the previous lesson
is this one's prerequisite.

## Three defences

**Diversity comes from a grid, not a temperature.** Ask one prompt 5,000 times at temperature
0.9 and you get FizzBuzz, palindromes and string-reversal 5,000 times with different variable
names. The prompt is assembled from 480 structurally different cells instead — topic × twist ×
difficulty — walked in shuffled order, so 200 samples use 200 different cells. That is a
coverage guarantee; a temperature is not.

**Near-duplicates, not just duplicates.** Two samples differing by a variable name are not
equal, so exact matching catches nothing while the set quietly becomes fifty paraphrases of
four problems. The check is on overlapping five-word runs compared by Jaccard similarity.

**The tests are executed.** This is the one that makes the Python recipe worth trusting: the
teacher writes asserts, the sandbox runs them, and a sample whose tests fail is dropped.

---

## Exercise: the check that checks the check

"The tests passed" is weaker than it sounds. Ask a model for a function and some tests and it
will occasionally write

```python
assert callable(dedupe)
assert dedupe.__name__ == "dedupe"
```

which passes, mentions the function by name — all a static check can confirm — and would pass
against a function with no body at all.

So every sample is run **twice**: once as written, and once with the entry point's body
replaced by `raise NotImplementedError`. The first must pass and the second must **fail**. If
the tests still pass with the implementation deleted, they never tested it.

1. Run the check. It passes — it feeds exactly the vacuous tests above and asserts the sample
   is rejected.
2. In `aksharallm/synth/verify.py`, find where the stubbed run's result decides the verdict
   and accept the sample regardless of what the stub did.
3. Run the check. **It should fail**: a sample that proves nothing is now being kept.
4. Put it back. Green.

> **What you just saw.** And the honest limit, which `verify.py` states in its own docstring:
> this catches tests that never exercise the function, and does *not* catch a weak-but-real
> assertion like `isinstance(f(x), list)`. Knowing where a check stops is worth more than
> believing it catches everything.

## The number that tells you what to fix

A pass rate alone says nothing. These three runs all report 30%:

| lost to | what to change |
|---|---|
| `tests_failed` | the prompt, or a better teacher |
| `near_duplicate` | a wider seed grid, or a different seed |
| `unparseable` | the output format — retrying will not help |

That is why every rejection is counted **by reason** and the rejected text is kept. On this
machine that loop paid off in an hour: the first batch kept 25%, every loss was
`tests_failed`, and reading three of them showed correct solutions with *wrong expected
values* in the tests. One rule added to the prompt took it to 58%.

```bash
python -m aksharallm.synth gen python --name my-first --n 6 --max-asks 12
python -m aksharallm.synth show my-first --samples 2 --rejects 3
```

Read the rejects. That is the lesson.
