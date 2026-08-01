"""The learning path: the repo as a course you can actually do.

Everything needed to teach this project already existed — `docs/00–14` explain the why, the
skills explain the how, 600-odd tests say what is true, the Playground runs a real
checkpoint and the Code tab explains any line back to you with a local model. What was
missing was **ordering, and something to do.**

A lesson is a triple: read the doc, open the file, break it and watch a real test go red.
The exercises are mostly bugs that actually happened here, which is why they are worth
doing — `is_causal` during single-token decode masks away the entire KV cache, and the model
still *trains* perfectly while generating garbage. Nobody would invent that.

* :mod:`~aksharallm.learn.lessons`  — the files, their frontmatter, the prereq graph, and the
  validation that keeps them from rotting.
* :mod:`~aksharallm.learn.progress` — who has done what, and the red-then-green rule that
  makes "done" mean the exercise happened.
* :mod:`~aksharallm.learn.check`    — running one pytest node and reporting it usefully.

`python -m aksharallm.learn` drives all of it from a terminal; the portal's **Learn** tab is
a view over the same three modules.
"""

from .check import CheckError, CheckResult, run as run_check
from .lessons import LearnError, Lesson, get, load_all, validate
from .progress import Progress, gate

__all__ = ["CheckError", "CheckResult", "LearnError", "Lesson", "Progress", "gate", "get",
           "load_all", "run_check", "validate"]
