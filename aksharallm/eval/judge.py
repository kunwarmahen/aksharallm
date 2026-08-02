"""Grading open-ended answers with a second, larger model.

Everything else in this harness has an objective answer: a gold letter, a number, a test
that passes. That covers a lot and misses the thing you would actually notice about a model
— whether its answers are any good. Fluency, following a length instruction, admitting it
does not know, not looping. Those need a reader, and the reader here is a local Ollama
model given a rubric and asked for a number.

This is the suite that earns its place *after* synthetic data exists, and it is why the
harness was built before the synthetic-data pipeline rather than after: training on
generated text is the easiest way to make a model that scores better on loss and worse to
talk to, and nothing else in this file drawer can see that happen.

What an LLM-judge is and is not
-------------------------------
It is a **consistent** reader, not a correct one. A judge is biased toward long answers,
toward its own writing style, and toward confident prose over hedged prose — the same
directions every time, which is precisely what makes the comparison between two of *our*
checkpoints meaningful even when the absolute number is not. Three things keep it honest
here:

* **temperature 0** and a fixed prompt, so re-running the same answers gives the same
  grades.
* **A stated rubric per prompt**, written when the prompt was, so the standard does not
  drift with the judge's mood or its model version.
* **The judge never sees which checkpoint produced the answer**, or any other answer to
  compare against. It grades one answer against one rubric.

The judge model is configured separately from the Code tab's explainer (`judge:` in
`configs/portal.yaml`) because they want opposite things: the explainer wants a small model
that can run beside a training run, and the judge wants the largest model on the machine
and does not care that it takes a minute per answer.

Read with: docs/12-eval.md -- the chapter this implements; it ends with the order to read these
files in.
"""

from __future__ import annotations

import json
import re
import statistics
from dataclasses import dataclass

from ..portal.explain import ExplainConfig, Ollama
from ..portal.runs import RunError
from .sources import EvalError


class JudgeConfig(ExplainConfig):
    """The explainer's config, reading `judge:` instead of `explain:`.

    Same client, same error messages, same reload-on-mtime contract — see
    :class:`~aksharallm.portal.explain.ExplainConfig`. Only the defaults differ, and they
    differ in the direction of "quality, slowly": a bigger model, no sampling, and a longer
    timeout, because a judge is run twelve times at the end of an evaluation rather than
    interactively while someone waits.
    """

    SECTION = "judge"
    ENV_PREFIX = "AKSHARALLM_JUDGE"


def default_config(root=None) -> JudgeConfig:
    cfg = JudgeConfig(path=None)
    cfg.model = "qwen3.5:27b"
    cfg.temperature = 0.0
    cfg.num_predict = 400
    cfg.num_ctx = 8192
    cfg.timeout_s = 600.0
    cfg.think = False
    from ..infer.checkpoints import repo_root
    cfg.path = (root or repo_root()) / "configs" / "portal.yaml"
    cfg.reload()
    return cfg


SYSTEM = """\
You grade answers written by a small language model that is being trained from scratch. You
are a strict but fair marker, and your grades are compared across versions of that model, so
consistency matters more than generosity.

You will be given a prompt, a rubric describing what a good answer contains, and the
model's answer. Grade the answer from 1 to 5:

  1  wrong, empty, off-topic, or degenerate (repeats itself, trails off mid-word)
  2  on topic but mostly incorrect, or ignores what the prompt asked for
  3  partly right: the gist is there with real errors or a missed instruction
  4  correct and useful, with a minor flaw in accuracy, format or length
  5  correct, complete, and does exactly what was asked, no more

Rules:
* Grade against the rubric, not against your own preferred answer.
* A confidently invented fact is worse than admitting ignorance. Mark it down hard.
* Ignoring an explicit instruction about length, count or format is at most a 3, however
  well written the answer is.
* Do not reward length. A short correct answer beats a long one.
* The answer comes from a very small model. That is context, not an excuse: grade what is
  in front of you.

Reply with JSON only, on one line, in exactly this form:
{"score": <1-5>, "reason": "<one short sentence>"}
"""


def build_messages(prompt: str, rubric: str, answer: str) -> list[dict]:
    body = (f"PROMPT GIVEN TO THE MODEL:\n{prompt}\n\n"
            f"RUBRIC — what a good answer contains:\n{rubric}\n\n"
            f"THE MODEL'S ANSWER:\n{answer if answer.strip() else '(empty)'}\n\n"
            "Grade it. JSON only.")
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": body}]


#: The first JSON object in the reply. A model told "JSON only" still sometimes writes a
#: sentence first, and re-asking would cost a minute per answer to fix punctuation.
JSON_RE = re.compile(r"\{.*?\}", re.DOTALL)
SCORE_RE = re.compile(r"\b([1-5])\b")


def parse_grade(text: str) -> tuple[int | None, str]:
    """The score and reason out of the judge's reply, however it wrapped them.

    Returns `(None, ...)` when nothing gradeable came back, which the caller records as an
    *ungraded* item rather than as a zero. A judge that failed to answer is missing data;
    scoring it 1 would quietly punish the model being tested for the judge's mistake.
    """
    found = JSON_RE.search(text)
    if found:
        try:
            data = json.loads(found.group(0))
            score = int(data.get("score"))
            if 1 <= score <= 5:
                return score, str(data.get("reason") or "").strip()
        except (ValueError, TypeError):
            pass
    loose = SCORE_RE.search(text)
    if loose:
        return int(loose.group(1)), text.strip()[:200]
    return None, text.strip()[:200]


@dataclass
class Grade:
    id: str
    group: str
    prompt: str
    answer: str
    score: int | None
    reason: str


def available(cfg: JudgeConfig) -> tuple[bool, str]:
    """Can the judge run at all? Checked before generating, not after.

    Generating twelve answers from a 300M model on the CPU takes minutes; discovering
    afterwards that `ollama serve` is not running would waste all of it.
    """
    try:
        models = Ollama(cfg).models()
    except RunError as exc:
        return False, str(exc)
    names = {m["name"] for m in models}
    if cfg.model not in names and cfg.model.split(":")[0] not in {n.split(":")[0] for n in names}:
        return False, (f"the judge model '{cfg.model}' is not pulled. Available: "
                       f"{', '.join(sorted(names)) or 'none'}.  "
                       f"Pull it with:  ollama pull {cfg.model}")
    return True, f"judging with {cfg.model} at {cfg.host}"


def grade_one(cfg: JudgeConfig, item, answer: str, model: str | None = None) -> Grade:
    client = Ollama(cfg)
    reply = ""
    stream = client.chat(build_messages(item.prompt, item.rubric, answer), model=model)
    try:
        for kind, piece in stream:
            if kind == "delta":
                reply += piece
    finally:
        stream.close()
    score, reason = parse_grade(reply)
    return Grade(id=item.id, group=item.group, prompt=item.prompt, answer=answer,
                 score=score, reason=reason)


def run(cfg: JudgeConfig, items, answers: list[str], model: str | None = None,
        progress=None) -> dict:
    """Grade every answer, and summarise.

    The headline `score` is the mean grade **rescaled to 0-1**, so it sits in the same
    table as an accuracy without anyone having to remember which column is out of five.
    `mean` keeps the raw 1-5 number, which is the one to quote in a sentence.
    """
    ok, note = available(cfg)
    if not ok:
        raise EvalError(note)

    grades: list[Grade] = []
    for i, (item, answer) in enumerate(zip(items, answers)):
        try:
            grades.append(grade_one(cfg, item, answer, model=model))
        except RunError as exc:
            grades.append(Grade(id=item.id, group=item.group, prompt=item.prompt,
                                answer=answer, score=None, reason=f"judge failed: {exc}"))
        if progress:
            progress(i + 1, len(items), "judge")

    valid = [g.score for g in grades if g.score is not None]
    by_group: dict[str, list[int]] = {}
    for g in grades:
        if g.score is not None:
            by_group.setdefault(g.group, []).append(g.score)

    mean = statistics.fmean(valid) if valid else None
    return {
        "n": len(grades),
        "graded": len(valid),
        "mean": mean,
        # 1-5 maps to 0-1 as (mean-1)/4: a model that scores 1 on everything is at the
        # floor, not at 20%. Anything else would flatter a model that answers nothing.
        "score": None if mean is None else (mean - 1) / 4,
        "judge_model": model or cfg.model,
        "judge_host": cfg.host,
        "groups": {name: {"n": len(v), "mean": statistics.fmean(v)}
                   for name, v in sorted(by_group.items())},
        "items": [{"id": g.id, "group": g.group, "score": g.score, "reason": g.reason,
                   "answer": g.answer[:2000]} for g in grades],
    }
