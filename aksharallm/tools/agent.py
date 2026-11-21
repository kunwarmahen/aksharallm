"""The agent loop: generate, run any tool call in the sandbox, hand back the output, repeat.

    render the chat so far, ask for the assistant's turn
      -> it contains <tool>...</tool>?  run the code (infer/sandbox.py, the same isolated
         subprocess GRPO's reward uses), append the call and a `tool` turn with its stdout,
         and ask again (at most `max_calls` times)
      -> otherwise it is the final answer

`score` compares the last number in the final answer with the problem's exact answer, and
runs the same problems DIRECTLY too (no tool turn executed: a call the model writes is left
unanswered and its answer is whatever it says after), so the report is paired: the same
model, the same questions, with and without the tool.

Read with: docs/06-posttraining.md -- the chapter this implements; it ends with the order to
read these files in.
"""

from __future__ import annotations

import re
from math import comb

import torch

from ..infer.generate import generate
from ..infer.sandbox import run_program
from .problems import TOOL_CLOSE, TOOL_OPEN, Problem

NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
TOOL = re.compile(re.escape(TOOL_OPEN) + r"\s*(.*?)\s*" + re.escape(TOOL_CLOSE), re.S)


def final_number(text: str) -> str | None:
    """The last number in a reply, commas removed: "The answer is 437,340,123." -> 437340123."""
    found = NUMBER.findall(text)
    return found[-1].replace(",", "") if found else None


def same_number(a: str | None, b: str) -> bool:
    if a is None:
        return False
    try:
        return abs(float(a) - float(b)) < 1e-6
    except ValueError:
        return False


def _turn(model, tok, messages, device, max_new=160) -> str:
    ids, _ = tok.render_chat(messages, add_generation_prompt=True)
    out = generate(model, ids, max_new_tokens=max_new, temperature=0.0, device=device,
                   eos_id=tok.im_end_id)
    reply = out[len(ids):]
    if reply and reply[-1] == tok.im_end_id:
        reply = reply[:-1]
    return tok.decode(reply)


@torch.no_grad()
def solve(model, tok, question: str, *, use_tool: bool = True, max_calls: int = 3,
          device: str = "cpu") -> dict:
    """One problem, greedily. Returns the transcript, the final text and how many calls ran."""
    messages = [{"role": "user", "content": question}]
    calls = 0
    while True:
        reply = _turn(model, tok, messages, device)
        m = TOOL.search(reply)
        if m and use_tool and calls < max_calls:
            res = run_program(m.group(1), timeout_s=5.0)
            out = (res.stdout.strip() or res.detail)[:300]
            messages += [{"role": "assistant", "content": f"{TOOL_OPEN}\n{m.group(1)}\n{TOOL_CLOSE}"},
                         {"role": "tool", "content": out}]
            calls += 1
            continue
        if m and not use_tool:
            # Direct mode: the call is not run. Ask once more with the call left unanswered,
            # so a model that only knows how to call still gets to state a number.
            messages += [{"role": "assistant", "content": reply},
                         {"role": "tool", "content": "(no tool available: answer directly)"}]
            reply = _turn(model, tok, messages, device)
        messages.append({"role": "assistant", "content": reply})
        return {"final": reply, "calls": calls, "messages": messages}


def score(model, tok, probs: list[Problem], *, device: str = "cpu") -> dict:
    """Paired: each problem with the tool and without. By kind, plus a sign test."""
    rows = []
    for p in probs:
        a = solve(model, tok, p.question, use_tool=True, device=device)
        b = solve(model, tok, p.question, use_tool=False, device=device)
        rows.append({"kind": p.kind, "question": p.question, "answer": p.answer,
                     "tool": same_number(final_number(a["final"]), p.answer),
                     "direct": same_number(final_number(b["final"]), p.answer),
                     "calls": a["calls"], "tool_final": a["final"][:200],
                     "direct_final": b["final"][:200]})
    by = {}
    for k in sorted({r["kind"] for r in rows}):
        rs = [r for r in rows if r["kind"] == k]
        by[k] = {"n": len(rs), "tool": _mean([r["tool"] for r in rs]),
                 "direct": _mean([r["direct"] for r in rs]),
                 "called": _mean([r["calls"] > 0 for r in rs])}
    wins = sum(r["tool"] and not r["direct"] for r in rows)
    losses = sum(r["direct"] and not r["tool"] for r in rows)
    return {"n": len(rows), "tool": _mean([r["tool"] for r in rows]),
            "direct": _mean([r["direct"] for r in rows]),
            "called": _mean([r["calls"] > 0 for r in rows]),
            "paired": {"tool_only": wins, "direct_only": losses, "p": _sign(wins, losses)},
            "by_kind": by, "rows": rows}


def compare(a: dict, b: dict) -> dict:
    """Two saved reports on the SAME problems, paired per problem (e.g. before and after the
    tool SFT). Refuses two reports on different problems: a comparison of totals over
    different questions is the mistake pairing exists to avoid."""
    ra, rb = a["rows"], b["rows"]
    if [r["question"] for r in ra] != [r["question"] for r in rb]:
        raise ValueError("the two reports are not on the same problems (different seed or "
                         "--per-kind), so they cannot be paired")
    out = {}
    for mode_a, mode_b, name in (("tool", "tool", "with the tool"),
                                 ("direct", "direct", "without the tool"),
                                 ("direct", "tool", "b with the tool vs a without")):
        w = sum(y[mode_b] and not x[mode_a] for x, y in zip(ra, rb))
        l = sum(x[mode_a] and not y[mode_b] for x, y in zip(ra, rb))
        out[name] = {"b_only": w, "a_only": l, "p": _sign(w, l)}
    kinds = sorted({r["kind"] for r in ra})
    by = {k: {"n": sum(r["kind"] == k for r in ra),
              "a_tool": sum(r["tool"] for r in ra if r["kind"] == k),
              "b_tool": sum(r["tool"] for r in rb if r["kind"] == k),
              "a_direct": sum(r["direct"] for r in ra if r["kind"] == k),
              "b_direct": sum(r["direct"] for r in rb if r["kind"] == k)} for k in kinds}
    return {"n": len(ra), "paired": out, "by_kind": by}


def _sign(a: int, b: int) -> float:
    n, k = a + b, min(a, b)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0


def _mean(xs) -> float:
    return float(sum(xs) / len(xs)) if xs else float("nan")
