"""GRPO with the tool in the loop: sample a turn, run the call, sample the answer, score that.

Tool SFT taught the model to call a tool and to read its output, by imitation. On problem
set v2 that produced a lookup from phrasing to code: 81% on the twenty trained kinds and
0 of 80 on the four held out, every held-out call being a neighbouring kind's code
(docs/06 § Set v2). Imitation cannot do better than the transcripts it copies. This file
lets GRPO reward the RESULT instead: the call is run for real in the middle of the
completion, the tool's output is handed back, the model writes its answer, and the reward
is whether that answer is right.

    prompt  (user turn + "assistant\\n")                          context, never trained
    turn 1  "<tool>\\nprint(...)\\n</tool>" <|im_end|>             SAMPLED -> trainable
    tool    "\\n<|im_start|>tool\\n4371<|im_end|>\\n<|im_start|>assistant\\n"
                                                                  the sandbox's, never trained
    turn 2  "The answer is 4371." <|im_end|>                      SAMPLED -> trainable

The two sampled turns carry the gradient; the tool's output is context, exactly as in SFT,
because a model rewarded for predicting the tool's output would be rewarded for not
needing the tool. That is what `trainable` encodes, token by token, and `grpo.build_batch`
turns it into the loss mask.

A turn with no call is the final answer (one sampled turn). A turn-1 that ran out of tokens
before `<|im_end|>` is not run: there is no finished call to run.

Both turns go through whatever sampler the trainer uses (`sample(prompts_ids, k)` returning
groups of `(full, gen)`), so the batched engine samples all turn-1s in one pass and all
turn-2s in a second; the sandbox runs in between, on a thread pool, because each call is
its own subprocess (~60 ms) and 32 of them in a row would be the slowest part of a step.

Read with: docs/06-posttraining.md -- the chapter this implements; it ends with the order to
read these files in.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from .agent import TOOL, final_number, same_number


def tool_segment(tok, output: str) -> list[int]:
    """The tokens between the end of turn 1 and the start of turn 2: the rest of turn 1's
    footer, the tool turn, and a new assistant header. Built the way `render_chat` builds
    each message, so a rollout's sequence is token for token what SFT trained on."""
    return (tok.encode("\n") + [tok.im_start_id] + tok.encode("tool\n") + tok.encode(output)
            + [tok.im_end_id] + tok.encode("\n") + [tok.im_start_id] + tok.encode("assistant\n"))


def run_call(code: str) -> str:
    from ..infer.sandbox import run_program
    res = run_program(code, timeout_s=5.0)
    return (res.stdout.strip() or res.detail)[:300]


def _strip_end(tok, ids: list[int]) -> list[int]:
    return ids[:-1] if ids and ids[-1] == tok.im_end_id else ids


def rollouts(sample, tok, prompts_ids: list[list[int]], k: int, run=None,
             workers: int = 8) -> list[list[tuple[list[int], list[int], list[int]]]]:
    """`k` tool rollouts per prompt: groups of `(full, final_gen, trainable)`.

    `full` is the whole sequence, `final_gen` the tokens of the turn the answer is read from
    (turn 2 if a call ran, else turn 1), and `trainable[i]` is 1 exactly where `full[i]` was
    sampled by the model. `final_gen` keeps the trainer's `(full, gen)` contract, so the
    reward and the held-out check decode the answer the way they decode any completion."""
    run = run or run_call
    first = sample(prompts_ids, k)
    calls, where = [], []
    for p, grp in enumerate(first):
        for g, (full, gen) in enumerate(grp):
            if not gen or gen[-1] != tok.im_end_id:
                continue
            m = TOOL.search(tok.decode(_strip_end(tok, gen)))
            if m:
                calls.append(m.group(1))
                where.append((p, g))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        outputs = list(ex.map(run, calls))
    ctx2 = []
    for (p, g), out in zip(where, outputs):
        full, _ = first[p][g]
        ctx2.append(list(full) + tool_segment(tok, out))
    second = sample(ctx2, 1) if ctx2 else []
    cont = {pg: grp[0] for pg, grp in zip(where, second)}

    groups = []
    for p, grp in enumerate(first):
        out = []
        for g, (full, gen) in enumerate(grp):
            plen = len(full) - len(gen)
            if (p, g) in cont:
                full2, gen2 = cont[(p, g)]
                seg = len(full2) - len(gen2) - len(full)
                trainable = [0] * plen + [1] * len(gen) + [0] * seg + [1] * len(gen2)
                out.append((list(full2), list(gen2), trainable))
            else:
                out.append((list(full), list(gen), [0] * plen + [1] * len(gen)))
        groups.append(out)
    return groups


class ToolReward:
    """1.0 if the last number of the final answer is the problem's exact answer, else 0.0.

    No partial credit for calling the tool, writing code that runs, or a number of the right
    size: CodeReward's docstring is the record of what a shaping term turns into once the
    group normalisation rescales it."""

    def __init__(self, answer: str):
        self.answer = answer

    def __call__(self, prompt: str, completion: str) -> float:
        return 1.0 if same_number(final_number(completion), self.answer) else 0.0
