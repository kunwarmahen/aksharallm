---
id: tool-use
title: Teaching a model to use a tool, and the tokens it must never be trained to predict
doc: docs/06-posttraining.md
files:
  - aksharallm/tools/rollout.py
  - aksharallm/tools/agent.py
  - aksharallm/train/grpo.py
verify: tests/test_tool_rl.py::test_a_tool_rollout_is_token_for_token_the_conversation_sft_trained_on
prereqs: [sft-mask]
minutes: 30
summary: The model writes a Python call, the sandbox runs it, the output comes back, and the model answers. Only what the model wrote carries loss. Train it on the tool's output too and it learns to guess the output instead of needing the tool.
---

# 27. Tool use, and the loss mask in the middle of a turn

A tool conversation has four turns ([doc 6](docs/06-posttraining.md) § "Tool use"):

```
user        What is 48213 times 9071?
assistant   <tool> print(48213 * 9071) </tool>          written by the model  -> trained
tool        437340123                                   written by the sandbox -> context
assistant   The answer is 437340123.                    written by the model  -> trained
```

[Lesson 8](docs/lessons/08-sft-mask.md) trained only on assistant turns. GRPO with the tool in
the loop needs the same rule *inside one completion*: `tools/rollout.py` samples turn 1, runs
the call, splices in the tool turn, and samples turn 2. It returns a `trainable` flag for
every token, and `grpo.build_batch` turns those flags into the loss mask.

Why not train on the tool's output too? A model rewarded for predicting `437340123` is being
rewarded for **not needing the tool**. Imitation of the result is exactly what tool use is
supposed to replace.

---

## Exercise: train on the tool's answer

1. Run the check. It builds a rollout with a fake sampler and compares it with the same four
   turns rendered by `render_chat`: identical tokens, and `trainable` set exactly on the two
   sampled turns.
2. In `aksharallm/tools/rollout.py`, in `rollouts`, find the line building `trainable` for a
   rollout that ran a call. Change `[0] * seg` to `[1] * seg`.
3. Run the check. **It should fail.** The tool's output (and the turn headers around it) are
   now trained on.
4. Put it back. Green.

> **What you just saw.** Nothing else would have caught this. The loss still falls, the
> reward still rises, and the model slowly learns to write the sandbox's answers itself. On
> problems it cannot do in its head, that is a model learning to make numbers up.

## Run it yourself

```bash
python -m aksharallm.tools eval checkpoints/small-code-py-tools-v2/sft_best.pt --set rl --label setrl
STAGE_RUN=small-code-py-tools-v2-grpo INIT=checkpoints/small-code-py-tools-v2/sft_best.pt TASKS=tools FILTER=8 STEPS=300 scripts/stage.sh grpo small-code-py-tools-v2
```

The Tools tab scores and pairs the reports; the dashboard shows the run's reward and the share
of rollouts that called the tool.
