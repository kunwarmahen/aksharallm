---
id: world-model
title: A model that predicts the next frame of a game, and a test that can be passed by watching the player
doc: docs/25-world-model.md
files:
  - aksharallm/world/keys.py
  - aksharallm/world/calibrate.py
  - configs/world-keys-blind.yaml
verify: tests/test_world.py::test_the_blind_agent_never_looks_at_what_it_carries
prereqs: [training-loop]
minutes: 30
summary: In the keys game a door opens only if the agent carries a key the picture never shows. A model can learn to remember the pickup, or it can learn that agents with keys walk to doors. Only one of those is memory.
---

# 26. A world model, and a leak in the test

The world model of [doc 25](docs/25-world-model.md) is the same transformer, trained on
games instead of text: each frame is 64 cell tokens, then an action. In the **keys** game a
key vanishes when picked up and is never drawn again, so whether the next door opens depends
on something no frame shows.

`world-keys` learned it: at doors whose pickup it had seen, it believed 0.95 that they would
open. But at doors whose pickup happened **before** what it could read, it believed 0.59,
where the window alone allows about 0.24.

```
the agent holds a key ──► the policy walks it to a door 60% of the time
                                   │
          the model sees an agent heading for a door ──► "it probably has a key"
```

That is real inference, from behaviour. It is not memory, and it was a leak in the test.

## The fix: an agent that cannot give itself away

`keys.policy_blind` chooses its moves from the frame alone: a door 35% of the time, a visible
key 25%, a box 12%, whatever it carries. Then the only evidence of a key is a pickup in view.
The scorer prints the **base rate** beside the door table, so a model that beats it on the
middle row is visibly using something other than its window.

---

## Exercise: let the agent peek

1. Run the check. It draws 400 random states, gives the blind policy the same random numbers
   with and without a key in hand, and requires the same move every time. It also requires
   the aware policy to differ, or the test would prove nothing.
2. In `aksharallm/world/keys.py`, in `policy_blind`, change the first condition from
   `if u < 0.35:` to `if u < (0.6 if s.held else 0.35):`.
3. Run the check. **It should fail.** The agent now heads for doors more often when it holds
   a key, which is exactly the leak.
4. Put it back. Green.

> **What you just saw.** The game, the frames and the model are unchanged. Only the agent
> that generated the data changed, and that was enough to let a model look like it
> remembers. A hidden variable can leak through anything that depends on it.

## Run it yourself

```bash
python -m aksharallm.world prepare --game keys --policy blind
scripts/experiment.sh world-keys-blind
python -m aksharallm.world eval checkpoints/world-keys-blind/ckpt_best.pt --episodes 1000
```

Or use the World tab: Make the data with "keys, blind agent", press Start on
`world-keys-blind`, then Evaluate.
