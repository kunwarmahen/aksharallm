"""The harder world: Sokoban plus keys, doors you cannot see through, and a slippery floor.

`world-sokoban` learned its game perfectly (100% exact frames, docs/25), which made it a
result about the game rather than the model. This variant adds the two things that make a
world hard to predict, each chosen so the right answer is still known exactly:

* **Hidden state.** A key (`k`) disappears when the agent walks onto it, and the frame never
  shows that the agent is carrying it. A door (`D`) opens when the agent walks into it with a
  key, and stays shut otherwise. Levels have 0-2 keys and 1-2 doors, so a frame with no key
  in it does not say whether the agent holds one: the only way to know whether the next move
  opens the door is to have SEEN the pickup, earlier in the context. A model that reads only
  the last frame is guessing.
* **Randomness.** With probability `SLIP` a move goes in one of the other three directions,
  chosen uniformly. The action token is what was INTENDED, so the same (frame, action) can be
  followed by different frames. Exact frames stop being the right target; the right target is
  the distribution, which `transition` returns exactly. The evaluation (`world/calibrate.py`)
  scores the model's probabilities against it.

Everything else is `game.py`'s: an 8x8 grid, a wall border, boxes pushed onto goals. A box
cannot be pushed into a key or a door.

Read with: docs/25-world-model.md -- the chapter this implements; it ends with the order to
read these files in.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .game import AGENT, AGENT_ON_GOAL, BOX, BOX_ON_GOAL, FLOOR, GOAL, MOVES, SIZE, WALL

KEY, DOOR = "k", "D"
CELLS = (WALL, FLOOR, AGENT, BOX, GOAL, BOX_ON_GOAL, AGENT_ON_GOAL, KEY, DOOR)
ACTIONS = ("up", "down", "left", "right", "stay")
DIRECTIONS = ACTIONS[:4]
SLIP = 0.1
OUTCOMES = ("moved", "pushed", "blocked", "stayed", "took_key", "unlocked", "locked")


@dataclass
class State:
    walls: np.ndarray
    goals: np.ndarray
    boxes: np.ndarray
    keys: np.ndarray
    doors: np.ndarray
    agent: tuple[int, int]
    held: int = 0            # keys carried: the hidden part, never rendered

    def copy(self) -> State:
        return State(self.walls.copy(), self.goals.copy(), self.boxes.copy(), self.keys.copy(),
                     self.doors.copy(), self.agent, self.held)


def random_level(rng: np.random.Generator) -> State:
    """Border walls, 0-4 interior walls, 0-2 boxes with as many goals, 0-2 keys (none in a
    fifth of levels, so a missing key never proves one is carried), 1-2 doors."""
    walls = np.zeros((SIZE, SIZE), bool)
    walls[0], walls[-1], walls[:, 0], walls[:, -1] = True, True, True, True
    inner = [(r, c) for r in range(1, SIZE - 1) for c in range(1, SIZE - 1)]
    picks = [inner[i] for i in rng.permutation(len(inner))]
    counts = {"walls": int(rng.integers(0, 5)), "boxes": int(rng.integers(0, 3)),
              "keys": int(rng.choice(3, p=[0.2, 0.4, 0.4])), "doors": int(rng.integers(1, 3))}
    layers = {n: np.zeros_like(walls) for n in ("boxes", "goals", "keys", "doors")}
    at = 0
    for r, c in picks[:counts["walls"]]:
        walls[r, c] = True
    at = counts["walls"]
    for name, n in (("boxes", counts["boxes"]), ("goals", counts["boxes"]),
                    ("keys", counts["keys"]), ("doors", counts["doors"])):
        for r, c in picks[at:at + n]:
            layers[name][r, c] = True
        at += n
    return State(walls, layers["goals"], layers["boxes"], layers["keys"], layers["doors"],
                 picks[at])


def apply(s: State, direction: str) -> tuple[State, str]:
    """The deterministic part: move one cell in `direction` (or stay). `(next, outcome)`."""
    dr, dc = MOVES[direction]
    if (dr, dc) == (0, 0):
        return s.copy(), "stayed"
    r, c = s.agent
    t = (r + dr, c + dc)
    if s.walls[t]:
        return s.copy(), "blocked"
    n = s.copy()
    if s.doors[t]:
        if s.held == 0:
            return n, "locked"
        n.doors[t], n.held, n.agent = False, s.held - 1, t
        return n, "unlocked"
    if s.keys[t]:
        n.keys[t], n.held, n.agent = False, s.held + 1, t
        return n, "took_key"
    if s.boxes[t]:
        b = (t[0] + dr, t[1] + dc)
        if s.walls[b] or s.boxes[b] or s.keys[b] or s.doors[b]:
            return n, "blocked"
        n.boxes[t], n.boxes[b], n.agent = False, True, t
        return n, "pushed"
    n.agent = t
    return n, "moved"


def transition(s: State, action: str, slip: float = SLIP) -> list[tuple[float, State, str, bool]]:
    """The exact distribution over what happens next: `[(p, next, outcome, slipped), ...]`.

    `stay` never slips. A move goes as intended with probability 1 - slip and in each other
    direction with slip / 3. Branches that render to the same frame are NOT merged here; the
    scorer merges by frame, because the model only ever sees frames."""
    if action == "stay" or slip <= 0:
        n, why = apply(s, action)
        return [(1.0, n, why, False)]
    out = [(1.0 - slip, *apply(s, action), False)]
    for d in DIRECTIONS:
        if d != action:
            out.append((slip / 3, *apply(s, d), True))
    return out


def render(s: State) -> list[str]:
    out = []
    for r in range(SIZE):
        for c in range(SIZE):
            if s.walls[r, c]:
                out.append(WALL)
            elif (r, c) == s.agent:
                out.append(AGENT_ON_GOAL if s.goals[r, c] else AGENT)
            elif s.boxes[r, c]:
                out.append(BOX_ON_GOAL if s.goals[r, c] else BOX)
            elif s.keys[r, c]:
                out.append(KEY)
            elif s.doors[r, c]:
                out.append(DOOR)
            else:
                out.append(GOAL if s.goals[r, c] else FLOOR)
    return out


def parse(cells: list[str]) -> State | None:
    """A frame back to a state, or None if no game could produce it. `held` is unknowable
    from a frame and comes back 0; this is for telling a possible frame from an impossible
    one, not for recovering the hidden state."""
    if len(cells) != SIZE * SIZE or any(x not in CELLS for x in cells):
        return None
    g = np.array(cells).reshape(SIZE, SIZE)
    walls = g == WALL
    if not (walls[0].all() and walls[-1].all() and walls[:, 0].all() and walls[:, -1].all()):
        return None
    agents = np.argwhere((g == AGENT) | (g == AGENT_ON_GOAL))
    if len(agents) != 1:
        return None
    return State(walls, (g == GOAL) | (g == BOX_ON_GOAL) | (g == AGENT_ON_GOAL),
                 (g == BOX) | (g == BOX_ON_GOAL), g == KEY, g == DOOR,
                 (int(agents[0][0]), int(agents[0][1])))


def _towards(rng, s: State, targets: np.ndarray) -> str | None:
    cells = np.argwhere(targets)
    if not len(cells):
        return None
    r, c = s.agent
    tr, tc = cells[int(np.abs(cells - [r, c]).sum(1).argmin())]
    opts = [a for a, ok in (("down", tr > r), ("up", tr < r), ("right", tc > c),
                            ("left", tc < c)) if ok]
    return opts[int(rng.integers(len(opts)))] if opts else None


POLICIES = ("aware", "blind")


def policy(rng: np.random.Generator, s: State, kind: str = "aware") -> str:
    """Coverage again, not winning. The agent knows what it carries (the model does not):
    holding a key it heads for a door, so `unlocked` is common; with none it heads for a key
    while one is left; and now and then it tries a door empty-handed, so `locked` happens
    too, in levels that look the same. Pushes where a box can move; random moves otherwise."""
    if kind == "blind":
        return policy_blind(rng, s)
    if rng.random() < 0.05:
        return "stay"
    u = rng.random()
    if s.held > 0 and u < 0.6:
        target = s.doors
    elif s.held == 0 and s.keys.any() and u < 0.5:
        target = s.keys
    elif u < 0.2:
        target = s.doors
    elif u < 0.35:
        target = s.boxes
    else:
        target = None
    a = _towards(rng, s, target) if target is not None else None
    return a or DIRECTIONS[int(rng.integers(4))]


def policy_blind(rng: np.random.Generator, s: State) -> str:
    """The same coverage, choosing only from what the FRAME shows: never `s.held`.

    The aware policy above heads for a door 60% of the time when it carries a key, so an
    agent walking to a door is itself evidence of a key, and `world-keys` used it: its belief
    at a door whose key was picked up before the context was 0.59, well above what the window
    alone allows. Here the agent heads for a door, a visible key or a box with fixed odds
    whatever it carries, so at a door the only evidence of a key is a pickup in the context.
    A test holds this to `policy(s) == policy(s with held changed)` for every state."""
    if rng.random() < 0.05:
        return "stay"
    u = rng.random()
    if u < 0.35:
        target = s.doors
    elif u < 0.6 and s.keys.any():
        target = s.keys
    elif u < 0.72:
        target = s.boxes
    else:
        target = None
    a = _towards(rng, s, target) if target is not None else None
    return a or DIRECTIONS[int(rng.integers(4))]


def sample(rng: np.random.Generator, s: State, action: str,
           slip: float = SLIP) -> tuple[State, str, bool]:
    branches = transition(s, action, slip)
    i = int(rng.choice(len(branches), p=[b[0] for b in branches]))
    return branches[i][1], branches[i][2], branches[i][3]


def episode_full(rng: np.random.Generator, n_steps: int, slip: float = SLIP,
                 policy_kind: str = "aware") -> dict:
    """Everything the scorer needs: the states (with the hidden `held`), frames, actions, what
    actually happened, whether it slipped, and the step at which each held key was taken."""
    s = random_level(rng)
    states, actions, outcomes, slipped = [s], [], [], []
    for _ in range(n_steps):
        a = policy(rng, s, policy_kind)
        s, why, sl = sample(rng, s, a, slip)
        states.append(s)
        actions.append(a)
        outcomes.append(why)
        slipped.append(sl)
    return {"states": states, "frames": [render(x) for x in states], "actions": actions,
            "outcomes": outcomes, "slipped": slipped}


def episode(rng: np.random.Generator, n_steps: int, slip: float = SLIP,
            policy_kind: str = "aware"):
    """`(frames, actions, outcomes)`, the shape `game.episode` has, so `data.prepare` writes
    either game."""
    e = episode_full(rng, n_steps, slip, policy_kind)
    return e["frames"], e["actions"], e["outcomes"]
