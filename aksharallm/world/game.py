"""The game the world model learns: push boxes onto goals on an 8x8 grid.

We write the game so its rules are known exactly. A world model is "a model that predicts
what the world does next", and with a game we wrote, every prediction can be checked against
the true next frame, cell by cell, with no judge and no ambiguity.

    # # # # # # # #        #  wall          o  goal
    # . . . . . . #        .  floor         *  box on a goal
    # . @ . $ . . #        @  the agent     +  the agent on a goal
    # . . # . o . #        $  a box
    # . . . . . . #
    # # # # # # # #  (8 rows)

The rules (Sokoban's): an action moves the agent one cell. Into a wall: nothing happens.
Into a box: the box moves one cell further the same way, if that cell is floor or a goal;
otherwise nothing happens. `stay` does nothing. Everything is deterministic, so the next
frame is a FUNCTION of (frame, action), and a perfect world model is possible.

What makes it a fair test rather than a lookup table: the layouts are random (walls, boxes,
goals and the agent placed fresh per episode), so the model has to learn the rule, not the
levels. The three outcomes of a move (`moved`, `pushed`, `blocked`) are reported separately
by the evaluation, because `blocked` is rare and `pushed` is the one with two cells changing.

Read with: docs/25-world-model.md -- the chapter this implements; it ends with the order to
read these files in.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SIZE = 8
WALL, FLOOR, AGENT, BOX, GOAL, BOX_ON_GOAL, AGENT_ON_GOAL = "#", ".", "@", "$", "o", "*", "+"
CELLS = (WALL, FLOOR, AGENT, BOX, GOAL, BOX_ON_GOAL, AGENT_ON_GOAL)
ACTIONS = ("up", "down", "left", "right", "stay")
MOVES = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1), "stay": (0, 0)}


@dataclass
class State:
    walls: np.ndarray        # (SIZE, SIZE) bool
    goals: np.ndarray        # (SIZE, SIZE) bool
    boxes: np.ndarray        # (SIZE, SIZE) bool
    agent: tuple[int, int]

    def copy(self) -> State:
        return State(self.walls.copy(), self.goals.copy(), self.boxes.copy(), self.agent)


def random_level(rng: np.random.Generator, *, n_boxes: int | None = None,
                 n_walls: int | None = None) -> State:
    """A fresh layout: border walls, a few interior walls, 1-3 boxes with as many goals."""
    walls = np.zeros((SIZE, SIZE), bool)
    walls[0], walls[-1], walls[:, 0], walls[:, -1] = True, True, True, True
    inner = [(r, c) for r in range(1, SIZE - 1) for c in range(1, SIZE - 1)]
    order = rng.permutation(len(inner))
    nw = int(rng.integers(0, 6)) if n_walls is None else n_walls
    nb = int(rng.integers(1, 4)) if n_boxes is None else n_boxes
    picks = [inner[i] for i in order]
    for r, c in picks[:nw]:
        walls[r, c] = True
    free = picks[nw:]
    boxes = np.zeros_like(walls)
    goals = np.zeros_like(walls)
    for r, c in free[:nb]:
        boxes[r, c] = True
    for r, c in free[nb:2 * nb]:
        goals[r, c] = True
    return State(walls, goals, boxes, free[2 * nb])


def step(s: State, action: str) -> tuple[State, str]:
    """`(next state, outcome)`, outcome one of `moved` / `pushed` / `blocked` / `stayed`."""
    dr, dc = MOVES[action]
    if (dr, dc) == (0, 0):
        return s.copy(), "stayed"
    r, c = s.agent
    t = (r + dr, c + dc)
    if s.walls[t]:
        return s.copy(), "blocked"
    n = s.copy()
    if s.boxes[t]:
        b = (t[0] + dr, t[1] + dc)
        if s.walls[b] or s.boxes[b]:
            return s.copy(), "blocked"
        n.boxes[t], n.boxes[b] = False, True
        n.agent = t
        return n, "pushed"
    n.agent = t
    return n, "moved"


def render(s: State) -> list[str]:
    """The frame as SIZE*SIZE cell symbols, row by row: what the model reads and writes."""
    out = []
    for r in range(SIZE):
        for c in range(SIZE):
            if s.walls[r, c]:
                out.append(WALL)
            elif (r, c) == s.agent:
                out.append(AGENT_ON_GOAL if s.goals[r, c] else AGENT)
            elif s.boxes[r, c]:
                out.append(BOX_ON_GOAL if s.goals[r, c] else BOX)
            else:
                out.append(GOAL if s.goals[r, c] else FLOOR)
    return out


def parse(cells: list[str]) -> State | None:
    """The inverse of `render`, or None for a frame no game could produce (no agent, two
    agents, a symbol that is not a cell, a broken border). A world model's output is parsed
    to tell "wrong frame" from "impossible frame"."""
    if len(cells) != SIZE * SIZE or any(x not in CELLS for x in cells):
        return None
    g = np.array(cells).reshape(SIZE, SIZE)
    walls = g == WALL
    if not (walls[0].all() and walls[-1].all() and walls[:, 0].all() and walls[:, -1].all()):
        return None
    agents = np.argwhere((g == AGENT) | (g == AGENT_ON_GOAL))
    if len(agents) != 1:
        return None
    boxes = (g == BOX) | (g == BOX_ON_GOAL)
    goals = (g == GOAL) | (g == BOX_ON_GOAL) | (g == AGENT_ON_GOAL)
    return State(walls, goals, boxes, (int(agents[0][0]), int(agents[0][1])))


def policy(rng: np.random.Generator, s: State) -> str:
    """The data's behaviour policy: mostly moves, a little `stay`, and a bias towards a box
    so pushes are not rare. Not trying to solve the level: the model learns the RULES, so
    the data should show every outcome often, not play well."""
    if rng.random() < 0.05:
        return "stay"
    r, c = s.agent
    pushable = [a for a, (dr, dc) in MOVES.items() if (dr, dc) != (0, 0)
                and s.boxes[r + dr, c + dc] and not s.walls[r + 2 * dr, c + 2 * dc]
                and not s.boxes[r + 2 * dr, c + 2 * dc]]
    if pushable and rng.random() < 0.7:
        return pushable[int(rng.integers(len(pushable)))]     # a push that will move the box
    if rng.random() < 0.6:
        boxes = np.argwhere(s.boxes)
        if len(boxes):
            br, bc = boxes[rng.integers(len(boxes))]
            r, c = s.agent
            opts = [a for a, ok in (("down", br > r), ("up", br < r), ("right", bc > c),
                                    ("left", bc < c)) if ok]
            if opts:
                return opts[int(rng.integers(len(opts)))]
    return ACTIONS[int(rng.integers(4))]


def episode(rng: np.random.Generator, n_steps: int) -> tuple[list[list[str]], list[str], list[str]]:
    """`(frames, actions, outcomes)`: n_steps + 1 frames with an action between each pair."""
    s = random_level(rng)
    frames, actions, outcomes = [render(s)], [], []
    for _ in range(n_steps):
        a = policy(rng, s)
        s, why = step(s, a)
        actions.append(a)
        outcomes.append(why)
        frames.append(render(s))
    return frames, actions, outcomes
