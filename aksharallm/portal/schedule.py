"""Scheduled starts and stops: "train 22:00–06:30 every night, and 13:00–17:00 at weekends".

A ~6-day run spent over evenings is a lot of remembering to press things. This turns that
into a handful of rules on disk, fired by a small clock loop.

The design follows the same rule as the rest of the portal: **state is a file, actions go
through the scripts.** A rule is a line in `schedule.json`; firing one calls exactly the
`RunStore.start` / `RunStore.stop` that the buttons call, which run `scripts/phase2.sh` and
`scripts/stop.sh`. So a scheduled start is indistinguishable from one you typed, and
`scripts/schedule.py` and the browser edit the same file.

    {
      "enabled": true,
      "rules": [
        {"id": "…", "run": "small-code", "action": "start", "at": "22:00",
         "days": [0,1,2,3,4], "skip_smoke": true, "stop_after": null},
        {"id": "…", "run": "small-code", "action": "stop",  "at": "06:30",
         "days": [1,2,3,4,5]}
      ]
    }

Times are **local wall-clock**, which is what "start it at ten at night" means. Days are
0=Monday … 6=Sunday, matching `datetime.weekday()`.

Two properties worth stating, because they are what make an unattended schedule safe:

* **Firing is idempotent.** A start when the run is already training is a no-op, as is a
  stop when nothing is running. Overlapping rules cannot compound into two trainers.
* **A missed fire stays missed.** If the machine was asleep at 22:00, the 22:00 start does
  not go off at 07:00 when you open the lid — only within a short grace window. Waking to
  find a run that started nine hours late, mid-workday, is worse than not starting.

Read with: docs/09-running-and-watching.md -- the chapter this implements; it ends with the
order to read these files in. See also docs/07-scaling.md.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timedelta
from pathlib import Path

from .runs import RunError, RunStore, _alive, _cmdline, _read_int

#: `small-code-grpo` -> ("small-code", "grpo"). The `<base>-<stage>` convention is built by
#: `Pipeline.stage_run` and parsed by `baseOf()` in dashboard.js, so reading it here adds no
#: new rule — it just teaches the clock which launcher a run belongs to.
_STAGE_RE = re.compile(r"^(?P<base>.+)-(?P<stage>sft|dpo|grpo)$")


def _stage_of(run: str) -> tuple[str, str] | None:
    """(base, stage) if this run name is a post-training stage, else None."""
    m = _STAGE_RE.match(run or "")
    return (m.group("base"), m.group("stage")) if m else None


TIME_RE = re.compile(r"^([01]\d|2[0-3]):([0-5]\d)$")
DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

#: How late a fire may be and still happen. Long enough to cover a slow tick, a portal
#: restart or a laptop lid, short enough that a rule never goes off at the wrong time of day.
GRACE_SECONDS = 15 * 60

#: How often the clock loop looks. A rule's resolution is a minute; there is no reason to
#: spin faster, and this keeps an idle portal at approximately zero CPU.
TICK_SECONDS = 20


@dataclass
class Rule:
    run: str
    action: str                       # "start" | "stop"
    at: str                           # "HH:MM", local time
    days: list[int] = field(default_factory=lambda: list(range(7)))  # 0=Mon … 6=Sun
    enabled: bool = True
    stop_after: int | None = None     # start only: bound the session it launches
    skip_smoke: bool = True           # start only: honoured only when ckpt_last.pt exists
    note: str | None = None
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    last_fired: str | None = None     # ISO timestamp of the occurrence, not of the attempt
    last_result: str | None = None

    def __post_init__(self):
        if self.action not in ("start", "stop"):
            raise RunError(f"action must be 'start' or 'stop', not {self.action!r}")
        if not TIME_RE.match(self.at or ""):
            raise RunError(f"time must be HH:MM (24-hour), not {self.at!r}")
        try:
            self.days = sorted({int(d) for d in self.days})
        except (TypeError, ValueError):
            raise RunError(f"days must be numbers 0-6 (Mon-Sun), not {self.days!r}")
        if not self.days or any(d < 0 or d > 6 for d in self.days):
            raise RunError("days must be a non-empty subset of 0-6 (Monday to Sunday)")
        if self.stop_after is not None:
            self.stop_after = int(self.stop_after)
            if self.stop_after < 1:
                raise RunError("stop_after must be at least 1 step")

    # ---- timing ------------------------------------------------------------------------
    def _at_on(self, day: datetime) -> datetime:
        hh, mm = (int(p) for p in self.at.split(":"))
        return day.replace(hour=hh, minute=mm, second=0, microsecond=0)

    def next_fire(self, now: datetime | None = None) -> datetime | None:
        """The next moment this rule is due, or None if it is disabled."""
        if not self.enabled:
            return None
        now = now or datetime.now()
        for delta in range(8):
            day = now + timedelta(days=delta)
            if day.weekday() not in self.days:
                continue
            when = self._at_on(day)
            if when > now:
                return when
        return None

    def previous_fire(self, now: datetime | None = None) -> datetime | None:
        """The most recent moment this rule was due (in the past week)."""
        now = now or datetime.now()
        for delta in range(8):
            day = now - timedelta(days=delta)
            if day.weekday() not in self.days:
                continue
            when = self._at_on(day)
            if when <= now:
                return when
        return None

    def due(self, now: datetime | None = None, grace: int = GRACE_SECONDS) -> datetime | None:
        """The occurrence to fire right now, or None.

        An occurrence fires once: `last_fired` records *which* occurrence, not when the
        attempt happened, so a restarted scheduler cannot re-fire one it already handled.
        """
        if not self.enabled:
            return None
        now = now or datetime.now()
        prev = self.previous_fire(now)
        if prev is None or (now - prev).total_seconds() > grace:
            return None
        if self.last_fired and self.last_fired >= prev.isoformat(timespec="seconds"):
            return None
        return prev

    def describe(self) -> str:
        days = ("every day" if len(self.days) == 7 else
                "weekdays" if self.days == [0, 1, 2, 3, 4] else
                "weekends" if self.days == [5, 6] else
                " ".join(DAY_NAMES[d] for d in self.days))
        extra = ""
        if self.action == "start":
            extra = f" ({self.stop_after} steps)" if self.stop_after else ""
        return f"{self.action} {self.run} at {self.at}, {days}{extra}"


class Schedule:
    """The rule file. Reloaded before every read so the CLI and the portal can both edit it."""

    def __init__(self, root: Path | str, path: Path | None = None):
        self.root = Path(root)
        self.path = Path(path) if path else self.root / "schedule.json"
        self.enabled = True
        self.rules: list[Rule] = []
        self._mtime: float | None = None
        self.load()

    # ---- persistence -------------------------------------------------------------------
    def load(self) -> "Schedule":
        try:
            raw = json.loads(self.path.read_text())
            self._mtime = self.path.stat().st_mtime
        except (OSError, ValueError):
            self.enabled, self.rules = True, []
            return self
        self.enabled = bool(raw.get("enabled", True))
        known = {f.name for f in fields(Rule)}
        self.rules = []
        for item in raw.get("rules", []):
            try:
                self.rules.append(Rule(**{k: v for k, v in item.items() if k in known}))
            except (RunError, TypeError):
                continue  # a hand-edited rule that no longer parses must not break the rest
        return self

    def reload_if_changed(self) -> "Schedule":
        try:
            mtime = self.path.stat().st_mtime
        except OSError:
            mtime = None
        if mtime != self._mtime:
            self.load()
        return self

    def save(self) -> None:
        payload = {"enabled": self.enabled,
                   "rules": [asdict(r) for r in self.rules]}
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2) + "\n")
        tmp.replace(self.path)  # atomic: a crash mid-write never truncates the schedule
        self._mtime = self.path.stat().st_mtime

    # ---- editing -----------------------------------------------------------------------
    def add(self, rule: Rule) -> Rule:
        self.rules.append(rule)
        self.save()
        return rule

    def add_window(self, run: str, start_at: str, stop_at: str, days: list[int],
                   stop_after: int | None = None, skip_smoke: bool = True) -> list[Rule]:
        """A training window as the pair of rules it really is.

        The stop's days are shifted when the window crosses midnight: "start 22:00, stop
        06:30, Mon-Fri" means the *stops* land Tue-Sat. Getting this wrong silently leaves
        the GPU running all Saturday, so it is done here rather than left to the user.
        """
        start = Rule(run=run, action="start", at=start_at, days=days,
                     stop_after=stop_after, skip_smoke=skip_smoke,
                     note=f"window {start_at}-{stop_at}")
        stop_days = days if stop_at > start_at else [(d + 1) % 7 for d in days]
        stop = Rule(run=run, action="stop", at=stop_at, days=stop_days,
                    note=f"window {start_at}-{stop_at}")
        self.rules += [start, stop]
        self.save()
        return [start, stop]

    def get(self, rule_id: str) -> Rule:
        for r in self.rules:
            if r.id == rule_id:
                return r
        raise RunError(f"no such schedule rule: {rule_id}")

    def remove(self, rule_id: str) -> Rule:
        rule = self.get(rule_id)
        self.rules.remove(rule)
        self.save()
        return rule

    def set_enabled(self, rule_id: str, enabled: bool) -> Rule:
        rule = self.get(rule_id)
        rule.enabled = bool(enabled)
        self.save()
        return rule

    # ---- view --------------------------------------------------------------------------
    def as_dict(self, now: datetime | None = None) -> dict:
        now = now or datetime.now()
        rules = []
        for r in sorted(self.rules, key=lambda r: (r.run, r.at, r.action)):
            nxt = r.next_fire(now)
            rules.append({**asdict(r),
                          "describe": r.describe(),
                          "next_fire": nxt.isoformat(timespec="minutes") if nxt else None,
                          "next_fire_in_s": (nxt - now).total_seconds() if nxt else None})
        return {"enabled": self.enabled, "rules": rules, "path": str(self.path),
                "now": now.isoformat(timespec="seconds")}


class Scheduler:
    """The clock loop. One per machine — see `lock()`."""

    def __init__(self, store: RunStore, schedule: Schedule | None = None,
                 tick: float = TICK_SECONDS):
        self.store = store
        self.schedule = schedule or Schedule(store.root)
        # Post-training stages are launched through the pipeline, not through LAUNCHERS;
        # see `_start`. Imported here rather than at module scope because pipeline.py
        # imports from runs.py, which this module also imports.
        from .pipeline import Pipeline
        self.pipeline = Pipeline(store.root)
        self.tick = tick
        self.log_path = store.root / "logs" / "scheduler.log"
        self.pid_path = store.root / "logs" / "scheduler.pid"
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # ---- one scheduler at a time -------------------------------------------------------
    def holder(self) -> int | None:
        """The pid of a live scheduler, if one already holds the lock."""
        pid = _read_int(self.pid_path)
        if pid and pid != os.getpid() and _alive(pid) and "aksharallm" in _cmdline(pid):
            return pid
        return None

    def lock(self) -> bool:
        """Claim the machine's one scheduler slot. False if somebody else has it.

        Two schedulers would not corrupt anything — firing is idempotent — but they would
        double every log line and make "did my 22:00 rule work?" unanswerable.
        """
        if self.holder():
            return False
        self.pid_path.parent.mkdir(parents=True, exist_ok=True)
        self.pid_path.write_text(f"{os.getpid()}\n")
        return True

    def release(self) -> None:
        if _read_int(self.pid_path) == os.getpid():
            self.pid_path.unlink(missing_ok=True)

    # ---- running -----------------------------------------------------------------------
    def log(self, message: str) -> None:
        line = f"{datetime.now():%Y-%m-%d %H:%M:%S}  {message}"
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "a") as fh:
            fh.write(line + "\n")

    def startable(self) -> list[str]:
        """Run names the schedule may be pointed at: the launchers, plus the post-training
        stages of the ones that are *language* models.

        Two filters, for two different reasons.

        Only language models get stages: a codec or an audio LM has no SFT, so offering
        `codec-lj-grpo` would be 24 lines of nonsense in a dropdown. That test is structural
        (does the config have a `model:` section) rather than temporal, so it does not
        flicker as checkpoints come and go.

        But stages ARE listed whether or not their prerequisite exists yet. A rule written
        today for a GRPO whose SFT finishes next week is exactly what a schedule is for; the
        gate is enforced at fire time, where `Pipeline.start` refuses with the reason and
        the clock records a skip.
        """
        from .pipeline import STAGES
        from .runs import LAUNCHERS
        names = set(LAUNCHERS)
        for base in LAUNCHERS:
            cfg = self.store.root / "configs" / f"{base}.yaml"
            try:
                if not re.search(r"^model:", cfg.read_text(errors="replace"), re.MULTILINE):
                    continue
            except OSError:
                continue
            names |= {f"{base}-{stage}" for stage in STAGES}
        return sorted(names)

    def _start(self, rule: Rule) -> dict:
        """Start a rule's run, through whichever launcher owns it.

        A post-training stage is launched by `scripts/stage.sh`, not `phase2.sh`, so it is
        not in `LAUNCHERS` and `RunStore.start` refuses it. Rather than teach `LAUNCHERS`
        about a second script — which would also put a second Start button on the dashboard,
        next to the Post-training panel's one, and duplicate the dependency gate — the
        scheduler dispatches on the run's shape and calls the same `Pipeline.start` the
        panel's button calls.

        `<base>-<stage>` is already load-bearing in both directions (`Pipeline.stage_run`
        builds it, `baseOf()` in dashboard.js parses it), so this adds no new convention.
        The gate comes along for free: `Pipeline.start` raises `RunError` naming the missing
        prerequisite, and `fire` records that as a skip without stopping the clock — which
        is the right behaviour for a rule written before its prerequisite exists.

        `stop_after` and `skip_smoke` are phase2.sh's and have no meaning for a stage; the
        window's paired stop rule is what bounds a stage session.
        """
        if (stage := _stage_of(rule.run)) is not None:
            base, name = stage
            return self.pipeline.start(base, name)
        return self.store.start(rule.run, stop_after=rule.stop_after,
                                skip_smoke=rule.skip_smoke)

    def _busy(self, run: str) -> str | None:
        """Another run's trainer holding the card, or None. Only consulted for starts.

        There is one GPU. Per-run idempotency ("already training" is a no-op) does not help
        across runs: a 22:00 GRPO window overlapping a 00:30 base-run window puts two
        trainers on the same 3090, and the second one dies in its first forward pass with
        nobody awake to read the traceback. A human pressing Start may have a reason to
        double up; an unattended clock does not, so this guard lives here rather than in
        `RunStore.start`.
        """
        for other in self.store.runs():
            if other != run and self.store.trainer_pid(other):
                return other
        return None

    def fire(self, rule: Rule, occurrence: datetime) -> str:
        """Run one rule. Never raises: a bad rule must not stop the clock."""
        try:
            if rule.action == "start":
                if (busy := self._busy(rule.run)):
                    # Raised, not returned: RunError below is what records last_fired,
                    # last_result and the log line. An early return here would leave the
                    # rule looking like it never came due.
                    raise RunError(f"'{busy}' is training; one GPU, one trainer")
                res = self._start(rule)
                result = f"started (launch pid {res['pid']})"
            else:
                res = self.store.stop(rule.run, "now")
                result = f"stop requested (pid {res['pid']})"
        except RunError as exc:
            # The common ones are not errors at all: "already training" for a start,
            # "not training" for a stop. Both mean the schedule's intent already holds.
            result = f"skipped — {exc}"
        except Exception as exc:  # noqa: BLE001 - the loop must survive anything
            result = f"failed — {type(exc).__name__}: {exc}"
        rule.last_fired = occurrence.isoformat(timespec="seconds")
        rule.last_result = result
        self.schedule.save()
        self.log(f"[{rule.id}] {rule.describe()} -> {result}")
        return result

    def check(self, now: datetime | None = None) -> list[tuple[Rule, str]]:
        """One pass: fire everything due. Returns what it did, for the tests and the CLI."""
        now = now or datetime.now()
        self.schedule.reload_if_changed()
        if not self.schedule.enabled:
            return []
        done = []
        for rule in list(self.schedule.rules):
            occurrence = rule.due(now)
            if occurrence is not None:
                done.append((rule, self.fire(rule, occurrence)))
        return done

    def run_forever(self) -> None:
        self.log(f"scheduler up (pid {os.getpid()}), {len(self.schedule.rules)} rules")
        try:
            while not self._stop.wait(self.tick):
                try:
                    self.check()
                except Exception as exc:  # noqa: BLE001
                    self.log(f"tick failed — {type(exc).__name__}: {exc}")
        finally:
            self.log("scheduler down")
            self.release()

    def start(self) -> bool:
        """Run the loop on a daemon thread. False if another scheduler already holds it."""
        if not self.lock():
            return False
        self._thread = threading.Thread(target=self.run_forever, name="scheduler",
                                        daemon=True)
        self._thread.start()
        return True

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=self.tick + 2)

    def recent(self, lines: int = 40) -> list[str]:
        try:
            return self.log_path.read_text(errors="replace").splitlines()[-lines:]
        except OSError:
            return []


def parse_days(spec: str | list | None) -> list[int]:
    """`"mon-fri"` / `"sat,sun"` / `"daily"` / `[0,1,2]` -> `[0,1,2,…]`."""
    if spec is None or spec == "":
        return list(range(7))
    if isinstance(spec, (list, tuple)):
        return [int(d) for d in spec]
    text = str(spec).strip().lower()
    if text in ("daily", "every", "everyday", "all"):
        return list(range(7))
    if text in ("weekdays", "week"):
        return [0, 1, 2, 3, 4]
    if text in ("weekends", "weekend"):
        return [5, 6]
    lookup = {n.lower(): i for i, n in enumerate(DAY_NAMES)}
    out: list[int] = []
    for part in re.split(r"[,\s]+", text):
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            if a[:3] not in lookup or b[:3] not in lookup:
                raise RunError(f"don't understand the day range {part!r}")
            i, j = lookup[a[:3]], lookup[b[:3]]
            out += [d % 7 for d in range(i, j + 1 if j >= i else j + 8)]
        elif part[:3] in lookup:
            out.append(lookup[part[:3]])
        elif part.isdigit():
            out.append(int(part))
        else:
            raise RunError(f"don't understand the day {part!r}")
    return sorted(set(out))


def fmt_in(seconds: float | None) -> str:
    """"in 3h20m" / "in 45m" — how the next fire reads in a table."""
    if seconds is None:
        return "—"
    from ..train.runlog import fmt_dur
    return f"in {fmt_dur(seconds)}"
