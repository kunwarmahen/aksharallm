#!/usr/bin/env python3
"""Schedule starts and stops: "train 22:00-06:30 on weeknights", several times a day.

The same rules the portal's Schedule panel edits — one `schedule.json` in the repo root,
read and written by both, so you can add a window in the browser and pause it from a
terminal. Firing a rule calls the same code the buttons call, which runs `scripts/phase2.sh`
and `scripts/stop.sh`; a scheduled start is indistinguishable from one you typed.

    scripts/schedule.sh                                  # what's scheduled, and what's next
    scripts/schedule.sh window small-code 22:00 06:30 --days mon-fri
    scripts/schedule.sh window small-code 13:00 17:30 --days sat,sun --steps 2000
    scripts/schedule.sh add start small-code 09:00 --days daily
    scripts/schedule.sh add stop  small-code 12:00
    scripts/schedule.sh pause 3f9a2b1c        # keep the rule, don't fire it
    scripts/schedule.sh resume 3f9a2b1c
    scripts/schedule.sh rm 3f9a2b1c
    scripts/schedule.sh off                   # master switch: nothing fires at all
    scripts/schedule.sh log                   # what the scheduler actually did

Something has to be watching the clock. Either of these does it:

    scripts/portal.sh                         # the portal runs the scheduler by default
    scripts/schedule.sh daemon                # a foreground clock loop, no web server
    * * * * * cd /path/to/repo && scripts/schedule.sh check    # or let cron do the ticking

All three take the same one-per-machine lock (`logs/scheduler.pid`), so running the portal
and the daemon together does not double-fire anything.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

try:
    from aksharallm.portal.runs import LAUNCHERS, RunError, RunStore
    from aksharallm.portal.schedule import DAY_NAMES, Rule, Schedule, Scheduler, parse_days
except ImportError:  # not the venv's python — re-exec with it rather than failing
    venv = ROOT / ".venv" / "bin" / "python"
    if venv.exists() and Path(sys.executable).resolve() != venv.resolve():
        os.execv(str(venv), [str(venv), str(Path(__file__).resolve()), *sys.argv[1:]])
    raise


def fmt_next(rule: Rule, now: datetime) -> str:
    from aksharallm.train.runlog import fmt_dur
    nxt = rule.next_fire(now)
    if nxt is None:
        return "paused"
    return f"{nxt:%a %H:%M} (in {fmt_dur((nxt - now).total_seconds())})"


def cmd_list(sched: Schedule, scheduler: Scheduler, args) -> int:
    now = datetime.now()
    holder = scheduler.holder()
    print(f"{sched.path}")
    print(f"schedule is {'ARMED' if sched.enabled else 'PAUSED (nothing will fire)'}; "
          f"clock {'running as pid ' + str(holder) if holder else 'NOT running'}"
          f"{'' if holder else ' — start scripts/portal.sh or scripts/schedule.sh daemon'}")
    if not sched.rules:
        print("\nno rules yet. try:  scripts/schedule.sh window small-code 22:00 06:30 --days mon-fri")
        return 0

    hdr = ("id", "run", "action", "at", "days", "next", "last result")
    rows = []
    for r in sorted(sched.rules, key=lambda r: (r.at, r.run)):
        days = ("daily" if len(r.days) == 7 else
                "mon-fri" if r.days == [0, 1, 2, 3, 4] else
                "sat,sun" if r.days == [5, 6] else
                ",".join(DAY_NAMES[d].lower() for d in r.days))
        extra = f" +{r.stop_after} steps" if r.stop_after else ""
        rows.append((r.id, r.run, r.action + extra, r.at, days,
                     fmt_next(r, now) if r.enabled else "paused",
                     (r.last_result or "—")[:44]))
    widths = [max(len(str(row[c])) for row in [hdr, *rows]) for c in range(len(hdr))]
    line = "  ".join(str(h).ljust(w) for h, w in zip(hdr, widths))
    print()
    print(line)
    print("-" * len(line))
    for row in rows:
        print("  ".join(str(c).ljust(w) for c, w in zip(row, widths)))
    return 0


def cmd_window(sched: Schedule, scheduler: Scheduler, args) -> int:
    rules = sched.add_window(args.run, args.start_at, args.stop_at, parse_days(args.days),
                             stop_after=args.steps, skip_smoke=not args.smoke)
    print(f"added: {rules[0].describe()}")
    print(f"       {rules[1].describe()}")
    if rules[1].days != rules[0].days:
        print("       (the window crosses midnight, so the stops land the following day)")
    return cmd_list(sched, scheduler, args)


def cmd_add(sched: Schedule, scheduler: Scheduler, args) -> int:
    rule = Rule(run=args.run, action=args.action, at=args.at, days=parse_days(args.days),
                stop_after=args.steps, skip_smoke=not args.smoke)
    sched.add(rule)
    print(f"added [{rule.id}]: {rule.describe()}")
    return cmd_list(sched, scheduler, args)


def cmd_rm(sched: Schedule, scheduler: Scheduler, args) -> int:
    rule = sched.remove(args.id)
    print(f"removed: {rule.describe()}")
    return cmd_list(sched, scheduler, args)


def cmd_pause(sched: Schedule, scheduler: Scheduler, args) -> int:
    rule = sched.set_enabled(args.id, args.command == "resume")
    print(f"{'resumed' if rule.enabled else 'paused'}: {rule.describe()}")
    return cmd_list(sched, scheduler, args)


def cmd_switch(sched: Schedule, scheduler: Scheduler, args) -> int:
    sched.enabled = args.command == "on"
    sched.save()
    print(f"schedule {'ARMED' if sched.enabled else 'PAUSED — rules kept, nothing fires'}")
    return cmd_list(sched, scheduler, args)


def cmd_check(sched: Schedule, scheduler: Scheduler, args) -> int:
    """One pass, then exit — the cron-friendly form."""
    fired = scheduler.check()
    if not fired:
        print("nothing due.")
    for rule, result in fired:
        print(f"{rule.describe()} -> {result}")
    return 0


def cmd_daemon(sched: Schedule, scheduler: Scheduler, args) -> int:
    if not scheduler.lock():
        print(f"a scheduler is already running as pid {scheduler.holder()} "
              "(the portal runs one by default). Nothing to do.", file=sys.stderr)
        return 1
    print(f"scheduler running (pid {os.getpid()}), {len(sched.rules)} rules, "
          f"{'armed' if sched.enabled else 'PAUSED'}. Ctrl-C to stop.")
    print(f"log: {scheduler.log_path}")
    try:
        scheduler.run_forever()
    except KeyboardInterrupt:
        print("\nscheduler stopped (training runs are unaffected).")
    finally:
        scheduler.release()
    return 0


def cmd_log(sched: Schedule, scheduler: Scheduler, args) -> int:
    lines = scheduler.recent(args.lines)
    print("\n".join(lines) if lines else f"nothing logged yet ({scheduler.log_path})")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="scripts/schedule.sh", description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join(__doc__.splitlines()[2:]))
    ap.add_argument("--root", default=str(ROOT), help=argparse.SUPPRESS)
    sub = ap.add_subparsers(dest="command")

    def with_days(parser, steps=False):
        parser.add_argument("--days", default="daily",
                            help="daily | mon-fri | sat,sun | mon,wed,fri  (default: daily)")
        if steps:
            parser.add_argument("--steps", type=int, default=None, metavar="N",
                                help="bound the session this start launches to N steps")
            parser.add_argument("--smoke", action="store_true",
                                help="run the 50-step smoke test on each scheduled start "
                                     "(default: skip it, since a scheduled start is a resume)")
        return parser

    sub.add_parser("list", help="show the rules and when they next fire")

    w = with_days(sub.add_parser("window", help="a start and its matching stop, in one go"),
                  steps=True)
    w.add_argument("run")
    w.add_argument("start_at", metavar="START", help="HH:MM")
    w.add_argument("stop_at", metavar="STOP", help="HH:MM")

    a = with_days(sub.add_parser("add", help="a single start or stop"), steps=True)
    a.add_argument("action", choices=["start", "stop"])
    a.add_argument("run")
    a.add_argument("at", metavar="HH:MM")

    for name, help_text in [("rm", "delete a rule"), ("pause", "keep a rule but don't fire it"),
                            ("resume", "un-pause a rule")]:
        p = sub.add_parser(name, help=help_text)
        p.add_argument("id")

    sub.add_parser("on", help="arm the schedule")
    sub.add_parser("off", help="pause everything (rules are kept)")
    sub.add_parser("check", help="fire anything due right now, then exit (for cron)")
    sub.add_parser("daemon", help="watch the clock in the foreground")
    lg = sub.add_parser("log", help="what the scheduler has done")
    lg.add_argument("-n", "--lines", type=int, default=40)

    args = ap.parse_args(argv)
    args.command = args.command or "list"
    if not hasattr(args, "days"):
        args.days = None
    if not hasattr(args, "steps"):
        args.steps = None

    root = Path(args.root)
    store = RunStore(root)
    sched = Schedule(root)
    scheduler = Scheduler(store, sched)

    handlers = {"list": cmd_list, "window": cmd_window, "add": cmd_add, "rm": cmd_rm,
                "pause": cmd_pause, "resume": cmd_pause, "on": cmd_switch, "off": cmd_switch,
                "check": cmd_check, "daemon": cmd_daemon, "log": cmd_log}
    try:
        if args.command in ("window", "add"):
            # Fail here rather than at 22:00: a start rule for a run phase2.sh can't build
            # would sit in the table looking correct and never work.
            store.check(args.run)
            wants_start = args.command == "window" or args.action == "start"
            if wants_start and args.run not in LAUNCHERS:
                raise RunError(f"'{args.run}' has no launcher, so a scheduled start could "
                               f"never work. Startable: {', '.join(sorted(LAUNCHERS))}.")
        return handlers[args.command](sched, scheduler, args)
    except RunError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
