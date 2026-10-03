"""`python -m aksharallm.dictate` — dictation on the desktop, and everything it learns.

    python -m aksharallm.dictate status                 # what is installed, running, learned
    python -m aksharallm.dictate daemon --bg            # load the models once, wait for the key
    python -m aksharallm.dictate install-shortcut       # GNOME: <Super><Alt>d runs `toggle`
    python -m aksharallm.dictate toggle                 # what the shortcut runs
    python -m aksharallm.dictate file me.wav            # the same pipeline on a file
    python -m aksharallm.dictate clean "um so i met shaun new line scratch that"
    python -m aksharallm.dictate correct --shown "I met Sean." --corrected "I met Shaun."
    python -m aksharallm.dictate dictionary add GitHub
    python -m aksharallm.dictate punct-eval checkpoints/punct/ckpt_best.pt
    python -m aksharallm.dictate stream-eval asr-libri100   # the live preview, measured

The punctuation tagger trains like any other run: `scripts/experiment.sh punct`.

Read with: docs/23-speech-recognition.md -- the chapter this implements; it ends with the
order to read these files in.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from . import daemon as dm
from .pipeline import Dictator, Settings

ROOT = Path.cwd()


def _settings(args) -> Settings:
    return Settings.load(ROOT, device=getattr(args, "device", None))


def cmd_status(args) -> int:
    s = _settings(args)
    pid = dm.running(ROOT, s)
    print(f"daemon       {'running, pid ' + str(pid) if pid else 'not running'}")
    if pid:
        st = dm.send(ROOT, s, "status", timeout=5)
        print(f"             state {st.get('state')}")
    sc = None
    try:
        sc = dm.shortcut()
    except Exception:   # noqa: BLE001 -- no gsettings: not GNOME
        pass
    print(f"shortcut     {sc['binding'] + ' -> ' + sc['command'] if sc else 'not installed (install-shortcut)'}")
    print(f"session      {os.environ.get('XDG_SESSION_TYPE', '?')}"
          + ("" if os.environ.get("XDG_SESSION_TYPE", "x11") == "x11" else
             "  <- typing into other windows needs X11"))
    for name, t in dm.tools().items():
        print(f"  {name:12s} {'ok' if t['found'] else 'MISSING':8s} {t['for']}"
              + ("" if t["found"] else f"   ({t['install']})"))
    d = Dictator(s, ROOT)
    print(f"recogniser   {s.recognizer}")
    p = ROOT / s.punctuator
    print(f"punctuation  {'tagger ' + s.punctuator if p.is_file() else 'rules only — train it: scripts/experiment.sh punct'}")
    print(f"word LM      {s.lm if (ROOT / s.lm).is_file() else 'none (greedy)'}")
    print(f"learned      {json.dumps(d.personal.summary())}")
    return 0


def cmd_daemon(args) -> int:
    s = _settings(args)
    if args.stop:
        pid = dm.running(ROOT, s)
        if not pid:
            print("no dictation daemon is running")
            return 1
        os.kill(pid, 15)
        print(f"stopped pid {pid}")
        return 0
    if args.bg:
        if pid := dm.running(ROOT, s):
            print(f"already running, pid {pid}")
            return 0
        log = dm.state_dir(ROOT, s) / "daemon.log"
        with open(log, "ab") as fh:
            p = subprocess.Popen([sys.executable, "-u", "-m", "aksharallm.dictate", "daemon"]
                                 + (["--device", args.device] if args.device else []),
                                 cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT,
                                 stdin=subprocess.DEVNULL, start_new_session=True)
        for _ in range(100):
            time.sleep(0.2)
            if p.poll() is not None:
                print(f"the daemon exited at once (code {p.returncode}); see {log}")
                return 1
            if dm.running(ROOT, s):
                break
        print(f"daemon started, pid {p.pid}; it loads the models in the background (log: {log})")
        return 0
    dm.Daemon(ROOT, s).serve()
    return 0


def _client(cmd: str):
    def run(args) -> int:
        r = dm.send(ROOT, _settings(args), cmd)
        if not r.get("ok"):
            print(r.get("error") or f"nothing to {cmd} (state: {r.get('state')})")
            if "no dictation daemon" in (r.get("error") or ""):
                dm.notify("Dictation is not running", "python -m aksharallm.dictate daemon --bg")
            return 1
        if "text" in r:
            print(r["text"] or f"(nothing written: {r.get('note') or 'no words heard'})")
        else:
            print(f"state: {r.get('state')}")
        return 0
    return run


def cmd_install_shortcut(args) -> int:
    command = f"{sys.executable} -m aksharallm.dictate --root {ROOT} toggle"
    try:
        dm.install_shortcut(command, args.binding)
    except (RuntimeError, OSError, subprocess.CalledProcessError) as e:
        print(f"not installed: {e}")
        return 1
    print(f"installed    {args.binding} -> {command}   (read back from GNOME: ok)")
    print("             press it once to start listening, again to type the text. The daemon must be")
    print("             running: python -m aksharallm.dictate daemon --bg")
    print("             see it in Settings > Keyboard > View and Customise Shortcuts > Custom Shortcuts")
    return 0


def cmd_uninstall_shortcut(args) -> int:
    print("removed" if dm.uninstall_shortcut() else "it was not installed")
    return 0


def cmd_file(args) -> int:
    from ..audio.io import load_audio
    d = Dictator(_settings(args), ROOT)
    for f in args.files:
        clip = load_audio(f, sample_rate=16_000)
        r = d.dictate(clip.samples, source="file")
        print(f"{f}:")
        print(f"  heard  {r.get('heard') or '(nothing)'}")
        print(f"  wrote  {r.get('text') or r.get('note', '')}")
        for st in r.get("steps", []):
            print(f"  step   {st}")
        print(f"  {r['seconds']} s of audio in {r['ms']:.0f} ms ({r['realtime']}x real time)")
    return 0


def cmd_clean(args) -> int:
    d = Dictator(_settings(args), ROOT)
    r = d.clean_text(" ".join(args.words))
    print(r["text"])
    for st in r["steps"]:
        print(f"  step   {st}", file=sys.stderr)
    print(f"  ({r['punctuator']})", file=sys.stderr)
    return 0


def cmd_correct(args) -> int:
    d = Dictator(_settings(args), ROOT)
    print(json.dumps(d.correct(args.shown, args.corrected), indent=1))
    return 0


def cmd_dictionary(args) -> int:
    d = Dictator(_settings(args), ROOT)
    p = d.personal
    if args.action == "add":
        for w in args.words:
            print(f"added {p.add_word(w)}")
        p.save()
    elif args.action == "remove":
        for w in args.words:
            print(("removed " if p.remove(w) else "not there: ") + w)
        p.save()
    else:
        for w, v in sorted(p.words.items()):
            print(f"  {v.get('spelling') or w:20s} {v.get('source', ''):10s} x{v.get('count', 0)}")
        for k, v in sorted(p.replacements.items()):
            state = "active" if v["count"] >= 2 else f"seen {v['count']}x, needs 2"
            print(f"  {k!r} -> {v['to']!r}  ({state})")
    return 0


def cmd_history(args) -> int:
    for r in reversed(Dictator(_settings(args), ROOT).history(args.limit)):
        print(f"{r['time']}  {r['seconds']:5.1f}s  {r.get('text') or '(' + r.get('note', 'nothing') + ')'}")
    return 0


def cmd_punct_eval(args) -> int:
    from .tagger import Punctuator, evaluate_tagger, heldout_passages
    p = Punctuator.load(args.checkpoint, args.device or "cpu")
    passages = heldout_passages(p.tok, args.val_bin, args.passages)
    r = evaluate_tagger(p, passages)
    print(f"tagger       {args.checkpoint} (step {p.step}) on {r['passages']} held-out passages, "
          f"{r['tagger']['words']:,} words")
    print(f"{'':12s} {'tagger P/R/F1':>24s}   {'rules only F1':>14s}")
    for k in ("comma", "period", "question", "capital"):
        t, b = r["tagger"][k], r["rules_only"][k]
        print(f"  {k:10s} {t['precision']:6.1%} {t['recall']:6.1%} {t['f1']:6.1%}   {b['f1']:12.1%}"
              f"   ({t['support']} in the text)")
    out = Path("logs/asr") / f"punct-{Path(args.checkpoint).parent.name}-step{p.step}.json"
    if not args.no_write:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"kind": "punct_eval", "checkpoint": args.checkpoint,
                                   "step": p.step, "val_bin": args.val_bin,
                                   "time": time.strftime("%Y-%m-%d %H:%M:%S"), **r}, indent=1))
        print(f"written      {out}")
    return 0


def cmd_stream_eval(args) -> int:
    """How the live preview behaves: commit latency, revisions, tick cost (dictate/stream.py)."""
    import torch
    from ..asr.__main__ import resolve
    from ..asr.data import Utterances
    from ..asr.train import feasibility, load_recognizer
    from .stream import evaluate
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    path = resolve(args.checkpoint)
    model, blob = load_recognizer(path, device)
    corpus = Utterances(args.corpus, max_seconds=40.0, limit=args.limit,
                        feasible=feasibility(model.cfg, model.cfg.sample_rate))
    t0 = time.time()
    r = evaluate(model, device, corpus, args.tick)
    L = r["latency_s"]
    print(f"checkpoint   {path} (step {blob.get('step')}) on {r['utts']} utterances of {args.corpus}, "
          f"a tick every {args.tick:g} s, {device}")
    print(f"latency      a word is committed {L['median']:.2f} s after it is spoken (median; "
          f"mean {L['mean']:.2f}, p90 {L['p90']:.2f}) — {r['words_committed_live']:,} of "
          f"{r['words_final']:,} words committed while still talking")
    print(f"revisions    {r['revised']} committed words differed from the final reading "
          f"({r['revision_rate']:.2%})")
    print(f"cost         {r['tick_ms']['mean']:.0f} ms per tick (p90 {r['tick_ms']['p90']:.0f})")
    print(f"took         {time.time() - t0:.0f}s")
    if not args.no_write:
        out = Path("logs/asr") / f"stream-{path.parent.name}-step{blob.get('step')}-{Path(args.corpus).name}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"kind": "asr_stream", "checkpoint": str(path), "run": path.parent.name,
                                   "step": blob.get("step"), "corpus": str(args.corpus), "device": device,
                                   "time": time.strftime("%Y-%m-%d %H:%M:%S"), **r}, indent=1))
        print(f"written      {out}")
    return 0


def main(argv=None) -> int:
    global ROOT
    ap = argparse.ArgumentParser(prog="python -m aksharallm.dictate",
                                 description="Dictation: hotkey, cleanup, corrections (docs/23).")
    ap.add_argument("--root", default=None, help="the repository (default: the current directory)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("status", help="what is installed, running and learned")
    s.set_defaults(fn=cmd_status)

    s = sub.add_parser("daemon", help="load the models once and wait for the shortcut")
    s.add_argument("--bg", action="store_true", help="start it in the background")
    s.add_argument("--stop", action="store_true", help="stop a running one")
    s.add_argument("--device", default=None)
    s.set_defaults(fn=cmd_daemon)

    for name, help_ in (("toggle", "start listening, or finish and type the text (the shortcut)"),
                        ("start", "start listening"), ("stop", "finish and type the text"),
                        ("cancel", "stop listening and throw the audio away")):
        s = sub.add_parser(name, help=help_)
        s.set_defaults(fn=_client(name))

    s = sub.add_parser("install-shortcut", help="add a GNOME keyboard shortcut that runs toggle")
    s.add_argument("--binding", default="<Super><Alt>d", metavar="KEYS",
                   help="GNOME key syntax, quoted: '<Super><Alt>d' (the default), "
                        "'<Ctrl><Alt>space', '<Super>F9'")
    s.set_defaults(fn=cmd_install_shortcut)
    s = sub.add_parser("uninstall-shortcut", help="remove it again")
    s.set_defaults(fn=cmd_uninstall_shortcut)

    s = sub.add_parser("file", help="dictate audio files through the same pipeline")
    s.add_argument("files", nargs="+")
    s.add_argument("--device", default=None)
    s.set_defaults(fn=cmd_file)

    s = sub.add_parser("clean", help="cleanup alone, on words you type")
    s.add_argument("words", nargs="+")
    s.set_defaults(fn=cmd_clean)

    s = sub.add_parser("correct", help="teach it: what it wrote, and what you meant")
    s.add_argument("--shown", required=True)
    s.add_argument("--corrected", required=True)
    s.set_defaults(fn=cmd_correct)

    s = sub.add_parser("dictionary", help="the personal dictionary and learned replacements")
    s.add_argument("action", nargs="?", choices=["list", "add", "remove"], default="list")
    s.add_argument("words", nargs="*")
    s.set_defaults(fn=cmd_dictionary)

    s = sub.add_parser("history", help="recent dictations")
    s.add_argument("--limit", type=int, default=20)
    s.set_defaults(fn=cmd_history)

    s = sub.add_parser("punct-eval", help="score the punctuation tagger on held-out text")
    s.add_argument("checkpoint")
    s.add_argument("--val-bin", default="data/blend/val.bin")
    s.add_argument("--passages", type=int, default=300)
    s.add_argument("--device", default=None)
    s.add_argument("--no-write", action="store_true")
    s.set_defaults(fn=cmd_punct_eval)

    s = sub.add_parser("stream-eval", help="measure the live preview: latency, revisions, cost")
    s.add_argument("checkpoint", help="a recogniser run name or .pt")
    s.add_argument("--corpus", default="data/asr/test-clean")
    s.add_argument("--limit", type=int, default=200)
    s.add_argument("--tick", type=float, default=0.5, help="seconds between re-readings")
    s.add_argument("--device", default=None)
    s.add_argument("--no-write", action="store_true")
    s.set_defaults(fn=cmd_stream_eval)

    args = ap.parse_args(argv)
    if args.root:
        ROOT = Path(args.root).resolve()
        os.chdir(ROOT)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
