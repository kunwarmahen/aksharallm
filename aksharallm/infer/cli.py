"""Talking to a trained checkpoint from a terminal.

    python -m aksharallm.infer.cli                       what has been trained so far
    python -m aksharallm.infer.cli small-code            chat/complete with it, interactively
    python -m aksharallm.infer.cli small-code --probes   the fixed prose suite
    python -m aksharallm.infer.cli small-code --tasks    the Python tasks, actually executed
    python -m aksharallm.infer.cli --compare fluency     one probe, across every step so far

Modes:
    complete  raw text continuation — what a *base* model does, and all it can do
    chat      multi-turn, ChatML — needs an SFT'd model (Phase 3); refused on a base one
    code      a function signature and docstring in, a function body out, optionally run

Everything here is a thin front end over `aksharallm.infer.playground`, which the portal's
Playground tab uses too. Same device policy (the GPU, unless a run is training), same
history file, same graded tasks — so a number you get here and a number you get in the
browser mean the same thing.

Read with: docs/07-inference.md -- the chapter this implements; it ends with the order to read
these files in.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from ..config import ModelConfig
from ..model.transformer import Transformer
from ..tokenizer.tokenizer import Tokenizer
from .checkpoints import CheckpointStore, InferError
from .engine import SamplingParams
from .generate import generate  # noqa: F401  (re-exported: imported by aksharallm.eval)
from .playground import Playground
from .tasks import CHAT_PROMPTS, PROBES, TASKS_BY_ID


# --------------------------------------------------------------------------------------
# the plain loaders, kept because `aksharallm.eval.evaluate` imports them
# --------------------------------------------------------------------------------------

def load_model(ckpt_path: str, device: str = "cuda") -> tuple[Transformer, dict]:
    """Load a checkpoint into a model, with no engine, no policy and no bookkeeping.

    Evaluation wants exactly this: one model, one device it chose itself, held for the
    duration of a benchmark. `engine.Engine` is the other shape — resident, swappable and
    careful about whose GPU it is.
    """
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    mcfg = ModelConfig(**ckpt["model_config"])
    model = Transformer(mcfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt


def resolve_tokenizer(ckpt: dict, override: str | None) -> str:
    if override:
        return override
    path = ckpt.get("config", {}).get("data", {}).get("tokenizer")
    if path and Path(path).exists():
        return path
    raise SystemExit("could not find the tokenizer; pass --tokenizer explicitly")


# --------------------------------------------------------------------------------------
# formatting
# --------------------------------------------------------------------------------------

def human(n: float | None, suffix: str = "") -> str:
    if n is None:
        return "–"
    for unit, size in (("B", 1e9), ("M", 1e6), ("k", 1e3)):
        if abs(n) >= size:
            return f"{n / size:.1f}{unit}{suffix}"
    return f"{n:.0f}{suffix}"


def describe(info: dict) -> str:
    bits = [f"{info['rel']:<32}", f"{info['stage_label']:<16}"]
    bits.append(f"step {info['step']:>6}" if info["step"] is not None else "step      ?")
    if info["max_steps"]:
        bits[-1] += f"/{info['max_steps']:<6}"
    bits.append(f"val {info['best_val']:.4f}" if info["best_val"] is not None else "val      ?")
    bits.append(f"{human(info['params'])} params")
    bits.append(f"{human(info['tokens_seen'])} tok seen")
    if info["error"]:
        bits.append(f"!! {info['error']}")
    return "  ".join(bits)


def print_checkpoints(store: CheckpointStore) -> int:
    rows = store.list()
    if not rows:
        print("no checkpoints yet — nothing under checkpoints/*/ has been trained.")
        print("start a run with scripts/phase2.sh, or scripts/portal.sh for a browser.")
        return 1
    print(f"{len(rows)} checkpoint(s) under {store.root / 'checkpoints'}:\n")
    for info in rows:
        print("  " + describe(info.as_dict()))
    default = store.default()
    if default:
        print(f"\ndefault: {default.rel}   (a bare run name also works: "
              f"`... {default.run}`)")
    return 0


def print_adapters(pg: Playground) -> int:
    rows = pg.adapters.list()
    if not rows:
        print("no adapters yet. Train one with:\n"
              "  python -m aksharallm.train.sft --base <ckpt> --data-dir data/sft \\\n"
              "      --tokenizer <tok> --out-dir checkpoints/<run> --qlora")
        return 0
    print(f"{len(rows)} adapter(s) under checkpoints/:\n")
    for a in rows:
        if a.error:
            print(f"  {a.rel:<34} {a.error}")
            continue
        print(f"  {a.rel:<34} r={a.r:<3} {a.targets:<11} "
              f"{a.params / 1e6:5.2f}M  {a.size / 1e6:6.1f} MB  stage={a.stage}"
              + (f"  val {a.val_loss:.4f}" if a.val_loss is not None else ""))
        print(f"  {'':<34} base: {a.base_path}"
              + (f"  [{a.base_quant}]" if a.base_quant else ""))
    print("\nuse one with:  ... <checkpoint> --adapter <run/name.lora.pt>")
    return 0


def print_plan(pg: Playground):
    status = pg.status()
    plan = status["plan"]
    where = "GPU" if plan["device"] == "cuda" else "CPU"
    print(f"[{where}] {plan['reason']}", file=sys.stderr)


# --------------------------------------------------------------------------------------
# the interactive and one-shot paths
# --------------------------------------------------------------------------------------

def stream_to_stdout(pg: Playground, **kw) -> dict:
    """Run one generation, printing tokens as they arrive. Returns the final stats."""
    stats: dict = {}
    stream = pg.stream(**kw)
    try:
        for kind, payload in stream:
            if kind == "delta":
                sys.stdout.write(payload)
                sys.stdout.flush()
            elif kind == "test":
                # Printed before `done` because that is the order it happened in.
                mark = "PASS" if payload["ok"] else payload["status"].upper()
                print(f"\n\n  [{mark}] {payload['detail']}", file=sys.stderr)
                if payload.get("stderr") and not payload["ok"]:
                    print("  " + payload["stderr"].strip().splitlines()[-1], file=sys.stderr)
            elif kind == "done":
                stats = payload
    except KeyboardInterrupt:
        # Ctrl-C stops the model, not the program: closing the generator ends the decode
        # loop and releases the engine lock, and you are back at the prompt.
        print("\n(interrupted)", file=sys.stderr)
    finally:
        stream.close()
    print()
    if stats:
        # "GPU"/"CPU", not torch's "cuda": the reader owns a graphics card, not a runtime.
        where = "GPU" if stats.get("device") == "cuda" else "CPU"
        spec = stats.get("speculative")
        # Drafting cannot change the text, so it is reported as a *speed* line rather than
        # mixed in with the model's own numbers: accepted guesses, and tokens per pass of
        # the real model (1.0 is what plain decoding gets).
        drafted = (f"  ·  drafted {spec['accept_rate'] * 100:.0f}% accepted, "
                   f"{spec['tokens_per_forward']:.2f} tokens per model pass" if spec else "")
        print(f"  {stats.get('tokens', 0)} tokens in {stats.get('elapsed_s', 0):.1f}s "
              f"({(stats.get('tok_per_s') or 0):.1f} tok/s on the {where}){drafted}",
              file=sys.stderr)
    return stats


def interactive(pg: Playground, ckpt_id: str, mode: str, args,
                adapter: str | None = None) -> int:
    info = pg.store.get(ckpt_id)
    banner = {
        "complete": "Completion mode: type a prompt and the model continues it.",
        "chat": "Chat mode: multi-turn conversation.",
        "code": "Code mode: paste a signature and docstring, the model writes the body.",
    }[mode]
    print(banner, file=sys.stderr)
    print("  /reset clears history, /probes lists the fixed prompts, /task <id> runs a "
          "graded task,\n  /quit exits. Ctrl-C stops a generation.", file=sys.stderr)
    messages: list[dict] = []

    while True:
        try:
            line = input("\nyou> " if mode == "chat" else "\n> ")
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        text = line.strip()
        if not text:
            continue
        if text in ("/quit", "/exit"):
            return 0
        if text == "/reset":
            messages = []
            print("(history cleared)", file=sys.stderr)
            continue
        if text == "/probes":
            pool = CHAT_PROMPTS if mode == "chat" else PROBES
            for p in pool:
                print(f"  {p.id:<12} {p.prompt.splitlines()[0][:60]}", file=sys.stderr)
            print("  run one with /probe <id>", file=sys.stderr)
            continue
        if text.startswith("/probe "):
            pool = {p.id: p for p in (CHAT_PROMPTS if mode == "chat" else PROBES)}
            probe = pool.get(text.split(None, 1)[1].strip())
            if not probe:
                print("no such probe; /probes lists them", file=sys.stderr)
                continue
            print(f"\n{probe.prompt}", file=sys.stderr)
            stream_to_stdout(pg, ckpt_id=ckpt_id, mode=mode, prompt=probe.prompt,
                             adapter=adapter,
                             probe=probe.id, params=sampling(args), device=args.device)
            print(f"  expected: {probe.expect}", file=sys.stderr)
            continue
        if text.startswith("/task "):
            task_id = text.split(None, 1)[1].strip()
            if task_id not in TASKS_BY_ID:
                print(f"no such task; known: {', '.join(sorted(TASKS_BY_ID))}",
                      file=sys.stderr)
                continue
            stream_to_stdout(pg, ckpt_id=ckpt_id, mode=mode, task=task_id,
                             adapter=adapter,
                             params=sampling(args), device=args.device)
            continue

        if mode == "chat":
            stats = stream_to_stdout(pg, ckpt_id=ckpt_id, mode="chat", prompt=text,
                                     adapter=adapter,
                                     messages=messages, system=args.system,
                                     params=sampling(args), device=args.device)
            messages.append({"role": "user", "content": text})
            messages.append({"role": "assistant", "content": stats.get("text", "")})
            # Keep the transcript from outgrowing a 1024-token window: the engine truncates
            # from the left anyway, but dropping old turns here keeps the prompt honest.
            if len(messages) > 20:
                messages = messages[-20:]
        else:
            stream_to_stdout(pg, ckpt_id=ckpt_id, mode=mode, prompt=text,
                             adapter=adapter,
                             params=sampling(args), device=args.device)
        _ = info


def sampling(args) -> SamplingParams:
    return SamplingParams(max_new_tokens=args.max_new_tokens, temperature=args.temperature,
                          top_k=args.top_k, top_p=args.top_p,
                          repetition_penalty=args.repetition_penalty, seed=args.seed,
                          ngram=getattr(args, "ngram", 0) or 0)


# --------------------------------------------------------------------------------------
# the suites and the history
# --------------------------------------------------------------------------------------

def run_probes(pg: Playground, ckpt_id: str, mode: str, args,
               adapter: str | None = None) -> int:
    def show(row):
        print(f"\n─── {row['probe']}  ({row['group']}) " + "─" * 30)
        print(f"prompt:   {row['prompt']}")
        print(f"expected: {row['expect']}")
        print(f"output:   {row['output'].strip()[:600]}")

    out = pg.run_probes(ckpt_id, mode=mode, params=sampling(args), device=args.device,
                        adapter=adapter,
                        on_result=show)
    prov = out["provenance"]
    val = f", val {prov['best_val']:.4f}" if prov.get("best_val") is not None else ""
    print(f"\n{len(out['rows'])} probes against {ckpt_id} (step {prov['step']}{val})")
    print("recorded in logs/playground.jsonl — run this again at a later step and compare "
          "with --compare <probe>.")
    return 0


def run_tasks(pg: Playground, ckpt_id: str, mode: str, args,
              adapter: str | None = None) -> int:
    ids = args.task or None

    def show(row):
        mark = "PASS" if row["ok"] else row["status"].upper()
        print(f"  {mark:<8} {row['task']:<16} {row['difficulty']:<8} {row['detail'][:70]}")

    print(f"running {len(ids or TASKS_BY_ID)} Python tasks against {ckpt_id}")
    if pg.cfg.run_tests:
        print("  (the generated code is executed — see aksharallm/infer/sandbox.py)")
    out = pg.run_tasks(ckpt_id, mode=mode, task_ids=ids, params=sampling(args),
                       adapter=adapter,
                       device=args.device, on_result=show)
    prov = out["provenance"]
    rate = out["pass_rate"]
    print(f"\n{out['passed']}/{out['total']} passed"
          f"{f' ({rate:.0%})' if rate is not None else ''}"
          f"  —  {prov['run']} step {prov['step']}, val {prov['best_val']}")
    if args.show_code:
        for row in out["rows"]:
            print(f"\n─── {row['task']} ({row['status']}) " + "─" * 30)
            # The executed program, not the raw generation: when a task fails this is what
            # tells you whether the model wrote bad code or the extraction trimmed it in
            # the wrong place.
            print((row.get("program") or row["output"]).rstrip()[:1500])
    return 0 if out["passed"] == out["total"] else 1


def show_compare(pg: Playground, probe: str, run: str | None) -> int:
    out = pg.history.compare(probe, run=run)
    if not out["count"]:
        seen = ", ".join(e["probe"] for e in pg.history.probes_seen()) or "none yet"
        print(f"no history for probe '{probe}'. Recorded so far: {seen}")
        print("run `--probes` against a checkpoint to start building it.")
        return 1
    print(f"probe '{probe}' — {out['count']} generation(s), oldest first\n")
    for row in out["rows"]:
        val = f"{row['best_val']:.4f}" if row["best_val"] is not None else "?"
        loss = f"{row['train_loss']:.4f}" if row["train_loss"] is not None else "?"
        print(f"─── {row['run']}/{row['checkpoint']}  step {row['step']}  "
              f"val {val}  ema {loss}  ({row['iso']}) " + "─" * 10)
        print((row["output"] or "").strip()[:600])
        if row.get("test"):
            print(f"    [{row['test'].get('status')}] {row['test'].get('detail', '')[:70]}")
        print()
    return 0


def show_history(pg: Playground, args) -> int:
    rows = pg.history.recent(args.limit, run=args.run, mode=args.mode_filter)
    if not rows:
        print("nothing recorded yet.")
        return 1
    print(f"{len(rows)} most recent generation(s), newest first "
          f"({pg.history.stats()['path']})\n")
    for r in rows:
        head = (f"{r.get('iso')}  {r.get('run')}/{r.get('checkpoint')}  "
                f"step {r.get('step')}  {r.get('mode')}")
        if r.get("probe"):
            head += f"  probe={r['probe']}"
        if r.get("task"):
            head += f"  task={r['task']}"
        if (r.get("test") or {}).get("status"):
            head += f"  [{r['test']['status']}]"
        print(head)
        print(f"    > {(r.get('prompt') or '').strip()[:100]}")
        print(f"    {(r.get('output') or '').strip()[:200]}\n")
    return 0


# --------------------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="python -m aksharallm.infer.cli",
        description="Generate from an aksharallm checkpoint, and check what it can do.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Modes:")[0].split("\n", 2)[2])
    ap.add_argument("checkpoint", nargs="?",
                    help="a run name (small-code), an id (small-code/ckpt_best.pt) or a "
                         "path. Omit to list what has been trained.")
    ap.add_argument("--list", action="store_true", help="list checkpoints and exit")
    ap.add_argument("--mode", choices=["complete", "chat", "code"], default="complete")
    ap.add_argument("--prompt", default=None, help="one-shot prompt; omit for interactive")
    ap.add_argument("--probes", action="store_true",
                    help="run the fixed prompt suite and record the results")
    ap.add_argument("--tasks", action="store_true",
                    help="run the Python tasks and execute the generated code")
    ap.add_argument("--task", action="append",
                    help="one task id (repeatable); implies --tasks")
    ap.add_argument("--show-code", action="store_true",
                    help="with --tasks, print what the model wrote")
    ap.add_argument("--compare", metavar="PROBE",
                    help="show one probe's history across checkpoints and exit")
    ap.add_argument("--history", action="store_true", help="show recent generations and exit")
    ap.add_argument("--limit", type=int, default=20, help="rows for --history")
    ap.add_argument("--run", default=None, help="filter --history/--compare to one run")
    ap.add_argument("--mode-filter", default=None, help="filter --history to one mode")

    gen = ap.add_argument_group("sampling")
    gen.add_argument("--max-new-tokens", type=int, default=256)
    gen.add_argument("--temperature", type=float, default=0.8)
    gen.add_argument("--top-k", type=int, default=50)
    gen.add_argument("--top-p", type=float, default=0.95)
    gen.add_argument("--repetition-penalty", type=float, default=1.0)
    gen.add_argument("--seed", type=int, default=None,
                     help="fixes sampling, so two checkpoints can be compared on one prompt")
    gen.add_argument("--ngram", type=int, default=0, metavar="N",
                     help="speculative decoding by lookup: draft the next tokens from the "
                          "last N of the text so far (3 is a good start, 0 is off). The "
                          "output is unchanged; only the speed differs")
    gen.add_argument("--system", default=None, help="system prompt for chat mode")

    env = ap.add_argument_group("where it runs")
    env.add_argument("--device", choices=["auto", "cuda", "cpu"], default=None,
                     help="default: the GPU, unless a run is training — then the CPU")
    env.add_argument("--adapter", default=None,
                     help="a LoRA adapter to put on top of the checkpoint (a path, or "
                          "run/name.lora.pt). One base, many specialisations — and an SFT "
                          "adapter unlocks --mode chat on a base checkpoint.")
    env.add_argument("--list-adapters", action="store_true",
                     help="list the adapters found under checkpoints/ and exit")
    env.add_argument("--tokenizer", default=None,
                     help="override the tokenizer recorded in the checkpoint (rarely right)")
    env.add_argument("--no-record", action="store_true",
                     help="do not append to logs/playground.jsonl")
    env.add_argument("--root", default=None, help="repo root (default: this checkout)")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.device == "auto":
        args.device = None
    if args.task:
        args.tasks = True

    pg = Playground(args.root)
    try:
        if args.compare:
            return show_compare(pg, args.compare, args.run)
        if args.history:
            return show_history(pg, args)
        if args.list_adapters:
            return print_adapters(pg)
        if args.list or not args.checkpoint:
            return print_checkpoints(pg.store)

        try:
            ckpt_id = pg.store.identify(args.checkpoint)
        except InferError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2

        adapter_id = None
        if args.adapter:
            try:
                adapter_id = pg.adapters.identify(args.adapter)
            except InferError as exc:
                print(f"error: {exc}", file=sys.stderr)
                return 2

        info = pg.store.get(ckpt_id)
        print(describe(info.as_dict()), file=sys.stderr)
        print(f"  {info.as_dict()['stage_note']}", file=sys.stderr)
        if adapter_id:
            ad = pg.adapters.get(adapter_id)
            print(f"+ adapter {ad.rel}  r={ad.r} {ad.targets} "
                  f"({ad.params:,} params, {ad.size / 1e6:.1f} MB) -> stage {ad.stage}",
                  file=sys.stderr)
        print_plan(pg)
        if args.tokenizer:
            print("  --tokenizer overrides the one this checkpoint was trained with; if it "
                  "is not the same tokenizer the output will be fluent nonsense.",
                  file=sys.stderr)

        if args.tasks:
            return run_tasks(pg, ckpt_id, args.mode if args.mode == "chat" else "complete",
                             args, adapter_id)
        if args.probes:
            return run_probes(pg, ckpt_id, args.mode, args, adapter_id)

        if args.prompt is not None:
            stream_to_stdout(pg, ckpt_id=ckpt_id, mode=args.mode, prompt=args.prompt,
                             system=args.system, params=sampling(args),
                             device=args.device, adapter=adapter_id,
                             record=not args.no_record)
            return 0
        return interactive(pg, ckpt_id, args.mode, args, adapter_id)
    except InferError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print()
        return 130
    finally:
        pg.close()


if __name__ == "__main__":
    sys.exit(main())
