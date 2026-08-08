"""The eval harness, from a terminal.

    python -m aksharallm.eval suites                       # what can be measured, and why
    python -m aksharallm.eval fetch --all                  # download the benchmarks, once
    python -m aksharallm.eval small-code                   # the default set, on the best ckpt
    python -m aksharallm.eval small-code --suite all --limit 0
    python -m aksharallm.eval small-code --suite judge --judge-model gemma4:31b
    python -m aksharallm.eval report --suite mmlu          # every score, across steps

The first positional is a checkpoint unless it names a subcommand, so the common case is
short. A checkpoint reference is a run name (`small-code` → its best checkpoint), an id
(`small-code/ckpt_last.pt`) or a path — the same resolution the Playground uses.

Everything writes to `logs/eval/<timestamp>-<run>-<label>.json`, which the portal's Eval tab
reads. A run started here appears there, and the other way round.

Read with: docs/13-eval.md -- the chapter this implements; it ends with the order to read these
files in.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
from pathlib import Path

from .jobs import announced
from .report import Results, compare_table, results_dir, summary_table
from .runner import Harness, Options, describe
from .sources import EvalError, SOURCES, fetch, load, status
from .suites import ALL_SUITES, DEFAULT_SUITES, SUITES, build, catalogue, resolve

SUBCOMMANDS = ("run", "fetch", "suites", "report", "contaminate", "domains",
                "calibrate")


def cmd_suites(args) -> int:
    print("\n  What can be measured, and what to expect at this project's scale.\n")
    cached = {row["name"] for row in status() if row["cached"]}
    for row in catalogue():
        tags = []
        if row["groups"]["default"]:
            tags.append("default")
        if row["groups"]["fast"]:
            tags.append("fast")
        head = f"  {row['name']}"
        print(f"{head:<16} [{row['kind']}]"
              + (f"  ({', '.join(tags)})" if tags else ""))
        print(f"                  {row['blurb']}")
        if row["source"]:
            state = "cached" if row["source"] in cached else "NOT downloaded"
            print(f"                  data: {row['source']} — {state}"
                  + (f", {row['shots']}-shot from {row['shot_source']}" if row["shot_source"] else ""))
        print(f"                  expect: {row['expect']}")
        print()
    print(f"  groups:  default = {', '.join(DEFAULT_SUITES)}")
    print(f"           all     = {', '.join(ALL_SUITES)}\n")
    return 0


def cmd_fetch(args) -> int:
    if args.list:
        print()
        for row in status():
            mark = "✓" if row["cached"] else " "
            size = f"{row['bytes'] / 1e6:.1f} MB" if row["cached"] else ""
            rows = f"{row['rows']:,} rows" if row.get("rows") else ""
            print(f"  {mark} {row['name']:<16} {rows:>12}  {size:>9}  {row['repo'] or row['repos'][0]}")
            print(f"     {row['note']}")
        print()
        return 0

    names = args.names or (list(SOURCES) if args.all else [])
    if not names:
        print("nothing to fetch. Name datasets, or use --all / --list.", file=sys.stderr)
        return 2
    for name in names:
        fetch(name, refresh=args.refresh)
    return 0


def cmd_report(args) -> int:
    results = Results(args.root)
    if args.suite:
        print(compare_table(results.compare(args.suite, run=args.run)))
        print()
        return 0
    rows = results.rows(limit=args.limit, run=args.run)
    if not rows:
        print("\n  no evaluations recorded yet. Run one:  python -m aksharallm.eval <run>\n")
        return 0
    names = [n for n in SUITES if any(n in r["scores"] for r in rows)]
    print()
    print("  " + f"{'when':<17}{'step':>9}  " + "".join(f"{n[:11]:>12}" for n in names))
    print("  " + "-" * (28 + 12 * len(names)))
    for row in rows:
        when = time.strftime("%Y-%m-%d %H:%M", time.localtime(row["when"]))
        step = f"{row['step']:,}" if row["step"] is not None else "–"
        cells = ""
        for name in names:
            entry = row["scores"].get(name)
            if not entry or entry.get("score") is None:
                cells += f"{'–':>12}"
            elif entry.get("kind") == "ppl":
                cells += f"{entry['score']:>12.3f}"
            else:
                cells += f"{entry['score'] * 100:>11.1f}%"
        print(f"  {when:<17}{step:>9}  {cells}   {row['checkpoint']}")
    print()
    print("  One suite across every step:  python -m aksharallm.eval report --suite mmlu\n")
    return 0


def cmd_run(args) -> int:
    harness = Harness(args.root)
    opts = Options(
        suites=resolve(args.suite), limit=args.limit, shots=args.shots,
        device=args.device, adapter=args.adapter, batch_tokens=args.batch_tokens,
        max_new_tokens=args.max_new_tokens, judge_model=args.judge_model,
        keep_items=not args.no_items, label=args.label)

    pre = harness.preflight(args.checkpoint, opts)
    if pre["missing"]:
        print(f"\n  Benchmark data is not downloaded: {', '.join(pre['missing'])}\n"
              f"  Fetch it once (it is cached under data/eval/ and reused forever):\n\n"
              f"      python -m aksharallm.eval fetch {' '.join(pre['missing'])}\n",
              file=sys.stderr)
        return 2
    for note in pre["notes"]:
        print(f"  note: {note}")

    last = [0.0]

    def progress(done: int, total: int, label: str):
        # A terminal line per second is enough. The portal reads the same stdout from a log
        # file, so this doubles as the browser's progress bar — hence the parseable shape.
        now = time.monotonic()
        if now - last[0] < 1.0 and done < total:
            return
        last[0] = now
        print(f"[eval] {label} {done}/{total} ({done / max(1, total) * 100:.0f}%)", flush=True)

    result = harness.run(args.checkpoint, opts, progress=progress, val_bin=args.val_bin)
    path = harness.save(result, args.json)
    print(summary_table(result))
    print(f"  wrote {path}\n")
    return 0


def cmd_contaminate(args) -> int:
    """n-gram overlap between the benchmark suites and the training corpus."""
    from ..config import load_config
    from ..tokenizer.tokenizer import Tokenizer
    from . import contamination as con
    from .runner import Harness

    harness = Harness(args.root)
    root = harness.root
    cfg = load_config(args.config)
    tok = Tokenizer(cfg.data.tokenizer)

    bins = [s["bin"] for s in (cfg.data.train_sources or [])] or [cfg.data.train_bin]
    bins = [str(root / b) if not Path(b).is_absolute() else b for b in bins]
    missing = [b for b in bins if not Path(b).is_file()]
    if missing:
        print(f"error: training data not found: {', '.join(missing)}")
        return 1

    # A full pass is half an hour, so its report has to be reusable: `--report` re-scores a
    # result against a scan that already happened rather than doing it again. Without this
    # the only way to answer "what does this do to my numbers?" was another 10B-token scan,
    # which is the sort of thing that stops a check from being run.
    if args.report:
        out = con.load_result(args.report)
        _rescore(out, args.against, con)
        return 0

    names = resolve(args.suite)
    texts = []
    for name in names:
        suite = SUITES[name]
        if not suite.source:
            continue                       # perplexity has no items to leak
        rows = load(suite.source, root, limit=args.limit)
        items = build(name, rows)
        texts.extend(con.item_texts(name, items))
    if not texts:
        print("error: no suites with items were selected")
        return 1

    probe = con.build_probe(texts, tok, n=args.n, keep_tokens=args.verify)
    total_tokens = sum(Path(b).stat().st_size // 2 for b in bins)
    print(f"probing {len(texts):,} texts ({len(probe):,} distinct {args.n}-grams) "
          f"against {total_tokens:,} training tokens\n")

    last = [0.0]

    def progress(done, total, label):
        now = time.monotonic()
        if now - last[0] < 1.0 and done < total:
            return
        last[0] = now
        print(f"[contam] {label} {done:,}/{total:,} ({done / max(1, total) * 100:.0f}%)",
              flush=True)

    hits: dict[str, int] = {}
    where: dict[str, tuple[str, int]] = {}
    for b in bins:
        for key, count in con.scan_bin(b, probe, max_tokens=args.max_tokens,
                                       progress=None if args.quiet else progress,
                                       where=where if args.verify else None).items():
            hits[key] = hits.get(key, 0) + count
    if args.verify and hits:
        before = len(hits)
        hits = con.verify(hits, probe, where, args.n)
        print(f"  verified {len(hits)}/{before} hits against the real token stream")

    out = con.summarise(hits, probe, args.n)

    # How much was actually looked at. Without this a partial scan is indistinguishable from
    # a full one -- same shape, same fields, a smaller dirty count -- and it is wrong in the
    # optimistic direction, which for a contamination check is the direction that matters.
    # `--limit` shrinks the probe (fewer benchmark items checked) and `--max-tokens` shrinks
    # the scan (less corpus read); both under-count, so both are recorded even when unset.
    scanned = sum(min(Path(b).stat().st_size // 2, args.max_tokens or total_tokens)
                  for b in bins)
    out["coverage"] = cov = con.coverage(
        total_tokens=total_tokens, max_tokens=None if scanned >= total_tokens else scanned,
        items_per_suite=args.limit, texts=len(texts), verified=args.verify)
    if cov["partial"]:
        print(f"\nPARTIAL SCAN — {scanned / max(1, total_tokens):.1%} of the corpus"
              f"{'' if args.limit is None else f', first {args.limit} items per suite'}. "
              f"Every number below is a LOWER BOUND on the real contamination.")

    print(f"\n{'suite':>12} {'part':>10} {'items':>7} {'dirty':>7} {'rate':>7}")
    for s in out["suites"]:
        for part, p in sorted(s["parts"].items()):
            rate = "–" if p["rate"] is None else f"{p['rate']:.1%}"
            print(f"{s['suite']:>12} {part:>10} {p['checkable']:>7} {p['dirty']:>7} {rate:>7}")
    print(f"\n{args.n}-gram overlap. 'question' leaking is common and mostly harmless — "
          f"benchmark questions are public text.\n'answered' is the one that matters: the "
          f"question WITH its answer, which is what a contaminated corpus memorises.")

    _rescore(out, args.against, con)

    path = Path(root) / "logs" / "eval" / f"contamination-{int(time.time())}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {path}")
    return 0


def _rescore(report: dict, against: str | None, con) -> None:
    """Print reported vs clean for a benchmark result, given a contamination report.

    `dirty_ids` is answer-leaks only. Dropping every item whose *question* appears in a web
    crawl would discard most of a public benchmark and then report a confident number on
    whatever was left — on our own blend that would have been 1,301 items instead of 156.
    """
    if not against:
        return
    result = con.load_result(against)
    dirty = set(report.get("dirty_ids") or [])
    print(f"\nre-scoring {Path(against).name} without the {len(dirty)} answer-leaked items:")
    any_row = False
    for name, res in (result.get("suites") or {}).items():
        clean = con.clean_score(res, dirty)
        if not clean:
            continue
        if clean["clean"] is None:
            # Say why, rather than dropping the row. A suite that quietly vanishes from
            # this table looks like a suite that was clean.
            any_row = True
            print(f"  {name:>14}  reported "
                  f"{'—' if clean['reported'] is None else format(clean['reported'], '.3f')}"
                  f"  clean —   ({clean.get('note', 'not computable')})")
            continue
        any_row = True
        delta = clean["clean"] - (clean["reported"] or 0)
        print(f"  {name:>14}  reported {clean['reported']:.3f}  "
              f"clean {clean['clean']:.3f}  ({delta:+.3f}, {clean['dropped']} dropped, "
              f"{clean['kept']} kept)")
    if not any_row:
        print("  (this result has no per-item verdicts — it was run with --no-items)")


def cmd_domains(args) -> int:
    """Held-out loss split by training source. One number is hiding two."""
    from ..infer.cli import load_model, resolve_tokenizer
    from ..infer.checkpoints import CheckpointStore
    from ..tokenizer.tokenizer import Tokenizer
    from . import domains as dom
    from .runner import Harness

    harness = Harness(args.root)
    store = CheckpointStore(args.root)
    ident = store.identify(args.checkpoint)
    path = store.resolve(*ident.split("/"))
    model, ckpt = load_model(str(path), device=args.device)
    tok = Tokenizer(resolve_tokenizer(ckpt, None))

    data_cfg = (ckpt.get("config") or {}).get("data", {})
    val_bin = args.val_bin or data_cfg.get("val_bin")
    if not val_bin:
        print("error: this checkpoint does not record a val_bin; pass --val-bin")
        return 1
    sources = data_cfg.get("train_sources")
    spans = dom.spans_for(val_bin, sources, tok)

    seq_len = args.seq_len or ckpt["model_config"].get("max_seq_len", 1024)
    rows = dom.per_domain_loss(model, val_bin, spans, seq_len, batches=args.batches,
                               batch_size=args.batch, device=args.device,
                               progress=_ticker("domains"))

    print(f"\n{args.checkpoint} on {val_bin}, {seq_len}-token windows\n")
    print(f"{'source':>22} {'tokens':>12} {'weight':>7} {'loss':>8} {'ppl':>9}  check")
    for r in rows:
        mark = {True: "ok", False: "MISMATCH", None: "unverified"}[r.get("verified")]
        loss = "–" if r["loss"] is None else f"{r['loss']:.4f}"
        ppl = "–" if r["loss"] is None else f"{r['perplexity']:.2f}"
        print(f"{r['name']:>22} {r['tokens']:>12,} "
              f"{(r.get('weight') or 0):>7.2f} {loss:>8} {ppl:>9}  {mark}")
    if any(r.get("verified") is False for r in rows):
        print("\nMISMATCH: a span's content does not match its name, so these boundaries are"
              "\nderived wrongly and the split above is meaningless. See docs/13.")
    b = dom.blended(rows)
    if b is not None:
        print(f"\nweight-blended: {b:.4f}  (compare with the run's own val loss; a big "
              f"disagreement means the spans are wrong)")

    # Written, not just printed. A split that only exists in a terminal cannot be compared
    # with the one you took ten thousand steps ago, and the portal has nothing to show —
    # which is exactly what happened: the Domains card stayed empty after a real run, from
    # the browser as well as from the CLI. The name is derived from the checkpoint's
    # *identity* rather than from whatever was typed, so `small-code` and the absolute path
    # the portal passes produce the same filename for the same checkpoint.
    slug = ident.replace("/", "-").removesuffix(".pt")
    out = {
        "checkpoint": ident,
        "val_bin": str(val_bin),
        "seq_len": seq_len,
        "batches": args.batches,
        "batch": args.batch,
        "device": args.device,
        "step": ckpt.get("step"),
        "rows": rows,
        "blended": b,
        "reading": "Sampled the same way inside every span, so these are comparable with "
                   "each other — but NOT with the trainer's val loss, which averages over "
                   "the whole file.",
    }
    dest = results_dir(args.root) / f"domains-{slug}-{time.strftime('%Y%m%d-%H%M%S')}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=1))
    print(f"written to {dest}")
    return 0


def cmd_calibrate(args) -> int:
    """Is the model's confidence honest? Accuracy asks a different question."""
    from ..data.loader import TokenDataset
    from ..infer.checkpoints import CheckpointStore
    from ..infer.cli import load_model
    from . import calibration as cal

    store = CheckpointStore(args.root)
    path = store.resolve(*store.identify(args.checkpoint).split("/"))
    model, ckpt = load_model(str(path), device=args.device)

    data_cfg = (ckpt.get("config") or {}).get("data", {})
    val_bin = args.val_bin or data_cfg.get("val_bin")
    if not val_bin:
        print("error: this checkpoint does not record a val_bin; pass --val-bin")
        return 1
    seq_len = args.seq_len or ckpt["model_config"].get("max_seq_len", 1024)
    ds = TokenDataset(val_bin, seq_len, args.device)

    print(f"{args.checkpoint} on {val_bin}, {seq_len}-token windows, on the {args.device}")
    print(f"collecting {args.batches} x {args.batch} batches, "
          f"subsampled to {args.positions:,} positions...")
    logits, targets = cal.collect(model, ds, args.batches, args.batch,
                                  max_positions=args.positions,
                                  progress=_ticker("calib"))
    res = cal.report(logits, targets)
    res["checkpoint"] = args.checkpoint
    res["val_bin"] = val_bin
    res["step"] = ckpt.get("step")
    # The familiar number beside the unfamiliar ones: if this disagrees with the run's own
    # recorded val loss, the calibration numbers are computed on something else.
    res["perplexity"] = cal.perplexity(logits.float(), targets)

    print()
    print(cal.format_report(res))
    print(f"\nperplexity on the same positions: {res['perplexity']:.3f}  "
          f"(cross-check against the run's own val loss)")

    # `results_dir`, not a bare relative "logs/eval": run from anywhere but the repo root
    # this wrote a logs/eval/ beside the shell's cwd, and the portal — which reads the
    # repo's — never saw the measurement.
    out = results_dir(args.root)
    out.mkdir(parents=True, exist_ok=True)
    dest = out / f"calibration-{Path(args.checkpoint).name}-{time.strftime('%Y%m%d-%H%M%S')}.json"
    dest.write_text(json.dumps(res, indent=1))
    print(f"written to {dest}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="python -m aksharallm.eval",
        description="Measure a checkpoint on real benchmarks.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    sub = ap.add_subparsers(dest="cmd")

    run = sub.add_parser("run", help="evaluate a checkpoint (the default)")
    run.add_argument("checkpoint", help="run name, run/name.pt, or a path")
    run.add_argument("--suite", default=None,
                     help="comma-separated, or a group: default, fast, mc, all. "
                          f"Suites: {', '.join(SUITES)}")
    run.add_argument("--limit", type=int, default=None,
                     help="items per suite; 0 for the whole cached split")
    run.add_argument("--shots", type=int, default=None, help="override the few-shot count")
    run.add_argument("--device", default=None, choices=("cuda", "cpu"),
                     help="default: the CPU while a run is training, else the GPU")
    run.add_argument("--adapter", default=None, help="a LoRA adapter to evaluate on top")
    run.add_argument("--batch-tokens", type=int, default=2048,
                     help="tokens per forward pass; lower it if scoring runs out of memory")
    run.add_argument("--max-new-tokens", type=int, default=256)
    run.add_argument("--judge-model", default=None, help="Ollama model for the judge suite")
    run.add_argument("--val-bin", default=None,
                     help="perplexity data; defaults to the checkpoint's own val split")
    run.add_argument("--label", default=None, help="goes in the result filename")
    run.add_argument("--no-items", action="store_true",
                     help="do not keep per-question verdicts in the JSON")
    run.add_argument("--json", default=None, help="write the result here instead")
    run.add_argument("--root", default=None)

    fetch_p = sub.add_parser("fetch", help="download benchmark data into data/eval/")
    fetch_p.add_argument("names", nargs="*", metavar="NAME",
                         help=f"one or more of: {', '.join(SOURCES)}")
    fetch_p.add_argument("--all", action="store_true")
    fetch_p.add_argument("--list", action="store_true", help="what is downloaded already")
    fetch_p.add_argument("--refresh", action="store_true", help="re-download even if cached")
    fetch_p.add_argument("--root", default=None)

    sub.add_parser("suites", help="what can be measured, and what to expect")

    rep = sub.add_parser("report", help="scores across every evaluation so far")
    rep.add_argument("--suite", default=None, help="one suite, across steps")
    rep.add_argument("--run", default=None, help="only this training run")
    rep.add_argument("--limit", type=int, default=25)
    rep.add_argument("--root", default=None)

    con_p = sub.add_parser("contaminate",
                           help="n-gram overlap between the suites and the training data")
    con_p.add_argument("--config", default="configs/small-code.yaml",
                       help="the run whose training data to check against")
    con_p.add_argument("--suite", default="mc", help="which suites to check")
    con_p.add_argument("--n", type=int, default=13, help="n-gram length (13 is standard)")
    con_p.add_argument("--limit", type=int, default=None, help="items per suite")
    con_p.add_argument("--max-tokens", type=int, default=None,
                       help="scan only the first N tokens of each bin (a quick look)")
    con_p.add_argument("--verify", action="store_true",
                       help="re-check every hit against the real tokens (drops collisions)")
    con_p.add_argument("--against", default=None,
                       help="a logs/eval/*.json result to re-score without dirty items")
    con_p.add_argument("--report", default=None,
                       help="re-score using an existing contamination report, skipping the "
                            "scan entirely")
    con_p.add_argument("--quiet", action="store_true")
    con_p.add_argument("--root", default=None)
    con_p.set_defaults(fn=cmd_contaminate)

    dom_p = sub.add_parser("domains", help="held-out loss split by training source")
    dom_p.add_argument("checkpoint")
    dom_p.add_argument("--val-bin", default=None)
    dom_p.add_argument("--seq-len", type=int, default=None)
    dom_p.add_argument("--batches", type=int, default=20)
    dom_p.add_argument("--batch", type=int, default=4)
    dom_p.add_argument("--device", default="cpu", choices=("cuda", "cpu"))
    dom_p.add_argument("--root", default=None)
    dom_p.set_defaults(fn=cmd_domains)

    cal_p = sub.add_parser("calibrate",
                           help="is the model's confidence honest? (ECE + temperature)")
    cal_p.add_argument("checkpoint")
    cal_p.add_argument("--val-bin", default=None)
    cal_p.add_argument("--seq-len", type=int, default=None)
    # Deliberately small. Calibration keeps the FULL logit vector per position, because
    # temperature scaling needs the whole distribution — see `calibration.collect`.
    cal_p.add_argument("--batches", type=int, default=8)
    cal_p.add_argument("--batch", type=int, default=4)
    cal_p.add_argument("--positions", type=int, default=20_000,
                       help="positions to keep logits for (memory is the constraint)")
    cal_p.add_argument("--device", default="cpu", choices=("cuda", "cpu"))
    cal_p.add_argument("--root", default=None)
    cal_p.set_defaults(fn=cmd_calibrate)

    return ap


def _ticker(tag: str, every: float = 1.0):
    """A progress callback in the one shape the portal can read.

    `[<tag>] <label> <done>/<total> (<pct>%)` is what `EvalJobs._PROGRESS_RE` matches, and
    matching it is the entire contract — a job whose lines do not is a job with no progress
    bar, which on a half-hour scan reads as a hang. Rate-limited so a fast inner loop does
    not turn the log into a flood.
    """
    last = [0.0]

    def tick(done: int, total: int, label: str = "") -> None:
        now = time.monotonic()
        if now - last[0] < every and done < total:
            return
        last[0] = now
        print(f"[{tag}] {label or tag} {done}/{total} "
              f"({done / max(1, total) * 100:.0f}%)", flush=True)

    return tick


#: Commands worth publishing to the portal. The rest finish before it could poll.
ANNOUNCED = {"run", "contaminate", "domains", "calibrate", "fetch"}


def _job_meta(args) -> dict:
    """What the Eval tab shows beside "running": enough to tell two jobs apart."""
    meta = {}
    for name, key in (("checkpoint", "checkpoint"), ("config", "config"),
                      ("label", "label")):
        value = getattr(args, name, None)
        if value:
            meta[key] = str(value)
    if suite := getattr(args, "suite", None):
        with contextlib.suppress(Exception):
            meta["suites"] = resolve(suite)
    return meta


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    # `python -m aksharallm.eval small-code` should work without typing `run`. Anything that
    # is not a subcommand and not a flag is a checkpoint.
    if argv and argv[0] not in SUBCOMMANDS and not argv[0].startswith("-"):
        argv = ["run"] + argv
    args = build_parser().parse_args(argv)
    if not args.cmd:
        build_parser().print_help()
        return 0
    try:
        # Newer subcommands carry their own handler on `fn`; the original four predate
        # that and are still dispatched by name.
        handler = getattr(args, "fn", None) or {
            "run": cmd_run, "fetch": cmd_fetch, "suites": cmd_suites,
            "report": cmd_report}[args.cmd]
        # Tell the portal what this terminal is doing, so the Eval tab shows a job started
        # here exactly as it shows one started there -- the results always appeared in both,
        # but the *running* state only ever did for the portal's own launches. Only the
        # long-running commands: announcing `suites` or `report` would flash a job that is
        # over before the tab's next poll. See jobs.py.
        if args.cmd not in ANNOUNCED:
            return handler(args)
        with announced(args.cmd, _job_meta(args), root=getattr(args, "root", None)):
            return handler(args)
    except EvalError as exc:
        print(f"\n  {exc}\n", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\n  stopped.\n", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
