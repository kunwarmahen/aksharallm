"""The generation loop: ask, parse, filter, verify, dedup, write — and stop when told.

The loop itself is thirty lines. What surrounds it is the two things that make a generation
run behave like everything else in this repo.

**It is a long, interruptible job, so it obeys the STOP contract.** `aksharallm/train/
stopfile.py` is already the single vocabulary for "stop now / stop at N / stop at this
time", shared by pretraining, SFT and QAT. Generating 5,000 verified Python samples is
hours, and the same three questions apply — so the same file answers them, with *kept
samples* in the place of training steps. A stop leaves a dataset that is complete as far as
it goes, with its provenance written, and the next run appends to it.

**Every drop is counted by reason.** The funnel is

    asked ──▶ answered ──▶ parsed ──▶ valid ──▶ verified ──▶ unique ──▶ kept

and a pass rate on its own tells you nothing about which of those five walls the samples hit.
The tally by reason does, and it is the number to read before touching anything: a run
losing 60% to `near_duplicate` needs a wider seed grid, one losing 60% to `tests_failed`
needs a better teacher, and one losing 60% to `unparseable` needs a clearer template. These
are three unrelated fixes behind one identical-looking percentage.

Attempts, and why they are capped low
-------------------------------------
A seed that comes back unparseable is retried once and then abandoned. It is tempting to
retry until it works, and it is wrong twice over: a model that ignored the output format at
temperature 0.9 usually ignores it again, and — more importantly — retrying the *same seed*
until it succeeds biases the dataset towards the cells the teacher finds easy. Abandoning it
and moving to the next cell spends the same budget on coverage instead.

Read with: docs/13-synthetic-data.md -- the chapter this implements; it ends with the order to
read these files in.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from ..train import stopfile
from . import filters
from .dataset import Dataset, SynthError
from .prompts import TEMPLATE_VERSION, seeds as seed_grid
from .recipes import Recipe, get_recipe
from .teacher import Teacher
from .verify import verify as verify_code


@dataclass
class GenerateOptions:
    """Everything a generation run needs that is not the teacher or the recipe."""

    n: int = 50
    seed: int = 0
    #: A ceiling on teacher calls, so a recipe that is failing every filter cannot run all
    #: night producing nothing. Default: six asks per wanted sample.
    max_asks: int | None = None
    max_attempts: int = 2
    #: Jaccard threshold for near-duplicates. Lower = stricter.
    dedup: float = 0.6
    #: Execute the tests (python recipe only). Turning this off is allowed and is recorded
    #: on every sample and in meta.json — unverified generated code is a different dataset,
    #: not the same one produced faster.
    verify: bool = True
    #: Run the tests a second time against a stubbed solution. See `verify.py`.
    mutate: bool = True
    sandbox_timeout_s: float = 10.0
    sandbox_memory_mb: int = 512
    stop_file: Path | None = None
    #: Wall-clock budget for this session, counted from the first ask.
    max_seconds: float | None = None
    min_chars: int = 20
    max_chars: int = 4000

    def as_dict(self) -> dict:
        return {"n": self.n, "seed": self.seed, "max_asks": self.max_asks,
                "max_attempts": self.max_attempts, "dedup": self.dedup,
                "verify": self.verify, "mutate": self.mutate,
                "sandbox_timeout_s": self.sandbox_timeout_s,
                "max_seconds": self.max_seconds}


@dataclass
class Stats:
    """The funnel, live. The same numbers `Dataset.stats()` reports afterwards."""

    asked: int = 0
    answered: int = 0
    parsed: int = 0
    kept: int = 0
    rejected: dict = field(default_factory=dict)
    started: float = field(default_factory=time.time)
    stopped: str | None = None
    last_error: str | None = None

    def drop(self, reason: str) -> None:
        self.rejected[reason] = self.rejected.get(reason, 0) + 1

    @property
    def elapsed(self) -> float:
        return time.time() - self.started

    @property
    def pass_rate(self) -> float | None:
        return (self.kept / self.asked) if self.asked else None

    def as_dict(self) -> dict:
        return {"asked": self.asked, "answered": self.answered, "parsed": self.parsed,
                "kept": self.kept, "rejected": dict(self.rejected),
                "elapsed_s": round(self.elapsed, 1), "pass_rate": self.pass_rate,
                "stopped": self.stopped, "last_error": self.last_error}


def preflight(recipe: Recipe, teacher: Teacher, opts: GenerateOptions,
              root: Path | None = None) -> list[str]:
    """Everything worth refusing *before* the first teacher call. Returns warnings.

    The two hard refusals are both about not producing a dataset that quietly means
    something other than what it says: a teacher that is not pulled, and the Python recipe
    with no working sandbox. The second one matters — "verified Python" whose tests were
    never executed is not a weaker version of the same dataset, it is a different dataset
    with the same name.
    """
    ok, note = teacher.available()
    if not ok:
        raise SynthError(note)

    warnings = [note]
    if recipe.verified and opts.verify:
        from ..infer import sandbox

        can, why = sandbox.available()
        if not can:
            raise SynthError(
                f"the '{recipe.name}' recipe verifies every sample by running its tests, "
                f"and this machine cannot: {why} Pass --no-verify to generate unverified "
                "samples — they are marked as such in every sample and in meta.json.")
    elif recipe.verified and not opts.verify:
        warnings.append(
            "VERIFICATION IS OFF. The tests will not be run, so nothing here checks that "
            "the generated code is correct. Every sample is stamped verified: false.")

    from .teacher import contention

    busy = contention(root, teacher.name)
    if not busy["safe"]:
        warnings.append(busy["reason"])
    return warnings


def generate(dataset: Dataset, recipe: Recipe | str, teacher: Teacher,
             opts: GenerateOptions | None = None, on_progress=None,
             root: Path | None = None) -> Stats:
    """Fill `dataset` with `opts.n` samples, or until stopped. Returns the funnel."""
    recipe = get_recipe(recipe) if isinstance(recipe, str) else recipe
    opts = opts or GenerateOptions()
    stats = Stats()
    max_asks = opts.max_asks or max(opts.n * 6, opts.n + 20)
    deadline = (time.time() + opts.max_seconds) if opts.max_seconds else None

    dataset.open(recipe.name, teacher.name, teacher.cfg.host, opts.as_dict(),
                 TEMPLATE_VERSION)

    # Seeded from everything already in the dataset, so appending to a dataset walks new
    # cells of the grid instead of starting the same shuffle again and duplicating its way
    # through the first hundred.
    already = dataset.n_samples()
    deduper = filters.Deduper(threshold=opts.dedup)
    for row in dataset.samples():
        try:
            deduper.add(recipe.dedup_key(row))
        except KeyError:
            continue

    plan = seed_grid(recipe.name, max_asks, seed=opts.seed + already)
    reason: str | None = None

    for seed in plan:
        if stats.kept >= opts.n:
            reason = reason or "done"
            break
        if stats.asked >= max_asks:
            reason = "ask budget spent"
            break
        if deadline and time.time() >= deadline:
            reason = "time budget spent"
            break
        stop = stopfile.read(opts.stop_file) if opts.stop_file else None
        fired = stopfile.reached(stop, already + stats.kept)
        if fired:
            reason = fired
            break

        for attempt in range(opts.max_attempts):
            stats.asked += 1
            dataset.count_asked()
            try:
                reply = teacher.ask(recipe.messages(seed))
            except Exception as exc:                      # noqa: BLE001
                # A teacher that fell over (Ollama restarted, model evicted, request timed
                # out) is not a bad sample — it is a failed call. Counted separately so a
                # pass rate is not quietly destroyed by an infrastructure problem.
                stats.drop("teacher_error")
                stats.last_error = str(exc)
                dataset.reject("teacher_error", seed.id, detail=str(exc))
                continue
            stats.answered += 1

            outcome = _handle(dataset, recipe, seed, reply, deduper, opts, stats)
            if outcome is None:                            # kept
                break
            if outcome not in ("unparseable", "identical_pair"):
                # Only a format failure is worth re-asking. A sample whose tests failed or
                # that duplicates an existing one is a fact about this cell, not a fluke.
                break
        if on_progress:
            on_progress(stats)

    else:
        # The loop ran out of seeds. `plan` is cut to `max_asks`, so that is almost always
        # the ask budget rather than the grid — saying "seed grid exhausted" when 480 cells
        # are untouched sends the reader to fix the wrong thing.
        reason = reason or ("done" if stats.kept >= opts.n else "ask budget spent")

    stats.stopped = reason or "done"
    dataset.close(stats.stopped)
    if on_progress:
        on_progress(stats)
    return stats


def _handle(dataset: Dataset, recipe: Recipe, seed, reply, deduper: filters.Deduper,
            opts: GenerateOptions, stats: Stats) -> str | None:
    """One reply, all the way down the funnel. Returns the reject reason, or None if kept."""
    try:
        sample = recipe.parse(reply.text, seed)
    except SynthError as exc:
        why = str(exc) if str(exc) in filters.REJECT_REASONS else "unparseable"
        stats.drop(why)
        dataset.reject(why, seed.id, text=reply.text)
        return why
    stats.parsed += 1
    dataset.count_parsed()

    # -- validity ------------------------------------------------------------------------
    if sample["kind"] == "python":
        why = filters.check_text(sample["problem"], opts.min_chars, opts.max_chars) \
            or filters.check_code(sample["solution"], sample["tests"],
                                  sample.get("entry_point"))
    elif sample["kind"] == "chat":
        why = filters.check_text(sample["prompt"], 10, 1000) \
            or filters.check_text(sample["answer"], opts.min_chars, opts.max_chars)
    else:
        why = filters.check_text(sample["prompt"], 10, 1000) \
            or filters.check_text(sample["chosen"], opts.min_chars, opts.max_chars) \
            or filters.check_text(sample["rejected"], opts.min_chars, opts.max_chars)
    if why:
        stats.drop(why)
        dataset.reject(why, seed.id, text=reply.text)
        return why

    # -- verification (python only) ------------------------------------------------------
    if recipe.verified and opts.verify:
        verdict = verify_code(sample["solution"], sample["tests"], sample["entry_point"],
                              timeout_s=opts.sandbox_timeout_s,
                              memory_mb=opts.sandbox_memory_mb, mutate=opts.mutate)
        sample["verify"] = verdict.as_dict()
        sample["verified"] = verdict.ok
        if not verdict.ok:
            stats.drop(verdict.status)
            dataset.reject(verdict.status, seed.id, detail=verdict.detail,
                           text=reply.text)
            return verdict.status
    elif recipe.verified:
        sample["verified"] = False

    # -- uniqueness ----------------------------------------------------------------------
    key = recipe.dedup_key(sample)
    why, similarity = deduper.check(key)
    if why:
        stats.drop(why)
        dataset.reject(why, seed.id, detail=f"jaccard {similarity:.2f}", text=key[:400])
        return why
    deduper.add(key)

    sample |= {
        "id": seed.id,
        "seed": seed.as_dict(),
        "teacher": reply.model,
        "template_version": TEMPLATE_VERSION,
        "gen_s": round(reply.duration_s, 2),
        "created": time.time(),
    }
    dataset.append(sample)
    dataset.save()
    stats.kept += 1
    return None
