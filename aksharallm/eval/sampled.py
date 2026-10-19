"""HumanEval with sampling: n completions per problem, scored as unbiased pass@k.

Greedy `humaneval` gives one completion per problem, so a model's score is a count of
problems it solved on its single best guess. At 300M that count is in single digits, and
two checkpoints at 5/164 and 8/164 cannot be told apart (McNemar p 0.375 on exactly that
pair, docs/13). Sampling `n` completions per problem turns each problem's verdict from a
bit into a fraction `c/n`, which is what gives the comparison its power.

**The estimator.** For `n` samples of which `c` pass, the probability that at least one of
`k` drawn without replacement passes is `1 - C(n-c, k) / C(n, k)` (Chen et al., 2021). It
is unbiased; the obvious alternative, `1 - (1 - c/n)**k`, is biased upwards for small n.
pass@1 under this estimator is just `c/n` -- the *sampled* pass rate, which is not the
greedy suite's pass@1 and must not be put in the same column. That is why this is a
separate suite (`humaneval-pass10`) and not a flag on `humaneval`: gotcha 4 in docs/13, a
change of decoding is a change of benchmark.

**Why it is batched.** 164 problems x 20 samples is 3,280 generations. One at a time at
~50 tok/s that is hours per checkpoint; through the server's `BatchEngine` (paged KV,
continuous batching) a whole chunk of problems decodes together. Each sequence is cancelled
the moment its text reaches a stop string, the batched equivalent of `generate_until`, so a
base model's "and now another function" costs nothing and frees its slot for the queue.

**Determinism.** Sampling is seeded (`seed`, recorded in the result), but a batched decode
is not bit-reproducible across batch compositions, so two runs can differ by a sample or
two. With n=20 the estimator's own standard error is what to read, and it is reported.

Read with: docs/13-eval.md -- the chapter this implements; it ends with the order to read these files in.
"""
from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor

import torch

STOPS = ["\ndef ", "\nclass ", "\nprint(", "\nif __name__"]


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k for one problem: n samples, c of them correct."""
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def _first_stop(text: str) -> int:
    hits = [text.find(s) for s in STOPS]
    hits = [h for h in hits if h >= 0]
    return min(hits) if hits else -1


def sample_completions(model, tok, prompts: list[str], n: int, *, temperature: float,
                       top_p: float, max_new: int, device: str, seed: int,
                       chunk: int = 8, max_batch: int = 64, progress=None) -> list[list[str]]:
    """`n` completions for every prompt, each cut at its first stop string."""
    from ..serve.batch import Request
    from ..train.grpo import build_sampler

    torch.manual_seed(seed)
    engine = build_sampler(model, device, max_batch=max_batch, max_new=max_new)
    out: list[list[str]] = [[] for _ in prompts]
    done = 0
    for lo in range(0, len(prompts), chunk):
        idx = list(range(lo, min(lo + chunk, len(prompts))))
        if engine is None:                                   # CPU: one at a time
            from ..infer.generate import stream_generate
            for p in idx:
                ids = tok.encode(prompts[p], bos=True)
                for _ in range(n):
                    toks = []
                    for t in stream_generate(model, ids, max_new_tokens=max_new,
                                             temperature=temperature, top_k=None, top_p=top_p,
                                             eos_id=tok.eos_id, device=device):
                        if t == tok.eos_id:
                            break
                        toks.append(t)
                        if len(toks) % 16 == 0 and _first_stop(tok.decode(toks)) >= 0:
                            break
                    out[p].append(_cut(tok.decode(toks)))
        else:
            owner, toks = {}, {}
            for p in idx:
                ids = tok.encode(prompts[p], bos=True)
                for _ in range(n):
                    req = Request(prompt_ids=list(ids), max_new_tokens=max_new,
                                  temperature=temperature, top_k=None, top_p=top_p,
                                  eos_id=tok.eos_id)
                    engine.submit(req)
                    owner[req.id], toks[req.id] = p, []
            with torch.no_grad():
                for seq_id, t in engine.run(max_steps=10 ** 7):
                    got = toks[seq_id]
                    if t == tok.eos_id:
                        continue
                    got.append(t)
                    # Decoding every token would dominate; every 16 is plenty, because the
                    # text is cut at the stop afterwards anyway -- this only saves compute.
                    if len(got) % 16 == 0 and _first_stop(tok.decode(got)) >= 0:
                        engine.cancel(seq_id)
            for seq_id, p in owner.items():
                out[p].append(_cut(tok.decode(toks[seq_id])))
        done += len(idx)
        if progress:
            progress(done, len(prompts), "humaneval:sample")
    return out


def _cut(text: str) -> str:
    at = _first_stop(text)
    return text if at < 0 else text[:at]


def score(items, completions: list[list[str]], ks=(1, 10), workers: int = 8,
          progress=None) -> dict:
    """Run every completion in the sandbox and fold the verdicts into pass@k."""
    from ..infer import sandbox
    from ..infer.tasks import extract_code

    def program(item, text):
        body = extract_code(text, item.prompt, item.entry_point)
        return (body if body.lstrip().startswith("def ")
                else item.prompt.rstrip("\n") + "\n" + body)

    jobs = [(i, program(item, text)) for i, (item, comps) in enumerate(zip(items, completions))
            for text in comps]
    results: list[list] = [[] for _ in items]
    with ThreadPoolExecutor(workers) as pool:
        verdicts = pool.map(lambda j: sandbox.run_program(
            f"{j[1]}\n\n{items[j[0]].tests}", timeout_s=10.0), jobs)
        for done, ((i, prog), v) in enumerate(zip(jobs, verdicts), 1):
            results[i].append((v.ok, v.status, prog))
            if progress and done % 100 == 0:
                progress(done, len(jobs), "humaneval:run")

    records, failures = [], {}
    per_k = {k: [] for k in ks}
    for item, res in zip(items, results):
        n, c = len(res), sum(ok for ok, _, _ in res)
        for k in ks:
            per_k[k].append(pass_at_k(n, c, k) if n >= k else float("nan"))
        for ok, status, _ in res:
            if not ok:
                failures[status] = failures.get(status, 0) + 1
        first_pass = next((prog for ok, _, prog in res if ok), None)
        records.append({"id": item.id, "n": n, "correct": c, "passed": c > 0,
                        "code": (first_pass or res[0][2])[:2000] if res else ""})

    out = {"n": len(items), "samples": len(results[0]) if results else 0,
           "solved_any": sum(r["correct"] > 0 for r in records),
           "failures": dict(sorted(failures.items())), "items": records}
    for k, vals in per_k.items():
        vals = [v for v in vals if v == v]
        mean = sum(vals) / max(1, len(vals))
        var = sum((v - mean) ** 2 for v in vals) / max(1, len(vals) - 1)
        out[f"pass@{k}"] = mean
        out[f"pass@{k}_stderr"] = math.sqrt(var / max(1, len(vals)))
    out["score"] = out[f"pass@{max(ks)}"]
    return out
