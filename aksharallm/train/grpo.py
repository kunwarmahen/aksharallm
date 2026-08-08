r"""GRPO — Group Relative Policy Optimization, from scratch.

The one idea. SFT and DPO both need a *dataset of good answers* (or good/bad pairs). But for
code there is something better than a human's opinion of an answer: **run it.** If the model
writes `is_palindrome` and the asserts pass, that is a reward of 1, no matter what the answer
looks like. GRPO turns that reward into a gradient.

How it works, in five steps per iteration:

    1. take a prompt (a coding task)
    2. sample a GROUP of G completions from the current policy
    3. score each with a reward (here: does the code pass its tests?)
    4. an answer's ADVANTAGE is how much better it did than its group-mates:
           A_i = (r_i - mean(r_group)) / std(r_group)
    5. push the policy up on the above-average answers and down on the below-average ones,
       with a KL leash to a frozen reference so it can't wander off and forget English.

Why "group relative": ordinary policy gradient needs a *value network* to say "was this
better than expected?". GRPO throws that away and uses the group's own mean as the baseline
— G samples of the same prompt are directly comparable, so their mean is exactly the
"expected" the baseline wanted. One model instead of two, and the baseline is unbiased by
construction.

    L = - 1/Σm · Σ_t m_t [ min(ρ_t A, clip(ρ_t,1±ε) A) - β·KL_t ]
        ρ_t = π(o_t)/π_old(o_t)          (=1 on the first inner step; the clip guards >1)
        KL_t = exp(r_t - p_t) - (r_t - p_t) - 1   (k3: unbiased, always ≥ 0)

The reward is pluggable (`RewardFn`): the real one runs the model's code in the sandbox; a
toy substring reward lets us prove the loop optimises *anything* on a model that can't yet
code — see tests.

Read with: docs/06-posttraining.md -- the chapter this implements; it ends with the order to
read these files in.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Protocol

import numpy as np
import torch
import torch.nn.functional as F

from ..config import ModelConfig
from ..infer.generate import generate
from ..model.transformer import Transformer
from ..tokenizer.tokenizer import Tokenizer
from . import report, resume, stopfile
from .pretrain import fmt_dur, human, save_checkpoint
from .sft import _rebuild_cfg


# ---- the two pure functions (unit-tested like dpo_loss) ----------------------------

def group_advantages(rewards: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """(P, G) rewards -> (P, G) advantages, normalised within each group (row).

    A group whose completions all earned the same reward yields all-zero advantages: there
    is nothing to learn from a prompt the policy already answers uniformly (all right, or
    all wrong). That is correct and important -- it's why GRPO spends its gradient on the
    prompts that are *on the boundary* of the policy's ability.
    """
    mean = rewards.mean(dim=1, keepdim=True)
    std = rewards.std(dim=1, keepdim=True)
    return (rewards - mean) / (std + eps)


def grpo_loss(new_lp, old_lp, ref_lp, adv, mask, beta: float = 0.04, clip_eps: float = 0.2):
    """Per-token GRPO surrogate. All of new_lp/old_lp/ref_lp/mask are (B, T); adv is (B,).

    Returns (loss, metrics). `mask` is 1 on completion tokens (we never train on the prompt).
    """
    adv = adv[:, None]  # broadcast one advantage across a completion's tokens
    ratio = torch.exp(new_lp - old_lp)  # = 1 on the first inner step; carries the gradient
    surr = torch.min(ratio * adv, torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv)

    # k3 KL estimator: unbiased and guaranteed non-negative, unlike (p-r) which can go
    # negative on a sample and yield a KL "reward".
    diff = ref_lp - new_lp
    kl = torch.exp(diff) - diff - 1.0

    per_tok = -(surr - beta * kl)
    denom = mask.sum().clamp(min=1)
    loss = (per_tok * mask).sum() / denom
    metrics = {
        "kl": (kl * mask).sum().item() / denom.item(),
        "ratio": (ratio * mask).sum().item() / denom.item(),
    }
    return loss, metrics


def token_logprobs(model, seq, ctx) -> torch.Tensor:
    """(B, L) token ids -> (B, L-1) log p(token_{t+1} | tokens_{<=t}) under `model`."""
    x, y = seq[:, :-1], seq[:, 1:]
    with ctx:
        logits, _ = model(x, targets=y)
    lp = F.log_softmax(logits.float(), dim=-1)
    return lp.gather(-1, y.unsqueeze(-1)).squeeze(-1)


# ---- rewards -----------------------------------------------------------------------

class RewardFn(Protocol):
    def __call__(self, prompt: str, completion: str) -> float: ...


class SubstringReward:
    """Toy, deterministic reward: 1.0 if `needle` appears in the completion, else 0.0.

    Exists so the *machinery* can be proven on a model that cannot yet code -- a TinyStories
    model really can raise the rate at which it emits a target word, which is exactly the
    end-to-end signal ("reward goes up, KL stays bounded") that a code reward can't give
    until the base model is trained. Not used for real training.
    """

    def __init__(self, needle: str):
        self.needle = needle

    def __call__(self, prompt: str, completion: str) -> float:
        return 1.0 if self.needle in completion else 0.0


class CodeReward:
    """Real reward: run the completion's code against a task's asserts in the sandbox.

    Shaped so a small model gets *some* gradient before it can fully solve anything:
        pass all tests            -> 1.0
        runs but asserts fail     -> 0.1   (it produced a real function, just wrong)
        error / timeout / syntax  -> 0.0
    """

    def __init__(self, task, chat: bool = False, enabled: bool = True):
        self.task = task
        self.chat = chat
        self.enabled = enabled

    def __call__(self, prompt: str, completion: str) -> float:
        from ..infer.sandbox import run_task

        r = run_task(self.task, completion, chat=self.chat, enabled=self.enabled)
        if r.ok:
            return 1.0
        return 0.1 if r.status == "fail" else 0.0


# ---- sampling ----------------------------------------------------------------------

def sample_group(model, prompt_ids, group_size, max_new, temperature, top_k, top_p,
                 eos_id, device):
    """Sample `group_size` completions for one prompt. Returns list of (full_ids, gen_ids)."""
    out = []
    for _ in range(group_size):
        full = generate(model, prompt_ids, max_new_tokens=max_new, temperature=temperature,
                        top_k=top_k, top_p=top_p, eos_id=eos_id, device=device)
        out.append((full, full[len(prompt_ids):]))
    return out


def build_batch(groups, pad_id, device):
    """groups: list over prompts of list over G of (full_ids, gen_ids).

    Returns (seq, mask) padded to the longest sequence. mask is aligned to the *targets*
    (seq[:,1:]): 1 exactly on the completion tokens, 0 on prompt and padding.
    """
    flat = [(full, len(full) - len(gen)) for grp in groups for (full, gen) in grp]
    maxlen = max(len(full) for full, _ in flat)
    B = len(flat)
    seq = torch.full((B, maxlen), pad_id, dtype=torch.long)
    mask = torch.zeros((B, maxlen - 1), dtype=torch.float32)
    for i, (full, plen) in enumerate(flat):
        seq[i, : len(full)] = torch.tensor(full, dtype=torch.long)
        # target position t corresponds to seq[t+1]; completion targets are t+1 >= plen.
        for t in range(maxlen - 1):
            if plen <= t + 1 < len(full):
                mask[i, t] = 1.0
    return seq.to(device), mask.to(device)


# ---- training loop -----------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="GRPO: reinforcement learning on a verifiable reward.")
    ap.add_argument("--init", required=True, help="checkpoint to start from (base or SFT)")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--reward", choices=["code", "substring"], default="code")
    ap.add_argument("--needle", default=" dragon", help="substring reward target (toy)")
    ap.add_argument("--chat", action="store_true", help="prompt code tasks in chat form")
    ap.add_argument("--group-size", type=int, default=8, help="G: completions per prompt")
    ap.add_argument("--prompts-per-step", type=int, default=4)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=1.0, help="exploration; keep >=0.7")
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--beta", type=float, default=0.04, help="KL leash to the reference")
    ap.add_argument("--clip-eps", type=float, default=0.2)
    ap.add_argument("--lr", type=float, default=1e-6, help="RL wants a very low LR")
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--log-every", type=int, default=1)
    ap.add_argument("--ckpt-every", type=int, default=50)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--stop-file", default=None,
                    help="poll this file for a stop request (default <out-dir>/STOP). "
                         "Empty = stop now, a number = stop at that step, @<epoch> = stop "
                         "at that wall-clock time.")
    ap.add_argument("--stop-in", default=None, metavar="DURATION",
                    help="train for this long, then save and exit: 30m / 90s / 2h / 1h30m, "
                         "or a bare number read as minutes.")
    ap.add_argument("--resume", default=None, metavar="PATH|auto",
                    help="continue a stopped run: 'auto' picks <out-dir>/grpo_last.pt if it "
                         "exists, so the same command starts and resumes. Restores the "
                         "POLICY, its optimizer, the step, the best reward and the prompt "
                         "sampler. The reference model always stays --init.")
    args = ap.parse_args()

    sys.stdout.reconfigure(line_buffering=True)
    torch.backends.cuda.matmul.allow_tf32 = True
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(args.init, map_location=args.device, weights_only=False)
    mcfg = ModelConfig(**ckpt["model_config"])
    mcfg.dropout = 0.0

    policy = Transformer(mcfg).to(args.device)
    policy.load_state_dict(ckpt["model"])
    ref = Transformer(mcfg).to(args.device)     # frozen reference, as in DPO
    ref.load_state_dict(ckpt["model"])
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)

    tok = Tokenizer(args.tokenizer)

    # Prompts + per-prompt reward functions.
    prompts: list[tuple[list[int], str, RewardFn]] = []
    if args.reward == "code":
        from ..infer.tasks import CODE_TASKS
        for task in CODE_TASKS:
            text = task.instruction() if args.chat else task.prompt
            ids = tok.encode(text, bos=True)
            prompts.append((ids, text, CodeReward(task, chat=args.chat)))
    else:
        rf = SubstringReward(args.needle)
        for seed in ["Once upon a time", "One day", "The little", "In the forest"]:
            prompts.append((tok.encode(seed, bos=True), seed, rf))

    optimizer, _ = policy.configure_optimizers(0.0, args.lr, (0.9, 0.95), args.device)

    # ---- resume -----------------------------------------------------------------------
    # `ref` above was built from `--init` and is deliberately NOT touched here. It is the
    # anchor the KL penalty measures drift from; re-pointing it at the resumed policy would
    # make the run measure itself against itself, so the KL collapses toward zero and the
    # policy is free to wander arbitrarily far from the SFT model — while the logged `kl`
    # still looks small. Nothing about the numbers would tell you.
    resumed = resume.resolve(args.resume, out_dir / "grpo_last.pt")
    start_step, resumed_state = 0, None
    if resumed:
        prev = resume.load(resumed, policy, optimizer, args.device)
        resumed_state = prev.get("grpo_progress") or {}
        start_step = int(resumed_state.get("step", prev.get("step", -1))) + 1
    ctx = (torch.autocast("cuda", dtype=torch.bfloat16)
           if args.device.startswith("cuda") else torch.autocast("cpu", enabled=False))

    print("=" * 78)
    print(f"init       {args.init} ({human(policy.num_params())} params, policy + frozen ref)")
    print(f"reward     {args.reward}" + (f" (needle={args.needle!r})" if args.reward == "substring"
                                         else f" ({len(prompts)} code tasks, chat={args.chat})"))
    print(f"group      G={args.group_size} x {args.prompts_per_step} prompts/step "
          f"= {args.group_size * args.prompts_per_step} samples/step")
    print(f"objective  beta(KL)={args.beta} clip={args.clip_eps} lr={args.lr}")
    print("=" * 78)

    logf = open(out_dir / "grpo_log.jsonl", "a")

    def log_session(event: str, **kw):
        """Same bracketing as sft.py/dpo.py — it is what gives the Sessions table its rows
        and the dashboard a `max_steps` for progress and ETA."""
        rec = {"event": event, "time": time.time(),
               "iso": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
               "run": out_dir.name, **kw}
        logf.write(json.dumps(rec) + "\n")
        logf.flush()

    policy.train()
    rng = np.random.default_rng(0)
    best_reward = -1.0
    if resumed_state:
        # best_reward must carry across sessions. Letting it reset to -1.0 makes the *first*
        # step of the next session "the best so far", overwriting grpo_best.pt with a policy
        # that has just been perturbed — the one failure mode here that destroys work.
        best_reward = float(resumed_state.get("best", -1.0))
        resume.restore_rng(rng, resumed_state.get("rng_state"), "the prompt sampler")
        print(f"resumed from {resumed} at step {start_step}, best reward "
              f"{best_reward:.3f} (reference still {args.init})")
    t0 = time.time()

    # ---- stopping early ------------------------------------------------------------
    # The same file contract pretraining, SFT and DPO obey (aksharallm/train/stopfile.py).
    # GRPO's step is expensive — a whole group is sampled, run in the sandbox and scored —
    # so an unstoppable run is worse here than anywhere: SIGKILL throws away the sampling
    # too, not just the update. Checkpoints are already written on every improvement, so
    # breaking out of the loop leaves the best model behind.
    run_t0 = t0
    prev_log_step = start_step - 1   # so the first window divides by the steps it covered
    stop_file = Path(args.stop_file) if args.stop_file else out_dir / "STOP"
    stop_by = run_t0 + stopfile.parse_duration(args.stop_in) if args.stop_in else None
    stop = {"now": False}

    def _request_stop(signum, frame):
        if stop["now"]:
            print("\n[stop] second signal -- exiting immediately (this step is lost)")
            raise KeyboardInterrupt
        print(f"\n[stop] signal {signum} received -- will save and exit after this step")
        stop["now"] = True

    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)
    if stop_by is not None:
        print(f"budget  {fmt_dur(stop_by - run_t0)} of training, then save and exit")

    # No throughput or MFU here on purpose: most of a GRPO step is *sampling* a group and
    # running it in the sandbox, not the one training update. A tokens/second taken from the
    # update would describe a few percent of the wall-clock and read as a throughput
    # collapse; reward and solve-rate are this run's real headline.
    # `prompts_per_step` alongside `group_size` because the two multiply into the only
    # "work done" figure this stage has: completions sampled and executed. The dashboard
    # derives it as `step x group_size x prompts_per_step` rather than the trainer keeping a
    # running total, so a resumed run needs nothing carried across.
    log_session("session_start", pid=os.getpid(), start_step=start_step,
                max_steps=args.steps, params=policy.num_params(), reward=args.reward,
                group_size=args.group_size, prompts_per_step=args.prompts_per_step,
                lr=args.lr, stage="grpo", resumed=bool(resumed))

    why = None
    announced = None
    last_step = args.steps - 1
    for step in range(start_step, args.steps):
        # 1-2. pick prompts, sample a group each
        idx = rng.choice(len(prompts), size=min(args.prompts_per_step, len(prompts)),
                         replace=False)
        groups, rewards_per_group = [], []
        policy.eval()
        with torch.no_grad():
            for j in idx:
                pids, ptext, rfn = prompts[j]
                grp = sample_group(policy, pids, args.group_size, args.max_new_tokens,
                                   args.temperature, args.top_k, args.top_p, tok.eos_id,
                                   args.device)
                groups.append(grp)
                # 3. reward each completion (decode the generated part only)
                rs = [rfn(ptext, tok.decode(gen)) for _, gen in grp]
                rewards_per_group.append(rs)
        policy.train()

        rewards = torch.tensor(rewards_per_group, dtype=torch.float32)  # (P, G)
        adv = group_advantages(rewards).reshape(-1).to(args.device)     # (P*G,)
        seq, mask = build_batch(groups, tok.pad_id, args.device)

        # old (sampling-time) and reference logprobs -- no grad
        with torch.no_grad():
            old_lp = token_logprobs(policy, seq, ctx)
            ref_lp = token_logprobs(ref, seq, ctx)
        # 4-5. one on-policy update
        new_lp = token_logprobs(policy, seq, ctx)
        loss, m = grpo_loss(new_lp, old_lp, ref_lp, adv, mask, args.beta, args.clip_eps)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(policy.parameters(), args.grad_clip)
        optimizer.step()

        mean_r = rewards.mean().item()
        solve = (rewards >= 1.0).float().mean().item()  # fraction that fully passed

        # Decided before the log line, so the step you stop on always gets logged.
        request = None if stop["now"] else stopfile.read(stop_file)
        if stop["now"]:
            why = "signal"
        else:
            why = stopfile.reached(request, step)
            if why is None and request is not None and request != announced:
                announced = request
                print(f"[stop] {stop_file.name} asks to {request.describe(step)}")
        if why is None and stop_by is not None and time.time() >= stop_by:
            why = f"reached the {fmt_dur(stop_by - run_t0)} budget for this run"

        if step % args.log_every == 0 or why:
            dt = time.time() - t0
            t0 = time.time()
            s_per_step = dt / max(1, step - prev_log_step)   # measured, not assumed
            prev_log_step = step
            up = time.time() - run_t0
            eta = (args.steps - step) * s_per_step
            print(f"step {step:>5}/{args.steps} | reward {mean_r:.3f} | solved {solve*100:4.0f}% | "
                  f"loss {loss.item():+.4f} | kl {m['kl']:.4f} | gnorm {gnorm:.2f} | "
                  f"{s_per_step:.1f}s/step | up {fmt_dur(up)} | eta {fmt_dur(eta)}")
            # Same rule as dpo.py: the browser only reads this file, so anything the line
            # above prints has to be in it. This one had drifted further — no gnorm, no
            # timing at all — so the dashboard could show neither an ETA nor a rate for the
            # longest-running stage of the three. Throughput/MFU stay out deliberately:
            # most of a GRPO step is sampling and sandbox execution rather than the update,
            # so tokens/sec would describe the wrong thing.
            logf.write(json.dumps({"step": step, "reward": mean_r, "solved": solve,
                                   "loss": loss.item(), "grad_norm": float(gnorm),
                                   "time": time.time(), "s_per_step": s_per_step,
                                   "elapsed": up, "eta_s": eta, **m}) + "\n")
            logf.flush()

        progress = resume.step_progress(step, rng.bit_generator.state, best_reward)
        if mean_r > best_reward:
            best_reward = mean_r
            progress = resume.step_progress(step, rng.bit_generator.state, best_reward)
            save_checkpoint(out_dir / "grpo_best.pt", policy, optimizer,
                            _rebuild_cfg(ckpt, mcfg, args), step, best_reward,
                            extra={"grpo_progress": progress})
        if step > 0 and step % args.ckpt_every == 0:
            save_checkpoint(out_dir / "grpo_last.pt", policy, optimizer,
                            _rebuild_cfg(ckpt, mcfg, args), step, best_reward,
                            extra={"grpo_progress": progress})
        if why:
            print(f"[stop] {why} -- saving at step {step}")
            last_step = step
            break

    save_checkpoint(out_dir / "grpo_last.pt", policy, optimizer,
                    _rebuild_cfg(ckpt, mcfg, args), last_step, best_reward,
                    extra={"grpo_progress": resume.step_progress(
                        last_step, rng.bit_generator.state, best_reward)})
    # Clear the honoured request: a stop file left behind would end the *next* run at step 0.
    if why and stop_file.exists():
        stop_file.unlink(missing_ok=True)
    log_session("session_end", reason=why or "steps", last_step=last_step,
                steps=last_step - start_step + 1, best_reward=best_reward,
                elapsed=time.time() - run_t0)
    print(f"\ndone. best mean reward {best_reward:.3f}"
          f"{f' (stopped early: {why})' if why else ''}. checkpoints in {out_dir}")
    logf.close()
    report.write_quietly(out_dir, log="grpo_log.jsonl")


if __name__ == "__main__":
    main()
