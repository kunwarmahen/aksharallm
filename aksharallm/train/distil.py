"""Knowledge distillation: a small student trained on a big teacher's predictions.

The loss is two terms, mixed by `distil.alpha`:

    soft = T^2 * KL( softmax(teacher / T) || softmax(student / T) )    per token, summed over the vocab
    hard = cross_entropy(student, next_token)                           ordinary pretraining
    loss = alpha * soft + (1 - alpha) * hard

The hard term says which token came next. The soft term says what else was plausible, and by
how much. "Paris" with 0.6 and "Lyon" with 0.2 is information a one-hot label throws away.
That is the whole case for distillation, and it is why a teacher that is still half-trained
is a bad one: its soft targets carry its confusions along with its knowledge.

Three details are the design, and each is something a plausible-looking version gets wrong:

1. **The tokenizers must match, and a mismatch is refused.** The KL compares probabilities
   per token *id*. If id 8,412 is " Paris" to the teacher and "ing" to the student, the loss
   still falls and the student learns nonsense. `check_teacher` compares the tokenizer files
   themselves, not their paths or the vocabulary sizes alone (two different 8,192-token
   vocabularies are the same size).
2. **The T^2.** The gradient of the softened KL scales as 1/T^2, so without the factor,
   raising the temperature would quietly turn the soft term down relative to the hard one,
   and `alpha` would stop meaning what it says. `test_distil.py` checks the gradient's
   scale is the same at T = 1 and T = 4.
3. **The logits are the memory, not the weights.** At (8, 1024, 32768), one fp32 copy of the
   logits is 1 GB, and a KL needs both softmaxes and their logs. So the two softmaxes are
   computed `distil.chunk` positions at a time, each chunk under `torch.utils.checkpoint`,
   so backward recomputes it instead of keeping it. Chunking alone does not save anything:
   autograd would keep every chunk's intermediates for backward anyway.

This is an *objective*, not a trainer (docs/05: "if a third paradigm arrives, add a third
objective, do not fork the loop"). `pretrain.py`'s loop runs it unchanged, so accumulation,
the schedule, resume, the stop file, the session log and the report all work as before.
Its validation number is the student's plain cross-entropy, the same number an ordinary run
reports, so a distilled student and a from-scratch one are compared directly.

Read with: docs/24-distillation.md -- the chapter this implements; it ends with the order to
read these files in. See also docs/05-pretraining.md.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from ..config import Config, ModelConfig
from ..tokenizer.tokenizer import Tokenizer
from ..model.transformer import Transformer

QUANT_CHOICES = (None, "int8", "nf4")


class DistilError(ValueError):
    """A teacher/student pairing that would train without error and learn nothing."""


def _file_digest(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def resolve_teacher(ref: str) -> Path:
    """A path as given, or `<run>/<file>.pt` (or just `<run>`, meaning its ckpt_best.pt)
    under checkpoints/."""
    p = Path(ref)
    if p.is_file():
        return p
    for cand in (Path("checkpoints") / ref, Path("checkpoints") / ref / "ckpt_best.pt"):
        if cand.is_file():
            return cand
    raise DistilError(f"cannot find the teacher checkpoint {ref!r} (tried it as a path and "
                      f"under checkpoints/)")


def check_teacher(teacher_ckpt: dict, cfg: Config, teacher_path: str | Path = "teacher") -> None:
    """Refuse a teacher whose predictions do not mean what the student's would.

    Raises DistilError. Every check here protects against a run that would go ahead and
    produce a loss curve, which is why none of them is a warning.
    """
    tcfg = teacher_ckpt.get("model_config") or {}
    if not tcfg:
        raise DistilError(f"{teacher_path} has no model_config; is it a model checkpoint?")
    if tcfg.get("vocab_size") != cfg.model.vocab_size:
        raise DistilError(
            f"vocabulary mismatch: the teacher has {tcfg.get('vocab_size'):,} token ids and "
            f"the student {cfg.model.vocab_size:,}. Logit distillation compares probabilities "
            f"per id, so both must use one tokenizer.")
    if not tcfg.get("causal", True) or tcfg.get("tag_classes"):
        raise DistilError(f"{teacher_path} is not a next-token model (a diffusion model or a "
                          f"tagger); its outputs are not next-token distributions.")
    theirs = ((teacher_ckpt.get("config") or {}).get("data") or {}).get("tokenizer")
    if not theirs:
        raise DistilError(f"{teacher_path} does not record its tokenizer, so a match cannot be "
                          f"checked. Refusing rather than guessing.")
    if not Path(theirs).is_file():
        raise DistilError(f"{teacher_path} was trained with {theirs}, which is not on disk, so "
                          f"it cannot be compared with this run's {cfg.data.tokenizer}.")
    # Contents, not paths: the same file in two places is fine, and two different files with
    # the same vocabulary size are not.
    if _file_digest(theirs) != _file_digest(cfg.data.tokenizer):
        raise DistilError(
            f"tokenizer mismatch: the teacher was trained with {theirs} and this run uses "
            f"{cfg.data.tokenizer}, and the files differ. A token id would mean a different "
            f"string to each model.")
    if tcfg.get("max_seq_len", 0) < cfg.train.seq_len:
        raise DistilError(f"the teacher's window is {tcfg.get('max_seq_len')} tokens and this "
                          f"run trains on {cfg.train.seq_len}; it cannot score that far.")


def load_teacher(path: str | Path, quant: str | None, device) -> Transformer:
    """The frozen teacher, on `device`, in eval mode, optionally quantized with our own RTN.

    A checkpoint that is already quantized loads as it was saved. bf16 on a GPU, fp32
    otherwise, exactly as inference loads a model."""
    from ..quant.convert import (_cast_float_only, build_from_checkpoint,
                                 is_quantized_checkpoint, quantize_model)
    from ..quant.qtensor import QuantScheme

    if quant not in QUANT_CHOICES:
        raise DistilError(f"distil.teacher_quant must be one of {QUANT_CHOICES}, got {quant!r}")
    device = torch.device(device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if is_quantized_checkpoint(ckpt):
        model = build_from_checkpoint(ckpt, device=str(device), dtype=dtype)
    else:
        model = Transformer(ModelConfig(**ckpt["model_config"]))
        model.load_state_dict(ckpt["model"])
        model = model.to(device)
        if quant:
            scheme = (QuantScheme(bits=8, group_size=-1) if quant == "int8" else
                      QuantScheme(bits=4, group_size=64, dtype="nf4", double_quant=True))
            quantize_model(model, scheme)
        _cast_float_only(model, dtype)
    del ckpt
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def _chunk_losses(s: torch.Tensor, t: torch.Tensor, y: torch.Tensor, temperature: float):
    """Summed (soft, hard) loss over one chunk of positions. s, t: (B, c, V); y: (B, c).

    Everything in fp32: the KL subtracts two log-probabilities that are often nearly equal,
    which is where bf16's eight mantissa bits lose the signal entirely."""
    s32 = s.float()
    keep = y != -100
    log_q = F.log_softmax(s32 / temperature, dim=-1)
    log_p = F.log_softmax(t.float() / temperature, dim=-1)
    # KL(p || q) = sum p (log p - log q), with p the teacher. `kl_div` wants log q and log p.
    kl = F.kl_div(log_q, log_p, log_target=True, reduction="none").sum(-1)   # (B, c)
    soft = (kl * keep).sum() * temperature ** 2
    hard = F.cross_entropy(s32.reshape(-1, s32.size(-1)), y.reshape(-1),
                           ignore_index=-100, reduction="sum")
    return soft, hard


def distil_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor,
                targets: torch.Tensor, temperature: float = 2.0, alpha: float = 0.5,
                chunk: int | None = 128):
    """The mixed loss and its two parts, each a mean over the counted tokens.

    Returns (loss, soft, hard). `chunk=None` computes it in one piece, which the tests use
    as the reference the chunked path must equal."""
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    teacher_logits = teacher_logits.detach()
    n = (targets != -100).sum().clamp(min=1)
    T = student_logits.size(1)
    step = T if not chunk or chunk >= T else chunk
    soft = hard = student_logits.new_zeros((), dtype=torch.float32)
    for i in range(0, T, step):
        s, t, y = (student_logits[:, i:i + step], teacher_logits[:, i:i + step],
                   targets[:, i:i + step])
        if step < T and torch.is_grad_enabled() and s.requires_grad:
            cs, ch = checkpoint(_chunk_losses, s, t, y, temperature, use_reentrant=False)
        else:
            cs, ch = _chunk_losses(s, t, y, temperature)
        soft, hard = soft + cs, hard + ch
    soft, hard = soft / n, hard / n
    return alpha * soft + (1 - alpha) * hard, soft, hard


class DistilObjective:
    """Next-token prediction with a teacher's distribution as a second target."""

    name = "distillation"
    metric = "cross-entropy"
    #: The validation number is the student's plain cross-entropy, so a distilled run and a
    #: from-scratch run of the same student go on one axis.
    comparable_to_ar = True

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.d = cfg.distil
        if not 0.0 <= self.d.alpha <= 1.0:
            raise DistilError(f"distil.alpha must be in [0, 1], got {self.d.alpha}")
        if self.d.temperature <= 0:
            raise DistilError(f"distil.temperature must be > 0, got {self.d.temperature}")
        if cfg.model.n_experts:
            # The router's balancing loss is added inside `Transformer.forward` only on the
            # `targets` path, which this objective does not take. Without it an MoE student
            # collapses silently (docs/15), so refuse rather than train one.
            raise DistilError("an MoE student is not supported: its load-balancing loss would "
                              "be skipped and the router could collapse unseen.")
        self.path = resolve_teacher(self.d.teacher)
        self.teacher: Transformer | None = None
        self.teacher_params: int | None = None
        self._stats: dict = {}

    def check(self, tok: Tokenizer) -> None:
        if tok.vocab_size != self.cfg.model.vocab_size:
            raise ValueError(f"config says vocab_size={self.cfg.model.vocab_size} but the "
                             f"tokenizer has {tok.vocab_size}")
        # mmap: reads the small fields without pulling the weights into memory yet.
        ckpt = torch.load(self.path, map_location="cpu", weights_only=False, mmap=True)
        check_teacher(ckpt, self.cfg, self.path)
        self.teacher_step = ckpt.get("step")
        del ckpt

    def describe(self) -> str:
        q = f", teacher in {self.d.teacher_quant}" if self.d.teacher_quant else ""
        return (f"distillation from {self.path} (step {self.teacher_step}), "
                f"T={self.d.temperature:g}, alpha={self.d.alpha:g}{q}")

    def _ensure_teacher(self, device) -> Transformer:
        # Loaded at the first batch rather than in `check`, because the device is only known
        # there -- under DDP it is a different card on every rank.
        if self.teacher is None:
            self.teacher = load_teacher(self.path, self.d.teacher_quant, device)
            self.teacher_params = sum(p.numel() for p in self.teacher.parameters())
            if self.cfg.train.compile and torch.device(device).type == "cuda" \
                    and not self.d.teacher_quant:
                # The teacher runs a forward pass on every batch, so it is half the step.
                # Compiled like the student; a quantized teacher is left alone, because
                # QuantLinear's dequantize is not something dynamo should be trusted with.
                self.teacher = torch.compile(self.teacher)
        return self.teacher

    def batch(self, dataset, batch_size: int):
        return dataset.get_batch(batch_size)

    def loss(self, model, batch):
        x, y = batch
        teacher = self._ensure_teacher(x.device)
        with torch.no_grad():
            t_logits, _ = teacher(x, full_logits=True)
        s_logits, _ = model(x, full_logits=True)
        loss, soft, hard = distil_loss(s_logits, t_logits, y, self.d.temperature,
                                       self.d.alpha, self.d.chunk)
        self._stats = {"kd": soft.item(), "ce": hard.item()}
        return loss

    def evaluate(self, model, dataset, batch_size: int, n_batches: int, ctx) -> float:
        from .pretrain import evaluate
        return evaluate(model, dataset, batch_size, n_batches, ctx)

    def sample(self, model, tok: Tokenizer, prompt: str, device: str) -> str:
        from .pretrain import sample_text
        return sample_text(model, tok, prompt, 80, device)

    def stats(self) -> dict:
        """`ce` is the hard term alone -- the number a from-scratch run would log as its loss,
        so it is the one to compare across the pair. `kd` is the soft term, already times T^2."""
        return dict(self._stats)

    def log_suffix(self) -> str:
        s = self._stats
        return f" | kd {s['kd']:.3f} | ce {s['ce']:.3f}" if s else ""
