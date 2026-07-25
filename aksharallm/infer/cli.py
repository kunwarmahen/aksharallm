"""Command-line interface for talking to a trained checkpoint.

Two modes:
  complete  - raw text continuation (what a *base* model does)
  chat      - multi-turn conversation using the ChatML template (needs an SFT'd model)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from ..config import ModelConfig
from ..model.transformer import Transformer
from ..tokenizer.tokenizer import Tokenizer
from .generate import generate


def load_model(ckpt_path: str, device: str = "cuda") -> tuple[Transformer, dict]:
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


def main():
    ap = argparse.ArgumentParser(description="Generate text from a aksharallm checkpoint.")
    ap.add_argument("checkpoint")
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--mode", choices=["complete", "chat"], default="complete")
    ap.add_argument("--prompt", default=None, help="one-shot prompt; omit for interactive")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--repetition-penalty", type=float, default=1.0)
    ap.add_argument("--system", default="You are a helpful assistant.")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    model, ckpt = load_model(args.checkpoint, args.device)
    tok = Tokenizer(resolve_tokenizer(ckpt, args.tokenizer))
    print(f"loaded {args.checkpoint}  ({model.num_params()/1e6:.1f}M params, "
          f"step {ckpt.get('step', '?')}, val {ckpt.get('best_val', float('nan')):.4f})",
          file=sys.stderr)

    gen_kw = dict(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        device=args.device,
    )

    class Streamer:
        """Prints tokens as they arrive, without mangling multi-byte characters.

        A single BPE token is often only *part* of a UTF-8 character (any emoji or
        accented letter spans several). So we decode the whole buffer each time and print
        only the delta rather than decoding tokens individually.

        That alone isn't enough: decoding a buffer that ends mid-character yields a
        trailing U+FFFD replacement char. If we printed it, the next decode would have the
        real character in that position but we'd already have emitted garbage and moved
        the cursor past it. So we hold back a trailing U+FFFD and let the next token
        complete it.
        """

        def __init__(self, stop_id):
            self.stop_id = stop_id
            self.ids: list[int] = []
            self.printed = ""

        def __call__(self, t):
            if t == self.stop_id:
                return
            self.ids.append(t)
            text = tok.decode(self.ids)
            if text.endswith("�"):
                return  # incomplete character -- wait for the rest of its bytes
            sys.stdout.write(text[len(self.printed):])
            sys.stdout.flush()
            self.printed = text

        @property
        def text(self) -> str:
            return tok.decode(self.ids)

    def stream(prompt_ids, stop_id):
        s = Streamer(stop_id)
        generate(model, prompt_ids, eos_id=stop_id, stream_cb=s, **gen_kw)
        # Flush anything held back (e.g. a genuinely invalid trailing byte).
        final = s.text
        if len(final) > len(s.printed):
            sys.stdout.write(final[len(s.printed):])
        print()
        return s

    if args.mode == "complete":
        if args.prompt is not None:
            stream(tok.encode(args.prompt, bos=True), tok.eos_id)
            return
        print("Base-model completion. Type a prompt (Ctrl-C to quit).", file=sys.stderr)
        while True:
            try:
                p = input("\n> ")
            except (EOFError, KeyboardInterrupt):
                print()
                return
            if p.strip():
                stream(tok.encode(p, bos=True), tok.eos_id)
        return

    # chat
    messages = [{"role": "system", "content": args.system}] if args.system else []
    if args.prompt is not None:
        messages.append({"role": "user", "content": args.prompt})
        ids, _ = tok.render_chat(messages, add_generation_prompt=True)
        stream(ids, tok.im_end_id)
        return

    print("Chat mode. Ctrl-C to quit, /reset to clear history.", file=sys.stderr)
    while True:
        try:
            user = input("\nyou> ")
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not user.strip():
            continue
        if user.strip() == "/reset":
            messages = [{"role": "system", "content": args.system}] if args.system else []
            print("(history cleared)", file=sys.stderr)
            continue

        messages.append({"role": "user", "content": user})
        ids, _ = tok.render_chat(messages, add_generation_prompt=True)
        # Keep the newest context if history overflows the window.
        if len(ids) > model.cfg.max_seq_len - args.max_new_tokens:
            ids = ids[-(model.cfg.max_seq_len - args.max_new_tokens):]

        print("bot> ", end="", flush=True)
        s = stream(ids, tok.im_end_id)
        messages.append({"role": "assistant", "content": s.text})


if __name__ == "__main__":
    main()
