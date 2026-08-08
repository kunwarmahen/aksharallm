"""Byte-level BPE tokenizer: training and a thin runtime wrapper.

Why we train our own instead of borrowing Llama's:
  The embedding matrix is vocab_size x d_model. For a 400M-param model with d_model=1024,
  a 128k vocab would be 131M params -- a third of the model spent on tokens we'll never
  see. A 32k vocab fitted to our own corpus is 33M params and compresses our text better.

Why *byte-level*:
  The base alphabet is the 256 byte values, so there is no such thing as an unknown
  character. Emoji, Cyrillic, control codes -- everything encodes, always round-trips.

Read with: docs/03-tokenizer.md -- the chapter this implements; it ends with the order to read
these files in.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Iterator

from tokenizers import Tokenizer as HFTokenizer
from tokenizers import decoders, models, pre_tokenizers, processors, trainers

# Reserved ids. Kept at the front so they're stable across vocab sizes, and so that
# code can hardcode `bos_id == 0` without surprises.
SPECIAL_TOKENS = [
    "<|endoftext|>",  # 0 - document boundary, also used as BOS and EOS
    "<|pad|>",        # 1 - padding (never contributes to loss)
    "<|im_start|>",   # 2 - chat: begins a role block
    "<|im_end|>",     # 3 - chat: ends a role block
]

# GPT-4's pre-tokenizer regex. This is what decides where BPE is *allowed* to merge.
# Reading it left to right, the alternatives are:
#   contractions ('s 're 'll ...) | a word with at most one leading letter-space
#   | a number of at most 3 digits | punctuation runs | trailing whitespace
# The 3-digit cap is what stops the tokenizer from memorising thousands of specific
# numbers, which is a real cause of bad arithmetic in small models.
SPLIT_PATTERN = (
    r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}++|\p{N}{1,3}"""
    r"""| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
)


def train_bpe(
    corpus: Iterator[str],
    vocab_size: int,
    out_path: str | Path,
    min_frequency: int = 2,
) -> HFTokenizer:
    """Train a byte-level BPE and save it as a single tokenizer.json."""
    tok = HFTokenizer(models.BPE(unk_token=None, byte_fallback=False))
    tok.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Split(pattern=__import__("tokenizers").Regex(SPLIT_PATTERN),
                                 behavior="isolated"),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
        ]
    )
    tok.decoder = decoders.ByteLevel()
    tok.post_processor = processors.ByteLevel(trim_offsets=False)

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),  # all 256 bytes, always
        show_progress=True,
    )
    tok.train_from_iterator(corpus, trainer=trainer)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tok.save(str(out_path))
    return tok


class Tokenizer:
    """Runtime wrapper. Adds the special-token ids and the chat template."""

    def __init__(self, path: str | Path):
        self.path = str(path)
        self._tok = HFTokenizer.from_file(self.path)
        self.bos_id = self._tok.token_to_id("<|endoftext|>")
        self.eos_id = self.bos_id
        self.pad_id = self._tok.token_to_id("<|pad|>")
        self.im_start_id = self._tok.token_to_id("<|im_start|>")
        self.im_end_id = self._tok.token_to_id("<|im_end|>")

    @property
    def vocab_size(self) -> int:
        return self._tok.get_vocab_size()

    def encode(self, text: str, bos: bool = False, eos: bool = False) -> list[int]:
        ids = self._tok.encode(text, add_special_tokens=False).ids
        if bos:
            ids = [self.bos_id] + ids
        if eos:
            ids = ids + [self.eos_id]
        return ids

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        return [e.ids for e in self._tok.encode_batch_fast(texts, add_special_tokens=False)]

    def decode(self, ids: Iterable[int], skip_special: bool = True) -> str:
        return self._tok.decode(list(ids), skip_special_tokens=skip_special)

    # ---- chat template -------------------------------------------------------------
    # ChatML. Rendered as:
    #   <|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\nHi<|im_end|>\n
    # We build it token-by-token rather than with a string template so that SFT can
    # know exactly which token indices belong to the assistant's reply.

    def render_chat(
        self, messages: list[dict], add_generation_prompt: bool = False
    ) -> tuple[list[int], list[int]]:
        """messages: [{"role": "user"|"assistant"|"system", "content": str}, ...]

        Returns (ids, mask) where mask[i] == 1 exactly for tokens the assistant produced
        -- i.e. the tokens we want to train on. Everything else is context.
        """
        ids: list[int] = []
        mask: list[int] = []

        def add(chunk: list[int], trainable: int):
            ids.extend(chunk)
            mask.extend([trainable] * len(chunk))

        for msg in messages:
            role, content = msg["role"], msg["content"]
            header = [self.im_start_id] + self.encode(f"{role}\n")
            body = self.encode(content)
            footer = [self.im_end_id] + self.encode("\n")
            # The header is a prompt regardless of role. The body+footer are trainable
            # only for the assistant: we never want the model rewarded for predicting
            # the *user's* words.
            trainable = 1 if role == "assistant" else 0
            add(header, 0)
            add(body, trainable)
            add(footer, trainable)

        if add_generation_prompt:
            add([self.im_start_id] + self.encode("assistant\n"), 0)

        return ids, mask
