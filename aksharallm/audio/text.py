"""Text for the audio phase — a character tokenizer, and the transcripts that go with it.

The repo already has a real BPE tokenizer (`tokenizer/`), trained on 10 B tokens of prose
and Python. It is the wrong tool here, for one reason: **TTS needs a spelling, not a
meaning.** A BPE merge turns "nation" into one token, which is exactly what a language model
wants and exactly what a text-to-speech model cannot use — the model has to know that the
sequence is n-a-t-i-o-n in order to produce those sounds in that order. Character level is
what small TTS systems use, and at ~70 symbols the embedding table is free.

For ASR the argument is weaker but the conclusion is the same: a character-level output can
spell a word it has never seen, and word error rate is measured on the spelling anyway.

`CharTokenizer` is deliberately built **from the corpus it will be used on**, so its
alphabet is a property of the data rather than a guess, and it is written into the
checkpoint. A model whose text ids mean something different from the ones it was trained on
does not fail — it politely transcribes everything as the wrong letters.

Read with: docs/21-audio.md -- the chapter this implements; it ends with the order to read
these files in. See also docs/03-tokenizer.md.
"""

from __future__ import annotations

import json
from pathlib import Path

#: `[PAD]` is never predicted, `[BOS]`/`[EOS]` bracket every transcript. Three ids before
#: the alphabet starts, exactly like the audio side's `[PAD]`/`[BOS]`.
PAD, BOS, EOS = 0, 1, 2
SPECIALS = 3


class CharTokenizer:
    """One id per character, plus three specials. Built from a corpus, saved with a model."""

    def __init__(self, alphabet: str):
        self.alphabet = alphabet
        self.stoi = {c: i + SPECIALS for i, c in enumerate(alphabet)}
        self.itos = {i + SPECIALS: c for i, c in enumerate(alphabet)}

    @property
    def vocab_size(self) -> int:
        return len(self.alphabet) + SPECIALS

    @classmethod
    def from_texts(cls, texts) -> CharTokenizer:
        """Sorted, so the same corpus always gives the same ids on any machine."""
        return cls("".join(sorted({c for t in texts for c in t})))

    def encode(self, text: str, *, bos: bool = True, eos: bool = True) -> list[int]:
        # Unknown characters are DROPPED rather than mapped to a shared `[UNK]`. A single
        # id standing for "some symbol" is a sound the model would have to invent; leaving
        # it out at least keeps the rest of the word pronounceable.
        ids = [self.stoi[c] for c in text if c in self.stoi]
        return ([BOS] if bos else []) + ids + ([EOS] if eos else [])

    def decode(self, ids) -> str:
        return "".join(self.itos.get(int(i), "") for i in ids)

    def to_dict(self) -> dict:
        return {"alphabet": self.alphabet}

    @classmethod
    def from_dict(cls, d: dict) -> CharTokenizer:
        return cls(d["alphabet"])


def load_transcripts(corpus: str | Path) -> dict[str, str]:
    """`{clip filename: transcript}` for a packed corpus.

    Two sources, both plain files:

    * `transcripts.json` beside `audio.bin` — what `synth_corpus` writes, and what
      `python -m aksharallm.audio transcripts` writes for anything else;
    * LJSpeech's `metadata.csv`, which is `id|raw|normalised` pipe-separated. The third
      field is the one to use: it has the numbers and abbreviations spelled out, which is
      most of the work a real TTS front end does.
    """
    corpus = Path(corpus)
    direct = corpus / "transcripts.json"
    if direct.is_file():
        return json.loads(direct.read_text())

    # A packed corpus usually lives somewhere else entirely — `data/audio/lj` built from
    # `data/audio/ljspeech/LJSpeech-1.1/wavs` — and the transcripts stay with the originals.
    # `pack` records where they came from for exactly this reason.
    candidates = [corpus / "metadata.csv", corpus.parent / "metadata.csv"]
    manifest = corpus / "manifest.json"
    if manifest.is_file():
        src = (json.loads(manifest.read_text()) or {}).get("source_dir")
        if src:
            candidates += [Path(src) / "metadata.csv", Path(src).parent / "metadata.csv"]

    meta = next((p for p in candidates if p.is_file()), None)
    if meta is None:
        raise FileNotFoundError(
            f"no transcripts for {corpus}. Looked for {direct}, and for a metadata.csv in: "
            + ", ".join(str(c.parent) for c in candidates)
        )
    out = {}
    for line in meta.read_text(encoding="utf-8").splitlines():
        parts = line.split("|")
        if len(parts) >= 2:
            out[f"{parts[0]}.wav"] = (parts[2] if len(parts) > 2 else parts[1]).strip()
    return out


def word_error_rate(reference: str, hypothesis: str) -> float:
    """WER — edit distance over *words*, divided by the reference's word count.

    The one number in this phase with a published range everyone can check, which is why ASR
    is the piece worth building for measurement rather than for the demo. It is the ordinary
    Levenshtein distance with words as the alphabet: substitutions, insertions and deletions
    each cost one.

    It can exceed 1.0 — a hypothesis longer than the reference can need more insertions than
    the reference has words — and clamping it would hide exactly the failure (a model that
    will not stop) that produces it.
    """
    r, h = reference.split(), hypothesis.split()
    if not r:
        return 0.0 if not h else float(len(h))
    prev = list(range(len(h) + 1))
    for i, rw in enumerate(r, 1):
        cur = [i]
        for j, hw in enumerate(h, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rw != hw)))
        prev = cur
    return prev[-1] / len(r)


def character_error_rate(reference: str, hypothesis: str) -> float:
    """The same, over characters. Worth reporting beside WER because a character-level model
    that gets one letter wrong loses a whole word to WER, which overstates how bad it is."""
    return word_error_rate(" ".join(reference), " ".join(hypothesis))
