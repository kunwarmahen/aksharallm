"""What the recogniser is allowed to write: blank, space, a–z, apostrophe. Twenty-nine symbols.

**A fixed alphabet, not one built from the corpus.** `audio/text.py`'s `CharTokenizer` derives
its alphabet from whatever corpus it is handed, which is right for TTS on one corpus. For a
recogniser it is wrong: LJSpeech has semicolons and LibriSpeech does not, so two runs would get
different ids for the same letter and a checkpoint could not be fine-tuned from one onto the
other — or onto the user's own corrections, which is day-two problem 5. One alphabet, chosen
once, written down here.

**Why no punctuation or capitals.** LibriSpeech transcripts have neither, and that is
typical of ASR corpora: punctuation is not *audible* in a way a frame classifier can learn. It
is a property of the sentence, which is a language model's job — in Phase 7 that is the
cleanup pass by our own chat model. Keeping the recogniser to sounds and the language model to
sentences is the division of labour every dictation system makes.

`normalise` is applied to every transcript, training and reference alike, and WER is computed
on normalised text on both sides. A WER that counts "Mr." against "mister" is measuring the
normaliser.

Read with: docs/23-speech-recognition.md -- the chapter this implements; it ends with the
order to read these files in.
"""

from __future__ import annotations

import re
import unicodedata

BLANK = 0
ALPHABET = " abcdefghijklmnopqrstuvwxyz'"
STOI = {c: i + 1 for i, c in enumerate(ALPHABET)}
ITOS = {i + 1: c for i, c in enumerate(ALPHABET)}
VOCAB_SIZE = len(ALPHABET) + 1

_KEEP = re.compile(r"[^a-z' ]+")
_SPACES = re.compile(r"\s+")


def normalise(text: str) -> str:
    """Lower-case, strip accents, drop everything outside the alphabet, squeeze spaces.

    Hyphens and other punctuation become spaces rather than vanishing, so "well-known" is two
    words (as LibriSpeech writes it) rather than the non-word "wellknown". Apostrophes are kept
    because "it's" and "its" are different words and both are common.
    """
    t = unicodedata.normalize("NFKD", text)
    t = "".join(c for c in t if not unicodedata.combining(c)).lower()
    t = t.replace("’", "'").replace("‘", "'")
    t = _KEEP.sub(" ", t)
    # A quote used as a quotation mark rather than inside a word is not a letter.
    t = re.sub(r"(?<![a-z])'|'(?![a-z])", " ", t)
    return _SPACES.sub(" ", t).strip()


def encode(text: str) -> list[int]:
    """Normalised text -> ids. Raises on anything outside the alphabet: `normalise` first."""
    try:
        return [STOI[c] for c in text]
    except KeyError as e:
        raise ValueError(f"{e.args[0]!r} is not in the ASR alphabet; call normalise() first") from None


def decode(ids) -> str:
    return "".join(ITOS.get(int(i), "") for i in ids)
