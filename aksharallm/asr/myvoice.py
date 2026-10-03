"""Your own voice, as a test set: read fixed sentences aloud, and get a WER instead of an impression.

Every number in docs/23 so far is on LibriSpeech — audiobooks read by hundreds of strangers.
The person this recogniser is for is not one of them, and "it seems to get me wrong a lot" is
not a measurement. This file makes it one: a short list of sentences written for dictation
(emails, plans, names, questions), each recorded once in your voice and stored as an ordinary
packed corpus, `data/asr/my-voice/` — so `asr eval`, `asr daytwo`, `asr robust` and the
portal's tables all work on it unchanged, and it can be compared with test-clean directly.

```mermaid
flowchart LR
    P["a fixed sentence<br/>(PROMPTS)"] --> R["you read it<br/>(portal or terminal)"]
    R --> C["data/asr/my-voice<br/>audio.bin + transcripts"]
    C --> E["asr eval --corpus data/asr/my-voice"]
    E --> W["WER for YOUR voice<br/>beside test-clean's"]
```

**The prompts are fixed and versioned** (`PROMPTS_VERSION`), for the same reason a benchmark's
prompt format is (gotcha 1 in AGENTS.md): a WER on different sentences is a different test.
They are written in the recogniser's alphabet — numbers as words, no symbols — because the
reference must be what a perfect transcript would say.

**Re-recording a sentence replaces it** rather than adding a second copy, so a bad take (a
cough, a door) can be redone without weighting that sentence twice. The corpus is rewritten in
full on each save; at a few minutes of audio that is milliseconds.

Recordings are personal, so they live under `data/` (never committed) and the portal only
ever serves them back to the person at this machine.

Read with: docs/23-speech-recognition.md -- the chapter this implements; it ends with the
order to read these files in.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from . import vocab

CORPUS = Path("data/asr/my-voice")
PROMPTS_VERSION = 1
PROMPTS = [
    "could you send me the report by friday afternoon",
    "i will be about ten minutes late to the meeting",
    "let us move the call to tuesday at three thirty",
    "thanks for your help yesterday it made a big difference",
    "please remind me to buy milk eggs and bread on the way home",
    "the quarterly numbers look better than we expected",
    "can we talk about the budget before the end of the week",
    "i think we should test this on a bigger data set first",
    "my flight lands at nine in the evening so i will call you then",
    "shaun and priya are joining us for dinner on saturday",
    "the train was delayed again because of the weather",
    "we need to finish the slides before the client arrives",
    "what time does the pharmacy close on sundays",
    "i have attached the signed contract to this email",
    "let me know if you have any questions about the plan",
    "the model trained overnight and the loss is still going down",
    "turn left at the second traffic light and park behind the bank",
    "could you double check the address before you post the letter",
    "she said the package would arrive sometime tomorrow morning",
    "i am working from home today because the office is closed",
    "the doctor moved my appointment to next wednesday",
    "remember to water the plants while we are away",
    "this paragraph needs a clearer opening sentence",
    "we sold three hundred and twenty tickets in the first hour",
    "my sister lives in a small village near the coast",
    "is there any coffee left or should i make a fresh pot",
    "the battery dies after about four hours of heavy use",
    "please forward this to everyone on the design team",
    "i would like a table for two at eight o'clock",
    "honestly i did not expect the results to be this good",
]


def prompts() -> list[dict]:
    out = []
    for i, p in enumerate(PROMPTS):
        assert vocab.normalise(p) == p, f"prompt {i} is not in the recogniser's alphabet: {p!r}"
        out.append({"id": f"me-v{PROMPTS_VERSION}-{i:03d}", "text": p})
    return out


def _load(corpus: Path) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    from ..audio.dataset import Manifest
    man_p = corpus / "manifest.json"
    if not man_p.is_file():
        return {}, {}
    man = Manifest.load(man_p)
    samples = np.fromfile(corpus / "audio.bin", dtype="<i2")
    texts = json.loads((corpus / "transcripts.json").read_text())
    clips = {n: samples[man.offsets[i] : man.offsets[i + 1]].copy() for i, n in enumerate(man.names)}
    return clips, texts


def status(corpus: str | Path = CORPUS) -> dict:
    clips, texts = _load(Path(corpus))
    have = {n.rsplit(".", 1)[0] for n in clips}
    ps = prompts()
    return {"corpus": str(corpus), "version": PROMPTS_VERSION,
            "prompts": [{**p, "recorded": p["id"] in have,
                         "seconds": round(len(clips.get(p["id"] + ".wav", [])) / 16_000, 1)}
                        for p in ps],
            "recorded": len(have & {p["id"] for p in ps}), "total": len(ps),
            "seconds": round(sum(len(c) for c in clips.values()) / 16_000, 1)}


def save(prompt_id: str, x: np.ndarray, sr: int, corpus: str | Path = CORPUS) -> dict:
    """Store one reading of one prompt (replacing an earlier take) and rewrite the corpus."""
    from ..audio.dataset import Manifest
    from ..audio.io import resample
    by_id = {p["id"]: p["text"] for p in prompts()}
    if prompt_id not in by_id:
        raise ValueError(f"unknown prompt {prompt_id!r}")
    x = np.asarray(x, dtype=np.float32)
    if sr != 16_000:
        x = resample(x, int(sr), 16_000).astype(np.float32)
    secs = len(x) / 16_000
    if not 0.5 <= secs <= 30:
        raise ValueError(f"{secs:.1f} s — read the sentence once, between 0.5 and 30 seconds")
    corpus = Path(corpus)
    corpus.mkdir(parents=True, exist_ok=True)
    clips, texts = _load(corpus)
    name = prompt_id + ".wav"
    clips[name] = (np.clip(x, -1, 1) * 32767).astype("<i2")
    texts[name] = by_id[prompt_id].upper()       # LibriSpeech's case, which Utterances normalises
    names = sorted(clips)
    offsets, total = [0], 0
    with open(corpus / "audio.bin.tmp", "wb") as f:
        for n in names:
            f.write(clips[n].tobytes())
            total += len(clips[n])
            offsets.append(total)
    (corpus / "audio.bin.tmp").replace(corpus / "audio.bin")
    Manifest(sample_rate=16_000, offsets=offsets, names=names,
             sources=[{"sr": 16_000, "channels": 1,
                       "peak": round(float(np.abs(clips[n]).max()) / 32767, 4)} for n in names],
             seconds=total / 16_000, built=time.strftime("%Y-%m-%d %H:%M:%S"),
             source_dir="recorded: python -m aksharallm.asr myvoice").save(corpus / "manifest.json")
    (corpus / "transcripts.json").write_text(json.dumps({n: texts[n] for n in names}, indent=0))
    return status(corpus)
