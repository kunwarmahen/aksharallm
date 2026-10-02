"""The day-two checks, as one suite: what a dictation clone gets wrong once someone uses it.

Phase 7 started from a list of five failures that a weekend dictation app shows on its second
day (PLAN.md § Phase 7). Each is a number here, measured on one corpus in one pass, written
to one JSON (`logs/asr/daytwo-*.json`, never `logs/eval/` — gotcha 18):

| # | failure | the check | pass |
|---|---|---|---|
| 1 | writes words on silence | characters written on five no-speech clips | 0 |
| 2 | accents / some voices | WER per speaker; the **worst** beside the median | reported, not graded |
| 3 | switching languages | — | **not built**: English-only, and said so |
| 4 | names | recall of words the LM has never seen, with and without them in a dictionary | reported, with false insertions |
| 5 | never learns | **a correction replay**: see below | the 4th occurrence beats the 1st |
| + | cleanup invents words | words in the cleaned text that the recogniser did not write | 0 |

**The correction replay (5) is the one worth reading.** For every word the LM has never seen
that occurs in at least `min_occurrences` utterances (mostly names — "montfichet",
"boolooroo"), the utterances containing it are dictated **in order**, each one with whatever a
fresh personal memory (`dictate/personal.py`) has learned so far, and after each one the
transcript is "corrected" to the reference exactly as a person would fix it. The result is a
curve: how often the word comes out right on its 1st, 2nd, 3rd… occurrence. A system that
learns has a rising curve; one that only *says* it learns has a flat one. Beside it,
**false insertions**: those learned words written into utterances where nobody said them —
the cost a bias pays, which recall alone would hide.

It replays with references the encoder never trained on (test-clean) and per-word memories
(each target learns alone), so the curve isolates that one word's learning rather than the
interaction of thirteen.

Read with: docs/23-speech-recognition.md -- the chapter this implements; it ends with the
order to read these files in.
"""

from __future__ import annotations

import tempfile
from collections import Counter, defaultdict

import numpy as np

from .measure import name_recall, score, silence_check


def _decode(lps, lm, tuning: dict, dictionary=(), workers: int = 8) -> list[str]:
    from .decode import BeamDecoder, decode_many
    dec = BeamDecoder(lm=lm, alpha=tuning["alpha"], beta=tuning["beta"],
                      unk_penalty=tuning["unk_penalty"], beam=tuning.get("beam", 16),
                      dictionary=set(dictionary))
    return decode_many(dec, lps, workers)


def correction_replay(texts: list[str], lps: list[np.ndarray], lm, tuning: dict,
                      min_occurrences: int = 4, max_occurrences: int = 6,
                      false_alarm_utts: int = 200) -> dict:
    """Day-two 5. See the module docstring."""
    from ..dictate.personal import Personal
    from .decode import BeamDecoder

    holders = defaultdict(list)
    for i, t in enumerate(texts):
        for w in set(t.split()):
            if w not in lm.index:
                holders[w].append(i)
    targets = sorted(w for w, ix in holders.items() if len(ix) >= min_occurrences)
    hits = defaultdict(list)          # occurrence number -> [0/1 ...]
    per_word = {}
    learned_words: set[str] = set()
    for w in targets:
        with tempfile.TemporaryDirectory() as tmp:
            p = Personal(tmp, known=lambda x: x in lm.index)
            curve = []
            for n, i in enumerate(holders[w][:max_occurrences], 1):
                dec = BeamDecoder(lm=lm, alpha=tuning["alpha"], beta=tuning["beta"],
                                  unk_penalty=tuning["unk_penalty"], beam=tuning.get("beam", 16),
                                  dictionary=p.dictionary())
                words, _ = p.rewrite(dec.decode(lps[i]).split())
                hyp = " ".join(words)
                ok = int(w in words)
                curve.append(ok)
                hits[n].append(ok)
                # The person fixes the transcript to what they said.
                p.learn(hyp, texts[i], save=False)
            per_word[w] = curve
            learned_words |= p.dictionary()
    # False insertions: every learned word, written where it was not said.
    clean_ix = [i for i, t in enumerate(texts) if not (set(t.split()) & set(targets))]
    rng = np.random.default_rng(0)
    pick = sorted(rng.choice(clean_ix, size=min(false_alarm_utts, len(clean_ix)), replace=False).tolist())
    learned_oov = {w for w in learned_words if w not in lm.index}
    hyps = _decode([lps[i] for i in pick], lm, tuning, learned_oov)
    false = sum(sum(1 for x in h.split() if x in learned_oov and x not in texts[i].split())
                for h, i in zip(hyps, pick, strict=True))
    return {
        "targets": len(targets), "min_occurrences": min_occurrences,
        "curve": [{"occurrence": n, "words": len(v), "recall": sum(v) / len(v)}
                  for n, v in sorted(hits.items())],
        "per_word": per_word,
        "false_insertions": false, "false_insertion_utts": len(pick),
        "learned_words": len(learned_oov),
    }


def cleanup_check(hyps: list[str], punctuator) -> dict:
    """Day-two +: does the cleanup ever write a word the recogniser did not? Fillers and
    spoken commands are removed on purpose and are excluded; anything else is a failure."""
    from ..dictate.cleanup import FILLERS, clean
    from ..dictate.punct import words_of
    invented = dropped = words = 0
    examples = []
    for h in hyps:
        src = [w for w in h.split() if w not in FILLERS]
        if any(tuple(src[i : i + 2]) in {("new", "line"), ("new", "paragraph"), ("scratch", "that")}
               for i in range(len(src) - 1)):
            continue
        out = words_of(clean(h, punctuator).text.replace("\n", " "))
        a, b = Counter(src), Counter(out)
        inv = sum((b - a).values())
        dro = sum((a - b).values())
        invented += inv
        dropped += dro
        words += len(src)
        if (inv or dro) and len(examples) < 5:
            examples.append({"heard": h, "wrote": clean(h, punctuator).text})
    return {"words": words, "invented": invented, "dropped": dropped, "examples": examples}


def run(model, device: str, corpus, lm, tuning: dict, punctuator=None, workers: int = 8,
        min_occurrences: int = 4, score_limit: int | None = None) -> dict:
    from .measure import log_probs

    utts = corpus.utts
    texts = [u.text for u in utts]
    lps = log_probs(model, [corpus.wave(u) for u in utts], device)
    sub = list(range(len(utts) if score_limit is None else min(score_limit, len(utts))))
    sub_lps = [lps[i] for i in sub]

    out = {"utts": len(utts), "scored_utts": len(sub)}
    # 1. silence
    sil = silence_check(model, device)
    out["silence"] = {"chars": sil["chars"], "clips": sil["clips"], "outputs": sil["outputs"],
                      "pass": sil["chars"] == 0}
    # 2. speakers (and the headline WER the rest is read against)
    hyps = _decode(sub_lps, lm, tuning, (), workers)
    s = score([(utts[i].speaker, texts[i], h) for i, h in zip(sub, hyps, strict=True)])
    rates = sorted(v["wer"] for v in s["speakers"].values())
    out["speakers"] = {"wer": s["wer"], "speakers": len(rates),
                       "best": rates[0] if rates else None,
                       "median": rates[len(rates) // 2] if rates else None,
                       "worst": rates[-1] if rates else None,
                       "worst_speakers": s["worst_speakers"][:3]}
    # 3. languages
    out["languages"] = {"status": "not built",
                        "why": "the recogniser is English-only (LibriSpeech); per-segment "
                               "language id is PLAN.md § Phase 7, still to do"}
    # 4. names
    targets = {w for i in sub for w in texts[i].split() if w not in lm.index}
    before = name_recall([texts[i] for i in sub], hyps, targets)
    with_dict = _decode(sub_lps, lm, tuning, targets, workers)
    after = name_recall([texts[i] for i in sub], with_dict, targets)
    out["names"] = {"words": len(targets), "without": before, "with_dictionary": after}
    # 5. corrections
    out["corrections"] = correction_replay(texts, lps, lm, tuning, min_occurrences)
    c = out["corrections"]["curve"]
    out["corrections"]["learns"] = bool(len(c) >= 2 and c[-1]["recall"] > c[0]["recall"])
    # + cleanup never invents
    out["cleanup"] = {**cleanup_check(hyps, punctuator),
                      "punctuator": "tagger" if punctuator is not None else "rules"}
    out["cleanup"]["pass"] = out["cleanup"]["invented"] == 0 and out["cleanup"]["dropped"] == 0
    return out
