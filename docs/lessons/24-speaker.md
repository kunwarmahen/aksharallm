---
id: speaker
title: Does it still sound like that person? Scoring a voice with an equal error rate
doc: docs/21-audio.md
files:
  - aksharallm/audio/speaker.py
  - aksharallm/audio/train_speaker.py
  - aksharallm/audio/transfer.py
verify: tests/test_speaker.py::test_eer_is_zero_when_separated_one_when_reversed_and_half_at_random
prereqs: [codec]
minutes: 30
summary: MCD measures the spectrum and WER measures the words, so neither can tell you whether a codec kept the voice. A speaker embedding can, and its one number, the equal error rate, is easy to compute backwards.
---

# 24. Scoring a voice

The [codec](docs/21-audio.md) has two scores. **MCD** measures the spectral envelope. **WER**
asks a recogniser to transcribe the reconstruction. A codec that turned every speaker into the
same voice could still do well on both. A multi-voice codec needs a third score, for the one
thing it exists to get right: does the output still sound like the person who spoke?

That score is a **speaker embedding**: a network that turns a clip into a unit vector, trained
so that two clips of one person point the same way and two different people point apart.

```
clip ─► log-mel (normalised per clip) ─► Conformer (the recogniser's) ─► mean + std over time
     ─► Linear ─► unit vector e

train:    tell 1,172 LibriSpeech speakers apart, with a margin (AM-softmax)
measure:  EER on speakers it never trained on
```

* **Normalised per clip**, the opposite of the recogniser's default. Otherwise the model could
  tell two speakers apart by how loud their microphones were.
* **The classifier is thrown away.** The 1,172 speaker weights only exist to train the
  embedding, so accuracy on the training speakers says little. The number that counts is the
  **equal error rate on new speakers**. Pick a similarity threshold and call everything above
  it "same speaker". The EER is the error rate at the threshold where false accepts equal
  false rejects. Chance is 50%.
* **A similarity needs two references beside it.** `cos(original, reconstruction)` means
  nothing alone. It is reported next to the mean similarity of two clips of the same speaker
  and of two different speakers, measured with the same model.

---

## Exercise: compute the EER backwards

`eer()` sorts the pairs from most similar to least, then walks down the list counting false
accepts and false rejects.

1. Run the check. It passes: perfectly separated scores give 0, perfectly reversed ones give
   1, and random ones give 0.5.
2. In `aksharallm/audio/speaker.py`, find `eer` and change `np.argsort(-scores)` to
   `np.argsort(scores)`, so the list runs from least similar to most.
3. Run the check. **It should fail**: perfectly separated scores now give an EER of 1.
4. Put it back. Green.

> **What you just saw.** A backwards EER gives `1 − EER` for any model. An untrained network
> is near 50%, where the two readings are almost the same. So the bug is invisible at the
> start of a run, and once the model learns it shows as a score getting *worse*. Ours scores
> 27% before any training and 4.5% after 2,000 steps. Read backwards, that would be 73% →
> 95.5%, a model apparently learning to be wrong. The random-scores case in the check is
> there to catch an error that leaves the 50% point alone.

## Run it yourself

```bash
scripts/audio.sh speaker-libri                                    # ~16 min on a 3090
.venv/bin/python -m aksharallm.audio speaker-eval checkpoints/speaker-libri/ckpt_best.pt
.venv/bin/python -m aksharallm.audio codec-transfer checkpoints/codec-lj/ckpt_best.pt \
    --speaker checkpoints/speaker-libri/ckpt_best.pt              # adds the voice column
```

The best checkpoint is chosen on dev-clean's speakers, and `speaker-eval` reports on
test-clean's. Neither set is in training, and they don't overlap each other, so the number you
quote was not used to choose the checkpoint.
