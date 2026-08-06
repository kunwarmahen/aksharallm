"""Audio — the same transformer, on sound instead of text.

The claim this package exists to test is that **the transformer does not care what its
tokens mean.** Nothing in `model/transformer.py` knows about words; it knows about integers
and their order. So if something can turn a waveform into a sequence of integers and back
again, the entire stack already built here — the pretraining loop, the KV cache, the
sampler, the quantizer, LoRA, the portal — works on sound without being told about it.

That something is a **codec**, and it is the only genuinely new piece of machinery in the
phase:

```mermaid
flowchart LR
    W["waveform<br/>16 kHz"] --> E["conv encoder<br/>320x downsample"]
    E --> Q["residual VQ<br/>N codebooks"]
    Q --> T["discrete tokens<br/>50 frames/s x N"]
    T --> LM["the SAME Transformer<br/>next-token prediction"]
    T --> D["conv decoder"]
    D --> W2["waveform back"]
```

Four pieces, strictly ordered because each needs the one before it:

| # | module | what it is |
|---|---|---|
| 1 | `io.py`, `features.py` | waveform in, log-mel out, and Griffin-Lim back to sound so you can **listen to what the model sees**. CPU only. |
| 2 | `codec.py`, `vq.py` | the RVQ-VAE. The one new training loop, and the expensive piece. |
| 3 | `delay.py`, `lm.py` | the existing `Transformer` over codec tokens, with the delay pattern that flattens N codebooks into one stream. |
| 4 | `tts.py`, `asr.py` | text -> audio tokens and audio -> text, reusing the SFT assistant-only loss mask with audio in the assistant's role. |

Read with: docs/20-audio.md -- the chapter this implements; it ends with the order to read
these files in.
"""

from .features import (
    MelConfig,
    griffin_lim,
    hz_to_mel,
    istft,
    log_mel,
    magnitude,
    mel_filterbank,
    mel_to_hz,
    mel_to_magnitude,
    spectral_convergence,
    stft,
)
from .io import Clip, load_audio, read_wav, resample, write_wav

__all__ = [
    "MelConfig", "stft", "istft", "magnitude", "log_mel", "mel_filterbank",
    "mel_to_magnitude", "griffin_lim", "spectral_convergence", "hz_to_mel", "mel_to_hz",
    "Clip", "load_audio", "read_wav", "write_wav", "resample",
]
