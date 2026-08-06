---
id: codec
title: Turning sound into integers, and the lie that lets it train
doc: docs/20-audio.md
files:
  - aksharallm/audio/vq.py
  - aksharallm/audio/codec.py
  - aksharallm/audio/features.py
verify: tests/test_codec.py::test_the_straight_through_gradient_reaches_the_encoder
prereqs: [training-loop]
minutes: 35
summary: The transformer does not care what its tokens mean — so a codec is all that stands between it and sound. Inside it is a gradient that does not exist, and a deliberate lie about it.
---

# 20. Turning sound into integers, and the lie that lets it train

Nothing in `model/transformer.py` knows about words. Open it and look: it knows about
integers, their order, and a vocabulary size. So the whole stack you have built —
pretraining, RoPE, GQA, the KV cache, the sampler, quantization, LoRA — will work on
**sound**, if something can turn a waveform into integers and back.

That something is a **codec**, and it is the only genuinely new machinery in the audio phase.

```
waveform  →  conv encoder  →  128 floats,  →  nearest codebook  →  8 integers,
16,000/s     320x down        50 times/s      entry, 8 times        50 times/s
```

The arithmetic decides everything downstream. 16,000 samples a second, downsampled by 320,
is **50 frames a second**. Eight codebooks of 1,024 entries is 10 bits each, so 80 bits a
frame — **4,000 bits a second**, against 256,000 for the 16-bit audio it came from. And
50 × 8 is the sequence length the transformer pays: ten seconds of speech is 4,000 tokens.

## The problem at the centre

Replacing a vector with the nearest of 1,024 learned vectors is an `argmin`. Differentiate an
`argmin` and you get **zero, almost everywhere** — nudge the encoder's output a little and
the chosen index does not change, so the loss does not change, so the gradient is nothing.
The encoder would never learn.

The **straight-through estimator** is the standard answer, and it is a deliberate lie:

```python
st = z + (q - z).detach()
```

Numerically that is exactly `q` — the codebook entry — because `z` cancels. But `.detach()`
means the second term contributes no gradient, so *differentiating* it gives `d(st)/d(z) = 1`.
Forward, the quantizer. Backward, the identity. The encoder is trained as though the
quantizer were not there.

The codebook then has to learn some other way, because straight-through gives it nothing.
Here it is an **exponential moving average** of the vectors assigned to each entry — a
k-means step, without the optimizer ever seeing it. That is why the codebook is a `buffer`
and not a `Parameter`: as a Parameter, weight decay would quietly shrink every entry towards
the origin between updates.

## And the failure you already know

A handful of entries win early, receive all the assignments, and the rest never train. A
1,024-entry codebook is quietly a 40-entry one, reconstruction plateaus — **and the loss
curve looks fine**, because a plateau is what every loss curve does eventually.

That is [lesson 13](docs/lessons/13-moe.md)'s router collapse, wearing a different hat, and it gets
the same treatment: usage counted from step one, and **dead-code restart** — an entry nobody
has chosen for a while is reinitialised to a random encoder output, which is somewhere data
actually is.

---

## Exercise: reverse the lie

The straight-through line has two halves and it is easy to detach the wrong one. Do it, and
watch what does *not* happen.

1. Run the check. It passes: the gradient arriving at `z` is exactly `1` everywhere.
2. In `aksharallm/audio/vq.py`, find `VectorQuantizer.forward` and change the
   straight-through line to detach the other side:

   ```python
   st = z.detach() + (q - z)      # <- wrong
   ```

3. Run the check. **It should fail** — and read *how*. The gradient is now `-1`, not zero,
   so nothing crashes and nothing warns.
4. Put it back. Green.

> **What you just saw.** Both versions produce a finite gradient of the right shape, and a
> training run with the wrong one will start, log a falling loss, and save checkpoints. What
> it will not do is learn a useful encoder — it is being pushed in the opposite direction on
> every step. This is the same family as the `is_causal` bug in
> [lesson 3](docs/lessons/03-attention.md): *trains fine, output is garbage*, and the only
> defence is a test that asserts the value rather than the absence of an exception.

## Two more things that were nearly wrong here

**The codebook must not run in bf16.** Under autocast the encoder's output is bf16, so the
EMA ran in bf16 too — and bf16 has eight bits of mantissa, while an EMA with decay 0.99 adds
one percent of a centroid per step. One percent is *below the dtype's own resolution*. The
codebook would have stopped moving with nothing to say so. `forward` now forces float32
internally and casts the straight-through output back, so the decoder keeps its fast dtype.

**The codebook has to be seeded from data.** A codebook initialised at `randn·0.02`, against
an encoder whose outputs happen to have a scale of 3, is a codebook where *one* entry is
nearest to everything. The first run here collapsed to an effective size of **1.0** by step
50. Seeding every entry from the first batch's encoder outputs fixes it, and it is four
lines.

## What it buys, and you can hear it

Residual VQ quantizes what the previous codebook got *wrong*, which means the **prefix of a
code is a valid code**. Decode one codebook instead of eight and you get a coarser but
listenable reconstruction — so bitrate is a dial you turn at decode time, not a property of
the checkpoint.

```bash
.venv/bin/python -m aksharallm.audio corpus --out data/audio/synth --clips 400
scripts/audio.sh codec-synth
.venv/bin/python -m aksharallm.audio reconstruct checkpoints/codec-synth/ckpt_best.pt \
    data/audio/synth/wavs/synth-0399.wav --codebooks 1,2,4,8
```

Or open the portal's **Audio** tab, which plays all four against the original. The trade you
are hearing is the same one [lesson 9](docs/lessons/09-quantization.md) makes silently in the weights.
