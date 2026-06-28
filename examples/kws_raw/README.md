# KWS Raw Waveform — PyTorch + C Parity Demo

Trains a 1D-CNN keyword-spotter on **raw 16 kHz SpeechCommands waveforms** in both
PyTorch (reference) and the ODT C framework. Companion to `kws_mfcc/`: same data
and harness, but instead of pre-computing MFCC features, the model consumes the
native `[1, 16000]` waveform and **downsamples in-framework** — its first layer is
`AvgPool1d(K=16, S=16)`, a decimation-by-16 box filter that turns 16 kHz into
1 kHz. Change `K` to change the effective rate (8 → 2 kHz, …) with no re-prep; the
`AdaptiveAvgPool1d(1)` head is length-agnostic so the rest of the model is
unchanged (only the three MaxPool nominal `inputLength`s in `train_c.c` need to
track the new lengths).

One binary, two modes — **bit-parity** (`BIT_PARITY=1`, the exact CI gate) and a
**train-from-scratch** informational demo. See `kws_mfcc/README.md` for the mode
explanation and the `KWS_CLASSES` knob; commands are identical with `kws_raw`
substituted.

## Why per-conv LayerNorm + a longer schedule

Raw waveforms are far harder to train than MFCC features: at the `kws_mfcc`
settings (lr=0.001) the raw model just trains *very* slowly and looks stuck at
random init within 15–20 epochs, which would make the bit-parity gate degenerate
(a one-class reference). The fix uses **LayerNorm**, the framework's only
bit-parity-covered normalizer (BatchNorm is not), at **lr=0.005, 50 epochs**.

A 10-seed sweep (3 placements × 3 learning rates × 10 seeds × 50 epochs) settled
*where* the LayerNorm goes:

| placement | mean ± std test_acc | seeds converged |
|---|---|---|
| no LayerNorm | 0.70 ± 0.02 | 10/10 |
| LayerNorm(64) after pooling | **0.47 ± 0.25** | **~6/10** |
| **per-conv `LayerNorm([C,L])`** | **0.72 ± 0.01** | **10/10** |

A single LayerNorm *after* global pooling is the **worst** option — it amplifies a
bad init and collapses on ~40 % of seeds. Per-conv LayerNorm (one over each conv's
full `[C, L]` feature map, pre-ReLU) normalises *inside* the stack and converges
reliably (`0.72 ± 0.01`, every seed, all six classes), so the gate genuinely
exercises the `AvgPool1d[1,16000]` + Conv + LayerNorm arithmetic (C reproduces
PyTorch's predictions int32-exactly). Even plain no-LayerNorm trains fine given 50
epochs — the model was never un-trainable, just slow.

## Run it (6-class)

```bash
uv run python examples/kws_raw/prepare_data.py
uv run python examples/kws_raw/train_pytorch.py
cmake --preset examples
cmake --build --preset examples --target train_c_kws_raw

BIT_PARITY=1 ./build/examples/examples/kws_raw/train_c_kws_raw
uv run python examples/_shared/compare_predictions.py \
  --pytorch examples/kws_raw/outputs/6class/pytorch_predictions.npy \
  --c examples/kws_raw/outputs/6class/c_predictions.npy --dtype int32
```

## Model

- Input: `[1, 16000]` → `reshapeItemsAddBatchDim` → `[1, 1, 16000]`
- `AvgPool1d(16) → 3× [Conv1d(K3,SAME) → LayerNorm([C,L]) → ReLU → MaxPool(4)] →
  AdaptiveAvgPool1d(1) → Flatten → Linear(64→C) → Softmax → CE`
  (channels 1→16→32→64; LayerNorm shapes `[16,1000]`, `[32,250]`, `[64,62]`)
- Lengths: 16000 → 1000 → 250 → 62 → 15 → 1; ~64 K params (the LayerNorm gamma/beta dominate)
- State-dict layers: `conv1`, `ln1`, `conv2`, `ln2`, `conv3`, `ln3`, `fc`
- Hyperparameters: SGD lr=0.005, momentum=0.9, batch=32, 50 epochs

The train-from-scratch demo is the slowest in the suite (raw `[1,16000]` is the
heaviest input even after the AvgPool downsample) — run it offline. Bit-parity
mode requires exact equality; the train-from-scratch tolerances are informational
and match `kws_mfcc/`.
