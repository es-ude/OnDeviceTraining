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

## Why LayerNorm + a longer schedule

Raw waveforms are far harder to train than MFCC features: at the `kws_mfcc`
settings (lr=0.001, 15 epochs) the raw model never escapes its random-init
fixed point (flat loss, every clip predicted as one class), which would make the
bit-parity gate degenerate. Two changes fix it without leaving the framework's
bit-parity-covered layers:

- a rate-agnostic **`LayerNorm(64)`** on the pooled features before the classifier
  (the C framework has bit-parity LayerNorm; BatchNorm is not covered), and
- **lr=0.005, 20 epochs** (the model breaks through around epoch 15).

The reference then reaches ~0.59 test accuracy with predictions spread across all
six classes, so the gate genuinely exercises the `AvgPool1d[1,16000]` + Conv +
LayerNorm arithmetic (C reproduces PyTorch's predictions int32-exactly).

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
- `AvgPool1d(16) → Conv1d(1→16,K3,SAME) → ReLU → MaxPool(4) → Conv1d(16→32,K3,SAME)
  → ReLU → MaxPool(4) → Conv1d(32→64,K3,SAME) → ReLU → MaxPool(4) →
  AdaptiveAvgPool1d(1) → Flatten → LayerNorm(64) → Linear(64→C) → Softmax → CE`
- Lengths: 16000 → 1000 → 250 → 62 → 15 → 1; ~10 K params
- State-dict layers: `conv1`, `conv2`, `conv3`, `ln`, `fc`
- Hyperparameters: SGD lr=0.005, momentum=0.9, batch=32, 20 epochs

The train-from-scratch demo is the slowest in the suite (raw `[1,16000]` is the
heaviest input even after the AvgPool downsample) — run it offline. Bit-parity
mode requires exact equality; the train-from-scratch tolerances are informational
and match `kws_mfcc/`.
