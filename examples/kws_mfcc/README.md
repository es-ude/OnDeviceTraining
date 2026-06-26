# KWS MFCC — PyTorch + C Parity Demo

Trains a small 1D-CNN keyword-spotter on Google SpeechCommands MFCC features in
both PyTorch (reference) and the ODT C framework. Stage 3 of the 1D-CNN example
suite. Each 1 s clip → log-MFCC `[40, 32]` (40 mel-cepstra × 32 frames); MFCC is
computed once in `prepare_data.py` so PyTorch and C read **identical** `.npy` —
feature extraction sits outside the parity check.

One binary, two verification modes — **bit-parity** (`BIT_PARITY=1`, the exact CI
gate: loads PyTorch's trained weights and runs inference only; C predictions must
be bit-identical) and a **train-from-scratch** informational demo (independent
random init; `compare.py` checks convergence within tolerance + emits plots).

## Class-count knob

`KWS_CLASSES` (default **6**) selects the subset. CI runs **6-class only**; 35 is
local-only. Per-config artifacts live under `<n>class/` subdirs.

- **6-class** (labels 0..5): `yes`, `no`, `up`, `down`, `silence` (synthetic
  low-amplitude Gaussian noise), `unknown` (random clips from the other 31 keywords).
- **35-class**: the 35 natural keywords, alphabetical.

## Run it (6-class)

```bash
uv run python examples/kws_mfcc/prepare_data.py        # downloads ~2.3 GB once (shared root)
uv run python examples/kws_mfcc/train_pytorch.py
cmake --preset examples
cmake --build --preset examples --target train_c_kws_mfcc

# Bit-parity (exact — the CI gate)
BIT_PARITY=1 ./build/examples/examples/kws_mfcc/train_c_kws_mfcc
uv run python examples/_shared/compare_predictions.py \
  --pytorch examples/kws_mfcc/outputs/6class/pytorch_predictions.npy \
  --c examples/kws_mfcc/outputs/6class/c_predictions.npy --dtype int32

# …or the train-from-scratch demo + plots (SLOW — C trains one sample at a time)
./build/examples/examples/kws_mfcc/train_c_kws_mfcc
uv run python examples/kws_mfcc/compare.py
```

Run the full 35-class set with `KWS_CLASSES=35 …` on every command (local-only).

## Model

- Input: `[40, 32]` (40 MFCC channels, 32 frames) → `reshapeItemsAddBatchDim` → `[1, 40, 32]`
- `Conv1d(40→32,K3,SAME) → ReLU → MaxPool(2) → Conv1d(32→64,K3,SAME) → ReLU →
  MaxPool(2) → AdaptiveAvgPool1d(1) → Flatten → Linear(64→C) → Softmax → CE`
- Lengths: 32 → 16 → 8 → 1; ~16 K params
- State-dict layers: `conv1`, `conv2`, `fc`

The train-from-scratch tolerances (`test_acc ±2.5 pp`, `test_loss ±0.15 nats`) are
informational; bit-parity mode requires exact equality. See
`examples/_shared/DETERMINISM.md` for the determinism contract.
