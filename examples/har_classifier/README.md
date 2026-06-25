# HAR Classifier — PyTorch + C Parity Demo

Trains a 6-class human-activity classifier on the UCI HAR dataset using the
1D-CNN layers exposed by both PyTorch (reference) and the ODT C framework. The
C model is built with the factory layer API (`conv1dLayerInit` + `layerQuant_t`)
and loads PyTorch weights through `StateDictApi`.

One binary, two verification modes:

- **Bit-parity** (what CI runs): `BIT_PARITY=1` loads PyTorch's trained weights
  into the C model and runs inference only — the C predictions must be
  **bit-identical** to PyTorch's. Deterministic and exact.
- **Train-from-scratch demo**: with no env var the C model trains from its own
  random init; `compare.py` checks final-state parity within tolerance and emits
  plots. Independent init, so it verifies *convergence*, not bits.

## Run it

```bash
# 1. Prepare data (downloads ~58 MB the first time; cached under data/raw/)
uv run python examples/har_classifier/prepare_data.py

# 2. Train the PyTorch reference + export weights (~30s on CPU)
uv run python examples/har_classifier/train_pytorch.py

# 3. Build the C trainer
cmake --preset examples
cmake --build --preset examples --target train_c_har_classifier

# 4a. Bit-parity check (exact — this is the CI gate)
BIT_PARITY=1 ./build/examples/examples/har_classifier/train_c_har_classifier
uv run python examples/_shared/compare_predictions.py \
  --pytorch examples/har_classifier/outputs/pytorch_predictions.npy \
  --c examples/har_classifier/outputs/c_predictions.npy --dtype int32

# 4b. …or the train-from-scratch demo + plots (several minutes)
./build/examples/examples/har_classifier/train_c_har_classifier
uv run python examples/har_classifier/compare.py
```

## Outputs

After the train-from-scratch demo, `examples/har_classifier/` contains:
- `data/{train,val,test}_{x,y}.npy`
- `logs/{pytorch,c}.json`
- `outputs/{pytorch,c}_predictions.npy`
- `plots/{loss_curves,accuracy_curves,confusion_matrix_pt,confusion_matrix_c}.png`

## Model

- Input: `[9, 128]` (9 IMU channels, 2.56 s window @ 50 Hz)
- 3 × `Conv1d → ReLU → MaxPool1d` blocks
- Global `AvgPool1d` → `Flatten → Linear → Softmax → CrossEntropy`
- ~10 K parameters

## Parity tolerance (train-from-scratch demo)

| Metric | Tolerance |
|---|---|
| test_acc  | ±2.5 pp absolute |
| test_loss | ±0.15 nats absolute |

The demo's two implementations use independent random init; the loss tolerance
is empirically calibrated. Bit-parity mode requires exact equality instead.
See `examples/_shared/DETERMINISM.md` for the full determinism contract.
