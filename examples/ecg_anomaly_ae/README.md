# ECG5000 Reconstruction Autoencoder — PyTorch + C Parity Demo

Trains a univariate 1D-CNN autoencoder on the ECG5000 dataset (UCR Time-Series
Classification archive). The training set is filtered to class-1 normals only;
at evaluation time, reconstruction MSE acts as an anomaly score against the
multi-class test set, with the threshold derived from training-set normals.

First example to exercise `Conv1dTransposed`. The C model is built with the
factory layer API and loads PyTorch weights through `StateDictApi`.

One binary, two verification modes:

- **Bit-parity** (what CI runs): `BIT_PARITY=1` loads PyTorch's trained weights
  into the C model and runs inference only — the C reconstructions must match
  PyTorch's within float tolerance (`rtol 1e-4, atol 1e-5`).
- **Train-from-scratch demo**: with no env var the C model trains from its own
  random init; `compare.py` checks final-state parity within tolerance and emits
  plots. Independent init, so it verifies *convergence*, not bits.

## Run it

```bash
# 1. Prepare data (downloads ~10 MB the first time; cached under data/raw/)
uv run python examples/ecg_anomaly_ae/prepare_data.py

# 2. Train the PyTorch reference + export weights (~4 minutes on CPU)
uv run python examples/ecg_anomaly_ae/train_pytorch.py

# 3. Build the C trainer
cmake --preset examples
cmake --build --preset examples --target train_c_ecg_anomaly_ae

# 4a. Bit-parity check (this is the CI gate)
BIT_PARITY=1 ./build/examples/examples/ecg_anomaly_ae/train_c_ecg_anomaly_ae
uv run python examples/_shared/compare_predictions.py \
  --pytorch examples/ecg_anomaly_ae/outputs/pytorch_reconstructions.npy \
  --c examples/ecg_anomaly_ae/outputs/c_reconstructions.npy \
  --dtype float32 --rtol 1e-4 --atol 1e-5

# 4b. …or the train-from-scratch demo + plots (~5s on this small dataset)
./build/examples/examples/ecg_anomaly_ae/train_c_ecg_anomaly_ae
uv run python examples/ecg_anomaly_ae/compare.py
```

## Outputs

After the train-from-scratch demo, `examples/ecg_anomaly_ae/` contains:
- `data/{train,val,test}_x.npy` and `data/test_y.npy`
- `logs/{pytorch,c}.json`
- `outputs/{pytorch,c}_reconstructions.npy` and `{pytorch,c}_train_recons.npy`
- `plots/{loss_curves,reconstructions,anomaly_score_hist}.png`

## Model

- Input: `[1, 140]` (univariate ECG beat, 140 samples)
- Encoder: `Conv1d(1→8, K=7, S=2, SAME)` → ReLU → `MaxPool1d(K=2, S=2)`
  → `Conv1d(8→16, K=5, SAME)` → ReLU → `AvgPool1d(K=5, S=5)` → `[16, 7]`
- Decoder: `Conv1dTransposed(16→8, K=5, S=5)` → ReLU
  → `Conv1dTransposed(8→4, K=2, S=2)` → ReLU
  → `Conv1dTransposed(4→1, K=2, S=2)` → `[1, 140]`
- ~1.5 K parameters

The decoder uses kernel size 2 (not the spec §4.2 K=4) for the two stride-2
layers because our framework's `Conv1dTransposed` only supports
`paddingType_t = VALID` plus an `outputPadding`; the only kernel/stride
combination that hits length 70 from 35 (and 140 from 70) without input-side
padding is K=2, S=2, op=0. The PyTorch model uses the same K=2 layout for parity.

## Training config

- Optimizer: SGD with momentum 0.9, lr 0.005
- Loss: MSE (reduction="mean", per-element)
- Batch size: 32 (training); 1 (val/test microbatch)
- Epochs: 200

The spec §4.2 originally projected 50 epochs, but the K=2 substitution slows
convergence enough that 50 epochs leave the model mid-descent. 200 epochs
provides a safety margin past the spec's expected `test_mse ≈ 0.05`.

## Parity tolerance (train-from-scratch demo)

| Metric | Tolerance | Notes |
|---|---|---|
| test_mse | ±20 % relative | ECG-specific override of spec §6's ±10 %; the K=2 substitution + independent random init produce a small test-set gap on out-of-distribution anomaly samples while train/val parity holds within ~7 % |
| anomaly AUC | ±3 pp absolute | Spec §6 default |

These tolerances are **informational** — `compare.py` reports them and writes
plots but does not fail. The two implementations use independent random init, and
this tiny AE amplifies a slight C-vs-PyTorch *training-dynamics* difference
(bit-parity tests inference only) into different optima, which can push the
anomaly AUC/MSE outside tolerance. The **exact gate is bit-parity mode** (load
PyTorch weights → matching reconstructions, run in CI). The training divergence is
a known open finding under separate investigation. See
`examples/_shared/DETERMINISM.md`.
