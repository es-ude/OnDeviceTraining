# Examples — End-to-End 1D-CNN Demos with PyTorch + C Parity

Each subdirectory is a self-contained example demonstrating the same model
in PyTorch (reference) and the ODT C framework, with final-state parity
checking and visualizations.

## Currently shipping

| Directory | Task | Status |
|---|---|---|
| `mnist_mlp/` | MNIST dense-MLP digit classification | ✅ |
| `mnist_cnn/` | MNIST 1D-CNN digit classification | ✅ |
| `har_classifier/` | UCI HAR 6-class activity classification | Stage 1 |
| `ecg_anomaly_ae/` | ECG5000 reconstruction-based anomaly detection | Stage 2 ✅ |
| `kws_classifier/` | SpeechCommands 6-class keyword spotting | Stage 3 (planned) |
| `kws_denoising_ae/` | SpeechCommands additive-noise denoising | Stage 4 (planned) |

## Running an example

```bash
uv run python examples/<name>/prepare_data.py
uv run python examples/<name>/train_pytorch.py
cmake --preset examples
cmake --build --preset examples --target train_c_<name>
./build/examples/examples/<name>/train_c_<name>
uv run python examples/<name>/compare.py
```

Each `train_c_<name>` binary also has a **bit-parity** mode: run it with
`BIT_PARITY=1` and it loads the PyTorch reference weights (instead of training
from scratch) and emits predictions that must match PyTorch exactly. This is
the deterministic check CI runs; see each example's README for the precise
`compare_predictions.py` invocation.

The C-side executables only build when configured with the `examples`
preset (`BUILD_EXAMPLES=ON`); the default `unit_test_*` presets do not
build them.

## Shared infrastructure

All examples share `examples/_shared/`: seed constants, the Python mirror
of the C `XorShift32` RNG (so train-set shuffle order matches between
implementations), JSON log schema, parity-tolerance helpers, matplotlib
plotting helpers, and a tiny C `.npy` writer for prediction output.

See `examples/_shared/DETERMINISM.md` for the determinism contract.
