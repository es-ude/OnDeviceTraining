# Examples — End-to-End Demos with PyTorch + C Parity

Each subdirectory is a self-contained example demonstrating the same model
in PyTorch (reference) and the ODT C framework, with final-state parity
checking and visualizations — except `mixed_width_mlp/`, which is a C-only
acceptance demo with no PyTorch twin.

## Currently shipping

| Directory | Task | Status |
|---|---|---|
| `mnist_mlp/` | MNIST dense-MLP digit classification | ✅ |
| `mnist_cnn/` | MNIST 1D-CNN digit classification | ✅ |
| `har_classifier/` | UCI HAR 6-class activity classification | Stage 1 ✅ |
| `ecg_anomaly_ae/` | ECG5000 reconstruction-based anomaly detection | Stage 2 ✅ |
| `kws_mfcc/` | SpeechCommands keyword spotting (MFCC features) | Stage 3 ✅ |
| `kws_raw/` | SpeechCommands keyword spotting (raw waveform + in-model downsample) | Stage 3 ✅ |
| `mixed_width_mlp/` | Mixed-width SYM_INT32 MLP on MNIST (C-only, no PyTorch twin) | ✅ |
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

`mixed_width_mlp/` has no Python steps: build its target and run the binary
directly. It reads `examples/mnist_mlp/data/`, so run the `mnist_mlp`
prepare step first (see [`mixed_width_mlp/README.md`](mixed_width_mlp/README.md)).

Each `train_c_<name>` binary except `train_c_mixed_width_mlp` also has a
**bit-parity** mode: run it with `BIT_PARITY=1` and it loads the PyTorch
reference weights (instead of training from scratch) and emits predictions
that CI compares against PyTorch's — classification predictions (int32) must
match exactly; the ECG autoencoder's float32 reconstructions are checked with
`allclose` (`rtol=1e-4`, `atol=1e-5`). This is the deterministic check CI
runs; see each example's README for the precise `compare_predictions.py`
invocation.

Some examples build additional targets beyond `train_c_<name>`:

- `har_classifier/`: `train_c_har_classifier_adamw`, `_finetune`, `_sym`,
  `_bfp` (optimizer, freezing and quantization studies) and
  `train_c_continual` (continual learning with PPCA replay) — see
  [`har_classifier/README.md`](har_classifier/README.md).
- `kws_raw/`: `trace_c_kws_raw`, a per-layer C-vs-PyTorch trace of one
  training step — see [`kws_raw/README.md`](kws_raw/README.md).

The C-side executables only build when configured with `BUILD_EXAMPLES=ON`
(the `examples` or `examples_memprofile` presets); the default `unit_test_*`
presets do not build them.

## Shared infrastructure

All examples share `examples/_shared/`: seed constants, the Python mirror
of the C `XorShift32` RNG (so train-set shuffle order matches between
implementations), JSON log schema, parity-tolerance helpers, matplotlib
plotting helpers, and a tiny C `.npy` writer for prediction output.

See `examples/_shared/DETERMINISM.md` for the determinism contract.
