# MNIST MLP — PyTorch + C Parity Demo

Trains a small dense classifier on MNIST using the factory layer API in both
PyTorch (reference) and the ODT C framework. Replaces the deleted legacy
`example/MnistExperiment`. The framework is 1D-only (no `Conv2d`); this example
treats each `[1,28,28]` image as a flat 784-vector — the `flatten` layer is the
model's first op (no preprocessing reshape).

One binary, two verification modes:

- **Bit-parity** (what CI runs): `BIT_PARITY=1` loads PyTorch's trained weights
  into the C model and runs inference only — C predictions must be
  **bit-identical** to PyTorch's. Deterministic and exact.
- **Train-from-scratch demo**: with no env var the C model trains from its own
  random init; `compare.py` checks final-state parity within tolerance and emits
  plots. Independent init, so it verifies *convergence*, not bits — informational.

## Run it

```bash
uv run python examples/mnist_mlp/prepare_data.py
uv run python examples/mnist_mlp/train_pytorch.py
cmake --preset examples
cmake --build --preset examples --target train_c_mnist_mlp

# Bit-parity (exact — the CI gate)
BIT_PARITY=1 ./build/examples/examples/mnist_mlp/train_c_mnist_mlp
uv run python examples/_shared/compare_predictions.py \
  --pytorch examples/mnist_mlp/outputs/pytorch_predictions.npy \
  --c examples/mnist_mlp/outputs/c_predictions.npy --dtype int32

# …or the train-from-scratch demo + plots (~75 min on full MNIST — slow; bit-parity above is the fast gate)
./build/examples/examples/mnist_mlp/train_c_mnist_mlp
uv run python examples/mnist_mlp/compare.py
```

## Model

- Input: `[1, 28, 28]` (collapsed to `784` by the first `flatten` layer)
- `Flatten → Linear(784→64) → ReLU → Linear(64→10) → Softmax → CrossEntropy`
- ~51 K parameters
- State-dict layers: `fc1` (784→64), `fc2` (64→10)

## Parity tolerance (train-from-scratch demo — informational)

| Metric | Tolerance |
|---|---|
| test_acc  | ±2.5 pp absolute |
| test_loss | ±0.15 nats absolute |

Bit-parity mode requires exact equality instead. See
`examples/_shared/DETERMINISM.md` for the determinism contract.
