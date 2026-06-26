# MNIST 1D-CNN — PyTorch + C Parity Demo

Trains a small 1D convolutional classifier on MNIST. The framework is 1D-only
(no `Conv2d`), so each `[1,28,28]` image is reshaped to a single-channel
length-784 signal — done as loader-side `shape_t` surgery in `train_c.c`
(`reshapeItemsToConv1d`), since the framework has no view/reshape layer and
`flatten` only produces 2D output. Companion to `mnist_mlp/`: same data and
harness, different topology (convolutional vs dense).

One binary, two modes — **bit-parity** (`BIT_PARITY=1`, the exact CI gate) and a
**train-from-scratch** informational demo. See `mnist_mlp/README.md` for the mode
explanation; the run commands are identical with `mnist_cnn` substituted.

## Run it

```bash
uv run python examples/mnist_cnn/prepare_data.py
uv run python examples/mnist_cnn/train_pytorch.py
cmake --preset examples
cmake --build --preset examples --target train_c_mnist_cnn

BIT_PARITY=1 ./build/examples/examples/mnist_cnn/train_c_mnist_cnn
uv run python examples/_shared/compare_predictions.py \
  --pytorch examples/mnist_cnn/outputs/pytorch_predictions.npy \
  --c examples/mnist_cnn/outputs/c_predictions.npy --dtype int32

# …or the train-from-scratch demo (~75 min on full MNIST — slow; bit-parity is the fast gate)
./build/examples/examples/mnist_cnn/train_c_mnist_cnn
uv run python examples/mnist_cnn/compare.py
```

## Model

- Input: `[1, 28, 28]` reshaped to `[1, 784]` (1 channel, length 784)
- `Conv1d(1→8,K3,SAME) → ReLU → MaxPool(2) → Conv1d(8→16,K3,SAME) → ReLU →
  MaxPool(2) → global AvgPool1d → Flatten → Linear(16→10) → Softmax → CE`
- Lengths: 784 → 392 → 196 → 1; ~600 parameters
- State-dict layers: `conv1`, `conv2`, `fc`

Bit-parity mode requires exact equality; the train-from-scratch tolerances match
`mnist_mlp/` and are informational.
