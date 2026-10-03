# Mixed-Width MLP — C-only Acceptance Demo

Trains the `mnist_mlp` topology with **mixed SYM_INT32 storage widths and
per-op arithmetic choices** — the acceptance run for the arithmetic/storage
split (`arithmetic_t` per op, declared-width storage per tensor). There is no
PyTorch twin and no bit-parity mode: mixed SYM_INT32/FLOAT32 wires diverge
from PyTorch by design, so the binary checks itself with in-process hard gates
instead. See [`../README.md`](../README.md) for the other examples and
[`docs/conventions/arithmetic-sym.md`](../../docs/conventions/arithmetic-sym.md)
for the SYM rules it exercises.

## Run it

```bash
# Reads examples/mnist_mlp/data/train_{x,y}.npy — prepare that dataset first.
uv run examples/mnist_mlp/prepare_data.py

cmake --preset examples
cmake --build --preset examples --target train_c_mixed_width_mlp
./build/examples/examples/mixed_width_mlp/train_c_mixed_width_mlp   # from the repo root
```

No env knobs: fixed seed 1, SGD lr 0.01 with FLOAT32 momentum 0.9, the first
256 training samples, batch 1, one pass. Output is stdout only
(`initial_loss`, `GATES PASS: …`, `final_loss`, `GATE PASS: …`); no log or
prediction files.

## Model and widths

`Flatten → Linear(784→64) → ReLU → Linear(64→10) → Quant → Softmax`, CrossEntropy loss.

| Tensor | Storage |
|---|---|
| Linear weights | SYM_INT32 @ 8 bits |
| Linear biases | SYM_INT32 @ 16 bits |
| Linear weight/bias grads | packed SYM @ 8 bits |
| Input, Linear/ReLU outputs, every dx wire | SYM_INT32 @ 12 bits |
| Quant / Softmax outputs | FLOAT32 |

Per-op arithmetic: both Linear layers run their forward in `ARITH_SYM_INT32`,
their weight-grad and dx ops in `ARITH_FLOAT32` and their bias-grad op in
`ARITH_SYM_INT32`; ReLU runs SYM_INT32 forward and backward; Softmax is
FLOAT32. The Quant layer is the one genuine SYM_INT32 → FLOAT32 dtype change,
so the model carries heterogeneous wire dtypes. Linear params are created with
FLOAT32 storage (the random-init factories require it) and then requantized
in place to the widths above — the documented route to SYM_INT32-native
params. The 256 input samples are requantized to SYM_INT32 @ 12 up front.

## Gates

On any failure the binary prints the reason and exits 1:

- after the first training step: the Linear0 forward wire is SYM_INT32 @ 12,
  every Linear param and grad has the expected dtype and width, and the packed
  grads total exactly 50 890 bytes;
- after training: the loss on the 256-sample subset is lower than before
  training (a sanity check — no pinned float values, since libm differs across
  platforms).

CI runs the binary in the `c-bit-parity` job as an acceptance step (not a
parity comparison).
