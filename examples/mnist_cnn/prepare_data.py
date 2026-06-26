"""Prepare MNIST for the mnist_cnn example.

Output (under examples/mnist_cnn/data/):
  train_x.npy [N,1,28,28] f32   train_y.npy [N] i32 (0..9)
  val_x.npy, val_y.npy   (10% of train, deterministic via SHUFFLE_SEED)
  test_x.npy, test_y.npy
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from examples._shared.mnist_data import load_mnist  # noqa: E402
from examples._shared.seeds import SHUFFLE_SEED  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
RAW_DIR = DATA_DIR / "raw"


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    train_x, train_y = load_mnist(RAW_DIR, "train")
    test_x, test_y = load_mnist(RAW_DIR, "test")

    rng = np.random.default_rng(SHUFFLE_SEED)
    perm = rng.permutation(train_x.shape[0])
    n_val = train_x.shape[0] // 10
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    val_x, val_y = train_x[val_idx], train_y[val_idx]
    train_x, train_y = train_x[train_idx], train_y[train_idx]

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.save(DATA_DIR / "train_x.npy", train_x)
    np.save(DATA_DIR / "train_y.npy", train_y)
    np.save(DATA_DIR / "val_x.npy", val_x)
    np.save(DATA_DIR / "val_y.npy", val_y)
    np.save(DATA_DIR / "test_x.npy", test_x)
    np.save(DATA_DIR / "test_y.npy", test_y)
    print(f"train: {train_x.shape}, val: {val_x.shape}, test: {test_x.shape}", flush=True)


if __name__ == "__main__":
    main()
