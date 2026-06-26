"""Shared MNIST loader for the mnist_mlp and mnist_cnn examples.

Wraps torchvision.datasets.MNIST so both examples download/cache once and
deliver identical arrays. Images are float32 [N,1,28,28] in [0,1]; labels are
int32 [N] (0..9). Reshaping into each model's input geometry is the first layer
of the model (flatten for the MLP) or loader-side shape surgery (the CNN), per
the repo's data-shape convention — not done here.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from torchvision import datasets, transforms

NUM_CLASSES = 10


def load_mnist(root: str | Path, split: str) -> tuple[np.ndarray, np.ndarray]:
    assert split in ("train", "test"), split
    ds = datasets.MNIST(
        root=str(root), train=(split == "train"),
        download=True, transform=transforms.ToTensor(),
    )
    n = len(ds)
    images = np.empty((n, 1, 28, 28), dtype=np.float32)
    labels = np.empty((n,), dtype=np.int32)
    for i in range(n):
        x, y = ds[i]
        images[i] = x.numpy()
        labels[i] = y
    return images, labels
