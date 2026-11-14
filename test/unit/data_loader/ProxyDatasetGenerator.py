"""Generate a tiny, deterministic proxy dataset as .npy files for CI-friendly
DataLoader tests. Produces proxy_x.npy (items) and proxy_y.npy (labels), each
of shape (4, 1, 2) and dtype float32, plus proxy_x_chw.npy, image-shaped items
(4, 3, 2, 2) holding arange(48). The (4, 1, 2) values mirror the in-memory proxy
dataset used in UnitTestDataLoader.c, so the two code paths can be kept in
sync by inspection.

Run with:  uv run --with numpy test/unit/data_loader/ProxyDatasetGenerator.py
"""

import os

import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

ITEMS = np.array(
    [
        [[1.0, 2.0]],
        [[3.0, 4.0]],
        [[5.0, 6.0]],
        [[7.0, 8.0]],
    ],
    dtype=np.float32,
)

LABELS = np.array(
    [
        [[0.0, 1.0]],
        [[1.0, 0.0]],
        [[0.0, 1.0]],
        [[1.0, 0.0]],
    ],
    dtype=np.float32,
)

ITEMS_CHW = np.arange(48, dtype=np.float32).reshape(4, 3, 2, 2)


if __name__ == "__main__":
    np.save(os.path.join(BASE_DIR, "proxy_x.npy"), ITEMS)
    np.save(os.path.join(BASE_DIR, "proxy_y.npy"), LABELS)
    np.save(os.path.join(BASE_DIR, "proxy_x_chw.npy"), ITEMS_CHW)
