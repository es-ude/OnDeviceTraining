"""Prepare the UCI HAR Dataset.

Output (under examples/har_classifier/data/):
  train_x.npy  shape [N_train, 9, 128] float32
  train_y.npy  shape [N_train]         int32  values 0..5
  val_x.npy, val_y.npy   (10% of train, by sample order -- not by subject)
  test_x.npy, test_y.npy

Channels: body_acc_{x,y,z}, body_gyro_{x,y,z}, total_acc_{x,y,z}.
"""
from __future__ import annotations

import sys
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from examples._shared.seeds import SPLIT_SEED  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
RAW_DIR = DATA_DIR / "raw"
ZIP_URL = "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip"
ZIP_PATH = RAW_DIR / "har.zip"
EXTRACTED_DIR = RAW_DIR / "UCI HAR Dataset"

CHANNELS = (
    "body_acc_x", "body_acc_y", "body_acc_z",
    "body_gyro_x", "body_gyro_y", "body_gyro_z",
    "total_acc_x", "total_acc_y", "total_acc_z",
)


def _download_if_missing() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if not ZIP_PATH.exists():
        print(f"downloading {ZIP_URL} -> {ZIP_PATH}", flush=True)
        urllib.request.urlretrieve(ZIP_URL, ZIP_PATH)
    if not EXTRACTED_DIR.exists():
        print(f"extracting {ZIP_PATH} -> {RAW_DIR}", flush=True)
        with zipfile.ZipFile(ZIP_PATH) as zf:
            zf.extractall(RAW_DIR)
        # The dataset zip nests another zip inside; modern UCI mirrors may not.
        nested = RAW_DIR / "UCI HAR Dataset.zip"
        if nested.exists():
            with zipfile.ZipFile(nested) as zf:
                zf.extractall(RAW_DIR)


def _load_partition(partition: str) -> tuple[np.ndarray, np.ndarray]:
    base = EXTRACTED_DIR / partition / "Inertial Signals"
    arrays = []
    for ch in CHANNELS:
        p = base / f"{ch}_{partition}.txt"
        arrays.append(np.loadtxt(p, dtype=np.float32))
    x = np.stack(arrays, axis=1)  # [N, 9, 128]
    assert x.shape[1] == 9 and x.shape[2] == 128, x.shape
    y = np.loadtxt(EXTRACTED_DIR / partition / f"y_{partition}.txt", dtype=np.int32) - 1  # 1..6 -> 0..5
    assert y.shape[0] == x.shape[0]
    return x, y


def main() -> None:
    _download_if_missing()
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    train_x, train_y = _load_partition("train")
    test_x, test_y = _load_partition("test")

    # Carve 10% of train as validation, deterministic via SPLIT_SEED.
    rng = np.random.default_rng(SPLIT_SEED)
    perm = rng.permutation(train_x.shape[0])
    n_val = train_x.shape[0] // 10
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    val_x, val_y = train_x[val_idx], train_y[val_idx]
    train_x, train_y = train_x[train_idx], train_y[train_idx]

    np.save(DATA_DIR / "train_x.npy", train_x)
    np.save(DATA_DIR / "train_y.npy", train_y)
    np.save(DATA_DIR / "val_x.npy", val_x)
    np.save(DATA_DIR / "val_y.npy", val_y)
    np.save(DATA_DIR / "test_x.npy", test_x)
    np.save(DATA_DIR / "test_y.npy", test_y)
    print(f"train: {train_x.shape}, val: {val_x.shape}, test: {test_x.shape}", flush=True)


if __name__ == "__main__":
    main()
