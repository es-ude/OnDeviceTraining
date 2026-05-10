"""Prepare the ECG5000 dataset (UCR Time-Series Classification archive).

Output (under examples/ecg_anomaly_ae/data/):
  train_x.npy  shape [N_train_normal, 1, 140] float32  (class-1 only)
  val_x.npy    shape [N_val,          1, 140] float32  (10 % carved off train_x)
  test_x.npy   shape [N_test,         1, 140] float32  (all 5 classes)
  test_y.npy   shape [N_test]                int32     values 1..5

The training set is filtered to class 1 ("normal") only — that is the
standard reconstruction-AE-as-anomaly-detector recipe. The test set
keeps all classes so anomaly detection can be evaluated against the
{normal, anomaly} binary label (test_y != 1).
"""
from __future__ import annotations

import sys
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from examples._shared.seeds import SHUFFLE_SEED  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
RAW_DIR = DATA_DIR / "raw"
ZIP_URL = "http://www.timeseriesclassification.com/aeon-toolkit/ECG5000.zip"
ZIP_PATH = RAW_DIR / "ECG5000.zip"
TRAIN_TXT = RAW_DIR / "ECG5000_TRAIN.txt"
TEST_TXT = RAW_DIR / "ECG5000_TEST.txt"

NORMAL_CLASS = 1
SAMPLE_LENGTH = 140
VAL_FRACTION = 0.1


def _download_if_missing() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if not ZIP_PATH.exists():
        print(f"downloading {ZIP_URL} -> {ZIP_PATH}", flush=True)
        urllib.request.urlretrieve(ZIP_URL, ZIP_PATH)
    if not TRAIN_TXT.exists() or not TEST_TXT.exists():
        print(f"extracting {ZIP_PATH} -> {RAW_DIR}", flush=True)
        with zipfile.ZipFile(ZIP_PATH) as zf:
            zf.extractall(RAW_DIR)


def _parse_partition(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Parse an ECG5000_*.txt file. Returns (x: float32 [N, 1, 140], y: int32 [N])."""
    raw = np.loadtxt(path, dtype=np.float32)
    assert raw.shape[1] == SAMPLE_LENGTH + 1, raw.shape
    y = raw[:, 0].astype(np.int32)
    x = raw[:, 1:].reshape(-1, 1, SAMPLE_LENGTH)
    return x, y


def main() -> None:
    _download_if_missing()
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    train_x_full, train_y_full = _parse_partition(TRAIN_TXT)
    test_x, test_y = _parse_partition(TEST_TXT)

    # Filter train to class-1 (normal) only.
    normal_mask = train_y_full == NORMAL_CLASS
    train_x_normal = train_x_full[normal_mask]
    print(
        f"train_full={train_x_full.shape[0]}, "
        f"train_normal={train_x_normal.shape[0]}, "
        f"test={test_x.shape[0]}",
        flush=True,
    )
    assert train_x_normal.shape[0] > 0, "train set has no class-1 (normal) samples?"

    # Carve VAL_FRACTION of the normal training samples as validation.
    rng = np.random.default_rng(SHUFFLE_SEED)
    perm = rng.permutation(train_x_normal.shape[0])
    n_val = max(1, int(round(VAL_FRACTION * train_x_normal.shape[0])))
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    train_x = train_x_normal[train_idx]
    val_x = train_x_normal[val_idx]

    np.save(DATA_DIR / "train_x.npy", train_x)
    np.save(DATA_DIR / "val_x.npy", val_x)
    np.save(DATA_DIR / "test_x.npy", test_x)
    np.save(DATA_DIR / "test_y.npy", test_y)
    print(
        f"train: {train_x.shape}, val: {val_x.shape}, "
        f"test: {test_x.shape}, test labels in {sorted(set(test_y.tolist()))}",
        flush=True,
    )


if __name__ == "__main__":
    main()
