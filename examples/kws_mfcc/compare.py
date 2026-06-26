"""Compare PyTorch and C runs of the kws_mfcc classifier.

Reads logs/<n>class/{pytorch,c}.json and outputs/<n>class/{pytorch,c}_predictions.npy.
Writes plots into plots/<n>class/. Prints a final-state parity report within tolerances.
INFORMATIONAL only — the bit-parity check (compare_predictions.py) is the gate.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from examples._shared.log_schema import load_log  # noqa: E402
from examples._shared.parity import ParityCheck, run_parity_checks  # noqa: E402
from examples._shared.plotting import (  # noqa: E402
    plot_accuracy_curves,
    plot_confusion_matrix,
    plot_loss_curves,
)

HERE = Path(__file__).resolve().parent
NUM_CLASSES = int(os.environ.get("KWS_CLASSES", "6"))
assert NUM_CLASSES in (6, 35), NUM_CLASSES
TAG = f"{NUM_CLASSES}class"
LOGS = HERE / "logs" / TAG
OUTPUTS = HERE / "outputs" / TAG
PLOTS = HERE / "plots" / TAG
DATA = HERE / "data" / TAG

CLASS_NAMES = (
    ["yes", "no", "up", "down", "silence", "unknown"]
    if NUM_CLASSES == 6
    else [str(i) for i in range(NUM_CLASSES)]
)

CHECKS = [
    ParityCheck("test_acc", abs_tol=0.025),   # ±2.5 pp
    ParityCheck("test_loss", abs_tol=0.15),   # ±0.15 nats (informational)
]


def confusion_matrix(preds: np.ndarray, labels: np.ndarray, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for p, a in zip(preds, labels):
        cm[int(p), int(a)] += 1
    return cm


def main() -> int:
    PLOTS.mkdir(parents=True, exist_ok=True)
    pt = load_log(LOGS / "pytorch.json")
    c = load_log(LOGS / "c.json")

    plot_loss_curves(PLOTS / "loss_curves.png", pt, c)
    plot_accuracy_curves(PLOTS / "accuracy_curves.png", pt, c)

    test_y = np.load(DATA / "test_y.npy")
    pt_pred = np.load(OUTPUTS / "pytorch_predictions.npy")
    c_pred = np.load(OUTPUTS / "c_predictions.npy")
    cm_pt = confusion_matrix(pt_pred, test_y, len(CLASS_NAMES))
    cm_c = confusion_matrix(c_pred, test_y, len(CLASS_NAMES))
    plot_confusion_matrix(PLOTS / "confusion_matrix_pt.png", cm_pt, CLASS_NAMES, "PyTorch KWS MFCC")
    plot_confusion_matrix(PLOTS / "confusion_matrix_c.png", cm_c, CLASS_NAMES, "C KWS MFCC")

    pt_finals = pt["final"]
    c_finals = c["final"]
    overall_pass, results = run_parity_checks(
        CHECKS,
        {"test_acc": pt_finals["test_acc"], "test_loss": pt_finals["test_loss"]},
        {"test_acc": c_finals["test_acc"], "test_loss": c_finals["test_loss"]},
    )

    print("\nParity report (PyTorch vs C) — INFORMATIONAL:")
    print(f"{'metric':<14} {'pt':>10} {'c':>10} {'diff':>10} {'tol':>8} {'type':>5} {'pass':>6}")
    for r in results:
        print(f"{r.metric:<14} {r.pt_value:>10.5f} {r.c_value:>10.5f} {r.diff:>10.5f} "
              f"{r.tolerance:>8.4f} {r.tolerance_type:>5} {str(r.passed):>6}")
    print(f"\nOverall: {'PASS' if overall_pass else 'FAIL'} (informational; not a CI gate)")
    return 0 if overall_pass else 1


if __name__ == "__main__":
    sys.exit(main())
