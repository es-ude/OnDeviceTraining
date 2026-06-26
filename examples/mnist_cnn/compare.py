"""Compare PyTorch and C runs of the MNIST 1D-CNN classifier.

Reads logs/{pytorch,c}.json and outputs/{pytorch,c}_predictions.npy.
Writes plots into plots/. Prints a final-state parity report within tolerances.
INFORMATIONAL only — the bit-parity check (compare_predictions.py) is the gate.
"""
from __future__ import annotations

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
LOGS = HERE / "logs"
OUTPUTS = HERE / "outputs"
PLOTS = HERE / "plots"
DATA = HERE / "data"

CLASS_NAMES = [str(d) for d in range(10)]

CHECKS = [
    ParityCheck("test_acc", abs_tol=0.025),   # ±2.5 pp
    ParityCheck("test_loss", abs_tol=0.15),   # ±0.15 nats (HAR-calibrated; informational)
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
    plot_confusion_matrix(PLOTS / "confusion_matrix_pt.png", cm_pt, CLASS_NAMES, "PyTorch MNIST CNN")
    plot_confusion_matrix(PLOTS / "confusion_matrix_c.png", cm_c, CLASS_NAMES, "C MNIST CNN")

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
