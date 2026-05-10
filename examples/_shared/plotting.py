"""Matplotlib helpers for compare.py across all four examples.

Convention: PyTorch curves are solid lines; C curves are dashed lines.
Two colors per chart, one per split (train/val).
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")  # no GUI in CI / headless dev shells
import matplotlib.pyplot as plt
import numpy as np

from examples._shared.log_schema import RunLog


def plot_loss_curves(out_path: Path | str, pt_log: RunLog, c_log: RunLog) -> None:
    pt_train = [e["train_loss"] for e in pt_log["epochs"]]
    pt_val   = [e["val_loss"]   for e in pt_log["epochs"]]
    c_train  = [e["train_loss"] for e in c_log["epochs"]]
    c_val    = [e["val_loss"]   for e in c_log["epochs"]]
    n = max(len(pt_train), len(c_train))
    x = list(range(n))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x[:len(pt_train)], pt_train, "-",  color="#1f77b4", label="PyTorch train")
    ax.plot(x[:len(pt_val)],   pt_val,   "-",  color="#ff7f0e", label="PyTorch val")
    ax.plot(x[:len(c_train)],  c_train,  "--", color="#1f77b4", label="C train")
    ax.plot(x[:len(c_val)],    c_val,    "--", color="#ff7f0e", label="C val")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title(f"{pt_log['example']} — train/val loss (PyTorch solid, C dashed)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_accuracy_curves(out_path: Path | str, pt_log: RunLog, c_log: RunLog) -> None:
    pt_acc = [e["val_acc"] for e in pt_log["epochs"]]
    c_acc  = [e["val_acc"] for e in c_log["epochs"]]
    n = max(len(pt_acc), len(c_acc))
    x = list(range(n))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x[:len(pt_acc)], pt_acc, "-",  color="#2ca02c", label="PyTorch val acc")
    ax.plot(x[:len(c_acc)],  c_acc,  "--", color="#2ca02c", label="C val acc")
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation accuracy")
    ax.set_title(f"{pt_log['example']} — validation accuracy (PyTorch solid, C dashed)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_confusion_matrix(
    out_path: Path | str,
    cm: np.ndarray,  # shape [num_classes, num_classes]; cm[i, j] = count predicted=i, actual=j
    class_names: Iterable[str],
    title: str,
) -> None:
    cm = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    names = list(class_names)
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_yticklabels(names)
    ax.set_xlabel("actual")
    ax.set_ylabel("predicted")
    ax.set_title(title)
    threshold = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > threshold else "black", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_reconstructions(
    out_path: Path | str,
    test_inputs: np.ndarray,    # shape [N, 1, L]
    pt_recons: np.ndarray,      # shape [N, 1, L]
    c_recons: np.ndarray,       # shape [N, 1, L]
    normal_idx: Iterable[int],  # length-K indices into test_inputs of class-1 samples
    anomaly_idx: Iterable[int], # length-K indices of non-class-1 samples
    title: str = "Reconstructions",
) -> None:
    """Overlay input / PyTorch-recon / C-recon for K normal + K anomaly samples.

    Layout: 2 rows × K columns. Row 0 = normal examples, row 1 = anomaly
    examples. Each subplot has three lines: input (black solid), PT-recon
    (blue dashed), C-recon (orange dashed).
    """
    normal_idx = list(normal_idx)
    anomaly_idx = list(anomaly_idx)
    K = min(len(normal_idx), len(anomaly_idx))
    if K == 0:
        raise ValueError("plot_reconstructions: need >=1 normal and >=1 anomaly sample")

    fig, axes = plt.subplots(2, K, figsize=(2.5 * K, 4), sharey=True)
    if K == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    for col, ix in enumerate(normal_idx[:K]):
        ax = axes[0, col]
        ax.plot(test_inputs[ix, 0], "-",  color="black",  linewidth=1.0, label="input")
        ax.plot(pt_recons[ix, 0],   "--", color="#1f77b4", linewidth=1.0, label="PT")
        ax.plot(c_recons[ix, 0],    "--", color="#ff7f0e", linewidth=1.0, label="C")
        ax.set_title(f"normal #{ix}", fontsize=8)
        ax.tick_params(labelsize=7)
        if col == 0:
            ax.set_ylabel("normal", fontsize=9)
            ax.legend(fontsize=7, loc="upper right")

    for col, ix in enumerate(anomaly_idx[:K]):
        ax = axes[1, col]
        ax.plot(test_inputs[ix, 0], "-",  color="black",  linewidth=1.0)
        ax.plot(pt_recons[ix, 0],   "--", color="#1f77b4", linewidth=1.0)
        ax.plot(c_recons[ix, 0],    "--", color="#ff7f0e", linewidth=1.0)
        ax.set_title(f"anomaly #{ix}", fontsize=8)
        ax.tick_params(labelsize=7)
        if col == 0:
            ax.set_ylabel("anomaly", fontsize=9)

    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_anomaly_score_hist(
    out_path: Path | str,
    pt_scores: np.ndarray,      # shape [N], per-test-sample reconstruction MSE (PyTorch)
    c_scores: np.ndarray,       # shape [N], same (C)
    test_labels: np.ndarray,    # shape [N], class IDs in {1, 2, 3, 4, 5}
    class_names: Iterable[str], # e.g. ["normal", "R-on-T", "PVC", "SP", "UB"]
    title: str = "Reconstruction-MSE histograms (per class)",
) -> None:
    """Side-by-side PyTorch / C histograms of reconstruction MSE, stacked by class.

    Two subplots; each has one histogram per class colored differently,
    superimposed in `histtype='step'` so distributions are comparable.
    """
    class_names = list(class_names)
    unique_labels = sorted(set(int(v) for v in test_labels))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True, sharey=True)

    cmap = plt.get_cmap("tab10")
    lo = float(min(pt_scores.min(), c_scores.min()))
    hi = float(max(pt_scores.max(), c_scores.max()))
    if hi <= lo:
        hi = lo + 1.0  # degenerate; render a single visible bar
    bins = np.linspace(lo, hi, 50)

    for ax, scores, label in zip(axes, [pt_scores, c_scores], ["PyTorch", "C"]):
        for ci, cls in enumerate(unique_labels):
            mask = test_labels == cls
            name = class_names[cls - 1] if 1 <= cls <= len(class_names) else f"class {cls}"
            ax.hist(
                scores[mask], bins=bins, histtype="step", linewidth=1.5,
                color=cmap(ci % 10), label=name,
            )
        ax.set_title(label)
        ax.set_xlabel("reconstruction MSE")
        ax.legend(fontsize=8)
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("count (log)")
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
