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


# ---------------------------------------------------------------------------
# HAR memory-sweep pictograms (consume compare_memory.py's aggregate dict).
#
# These three are theme-aware (accept dark=) — the older five above are not, so
# theming is opt-in and additive. Colors are the Okabe-Ito colorblind-safe set.
# The aggregate `agg` shape is {"per_config": {cfg: {"stats": {metric:
# {"mean","std"}}, ...}}, "comparisons": {cfg: {"weight_bytes_drop_pct", ...}}}.
# ---------------------------------------------------------------------------

# Real run_matrix.py sweep set (#322 — was drifted: sym16 never ran, sym10/sym6 missing).
_MEM_CONFIG_ORDER = [
    "float", "sym12", "sym10", "sym8", "sym6", "sym4", "sym8cos", "sym4cos",
    # #300 granularity axis — keep in sync with CONFIG_ORDER in compare_memory.py.
    "sym6pc", "sym6g64", "sym6g32",
    "sym4pc", "sym4g64", "sym4g32",
    "asym6", "asym6pc", "asym6g64", "asym6g32",
    "asym4", "asym4pc", "asym4g64", "asym4g32",
]
# MUST stay in sync with CATEGORIES in compare_memory.py: the stacked bar labels each
# bar with the sum of these, so an omission here silently prints a total that
# disagrees with mcu_total_b in the summary JSON.
_MEM_CATEGORIES = ["params_b", "group_overhead_b", "grads_b", "optstate_analytic_b",
                   "activations_b", "io_b", "pool_backward_b", "dx_peak_b"]
# Okabe-Ito palette, one per analytic category (colorblind-safe, distinguishable).
_MEM_CAT_COLORS = {
    "params_b": "#0072B2",             # blue   — the category that shrinks
    # Brown, NOT the 8th Okabe-Ito yellow (#F0E442 fails the lightness band at
    # 1.29:1 contrast on the light surface) and NOT vermillion (already taken by
    # pool_backward_b below — a duplicate would render two segments identically).
    # Validated at its insertion slot: params | THIS | grads.
    "group_overhead_b": "#A6761D",     # brown — group scale/zp metadata (#300)
    "grads_b": "#E69F00",              # orange
    "optstate_analytic_b": "#009E73",  # green
    "activations_b": "#CC79A7",        # pink   — dominates at batch 64
    "io_b": "#999999",                 # grey
    "pool_backward_b": "#D55E00",      # vermillion — MaxPool argmax backward state (#321)
    "dx_peak_b": "#56B4E9",            # sky blue   — dx ping-pong transient (#321)
}
# The weight chart is a TWO-series part-to-whole, not a slice of the 8-category
# stack, so it gets its own validated pair (all checks pass, light AND dark)
# rather than borrowing two hues tuned for 8-way separation.
_WEIGHT_SPLIT_COLORS = {"payload": "#0072B2", "metadata": "#D55E00"}


def _mem_theme(dark: bool) -> tuple[str, str, str]:
    """Return (figure/axes facecolor, foreground text/line color, grid color)."""
    if dark:
        return "#1e1e1e", "#e0e0e0", "#555555"
    return "#ffffff", "#222222", "#cccccc"


def _mem_configs_present(agg: dict) -> list[str]:
    per = agg["per_config"]
    ordered = [c for c in _MEM_CONFIG_ORDER if c in per]
    return ordered + [c for c in per if c not in _MEM_CONFIG_ORDER]


def _mem_apply_theme(fig, ax, bg: str, fg: str, grid: str) -> None:
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)
    for spine in ax.spines.values():
        spine.set_color(fg)
    ax.tick_params(colors=fg)
    ax.xaxis.label.set_color(fg)
    ax.yaxis.label.set_color(fg)
    ax.title.set_color(fg)
    ax.grid(True, alpha=0.3, color=grid)


def plot_peak_ram_stacked_bar(out_path: Path | str, agg: dict, dark: bool = False) -> None:
    """One stacked bar per config: analytic MCU footprint by category (mean bytes).

    Activations are sized at micro-batch B=1 (the loop streams the macro-batch one
    sample at a time), so the categories are roughly balanced — params/grads/
    momentum each carry ~a third — and the packed-SYM weight cut is a visible slice
    of the whole bar (the weight-compression bar shows that slice on its own axis).
    """
    bg, fg, grid = _mem_theme(dark)
    configs = _mem_configs_present(agg)
    per = agg["per_config"]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(configs))
    bottom = np.zeros(len(configs))
    for cat in _MEM_CATEGORIES:
        vals = np.array([per[c]["stats"][cat]["mean"] for c in configs])
        ax.bar(x, vals, bottom=bottom, width=0.62,
               # removesuffix, not replace: "pool_backward_b".replace("_b","") also
               # eats the "_b" inside "_backward" and renders as "poolackward".
               color=_MEM_CAT_COLORS[cat], label=cat.removesuffix("_b"))
        bottom += vals
    for xi, total in zip(x, bottom):
        ax.text(xi, total, f"{total / (1 << 20):.2f}MB", ha="center", va="bottom",
                fontsize=8, color=fg)

    ax.set_xticks(x)
    ax.set_xticklabels(configs)
    ax.set_ylabel("bytes")
    ax.set_ylim(0, bottom.max() * 1.12)
    ax.set_title("HAR MCU memory footprint by category (mean over seeds, micro-batch B=1)")
    _mem_apply_theme(fig, ax, bg, fg, grid)
    leg = ax.legend(fontsize=8, ncol=5, loc="upper center", framealpha=0.3)
    for text in leg.get_texts():
        text.set_color(fg)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, facecolor=bg)
    plt.close(fig)


def plot_accuracy_vs_memory_scatter(out_path: Path | str, agg: dict, dark: bool = False) -> None:
    """Test accuracy vs total on-device footprint (mcu_total_b) — the memory/accuracy Pareto.

    x is the analytic on-device footprint (params+grads+momentum+activations+io at
    micro-batch B=1) — the honest 'how much RAM' axis, which moves materially once
    activations are sized per-sample. Mean point per config with +/- std error bars.
    (The weight-compression bar shows the weight category alone.)
    """
    bg, fg, grid = _mem_theme(dark)
    configs = _mem_configs_present(agg)
    per = agg["per_config"]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    cmap = plt.get_cmap("viridis")
    for i, c in enumerate(configs):
        st = per[c]["stats"]
        # #322: mcu_total_b is an analytic formula — bit-identical across seeds, so it has
        # no measurement error. Only y (test accuracy) carries a seed-std bar; a deterministic
        # xerr would falsely imply the footprint was measured-with-uncertainty.
        xm = st["mcu_total_b"]["mean"]
        ym, ys = st["test_acc"]["mean"], st["test_acc"]["std"]
        color = cmap(i / max(len(configs) - 1, 1))
        ax.errorbar(xm / 1024.0, ym, yerr=ys, fmt="o", markersize=8,
                    color=color, ecolor=color, capsize=3, elinewidth=1)
        ax.annotate(c, (xm / 1024.0, ym), textcoords="offset points", xytext=(8, 4),
                    fontsize=9, color=fg)

    # #322: mcu_total_b is the HEAP categories only; the training-step stack high-water
    # (stack_peak_b, reported separately) is excluded — say so on the axis.
    ax.set_xlabel("on-device heap footprint (KB, B=1) — excl. training-step stack")
    ax.set_ylabel("test accuracy")
    ax.set_title("Accuracy vs on-device memory (packed SYM@x vs FLOAT32)")
    _mem_apply_theme(fig, ax, bg, fg, grid)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, facecolor=bg)
    plt.close(fig)


def plot_weight_compression_bar(out_path: Path | str, agg: dict, dark: bool = False) -> None:
    """Weight storage per config, split into packed payload + group metadata.

    The compression 'money shot' on its own axis — invisible inside the
    activation-dominated stacked bar. STACKED on purpose (#300): a group config
    buys accuracy with scale/zero-point bytes, and at 4 bits that surcharge
    reaches ~24% of the payload. Drawing params_b alone made per-tensor,
    per-channel, g64 and g32 render as four identical bars, i.e. it showed finer
    granularity as free — the exact claim this chart exists to test.

    Bars annotated with the mean TOTAL weight bytes (payload + metadata) and,
    where a baseline exists, the percent reduction vs FLOAT32.
    """
    bg, fg, grid = _mem_theme(dark)
    configs = _mem_configs_present(agg)
    per = agg["per_config"]
    comps = agg.get("comparisons", {})

    payload = np.array([per[c]["stats"]["params_b"]["mean"] for c in configs])
    meta = np.array([per[c]["stats"]["group_overhead_b"]["mean"] for c in configs])
    totals = payload + meta

    # Widen with the config count: the #300 sweep is 16 arms, the pre-#300 set was 8.
    fig, ax = plt.subplots(figsize=(max(7.0, 0.62 * len(configs) + 2.0), 4.8))
    x = np.arange(len(configs))
    # edgecolor=bg draws the surface-colored gap the design system wants between
    # stacked segments, so the two fills never touch.
    ax.bar(x, payload, width=0.6, color=_WEIGHT_SPLIT_COLORS["payload"],
           edgecolor=bg, linewidth=1.2, label="packed payload")
    ax.bar(x, meta, bottom=payload, width=0.6, color=_WEIGHT_SPLIT_COLORS["metadata"],
           edgecolor=bg, linewidth=1.2, label="group scale/zp metadata")

    for xi, c, total in zip(x, configs, totals):
        label = f"{total / 1024:.1f}KB"
        if c in comps:
            label += f"\n-{comps[c]['weight_bytes_drop_pct']:.0f}%"
        ax.text(xi, total, label, ha="center", va="bottom", fontsize=7.5, color=fg)

    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("weight storage (bytes)")
    ax.set_title("Weight storage: packed payload + group metadata")
    ax.set_ylim(0, totals.max() * 1.20)
    _mem_apply_theme(fig, ax, bg, fg, grid)
    leg = ax.legend(fontsize=8, loc="upper right", framealpha=0.3)
    for text in leg.get_texts():
        text.set_color(fg)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, facecolor=bg)
    plt.close(fig)
