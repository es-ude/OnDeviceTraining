"""Paper-ready figures (PDF + SVG) for the #326 continual-learning study.

Usage (from examples/har_classifier/):
  uv run plot_continual.py --logs logs --out logs/figures --seeds 42 43 44 45 46 47 48 49 50 51

Expects the per-arm logs written by train_c_continual:
  logs/continual_main/replay{0,1}_seed{s}.json    no-replay / PPCA replay
  logs/continual_mean/mean_seed{s}.json           mean-replay arm (optional)
  logs/continual_exemplar/exemplar_seed{s}.json   exemplar-replay arm (optional)
  logs/continual_calib/ep{e}_replay{r}_seed{s}.json  calibration sweep (optional)

Emits vector figures (editable text, no rasterization):
  fig_forgetting_curves, fig_accuracy_matrices, fig_delta_matrices,
  fig_paired_bwt, fig_calibration  (each as .pdf and .svg)
"""

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# arm key -> (subdir, filename pattern, display label, hex color)
ARMS = {
    "none": ("continual_main", "replay0_seed{s}.json", "no replay", "#2a78d6"),
    "mean": ("continual_mean", "mean_seed{s}.json", "mean replay", "#eda100"),
    "exemplar": ("continual_exemplar", "exemplar_seed{s}.json", "exemplar replay", "#008300"),
    "ppca": ("continual_main", "replay1_seed{s}.json", "PPCA replay", "#1baf7a"),
}

FULL_W = 7.16  # IEEE double-column text width in inches
INK = "#333333"


def paper_style() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": INK,
        "text.color": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "pdf.fonttype": 42,  # embed TrueType -> text stays editable
        "svg.fonttype": "none",
        "figure.dpi": 200,
    })


def load_arms(logs: Path, seeds: list[int]) -> dict[str, list[list[list[float]]]]:
    arms = {}
    for key, (subdir, pattern, _, _) in ARMS.items():
        paths = [logs / subdir / pattern.format(s=s) for s in seeds]
        if not all(p.is_file() for p in paths):
            print(f"  arm '{key}': logs missing under {logs / subdir} -> skipped")
            continue
        arms[key] = [json.loads(p.read_text())["accuracy_matrix"] for p in paths]
    return arms


def metrics(R: list[list[float]]) -> tuple[float, float]:
    T = len(R)
    acc = float(np.mean(R[T - 1]))
    bwt = float(np.mean([R[T - 1][j] - R[j][j] for j in range(T - 1)]))
    return acc, bwt


def mean_matrix(mats: list[list[list[float]]]) -> np.ndarray:
    T = len(mats[0])
    M = np.full((T, T), np.nan)
    for t in range(T):
        for j in range(t + 1):
            M[t, j] = float(np.mean([m[t][j] for m in mats]))
    return M


def save(fig: plt.Figure, out: Path, name: str) -> None:
    for ext in ("pdf", "svg"):
        fig.savefig(out / f"{name}.{ext}")
    plt.close(fig)
    print(f"  wrote {out / name}.pdf/.svg")


def fig_forgetting_curves(arms: dict, out: Path) -> None:
    T = len(next(iter(arms.values()))[0])
    fig, axes = plt.subplots(1, T - 1, figsize=(FULL_W, 1.9), sharey=True,
                             constrained_layout=True)
    for j, ax in enumerate(np.atleast_1d(axes)):
        for key, mats in arms.items():
            _, _, label, color = ARMS[key]
            ts = list(range(j, T))
            m = np.array([[mm[t][j] for mm in mats] for t in ts])
            mean, std = m.mean(axis=1), m.std(axis=1, ddof=1)
            ax.fill_between(ts, mean - std, mean + std, color=color, alpha=0.15, lw=0)
            ax.plot(ts, mean, color=color, lw=1.1, marker="o", ms=2.2, label=label)
        ax.set_title(f"evaluated on $D_{j}$")
        ax.set_xticks(range(T), [f"$D_{t}$" for t in range(T)])
        ax.set_xlabel("trained through domain")
        ax.set_xlim(-0.2, T - 0.8)
    np.atleast_1d(axes)[0].set_ylabel("held-out accuracy")
    np.atleast_1d(axes)[0].legend(loc="lower left", frameon=False, handlelength=1.4)
    save(fig, out, "fig_forgetting_curves")


def fig_accuracy_matrices(arms: dict, out: Path) -> None:
    n = len(arms)
    fig, axes = plt.subplots(1, n, figsize=(FULL_W, FULL_W / n * 1.12),
                             constrained_layout=True)
    cmap = LinearSegmentedColormap.from_list("acc", ["#cde2fb", "#0d366b"])
    vmin, vmax = 0.66, 1.0
    im = None
    for ax, (key, mats) in zip(np.atleast_1d(axes), arms.items()):
        _, _, label, _ = ARMS[key]
        M = mean_matrix(mats)
        T = M.shape[0]
        im = ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax)
        for t in range(T):
            for j in range(t + 1):
                dark = (M[t, j] - vmin) / (vmax - vmin) > 0.55
                ax.text(j, t, f"{M[t, j]:.3f}".replace("0.", "."),
                        ha="center", va="center", fontsize=5.5,
                        color="white" if dark else INK)
        ax.set_title(label)
        ax.set_xticks(range(T), [f"$D_{j}$" for j in range(T)])
        ax.set_yticks(range(T), [f"$D_{t}$" for t in range(T)])
        ax.set_xlabel("evaluated on domain")
        if ax is np.atleast_1d(axes)[0]:
            ax.set_ylabel("after training domain")
        ax.spines[:].set_visible(False)
        ax.tick_params(length=0)
    fig.colorbar(im, ax=axes, shrink=0.8, pad=0.015, label="accuracy")
    save(fig, out, "fig_accuracy_matrices")


def fig_delta_matrices(arms: dict, out: Path) -> None:
    others = [k for k in arms if k != "none"]
    if "none" not in arms or not others:
        return
    base = mean_matrix(arms["none"])
    fig, axes = plt.subplots(1, len(others),
                             figsize=(FULL_W * len(others) / 3, FULL_W / 3 * 1.12),
                             constrained_layout=True)
    cmap = LinearSegmentedColormap.from_list("div", ["#e34948", "#f7f6f3", "#2a78d6"])
    lim = 0.15
    im = None
    for ax, key in zip(np.atleast_1d(axes), others):
        _, _, label, _ = ARMS[key]
        D = mean_matrix(arms[key]) - base
        T = D.shape[0]
        im = ax.imshow(D, cmap=cmap, vmin=-lim, vmax=lim)
        for t in range(T):
            for j in range(t + 1):
                ax.text(j, t, f"{D[t, j]:+.3f}".replace("0.", "."),
                        ha="center", va="center", fontsize=5.5,
                        color="white" if abs(D[t, j]) > 0.6 * lim else INK)
        ax.set_title(f"{label} $-$ no replay")
        ax.set_xticks(range(T), [f"$D_{j}$" for j in range(T)])
        ax.set_yticks(range(T), [f"$D_{t}$" for t in range(T)])
        ax.set_xlabel("evaluated on domain")
        if ax is np.atleast_1d(axes)[0]:
            ax.set_ylabel("after training domain")
        ax.spines[:].set_visible(False)
        ax.tick_params(length=0)
    fig.colorbar(im, ax=axes, shrink=0.8, pad=0.015, label="$\\Delta$ accuracy")
    save(fig, out, "fig_delta_matrices")


def fig_paired_bwt(arms: dict, seeds: list[int], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(FULL_W * 0.55, 2.6), constrained_layout=True)
    ys = np.arange(len(seeds))[::-1]
    per_arm = {k: [metrics(R)[1] for R in mats] for k, mats in arms.items()}
    for i, y in enumerate(ys):
        vals = [per_arm[k][i] for k in arms]
        ax.plot([min(vals), max(vals)], [y, y], color="#c3c2b7", lw=0.8, zorder=1)
    for key in arms:
        _, _, label, color = ARMS[key]
        ax.scatter(per_arm[key], ys, s=14, color=color, label=label, zorder=2,
                   edgecolors="white", linewidths=0.5)
    ax.set_yticks(ys, [f"seed {s}" for s in seeds])
    ax.set_xlabel("backward transfer (0 = no lasting forgetting)")
    ax.legend(loc="lower left", frameon=False, handletextpad=0.2)
    save(fig, out, "fig_paired_bwt")


def fig_calibration(logs: Path, out: Path) -> None:
    calib = logs / "continual_calib"
    eps, cal_seeds = [1, 5, 10], [42, 43, 44]
    if not calib.is_dir():
        print(f"  calibration logs missing under {calib} -> skipped")
        return
    fig, ax = plt.subplots(figsize=(FULL_W * 0.45, 1.9), constrained_layout=True)
    for r, key in ((0, "none"), (1, "ppca")):
        _, _, label, color = ARMS[key]
        bwts = np.array([[metrics(json.loads(
            (calib / f"ep{e}_replay{r}_seed{s}.json").read_text())["accuracy_matrix"])[1]
            for s in cal_seeds] for e in eps])
        ax.errorbar(eps, bwts.mean(axis=1), yerr=bwts.std(axis=1, ddof=1),
                    color=color, lw=1.1, marker="o", ms=2.5, capsize=2,
                    elinewidth=0.8, label=label)
    ax.axhline(0, color="#c3c2b7", lw=0.6, zorder=0)
    ax.set_xticks(eps)
    ax.set_xlabel("fine-tune epochs per domain")
    ax.set_ylabel("BWT")
    ax.legend(loc="lower left", frameon=False)
    save(fig, out, "fig_calibration")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--logs", type=Path, default=Path("logs"))
    ap.add_argument("--out", type=Path, default=Path("logs/figures"))
    ap.add_argument("--seeds", type=int, nargs="+",
                    default=[42, 43, 44, 45, 46, 47, 48, 49, 50, 51])
    args = ap.parse_args()
    paper_style()
    args.out.mkdir(parents=True, exist_ok=True)
    arms = load_arms(args.logs, args.seeds)
    if not arms:
        raise SystemExit(f"no study logs found under {args.logs}")
    fig_forgetting_curves(arms, args.out)
    fig_accuracy_matrices(arms, args.out)
    fig_delta_matrices(arms, args.out)
    fig_paired_bwt(arms, args.seeds, args.out)
    fig_calibration(args.logs, args.out)


if __name__ == "__main__":
    main()
