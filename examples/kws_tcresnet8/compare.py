"""Compare the TC-ResNet8 sweep (run_sweep.py): C against the PyTorch twin over 10 seeds.

Per configuration (data A/B x width full/half): mean +- std of the final-epoch test accuracy for
C at micro-batch 100, PyTorch, and C at micro-batch 4; the hypothesis test (the C and PyTorch
means at micro-batch 100 differ by less than one pooled standard deviation); mean loss curves;
and the training-memory table of the ODT_MEM_PROFILE runs against the Pico 2's 520 KiB SRAM.

    uv run examples/kws_tcresnet8/compare.py [--logs DIR] [--plots DIR]

Exits 1 when the hypothesis test fails in any configuration with logs on both sides.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT))
from examples._shared.log_schema import RunLog, load_log  # noqa: E402
from examples.kws_tcresnet8.run_sweep import (  # noqa: E402
    ARMS,
    CONFIGS,
    DEFAULT_LOGS,
    MEM_MICRO_BATCHES,
    SEEDS,
    Run,
)

PICO2_SRAM_B = 520 * 1024  # RP2350


def pooled_std(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    dof = len(a) + len(b) - 2
    return float(np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / dof))


def gap_within_pooled_std(c: np.ndarray, py: np.ndarray) -> bool:
    """The spec's pass rule for one configuration: |mean(C) - mean(PyTorch)| < pooled std.
    Both sets need at least two seeds."""
    return bool(abs(np.mean(c) - np.mean(py)) < pooled_std(c, py))


def load_arm(root: Path, data: str, width: str, side: str, mb: int) -> tuple[list[RunLog], list[int]]:
    """The arm's logs that finished training (diverged ones included), and the missing seeds."""
    logs, missing = [], []
    for seed in SEEDS:
        path = Run(data, width, side, mb, seed).log_path(root)
        try:
            log = load_log(path)
        except (OSError, ValueError, KeyError):
            missing.append(seed)
            continue
        if "final" not in log:
            missing.append(seed)
            continue
        logs.append(log)
    return logs, missing


def fmt_acc(accs: np.ndarray) -> str:
    if len(accs) == 0:
        return "-"
    std = f"{100 * accs.std(ddof=1):.2f}" if len(accs) > 1 else "n/a"
    return f"{100 * accs.mean():.2f} +- {std} (n={len(accs)})"


def accuracy_report(root: Path) -> tuple[bool, dict]:
    print("Final-epoch test accuracy [%], mean +- std over seeds:")
    print(f"{'config':<8} {'C m100':>22} {'PyTorch m100':>22} {'C-Py':>7} {'pooled':>7} "
          f"{'verdict':>8} {'C m4':>22}")
    all_pass, arms = True, {}
    for data, width in CONFIGS:
        cells = {}
        for side, mb in ARMS:
            logs, missing = load_arm(root, data, width, side, mb)
            arms[(data, width, side, mb)] = logs
            diverged = sum(log["final"].get("diverged", 0) for log in logs)
            accs = np.array([log["final"]["test_acc"] for log in logs], dtype=float)
            cells[(side, mb)] = accs
            if missing or diverged:
                print(f"  note {data}/{width}/{side}/m{mb}: missing seeds {missing}, "
                      f"{diverged} diverged")
        c, py = cells[("c", 100)], cells[("pytorch", 100)]
        if len(c) >= 2 and len(py) >= 2:
            ok = gap_within_pooled_std(c, py)
            all_pass &= ok
            gap = f"{100 * (c.mean() - py.mean()):+.2f}"
            pooled = f"{100 * pooled_std(c, py):.2f}"
            verdict = "pass" if ok else "FAIL"
        else:
            gap = pooled = verdict = "n/a"
        print(f"{data}/{width:<6} {fmt_acc(c):>22} {fmt_acc(py):>22} {gap:>7} {pooled:>7} "
              f"{verdict:>8} {fmt_acc(cells[('c', 4)]):>22}")
    return all_pass, arms


def plot_loss_curves(out_path: Path, arms: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    styles = {("c", 100): ("C m100", "#1f77b4"), ("pytorch", 100): ("PyTorch m100", "#ff7f0e"),
              ("c", 4): ("C m4", "#2ca02c")}
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharey=False)
    for ax, (data, width) in zip(axes.flat, CONFIGS):
        for (side, mb), (label, color) in styles.items():
            logs = arms[(data, width, side, mb)]
            if not logs:
                continue
            n = min(len(log["epochs"]) for log in logs)
            for key, ls in (("train_loss", "-"), ("val_loss", "--")):
                curves = np.array([[e[key] for e in log["epochs"][:n]] for log in logs])
                mean, std = curves.mean(axis=0), curves.std(axis=0)
                ax.plot(mean, ls, color=color,
                        label=f"{label} {key.split('_')[0]} (n={len(logs)})")
                if key == "train_loss":
                    ax.fill_between(range(n), mean - std, mean + std, color=color, alpha=0.15)
        ax.set_title(f"data {data}, {width} width")
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle("TC-ResNet8: mean loss over seeds (train solid with +-1 std, val dashed)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def memory_report(root: Path) -> None:
    print(f"\nTraining memory (ODT_MEM_PROFILE, heap peak minus the dataset and the loader index "
          f"arrays) against the Pico 2 SRAM ({PICO2_SRAM_B // 1024} KiB = {PICO2_SRAM_B} B):")
    print(f"{'config':<8} {'m':>4} {'training KiB':>13} {'Pico 2':>7} {'loader idx KiB':>15} "
          f"{'dataset MiB':>12}")
    for data, width in CONFIGS:
        for mb in MEM_MICRO_BATCHES:
            path = Run(data, width, "c", mb, 0, memory=True, epochs=1).log_path(root)
            try:
                mem = load_log(path)["memory"]
            except (OSError, ValueError, KeyError):
                print(f"{data}/{width:<6} {mb:>4} {'-':>13}")
                continue
            train_b = mem["train_heap_peak_b"]
            fits = "fits" if train_b <= PICO2_SRAM_B else "over"
            print(f"{data}/{width:<6} {mb:>4} {train_b / 1024:>13.1f} {fits:>7} "
                  f"{mem['loader_index_b'] / 1024:>15.1f} {mem['dataset_b'] / 2**20:>12.1f}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--logs", type=Path, default=DEFAULT_LOGS)
    parser.add_argument("--plots", type=Path, default=HERE / "plots" / "sweep")
    args = parser.parse_args()

    all_pass, arms = accuracy_report(args.logs)
    args.plots.mkdir(parents=True, exist_ok=True)
    plot_loss_curves(args.plots / "loss_curves.png", arms)
    print(f"loss curves: {args.plots / 'loss_curves.png'}")
    memory_report(args.logs)
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
