"""Aggregate the HAR FLOAT32-vs-packed-SYM sweep into an honest comparison.

Reads the per-run ``logs/{config}_seed{N}.json`` files written by run_matrix.py,
computes per-config mean +/- sample-std over the seeds, and prints a headline
comparison table + writes ``outputs/memory_summary.json``.

INTEGRITY (this is the whole point of the example, so it is enforced here):

* Headline numbers come ONLY from the multi-seed aggregate; a --min-seeds guard
  warns loudly when fewer than 10 seeds are present (smoke, not a shippable claim).
* The weight-compression win (params_b) and the TOTAL on-device footprint
  (mcu_total_b) are reported SEPARATELY: at a training batch of 64 the activation
  tensors dominate, so a 75%-at-SYM@8 WEIGHT drop is only a small total-footprint
  drop. Conflating the two would be dishonest; the table shows the full category
  breakdown so the reader sees where the win lands.
* Group scale/zero-point metadata (group_overhead_b) is its OWN category, never
  folded into params_b. Finer granularity buys accuracy with metadata bytes, so
  reporting the packed payload alone would make per-channel/g64/g32 look free —
  it is up to a 24% surcharge on the weights at 4 bits (#300).
* The reconciliation gap (heap_peak_b - mcu_total_b, ~= the host-resident dataset
  the MCU would stream) is reported as-is, never massaged.
* "Convergence" (did train loss descend) is reported as k/N across seeds — a
  coarse config that fails to descend is a recorded finding, not a dropped run.

CLI::

    uv run examples/har_classifier/compare_memory.py [--logs DIR] [--min-seeds N] [--plots]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

from examples._shared.log_schema import RunLog, load_log  # noqa: E402

# Config display order (float first as the reference, then descending SYM width).
# The actual run_matrix.py sweep set (#322: was ["float","sym16","sym12","sym8","sym4"] —
# sym16 never ran and sym10/sym6 were omitted, so those rows appended in dict order and
# scrambled the table + color ramp). sym8cos/sym4cos are LR-schedule variants (memory-
# identical to sym8/sym4) kept here so they order deterministically instead of appending.
CONFIG_ORDER = [
    "float", "sym12", "sym10", "sym8", "sym6", "sym4", "sym8cos", "sym4cos",
    # #300 granularity axis. The rows above are per-TENSOR; these are the
    # per-channel / fixed-group variants, widest-first and coarsest-to-finest
    # within each width so the metadata surcharge reads monotonically downward.
    "sym6pc", "sym6g64", "sym6g32",
    "sym4pc", "sym4g64", "sym4g32",
    "asym6", "asym6pc", "asym6g64", "asym6g32",
    "asym4", "asym4pc", "asym4g64", "asym4g32",
]

# Analytic MCU categories, in report order. mcu_total_b is their sum.
CATEGORIES = ["params_b", "group_overhead_b", "grads_b", "optstate_analytic_b", "activations_b",
              "io_b", "pool_backward_b", "dx_peak_b"]

# Scalars aggregated per config (mean +/- std over seeds).
SCALARS = [
    "test_acc",
    "params_b",
    "group_overhead_b",
    "grads_b",
    "optstate_analytic_b",
    "activations_b",
    "io_b",
    "pool_backward_b",
    "dx_peak_b",
    "mcu_total_b",
    "heap_peak_b",
    "stack_peak_b",
    "rss_peak_kb",
    "reconciliation_gap_b",
    "wall_s",
]

ACC_TOL = 0.01  # "keeps up with FLOAT32" = within 1 accuracy point

# Config keys that define the TRAINING BUDGET. A baseline differing in any of them
# is not a baseline: the accuracy gap would then measure the budget, not the
# quantization. (The #300 sweep ran 50 epochs; the older logs/ float ran 20.)
# Absent keys carry a defined meaning (see log_schema.TrainConfig), so normalise
# before comparing: an older log that simply omits lr_schedule is running the same
# constant LR as one that writes "none". A gate with false positives gets bypassed.
_BUDGET_DEFAULTS = {
    "epochs": None, "batch": None, "lr": None,
    "momentum": 0.0,        # absent for optimizers without momentum (#328)
    "optimizer": "sgd",     # absent = sgd
    "lr_schedule": "none",  # absent = constant LR (#327)
    "weight_decay": 0.0,
}


def _budget(log: RunLog) -> tuple:
    cfg = log["config"]
    return tuple((k, cfg.get(k, default)) for k, default in _BUDGET_DEFAULTS.items())


def baseline_incompatibilities(
    sweep: dict[str, dict[int, RunLog]], baseline: dict[str, dict[int, RunLog]]
) -> list[str]:
    """Reasons ``baseline`` may not be paired with ``sweep`` (empty = compatible).

    Two independent checks, because a cross-directory baseline can be wrong in two
    unrelated ways:

    1. TRAINING BUDGET — differing epochs/lr/optimizer make the arms incomparable
       on their face.
    2. CODE DRIFT — C training is deterministic per seed, so any config appearing
       in BOTH sets must reproduce EXACTLY. A mismatch proves the framework changed
       between the runs, so the two accuracy scales are not the same scale. This is
       the check that matters: it needs no metadata to be correct and it catches
       silent numerical change that config keys cannot express.
    """
    problems: list[str] = []

    sweep_budgets = {_budget(log) for by_seed in sweep.values() for log in by_seed.values()}
    for config, by_seed in sorted(baseline.items()):
        for seed, log in sorted(by_seed.items()):
            b = _budget(log)
            if b in sweep_budgets:
                continue
            # key=repr: the values are heterogeneous (int/float/str/None once a key is
            # absent everywhere), so a natural sort raises TypeError on mixed types.
            diffs = sorted(
                f"{k}={v!r} vs sweep {sorted({dict(sb)[k] for sb in sweep_budgets}, key=repr)!r}"
                for k, v in b
                if all(dict(sb)[k] != v for sb in sweep_budgets)
            )
            if not diffs:
                # Mixed-budget sweep: every individual value occurs somewhere, but no
                # single sweep config combines them the way the baseline does. Without
                # this branch the message ended at the dash and stated no reason at all.
                diffs = [
                    f"no single sweep config combines {dict(b)} — sweep budgets are "
                    f"{[dict(sb) for sb in sweep_budgets]}"
                ]
            problems.append(
                f"baseline {config}_seed{seed}: training budget differs — " + "; ".join(diffs)
            )
            break  # one report per config is enough

    for config in sorted(set(sweep) & set(baseline)):
        for seed in sorted(set(sweep[config]) & set(baseline[config])):
            a = sweep[config][seed]["final"]["test_acc"]
            b = baseline[config][seed]["final"]["test_acc"]
            if a != b:
                problems.append(
                    f"code drift: {config}_seed{seed} is in both sets but scored "
                    f"{a:.6f} (sweep) vs {b:.6f} (baseline) — identical config and seed "
                    f"must reproduce exactly, so the runs are not on a common scale"
                )
                break

    return problems


def _converged(log: RunLog) -> bool:
    """True iff train loss descended from the first to the last epoch."""
    epochs = log.get("epochs") or []
    if len(epochs) < 2:
        return False  # single-epoch runs cannot show a first!=last trend
    return float(epochs[-1]["train_loss"]) < float(epochs[0]["train_loss"])


def _run_scalars(log: RunLog) -> dict[str, float]:
    """Flatten one RunLog into the scalar metrics we aggregate."""
    mem = log.get("memory")
    if mem is None:
        raise KeyError("run log has no 'memory' block — was it built with ODT_MEM_PROFILE?")
    final = log["final"]
    out: dict[str, float] = {
        "test_acc": float(final["test_acc"]) if final.get("test_acc") is not None else float("nan"),
        "wall_s": float(sum(e["wall_s"] for e in log.get("epochs", []))),
        # Group scale/zero-point metadata is resident on-device RAM, but the C
        # harness derives it from the post-requantize param tensors and emits it
        # under "config", NOT in the memory report — so it must be picked up
        # here or the granularity axis vanishes from the summary. Absent in
        # pre-#300 logs (float/sym8 baselines) => 0.
        "group_overhead_b": float(log["config"].get("group_overhead_b", 0)),
    }
    for key in SCALARS:
        if key in out or key == "mcu_total_b":
            continue
        out[key] = float(mem[key])

    # mem_instrument.c sums the SEVEN analytic categories and knows nothing about
    # the metadata block, so mcu_total_b is recomputed here to keep the headline
    # total equal to the breakdown row. Cross-check the C value against the
    # categories first: a mismatch means harness and aggregator have drifted, and
    # silently preferring either number would corrupt every headline claim.
    c_total = float(mem["mcu_total_b"])
    analytic = sum(out[k] for k in CATEGORIES if k != "group_overhead_b")
    if analytic != c_total:
        raise ValueError(
            f"mcu_total_b drift: mem_instrument.c reported {c_total:.0f} but its analytic "
            f"categories sum to {analytic:.0f}. The C harness and compare_memory.py disagree "
            f"about the MCU model — reconcile them, do not paper over the gap."
        )
    out["mcu_total_b"] = analytic + out["group_overhead_b"]
    # gap == heap_peak - mcu_total is a documented identity (log_schema.MemoryLog), and
    # heap_peak/mcu_total are the primitives — the gap is derived from them. Copying the
    # C-computed gap while widening the total would break the identity by exactly
    # group_overhead_b and understate the unaccounted host-resident bytes, which is the
    # one number the header promises is never massaged.
    out["reconciliation_gap_b"] = out["heap_peak_b"] - out["mcu_total_b"]
    return out


def load_runs(logs_dir: Path) -> dict[str, dict[int, RunLog]]:
    """Load logs into {config: {seed: RunLog}} from ``{config}_seed{N}.json``."""
    runs: dict[str, dict[int, RunLog]] = {}
    for path in sorted(logs_dir.glob("*_seed*.json")):
        stem = path.stem  # e.g. "sym8_seed3"
        config, _, seed_part = stem.rpartition("_seed")
        if not config or not seed_part.isdigit():
            continue
        log = load_log(path)
        runs.setdefault(config, {})[int(seed_part)] = log
    return runs


def aggregate(runs: dict[str, dict[int, RunLog]]) -> dict:
    """Per-config mean/std over seeds + per-config convergence + directional deltas."""
    per_config: dict[str, dict] = {}
    for config, by_seed in runs.items():
        seeds = sorted(by_seed)
        scal = [_run_scalars(by_seed[s]) for s in seeds]
        stats: dict[str, dict[str, float]] = {}
        for key in SCALARS:
            vals = np.array([r[key] for r in scal], dtype=float)
            stats[key] = {
                "mean": float(np.nanmean(vals)),
                # sample std (ddof=1) needs >=2 points; else 0.0
                "std": float(np.nanstd(vals, ddof=1)) if np.count_nonzero(~np.isnan(vals)) > 1 else 0.0,
            }
        converged = [(_converged(by_seed[s])) for s in seeds]
        per_config[config] = {
            "n_seeds": len(seeds),
            "seeds": seeds,
            "stats": stats,
            "converged_k": int(sum(converged)),
        }

    # Directional consistency vs FLOAT32, paired by seed (accuracy tradeoff).
    comparisons: dict[str, dict] = {}
    if "float" in runs:
        float_by_seed = runs["float"]
        for config in runs:
            if config == "float":
                continue
            common = sorted(set(runs[config]) & set(float_by_seed))
            keeps_up = 0
            acc_gaps = []
            for s in common:
                sym_acc = float(runs[config][s]["final"]["test_acc"])
                flt_acc = float(float_by_seed[s]["final"]["test_acc"])
                acc_gaps.append(flt_acc - sym_acc)
                if sym_acc >= flt_acc - ACC_TOL:
                    keeps_up += 1
            fp = per_config["float"]["stats"]
            cp = per_config[config]["stats"]
            comparisons[config] = {
                "n_pairs": len(common),
                "acc_keeps_up_k": keeps_up,  # SYM within ACC_TOL of FLOAT32
                "acc_gap_mean": float(np.mean(acc_gaps)) if acc_gaps else float("nan"),
                # Weight cost = packed payload + its group metadata. Comparing bare
                # payloads would credit a group config with a compression win it
                # does not have (float carries no metadata, so its overhead is 0).
                "weight_bytes_drop_pct": _drop_pct(
                    fp["params_b"]["mean"] + fp["group_overhead_b"]["mean"],
                    cp["params_b"]["mean"] + cp["group_overhead_b"]["mean"],
                ),
                "mcu_total_drop_pct": _drop_pct(fp["mcu_total_b"]["mean"], cp["mcu_total_b"]["mean"]),
            }
    return {"per_config": per_config, "comparisons": comparisons}


def merge_baseline(
    runs: dict[str, dict[int, RunLog]], baseline: dict[str, dict[int, RunLog]], config: str
) -> None:
    """Splice ``config`` from ``baseline`` into ``runs``, refusing to shadow one.

    An unconditional assignment would discard the sweep's OWN runs of that name in
    favour of the cross-run ones — and it would do so right after the drift check
    reported the collision, so the tool would diagnose the problem and then commit
    it. If the sweep already carries the reference arm, that arm is the better data
    by construction: same build, same session.
    """
    if config in runs:
        raise ValueError(
            f"--logs already contains '{config}' runs, so merging a cross-run baseline would "
            f"replace same-build data with older data. Drop --baseline-logs and use the "
            f"in-sweep '{config}' arm, or point --baseline-config at a name the sweep lacks."
        )
    runs[config] = baseline[config]


def _drop_pct(ref: float, val: float) -> float:
    """Percent reduction of ``val`` relative to ``ref`` (positive = smaller)."""
    return (1.0 - val / ref) * 100.0 if ref else float("nan")


def _fmt_bytes(n: float) -> str:
    for unit, div in (("MB", 1 << 20), ("KB", 1 << 10)):
        if n >= div:
            return f"{n / div:.2f}{unit}"
    return f"{n:.0f}B"


def print_table(agg: dict, min_seeds: int) -> None:
    per_config = agg["per_config"]
    comparisons = agg["comparisons"]
    present = [c for c in CONFIG_ORDER if c in per_config] + [
        c for c in per_config if c not in CONFIG_ORDER
    ]

    # Trip on the MINIMUM seed count across configs, not the max: a single well-
    # seeded config must NOT suppress the warning for an under-seeded one (the SYM
    # configs — most likely to be under-seeded after a crash — are the whole point).
    seed_counts = {c: per_config[c]["n_seeds"] for c in present}
    under = {c: n for c, n in seed_counts.items() if n < min_seeds}
    if under:
        offenders = ", ".join(f"{c}={n}" for c, n in under.items())
        print(
            f"\n*** SMOKE ONLY: config(s) [{offenders}] have < --min-seeds={min_seeds} seeds. "
            "Their mean/std rows and comparisons are a pipeline check, NOT a shippable claim. ***\n",
            file=sys.stderr,
        )

    # --- Headline scalar table (mean +/- std) ---
    print("\n" + "=" * 92)
    print("HAR memory / accuracy sweep — per-config mean +/- std over seeds")
    print("=" * 92)
    cols = [
        ("config", lambda c: c, 8, "s"),
        ("seeds", lambda c: str(per_config[c]["n_seeds"]), 6, "s"),
        ("conv", lambda c: f"{per_config[c]['converged_k']}/{per_config[c]['n_seeds']}", 6, "s"),
        ("test_acc", lambda c: _pm(per_config[c]["stats"]["test_acc"], ".4f"), 17, "s"),
        ("weight_B", lambda c: _pm(per_config[c]["stats"]["params_b"], ".0f"), 16, "s"),
        # Metadata is deterministic per config (std always 0), so mean-only.
        ("grp_meta_B", lambda c: f"{per_config[c]['stats']['group_overhead_b']['mean']:.0f}", 11, "s"),
        ("mcu_total", lambda c: _fmt_bytes(per_config[c]["stats"]["mcu_total_b"]["mean"]), 10, "s"),
        ("heap_peak", lambda c: _fmt_bytes(per_config[c]["stats"]["heap_peak_b"]["mean"]), 10, "s"),
        ("stack_B", lambda c: f"{per_config[c]['stats']['stack_peak_b']['mean']:.0f}", 9, "s"),
    ]
    header = "".join(f"{name:>{w}}" for name, _, w, _ in cols)
    print(header)
    print("-" * len(header))
    for c in present:
        print("".join(f"{fn(c):>{w}}" for _, fn, w, _ in cols))

    # --- Category breakdown (means, bytes) — micro-batch B=1 concurrent peak ---
    print("\nAnalytic MCU category means (bytes) — activations sized at micro-batch B=1 "
          "(the loop streams the macro-batch one sample at a time):")
    cat_labels = {
        "params_b": "params", "group_overhead_b": "grp_meta", "grads_b": "grads",
        "optstate_analytic_b": "optstate", "activations_b": "activations", "io_b": "io",
        "pool_backward_b": "pool_bwd", "dx_peak_b": "dx_peak",
    }
    chdr = f"{'config':>8}" + "".join(f"{cat_labels[k]:>14}" for k in CATEGORIES) + f"{'mcu_total':>14}"
    print(chdr)
    print("-" * len(chdr))
    for c in present:
        st = per_config[c]["stats"]
        row = f"{c:>8}" + "".join(f"{st[k]['mean']:>14.0f}" for k in CATEGORIES)
        row += f"{st['mcu_total_b']['mean']:>14.0f}"
        print(row)

    # --- Comparison vs FLOAT32 ---
    if comparisons:
        print("\nvs FLOAT32 (paired by seed):")
        chdr = (
            f"{'config':>8}{'weight_drop%':>14}{'mcu_total_drop%':>16}"
            f"{'acc_gap':>10}{'keeps_up(<=1pt)':>17}"
        )
        print(chdr)
        print("-" * len(chdr))
        for c in present:
            if c not in comparisons:
                continue
            cm = comparisons[c]
            acc = (f"{cm['acc_gap_mean']:>10.4f}" if cm["acc_gap_mean"] is not None
                   else f"{'n/a':>10}")
            keeps = (f"{cm['acc_keeps_up_k']:>10}/{cm['n_pairs']:<6}"
                     if cm["acc_keeps_up_k"] is not None else f"{'n/a':>10}{'':<7}")
            print(
                f"{c:>8}{cm['weight_bytes_drop_pct']:>13.1f}%{cm['mcu_total_drop_pct']:>15.1f}%"
                f"{acc}{keeps}"
            )
        if any(c in comparisons and comparisons[c].get("acc_suppressed_reason") for c in present):
            print(
                "\n*** accuracy columns SUPPRESSED: the baseline is not comparable (different\n"
                "training budget and/or a framework change between the runs). The BYTE columns\n"
                "remain valid — they are arithmetic over the model geometry, which is identical.\n"
                "See 'baseline.incompatibilities' in the summary JSON. ***"
            )
        print(
            "\nHONESTY: 'weight_drop%' is the packed-SYM compression of the WEIGHT category;\n"
            "'mcu_total_drop%' is the effect on the FULL on-device footprint. Both are real;\n"
            "they answer different questions. Because activations are sized at micro-batch\n"
            "B=1 (not the macro-batch), params/grads/momentum each carry ~a third of the\n"
            "footprint, so a 75% weight cut yields a MATERIAL total-RAM drop (not a rounding\n"
            "error). The reconciliation gap (heap_peak - mcu_total ~= host dataset) is in the JSON."
        )


def _pm(stat: dict[str, float], fmt: str) -> str:
    """Format a mean +/- std stat like '0.5857+/-0.0123'."""
    return f"{stat['mean']:{fmt}}+/-{stat['std']:{fmt}}"


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--logs", default=str(HERE / "logs"), help="dir of per-run JSON")
    ap.add_argument(
        "--min-seeds", type=int, default=10,
        help="warn loudly if the richest config has fewer seeds (default 10)",
    )
    ap.add_argument(
        "--out", default=str(HERE / "outputs" / "memory_summary.json"),
        help="path for the aggregated summary JSON",
    )
    ap.add_argument("--plots", action="store_true", help="also render pictograms into plots/")
    ap.add_argument(
        "--baseline-logs", default=None,
        help="dir holding the FLOAT32 reference runs, when they live outside --logs",
    )
    ap.add_argument(
        "--baseline-config", default="float", help="config name to pull from --baseline-logs",
    )
    ap.add_argument(
        "--allow-incomparable-baseline", action="store_true",
        help="merge the baseline even when the compatibility gate rejects it; the reasons are "
             "recorded in the summary JSON so the caveat travels with the numbers",
    )
    args = ap.parse_args()

    logs_dir = Path(args.logs)
    runs = load_runs(logs_dir)
    if not runs:
        raise SystemExit(f"no *_seed*.json logs found in {logs_dir} — run run_matrix.py first")

    baseline_meta = None
    if args.baseline_logs:
        bdir = Path(args.baseline_logs)
        braw = load_runs(bdir)
        if args.baseline_config not in braw:
            raise SystemExit(f"{bdir} has no '{args.baseline_config}_seed*.json' runs")
        # Gate on the config being merged PLUS every config the two dirs share — the
        # shared ones are the drift evidence, and they cost nothing to check.
        relevant = {c: r for c, r in braw.items() if c == args.baseline_config or c in runs}
        problems = baseline_incompatibilities(runs, relevant)
        merged = not problems or args.allow_incomparable_baseline
        baseline_meta = {
            "config": args.baseline_config,
            "source": str(bdir),
            "cross_run": True,
            "incompatibilities": problems,
            "merged": merged,
        }
        if problems:
            banner = ("MERGED ANYWAY (--allow-incomparable-baseline)" if merged
                      else "REJECTED — not merged")
            print(f"\n*** BASELINE {banner} ***", file=sys.stderr)
            for p in problems:
                print(f"  - {p}", file=sys.stderr)
            if not merged:
                print("  Pass --allow-incomparable-baseline to merge anyway: you then get the "
                      "BYTE comparison (valid — same model geometry) with the accuracy columns "
                      "suppressed, and the reasons stamped into the summary JSON.",
                      file=sys.stderr)
            print("", file=sys.stderr)
        if merged:
            try:
                merge_baseline(runs, braw, args.baseline_config)
            except ValueError as exc:
                raise SystemExit(str(exc)) from exc

    agg = aggregate(runs)
    if baseline_meta is not None:
        agg["baseline"] = baseline_meta
        if baseline_meta["merged"] and baseline_meta["incompatibilities"]:
            # The two halves of the comparison do NOT fail together. Byte columns are
            # pure arithmetic over the model geometry (identical across the runs), so
            # they stay valid. The accuracy columns are exactly what a budget or code
            # difference contaminates — suppress them rather than print a number that
            # says "4-bit beats FLOAT32" when it only means "float trained 20 epochs".
            for cm in agg["comparisons"].values():
                cm["acc_gap_mean"] = None
                cm["acc_keeps_up_k"] = None
                cm["acc_suppressed_reason"] = "incomparable baseline; see agg['baseline']"
    print_table(agg, args.min_seeds)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(agg, indent=2))
    print(f"\nwrote {out_path}", flush=True)

    if args.plots:
        from examples._shared import plotting

        plots_dir = HERE / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        plotting.plot_peak_ram_stacked_bar(plots_dir / "har_mem_peak_ram_bar.png", agg)
        plotting.plot_accuracy_vs_memory_scatter(plots_dir / "har_mem_acc_vs_mem.png", agg)
        plotting.plot_weight_compression_bar(plots_dir / "har_mem_weight_compression.png", agg)
        print(f"wrote 3 pictograms to {plots_dir}", flush=True)


if __name__ == "__main__":
    main()
