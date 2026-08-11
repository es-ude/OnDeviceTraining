"""Test examples/har_classifier/compare_memory.py aggregation.

The regression under test (#300 group-quant sweep): group_overhead_b — the
per-group scale/zero-point metadata — is emitted per run by the C harness but
lives under log["config"], while _run_scalars only read log["memory"]. The
16-config group sweep therefore reported IDENTICAL params_b and mcu_total_b for
per-tensor / per-channel / g64 / g32 at the same bit width, i.e. it made finer
granularity look free — the exact question the sweep exists to answer.
"""
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from examples.har_classifier.compare_memory import (  # noqa: E402
    CATEGORIES,
    aggregate,
    baseline_incompatibilities,
    merge_baseline,
)
from examples._shared.log_schema import RunLog  # noqa: E402

# Analytic categories other than params/group_overhead, held fixed across the
# synthetic configs so any mcu_total_b delta is attributable to metadata alone.
_FIXED = {
    "grads_b": 40856,
    "optstate_analytic_b": 40856,
    "activations_b": 57904,
    "io_b": 4632,
    "pool_backward_b": 8192,
    "dx_peak_b": 16384,
}


def _log(*, params_b: int, overhead: int | None, acc: float, mcu_total_b: int | None = None) -> RunLog:
    """One synthetic -DODT_MEM_PROFILE run log.

    ``overhead=None`` omits group_overhead_b entirely (a pre-#300 log).
    ``mcu_total_b=None`` derives the C-side total the way mem_instrument.c does:
    the seven analytic categories, WITHOUT the metadata block.
    """
    config: dict = {"epochs": 2, "batch": 64, "lr": 0.01, "seed": 1, "shuffle_seed": 1}
    if overhead is not None:
        config["group_overhead_b"] = overhead
    c_total = mcu_total_b if mcu_total_b is not None else params_b + sum(_FIXED.values())
    return {
        "impl": "c-sym-weights",
        "example": "har_classifier",
        "config": config,
        "epochs": [
            {"epoch": 0, "step_losses": [1.0], "train_loss": 1.0,
             "val_loss": 0.9, "val_acc": 0.5, "wall_s": 1.0},
            {"epoch": 1, "step_losses": [0.5], "train_loss": 0.5,
             "val_loss": 0.4, "val_acc": 0.8, "wall_s": 1.0},
        ],
        "final": {"test_loss": 0.4, "test_acc": acc, "test_auc": None},
        "memory": {
            "sym_bits": 4, "dataset_b": 50877204, "params_grads_b": 58835, "optstate_b": 42084,
            "params_b": params_b, **_FIXED,
            "mcu_total_b": c_total,
            "heap_peak_b": 51138367, "stack_peak_b": 52416, "rss_peak_kb": 60608,
            "reconciliation_gap_b": 51138367 - c_total,
        },
    }


def test_group_overhead_is_aggregated_and_separated_from_payload():
    """Metadata must show up as its own category, never folded into params_b."""
    runs = {
        "sym4": {1: _log(params_b=5107, overhead=32, acc=0.8957)},
        "sym4g32": {1: _log(params_b=5107, overhead=1216, acc=0.8976)},
    }
    agg = aggregate(runs)

    # Payload is genuinely identical — that part of the old output was correct.
    assert agg["per_config"]["sym4"]["stats"]["params_b"]["mean"] == 5107
    assert agg["per_config"]["sym4g32"]["stats"]["params_b"]["mean"] == 5107

    # ...but the metadata cost must be visible and distinct.
    assert agg["per_config"]["sym4"]["stats"]["group_overhead_b"]["mean"] == 32
    assert agg["per_config"]["sym4g32"]["stats"]["group_overhead_b"]["mean"] == 1216


def test_mcu_total_includes_group_overhead():
    """Two configs differing ONLY in metadata must differ in total footprint."""
    runs = {
        "sym4": {1: _log(params_b=5107, overhead=32, acc=0.8957)},
        "sym4g32": {1: _log(params_b=5107, overhead=1216, acc=0.8976)},
    }
    agg = aggregate(runs)
    tensor = agg["per_config"]["sym4"]["stats"]["mcu_total_b"]["mean"]
    g32 = agg["per_config"]["sym4g32"]["stats"]["mcu_total_b"]["mean"]
    assert g32 - tensor == 1216 - 32


def test_mcu_total_equals_category_sum():
    """The printed breakdown row must sum to the printed total (no silent gap)."""
    runs = {"sym4g32": {1: _log(params_b=5107, overhead=1216, acc=0.8976)}}
    stats = aggregate(runs)["per_config"]["sym4g32"]["stats"]
    assert stats["mcu_total_b"]["mean"] == sum(stats[k]["mean"] for k in CATEGORIES)


def test_log_without_group_overhead_defaults_to_zero():
    """Pre-#300 logs (float/sym8 baselines) must still aggregate, contributing 0."""
    runs = {"float": {1: _log(params_b=40856, overhead=None, acc=0.91)}}
    stats = aggregate(runs)["per_config"]["float"]["stats"]
    assert stats["group_overhead_b"]["mean"] == 0
    assert stats["mcu_total_b"]["mean"] == 40856 + sum(_FIXED.values())


def _budget_log(*, epochs: int, acc: float, params_b: int = 5107) -> RunLog:
    log = _log(params_b=params_b, overhead=32, acc=acc)
    log["config"]["epochs"] = epochs
    return log


def test_baseline_with_different_epoch_budget_is_rejected():
    """A baseline trained for a different number of epochs is not a baseline: the
    accuracy gap would measure the training budget, not the quantization."""
    sweep = {"sym4": {s: _budget_log(epochs=50, acc=0.895) for s in (1, 2)}}
    baseline = {"float": {s: _budget_log(epochs=20, acc=0.890, params_b=40856) for s in (1, 2)}}
    problems = baseline_incompatibilities(sweep, baseline)
    assert any("epochs" in p for p in problems), problems


def test_baseline_from_a_drifted_run_is_rejected():
    """Training is deterministic per seed, so a config present in BOTH sets must
    reproduce exactly. A mismatch means the code changed between the two runs and
    the accuracies are not on a common scale."""
    sweep = {
        "sym4": {1: _budget_log(epochs=50, acc=0.8957)},
        "float": {1: _budget_log(epochs=50, acc=0.8953, params_b=40856)},
    }
    baseline = {
        "sym4": {1: _budget_log(epochs=50, acc=0.8761)},  # same config, different result
        "float": {1: _budget_log(epochs=50, acc=0.8953, params_b=40856)},
    }
    problems = baseline_incompatibilities(sweep, baseline)
    assert any("sym4" in p for p in problems), problems


def test_absent_optional_config_keys_do_not_trip_the_gate():
    """An older log that omits lr_schedule runs the same constant LR as one that
    writes "none" (log_schema.TrainConfig). A gate that cries wolf gets bypassed."""
    sweep_log = _budget_log(epochs=50, acc=0.8957)
    sweep_log["config"]["lr_schedule"] = "none"
    baseline_log = _budget_log(epochs=50, acc=0.8953, params_b=40856)
    baseline_log["config"].pop("lr_schedule", None)  # older log: key simply absent
    assert baseline_incompatibilities({"sym4": {1: sweep_log}},
                                      {"float": {1: baseline_log}}) == []


def test_compatible_baseline_passes():
    sweep = {"sym4": {1: _budget_log(epochs=50, acc=0.8957)}}
    baseline = {"float": {1: _budget_log(epochs=50, acc=0.8953, params_b=40856)}}
    assert baseline_incompatibilities(sweep, baseline) == []


def test_comparisons_populated_once_baseline_is_merged():
    """The end the whole gate serves: a compatible float baseline fills in the
    vs-FLOAT32 columns, and the weight drop accounts for metadata."""
    runs = {
        "sym4": {1: _log(params_b=5107, overhead=32, acc=0.8957)},
        "float": {1: _log(params_b=40856, overhead=None, acc=0.8953)},
    }
    comps = aggregate(runs)["comparisons"]
    assert "sym4" in comps
    # 40856+0 -> 5107+32 is a 87.4% drop; using bare payload would overstate it.
    assert comps["sym4"]["weight_bytes_drop_pct"] == pytest.approx(
        (1 - 5139 / 40856) * 100, abs=1e-9
    )


def test_reconciliation_gap_tracks_the_widened_total():
    """gap == heap_peak - mcu_total is a documented identity (log_schema.MemoryLog).
    Widening mcu_total_b with the metadata while copying the C-computed gap verbatim
    broke it by exactly group_overhead_b, understating the unaccounted host bytes."""
    runs = {"sym4g32": {1: _log(params_b=5107, overhead=1216, acc=0.8976)}}
    stats = aggregate(runs)["per_config"]["sym4g32"]["stats"]
    assert stats["reconciliation_gap_b"]["mean"] == (
        stats["heap_peak_b"]["mean"] - stats["mcu_total_b"]["mean"]
    )


def test_merge_baseline_refuses_to_shadow_in_sweep_runs():
    """A cross-run baseline must never silently replace the sweep's own runs of the
    same name — the drift check would flag the collision and the merge would then
    discard the fresher data anyway."""
    runs = {"float": {1: _log(params_b=40856, overhead=None, acc=0.9053)}}
    braw = {"float": {1: _log(params_b=40856, overhead=None, acc=0.8953)}}
    with pytest.raises(ValueError, match="already contains"):
        merge_baseline(runs, braw, "float")
    assert runs["float"][1]["final"]["test_acc"] == 0.9053  # untouched


def test_merge_baseline_splices_a_config_absent_from_the_sweep():
    runs = {"sym4": {1: _log(params_b=5107, overhead=32, acc=0.8957)}}
    braw = {"float": {1: _log(params_b=40856, overhead=None, acc=0.8953)}}
    merge_baseline(runs, braw, "float")
    assert runs["float"][1]["final"]["test_acc"] == 0.8953


def test_budget_mismatch_always_states_a_reason():
    """With a mixed-budget sweep a baseline can match no single budget while every
    individual key value appears somewhere, which produced a bare
    'training budget differs — ' with nothing after the dash."""
    a = _budget_log(epochs=50, acc=0.89)
    a["config"]["momentum"] = 0.9
    b = _budget_log(epochs=20, acc=0.88)  # no momentum key -> normalises to 0.0
    base = _budget_log(epochs=20, acc=0.90, params_b=40856)
    base["config"]["momentum"] = 0.9  # matches a's momentum, b's epochs, neither tuple

    problems = baseline_incompatibilities({"symA": {1: a}, "symB": {1: b}}, {"float": {1: base}})
    assert len(problems) == 1
    assert not problems[0].rstrip().endswith("—"), problems[0]
    assert "epochs" in problems[0] and "momentum" in problems[0], problems[0]


def test_c_python_total_drift_is_rejected():
    """If the C total stops matching the analytic categories, fail loudly rather
    than silently reporting a Python-side number that drifted from the harness."""
    bad = _log(params_b=5107, overhead=32, acc=0.89, mcu_total_b=999999)
    with pytest.raises(ValueError, match="mcu_total_b"):
        aggregate({"sym4": {1: bad}})
