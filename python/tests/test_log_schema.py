"""Test examples/_shared/log_schema.py."""
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from examples._shared.log_schema import EpochLog, FinalLog, RunLog, TrainConfig, dump_log, load_log


def test_dump_then_load_roundtrips(tmp_path):
    log: RunLog = {
        "impl": "pytorch",
        "example": "har_classifier",
        "config": {
            "epochs": 2, "batch": 64, "lr": 0.01, "momentum": 0.9,
            "seed": 42, "shuffle_seed": 42,
        },
        "epochs": [
            {"epoch": 0, "step_losses": [1.5, 1.2, 1.0], "train_loss": 1.23,
             "val_loss": 0.85, "val_acc": 0.71, "wall_s": 1.4},
            {"epoch": 1, "step_losses": [0.8, 0.6, 0.5], "train_loss": 0.62,
             "val_loss": 0.42, "val_acc": 0.88, "wall_s": 1.3},
        ],
        "final": {"test_loss": 0.4, "test_acc": 0.9, "test_auc": None},
    }
    path = tmp_path / "log.json"
    dump_log(path, log)
    loaded = load_log(path)
    assert loaded == log


def test_load_validates_required_top_level_keys(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"impl": "pytorch"}))  # missing fields
    try:
        load_log(path)
    except KeyError as e:
        assert "config" in str(e) or "epochs" in str(e)
    else:
        raise AssertionError("expected KeyError on missing required top-level key")


def test_scheduler_and_provenance_keys_are_declared_optional():
    """The C HAR harness writes these (batch-size scheduler port, design A4). A key the
    schema does not declare is a silent contract drift, so pin them as NotRequired."""
    expected_config = {
        "lr_schedule", "bs_schedule", "bs_lr_compensation", "gamma", "step_size",
        "max_batch_size", "reshuffle", "toolchain",
    }
    assert expected_config <= TrainConfig.__optional_keys__
    assert {"lr", "batch_size", "parameter_updates"} <= EpochLog.__optional_keys__


def test_extended_c_log_roundtrips(tmp_path):
    log: RunLog = {
        "impl": "c",
        "example": "har_classifier",
        "config": {
            "epochs": 2, "batch": 8, "lr": 0.01, "momentum": 0.9,
            "seed": 42, "shuffle_seed": 42,
            "lr_schedule": "none", "bs_schedule": "exp", "bs_lr_compensation": 1,
            "gamma": 0.5, "step_size": 1, "max_batch_size": 661, "reshuffle": 1,
            "toolchain": "Apple LLVM 21.0.0 (clang-2100.3.34.2)",
        },
        "epochs": [
            {"epoch": 0, "step_losses": [], "train_loss": 1.2, "val_loss": 0.9,
             "val_acc": 0.7, "wall_s": 1.0, "lr": 0.01, "batch_size": 8,
             "parameter_updates": 827},
            {"epoch": 1, "step_losses": [], "train_loss": 0.8, "val_loss": 0.6,
             "val_acc": 0.8, "wall_s": 1.0, "lr": 0.01, "batch_size": 16,
             "parameter_updates": 413},
        ],
        "final": {"test_loss": 0.5, "test_acc": 0.85, "test_auc": None},
    }
    path = tmp_path / "c.json"
    dump_log(path, log)
    assert load_log(path) == log


def test_c_bfp_log_without_sym_bits_loads(tmp_path):
    log = {
        "impl": "c-bfp", "example": "har_classifier",
        "config": {
            "epochs": 1, "batch": 64, "lr": 0.01, "momentum": 0.9, "seed": 1, "shuffle_seed": 1,
            "weight_dtype": "bfp", "mantissa_bits": 6, "exponent_bits": 8,
            "weight_block": "32", "wire_block": "16", "bfp_math": "native",
            "bfp_grads": 0, "bfp_state": 0, "bfp_rounding": "sr",
            "groups_resolved": {"conv1": [16, 63]}, "wires_resolved": {"conv1.out": [128, 16]},
            "group_overhead_b": 300,
        },
        "epochs": [{"epoch": 0, "step_losses": [1.0], "train_loss": 1.0,
                    "val_loss": 1.0, "val_acc": 0.2, "wall_s": 1.0}],
        "final": {"test_loss": 1.0, "test_acc": 0.2, "test_auc": None},
        "memory": {"storage_dtype": "bfp", "dataset_b": 1, "params_grads_b": 1, "optstate_b": 1,
                   "params_b": 1, "group_overhead_b": 300, "grads_b": 1, "grad_overhead_b": 0,
                   "optstate_analytic_b": 1, "optstate_overhead_b": 0, "activations_b": 1,
                   "wire_overhead_b": 2, "io_b": 1, "pool_backward_b": 1, "dx_peak_b": 1,
                   "mcu_total_b": 311, "heap_peak_b": 400, "stack_peak_b": 5, "rss_peak_kb": 1,
                   "reconciliation_gap_b": 89},
    }
    path = tmp_path / "bfp.json"
    dump_log(path, log)
    loaded = load_log(path)
    assert loaded == log
    assert "sym_bits" not in loaded["memory"]
