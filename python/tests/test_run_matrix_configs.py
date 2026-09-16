"""Contract test for the FIVE config consumers (spec 2026-09-14 §8): every config
name in run_matrix.CONFIGS must appear in compare_memory.CONFIG_ORDER and
plotting._MEM_CONFIG_ORDER, BFP names must round-trip to their env through the
name grammar, and the trainer's log keys must be the TrainConfig keys."""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from examples.har_classifier import run_matrix  # noqa: E402
from examples.har_classifier.compare_memory import CONFIG_ORDER  # noqa: E402
from examples._shared import plotting  # noqa: E402
from examples._shared.log_schema import TrainConfig  # noqa: E402

BFP_KNOBS = ("BFP_WEIGHT_BLOCK", "BFP_WIRE_BLOCK", "BFP_MANTISSA_BITS", "BFP_EXPONENT_BITS",
             "BFP_MATH", "BFP_GRADS", "BFP_STATE", "BFP_ROUNDING", "LR_SCHEDULE")


def test_every_config_has_a_display_order_and_vice_versa():
    assert set(run_matrix.CONFIGS) == set(CONFIG_ORDER)
    assert list(plotting._MEM_CONFIG_ORDER) == list(CONFIG_ORDER)
    assert len(CONFIG_ORDER) == len(set(CONFIG_ORDER))


def test_bfp_names_encode_every_knob_and_round_trip():
    bfp = {n: v for n, v in run_matrix.CONFIGS.items() if n.startswith("bfp_")}
    assert len(bfp) == 14
    for name, (binary, env) in bfp.items():
        assert binary == "train_c_har_classifier_bfp"
        assert set(env) == set(BFP_KNOBS), name
        assert run_matrix.bfp_config_name(env) == name


def test_legacy_names_are_untouched():
    for legacy in ("float", "sym8", "sym8det", "sym4g32", "asym6pc", "adamw"):
        assert legacy in run_matrix.CONFIGS
        assert run_matrix.CONFIGS[legacy][0] != "train_c_har_classifier_bfp"


def test_bfp_train_config_keys_are_declared_in_the_schema():
    declared = set(TrainConfig.__annotations__)
    for key in ("weight_dtype", "mantissa_bits", "exponent_bits", "weight_block", "wire_block",
                "wires_resolved", "bfp_math", "bfp_grads", "bfp_state", "bfp_rounding",
                "groups_resolved", "group_overhead_b"):
        assert key in declared, key
