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
             "BFP_MATH", "BFP_GRADS", "BFP_STATE", "BFP_ROUNDING", "LR_SCHEDULE", "BFP_HEAD")


def test_every_config_has_a_display_order_and_vice_versa():
    assert set(run_matrix.CONFIGS) == set(CONFIG_ORDER)
    assert list(plotting._MEM_CONFIG_ORDER) == list(CONFIG_ORDER)
    assert len(CONFIG_ORDER) == len(set(CONFIG_ORDER))


def test_bfp_names_encode_every_knob_and_round_trip():
    bfp = {n: v for n, v in run_matrix.CONFIGS.items() if n.startswith("bfp_")}
    assert len(bfp) == 15
    for name, (binary, env) in bfp.items():
        assert binary == "train_c_har_classifier_bfp"
        assert set(env) == set(BFP_KNOBS), name
        assert run_matrix.bfp_config_name(env) == name


def test_bfp8full_aliases_pin_the_full_integer_path_with_an_explicit_head():
    """bfp8full = bfp8 (m8 e8, per-channel weights, ab16, native, sr) + BFP grads
    + BFP momentum. The head stays an explicit choice: one alias per BFP_HEAD
    value, no bare `bfp8full`."""
    assert "bfp8full" not in run_matrix.CONFIGS
    bfp8 = {"BFP_WEIGHT_BLOCK": "channel", "BFP_WIRE_BLOCK": "16", "BFP_MANTISSA_BITS": "8",
            "BFP_EXPONENT_BITS": "8", "BFP_MATH": "native", "BFP_GRADS": "1",
            "BFP_STATE": "1", "BFP_ROUNDING": "sr", "LR_SCHEDULE": "none"}
    for alias, head in (("bfp8full_hflt", "float"), ("bfp8full_hbfp", "bfp")):
        binary, env = run_matrix.CONFIGS[alias]
        assert binary == "train_c_har_classifier_bfp"
        assert set(env) == set(BFP_KNOBS), alias
        assert env == {**bfp8, "BFP_HEAD": head}, alias


def test_legacy_names_are_untouched():
    for legacy in ("float", "sym8", "sym8det", "sym4g32", "asym6pc", "adamw"):
        assert legacy in run_matrix.CONFIGS
        assert run_matrix.CONFIGS[legacy][0] != "train_c_har_classifier_bfp"


def test_bfp_train_config_keys_are_declared_in_the_schema():
    declared = set(TrainConfig.__annotations__)
    for key in ("weight_dtype", "mantissa_bits", "exponent_bits", "weight_block", "wire_block",
                "wires_resolved", "bfp_math", "bfp_grads", "bfp_state", "bfp_rounding", "bfp_head",
                "groups_resolved", "group_overhead_b"):
        assert key in declared, key


def test_full_integer_configs_need_pre_converted_bfp_data():
    """The trainer's FULL predicate (native math, BFP wires, BFP_GRADS=1,
    BFP_STATE=1) trains on data written by `--write-bfp-data`; one write per
    data format (m, e, wire block), whatever the head."""
    full = {n for n, (_, env) in run_matrix.CONFIGS.items() if run_matrix.needs_bfp_data(env)}
    assert {"bfp8full_hflt", "bfp8full_hbfp"} <= full
    assert run_matrix.needs_bfp_data(run_matrix._bfp(BFP_GRADS="1", BFP_STATE="1")[1])
    for env in (run_matrix._bfp()[1],
                run_matrix._bfp(BFP_GRADS="1")[1],
                run_matrix._bfp(BFP_GRADS="1", BFP_STATE="1", BFP_MATH="fq")[1],
                run_matrix._bfp(BFP_GRADS="1", BFP_STATE="1", BFP_WIRE_BLOCK="float",
                                BFP_HEAD="float")[1],
                {"SYM_BITS": "8"}, {}):
        assert not run_matrix.needs_bfp_data(env)
    keys = {run_matrix.bfp_data_key(run_matrix.CONFIGS[n][1])
            for n in ("bfp8full_hflt", "bfp8full_hbfp")}
    assert keys == {("8", "8", "16")}


def test_crashed_run_counts_as_diverged_not_dropped(tmp_path, monkeypatch):
    """A non-zero trainer exit (SYM scale fail-fast on a non-finite fit) must keep
    its seed in the per-config table as diverged=1 with no accuracy."""
    import subprocess

    def boom(*args, **kwargs):
        raise subprocess.CalledProcessError(1, "trainer", stderr="scale fail-fast")

    monkeypatch.setattr(run_matrix, "run_one", boom)
    row = run_matrix.run_pair("sym8", 3, 2, tmp_path, tmp_path)
    assert row["diverged"] == 1 and row["test_acc"] is None and row["seed"] == 3
    ok = {"config": "sym8", "seed": 1, "test_acc": 0.9, "wall": 1.0, "diverged": 0,
          "returncode": 0, "stderr": ""}
    text = run_matrix.format_summary([ok, row])
    assert "sym8: 2 seeds, 1 diverged" in text
    assert "seed3   test_acc=N/A diverged=1" in text
    assert "diverged: 1/2 runs" in text


def test_clean_run_reads_diverged_from_its_log(tmp_path, monkeypatch):
    monkeypatch.setattr(run_matrix, "run_one", lambda *a, **k: (0.5, 1.0))
    (tmp_path / "float_seed1.json").write_text('{"final": {"diverged": 1}}')
    assert run_matrix.run_pair("float", 1, 2, tmp_path, tmp_path)["diverged"] == 1
    assert run_matrix.run_pair("float", 2, 2, tmp_path, tmp_path)["diverged"] == 0


LEAKY_PARENT_KNOBS = {"SYM_BWD": "1", "SYM_WIRES": "1", "GROUP_MODE": "channel"}


def _capture_child_envs(tmp_path, monkeypatch):
    """Fake trainer binaries + a subprocess.run spy recording each child env."""
    import subprocess

    for binary in {b for b, _ in run_matrix.CONFIGS.values()}:
        (tmp_path / binary).write_text("")
    envs = []

    def spy(cmd, **kwargs):
        envs.append(kwargs["env"])
        return subprocess.CompletedProcess(cmd, 0, stdout="FINAL test_acc=0.5\n", stderr="")

    monkeypatch.setattr(run_matrix.subprocess, "run", spy)
    for key, value in LEAKY_PARENT_KNOBS.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("PATH_SENTINEL_FOR_TEST", "kept")
    return envs


def test_parent_shell_knobs_do_not_leak_into_arms_that_do_not_set_them(tmp_path, monkeypatch):
    """An exported SYM_BWD=1 must not turn the `sym8` arm into an integer-backward
    run under the sym8 label: each arm gets exactly its own config knobs."""
    envs = _capture_child_envs(tmp_path, monkeypatch)
    run_matrix.run_one("sym8", 1, 2, tmp_path, tmp_path)
    sym8 = envs[-1]
    for key in LEAKY_PARENT_KNOBS:
        assert key not in sym8, key
    assert sym8["SYM_BITS"] == "8"
    assert sym8["PATH_SENTINEL_FOR_TEST"] == "kept"

    run_matrix.run_one("sym8bw", 1, 2, tmp_path, tmp_path)
    sym8bw = envs[-1]
    assert sym8bw["SYM_BWD"] == "1" and sym8bw["SYM_WIRES"] == "1"
    assert "GROUP_MODE" not in sym8bw


def test_bfp_data_writer_does_not_inherit_parent_shell_knobs(tmp_path, monkeypatch):
    envs = _capture_child_envs(tmp_path, monkeypatch)
    monkeypatch.setenv("BFP_HEAD", "float")
    monkeypatch.setenv("SYM_BITS", "4")
    run_matrix.write_bfp_data(["bfp8full_hbfp"], tmp_path)
    env = envs[-1]
    assert env["BFP_HEAD"] == "bfp"
    for key in (*LEAKY_PARENT_KNOBS, "SYM_BITS"):
        assert key not in env, key
