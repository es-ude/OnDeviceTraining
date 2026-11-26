"""The TC-ResNet8 same-start rule (examples/kws_tcresnet8/same_start.py): step-1 loss, the state
after one update against a float64 twin, and eval argmax agreement. Boundary values are exactly
representable so each case sits on, or just past, the rule's edge."""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from examples.kws_tcresnet8.same_start import KINDS, same_start_verdict  # noqa: E402

ROWS = 200


def _state(value: float) -> dict[str, list[np.ndarray]]:
    """Every kind holds two tensors; the first one's largest entry is `value`, the second is 0.5."""
    return {k: [np.array([value, 0.25]), np.array([0.5])] for k in KINDS}


def _probs(labels: np.ndarray) -> np.ndarray:
    p = np.full((len(labels), 3), 0.1)
    p[np.arange(len(labels)), labels] = 0.8
    return p


def _verdict(step1_c=1.0, state_c=None, state_t32=None, agree=ROWS):
    labels = np.zeros(ROWS, dtype=np.int64)
    c_labels = labels.copy()
    c_labels[agree:] = 1
    return same_start_verdict(
        step1_c=step1_c, step1_t32=1.0,
        state_c=state_c or _state(1.0), state_t32=state_t32 or _state(1.0), state_t64=_state(1.0),
        probs_c=_probs(c_labels), probs_t32=_probs(labels))


def test_identical_runs_pass():
    assert _verdict()["pass"]


@pytest.mark.parametrize(("c", "passes"), [(10001.0, True), (10001.01, False), (9998.99, False)])
def test_step1_relative_tolerance(c, passes):
    labels = np.zeros(ROWS, dtype=np.int64)
    v = same_start_verdict(step1_c=c, step1_t32=10000.0, state_c=_state(1.0),
                           state_t32=_state(1.0), state_t64=_state(1.0),
                           probs_c=_probs(labels), probs_t32=_probs(labels))
    assert v["step1"]["pass"] is passes
    assert v["pass"] is passes


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize(("c_err", "passes"), [(10 * 2**-10, True), (11 * 2**-10, False)])
def test_state_bound_is_ten_times_t32(kind, c_err, passes):
    t32 = _state(1.0)
    t32[kind] = [np.array([1.0 + 2**-10, 0.25]), np.array([0.5])]
    c = _state(1.0)
    c[kind] = [np.array([1.0 + c_err, 0.25]), np.array([0.5])]
    v = _verdict(state_c=c, state_t32=t32)
    assert v["state"][kind]["pass"] is passes
    assert v["pass"] is passes


@pytest.mark.parametrize(("c_err", "passes"), [(9e-6, True), (1.1e-5, False)])
def test_state_floor_when_t32_equals_t64(c_err, passes):
    # The error sits in the second tensor (max |t64| 0.5): normalised per tensor it is c_err;
    # normalising by the kind's overall max (1.0) would halve it and let 1.1e-5 pass.
    c = _state(1.0)
    c["running_var"] = [np.array([1.0, 0.25]), np.array([0.5 + 0.5 * c_err])]
    v = _verdict(state_c=c)
    assert v["state"]["running_var"]["bound"] == pytest.approx(1e-5)
    assert v["state"]["running_var"]["pass"] is passes
    assert v["pass"] is passes


@pytest.mark.parametrize(("agree", "passes"), [(199, True), (198, False)])
def test_eval_argmax_agreement(agree, passes):
    v = _verdict(agree=agree)
    assert v["eval"]["agree"] == agree
    assert v["eval"]["pass"] is passes
    assert v["pass"] is passes


def test_nan_c_state_fails():
    # In the second tensor: a plain max() over Python floats drops a NaN that is not first.
    c = _state(1.0)
    c["param"] = [np.array([1.0, 0.25]), np.array([np.nan])]
    v = _verdict(state_c=c)
    assert v["state"]["param"]["pass"] is False
    assert v["pass"] is False


def test_all_zero_reference_tensor_fails():
    # 0/0 gives a NaN distance, which must not read as "within the bound".
    zero = {k: [np.array([1.0, 0.25]), np.array([0.0])] for k in KINDS}
    labels = np.zeros(ROWS, dtype=np.int64)
    v = same_start_verdict(step1_c=1.0, step1_t32=1.0, state_c=zero, state_t32=zero,
                           state_t64=zero, probs_c=_probs(labels), probs_t32=_probs(labels))
    assert not any(s["pass"] for s in v["state"].values())
    assert v["pass"] is False


def test_non_finite_c_probabilities_fail():
    labels = np.zeros(ROWS, dtype=np.int64)
    probs_c = _probs(labels)
    probs_c[5, 1] = np.nan  # one disagreeing row: 199/200 would still pass the argmax count
    v = same_start_verdict(step1_c=1.0, step1_t32=1.0, state_c=_state(1.0),
                           state_t32=_state(1.0), state_t64=_state(1.0),
                           probs_c=probs_c, probs_t32=_probs(labels))
    assert v["eval"]["pass"] is False
    assert v["pass"] is False


@pytest.mark.parametrize(("c_rows", "t32_rows"), [(1, ROWS), (150, ROWS), (ROWS - 1, ROWS - 1)])
def test_eval_needs_both_sides_to_hold_exactly_the_eval_rows(c_rows, t32_rows):
    # A one-row C artifact broadcasts against the twin's 200 rows; a short pair of identical
    # artifacts would agree on every row it has.
    probs_c = _probs(np.zeros(c_rows, dtype=np.int64))
    probs_t32 = _probs(np.zeros(t32_rows, dtype=np.int64))
    v = same_start_verdict(step1_c=1.0, step1_t32=1.0, state_c=_state(1.0),
                           state_t32=_state(1.0), state_t64=_state(1.0),
                           probs_c=probs_c, probs_t32=probs_t32)
    assert v["eval"]["pass"] is False
    assert v["pass"] is False


def test_relative_paths_resolve_against_the_invocation_directory(tmp_path, monkeypatch):
    # The C child runs from the repo root, so it must receive the paths the driver writes to.
    from examples.kws_tcresnet8.same_start import parse_args

    monkeypatch.chdir(tmp_path)
    args = parse_args(["--data", "B", "--width", "half", "--out", "check", "--c-binary", "bin/c"])
    assert args.out == tmp_path.resolve() / "check"
    assert args.c_binary == tmp_path.resolve() / "bin" / "c"
