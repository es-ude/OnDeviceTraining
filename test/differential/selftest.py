#!/usr/bin/env python3
"""Self-test of the harness's own judgement code (no C build needed).

    uv run test/differential/selftest.py

Each check pins one place where a wrong harness would call a broken C result
good: the row verdict on a failed gold check, the guard-banner pattern on
grouped banners, the zero-length conv reference, the int32 fit range of the
overflow report and the float64 clamp in the BFP reference pack.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import run_cells  # noqa: E402
from ref import bfp_ref, exact_ref, sym_ref  # noqa: E402

CHECKS = []


def check(fn):
    CHECKS.append(fn)
    return fn


def _passing_row(**extra):
    leg = {"taxonomy": "EXACT", "scale": {"ok": True}}
    row = {"legA": dict(leg), "legB": {"taxonomy": "BOUNDED"}}
    row.update(extra)
    return row


@check
def failed_gold_check_fails_the_row():
    assert run_cells.verdict_of(_passing_row(gold=None), {}) == "PASS"
    assert run_cells.verdict_of(_passing_row(gold={"status": "match"}), {}) == "PASS"
    assert run_cells.verdict_of(_passing_row(gold={"status": "mismatch"}), {}) == "FAIL"
    assert run_cells.verdict_of(_passing_row(gold={"status": "n/a"}), {}) == "FAIL"


@check
def guard_banner_pattern_matches_grouped_and_weight_grad_banners():
    for op in (
        "matmul SYM_INT32",
        "matmul SYM_INT32 grouped (raw group partial)",
        "matmul SYM_INT32 grouped (rescaled accumulator)",
        "conv1dKernelSymInt32Grouped (rescaled accumulator)",
        "conv1dCalcWeightGradsSymInt32 (batch*outputLength)",
    ):
        text = (
            f"\x1b[0;31m[MATMUL: f] {op}: SYM int32 accumulator overflow -- N=7 terms, "
            "operand widths (12, 13), max|seed|=0; narrow ODT_SYM_OPERAND_QMAXBITS\x1b[0m"
        )
        got = run_cells.check_sym_guard_banner(text, op, 7, (12, 13), (0, 0))
        assert got["ok"], (op, got)


@check
def conv1d_exact_returns_an_empty_result_for_a_zero_length_output():
    x = np.ones((2, 3, 4))
    w = np.ones((5, 3, 6))  # kernel 6 > length 4
    y = exact_ref.conv1d_exact(x, w, None, 1, 1, 1, 0, 0)
    assert y.shape == (2, 5, 0), y.shape
    y = exact_ref.conv1d_exact(x, np.ones((5, 3, 3)), None, 1, 2, 1, 0, 0)  # effective kernel 5 > 4
    assert y.shape == (2, 5, 0), y.shape
    assert exact_ref.conv1d_exact(x, np.ones((5, 3, 3)), None, 1, 1, 1, 0, 0).shape == (2, 5, 2)


@check
def overflow_report_accepts_int32_min():
    raw = np.array([sym_ref.INT32_MIN, 5], dtype=np.int64)
    assert sym_ref.overflow_report(raw, 2, 12, 12).fits
    below = np.array([sym_ref.INT32_MIN - 1], dtype=np.int64)
    assert not sym_ref.overflow_report(below, 1, 12, 12).fits
    above = np.array([sym_ref.INT32_MAX + 1], dtype=np.int64)
    assert not sym_ref.overflow_report(above, 1, 12, 12).fits


@check
def bfp_reference_pack_clamps_before_the_integer_cast():
    huge = np.array([3.0e38, -3.0e38, 1.0], dtype=np.float32)
    stored = np.array([0], dtype=np.int64)  # 3e38 / 2^(0 - 127) ~ 5e76, past int64
    with np.errstate(all="raise"):  # the platform-dependent int64 cast must not even happen
        got = bfp_ref.repack_under(huge, stored, 8, 8, 0)
    q, floor = bfp_ref.qmax(8), bfp_ref.code_floor_twos_complement(8)
    assert list(got.codes) == [q, floor, q], list(got.codes)
    assert got.n_clamped == 3, got.n_clamped
    # an in-range value still rounds half away from zero
    mid = bfp_ref.repack_under(np.array([2.5, -2.5], dtype=np.float32), np.array([127]), 8, 8, 0)
    assert list(mid.codes) == [3, -3], list(mid.codes)


def main() -> int:
    failed = 0
    for fn in CHECKS:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL {fn.__name__}: {type(exc).__name__}: {exc}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
