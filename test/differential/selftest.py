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
from fractions import Fraction
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


@check
def integer_fold_helpers_follow_section_13():
    # HALF_AWAY ties both signs, k = 0 identity, k >= 32 drops (the exact -1/2 rounds away)
    assert [bfp_ref.shr_half_away(v, 1) for v in (7, -7, 5, -5, 1, -1)] == [4, -4, 3, -3, 1, -1]
    assert bfp_ref.shr_half_away(11, 2) == 3 and bfp_ref.shr_half_away(9, 2) == 2
    assert bfp_ref.shr_half_away(bfp_ref.INT32_MIN, 0) == bfp_ref.INT32_MIN
    assert bfp_ref.shr_half_away(bfp_ref.INT32_MAX, 32) == 0 and bfp_ref.shr_half_away(bfp_ref.INT32_MIN, 32) == -1
    assert bfp_ref.align_partial(-5, -3) == -40 and bfp_ref.align_partial(7, 1) == 4
    # the guard from the product bound: HAR widths, the conv gold widths, the m16 limit cases
    assert bfp_ref.fold_guard_bits(80, 8, 8, 8) == -10
    assert bfp_ref.fold_guard_bits(6, 6, 4, 8) == -20
    assert bfp_ref.fold_guard_bits(2, 16, 16, 0) == 1 and bfp_ref.fold_guard_bits(600, 16, 8, 0) == 1
    assert bfp_ref.fold_guard_bits(511, 16, 8, 0) == 0
    # g >= 2 (a search stepping back from g = 1 with an out-of-range shift returned 1 / raised)
    assert bfp_ref.fold_guard_bits(4, 16, 16, 0) == 2
    assert bfp_ref.fold_guard_bits(80, 16, 16, 0) == 6
    assert bfp_ref.fold_guard_bits(2400, 16, 8, 0) == 3
    assert bfp_ref.fold_guard_bits(4, 16, 16, 16) == 2
    assert bfp_ref.fold_guard_bits(2, 4, 4, 16) == -15  # bias-dominated
    # the inequality itself and minimality, exactly, over a sweep
    for n in (1, 2, 4, 7, 80, 511, 512, 2400, 100000):
        for ma in (2, 8, 16):
            for mb in (4, 16):
                for m_bias in (0, 8, 16):
                    total = n * 2 ** (ma + mb - 2) + (2 ** (m_bias - 1) if m_bias else 0)

                    def holds(gg):
                        return Fraction(total) * Fraction(2) ** (-gg) + (n + 1) <= bfp_ref.INT32_MAX

                    g = bfp_ref.fold_guard_bits(n, ma, mb, m_bias)
                    assert holds(g), (n, ma, mb, m_bias, g)
                    assert g == -30 or not holds(g - 1), (n, ma, mb, m_bias, g)


@check
def integer_fold_reblock_matches_the_pack_where_it_is_exact():
    # A raw whose mantissas already fit m=8 under rawE: stored = rawE + bias, codes verbatim.
    mant = np.array([127, -127, 3, 0, 64, -1, 2, 9], dtype=np.int64)
    codes, stored = bfp_ref.reblock_raw(mant, np.array([-3, 5]), 4, 8, 8, 4)
    assert list(stored) == [124, 132] and list(codes) == list(mant)
    # A zero group stores the bias whatever rawE says; a wide group shifts with HALF_AWAY.
    codes, stored = bfp_ref.reblock_raw(np.array([0, 0, 1000, -1001]), np.array([9, 0]), 2, 8, 8, 2)
    assert list(stored) == [127, 127 + 3] and list(codes) == [0, 0, 125, -125]
    # The dequant is float32: a 25-bit mantissa rounds to nearest even.
    deq = bfp_ref.dequant_raw_f32(np.array([2**24 + 1, 2**24 + 3]), np.array([0]), 0)
    assert list(deq) == [float(2**24), float(2**24 + 4)]


@check
def linear_int_fold_follows_section_13_by_hand():
    # x [1, 2] m16 floors at E = 0, W [1, 2] m16 floors: N = 2 > limit 1, g = 1,
    # two segments of 2^30 each halved exactly -> mant 2^30 at E_acc = 1.
    bits16 = (16, 8, 0)
    f = bfp_ref.linear_int_fold(np.array([-32768, -32768]), np.array([127]), bits16,
                                np.array([-32768, -32768]), np.array([127]), bits16, None, 1, 2, 1, 0)
    assert f.g == 1 and list(f.exps) == [1] and list(f.mant) == [2**30] and f.n_splits == 1
    # E_top = row max of x + row max of W, max'ed with the bias exponent; two outputs
    # in one raw group take the larger. x row: E {+3, -2}; W rows: {0} and {+4}
    # (gs 2, one group per row); bias E {+10, -9} -> tops {10, 7}, E_acc = 10 + g.
    bits8 = (8, 8, 2)
    f = bfp_ref.linear_int_fold(np.array([1, 1, 1, 1]), np.array([130, 125]), bits8,
                                np.array([1, 1, 1, 1, 1, 1, 1, 1]), np.array([127, 131]), (8, 8, 4),
                                (np.array([3, -3]), np.array([137, 118]), (8, 8, 1)), 1, 4, 2, 0)
    assert f.g == bfp_ref.fold_guard_bits(4, 8, 8, 8) and list(f.exps) == [10 + f.g]
    # the exact sums: out0 = 3*2^10 + 2*(2^3 + 2^-2), out1 = -3*2^-9 + 2*(2^7 + 2^2)
    assert list(f.exact) == [3 * 2**10 + 2 * (2**3 + 2**-2), -3 * 2**-9 + 2 * (2**7 + 2**2)]


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
