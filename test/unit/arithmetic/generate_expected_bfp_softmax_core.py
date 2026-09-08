#!/usr/bin/env python3
"""Generate expected_bfp_softmax_core.h for UnitTestBfpSoftmaxExp (BFP epic
PR6, Task 3 -- integer i-exp core, I-BERT Algorithm 3 on the fixed 2^-14 work
grid, plus the 3-mode shift-rounding helper).

Everything integer is mirrored in Python ints (arbitrary precision, >> is a
floor shift -- identical to C two's-complement arithmetic >> for every value
that fits int32), so the emitted expectations are bit-exact.

Constants are RE-DERIVED here (math.floor over IEEE-754 doubles, the same
arithmetic C's floor(log(2.0) * 16384.0) performs) and emitted as EXPECTED_*;
the C test pins the header #defines against BOTH these and an independent
in-C double recomputation. A hard assert against the spec literals implements
the "if the derivation disagrees, STOP" rule from the numerics spec.

Self-checks (abort generation rather than emit a vacuous fixture):
  - constants match the numerics-spec literals (11356 / 22167 / 257578233)
    and qL(0) = QB^2 + QC = 748,954,122 (the exp(0) ~ 1.000241 reference);
  - accuracy sweep: max |float32(iexp * 0.3585 * 2^-28) - exp(x)| < 2.5e-3
    over 64 points on [-20, 0] (kills wrong-constant and wrong-z mutants);
  - vacuity: at least one accuracy vector has z >= 2 (a z<2-only sweep would
    never exercise the multi-bit renormalization shift);
  - at least one i-exp vector where HALF_AWAY differs from TRUNC (otherwise
    the mode dispatch is unobservable in the golden vectors);
  - at least one shift vector where the floor shift differs from C
    truncating division v / 2^k (pins arithmetic->> semantics on negatives).

Run via `uv run` (CMake wires this automatically, see CMakeLists.txt).
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np

F = 14
QLN2 = math.floor(math.log(2) * 2**F)
QB = math.floor(1.353 * 2**F)
QC = math.floor(0.344 * 2**28 / 0.3585)
A = 0.3585
ZMAX = 31
QFLOOR = -(ZMAX + 1) * QLN2

INT32_MIN = -(2**31)

SHIFT_V = [13, -13, 22167, -22167, 5, -5, 0, INT32_MIN // 4]
SHIFT_K = [0, 1, 3, 14, 31]

IEXP_QW = [
    0,
    -1,
    -QLN2,
    -QLN2 - 1,
    -3 * QLN2 - 500,
    -100000,
    QFLOOR + 1,
    QFLOOR,
    QFLOOR - 5,
    INT32_MIN,
]

ACC_N = 64


def shift_trunc(v: int, k: int) -> int:
    if k == 0:
        return v
    return v >> min(k, 31)


def shift_half_away(v: int, k: int) -> int:
    if k == 0:
        return v
    kk = min(k, 31)
    sign = -1 if v < 0 else 1
    mag = -v if v < 0 else v
    return sign * ((mag + (1 << (kk - 1))) >> kk)


def shift_rounded(v: int, k: int, mode: str) -> int:
    return shift_trunc(v, k) if mode == "TRUNC" else shift_half_away(v, k)


def iexp(qw: int, mode: str) -> int:
    """Mirror of bfpIExpQ (numerics spec, i-exp core block) in Python ints."""
    if qw <= QFLOOR:
        return 0
    z = (-qw) // QLN2  # both operands non-negative -> C division is floor too
    qp = qw + z * QLN2
    ql = (qp + QB) * (qp + QB) + QC
    assert -QLN2 < qp <= 0, f"qp {qp} out of (-QLN2, 0] for qW {qw}"
    assert 0 <= z <= ZMAX, f"z {z} out of [0, ZMAX] for qW {qw}"
    assert ql < 2**31, f"qL {ql} overflows int32 for qW {qw}"
    return shift_rounded(ql, z, mode)


def fmt_i32(v: int) -> str:
    assert -(2**31) <= v < 2**31, f"{v} does not fit int32"
    # INT32_MIN literal would be parsed as -(2147483648) -> long; emit the
    # canonical C idiom instead.
    return "(-2147483647 - 1)" if v == INT32_MIN else str(v)


def emit_i32_array(name: str, values: list[int]) -> str:
    body = ", ".join(fmt_i32(v) for v in values)
    return f"static const int32_t {name}[{len(values)}] = {{{body}}};\n"


def emit_u32_array(name: str, values: list[int]) -> str:
    body = ", ".join(f"{v}u" for v in values)
    return f"static const uint32_t {name}[{len(values)}] = {{{body}}};\n"


def emit_i32_matrix(name: str, rows: list[list[int]]) -> str:
    inner = ",\n    ".join("{" + ", ".join(fmt_i32(v) for v in r) + "}" for r in rows)
    return (
        f"static const int32_t {name}[{len(rows)}][{len(rows[0])}] = {{\n"
        f"    {inner}}};\n"
    )


def emit_double_array(name: str, values: list[float]) -> str:
    body = ", ".join(repr(v) for v in values)
    return f"static const double {name}[{len(values)}] = {{{body}}};\n"


def emit_float_array(name: str, values: list[float]) -> str:
    body = ", ".join(repr(float(np.float32(v))) + "f" for v in values)
    return f"static const float {name}[{len(values)}] = {{{body}}};\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    # --- STOP gate: derivation must reproduce the numerics-spec literals ---
    assert QLN2 == 11356, f"QLN2 derivation drifted: {QLN2} != 11356 -- STOP"
    assert QB == 22167, f"QB derivation drifted: {QB} != 22167 -- STOP"
    assert QC == 257578233, f"QC derivation drifted: {QC} != 257578233 -- STOP"
    assert QFLOOR == -363392, f"QFLOOR {QFLOOR} != -363392 -- STOP"
    assert QB * QB + QC == 748954122, "qL(0) reference value drifted -- STOP"

    # --- shift vectors ---
    trunc_rows = [[shift_trunc(v, k) for k in SHIFT_K] for v in SHIFT_V]
    half_rows = [[shift_half_away(v, k) for k in SHIFT_K] for v in SHIFT_V]

    assert any(
        v < 0 and k > 0 and shift_trunc(v, k) != -((-v) >> k)
        for v in SHIFT_V
        for k in SHIFT_K
    ), "vacuous: no vector distinguishes floor shift from truncating division"

    # --- i-exp vectors ---
    iexp_trunc = [iexp(qw, "TRUNC") for qw in IEXP_QW]
    iexp_half = [iexp(qw, "HALF_AWAY") for qw in IEXP_QW]
    assert iexp_trunc[0] == 748954122, "qW=0 must hit the qL(0) reference"
    assert any(
        t != h for t, h in zip(iexp_trunc, iexp_half)
    ), "vacuous: HALF_AWAY indistinguishable from TRUNC in the i-exp vectors"

    # --- value-accuracy sweep: 64 evenly spaced x on [-20, 0], TRUNC ---
    acc_x = [-20.0 + 20.0 * i / (ACC_N - 1) for i in range(ACC_N)]
    acc_qw = [round(x * 2**F) for x in acc_x]
    acc_q = [iexp(qw, "TRUNC") for qw in acc_qw]
    acc_val = [float(np.float32(q * A * 2**-28)) for q in acc_q]

    gaps = [abs(v - math.exp(x)) for v, x in zip(acc_val, acc_x)]
    assert max(gaps) < 2.5e-3, (
        f"accuracy envelope violated: max |iexp - exp| = {max(gaps):.3e}"
    )
    assert any(
        (-qw) // QLN2 >= 2 for qw in acc_qw if qw > QFLOOR
    ), "vacuous: no accuracy vector exercises a renormalization shift z >= 2"

    parts = [
        "// Generated by generate_expected_bfp_softmax_core.py -- do not edit.\n",
        "#ifndef ODT_EXPECTED_BFP_SOFTMAX_CORE_H\n",
        "#define ODT_EXPECTED_BFP_SOFTMAX_CORE_H\n",
        "#include <stdint.h>\n\n",
        f"#define EXPECTED_QLN2 {QLN2}\n",
        f"#define EXPECTED_QB {QB}\n",
        f"#define EXPECTED_QC {QC}\n",
        f"#define EXPECTED_QFLOOR ({QFLOOR})\n\n",
        f"#define SHIFT_NUM_V {len(SHIFT_V)}\n",
        f"#define SHIFT_NUM_K {len(SHIFT_K)}\n",
        emit_i32_array("kShiftV", SHIFT_V),
        emit_u32_array("kShiftK", SHIFT_K),
        emit_i32_matrix("kShiftTruncExpected", trunc_rows),
        emit_i32_matrix("kShiftHalfAwayExpected", half_rows),
        "\n",
        f"#define IEXP_NUM_QW {len(IEXP_QW)}\n",
        emit_i32_array("kIExpQw", IEXP_QW),
        emit_i32_array("kIExpTruncExpected", iexp_trunc),
        emit_i32_array("kIExpHalfAwayExpected", iexp_half),
        "\n",
        f"#define IEXP_ACC_N {ACC_N}\n",
        emit_double_array("kIExpAccX", acc_x),
        emit_i32_array("kIExpAccQw", acc_qw),
        emit_i32_array("kIExpAccTruncExpected", acc_q),
        emit_float_array("kIExpAccValue", acc_val),
        "\n#endif // ODT_EXPECTED_BFP_SOFTMAX_CORE_H\n",
    ]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(parts))
    return 0


if __name__ == "__main__":
    sys.exit(main())
