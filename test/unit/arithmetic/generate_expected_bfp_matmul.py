#!/usr/bin/env python3
"""Generate expected_bfp_matmul.h for UnitTestMatmul (BFP epic PR2, Task 3 --
spec docs/superpowers/specs/2026-07-29-block-floating-point-design.md).

Pins matmulBfpTensors' fold order (Matmul.c): per output element ONE int32
partial; per reduction step both operands' storage indices -> group ids
(bfpGroupOf); when EITHER id changes the partial folds into a float32
accumulator via ldexpf((float)partial, Ea + Eb - biasA - biasB) and resets;
tail fold after the loop; bias is a value-seed dequantized to float BEFORE
the reduction ((float)mantissa * bfpGroupScale). The kernel never rounds --
see sym_gold.matmul_bfp_ref for the exact np.float32-mirrored emulation.

Fixture geometry (2x6 @ 6x3 -> 2x3): `a` grouped numGroups=3/groupSize=4
(m=6, e=8) so each reduction row crosses one a-group boundary; `b` grouped
numGroups=9/groupSize=2 (m=4, e=8) in the GEMM-weight storage order
[outCols=3, reduceLen=6] behind a bOrder {1,0} transposed logical view (the
same strided-walk wiring the twin test at UnitTestMatmul.c:643 uses), so each
column's walk crosses two b-group boundaries; bias per-tensor (m=8, e=8).
Input values are SMALL and grid-exact (every code * scale reproduces the
input float bit-for-bit -- asserted below), so every fold is exact float32
arithmetic and the expected outputs are bit-pinned via
TEST_ASSERT_EQUAL_MEMORY, not a tolerance.

Self-checks (abort generation rather than emit a vacuous fixture):
  - matmul_bfp_ref's built-in (i)-(iii): >= 2 groups crossed on EACH operand
    somewhere; >= 1 fold with a nonzero exactly-float-convertible partial;
    the grouped result differs from an all-per-tensor (exponents[0]) collapse
    -- (iii) is what makes the boundary-fold mutation (folding only at the
    tail) observable in the gold test.
  - exact-quantization roundtrip: dequantizing the emitted codes reproduces
    the input floats bit-for-bit (the exact-float-regime claim).
  - both operands' exponent arrays are non-uniform (a uniform array would
    make every fold shift identical, hiding group-exponent mix-ups).
  - the with-bias and no-bias expected outputs differ elementwise (a kernel
    that ignores the bias seed cannot pass both gold tests).

Run via `uv run` (CMake wires this automatically, see CMakeLists.txt).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "goldgen"))

from sym_gold import (assert_rounding_canary, bfp_quantize_grouped, emit_float_array,
                      emit_int32_array, emit_int32_scalar, matmul_bfp_ref)

OUT_ROWS = 2
OUT_COLS = 3
REDUCE_LEN = 6

# a: [out_rows=2, reduce_len=6] row-major, quantization groups of 4 -> group 1
# straddles the row boundary (row 0 crosses groups {0,1}, row 1 {1,2}).
A_VALUES = [1.0, -2.0, 3.0, -1.5, 10.0, -6.0,
            4.0, 8.0, 0.5, -0.25, 0.75, 1.0]
A_QC = {"mantissa_bits": 6, "exponent_bits": 8, "group_size": 4}
A_NUM_GROUPS = 3

# b: [out_cols=3, reduce_len=6] STORAGE order (row c = output channel c's
# weights -- the GEMM-weight layout the bOrder {1,0} view exposes with the
# reduction axis logical-first), groups of 2 -> three groups per channel.
B_VALUES = [7.0, -3.0, 1.0, 2.5, -1.75, 0.75,
            3.0, -2.0, 0.5, -0.5, -6.0, 2.0,
            1.5, 1.0, -0.75, -1.25, 2.0, -1.0]
B_QC = {"mantissa_bits": 4, "exponent_bits": 8, "group_size": 2}
B_NUM_GROUPS = 9

BIAS_VALUES = [2.0, -1.5, 0.5]
BIAS_QC = {"mantissa_bits": 8, "exponent_bits": 8, "group_size": 0}


def emit_uint8_array(name: str, values) -> str:
    vals = [int(v) for v in values]
    assert all(0 <= v <= 255 for v in vals), f"{name}: value outside uint8 range"
    body = ", ".join(str(v) for v in vals)
    return (
        f"static const uint8_t {name}[] = {{ {body} }};\n"
        f"static const size_t {name}_len = {len(vals)};\n"
    )


def check_exact_roundtrip(name, values, codes, exps, qc):
    """Exact-float-regime pin: code * 2^(stored - bias) must reproduce the
    input float bit-for-bit (float32 multiply by a power of two is exact)."""
    bias = 2 ** (qc["exponent_bits"] - 1) - 1
    gsz = len(values) if qc["group_size"] == 0 else qc["group_size"]
    for i, v in enumerate(values):
        scale = np.float32(np.ldexp(np.float32(1.0), np.int32(exps[i // gsz] - bias)))
        deq = float(np.float32(np.float32(codes[i]) * scale))
        assert deq == v, (
            f"{name}: element {i} dequantizes to {deq}, not {v} -- fixture left "
            "the exact float regime; pick grid-exact values")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    assert_rounding_canary()

    a_codes, a_exps = bfp_quantize_grouped(A_VALUES, A_QC["mantissa_bits"],
                                           A_QC["exponent_bits"], A_QC["group_size"])
    b_codes, b_exps = bfp_quantize_grouped(B_VALUES, B_QC["mantissa_bits"],
                                           B_QC["exponent_bits"], B_QC["group_size"])
    bias_codes, bias_exps = bfp_quantize_grouped(BIAS_VALUES, BIAS_QC["mantissa_bits"],
                                                 BIAS_QC["exponent_bits"],
                                                 BIAS_QC["group_size"])
    assert len(a_exps) == A_NUM_GROUPS and len(b_exps) == B_NUM_GROUPS
    assert len(bias_exps) == 1

    check_exact_roundtrip("a", A_VALUES, a_codes, a_exps, A_QC)
    check_exact_roundtrip("b", B_VALUES, b_codes, b_exps, B_QC)
    check_exact_roundtrip("bias", BIAS_VALUES, bias_codes, bias_exps, BIAS_QC)

    # Non-uniform exponents per operand: uniform arrays would make every fold
    # shift identical, hiding a curGa/curGb mix-up behind equal scales.
    assert len(set(a_exps)) >= 2, "a: exponent array is uniform -- fixture too weak"
    assert len(set(b_exps)) >= 2, "b: exponent array is uniform -- fixture too weak"

    expected = matmul_bfp_ref(a_codes, a_exps, A_QC, b_codes, b_exps, B_QC,
                              bias_codes, bias_exps, BIAS_QC,
                              OUT_ROWS, OUT_COLS, REDUCE_LEN, b_transposed=True)
    expected_no_bias = matmul_bfp_ref(a_codes, a_exps, A_QC, b_codes, b_exps, B_QC,
                                      None, None, None,
                                      OUT_ROWS, OUT_COLS, REDUCE_LEN, b_transposed=True)

    # A bias-ignoring kernel must not be able to pass both gold tests.
    assert all(w != n for w, n in zip(expected, expected_no_bias)), (
        "with-bias and no-bias expectations coincide somewhere -- bias seed "
        "would be unobservable there")

    parts = [
        "// AUTOGENERATED by generate_expected_bfp_matmul.py — DO NOT EDIT\n",
        "#ifndef ODT_EXPECTED_BFP_MATMUL_H\n",
        "#define ODT_EXPECTED_BFP_MATMUL_H\n",
        "#include <stdint.h>\n",
        "#include <stdlib.h>\n\n",
        emit_int32_scalar("kBfpOutRows", OUT_ROWS),
        emit_int32_scalar("kBfpOutCols", OUT_COLS),
        emit_int32_scalar("kBfpReduceLen", REDUCE_LEN),
        emit_int32_array("kBfpACodes", torch.tensor(a_codes)),
        emit_uint8_array("kBfpAExponents", a_exps),
        emit_int32_scalar("kBfpANumGroups", A_NUM_GROUPS),
        emit_int32_scalar("kBfpAGroupSize", A_QC["group_size"]),
        emit_int32_scalar("kBfpAMantissaBits", A_QC["mantissa_bits"]),
        emit_int32_scalar("kBfpAExponentBits", A_QC["exponent_bits"]),
        emit_int32_array("kBfpBCodes", torch.tensor(b_codes)),
        emit_uint8_array("kBfpBExponents", b_exps),
        emit_int32_scalar("kBfpBNumGroups", B_NUM_GROUPS),
        emit_int32_scalar("kBfpBGroupSize", B_QC["group_size"]),
        emit_int32_scalar("kBfpBMantissaBits", B_QC["mantissa_bits"]),
        emit_int32_scalar("kBfpBExponentBits", B_QC["exponent_bits"]),
        emit_int32_array("kBfpBiasCodes", torch.tensor(bias_codes)),
        emit_uint8_array("kBfpBiasExponents", bias_exps),
        emit_int32_scalar("kBfpBiasMantissaBits", BIAS_QC["mantissa_bits"]),
        emit_int32_scalar("kBfpBiasExponentBits", BIAS_QC["exponent_bits"]),
        emit_float_array("kBfpMatmulExpected", torch.tensor(expected, dtype=torch.float32)),
        emit_float_array("kBfpMatmulNoBiasExpected",
                         torch.tensor(expected_no_bias, dtype=torch.float32)),
        "\n#endif // ODT_EXPECTED_BFP_MATMUL_H\n",
    ]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(parts))
    return 0


if __name__ == "__main__":
    sys.exit(main())
