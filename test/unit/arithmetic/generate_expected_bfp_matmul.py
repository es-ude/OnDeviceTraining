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

Fixture geometry (2x6 @ 6x3 -> 2x3): `a` grouped numGroups=4/groupSize=3
(m=6, e=8) so each reduction row crosses one a-group boundary at k=3 --
DISJOINT from b's boundaries {2, 4} by construction, so a fold that only
watches b's group id is observably wrong (review finding 1: with the earlier
groupSize=4 every a-boundary coincided with a b-boundary and the either-
operand fold clause was untested); `b` grouped numGroups=9/groupSize=2
(m=4, e=8) in the GEMM-weight storage order [outCols=3, reduceLen=6] behind
a bOrder {1,0} transposed logical view (the same strided-walk wiring the
twin test at UnitTestMatmul.c:643 uses), so each column's walk crosses two
b-group boundaries; bias per-tensor (m=8, e=8).
Input values are SMALL and grid-exact (every code * scale reproduces the
input float bit-for-bit -- asserted below), so every fold is exact float32
arithmetic and the expected outputs are bit-pinned via
TEST_ASSERT_EQUAL_MEMORY, not a tolerance.

Self-checks (abort generation rather than emit a vacuous fixture):
  - matmul_bfp_ref's built-in (i)-(iv): >= 2 groups crossed on EACH operand
    somewhere; >= 1 fold with a nonzero exactly-float-convertible partial;
    the grouped result differs from an all-per-tensor (exponents[0]) collapse
    -- (iii) is what makes the boundary-fold mutation (folding only at the
    tail) observable in the gold test; (iv) >= 1 reduction step where a's
    group changes while b's does NOT -- pins the EITHER-operand fold clause
    (a b-only fold condition is observably wrong).
  - exact-quantization roundtrip: dequantizing the emitted codes reproduces
    the input floats bit-for-bit (the exact-float-regime claim).
  - both operands' exponent arrays are non-uniform (a uniform array would
    make every fold shift identical, hiding group-exponent mix-ups).
  - the with-bias and no-bias expected outputs differ elementwise (a kernel
    that ignores the bias seed cannot pass both gold tests).

PR3 backward fixtures (BFP epic PR3, Task 1): one consistent Linear-backward
operand triple -- loss [batch=3, outF=4], W [outF=4, inF=4], x [batch=3,
inF=4] -- feeds three expectations: dx = loss @ W (RAW weight storage, the
reduction strides W by inF), dW = loss^T @ x (matmul_bfp_ref's a_transposed
view), and db[f] = sum_n loss[n][f] (matmul_bfp_bias_grad_ref). Geometry is
chosen so matmul_bfp_ref's self-check (iv) holds in BOTH orientations: with a
strided b walk, an a-only group boundary (a's group changes while b's does
NOT) requires the b operand's groupSize to EXCEED the b stride (inF), so W
uses groupSize 8 > 4 and x uses groupSize 6 > 4. Smaller group sizes make
every a-boundary coincide with a b-boundary, which is exactly the review-
finding-1 vacuity documented for the forward fixture above: the either-
operand fold clause (and the drop-a-clause mutant) would be unobservable.
batch is 3 because the weightGrad reduction (K = batch) needs one step where
x stays inside a group (Lx > inF) and another where it crosses -- impossible
with only one reduction step at batch=2.

Run via `uv run` (CMake wires this automatically, see CMakeLists.txt).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "goldgen"))

from sym_gold import (assert_rounding_canary, bfp_quantize_grouped, check_exact_roundtrip,
                      emit_float_array, emit_int32_array, emit_int32_scalar, emit_uint8_array,
                      matmul_bfp_bias_grad_ref, matmul_bfp_ref)

OUT_ROWS = 2
OUT_COLS = 3
REDUCE_LEN = 6

# a: [out_rows=2, reduce_len=6] row-major, quantization groups of 3 -> each
# row crosses ONE a-group boundary at k=3 (row 0: groups {0,1}, row 1:
# {2,3}), deliberately disjoint from b's per-column boundaries {2, 4}.
A_VALUES = [1.0, -2.0, 3.0, -1.5, 10.0, -6.0,
            4.0, 8.0, 0.5, -0.25, 0.75, 1.0]
A_QC = {"mantissa_bits": 6, "exponent_bits": 8, "group_size": 3}
A_NUM_GROUPS = 4

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

    # Grouped-bias fixture (PR2 self-review finding 3): the SAME bias values
    # stored grouped {numGroups=3, groupSize=1} -- each output column its own
    # exponent. The values are exact under their per-value grids too, so the
    # expected output is BIT-IDENTICAL to the per-tensor-bias gold (asserted);
    # only WHICH exponent each column's seed dequantizes through changes. A
    # kernel that reads every bias seed through group 0 (the bfpGroupOf-drop
    # mutant at Matmul.c's bias seed) dequantizes columns whose exponent
    # differs from group 0's off by a power of two -- asserted non-vacuous.
    bias_g_qc = {"mantissa_bits": 8, "exponent_bits": 8, "group_size": 1}
    bias_g_codes, bias_g_exps = bfp_quantize_grouped(BIAS_VALUES, bias_g_qc["mantissa_bits"],
                                                     bias_g_qc["exponent_bits"],
                                                     bias_g_qc["group_size"])
    assert len(bias_g_exps) == len(BIAS_VALUES)
    check_exact_roundtrip("bias_grouped", BIAS_VALUES, bias_g_codes, bias_g_exps, bias_g_qc)
    assert len(set(bias_g_exps)) >= 2, (
        "grouped bias: exponent array is uniform -- group binding unobservable")
    ebias = 2 ** (bias_g_qc["exponent_bits"] - 1) - 1
    scale0 = np.float32(np.ldexp(np.float32(1.0), np.int32(bias_g_exps[0] - ebias)))
    collapsed = [float(np.float32(np.float32(c) * scale0)) for c in bias_g_codes]
    assert any(cv != v for cv, v in zip(collapsed, BIAS_VALUES)), (
        "grouped bias: group-0 collapse reproduces every value -- the "
        "bfpGroupOf mutant would be unobservable")
    expected_grouped_bias = matmul_bfp_ref(a_codes, a_exps, A_QC, b_codes, b_exps, B_QC,
                                           bias_g_codes, bias_g_exps, bias_g_qc,
                                           OUT_ROWS, OUT_COLS, REDUCE_LEN, b_transposed=True)
    assert expected_grouped_bias == expected, (
        "grouped-bias expectation must be bit-identical to the per-tensor gold "
        "-- both grids are exact for these values")

    # ---- PR3 backward fixtures: loss [3,4] (batch=3, outF=4), W [4,4]
    # ([outF, inF] row-major), x [3,4]. Group sizes per the module docstring:
    # loss gsz 3 (4 groups, misaligned with the outF=4 rows so both the
    # contiguous dx walk and the strided weightGrad walk cross groups),
    # W gsz 8 > inF and x gsz 6 > inF (a-only boundaries exist in both
    # orientations). Every group's values sit exactly on its derived grid.
    LOSS_VALUES = [1.0, -2.0, 5.0, -1.5,
                   10.0, -6.0, 2.5, -0.25,
                   0.75, 16.0, -3.0, 21.0]
    LOSS_QC = {"mantissa_bits": 6, "exponent_bits": 8, "group_size": 3}
    WB_VALUES = [3.0, -1.0, 0.5, 2.5, 1.0, -2.0, 0.5, -0.5,
                 1.5, 1.0, -0.75, -1.25, 0.75, -0.25, 1.0, -1.5]
    WB_QC = {"mantissa_bits": 4, "exponent_bits": 8, "group_size": 8}
    XB_VALUES = [1.0, -2.0, 0.5, 4.0, -0.25, 0.75,
                 1.0, -3.0, 0.625, 2.25, -0.125, 1.75]
    XB_QC = {"mantissa_bits": 6, "exponent_bits": 8, "group_size": 6}
    BWD_BATCH = 3
    BWD_OUT_F = 4
    BWD_IN_F = 4
    loss_codes, loss_exps = bfp_quantize_grouped(LOSS_VALUES, LOSS_QC["mantissa_bits"],
                                                 LOSS_QC["exponent_bits"],
                                                 LOSS_QC["group_size"])
    wb_codes, wb_exps = bfp_quantize_grouped(WB_VALUES, WB_QC["mantissa_bits"],
                                             WB_QC["exponent_bits"], WB_QC["group_size"])
    xb_codes, xb_exps = bfp_quantize_grouped(XB_VALUES, XB_QC["mantissa_bits"],
                                             XB_QC["exponent_bits"], XB_QC["group_size"])
    for nm, vals, cds, exs, qc in (("loss", LOSS_VALUES, loss_codes, loss_exps, LOSS_QC),
                                   ("wb", WB_VALUES, wb_codes, wb_exps, WB_QC),
                                   ("xb", XB_VALUES, xb_codes, xb_exps, XB_QC)):
        check_exact_roundtrip(nm, vals, cds, exs, qc)
        assert len(set(exs)) >= 2, f"{nm}: uniform exponents -- fixture too weak"
    # dx = loss @ W  ([3,4]@[4,4] -> [3,4]); b NOT transposed: b_idx = k*cols + c
    # walks W [outF, inF] storage strided by inF -- the canonical strided-dx walk.
    expected_dx = matmul_bfp_ref(loss_codes, loss_exps, LOSS_QC,
                                 wb_codes, wb_exps, WB_QC, None, None, None,
                                 BWD_BATCH, BWD_IN_F, BWD_OUT_F, b_transposed=False)
    # dW = loss^T @ x ([4,3]@[3,4] -> [4,4]); a is the loss^T VIEW (a_transposed).
    expected_wg = matmul_bfp_ref(loss_codes, loss_exps, LOSS_QC,
                                 xb_codes, xb_exps, XB_QC, None, None, None,
                                 BWD_OUT_F, BWD_IN_F, BWD_BATCH,
                                 b_transposed=False, a_transposed=True)
    # db[f] = sum_n loss[n][f]
    expected_bg = matmul_bfp_bias_grad_ref(loss_codes, loss_exps, LOSS_QC,
                                           BWD_BATCH, BWD_OUT_F)

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
        emit_int32_array("kBfpBiasGroupedCodes", torch.tensor(bias_g_codes)),
        emit_uint8_array("kBfpBiasGroupedExponents", bias_g_exps),
        emit_int32_scalar("kBfpBiasGroupedNumGroups", len(bias_g_exps)),
        emit_float_array("kBfpMatmulExpected", torch.tensor(expected, dtype=torch.float32)),
        emit_float_array("kBfpMatmulNoBiasExpected",
                         torch.tensor(expected_no_bias, dtype=torch.float32)),
        emit_int32_scalar("kBfpBwdBatch", BWD_BATCH),
        emit_int32_scalar("kBfpBwdOutF", BWD_OUT_F),
        emit_int32_scalar("kBfpBwdInF", BWD_IN_F),
        emit_int32_array("kBfpLossCodes", torch.tensor(loss_codes)),
        emit_uint8_array("kBfpLossExponents", loss_exps),
        emit_int32_scalar("kBfpLossNumGroups", len(loss_exps)),
        emit_int32_scalar("kBfpLossGroupSize", LOSS_QC["group_size"]),
        emit_int32_scalar("kBfpLossMantissaBits", LOSS_QC["mantissa_bits"]),
        emit_int32_scalar("kBfpLossExponentBits", LOSS_QC["exponent_bits"]),
        emit_int32_array("kBfpWbCodes", torch.tensor(wb_codes)),
        emit_uint8_array("kBfpWbExponents", wb_exps),
        emit_int32_scalar("kBfpWbNumGroups", len(wb_exps)),
        emit_int32_scalar("kBfpWbGroupSize", WB_QC["group_size"]),
        emit_int32_scalar("kBfpWbMantissaBits", WB_QC["mantissa_bits"]),
        emit_int32_scalar("kBfpWbExponentBits", WB_QC["exponent_bits"]),
        emit_int32_array("kBfpXbCodes", torch.tensor(xb_codes)),
        emit_uint8_array("kBfpXbExponents", xb_exps),
        emit_int32_scalar("kBfpXbNumGroups", len(xb_exps)),
        emit_int32_scalar("kBfpXbGroupSize", XB_QC["group_size"]),
        emit_int32_scalar("kBfpXbMantissaBits", XB_QC["mantissa_bits"]),
        emit_int32_scalar("kBfpXbExponentBits", XB_QC["exponent_bits"]),
        emit_float_array("kBfpDxExpected", torch.tensor(expected_dx, dtype=torch.float32)),
        emit_float_array("kBfpWgExpected", torch.tensor(expected_wg, dtype=torch.float32)),
        emit_float_array("kBfpBgExpected", torch.tensor(expected_bg, dtype=torch.float32)),
        "\n#endif // ODT_EXPECTED_BFP_MATMUL_H\n",
    ]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(parts))
    return 0


if __name__ == "__main__":
    sys.exit(main())
