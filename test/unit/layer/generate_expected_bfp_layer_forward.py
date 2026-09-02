#!/usr/bin/env python3
"""Generate expected_bfp_layer_forward.h for the three GEMM-family LAYER
forward tests (BFP epic PR2, Task 7 -- spec
docs/superpowers/specs/2026-07-29-block-floating-point-design.md). One header,
three fixtures (Linear / Conv1d / ConvT1d), shared by UnitTestLinear,
UnitTestConv1d and UnitTestConv1dTransposed (same binary dir; each target adds
its own dependency on the one generate target -- a separate small generator
per the Task 7 controller authorization, instead of growing the three
kernel-level generators in test/unit/arithmetic/).

What the fixtures pin (the LAYER wiring, not the kernel fold -- Tasks 3-5's
generators own that):
  1. The layer requantizes FLOAT32 weights into a GROUPED BFP config
     (requantizeTensorInPlace); the gold mirrors that with
     sym_gold.bfp_quantize_grouped (the bit-exact quantizeFloatBufferToBfpCodes
     twin, HALF_AWAY).
  2. The funnel stages the FLOAT32-stored input (and Linear's FLOAT32 bias)
     PER-TENSOR at the WEIGHTS' widths (plan Decision 1/2): the gold stages
     with group_size=0 at the weight fixture's mantissa/exponent bits. Input
     values are chosen LOSSY at m=6 so a layer that stages at a hardcoded
     m=8 (the Task 7 rule-2 mutation) produces observably different outputs
     -- asserted below per fixture.
  3. The FLOAT32 output wire is bit-exact: ARITH_BFP's raw intermediate is
     FLOAT32 (D7) and the OUT_WRITE epilogue's FLOAT32->FLOAT32 write is a
     memmove, so the C tests compare via TEST_ASSERT_EQUAL_MEMORY against
     the np.float32-mirrored *_bfp_ref outputs.

The *_bfp_ref self_check=False everywhere: a per-tensor staged input can
never cross >= 2 input groups, so the kernel-fixture vacuity checks cannot
hold here BY CONSTRUCTION. Layer-relevant replacements (abort generation
rather than emit a vacuous fixture):
  - staging the input at m=8 instead of the weights' m=6 changes the
    expected output (rule-2 width-copy mutation observable);
  - collapsing the weight exponents to per-tensor changes the expected
    output (the layer path preserves the weights' group structure);
  - the weight exponent array is non-uniform;
  - Linear: with-bias and no-bias expectations differ elementwise (bias
    staging observable);
  - ConvT1d: >= 1 outputPadding tail position (expected 0.0 without bias)
    AND >= 1 nonzero output (a dropped cfg->outputPadding pass-through in
    the layer adapter is observable against the sentinel-prefilled wire).

PR3 Task 3 extension -- Conv1d BACKWARD fixtures (kConvBfpWg*/kConvBfpBg*/
kConvBfpDx*): unlike the forward fixtures' staged-FLOAT32 inputs, BOTH
backward operands (forwardInput x and lossGrad gy) are BFP-STORED wires, so
the new weightGrad/biasGrad kernels' fold contract is exercised end-to-end
(borrow arm) and conv1d_bfp_weight_grad_ref / conv_bfp_bias_grad_ref run
with their FULL kernel-grade self-checks. The dx gold delegates to the D9
gather ref through conv1d_bfp_dx_ref with self_check=False -- the gather's
inner walk moves the WEIGHT storage index by +outChPerGroup*K within a tap
and jumps negatively at tap transitions, never landing in the same
groupSize-2-aligned weight group, so no step is weight-quiet and a gy-only
boundary event (the ref's x_only pin) can never fire for ANY legal gy
grouping; the gather fold itself is kernel-pinned in PR2's ConvT1d golds,
and layer-relevant replacements are asserted below (per-operand collapses
differ, adjoint-hole zeros, nonzero output).

Run via `uv run` (CMake wires this automatically, see CMakeLists.txt).
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "goldgen"))

from sym_gold import (assert_rounding_canary, bfp_quantize_grouped, check_exact_roundtrip,
                      conv1d_bfp_ref, conv1d_bfp_dx_ref, conv1d_bfp_weight_grad_ref,
                      conv_bfp_bias_grad_ref, convT1d_bfp_gather_ref, convT1d_bfp_dx_ref,
                      convT1d_bfp_weight_grad_ref, emit_float_array, emit_int32_array,
                      emit_int32_scalar, emit_uint8_array, matmul_bfp_ref)

# Weights' widths (Decision 1: the funnel stages FLOAT32 operands at these).
# m=6 deliberately != 8 so the hardcode-m=8 mutation is observable.
W_MANTISSA_BITS = 6
W_EXPONENT_BITS = 8

# ---- Linear: weight [3, 6] grouped {9, 2}; input [2, 6]; bias [3] ----------
LIN_OUT_FEATURES = 3
LIN_IN_FEATURES = 6
LIN_BATCH = 2
LIN_W_NUM_GROUPS = 9
LIN_W_GROUP_SIZE = 2
LIN_W_VALUES = [3.5, -1.5, 1.0, -0.5, -2.0, 1.25,
                0.75, -0.25, 1.5, 2.5, -0.5, 0.25,
                1.5, 1.0, -0.75, -1.25, 2.0, -1.0]
LIN_X_VALUES = [1.3, -2.6, 3.3, -1.55, 10.0, -6.1,
                4.2, 7.9, 0.55, -0.35, 0.85, 1.15]
LIN_BIAS_VALUES = [2.3, -1.55, 0.65]

# ---- Conv1d: weight [2, 2, 3] grouped {6, 2}; input [1, 2, 10];
#      VALID, stride 2 -> outLen 4; no bias (bias staging is Linear-only) ----
CONV_BATCH = 1
CONV_IN_CHANNELS = 2
CONV_OUT_CHANNELS = 2
CONV_KERNEL_SIZE = 3
CONV_INPUT_LENGTH = 10
CONV_STRIDE = 2
CONV_OUT_LEN = 4  # (10 - 3) // 2 + 1
CONV_W_NUM_GROUPS = 6
CONV_W_GROUP_SIZE = 2
CONV_W_VALUES = [3.5, -1.5, 1.0, -0.5, -2.0, 1.25,
                 0.75, -0.25, 1.5, 2.5, -0.5, 0.25]
CONV_X_VALUES = [1.3, -2.6, 3.3, -1.55, 10.0, -6.1, 4.2, 7.9, 0.55, -0.35,
                 0.85, 1.15, 4.1, 2.3, -1.05, 0.65, -3.2, 1.45, 2.55, -0.65]

# ---- Conv1d BACKWARD operand set (PR3 Task 3): same weight tensor as the
#      forward fixture ([2, 2, 3] grouped {6, 2}, requantized by the layer),
#      but a DEDICATED kernel geometry: K=3, VALID, stride 2, DILATION 2
#      (dilation must be load-bearing: the weightGrad mutant that drops
#      `* geom.dilation` from the tap arithmetic is identity at dilation 1,
#      and stride cannot expose it) -> x [1, 2, 9], outLen (9-5)//2+1 = 3,
#      gy [1, 2, 3]. VALID + exact stride inversion ((3-1)*2 + 5 == 9) keeps
#      the SAME operand set valid for the dx adjoint gold. Both operands
#      BFP-stored m=6/e=8: x grouped {6 groups x 3}, gy grouped
#      {3 groups x 2} (the brief's {groupSize 2, outLen groups} shape --
#      each output channel spans one-and-a-half groups, so every per-oc walk
#      crosses a boundary and gy has >= 3 groups, keeping crossing-site
#      exponent bindings observable at MIXED sites, not only group 0).
#      Values are per-group grid-exact (roundtrip-asserted below). ----------
CONV_BWD_INPUT_LENGTH = 9
CONV_BWD_STRIDE = 2
CONV_BWD_DILATION = 2
CONV_BWD_OUT_LEN = 3  # (9 - (2*(3-1)+1)) // 2 + 1
CONV_BWD_X_VALUES = [1.0, -2.0, 3.0, -1.5, 6.0, -4.0, 2.0, 0.5, -0.25,
                     8.0, -2.0, 4.0, -0.5, 1.0, 0.75, 3.0, -1.0, 2.5]
CONV_BWD_X_QC = {"mantissa_bits": 6, "exponent_bits": 8, "group_size": 3}
CONV_BWD_X_NUM_GROUPS = 6
CONV_BWD_GY_VALUES = [0.5, -1.0, 2.0, 1.5, -4.0, 3.0]
CONV_BWD_GY_QC = {"mantissa_bits": 6, "exponent_bits": 8, "group_size": 2}
CONV_BWD_GY_NUM_GROUPS = 3

# ---- ConvT1d: weight [2, 2, 3] ([Cin, Cout, K]) grouped {6, 2};
#      input [1, 2, 5]; stride 2, outputPadding 1 -> outLen 12; no bias -----
CONVT_BATCH = 1
CONVT_IN_CHANNELS = 2
CONVT_OUT_CHANNELS = 2
CONVT_KERNEL_SIZE = 3
CONVT_INPUT_LENGTH = 5
CONVT_STRIDE = 2
CONVT_OUTPUT_PADDING = 1
CONVT_OUT_LEN = 12  # (5-1)*2 + 1*(3-1) + 1 + 1
CONVT_W_NUM_GROUPS = 6
CONVT_W_GROUP_SIZE = 2
CONVT_W_VALUES = [3.5, -1.5, 1.0, -0.5, -2.0, 1.25,
                  0.75, -0.25, 1.5, 2.5, -0.5, 0.25]
CONVT_X_VALUES = [1.3, -2.6, 3.3, -1.55, 10.0,
                  -6.1, 4.2, 7.9, 0.55, -0.35]

# ---- ConvT1d BACKWARD operand set (PR3 Task 4): the forward fixture's
#      weight ([2,2,3] = [Cin, Cout, K] grouped {6,2}, requantized by the
#      layer -- the rule-1 width anchor) under the forward kernel geometry
#      (K=3, VALID, stride 2, outputPadding 1, dilation 1) at a DEDICATED
#      input length 4 -> x [1,2,4], outLen (4-1)*2 + (3-1) + 1 + 1 = 10,
#      gy [1,2,10]. The outputPadding tail (gy position 9 per channel) has
#      NO weightGrad/dx contributors (max out_idx = outLen - outputPadding
#      - 1 = 8) -- it is load-bearing ONLY through biasGrad's full sum,
#      separating the two walks. Both operands BFP-stored m=6/e=8:
#      x grouped {4 groups x 2}, gy grouped {5 groups x 4} (>= 3 groups,
#      PAIRWISE-DISTINCT exponents -- crossing-site exponent bindings stay
#      observable at MIXED sites, not only group 0). Values are per-group
#      grid-exact (roundtrip-asserted below). ------------------------------
CONVT_BWD_INPUT_LENGTH = 4
CONVT_BWD_OUT_LEN = 10  # (4-1)*2 + 1*(3-1) + 1 + 1
CONVT_BWD_X_VALUES = [1.0, -2.5, 6.0, -3.75, 0.5, 0.75, -12.0, 7.5]
CONVT_BWD_X_QC = {"mantissa_bits": 6, "exponent_bits": 8, "group_size": 2}
CONVT_BWD_X_NUM_GROUPS = 4
CONVT_BWD_GY_VALUES = [0.5, -1.0, 2.0, 1.5, -4.0, 3.0, -0.5, 1.0, 8.0, -6.0,
                       0.5, -1.5, 16.0, -12.0, 2.0, -3.0, 0.125, -0.25, 0.0625, 0.1875]
CONVT_BWD_GY_QC = {"mantissa_bits": 6, "exponent_bits": 8, "group_size": 4}
CONVT_BWD_GY_NUM_GROUPS = 5

# dilation-2 weightGrad sub-fixture: the affine out_idx = in_pos*stride +
# k*dilation is the ONLY place dilation enters the new ConvT weightGrad
# kernel, and the main fixture's dilation 1 makes a dropped factor an
# arithmetic identity (the Conv1d backward fixture's dilation lesson) --
# same x operand, gy_dil [1,2,12] for outLen (4-1)*2 + 2*2 + 1 + 1 = 12.
CONVT_BWD_DIL_DILATION = 2
CONVT_BWD_DIL_OUT_LEN = 12
CONVT_BWD_DIL_GY_VALUES = CONVT_BWD_GY_VALUES + [0.5, 1.0, -2.0, 1.75]
CONVT_BWD_DIL_GY_NUM_GROUPS = 6


def quantize_weights(values, num_groups, group_size):
    """Mirror of the C test's requantizeTensorInPlace(FLOAT32 -> grouped BFP,
    HALF_AWAY) at the weights' widths."""
    codes, exps = bfp_quantize_grouped(values, W_MANTISSA_BITS, W_EXPONENT_BITS, group_size)
    assert len(exps) == num_groups
    qc = {"mantissa_bits": W_MANTISSA_BITS, "exponent_bits": W_EXPONENT_BITS,
          "group_size": group_size}
    return codes, exps, qc


def stage_per_tensor(values, mantissa_bits):
    """Mirror of the funnel's FLOAT32-operand staging: per-tensor {1,0} at the
    given mantissa width (the weights' exponent width rides along)."""
    codes, exps = bfp_quantize_grouped(values, mantissa_bits, W_EXPONENT_BITS, 0)
    assert len(exps) == 1
    qc = {"mantissa_bits": mantissa_bits, "exponent_bits": W_EXPONENT_BITS, "group_size": 0}
    return codes, exps, qc


def check_fixture_strength(name, expected, expected_m8, expected_collapsed, w_exps):
    assert any(a != b for a, b in zip(expected, expected_m8)), (
        f"{name}: staging the input at m=8 is indistinguishable from m={W_MANTISSA_BITS} "
        "-- the rule-2 width-copy mutation would be unobservable; pick lossier inputs")
    assert any(a != b for a, b in zip(expected, expected_collapsed)), (
        f"{name}: per-tensor weight-exponent collapse is indistinguishable from the "
        "grouped weights -- group structure is unobservable at the layer level")
    assert len(set(w_exps)) >= 2, (
        f"{name}: weight exponent array is uniform -- fixture too weak")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    assert_rounding_canary()

    # ---- Linear -----------------------------------------------------------
    lw_codes, lw_exps, lw_qc = quantize_weights(LIN_W_VALUES, LIN_W_NUM_GROUPS, LIN_W_GROUP_SIZE)
    lx_codes, lx_exps, lx_qc = stage_per_tensor(LIN_X_VALUES, W_MANTISSA_BITS)
    lb_codes, lb_exps, lb_qc = stage_per_tensor(LIN_BIAS_VALUES, W_MANTISSA_BITS)

    lin_no_bias = matmul_bfp_ref(lx_codes, lx_exps, lx_qc, lw_codes, lw_exps, lw_qc,
                                 None, None, None, LIN_BATCH, LIN_OUT_FEATURES,
                                 LIN_IN_FEATURES, b_transposed=True, self_check=False)
    lin_with_bias = matmul_bfp_ref(lx_codes, lx_exps, lx_qc, lw_codes, lw_exps, lw_qc,
                                   lb_codes, lb_exps, lb_qc, LIN_BATCH, LIN_OUT_FEATURES,
                                   LIN_IN_FEATURES, b_transposed=True, self_check=False)

    lx8_codes, lx8_exps, lx8_qc = stage_per_tensor(LIN_X_VALUES, 8)
    lin_m8 = matmul_bfp_ref(lx8_codes, lx8_exps, lx8_qc, lw_codes, lw_exps, lw_qc,
                            None, None, None, LIN_BATCH, LIN_OUT_FEATURES,
                            LIN_IN_FEATURES, b_transposed=True, self_check=False)
    lin_collapsed = matmul_bfp_ref(lx_codes, lx_exps, lx_qc, lw_codes, [lw_exps[0]],
                                   {**lw_qc, "group_size": 0}, None, None, None,
                                   LIN_BATCH, LIN_OUT_FEATURES, LIN_IN_FEATURES,
                                   b_transposed=True, self_check=False)
    check_fixture_strength("linear", lin_no_bias, lin_m8, lin_collapsed, lw_exps)
    assert all(w != n for w, n in zip(lin_with_bias, lin_no_bias)), (
        "linear: with-bias and no-bias expectations coincide somewhere -- bias "
        "staging would be unobservable there")

    # ---- Conv1d -----------------------------------------------------------
    cw_codes, cw_exps, cw_qc = quantize_weights(CONV_W_VALUES, CONV_W_NUM_GROUPS,
                                                CONV_W_GROUP_SIZE)
    cx_codes, cx_exps, cx_qc = stage_per_tensor(CONV_X_VALUES, W_MANTISSA_BITS)

    conv_expected = conv1d_bfp_ref(cx_codes, cx_exps, cx_qc, cw_codes, cw_exps, cw_qc,
                                   None, None, None, CONV_BATCH, CONV_IN_CHANNELS,
                                   CONV_OUT_CHANNELS, CONV_KERNEL_SIZE, CONV_INPUT_LENGTH,
                                   stride=CONV_STRIDE, self_check=False)
    assert len(conv_expected) == CONV_BATCH * CONV_OUT_CHANNELS * CONV_OUT_LEN

    cx8_codes, cx8_exps, cx8_qc = stage_per_tensor(CONV_X_VALUES, 8)
    conv_m8 = conv1d_bfp_ref(cx8_codes, cx8_exps, cx8_qc, cw_codes, cw_exps, cw_qc,
                             None, None, None, CONV_BATCH, CONV_IN_CHANNELS,
                             CONV_OUT_CHANNELS, CONV_KERNEL_SIZE, CONV_INPUT_LENGTH,
                             stride=CONV_STRIDE, self_check=False)
    conv_collapsed = conv1d_bfp_ref(cx_codes, cx_exps, cx_qc, cw_codes, [cw_exps[0]],
                                    {**cw_qc, "group_size": 0}, None, None, None,
                                    CONV_BATCH, CONV_IN_CHANNELS, CONV_OUT_CHANNELS,
                                    CONV_KERNEL_SIZE, CONV_INPUT_LENGTH,
                                    stride=CONV_STRIDE, self_check=False)
    check_fixture_strength("conv1d", conv_expected, conv_m8, conv_collapsed, cw_exps)

    # ---- Conv1d backward (PR3 Task 3) -------------------------------------
    bx_codes, bx_exps = bfp_quantize_grouped(
        CONV_BWD_X_VALUES, CONV_BWD_X_QC["mantissa_bits"], CONV_BWD_X_QC["exponent_bits"],
        CONV_BWD_X_QC["group_size"])
    bgy_codes, bgy_exps = bfp_quantize_grouped(
        CONV_BWD_GY_VALUES, CONV_BWD_GY_QC["mantissa_bits"], CONV_BWD_GY_QC["exponent_bits"],
        CONV_BWD_GY_QC["group_size"])
    assert len(bx_exps) == CONV_BWD_X_NUM_GROUPS
    assert len(bgy_exps) == CONV_BWD_GY_NUM_GROUPS
    check_exact_roundtrip("conv1d bwd x", CONV_BWD_X_VALUES, bx_codes, bx_exps, CONV_BWD_X_QC)
    check_exact_roundtrip("conv1d bwd gy", CONV_BWD_GY_VALUES, bgy_codes, bgy_exps,
                          CONV_BWD_GY_QC)
    assert len(set(bx_exps)) >= 2, "conv1d bwd x: exponent array is uniform -- fixture too weak"
    # gy needs PAIRWISE distinct group exponents: the biasGrad fold-binding
    # mutant (exponents[g] at the crossing site instead of currentGroup) folds
    # segment(g0) with E(g1) on oc=0's walk and segment(g1) with E(g2) on
    # oc=1's -- both must be power-of-two-off, and both mutated segments must
    # carry a nonzero partial.
    assert len(set(bgy_exps)) == CONV_BWD_GY_NUM_GROUPS, (
        "conv1d bwd gy: group exponents not pairwise distinct -- crossing-site "
        "fold-binding mutants would be unobservable at some site")
    assert bgy_codes[0] + bgy_codes[1] != 0 and bgy_codes[3] != 0, (
        "conv1d bwd gy: a crossing-fold segment partial is zero -- the "
        "fold-binding mutant would be unobservable there")

    conv_wg = conv1d_bfp_weight_grad_ref(
        bx_codes, bx_exps, CONV_BWD_X_QC, bgy_codes, bgy_exps, CONV_BWD_GY_QC,
        CONV_BATCH, CONV_IN_CHANNELS, CONV_OUT_CHANNELS, CONV_KERNEL_SIZE,
        CONV_BWD_INPUT_LENGTH, stride=CONV_BWD_STRIDE, dilation=CONV_BWD_DILATION)
    assert len(conv_wg) == CONV_OUT_CHANNELS * CONV_IN_CHANNELS * CONV_KERNEL_SIZE

    conv_bg = conv_bfp_bias_grad_ref(bgy_codes, bgy_exps, CONV_BWD_GY_QC,
                                     CONV_BATCH, CONV_OUT_CHANNELS, CONV_BWD_OUT_LEN)
    assert len(conv_bg) == CONV_OUT_CHANNELS

    # dx: gather-ref built-ins are structurally unsatisfiable here (module
    # docstring), so self_check=False + the layer-relevant replacements:
    conv_dx = conv1d_bfp_dx_ref(
        bgy_codes, bgy_exps, CONV_BWD_GY_QC, cw_codes, cw_exps, cw_qc,
        CONV_BATCH, CONV_IN_CHANNELS, CONV_OUT_CHANNELS, CONV_KERNEL_SIZE,
        CONV_BWD_INPUT_LENGTH, stride=CONV_BWD_STRIDE, dilation=CONV_BWD_DILATION,
        self_check=False)
    # per-operand collapses must both change the result (group structure
    # observable from EACH operand through the layer dx path);
    conv_dx_gy_collapsed = conv1d_bfp_dx_ref(
        bgy_codes, [bgy_exps[0]], {**CONV_BWD_GY_QC, "group_size": 0},
        cw_codes, cw_exps, cw_qc,
        CONV_BATCH, CONV_IN_CHANNELS, CONV_OUT_CHANNELS, CONV_KERNEL_SIZE,
        CONV_BWD_INPUT_LENGTH, stride=CONV_BWD_STRIDE, dilation=CONV_BWD_DILATION,
        self_check=False)
    conv_dx_w_collapsed = conv1d_bfp_dx_ref(
        bgy_codes, bgy_exps, CONV_BWD_GY_QC, cw_codes, [cw_exps[0]],
        {**cw_qc, "group_size": 0},
        CONV_BATCH, CONV_IN_CHANNELS, CONV_OUT_CHANNELS, CONV_KERNEL_SIZE,
        CONV_BWD_INPUT_LENGTH, stride=CONV_BWD_STRIDE, dilation=CONV_BWD_DILATION,
        self_check=False)
    assert conv_dx_gy_collapsed != conv_dx, (
        "conv1d bwd dx: loss-exponent collapse is indistinguishable -- loss "
        "group structure unobservable through the dx path")
    assert conv_dx_w_collapsed != conv_dx, (
        "conv1d bwd dx: weight-exponent collapse is indistinguishable -- "
        "weight group structure unobservable through the dx path")
    # adjoint-hole pin: stride 2 x dilation 2 reach only EVEN input positions
    # (in_idx = 2*outPos + 2*k), so every odd dx position is tap-free and must
    # be EXACTLY 0.0 -- a layer adapter that drops kernel geometry cannot
    # reproduce the holes against the test's sentinel-prefilled wire.
    for c in range(CONV_IN_CHANNELS):
        for pos in range(1, CONV_BWD_INPUT_LENGTH, 2):
            assert conv_dx[c * CONV_BWD_INPUT_LENGTH + pos] == 0.0, (
                "conv1d bwd dx: adjoint-hole position is nonzero -- fixture "
                "broke the tap-free-hole assumption")
    assert any(v != 0.0 for v in conv_dx), "conv1d bwd dx: all-zero expected output"

    # ---- ConvT1d ----------------------------------------------------------
    tw_codes, tw_exps, tw_qc = quantize_weights(CONVT_W_VALUES, CONVT_W_NUM_GROUPS,
                                                CONVT_W_GROUP_SIZE)
    tx_codes, tx_exps, tx_qc = stage_per_tensor(CONVT_X_VALUES, W_MANTISSA_BITS)

    convt_expected = convT1d_bfp_gather_ref(tx_codes, tx_exps, tx_qc, tw_codes, tw_exps, tw_qc,
                                            None, None, None, CONVT_BATCH, CONVT_IN_CHANNELS,
                                            CONVT_OUT_CHANNELS, CONVT_KERNEL_SIZE,
                                            CONVT_INPUT_LENGTH, stride=CONVT_STRIDE,
                                            output_padding=CONVT_OUTPUT_PADDING,
                                            self_check=False)
    assert len(convt_expected) == CONVT_BATCH * CONVT_OUT_CHANNELS * CONVT_OUT_LEN

    tx8_codes, tx8_exps, tx8_qc = stage_per_tensor(CONVT_X_VALUES, 8)
    convt_m8 = convT1d_bfp_gather_ref(tx8_codes, tx8_exps, tx8_qc, tw_codes, tw_exps, tw_qc,
                                      None, None, None, CONVT_BATCH, CONVT_IN_CHANNELS,
                                      CONVT_OUT_CHANNELS, CONVT_KERNEL_SIZE,
                                      CONVT_INPUT_LENGTH, stride=CONVT_STRIDE,
                                      output_padding=CONVT_OUTPUT_PADDING, self_check=False)
    convt_collapsed = convT1d_bfp_gather_ref(tx_codes, tx_exps, tx_qc, tw_codes, [tw_exps[0]],
                                             {**tw_qc, "group_size": 0}, None, None, None,
                                             CONVT_BATCH, CONVT_IN_CHANNELS,
                                             CONVT_OUT_CHANNELS, CONVT_KERNEL_SIZE,
                                             CONVT_INPUT_LENGTH, stride=CONVT_STRIDE,
                                             output_padding=CONVT_OUTPUT_PADDING,
                                             self_check=False)
    check_fixture_strength("convT1d", convt_expected, convt_m8, convt_collapsed, tw_exps)
    # outputPadding tail (per output channel the LAST position has zero taps
    # and no bias): expected exactly 0.0 there, nonzero elsewhere -- a layer
    # adapter that drops cfg->outputPadding cannot reproduce the tail against
    # the sentinel-prefilled wire.
    for oc in range(CONVT_OUT_CHANNELS):
        assert convt_expected[(oc + 1) * CONVT_OUT_LEN - 1] == 0.0, (
            "convT1d: outputPadding tail position is not 0.0 -- fixture broke "
            "the tap-free-tail assumption")
    assert any(v != 0.0 for v in convt_expected), "convT1d: all-zero expected output"

    # ---- ConvT1d backward (PR3 Task 4) ------------------------------------
    tbx_codes, tbx_exps = bfp_quantize_grouped(
        CONVT_BWD_X_VALUES, CONVT_BWD_X_QC["mantissa_bits"], CONVT_BWD_X_QC["exponent_bits"],
        CONVT_BWD_X_QC["group_size"])
    tbgy_codes, tbgy_exps = bfp_quantize_grouped(
        CONVT_BWD_GY_VALUES, CONVT_BWD_GY_QC["mantissa_bits"], CONVT_BWD_GY_QC["exponent_bits"],
        CONVT_BWD_GY_QC["group_size"])
    assert len(tbx_exps) == CONVT_BWD_X_NUM_GROUPS
    assert len(tbgy_exps) == CONVT_BWD_GY_NUM_GROUPS
    check_exact_roundtrip("convT1d bwd x", CONVT_BWD_X_VALUES, tbx_codes, tbx_exps,
                          CONVT_BWD_X_QC)
    check_exact_roundtrip("convT1d bwd gy", CONVT_BWD_GY_VALUES, tbgy_codes, tbgy_exps,
                          CONVT_BWD_GY_QC)
    assert len(set(tbx_exps)) >= 2, "convT1d bwd x: exponent array is uniform -- fixture too weak"
    # gy needs PAIRWISE distinct group exponents plus nonzero crossing-fold
    # segment partials: the biasGrad fold-binding mutant (exponents[g] at the
    # crossing site instead of currentGroup) folds segment(g0) with E(g1) and
    # segment(g1) with E(g2) on oc=0's walk (crossings at gy_idx 4 and 8) and
    # segment(g2) with E(g3), segment(g3) with E(g4) on oc=1's (crossings at
    # 12 and 16) -- every mutated fold must be power-of-two-off on a nonzero
    # partial.
    assert len(set(tbgy_exps)) == CONVT_BWD_GY_NUM_GROUPS, (
        "convT1d bwd gy: group exponents not pairwise distinct -- crossing-site "
        "fold-binding mutants would be unobservable at some site")
    for lo, hi in ((0, 4), (4, 8), (10, 12), (12, 16)):
        assert sum(tbgy_codes[lo:hi]) != 0, (
            "convT1d bwd gy: a crossing-fold segment partial is zero -- the "
            "fold-binding mutant would be unobservable there")

    convt_wg = convT1d_bfp_weight_grad_ref(
        tbx_codes, tbx_exps, CONVT_BWD_X_QC, tbgy_codes, tbgy_exps, CONVT_BWD_GY_QC,
        CONVT_BATCH, CONVT_IN_CHANNELS, CONVT_OUT_CHANNELS, CONVT_KERNEL_SIZE,
        CONVT_BWD_INPUT_LENGTH, stride=CONVT_STRIDE, output_padding=CONVT_OUTPUT_PADDING)
    assert len(convt_wg) == CONVT_IN_CHANNELS * CONVT_OUT_CHANNELS * CONVT_KERNEL_SIZE

    convt_bg = conv_bfp_bias_grad_ref(tbgy_codes, tbgy_exps, CONVT_BWD_GY_QC,
                                      CONVT_BATCH, CONVT_OUT_CHANNELS, CONVT_BWD_OUT_LEN)
    assert len(convt_bg) == CONVT_OUT_CHANNELS

    # dilation-2 weightGrad sub-fixture (see the operand block's comment).
    tbgyd_codes, tbgyd_exps = bfp_quantize_grouped(
        CONVT_BWD_DIL_GY_VALUES, CONVT_BWD_GY_QC["mantissa_bits"],
        CONVT_BWD_GY_QC["exponent_bits"], CONVT_BWD_GY_QC["group_size"])
    assert len(tbgyd_exps) == CONVT_BWD_DIL_GY_NUM_GROUPS
    check_exact_roundtrip("convT1d bwd dil gy", CONVT_BWD_DIL_GY_VALUES, tbgyd_codes,
                          tbgyd_exps, CONVT_BWD_GY_QC)
    assert len(set(tbgyd_exps)) >= 2, (
        "convT1d bwd dil gy: exponent array is uniform -- fixture too weak")
    convt_wg_dil = convT1d_bfp_weight_grad_ref(
        tbx_codes, tbx_exps, CONVT_BWD_X_QC, tbgyd_codes, tbgyd_exps, CONVT_BWD_GY_QC,
        CONVT_BATCH, CONVT_IN_CHANNELS, CONVT_OUT_CHANNELS, CONVT_KERNEL_SIZE,
        CONVT_BWD_INPUT_LENGTH, stride=CONVT_STRIDE, dilation=CONVT_BWD_DIL_DILATION,
        output_padding=CONVT_OUTPUT_PADDING)
    assert len(convt_wg_dil) == CONVT_IN_CHANNELS * CONVT_OUT_CHANNELS * CONVT_KERNEL_SIZE
    assert len(set(convt_wg_dil)) >= 2, "convT1d bwd dil: grads degenerate"

    # dx: the delegate's clipped-window pin cannot hold under the VALID-only
    # adjoint (convT1d_bfp_dx_ref's docstring), so self_check=False + the
    # layer-relevant replacements:
    convt_dx = convT1d_bfp_dx_ref(
        tbgy_codes, tbgy_exps, CONVT_BWD_GY_QC, tw_codes, tw_exps, tw_qc,
        CONVT_BATCH, CONVT_IN_CHANNELS, CONVT_OUT_CHANNELS, CONVT_KERNEL_SIZE,
        CONVT_BWD_INPUT_LENGTH, stride=CONVT_STRIDE,
        output_padding=CONVT_OUTPUT_PADDING, self_check=False)
    convt_dx_gy_collapsed = convT1d_bfp_dx_ref(
        tbgy_codes, [tbgy_exps[0]], {**CONVT_BWD_GY_QC, "group_size": 0},
        tw_codes, tw_exps, tw_qc,
        CONVT_BATCH, CONVT_IN_CHANNELS, CONVT_OUT_CHANNELS, CONVT_KERNEL_SIZE,
        CONVT_BWD_INPUT_LENGTH, stride=CONVT_STRIDE,
        output_padding=CONVT_OUTPUT_PADDING, self_check=False)
    convt_dx_w_collapsed = convT1d_bfp_dx_ref(
        tbgy_codes, tbgy_exps, CONVT_BWD_GY_QC, tw_codes, [tw_exps[0]],
        {**tw_qc, "group_size": 0},
        CONVT_BATCH, CONVT_IN_CHANNELS, CONVT_OUT_CHANNELS, CONVT_KERNEL_SIZE,
        CONVT_BWD_INPUT_LENGTH, stride=CONVT_STRIDE,
        output_padding=CONVT_OUTPUT_PADDING, self_check=False)
    assert convt_dx_gy_collapsed != convt_dx, (
        "convT1d bwd dx: loss-exponent collapse is indistinguishable -- loss "
        "group structure unobservable through the dx path")
    assert convt_dx_w_collapsed != convt_dx, (
        "convT1d bwd dx: weight-exponent collapse is indistinguishable -- "
        "weight group structure unobservable through the dx path")
    assert any(v != 0.0 for v in convt_dx), "convT1d bwd dx: all-zero expected output"

    parts = [
        "// AUTOGENERATED by generate_expected_bfp_layer_forward.py — DO NOT EDIT\n",
        "#ifndef ODT_EXPECTED_BFP_LAYER_FORWARD_H\n",
        "#define ODT_EXPECTED_BFP_LAYER_FORWARD_H\n",
        "#include <stdint.h>\n",
        "#include <stdlib.h>\n\n",
        emit_int32_scalar("kLinBfpOutFeatures", LIN_OUT_FEATURES),
        emit_int32_scalar("kLinBfpInFeatures", LIN_IN_FEATURES),
        emit_int32_scalar("kLinBfpBatch", LIN_BATCH),
        emit_int32_scalar("kLinBfpWNumGroups", LIN_W_NUM_GROUPS),
        emit_int32_scalar("kLinBfpWGroupSize", LIN_W_GROUP_SIZE),
        emit_int32_scalar("kLinBfpWMantissaBits", W_MANTISSA_BITS),
        emit_int32_scalar("kLinBfpWExponentBits", W_EXPONENT_BITS),
        emit_float_array("kLinBfpWValues", torch.tensor(LIN_W_VALUES, dtype=torch.float32)),
        emit_float_array("kLinBfpXValues", torch.tensor(LIN_X_VALUES, dtype=torch.float32)),
        emit_float_array("kLinBfpBiasValues",
                         torch.tensor(LIN_BIAS_VALUES, dtype=torch.float32)),
        emit_float_array("kLinBfpExpectedNoBias",
                         torch.tensor(lin_no_bias, dtype=torch.float32)),
        emit_float_array("kLinBfpExpectedWithBias",
                         torch.tensor(lin_with_bias, dtype=torch.float32)),
        emit_int32_scalar("kConvBfpBatch", CONV_BATCH),
        emit_int32_scalar("kConvBfpInChannels", CONV_IN_CHANNELS),
        emit_int32_scalar("kConvBfpOutChannels", CONV_OUT_CHANNELS),
        emit_int32_scalar("kConvBfpKernelSize", CONV_KERNEL_SIZE),
        emit_int32_scalar("kConvBfpInputLength", CONV_INPUT_LENGTH),
        emit_int32_scalar("kConvBfpStride", CONV_STRIDE),
        emit_int32_scalar("kConvBfpOutLen", CONV_OUT_LEN),
        emit_int32_scalar("kConvBfpWNumGroups", CONV_W_NUM_GROUPS),
        emit_int32_scalar("kConvBfpWGroupSize", CONV_W_GROUP_SIZE),
        emit_int32_scalar("kConvBfpWMantissaBits", W_MANTISSA_BITS),
        emit_int32_scalar("kConvBfpWExponentBits", W_EXPONENT_BITS),
        emit_float_array("kConvBfpWValues", torch.tensor(CONV_W_VALUES, dtype=torch.float32)),
        emit_float_array("kConvBfpXValues", torch.tensor(CONV_X_VALUES, dtype=torch.float32)),
        emit_float_array("kConvBfpExpected",
                         torch.tensor(conv_expected, dtype=torch.float32)),
        emit_int32_scalar("kConvBfpBwdInputLength", CONV_BWD_INPUT_LENGTH),
        emit_int32_scalar("kConvBfpBwdStride", CONV_BWD_STRIDE),
        emit_int32_scalar("kConvBfpBwdDilation", CONV_BWD_DILATION),
        emit_int32_scalar("kConvBfpBwdOutLen", CONV_BWD_OUT_LEN),
        emit_int32_scalar("kConvBfpBwdMantissaBits", CONV_BWD_X_QC["mantissa_bits"]),
        emit_int32_scalar("kConvBfpBwdExponentBits", CONV_BWD_X_QC["exponent_bits"]),
        emit_int32_array("kConvBfpBwdXCodes", torch.tensor(bx_codes)),
        emit_uint8_array("kConvBfpBwdXExponents", bx_exps),
        emit_int32_scalar("kConvBfpBwdXNumGroups", CONV_BWD_X_NUM_GROUPS),
        emit_int32_scalar("kConvBfpBwdXGroupSize", CONV_BWD_X_QC["group_size"]),
        emit_int32_array("kConvBfpBwdGyCodes", torch.tensor(bgy_codes)),
        emit_uint8_array("kConvBfpBwdGyExponents", bgy_exps),
        emit_int32_scalar("kConvBfpBwdGyNumGroups", CONV_BWD_GY_NUM_GROUPS),
        emit_int32_scalar("kConvBfpBwdGyGroupSize", CONV_BWD_GY_QC["group_size"]),
        emit_float_array("kConvBfpWgExpected", torch.tensor(conv_wg, dtype=torch.float32)),
        emit_float_array("kConvBfpBgExpected", torch.tensor(conv_bg, dtype=torch.float32)),
        emit_float_array("kConvBfpDxExpected", torch.tensor(conv_dx, dtype=torch.float32)),
        emit_int32_scalar("kConvTBfpBatch", CONVT_BATCH),
        emit_int32_scalar("kConvTBfpInChannels", CONVT_IN_CHANNELS),
        emit_int32_scalar("kConvTBfpOutChannels", CONVT_OUT_CHANNELS),
        emit_int32_scalar("kConvTBfpKernelSize", CONVT_KERNEL_SIZE),
        emit_int32_scalar("kConvTBfpInputLength", CONVT_INPUT_LENGTH),
        emit_int32_scalar("kConvTBfpStride", CONVT_STRIDE),
        emit_int32_scalar("kConvTBfpOutputPadding", CONVT_OUTPUT_PADDING),
        emit_int32_scalar("kConvTBfpOutLen", CONVT_OUT_LEN),
        emit_int32_scalar("kConvTBfpWNumGroups", CONVT_W_NUM_GROUPS),
        emit_int32_scalar("kConvTBfpWGroupSize", CONVT_W_GROUP_SIZE),
        emit_int32_scalar("kConvTBfpWMantissaBits", W_MANTISSA_BITS),
        emit_int32_scalar("kConvTBfpWExponentBits", W_EXPONENT_BITS),
        emit_float_array("kConvTBfpWValues", torch.tensor(CONVT_W_VALUES, dtype=torch.float32)),
        emit_float_array("kConvTBfpXValues", torch.tensor(CONVT_X_VALUES, dtype=torch.float32)),
        emit_float_array("kConvTBfpExpected",
                         torch.tensor(convt_expected, dtype=torch.float32)),
        emit_int32_scalar("kConvTBfpBwdInputLength", CONVT_BWD_INPUT_LENGTH),
        emit_int32_scalar("kConvTBfpBwdOutLen", CONVT_BWD_OUT_LEN),
        emit_int32_scalar("kConvTBfpBwdMantissaBits", CONVT_BWD_X_QC["mantissa_bits"]),
        emit_int32_scalar("kConvTBfpBwdExponentBits", CONVT_BWD_X_QC["exponent_bits"]),
        emit_int32_array("kConvTBfpBwdXCodes", torch.tensor(tbx_codes)),
        emit_uint8_array("kConvTBfpBwdXExponents", tbx_exps),
        emit_int32_scalar("kConvTBfpBwdXNumGroups", CONVT_BWD_X_NUM_GROUPS),
        emit_int32_scalar("kConvTBfpBwdXGroupSize", CONVT_BWD_X_QC["group_size"]),
        emit_int32_array("kConvTBfpBwdGyCodes", torch.tensor(tbgy_codes)),
        emit_uint8_array("kConvTBfpBwdGyExponents", tbgy_exps),
        emit_int32_scalar("kConvTBfpBwdGyNumGroups", CONVT_BWD_GY_NUM_GROUPS),
        emit_int32_scalar("kConvTBfpBwdGyGroupSize", CONVT_BWD_GY_QC["group_size"]),
        emit_int32_scalar("kConvTBfpBwdDilDilation", CONVT_BWD_DIL_DILATION),
        emit_int32_scalar("kConvTBfpBwdDilOutLen", CONVT_BWD_DIL_OUT_LEN),
        emit_int32_array("kConvTBfpBwdDilGyCodes", torch.tensor(tbgyd_codes)),
        emit_uint8_array("kConvTBfpBwdDilGyExponents", tbgyd_exps),
        emit_int32_scalar("kConvTBfpBwdDilGyNumGroups", CONVT_BWD_DIL_GY_NUM_GROUPS),
        emit_float_array("kConvTBfpWgExpected", torch.tensor(convt_wg, dtype=torch.float32)),
        emit_float_array("kConvTBfpWgDilExpected",
                         torch.tensor(convt_wg_dil, dtype=torch.float32)),
        emit_float_array("kConvTBfpBgExpected", torch.tensor(convt_bg, dtype=torch.float32)),
        emit_float_array("kConvTBfpDxExpected", torch.tensor(convt_dx, dtype=torch.float32)),
        "\n#endif // ODT_EXPECTED_BFP_LAYER_FORWARD_H\n",
    ]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(parts))
    return 0


if __name__ == "__main__":
    sys.exit(main())
