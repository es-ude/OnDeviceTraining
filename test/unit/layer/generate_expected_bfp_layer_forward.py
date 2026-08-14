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

Run via `uv run` (CMake wires this automatically, see CMakeLists.txt).
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "goldgen"))

from sym_gold import (assert_rounding_canary, bfp_quantize_grouped, conv1d_bfp_ref,
                      convT1d_bfp_gather_ref, emit_float_array, emit_int32_scalar,
                      matmul_bfp_ref)

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
        "\n#endif // ODT_EXPECTED_BFP_LAYER_FORWARD_H\n",
    ]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(parts))
    return 0


if __name__ == "__main__":
    sys.exit(main())
