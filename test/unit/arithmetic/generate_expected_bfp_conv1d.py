#!/usr/bin/env python3
"""Generate expected_bfp_conv1d.h for UnitTestConv1dKernel (BFP epic PR2,
Task 4 -- docs/conventions/arithmetic-bfp.md §5).

Pins conv1dKernelBfp's fold order (Conv1dKernel.c): per (b, oc, outPos) ONE
int32 partial; per visited tap BOTH operands' storage indices -> group ids
(bfpGroupOf -- per-element division, gap-robust across the index gaps that
clipped windows create); when EITHER id changes the partial folds into a
float32 accumulator via ldexpf((float)partial, Ein + Ew - biasIn - biasW)
and resets; tail fold after the walk; bias is a value-seed dequantized to
float BEFORE the reduction ((float)mantissa * bfpGroupScale). The kernel
never rounds -- see sym_gold.conv1d_bfp_ref for the exact np.float32-mirrored
emulation.

Fixture geometry (input [1,2,10], weight [2,2,3], stride 2, EXPLICIT
padding 1 -> output [1,2,5]): input grouped numGroups=5/groupSize=4 (m=6,
e=8); weight grouped numGroups=6/groupSize=2 (m=4, e=8) -- deliberately NOT
the plan's original {numGroups=4, groupSize=3}: with groupSize == kernelSize
every weight-group boundary sits at an icOffset transition, where the input's
storage index jumps by inputLength and (for these shapes) ALWAYS changes its
group too, so no boundary would be weight-only and a fold that only watches
the input's group id would be untestable. With groupSize=2 the weight crosses
groups MID-run (e.g. oc=0, outPos=1: tap (in 2, w 1)->(in 3, w 2) is weight-only, the
following icOffset transition (in 3, w 2)->(in 11, w 3) is input-only).
outPos=0's window is left-clipped (taps k in {1,2} only), exercising the
gap-robust per-element lookup; bias per-tensor (m=8, e=8).
Input values are SMALL and grid-exact (every code * scale reproduces the
input float bit-for-bit -- asserted below), so every fold is exact float32
arithmetic and the expected outputs are bit-pinned via
TEST_ASSERT_EQUAL_MEMORY, not a tolerance.

Self-checks (abort generation rather than emit a vacuous fixture):
  - conv1d_bfp_ref's built-in (i)-(iv) + disjoint-boundary pins: >= 2 groups
    crossed on EACH operand within a single reduction; >= 1 fold with a
    nonzero exactly-float-convertible partial; the grouped result differs
    from an all-per-tensor (exponents[0]) collapse; >= 1 output element with
    a CLIPPED tap window; >= 1 input-only AND >= 1 weight-only boundary
    event (either-operand fold clause observable from both sides).
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

from sym_gold import (assert_rounding_canary, bfp_quantize_grouped, check_exact_roundtrip,
                      conv1d_bfp_dx_ref, conv1d_bfp_raw_ref, conv1d_bfp_ref, emit_double_array,
                      emit_float_array, emit_int16_array, emit_int32_array, emit_int32_scalar,
                      emit_uint8_array)

BATCH = 1
IN_CHANNELS = 2
INPUT_LENGTH = 10
OUT_CHANNELS = 2
KERNEL_SIZE = 3
STRIDE = 2
PADDING = 1  # EXPLICIT
OUT_LEN = 5  # (10 + 2*1 - 3) // 2 + 1

# input: [1, 2, 10] row-major flat, quantization groups of 4 -> ic0 spans
# groups {0,1,2}, ic1 spans {2,3,4}; a stride-2 window's icOffset transition
# always changes the input group, mid-run crossings happen at storage
# multiples of 4 (e.g. pos 3->4 in ic0).
X_VALUES = [1.0, -2.0, 3.0, -1.5, 10.0, -6.0, 4.0, 8.0, 0.5, -0.25,
            0.75, 1.0, 4.0, 2.0, -1.0, 0.5, -3.0, 1.5, 2.5, -0.5]
X_QC = {"mantissa_bits": 6, "exponent_bits": 8, "group_size": 4}
X_NUM_GROUPS = 5

# weight: [2, 2, 3] row-major flat, quantization groups of 2 (see module
# docstring for why NOT groupSize 3) -> per output channel three groups.
W_VALUES = [3.5, -1.5, 1.0, -0.5, -2.0, 1.0,
            0.75, -0.25, 1.5, 2.5, -0.5, 0.25]
W_QC = {"mantissa_bits": 4, "exponent_bits": 8, "group_size": 2}
W_NUM_GROUPS = 6

BIAS_VALUES = [2.0, -1.5]
BIAS_QC = {"mantissa_bits": 8, "exponent_bits": 8, "group_size": 0}

# ---- conv-groups=2 fixture (#416, BFP PR3 Task 3): split channels, i.e.
# inChPerGroup = outChPerGroup = 2 (NOT depthwise) so the group loop's
# in_lo/w_base channel arithmetic is load-bearing. Input [1, 4, 10], weight
# [4, 2, 3], stride 2, EXPLICIT padding 1 -> output [1, 4, 5]; same vacuity
# self-checks as the main fixture, now under split channels (a reduction
# spans ic_offset {0,1} x <= 3 taps: the +10 input hop at the ic transition
# and the +-1 mid-run steps put x boundaries (mod 4) and w boundaries
# (mod 2) at DISJOINT phases). Values derive from the main fixture's
# grid-exact sets: x = X_VALUES ++ X_VALUES*0.5, w = W_VALUES ++ W_VALUES*2
# (power-of-two scalings keep every group grid-exact and force NON-UNIFORM
# exponent arrays across the two halves). The float w values are ALSO
# emitted: the layer-level dx test rebuilds the weight the user way
# (FLOAT32 init + requantizeTensorInPlace), whose bfp_quantize_grouped
# mirror reproduces exactly these codes (roundtrip-asserted).
G2_CONV_GROUPS = 2
G2_IN_CHANNELS = 4
G2_OUT_CHANNELS = 4
G2_INPUT_LENGTH = 10
G2_OUT_LEN = 5  # (10 + 2*1 - 3) // 2 + 1
G2_X_VALUES = X_VALUES + [v * 0.5 for v in X_VALUES]
G2_X_QC = {"mantissa_bits": 6, "exponent_bits": 8, "group_size": 4}
G2_X_NUM_GROUPS = 10
G2_W_VALUES = W_VALUES + [v * 2.0 for v in W_VALUES]
G2_W_QC = {"mantissa_bits": 4, "exponent_bits": 8, "group_size": 2}
G2_W_NUM_GROUPS = 12
G2_BIAS_VALUES = [2.0, -1.5, 1.0, 0.5]
G2_BIAS_QC = {"mantissa_bits": 8, "exponent_bits": 8, "group_size": 0}

# dx sub-fixture (layer-level testConv1dBackwardBfpDxConvGroups2MatchesGold):
# SAME weight operand through the adjoint gather, but a VALID unit-stride
# geometry (the dx delegation asserts exact stride inversion, which the
# EXPLICIT-padded forward geometry above cannot satisfy): dx [1, 4, 6],
# loss [1, 4, 4], per-channel loss groups {4 x 4} with pairwise-distinct
# exponents. The gather ref's disjoint-boundary built-ins are structurally
# unsatisfiable in the dx role: its inner walk strides the weight index by
# whole channel-rows (+outChPerGroup*K per step), never staying inside one
# groupSize-2 weight group, so no step is weight-quiet (see
# generate_expected_bfp_layer_forward.py's docstring) -- the delegation runs
# self_check=False with per-operand collapse asserts below instead.
G2_DX_INPUT_LENGTH = 6
G2_DX_OUT_LEN = 4  # (6 - 3) // 1 + 1
G2_DX_LOSS_VALUES = [0.5, -1.0, 0.75, 0.25,
                     2.0, -1.5, 1.0, -0.5,
                     -4.0, 3.0, -2.0, 1.5,
                     8.0, -6.0, 4.0, -2.0]
G2_DX_LOSS_QC = {"mantissa_bits": 6, "exponent_bits": 8, "group_size": 4}
G2_DX_LOSS_NUM_GROUPS = 4


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    assert_rounding_canary()

    x_codes, x_exps = bfp_quantize_grouped(X_VALUES, X_QC["mantissa_bits"],
                                           X_QC["exponent_bits"], X_QC["group_size"])
    w_codes, w_exps = bfp_quantize_grouped(W_VALUES, W_QC["mantissa_bits"],
                                           W_QC["exponent_bits"], W_QC["group_size"])
    bias_codes, bias_exps = bfp_quantize_grouped(BIAS_VALUES, BIAS_QC["mantissa_bits"],
                                                 BIAS_QC["exponent_bits"],
                                                 BIAS_QC["group_size"])
    assert len(x_exps) == X_NUM_GROUPS and len(w_exps) == W_NUM_GROUPS
    assert len(bias_exps) == 1

    check_exact_roundtrip("input", X_VALUES, x_codes, x_exps, X_QC)
    check_exact_roundtrip("weight", W_VALUES, w_codes, w_exps, W_QC)
    check_exact_roundtrip("bias", BIAS_VALUES, bias_codes, bias_exps, BIAS_QC)

    # Non-uniform exponents per operand: uniform arrays would make every fold
    # shift identical, hiding a curGx/curGw mix-up behind equal scales.
    assert len(set(x_exps)) >= 2, "input: exponent array is uniform -- fixture too weak"
    assert len(set(w_exps)) >= 2, "weight: exponent array is uniform -- fixture too weak"

    expected = conv1d_bfp_ref(x_codes, x_exps, X_QC, w_codes, w_exps, W_QC,
                              bias_codes, bias_exps, BIAS_QC,
                              BATCH, IN_CHANNELS, OUT_CHANNELS, KERNEL_SIZE, INPUT_LENGTH,
                              stride=STRIDE, padding_type="EXPLICIT", padding=PADDING)
    expected_no_bias = conv1d_bfp_ref(x_codes, x_exps, X_QC, w_codes, w_exps, W_QC,
                                      None, None, None,
                                      BATCH, IN_CHANNELS, OUT_CHANNELS, KERNEL_SIZE,
                                      INPUT_LENGTH, stride=STRIDE,
                                      padding_type="EXPLICIT", padding=PADDING)
    assert len(expected) == BATCH * OUT_CHANNELS * OUT_LEN

    # A bias-ignoring kernel must not be able to pass both gold tests.
    assert all(w != n for w, n in zip(expected, expected_no_bias)), (
        "with-bias and no-bias expectations coincide somewhere -- bias seed "
        "would be unobservable there")

    # Dilation fixture: same operands, dilation 2.
    # The dilation arithmetic is BFP-arm-local (inputIdx = firstValidInputIdx
    # + i * geom.dilation), so the main dilation=1 gold cannot see a dropped
    # dilation factor -- this fixture's taps step by 2 and a hardcode-1 mutant
    # reads p, p+1, p+2 instead of p, p+2, p+4.
    dil_dilation = 2
    dil_out_len = (INPUT_LENGTH + 2 * PADDING - (dil_dilation * (KERNEL_SIZE - 1) + 1)) \
        // STRIDE + 1
    expected_dil = conv1d_bfp_ref(x_codes, x_exps, X_QC, w_codes, w_exps, W_QC,
                                  bias_codes, bias_exps, BIAS_QC,
                                  BATCH, IN_CHANNELS, OUT_CHANNELS, KERNEL_SIZE, INPUT_LENGTH,
                                  stride=STRIDE, dilation=dil_dilation,
                                  padding_type="EXPLICIT", padding=PADDING)
    assert len(expected_dil) == BATCH * OUT_CHANNELS * dil_out_len
    assert len(set(expected_dil)) >= 2, "dilation fixture: outputs degenerate"

    # Grouped-bias fixture: the SAME bias values
    # stored grouped {numGroups=2, groupSize=1} -- each output channel its own
    # exponent. Values exact under their per-value grids too, so the expected
    # output is BIT-IDENTICAL to the per-tensor-bias gold (asserted); a kernel
    # reading every bias seed through group 0 (the bfpGroupOf-drop mutant at
    # Conv1dKernel.c's bias seed) is off by a power of two on the other
    # channel -- asserted non-vacuous.
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
    expected_grouped_bias = conv1d_bfp_ref(x_codes, x_exps, X_QC, w_codes, w_exps, W_QC,
                                           bias_g_codes, bias_g_exps, bias_g_qc,
                                           BATCH, IN_CHANNELS, OUT_CHANNELS, KERNEL_SIZE,
                                           INPUT_LENGTH, stride=STRIDE,
                                           padding_type="EXPLICIT", padding=PADDING)
    assert expected_grouped_bias == expected, (
        "grouped-bias expectation must be bit-identical to the per-tensor gold "
        "-- both grids are exact for these values")

    # ---- conv-groups=2 (#416) --------------------------------------------
    g2x_codes, g2x_exps = bfp_quantize_grouped(G2_X_VALUES, G2_X_QC["mantissa_bits"],
                                               G2_X_QC["exponent_bits"], G2_X_QC["group_size"])
    g2w_codes, g2w_exps = bfp_quantize_grouped(G2_W_VALUES, G2_W_QC["mantissa_bits"],
                                               G2_W_QC["exponent_bits"], G2_W_QC["group_size"])
    g2b_codes, g2b_exps = bfp_quantize_grouped(G2_BIAS_VALUES, G2_BIAS_QC["mantissa_bits"],
                                               G2_BIAS_QC["exponent_bits"],
                                               G2_BIAS_QC["group_size"])
    assert len(g2x_exps) == G2_X_NUM_GROUPS and len(g2w_exps) == G2_W_NUM_GROUPS
    check_exact_roundtrip("g2 input", G2_X_VALUES, g2x_codes, g2x_exps, G2_X_QC)
    check_exact_roundtrip("g2 weight", G2_W_VALUES, g2w_codes, g2w_exps, G2_W_QC)
    check_exact_roundtrip("g2 bias", G2_BIAS_VALUES, g2b_codes, g2b_exps, G2_BIAS_QC)
    assert len(set(g2x_exps)) >= 2, "g2 input: exponent array is uniform -- fixture too weak"
    assert len(set(g2w_exps)) >= 2, "g2 weight: exponent array is uniform -- fixture too weak"

    g2_expected = conv1d_bfp_ref(g2x_codes, g2x_exps, G2_X_QC, g2w_codes, g2w_exps, G2_W_QC,
                                 g2b_codes, g2b_exps, G2_BIAS_QC,
                                 BATCH, G2_IN_CHANNELS, G2_OUT_CHANNELS, KERNEL_SIZE,
                                 G2_INPUT_LENGTH, stride=STRIDE, padding_type="EXPLICIT",
                                 padding=PADDING, conv_groups=G2_CONV_GROUPS)
    assert len(g2_expected) == BATCH * G2_OUT_CHANNELS * G2_OUT_LEN

    # dx sub-fixture: same weight, VALID unit-stride adjoint (see the operand
    # block's comment for the self_check=False rationale).
    g2dx_loss_codes, g2dx_loss_exps = bfp_quantize_grouped(
        G2_DX_LOSS_VALUES, G2_DX_LOSS_QC["mantissa_bits"], G2_DX_LOSS_QC["exponent_bits"],
        G2_DX_LOSS_QC["group_size"])
    assert len(g2dx_loss_exps) == G2_DX_LOSS_NUM_GROUPS
    check_exact_roundtrip("g2 dx loss", G2_DX_LOSS_VALUES, g2dx_loss_codes, g2dx_loss_exps,
                          G2_DX_LOSS_QC)
    assert len(set(g2dx_loss_exps)) == G2_DX_LOSS_NUM_GROUPS, (
        "g2 dx loss: group exponents not pairwise distinct -- per-channel "
        "exponent binding unobservable")
    g2_dx = conv1d_bfp_dx_ref(g2dx_loss_codes, g2dx_loss_exps, G2_DX_LOSS_QC,
                              g2w_codes, g2w_exps, G2_W_QC,
                              BATCH, G2_IN_CHANNELS, G2_OUT_CHANNELS, KERNEL_SIZE,
                              G2_DX_INPUT_LENGTH, stride=1, dilation=1,
                              conv_groups=G2_CONV_GROUPS, self_check=False)
    g2_dx_loss_collapsed = conv1d_bfp_dx_ref(
        g2dx_loss_codes, [g2dx_loss_exps[0]], {**G2_DX_LOSS_QC, "group_size": 0},
        g2w_codes, g2w_exps, G2_W_QC, BATCH, G2_IN_CHANNELS, G2_OUT_CHANNELS,
        KERNEL_SIZE, G2_DX_INPUT_LENGTH, stride=1, dilation=1,
        conv_groups=G2_CONV_GROUPS, self_check=False)
    g2_dx_w_collapsed = conv1d_bfp_dx_ref(
        g2dx_loss_codes, g2dx_loss_exps, G2_DX_LOSS_QC,
        g2w_codes, [g2w_exps[0]], {**G2_W_QC, "group_size": 0},
        BATCH, G2_IN_CHANNELS, G2_OUT_CHANNELS, KERNEL_SIZE, G2_DX_INPUT_LENGTH,
        stride=1, dilation=1, conv_groups=G2_CONV_GROUPS, self_check=False)
    assert g2_dx_loss_collapsed != g2_dx, (
        "g2 dx: loss-exponent collapse is indistinguishable -- loss group "
        "structure unobservable through the grouped dx path")
    assert g2_dx_w_collapsed != g2_dx, (
        "g2 dx: weight-exponent collapse is indistinguishable -- weight group "
        "structure unobservable through the grouped dx path")
    assert any(v != 0.0 for v in g2_dx), "g2 dx: all-zero expected output"

    # ---- Integer-only forward (conv1dKernelBfpRaw, arithmetic-bfp section 13)
    # The same operands through the integer fold: int32 mantissas + one
    # UNBIASED exponent per raw target group, pinned bit for bit, plus the
    # float64 exact values and the per-element fold bound 1/2 (S+1) 2^E_acc.
    def raw_main(bias_triple, raw_gs, **kw):
        bc, be, bq = bias_triple if bias_triple is not None else (None, None, None)
        return conv1d_bfp_raw_ref(x_codes, x_exps, X_QC, w_codes, w_exps, W_QC, bc, be, bq,
                                  BATCH, IN_CHANNELS, OUT_CHANNELS, KERNEL_SIZE, INPUT_LENGTH,
                                  stride=STRIDE, padding_type="EXPLICIT", padding=PADDING,
                                  raw_group_size=raw_gs, **kw)

    raw = raw_main((bias_codes, bias_exps, BIAS_QC), 0)
    raw_no_bias = raw_main(None, 0)
    raw_bias_grouped = raw_main((bias_g_codes, bias_g_exps, bias_g_qc), 0)
    # Raw target groups of 2 over the [1, 2, 5] output: group 2 straddles the
    # two output rows (channels), so its E_acc must cover BOTH rows' maxima.
    # Output channel 1's weight groups (3..5) get their stored exponent bumped
    # by 3 (codes unchanged, values x8) so the two rows' maxima DIFFER and the
    # straddling group observably takes the larger one.
    w_exps_gs2 = [e + (3 if g >= 3 else 0) for g, e in enumerate(w_exps)]
    raw_gs2 = conv1d_bfp_raw_ref(x_codes, x_exps, X_QC, w_codes, w_exps_gs2, W_QC,
                                 bias_codes, bias_exps, BIAS_QC,
                                 BATCH, IN_CHANNELS, OUT_CHANNELS, KERNEL_SIZE, INPUT_LENGTH,
                                 stride=STRIDE, padding_type="EXPLICIT", padding=PADDING,
                                 raw_group_size=2)
    assert len(raw_gs2["exps"]) == 5
    assert raw_gs2["exps"][3] == raw_gs2["exps"][1] + 3, "raw gs2: rows do not differ"
    assert raw_gs2["exps"][2] == raw_gs2["exps"][3], (
        "raw gs2: the straddling group's exponent does not cover the larger row")
    assert raw["mant"] != raw_no_bias["mant"], "raw: bias segment unobservable"
    raw_dil = raw_main((bias_codes, bias_exps, BIAS_QC), 0, dilation=dil_dilation)
    assert len(raw_dil["mant"]) == BATCH * OUT_CHANNELS * dil_out_len
    g2_raw = conv1d_bfp_raw_ref(g2x_codes, g2x_exps, G2_X_QC, g2w_codes, g2w_exps, G2_W_QC,
                                g2b_codes, g2b_exps, G2_BIAS_QC,
                                BATCH, G2_IN_CHANNELS, G2_OUT_CHANNELS, KERNEL_SIZE,
                                G2_INPUT_LENGTH, stride=STRIDE, padding_type="EXPLICIT",
                                padding=PADDING, conv_groups=G2_CONV_GROUPS)
    assert len(g2_raw["mant"]) == BATCH * G2_OUT_CHANNELS * G2_OUT_LEN
    # The grouped-weight / grouped-input structure must be observable through
    # the integer fold too: a per-tensor collapse of either operand changes
    # the mantissas (the exponent pre-pass alone would hide a group mix-up
    # only if the fold were also wrong in the same way).
    raw_x_collapsed = conv1d_bfp_raw_ref(x_codes, [x_exps[0]], {**X_QC, "group_size": 0},
                                         w_codes, w_exps, W_QC, bias_codes, bias_exps, BIAS_QC,
                                         BATCH, IN_CHANNELS, OUT_CHANNELS, KERNEL_SIZE,
                                         INPUT_LENGTH, stride=STRIDE, padding_type="EXPLICIT",
                                         padding=PADDING)
    raw_w_collapsed = conv1d_bfp_raw_ref(x_codes, x_exps, X_QC, w_codes, [w_exps[0]],
                                         {**W_QC, "group_size": 0}, bias_codes, bias_exps,
                                         BIAS_QC, BATCH, IN_CHANNELS, OUT_CHANNELS, KERNEL_SIZE,
                                         INPUT_LENGTH, stride=STRIDE, padding_type="EXPLICIT",
                                         padding=PADDING)
    assert raw_x_collapsed["mant"] != raw["mant"], "raw: input group structure unobservable"
    assert raw_w_collapsed["mant"] != raw["mant"], "raw: weight group structure unobservable"

    # Two-sided alignment fixture: input groups 24 binades apart, so with the
    # (negative) fold guard of this reduction the max-exponent segment is
    # LEFT-shifted exactly and the low-exponent segment is RIGHT-shifted and
    # rounded -- both arms inside ONE output (position 3, taps 3 and 4 sit in
    # different input groups). Group 1's last code is -7 so the right shift
    # lands on a HALF_AWAY tie (3.5 -> 4), not an exact quotient.
    spread_x_values = [16777216.0, -33554432.0, 50331648.0, -16777216.0,
                       -0.875, -2.0, 3.0, 1.0]
    spread_x_qc = {"mantissa_bits": 6, "exponent_bits": 8, "group_size": 4}
    spread_w_values = [1.0, -0.5]
    spread_w_qc = {"mantissa_bits": 4, "exponent_bits": 8, "group_size": 0}
    spread_x_codes, spread_x_exps = bfp_quantize_grouped(
        spread_x_values, spread_x_qc["mantissa_bits"], spread_x_qc["exponent_bits"],
        spread_x_qc["group_size"])
    spread_w_codes, spread_w_exps = bfp_quantize_grouped(
        spread_w_values, spread_w_qc["mantissa_bits"], spread_w_qc["exponent_bits"], 0)
    check_exact_roundtrip("spread input", spread_x_values, spread_x_codes, spread_x_exps,
                          spread_x_qc)
    check_exact_roundtrip("spread weight", spread_w_values, spread_w_codes, spread_w_exps,
                          spread_w_qc)
    assert spread_x_exps[0] - spread_x_exps[1] >= 21, "spread: groups not far enough apart"
    spread_raw = conv1d_bfp_raw_ref(spread_x_codes, spread_x_exps, spread_x_qc,
                                    spread_w_codes, spread_w_exps, spread_w_qc,
                                    None, None, None, 1, 1, 1, 2, 8)
    assert spread_raw["g"] < 0, "spread: the fold guard is not negative -- no left shift"
    e_acc = spread_raw["exps"][0]
    assert spread_raw["mant"][3] * 2.0 ** e_acc != spread_raw["exact"][3], (
        "spread: output 3 is exact -- the right-shift rounding arm is unobserved")
    assert abs(spread_raw["mant"][3]) > 2 ** 20, (
        "spread: output 3 is small -- the left-shift arm is unobserved")

    # m16 x m16: bfpSegmentLimit == 1, so EVERY product is its own segment and
    # the run splitting (not a fail-fast) carries the reduction; g = 1.
    m16_x_codes = [3, -5, 7, 2]
    m16_w_codes = [2, -3]
    m16_qc = {"mantissa_bits": 16, "exponent_bits": 8, "group_size": 0}
    m16_raw = conv1d_bfp_raw_ref(m16_x_codes, [127], m16_qc, m16_w_codes, [127], m16_qc,
                                 None, None, None, 1, 1, 1, 2, 4)
    assert m16_raw["g"] == 1 and m16_raw["exps"] == [1]
    assert m16_raw["mant"] == [11, -16, 4], m16_raw["mant"]

    # m16 x m8, one 600-tap reduction at the code floors: the product is
    # exactly 2^22 and 600 of them overflow int32, so the run splits at 511;
    # g = 1 aligns both partials exactly (no rounding: both are even).
    long_raw = conv1d_bfp_raw_ref([-32768] * 600, [127], m16_qc, [-128] * 600, [127],
                                  {"mantissa_bits": 8, "exponent_bits": 8, "group_size": 0},
                                  None, None, None, 1, 1, 1, 600, 600)
    assert long_raw["mant"] == [600 * 2 ** 21] and long_raw["exps"] == [1], long_raw

    parts = [
        "// AUTOGENERATED by generate_expected_bfp_conv1d.py — DO NOT EDIT\n",
        "#ifndef ODT_EXPECTED_BFP_CONV1D_H\n",
        "#define ODT_EXPECTED_BFP_CONV1D_H\n",
        "#include <stdint.h>\n",
        "#include <stdlib.h>\n\n",
        emit_int32_scalar("kBfpConvBatch", BATCH),
        emit_int32_scalar("kBfpConvInChannels", IN_CHANNELS),
        emit_int32_scalar("kBfpConvInputLength", INPUT_LENGTH),
        emit_int32_scalar("kBfpConvOutChannels", OUT_CHANNELS),
        emit_int32_scalar("kBfpConvKernelSize", KERNEL_SIZE),
        emit_int32_scalar("kBfpConvStride", STRIDE),
        emit_int32_scalar("kBfpConvPadding", PADDING),
        emit_int32_scalar("kBfpConvOutLen", OUT_LEN),
        emit_int32_array("kBfpConvInCodes", torch.tensor(x_codes)),
        emit_uint8_array("kBfpConvInExponents", x_exps),
        emit_int32_scalar("kBfpConvInNumGroups", X_NUM_GROUPS),
        emit_int32_scalar("kBfpConvInGroupSize", X_QC["group_size"]),
        emit_int32_scalar("kBfpConvInMantissaBits", X_QC["mantissa_bits"]),
        emit_int32_scalar("kBfpConvInExponentBits", X_QC["exponent_bits"]),
        emit_int32_array("kBfpConvWCodes", torch.tensor(w_codes)),
        emit_uint8_array("kBfpConvWExponents", w_exps),
        emit_int32_scalar("kBfpConvWNumGroups", W_NUM_GROUPS),
        emit_int32_scalar("kBfpConvWGroupSize", W_QC["group_size"]),
        emit_int32_scalar("kBfpConvWMantissaBits", W_QC["mantissa_bits"]),
        emit_int32_scalar("kBfpConvWExponentBits", W_QC["exponent_bits"]),
        emit_int32_array("kBfpConvBiasCodes", torch.tensor(bias_codes)),
        emit_uint8_array("kBfpConvBiasExponents", bias_exps),
        emit_int32_scalar("kBfpConvBiasMantissaBits", BIAS_QC["mantissa_bits"]),
        emit_int32_scalar("kBfpConvBiasExponentBits", BIAS_QC["exponent_bits"]),
        emit_int32_array("kBfpConvBiasGroupedCodes", torch.tensor(bias_g_codes)),
        emit_uint8_array("kBfpConvBiasGroupedExponents", bias_g_exps),
        emit_int32_scalar("kBfpConvBiasGroupedNumGroups", len(bias_g_exps)),
        emit_int32_scalar("kBfpConvDilDilation", dil_dilation),
        emit_int32_scalar("kBfpConvDilOutLen", dil_out_len),
        emit_float_array("kBfpConvDilExpected",
                         torch.tensor(expected_dil, dtype=torch.float32)),
        emit_float_array("kBfpConvExpected", torch.tensor(expected, dtype=torch.float32)),
        emit_float_array("kBfpConvNoBiasExpected",
                         torch.tensor(expected_no_bias, dtype=torch.float32)),
        emit_int32_scalar("kBfpConvG2ConvGroups", G2_CONV_GROUPS),
        emit_int32_scalar("kBfpConvG2InChannels", G2_IN_CHANNELS),
        emit_int32_scalar("kBfpConvG2OutChannels", G2_OUT_CHANNELS),
        emit_int32_scalar("kBfpConvG2InputLength", G2_INPUT_LENGTH),
        emit_int32_scalar("kBfpConvG2OutLen", G2_OUT_LEN),
        emit_int32_array("kBfpConvG2InCodes", torch.tensor(g2x_codes)),
        emit_uint8_array("kBfpConvG2InExponents", g2x_exps),
        emit_int32_scalar("kBfpConvG2InNumGroups", G2_X_NUM_GROUPS),
        emit_int32_scalar("kBfpConvG2InGroupSize", G2_X_QC["group_size"]),
        emit_int32_scalar("kBfpConvG2InMantissaBits", G2_X_QC["mantissa_bits"]),
        emit_int32_scalar("kBfpConvG2InExponentBits", G2_X_QC["exponent_bits"]),
        emit_float_array("kBfpConvG2WValues", torch.tensor(G2_W_VALUES, dtype=torch.float32)),
        emit_int32_array("kBfpConvG2WCodes", torch.tensor(g2w_codes)),
        emit_uint8_array("kBfpConvG2WExponents", g2w_exps),
        emit_int32_scalar("kBfpConvG2WNumGroups", G2_W_NUM_GROUPS),
        emit_int32_scalar("kBfpConvG2WGroupSize", G2_W_QC["group_size"]),
        emit_int32_scalar("kBfpConvG2WMantissaBits", G2_W_QC["mantissa_bits"]),
        emit_int32_scalar("kBfpConvG2WExponentBits", G2_W_QC["exponent_bits"]),
        emit_int32_array("kBfpConvG2BiasCodes", torch.tensor(g2b_codes)),
        emit_uint8_array("kBfpConvG2BiasExponents", g2b_exps),
        emit_int32_scalar("kBfpConvG2BiasMantissaBits", G2_BIAS_QC["mantissa_bits"]),
        emit_int32_scalar("kBfpConvG2BiasExponentBits", G2_BIAS_QC["exponent_bits"]),
        emit_float_array("kBfpConvG2Expected", torch.tensor(g2_expected, dtype=torch.float32)),
        emit_int32_scalar("kBfpConvG2DxInputLength", G2_DX_INPUT_LENGTH),
        emit_int32_scalar("kBfpConvG2DxOutLen", G2_DX_OUT_LEN),
        emit_int32_array("kBfpConvG2DxLossCodes", torch.tensor(g2dx_loss_codes)),
        emit_uint8_array("kBfpConvG2DxLossExponents", g2dx_loss_exps),
        emit_int32_scalar("kBfpConvG2DxLossNumGroups", G2_DX_LOSS_NUM_GROUPS),
        emit_int32_scalar("kBfpConvG2DxLossGroupSize", G2_DX_LOSS_QC["group_size"]),
        emit_int32_scalar("kBfpConvG2DxLossMantissaBits", G2_DX_LOSS_QC["mantissa_bits"]),
        emit_int32_scalar("kBfpConvG2DxLossExponentBits", G2_DX_LOSS_QC["exponent_bits"]),
        emit_float_array("kBfpConvG2DxExpected", torch.tensor(g2_dx, dtype=torch.float32)),
        # ---- integer-only forward (conv1dKernelBfpRaw) ----
        emit_int32_scalar("kBfpConvRawG", raw["g"]),
        emit_int32_array("kBfpConvRawMant", torch.tensor(raw["mant"])),
        emit_int16_array("kBfpConvRawExps", raw["exps"]),
        emit_double_array("kBfpConvRawExact", raw["exact"]),
        emit_double_array("kBfpConvRawBound", raw["bound"]),
        emit_int32_array("kBfpConvRawNoBiasMant", torch.tensor(raw_no_bias["mant"])),
        emit_int16_array("kBfpConvRawNoBiasExps", raw_no_bias["exps"]),
        emit_int32_array("kBfpConvRawBiasGroupedMant", torch.tensor(raw_bias_grouped["mant"])),
        emit_int16_array("kBfpConvRawBiasGroupedExps", raw_bias_grouped["exps"]),
        emit_int32_scalar("kBfpConvRawGs2GroupSize", 2),
        emit_uint8_array("kBfpConvRawGs2WExponents", w_exps_gs2),
        emit_int32_array("kBfpConvRawGs2Mant", torch.tensor(raw_gs2["mant"])),
        emit_int16_array("kBfpConvRawGs2Exps", raw_gs2["exps"]),
        emit_int32_array("kBfpConvRawDilMant", torch.tensor(raw_dil["mant"])),
        emit_int16_array("kBfpConvRawDilExps", raw_dil["exps"]),
        emit_int32_array("kBfpConvG2RawMant", torch.tensor(g2_raw["mant"])),
        emit_int16_array("kBfpConvG2RawExps", g2_raw["exps"]),
        emit_int32_scalar("kBfpConvSpreadInputLength", 8),
        emit_int32_scalar("kBfpConvSpreadOutLen", 7),
        emit_int32_array("kBfpConvSpreadInCodes", torch.tensor(spread_x_codes)),
        emit_uint8_array("kBfpConvSpreadInExponents", spread_x_exps),
        emit_int32_scalar("kBfpConvSpreadInMantissaBits", spread_x_qc["mantissa_bits"]),
        emit_int32_scalar("kBfpConvSpreadInGroupSize", spread_x_qc["group_size"]),
        emit_int32_array("kBfpConvSpreadWCodes", torch.tensor(spread_w_codes)),
        emit_uint8_array("kBfpConvSpreadWExponents", spread_w_exps),
        emit_int32_scalar("kBfpConvSpreadWMantissaBits", spread_w_qc["mantissa_bits"]),
        emit_int32_scalar("kBfpConvSpreadG", spread_raw["g"]),
        emit_int32_array("kBfpConvSpreadRawMant", torch.tensor(spread_raw["mant"])),
        emit_int16_array("kBfpConvSpreadRawExps", spread_raw["exps"]),
        emit_double_array("kBfpConvSpreadRawExact", spread_raw["exact"]),
        emit_double_array("kBfpConvSpreadRawBound", spread_raw["bound"]),
        emit_int32_array("kBfpConvM16InCodes", torch.tensor(m16_x_codes)),
        emit_int32_array("kBfpConvM16WCodes", torch.tensor(m16_w_codes)),
        emit_int32_array("kBfpConvM16RawMant", torch.tensor(m16_raw["mant"])),
        emit_int16_array("kBfpConvM16RawExps", m16_raw["exps"]),
        emit_double_array("kBfpConvM16RawExact", m16_raw["exact"]),
        emit_double_array("kBfpConvM16RawBound", m16_raw["bound"]),
        emit_int32_scalar("kBfpConvLongTaps", 600),
        emit_int32_scalar("kBfpConvLongRawMant", long_raw["mant"][0]),
        emit_int32_scalar("kBfpConvLongRawExp", long_raw["exps"][0]),
        "\n#endif // ODT_EXPECTED_BFP_CONV1D_H\n",
    ]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(parts))
    return 0


if __name__ == "__main__":
    sys.exit(main())
