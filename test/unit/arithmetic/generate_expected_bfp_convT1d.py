#!/usr/bin/env python3
"""Generate expected_bfp_convT1d.h for UnitTestConvTranspose1dKernel (BFP epic
PR2, Task 5 -- spec docs/superpowers/specs/2026-07-29-block-floating-point-design.md,
decision D9).

Pins convTranspose1dKernelBfpGather's fold order (ConvTranspose1dKernel.c):
per (b, conv-group, oc, outPos) ONE int32 partial over the contributors
convTranspose1dTapsAt enumerates, walked taps OUTER / icOffset INNER; per
visited step BOTH operands' storage indices -> group ids (bfpGroupOf); when
EITHER id changes the partial folds into a float32 accumulator via
ldexpf((float)partial, Ein + Ew - biasIn - biasW) and resets; tail fold after
the walk; outputPadding tail positions have ZERO taps and stay at the bias
seed; bias is a value-seed dequantized to float BEFORE the reduction. The
kernel never rounds -- see sym_gold.convT1d_bfp_gather_ref for the exact
np.float32-mirrored emulation (including its float-scatter cross-check).

Fixture geometry (input [1,4,5], weight [4,2,3] = [Cin, Cout/groups, K],
stride 2, dilation 1, outputPadding 1, VALID, conv groups 1 -> output
[1,2,12]): input grouped numGroups=2/groupSize=10 (m=6, e=8); weight grouped
numGroups=3/groupSize=8 (m=4, e=8). This pairing puts boundaries where the
gather walk (taps outer, ic inner: input jumps by Lin=5, weight by
outChPerGroup*K=6 per ic step) crosses them DISJOINTLY: e.g. oc=0, outPos=2
(taps k={0,2}) hits a weight-only crossing at ic2->ic3 within the k=0 tap
(w 12->18 crosses g1->g2 while the input stays in its 10-wide group) and an
input-only crossing at ic1->ic2 within the k=2 tap (in 5->10 crosses g0->g1
while w 8->14 stays in g1) -- both fold-clause directions observable (Task 3/4
review lesson). outPos=11 (the outputPadding tail) has NO taps -> bias-only.
Bias per-tensor (m=8, e=8), both values nonzero so the with-bias/no-bias
expectations differ even at the tap-free tail. Values are SMALL and grid-exact
(every code * scale reproduces the input float bit-for-bit -- asserted below),
so every fold is exact float32 arithmetic and the expected outputs are
bit-pinned via TEST_ASSERT_EQUAL_MEMORY, not a tolerance.

Self-checks (abort generation rather than emit a vacuous fixture):
  - convT1d_bfp_gather_ref's built-ins: (i) >= 2 groups crossed on EACH
    operand within a single reduction; (ii) >= 1 fold with a nonzero
    exactly-float-convertible partial; (iii) the grouped result differs from
    an all-per-tensor (exponents[0]) collapse; >= 1 input-only AND >= 1
    weight-only boundary event; >= 1 tap-free output position; and the
    SCATTER CROSS-CHECK (a float32 scatter on dequantized values reproduces
    the gather bit-for-bit -- the D9 gather-equals-scatter pin).
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

from sym_gold import (assert_rounding_canary, bfp_quantize_grouped, convT1d_bfp_gather_ref,
                      emit_float_array, emit_int32_array, emit_int32_scalar)

BATCH = 1
IN_CHANNELS = 4
INPUT_LENGTH = 5
OUT_CHANNELS = 2
KERNEL_SIZE = 3
STRIDE = 2
OUTPUT_PADDING = 1
OUT_LEN = 12  # (5-1)*2 + 1*(3-1) + 1 + 1

# input: [1, 4, 5] row-major flat, quantization groups of 10 -> group == ic/2
# ({ic0,ic1} share group 0, {ic2,ic3} group 1): within-tap ic steps (+5) cross
# the input boundary only at ic1->ic2, leaving ic0->ic1 / ic2->ic3 input-quiet
# for the weight-only events (see module docstring). Group absmaxes 6.0 / 12.0
# land scales 0.25 / 0.5 (m=6, snap-up grid).
X_VALUES = [1.0, -2.25, 3.5, -0.75, 6.0,
            0.5, 2.0, -1.25, 4.75, -3.0,
            2.5, -12.0, 7.5, 0.5, -4.0,
            -1.5, 3.0, 9.5, -6.0, 10.0]
X_QC = {"mantissa_bits": 6, "exponent_bits": 8, "group_size": 10}
X_NUM_GROUPS = 2

# weight: [4, 2, 3] row-major flat ([Cin, Cout/groups, K]), quantization groups
# of 8: boundaries at flat 8 and 16 fall MID-ic-slab (slabs are 6 wide), so
# within-tap ic steps (+6) sometimes stay in a weight group (8->14) and
# sometimes cross one (12->18). Group absmaxes 3.0 / 1.5 / 6.0 land scales
# 0.5 / 0.25 / 1.0 (m=4).
W_VALUES = [1.5, -0.5, 2.0, -1.0, 3.0, 0.5,
            2.5, -2.0, 0.75, -1.5, 1.0, 0.25,
            1.25, -0.75, 0.5, 1.5, -2.0, 4.0,
            1.0, -3.0, 6.0, -4.0, 2.0, -1.0]
W_QC = {"mantissa_bits": 4, "exponent_bits": 8, "group_size": 8}
W_NUM_GROUPS = 3

BIAS_VALUES = [1.5, -0.75]
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

    expected = convT1d_bfp_gather_ref(x_codes, x_exps, X_QC, w_codes, w_exps, W_QC,
                                      bias_codes, bias_exps, BIAS_QC,
                                      BATCH, IN_CHANNELS, OUT_CHANNELS, KERNEL_SIZE,
                                      INPUT_LENGTH, stride=STRIDE,
                                      output_padding=OUTPUT_PADDING)
    expected_no_bias = convT1d_bfp_gather_ref(x_codes, x_exps, X_QC, w_codes, w_exps, W_QC,
                                              None, None, None,
                                              BATCH, IN_CHANNELS, OUT_CHANNELS, KERNEL_SIZE,
                                              INPUT_LENGTH, stride=STRIDE,
                                              output_padding=OUTPUT_PADDING)
    assert len(expected) == BATCH * OUT_CHANNELS * OUT_LEN

    # A bias-ignoring kernel must not be able to pass both gold tests -- this
    # holds even at the tap-free outputPadding tail because both bias values
    # are nonzero (tail = bias seed vs 0.0).
    assert all(w != n for w, n in zip(expected, expected_no_bias)), (
        "with-bias and no-bias expectations coincide somewhere -- bias seed "
        "would be unobservable there")

    # Dilation fixture (PR2 self-review finding 1): same operands, dilation 2.
    # The gather forwards kernel->dilation into convTranspose1dTapsAt only in
    # the BFP arm, so the main dilation=1 gold cannot see a hardcoded 1 there
    # -- this fixture's contributor enumeration genuinely depends on it.
    dil_dilation = 2
    dil_out_len = (INPUT_LENGTH - 1) * STRIDE + dil_dilation * (KERNEL_SIZE - 1) \
        + OUTPUT_PADDING + 1
    expected_dil = convT1d_bfp_gather_ref(x_codes, x_exps, X_QC, w_codes, w_exps, W_QC,
                                          bias_codes, bias_exps, BIAS_QC,
                                          BATCH, IN_CHANNELS, OUT_CHANNELS, KERNEL_SIZE,
                                          INPUT_LENGTH, stride=STRIDE, dilation=dil_dilation,
                                          output_padding=OUTPUT_PADDING)
    assert len(expected_dil) == BATCH * OUT_CHANNELS * dil_out_len
    assert len(set(expected_dil)) >= 2, "dilation fixture: outputs degenerate"

    # Grouped-bias fixture (PR2 self-review finding 3): the SAME bias values
    # stored grouped {numGroups=2, groupSize=1} -- each output channel its own
    # exponent. Values exact under their per-value grids too, so the expected
    # output is BIT-IDENTICAL to the per-tensor-bias gold (asserted); a kernel
    # reading every bias seed through group 0 (the bfpGroupOf-drop mutant at
    # ConvTranspose1dKernel.c's bias seed) is off by a power of two on the
    # other channel -- asserted non-vacuous.
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
    expected_grouped_bias = convT1d_bfp_gather_ref(x_codes, x_exps, X_QC, w_codes, w_exps,
                                                   W_QC, bias_g_codes, bias_g_exps, bias_g_qc,
                                                   BATCH, IN_CHANNELS, OUT_CHANNELS,
                                                   KERNEL_SIZE, INPUT_LENGTH, stride=STRIDE,
                                                   output_padding=OUTPUT_PADDING)
    assert expected_grouped_bias == expected, (
        "grouped-bias expectation must be bit-identical to the per-tensor gold "
        "-- both grids are exact for these values")

    parts = [
        "// AUTOGENERATED by generate_expected_bfp_convT1d.py — DO NOT EDIT\n",
        "#ifndef ODT_EXPECTED_BFP_CONVT1D_H\n",
        "#define ODT_EXPECTED_BFP_CONVT1D_H\n",
        "#include <stdint.h>\n",
        "#include <stdlib.h>\n\n",
        emit_int32_scalar("kBfpConvTBatch", BATCH),
        emit_int32_scalar("kBfpConvTInChannels", IN_CHANNELS),
        emit_int32_scalar("kBfpConvTInputLength", INPUT_LENGTH),
        emit_int32_scalar("kBfpConvTOutChannels", OUT_CHANNELS),
        emit_int32_scalar("kBfpConvTKernelSize", KERNEL_SIZE),
        emit_int32_scalar("kBfpConvTStride", STRIDE),
        emit_int32_scalar("kBfpConvTOutputPadding", OUTPUT_PADDING),
        emit_int32_scalar("kBfpConvTOutLen", OUT_LEN),
        emit_int32_array("kBfpConvTInCodes", torch.tensor(x_codes)),
        emit_uint8_array("kBfpConvTInExponents", x_exps),
        emit_int32_scalar("kBfpConvTInNumGroups", X_NUM_GROUPS),
        emit_int32_scalar("kBfpConvTInGroupSize", X_QC["group_size"]),
        emit_int32_scalar("kBfpConvTInMantissaBits", X_QC["mantissa_bits"]),
        emit_int32_scalar("kBfpConvTInExponentBits", X_QC["exponent_bits"]),
        emit_int32_array("kBfpConvTWCodes", torch.tensor(w_codes)),
        emit_uint8_array("kBfpConvTWExponents", w_exps),
        emit_int32_scalar("kBfpConvTWNumGroups", W_NUM_GROUPS),
        emit_int32_scalar("kBfpConvTWGroupSize", W_QC["group_size"]),
        emit_int32_scalar("kBfpConvTWMantissaBits", W_QC["mantissa_bits"]),
        emit_int32_scalar("kBfpConvTWExponentBits", W_QC["exponent_bits"]),
        emit_int32_array("kBfpConvTBiasCodes", torch.tensor(bias_codes)),
        emit_uint8_array("kBfpConvTBiasExponents", bias_exps),
        emit_int32_scalar("kBfpConvTBiasMantissaBits", BIAS_QC["mantissa_bits"]),
        emit_int32_scalar("kBfpConvTBiasExponentBits", BIAS_QC["exponent_bits"]),
        emit_int32_array("kBfpConvTBiasGroupedCodes", torch.tensor(bias_g_codes)),
        emit_uint8_array("kBfpConvTBiasGroupedExponents", bias_g_exps),
        emit_int32_scalar("kBfpConvTBiasGroupedNumGroups", len(bias_g_exps)),
        emit_int32_scalar("kBfpConvTDilDilation", dil_dilation),
        emit_int32_scalar("kBfpConvTDilOutLen", dil_out_len),
        emit_float_array("kBfpConvTDilExpected",
                         torch.tensor(expected_dil, dtype=torch.float32)),
        emit_float_array("kBfpConvTExpected", torch.tensor(expected, dtype=torch.float32)),
        emit_float_array("kBfpConvTNoBiasExpected",
                         torch.tensor(expected_no_bias, dtype=torch.float32)),
        "\n#endif // ODT_EXPECTED_BFP_CONVT1D_H\n",
    ]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(parts))
    return 0


if __name__ == "__main__":
    sys.exit(main())
