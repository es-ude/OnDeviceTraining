#!/usr/bin/env python3
"""Generate expected_bfp_conv1d.h for UnitTestConv1dKernel (BFP epic PR2,
Task 4 -- spec docs/superpowers/specs/2026-07-29-block-floating-point-design.md).

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
the input's group id would be untestable (Task 3 review lesson, controller-
authorized adjustment). With groupSize=2 the weight crosses groups MID-run
(e.g. oc=0, outPos=1: tap (in 2, w 1)->(in 3, w 2) is weight-only, the
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

from sym_gold import (assert_rounding_canary, bfp_quantize_grouped, conv1d_bfp_ref,
                      emit_float_array, emit_int32_array, emit_int32_scalar)

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
        emit_float_array("kBfpConvExpected", torch.tensor(expected, dtype=torch.float32)),
        emit_float_array("kBfpConvNoBiasExpected",
                         torch.tensor(expected_no_bias, dtype=torch.float32)),
        "\n#endif // ODT_EXPECTED_BFP_CONV1D_H\n",
    ]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(parts))
    return 0


if __name__ == "__main__":
    sys.exit(main())
