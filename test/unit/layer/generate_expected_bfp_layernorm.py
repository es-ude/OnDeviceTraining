#!/usr/bin/env python3
"""Generate expected_bfp_layernorm.h for UnitTestLayerNorm's native ARITH_BFP
tests (BFP epic PR5, R-N1/R-N2/R-N3 -- spec docs/superpowers/specs/
2026-07-29-block-floating-point-design.md).

Two forward fixtures, both on shape [2, 4] with numNormDims = 1 (G = 2 blocks
of N = 4):

  LN-A "native": input, gamma and beta are all BFP-STORED, so the funnel
    borrows their exponents zero-copy and the kernel folds on the fixture's
    own grid. The input's four stored exponents are all DIFFERENT (125, 127,
    128, 124) and its {numGroups=4, groupSize=2} blocking cuts each norm group
    in half, so the mean's segment-fold crosses a group boundary INSIDE every
    reduction -- a per-tensor collapse, a dropped fold or a wrong group index
    all move the numbers. gamma is grouped {2, 2} (exercises bfpGroupOf on the
    PARAM axis, where the kernel indexes with the inner index j, not the
    input's physical offset) and beta is per-tensor {1, 0} (the other grid
    shape in the same fixture).

  LN-B "staged": the same geometry with FLOAT32-stored operands. The funnel
    stages each of them per-tensor at the ANCHOR widths -- the layer's own
    produced-wire config (outputQ), m = 6 here, NOT the operand's own width
    and not a hardcoded 8 -- and only then runs the same kernel. The
    width-discrimination self-check below proves the fixture can tell those
    apart.

Every scalar step is mirrored in np.float32 (never float64): the C kernels are
bit-exact targets, so an accumulation done in double would emit gold the C can
only approximate. The emulation order is the C order, statement for statement
(meanOverTrailingAxesBfp / varianceBiasedOverTrailingAxesBfp in Reduce.c,
layerNormForwardBfp in LayerNorm.c).

Task 3 (LayerNorm BFP backward) EXTENDS this file: add the backward emulation
plus a `backward_parts()` builder and append it to `parts` in main().

Run via `uv run` (CMake wires this automatically, see CMakeLists.txt)."""
import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "goldgen"))

from sym_gold import (_bfp_group_of, assert_rounding_canary, bfp_quantize_grouped,
                      emit_float_array, emit_float_scalar, emit_int32_array, emit_int32_scalar,
                      emit_uint8_array)


# ---- float32-exact emulation primitives: one helper per C operation so the
# rounding happens at exactly the points the C kernel rounds at. ----


def f32(x):
    return np.float32(x)


def f32_add(a, b):
    return np.float32(np.float32(a) + np.float32(b))


def f32_sub(a, b):
    return np.float32(np.float32(a) - np.float32(b))


def f32_mul(a, b):
    return np.float32(np.float32(a) * np.float32(b))


def f32_div(a, b):
    return np.float32(np.float32(a) / np.float32(b))


def ldexp_f32(mant, e):
    """ldexpf: exact (a float32 scaling by a power of two loses nothing until
    it under/overflows, which no fixture here does)."""
    return np.float32(np.ldexp(np.float32(mant), int(e)))


def sqrt_f32(x):
    """sqrtf is correctly rounded in float32 by IEEE-754, so np.sqrt on a
    float32 matches the C bit-for-bit."""
    return np.float32(np.sqrt(np.float32(x)))


# ---- geometry ----

ROWS = 2  # G: number of norm groups
COLS = 4  # N: elements per norm group
NUM_NORM_DIMS = 1
EPS = np.float32(1e-5)

OUT_MANTISSA_BITS = 8 - 2  # 6: the produced wire is NARROWER than the operands
OUT_EXPONENT_BITS = 8
OUT_NUM_GROUPS = 4
OUT_GROUP_SIZE = 2

# ---- LN-A: all-BFP operands ----
#
# Codes are hand-picked, pairwise distinct and inside [-100, 100] (m = 8 holds
# [-128, 127], so nothing saturates and every code survives the pack/unpack
# round trip). Together with the exponents below every dequant is a small
# dyadic rational -- 3, -8.75, 48, 7 / -120, 50, 11.25, -1.75 -- so the whole
# stats chain is EXACT in float32. That exactness is what the fake-quant twin
# test rests on: the BFP segment-fold mean and the FLOAT32 sequential-sum mean
# must agree bit-for-bit, which the `parity` self-check below verifies here.
LN_A_X_CODES = [12, -35, 48, 7, -60, 25, 90, -14]
LN_A_X_EXPS = [125, 127, 128, 124]
LN_A_X_QC = {"mantissa_bits": 8, "exponent_bits": 8, "group_size": 2}

LN_A_GAMMA_CODES = [64, 80, -48, 96]
LN_A_GAMMA_EXPS = [126, 127]
LN_A_GAMMA_QC = {"mantissa_bits": 8, "exponent_bits": 8, "group_size": 2}

LN_A_BETA_CODES = [16, -8, 24, 4]
LN_A_BETA_EXPS = [126]
LN_A_BETA_QC = {"mantissa_bits": 8, "exponent_bits": 8, "group_size": 0}

# ---- LN-B: FLOAT32 operands, staged by the funnel ----
#
# None of these values sits on the m = 6 grid the anchor stages at, so the
# staging rounding is load-bearing (a wrong staging width lands on a different
# code, which the width-discrimination self-check pins). The input's magnitudes
# differ by ~30x across the two rows so the two norm groups have genuinely
# different means and sigmas.
LN_B_X_VALS = [1.3, -2.7, 0.45, 3.9, -12.5, 7.25, 0.8, -5.1]
LN_B_GAMMA_VALS = [1.5, -0.75, 2.25, 0.5]
LN_B_BETA_VALS = [0.1, -0.2, 0.3, -0.4]
# The anchor is the PRODUCED WIRE's config (R-N1), not the operands'.
STAGE_MANTISSA_BITS = OUT_MANTISSA_BITS
STAGE_EXPONENT_BITS = OUT_EXPONENT_BITS


def phys_offset(g, j, n):
    """layerNormPhysOffset for a default-order rank-2 tensor with
    numNormDims = 1: the logical index (g, j) is the physical offset g*N + j.
    Both fixtures use that layout (transposed layouts are covered by the
    FLOAT32/SYM divergent-layout tests, which share the same offset helper)."""
    return g * n + j


def bfp_bias(qc):
    return 2 ** (qc["exponent_bits"] - 1) - 1


def deq(code, exps, qc, idx):
    """dequantChunkToFloat's BFP arm: code * 2^(stored - bias)."""
    return ldexp_f32(code, exps[_bfp_group_of(idx, qc["group_size"])] - bfp_bias(qc))


def block_mean_bfp(codes, exps, qc, g_count, n):
    """meanOverTrailingAxesBfp (Reduce.c): ONE int32 partial per same-exponent
    segment, folded through ldexpf into a float32 accumulator on every group
    crossing plus the tail, then divided by N. Python ints are exact, which is
    what the C's int32 partial is (guarded by bfpValidateSumHeadroom) -- the
    assert below stands in for that guard."""
    bias = bfp_bias(qc)
    limit = 2 ** 31 - 1
    means = []
    for g in range(g_count):
        acc = f32(0.0)
        partial = 0
        current_group = 0
        for j in range(n):
            off = phys_offset(g, j, n)
            grp = _bfp_group_of(off, qc["group_size"])
            if j == 0:
                current_group = grp
            elif grp != current_group:
                acc = f32_add(acc, ldexp_f32(float(partial), exps[current_group] - bias))
                partial = 0
                current_group = grp
            partial += codes[off]
            assert abs(partial) <= limit, (
                f"block_mean_bfp: segment partial {partial} leaves int32 -- the C kernel "
                "guarantees this bound via bfpValidateSumHeadroom, never via int64")
        acc = f32_add(acc, ldexp_f32(float(partial), exps[current_group] - bias))
        means.append(f32_div(acc, f32(n)))
    return means


def block_mean_float32(codes, exps, qc, g_count, n):
    """meanOverTrailingAxesFloat32 on the EXACT dequants of the same codes --
    the fake-quant twin's other half. Not emitted; used by the parity
    self-check that licenses testLayerNormForwardBfpFakeQuantPinTwin."""
    means = []
    for g in range(g_count):
        acc = f32(0.0)
        for j in range(n):
            off = phys_offset(g, j, n)
            acc = f32_add(acc, deq(codes[off], exps, qc, off))
        means.append(f32_div(acc, f32(n)))
    return means


def block_var_bfp(codes, exps, qc, means, g_count, n):
    """varianceBiasedOverTrailingAxesBfp: per-element exact dequant, center
    against the float32 mean, accumulate the square in float32, divide by N
    (BIASED)."""
    variances = []
    for g in range(g_count):
        acc = f32(0.0)
        for j in range(n):
            off = phys_offset(g, j, n)
            d = f32_sub(deq(codes[off], exps, qc, off), means[g])
            acc = f32_add(acc, f32_mul(d, d))
        variances.append(f32_div(acc, f32(n)))
    return variances


def forward_bfp(x_codes, x_exps, x_qc, gamma_codes, gamma_exps, gamma_qc, beta_codes, beta_exps,
                beta_qc, g_count, n):
    """layerNormForwardBfp: stats through the Reduce BFP arms, then
    nval = (x - mean) * invSigma and y = gamma*nval + beta in float32. Returns
    the RAW float32 output (D7) -- the OUT_WRITE epilogue packs it."""
    means = block_mean_bfp(x_codes, x_exps, x_qc, g_count, n)
    variances = block_var_bfp(x_codes, x_exps, x_qc, means, g_count, n)
    inv_sigma = [f32_div(f32(1.0), sqrt_f32(f32_add(v, EPS))) for v in variances]
    y = [f32(0.0)] * (g_count * n)
    for g in range(g_count):
        for j in range(n):
            off = phys_offset(g, j, n)
            nval = f32_mul(f32_sub(deq(x_codes[off], x_exps, x_qc, off), means[g]), inv_sigma[g])
            gv = deq(gamma_codes[j], gamma_exps, gamma_qc, j)
            bv = deq(beta_codes[j], beta_exps, beta_qc, j)
            y[off] = f32_add(f32_mul(gv, nval), bv)
    return y, means, variances, inv_sigma


def pack_output(y):
    """The OUT_WRITE epilogue: conversionMatrix[FLOAT32][BFP] re-derives the
    TARGET's per-group exponents and rounds with the OP's mode (HALF_AWAY for
    every forward fixture here -- SR is exercised in C, never in gold)."""
    return bfp_quantize_grouped([float(v) for v in y], OUT_MANTISSA_BITS, OUT_EXPONENT_BITS,
                                OUT_GROUP_SIZE)


def stage_per_tensor(vals, mantissa_bits):
    """quantizeFloatBufferToBfpCodes on the funnel's {1, 0} staging template at
    the anchor widths."""
    codes, exps = bfp_quantize_grouped(vals, mantissa_bits, STAGE_EXPONENT_BITS, 0)
    return codes, exps, {"mantissa_bits": mantissa_bits, "exponent_bits": STAGE_EXPONENT_BITS,
                         "group_size": 0}


def reference_layer_norm(x_deq, gamma_deq, beta_deq):
    """torch.nn.functional.layer_norm in float64 on the exact dequants -- an
    INDEPENDENT oracle for the emulation (it shares no code with it)."""
    x = torch.tensor(x_deq, dtype=torch.float64).reshape(ROWS, COLS)
    w = torch.tensor(gamma_deq, dtype=torch.float64)
    b = torch.tensor(beta_deq, dtype=torch.float64)
    return torch.nn.functional.layer_norm(x, (COLS,), weight=w, bias=b,
                                          eps=float(EPS)).flatten().tolist()


def max_group_scale(exps, exponent_bits):
    bias = 2 ** (exponent_bits - 1) - 1
    return max(math.ldexp(1.0, int(e) - bias) for e in exps)


def check_non_uniform(name, values):
    assert len(set(int(v) for v in values)) > 1, (
        f"{name}: all entries equal -- the fixture cannot distinguish a per-group walk "
        "from an exponent collapse")


def forward_parts():
    """Header body for the two forward fixtures. Task 3 appends a
    backward_parts() next to this."""
    # ---- LN-A ----
    check_non_uniform("LN-A input exponents", LN_A_X_EXPS)
    check_non_uniform("LN-A gamma exponents", LN_A_GAMMA_EXPS)
    assert len(set(LN_A_X_CODES)) == len(LN_A_X_CODES), "LN-A input codes must be pairwise distinct"
    assert all(abs(c) <= 100 for c in LN_A_X_CODES), "LN-A input codes must stay inside [-100, 100]"

    a_y, a_means, _, _ = forward_bfp(LN_A_X_CODES, LN_A_X_EXPS, LN_A_X_QC, LN_A_GAMMA_CODES,
                                     LN_A_GAMMA_EXPS, LN_A_GAMMA_QC, LN_A_BETA_CODES,
                                     LN_A_BETA_EXPS, LN_A_BETA_QC, ROWS, COLS)
    a_out_codes, a_out_exps = pack_output(a_y)

    # (i) The fake-quant twin's licence: on THIS fixture the segment-fold mean
    # and the plain float32 sum of the same dequants agree bit-for-bit, so the
    # native and fake-quant forwards must produce identical wires.
    a_means_float = block_mean_float32(LN_A_X_CODES, LN_A_X_EXPS, LN_A_X_QC, ROWS, COLS)
    for g, (mb, mf) in enumerate(zip(a_means, a_means_float)):
        assert mb == mf, (
            f"LN-A group {g}: BFP segment-fold mean {mb} != FLOAT32 sequential-sum mean {mf} -- "
            "the fixture left the exact regime, so the fake-quant twin test cannot assert "
            "bit-equality; pick codes/exponents whose partial sums are exact")

    # (ii) Independent oracle: the packed output, dequantized, must sit within
    # half a pack step of PyTorch's float64 layer_norm on the same dequants.
    a_x_deq = [float(deq(c, LN_A_X_EXPS, LN_A_X_QC, i)) for i, c in enumerate(LN_A_X_CODES)]
    a_gamma_deq = [float(deq(c, LN_A_GAMMA_EXPS, LN_A_GAMMA_QC, i))
                   for i, c in enumerate(LN_A_GAMMA_CODES)]
    a_beta_deq = [float(deq(c, LN_A_BETA_EXPS, LN_A_BETA_QC, i))
                  for i, c in enumerate(LN_A_BETA_CODES)]
    a_ref = reference_layer_norm(a_x_deq, a_gamma_deq, a_beta_deq)
    a_out_qc = {"mantissa_bits": OUT_MANTISSA_BITS, "exponent_bits": OUT_EXPONENT_BITS,
                "group_size": OUT_GROUP_SIZE}
    a_out_deq = [float(deq(c, a_out_exps, a_out_qc, i)) for i, c in enumerate(a_out_codes)]
    bound = 0.5 * max_group_scale(a_out_exps, OUT_EXPONENT_BITS) + 1e-3
    for i, (got, want) in enumerate(zip(a_out_deq, a_ref)):
        assert abs(got - want) <= bound, (
            f"LN-A element {i}: packed output dequantizes to {got}, PyTorch layer_norm says "
            f"{want} (bound {bound}) -- the emulation and the definition disagree")

    # (iii) The grouped input must MATTER: collapsing its four exponents onto
    # one value has to move the wire, or a per-tensor read of the input grid
    # would pass this fixture.
    collapsed_exps = [LN_A_X_EXPS[0]] * len(LN_A_X_EXPS)
    c_y, _, _, _ = forward_bfp(LN_A_X_CODES, collapsed_exps, LN_A_X_QC, LN_A_GAMMA_CODES,
                               LN_A_GAMMA_EXPS, LN_A_GAMMA_QC, LN_A_BETA_CODES, LN_A_BETA_EXPS,
                               LN_A_BETA_QC, ROWS, COLS)
    c_out_codes, c_out_exps = pack_output(c_y)
    assert (c_out_codes, c_out_exps) != (a_out_codes, a_out_exps), (
        "LN-A: forcing every input exponent to one value leaves the output unchanged -- the "
        "fixture cannot tell a grouped input walk from a per-tensor collapse")

    # ---- LN-B ----
    b_x_vals = torch.tensor(LN_B_X_VALS, dtype=torch.float32)
    b_gamma_vals = torch.tensor(LN_B_GAMMA_VALS, dtype=torch.float32)
    b_beta_vals = torch.tensor(LN_B_BETA_VALS, dtype=torch.float32)

    def staged_forward(mantissa_bits):
        xc, xe, xq = stage_per_tensor(b_x_vals.tolist(), mantissa_bits)
        gc, ge, gq = stage_per_tensor(b_gamma_vals.tolist(), mantissa_bits)
        bc, be, bq = stage_per_tensor(b_beta_vals.tolist(), mantissa_bits)
        y, _, _, _ = forward_bfp(xc, xe, xq, gc, ge, gq, bc, be, bq, ROWS, COLS)
        return pack_output(y)

    b_out_codes, b_out_exps = staged_forward(STAGE_MANTISSA_BITS)

    # (iv) Width discrimination: staging at the operands' own width (8) instead
    # of the anchor's (6) must change the wire, or the test could not tell a
    # hardcoded-8 staging from the anchored one.
    b_wide_codes, b_wide_exps = staged_forward(8)
    assert (b_wide_codes, b_wide_exps) != (b_out_codes, b_out_exps), (
        "LN-B: staging at m=8 produces the same wire as staging at the anchor's m=6 -- the "
        "fixture cannot pin the anchor widths; pick values further off the m=6 grid")

    zero_codes = [0] * (ROWS * COLS)
    zero_exps = [2 ** (OUT_EXPONENT_BITS - 1) - 1] * OUT_NUM_GROUPS

    return [
        "/* GENERATED by generate_expected_bfp_layernorm.py -- do not edit. */\n",
        "#ifndef ODT_EXPECTED_BFP_LAYERNORM_H\n#define ODT_EXPECTED_BFP_LAYERNORM_H\n\n",
        "#include <stddef.h>\n#include <stdint.h>\n\n",
        "/* shared geometry: [2, 4], numNormDims = 1 -> G = 2, N = 4 */\n",
        emit_int32_scalar("kLnBfpRows", ROWS),
        emit_int32_scalar("kLnBfpCols", COLS),
        emit_int32_scalar("kLnBfpNumNormDims", NUM_NORM_DIMS),
        emit_float_scalar("kLnBfpEps", float(EPS)),
        "\n/* LN-A: BFP-stored input (grouped {4, 2}, four distinct exponents),\n"
        " * BFP grouped gamma {2, 2}, BFP per-tensor beta {1, 0}. */\n",
        emit_int32_scalar("kLnBfpAXMantissaBits", LN_A_X_QC["mantissa_bits"]),
        emit_int32_scalar("kLnBfpAXExponentBits", LN_A_X_QC["exponent_bits"]),
        emit_int32_scalar("kLnBfpAXNumGroups", len(LN_A_X_EXPS)),
        emit_int32_scalar("kLnBfpAXGroupSize", LN_A_X_QC["group_size"]),
        emit_int32_array("kLnBfpAXCodes", torch.tensor(LN_A_X_CODES)),
        emit_uint8_array("kLnBfpAXExponents", LN_A_X_EXPS),
        emit_int32_scalar("kLnBfpAGammaMantissaBits", LN_A_GAMMA_QC["mantissa_bits"]),
        emit_int32_scalar("kLnBfpAGammaExponentBits", LN_A_GAMMA_QC["exponent_bits"]),
        emit_int32_scalar("kLnBfpAGammaNumGroups", len(LN_A_GAMMA_EXPS)),
        emit_int32_scalar("kLnBfpAGammaGroupSize", LN_A_GAMMA_QC["group_size"]),
        emit_int32_array("kLnBfpAGammaCodes", torch.tensor(LN_A_GAMMA_CODES)),
        emit_uint8_array("kLnBfpAGammaExponents", LN_A_GAMMA_EXPS),
        emit_int32_scalar("kLnBfpABetaMantissaBits", LN_A_BETA_QC["mantissa_bits"]),
        emit_int32_scalar("kLnBfpABetaExponentBits", LN_A_BETA_QC["exponent_bits"]),
        emit_int32_scalar("kLnBfpABetaNumGroups", len(LN_A_BETA_EXPS)),
        emit_int32_scalar("kLnBfpABetaGroupSize", LN_A_BETA_QC["group_size"]),
        emit_int32_array("kLnBfpABetaCodes", torch.tensor(LN_A_BETA_CODES)),
        emit_uint8_array("kLnBfpABetaExponents", LN_A_BETA_EXPS),
        "\n/* produced wire (both fixtures): narrower than the operands, grouped {4, 2} */\n",
        emit_int32_scalar("kLnBfpAOutMantissaBits", OUT_MANTISSA_BITS),
        emit_int32_scalar("kLnBfpAOutExponentBits", OUT_EXPONENT_BITS),
        emit_int32_scalar("kLnBfpAOutNumGroups", OUT_NUM_GROUPS),
        emit_int32_scalar("kLnBfpAOutGroupSize", OUT_GROUP_SIZE),
        emit_int32_array("kLnBfpAOutCodes", torch.tensor(a_out_codes)),
        emit_uint8_array("kLnBfpAOutExponents", a_out_exps),
        "/* canonical zero state for a freshly seeded output wire */\n",
        emit_int32_array("kLnBfpAOutZeroCodes", torch.tensor(zero_codes)),
        emit_uint8_array("kLnBfpAOutZeroExponents", zero_exps),
        "\n/* LN-B: FLOAT32-stored operands; the funnel stages each per-tensor at\n"
        " * the anchor (produced-wire) widths m = 6, e = 8. */\n",
        emit_float_array("kLnBfpBXValues", b_x_vals),
        emit_float_array("kLnBfpBGammaValues", b_gamma_vals),
        emit_float_array("kLnBfpBBetaValues", b_beta_vals),
        emit_int32_array("kLnBfpBOutCodes", torch.tensor(b_out_codes)),
        emit_uint8_array("kLnBfpBOutExponents", b_out_exps),
        "\n#endif /* ODT_EXPECTED_BFP_LAYERNORM_H */\n",
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    assert_rounding_canary()

    parts = forward_parts()
    Path(args.out).write_text("".join(parts))


if __name__ == "__main__":
    main()
