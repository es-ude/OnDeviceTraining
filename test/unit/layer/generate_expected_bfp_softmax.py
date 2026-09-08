#!/usr/bin/env python3
"""Generate expected_bfp_softmax.h for UnitTestSoftmax's native ARITH_BFP
forward tests (BFP epic PR6 Task 4 -- normative pipeline:
.superpowers/sdd/2026-09-08-bfp-pr6-softmax/numerics-spec.md, steps 1-5).

Two forward fixtures, both n = 8 (whole-tensor softmax, microbatch B=1):

  SM-A "native": the input is BFP-STORED, grouped {numGroups=2, groupSize=4},
    m = 8 / e = 8, with DIFFERENT stored exponents per block (122 -> E=-5
    around +-3.0, 118 -> E=-9 around +-0.1), so block B's alignment shift is
    (EMax - E_i) = 4 bits and codes 44 (2.75) and -33 (-2.0625) leave nonzero
    remainders on which TRUNC and HALF_AWAY genuinely diverge. Gold is
    emitted for BOTH knob positions (the knob-discrimination vacuity check
    below asserts the two packed wires differ). The argmax element (code 96,
    block A) sits in the max-exponent block, so the alignment invariant
    E_i <= EMax holds as the spec requires.

  SM-B "staged": the SAME logits as FLOAT32 input values; the funnel stages
    them per-tensor at the ANCHOR widths -- the layer's own produced-wire
    config (outputQ), m = 6 here, NOT the operand's width and not a
    hardcoded 8 (the width-discrimination self-check pins that). Knob stays
    the factory default TRUNC.

Every integer step runs in exact Python ints (the C kernel's int32 bounds are
asserted, never widened); every float step is mirrored in np.float32 with the
C's exact op order: ldexpf max-compare (strict >, first max wins), (float)qe *
0.3585f then ldexpf(.., -28), sum accumulated in index order, per-element
divide. Pack/staging emulation comes from sym_gold's bfp_quantize_grouped
(HALF_AWAY -- the fixtures' storage-derived op rounding); the small
dequant/f32 helpers are copied per script from generate_expected_bfp_layernorm
per the repo goldgen convention (no cross-imports between generators).

Run via `uv run` (CMake wires this automatically, see CMakeLists.txt)."""
import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "goldgen"))

from sym_gold import (_bfp_group_of, assert_rounding_canary, bfp_quantize_grouped,
                      emit_float_array, emit_int32_array, emit_int32_scalar, emit_uint8_array)

# ---- Task 3 constants (BfpSoftmaxExp.h). QLN2/QB/QC/QFLOOR are RE-DERIVED
# below (main) and the script aborts on mismatch -- the numerics spec forbids
# trusting the literals. ----

F = 14
QLN2 = 11356
QB = 22167
QC = 257578233
A_OUT = np.float32(0.3585)
ZMAX = 31
QFLOOR = -(ZMAX + 1) * QLN2  # -363392
INT32_MIN, INT32_MAX = -2 ** 31, 2 ** 31 - 1


# ---- float32-exact emulation primitives (copied per repo goldgen convention
# from generate_expected_bfp_layernorm.py): one helper per C operation so the
# rounding happens exactly where the C kernel rounds. ----


def f32(x):
    return np.float32(x)


def f32_add(a, b):
    return np.float32(np.float32(a) + np.float32(b))


def f32_mul(a, b):
    return np.float32(np.float32(a) * np.float32(b))


def f32_div(a, b):
    return np.float32(np.float32(a) / np.float32(b))


def ldexp_f32(mant, e):
    """ldexpf: exact (a float32 scaling by a power of two loses nothing until
    it under/overflows, which no fixture here does)."""
    return np.float32(np.ldexp(np.float32(mant), int(e)))


def bfp_bias(qc):
    return 2 ** (qc["exponent_bits"] - 1) - 1


def deq(code, exps, qc, idx):
    """dequantChunkToFloat's BFP arm: code * 2^(stored - bias)."""
    return ldexp_f32(code, exps[_bfp_group_of(idx, qc["group_size"])] - bfp_bias(qc))


# ---- integer pipeline emulation (BfpSoftmaxExp.c + softmaxValuesBfp),
# statement for statement. Deterministic modes only: SR needs the C RNG
# stream and is exercised in C, never in gold. ----


def shift_right_rounded(v, k, mode):
    """bfpShiftRightRounded: k==0 identity, k>31 clamped; TRUNC is Python's
    floor >> (== C arithmetic shift); HALF_AWAY via the magnitude add."""
    assert mode in ("trunc", "half_away")
    if k == 0:
        return v
    if k > 31:
        k = 31
    if mode == "trunc":
        return v >> k
    mag = -v if v < 0 else v
    assert mag <= 2 ** 30, f"HALF_AWAY magnitude {mag} breaks the 2^30 precondition"
    r = (mag + (1 << (k - 1))) >> k
    return -r if v < 0 else r


def iexp_q(qw, mode):
    """bfpIExpQ: I-BERT Algorithm 3 on the fixed 2^-F grid."""
    assert qw <= 0, f"iexp_q: qW {qw} > 0 reached the core (the kernel clamp failed)"
    if qw <= QFLOOR:
        return 0
    z = (-qw) // QLN2  # non-negative operands: Python floor == C division
    qp = qw + z * QLN2
    ql = (qp + QB) * (qp + QB) + QC
    assert ql <= INT32_MAX, f"iexp_q: qL {ql} leaves int32 (headroom proof violated)"
    return shift_right_rounded(ql, z, mode)


def softmax_values_bfp(codes, exps, qc, mode):
    """softmaxValuesBfp (Softmax.c): numerics-spec steps 1-5 on unpacked
    mantissa codes with a live grid. Returns the RAW float32 outputs (D7) --
    the OUT_WRITE epilogue packs them."""
    n = len(codes)
    bias = bfp_bias(qc)
    E = [exps[_bfp_group_of(i, qc["group_size"])] - bias for i in range(n)]

    # (1) Max: ldexpf compare, strict > keeps the FIRST max.
    m_max, e_max = codes[0], E[0]
    x_max = ldexp_f32(codes[0], E[0])
    for i in range(1, n):
        xi = ldexp_f32(codes[i], E[i])
        if xi > x_max:
            x_max, m_max, e_max = xi, codes[i], E[i]

    # Fixture validation (NOT a kernel mirror -- the C does not assert this):
    # the alignment invariant the spec proves for absmax-minimal grids. A
    # fixture violating it would make this emulation silently diverge from
    # the C's uint32-wrap/clamp behavior, so abort loudly instead.
    for i in range(n):
        assert e_max - E[i] >= 0, (
            f"fixture breaks the alignment invariant at element {i}: "
            f"EMax={e_max} < E_i={E[i]} -- pick the argmax in the max-exponent block")

    # (2) Align onto the work-or-coarser grid: ONE rounded shift per element.
    sigma = e_max + F
    down = -sigma if sigma < 0 else 0
    m_max_w = shift_right_rounded(m_max, down, mode)
    s = []
    total = f32(0.0)
    for i in range(n):
        aligned = shift_right_rounded(codes[i], (e_max - E[i]) + down, mode)
        qt = aligned - m_max_w
        if qt > 0:  # min(qT, 0): load-bearing under SR, no-op for these modes
            qt = 0
        # (3) Work-grid promotion (exact).
        if sigma >= 31:
            qw = QFLOOR if qt < 0 else 0
        elif sigma >= 0:
            thr = -((363392 + (1 << sigma) - 1) >> sigma)
            qw = QFLOOR if qt <= thr else qt << sigma  # Python << is exact; C uses
            # the guarded two's-complement image, value-identical inside int32
        else:
            qw = qt
        assert INT32_MIN <= qw <= INT32_MAX, f"qW {qw} leaves int32"
        # (4) Integer core; (5) float boundary.
        qe = iexp_q(qw, mode)
        e = ldexp_f32(f32_mul(f32(qe), A_OUT), -28)
        s.append(e)
        total = f32_add(total, e)
    return [f32_div(v, total) for v in s]


def softmax_float64(vals):
    """Independent proximity oracle (float64, shares nothing with the
    emulation): softmax of the exact dequants."""
    x = np.asarray(vals, dtype=np.float64)
    e = np.exp(x - x.max())
    return (e / e.sum()).tolist()


# ---- geometry + fixtures ----

N = 8

SM_X_CODES = [96, -50, 40, 77, 51, -33, 44, 100]
SM_X_EXPS = [122, 118]  # E = -5 (block A, ~+-3.0), E = -9 (block B, ~+-0.1)
SM_X_QC = {"mantissa_bits": 8, "exponent_bits": 8, "group_size": 4}

OUT_QC = {"mantissa_bits": 8, "exponent_bits": 8, "group_size": 4}
OUT_NUM_GROUPS = 2

STAGED_OUT_QC = {"mantissa_bits": 6, "exponent_bits": 8, "group_size": 4}


def pack(raw, qc):
    """The OUT_WRITE epilogue: conversionMatrix[FLOAT32][BFP] re-derives the
    TARGET's per-group exponents and rounds with the OP's mode (HALF_AWAY for
    every fixture here -- SR is exercised in C, never in gold)."""
    return bfp_quantize_grouped([float(v) for v in raw], qc["mantissa_bits"],
                                qc["exponent_bits"], qc["group_size"])


def max_group_scale(exps, exponent_bits):
    bias = 2 ** (exponent_bits - 1) - 1
    return max(math.ldexp(1.0, int(e) - bias) for e in exps)


def check_proximity(name, out_codes, out_exps, out_qc, ref, i_exp_bound=2.5e-3):
    """Value-level acceptance: dequant(packed out) within the i-exp accuracy
    bound (2.5e-3, numerics spec) + half a pack step of the float softmax,
    and sum(dequant(out)) ~= 1 within 3e-3 + the summed half-pack-steps (the
    m=6 SM-B wire's pack rounding alone exceeds a bare 3e-3; each element
    contributes up to half ITS group's scale to the sum)."""
    bias = 2 ** (out_qc["exponent_bits"] - 1) - 1
    out_deq = [float(deq(c, out_exps, out_qc, i)) for i, c in enumerate(out_codes)]
    bound = i_exp_bound + 0.5 * max_group_scale(out_exps, out_qc["exponent_bits"])
    for i, (got, want) in enumerate(zip(out_deq, ref)):
        assert abs(got - want) <= bound, (
            f"{name} element {i}: packed output dequantizes to {got}, float softmax says "
            f"{want} (bound {bound}) -- the emulation and the definition disagree")
    total = sum(out_deq)
    half_steps = sum(
        0.5 * math.ldexp(1.0, int(out_exps[_bfp_group_of(i, out_qc["group_size"])]) - bias)
        for i in range(len(out_codes)))
    assert abs(total - 1.0) <= 3e-3 + half_steps, (
        f"{name}: dequantized outputs sum to {total}, not 1 (+-{3e-3 + half_steps})")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    assert_rounding_canary()

    # -- constants re-derivation (numerics spec: never trust the literals) --
    assert QLN2 == math.floor(math.log(2) * 2 ** F), "QLN2 literal wrong"
    assert QB == math.floor(1.353 * 2 ** F), "QB literal wrong"
    assert QC == math.floor(0.344 * 2 ** 28 / 0.3585), "QC literal wrong"
    assert QFLOOR == -363392, "QFLOOR literal wrong"
    # Reference check from the spec: qW = 0 -> qL = QB^2 + QC.
    assert iexp_q(0, "trunc") == QB * QB + QC == 748954122

    # -- SM-A fixture validation --
    assert len(SM_X_CODES) == N and len(SM_X_EXPS) == OUT_NUM_GROUPS
    assert all(-128 <= c <= 127 for c in SM_X_CODES), "SM-A codes must fit m=8"
    assert len(set(SM_X_EXPS)) > 1, "SM-A blocks must have DIFFERENT exponents"
    # At least one alignment shift must carry a remainder on which TRUNC and
    # HALF_AWAY diverge (the knob's lossy site #1).
    shift = SM_X_EXPS[0] - SM_X_EXPS[1]
    assert shift > 0
    diverging = [c for c in SM_X_CODES[4:]
                 if shift_right_rounded(c, shift, "trunc")
                 != shift_right_rounded(c, shift, "half_away")]
    assert diverging, ("SM-A: no block-B code diverges between TRUNC and HALF_AWAY "
                       "on the alignment shift -- the knob tests would be vacuous")

    # -- SM-A gold: both knob positions --
    a_raw_trunc = softmax_values_bfp(SM_X_CODES, SM_X_EXPS, SM_X_QC, "trunc")
    a_raw_ha = softmax_values_bfp(SM_X_CODES, SM_X_EXPS, SM_X_QC, "half_away")
    a_codes_trunc, a_exps_trunc = pack(a_raw_trunc, OUT_QC)
    a_codes_ha, a_exps_ha = pack(a_raw_ha, OUT_QC)

    # (i) Knob discrimination: the packed wires MUST differ.
    assert (a_codes_trunc, a_exps_trunc) != (a_codes_ha, a_exps_ha), (
        "SM-A: TRUNC and HALF_AWAY produce the same packed wire -- the knob tests "
        "cannot discriminate; pick codes with larger alignment remainders")

    # (ii) RED guard / native-vs-fake divergence: the fake-quant path (float
    # softmax of the exact dequants, packed the same way) must NOT collide
    # with the native TRUNC wire, or the pre-implementation test run could
    # pass by accident.
    x_deq = [float(deq(c, SM_X_EXPS, SM_X_QC, i)) for i, c in enumerate(SM_X_CODES)]
    fake = softmax_float64(x_deq)
    fake_codes, fake_exps = pack([np.float32(v) for v in fake], OUT_QC)
    assert (fake_codes, fake_exps) != (a_codes_trunc, a_exps_trunc), (
        "SM-A: the native TRUNC wire equals the fake-quant float wire -- the native "
        "tests cannot discriminate the new kernel from the old float path")

    # (iii) Value-level acceptance for both knobs.
    check_proximity("SM-A TRUNC", a_codes_trunc, a_exps_trunc, OUT_QC, fake)
    check_proximity("SM-A HALF_AWAY", a_codes_ha, a_exps_ha, OUT_QC, fake)

    # -- SM-B gold: staged FLOAT32 input at the m=6 anchor, knob TRUNC --
    b_vals = [np.float32(v) for v in x_deq]
    for v, orig in zip(b_vals, x_deq):
        assert float(v) == orig, "SM-B float fixture must be exact (dyadic logits)"

    def staged_forward(mantissa_bits):
        """The funnel's {1, 0} staging template at the given widths (HALF_AWAY
        -- the OP's storage-derived rounding), then the TRUNC pipeline and the
        outputQ pack. Returns (packed (codes, exps), staged codes, exps, qc)."""
        stage_qc = {"mantissa_bits": mantissa_bits,
                    "exponent_bits": STAGED_OUT_QC["exponent_bits"], "group_size": 0}
        codes, exps = bfp_quantize_grouped([float(v) for v in b_vals], mantissa_bits,
                                           stage_qc["exponent_bits"], 0)
        raw = softmax_values_bfp(codes, exps, stage_qc, "trunc")
        return pack(raw, STAGED_OUT_QC), codes, exps, stage_qc

    b_out, b_staged_codes, b_staged_exps, b_stage_qc = staged_forward(
        STAGED_OUT_QC["mantissa_bits"])
    b_codes, b_exps = b_out

    # (iv) Width discrimination: staging at the operand's own width (8)
    # instead of the anchor's (6) must change the wire.
    b_wide_out = staged_forward(8)[0]
    assert b_wide_out != (b_codes, b_exps), (
        "SM-B: staging at m=8 produces the same wire as the m=6 anchor -- the fixture "
        "cannot pin the anchor width")

    # (v) Staged proximity: against the float softmax of the STAGED values
    # (staging loss is not the i-exp core's error budget).
    staged_deq = [float(deq(c, b_staged_exps, b_stage_qc, i))
                  for i, c in enumerate(b_staged_codes)]
    check_proximity("SM-B", b_codes, b_exps, STAGED_OUT_QC, softmax_float64(staged_deq))

    zero_codes = [0] * N
    zero_exps = [2 ** (OUT_QC["exponent_bits"] - 1) - 1] * OUT_NUM_GROUPS

    parts = [
        "/* GENERATED by generate_expected_bfp_softmax.py -- do not edit. */\n",
        "#ifndef ODT_EXPECTED_BFP_SOFTMAX_H\n#define ODT_EXPECTED_BFP_SOFTMAX_H\n\n",
        "#include <stddef.h>\n#include <stdint.h>\n\n",
        "/* SM-A: BFP-stored input, grouped {2, 4}, m=8/e=8, two DIFFERENT block\n"
        " * exponents (E=-5 / E=-9) -- the 4-bit alignment shift carries nonzero\n"
        " * remainders, so the TRUNC and HALF_AWAY wires differ (script-asserted). */\n",
        emit_int32_scalar("kSmBfpN", N),
        emit_int32_scalar("kSmBfpXMantissaBits", SM_X_QC["mantissa_bits"]),
        emit_int32_scalar("kSmBfpXExponentBits", SM_X_QC["exponent_bits"]),
        emit_int32_scalar("kSmBfpXNumGroups", len(SM_X_EXPS)),
        emit_int32_scalar("kSmBfpXGroupSize", SM_X_QC["group_size"]),
        emit_int32_array("kSmBfpXCodes", torch.tensor(SM_X_CODES)),
        emit_uint8_array("kSmBfpXExponents", SM_X_EXPS),
        "\n/* produced wire (SM-A): m=8/e=8, grouped {2, 4}; HALF_AWAY pack (the\n"
        " * OP's storage-derived rounding) for BOTH knob positions -- the knob\n"
        " * governs only the kernel-internal integer shifts. */\n",
        emit_int32_scalar("kSmBfpOutMantissaBits", OUT_QC["mantissa_bits"]),
        emit_int32_scalar("kSmBfpOutExponentBits", OUT_QC["exponent_bits"]),
        emit_int32_scalar("kSmBfpOutNumGroups", OUT_NUM_GROUPS),
        emit_int32_scalar("kSmBfpOutGroupSize", OUT_QC["group_size"]),
        emit_int32_array("kSmBfpOutCodesTrunc", torch.tensor(a_codes_trunc)),
        emit_uint8_array("kSmBfpOutExponentsTrunc", a_exps_trunc),
        emit_int32_array("kSmBfpOutCodesHalfAway", torch.tensor(a_codes_ha)),
        emit_uint8_array("kSmBfpOutExponentsHalfAway", a_exps_ha),
        "/* canonical zero state for a freshly seeded output wire (both fixtures'\n"
        " * wires share {2, 4} / e=8, so one zero set serves both) */\n",
        emit_int32_array("kSmBfpOutZeroCodes", torch.tensor(zero_codes)),
        emit_uint8_array("kSmBfpOutZeroExponents", zero_exps),
        "\n/* SM-B: the SAME logits as FLOAT32 input values; the funnel stages them\n"
        " * per-tensor at the ANCHOR widths (outputQ: m=6/e=8 -- script-asserted\n"
        " * to differ from an m=8 staging). Knob stays the default TRUNC. */\n",
        emit_float_array("kSmBfpBXValues", torch.tensor(b_vals, dtype=torch.float32)),
        emit_int32_scalar("kSmBfpBOutMantissaBits", STAGED_OUT_QC["mantissa_bits"]),
        emit_int32_scalar("kSmBfpBOutExponentBits", STAGED_OUT_QC["exponent_bits"]),
        emit_int32_scalar("kSmBfpBOutNumGroups", OUT_NUM_GROUPS),
        emit_int32_scalar("kSmBfpBOutGroupSize", STAGED_OUT_QC["group_size"]),
        emit_int32_array("kSmBfpBOutCodes", torch.tensor(b_codes)),
        emit_uint8_array("kSmBfpBOutExponents", b_exps),
        "\n#endif /* ODT_EXPECTED_BFP_SOFTMAX_H */\n",
    ]
    Path(args.out).write_text("".join(parts))


if __name__ == "__main__":
    main()
