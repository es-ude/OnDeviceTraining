#!/usr/bin/env python3
"""Generate expected_bfp_softmax.h for UnitTestSoftmax's native ARITH_BFP
forward AND backward tests (BFP epic PR6 Tasks 4+5 -- pipeline contract:
docs/conventions/arithmetic-bfp.md §5.9, R-S2 steps 1-5 and the R-S4
backward paragraph).

Four forward fixtures, all n = 8 (whole-tensor softmax, microbatch B=1):

  SM-A "native": the input is BFP-STORED, grouped {numGroups=2, groupSize=4},
    m = 8 / e = 8, with DIFFERENT stored exponents per block (122 -> E=-5
    around +-3.0, 118 -> E=-9 around +-0.1), so block B's alignment shift is
    (EMax - E_i) = 4 bits and codes 44 (2.75) and -33 (-2.0625) leave nonzero
    remainders on which TRUNC and HALF_AWAY genuinely diverge. Gold is
    emitted for BOTH knob positions (the knob-discrimination vacuity check
    below asserts the two packed wires differ). The argmax element (code 96,
    block A) sits in the max-exponent block, so the alignment invariant
    E_i <= EMax holds as the spec requires.

  SM-CN "coarse negative block" (fix round 1, amended spec step 2): block A
    holds the SIGNED max while the negative-dominated block B legitimately
    carries a COARSER absmax-minimal grid (E_i > EMax) -- its elements take
    the amended exact-left-shift alignment. Pins both regimes: large
    negatives (mass exactly 0, packed code 0) and a small positive logit
    the old clamp regime crushed by tens of percent.

  SM-CS "coarse saturation" (fix round 2): grouped {4, 2}; one coarse block
    per saturation disjunct (magnitude at up=24, up>=31 at up=32), each next
    to a ZERO code -- the up=32 zero pins the kernel's up>=31 clause (its
    0 << 32 is formal UB; the zero elements keep their true nonzero mass).

  SM-B "staged": the SAME logits as FLOAT32 input values; the funnel stages
    them per-tensor at the ANCHOR widths -- the layer's own produced-wire
    config (outputQ), m = 6 here, NOT the operand's width and not a
    hardcoded 8 (the width-discrimination self-check pins that). Knob stays
    the factory default TRUNC.

Two backward fixtures (Task 5 -- recompute s via the forward pipeline from
the LOGITS, exact-dequant dLds, float32 dot in index order, raw = s * (dLds
- dot), OUT_WRITE pack at propLossQ):

  BWD-N "native": x = SM-A's logits (BFP codes/exponents), dLds a SECOND BFP
    wire on a DIFFERENT grid (per-tensor {1, 0}, m=8/e=8) with NON-uniform
    values -- uniform loss grads make the -dot term vacuous (the
    uniform-lossGrad lesson); propLossQ is the grouped {2, 4} m=8/e=8 grid.

  BWD-S "staged loss": same x, dLds as FLOAT32 values the funnel stages
    per-tensor at the propLossQ ANCHOR widths (m=8/e=8, HALF_AWAY -- the
    op's storage-derived rounding). The values are deliberately NOT exactly
    representable on that grid, so the staging rounding shows in the packed
    gold (script-asserted: skipping the staging changes the wire, and this
    gold differs from BWD-N's).

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

    # (2) Align onto the work-or-coarser grid: at most ONE rounded shift per
    # element. The net shift s_i MAY be NEGATIVE (spec amended 2026-09-08,
    # Task-4 fix round 1): EMax follows the SIGNED argmax while block
    # exponents follow the block ABSMAX, so a negative-dominated block is
    # legitimately coarser than the argmax block (E_i > EMax); such elements
    # take an EXACT saturating left shift instead of a rounded right shift.
    sigma = e_max + F
    down = -sigma if sigma < 0 else 0
    m_max_w = shift_right_rounded(m_max, down, mode)
    s = []
    total = f32(0.0)
    for i in range(n):
        si = (e_max - E[i]) + down
        if si >= 0:
            aligned = shift_right_rounded(codes[i], si, mode)
        else:
            up = -si
            if codes[i] < 0 and (up >= 31 or codes[i] < -(INT32_MAX >> up)):
                # Saturating sentinel INT32_MIN/2: the element's true value
                # sits below the exp-underflow floor up to a residual
                # <= exp(-16); the sentinel rides the qT clamp + thr/QFLOOR
                # handling downstream.
                aligned = INT32_MIN // 2
            elif up >= 31:
                # Fix round 2: the only NON-negative code that can reach
                # up >= 31 is m_i == 0 (the m_i >= 1 proof bounds up <= 30),
                # and 0's exact shift is 0 -- the explicit clause exists
                # because 0 << up with up >= 32 is formal UB in C.
                aligned = 0
            else:
                aligned = codes[i] << up  # exact left shift (C: unsigned image)
                assert INT32_MIN <= aligned <= INT32_MAX, (
                    f"aligned {aligned} leaves int32 -- the m_i > 0 no-overflow "
                    "proof was violated (grid corruption in the fixture)")
        qt = aligned - m_max_w
        assert INT32_MIN <= qt <= INT32_MAX, f"qT {qt} leaves int32"
        if qt > 0:  # min(qT, 0): load-bearing under SR, no-op for these modes
            qt = 0
        # (3) Work-grid promotion (exact).
        if sigma >= 31:
            qw = QFLOOR if qt < 0 else 0
        elif sigma >= 0:
            thr = -(((-QFLOOR) + (1 << sigma) - 1) >> sigma)
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


def softmax_backward_bfp(x_codes, x_exps, x_qc, dlds_deq, mode):
    """softmaxBackwardKernelBfp (Softmax.c, Task 5): recompute s via the
    forward pipeline from the LOGITS (the backward differentiates the
    UNpacked forward values -- R-N4), then float32 dot in index order and
    raw_i = s_i * (dLds_i - dot). Returns (raw floats, dot, s) -- the extra
    two feed the mutation-killability self-checks."""
    s = softmax_values_bfp(x_codes, x_exps, x_qc, mode)
    dot = f32(0.0)
    for si, d in zip(s, dlds_deq):
        dot = f32_add(dot, f32_mul(si, d))
    return [f32_mul(si, f32_sub(d, dot)) for si, d in zip(s, dlds_deq)], dot, s


def softmax_backward_autograd(x_deq, dlds_deq):
    """Independent proximity oracle: torch.autograd through a float64 softmax
    of the exact logit dequants, upstream grad dLds (float64)."""
    xt = torch.tensor(x_deq, dtype=torch.float64, requires_grad=True)
    st = torch.softmax(xt, dim=0)
    st.backward(torch.tensor(dlds_deq, dtype=torch.float64))
    return xt.grad.tolist()


# ---- geometry + fixtures ----

N = 8

SM_X_CODES = [96, -50, 40, 77, 51, -33, 44, 100]
SM_X_EXPS = [122, 118]  # E = -5 (block A, ~+-3.0), E = -9 (block B, ~+-0.1)
SM_X_QC = {"mantissa_bits": 8, "exponent_bits": 8, "group_size": 4}

OUT_QC = {"mantissa_bits": 8, "exponent_bits": 8, "group_size": 4}
OUT_NUM_GROUPS = 2

STAGED_OUT_QC = {"mantissa_bits": 6, "exponent_bits": 8, "group_size": 4}

# SM-CN "coarse negative block" (fix round 1): block A {2.0, 0.5, -0.3125,
# 0.125} holds the SIGNED max at E=-5; block B {-100, -80, +1.0, -50} is
# negative-DOMINATED, so its absmax-minimal exponent E=0 is COARSER than the
# argmax block's (E_i > EMax) -- the fixture deliberately violates the old
# (wrong) invariant and exercises BOTH negative-net-shift regimes: large
# negatives whose true mass is exactly 0, and a small POSITIVE logit whose
# mass (~exp(1-2)/sum ~ 0.2) the old clamp regime crushed to ~exp(-x_max).
SM_CN_X_CODES = [64, 16, -10, 4, -100, -80, 1, -50]
SM_CN_X_EXPS = [122, 127]  # E = -5 (argmax block), E = 0 (coarser)
SM_CN_X_QC = dict(SM_X_QC)

# SM-CS "coarse saturation" (fix round 2): grouped {4, 2} so ONE fixture can
# reach every arm of the negative-net-shift branch. Block g0 (E=-5) holds the
# small positive argmax 0.25; g1 (E=19, up=24) carries code -128 -- the
# MAGNITUDE-disjunct saturation (|m| > INT32_MAX >> 24 = 127) -- next to a
# zero code whose 0 << 24 stays a defined shift; g2 (E=27, up=32) carries
# code -128 -- the up>=31-disjunct saturation -- next to a ZERO code that
# needs the new up>=31 clause (0 << 32 is formal UB, C11 6.5.7p3; masked
# shifts return the correct 0 on real targets, so the behavioral oracle here
# is that the zero elements KEEP their true mass exp(0 - 0.25)/sum -- any
# garbage alignment would zero or move them -- while the UB itself is pinned
# by UBSan, see the C test comment); g3 (E=-5) is a normal filler block.
SM_CS_X_CODES = [8, 2, -128, 0, -128, 0, -16, 4]
SM_CS_X_EXPS = [122, 146, 154, 122]  # E = -5, 19, 27, -5
SM_CS_X_QC = {"mantissa_bits": 8, "exponent_bits": 8, "group_size": 2}

# BWD-N (Task 5): x = SM-A's logits; dLds a per-tensor {1, 0} m=8/e=8 wire on
# its OWN grid (E = -8, absmax-minimal for code 90) with NON-uniform values in
# ~[-0.31, 0.35] -- uniform loss grads make the -dot term vacuous (the
# uniform-lossGrad lesson). propLossQ is the same grouped {2, 4} m=8/e=8 grid
# as the forward outputQ (OUT_QC), so the C test reuses the zero-seed wire.
BWD_DLDS_CODES = [40, -25, 60, 10, -80, 33, -5, 90]
BWD_DLDS_EXPS = [119]  # per-tensor: ONE stored exponent, E = -8
BWD_DLDS_QC = {"mantissa_bits": 8, "exponent_bits": 8, "group_size": 0}

# BWD-S (Task 5): the dLds values arrive FLOAT32 and the funnel stages them
# per-tensor at the propLossQ anchor widths (m=8/e=8). Deliberately lossy on
# that grid: elements 0 and 3 sit ~0.49 codes off their staged values (the
# largest staging error the grid allows, opposite rounding directions), so
# skipping the staging visibly moves the packed gold; elements 3 and 7 also
# stage to DIFFERENT codes than BWD-N's dLds, so this gold cannot collide
# with BWD-N's (both script-asserted below).
BWD_STAGED_DLDS_VALS = [0.1582, -0.099, 0.233, 0.04105, -0.311, 0.13, -0.021, 0.323]


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

    # -- SM-CN fixture validation (fix round 1): the fixture must EXERCISE the
    # amended negative-net-shift branch, i.e. deliberately contain a block
    # coarser than the argmax block, with both regimes inside it. --
    assert SM_CN_X_QC == SM_X_QC and len(SM_CN_X_EXPS) == len(SM_X_EXPS), (
        "SM-CN must share SM-A's input geometry -- the C test reuses the kSmBfpX* "
        "geometry scalars")
    cn_deq = [float(deq(c, SM_CN_X_EXPS, SM_CN_X_QC, i)) for i, c in enumerate(SM_CN_X_CODES)]
    cn_bias = bfp_bias(SM_CN_X_QC)
    cn_argmax = max(range(N), key=lambda i: (cn_deq[i], -i))  # first-wins max
    cn_e_max = SM_CN_X_EXPS[_bfp_group_of(cn_argmax, SM_CN_X_QC["group_size"])] - cn_bias
    cn_x_max = cn_deq[cn_argmax]
    coarse_blocks = [g for g, se in enumerate(SM_CN_X_EXPS) if se - cn_bias > cn_e_max]
    assert coarse_blocks, (
        "SM-CN: no block is coarser than the argmax block (E_i > EMax) -- the fixture "
        "no longer exercises the amended negative-net-shift branch")
    gsz = SM_CN_X_QC["group_size"]
    for g in coarse_blocks:
        block_deq = cn_deq[g * gsz:(g + 1) * gsz]
        # exp-underflow regime: x_i - x_max <= QFLOOR * 2^-F = -22.18 -> mass 0
        assert any(v - cn_x_max <= QFLOOR * 2.0 ** -F for v in block_deq), (
            f"SM-CN: coarse block {g} has no large-negative element (mass exactly 0)")
        assert any(0.0 < v < cn_x_max for v in block_deq), (
            f"SM-CN: coarse block {g} has no small-positive element -- the "
            "positive-in-coarse-block regime is unexercised")

    # -- SM-CN gold: both knob positions; pack at SM-A's outputQ geometry --
    cn_raw_trunc = softmax_values_bfp(SM_CN_X_CODES, SM_CN_X_EXPS, SM_CN_X_QC, "trunc")
    cn_raw_ha = softmax_values_bfp(SM_CN_X_CODES, SM_CN_X_EXPS, SM_CN_X_QC, "half_away")
    cn_codes_trunc, cn_exps_trunc = pack(cn_raw_trunc, OUT_QC)
    cn_codes_ha, cn_exps_ha = pack(cn_raw_ha, OUT_QC)

    # (vi) The underflow elements must land at code 0 (their e_i is EXACTLY 0),
    # and every element -- the +1.0 one included -- must sit within the i-exp
    # + pack bound of the float softmax (the old clamp regime missed the +1.0
    # element by tens of percent, which is what the C RED run captures).
    cn_ref = softmax_float64(cn_deq)
    for cn_codes, cn_exps, knob in ((cn_codes_trunc, cn_exps_trunc, "TRUNC"),
                                    (cn_codes_ha, cn_exps_ha, "HALF_AWAY")):
        for i in range(N):
            if cn_deq[i] - cn_x_max <= QFLOOR * 2.0 ** -F:
                assert cn_codes[i] == 0, (
                    f"SM-CN {knob}: underflow element {i} packs to code {cn_codes[i]}, "
                    "not 0 -- the exp-underflow floor leaked mass")
        check_proximity(f"SM-CN {knob}", cn_codes, cn_exps, OUT_QC, cn_ref)

    # -- SM-CS fixture validation (fix round 2): every arm of the negative-
    # net-shift branch must be exercised -- both saturation disjuncts AND the
    # new zero-code up>=31 clause. The branch conditions are re-derived here
    # from the fixture data alone (same arithmetic as the emulation). --
    assert (SM_CS_X_QC["mantissa_bits"] == SM_X_QC["mantissa_bits"]
            and SM_CS_X_QC["exponent_bits"] == SM_X_QC["exponent_bits"]), (
        "SM-CS must share SM-A's widths -- the C test reuses the kSmBfpX* width scalars")
    cs_bias = bfp_bias(SM_CS_X_QC)
    cs_deq = [float(deq(c, SM_CS_X_EXPS, SM_CS_X_QC, i)) for i, c in enumerate(SM_CS_X_CODES)]
    cs_E = [SM_CS_X_EXPS[_bfp_group_of(i, SM_CS_X_QC["group_size"])] - cs_bias for i in range(N)]
    cs_argmax = max(range(N), key=lambda i: (cs_deq[i], -i))  # first-wins max
    cs_e_max = cs_E[cs_argmax]
    cs_x_max = cs_deq[cs_argmax]
    assert 0.0 < cs_x_max < 22.0, "SM-CS: the argmax must be small-positive"
    cs_sigma = cs_e_max + F
    cs_down = -cs_sigma if cs_sigma < 0 else 0
    sat_up31 = sat_magnitude = zero_up31 = False
    for i in range(N):
        si = (cs_e_max - cs_E[i]) + cs_down
        if si >= 0:
            continue
        up = -si
        if SM_CS_X_CODES[i] < 0 and up >= 31:
            sat_up31 = True
        elif SM_CS_X_CODES[i] < 0 and SM_CS_X_CODES[i] < -(INT32_MAX >> up):
            sat_magnitude = True
        elif SM_CS_X_CODES[i] == 0 and up >= 31:
            zero_up31 = True
    assert sat_up31, "SM-CS: no element reaches the up>=31 saturation disjunct"
    assert sat_magnitude, (
        "SM-CS: no element reaches the magnitude saturation disjunct (|m| > INT32_MAX >> up)")
    assert zero_up31, "SM-CS: no ZERO code reaches the up>=31 clause (the UB fix's pin)"

    # -- SM-CS gold (knob TRUNC; SM-A remains the knob discriminator) --
    cs_raw = softmax_values_bfp(SM_CS_X_CODES, SM_CS_X_EXPS, SM_CS_X_QC, "trunc")
    cs_codes, cs_exps = pack(cs_raw, OUT_QC)
    cs_ref = softmax_float64(cs_deq)
    # (vii) Saturated elements pack to code 0 exactly; the ZERO-code elements
    # keep NONZERO mass matching the float softmax (exp(-x_max)/sum) -- a
    # STRONGER pin than "zero elements pack to 0" (any corrupted alignment
    # for the zero code would zero or move a live output code).
    for i in range(N):
        if cs_deq[i] - cs_x_max <= QFLOOR * 2.0 ** -F:
            assert cs_codes[i] == 0, (
                f"SM-CS: saturated/underflow element {i} packs to {cs_codes[i]}, not 0")
        if SM_CS_X_CODES[i] == 0:
            assert cs_codes[i] != 0, (
                f"SM-CS: zero-code element {i} packs to 0 -- the zero-clause pin is vacuous "
                "(its true mass exp(-x_max)/sum must survive the pack)")
    check_proximity("SM-CS", cs_codes, cs_exps, OUT_QC, cs_ref)

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

    # ---- Task 5: backward fixtures ----

    # -- BWD-N fixture validation + gold --
    assert len(BWD_DLDS_CODES) == N and len(BWD_DLDS_EXPS) == 1
    assert all(-128 <= c <= 127 for c in BWD_DLDS_CODES), "BWD-N dLds codes must fit m=8"
    assert len(set(BWD_DLDS_CODES)) > 1, (
        "BWD-N: dLds is uniform -- the -dot term collapses to dot = c * sum(s) = c and "
        "raw = 0 everywhere (the uniform-lossGrad vacuity)")
    bwd_dlds_deq = [deq(c, BWD_DLDS_EXPS, BWD_DLDS_QC, i) for i, c in enumerate(BWD_DLDS_CODES)]
    bwd_raw, bwd_dot, bwd_s = softmax_backward_bfp(SM_X_CODES, SM_X_EXPS, SM_X_QC,
                                                   bwd_dlds_deq, "trunc")
    assert float(bwd_dot) != 0.0, (
        "BWD-N: dot == 0 -- the -dot term is vacuous, mutation (a) would survive")
    bwd_out = pack(bwd_raw, OUT_QC)
    bwd_codes, bwd_exps = bwd_out

    # (viii) Mutation-(a) killability: dropping -dot must move the PACKED wire.
    no_dot_out = pack([f32_mul(si, d) for si, d in zip(bwd_s, bwd_dlds_deq)], OUT_QC)
    assert no_dot_out != bwd_out, (
        "BWD-N: raw without the -dot term packs to the same wire -- mutation (a) "
        "would survive the gold")

    # (ix) Mutation-(b) killability: recomputing s from dLds instead of x
    # (softmaxValuesBfp(dLdsT, ...)) must move the PACKED wire.
    bad_raw, _, _ = softmax_backward_bfp(BWD_DLDS_CODES, BWD_DLDS_EXPS, BWD_DLDS_QC,
                                         bwd_dlds_deq, "trunc")
    assert pack(bad_raw, OUT_QC) != bwd_out, (
        "BWD-N: s recomputed from dLds packs to the same wire -- mutation (b) "
        "would survive the gold")

    # (x) Autograd proximity (float64 torch, shares nothing with the
    # emulation): 6e-3 -- loose, the expected wire is two quantization layers
    # deep (i-exp s + propLossQ pack).
    bwd_ref = softmax_backward_autograd(x_deq, [float(v) for v in bwd_dlds_deq])
    bwd_out_deq = [float(deq(c, bwd_exps, OUT_QC, i)) for i, c in enumerate(bwd_codes)]
    for i, (got, want) in enumerate(zip(bwd_out_deq, bwd_ref)):
        assert abs(got - want) <= 6e-3, (
            f"BWD-N element {i}: packed dx dequantizes to {got}, torch autograd says "
            f"{want} (bound 6e-3) -- the emulation and the definition disagree")

    # -- BWD-S fixture validation + gold --
    bwd_st_vals = [np.float32(v) for v in BWD_STAGED_DLDS_VALS]
    st_codes, st_exps = bfp_quantize_grouped([float(v) for v in bwd_st_vals],
                                             OUT_QC["mantissa_bits"], OUT_QC["exponent_bits"], 0)
    st_qc = {"mantissa_bits": OUT_QC["mantissa_bits"],
             "exponent_bits": OUT_QC["exponent_bits"], "group_size": 0}
    st_deq = [deq(c, st_exps, st_qc, i) for i, c in enumerate(st_codes)]
    assert any(float(d) != float(v) for d, v in zip(st_deq, bwd_st_vals)), (
        "BWD-S: every dLds value is exactly representable at the staging grid -- the "
        "fixture cannot pin the staging rounding")
    st_raw, st_dot, _ = softmax_backward_bfp(SM_X_CODES, SM_X_EXPS, SM_X_QC, st_deq, "trunc")
    assert float(st_dot) != 0.0, "BWD-S: dot == 0 -- the -dot term is vacuous"
    st_out = pack(st_raw, OUT_QC)
    st_out_codes, st_out_exps = st_out

    # (xi) Staging non-vacuity: feeding the kernel the EXACT float values
    # (i.e. skipping the staging quantization) must move the packed wire.
    unstaged_raw, _, _ = softmax_backward_bfp(SM_X_CODES, SM_X_EXPS, SM_X_QC,
                                              bwd_st_vals, "trunc")
    assert pack(unstaged_raw, OUT_QC) != st_out, (
        "BWD-S: the unstaged floats pack to the same wire -- the fixture cannot pin "
        "the staging step; increase the values' distance from the m=8 grid")

    # (xii) Cross-fixture guard: the staged gold must differ from BWD-N's, so
    # a test wiring mix-up between the two backward tests cannot pass.
    assert st_out != bwd_out, (
        "BWD-S: staged gold equals BWD-N's -- pick staged values whose codes differ")

    # (xiii) Autograd proximity with the STAGED upstream grads (staging loss
    # is the fixture's doing, not the kernel's error budget).
    st_ref = softmax_backward_autograd(x_deq, [float(v) for v in st_deq])
    st_out_deq = [float(deq(c, st_out_exps, OUT_QC, i)) for i, c in enumerate(st_out_codes)]
    for i, (got, want) in enumerate(zip(st_out_deq, st_ref)):
        assert abs(got - want) <= 6e-3, (
            f"BWD-S element {i}: packed dx dequantizes to {got}, torch autograd says "
            f"{want} (bound 6e-3) -- the emulation and the definition disagree")

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
        "\n/* SM-CN (fix round 1): the coarse-negative-block fixture -- block A holds\n"
        " * the SIGNED max at E=-5, the negative-dominated block B is legitimately\n"
        " * COARSER (E=0 > EMax), so its elements take the amended EXACT left-shift\n"
        " * alignment. Input geometry is SM-A's (script-asserted); pack at the same\n"
        " * m=8 {2, 4} outputQ. Underflow elements pack to code 0 exactly. */\n",
        emit_int32_array("kSmBfpCnXCodes", torch.tensor(SM_CN_X_CODES)),
        emit_uint8_array("kSmBfpCnXExponents", SM_CN_X_EXPS),
        emit_int32_array("kSmBfpCnOutCodesTrunc", torch.tensor(cn_codes_trunc)),
        emit_uint8_array("kSmBfpCnOutExponentsTrunc", cn_exps_trunc),
        emit_int32_array("kSmBfpCnOutCodesHalfAway", torch.tensor(cn_codes_ha)),
        emit_uint8_array("kSmBfpCnOutExponentsHalfAway", cn_exps_ha),
        "\n/* SM-CS (fix round 2): grouped {4, 2} -- one coarse block per saturation\n"
        " * disjunct (up=24 magnitude, up=32 up>=31) plus a ZERO code in each; the\n"
        " * up=32 zero code needs the kernel's up>=31 clause (0 << 32 is formal UB).\n"
        " * Saturated elements pack to 0; the zero-code elements KEEP their true\n"
        " * mass exp(-x_max)/sum (script-asserted nonzero). Widths are SM-A's;\n"
        " * knob TRUNC; pack at the same m=8 {2, 4} outputQ. */\n",
        emit_int32_scalar("kSmBfpCsXNumGroups", len(SM_CS_X_EXPS)),
        emit_int32_scalar("kSmBfpCsXGroupSize", SM_CS_X_QC["group_size"]),
        emit_int32_array("kSmBfpCsXCodes", torch.tensor(SM_CS_X_CODES)),
        emit_uint8_array("kSmBfpCsXExponents", SM_CS_X_EXPS),
        emit_int32_array("kSmBfpCsOutCodesTrunc", torch.tensor(cs_codes)),
        emit_uint8_array("kSmBfpCsOutExponentsTrunc", cs_exps),
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
        "\n/* BWD-N (Task 5): x = SM-A's logits; dLds a per-tensor {1, 0} m=8/e=8\n"
        " * BFP wire on its OWN grid (E=-8) with NON-uniform values (a uniform dLds\n"
        " * makes the -dot term vacuous). The backward recomputes s from the LOGITS\n"
        " * (R-N4), dots in index order, and OUT_WRITE-packs dx at the grouped\n"
        " * {2, 4} m=8 propLossQ. Knob TRUNC (the recompute's shift sites). */\n",
        emit_int32_array("kSmBfpBwdDLdsCodes", torch.tensor(BWD_DLDS_CODES)),
        emit_uint8_array("kSmBfpBwdDLdsExponents", BWD_DLDS_EXPS),
        emit_int32_array("kSmBfpBwdOutCodes", torch.tensor(bwd_codes)),
        emit_uint8_array("kSmBfpBwdOutExponents", bwd_exps),
        "\n/* BWD-S (Task 5): same x, dLds as FLOAT32 values the funnel stages\n"
        " * per-tensor at the propLossQ ANCHOR widths (m=8/e=8, HALF_AWAY -- the\n"
        " * op's storage-derived rounding). Deliberately lossy on that grid: the\n"
        " * script asserts skipping the staging moves the packed wire, and that\n"
        " * this gold differs from BWD-N's. */\n",
        emit_float_array("kSmBfpBwdStagedDLdsValues",
                         torch.tensor(bwd_st_vals, dtype=torch.float32)),
        emit_int32_array("kSmBfpBwdStagedOutCodes", torch.tensor(st_out_codes)),
        emit_uint8_array("kSmBfpBwdStagedOutExponents", st_out_exps),
        "\n#endif /* ODT_EXPECTED_BFP_SOFTMAX_H */\n",
    ]
    Path(args.out).write_text("".join(parts))


if __name__ == "__main__":
    main()
