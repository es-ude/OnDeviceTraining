"""Leg A for BFP: the documented scheme, re-implemented from the docs only.

Source of truth: ``docs/conventions/arithmetic-bfp.md`` (cited below as
``arithmetic-bfp``, section numbers as in that file) and, for the storage
grammar, the BFP and group-quant sections of ``docs/conventions/tensor.md``
(``tensor``). Nothing here was written from the C sources or from
``test/unit/goldgen/sym_gold.py``; where the documentation does not pin a
detail, the choice made is stated in the docstring and listed in ``DOC_GAPS``.

What the doc fixes exactly, and this leg therefore computes exactly:

- the exponent rule (arithmetic-bfp section 3), evaluated in RATIONAL
  arithmetic (``fractions.Fraction`` of the float32 absmax) -- the doc states
  the rule in reals and names ``frexpf`` only as the mechanism;
- the D6 clamp of the stored exponent (section 2) and the zero state;
- the two's-complement mantissa grid and its clamp ``[-2^(m-1), qMax]`` (section 1);
- the fold contract of section 5.3 (int32 same-exponent partials, exact
  ``ldexpf`` fold, float32 accumulation, bias seeded first per section 5.4
  Decision 2), sections 7 and 8 included: for a reduction that is ONE
  same-exponent segment the raw float32 output is predicted to the bit
  (``fold_single_segment``), whatever the walk order; for several segments
  ``fold_walk`` evaluates the documented algorithm along an ASSUMED walk order
  (``DOC_GAPS``), which the harness reports but never gates on (F10);
- the headroom guards of sections 5.3 / 5.6, to predict a fail-fast.

What the doc leaves open, and the harness therefore only bounds: the ORDER in
which several segments fold into the float32 accumulator. In that regime the
exact float64 dot product of the dequantized operands (``exact_ref``) is the
reference and the float32 accumulation bound is the acceptance, so the
two legs coincide.

Imports allowed: numpy, stdlib. Never ``sym_gold`` or a ``generate_expected_*``
script.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

import numpy as np

INT32_MAX = 2**31 - 1
INT32_MIN = -(2**31)
F32_U = 2.0**-24

# What the fixed arithmetic-bfp.md now states, with its section. Leg A cites
# these instead of assuming them; they are not gaps.
DOC_COVERED: dict[str, str] = {
    "grid": "arithmetic-bfp section 1: the shipped grid is the full two's-complement one, "
    "mantissas clamp to [-2^(m-1), qMax] with qMax = 2^(m-1) - 1; -2^(m-1) is reachable (HALF_AWAY: "
    "only when D6's exponent cap pins the scale, section 2; SR_HALF_AWAY: no extra route, "
    "floor(x + u) with x >= -qMax cannot go below -qMax). Section 2 states the saturation grid as the same "
    "[-2^(m-1), qMax].",
    "rounding": "arithmetic-sym 'Rounding modes, the SYM_INT32 grid, encode and requant' (referenced "
    "from arithmetic-bfp section 5.3): HALF_AWAY = C round() (an exact .5 goes away from zero); "
    "SR_HALF_AWAY = exact floor(x + u) (the sum is not rounded in float32), u = top-24-bits / 2^24.",
    "headroom": "arithmetic-bfp section 5.3 'Headroom, exact form': bfpSegmentLimit(ma, mb) = "
    "INT32_MAX >> (ma + mb - 2) (shift >= 31 clamps to 1; equal widths 2^(33-2m) - 1: 131071 at "
    "m=8, 511 at m=12, 1 at m=16), checked over min(runA, runB, reductionLen) with run = groupSize "
    "or the reduction length per-tensor; the bias operand is exempt. The sum twin "
    "INT32_MAX >> (m - 1) is section 5.6.",
    "zero_state": "arithmetic-bfp section 3 (absMax == 0 stores E = 0, storedExponent = bias, scale 1.0), "
    "section 5.5 / 5.6 (a freshly allocated wire and an empty tensor start in the zero state).",
}

# Genuinely open: the fixed doc is still silent. Each is an assumption Leg A
# had to make.
DOC_GAPS: list[str] = [
    "Which float the pack divides: section 3 states E from absMax in reals and the mantissa "
    "as 'x / 2^E rounded'; Leg A takes the division as exact (a power-of-two division of a "
    "float32 value is exact unless it leaves the normal range), so codes carry NO float32 "
    "expression-order ambiguity -- only E may differ (F2: the frexpf(absMax / qMax) float32 "
    "quotient at absMax within an ulp of qMax * 2^k). A code that differs with equal E is a "
    "MISMATCH, not a tie.",
    "Bias seed in the BFP GEMM: section 5.4 Decision 2 says the kernel 'dequant-seeds the "
    "float accumulator with the bias value BEFORE the reduction (exact mantissa * 2^E)'. For "
    "ONE segment the seed and the fold meet in a single float32 addition, which is all "
    "fold_single_segment relies on; with several segments the seed-first order is taken as "
    "written and the fold order is F10 (next entry).",
    "Fold order across segments (F10): section 5.3 'fold ... on ANY group-boundary crossing "
    "(either operand) ... and once more at the tail' fixes the algorithm but not the walk "
    "order of every op; section 7 admits (float)partial above 2^24 adds a rounding the D7 "
    "claim omits. Leg A predicts the exact float32 result only for single-segment reductions; "
    "for several segments fold_walk assumes: Linear -- the reduction index ascending (section "
    "5.3: storage groups are contiguous along the axis the forward reduction walks); Conv1d "
    "forward -- input channel outer, kernel tap inner (the weight's [Cout, Cin/g, K] storage "
    "order); the gather-formulated ConvT1d forward and the Conv1d dx it serves (section 6) -- "
    "kernel tap outer, channel inner; conv weightGrad -- '(batch, outputPosition) as the "
    "reduction' and biasGrad 'b-outer/outPos-inner' (section 5.6, documented). The result is "
    "reported (bit-exact or not) and never gates: the walk-order-free bound is the acceptance.",
    "(float)partial rounding mode (section 7): the doc says the int32 -> float conversion "
    "'rounds' above 2^24 without naming the mode. Assumed IEEE round-to-nearest-even (C's "
    "default conversion), reproduced by numpy's int64 -> float32 cast.",
    "Staging of a FLOAT32 operand under ARITH_BFP (section 5.4 Decision 1): per-tensor at the "
    "weight's (mantissaBits, exponentBits), rounded by the op's mode -- documented; that the "
    "staged grid's exponent follows the SAME section-3 rule as a storage pack is assumed (the "
    "doc says 'quantizes them into transient exponent-backed scratch' without naming the rule).",
]


# --------------------------------------------------------------------------- #
# Grid (arithmetic-bfp section 1; tensor BFP section)                          #
# --------------------------------------------------------------------------- #


def qmax(mantissa_bits: int) -> int:
    """arithmetic-bfp section 1: qMax = 2^(mantissaBits-1) - 1."""
    return (1 << (mantissa_bits - 1)) - 1


def code_floor_twos_complement(mantissa_bits: int) -> int:
    """The lower end of the mantissa grid, -2^(m-1) (arithmetic-bfp section
    1: 'clamps mantissas to [-2^(m-1), qMax]'; section 2 states the
    saturation grid the same way)."""
    return -(1 << (mantissa_bits - 1))


def exponent_bias(exponent_bits: int) -> int:
    """tensor, BFP section: bias = 2^(exponentBits-1) - 1 (arithmetic-bfp
    section 2 by the exponentBits = 8 example: stored 255 <-> E = 128)."""
    return (1 << (exponent_bits - 1)) - 1


def stored_exponent_cap(exponent_bits: int) -> int:
    """Largest stored exponent: 2^exponentBits - 1, but 'never exceeds bias +
    127' (arithmetic-bfp section 2 -- only binding at exponentBits = 8, where
    the natural top stored 255 has no finite float32 scale)."""
    return min((1 << exponent_bits) - 1, exponent_bias(exponent_bits) + 127)


def round_half_away(v: np.ndarray) -> np.ndarray:
    """Half away from zero (arithmetic-sym 'Rounding modes, the SYM_INT32 grid...': C round()); float64 -> int64."""
    v = np.asarray(v, dtype=np.float64)
    return np.where(v >= 0.0, np.floor(v + 0.5), np.ceil(v - 0.5)).astype(np.int64)


def group_ids(n: int, group_size: int) -> np.ndarray:
    """Flat-storage group of each element: tensor, 'Storage-order binding': a
    group is groupSize consecutive elements in STORAGE order, id = flatIdx /
    groupSize; the BFP section says its geometry 'mirrors symQConfig_t
    exactly'. group_size 0 is the per-tensor sentinel ({1,0}, arithmetic-bfp
    section 5.5)."""
    if group_size == 0:
        return np.zeros(n, dtype=np.int64)
    return np.arange(n, dtype=np.int64) // group_size


def num_groups(n: int, group_size: int) -> int:
    """arithmetic-bfp section 5.5: numGroups = numberOfValues / groupSize,
    derived from the actual element count; groupSize == n normalizes to the
    per-tensor {1,0} config; a non-divisor fails fast."""
    if group_size == 0 or group_size >= n:
        return 1
    if n % group_size != 0:
        raise ValueError(f"numberOfValues {n} % groupSize {group_size} != 0 (section 5.5 fails fast)")
    return n // group_size


# --------------------------------------------------------------------------- #
# Exponent rule (section 3) and D6 clamp (section 2), exact                   #
# --------------------------------------------------------------------------- #


@dataclass
class GroupExponent:
    absmax: float  # the float32 absmax of the group as a Python float
    e_unclamped: int | None  # smallest E with absmax / 2^E <= qMax; None for absmax == 0
    stored: int  # after the D6 clamp into [0, cap]
    regime: str  # "zero" | "normal" | "saturate" | "flush"
    boundary_rel: float | None  # q / 2^E_unclamped - 1 in (-1/2, 0]: 0 = absmax IS qMax * 2^E


def exponent_rule_exact(absmax: float, mantissa_bits: int) -> tuple[int | None, float | None]:
    """arithmetic-bfp section 3: E = smallest E with absMax / 2^E <= qMax,
    evaluated exactly: absmax is a float32 (a dyadic rational), qMax an integer,
    so q = absmax / qMax is an exact Fraction and E = ceil(log2 q) is found by
    comparing q with powers of two -- no float rounding anywhere. Returns
    (E, boundary_rel) with boundary_rel = q / 2^E - 1 in (-1/2, 0], the exact
    relative distance of the quotient BELOW the power of two 2^E (0 means the
    quotient IS a power of two: absmax == qMax * 2^E). For absmax == 0 the
    doc's zero state applies and (None, None) is returned."""
    if absmax == 0.0:
        return None, None
    q = Fraction(absmax) / qmax(mantissa_bits)
    e = int(np.floor(np.log2(float(q))))  # float estimate, corrected exactly below
    while Fraction(2) ** e < q:
        e += 1
    while e > -2000 and Fraction(2) ** (e - 1) >= q:
        e -= 1
    rel = float(q / (Fraction(2) ** e) - 1)
    return e, rel


def exponent_of(absmax: float, mantissa_bits: int, exponent_bits: int) -> GroupExponent:
    """Section 3 rule + section 2 D6 clamp: stored = clamp(E + bias, 0, cap);
    absmax == 0 -> stored = bias (zero state, section 3)."""
    bias = exponent_bias(exponent_bits)
    cap = stored_exponent_cap(exponent_bits)
    e, rel = exponent_rule_exact(absmax, mantissa_bits)
    if e is None:
        return GroupExponent(absmax, None, bias, "zero", None)
    stored = e + bias
    regime = "normal"
    if stored > cap:
        stored, regime = cap, "saturate"
    elif stored < 0:
        stored, regime = 0, "flush"
    return GroupExponent(absmax, e, stored, regime, rel)


def boundary_window_rel(absmax: float, k: int) -> float:
    """Relative half-width of the F2 window at this absmax: k float32 ulps of
    the quotient absmax / qMax, expressed relative to the quotient -- the
    float32 division the doc names as the mechanism can land on the other
    side of a power of two only within this distance."""
    a = max(abs(float(absmax)), float(np.finfo(np.float32).tiny))
    return k * float(np.spacing(np.float32(a))) / a


# --------------------------------------------------------------------------- #
# Pack FLOAT32 -> BFP (sections 1-3, per group)                               #
# --------------------------------------------------------------------------- #


@dataclass
class Packed:
    codes: np.ndarray  # int64, flat
    stored: np.ndarray  # int64 [numGroups]
    pre: np.ndarray  # float64 x / 2^E (exact), flat -- before rounding and clamp
    groups: list[GroupExponent]
    n_clamped: int  # elements whose rounded value leaves [-2^(m-1), qMax] (D6 saturation only)


def pack_bfp(values: np.ndarray, mantissa_bits: int, exponent_bits: int, group_size: int) -> Packed:
    """FLOAT32 -> BFP, the documented rule per group: E by ``exponent_of`` on
    the group's absmax (section 3 + D6), mantissa = round(x / 2^E) clamped to
    [-2^(m-1), qMax] (section 1). The division is exact
    (DOC_GAPS[0]) so ``pre`` is the real pre-rounding value. ``values`` are
    float32 (any shape; flat storage order is the group order)."""
    x = np.asarray(values, dtype=np.float32).ravel().astype(np.float64)
    if not np.all(np.isfinite(x)):
        raise ValueError("non-finite input value: the pack entries fail fast (section 5.6, R9)")
    n = x.size
    g_of = group_ids(n, group_size)
    n_g = num_groups(n, group_size) if n else 1
    bias = exponent_bias(exponent_bits)
    q = qmax(mantissa_bits)
    floor = code_floor_twos_complement(mantissa_bits)
    codes = np.zeros(n, dtype=np.int64)
    pre = np.zeros(n, dtype=np.float64)
    stored = np.zeros(n_g, dtype=np.int64)
    groups: list[GroupExponent] = []
    n_clamped = 0
    for g in range(n_g):
        sel = g_of == g
        xs = x[sel]
        absmax = float(np.float32(np.max(np.abs(xs)))) if xs.size else 0.0
        ge = exponent_of(absmax, mantissa_bits, exponent_bits)
        groups.append(ge)
        stored[g] = ge.stored
        p = xs / (2.0 ** (ge.stored - bias))
        pre[sel] = p
        # Clamp in float64 BEFORE the integer cast (section 1: the grid clamp
        # is part of the pack; a huge finite value must not reach the int64
        # cast). Integer bounds make clamp-then-round equal round-then-clamp;
        # the clamp count reads the pre-clamp value (round half away leaves
        # the grid exactly at q + 0.5 / floor - 0.5).
        n_clamped += int(((p >= q + 0.5) | (p <= floor - 0.5)).sum())
        codes[sel] = round_half_away(np.clip(p, floor, q))
    return Packed(codes, stored, pre, groups, n_clamped)


def repack_under(
    values: np.ndarray, stored: np.ndarray, mantissa_bits: int, exponent_bits: int, group_size: int
) -> Packed:
    """The pack of section 1 under GIVEN stored exponents (the C's), for the
    F2 census: when the C's E differs from the exact rule at a boundary, its
    codes must still be round(x / 2^E_C) clamped -- self-consistent under its
    own exponent."""
    x = np.asarray(values, dtype=np.float32).ravel().astype(np.float64)
    n = x.size
    g_of = group_ids(n, group_size)
    bias = exponent_bias(exponent_bits)
    q = qmax(mantissa_bits)
    floor = code_floor_twos_complement(mantissa_bits)
    stored = np.asarray(stored, dtype=np.int64)
    pre = x / (2.0 ** (stored[g_of] - bias))
    n_clamped = int(((pre >= q + 0.5) | (pre <= floor - 0.5)).sum())  # see pack_bfp
    return Packed(round_half_away(np.clip(pre, floor, q)), stored.copy(), pre, [], n_clamped)


def dequant(codes: np.ndarray, stored: np.ndarray, exponent_bits: int, group_size: int) -> np.ndarray:
    """value = mantissa * 2^(storedExponent - bias) per group (tensor BFP
    section; arithmetic-bfp section 5.6 'BFP dequant is mantissa * 2^(E-bias)
    per GROUP'). Exact in float64 (and in float32: at most 16 significant
    bits)."""
    c = np.asarray(codes, dtype=np.int64).ravel()
    g_of = group_ids(c.size, group_size)
    e = np.asarray(stored, dtype=np.int64)[g_of] - exponent_bias(exponent_bits)
    return c.astype(np.float64) * np.ldexp(1.0, e.astype(np.int64))


# --------------------------------------------------------------------------- #
# Kernel contract (section 5.3): headroom guard and the fold                   #
# --------------------------------------------------------------------------- #


def segment_limit(ma: int, mb: int) -> int:
    """bfpSegmentLimit(ma, mb) = INT32_MAX >> (ma+mb-2); 'shift >= 31 clamps
    to 1' (section 5.3)."""
    shift = ma + mb - 2
    if shift >= 31:
        return 1
    return INT32_MAX >> shift


def sum_segment_limit(m: int) -> int:
    """bfpSumSegmentLimit(m) = INT32_MAX >> (m-1) (section 5.6, biasGrad)."""
    return INT32_MAX >> (m - 1)


def max_segment(run_a: int, run_b: int | None, reduction: int) -> int:
    """min(runA, runB, reductionLen) with run = groupSize, or the reduction
    length for a per-tensor operand (sections 5.3 / 5.6)."""
    ra = reduction if run_a == 0 else run_a
    rb = reduction if (run_b is None or run_b == 0) else run_b
    return min(ra, rb, reduction)


def headroom_ok(run_a: int, run_b: int, reduction: int, ma: int, mb: int) -> bool:
    """bfpValidateBlockHeadroom (section 5.3): the largest same-exponent
    segment must not exceed segment_limit(ma, mb) (arithmetic-bfp section 5.3)."""
    return max_segment(run_a, run_b, reduction) <= segment_limit(ma, mb)


def sum_headroom_ok(run: int, reduction: int, m: int) -> bool:
    """bfpValidateSumHeadroom (section 5.6): min(groupSize, reductionLen) (or
    the reduction length per-tensor) within sum_segment_limit(m)."""
    return max_segment(run, None, reduction) <= sum_segment_limit(m)


def f32(x: float) -> float:
    return float(np.float32(x))


@dataclass
class SingleSegmentFold:
    partial: np.ndarray  # exact int64 same-exponent partial per output element
    predicted: np.ndarray  # the documented fold evaluated in float32, per element
    exact: np.ndarray  # float64 value of partial * 2^shift (+ bias)
    s7_loss: np.ndarray  # predicted - exact: the (float)partial rounding section 7 admits


def fold_single_segment(partial: np.ndarray, shift: int, bias_values: np.ndarray | None) -> SingleSegmentFold:
    """Section 5.3 fold for a reduction that is ONE same-exponent segment
    (both operands per-tensor, or one group covering the whole reduction):
    acc = (float)bias (exact, section 5.4 Decision 2) ; acc += ldexpf((float)
    partial, Ea + Eb - biasA - biasB). Two float32 roundings at most -- the
    int->float conversion of the partial (exact below 2^24, section 7 above
    it, DOC_GAPS[3]) and the single addition -- so the result is predicted to
    the bit, whatever the walk order (integer addition commutes). ``shift`` =
    Ea + Eb - biasA - biasB. Section 8 (ldexpf overflow to inf) is reproduced
    by numpy's float32 ldexp."""
    p = np.asarray(partial, dtype=np.int64)
    p32 = p.astype(np.float32)  # (float)partial -- section 7's rounding
    with np.errstate(over="ignore"):
        folded = np.ldexp(p32, np.int32(shift)).astype(np.float32)
        if bias_values is not None:
            b32 = np.asarray(bias_values, dtype=np.float64).astype(np.float32)
            predicted = (b32 + folded).astype(np.float32)
        else:
            predicted = folded
    exact = p.astype(np.float64) * (2.0**shift)
    if bias_values is not None:
        exact = exact + np.asarray(bias_values, dtype=np.float64)
    return SingleSegmentFold(p, predicted.astype(np.float64), exact, predicted.astype(np.float64) - exact)


@dataclass
class FoldStats:
    n_outputs: int
    n_segments_max: int  # most segments any output element folded
    n_segments_total: int
    partial_abs_max: int  # largest |int32 partial| closed anywhere
    n_partials_gt_2p24: int  # segments whose (float)partial rounded (section 7)
    cast_loss_max_value: float  # largest |(float)partial - partial| * 2^shift in value units
    partial_fits_int32: bool
    single_segment: bool  # every output element folded at most one segment

    def to_json(self) -> dict:
        return dict(self.__dict__)


@dataclass
class WalkFold:
    predicted: np.ndarray  # float32 (as float64) raw per output element, documented fold along the walk
    exact: np.ndarray  # float64 exact sum of the same terms (+ bias)
    stats: FoldStats


def fold_walk(
    a_codes: np.ndarray,
    a_stored: np.ndarray,
    a_bits: tuple[int, int, int],
    ia: list[np.ndarray],
    b_codes: np.ndarray | None,
    b_stored: np.ndarray | None,
    b_bits: tuple[int, int, int] | None,
    ib: list[np.ndarray] | None,
    bias_values: np.ndarray | None,
) -> WalkFold:
    """The section 5.3 / 5.6 algorithm along a given walk (DOC_GAPS[2]):
    ``ia[o]`` (and ``ib[o]``) are the flat storage indices of the terms of
    output element ``o`` in walk order. Per element: acc = bias seed (float32,
    section 5.4 Decision 2; 0 without bias); partial = 0; for every term, if
    EITHER operand's group id differs from the previous term's, fold
    (acc += ldexpf((float)partial, Ea + Eb - biasA - biasB), partial = 0);
    partial += a * b (int32 -- the guard of section 5.3 excludes overflow);
    tail fold. ``b_codes`` None is the SUM form of section 5.6 (biasGrad:
    partial += a, shift = Ea - biasA). ``*_bits`` = (mantissaBits,
    exponentBits, groupSize)."""
    a = np.asarray(a_codes, dtype=np.int64).ravel()
    a_st = np.asarray(a_stored, dtype=np.int64)
    ga_all = group_ids(a.size, a_bits[2])
    bias_a = exponent_bias(a_bits[1])
    is_sum = b_codes is None
    if not is_sum:
        b = np.asarray(b_codes, dtype=np.int64).ravel()
        b_st = np.asarray(b_stored, dtype=np.int64)
        gb_all = group_ids(b.size, b_bits[2])
        bias_b = exponent_bias(b_bits[1])
    n_out = len(ia)
    predicted = np.zeros(n_out, dtype=np.float64)
    exact = np.zeros(n_out, dtype=np.float64)
    seg_max = 0
    seg_total = 0
    partial_max = 0
    n_gt = 0
    cast_loss_max = 0.0
    fits = True
    for o in range(n_out):
        idx_a = np.asarray(ia[o], dtype=np.int64)
        acc = np.float32(0.0)
        ex = 0.0
        if bias_values is not None:
            acc = np.float32(bias_values[o])
            ex = float(bias_values[o])
        if idx_a.size == 0:
            predicted[o] = float(acc)
            exact[o] = ex
            continue
        av = a[idx_a]
        ga = ga_all[idx_a]
        if is_sum:
            terms = av
            shifts = a_st[ga] - bias_a
            change = np.flatnonzero(ga[1:] != ga[:-1]) + 1
        else:
            idx_b = np.asarray(ib[o], dtype=np.int64)
            bv = b[idx_b]
            gb = gb_all[idx_b]
            terms = av * bv
            shifts = (a_st[ga] - bias_a) + (b_st[gb] - bias_b)
            change = np.flatnonzero((ga[1:] != ga[:-1]) | (gb[1:] != gb[:-1])) + 1
        bounds = np.concatenate(([0], change, [idx_a.size]))
        n_seg = bounds.size - 1
        seg_max = max(seg_max, n_seg)
        seg_total += n_seg
        with np.errstate(over="ignore"):
            for s in range(n_seg):
                lo, hi = int(bounds[s]), int(bounds[s + 1])
                partial = int(terms[lo:hi].sum())
                shift = int(shifts[lo])
                partial_max = max(partial_max, abs(partial))
                if partial > INT32_MAX or partial < INT32_MIN:
                    fits = False
                p32 = np.float32(partial)
                if abs(partial) > 2**24:
                    n_gt += 1
                    cast_loss_max = max(cast_loss_max, abs(float(p32) - partial) * 2.0**shift)
                acc = np.float32(acc + np.float32(np.ldexp(p32, np.int32(shift))))
                ex += partial * 2.0**shift
        predicted[o] = float(acc)
        exact[o] = ex
    stats = FoldStats(
        n_outputs=n_out,
        n_segments_max=seg_max,
        n_segments_total=seg_total,
        partial_abs_max=partial_max,
        n_partials_gt_2p24=n_gt,
        cast_loss_max_value=cast_loss_max,
        partial_fits_int32=fits,
        single_segment=(seg_max <= 1),
    )
    return WalkFold(predicted, exact, stats)


# --------------------------------------------------------------------------- #
# Walk orders of the GEMM family (DOC_GAPS[2]): flat storage indices per      #
# output element, in the order the fold is evaluated                           #
# --------------------------------------------------------------------------- #


@dataclass
class Walk:
    out_shape: tuple
    ia: list[np.ndarray]  # operand a (input / loss) flat indices per output element
    ib: list[np.ndarray] | None  # operand b (weight / input) flat indices; None for a sum
    reduction: int  # the nominal reduction length N (terms per output, padding included)
    bias_index: np.ndarray | None  # which bias element seeds each output (None without bias)


def linear_walk(batch: int, in_f: int, out_f: int, direction: str, with_bias: bool) -> Walk:
    """Linear y = x @ W^T + b with x [B, inF], W [outF, inF] (the PyTorch
    layout, data-shape 'Weight layouts, orientation and padding'); forward reduces over inF, dx = L @ W over
    outF, weightGrad = L^T @ x over the batch, biasGrad = batch sum of L
    (section 5.6: 'per-feature and the inner loop walks the batch')."""
    ia: list[np.ndarray] = []
    ib: list[np.ndarray] = []
    if direction == "forward":
        for bb in range(batch):
            for o in range(out_f):
                i = np.arange(in_f)
                ia.append(bb * in_f + i)
                ib.append(o * in_f + i)
        bias_index = np.tile(np.arange(out_f), batch) if with_bias else None
        return Walk((batch, out_f), ia, ib, in_f, bias_index)
    if direction == "dx":
        for bb in range(batch):
            for i in range(in_f):
                o = np.arange(out_f)
                ia.append(bb * out_f + o)
                ib.append(o * in_f + i)
        return Walk((batch, in_f), ia, ib, out_f, None)
    if direction == "weightGrad":
        for o in range(out_f):
            for i in range(in_f):
                bb = np.arange(batch)
                ia.append(bb * out_f + o)
                ib.append(bb * in_f + i)
        return Walk((out_f, in_f), ia, ib, batch, None)
    if direction == "biasGrad":
        for o in range(out_f):
            ia.append(np.arange(batch) * out_f + o)
        return Walk((out_f,), ia, None, batch, None)
    raise ValueError(direction)


@dataclass(frozen=True)
class ConvGeom:
    """Conv1d / ConvT1d geometry (PyTorch semantics, data-shape 'Weight layouts, orientation and padding'):
    stride, dilation, conv groups, explicit left/right zero padding (Conv1d)
    or outputPadding (ConvT1d, VALID)."""

    kernel_size: int
    stride: int
    dilation: int
    groups: int
    pad_left: int = 0
    pad_right: int = 0
    output_padding: int = 0

    def conv_out_len(self, length: int) -> int:
        eff = self.dilation * (self.kernel_size - 1) + 1
        padded = length + self.pad_left + self.pad_right
        return 0 if padded < eff else (padded - eff) // self.stride + 1

    def convt_out_len(self, length: int) -> int:
        return (length - 1) * self.stride + self.dilation * (self.kernel_size - 1) + self.output_padding + 1


def conv1d_walk(
    batch: int, c_in: int, length: int, c_out: int, g: ConvGeom, direction: str, with_bias: bool
) -> Walk:
    """Conv1d y[b, oc, t] = sum_{ic in group(oc), k} x[b, ic, t*s + k*d - pl] *
    W[oc, ic', k] (cross-correlation, PyTorch layout). Walk orders: forward
    (ic outer, k inner); weightGrad '(batch, outputPosition) as the reduction'
    and biasGrad 'per-output-channel over (batch, outputPosition)' (section
    5.6, documented); dx = the gather adjoint (section 6; kernel tap outer,
    channel inner, DOC_GAPS[2]). Padded taps are never visited."""
    cin_g = c_in // g.groups
    cout_g = c_out // g.groups
    l_out = g.conv_out_len(length)

    def x_idx(b, ic, pos):
        return (b * c_in + ic) * length + pos

    def w_idx(oc, icl, k):
        return (oc * cin_g + icl) * g.kernel_size + k

    def l_idx(b, oc, t):
        return (b * c_out + oc) * l_out + t

    ia: list[np.ndarray] = []
    ib: list[np.ndarray] = []
    if direction == "forward":
        for b in range(batch):
            for oc in range(c_out):
                grp = oc // cout_g
                for t in range(l_out):
                    xa, wb = [], []
                    for icl in range(cin_g):
                        ic = grp * cin_g + icl
                        for k in range(g.kernel_size):
                            pos = t * g.stride + k * g.dilation - g.pad_left
                            if 0 <= pos < length:
                                xa.append(x_idx(b, ic, pos))
                                wb.append(w_idx(oc, icl, k))
                    ia.append(np.asarray(xa, dtype=np.int64))
                    ib.append(np.asarray(wb, dtype=np.int64))
        bias_index = np.repeat(np.tile(np.arange(c_out), batch), l_out) if with_bias else None
        return Walk((batch, c_out, l_out), ia, ib, cin_g * g.kernel_size, bias_index)
    if direction == "weightGrad":
        for oc in range(c_out):
            grp = oc // cout_g
            for icl in range(cin_g):
                ic = grp * cin_g + icl
                for k in range(g.kernel_size):
                    la, xb = [], []
                    for b in range(batch):
                        for t in range(l_out):
                            pos = t * g.stride + k * g.dilation - g.pad_left
                            if 0 <= pos < length:
                                la.append(l_idx(b, oc, t))
                                xb.append(x_idx(b, ic, pos))
                    ia.append(np.asarray(la, dtype=np.int64))
                    ib.append(np.asarray(xb, dtype=np.int64))
        return Walk((c_out, cin_g, g.kernel_size), ia, ib, batch * l_out, None)
    if direction == "biasGrad":
        for oc in range(c_out):
            la = [l_idx(b, oc, t) for b in range(batch) for t in range(l_out)]
            ia.append(np.asarray(la, dtype=np.int64))
        return Walk((c_out,), ia, None, batch * l_out, None)
    if direction == "dx":
        for b in range(batch):
            for ic in range(c_in):
                grp = ic // cin_g
                icl = ic - grp * cin_g
                for i in range(length):
                    la, wb = [], []
                    for k in range(g.kernel_size):
                        num = i + g.pad_left - k * g.dilation
                        if num < 0 or num % g.stride:
                            continue
                        t = num // g.stride
                        if t >= l_out:
                            continue
                        for ocl in range(cout_g):
                            oc = grp * cout_g + ocl
                            la.append(l_idx(b, oc, t))
                            wb.append(w_idx(oc, icl, k))
                    ia.append(np.asarray(la, dtype=np.int64))
                    ib.append(np.asarray(wb, dtype=np.int64))
        return Walk((batch, c_in, length), ia, ib, cout_g * g.kernel_size, None)
    raise ValueError(direction)


def convt1d_walk(
    batch: int, c_in: int, length: int, c_out: int, g: ConvGeom, direction: str, with_bias: bool
) -> Walk:
    """Conv1dTransposed, VALID: y[b, oc, t] = sum_{ic in group(oc), k, i: i*s +
    k*d == t} x[b, ic, i] * W[ic, oc', k], W [Cin, Cout/g, K], Lout = (L-1)*s +
    d*(K-1) + outputPadding + 1 (PyTorch). The forward is gather-formulated
    (section 6): per output element, taps outer, channel inner (DOC_GAPS[2]);
    outputPadding tail positions have no contributors and stay at the bias
    seed. weightGrad / biasGrad walk (batch, outputPosition) (section 5.6);
    dx is a Conv1d of the loss with W (section 5.6: 'Conv1dTransposed's dx is
    mathematically a Conv1d op'), channel outer, tap inner."""
    cin_g = c_in // g.groups
    cout_g = c_out // g.groups
    l_out = g.convt_out_len(length)

    def x_idx(b, ic, i):
        return (b * c_in + ic) * length + i

    def w_idx(ic, ocl, k):
        return (ic * cout_g + ocl) * g.kernel_size + k

    def l_idx(b, oc, t):
        return (b * c_out + oc) * l_out + t

    ia: list[np.ndarray] = []
    ib: list[np.ndarray] = []
    if direction == "forward":
        for b in range(batch):
            for oc in range(c_out):
                grp = oc // cout_g
                ocl = oc - grp * cout_g
                for t in range(l_out):
                    xa, wb = [], []
                    for k in range(g.kernel_size):
                        num = t - k * g.dilation
                        if num < 0 or num % g.stride:
                            continue
                        i = num // g.stride
                        if i >= length:
                            continue
                        for icl in range(cin_g):
                            ic = grp * cin_g + icl
                            xa.append(x_idx(b, ic, i))
                            wb.append(w_idx(ic, ocl, k))
                    ia.append(np.asarray(xa, dtype=np.int64))
                    ib.append(np.asarray(wb, dtype=np.int64))
        bias_index = np.repeat(np.tile(np.arange(c_out), batch), l_out) if with_bias else None
        return Walk((batch, c_out, l_out), ia, ib, cin_g * g.kernel_size, bias_index)
    if direction == "weightGrad":
        for ic in range(c_in):
            grp = ic // cin_g
            for ocl in range(cout_g):
                oc = grp * cout_g + ocl
                for k in range(g.kernel_size):
                    la, xb = [], []
                    for b in range(batch):
                        for t in range(l_out):
                            num = t - k * g.dilation
                            if num < 0 or num % g.stride:
                                continue
                            i = num // g.stride
                            if i >= length:
                                continue
                            la.append(l_idx(b, oc, t))
                            xb.append(x_idx(b, ic, i))
                    ia.append(np.asarray(la, dtype=np.int64))
                    ib.append(np.asarray(xb, dtype=np.int64))
        return Walk((c_in, cout_g, g.kernel_size), ia, ib, batch * length, None)
    if direction == "biasGrad":
        for oc in range(c_out):
            la = [l_idx(b, oc, t) for b in range(batch) for t in range(l_out)]
            ia.append(np.asarray(la, dtype=np.int64))
        return Walk((c_out,), ia, None, batch * l_out, None)
    if direction == "dx":
        for b in range(batch):
            for ic in range(c_in):
                grp = ic // cin_g
                for i in range(length):
                    la, wb = [], []
                    for ocl in range(cout_g):
                        oc = grp * cout_g + ocl
                        for k in range(g.kernel_size):
                            t = i * g.stride + k * g.dilation
                            if t < l_out:
                                la.append(l_idx(b, oc, t))
                                wb.append(w_idx(ic, ocl, k))
                    ia.append(np.asarray(la, dtype=np.int64))
                    ib.append(np.asarray(wb, dtype=np.int64))
        return Walk((batch, c_in, length), ia, ib, cout_g * g.kernel_size, None)
    raise ValueError(direction)
