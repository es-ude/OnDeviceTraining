"""Leg A for SYM_INT32: the documented scheme, re-implemented from the docs only.

Source of truth: ``docs/conventions/arithmetic-sym.md`` (cited below as
``arithmetic-sym``) and ``docs/conventions/tensor.md`` (``tensor``). Every
function docstring names the section it implements. Nothing here was written
from the C sources or from ``test/unit/goldgen/sym_gold.py``; where the
documentation does not pin a detail, the choice made is stated in the
docstring and listed in ``DOC_GAPS`` so the report can surface it.

All arithmetic is exact (int64 MACs) or float64. The float32 expression order
of the C requant is NOT part of the documented scheme, so this leg cannot be
bit-exact at float32 rounding ties; ``census.py`` classifies those.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# What the fixed docs now state, with the section that states it. Leg A cites
# these instead of assuming them; they are not gaps. ``docs/conventions/``
# names below are file + section heading.
DOC_COVERED: dict[str, str] = {
    "grid": "arithmetic-sym 'Rounding modes, the SYM_INT32 grid, encode and requant', Grid: "
    "[qMin, qMax] = [-2^(b-1), 2^(b-1) - 1] (2047 and -2048 at int12); every SYM_INT32 requant "
    "clamps to it; HALF_AWAY never produces qMin from an absmax scale, SR_HALF_AWAY can (u == 0).",
    "zero_absmax": "same section, dynamic absmax encode: an all-zero tensor gets scale = 1.0 and "
    "all-zero codes.",
    "half_away": "same section, Rounding modes: HALF_AWAY = C round() (an exact .5 goes away from "
    "zero), result converted to int32.",
    "staging": "same document, 'Staging a FLOAT32 operand under ARITH_SYM_INT32': the dynamic absmax "
    "encode at ODT_SYM_OPERAND_QMAXBITS (12), per TENSOR, rounded by the op's mode.",
    "fixed_requant": "same section, fixed-scale requant: clamp(code * inScale / targetScale, qMin, "
    "qMax) in the float domain, THEN round; no integer clamp after the rounding (under "
    "SR_HALF_AWAY a value saturated at qMin can land on qMin - 1; the cells here are HALF_AWAY, "
    "where the clamp bounds are integers and the rounding cannot leave the grid).",
    "layouts": "data-shape 'Weight layouts, orientation and padding': Linear weight [outF, inF], "
    "y = x W^T + b, dx = dL W, weightGrad = dL^T x, biasGrad = sum of dL over the batch; Conv1d "
    "weight [Cout, Cin/groups, K], cross-correlation, tap k reads outPos*stride - padLeft + "
    "k*dilation; Conv1dTransposed weight [Cin, Cout/groups, K].",
    "padding": "data-shape, same section: VALID outputLength = (L - effK)/stride + 1, or 0 when "
    "L < effK; EXPLICIT pads `padding` zeros per side; SAME outputLength = ceil(L/stride), "
    "totalPad = max(0, effK + (outputLength-1)*stride - L), padLeft = totalPad/2 (floor), "
    "padRight = totalPad - padLeft.",
    "accumulator_bound": "arithmetic-sym 'Operand bit-width: int12, not int16', 'SYM int32 accumulator "
    "guard (shipped)': a product of widths (qa, qb) is bounded by 2^(qa-1) * 2^(qb-1); int12 is "
    "provably sound for N <= 511; when BOTH operands are absmax-encoded (|code| <= 2^(b-1) - 1) "
    "the budget is 512 int12 terms (512 * 2047^2 < INT32_MAX, 513 wraps). The guard's static "
    "proof is N_max * 2^(qa-1) * 2^(qb-1) + max|seed| <= INT32_MAX; an unprovable reduction runs "
    "checked (exact when it fits, abort at the first real overflow).",
}

# Genuinely open: the fixed docs are still silent. Each is an assumption Leg A
# had to make; a C/Leg A disagreement on one of them is a documentation
# finding, not an implementation bug.
DOC_GAPS: list[str] = [
    "Float32 evaluation order of the layer-tier width restore 'round(raw * s_raw / s_out)' and of "
    "the bias refold: arithmetic-sym writes the encode as roundByMode(x / scale) and the dynamic "
    "requant as (float)code * (inScale / scale), but not the float32 expression of the restore "
    "chain. Leg A evaluates in float64; disagreements within k*ulp32 of a .5 boundary are ties.",
    "Rounding owner of the forward bias-seed refold: arithmetic-sym says 'seed = "
    "round(param_q * param_scale / accumulator_scale)' without naming the rounding "
    "mode's owner (op vs bias storage). Assumed HALF_AWAY (both coincide in det cells).",
    "Bias seed beyond the float32 tie window: the seed rule 'round(param_q * param_scale / "
    "accumulator_scale)' is stated in reals, but at |seed| >= 2^20 (k = 4: k*ulp32 >= 0.5) no "
    "float32 evaluation can place the value relative to a .5 boundary, so 'round' is not "
    "checkable to +-1 from the doc alone. INTERPRETATION used as the gate there: the expression "
    "evaluated left-to-right in float32 (q*param_scale, accumulator_scale = s_in*s_w as one "
    "float32 product, then the quotient). A C seed that differs from this reading on such a "
    "channel is reported as a MISMATCH -- an interpretation/doc finding, not waived.",
    "FLOAT32 grad storage receiving a SYM_INT32 raw increment: arithmetic-sym 'Quantized gradient "
    "accumulation' says the accumulate modes 'collapse to the exact float add (no quantization)' "
    "on FLOAT32 targets, but not how the raw int32 at s_raw becomes a float. Assumed value = raw * "
    "s_raw with at most three float32 roundings (the int->float cast, the s_loss*s_in product, the "
    "multiply; the cast is exact for |raw| < 2^24, leaving two); the first store into a zeroed grad "
    "is then that value exactly.",
]



# --------------------------------------------------------------------------- #
# Grid and rounding                                                           #
# --------------------------------------------------------------------------- #


def qmax(q_max_bits: int) -> int:
    """Largest code of a symmetric b-bit grid, 2^(b-1) - 1.

    arithmetic-sym, 'Operand bit-width: int12, not int16': 'For symmetric b-bit
    operands each product is <= 2^(2b-2)' and the table row 'int12 (qMaxBits=12):
    products <= 2047^2' -- i.e. |code| <= 2^(b-1) - 1 = 2047 at b = 12.
    """
    return (1 << (q_max_bits - 1)) - 1


def qmin(q_max_bits: int) -> int:
    """Smallest code of the grid, -2^(b-1) (arithmetic-sym 'Rounding modes, the SYM_INT32 grid...', Grid)."""
    return -(1 << (q_max_bits - 1))


def round_half_away(v: np.ndarray) -> np.ndarray:
    """Round half away from zero, elementwise, float64 -> int64.

    Semantics: arithmetic-sym 'Rounding modes, the SYM_INT32 grid, encode and
    requant' defines HALF_AWAY as C ``round()`` (an exact .5 goes away from
    zero), i.e. sign(v) * floor(|v| + 0.5). WHICH mode a cell rounds with is
    the cell's own ``arith.rounding`` (operation-owned per #282); the
    deterministic cells of this harness declare HALF_AWAY.
    """
    v = np.asarray(v, dtype=np.float64)
    out = np.where(v >= 0.0, np.floor(v + 0.5), np.ceil(v - 0.5))
    return out.astype(np.int64)


@dataclass
class Encoded:
    codes: np.ndarray  # int64
    scale: float  # float64, the documented real-valued scale
    pre: np.ndarray  # float64 pre-rounding values x / scale (for the tie census)


def encode_sym(x: np.ndarray, q_max_bits: int) -> Encoded:
    """Dynamic absmax encode of real values onto a symmetric b-bit grid.

    arithmetic-sym, decision register '#192 D1': 'a dynamic one (dequant ->
    absmax -> fresh scale ...)' and 'D3': 'forward requant maps absmax exactly
    onto qMax, so it never saturates'; arithmetic-sym 'Rounding modes, the
    SYM_INT32 grid, encode and requant' gives scale = absmax / qMax and
    code = clampInt32(roundByMode(x / scale), qMin, qMax). The clamp is a
    no-op under HALF_AWAY (qMin is only produced by SR_HALF_AWAY at u == 0).
    absmax == 0 -> scale 1.0 and all-zero codes (same section).
    """
    x = np.asarray(x, dtype=np.float64)
    absmax = float(np.max(np.abs(x))) if x.size else 0.0
    scale = 1.0 if absmax == 0.0 else absmax / qmax(q_max_bits)
    pre = x / scale
    codes = np.clip(round_half_away(pre), qmin(q_max_bits), qmax(q_max_bits))
    return Encoded(codes=codes, scale=scale, pre=pre)


def dequant(codes: np.ndarray, scale: float) -> np.ndarray:
    """value = code * scale in float64 (tensor, 'int_repr vs dequantize')."""
    return np.asarray(codes, dtype=np.float64) * float(scale)


# --------------------------------------------------------------------------- #
# Conv1d geometry                                                             #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ConvGeometry:
    kernel_size: int
    stride: int = 1
    dilation: int = 1
    padding: str = "VALID"  # VALID | SAME | EXPLICIT
    padding_amount: int = 0  # EXPLICIT only, per side
    groups: int = 1

    @staticmethod
    def from_spec(g: dict) -> "ConvGeometry":
        """Geometry from the cell/driver spec vocabulary (kernelSize, stride,
        dilation, padding, paddingAmount, groups); defaults are PyTorch's."""
        return ConvGeometry(
            kernel_size=int(g["kernelSize"]),
            stride=int(g.get("stride", 1)),
            dilation=int(g.get("dilation", 1)),
            padding=str(g.get("padding", "VALID")),
            padding_amount=int(g.get("paddingAmount", 0)),
            groups=int(g.get("groups", 1)),
        )

    def effective_kernel(self) -> int:
        """Span of a dilated kernel, dilation * (K - 1) + 1 -- the PyTorch
        definition (data-shape 'Weight layouts, orientation and padding')."""
        return self.dilation * (self.kernel_size - 1) + 1

    def pads(self, length: int) -> tuple[int, int]:
        """(padLeft, padRight). VALID: none. EXPLICIT: the amount on each side
        (PyTorch ``padding=N``). SAME: output length ceil(L/stride) with the
        totalPad = max(0, effK + (outLen - 1) * stride - L) split padLeft = floor(totalPad / 2) (data-shape 'Weight layouts, orientation and padding')."""
        if self.padding == "VALID":
            return 0, 0
        if self.padding == "EXPLICIT":
            return self.padding_amount, self.padding_amount
        if self.padding == "SAME":
            out_len = -(-length // self.stride)
            needed = self.effective_kernel() + (out_len - 1) * self.stride
            total = max(needed - length, 0)
            left = total // 2
            return left, total - left
        raise ValueError(f"unknown padding {self.padding!r}")

    def output_length(self, length: int) -> int:
        """floor((L + padLeft + padRight - effectiveKernel) / stride) + 1, the
        standard (PyTorch) formula; 0 when the padded input is shorter than
        the effective kernel (data-shape, same section: 0 when L < effK)."""
        left, right = self.pads(length)
        padded = length + left + right
        eff = self.effective_kernel()
        if padded < eff:
            return 0
        return (padded - eff) // self.stride + 1


def conv1d_int_mac(x_codes: np.ndarray, w_codes: np.ndarray, geom: ConvGeometry) -> np.ndarray:
    """Exact integer cross-correlation of code tensors (int64 accumulation).

    arithmetic-sym, 'Conv1d / Conv1dTransposed SYM_INT32 (#45)': the gather
    forward emits 'raw accumulator-range int32 mantissas at output scale
    s_in*s_w' -- an exact integer sum of code products. Layout and orientation
    per data-shape 'Weight layouts, orientation and padding': x [B, Cin, L], w [Cout, Cin/groups, K],
    y[b, oc, t] = sum_{ic in group(oc), k} x[b, ic, t*stride + k*dilation - padLeft]
    * w[oc, ic - groupBase, k], with out-of-range taps contributing 0 (padding).
    """
    x = np.asarray(x_codes, dtype=np.int64)
    w = np.asarray(w_codes, dtype=np.int64)
    if x.ndim != 3 or w.ndim != 3:
        raise ValueError("x must be [B, Cin, L] and w [Cout, Cin/groups, K]")
    batch, c_in, length = x.shape
    c_out, c_in_per_group, k_size = w.shape
    g = geom.groups
    if c_in % g or c_out % g or c_in // g != c_in_per_group or k_size != geom.kernel_size:
        raise ValueError("groups/kernel geometry inconsistent with the operand shapes")
    left, right = geom.pads(length)
    out_len = geom.output_length(length)
    xp = np.zeros((batch, c_in, length + left + right), dtype=np.int64)
    xp[:, :, left : left + length] = x
    y = np.zeros((batch, c_out, out_len), dtype=np.int64)
    c_out_per_group = c_out // g
    for t in range(out_len):
        start = t * geom.stride
        taps = xp[:, :, start : start + geom.effective_kernel() : geom.dilation]  # [B, Cin, K]
        for grp in range(g):
            ic0 = grp * c_in_per_group
            oc0 = grp * c_out_per_group
            xs = taps[:, ic0 : ic0 + c_in_per_group, :]  # [B, Cin/g, K]
            ws = w[oc0 : oc0 + c_out_per_group]  # [Cout/g, Cin/g, K]
            y[:, oc0 : oc0 + c_out_per_group, t] = np.einsum("bik,oik->bo", xs, ws)
    return y


# --------------------------------------------------------------------------- #
# Forward: raw accumulator wire, bias seed, width restore                     #
# --------------------------------------------------------------------------- #


@dataclass
class BiasSeed:
    seed: np.ndarray  # int64 [Cout]
    pre: np.ndarray  # float64 [Cout], pre-rounding b_q * s_b / s_acc


def bias_seed(b_codes: np.ndarray, s_b: float, s_acc: float) -> BiasSeed:
    """Per-channel bias seed refolded into the accumulator scale.

    arithmetic-sym, 'SYM_INT32 seed-rescale + the #189 guard': 'seed =
    round(param_q * param_scale / accumulator_scale)', recomputed each forward
    and added into the integer accumulator; 'never raw-added' ('Conv1d /
    Conv1dTransposed SYM_INT32'). Rounding mode assumed HALF_AWAY (DOC_GAPS[1]).
    """
    pre = np.asarray(b_codes, dtype=np.float64) * float(s_b) / float(s_acc)
    return BiasSeed(seed=round_half_away(pre), pre=pre)


def bias_seed_f32_literal(
    b_codes: np.ndarray, s_b: float, s_x: float, s_w: float
) -> np.ndarray:
    """The documented seed expression read left-to-right in float32.

    'seed = round(param_q * param_scale / accumulator_scale)' (arithmetic-sym,
    'SYM_INT32 seed-rescale') with accumulator_scale = s_in * s_w ('Conv1d /
    Conv1dTransposed SYM_INT32'): q * s_b first, s_in * s_w as one float32
    product, then the quotient, rounded half away. The doc states the rule
    in reals (DOC_GAPS[0]); this reading is the harness's INTERPRETATION
    (DOC_GAPS[2]) and gates a channel only where the real-valued tie window
    cannot discriminate (k * ulp32(|seed|) >= 0.5, i.e. |seed| >= 2^20 at
    k = 4); elsewhere it is informational. Returns float64 with NaN where
    the float32 evaluation is not finite (the reading is then undefined).
    """
    f = np.float32
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        pre32 = (np.asarray(b_codes).astype(f) * f(s_b)) / (f(s_x) * f(s_w))
    pre64 = pre32.astype(np.float64)
    finite = np.isfinite(pre64)
    out = np.full(pre64.shape, np.nan, dtype=np.float64)
    out[finite] = round_half_away(pre64[finite]).astype(np.float64)
    return out


@dataclass
class RawForward:
    mac: np.ndarray  # int64 [B, Cout, Lout], the pure integer MAC (no seed)
    raw: np.ndarray  # int64, mac + seed[oc]
    s_raw: float  # float64 s_in * s_w
    seed: BiasSeed | None


def conv1d_sym_forward_raw(
    x_codes: np.ndarray,
    s_x: float,
    w_codes: np.ndarray,
    s_w: float,
    b_codes: np.ndarray | None,
    s_b: float | None,
    geom: ConvGeometry,
) -> RawForward:
    """The kernel contract: raw int32-range mantissas at s_in * s_w.

    arithmetic-sym, 'Conv1d / Conv1dTransposed SYM_INT32 (#45)': 'Both emit raw
    accumulator-range int32 mantissas at output scale s_in*s_w (NOT
    range-restored) -- that's the kernel's contract'; the bias enters as the
    refolded per-output-channel seed (``bias_seed``).
    """
    mac = conv1d_int_mac(x_codes, w_codes, geom)
    s_raw = float(s_x) * float(s_w)
    seed = None
    raw = mac
    if b_codes is not None:
        seed = bias_seed(b_codes, s_b, s_raw)
        raw = mac + seed.seed.reshape(1, -1, 1)
    return RawForward(mac=mac, raw=raw, s_raw=s_raw, seed=seed)


@dataclass
class Restored:
    codes: np.ndarray  # int64, width-restored codes
    scale: float  # float64 fresh scale absmax / qMax
    pre: np.ndarray  # float64 pre-rounding values raw * s_raw / scale


def restore_sym(raw: np.ndarray, s_raw: float, q_max_bits: int) -> Restored:
    """Producer-side width restore of a raw wire (dynamic requant).

    arithmetic-sym, 'Conv1d / Conv1dTransposed SYM_INT32 (#45)': 'The executeOp
    epilogue restores operand width at the producer ... OUT_WRITE routes
    SYM->SYM through the conversionMatrix diagonal', and the decision register
    '#192 D1' defines that diagonal as the dynamic requant 'dequant -> absmax
    -> fresh scale'; 'D3': absmax maps exactly onto qMax (no saturation). So
    scale = max|raw * s_raw| / qMax, codes = round_half_away(raw * s_raw / scale).
    absmax == 0 -> scale 1.0 (arithmetic-sym, same section as the encode). The width (q_max_bits) is the
    produced wire's storage config; the rounding mode is the OP's
    ('OUT_WRITE epilogue rounding is operation-owned (#282)': the op's
    arithmetic.roundingMode, for which the storage config is only the
    default) -- HALF_AWAY in every deterministic cell.
    """
    values = np.asarray(raw, dtype=np.float64) * float(s_raw)
    absmax = float(np.max(np.abs(values))) if values.size else 0.0
    scale = 1.0 if absmax == 0.0 else absmax / qmax(q_max_bits)
    pre = values / scale
    codes = np.clip(round_half_away(pre), qmin(q_max_bits), qmax(q_max_bits))
    return Restored(codes=codes, scale=scale, pre=pre)


# --------------------------------------------------------------------------- #
# Linear (step 2): forward, dx, weightGrad raw contracts                      #
# --------------------------------------------------------------------------- #


def linear_int_mac(x_codes: np.ndarray, w_codes: np.ndarray) -> np.ndarray:
    """Exact integer x @ W^T of code tensors (int64), x [B, inF], W [outF, inF].

    arithmetic-sym, 'Operand bit-width': 'matmulIntCore (Linear forward /
    propLoss / weightGrad) ... run on int12 operands' with an int32
    accumulator of code products; the layout [outF, inF] and the W^T
    orientation are data-shape 'Weight layouts, orientation and padding'.
    """
    x = np.asarray(x_codes, dtype=np.int64)
    w = np.asarray(w_codes, dtype=np.int64)
    if x.ndim != 2 or w.ndim != 2 or x.shape[1] != w.shape[1]:
        raise ValueError("x must be [B, inF] and W [outF, inF]")
    return x @ w.T


def linear_sym_forward_raw(
    x_codes: np.ndarray,
    s_x: float,
    w_codes: np.ndarray,
    s_w: float,
    b_codes: np.ndarray | None,
    s_b: float | None,
) -> RawForward:
    """The Linear kernel contract: raw int32-range mantissas at s_in * s_w.

    arithmetic-sym, 'SYM_INT32 seed-rescale': the forward bias seed ('Matmul
    today') is refolded per output column with ``bias_seed`` and added into
    the integer accumulator; the product scale is s_in * s_w (same contract
    as 'Conv1d / Conv1dTransposed SYM_INT32', 'raw accumulator-range int32
    mantissas at output scale s_in*s_w'). The result is returned as
    [B, outF, 1] so the per-output-channel seed/restore resolvers shared with
    Conv1d apply unchanged (channel axis 1).
    """
    mac = linear_int_mac(x_codes, w_codes)[:, :, None]
    s_raw = float(s_x) * float(s_w)
    seed = None
    raw = mac
    if b_codes is not None:
        seed = bias_seed(b_codes, s_b, s_raw)
        raw = mac + seed.seed.reshape(1, -1, 1)
    return RawForward(mac=mac, raw=raw, s_raw=s_raw, seed=seed)


def linear_sym_dx_raw(l_codes: np.ndarray, s_l: float, w_codes: np.ndarray, s_w: float) -> RawForward:
    """dx = L @ W, raw at s_loss * s_w, L [B, outF], W [outF, inF] -> [B, inF, 1].

    arithmetic-sym, 'dx / propLoss (SYM)' (Conv section; Linear 'propLoss' is
    the matmulIntCore sibling): the kernel 'emits the raw s_loss*s_w mantissa
    into the funnel intermediate; the OUT_WRITE epilogue then width-restores
    it at the producer'. The reduction runs over outFeatures ('per-channel
    Linear dx = outFeatures', 'Grouped backward & update'); no bias enters
    the dx (data-shape 'Weight layouts, orientation and padding': dx = dL W).
    """
    l = np.asarray(l_codes, dtype=np.int64)
    w = np.asarray(w_codes, dtype=np.int64)
    if l.ndim != 2 or w.ndim != 2 or l.shape[1] != w.shape[0]:
        raise ValueError("L must be [B, outF] and W [outF, inF]")
    mac = (l @ w)[:, :, None]
    return RawForward(mac=mac, raw=mac, s_raw=float(s_l) * float(s_w), seed=None)


def linear_sym_weight_grad_raw(
    l_codes: np.ndarray, s_l: float, x_codes: np.ndarray, s_x: float
) -> RawForward:
    """weightGrad = L^T @ x, raw at s_loss * s_in, -> [1, outF, inF].

    arithmetic-sym, 'weightGrad (SYM) = strategy A: integer gather into the
    funnel's Phase-2 intermediate at scale s_loss*s_in, then the
    OUT_ACC_DYNAMIC_RESCALE epilogue' (Conv section; 'Two accumulation
    schemes': Linear SYM weight grads are Strategy A too). The reduction is
    over the batch. Returned with a leading batch axis of 1 so the shared
    resolvers see [B=1, C=outF, L=inF].
    """
    l = np.asarray(l_codes, dtype=np.int64)
    x = np.asarray(x_codes, dtype=np.int64)
    if l.ndim != 2 or x.ndim != 2 or l.shape[0] != x.shape[0]:
        raise ValueError("L must be [B, outF] and x [B, inF]")
    mac = (l.T @ x)[None, :, :]
    return RawForward(mac=mac, raw=mac, s_raw=float(s_l) * float(s_x), seed=None)


def linear_sym_bias_grad_raw(l_codes: np.ndarray) -> np.ndarray:
    """biasGrad raw value sum over the batch, int64 [outF], at s_loss.

    arithmetic-sym, 'biasGrad (SYM) = an int32 (batch x outputLength)
    accumulator per output channel' (Conv section; Linear is the batch-only
    case), 'a value-sum' ('Realized framework-wide int12 contract').
    """
    return np.asarray(l_codes, dtype=np.int64).sum(axis=0)


# --------------------------------------------------------------------------- #
# int32 accumulator hazard (F9)                                               #
# --------------------------------------------------------------------------- #

INT32_MAX = 2**31 - 1
INT32_MIN = -(2**31)


def wrap_int32(v: np.ndarray) -> np.ndarray:
    """Two's-complement wrap of int64 values into int32 (what a wrapping
    accumulator ends up holding; signed overflow is UB in C, so this is the
    prediction for a wrapping build, not a documented behaviour)."""
    v = np.asarray(v, dtype=np.int64)
    return ((v + 2**31) % 2**32) - 2**31


def guard_static_proof(n_max: int, bits_a: int, bits_b: int, seed_abs_max: int = 0) -> bool:
    """The SYM int32 accumulator guard's static proof, as the doc states it.

    arithmetic-sym 'SYM int32 accumulator guard (shipped)', item 1: safe iff
    ``N_max * 2^(qa-1) * 2^(qb-1) + max|seed| <= INT32_MAX`` with qa / qb the
    operands' actual widths (the full two's-complement magnitude, because
    -2^(b-1) is reachable through fixed-scale requant saturation) and
    max|seed| the op's actual bias-seed array. Python ints are exact, so
    neither a huge N nor a product past 2^31 wraps. When the proof holds the
    kernel runs its original loop; otherwise the same loop runs checked
    (exact when the values fit, abort at the first real overflow).
    """
    return int(n_max) * (1 << (bits_a - 1)) * (1 << (bits_b - 1)) + int(seed_abs_max) <= INT32_MAX


@dataclass
class OverflowReport:
    fits: bool  # every exact raw accumulator value in [INT32_MIN, INT32_MAX]
    max_abs_raw: int
    n_wrap_full_scale: int  # first reduction length at which sign-aligned full-scale
    # products (qMax_a * qMax_b each) plus the largest |seed| exceed INT32_MAX
    reduction: int
    static_proof: bool = True  # the guard's static bound admits this reduction
    bits: tuple[int, int] = (0, 0)
    seed_abs_max: int = 0

    @property
    def path(self) -> str:
        """Which guard path Leg A predicts: ``proved`` (original loop),
        ``checked`` (checked loop, the values fit: exact result) or
        ``abort`` (the exact accumulator leaves int32: the checked loop
        fires). A fitting reduction whose PARTIAL sums leave int32 would
        also abort on the checked path; the walk order is not documented,
        so the cells keep the partials small (random signs) or one-signed
        and fitting (the all-+qMax probes)."""
        if not self.fits:
            return "abort"
        return "proved" if self.static_proof else "checked"

    def to_json(self) -> dict:
        return {
            "fits": self.fits,
            "max_abs_raw": self.max_abs_raw,
            "n_wrap_full_scale": self.n_wrap_full_scale,
            "reduction": self.reduction,
            "static_proof": self.static_proof,
            "path": self.path,
            "bits": list(self.bits),
            "seed_abs_max": self.seed_abs_max,
        }


def overflow_report(raw: np.ndarray, reduction: int, bits_a: int, bits_b: int, seed_abs_max: int = 0) -> OverflowReport:
    """Does the exact integer accumulator fit int32, would the guard's static
    proof admit the reduction, and where would full-scale operands first wrap?

    arithmetic-sym, 'Operand bit-width': 'an int32 accumulator (~2^31) holds
    only ~2^(33-2b) worst-case product terms', 'int12 is sound only for
    reductions N <= 511', and 'When both operands are absmax-encoded
    (|code| <= 2^(b-1) - 1) the budget is 512 int12 terms (512 * 2047^2 <
    INT32_MAX, 513 wraps)'. So for codes on the HALF_AWAY absmax grid
    n_wrap = floor((INT32_MAX - |seed|) / (qMax_a * qMax_b)) + 1.
    """
    r = np.asarray(raw, dtype=np.int64)
    max_abs = int(np.abs(r).max()) if r.size else 0
    product = qmax(bits_a) * qmax(bits_b)
    n_wrap = (INT32_MAX - int(seed_abs_max)) // product + 1
    return OverflowReport(
        fits=bool(r.size == 0 or (int(r.min()) >= INT32_MIN and int(r.max()) <= INT32_MAX)),
        max_abs_raw=max_abs,
        n_wrap_full_scale=n_wrap,
        reduction=reduction,
        static_proof=guard_static_proof(reduction, bits_a, bits_b, seed_abs_max),
        bits=(bits_a, bits_b),
        seed_abs_max=int(seed_abs_max),
    )


# --------------------------------------------------------------------------- #
# Fixed-scale requant (S11)                                                   #
# --------------------------------------------------------------------------- #


def requant_fixed(codes: np.ndarray, s_in: float, s_target: float, q_max_bits: int) -> Restored:
    """SYM_INT32 -> SYM_INT32 requant into a PRESET target scale.

    arithmetic-sym decision register '#192 D1': 'a fixed-scale one (requant
    into a preset target scale)'; arithmetic-sym 'Rounding modes, the
    SYM_INT32 grid, encode and requant' writes it as roundByMode(clamp(code *
    inScale / targetScale, qMin, qMax)): the clamp is in the float domain
    BEFORE the rounding and there is no integer clamp after it. Under
    HALF_AWAY (every cell here) the clamp bounds are integers, so the rounded
    value stays on the grid; under SR_HALF_AWAY a value saturated at qMin
    could round to qMin - 1 (not exercised: no SR cell). The scale is the
    given target, unchanged.
    """
    values = np.asarray(codes, dtype=np.float64) * float(s_in)
    pre = values / float(s_target)
    out = round_half_away(np.clip(pre, qmin(q_max_bits), qmax(q_max_bits)))
    return Restored(codes=out, scale=float(s_target), pre=pre)
