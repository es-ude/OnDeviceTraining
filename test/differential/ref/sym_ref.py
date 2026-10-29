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

DOC_GAPS: list[str] = [
    # Each entry: what Leg A needed, where the doc is silent, what was assumed.
    "qMin (lower clamp bound) of a SYM_INT32 grid: arithmetic-sym states the product "
    "bound 2^(2b-2) and qMax = 2^(b-1)-1 only by example (2047 for int12); the lower "
    "clamp is nowhere stated. Assumed the symmetric two's-complement range "
    "[-2^(b-1), 2^(b-1)-1]; never reached under HALF_AWAY with an absmax-derived scale.",
    "absmax == 0: the dynamic requant (#192 D1 'dequant -> absmax -> fresh scale') does "
    "not say what scale an all-zero tensor gets. Assumed scale = 1.0 (codes all zero).",
    "Float32 expression order of 'round(value / scale)' (encode, dynamic requant, bias "
    "refold): the rule is stated in reals; the float32 evaluation order is not. Leg A "
    "evaluates in float64; disagreements within k*ulp32 of a .5 boundary are ties.",
    "Rounding owner of the forward bias-seed refold: arithmetic-sym says 'seed = "
    "round(param_q * param_scale / accumulator_scale)' without naming the rounding "
    "mode's owner (op vs bias storage). Assumed HALF_AWAY (both coincide in det cells).",
    "Conv1d orientation and weight layout ([Cout, Cin/groups, K], cross-correlation, no "
    "kernel flip) are not stated in docs/conventions; data-shape only says shapes follow "
    "the PyTorch convention. Assumed PyTorch F.conv1d semantics.",
    "SAME padding split (left/right for an even total) is not in docs/conventions "
    "(data-shape.md has no padding section). Assumed TF-minimal: left = floor(total/2), "
    "right = total - left. Not exercised by cell 0 (VALID/EXPLICIT only).",
    "Output length under VALID when L < effective kernel: not documented. Assumed 0.",
    "Bias seed beyond the float32 tie window: the seed rule 'round(param_q * param_scale / "
    "accumulator_scale)' is stated in reals, but at |seed| >= 2^20 (k = 4: k*ulp32 >= 0.5) no "
    "float32 evaluation can place the value relative to a .5 boundary, so 'round' is not "
    "checkable to +-1 from the doc alone. INTERPRETATION used as the gate there: the expression "
    "evaluated left-to-right in float32 (q*param_scale, accumulator_scale = s_in*s_w as one "
    "float32 product, then the quotient). A C seed that differs from this reading on such a "
    "channel is reported as a MISMATCH -- an interpretation/doc finding, not waived.",
    "Definition of HALF_AWAY: the doc names the mode ('half-away-from-zero = roundByMode("
    "HALF_AWAY) semantics', Pooling SYM_INT32) but never writes the function. Assumed the "
    "standard one: sign(v) * floor(|v| + 0.5) (an exact .5 goes away from zero).",
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
    """Smallest code of the grid, assumed -2^(b-1) (see DOC_GAPS[0])."""
    return -(1 << (q_max_bits - 1))


def round_half_away(v: np.ndarray) -> np.ndarray:
    """Round half away from zero, elementwise, float64 -> int64.

    Semantics: arithmetic-sym, 'Pooling SYM_INT32', names the mode
    ('half-away-from-zero = roundByMode(HALF_AWAY) semantics') but does not
    write the function; the standard definition sign(v) * floor(|v| + 0.5)
    is assumed (DOC_GAPS[8]). WHICH mode a cell rounds with is the cell's
    own ``arith.rounding`` (operation-owned per #282), not something the
    doc fixes; the deterministic cells of this harness declare HALF_AWAY.
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
    onto qMax, so it never saturates'. Hence scale = absmax / qMax and
    codes = round_half_away(x / scale); the clamp to [qMin, qMax] is a no-op
    by construction. absmax == 0 -> scale 1.0 (DOC_GAPS[1]).
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
        definition the harness assumes (DOC_GAPS[4]: not in docs/conventions)."""
        return self.dilation * (self.kernel_size - 1) + 1

    def pads(self, length: int) -> tuple[int, int]:
        """(padLeft, padRight). VALID: none. EXPLICIT: the amount on each side
        (PyTorch ``padding=N``). SAME: output length ceil(L/stride) with the
        minimal total pad split floor/ceil (DOC_GAPS[5])."""
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
        the effective kernel (DOC_GAPS[6])."""
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
    assumed PyTorch (DOC_GAPS[4]): x [B, Cin, L], w [Cout, Cin/groups, K],
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
    Conv1dTransposed SYM_INT32'). Rounding mode assumed HALF_AWAY (DOC_GAPS[3]).
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
    in reals (DOC_GAPS[2]); this reading is the harness's INTERPRETATION
    (DOC_GAPS[7]) and gates a channel only where the real-valued tie window
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
    absmax == 0 -> scale 1.0 (DOC_GAPS[1]). The width (q_max_bits) is the
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
