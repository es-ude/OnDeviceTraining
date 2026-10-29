"""Leg B: exact (float64) math on dequantized values -- no quantization scheme.

The reference is ``torch.nn.functional`` in float64 on ``code * scale``. The
analytic bound per op counts 0.5 LSB per documented rounding point, the
float32 representation term of the produced scale, and -- so that an element
Leg A accepts can never be a Leg B EXCESS -- exactly the deviation the Leg A
policy grants at each rounding point (derived in ``bound_conv1d_forward``).

Imports allowed: numpy, torch, stdlib.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

F32_U = 2.0**-24  # float32 unit roundoff
F32_REL = 2.0**-23  # two units: covers the float32 scale product s_in * s_w


def dequant(codes: np.ndarray, scale: float) -> np.ndarray:
    return np.asarray(codes, dtype=np.float64) * float(scale)


def ulp32(v: float) -> float:
    a = max(abs(float(v)), float(np.finfo(np.float32).tiny))
    return float(np.spacing(np.float32(a)))


def conv1d_exact(
    x: np.ndarray,
    w: np.ndarray,
    b: np.ndarray | None,
    stride: int,
    dilation: int,
    groups: int,
    pad_left: int,
    pad_right: int,
) -> np.ndarray:
    """float64 F.conv1d on dequantized operands with explicit (possibly
    asymmetric) zero padding; returns [B, Cout, Lout] float64."""
    xt = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float64))
    wt = torch.from_numpy(np.ascontiguousarray(w, dtype=np.float64))
    bt = None if b is None else torch.from_numpy(np.ascontiguousarray(b, dtype=np.float64))
    if pad_left or pad_right:
        xt = F.pad(xt, (pad_left, pad_right))
    y = F.conv1d(xt, wt, bias=bt, stride=stride, padding=0, dilation=dilation, groups=groups)
    return y.numpy()


def restore_slack_codes(q_max: int, k: int) -> float:
    """Deviation (in restored codes, beyond the 0.5 of the rounding) a
    layer-tier element may legitimately carry. Derivation:

    The restored code is round(pre32) with pre32 the float32 evaluation of
    pre = raw * s_raw / s_out. If round(pre32) == round(pre) the error
    against pre is <= 0.5 (that is what round(pre) means). If it differs, a
    .5 boundary lies between pre and pre32, so |code - pre| <= 0.5 +
    |pre32 - pre|. The evaluation is at most four float32 roundings
    ((float)raw, the s_in * s_w product, the multiply, the divide):
    |pre32 - pre| <= ((1+u)^4 - 1) * |pre|, and |pre| <= qMax because the
    absmax maps onto qMax. So the physical slack is ((1+u)^4 - 1) * qMax.

    Leg A's census grants a wider window: a mismatch is a tie when pre is
    within k * ulp32(|pre|) of the boundary, i.e. |code - pre| <= 0.5 +
    k * ulp32(|pre|), at most 0.5 + k * ulp32(qMax). A TIE_ONLY element must
    not be an EXCESS, so the bound takes the larger of the two. They are
    not added: the float32 roundings are the MECHANISM that moves pre across
    the boundary, not a second error on top of the window. For k >= 4 the
    window is the larger term (k * ulp32(x) > 4u * x, while the physical
    term is 4u * x + O(u^2) * x; at qMax = 2047 the window is 2^-11 against
    4.8804e-4); the max keeps the bound honest for any k and qMax.
    """
    physical = ((1.0 + F32_U) ** 4 - 1.0) * q_max
    window = k * ulp32(q_max)
    return max(physical, window)


def bound_conv1d_forward(
    tier: str,
    s_out: float | None,
    s_raw: float,
    seed_granted_lsb: float | None,
    y_absmax: float,
    q_max: int | None = None,
    k: int = 4,
) -> float:
    """Analytic bound on |dequant(C output) - exact| for the Conv1d forward.

    Terms (all in value units):
      y_absmax * F32_REL          s_raw is a float32 product in C (<= u
                                  relative on raw * s_raw; two units kept)
      s_raw * (0.5 + g)           bias present: the seed rounding (0.5 raw
                                  LSB) plus g = the deviation the seed policy
                                  grants on the widest channel
                                  (census.ChannelSeed.granted_lsb: 0 pinned,
                                  k * ulp32(|seed|) in the tie window,
                                  SEED_F32_REL * |seed| when the float32
                                  reading gates); again a max over the
                                  channel's alternatives, not a sum
      s_out * (0.5 + slack)       layer tier: the restore rounding plus
                                  ``restore_slack_codes`` (derived there)
    ``seed_granted_lsb`` is None without a bias.
    """
    bound = y_absmax * F32_REL
    if seed_granted_lsb is not None:
        bound += s_raw * (0.5 + float(seed_granted_lsb))
    if tier == "layer":
        if s_out is None or q_max is None:
            raise ValueError("layer tier needs the restored scale and qMax")
        bound += s_out * (0.5 + restore_slack_codes(q_max, k))
    elif tier != "kernel":
        raise ValueError(f"unknown tier {tier!r}")
    return bound


# --------------------------------------------------------------------------- #
# Linear (step 2)                                                             #
# --------------------------------------------------------------------------- #


def linear_exact(x: np.ndarray, w: np.ndarray, b: np.ndarray | None) -> np.ndarray:
    """float64 x @ W^T (+ b): x [B, inF], W [outF, inF] -> [B, outF]."""
    y = np.asarray(x, dtype=np.float64) @ np.asarray(w, dtype=np.float64).T
    if b is not None:
        y = y + np.asarray(b, dtype=np.float64)[None, :]
    return y


def linear_dx_exact(l: np.ndarray, w: np.ndarray) -> np.ndarray:
    """float64 L @ W: L [B, outF], W [outF, inF] -> [B, inF]."""
    return np.asarray(l, dtype=np.float64) @ np.asarray(w, dtype=np.float64)


def linear_weight_grad_exact(l: np.ndarray, x: np.ndarray) -> np.ndarray:
    """float64 L^T @ x: -> [outF, inF]."""
    return np.asarray(l, dtype=np.float64).T @ np.asarray(x, dtype=np.float64)


def bias_grad_exact(l: np.ndarray) -> np.ndarray:
    """float64 batch sum of L -> [outF]."""
    return np.asarray(l, dtype=np.float64).sum(axis=0)


def staging_term(s_in: float, q_max_in: int, other_abs_sum_max: float, k: int) -> float:
    """Value-unit error a FLOAT32 operand staged onto the int12 grid can put
    into one output element: every staged code is at most 0.5 + slack LSB
    (``restore_slack_codes`` -- the encode is the same rounding point as the
    restore) from its real value, and the element multiplies it by the other
    operand, so |sum_i (x_i - x^_i) w_i| <= (0.5 + slack) * s_in * sum_i |w_i|,
    maximised over the output elements. Zero without a FLOAT32 operand.
    """
    return (0.5 + restore_slack_codes(q_max_in, k)) * float(s_in) * float(other_abs_sum_max)


def bound_gemm_raw(
    tier: str,
    s_out: float | None,
    s_raw: float,
    seed_granted_lsb: float | None,
    y_absmax: float,
    q_max: int | None = None,
    k: int = 4,
    staging: float = 0.0,
) -> float:
    """``bound_conv1d_forward`` for any GEMM-family raw wire (Linear forward,
    dx, Conv1d forward) plus the staging term of a FLOAT32 operand encoded
    in the funnel prologue (value units, ``staging_term``). Same terms as
    before: the float32 product scale, the bias seed (0.5 + granted), and on
    the layer tier the restore (0.5 + slack) in the restored scale.
    """
    return bound_conv1d_forward(tier, s_out, s_raw, seed_granted_lsb, y_absmax, q_max, k) + staging


def bound_float32_grad(g_absmax: float, n_roundings: int, staging: float = 0.0) -> float:
    """Bound for a FLOAT32-stored grad produced from a SYM_INT32 raw increment:
    the first store into a zeroed grad is the increment's float32 value,
    at most ``n_roundings`` float32 roundings away from raw * s_raw (the
    int->float cast, the s_loss * s_in product, the multiply -- three; see
    sym_ref.DOC_GAPS[11]) -- ((1+u)^n - 1) relative -- plus the staging term
    of a FLOAT32 forward input. The kernel tier has no float at all: its raw
    int32 is exact and only the float32 product scale (F32_REL) applies.
    """
    return g_absmax * ((1.0 + F32_U) ** n_roundings - 1.0) + staging


def bound_requant_roundtrip(s_out: float, q_max: int, k: int) -> float:
    """|value - code * s_out| for an encode / requant onto a fresh or preset
    grid: the rounding (0.5 LSB), the census-legal tie flip (slack), and the
    float32 representation of the scale the codes are read back with
    (|code| <= qMax, scale within u relative): s_out * (0.5 + slack + qMax*u).
    Clamped elements are outside this bound by construction and are
    reported separately.
    """
    return float(s_out) * (0.5 + restore_slack_codes(q_max, k) + q_max * F32_U)


# --------------------------------------------------------------------------- #
# BFP (step 3): dequant, the GEMM family in float64, the fold bound           #
# --------------------------------------------------------------------------- #


def bfp_dequant(codes: np.ndarray, stored: np.ndarray, exponent_bits: int, group_size: int) -> np.ndarray:
    """mantissa * 2^(stored - bias) per flat-storage group, bias =
    2^(exponentBits-1) - 1; exact in float64. Returns the flat float64 image
    (the caller reshapes)."""
    c = np.asarray(codes, dtype=np.int64).ravel()
    g = np.zeros(c.size, dtype=np.int64) if group_size == 0 else np.arange(c.size) // group_size
    e = np.asarray(stored, dtype=np.int64)[g] - ((1 << (exponent_bits - 1)) - 1)
    return c.astype(np.float64) * np.ldexp(1.0, e)


def _t(a: np.ndarray, grad: bool = False) -> torch.Tensor:
    t = torch.from_numpy(np.ascontiguousarray(a, dtype=np.float64))
    return t.requires_grad_(True) if grad else t


def conv_transpose1d_exact(
    x: np.ndarray, w: np.ndarray, b: np.ndarray | None, stride: int, dilation: int, groups: int, output_padding: int
) -> np.ndarray:
    """float64 F.conv_transpose1d, VALID geometry: x [B, Cin, L], w [Cin,
    Cout/groups, K] -> [B, Cout, (L-1)*stride + dilation*(K-1) + output_padding + 1]."""
    y = F.conv_transpose1d(
        _t(x), _t(w), bias=None if b is None else _t(b), stride=stride, padding=0,
        output_padding=output_padding, groups=groups, dilation=dilation,
    )
    return y.numpy()


def gemm_family_exact(op: str, x: np.ndarray, w: np.ndarray, b: np.ndarray | None, geom: dict) -> np.ndarray:
    """Forward of the three GEMM-family ops in float64. ``geom`` carries
    stride/dilation/groups/pad_left/pad_right (conv1d) or
    stride/dilation/groups/output_padding (convt1d); ignored for linear."""
    if op == "linear":
        return linear_exact(x, w, b)
    if op == "conv1d":
        return conv1d_exact(x, w, b, geom["stride"], geom["dilation"], geom["groups"], geom["pad_left"], geom["pad_right"])
    if op == "convt1d":
        return conv_transpose1d_exact(x, w, b, geom["stride"], geom["dilation"], geom["groups"], geom["output_padding"])
    raise ValueError(op)


def gemm_family_backward_exact(
    op: str, x: np.ndarray, w: np.ndarray, has_bias: bool, loss: np.ndarray, geom: dict
) -> dict[str, np.ndarray]:
    """float64 autograd of sum(y * loss) for the op's forward on the dequantized
    operands: {'dx': dL/dx, 'weightGrad': dL/dW, 'biasGrad': dL/db}. The
    reductions are exactly the documented backward GEMMs (Linear dx = L @ W,
    weightGrad = L^T @ x, biasGrad = batch sum; conv twins per autograd), so
    the same function serves as the abs-sum oracle when called on |x|, |W|,
    |L| (every term is then non-negative and the result IS sum|a_i b_i|)."""
    xt = _t(x, True)
    wt = _t(w, True)
    bt = _t(np.zeros(_bias_len(op, w, geom)), True) if has_bias else None
    if op == "linear":
        y = xt @ wt.T + (bt[None, :] if bt is not None else 0.0)
    elif op == "conv1d":
        xp = F.pad(xt, (geom["pad_left"], geom["pad_right"])) if (geom["pad_left"] or geom["pad_right"]) else xt
        y = F.conv1d(xp, wt, bias=bt, stride=geom["stride"], padding=0, dilation=geom["dilation"], groups=geom["groups"])
    elif op == "convt1d":
        y = F.conv_transpose1d(
            xt, wt, bias=bt, stride=geom["stride"], padding=0, output_padding=geom["output_padding"],
            groups=geom["groups"], dilation=geom["dilation"],
        )
    else:
        raise ValueError(op)
    (y * _t(loss)).sum().backward()
    out = {"dx": xt.grad.numpy(), "weightGrad": wt.grad.numpy()}
    if bt is not None:
        out["biasGrad"] = bt.grad.numpy()
    return out


def _bias_len(op: str, w: np.ndarray, geom: dict) -> int:
    if op == "convt1d":
        return int(w.shape[1]) * int(geom["groups"])
    return int(w.shape[0])


def bound_bfp_fold(n_terms: int, abs_sum: np.ndarray) -> np.ndarray:
    """The design's walk-order-free bound on |C - exact| for a rounding-free
    BFP reduction (arithmetic-bfp 5.3/7/8): N float32 additions, each rounding
    once (relative 2^-24) on a partial no larger than the absolute sum of the
    terms, the (float)partial rounding of every segment being 2^-24 relative
    to a value bounded by the same absolute sum: N * 2^-24 * (sum_i |a_i b_i|
    + |bias|), elementwise, floored at float32 tiny so an all-zero element is
    not divided by zero. ``abs_sum`` is the same op applied to |operands|
    (plus |bias|). This is the acceptance bound of every BFP GEMM row (the
    design fixes it; it is not widened after a run). Its rounding COUNT is
    heuristic -- see ``bound_bfp_fold_rigorous`` for the derivation that
    counts every rounding point; both figures are reported per row."""
    return np.maximum(n_terms, 1) * F32_U * np.abs(np.asarray(abs_sum, dtype=np.float64)) + np.finfo(np.float32).tiny


def bound_bfp_fold_rigorous(n_terms: int, abs_sum: np.ndarray, with_bias: bool) -> np.ndarray:
    """Every rounding point of the section-5.3 fold counted, walk-order-free:
    with n_seg <= N segments there are n_seg (float)partial casts (each at
    most u * |partial| <= u * sum over that segment, so all casts together at
    most u * S), n_seg fold additions (each at most u * |running sum| <= u *
    S) and one bias-seed addition when a bias is present: (2 * N + 1 + bias)
    * u * S with S = sum_i |a_i b_i| + |bias|. Informational: it shows when an
    EXCESS against the design's N-count bound is a float32 rounding-count
    effect rather than a scheme loss (reachable only at tiny N with a bias,
    where 2 or 3 roundings meet a bound of N = 1 or 2 units)."""
    n = max(int(n_terms), 1)
    count = 2 * n + (1 if with_bias else 0)
    return count * F32_U * np.abs(np.asarray(abs_sum, dtype=np.float64)) + np.finfo(np.float32).tiny


def bound_float32_kernel(n_terms: int, abs_sum: np.ndarray, with_bias: bool) -> np.ndarray:
    """Bound for the plain float32 GEMM kernel (the fake-quant twin of B5): N
    products each rounding once (u * |a_i b_i|, together <= u * S), N
    additions each rounding once on a running sum <= S, plus the bias
    addition: (2 * N + 1 + bias) * u * S -- stated before the run; the twin's
    own gate is bit-equality between the two C runs, not this bound."""
    n = max(int(n_terms), 1)
    count = 2 * n + (1 if with_bias else 0)
    return count * F32_U * np.abs(np.asarray(abs_sum, dtype=np.float64)) + np.finfo(np.float32).tiny
