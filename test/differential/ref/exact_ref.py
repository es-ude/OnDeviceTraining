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
