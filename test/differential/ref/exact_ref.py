"""Leg B: exact (float64) math on dequantized values -- no quantization scheme.

The reference is ``torch.nn.functional`` in float64 on ``code * scale``. The
analytic bound per op counts 0.5 LSB per documented rounding point plus the
float32 representation term of the produced scale:

- kernel tier (raw wire at s_raw = s_in * s_w): the integer MAC is exact in
  reals, so |raw * s_raw - exact| <= 0.5 * s_raw * [bias present] (the seed
  rounding) + |exact|max * 2^-23 (s_raw is a float32 product in C);
- layer tier (restored wire at s_out): + 0.5 * s_out for the restore rounding.

Imports allowed: numpy, torch, stdlib.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

F32_REL = 2.0**-23  # one float32 ulp, relative: covers the float32 scale and its product


def dequant(codes: np.ndarray, scale: float) -> np.ndarray:
    return np.asarray(codes, dtype=np.float64) * float(scale)


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


def bound_conv1d_forward(
    tier: str, s_out: float | None, s_raw: float, has_bias: bool, y_absmax: float
) -> float:
    """Analytic bound on |dequant(C output) - exact| for the Conv1d forward."""
    bound = y_absmax * F32_REL
    if has_bias:
        bound += 0.5 * s_raw
    if tier == "layer":
        if s_out is None:
            raise ValueError("layer tier needs the restored scale")
        bound += 0.5 * s_out
    elif tier != "kernel":
        raise ValueError(f"unknown tier {tier!r}")
    return bound
