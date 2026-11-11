#!/usr/bin/env python3
"""The one Python copy of the canonical dyadic-scale rule (DyadicScale.h,
ScaleCanon.h), imported by sym_gold.py and the gold generators.

A stored SYM / SYM_INT32 / ASYM scale is an exact 15-bit dyadic value
m * 2^e with 16384 <= m <= 32767, held in a float32. Each rule rounds ONCE:

- fit_scale(absmax, hi): a grid fitted to float data. Mirrors
  canonFitScale: the float32 quotient absmax / hi FIRST, then its float32
  bits rounded UP to 15 bits (CEIL). 1.0 for absmax == 0.
- fit_scale_from_codes(in_scale, max_abs_code, hi): a grid fitted to integer
  codes, CEIL of the exact rational in_scale * max_abs_code / hi
  (canonFitScaleFromCodes / odtScaleMulRatio).
- product_scale(a, b, mode): a stored product, rounded once from the exact
  mantissa product (odtScaleMul) -- never the float32 product rounded again.
- ratio_scale(s, num, den): s * num / den, NEAREST (odtScaleMulRatio).
- coeff_scale(s, factor): s times a coefficient canonicalized NEAREST.
- canon(x, mode): odtScaleCanonFloat, the float32 bits rounded to 15 bits.

NEAREST rounds half away from zero (scales are positive: half up); CEIL and
FLOOR are directed. Every helper returns a Python float that is exactly the
float32 the C side stores.
"""
from fractions import Fraction

import numpy as np

NEAREST = "NEAREST"
CEIL = "CEIL"
FLOOR = "FLOOR"

M_MIN = 16384
M_MAX = 32767


def _round15(q: Fraction, mode: str):
    """Round the exact positive rational q to m * 2^e, m in [2^14, 2^15)."""
    assert q > 0, f"scale must be positive, got {q}"
    e = q.numerator.bit_length() - q.denominator.bit_length() - 15
    # normalize so 2^14 <= q / 2^e < 2^15
    while q / Fraction(2) ** e >= 2 ** 15:
        e += 1
    while q / Fraction(2) ** e < 2 ** 14:
        e -= 1
    mq = q / Fraction(2) ** e
    m = mq.numerator // mq.denominator
    rem = mq - m
    if mode == NEAREST:
        m += 1 if rem >= Fraction(1, 2) else 0
    elif mode == CEIL:
        m += 1 if rem > 0 else 0
    elif mode == FLOOR:
        pass
    else:
        raise ValueError(mode)
    if m > M_MAX:
        m = M_MIN
        e += 1
    assert -140 <= e <= 113, f"scale {q} outside the normal float range"
    return m, e


def _value(m: int, e: int) -> float:
    v = float(Fraction(m) * Fraction(2) ** e)
    assert float(np.float32(v)) == v
    return v


def decompose(x: float):
    """(m, e) of a canonical float32 scale; asserts it IS canonical."""
    m, e = _round15(Fraction(float(np.float32(x))), FLOOR)
    assert _value(m, e) == float(np.float32(x)), f"{x!r} is not a canonical scale"
    return m, e


def is_canonical(x: float) -> bool:
    x32 = float(np.float32(x))
    if not x32 > 0:
        return False
    return canon(x32, NEAREST) == x32


def canon(x: float, mode: str = NEAREST) -> float:
    """odtScaleCanonFloat: x is a float32 value; its bits rounded to 15 bits."""
    x32 = np.float32(x)
    assert float(x32) > 0 and np.isfinite(x32), f"not a positive finite scale: {x!r}"
    return _value(*_round15(Fraction(float(x32)), mode))


def fit_scale(absmax: float, hi: float) -> float:
    """canonFitScale(absMax, hi): float32 quotient, then CEIL of its bits."""
    a32 = np.float32(absmax)
    if a32 == 0:
        return 1.0
    q32 = np.float32(a32 / np.float32(hi))
    return canon(float(q32), CEIL)


def fit_scale_from_codes(in_scale: float, max_abs_code: int, hi: int) -> float:
    """canonFitScaleFromCodes: CEIL(in_scale * max_abs_code / hi), exact."""
    if max_abs_code == 0:
        return 1.0
    s = canon(in_scale, NEAREST)
    return _value(*_round15(Fraction(s) * max_abs_code / hi, CEIL))


def product_scale(a: float, b: float, mode: str = NEAREST) -> float:
    """canonProductScale: odtScaleMul on the canonical operands."""
    return _value(*_round15(Fraction(canon(a)) * Fraction(canon(b)), mode))


def ratio_scale(s: float, num: int, den: int) -> float:
    """canonRatioScale: s * num / den, NEAREST, rounded once."""
    return _value(*_round15(Fraction(canon(s)) * num / den, NEAREST))


def coeff_scale(s: float, factor: float) -> float:
    """canonCoeffScale: s times the NEAREST-canonical factor, NEAREST."""
    return product_scale(s, canon(factor, NEAREST), NEAREST)


def _self_check():
    # S1's directed product: 16385 * 24560 = 24561.49.. * 2^14 -> 24561
    # NEAREST, 24562 CEIL (the float32 product would tie up to 24562).
    a = _value(16385, -14)
    b = _value(24560, -14)
    assert decompose(product_scale(a, b, NEAREST)) == (24561, -14)
    assert decompose(product_scale(a, b, CEIL)) == (24562, -14)
    assert canon(1.0, CEIL) == 1.0 and decompose(1.0) == (16384, -14)
    # CEIL of 1/127 as float32 bits: strictly above the float32 quotient
    f = fit_scale(1.0, 127.0)
    assert is_canonical(f) and f >= float(np.float32(1.0) / np.float32(127.0))
    # carry at m = 32767
    assert decompose(canon(_value(32767, -14) + 2.0 ** -16, CEIL)) == (16384, -13)
    assert decompose(fit_scale_from_codes(1.0, 2 ** 31, 127)) == _round15(
        Fraction(2 ** 31, 127), CEIL)


_self_check()
