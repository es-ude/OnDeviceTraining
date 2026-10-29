"""Leg A for BFP -- STUB (design step 3 fills this in).

To be written from ``docs/conventions/arithmetic-bfp.md`` only:

- pack FLOAT32 -> BFP: the exponent rule of section 3 ("absmax-snap-up": the
  smallest E with absmax / 2^E <= qMax), the two's-complement mantissa of
  section 1, the saturation rule D6 of section 2;
- GEMM-family forward/backward: the rounding-free kernel contract of
  section 5.3 (int32 same-exponent partials, exact power-of-two fold, float32
  accumulation), whose exact-regime reference is the float64 dot product of
  the dequantized operands (``exact_ref``).

Imports allowed: numpy, torch, stdlib. Never ``sym_gold`` or a
``generate_expected_*`` script.
"""

DOC_GAPS: list[str] = []


def not_implemented(*_args, **_kwargs):
    raise NotImplementedError("bfp_ref: BFP Leg A lands in design step 3")
