"""Stochastic-rounding statistical cells -- STUB (not yet implemented).

Planned checks (every result must lie in {floor(x), ceil(x)}; the mean must be
unbiased within 4 sigma): ``roundByMode(x, SR_HALF_AWAY)`` across magnitudes,
the BFP softmax SR shift, and SR write-back drift over many optimizer steps.

Imports allowed: numpy, stdlib.
"""

DOC_GAPS: list[str] = [
    "SR_HALF_AWAY: docs/conventions/arithmetic-sym.md names the formula floorf(x + u) "
    "and the dither (u = top 24 bits of a U32 / 2^24, in [0, 1 - 2^-24]) and calls the "
    "mode unbiased in exact arithmetic, but states no unbiasedness bound for the float32 "
    "add (a large |x| plus a u near 1 rounds up inside the add), so the statistical "
    "tolerance of the planned mean check is not a documented contract.",
]


def not_implemented(*_args, **_kwargs):
    raise NotImplementedError("sr_stats: SR cells are not implemented yet")
