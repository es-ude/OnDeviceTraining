"""Stochastic-rounding statistical cells -- STUB (not yet implemented).

Planned checks (every result must lie in {floor(x), ceil(x)}; the mean must be
unbiased within 4 sigma): ``roundByMode(x, SR_HALF_AWAY)`` across magnitudes,
the BFP softmax SR shift, and SR write-back drift over many optimizer steps.

Imports allowed: numpy, stdlib.
"""

DOC_GAPS: list[str] = [
    "SR_HALF_AWAY: docs/conventions/arithmetic-sym.md names the exact formula floor(x + u) "
    "and the dither (u = top 24 bits of a U32 / 2^24, in [0, 1 - 2^-24]) and states it is "
    "computed exactly (no rounded sum x + u), so the mean is unbiased up to the float32 "
    "representation of x; the statistical tolerance of the planned mean check is still not "
    "a documented contract.",
]


def not_implemented(*_args, **_kwargs):
    raise NotImplementedError("sr_stats: SR cells are not implemented yet")
