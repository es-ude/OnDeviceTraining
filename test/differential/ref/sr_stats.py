"""Stochastic-rounding statistical cells -- STUB (not yet implemented).

Planned checks (every result must lie in {floor(x), ceil(x)}; the mean must be
unbiased within 4 sigma): ``roundByMode(x, SR_HALF_AWAY)`` across magnitudes,
the BFP softmax SR shift, and SR write-back drift over many optimizer steps.

Imports allowed: numpy, stdlib.
"""

DOC_GAPS: list[str] = [
    "SR_HALF_AWAY: docs/conventions names the mode and its purpose (#279 dead-zone "
    "escape) but not the dither distribution, its precision domain, or unbiasedness "
    "as a contract.",
]


def not_implemented(*_args, **_kwargs):
    raise NotImplementedError("sr_stats: SR cells are not implemented yet")
