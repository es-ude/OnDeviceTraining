"""Reference legs of the differential numerics harness (test/differential).

Discipline (what makes Leg A independent): the modules in this package import
only numpy, torch and the standard library. They never import
``test/unit/goldgen/sym_gold.py`` or any ``generate_expected_*.py``, and they
are written from ``docs/conventions/*.md`` only -- every function docstring
cites the section it implements. Where the documentation is silent the gap is
recorded in ``DOC_GAPS`` of the module (and surfaced in the report), not
filled in by reading the C sources.

- ``sym_ref``   Leg A for SYM_INT32 (``docs/conventions/arithmetic-sym.md``)
- ``bfp_ref``   Leg A for BFP (``docs/conventions/arithmetic-bfp.md``)
- ``exact_ref`` Leg B: float64 math on dequantized values
- ``census``    mismatch census and the result taxonomy
- ``sr_stats``  stochastic-rounding statistics -- stub
"""
