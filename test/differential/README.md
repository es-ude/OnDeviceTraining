# Differential numerics harness

Op-level check of the SYM_INT32 (and, in later steps, BFP) implementations
against two references that are independent of the C code and of the Python
mirror in `test/unit/goldgen/sym_gold.py`:

- **Leg A** -- the *documented* scheme re-implemented from
  `docs/conventions/*.md` only (`ref/sym_ref.py`, `ref/bfp_ref.py`). Catches
  "C does not do what the doc says". Agreement is bit-exact except for +-1 at
  float32 rounding ties (the doc fixes the rounding rule, not the float32
  expression order).
- **Leg B** -- exact float64 math on the dequantized values
  (`ref/exact_ref.py`). Catches "the documented scheme loses more than it
  claims". Agreement is an analytic bound: 0.5 LSB per documented rounding
  point plus the float32 representation of the produced scale.

The C side is one small driver, `odt_diffdrive`, that links the same static
libraries the unit tests use and runs exactly one op per invocation: JSON spec
+ `.npy` integer codes in, codes + scale out (never dequantized floats). Two
tiers: `kernel` calls the raw arithmetic entry (accumulator-range int32 the
funnel never exposes), `layer` goes through `layerFunctions[].forward` and the
`executeOp` funnel (the scheme tier: staging, width restore, rounding owner).

Not a ctest. Outputs under `out/` are gitignored.

## Build and run

```bash
cmake --preset differential                 # ODT_BUILD_DIFFERENTIAL=ON, no unit tests
cmake --build --preset differential         # -> build/differential/test/differential/odt_diffdrive
uv run test/differential/run_cells.py       # all cells; writes out/report.md + out/report.json
```

`run_cells.py` options: `--cells 'test/differential/cells/cell0a_*.json'` to
select cells, `--gold skip` to drop the cross-check against the generated
PyTorch gold header (`--gold auto` regenerates it into `out/gold/` via
`test/unit/layer/generate_expected_conv1d.py`), `--k N` for the tie window
(default 4 ulp32). Exit status 1 if any row is not PASS.

The driver alone: `odt_diffdrive <spec.json> <in_dir> <out_dir>`; every run's
inputs and outputs are kept under `out/runs/<cell>__<tier>/` for inspection.

## Result taxonomy (per row = cell x tier)

| Leg A | meaning |
|---|---|
| `EXACT` | codes bit-equal, scale within 2^-22 relative |
| `TIE_ONLY` | every differing code is +-1 with the reference's pre-rounding value within k ulp32 of a .5 boundary (or a bias-seed channel whose seed sits at such a tie) -- the doc underspecifies float32 order, not a bug |
| `MISMATCH` | a difference away from a tie, or a scale miss -- implementation finding |

| Leg B | meaning |
|---|---|
| `BOUNDED` | max dequantized error inside the analytic bound |
| `EXCESS` | outside the bound -- scheme finding |

A row PASSes when Leg A is `EXACT`/`TIE_ONLY` and Leg B is `BOUNDED` (and the
cell's `expect` block, if any, agrees).

Known limit of the tie rule: the forward bias seed `round(b_q*s_b/(s_in*s_w))`
is large (int12*int12 product scale), and at |seed| >= 2^21 one float32 ulp
is >= 0.25, so *every* seed sits inside the k=4 window and both roundings are
accepted for that channel. The report therefore also prints, informationally
(never gating), how many channels match the documented expression evaluated
left-to-right in float32 (`seed f32-literal`); on cell 0 that reading matched
the C seeds channel-for-channel.

## Cells

`cells/*.json`, one op configuration each (op, direction, tiers, arithmetic,
geometry, operand/output widths, fixture, expectation, optional gold name).

Cell 0 is the calibration cell (Conv1d SYM_INT32 per-tensor forward,
HALF_AWAY, int12 operands and wire): (a) the four fixtures of the real-torch
gold in `test/unit/layer/UnitTestConv1d.c` (`testConv1dForwardSym*`; three
VALID, one EXPLICIT-padding), (b) seeded random codes with random
non-power-of-two scales at batch 1 and 2. If cell 0 does not pass, the harness
is wrong before the C is.

Fixture kinds: `float_values` and `torch_randn` encode floats with the
documented absmax rule (Leg A's own encoder) so the gold fixtures are
reproduced code-for-code; `random_codes` draws int12 codes and float32 scales
directly (no encode semantics involved).

## Adding ops / tiers

- C: add a row to `opTable[]` in `odt_diffdrive.c` (op/tier/direction ->
  runner); runners reuse `loadOperand` (add a dtype arm for packed SYM / BFP
  codes + side-car), `makeQuantization`, and `addOutput` (several outputs per
  run are fine: backward emits dx/dW/db).
- Python: a Leg A function per op in `ref/*.py` (docstring cites the doc
  section; doc silence goes into `DOC_GAPS`, never filled from C), a Leg B
  function + bound in `ref/exact_ref.py`, and the cell JSON.

## Documentation gaps

Leg A records every place `docs/conventions` was silent in
`ref/sym_ref.py::DOC_GAPS` (also printed at the end of `out/report.md`).
Each is an assumption the reference had to make; a C/Leg A disagreement on
one of them is a documentation finding, not an implementation bug.
