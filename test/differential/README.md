# Differential numerics harness

Op-level check of the SYM_INT32 (and, in later steps, BFP) implementations
against two references that are independent of the C code and of the Python
mirror in `test/unit/goldgen/sym_gold.py`:

- **Leg A** -- the *documented* scheme re-implemented from
  `docs/conventions/*.md` only (`ref/sym_ref.py`, `ref/bfp_ref.py`). Catches
  "C does not do what the doc says". On the kernel tier the raw int32 wire
  must agree bit-exactly; the only deviation accepted anywhere is what the
  doc leaves to float32 -- the bias seed (per output channel, by the seed
  policy below) and, on the layer tier, +-1 at a restore rounding tie (the
  doc fixes the rounding rule, not the float32 expression order).
- **Leg B** -- exact float64 math on the dequantized values
  (`ref/exact_ref.py`). Catches "the documented scheme loses more than it
  claims". Agreement is an analytic bound: 0.5 LSB per documented rounding
  point, the float32 representation of the produced scale, and exactly the
  deviation Leg A grants at each rounding point (derived in
  `ref/exact_ref.py`), so an element Leg A accepts is never a Leg B EXCESS.

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
The driver validates the spec before touching a library entry and exits 2
with the offending key path: operand dtypes and ranks, `kernelSize` vs weight
dim 2, `Cin == groups * weightDim1`, `Cout % groups == 0`, bias `[Cout]`, and
every SYM_INT32 code within `[-qMax, qMax]` of its declared `qMaxBits`.

## Result taxonomy (per row = cell x tier)

| Leg A | meaning |
|---|---|
| `EXACT` | codes bit-equal, scale within 2^-22 relative, every bias channel at the real-valued rounding of the documented seed rule |
| `TIE_ONLY` | the only differences are what the doc leaves to float32: on the layer tier, +-1 codes whose reference pre-rounding value is within k ulp32 of a .5 boundary; on either tier, a bias channel resolved to the other side of a float32-tied seed or to the float32 reading (see the seed policy -- such a channel can deviate by more than one raw LSB from the real-valued rounding; the report prints the largest deviation as `seed dev`) |
| `MISMATCH` | a difference away from a tie, a seed outside the policy, or a scale miss -- implementation finding. The kernel tier also prints the `class`: `seed` (a constant per-channel residual that is not an allowed seed) or `mac` (a non-constant residual, i.e. the integer sum itself) |

| Leg B | meaning |
|---|---|
| `BOUNDED` | max dequantized error inside the analytic bound |
| `EXCESS` | outside the bound -- scheme finding |

A row PASSes when Leg A is `EXACT`/`TIE_ONLY` and Leg B is `BOUNDED` (and the
cell's `expect` block, if any, agrees).

The kernel tier compares raw int32 EXACTLY: an integer accumulator has no
rounding point of its own, so no tie window is applied to it (one was, in an
earlier version, and at |acc| >= 2^20 it waived every +-1 -- a planted
seed+1 mutant passed). The only latitude on that tier is the bias seed, per
channel, judged on the seed's own pre-rounding value.

## Bias-seed policy (an interpretation, recorded as a doc finding)

The forward bias seed `round(b_q * s_b / (s_in * s_w))` is large (an
int12*int12 product scale) and rounded once per output channel in C; its
real-valued pre-rounding value is where the documented rule meets float32.
Per channel (`ref/census.py::seed_policy`):

- the k-ulp32 tie window discriminates (`k * ulp32(|seed|) < 0.5`, i.e.
  |seed| < 2^20 at k = 4) and the value is inside it: either rounding is
  accepted (`alt`);
- it discriminates and the value is outside: the real-valued rounding only
  (`ref`);
- it does not discriminate (|seed| >= 2^20 at k = 4: one ulp32 is >= 0.125,
  so the window covers the whole unit interval and says nothing): the
  documented expression evaluated left-to-right in float32 -- `q * s_b`,
  `s_in * s_w` as one float32 product, then the quotient, rounded half away
  -- is the gate (`f32`). This is an INTERPRETATION the doc does not make
  (it states the rule in reals; `DOC_GAPS[7]` in `ref/sym_ref.py`); a C seed
  that differs from this reading on such a channel is a `MISMATCH` and an
  interpretation/doc finding, not a waiver. Where the float32 reading is not
  finite the channel falls back to either rounding.

The report's `seed ref/alt/f32` column counts channels per outcome, `seed
dev` is the largest |used - real-valued rounding| in raw LSB, and `f32=real`
(informational) counts channels whose float32 reading equals the real-valued
rounding. The policy never enumerates seed combinations: on the kernel tier
the seed decouples per channel; on the layer tier the restore couples channels
only through the fresh scale, so the realizable absmax values (at most
2 x Cout) are enumerated, the observed C scale selects those within 2^-22
relative, and channels are resolved independently under each -- linear in
Cout, so wide layers run.

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

## Leg B bound (Conv1d forward)

`ref/exact_ref.py::bound_conv1d_forward`, in value units:

- `y_absmax * 2^-23` -- `s_raw = s_in * s_w` is a float32 product in C;
- with a bias, `s_raw * (0.5 + g)` -- the seed rounding plus `g`, the
  deviation the seed policy grants on the widest channel (0 pinned,
  `k * ulp32(|seed|)` in the window, `((1+u)^2/(1-u) - 1) * |seed|` for the
  three float32 roundings of the gated reading);
- layer tier, `s_out * (0.5 + max(k * ulp32(qMax), ((1+u)^4 - 1) * qMax))`
  -- the restore rounding plus the larger of the census-legal tie width and
  the four float32 roundings of the restore chain, not their sum: the
  roundings are the mechanism that moves a value across the .5 boundary, not
  a second error on top of the window (derivation in the source).

## Adding ops / tiers

- C: add a row to `opTable[]` in `odt_diffdrive.c` (op/tier/direction ->
  runner); runners reuse `loadOperand` (add a dtype arm for packed SYM / BFP
  codes + side-car), `makeQuantization`, `validate*Operands`, and `addOutput`
  (several outputs per run are fine: backward emits dx/dW/db).
- Python: a Leg A function per op in `ref/*.py` (docstring cites the doc
  section; doc silence goes into `DOC_GAPS`, never filled from C), a Leg B
  function + bound in `ref/exact_ref.py`, and the cell JSON.
- Non-vacuity: every new cell family must be shown to turn red under at
  least one planted `src/` mutation (restored afterwards; `jj diff --stat
  src/` empty). For cell 0: `rescaleIntoAccumulatorScale` returning
  `roundByMode(...) + 1` fails every biased kernel row; a truncating seed
  fails every biased kernel row whose fixture has a seed fraction >= 0.5
  (`cell0a_gold_singleChannelWithBias` has 0.0625 on its single channel, so
  that mutant is numerically identical to the correct code there).

## Documentation gaps

Leg A records every place `docs/conventions` was silent in
`ref/sym_ref.py::DOC_GAPS` (also printed at the end of `out/report.md`).
Each is an assumption the reference had to make; a C/Leg A disagreement on
one of them is a documentation finding, not an implementation bug.
