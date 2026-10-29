# Differential numerics harness

Op-level check of the SYM_INT32 (and, in later steps, BFP) implementations
against two references that are independent of the C code and of the Python
mirror in `test/unit/goldgen/sym_gold.py`:

- **Leg A** -- the *documented* scheme re-implemented from
  `docs/conventions/*.md` only (`ref/sym_ref.py`, `ref/bfp_ref.py`). Catches
  "C does not do what the doc says". On the kernel tier the raw int32 wire
  must agree bit-exactly; the only deviation accepted anywhere is what the
  doc leaves to float32 -- the bias seed (per output channel, by the seed
  policy below), on the layer tier +-1 at a restore rounding tie (the doc
  fixes the rounding rule, not the float32 expression order), and for a
  FLOAT32-stored grad the float32 roundings its production can take.
- **Leg B** -- exact float64 math on the dequantized values
  (`ref/exact_ref.py`). Catches "the documented scheme loses more than it
  claims". Agreement is an analytic bound: 0.5 LSB per documented rounding
  point, the float32 representation of the produced scale, and exactly the
  deviation Leg A grants at each rounding point (derived in
  `ref/exact_ref.py`), so an element Leg A accepts is never a Leg B EXCESS.

The C side is one small driver, `odt_diffdrive`, that links the same static
libraries the unit tests use and runs exactly one op per invocation: JSON spec
+ `.npy` integer codes (or float32 values for a FLOAT32-stored operand) in,
codes + scale (or float32 values for a FLOAT32 grad) out -- never dequantized
floats where a scale exists. Three tiers: `kernel` calls the raw arithmetic
entry (accumulator-range int32 the funnel never exposes), `layer` goes
through `layerFunctions[].forward/backward` and the `executeOp` funnel (the
scheme tier: staging, width restore, rounding owner, grad accumulate), and
`convert` runs one conversion primitive in isolation (encode, dynamic
requant, fixed-scale requant).

Not a ctest. Outputs under `out/` are gitignored.

## Build and run

```bash
cmake --preset differential                 # ODT_BUILD_DIFFERENTIAL=ON, no unit tests
cmake --build --preset differential         # -> build/differential/test/differential/odt_diffdrive
uv run test/differential/run_cells.py       # all cells; writes out/report.md + out/report.json
```

`run_cells.py` options: `--cells 'test/differential/cells/s1_*.json'` to
select cells, `--gold skip` to drop the cross-check against the generated
PyTorch gold header (`--gold auto` regenerates it into `out/gold/` via
`test/unit/layer/generate_expected_conv1d.py`; Conv1d cells only), `--k N`
for the tie window (default 4 ulp32). Exit status 1 if any row is not PASS.

The driver alone: `odt_diffdrive <spec.json> <in_dir> <out_dir>`; every run's
inputs and outputs are kept under `out/runs/<cell>__<tier>/` for inspection.
The driver validates the spec before touching a library entry and exits 2
with the offending key path: operand dtypes and ranks, Conv1d `kernelSize`
vs weight dim 2, `Cin == groups * weightDim1`, `Cout % groups == 0`, bias
`[Cout]`; Linear input `[B, inF]`, weight `[outF, inF]`, bias `[outF]`, loss
`[B, outF]`, per-tier storage dtypes; and every SYM_INT32 / packed-SYM code
within its declared grid. The `differential` preset carries no sanitizer and
no optimisation level (host default), so a library fail-fast is exit 1 and
signed-overflow behaviour is that of a plain wrapping build.

## Result taxonomy (per row = cell x tier x output)

| Leg A | meaning |
|---|---|
| `EXACT` | codes bit-equal, scale within the path's derived tolerance, every bias channel at the real-valued rounding of the documented seed rule; a FLOAT32 output equal to the float32 rounding of the reference everywhere |
| `TIE_ONLY` | the only differences are what the doc leaves to float32: on the layer tier, +-1 codes whose reference pre-rounding value is within k ulp32 of a .5 boundary; on either tier, a bias channel resolved to the other side of a float32-tied seed or to the float32 reading (see the seed policy); a FLOAT32 output within the float32 roundings its documented production takes (3 for a weight grad, 2 for a bias grad -- `sym_ref.DOC_GAPS[11]`) |
| `MISMATCH` | a difference away from a tie, a seed outside the policy, or a scale miss -- implementation finding. The kernel tier also prints the `class`: `seed` (a constant per-channel residual that is not an allowed seed) or `mac` (a non-constant residual, i.e. the integer sum itself); `float` for a FLOAT32 output beyond its roundings |
| `MATCH` / `ABORT` / `WRAP` / `SATURATE` | F9 rows only -- a reduction whose EXACT integer accumulator does not fit int32 (Leg A computes this per fixture). The C outcome is classified against `restore(exact)`, a library fail-fast (exit 1), `restore(wrap_int32(exact))` and `restore(clip(exact))`; `WRAP` is the silent overflow the doc's deferred runtime check would catch and is a finding (FAIL). The kernel tier is the authoritative row: on the layer tier the width restore can map different raw wires onto the same codes |

| Leg B | meaning |
|---|---|
| `BOUNDED` | max dequantized error inside the analytic bound |
| `EXCESS` | outside the bound -- scheme finding |

A row PASSes when Leg A is `EXACT`/`TIE_ONLY` and Leg B is `BOUNDED` (and the
cell's `expect` block, if any, agrees); an F9 row PASSes on `MATCH`/`ABORT`.

The kernel tier compares raw int32 EXACTLY: an integer accumulator has no
rounding point of its own, so no tie window is applied to it (one was, in an
earlier version, and at |acc| >= 2^20 it waived every +-1 -- a planted
seed+1 mutant passed). The only latitude on that tier is the bias seed, per
channel, judged on the seed's own pre-rounding value.

Scale tolerances are derived per path from the float32 rounding count BEFORE
a run, never widened after one (`run_cells.py::scale_tol`): kernel tier
2^-22 (one product, two units), layer tier `(1+u)^4 - 1` (restore chain),
staged layer tier `(1+u)^5 - 1` (the staging scale multiplies in), encode
`(1+u)^1 - 1`, dynamic requant `(1+u)^3 - 1`, fixed requant 0 (preset).

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
dev` is the largest |used - real-valued rounding| in raw LSB. The policy
never enumerates seed combinations: on the kernel tier the seed decouples
per channel; on the layer tier the restore couples channels only through the
fresh scale, so the realizable absmax values (at most 2 x Cout) are
enumerated, the observed C scale selects those within the path's tolerance,
and channels are resolved independently under each -- linear in Cout, so
wide layers run. The same resolvers serve Conv1d (`[B, Cout, Lout]`) and
Linear (`[B, outF, 1]`; dx `[B, inF, 1]`; weightGrad `[1, outF, inF]`).

## Cells

`cells/*.json`, one op configuration each (op, direction, tiers, arithmetic,
geometry, operand/output widths, fixture, expectation, optional gold name).

**Cell 0** is the calibration cell (Conv1d SYM_INT32 per-tensor forward,
HALF_AWAY, int12 operands and wire): (a) the four fixtures of the real-torch
gold in `test/unit/layer/UnitTestConv1d.c` (`testConv1dForwardSym*`; three
VALID, one EXPLICIT-padding), (b) seeded random codes with random
non-power-of-two scales at batch 1 and 2. If cell 0 does not pass, the harness
is wrong before the C is.

**Step 2 (Linear SYM_INT32 every direction, encode/requant):**

- `s1_fwd_n{N}_b{B}` -- Linear forward at the production config: packed
  SYM@8 weights and bias, a FLOAT32 input the funnel stages to int12, int12
  output wire; N = inF in {3, 64, 500, 511, 512, 600}, B in {1, 4}, outF 6.
  Kernel tier = `linearForwardSymInt32` on the same codes (SYM_INT32 at 8 /
  12 bits), layer tier = `layerFunctions[LINEAR].forward`. The FLOAT32 input
  is **grid-exact** (values = codes x a 13-bit-significand non-power-of-two
  scale, exact in float32, with a +-2047 in every sample row), so the staged
  codes are deterministic: a staging +-1 shifts the accumulator by a weight
  and is not a tie, so float32 encode ties are kept out of the GEMM cells and
  exercised in S11 instead (the design table's "random non-integer floats"
  is realized as random non-integer dequantized VALUES).
- `s1p_fwd_staging_rows_n64_b4` -- only sample row 0 reaches the grid edge:
  Leg A stages per TENSOR (`DOC_GAPS[10]`); a per-sample staging in C would
  show as a code/scale MISMATCH. Observed: per tensor (EXACT).
- `s1x_fwd_overflow_int12_n{511,512,513,600}`, `s2x_dx_overflow_int12_n{512,513,600}`
  -- the F9 hazard: int12 x int12 with every code +2047, so the exact
  accumulator is N x 2047^2. SYM@8 weights make the hazard unreachable
  (first full-scale wrap at N = 8261, `s1x_fwd_fullscale_sym8_n600` shows
  N = 600 fits), hence int12 weights here. Leg A predicts the first wrap at
  N = 513 (2047^2 x 512 = 2,145,387,008 <= INT32_MAX; the doc's "N <= 511"
  bounds the product by 2^22 and is conservative by one term,
  `DOC_GAPS[12]`); N = 513 is an addition to the design's N set for exactly
  that boundary.
- `s2_dx_n{N}_b{B}` -- Linear dx, the reduction is over outF = N (inF 6),
  int12 loss, packed SYM@8 weights, int12 dx wire.
- `s3_dw_n{N}_b{B}` -- Linear weightGrad with FLOAT32 grad storage (the
  production default), N = inF in {3, 64, 600}, B in {1, 4} (the reduction
  is over B); the biasGrad rides along as a second row. Kernel tier = the
  raw int32 `loss^T @ x` and the raw bias value sum; layer tier = one
  `layerFunctions[LINEAR].backward` whose ACC epilogue stores the float32
  value of the increment into the zeroed FLOAT32 grads.
- `s11a_encode_b{12,16}`, `s11b_requant_dyn_{16to12,12to16,12to12,16to16}`,
  `s11c_requant_fixed_{12to12,16to12,16to16}` -- the convert tier. Encode:
  64 float32 values, 17 within +-4 ulp32 of a .5 boundary (incl. qMax-0.5),
  absmax = qMax x scale with a 13-bit-exact scale. Dynamic requant: the
  absmax code M makes `m * qMax_out / M` an EXACT .5 for m = 4 (mod 8) and a
  k/8 fraction elsewhere (an exact-ties-only fixture could not tell a
  truncating requant from a legitimate tie flip -- found by the mutation
  below). Fixed requant: interior ties, codes around the +-(qMax + 0.5)
  clamp edge and values far outside the target grid. Observed: the fixed
  requant saturates to the two's-complement grid [-2^(b-1), 2^(b-1)-1]
  (negative overflow lands on -2048, i.e. qMin is reachable through it --
  `DOC_GAPS[0]`, `[13]`).

Fixture kinds: `float_values` and `torch_randn` encode floats with the
documented absmax rule (Leg A's own encoder) so the gold fixtures are
reproduced code-for-code; `random_codes` draws int12 codes and float32 scales
directly (Conv1d); `linear_grid` / `full_scale` (Linear, above); `convert`
(S11, above).

## Leg B bounds

`ref/exact_ref.py`, in value units:

- GEMM raw wire (Conv1d forward, Linear forward, Linear dx;
  `bound_gemm_raw`): `y_absmax * 2^-23` (`s_raw = s_a * s_b` is a float32
  product in C); with a bias `s_raw * (0.5 + g)` (the seed rounding plus
  `g`, the deviation the seed policy grants on the widest channel); layer
  tier `s_out * (0.5 + max(k * ulp32(qMax), ((1+u)^4 - 1) * qMax))` (the
  restore rounding plus the larger of the census-legal tie width and the
  four float32 roundings of the restore chain, not their sum: the roundings
  are the mechanism that moves a value across the .5 boundary, not a second
  error on top of the window); plus, for a FLOAT32 input staged to int12 on
  the layer tier, `staging_term` = `(0.5 + slack) * s_in * max_o sum_i |W[o,i]|`
  (every staged code is at most 0.5 + slack LSB from its real value, times
  the other operand). Grid-exact fixtures never use this term; it is kept so
  the bound is honest for any FLOAT32 input.
- FLOAT32 grad (`bound_float32_grad`): `g_absmax * ((1+u)^n - 1)` with n = 3
  (weight grad: int->float cast, scale product, multiply) or 2 (bias grad),
  plus the staging term with `max_o sum_b |L[b,o]|` for the weight grad. The
  kernel-tier grads are raw int32 at `s_raw`: only the float32 product scale
  applies (`y_absmax * 2^-23`; the bias-grad scale is copied verbatim).
- Encode / requant round trip (`bound_requant_roundtrip`):
  `s_out * (0.5 + slack + qMax * u)`; elements the fixed requant clamps are
  outside any round-trip bound by construction and are excluded (reported
  as `n_clamped_excluded` with their own max error).

## Adding ops / tiers

- C: add a row to `opTable[]` in `odt_diffdrive.c` (op/tier/direction ->
  runner); runners reuse `loadOperand` (dtype arms FLOAT32 / SYM_INT32 /
  packed SYM; add BFP codes + side-car), `makeQuantization`,
  `validate*Operands`, and `addOutput` (several outputs per run are fine:
  the Linear backward emits propLoss/weightGrad/biasGrad; a FLOAT32 output
  is written as float32 values).
- Python: a Leg A function per op in `ref/*.py` (docstring cites the doc
  section; doc silence goes into `DOC_GAPS`, never filled from C), a Leg B
  function + bound in `ref/exact_ref.py`, a fixture builder + row evaluator
  per op family in `run_cells.py`, and the cell JSON.
- Non-vacuity: every new cell family must be shown to turn red under at
  least one planted `src/` mutation (restored afterwards; `jj diff --stat
  src/` empty). Record so far:
  - cell 0: `rescaleIntoAccumulatorScale` returning `roundByMode(...) + 1`
    fails every biased kernel row; a truncating seed fails every biased
    kernel row whose fixture has a seed fraction >= 0.5
    (`cell0a_gold_singleChannelWithBias` has 0.0625 on its single channel, so
    that mutant is numerically identical to the correct code there).
  - S1/S2/S3 (both tiers): `matmulIntCore` adding `aValue` to the first
    term -> MISMATCH class `mac` on every forward/dx/weightGrad row (the
    bias-grad rows stay green: not their path); `s1x_..._n512` (fits) fails
    its scale check, `s1x_..._n513` reclassifies from WRAP to MISMATCH.
    `matmulSymInt32TensorsWithBias` scale `* 1.0001f` -> every S1 row fails
    the scale check (Leg B stays BOUNDED on the layer tier: a 1e-4 scale
    error is below int12 resolution -- the scale check, not the bound, is
    what sees it).
  - S1 layer staging and S11a: `convertFloatTensorToSymInt32Tensor` scale
    `absMax / (qMax + 1)` -> both encode cells and the staged S1 layer row
    MISMATCH; the S1 kernel row (codes in) stays green as it must.
  - S11b and the layer-tier restore: `requantSymInt32Tensor` ratio
    `* 1.0001f` -> all four dynamic cells and the S1 layer row MISMATCH;
    truncating the requant (`(int32_t)` for `roundByMode`) -> all four
    dynamic cells and the S2 layer row MISMATCH. Before the k/8 fixture
    change both mutants passed the 12-bit-output dynamic cells (every value
    an exact integer or half: truncation at a tie IS the alt rounding).
  - S11c: removing the clamp in `requantSymInt32TensorToScale` -> all three
    fixed cells MISMATCH.
  - S3 layer (FLOAT32 grads): `dequantChunkToFloat`'s SYM_INT32 arm
    `* 1.0001f` -> every layer weightGrad and biasGrad row MISMATCH; kernel
    rows unaffected.
  - F9 classification: a saturating int64 accumulator in `matmulIntCore`
    turns the kernel-tier WRAP rows into SATURATE (the fitting N = 512 rows
    stay EXACT); the layer rows read MATCH because every output of the
    all-+2047 fixture is identical and the restore maps clip and exact onto
    the same codes -- the kernel tier is the authoritative F9 row.

## Documentation gaps

Leg A records every place `docs/conventions` was silent in
`ref/sym_ref.py::DOC_GAPS` (also printed at the end of `out/report.md`).
Each is an assumption the reference had to make; a C/Leg A disagreement on
one of them is a documentation finding, not an implementation bug. Step 2
added: the Linear weight layout, the staging rule for a FLOAT32 operand under
SYM math (observed per tensor, int12, HALF_AWAY), the float32 dequant of a
raw increment into a FLOAT32 grad, the doc's `N <= 511` versus the exact
first wrap at 513, and the fixed-scale requant's clamp (observed:
two's-complement grid).
