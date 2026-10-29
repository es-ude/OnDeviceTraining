# Differential numerics harness

Op-level check of the SYM_INT32 and BFP implementations against two
references that are independent of the C code and of the Python mirror in
`test/unit/goldgen/sym_gold.py`:

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
codes + scale (or float32 values for a FLOAT32 grad; codes + per-group
exponents for a BFP wire) out -- never dequantized floats where a grid
exists. Three tiers: `kernel` calls the raw arithmetic entry (accumulator-
range int32 the funnel never exposes; the raw FLOAT32 of a BFP kernel),
`layer` goes through `layerFunctions[].forward/backward` and the `executeOp`
funnel (the scheme tier: staging, width restore, rounding owner, grad
accumulate), and `convert` runs one conversion primitive in isolation
(encode, dynamic requant, fixed-scale requant, BFP pack). Ops: `linear`,
`conv1d`, `convt1d` (BFP only), `convert`.

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
| BFP rows (`bfp_rows.py`) | the same words, read on the BFP contract: `EXACT` = the raw float32 equals the documented fold bit for bit (single-segment reductions are predicted to the bit; multi-segment ones along the assumed walk, `bfp_ref.DOC_GAPS[4]`), a BFP wire / pack equals the exact exponent rule + round/clamp, an fq twin is bit-identical; `TIE_ONLY` = only what the doc leaves open -- class `fold_order` (a multi-segment raw inside the fold bound but not bit-equal to the assumed walk), `E_boundary` (a group exponent off by one inside the F2 window and self-consistent codes), `clamp_floor` (C `-2^(m-1)` where sections 1/2 say `-qMax` and 5.6 says `-2^(m-1)` -- the doc's own inconsistency); `MISMATCH` classes `fold` (raw differs beyond that), `E` / `code` (pack), `guard` (Leg A predicts the headroom fail-fast, the run completed), `abort` (the run failed fast inside the guard), `fq` (the twins differ); `ABORT` = the headroom guard fired where Leg A predicted it (PASS) |

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

**Step 3 (BFP, design cells B1-B5; `family` key dispatches to
`bfp_rows.py`):**

- `b1_pack_m{m}_e{e}_pt` / `_g8` (m in {2,4,6,8,12,16}, e in {2,4,8}) and
  `b1x_census_m{m}_e8_g16` -- B1, the FLOAT32 -> BFP pack on the convert
  tier (`conversionMatrix[FLOAT32][BFP]`). Per-tensor: one random binade.
  Grouped (16 groups of 8): random groups, absmax exactly / +1 / -1 ulp32 of
  `qMax * 2^k` (F2), exact-half ties (HALF_AWAY vs half-even), an all-zero
  group (zero state), the D6 saturation and flush edges (reachable at e < 8;
  at e = 8 the cap is `2^127` and every finite float32 fits for m > 2), and
  a value far below `-qMax` at the cap (`negfloor`). The census cells put 18
  groups at `j` ulp32 off a boundary, `j` in -4..4. Leg A = the section-3
  rule in rational arithmetic + D6 + `+-qMax` clamp; Leg B = round trip
  within half a code at the C's own exponent (clamped elements excluded,
  counted). Observed: every E equals the exact rule (the F2 window never
  opened: a float32 absmax divided by the integer qMax is correctly rounded
  and lands on a power of two only when the exact quotient is one), codes
  exact, and the saturating NEGATIVE values come back as `-2^(m-1)` where
  sections 1/2 say `-qMax` (class `clamp_floor`, 3-5 elements per e < 8
  grouped cell, e.g. m8 e4: value -130048, pre-rounding -508 at the cap
  scale, C -128, Leg A -127) -- the inconsistency section 5.6 already names
  as open (#420), reported, not waived into EXACT.
- `b2_lin_fwd_*`, `b2_conv_fwd_*`, `b2_convt_fwd_*` -- B2, the GEMM-family
  forward under `ARITH_BFP`, kernel tier (`linearForwardBfp`,
  `conv1dKernelBfp`, `convTranspose1dKernelBfpGather` on the funnel's
  unpacked scratch form) and layer tier (packed BFP operands borrowed by the
  funnel, FLOAT32 `outputQ` = the raw). Per-tensor fixtures (one segment per
  output: the fold is predicted to the bit, `expect EXACT`), grouped on both
  operands with group sizes that do and do not divide the row length, mixed
  widths (m12 x m8, m12 x m6, m12 x m4, m8 x m6), e = 4 and 8, conv groups,
  SAME / EXPLICIT padding, stride and dilation, ConvT outputPadding (tail
  positions stay at the bias seed), and the lossy regime of section 7: m16 x
  m8 with a 400-term segment (int32 partial ~1.7e9 > 2^24: the `(float)
  partial` cast rounds), two 200-term segments, and m16 x m16 with groupSize
  1 (every product its own segment, 128 casts above 2^24). `b2x_*_abort_*`
  sit one past `bfpSegmentLimit` (16 x 8 at N = 512; 16 x 16 at N = 2) and
  must fail fast. `b2s_lin_fwd_f32in_*` gives the layer tier a grid-exact
  FLOAT32 input: Decision 1 stages it per-tensor at the weight's widths, so
  its codes must equal the ones the kernel tier receives.
- `b3_*_wire_*` -- B3, the same forwards (and a Linear dx) onto a BFP wire
  (`outputQ` / `propLossQ` BFP, per-tensor and grouped): the OUT_WRITE pack
  of the raw. Leg A packs the PREDICTED raw (bit-exact for the
  single-segment fixtures; for the one multi-segment fixture the fold bound,
  in code units, is the tie window); Leg B = fold bound + half a code.
- `b4_*` -- B4, native backward: Linear dx / weightGrad / biasGrad on both
  tiers (`linearCalcPropLossBfp`, `linearCalcWeightGradsBfp`,
  `linearCalcBiasGradsBfp` raw; the layer's ONE backward into a FLOAT32 dx
  wire and FLOAT32 grads -- the ACC epilogue is the exact float add into a
  zeroed grad, so the layer tier IS the raw tier), Conv1d / ConvT1d dx,
  weightGrad and biasGrad on the layer tier (their grad kernels have no
  public raw entry), batch 2, conv groups, SAME padding (padded weightGrad),
  stride / dilation / outputPadding, per-tensor and grouped operands. Leg B
  = float64 autograd on the dequantized operands within the fold bound.
  `b4x_lin_dw_abort_sum_m16_b2` is the sum-headroom control (fits).
- `b5_fq_*` -- B5, the fake-quant twin: the layer under `ARITH_FLOAT32`
  with BFP-stored weights / bias / loss (m12, grouped) against the same
  layer with every BFP operand replaced by its exact float32
  dequantization. Both runs' outputs must be bit-identical (Leg A); Leg B is
  float64 within the float32 kernel's (2N+1) roundings. Softmax fq (design
  B5 column) is not in this step: the step text names the GEMM family.

Observed in step 3 (all 119 rows PASS, none TIE_ONLY except the `clamp_floor`
pack rows): every raw float32 output of every BFP GEMM-family op, forward
and backward, kernel and layer tier, is BIT-EXACT against the documented fold
evaluated along the assumed walk -- including the section-7 lossy rows, where
the `(float)partial` cast is reproduced by the int64 -> float32 cast. The
assumed walk orders of `bfp_ref.DOC_GAPS[4]` (Linear: reduction index
ascending; Conv1d forward: input channel outer, tap inner; gather: tap outer,
channel inner; conv weightGrad / biasGrad: batch outer, position inner) are
therefore the shipped orders; they remain undocumented. The fq twins are
bit-identical. The design's bound `N * 2^-24 * (sum|a_i b_i| + |bias|)` is
met with ratio <= 4e-4 on every raw FLOAT32 row (the lossy rows' errors are
exactly the cast losses Leg A predicts; the BFP wire rows print ratios up
to 1.00 because their bound is dominated by the half-code pack term, which
the pack rounding uses up); the every-rounding-point count `(2N+1+bias)` is
reported alongside (`ratio_rigorous`) and never needed.

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
- BFP GEMM family (`bound_bfp_fold`, elementwise): the design's
  walk-order-free `N * 2^-24 * (sum_i |a_i b_i| + |bias|)` with `S = sum|.|`
  computed by the same op on the absolute operands (autograd for the
  backward directions), N the nominal reduction length. It is the gate; the
  every-rounding-point count `(2N + 1 + bias) * u * S`
  (`bound_bfp_fold_rigorous`: N casts, N fold additions, the bias add) is
  reported as `ratio_rigorous` so an EXCESS at tiny N with a bias could be
  told apart from a scheme loss (none occurred). A BFP wire adds half a code
  at the C's exponent (the pack rounding). The pack round trip is half a
  code at the C's exponent; the fq twin uses the float32 kernel's
  `(2N + 1 + bias) * u * S` (`bound_float32_kernel`).

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
  per op family in `run_cells.py` (SYM) or `bfp_rows.py` (BFP; a cell's
  `family` key routes it there), and the cell JSON. A BFP operand spec
  carries int32 codes plus `mantissaBits`, `exponentBits`, `groupSize` and
  the stored `exponents`; the kernel tier receives the unpacked scratch form,
  the layer tier a packed tensor (INT32 -> BFP code-domain pack, exponents
  set verbatim), and a BFP output comes back as codes (BFP -> INT32
  int_repr) + exponents.
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
  - Step 3 (BFP). B2/B3/B4 Linear: `matmulBfpTensors`' tail fold shifted by
    one (`ldexpf(..., shift + 1)`) -> every Linear forward / dx / weightGrad
    row on both tiers MISMATCH class `fold` + Leg B EXCESS, the B3 wire row
    MISMATCH class `E+code`; the biasGrad rows (a sum fold in `Linear.c`)
    and the fq twin (a FLOAT32 path) stay green as they must. B2/B4 conv:
    `conv1dKernelBfp`'s tail fold `(float)partial + 1.0f` -> every conv1d
    forward row and both ConvT1d dx rows (which use that kernel) MISMATCH;
    the m8 B3 conv wire stays EXACT (a one-product perturbation of the raw
    is ~1e-3 of a wire code: a packed output cannot see errors below its
    own resolution, by construction). `convTranspose1dKernelBfpGather`'s
    tail fold likewise -> every ConvT1d forward row and both Conv1d dx rows
    MISMATCH. B4 biasGrad: `linearCalcBiasGradsBfp`'s tail fold shift + 1
    -> only the biasGrad rows MISMATCH. B4 weightGrad orientation: removing
    the transpose view in `linearCalcWeightGradsBfp` -> the kernel aborts on
    the shape (MISMATCH class `abort`). B1: dropping the `frac == 0.5`
    correction in `deriveBfpStoredExponent` -> the census cell and the g8
    cell MISMATCH class `E` (every exact-power-of-two absmax gets E + 1),
    the per-tensor random cell stays EXACT (no boundary absmax), and the
    staging probe `b2s_*` MISMATCHes on the layer tier only (the staged E
    moves; the kernel tier gets codes). B1/B3: truncating the pack rounding
    in `packFloatBufferAsBfp` -> every pack cell and both B3 wire cells
    MISMATCH class `code`; `b2s_*` stays EXACT (grid-exact values truncate
    to themselves -- the fixture's determinism, as intended). Headroom:
    loosening `bfpValidateBlockHeadroom` by 100000 -> both `b2x_*_abort_*`
    rows MISMATCH class `guard` (the run completes with wrapped partials).
    B5: `dequantChunkToFloat`'s per-tensor BFP arm `* 1.0001f` -> the Linear
    fq forward (per-tensor bias) MISMATCH class `fq`, the grouped-operand
    twins unaffected; the grouped arm `* 1.0001f` -> every conv / convT /
    Linear-dx twin MISMATCH class `fq`.

## Documentation gaps

Leg A records every place `docs/conventions` was silent in
`ref/sym_ref.py::DOC_GAPS` (also printed at the end of `out/report.md`).
Each is an assumption the reference had to make; a C/Leg A disagreement on
one of them is a documentation finding, not an implementation bug. Step 2
added: the Linear weight layout, the staging rule for a FLOAT32 operand under
SYM math (observed per tensor, int12, HALF_AWAY), the float32 dequant of a
raw increment into a FLOAT32 grad, the doc's `N <= 511` versus the exact
first wrap at 513, and the fixed-scale requant's clamp (observed:
two's-complement grid). Step 3 records the BFP gaps in
`ref/bfp_ref.py::DOC_GAPS`: the lower-clamp inconsistency (`-qMax` in
sections 1/2 vs the shipped `-2^(m-1)` of 5.6 -- observed: `-2^(m-1)`),
HALF_AWAY never written out, the exactness of the pack division, the bias
seed order, the fold / walk order of every op (F10 -- observed: the
in-order walks listed there, all bit-exact), the per-group zero state, the
headroom guard wording, the `(float)partial` rounding mode and the staged
grid's exponent rule.
