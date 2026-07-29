# Arithmetic & SYM_INT32 kernels

Conventions for the integer-math path: `src/arithmetic/**` and the SYM kernels of
`src/layer/{Conv1d,Conv1dTransposed,Linear,LayerNorm}*`. Path-scoped for Claude
via `.claude/rules/arithmetic-sym.md`.

## SYM_INT32 seed-rescale + the #189 guard

A SYM_INT32 parameter that must enter an integer accumulator at a *different*
scale — the forward bias seed (Matmul today; Conv when #45 lands) and the
LayerNorm affine beta seed — is converted via `rescaleIntoAccumulatorScale`
(`src/arithmetic/Rounding.c`): `seed = round(param_q * param_scale /
accumulator_scale)`. The `float -> int32` cast is data-dependent and is UB on
overflow (#189); the helper guards it NaN-robustly (`!(x <= T)`, reserving one
worst-case int16 product `32768*32767` of headroom) under `-DODT_SEED_GUARD`
(default ON; a future MCU/release build disables it, with UBSan #204 covering
occurrences). All seed-rescale sites route through this one helper.

This refold is deliberate, not a wart: it holds the real-valued bias **constant**
under ODT's dynamic per-input activation scaling. A fixed integer added raw
(`seed = b_int`, ignoring the bias scale) would apply the bias at
`s_acc / s_bias` of its value (≈0.01-0.05% on real layers — effectively deleting
it) and make it co-scale with input magnitude; the refold recomputes the seed
each forward (`∝ 1/s_acc`) so the bias stays a constant offset. The bias stays
SYM_INT32 (never a float master — the optimizer is single-dtype); a wide
raw-integer bias (qMaxBits=32, scale=1) would need a structurally different
scheme and is out of scope.

## Conv1d / Conv1dTransposed SYM_INT32 (#45)

Two integer sliding-window cores live in `src/arithmetic/`, siblings of the
FLOAT kernels with identical loop nest + `SlidingWindow1d` geometry:

- `conv1dKernelSymInt32` — gather forward; Conv1d forward, and Conv1dTransposed's
  `dx` adjoint in PR3.
- `convTranspose1dKernelSymInt32` — scatter forward; Conv1d's `dx` adjoint, and
  Conv1dTransposed's forward in PR3.

Both emit **raw accumulator-range int32 mantissas** at output scale `s_in·s_w`
(NOT range-restored) — that's the *kernel's* contract. The `executeOp` epilogue
restores operand width at the **producer**, on every migrated path (forward
AND backward: Linear/Conv1d/Conv1dTransposed/LayerNorm forward; the dx wire;
the Quantization layer itself — `OUT_WRITE` routes SYM→SYM through the
conversionMatrix diagonal). A chained Quantization layer after a SYM producer
is therefore **optional**, not required (`validateModelQuantization`'s old
"next layer must be QUANTIZATION" rule is retired, PR1b.2/spec D3) — but
"optional" does not mean "harmless to add anyway". The precise anti-pattern:
**a Quant node with an IDENTICAL config directly consuming a funnel-restored
producer wire** repeats the exact restoration the producer epilogue just
did — under HALF_AWAY this is a redundant no-op computation; under SR it
re-draws stochastic jitter, i.e. pure noise. Three uses of a following
Quant/requant node stay legitimate single-requants, not double-requants:
(a) **width change** `@W1 -> @W2` — though when the target width is known at
the producer, prefer declaring it directly in `outputQ` and letting the
`OUT_WRITE` epilogue requant straight to `@W2`, skipping the Quant node
entirely; (b) **same-width re-normalization after a scale-transparent
segment** (Relu/Dropout/Flatten forward are NOT funnel-routed by design —
element-wise scale-transparent ops that copy/fold the input's scale rather
than deriving a fresh one, `2026-07-03-pr1b2-forward-funnel-design.md` D2 —
so they can leave a tensor underfilled, e.g. Relu zeroing the absmax element;
a Quant node there is the *first* requant of that value, not a second one);
(c) **dtype changes** (SYM -> FLOAT32 etc.). There is no runtime "was this wire already
restored?" check — raw vs. restored isn't reliably detectable from the tensor
alone, and requant legitimately re-normalizes underfilled tensors, so this is
a design discipline for model authors, not an enforced invariant. Per-output-
channel bias is refolded into the product scale via `rescaleIntoAccumulatorScale`
(the #189 guarded helper); never raw-added.

Conv1d backward declares **per-op arithmetic** (`weightGradMath`, `biasGradMath`,
`propLossMath`, by-value `arithmetic_t`), like `linearBackward`, and — like
forward (above) — routes all three through the `executeOp` funnel:

- **weightGrad (SYM)** = strategy A: integer gather into the funnel's Phase-2
  intermediate (a stack-allocated, data-sized VLA scratch inside `executeOp` —
  no `reserveMemory`) at scale `s_loss·s_in`, then the `OUT_ACC_DYNAMIC_RESCALE`
  epilogue's `addSymInt32TensorsInplace` into the SYM grad accumulator (fresh
  absmax scale).
- **biasGrad (SYM)** = an int32 `(batch × outputLength)` accumulator per output
  channel, then the `OUT_ACC_FIXED_SCALE` epilogue's
  `rescaleIntoAccumulatorScale(sum, s_loss, s_bg, mode)` at the bias-grad's
  fixed scale (the #218 scheme).
- **dx / propLoss (SYM)** = `convTranspose1dKernelSymInt32(lossGrad, weights)`
  emits the raw `s_loss·s_w` mantissa into the funnel intermediate; the
  `OUT_WRITE` epilogue then width-restores it at the producer through the
  conversionMatrix diagonal — the dx wire gets the same treatment as every
  other producer wire (above). The old #187 dtype fail-fast (produced propLoss
  tensor must be SYM) is retired: superseded by the funnel's own
  prologue/epilogue and a confirmed tautology post-#221 (zero test coverage
  exercised it).

### Operand bit-width: int12, not int16 (int32-accumulator soundness)

SYM kernels accumulate **products** of operands in an **int32** accumulator (no
int64 — hard rule). For symmetric `b`-bit operands each product is ≤ 2^(2b−2),
so an int32 accumulator (~2^31) holds only ~2^(33−2b) worst-case product terms
before signed overflow (UB):

| operand width | max product | int32 term at which overflow first occurs |
|---|---|---|
| int16 (qMaxBits=16) | 2^30 | 2 |
| int12 (qMaxBits=12) | 2^22 | 512 |
| int8  (qMaxBits=8)  | 2^14 | 131072 |

The number of worst-case terms that still **fit** is one less: int16 survives 1,
int12 survives **511**, int8 survives 131071 — i.e. int12 is sound for reductions
of length **N ≤ 511** (`512·2^22 = 2^31 > INT32_MAX`).

int16×int16→int32 is **unsound for product-accumulation** (forward, dx,
weightGrad) — it overflows after ~2 full-scale terms; it is sound only for
*value* sums (biasGrad). Conv SYM therefore uses **int12 operands**
(`quantizationInitSymInt32WithBits(rm, 12)`): products ≤ 2047² ≈ 4.2e6, ~512-term
int32 headroom — ample for the batch=1 MCU regime ODT targets, matching the
low-bit×low-bit→int32 arithmetic of the Deutel FQT paper (arXiv:2407.10734) /
TFLite. The **grad accumulators stay int16** (wider accumulator, free since SYM
stores int32 regardless of qMaxBits). The **kernels are bit-width-agnostic** —
only the quantization configs change; the int32 accumulator (no int64) is kept.

**Realized framework-wide int12 contract (PR-A, #227):**

- The SYM_INT32 **operand** default is int12 via the compile-time knob
  `ODT_SYM_OPERAND_QMAXBITS` (=12), set in `initSymInt32QConfig`
  (`src/tensor/include/Quantization.h`). Override per-build with
  `-DODT_SYM_OPERAND_QMAXBITS=N` (e.g. =8 for layers wider than 511).
- `matmulIntCore` (Linear forward / propLoss / weightGrad) and the LayerNorm
  **affine product** now run on int12 operands, enforced by op-entry guards
  (`matmulValidateSymOperand` at both Matmul SYM entries;
  `layerNormValidateSymTensor` lowered to the knob). LayerNorm's per-group
  mantissa-sum is a value-sum and stays sound at any qMaxBits ≤ 16.
- **Grad accumulators stay int16** via `ODT_SYM_GRAD_QMAXBITS` (=16), pinned
  in `gradInitSymInt32` (`getQLike` preserves the source width). biasGrad is a
  value-sum; weightGrad is a sum of products (int32 accumulate → requantize).
  Whether grads should be stored SYM_INT32 at all is under redesign — #261.
- int12 is sound only for reductions **N ≤ 511**; the runtime N-vs-budget check
  is a deferred follow-up. The #189 policy (release runs free, CI UBSan #204)
  backstops residual overflow.
- Note: the conv weightGrad product mixes an int12 input with an int16 grad
  operand under the #218 grad-accumulator scheme — its budget is governed by
  #218/#45, not closed by this operand flip.
- The unit-test gold suite validates the **default** int12/int16 contract
  (`ODT_SYM_OPERAND_QMAXBITS=12`, `ODT_SYM_GRAD_QMAXBITS=16`); building with a
  knob override (e.g. `-DODT_SYM_OPERAND_QMAXBITS=8`) diverges from those gold
  fixtures, which is expected and intentional.

The training loop (`CalculateGradsSequential.c`) allocates each dx-wire buffer
from the **producing layer's declared backward config** (`backwardWireQ`: reads
`propLossQ` storage uniformly for every layer type — Linear/Conv/pools/Relu/
Softmax/Dropout/LayerNorm/Quantization all resolve through the same field;
Flatten and the loss seed pass through the upstream dtype) — #221.
Uniform chains behave exactly as before. The Conv→Quant→…→MSE chain
wiring + FLOAT32-twin convergence check is PR3.

### Conv1dTransposed SYM_INT32 (PR3)

Conv1dTransposed is Conv1d's adjoint with roles swapped, so it reuses BOTH PR2
cores — no new kernels:

- **forward** = `convTranspose1dKernelSymInt32` (the scatter core; its internal
  per-channel bias-seed refold gives ConvT bias for free). Pass `outputPadding`.
- **dx / propLoss** = `conv1dKernelSymInt32` (the gather core, the VALID adjoint);
  the `OUT_WRITE` epilogue width-restores the raw mantissa at the producer,
  same as Conv1d's dx wire above — the old #187 dtype fail-fast is retired
  (superseded by the funnel, PR1b.2).
- **weightGrad** = strategy A: a scatter-style integer gather (ConvT weight layout
  `[Cin, Cout/groups, K]`, index `(ic·outChPerGroup + ocOffset)·K + k`) into the
  funnel's Phase-2 stack-VLA intermediate (no `reserveMemory`) at scale
  `s_in·s_loss`, then the `OUT_ACC_DYNAMIC_RESCALE` epilogue's
  `addSymInt32TensorsInplace` into the SYM grad accumulator.
- **biasGrad** = the same fixed-scale refold as Conv1d (`rescaleIntoAccumulatorScale`
  over the `batch × outputLength` int32 sum).

Backward declares per-op arithmetic (`weightGradMath`/`biasGradMath`/
`propLossMath`, by-value `arithmetic_t`), like `conv1dBackward`/`linearBackward`. Operands are int12, grad
accumulators int16, accumulators int32 — no int64. Conv1dTransposed is VALID-only
(Phase 1), so the adjoint never hits a SAME/EXPLICIT padLeft.

### Arithmetic vs. storage field roles — the executeConvert funnel

As of PR1b (executeOp) and this design split, layer configs carry two conceptual
roles for their quantization fields: **arithmetic** (by-value `arithmetic_t`,
declares the compute representation `forwardMath`/`{weightGradMath, biasGradMath, propLossMath}`)
and **storage** (pointer `quantization_t *`, specifies the dtype/scale/qMaxBits of
produced wires: `outputQ`/`propLossQ`). The `executeOp` funnel reads arithmetic to
dispatch float vs. SYM kernels and configure intermediate scratch; it then routes
the result through `OUT_WRITE` or `OUT_ACC_*` epilogues that honor the storage config's
dtype and qMaxBits on the produced wire. When input and output dtypes differ, a
kernel-less form `executeConvert(input, target)` performs the storage-to-storage
conversion for any populated `conversionMatrix` cell — including previously-aborting
paths (ASYM, BOOL, SYM width-restore), unifying the Quantization layer into a pure
conversion node (#266). This separation keeps arithmetic orthogonal to storage,
enabling per-op compute-dtype divergence (#218 knobs) without ownership complexity.

**`OUT_WRITE` epilogue rounding is operation-owned (#282).** The funnel's
`OUT_WRITE` requant into a quantized target rounds by the op's
`arithmetic.roundingMode`, not the target's storage qConfig. The storage
qConfig's `roundingMode` remains as the DEFAULT the op's rounding derives from
(`arithmeticFromQuantization` copies it), so every layer that derives its
declared math from a quantization behaves exactly as before — the two modes
coincide there. Ops that need a rounding of their own overwrite the field
after deriving; the in-tree case is the optimizer (#279): a deterministic
`HALF_AWAY` write-back silently freezes any parameter whose per-step update is
sub-ULP at its fixed scale (the dead-zone; catastrophic at coarse widths —
sweep: sym4 HALF_AWAY = random-guessing from step 0), so the step functions
set `arithmetic.roundingMode = optim->writeBackRounding` on every param/state
write-back. Factories default the knob to seeded `SR_HALF_AWAY`;
`optimizerSetWriteBackRounding(optim, HALF_AWAY)` is the explicit opt-out.
The storage qConfig stays authoritative everywhere else: storage, inference
and serialization encodes, bare conversions (`executeConvert`/`convertTensor`
— a conversion node's rounding IS a storage encode), and the `OUT_ACC_*`
epilogues (accumulate is a read-modify-write under the accumulator's own
storage grid, whose rounding is part of the grid discipline like scale, D4 —
grad-ACC callers also derive their declared math from the backward WIRE
quantization, not the grad-storage config, so operation-owning ACC would
silently re-route existing grad rounding). The tensor's own qConfig is never
left mutated by an op.

### Validator retirement (PR1b.2)

`ModelValidationApi.c`'s `isAccumulatorRangeSymProducer` + `validateModelQuantization`'s
"a SYM accumulator-range producer (LINEAR/LAYERNORM/CONV1D/CONV1D_TRANSPOSED) not in
the last position must be followed by a QUANTIZATION layer" rule is **retired**: once
every producer's forward restores width at its own wire (above), there is nothing left
for that rule to catch — a following Quant layer is optional, and forcing one directly
after a producer is the double-requant anti-pattern, not a requirement. `validateModelQuantization`
stays as the model-wide validation entry point for future rules; today it only guards
the model array shape (non-NULL model/elements).

### SYM training chains

The training loop allocates each dx-wire buffer from the **producing layer's
declared backward config** (`backwardWireQ`: `propLossQ` storage uniformly for
every layer type — Linear/Conv/pools/Relu/Softmax/Dropout/LayerNorm/Quantization
all resolve through the same field; Flatten and the loss seed pass through the
upstream dtype) — #221. Uniform chains behave exactly as before: a
uniformly-SYM chain (every layer's `forwardMath` declaring `ARITH_SYM_INT32`)
keeps every layer's `propLossQ` match consistent — each producer's
`OUT_WRITE` epilogue (above) width-restores its own dx wire independently,
so the chain stays consistent without the retired #187 cross-layer guard.
As of PR1c, grad *tensor* dtype is decoupled from this: the factory's
grad-storage default is `FLOAT32` regardless of how uniformly SYM the
chain's forward/propLoss path is; a uniform-SYM chain only gets `SYM_INT32`
parameter grads if each layer's `weightGradStorage`/`biasGradStorage` knob is
set explicitly. SYM-trainable conv layers are built via the low-level
`initConv1dTransposedConfigWithWeightsAndBias` with SYM `parameter_t`s (the
high-level factory's KAIMING/uniform init still requires FLOAT32-native
weight/bias storage; grad storage there is a hard-pinned FLOAT32 default too,
overridable by the same knob, #261).
`Conv1dTransposed → Quant → MSE` trains under
`calculateGradsSequential` + `sgdStepM(SYM_INT32)`.

## Pooling SYM_INT32 (#205)

All three pooling layers (MaxPool1d / AvgPool1d / AdaptiveAvgPool1d) carry
funnel-routed `ARITH_SYM_INT32` arms, forward AND backward (dx via `OUT_WRITE`,
Conv1d-dx precedent). The kernels are **integer-exact** — no products, so the
no-int64 rule is trivially satisfied — and differ only in how the divisor is
handled:

- **MaxPool1d** — pure mantissa select: argmax over int32 mantissas IS argmax
  over values (scale > 0 preserves order); the scale is copied to the raw
  intermediate (ReLU idiom). Tie-break matches the FLOAT32 arm (strict `>`,
  first occurrence wins), but ties live in the QUANTIZED domain: inputs that
  quantize to the same mantissa tie here even when their float values differ,
  so the recorded argmax may deviate from float argmax within quantization
  noise — documented semantics, pinned by an argmax-gold tie fixture.
  Backward scatters loss-grad mantissas to the argmax positions (scale copy).
- **AvgPool1d** — constant divisor K: exact scale fold `s_out = s_in / K`
  (Dropout idiom), mantissas carry the raw window sum. Zero kernel rounding;
  `count_include_pad=true` falls out for free (padded positions add 0 to the
  sum, the fold keeps the divisor at K). Backward is the transpose with the
  same fold (`s_dx = s_gy / K`). The raw wire is accumulator-range (bounded
  by K·qMax — plain sums, no headroom concern for realistic K).
- **AdaptiveAvgPool1d** — per-window element count varies, so no single fold
  exists: rounded INTEGER division of the mantissa sum
  (`roundedDivHalfAwayInt32`, half-away-from-zero = `roundByMode(HALF_AWAY)`
  semantics without leaving the integer domain), scale unchanged. At most
  0.5 LSB error per element — standard fixed-point practice, not homegrown
  numerics. Backward divides the loss-grad mantissa per window and scatters.

Every arm's producer output is width-restored by the `OUT_WRITE` epilogue
(SYM→SYM conversionMatrix diagonal) like all funnel SYM paths. Gold values are
emulated bit-exactly in the pool generators (`generate_expected_*_pool_1d.py`,
shared int12 helpers + `windowSlice1dAt` emulation in
`test/unit/goldgen/sym_gold.py`); the only C-vs-emulation tolerance is the
restore's float32 expression-order ULP (mantissa ±1, scale rel 1e-4).

The summing arms carry the same **value-sum guard** as every other value-sum
path in-tree (`poolValidateSymValueSum`, per-file like the Reduce/LayerNorm
validators): operand `qMaxBits` in [1,16] AND worst-case summed terms
`< 2^(32-qMaxBits)`, checked against K (AvgPool forward), `ceil(L/O)+1`
(AdaptiveAvgPool forward), and the covering-window count `(effK-1)/stride+1`
resp. `ceil(O/L)+1` for the backward scatters (MaxPool's scatter shares the
AvgPool bound: an argmax hit implies window membership). At the int12 operand
default these bounds are unreachable; they exist for the legal wider-wire
configs (`quantizationInitSymInt32WithBits` allows up to 31 bits, where even a
2-term window sum overflows int32). Death-tested per call site.

## Grouped backward & update — error analysis and path choice

Group-quantized weights (group-quant epic PR2/PR3, spec
`docs/superpowers/specs/2026-07-28-group-quantization-design.md`) keep raw
int32 MACs *within* a group and fold partials into the common accumulator
scale `s_acc = s_in · max_g(scales[g])` via `rescaleIntoAccumulatorScale`
(factors `scales[g]/s_wmax ≤ 1`, so combines never grow mantissas). Every
combine is one rounding of ≤ 0.5·ulp(s_acc). Per output element with C
combine-roundings: worst case |err| ≤ 0.5·C·s_acc; under the
independent-rounding assumption RMS ≈ 0.5·√C·s_acc. C by path:

| path | structure | C per output element |
|---|---|---|
| forward gather (Linear/Conv1d) | running group-partial | groups crossed by the reduction; per-channel = **1** |
| dx gather (Linear dx, ConvT1d dx) | running group-partial, strided/adjoint orientation | distinct groups crossed; per-channel Linear dx = **outFeatures** |
| scatter (ConvT1d forward, Conv1d dx) | per-product rescale (scatter targets change per product — no running partial exists) | contributing products; worst case ≈ IC·K |

The RELATIVE error stays in the single-quantization-step class: the signal is
itself a sum of C-many O(qMax²·s_acc) terms, so √C rounding noise rides on a
√C-growing signal. This is a deliberate, documented deviation with no
literature precedent for per-group *backward* (TFLite/CMSIS have no backward;
Deutel arXiv:2407.10734 is per-tensor) — per the research-deviation rule.
When the noise matters (coarse groups, deep chains), `propLossMath =
ARITH_FLOAT32` is the exact alternative: the funnel prologue dequantizes the
grouped weight per-group (exact per element) and computes in float. The
integer path exists per ratified spec decision D2; convergence validation at
scale is the #300-methodology sweep (PR5, ≥10 seeds, Leo).

**Optimizer updates on grouped params** (SGD/AdamW, PR3): the value-space
update dequantizes the param per-group in the FLOAT32 prologue, updates in
float, and the OUT_WRITE epilogue re-derives **fresh per-group absmax scales**
on every write-back (grouped `packFloatBufferAsSym`). This is Deutel's
dynamic scale re-derivation (Sec. IV-E, weights every SGD update) generalized
per-group — scales track the data every step, consistent with the dynamic
Strategy-A discipline below. A fixed-grid per-group write-back (scale held
constant, mantissa-only update) is future work if the PR5 sweep shows
scale-churn artifacts.

## Quantized gradient accumulation — known precision Open Problem

As of the quantized-gradient prerequisite (`gradInit`, 2026-06-05) a trainable
layer's parameter gradient can be stored in `param->grad->quantization`'s dtype
(grad-storage axis); the three per-op fields `weightGradMath`/`biasGradMath`/
`propLossMath` are the declared backward *arithmetic*.
Accumulation is the `OUT_ACC_DYNAMIC_RESCALE` epilogue mode of `executeOp` (no
longer inside kernels). For SYM_INT32 grads, the per-microbatch accumulation
delegates to the funnel's rescaling epilogue, which dequantizes both the running
grad and the new microbatch grad to float, adds, and re-quantizes the running sum
to a new absmax-derived scale **on every microbatch**.

This is functionally correct end-to-end today, but **not** numerically ideal:

- Quantization noise compounds with the number of microbatches M.
- The running-sum absmax is pinned by the heaviest microbatch, coarsening the
  LSB for the accumulated small-gradient mass.

Preliminary characterization (internal simulation, M=100, N=64, σ=1e-3 with a
10% ×50 heavy tail — *problem characterization only, not a basis for a chosen
solution*):

| Strategy | Final rel. error vs float64 | Float-free? |
|---|---|---|
| A — dynamic-rescale (current) | ~1.5e-4, **grows with M** (2.0e-5 @ step1 → 1.7e-4 @ step100) | No |
| B — fixed-scale integer accum | ~9.9e-5 | Yes |
| C — float accum, quantize-at-read | ~2.2e-5 | No |

We deliberately ship strategy A now and do **not** adopt B/C or any homegrown
numerical scheme. The resolution path is a literature review (stochastic-rounding
accumulators, error-feedback / residual accumulation, higher-precision master
grads, block/group scaling, …) → implement or improve a **published** technique.
Tracked as a separate research task (#218). This note is intentionally public
(not buried in a private spec) so contributors hitting accuracy issues in
quantized training know this is a known, expected limitation rather than a bug.

### Two accumulation schemes in-tree (both intentional)

- **Strategy A (dynamic-rescale)** — Linear SYM weight grads and LayerNorm
  gamma/beta grads: per-microbatch `addSymInt32TensorsInplace` (dequantize
  both operands with their own scales, float-add, requantize the running sum
  to a fresh absmax scale). Not float-free.
- **Fixed-scale integer accumulation** — Linear SYM bias grads
  (`linearCalcBiasGradsSymInt32`): increments are rescaled into the running
  grad's EXISTING scale and added in integer arithmetic; the scale is never
  re-derived during accumulation. The coarser resolution (LSB pinned by the
  running scale, which inits to 1.0) is inherent to the scheme.

The two schemes are now the named funnel epilogue modes chosen per call in layer
code: `OUT_ACC_DYNAMIC_RESCALE` (Strategy A) and `OUT_ACC_FIXED_SCALE` (Linear
SYM bias). On FLOAT32 targets, both collapse to the exact float add (no
quantization).

  **Attribution note:** this fixed-scale integer bias-GRADIENT accumulation is
  ODT's own construction and is NOT prescribed by Deutel et al.
  (arXiv:2407.10734). The paper's quantization is *dynamic*: scales are
  re-derived from observed data — weights every SGD update (Eqs. 6-7) — and the
  method is framed throughout as "dynamic adaptation of the zero-point and
  scale parameters" (Sec. IV-E). The paper has a forward bias (int32 bias on
  the int32 MAC accumulator, Fig. 2) but describes no bias-*gradient*
  accumulation, and it nowhere states that any scale is held static *during
  training* (the only static/PTQ mention is post-training, at deployment) — so
  absent evidence to the contrary, assume its scales are dynamic. ODT's
  fixed-scale bias-grad scheme, which never re-derives the scale during
  accumulation, therefore DEVIATES from the paper's dynamic scaling; the ODT
  scheme that corresponds to Deutel is Strategy A (dynamic-rescale, above).
  What ODT also follows from Deutel: per-layer error requant (~Eq. 4) and the
  float-space SGD step (~Eqs. 5-7). Scheme choice + the init-scale resolution
  limit: #218.

This is a research framework: deliberate scheme differences like this one
MUST be documented here, so experimental design stays separable from
accidental inconsistency. LayerNorm uses strategy A for BOTH gamma and beta
per the 2026-06-05 LayerNorm spec.


