# Block-Floating-Point (BFP) arithmetic — PR1–PR6 conventions + deviations register

Conventions for the `BFP` qtype's dtype-core scope (`src/tensor/Quantization*`,
`src/tensor/TensorConversion.c`'s BFP cells) and, since epic PR2, the native
`ARITH_BFP` compute contract (`src/arithmetic/include/BfpKernelSupport.h`,
`Matmul.c`, `Conv1dKernel.c`, `ConvTranspose1dKernel.c`, `ExecuteOp.c`'s funnel
arms, and the `ARITH_BFP` arms in `src/layer/Linear.c`/`Conv1d.c`/
`Conv1dTransposed.c`). Since epic PR3 this also covers those same layers'
native `ARITH_BFP` **backward** arms (`weightGrad`/`biasGrad`/`dx`), the BFP
grad-accumulate and scale engines in `src/tensor/TensorConversion.c`
(`accumulateIntoBfpFixedGridEngine` and the unified `bfpRescaleWalk` behind
`scaleBfpTensorInPlace` and the rescale accumulate arms), and the
per-tensor-only BFP grad/optimizer-state storage knob (`src/userApi/tensor/TensorApi.c`'s `gradInit`,
`src/userApi/optimizer/{SgdApi,AdamWApi,OptimizerApi}.c`,
`src/userApi/continual_learning/PpcaReplayApi.c`). Since epic PR4 it also
covers the weight-less layers — the native `ARITH_BFP` arms in
`src/layer/{MaxPool1d,AvgPool1d,AdaptiveAvgPool1d}.c`, the packed-domain
(outside-funnel) BFP paths in `src/layer/{Relu,Flatten,Dropout}.c`, and the
BFP fake-quant arms in `src/loss_functions/{MSE,CrossEntropy}.c` — all
documented in §5.7. Since epic PR5 it also covers the norm layers — the
native `ARITH_BFP` arms (forward + `dgamma`/`dbeta`/`dx`) in
`src/layer/{LayerNorm,GroupNorm}.c`, the BFP stats reducers
`meanOverTrailingAxesBfp`/`varianceBiasedOverTrailingAxesBfp` in
`src/arithmetic/Reduce.c`, and the factory coherence rules in
`src/userApi/layer/{LayerNormApi,GroupNormApi}.c` — all documented in §5.8.
Since epic PR6 it also covers Softmax — the native `ARITH_BFP` forward and
backward in `src/layer/Softmax.c`, the I-BERT integer exp
`src/arithmetic/BfpSoftmaxExp.c`, and the `softmaxSetBfpExpShiftRounding`
knob (`SoftmaxApi.h`) — documented in §5.9.

Decision IDs: the epic's design decisions are cited as `D<n>` (code
comments outside this document write `BFP D8` etc.), the epic-PR2
implementation decisions as `Decision N` (outside this document:
`PR2-Decision N`), the epic-PR6 kickoff rulings as `P6-N`, and the epic
design's deviations list as "deviations register N". §12 states every one
that is cited anywhere in the repository; the `R-P*`/`R-N*`/`R-S*` entries are
defined in place in §5.7–§5.9. The design documents themselves are not
published; everything this document relies on from them is stated here.

§§1–4 track where the shipped PR1 dtype-core deliberately
deviates from the cited literature and from ODT's own `#227` discipline — the
Deutel-note format used by `docs/conventions/arithmetic-sym.md`'s attribution
notes, applied to the BFP anchors (HBFP, MSFP, MX, FAST) instead. §5
documents the compute contract itself (PR2 forward + PR3 backward + PR4's
weight-less layers in §5.7 + PR5's norm layers in §5.8 + PR6's Softmax in
§5.9, all shipped, not a
forward pointer); §§6–8 extend
the deviations register with three deviations the PR2 kernels introduced; §9
amends design decision D8 with the PR3 backward's own deviation (exact fold
segmentation instead of op-local re-blocking); §10 records the PR5 norms'
own register entries (the 3× stats recompute, and the float32 stats bridges
that bound what "native" means for a norm); §11 records PR6 Softmax's (the
I-BERT adaptations, and two corrections); §12 is the decision register.

## 1. Two's-complement mantissas, not sign-magnitude

HBFP/MSFP/MX all describe the BFP mantissa as **sign-magnitude**: one sign bit
plus an unsigned magnitude. `bfpQConfig_t`'s packed mantissa payload is
**two's complement** instead — packed via the existing `byteConversionAppend`
codec and read back via the shared `unpackSignExtend(src, srcBits, srcStartBit,
dst, n)` helper, the same sign-extend-on-unpack idiom the `SYM ↔ *` conversion
bridge already uses (`docs/conventions/tensor.md`, #227).

**Attribution note:** this is a deliberate, numerically-equivalent deviation,
not an approximation. Both encodings represent the same real value at the same
mantissa width up to the asymmetric range (`[-2^(m-1), 2^(m-1)-1]` for two's
complement vs. `[-(2^(m-1)-1), 2^(m-1)-1]` for sign-magnitude, ignoring
sign-magnitude's redundant `-0`). The shipped grid is the full two's-complement
one: every value-domain BFP quantize path clamps mantissas to
`[-2^(m-1), qMax]` with `qMax = 2^(mantissaBits-1) - 1` — positive cap `qMax`,
negative floor `-2^(m-1)` (a float-domain `clamp` followed by `clampInt32` in
`packFloatBufferAsBfp`, `quantizeFloatBufferToBfpCodes`, `packStreamAsBfp` and
the rescale walker). The `-2^(m-1)` code is therefore **reachable**, not inert:
the derived exponent keeps `|v / scale| <= qMax` (§3), so under `HALF_AWAY` it
is emitted only when D6's exponent cap pins the scale below what the data needs
(§2). Under `SR_HALF_AWAY` (lift iff `u < frac`, see `arithmetic-sym.md`) a value in
`[-qMax, qMax]` cannot round below `-qMax`, so it adds no route to it. The reason to deviate is infrastructure reuse, not a
numerical one: two's complement lets BFP share `byteConversionAppend` /
`unpackSignExtend` verbatim with `SYM`'s pack/unpack path instead of adding a
second signed-payload codec (a sign-bit-plus-magnitude packer/unpacker) purely
for BFP. The bit-exact NumPy gold emulation (`sym_gold.py` pattern) mirrors
two's complement, not sign-magnitude, for the same reason.

## 2. D6: exponent saturation, not abort

ODT's `#227` discipline for every other `packChunkGuarded` caller (SYM, ASYM,
BFP code-domain packing — see below) is to **abort** on overflow
(`PRINT_ERROR` + `exit(1)`): a research framework should surface a config that
can't represent its data, not silently corrupt it.

BFP's **value-domain** quantize path (deriving a block's stored exponent from
float magnitudes, `deriveBfpStoredExponent` in `src/tensor/TensorConversion.c`)
deviates from that discipline by design (D6): when the derived exponent
would fall outside `[0, 2^exponentBits - 1]`, it **clamps** instead of
aborting — high side clamps the stored exponent to its max, so the block's
mantissas saturate to the grid `[-2^(m-1), qMax]` (§1); low side clamps the stored exponent to `0`, so
small-magnitude quotients round toward zero (flush-to-zero). Both regimes are
implemented as a plain clamp on the `stored` value inside
`deriveBfpStoredExponent`, immediately before the biased byte is written.
The high clamp additionally never exceeds `bias + 127` — only reachable at
`exponentBits=8`, whose natural top (stored 255, E=128) has **no finite
float32 scale**: `ldexpf(1, 128)` is `+inf`, which would quantize every code
to 0 and dequantize the whole block to NaN (`0 * inf`) instead of
saturating. The saturation regime therefore engages at scale `2^127`, the
largest finite float32 power of two. A **non-finite `absMax`** (a caller's
pass 1 overflowed — see §5.6's scale bullet) takes the same high regime
directly: `deriveBfpStoredExponent` returns the cap without consulting
`frexpf`, whose result and `*exp` are both unspecified for inf/NaN (C17
7.12.6.4). A magnitude too large for any grid saturates at the largest finite
one; the alternative was an arbitrary exponent leaking into the caller's emit
pass. Since `#421` that arm is reached only by the ACCUMULATE/SCALE engines,
whose absmax can legitimately overflow to `±inf` from finite operands — the
three value-domain pack entries (`packFloatBufferAsBfp`,
`quantizeFloatBufferToBfpCodes`, `packStreamAsBfp`) now fail fast on a
non-finite INPUT VALUE before their absmax is ever formed (§5.6's R9
bullet), because a non-finite input is the caller handing over something BFP
cannot represent, not an overflow this format's own saturation regime is
meant to model.

**Rationale for the deviation:** `exponentBits` (range `[2,8]`) is one of the
epic's first-class HAR sweep axes — the sweep exists specifically to
measure the accuracy/loss cost of narrowing the exponent range. Saturation
*is* the phenomenon under study at that axis's coarse end, not a bug to guard
against: aborting on the first block whose absmax needs an out-of-range
exponent would make every narrow-`exponentBits` sweep point unrunnable instead
of producing a data point. This is scoped narrowly — it does not relax `#227`
generally.

**Code-domain packing still aborts.** `INT32 → BFP` (`convertInt32TensorToBfpTensor`)
packs raw integer codes directly into BFP mantissas with **no exponent
derivation** (exponents are set to the zero-state bias, `E = 0`, verbatim) —
there is no absmax to saturate against, so overflow of a code into
`mantissaBits` is a genuine "value doesn't fit the declared width" config
error, and `packChunkGuarded` aborts exactly as it does for every other
code-domain packer. D6 saturation covers value-domain quantization only.

## 3. Exponent rule: absmax-snap-up, not MX/MSFP's floor conventions

ODT's rule (`deriveBfpStoredExponent`): **`E = smallest E with absMax/2^E ≤
qMax`**, `qMax = 2^(mantissaBits-1) - 1`, derived via `frexpf` rather than
`log2f` so an absmax that is itself an exact power of two lands on the
boundary without float-rounding surprises (`frexpf` returns `frac ∈ [0.5,1)`;
`E = e` unless `frac == 0.5`, in which case `E = e-1`). Zero-state (`absMax ==
0`, e.g. a freshly-initialized config, mirroring `SYM`/`SYM_INT32`'s `scale =
1.f` zero-state) stores `E = 0`, i.e. `storedExponent = bias`, scale `1.0` —
same convention as the rest of the framework's "no data seen yet" grids, not a
value taken from the cited literature.

This is a genuinely different rule from the two literature anchors it is most
often compared against:

| Convention | Exponent derivation | Relation to ODT's rule |
|---|---|---|
| **MX** (OCP MX v1.0) | `E = floor(log2(absMax)) − emax`, where `emax` is the target *sub-format's* own max representable exponent (the shared scale re-centers each element's private float exponent into that sub-format's native range) | ODT has no per-element sub-format exponent to re-center — the mantissa is a plain signed integer, not a mini-float. `qMax` plays the role `emax` plays for MX: both express "how large the largest representable mantissa magnitude is," so dividing by `qMax` before snapping is ODT's integer-mantissa analog of MX's `-emax` shift |
| **MSFP** | `E = max_i(exponent(element_i))` — the block's shared exponent is literally the *largest natural FP32 exponent* among the block's elements, read directly off each element's IEEE-754 representation | ODT never inspects element-wise FP32 exponents; it computes one absmax over the block and snaps `absMax/qMax` up to a power of two. The two coincide only in the degenerate case where the block's max-magnitude element already sits exactly on a power-of-two boundary matching `qMax`'s bit position — in general they diverge because MSFP's rule is FP32-exponent-native (reads bits) while ODT's is integer-qMax-native (divides then snaps) |

Both comparisons are informational (mapping ODT's convention onto the
literature's vocabulary for readers coming from those papers), not a claim of
equivalence — the epic's sweep methodology treats ODT's rule as its own
convention, validated by its own gold emulation, not by parity with MX/MSFP.

## 4. Clone semantics: per-tensor reset, grouped deep-copy

`getQLike`'s BFP arm mirrors the `SYM` precedent exactly (refined
during PR1 planning from an earlier blanket-zero-state proposal): a
**per-tensor** clone (`numGroups == 1`) resets to the zero-state (`exponents[0]
= bias`, i.e. scale `1.0`) — the clone is a fresh, never-yet-quantized target,
so carrying over the source's single exponent would be a stale scale
masquerading as a derived one. A **grouped** clone (`numGroups > 1`) instead
**deep-copies** the source's `exponents[]` values verbatim, alongside
`numGroups`/`groupSize` — the group grid is an attach-time geometric fact
(which elements belong to which block), not a per-quantize derived value, so
`deepCopyQuantization`-style semantics (copy the fact, not reset it) apply,
exactly as they do for grouped `SYM`. This clone rule is what `gradInit` and
optimizer per-parameter state cloning (`m`/`v` buffers) both go through when
handed a BFP template.

## 5. Compute contract (epic PR2 forward + epic PR3 backward + epic PR4 weight-less layers + epic PR5 norms + epic PR6 softmax) — shipped

### 5.1 The flip is done — backward is now native, pinning is an optional fake-quant mode

`arithmeticFromQuantization`'s `BFP` arm (`src/arithmetic/ArithmeticType.c`)
derives native `ARITH_BFP`, not the PR1 float bridge:

```c
case BFP:
    a.type = ARITH_BFP;
    a.roundingMode = ((bfpQConfig_t *)q->qConfig)->roundingMode;
    break;
```

This is the documented breaking change of epic PR2 (the PR1 "D5 float-bridge
staging rule" — arithmetic-type-split D5, `docs/conventions/arithmetic-sym.md`
§"Decision register" — retired, in the codebase's own naming; see the comment
at this seam; Decision 8): a `layerQuant_t` profile built with
`layerQuantInitUniform(bfpConfig)` derives `ARITH_BFP` in **all four** math
slots (`forwardMath`/`weightGradMath`/`biasGradMath`/`propLossMath`). Through
epic PR2 the framework shipped the forward only, so a uniform-BFP model that
left a backward slot derived died at the layer's backward kernel dispatch:
every GEMM-family layer guards its three backward slots with a fail-fast
`switch` (the layer gate is load-bearing — for BFP-STORED operands the
funnel's missing-`bfpStage` gate of §5.3 never fires, it backstops only
FLOAT32-stored operands). **Epic PR3 lifts that gate**: `weightGrad`,
`biasGrad`, and `dx` (`propLoss`) all run native `ARITH_BFP` kernels for
Linear/Conv1d/Conv1dTransposed (§5.6 documents the backward contract itself).
A uniform-BFP model now trains its ENTIRE loop natively with no pins at all —
the capstone, `testBfpNativeForwardTrainingLossDecreasesAndGridMoves`
(`test/unit/userAPI/UnitTestMultiLayerTraining.c`), runs 25 fully-native steps
end to end and asserts convergence, the derived-`ARITH_BFP` slots, and that
both the forward wire's and the weights' grids move off the zero state.

Pinning the three backward slots to `ARITH_FLOAT32` is no longer required to
keep a uniform-BFP model trainable — it is now a **deliberate, still-supported
fake-quant-backward mode** (dequantize the BFP operands, compute the gradient
in plain float32, requantize the write-back), useful when the native kernels'
block-partial rounding profile is undesirable for a given experiment:

```c
lq0.weightGradMath = (arithmetic_t){.type = ARITH_FLOAT32, .roundingMode = HALF_AWAY};
lq0.biasGradMath   = (arithmetic_t){.type = ARITH_FLOAT32, .roundingMode = HALF_AWAY};
lq0.propLossMath   = (arithmetic_t){.type = ARITH_FLOAT32, .roundingMode = HALF_AWAY};
lq0.propLossQ      = f->floatQ;
```

(the three-slot-plus-wire pin from
`testBfpPinnedFloat32BackwardTrainingLossDecreases`,
`test/unit/userAPI/UnitTestMultiLayerTraining.c` — `propLossQ` must move
together with `propLossMath`, since a derived-BFP `propLossQ` alone would not
by itself make the backward op run in float. The shared fixture
(`buildBfpNativeFixture`, same file) gates all three math-slot pins plus the
wire behind one `pinWeightGradMath` flag: `false` builds the fully-native
capstone above, `true` builds this pinned fake-quant-backward variant — both
share everything else, so the flag isolates exactly the native-vs-pinned
difference.) Fake-quant over BFP storage was always available and remains so
post-PR3: pin the math slot(s) to `ARITH_FLOAT32` explicitly and the funnel
dequantizes the BFP operand like any other storage-only dtype, exactly as
PR1's derivation did automatically before the PR2 flip.
`testBfpFakeQuantTrainingLossDecreasesAndGridMoves` (the PR1 capstone) still
passes unchanged: its math slots all derive from a `FLOAT32`
`quantization_t`, which the flip never touches. Two later families narrow the
pin on the BACKWARD only, because their FLOAT32 backwards run outside the
funnel and raw-cast their wires: the norms reject the profile at the factory
(§5.8 R-N6) and softmax fails fast at the arm's guard (§5.9 R-S6, forward pin
unaffected).

### 5.2 BFP-native weights workflow

`ARITH_BFP` forward (Linear/Conv1d/Conv1dTransposed) requires BFP-stored
weights — the weight is the operand every FLOAT32 operand stages at (§5.4),
so a FLOAT32 weight has no width source to stage at. The layer factories only
ever allocate FLOAT32 parameters (#270 — the factory rejects any other init
dtype), so getting a native-BFP layer is always two steps:

1. Build the layer normally (FLOAT32 weights/bias).
2. `requantizeTensorInPlace(getParamFromParameter(layer->config->linear->weights), bfpQ)`
   (and the same for `bias`, if the layer has one) — converts the FLOAT32
   parameter to BFP storage at the widths/geometry `bfpQ` describes, in
   place.

A weight left FLOAT32 under `forwardMath.type == ARITH_BFP` fails fast with
`"<Layer>: ARITH_BFP forward requires BFP-stored weights (FLOAT32-init +
requantizeTensorInPlace, see docs/conventions/arithmetic-bfp.md); got dtype
%d"` — the message every guided error at this seam points back at this
section.

### 5.3 Kernel contract

Every `ARITH_BFP` kernel (`matmulBfpTensors`, `conv1dKernelBfp`,
`convTranspose1dKernelBfpGather`) shares one contract, enforced by the funnel
(`executeOp`'s prologue, `src/arithmetic/ExecuteOp.c`) before any kernel runs:

- **Unpacked-BFP scratch form.** Every operand reaches the kernel as
  sign-extended `int32` mantissa codes under a live `bfpQConfig_t` — never
  packed bytes (`opSpec_t.bfpStage`'s doc block, `ExecuteOp.h`). BFP-stored
  operands **borrow** their exponents array (pointer-aliased, read-only —
  writing through it corrupts the source tensor) and carry the source's own
  STORAGE `roundingMode`. FLOAT32-stored operands are **staged**: the funnel
  quantizes them into transient exponent-backed scratch at a caller-supplied
  geometry template (`opSpec_t.bfpStage[i]`), carrying the OP's
  `arithmetic.roundingMode` — this is the borrow/staged rounding-mode
  asymmetry: a borrowed config's rounding describes how the SOURCE was
  quantized (a historical fact), a staged config's rounding is a live choice
  the current op makes.
- **Both operands may be blocked.** Unlike the SYM/ASYM
  `groupedSymOperandPos` gate (at most one grouped operand at a time),
  `ARITH_BFP` blocking is legal on every operand simultaneously — the
  prologue hands each operand its own live `bfpQConfig_t`, so there is no
  scalar collapse to guard against.
- **Fold-on-boundary via `ldexpf`.** Kernels accumulate one `int32` partial
  per same-exponent-group segment (both operands' group ids tracked
  per-element), then fold into a `float32` accumulator on ANY group-boundary
  crossing (either operand) and once more at the tail:
  `acc += ldexpf((float)partial, Ea + Eb − biasA − biasB)`. The power-of-two
  multiply is exact (D7) — the only rounding this step introduces is
  ordinary float32 addition across blocks, plus the two exceptions in §§7–8.
  Kernels write the fold as `bfpLdexpInt32(partial, e)` (BfpKernelSupport.h):
  bit-identical to that ldexpf, a plain multiply by an exact 2^e on the hot path.
  Amended for the forward by §13: an op that opts into the integer raw
  (`opSpec_t.bfpRawOut`) emits `BFP_RAW` and the epilogue re-blocks in
  integers; without the opt-in (every backward op) this bullet stands.
- **Kernels are rounding-free** (Decision 7). Every `roundByMode` call in
  the BFP path (`HALF_AWAY` and `SR_HALF_AWAY` are defined once, in
  [arithmetic-sym.md](arithmetic-sym.md) "Rounding modes, the SYM_INT32 grid,
  encode and requant") lives at exactly two seams: staging (quantizing a FLOAT32
  operand into scratch) and the `OUT_WRITE` epilogue (packing the raw FLOAT32
  kernel output back into a BFP target). Nothing inside a kernel's reduction
  loop rounds. Exception, forward only (§13 "The integer fold"): an
  integer-raw kernel aligns each finished segment to the accumulation
  exponent with a HALF_AWAY right shift (never SR, never the op's mode), and
  the integer Softmax forward rounds its normalisation HALF_AWAY (§5.9 step
  5) — the re-block still owns the op's rounding mode.
- **Headroom, exact form.** A same-exponent `int32` partial is sound only up
  to `bfpSegmentLimit(ma, mb) = INT32_MAX >> (ma+mb−2)`
  (`src/arithmetic/include/BfpKernelSupport.h`; `shift >= 31` clamps to `1` —
  two `2^30` products already overflow `int32`). `bfpValidateBlockHeadroom`
  fail-fasts at kernel entry over `min(runA, runB, reductionLen)` vs. that
  limit, where `runA`/`runB` are each operand's `groupSize` (or the full
  reduction length for a per-tensor operand) — the bound holds for strided
  walks too, since distinct storage indices inside one group number at most
  `groupSize` regardless of walk order. The design's SYMBOLIC closed form
  `g ≤ 2^(33−2m)` (equal-width case, `ma = mb = m`) reads one product looser
  than what shipped if evaluated literally (`2^(33−2m)` is `131072` at m=8,
  `512` at m=12) — but the design's own worked examples already state the
  TIGHT values that match the shipped formula exactly: "m=8 → 131 071;
  m=12 → 511". The shipped `INT32_MAX >> (ma+mb−2)` IS that tight
  form, expressed exactly rather than symbolically: for equal widths it
  reduces algebraically to `2^(33−2m) − 1` — one below the naive power of
  two, because `INT32_MAX = 2^31−1`, not `2^31` — which is why the design's
  worked examples already land on the shipped values rather than the
  symbolic form's literal evaluation. Pinned by
  `testBfpSegmentLimitTableValues`, `UnitTestMatmul.c`, which also covers
  **m=16 → 1** (shipped-only; the design's own worked examples stop at m=12).
  In practice operand widths often differ (e.g. Task 3's gold fixture mixes
  `ma=6`/`mb=4`), which is exactly why the guard's normative form is
  `INT32_MAX >> (ma+mb−2)`, not the equal-width closed form. Bias operands
  are exempt from this guard: they are value-seeds dequantized to float
  BEFORE the reduction, never product operands.
- **Neither forward nor backward ever re-blocks a BFP-stored operand.**
  Storage groups are contiguous along exactly the axis the forward reduction
  walks (the row/column a GEMM reduces over, or a conv's input-channel ×
  kernel-tap axis), so the forward kernels consume the tensors' STORAGE
  blocking directly, unmodified. The backward GEMMs reduce along a DIFFERENT
  axis (e.g. a transposed reduction), which is exactly the situation design
  decision D8's original wording ("op-local blocking is a property of the
  op") pointed at — read literally, that wording says to recompute fresh
  block exponents for the backward's own reduction axis before each dot
  product, i.e. HBFP's per-dot-product recompute rule. **Epic PR3 does not do
  that.** D8 was amended 2026-09-02 (§9) to spell out that for
  ALREADY-QUANTIZED BFP-stored operands, op-local blocking realizes as the
  SAME PR2 fold contract instead (§5.6): per-element `bfpGroupOf` lookup,
  folding on EITHER operand's group change, never a re-quantize onto fresh
  reduction-axis blocks. Only the FOLD FREQUENCY differs between forward and
  backward — a transposed or output-centric walk crosses group boundaries
  more often than the forward's storage-order walk does, so it folds more
  often — never what the exponents mean or where they come from. The
  regression coverage for this contract lives entirely in the PR3 backward
  test suite now that native backward ships end to end; the PR2-era pins that
  asserted a uniform-BFP model dies at backward were retired along with the
  dead code path they exercised.

### 5.4 Staging rule (Decisions 1–3, marked revisable)

- **Decision 1**: a FLOAT32-stored operand stages **per-tensor** (`{1,0}`) at
  the **weight operand's** `(mantissaBits, exponentBits)`, rounded by the
  op's `arithmetic.roundingMode`. This mirrors SYM's scratch rule
  (`initSymInt32QConfig(arithmetic.roundingMode, ...)`) and covers the
  first-layer/dataset-input case — a serious wire-width sweep stores wires
  BFP instead (§5.5) and never reaches this path. **Explicitly marked
  revisable**: nothing in the funnel or the kernels assumes per-tensor
  staging specifically — a future PR could stage at a grouped template
  without touching the kernel contract. Since 2026-11-04 the FORWARD stages
  only into FLOAT32 targets: an integer forward into a BFP wire rejects a
  FLOAT32 operand, and the model input goes through a Quantization layer
  (§13, "No float in the forward after the input layer").
- **Decision 1 covers backward identically (epic PR3).** Every layer's
  backward dispatch builds the SAME per-tensor `{1,0}` staging template, at
  the weight operand's own `(mantissaBits, exponentBits)`, once before its
  three `executeOp` backward calls (`weightGrad`/`biasGrad`/`propLoss`) — one
  template, reused across all three, rounded by each op's own
  `arithmetic.roundingMode` (never the template's), exactly mirroring the
  forward. (Construction detail: Linear literally builds the ONE template in
  `linearBackward` and reuses it across its three calls, while the
  Conv1d/Conv1dTransposed grad wrappers each construct it per-wrapper — the
  field values are identical, and since staging rounds by the op's
  `arithmetic.roundingMode`, the template's own `roundingMode` field is
  inert.) This is the **rule-1 mirror**: `ARITH_BFP` forward requires
  BFP-stored weights because the weight is the only width source a
  FLOAT32-stored operand can stage at (§5.2); the SAME requirement applies to
  backward, independently — ANY of the three backward math slots deriving
  `ARITH_BFP` fails fast unless that layer's OWN weights are BFP-stored,
  checked once per backward call before the staging template is even built
  (`Linear backward: ARITH_BFP math slots require BFP-stored weights ...`,
  and the Conv1d/Conv1dTransposed twins of that message). The forward gate
  (§5.2) and this backward gate are independent checks on the SAME
  underlying fact (BFP-stored weights are the only width source a
  FLOAT32-stored operand can stage at) — whichever side of a layer's graph
  runs `ARITH_BFP`, that side's gate fires, so the two-step §5.2 recipe
  (FLOAT32-init, then `requantizeTensorInPlace`) is the one prerequisite
  either direction depends on.
- **Decision 2**: bias is staged **uniformly**, like any other FLOAT32-stored
  operand (no funnel special case) — a FLOAT32 bias stages into BFP scratch;
  a BFP-stored bias is unpacked and borrowed. Either way the kernel
  dequant-seeds the float accumulator with the bias value BEFORE the
  reduction (exact `mantissa · 2^E`), so bias never touches the `int32`
  partial and is exempt from the headroom guard.
- **Decision 3 (v1 operand-dtype scope)**: under `ARITH_BFP`, only
  FLOAT32-stored (staged) and BFP-stored (unpacked, borrowed) operands are
  legal. `SYM`/`ASYM`/`INT32`/`BOOL`-stored operands fail fast with a guided
  message ("insert a Quantization layer" — convert explicitly first).
- **Decision 11 deny**: a BFP-stored operand under `ARITH_SYM_INT32` is
  denied in the funnel's SYM prologue arm —
  `convertBfpTensorToSymInt32Tensor` would requantize the group structure
  away to a fresh per-tensor scalar grid; values stay correct (the dequant
  path is grouped-capable) but the BLOCKING silently disappears, which is
  exactly the accuracy surprise the grouped-operand gate exists to prevent.
  Legal combinations for BFP storage are `ARITH_FLOAT32` (fake-quant) and
  `ARITH_BFP` (native) only.

### 5.5 Wire rule (Decision 5)

The training wire table (`rematBfpWireGrouping`, applied to every ACT and GRAD
wire, `src/userApi/training_loop/remat/RematWireTable.c`) and the two inference
allocators (`initBufferOutput`/`initBufferInput`, `src/userApi/InferenceApi.c`)
each carry a BFP arm keyed off the layer's `layerQuant_t` template (a
`quantizationInitBfp*` config):

- The template's `groupSize` is the ONE knob a wire allocator honors. `0`
  (the `{1,0}` sentinel) allocates a per-tensor wire; a nonzero `groupSize`
  allocates a grouped wire whose `numGroups = numberOfValues / groupSize` is
  **derived at allocation time from the wire's actual runtime element
  count**, never taken from the template. A `groupSize` equal to the wire's
  element count derives `numGroups == 1` and **normalizes to the per-tensor
  `{1,0}` config** — one group spanning the tensor IS per-tensor blocking,
  and the `{1,N}` spelling would violate the config grammar.
- The template's own `numGroups` is deliberately IGNORED: a `layerQuant_t`
  profile is shape-agnostic and routinely shared across layers/wires whose
  sizes differ, so a template `numGroups` can only be a guess — and honoring
  it would size `exponents[]` against a count that does not describe the
  wire while the packer's group index stays unbounded by it (heap overflow).
- `numberOfValues % groupSize != 0` fails fast with a guided message ("pick a
  divisor or a per-tensor `{1,0}` template") rather than silently flooring.
- A freshly allocated wire starts at the **zero-state** exponents (bias,
  scale `1.0`) — the forward's first `OUT_WRITE` derives the real grid.

### 5.6 Backward contract (epic PR3)

Native `ARITH_BFP` backward ships for the three GEMM-family layers
(Linear/Conv1d/Conv1dTransposed): `weightGrad`, `biasGrad`, and `dx`
(`propLoss`) all run `ARITH_BFP` kernels when their math slot derives (or is
explicitly configured) `ARITH_BFP`, gated by the layer-side dispatch
(`linearBackwardKernelForArithmetic` and its Conv1d/Conv1dTransposed twins)
the same way the forward is — any future arithmetic type must die at that
dispatch, not fall through to the funnel: for BFP-STORED operands the
funnel's missing-`bfpStage` gate (§5.3) never fires, so a fall-through would
hand a float kernel unpacked int32 mantissa scratch through a `float*` cast —
silent wrong arithmetic, not a crash.

- **`weightGrad` is an output-centric walk**, not the forward's input-centric
  one. Linear reuses the GEMM entries directly — the kernels are
  orientation-agnostic (per-element group lookup honors `orderOfDimensions`),
  so `linearCalcWeightGradsBfp` (float fold, `matmulBfpTensors`) and
  `linearCalcWeightGradsBfpRaw` (integer, `matmulBfpRawTensors`, into BFP
  grads, §14) are thin transpose-view wrappers (`transposeTensor(loss, 0,
  1)` around the same GEMM walk the forward and `dx` use). Conv1d runs the FLOAT32 weightGrad loop on integer
  accumulators (§14). Conv1dTransposed instead walks `(outChannel,
  inChannel, kernelTap)` as the OUTER loops and `(batch, outputPosition)` as
  the reduction — each `(oc, ic, k)` cell accumulates its own same-exponent
  `int32` partial across every contributing `(b, outPos)` pair, folding on
  EITHER operand's group change via `ldexpf` (the same fold contract as
  §5.3), just walked in a different order than the forward's per-output-
  element loop.
- **`biasGrad` is a b-outer/outPos-inner sum-fold.** For Linear the outer
  loop is per-feature and the inner loop walks the batch; for
  Conv1d/Conv1dTransposed the same shape walks per-output-channel over
  `(batch, outputPosition)`. This reduction SUMS one BFP operand's mantissas
  (no product), so it validates against the sum-headroom twin,
  `bfpValidateSumHeadroom`/`bfpSumSegmentLimit`
  (`src/arithmetic/include/BfpKernelSupport.h`: `INT32_MAX >> (m-1)` for a
  pure sum, vs. the product guard's `INT32_MAX >> (ma+mb-2)`), evaluated over
  `min(groupSize, reductionLen)` (or the full reduction length per-tensor,
  i.e. `min(n, reductionLen)`) — a same-exponent segment never accumulates
  more codes than one group holds, by the same "no walk visits more of one
  group than its `groupSize`" argument the product guard uses. All three
  layers' `biasGrad` share this one bound
  (`linearCalcBiasGradsBfp`/`conv1dCalcBiasGradsBfp`/
  `conv1dTransposedCalcBiasGradsBfp`). Into BFP grads Conv1d and Linear
  fold the segment partials in integers instead of `ldexpf` (§14).
- **`dx` (`propLoss`) reuses the OTHER family's own forward kernel — the D9
  gather adjoint (§6), not a dedicated backward kernel** — except Conv1d.
  Conv1d's `dx` is mathematically a Conv1dTransposed op; its
  `propLossKernelBfp` adapter runs `convTranspose1dKernelBfpScatter`, the
  FLOAT32 ConvT scatter loop on integer accumulators (§14), not the gather.
  Conv1dTransposed's `dx` is mathematically
  a Conv1d op, so ITS `propLossKernelBfp` dispatches to `conv1dKernelBfp`.
  Linear's `dx` (`linearCalcPropLossBfp`) is
  `matmulBfpRawTensors(loss, weights, NULL, propLoss)` — the same
  orientation-agnostic integer GEMM the forward calls, with `weights` as
  the second operand instead of the forward input (integer end to end,
  §14).
- **BFP-stored operands entering any backward op go through the EXACT SAME
  fold contract as the forward (§5.3), never a re-quantize** — except Conv1d
  `dx` and `weightGrad`, which align every product on its own (§14). Every operand
  is looked up per-element (`bfpGroupOf`); the running `int32` partial folds
  into the `float32` accumulator via a lossless `ldexpf` power-of-two shift
  on ANY group-boundary crossing (either operand) and once more at the tail
  — identical mechanics whether the reduction axis matches storage order
  (forward) or not (backward's transposed/output-centric walks). This is the
  **D8 amendment** (§9). FLOAT32-stored operands still STAGE — per-tensor, at
  the weight's own widths (§5.4's Decision-1 mirror) — because staging is the
  one point where quantization legitimately happens; a BFP-stored operand
  already paid that cost once at its own storage time, so re-deriving a
  second grid for it would only add a needless second quantization step
  (§9's double-quantization rationale).
- **Per-tensor-only grad/optimizer-state knob (epic PR3 Tasks 6–7).**
  `gradInit`'s carrier gate (`src/userApi/tensor/TensorApi.c`), `SgdApi.c`'s
  `momentumStateInit`, `AdamWApi.c`'s `momentStateInit`, and
  `PpcaReplayApi.c`'s state-template validation each accept a per-tensor
  (`{1,0}`) BFP template and reject a grouped one with the same "unsupported
  — per-tensor only (#300 axis)" shape of message: a template is deep-cloned
  per parameter/state buffer via `getQLike`, and a grouped template only
  ever fits the ONE element count it was built against, so honoring
  `numGroups > 1` generically would be a silent shape trap the moment the
  same template is reused against a differently-sized tensor. This is a
  deliberate SCOPE decision, not a limitation of the accumulate/scale
  primitives themselves — Task 5's accumulate engines and Task 8's
  `scaleBfpTensorInPlace` are both written generically over `numGroups` and
  would work correctly on a grouped grad were one ever constructed — grouped
  BFP grad/state storage is left for a future `#300` axis, mirroring the
  SYM/ASYM grad-storage gates it was modeled on exactly.
- **`optimizerZeroGrad`'s `BFP` arm resets exponents to bias, not just codes
  to zero.** Byte-zeroing the packed mantissa storage alone already decodes
  every code to `0.0f` under any FINITE exponent (`0 · inf` would be NaN,
  which is why `deriveBfpStoredExponent` caps derived exponents at
  `bias + 127` and, since #420 G5, the deserializer rejects records above
  that cap) — the reset is *value*-inert —
  and the NEXT accumulate does not key on it either: the `FixedGrid`
  engine's "fresh vs. already-gridded" carry decision (below) is a
  codes-only all-zero scan, so a byte-zeroed grad is classified fresh
  whatever its exponents say. The exponent reset is SYM/ASYM-parity hygiene
  (the `scales[0] = 1.f` / `zeroPoints[0] = 0` analogs): it restores the
  canonical zero state (stored = bias) for serialization/inspection and for
  any future consumer that DOES read a grad's exponents; pinned by the Task
  6 e2e's exponent assertion
  (`testBfpGradStorageTrainingAccumulatesAndSteps`).
- **`accumulateOut`'s BFP-target arm (epic PR3 Task 5,
  `src/tensor/TensorConversion.c`)** backs the `executeOp` epilogue's
  grad-accumulation modes for a BFP-typed grad target:
  - `OUT_ACC_FIXED_SCALE` (`accumulateIntoBfpFixedGridEngine`): a codes-only
    all-zero scan over the WHOLE target decides fresh-vs-gridded — an
    all-zero accumulator (post-`initTensor` or post-`optimizerZeroGrad`)
    derives a brand-new per-group grid from the increment alone (through
    `deriveBfpStoredExponent`, the zero-state convention: absmax 0 → stored =
    bias); any other state carries the EXISTING grid verbatim, fit-preserving
    like the SYM `FixedGrid` twin. The read-modify-write pass then
    `packChunkGuarded`-**aborts** on mantissa overflow (#227 code-domain
    discipline — this is a CODE-domain pack, so D6's value-domain saturation
    does not apply here). Its float-domain pre-clamp (#421) is therefore
    deliberately WIDER than the legal code band — two codes on each side,
    enough that `(int32_t)round(...)` is always defined but never enough to
    pull an overflowing value back onto the grid, so the abort still fires.
    Under `SR_HALF_AWAY` (lift iff `u < frac`; no sum is formed) a value
    clamped exactly onto `qMin-1` or `qMax+1` has no fraction and stays put, so
    a one-code band would already preserve the abort; the second code is
    headroom, and it keeps the `(int32_t)` cast defined.
  - `OUT_ACC_DYNAMIC_RESCALE` (`bfpRescaleWalk`): re-derives a FRESH
    per-group grid on every call from `|mant·oldScale·factor + inc|`'s absmax
    (value-domain, so D6 CLAMPS rather than aborts at the exponent range's
    edges). Since the #421 unification this is the SAME function
    `scaleBfpTensorInPlace` runs — the accumulate arms pass `factor = 1.f`,
    the scale arm passes a NULL increment source; both degenerate knobs are
    exact, so neither path's output moved. Its two passes are INTERLEAVED,
    pass 1 running exactly one group boundary ahead of pass 2:
    `qc->exponents` keeps holding the OLD grid for the whole walk (so pass 2's
    dequant of the pre-existing mantissas always decodes under the grid they
    were actually stored under, never the freshly derived one) and pass 2
    publishes each group's fresh exponent only after consuming that group's
    LAST element. The only out-of-config scratch is therefore a fixed
    `ODT_CONVERSION_CHUNK_ELEMS + 1` byte window of fresh exponents (the live
    set spans at most `ODT_CONVERSION_CHUNK_ELEMS` groups, so that many slots
    are already exactly enough; the extra one is slack) — the PR3
    whole-tensor two-pass instead latched the ENTIRE old grid into an
    `O(numGroups)` stack VLA. The frame is deterministic rather than smaller:
    sharing one walker costs the SCALE arm an increment buffer it never had,
    so at `numGroups == 1` that arm grows by ~1280 B (~2 KB → ~3.3 KB), a
    deliberate trade — a scratch split stays mechanical if MCU stack budgets
    ever complain.
  - Both engines are **two-pass** (one pass to decide/derive the grid, a
    second chunk-aligned read-modify-write pass to write it) rather than a
    single group-sequential walk, because a group's bit offset
    (`g·groupSize·mantissaBits`) is not byte-aligned for arbitrary
    geometries — the walk has to stay chunk-aligned
    (`ODT_CONVERSION_CHUNK_ELEMS`) and let a run-walk inside each chunk
    handle group boundaries, the same shape `packStreamAsBfp`/
    `dequantChunkToFloat` already use. `FixedGrid` runs its two passes over
    the whole tensor; `bfpRescaleWalk` interleaves them (above).
  - Both engines **fail fast on a non-finite value in the INCREMENT** (#421,
    ruling R6 — the increment-side twin of the scale arm's
    non-finite-`factor` guard): a BFP grid has no NaN/inf code, so mapping it
    to 0 would drop the caller's data and saturating it would invent data.
    Deliberate contract asymmetry — the FLOAT32 grad-storage path keeps
    propagating a NaN gradient loudly, because FLOAT32 can represent it.
    Non-finite INTERMEDIATES arising from finite inputs (a grid at the
    exponent cap overflowing its scaled values) are NOT rejected: those
    saturate through the emit clamp as ordinary D6 value-domain behaviour.
  - Both engines leave an **empty target** (`n == 0`) in the canonical zero
    state (every group's stored exponent = bias) — the semantic the scale arm
    already had (#421, ruling R7; before it the rescale arm kept the stale
    grid and the fixed-grid arm's two increment kinds disagreed with each
    other).
- **`scaleBfpTensorInPlace` + `scaleOptimizerGradients`'s `BFP` arm (epic PR3
  Task 8)** close the last gap in the default training-loop epoch:
  `TrainingEpochDefault.c`'s mean-scale branch (`computeMeanScale` →
  `scaleOptimizerGradients` → step → zero) now handles a `BFP` grad without
  dying. There is no O(1) scale fold the way SYM_INT32/SYM/ASYM get one (BFP
  dequant is `mantissa · 2^(E−bias)` per GROUP, and an arbitrary mean-scale
  factor is not a power of two in general), so the `BFP` arm calls
  `scaleBfpTensorInPlace` for an honest O(n) value-domain repack: since #421
  literally the `DYNAMIC_RESCALE` engine itself (`bfpRescaleWalk` with a NULL
  increment source), fresh exponents derived from the SCALED absmax, one `roundByMode` per
  element by the grad's own STORAGE `roundingMode` (scaling is a storage
  requantization, not an op — #282's target-owned convention, the SAME
  ownership the ACC epilogues keep: `accumulateOut` deliberately rounds by
  the TARGET's storage mode (#282; `ExecuteOp.c`'s ACC-epilogue comment —
  accumulate is a read-modify-write under the accumulator's own storage
  grid), while the OP's `arithmetic.roundingMode` governs staging and the
  OUT_WRITE epilogue only). A power-of-two `factor` is exact end to end
  (multiplying by an exact power of two only shifts the exponent, never
  rounds the mantissa); any other factor is exact up to ordinary float32
  rounding — the one lossy case is a group whose SCALED absmax pushes its
  derived exponent past `[0, 2^exponentBits − 1]`, which D6 clamps
  (saturates) rather than aborts, same as every other value-domain BFP
  quantize path. `REDUCTION_MEAN` — the framework's default forward
  reduction — and the default `TrainingEpochDefault` epoch path both now
  work end to end with a BFP-stored grad
  (`testBfpGradStorageTrainsUnderReductionMean`,
  `test/unit/userAPI/UnitTestMultiLayerTraining.c`).
  Three defined-behavior rules close this path's edges (PR3 follow-up batch):
  the `factor` must be **finite** — a non-finite one fail-fasts in the
  primitive, the ONE `scaleOptimizerGradients` arm that does, because BFP has
  no non-finite code to propagate into while FLOAT32/SYM_INT32/SYM/ASYM all
  warn and propagate; a group already AT the exponent cap has no headroom
  left, so even a finite factor can overflow its scaled values to `±inf`,
  which **saturate to the code range** (that clamp runs in the float domain,
  BEFORE `roundByMode` — `(int32_t)round(±inf)` is undefined, C17 6.3.1.4);
  and an **empty** tensor (`n == 0`) is left in the canonical zero state
  (stored = bias) instead of keeping a grid that describes data it does not
  have. #421 closed the engine-unification follow-up this bullet used to
  defer: the float-domain pre-clamp now runs at EVERY BFP emit site
  (`packFloatBufferAsBfp`, `quantizeFloatBufferToBfpCodes`,
  `packStreamAsBfp`, the unified rescale walker, and — in its
  abort-preserving wide form — the `FixedGrid` engine), because the regime
  needs no `inf` at all: at a narrow `exponentBits` the derived exponent
  saturates, so `v / scale` leaves `int32` range for entirely FINITE inputs.
  The saturation grid is the full two's-complement one at every value-domain
  site: positive cap `qMax`, negative floor `−2^(m−1)` (§1). The `FixedGrid`
  engine's wider pre-clamp is the one deliberate exception (above).
- **A non-finite INPUT VALUE fails fast at the three value-domain pack
  entries** (`packFloatBufferAsBfp`, `quantizeFloatBufferToBfpCodes`,
  `packStreamAsBfp`) — `#421`, ruling R9, the value-side twin of the
  accumulate engines' R6 increment guard, with the same rationale: BFP has no
  NaN/inf code, so mapping the value to `0` drops the caller's data and
  saturating it onto the grid invents data, while the FLOAT32 wire keeps
  propagating a NaN loudly because FLOAT32 can represent it. This is not a
  theoretical corner: a diverged training step's NaN loss reaches
  `quantizeFloatBufferToBfpCodes` through the funnel's FLOAT32-operand
  staging, and the previous behaviour was PLATFORM-DIVERGENT rather than
  merely undefined — pass 1's `v > absMax` comparison is false for NaN so the
  grid ignored it, the comparison-based float clamp could not map it either,
  and `(int32_t)round(NaN)` yields `0` on arm64 but `INT32_MIN` (clamped to
  `qMin`) on x86-64, i.e. a bit-parity hazard. The check lives in the pass
  that already reads every element, so it costs no extra walk, and it names
  the public entry through the same `what` label the abort messages use.
  **Only INPUTS**: computed intermediates inside the engines keep saturating
  through the emit clamp, and `deriveBfpStoredExponent`'s non-finite-absMax
  cap branch (§2) stays live for them.
- **Native BFP `dx` wires are legal.** A layer's `dx` (`propLoss`) output can
  itself be a BFP-stored wire, written by the native backward kernel's own
  `OUT_WRITE` epilogue (deriving that wire's grid directly, no float bridge)
  and read back by the PREVIOUS layer's `weightGrad`/`biasGrad` as an
  ordinary BFP-borrowed operand — pinned end to end by
  `testBfpDxWireNativeBackwardTrains`
  (`test/unit/userAPI/UnitTestMultiLayerTraining.c`).

**Still gated (deferred past PR4, not a PR3/PR4 gap):**

- Grouped BFP grad/optimizer-state templates (per-tensor-only decision above
  — a scope decision, not a kernel limitation; a future `#300` axis).
- NATIVE BFP losses (integer CE/MSE). Every LAYER now has an `ARITH_BFP`
  arm: Softmax SHIPPED with PR6 on BOTH sides — native i-exp forward and a
  native funnel backward, see §5.9 — the norms (LayerNorm/GroupNorm) with
  PR5 (§5.8), and pools, Relu, Flatten and Dropout with PR4 (§5.7). The
  losses have only their PR4 FAKE-QUANT arm (`case BFP:` joining
  `case SYM_INT32:` on the dtype-generic helpers), which closed
  `docs/FEATURES.md`'s "no loss function has a BFP arm" gap; integer CE/MSE
  stay explicitly OUT of this epic's committed scope — the epic design lists them as
  an optional stretch goal, filed as a follow-up issue when PR6 lands.
- The optimizer's `updateMath` stays `ARITH_FLOAT32`-only (#310) — BFP
  backward produces grads and lets them be STORED BFP, but the parameter
  update step itself is unchanged.
- `optimizerClipGradNorm` rejects packed SYM/ASYM/BFP grad storage (computing
  a norm needs unpacked element values; the O(1) scale-fold trick that works
  for *applying* an already-known coefficient does not provide that) — a
  deliberate v1 limitation, untouched by this epic.
- An opt-in reduction-axis re-block as a perf knob (trading the accuracy cost
  of double-quantizing grads for longer same-exponent fold segments) is a
  documented, deferred follow-up (§9) — file the issue only once a real MCU
  measurement motivates it.

### 5.7 Weight-less layers (epic PR4)

**Staging anchor (R-P1).** GEMM rules 1/2 are weight-anchored: a FLOAT32
operand stages at the *weights'* widths. Pools have no weight operand, so the
anchor is the layer's OWN produced-wire config — `outputQ` for the forward op,
`propLossQ` for the backward op. The check is EAGER at op entry (`ARITH_BFP`
math slot ⇒ the corresponding produced-wire config must be BFP-typed), because
without it there is no width source at all. The stage template is always
per-tensor `{1,0}`; only the anchor's `mantissaBits`/`exponentBits` are read.
A BFP-STORED operand is borrowed zero-copy as everywhere (D8) and never
re-blocked.

**Pool numerics (R-P4).** Raw mantissas are not comparable across groups,
so MaxPool compares exact `(code, E)` values; argmax/auxOut semantics and the
−1 empty-window sentinel are unchanged. AvgPool sums int32 partials within
same-group segments and folds on every group crossing plus the tail. Both
FORWARDS are integer-only (`BFP_RAW`, §13 "The pool forwards"): the fold is
an integer shift and the `/K` is an exponent shift: BFP math takes a
power-of-two K only (ruling R1, §14). AdaptiveAvgPool still folds in float and
divides by each window's own count in the FLOAT32 raw (the SYM `s/K` scale
fold has no BFP analog: a BFP scale is `2^E`). Forward window sums carry
`bfpValidateSumHeadroom`; the backwards do NOT, because contributions
accumulate directly in the FLOAT32 raw with no int32 partials.

**Packed-domain transparency (R-P2, R-P5).** Relu and Flatten carry the
per-group exponents VERBATIM and stay outside the funnel: routing them through
`executeOp` would re-derive exponents at OUT_WRITE, a second quantization of
UNCHANGED values, which §9/D8 forbid. The CODES are verbatim for Flatten only —
a reshape moves the payload byte-for-byte. Relu's are not: its forward clamps
negative codes to 0 (the block's grid stays valid, just no longer absmax-tight
— the accepted utilization drop, deviations register 6) and its backward masks
the loss codes by the SIGN of the forward input's codes. What is transparent in
both cases is the GRID, never necessarily the code values.

Both require the wire pair to share the same **element count** AND
`{numGroups, groupSize, mantissaBits, exponentBits}`. The element count is a
SEPARATE check (`bfpRequireElementCount`, not `bfpRequireSameGeometry`) for a
specific reason: a per-tensor `{1, 0}` grid is valid for ANY length, so two
wires of different sizes can pass the grid check, after which a `memcpy` sized
off one side overruns the other. A differing grid is a re-block and belongs to
the Quantization layer.

**Dropout bridge (R-P3).** Non-native BY DECISION (D4): `1/(1−p)` is not a
power of two and cannot fold into exponents. One float bridge, two passes over
the packed payload (`scaleBfpTensorInPlace`'s skeleton with the BOOL mask fused
in): fresh per-group exponents from the masked-and-scaled absmax, then
requantize with the DESTINATION config's own storage rounding (#282). This
fresh derive is not double quantization — the multiply changed the values.
Eval mode is a verbatim copy. `p = 0.5` is exact and needs no special case.

**Losses (R-P6).** Fake-quant only: `case BFP:` joins the existing
dtype-generic helpers, whose `convertTensor` calls already own
`conversionMatrix[BFP][FLOAT32]` (read) and `conversionMatrix[FLOAT32][BFP]`
(write, fresh exponents). Native BFP CE/MSE remain the optional stretch that
PR6 files as a follow-up.

**Outside-funnel carve-out (PR4 scope).** The funnel is still the ONE place
storage conversion happens (`docs/CONVENTIONS.md`), and PR4 does not widen
that rule — it records which layers were already outside it and why. Relu,
Flatten and Dropout are scale-transparent or bridge-local, so they read and
write the PACKED payload and the exponent array DIRECTLY, by design: for Relu
and Flatten because a funnel round-trip would re-quantize unchanged values
(above), and for Dropout because its own two-pass bridge is the conversion.
Pools and losses take the opposite route — pools run INSIDE `executeOp` on the
unpacked-BFP scratch under the anchor rule above, and the losses go through
`convertTensor`, i.e. both stay on documented arms. Softmax's BACKWARD is the
FOURTH outside-funnel site (`Softmax.c`'s own wording) — its FLOAT32 and SYM
arms raw-cast all three wires to `float*`/`int32*` — but they handle no BFP at
all: each rejects a BFP wire on any of the three (per-arm
`bfpRequireNoBfpWire`, §5.9 R-S5), so they walk no packed payload. Since PR6
the `ARITH_BFP` backward arm runs INSIDE the funnel like the pools, and so
does the native BFP forward (§5.9).

Consequence for reviewers — the COMPLETE inventory of code that may walk packed
BFP bytes, so that the rule applied literally does not flag correct code:

1. the conversion authority `src/tensor/TensorConversion.c` (the BFP cells, the
   accumulate engines, `scaleBfpTensorInPlace`) **plus the bit-level codec it
   is built on**, `byteConversion`/`byteConversionAppend` in
   `src/tensor/Tensor.c`;
2. the funnel's own operand staging in `src/arithmetic/ExecuteOp.c`, which
   unpacks BFP payloads into the int32 scratch every `ARITH_BFP` kernel
   consumes;
3. `gteBfpZero` in `src/arithmetic/Comparison.c` — **Relu's FORWARD packed walk
   lives there**, not in `Relu.c`, which only calls it;
4. the three outside-funnel layers themselves: `Relu.c`'s backward sign mask,
   `Dropout.c`'s two-pass bridge, and `Flatten.c`'s whole-payload `memcpy`.

Anywhere else, a raw `->data` walk over BFP bytes is a bug. Test fixtures are
the one non-`src/` exception, and they go through
`byteConversion`/`unpackSignExtend` rather than indexing raw.

Because the funnel's prologue is not there to reject a mismatched wire, each
of the three layers carries its own gate, and they are NOT the same shape.
Relu and Dropout have an `arithmetic_t` slot, so they guard PER ARM:
`requireNoBfpWire` on the FLOAT32/SYM arms (which raw-cast to `float*` /
`int32*`) and `requireBfpWire` on the `ARITH_BFP` arm. Flatten has NO
arithmetic slot at all (`layerForwardMath` returns the `NO_ARITHMETIC`
constant) and therefore no arms to guard: its gate is a pair check —
a BFP wire on one side demands a BFP wire on the other, and the two must then
share the same element count and `{numGroups, groupSize, mantissaBits,
exponentBits}`.

### 5.8 Norm layers (epic PR5) — LayerNorm/GroupNorm contract + error analysis

Native `ARITH_BFP` ships on BOTH sides of both norms: the forward and all
three backward ops (`dgamma`/`dbeta`/`dx`) run as `executeOp` funnel ops for
LayerNorm and GroupNorm. Everything is funnel-routed — **the §5.7 packed-walk
reviewer inventory does NOT grow**: the four norm ops consume the funnel's
unpacked-BFP scratch (borrowed or staged, §5.3) and produce through the
OUT_WRITE/ACC epilogues, and the BFP stats reducers read that same unpacked
scratch. A raw `->data` walk over PACKED BFP bytes in `LayerNorm.c` or
`GroupNorm.c` remains a bug, exactly as §5.7's closing rule says. The
end-to-end capstone — a uniform-BFP
conv → groupNorm → relu → adaptiveAvgPool → flatten → layerNorm → linear →
softmax model training natively with NO pins anywhere (since PR6 that includes
softmax, which runs its native forward there; its native backward is not
reached, because CrossEntropy's fused gradient makes the loop skip the layer),
both norms' dx EXECUTING — is `testBfpUniformNormModelTrainsAndGridsMove`
(`test/unit/userAPI/UnitTestMultiLayerTraining.c`).

**R-N1 — wire-anchored staging.** Norms have no reduction-weight operand, so
GEMM rule 1 ("stage at the weights' widths", §§5.2/5.4) has nothing to anchor
on; the anchor is the layer's OWN produced-wire config — the R-P1 pool rule
at the norm layer. `outputQ` anchors the forward op; `propLossQ` anchors ALL
THREE backward ops, **including at `propLoss == NULL`** (a grads-only call
still stages `forwardInput`/`loss` at the `propLossQ` widths — gamma is a
dx-only operand — the anchor is a width source, not a write target). The check is EAGER at op
entry (`layerNormBfpWireAnchor`/`groupNormBfpWireAnchor`): an `ARITH_BFP`
math slot whose produced-wire config is NULL or non-BFP fails fast with a
guided message, because without it there is no width source at all. The
stage template is per-tensor `{1,0}` at the anchor's
`mantissaBits`/`exponentBits`, rounded by the op's own
`arithmetic.roundingMode` (§5.3's borrow/staged asymmetry: staging is a live
op choice) — ONE template per direction, shared by every FLOAT32-stored
operand of that direction's ops. Gamma and beta are funnel OPERANDS
(forward `{input, gamma, beta}`; dgamma `{forwardInput, loss}`, dbeta
`{loss}`, dx `{forwardInput, loss, gamma}`), so all three operands are
staged-or-borrowed under exactly the same rule; a BFP-stored operand is
borrowed zero-copy as everywhere (D8) and never re-blocked.

**R-N2 — stats: one Reduce authority, two mechanisms.** Both layers compute
mean and biased variance through the Reduce module's BFP arms
(`meanOverTrailingAxesBfp`/`varianceBiasedOverTrailingAxesBfp`,
`src/arithmetic/Reduce.c`) via the SAME shared per-layer helper
(`layerNormAllGroupStats`/`groupNormAllGroupStats`) in forward AND backward,
so a layer's passes can never desync on the stats definition (variance
BIASED, ÷N; eps INSIDE the sqrt). GroupNorm hands Reduce a `[B,G,cpg,T]`
alias VIEW whose flat index equals the storage index (the layer's
rank-3/identity-order gate makes the split of C a pure relabeling), so
`bfpGroupOf` still resolves the exponent group each code was actually packed
into.

- The MEAN is the R-P4 sum contract: one `int32` partial per same-exponent
  segment, folded into the float accumulator by an exact `ldexpf`
  power-of-two shift on every group crossing plus the tail, then one float
  divide by N. `bfpValidateSumHeadroom` fail-fasts at entry (the pure-sum
  bound `INT32_MAX >> (m−1)`, §5.6's biasGrad twin).
- The VARIANCE is per-element dequant-center-square in float32: dequantize
  exactly (`mantissa · 2^E`), subtract the block mean, square, accumulate —
  no `int32` partials, no headroom guard. A mantissa-domain variance is
  REJECTED for BFP: centering subtracts an arbitrary float mean, so the
  centered values sit on no shared `2^E` grid and no same-exponent integer
  partial can represent `(x−μ)²`. (The uncentered `Σx² − Nμ²` form does
  admit an int32 partial and is rejected for the other reason: it is
  catastrophically cancelling whenever `|μ|` dominates the spread.) This is the SYM_INT32 variance's own shape
  (dequant-center-square) carried onto the BFP grid, not a BFP shortcut
  forgone.

The Reduce module polices its own operands per contract: the BFP arms
require BFP in the unpacked scratch form, and — new with PR5 — **the FLOAT32
reducers fail fast on any non-FLOAT32 operand**
(`reduceValidateFloat32Operand`). That guard closes the silent-`float*`
hazard the widened norm dtype dispatch created: an unpacked BFP (or SYM)
int32 scratch operand reaching a `float*` read is the same 4 bytes per
element, so the failure mode is silent wrong arithmetic, not a crash. The
layer-side twin of the same ruling: both layers' stats dispatch and forward
kernel selects are explicit `switch`es with fail-fast defaults, never
ternaries (the PR2 Task 9 ruling; a ternary hands every unknown arithmetic
to the float kernel).

**R-N3 — normalize + affine in float32, pack once at OUT_WRITE.** After the
stats, the kernels dequantize each operand element EXACTLY
(`ldexpf((float)code, E − bias)` — a power-of-two multiply, no rounding),
normalize (`(x − μ)·invσ`) and apply the affine (`γ·n + β`) in the FLOAT32
raw, and let the OUT_WRITE epilogue derive the wire's fresh per-group
exponents and pack ONCE. There is no BFP analog of the SYM path's
`rescaleIntoAccumulatorScale` beta-seed / integer-affine bookkeeping, for
the same reason AvgPool's `s/K` fold has no BFP analog (R-P4): a BFP scale
is `2^E`, and neither the data-dependent `1/σ` stretch nor the β-into-`s_y`
rescale is a power of two in general — an "integer affine" would smuggle
the same float multiplies back in as rounded rescale factors, ADDING
roundings instead of saving any. The raw is FLOAT32 (D7); the only
BFP-specific rounding on the whole forward path is the final pack — the
error analysis below leans on exactly this.

**R-N4 — backward = three funnel ops (GEMM parity).** The `ARITH_BFP`
backward issues up to three `executeOp` calls, mirroring the GEMM family's
op split: `dgamma` (mode = `weightGradAccMode`, ACC), `dbeta` (mode =
`biasGradAccMode`, ACC), `dx` (mode = `OUT_WRITE` into the `propLoss`
wire). `frozen` skips both grad ops (the grad tensors do not exist, #380);
`propLoss == NULL` skips the dx op only. The ACC epilogues round by the
TARGET grad tensor's own storage config (`accumulateOut` deliberately keeps
the target's mode — #282's ownership split, per `ExecuteOp.c`'s
ACC-epilogue comment; a FLOAT32 grad target is a plain float add, i.e. no
REQUANTIZATION — the float add itself still rounds, and part 3 counts
those roundings), while staging and dx's OUT_WRITE round by
`propLossMath.roundingMode` (#282).

- **Stats are recomputed per op** — dgamma and dx each run the shared stats
  helper again; dbeta needs no stats at all (`dβ_j = Σ dy`). §10 owns this
  as the 3×-recompute register entry: the deliberate cost of keeping every
  funnel op self-contained (no cross-op stats cache to invalidate, no way
  for a cached μ/σ to desync from the forward definition).
- **dbeta is the R-P4 segment-fold** over the loss mantissas
  (`bfpValidateSumHeadroom` over the cross-block walk). LayerNorm's strided
  j-outer/g-inner walk typically closes a segment on every element — the
  contract degrades to per-element folds gracefully, never wrongly — while
  GroupNorm's contiguous per-(b,c)-row walk gets real multi-element
  segments.
- **dgamma is per-element float32**: `dγ_j = Σ dy·n` multiplies a
  mantissa-backed `dy` into the float `n` — a mixed product has no
  same-exponent `int32` partial — so each term is dequantized exactly and
  accumulated in float.
- **dgamma/dbeta memset their FLOAT32 raw** before writing: the funnel's
  Phase-2 raw is uninitialized scratch (#427). The two memsets are NOT
  equally load-bearing, documented honestly: dgamma ACCUMULATES into the
  raw (`out[j] += …` across blocks), so its memset guards every call;
  dbeta OVERWRITES every element (`out[j] = acc`), so its memset is
  load-bearing ONLY on the empty-geometry early-out (`G == 0`/`N == 0`,
  resp. `K == 0`), where the ACC epilogue would otherwise fold
  uninitialized raw into the grad — on every non-empty path removing it is
  an equivalent mutant (the Task 5 finding, recorded here instead of
  faked test coverage).
- **dx validates ALL operand grids, including gamma's**
  (`validateBfpQConfigShape` per operand): the element-count gate alone
  cannot catch a malformed `{numGroups, groupSize}`, and `bfpGroupOf`
  would then index `exponents[]` out of bounds.

**R-N5 — grad storage: the norms join the per-tensor BFP knob.** Because
dgamma/dbeta land through the funnel's ACC epilogues, the PR3 grad-storage
machinery covers the norms with NO new code: a per-tensor `{1,0}` BFP
`weightGradStorage`/`biasGradStorage` template flows through `gradInit`'s
UNCHANGED carrier gate (grouped templates still rejected — "per-tensor
only, #300 axis"), and the `accumulateOut` BFP arm (§5.6) does the landing
(FLOAT32 raw increments requantized into the BFP grad under the target's
own storage rounding). `docs/FEATURES.md`'s former "BFP grad storage is a
GEMM-family-only feature" caveat is retired; the full round trip
(ACC → optimizer step → `optimizerZeroGrad`) is pinned by
`testBfpNormGradStorageAccumulatesAndSteps`
(`test/unit/userAPI/UnitTestMultiLayerTraining.c`). The knob rides the
`ARITH_BFP` backward — see R-N6's gap (a) for what happens when it is
combined with the FLOAT32 backward instead.

**R-N6 — factory coherence rules (the 8-rule set) + param geometry.** Both
norm factories (`layerNormLayerInit*`/`groupNormLayerInit*`,
`src/userApi/layer/{LayerNormApi,GroupNormApi}.c`) validate the
`layerQuant_t` profile with the same eight rules: **R1** gamma/beta storage
∈ {FLOAT32, SYM_INT32, BFP}; **R2** SYM_INT32 `forwardMath` ⇒ SYM_INT32
params; **R3** `ARITH_BFP` `forwardMath` ⇒ FLOAT32-or-BFP params AND a
BFP-typed `outputQ` (the R-N1 anchor, checked at construction so a
mis-wired profile dies with a factory message instead of at the first
forward); **R4** `ARITH_FLOAT32` `forwardMath` ⇒ FLOAT32 params; **R5**
SYM_INT32 `propLossMath` ⇒ SYM_INT32 `forwardMath` (the reverse stays
constructible — the SYM inference-only profile); **R6** `ARITH_BFP`
`propLossMath` ⇒ FLOAT32-or-BFP params AND a BFP-typed `propLossQ` (R3's
backward twin — a FLOAT32 forward with a BFP backward is mechanically legal
and stays allowed); **R7** `ARITH_FLOAT32` `propLossMath` rejects BFP
ANYWHERE among gamma/beta storage and the two wire configs; **R8** grad
storage is delegated to `gradInit`'s carrier gate (no duplicate check).

BFP param allocation is **constant-fill, Decision-5-derived**. The norms are
the one factory family that allocates non-FLOAT32 params directly (no
`requireFloat32` gate — #270 covers RANDOM init, and gamma = 1 / beta = 0
are constants that are exact grid points on any BFP grid). Geometry follows
the wire-allocator Decision 5 rule applied to params
(`layerNormParamQLike`/`groupNormParamQLike`): the storage template's
`groupSize` is honored, `numGroups` is derived from THIS parameter's
element count (never taken from the shape-agnostic template),
`groupSize == N` normalizes to the per-tensor `{1,0}` config (a `{1,N}`
spelling violates the config grammar), and a non-divisor `groupSize` aborts
with a guided message.

**Rule 4/7's honest asymmetry — fake-quant-over-BFP-params is
factory-unconstructible for norms.** The GEMM family supports the
fake-quant profile (FLOAT32 math over BFP-stored params/wires) because its
ENTIRE backward is funnel-routed: the prologue dequantizes any storage
dtype. The norms' FLOAT32 backward
(`layerNormBackwardFloat`/`groupNormBackwardFloat`) is NOT funnel-routed —
it raw-casts every operand, the grads and the dx wire to `float*` — so a
FLOAT32-math norm over BFP params would forward fine and die at the first
backward: a trap, not a feature, hence rules 4 and 7 reject it at
construction. Escape hatches, in honesty order: keep the norm's params and
wires FLOAT32 (the profile the guided messages steer toward); insert a
Quantization layer around the norm so the explicit converter owns the dtype
change; or, for a forward-only experiment, pin `cfg->forwardMath` directly
on a hand-wired config (the config path has no factory rules — the
test-side forward pin twins do exactly this).

Two gaps the rule set deliberately does NOT close — owned here rather than
papered over:

- **(a) R7 covers params + wires, NOT grad storage.** A FLOAT32-everything
  profile (FLOAT32 math, params, wires) with a per-tensor-BFP
  `weightGradStorage`/`biasGradStorage` passes every factory rule — R8
  delegates to `gradInit`, whose carrier gate accepts a per-tensor BFP
  template — builds fine, and dies at the FIRST backward at the runtime
  #261 guard ("packed grad storage requires the funnel route"). This is
  the PRE-EXISTING #261 packed-grad hole of the norms' FLOAT32 backward,
  with BFP now joining SYM/ASYM in it; closing it means funnel-routing the
  float backward's grads (the #261 follow-up), not another factory rule.
- **(b) The R5-vs-R7 asymmetry + the zero-init trap.** R5 leaves the SYM
  inference-only profile constructible (SYM_INT32 forward + FLOAT32
  backward: the runtime backward guard rejects training it, inference
  works). Its BFP twin (`ARITH_BFP` forward + FLOAT32 backward) is
  DELIBERATELY unconstructible — R7 rejects it, because the BFP profile
  requires BFP wires (R3) and the FLOAT32 backward raw-casts them. And
  since `ARITH_FLOAT32 == 0`, a hand-built `layerQuant_t` that sets the
  BFP wires but FORGETS to set `propLossMath` arrives with a
  zero-initialized slot that reads as `ARITH_FLOAT32` and trips R7 — a
  zero-init trap whose guided message points at declaring `ARITH_BFP`,
  which is almost always what such a caller meant.
  `layerQuantInitUniform` sidesteps the trap entirely (it derives all four
  slots from the one config); it exists only for field-wise hand
  construction.

**R-N7 — two math slots, not four.**
`layerNormConfig_t`/`groupNormConfig_t` carry `forwardMath` and
`propLossMath` only; `propLossMath` governs ALL THREE backward ops (their
`executeOp` calls all pass `.arithmetic = cfg->propLossMath`). This is a
deliberate asymmetry against the GEMM family's four slots
(`weightGradMath`/`biasGradMath` there): the norms' three backward ops are
three views of ONE derivation (same stats, same dequants, same arithmetic),
no experiment has asked for a mixed-arithmetic norm backward, and growing
the config struct would ripple through every field-assign fixture in the
tree (the struct-growth lesson) for a knob with no user. Rounding still
splits per seam as everywhere: staging and dx's OUT_WRITE by
`propLossMath.roundingMode`, the two ACC landings by the grad target's own
storage mode (R-N4).

**Proof ladder: no bit-identity twin — and what replaces it.** The
epic's bit-identity proof rung (a BFP config expressible as grouped
SYM with `2^E` scales runs bit-identical to the grouped-SYM path) exists
for the GEMM family and CANNOT exist for the norms: the SYM norm kernels
derive DYNAMIC, data-dependent output scales (forward
`s_norm = absmax/qMax` via the global-stretch scheme; backward dx's scale
likewise, refreshed every call), so even a power-of-two-input config
diverges structurally — the SYM path's produced grid is an arbitrary float
scale, the BFP path's is a power of two, and the two paths are not the same
sum in a different order but different QUANTIZERS. What replaces the rung:

- **config-pin forward twins on grid-exact fixtures**
  (`testLayerNormForwardBfpFakeQuantPinTwin`,
  `testGroupNormForwardBfpFakeQuantPinTwin`): the same layer runs native
  vs. `ARITH_FLOAT32`-pinned fake-quant on fixtures whose values are exact
  grid points, and the packed outputs must agree;
- **a test-side backward cross-check**
  (`testGroupNormBackwardBfpFakeQuantPinCrossCheck`): native dgamma/dbeta
  asserted memory-equal against a test-side FLOAT32 twin; native dx
  asserted payload- AND exponent-equal against `convertTensor` over the
  twin's float dx;
- **bit-exact NumPy gold for LayerNorm**
  (`test/unit/layer/generate_expected_bfp_layernorm.py` → the
  `testLayerNorm{Forward,Backward}BfpGold*` fixtures): every scalar step
  emulated in `np.float32` in the C statement order, forward (native +
  staged) and backward, asserted code- and exponent-exact;
- **twin-sanity for GroupNorm**
  (`testGroupNormForwardBfpTwinSanityTwoGroups`,
  `testGroupNormBackwardBfpTwinSanityTwoGroups`): dequantized outputs
  against the existing FLOAT32 gold within a small multiple of the pack
  band. A dedicated bit-exact GroupNorm gold generator (mirroring
  LayerNorm's) is the filed follow-up if twin-sanity ever proves too
  coarse — the same precedent as the missing SYM GroupNorm gold.

**Error analysis (the `0.5·C·s_acc`-style deliverable,
mirroring `docs/conventions/arithmetic-sym.md` §"Grouped backward").**

1. *Mechanism.* Every norm op computes its stats, normalization and affine
   in float32 FROM EXACT DEQUANTS: `mantissa · 2^E` is a power-of-two
   multiply (`ldexpf`), exact in float32, and the mean's/dbeta's `int32`
   segment partials and folds are exact under the sum-headroom guard — up
   to §7's `(float)partial` conversion point above `2^24`, the one place
   a headroom-legal partial can still round (cross-referenced in §10). §7's
   "concrete norm corner" works that case through on a norm: at `m = 16` a
   constant-code input long enough to push its segment past `2^24` shifts the
   mean off its exact value and makes the norm emit ≈ `0.525` where the true
   answer is `0`. The ONLY BFP-specific rounding on any norm path is the
   OUT_WRITE pack of the produced wire (forward y, backward dx) — one
   `roundByMode` per element — plus, when a grad tensor is STORED BFP, the
   ACC landing's own requant (§5.6's engines, a storage property, not a
   kernel one). Everything else is ordinary float32 arithmetic noise.
2. *Per-element pack bound.* The pack rounds each element to its group's
   grid: |err| ≤ `0.5·s_out` per element under HALF_AWAY, where
   `s_out = 2^{E_g}` is the scale of the element's group in the DERIVED
   wire grid (`E_g` the unbiased exponent `deriveBfpStoredExponent` snaps
   to from the raw's per-group absmax). `SR_HALF_AWAY` (lift iff `u < frac`) rounds to
   one of the two neighbouring grid points — unbiased in expectation, per-draw
   error < `1·s_out`.
   Because `s_out ≈ absmax_g/qMax`, the bound is RELATIVE ≈ `2^{−(m−1)}`
   of the group's largest magnitude — the mantissa-width sweep axis,
   directly.
3. *Aggregate float32 noise.* An N-term float32 sum carries worst-case
   relative error ≤ `(N−1)·2^{−24}`, RMS ≈ `√N·2^{−24}` under the
   independent-rounding assumption. The BFP MEAN does no worse than
   the FLOAT32 path's own mean: its segment folds are EXACT (`ldexpf`
   power-of-two shifts of exact `int32` partials), so its float roundings
   number `C ≈ ⌈N/groupSize⌉` fold-adds (plus one divide) instead of the
   FLOAT32 reduction's N adds — never MORE, and equal only at
   `groupSize == 1`, where every element is its own segment. The variance and the backward reductions
   accumulate per element in float32 (part 5's table) — the SAME rounding
   counts as the FLOAT32 kernels' own loops; BFP adds nothing there.
4. *Dominant term.* At every practical `(m, N)` the pack term dominates
   the stats term. On the same relative footing: the pack is
   ≈ `2^{−(m−1)}` of the group absmax (≥ `2^{−15}` even at the m = 16
   ceiling), the float32 stats noise is ≈ `√N·2^{−24}` (≈ `2^{−20}` at
   N = 256; still only `2^{−16}` at an absurd N = 2^16). In absolute form:
   an m = 8 wire whose output block absmax is O(1) — anywhere down to
   ~`2^{−6}` — derives `s_out ≥ 2^{−13}`, so the pack band
   `0.5·s_out ≥ 2^{−14}` sits ~`2^6` ABOVE the ≈`2^{−20}` stats noise. The
   norms' BFP error is therefore **quantization-bound, not
   accumulation-bound — the OPPOSITE of the GEMM finding** (§§7–8: there
   the interesting error lives in the accumulation seams — the >2^24
   partial conversion, the ±inf fold — while the pack is routine).
   Practical consequence: sweeping `mantissaBits` moves norm accuracy
   directly, and sweeping the wire's `groupSize` moves it too — through
   the SAME pack term, since `s_out ≈ absmax_g/qMax` is PER-GROUP and a
   finer blocking tightens each group's absmax (D5's rationale, the
   wire-block-size sweep axis); what barely matters is N and
   the fold geometry, which touch only the subdominant stats term.
5. *Rounding counts per path* (float32 roundings per OUTPUT element; exact
   dequants and exact `int32` partials/folds contribute none; per-group
   stats terms are shared by that group's N outputs and listed as group
   totals):

   | op | float32 roundings per output element |
   |---|---|
   | forward | 4 per-element (center, ·invσ, ·γ, +β) + 1 pack; per group: mean `⌈N/groupSize⌉` fold-adds + 1 divide, variance ~3N, rsqrt ~3 |
   | dgamma | per param element: G (LayerNorm) resp. B·T (GroupNorm) adds plus the per-term products; no pack (FLOAT32 raw; ACC landing per §5.6 if the grad is stored BFP) |
   | dbeta | per param element: one fold-add per closed segment — between `⌈#terms/groupSize⌉` (GroupNorm's contiguous rows) and #terms (LayerNorm's strided walk); the `int32` partials are exact; no pack |
   | dx | ~7 per-element (n = 2, dn = 1, two subs, two muls) + 1 pack; per group: 2N sum roundings (Σdn, Σdn·n) plus each sum term's own products, + 2 divides + the stats recompute (as forward) |

6. *Relative-error class (dx).* dx's signal is O(invσ·|dy|) per element;
   its noise is the deterministic pack band O(`0.5·s_dx`) — `s_dx` derived
   from the dx raw's own per-group absmax, i.e. relative ≈ `2^{−(m−1)}` of
   the block's largest dx — plus float32 dust of relative order
   `√N·2^{−24}`. dx therefore stays in the single-quantization-step class:
   ONE pack of the true float32 dx, never an accumulation of quantized
   partials. The shipped tests derive their tolerances from exactly this:
   the LayerNorm gold needs NO tolerance at all (the generator emulates
   every float32 rounding statement-for-statement, so gold is code- and
   exponent-exact), and GroupNorm's twin-sanity window (5e-2 absolute
   against the FLOAT32 gold) is a small multiple of the fixture grid's
   worst-case pack step.
7. *Research deviation.* There is no literature template for NATIVE BFP
   norm layers — HBFP/MSFP/FAST/MX quantize GEMM operands and leave norms
   in higher precision; the epic's goal of native LN/GN beyond the
   literature, each with its own error analysis, is realized HERE. The
   documented boundary of "native": mantissas enter the ops unpacked and
   exact, but the stats/normalize/affine arithmetic is float32 (bridges),
   and the block structure re-enters at exactly one point per op — the
   produced-wire pack. §10 records this boundary as a register entry.
8. *Escape hatch.* `propLossMath = ARITH_FLOAT32` is the exact backward —
   but ONLY on FLOAT32 wires and params: that arm raw-casts every operand
   (R-N6's rules 4/7), so it is not reachable over BFP storage. On BFP
   wires the honest alternatives are a Quantization layer around the norm
   (the explicit converter owns the dtype change) or FLOAT32 storage for
   the norm's own wires; there is no norm equivalent of the GEMM family's
   pin-the-slot fake-quant-over-BFP profile — deliberately (R-N6).

### 5.9 Softmax (epic PR6) — I-BERT adaptation contract + error analysis

Native `ARITH_BFP` ships on BOTH sides. The forward runs an INTEGER pipeline —
I-BERT's Algorithm 3 (`bfpIExpQ`, `src/arithmetic/BfpSoftmaxExp.c`) on a fixed
dyadic work grid `2^-14`, fed by a block-alignment step that has no I-BERT
analog (`softmaxExpCodesBfp`, `src/layer/Softmax.c`) and closed by an
integer normalisation into the `BFP_RAW` raw (`softmaxForwardBfpRaw`, §13)
— and the backward is ONE `executeOp` op that recomputes steps 1-4 through
the same helper and normalises in float32 (`softmaxValuesBfp`). Both are
funnel-routed, so **the §5.7 packed-walk reviewer inventory does NOT grow**:
the kernels consume the funnel's unpacked-BFP scratch (borrowed or staged,
§5.3) and produce through the OUT_WRITE epilogue; a raw `->data` walk over
PACKED BFP bytes in `Softmax.c` is still a bug. §5.7's "Softmax's BACKWARD is
the FOURTH outside-funnel site" now describes its FLOAT32/SYM arms only —
those raw-cast their wires and therefore reject BFP storage outright (R-S5).
End-to-end evidence: `unitTestUniformBfpSoftmaxMseTrains`
(`test/unit/userAPI/UnitTestMultiLayerTraining.c`) is the model that actually
drives the native BACKWARD through the training loop — MSE, not CE, because
CrossEntropy's fused gradient makes the loop skip the softmax layer — while
the two CE capstones (`testBfpUniformPoolActivationModelTrains`,
`testBfpUniformNormModelTrainsAndGridsMove`) now run the native forward with
no softmax pins at all.

**R-S1 — wire-anchored staging, ONE op per direction.** Softmax has no weight
operand, so the staging anchor is the layer's OWN produced-wire config, the
R-P1/R-N1 rule at this layer: `outputQ` anchors the forward op, `propLossQ`
the backward op, checked EAGERLY at op entry through the now-shared
`bfpWireAnchor` (`BfpKernelSupport.h`) — an `ARITH_BFP` math slot whose
produced-wire config is NULL or non-BFP fails fast, because without it there
is no width source at all. The stage template is per-tensor `{1,0}` at the
anchor's `mantissaBits`/`exponentBits`, rounded by the op's own
`arithmetic.roundingMode`; a BFP-stored operand is borrowed zero-copy (D8) and
never re-blocked. The backward's operands are `{x (logits), loss}`, both under
that one template. Pinned by `unitTestSoftmaxForwardBfpStagedWidths` /
`unitTestSoftmaxBackwardBfpStagedLoss` (widths) and the two
`…RequiresBfp{OutputQ,PropLossQ}` death tests (the eager gate).

**R-S2 — the integer pipeline and its three lossy shift sites.** Per element,
with `E_i` the element's unbiased block exponent and `EMax` the exponent of
the block holding the SIGNED max:

1. *Max* — one pass of the exact `(code, E)` compare `bfpValueGreater`
   (§13 "The pool forwards" item 1; needs `m <= 16`, checked), strict `>`
   keeps the FIRST max, ties across blocks included. Until 2026-11-04 this
   was an `ldexpf` compare, equal on every input where the dequant stays a
   finite normal float32 (it overflowed to `inf` above `2^128` at e = 8).
2. *Align* onto the work-grid-or-coarser common grid: `sigma = EMax + 14`,
   `down = max(0, -sigma)`, net shift `s_i = (EMax - E_i) + down`. `s_i >= 0`
   is a **rounded right shift** (`bfpShiftRightRounded`, exact at `s_i == 0`)
   — the FIRST lossy site; `s_i < 0` is an EXACT saturating left shift, which
   is the case a block COARSER than the argmax block lands in. `mMaxW` is the same
   rounded shift of the max mantissa by `down` — the SECOND lossy site (inert
   at `down == 0`, i.e. whenever `EMax >= -14`). `down` folds the descent onto
   a finer-than-work storage grid into that same single shift: never two
   chained roundings per element.
3. *Promote* `qT_i = min(aligned_i - mMaxW, 0)` onto the work grid — EXACT by
   construction (`sigma >= 31` ⇒ any nonzero deficit underflows; `0 <= sigma <
   31` ⇒ a left shift guarded by `thr = -ceil(363392 / 2^sigma)` so the shift
   stays inside int32; `sigma < 0` ⇒ `qT` already IS on the work grid).
4. *i-exp* `bfpIExpQ(qW_i, mode)` — its `>> z` renormalization is the THIRD
   lossy site.
5. *Normalise.* FORWARD (integer, 2026-11-04, `softmaxNormalizeRaw`):
   the common scale `0.3585 · 2^-28` cancels, so `s_i = qe_i / Σ qe`
   directly. `Σ qe` reaches `n · QE_MAX` (`QE_MAX = QB² + QC = 748 954 122`,
   the i-exp's largest output) and leaves int32 for `n >= 3`, so the sum
   is kept as `T = hi · 2^p + lo` (`lo < 2^p`, one carry per term) with `p`
   the smallest `p >= 0` such that `n · QE_MAX <= INT32_MAX · 2^p` (a
   compile-time table; HAR's `n = 6`: `p = 2`) — `hi` fits by that choice —
   and the divisor is `S = round_half_up(T / 2^p)`. Then
   `mant_i = round_half_up(qe_i · 2^(30-p) / S)` by a 32-step restoring
   division (no hardware divide; `30 = BFP_SOFTMAX_RAW_FRAC_BITS`), at raw
   exponent `-30` in EVERY raw group (the output geometry does not matter:
   all elements share one scale). The division starts at
   `r = qe_i >> (p+1)`, which must be `< S` for the quotient to fit 32
   bits; `S >= T/2^p - 1/2` and `T > 2^p` (the argmax's `qe` is `≈ QE_MAX`)
   give it, and it is CHECKED once per call (fail-fast). `|mant| <= 2^30 (1
   + 2^(p-1)/T) + 1 < 2^31`. Rounding is HALF_AWAY (all values `>= 0`, so
   half up), never SR and never the knob's: it is a fourth rounding site,
   outside the knob, and the re-block owns the op's mode. Error per output
   `<= 2^-31` from the division plus a common relative `<= 2^(p-1) / T <=
   2^(p-30)` from `S`; `|Σ mant − 2^30| <= n/2 + 2^p + 1`.
   BACKWARD recompute (R-S4, float32 as before): `e_i = ldexpf((float)qe_i *
   0.3585f, -28)`, float32 sum in index order, divide (R-S3).

The `min(qT, 0)` clamp is load-bearing under SR only: the deterministic modes
are monotone, but an SR draw can jitter to `+1`, which would enter `bfpIExpQ`
as `qW > 0` and make `z` negative; the clamp maps such jitter to `exp(0)`.
Two saturation corners are accepted by design rather than guarded: a NEGATIVE
mantissa whose exact left shift leaves int32 becomes the `INT32_MIN / 2`
sentinel (part 4 bounds its true weight), and a ZERO mantissa with `up >= 31`
returns its exact `0` through an explicit clause, because `0 << 32` is
shift-count UB (C11 6.5.7p3), not merely a value question.

**The only int32 accumulator is the forward's step-5 sum**, whose headroom
is the static `p` of step 5 (no `bfpValidate*Headroom` has a role here —
unlike every other BFP kernel family). The other int32 headroom argument is
the polynomial's, and it is what fixes `F = 14`: `q_p ∈ (-QLN2, 0]` gives
`|q_p + q_b| <= q_b = 22167`, so `qL <= 22167² + QC = 748 954 122 < 2^31`; at
`F = 15` the square alone is ≈ 1.97e9 and overflows. The constants
(`QLN2 = 11356`, `QB = 22167`, `QC = 257 578 233`, `QFLOOR = -363 392`) are
COMPILE-TIME floors of their real values, re-derived by the goldgen and pinned
in C against a double-precision recomputation (`testShiftConstantsMatchDerivation`).

**The knob (`bfpExpShiftRounding`).** `BFP_SHIFT_TRUNC` (default,
I-BERT-faithful `>>`), `BFP_SHIFT_HALF_AWAY`, `BFP_SHIFT_SR` (the integer
draw since 2026-11-04: lift iff the draw's top `k` bits `<` the `k` dropped
bits, one `rngNextU32` per shift — for `k <= 24` the decision the 24-bit
float draw took from the same word, for `25 <= k <= 31` the exact one) — set through
`softmaxSetBfpExpShiftRounding` (`SoftmaxApi.h`). It governs the three
INTEGER shift sites above and NOTHING else: it is ORTHOGONAL to every
`roundingMode_t` in the config, never derives from one, and staging plus the
OUT_WRITE pack keep the ordinary `roundingMode_t` machinery. It is inert
unless a math slot is `ARITH_BFP`, and it is **not serialized** — the ODTS
record carries the two arithmetics and the two wire configs only, the same
class as Dropout's `training` flag: an experiment knob, not a model property.
Deserialization fills an ALREADY-CONSTRUCTED layer, so a reloaded model runs
at whatever its factory set (`BFP_SHIFT_TRUNC`); a sweep that varies the knob
must re-apply the setter after loading.

**R-S3 — the float32 boundary (what "native" means here).** Exact:
the max compare, the alignment/promotion arithmetic, the `dLds` dequant in
the backward. The FORWARD has no float32 left (2026-11-04): step 5 is the
integer normalisation and the raw is `BFP_RAW` (§13), re-blocked in
integers. Float32 on the BACKWARD only: the exp weights `e_i` after
`(float)qe_i`, the cross-element sum, the normalize divide, and the whole
elementwise/dot arithmetic; its raw stays FLOAT32 (D7). The block structure
re-enters at exactly ONE point per op — the produced wire's OUT_WRITE pack,
i.e. §10's "native OPERAND handling plus a natively derived produced grid"
boundary in softmax's shape (§11 records it).

**R-S4 — backward = ONE funnel op that recomputes from the LOGITS.** The
`ARITH_BFP` arm issues a single `executeOp` (`OUT_WRITE` into `propLoss`,
anchored at `propLossQ`, arithmetic `propLossMath`): it recomputes `s` from
the logits via `softmaxValuesBfp` with the layer's OWN knob, dequantizes
`dLds` exactly, accumulates `dot = Σ s·dLds` in float32 index order and emits
`s·(dLds - dot)`. The recompute means the backward differentiates the
UNPACKED forward values — the packed forward wire differs from them by one
pack rounding — which is the norms' R-N4 cost note verbatim (since the
integer forward, the backward's `s` is the float32 normalisation of the
SAME `qe` codes, so it differs from the forward's raw by at most
`2^-31 + 2^-22 · s` — the goldgen asserts it per fixture), and it is what
makes the layer stateless between passes. Two gates: the operand grid
(`validateBfpQConfigShape`) and the loss ELEMENT COUNT (the #436 OOB class —
both walks run `x`'s flat `n`, and a shorter per-tensor `{1,0}` loss passes
every grid check while the loss read runs off its scratch). Softmax has no
param grads, so **`propLoss == NULL` returns before the anchor gate binds** —
no op runs at all. This is a DELIBERATE deviation from R-N1's
anchor-binds-even-when-NULL rule and is sound for the same reason the rule
exists: the norms still have `dgamma`/`dbeta` needing a width source at
`propLoss == NULL`, softmax has nothing left to run. The ordering is pinned
(`unitTestSoftmaxBackwardBfpNullPropLossIsNoOp` carries the same broken config
the anchor death test dies on). The recompute is NOT a BFP-specific cost: ALL
THREE arms recompute `s` from the logits, because the training loop hands
every backward the layer's INPUT — §11's correction 1 records why that had to
be fixed first.

**R-S5 — the non-BFP arms.** Dispatch is an explicit `switch` with a
fail-fast default in BOTH directions (the PR2 Task 9 ruling: a ternary hands
every unknown arithmetic to the float kernel). `ARITH_FLOAT32` is the exact
float path. `ARITH_SYM_INT32` keeps its documented FAKE-QUANT meaning —
softmax never had a native SYM kernel — and the forward's `opSpec` therefore
hardcodes `ARITH_FLOAT32` rather than passing `cfg->forwardMath`: passing it
would make the funnel prologue unpack into int32 scratch that the float kernel
then misreads through a `float*` cast. Both backward arms run OUTSIDE the
funnel and raw-cast their wires, so each carries `bfpRequireNoBfpWire` on all
three (`testSoftmaxBackward{Float,Sym}ArmRejectsBfpWire`). For the FLOAT32 arm
that guard is memory safety (a ~4× over-read of a packed payload, an over-write
into it); for the SYM arm it is POLICY — `conversionMatrix[BFP][FLOAT32]`
exists, so the arm would silently "work" — the outside-funnel twin of the
funnel's Decision-11 deny.

**R-S6 — the silent flip, and a FORWARD-ONLY escape hatch.** Because
`arithmeticFromQuantization` derives `ARITH_BFP` from BFP storage, an existing
model whose softmax merely CROSSED a BFP wire as a fake-quant bridge now runs
the native i-exp kernel on both sides, with no config change — the breaking
change of this PR. Pinning `forwardMath = ARITH_FLOAT32` restores the
fake-quant forward over BFP wires (the funnel dequantizes any storage dtype).
The BACKWARD has no such hatch: pinning `propLossMath = ARITH_FLOAT32` over a
BFP wire FAILS FAST at the arm's own guard, so a fake-quant softmax backward
is not constructible over BFP storage at all. That asymmetry is deliberate and
is the norms' §5.8-part-8 asymmetry in a different shape (there the FLOAT32
backward is unreachable over BFP storage too, only via the factory rules
instead of a runtime guard). The honest backward opt-outs: CrossEntropy
coupling (the fused gradient makes the loop skip the layer entirely), a
Quantization layer around the softmax, or FLOAT32 wires.

**R-S7 — single row only (#152).** The native pipeline runs ONE
max/alignment grid and ONE partition sum over the whole wire, while the
FLOAT32/SYM arms normalize per row (row = axis 0 over everything after it; a
rank-1 input is one row). Per-row BFP softmax is out of scope of the
FLOAT32-only micro-batch work, so both `ARITH_BFP` arms fail fast at op entry
when the input has more than one row (`softmaxBfpRequireSingleRow`; the
backward check sits after the `propLoss == NULL` return, R-S4's ordering).
`[N]` and `[1, N]` are one row and run unchanged
(`unitTestSoftmax{Forward,Backward}BfpSingleRowRank2MatchesRank1`,
`testSoftmax{Forward,Backward}BfpRejectsMultiRow`).

**Error analysis (softmax, mirroring
§5.8's parts).**

1. *Mechanism.* Four error sources, and no others: (i) the alignment/`mMaxW`
   shifts (part 4), (ii) the i-exp polynomial plus its `>> z` (part 2),
   (iii) float32 dust at the boundary (part 6), (iv) the ONE BFP rounding, the
   OUT_WRITE pack of the produced wire (part 5) — plus, when the operand is
   FLOAT32-STORED, its staging quantize, which is the ordinary §5.4 cost and
   not softmax-specific. Everything else is exact: the max-compare dequants,
   the `s_i < 0` left shifts, the work-grid promotion, the backward's `dLds`
   dequant.
2. *i-exp accuracy.* I-BERT reports `max |i-exp - exp| ≈ 1.9e-3` for its own
   calibrated setting (paper §3.5). Our FIXED-grid variant at the default
   TRUNC knob measures **2.0896e-3** worst-case over 2001 points on `[-20, 0]`
   (goldgen sweep, 2026-09-08); `2.5e-3` is the acceptance bound every
   value-level test uses (`testIExpAccuracyEnvelope`, the layer goldgen's
   proximity self-checks). It is an ABSOLUTE error on the unnormalized weight
   (whose maximum is `exp(0) ≈ 1.000241` in this representation), it is FIXED
   by `F = 14`, and no knob moves it.
3. *Headroom / why no int64.* R-S2's `qL <= 748 954 122 < 2^31` bound holds
   for every reachable input because the `QFLOOR` gate caps `z <= 31` and
   `q_p ∈ (-QLN2, 0]`. The whole pipeline is int32 with the framework's
   no-int64 rule intact by construction, not by wider accumulators.
4. *Alignment error — the term the knob moves.* Write `u = 2^max(EMax, -14)`
   for one aligned unit. Because block exponents are absmax-snap-up minimal
   (§3), `a/qMax <= 2^EMax < 2a/qMax` for the argmax block's absmax `a`, so
   `u ≈ a·2^-(m-1) … a·2^-(m-2)` in the ordinary `EMax >= -14` regime. The
   error DISTRIBUTION is what matters, and it is not uniform over elements:
   - the argmax element carries ZERO alignment error under both deterministic
     modes — identical mantissa, shift and mode as `mMaxW`, so the two round
     the same way and `qT = 0` exactly; under SR its two INDEPENDENT draws can
     differ by one unit, which the `min(qT, 0)` clamp maps to `exp(0)`;
   - every element of a block with `E_i == EMax` is exact when `down == 0`;
   - elements in blocks COARSER than the argmax block (`E_i > EMax` — the
     legitimate negative-dominated case) take the exact left shift: also zero
     alignment error;
   - the error therefore CONCENTRATES on elements in blocks FINER than the
     argmax block (`E_i < EMax`), where the shift rounds: error in `(-u, 0]`
     under TRUNC, `|err| <= u/2` under HALF_AWAY, `< u` unbiased in
     expectation under SR.
   Since `u` is an absolute error on the EXPONENT ARGUMENT, it multiplies that
   element's unnormalized weight by `e^δ ≈ 1 + δ`. TRUNC floors toward `-∞`,
   so it systematically UNDERESTIMATES every rounded element while leaving the
   argmax exact: it SHRINKS non-max mass and sharpens the distribution toward
   the argmax — the measurable knob effect, and the reason SR (unbiased in
   expectation) is the interesting sweep arm and HALF_AWAY the symmetric
   control. Note the softmax-specific consequence: `u` scales with the LOGIT
   MAGNITUDE `a`, so the argument error is a fixed fraction ≈`2^-(m-2)` of the
   logit block's absmax — a model with large logits needs more mantissa bits
   on the softmax INPUT wire, where a norm's pack error would stay relative.
5. *The single pack.* The OUT_WRITE pack rounds each output element to its
   group's derived grid: `|err| <= 0.5·2^{E_g}` under HALF_AWAY, where
   `2^{E_g} ≈ absmax_g/qMax` — relative ≈`2^-(m-1)` of the group's largest
   PROBABILITY, i.e. §5.8 part 2 verbatim, the same mantissa-width sweep axis.
   `SR_HALF_AWAY` rounds to one of the two neighbouring grid points. This is the ONE BFP-specific
   rounding of the whole path (backward: of `dx`).
6. *Normalisation dust.* Forward: the integer step 5's `<= 2^-31` per
   output plus the common `<= 2^(p-30)` relative of `S` (step 5). Backward:
   `(float)qe_i` rounds above `2^24` (`qe` reaches ≈`7.49e8`) — §7's
   documented `int32→float` conversion class at a different site, relative
   `2^-24`, absorbed by the `2.5e-3` bound; the `N`-term sum in index order
   carries the usual `<= (N-1)·2^-24` relative worst case, the divide one
   rounding, and the backward's `dot` the same shape. All of it is orders of
   magnitude below parts 4 and 5 at any practical width.
7. *Dominant term.* Worked at `m = 8` on both wires with logits of order 1
   (argmax block absmax `a = 2` ⇒ `EMax = -5`): the alignment band is
   `u = 2^-5 ≈ 3.1e-2` on the ARGUMENT, i.e. up to ≈3% of a rounded element's
   weight — ≈`6e-3` absolute on a probability of 0.2; the pack band on an
   output group whose absmax is `p_max ≈ 0.5` (`2^{E_g} = 2^-7`) is
   `0.5·2^-7 ≈ 3.9e-3` absolute; the i-exp term is ≤`2.1e-3` absolute. **The
   BFP softmax is ALIGNMENT-bound** — the INPUT logit wire's `mantissaBits`
   (and, through the per-group absmax, its `groupSize`) moves accuracy first,
   the output wire's pack second, and the i-exp floor last; the margins are
   small at `m = 8` and widen with the logit magnitude, since only the
   alignment term scales with `a`. That is a third ordering: the GEMMs are
   accumulation-bound (§§7–8), the norms quantization-bound (§5.8 part 4),
   softmax alignment-bound. The backward inherits ALL of it — it recomputes
   the same `s` — and adds only its own pack, so `dx` stays in the
   single-quantization-step class.
8. *Degenerate corner.* When `EMax <= -46` (`sigma < -31`), `down` exceeds
   `bfpShiftRightRounded`'s documented `k <= 31` clamp and every code on that
   scale collapses to `{-1, 0}`, so every such element maps to `exp(0)` and
   the layer emits the near-uniform distribution — which for logits below
   `2^-31` in magnitude IS the true answer to ~`2^-30`; elements in far
   coarser blocks still take the exact left-shift route and underflow
   correctly. The `INT32_MIN / 2` saturation sentinel is likewise benign: it
   is only reachable when the aligned deficit exceeds `2^31` units, i.e. more
   than `2^17` in the exponent argument against an underflow floor of 22.18,
   so routing it to an exact `0` weight is correct, not approximate (the
   kernel comment's `exp(-16)` residual is a conservative restatement of this
   bound).

**Proof ladder: no bit-identity twin here either — and what replaces
it.** The bit-identity rung (a power-of-two BFP config running bit-identical to
grouped SYM) cannot exist for softmax for a STRONGER reason than the norms':
not only does the produced grid differ (the pack derives a dynamic `2^E` from
the raw's absmax, §5.8's argument verbatim), the two paths compute different
FUNCTIONS — the SYM/FLOAT32 arms evaluate `expf` in float, the BFP arm
evaluates a quadratic i-exp on an integer grid. What replaces the rung:

- **a bit-exact NumPy/`np.float32` goldgen**
  (`test/unit/layer/generate_expected_bfp_softmax.py`) mirroring the kernel
  statement for statement, for the forward at BOTH deterministic knob
  positions and for the backward — the two knob golds are script-asserted to
  DIFFER, so passing both proves the knob reaches the shift sites;
- **a native-vs-fake-quant RED guard** in that generator (the pinned
  `ARITH_FLOAT32` wire must differ from the native one, so a silent fallback
  to the float kernel cannot pass the gold);
- **corner fixtures with their own gold**: the coarse-negative-block regime
  (`unitTestSoftmaxForwardBfpCoarseNegativeBlock`, both knobs) and the
  saturation/zero-code clauses (`unitTestSoftmaxForwardBfpCoarseSaturation`,
  whose zero-shift clause is behavioral only under UBSan);
- **the integer core's own vectors + envelope**
  (`test/unit/arithmetic/UnitTestBfpSoftmaxExp.c`: constant re-derivation, per-mode
  shift vectors, SR bracket/determinism, the `2.5e-3` accuracy envelope);
- **value-level proximity self-checks** in the goldgen (every element within
  `2.5e-3` + half a pack step of a float64 softmax; the dequantized wire sums
  to 1 within the summed pack steps) — the layer-level statement that the
  integer pipeline computes a softmax at all.

---

## 6. D9: gather-formulated ConvT1d, not scatter

Every other ConvT1d kernel in the framework (`Float32`, `SymInt32`,
`SymInt32Grouped`) is **scatter-formulated**: the outer loop walks input
positions and each input element scatters its contribution into several
output positions. Under `ARITH_BFP` that formulation has no usable block
structure — consecutive products from one scatter step land in DIFFERENT
output elements, so there is no per-(output, exponent-group) run across which
one `int32` block partial could accumulate (§5.3's fold contract needs a
stable target across a whole segment).

`convTranspose1dKernelBfpGather` (D9; implemented in epic PR2) deviates from ODT's own scatter precedent by computing **output-centric**:
for every output position, `convTranspose1dTapsAt` enumerates its
contributing (input position, kernel tap) pairs, and the reduction walks
those taps like any other GEMM-family reduction — restoring the same
block-partial contract Matmul/Conv1d use. The SYM scatter cores are
untouched; this is an `ARITH_BFP`-only reformulation, not a topology change
(scatter and gather compute the identical mathematical result — enumeration
order only, pinned equal by
`testConvTranspose1dKernelBfpGatherAdjointSameParityWithScatter` and the D9
scatter-vs-gather cross-check built into the gold generator).

**Bias folded in after the reduction, not seeded before it — the reverse of
the scatter kernels.** The `Float32`/`SymInt32` scatter kernels zero-init
`output`, run the pure `+=` scatter accumulation over every (input, tap)
pair, and only THEN refold the bias in a separate post-pass over every output
element (the SYM kernel's own comment at the site: "Bias seed pass (refold),
separate from the pure-+= scatter"). The gather kernel inverts that order: it
dequant-seeds each output element's float accumulator with its bias value
(per §5.4 Decision 2) as the FIRST thing the per-output-element loop body
does, BEFORE its own reduction — there is no separate pass at all, because
the gather formulation already visits every output element exactly once and
folds the bias straight into the accumulator the reduction adds into.
`outputPadding` tail positions (no taps at all) simply stay at the bias seed,
which the gold fixture's tap-free output position pins.

## 7. Float32 `int32→float` partial conversion can round above 2^24

§5.3's fold step converts the finished segment's raw `int32 partial` to
`float` before the `ldexpf` power-of-two shift: `(float)partial`. `float32`
represents integers exactly only up to `2^24`; a `partial` whose magnitude
exceeds that loses precision in the conversion itself — an extra rounding
point beyond the design's stated claim that "the only added rounding point is
float32 addition across blocks" (D7).

This is unreachable at the `(mantissaBits, groupSize)` pairs the epic's sweep
axes exercise in practice: `bfpSegmentLimit` already caps a segment's product
COUNT well below where the partial's magnitude could reach `2^24` for any
headroom-legal segment at typical mantissa widths. It is real, not
hypothetical, at the wide-`m` end of the sweep, combined with long
same-exponent runs near the headroom ceiling — documented here rather than
guarded against, because guarding it would mean a second fail-fast stacked on
top of the headroom guard for a case the sweep may legitimately want to
explore (mantissa-width sensitivity is exactly what the epic measures).

**The concrete norm corner (PR5).** The norms' mean/dbeta segment folds sum
single mantissas rather than products, so `bfpSumSegmentLimit` caps a segment
at `INT32_MAX >> (m−1)` codes — orders of magnitude looser than the product
form, which is what lets a headroom-legal segment actually reach `2^24` here.
At `mantissaBits = 16` (`qMax = 32 767`) a per-tensor wire holding the
CONSTANT code `32765` over `N = 513` elements folds one segment of
`16 808 445 > 2^24`; `(float)partial` rounds it to `16 808 444`, the mean
lands at `32764.998…` instead of `32765`, and the normalize step then emits
≈ `0.525` (at unit scale with `eps = 1e-5`) for an input that is exactly
constant and whose true normalized output is `0` — §7's documented deviation
showing up at norm scale, on the one input where the answer is obvious. The
sweep-practical range is unaffected: at `m ≤ 8` (`qMax = 127`) a per-tensor
sum stays exact below `N ≈ 132 000`, far above any norm block the epic
exercises.

**The softmax site (PR6).** The same phenomenon at a place that is not a fold:
the BFP softmax converts each i-exp result with `(float)qe_i`, and `qe` ranges
up to `748 954 122` — ALWAYS above `2^24`, by construction rather than at a
sweep extreme. It is harmless there because the value is a probability weight
consumed immediately in float32: the relative `2^-24` is four orders below the
i-exp accuracy bound it feeds into (§5.9 parts 2 and 6). No headroom guard
applies to that path at all — softmax accumulates nothing in `int32`.

## 8. Exponent fold can overflow to ±inf at extreme combined exponents

`ldexpf(x, shift)` overflows to `±inf` when `shift` drives the result outside
`float`'s representable range. `shift = Ea + Eb − biasA − biasB` combines two
operands' STORED exponents; at the extreme ends of a wide `exponentBits`
sweep axis (up to 8 bits, bias up to 127 per operand) a segment whose
combined true exponent is large enough overflows the fold the same way
`ldexpf` overflows anywhere else in C.

This is consistent with D6's saturation-not-abort spirit (§2): D6 already
established that BFP's value-domain quantize path clamps rather than aborts
because the `exponentBits` sweep axis exists specifically to measure the
accuracy cost of exponent-range narrowing, including its saturation
behavior. The kernel fold reaching `±inf` under the same sweep pressure is
the SAME phenomenon showing up at the compute step instead of the quantize
step — not a new failure mode invented by the kernels, and not guarded
against for the same reason §2's saturation is not aborted: guarding it would
make the sweep's coarse end unrunnable instead of producing a data point.

## 9. D8 amendment (PR3): exact fold segmentation, not op-local re-blocking

**Amended 2026-09-02 (Leo, PR3 kickoff).** The original D8 ("op-local
blocking is a property of the op, storage blocking is a property of the
tensor") was written before any backward kernel existed. Read literally it
points at HBFP's own rule: recompute a block's exponents fresh, immediately
before a dot product whose reduction axis differs from storage order — which
is exactly the situation the backward GEMMs are in (a transposed or
output-centric reduction reads storage elements in a different order than
they were grouped in). §5.3's original text called op-local re-blocking
"exclusively a PR3/backward concern" and left the question open; this entry
records how PR3 actually resolved it.

Epic PR3 does **not** implement that recompute. For an ALREADY-QUANTIZED
(BFP-stored) operand, op-local blocking realizes instead as the PR2 **exact
fold-segmentation contract** carried over verbatim from the forward (§5.3,
§5.6): per-element `bfpGroupOf` lookup, folding the running `int32` partial
into the `float32` accumulator on EITHER operand's group change (plus a tail
fold) — never a re-quantize onto fresh reduction-axis blocks. A BFP-stored
operand's exponents are borrowed read-only (zero-copy) in the backward
exactly as in the forward; the only thing that changes between forward and
backward is HOW OFTEN a segment closes (the forward folds once per storage
group, since storage groups are contiguous along the forward's reduction
axis; the backward's transposed/output-centric walks cross group boundaries
more often, so they fold more often) — never what the exponents mean or
where they come from.

**Rationale — why re-blocking would be wrong here, not merely unimplemented.**
HBFP's recompute rule addresses giving a FRESH block structure to
UNQUANTIZED FP32 values along whatever axis a given dot product happens to
reduce over — there is no pre-existing quantization to disturb in HBFP's
setting. ODT's BFP-stored operands are already quantized AT STORAGE TIME,
under the tensor's OWN group grid. A literal op-local re-block would
requantize an already-quantized mantissa a SECOND time onto a different grid
immediately before the reduction: pure information loss with no compensating
benefit (a re-derived block exponent cannot recover precision the first
quantization already discarded) — exactly the **double-quantization** of the
gradient signal FAST names gradient precision as the sensitive axis for. It
would also break the backward's bit-identity twin against the grouped-SYM
path (the shipped proof-ladder mechanism): a re-block changes which
values participate in which `int32` partial, so the two paths would no
longer be computing the identical sum in a different order — they would be
computing genuinely different sums.
Fresh blocking — i.e. actual quantization — still happens exactly where it
always did: the FLOAT32-staging path (§5.4 Decision 1), per-tensor, at the
weight operand's own widths, identically for forward and backward.

**Deferred, not abandoned.** An opt-in reduction-axis re-block remains a
possible FUTURE perf knob — trading the accuracy cost of double-quantizing
grads for longer same-exponent fold segments, on hardware where many short
folds turn out to be measurably expensive. It is deliberately NOT
implemented here and is gated on a real MCU measurement motivating it, not filed speculatively.

## 10. Norm register entries (PR5): 3× stats recompute; float32 stats bridges bound "native"

Two register entries the PR5 norms add. The fold-related corners — §7's
`(float)partial` conversion above `2^24` and §8's `ldexpf` ±inf at extreme
combined exponents — apply to the norms' mean/dbeta segment folds exactly as
written there and are NOT re-derived here.

**3× stats recompute.** A full norm training step computes μ/invσ three
times per layer: once in the forward, once in `dgamma`, once in `dx`
(`dbeta` needs no stats) — and none of them is cached across ops. This
deviates from the obvious engineering move (compute once in the forward,
carry to the backward) deliberately: every funnel op is self-contained
(operands + ctx in, raw out), a stats cache would need a side channel
through the layer config, and a cached μ/σ can silently desync from the
recompute the moment any operand changes between calls — exactly the desync
class the shared Reduce helper (R-N2) exists to kill. The cost is bounded
(one extra O(n) Reduce pass per grad op) and is the price of the "one stats
authority" rule. An opt-in stats cache is a perf follow-up of the same
shape as §9's re-block knob: file it when an MCU measurement motivates it,
not before.

**Float32 stats bridges bound what "native" means.** The epic design
promised native LN/GN kernels "beyond literature … with a documented
boundary of what native means (float32 stats/accumulation bridges)". The
shipped boundary: operands are consumed as exact unpacked mantissas under
their own grids (never re-quantized, D8/§9), the mean's and dbeta's
within-segment arithmetic is exact `int32`, and everything else — variance,
rsqrt, normalize, affine, all backward reductions — is float32 from exact
dequants, with the block structure re-entering at exactly one point per op
(the produced-wire pack; §5.8's error analysis, part 1). "Native" for the
norms therefore means native OPERAND handling plus a natively derived
produced grid, NOT integer-only arithmetic — the honest reading the §5.8
error analysis quantifies, and the research deviation (no literature
template for BFP norms at all) it documents.

## 11. Softmax register entries (PR6): the I-BERT adaptations, and two corrections

Where §5.9 states the shipped contract, this section records what the softmax
work DEVIATES from — the source it adapts, and two claims that turned out to
be wrong. §7's `(float)partial` class and §8's `ldexpf` ±inf do not apply
here: softmax accumulates nothing in int32 and folds no exponent pair. The
`(float)qe` conversion above `2^24` is §7's phenomenon at a new site — §7
records it ("The softmax site"), §5.9 part 6 quantifies it, and neither is
re-derived here.

**Static input quantization → a fixed COMPILE-TIME grid.** I-BERT (Kim et
al. 2021, Alg. 3) runs its i-exp on the input tensor's calibrated static scale
and derives `q_ln2`/`q_b`/`q_c` from it. We pin the work
grid at `2^-F`, `F = 14`, and floor-quantize the three constants at BUILD
time. This is stronger than the paper's construction in one respect — the
constants carry no calibration error, they are exactly
`floor(ln2·2^14)`, `floor(1.353·2^14)` and `floor(0.344·2^28/0.3585)`,
re-derived by the goldgen and pinned in C against a double recomputation — and
it moves a cost elsewhere: the input must be brought ONTO that fixed grid,
which is the alignment step (§5.9 R-S2 step 2), the pipeline's dominant error
term (§5.9 part 4) and the piece with no I-BERT analog at all, because a BFP
tensor has per-group exponents where I-BERT has one tensor scale. `F = 14` is
not a tuning choice: it is the maximum for which the I-POLY square stays in
int32 (§5.9 part 3).

**TRUNC is the paper-faithful default; the knob is a research axis.** I-BERT's
Algorithm 3 renormalizes with an arithmetic right shift, i.e. floor. That is
`BFP_SHIFT_TRUNC` and it is the factory default. `HALF_AWAY` and `SR` are
ADDED positions, not corrections of the paper — they exist because the
alignment shift makes the rounding regime measurable at the layer's output
(§5.9 part 4: TRUNC sharpens toward the argmax, SR is unbiased in
expectation). The knob does not participate in serialization (§5.9 R-S2).

**A backward beyond the literature.** I-BERT is inference-only, and the BFP
literature (HBFP/MSFP/FAST/MX) quantizes GEMM operands while leaving softmax
in higher precision. A native BFP softmax BACKWARD therefore has no template
anywhere in the cited work; it is realized here as the R-S4 single funnel op
that recomputes `s` from the logits, with its own error statement (§5.9
part 7). The same sentence as the norms' entry applies to what "native" means
(§10): native operand handling and a natively derived produced grid, with a
float32 boundary that §5.9 R-S3 names exactly.

**Correction 1 (P6-1): the backward consumed logits as probabilities.** Every
softmax backward arm applied the Jacobian `s·(dLds − Σ s·dLds)` to the tensor
the training loop hands it — which is `layerOutputs[i]`, the layer's INPUT
(the logits), not the softmax OUTPUT the formula needs. The bug predates this
epic and was invisible because the unit fixtures fed PROBABILITIES, i.e. the
tests agreed with the bug. The root fix recomputes `s` inside the backward in
ALL arms (float, SYM, BFP), which is also why the BFP backward's recompute is
not an extra cost invented for BFP — it is the contract every arm now follows.
Consequence worth knowing: a wrong `dx` through softmax poisons the UPSTREAM
layer's weight grad, so the regression pin is an end-to-end one
(`unitTestSoftmaxMseBackwardThroughLoop`, torch-autograd gold through
`Linear → Softmax → MSE`).

**Correction 2: the `E_i <= EMax` alignment invariant was FALSE.** The design
claimed every block exponent is `<= EMax`, so the alignment shift could only
ever be a right shift, by "absmax monotonicity". That proof is about the
GLOBAL-absmax block; `EMax` is the exponent of the block containing the SIGNED
max. A block whose absmax is a large NEGATIVE value is legitimately COARSER
than the argmax block (`E_i > EMax`), and the first implementation mis-handled
exactly that case — the shift count went negative, wrapped through `uint32`,
and the affected elements received `≈exp(-x_max)` mass instead of their true
share (tens of percent wrong on the corner fixture). The shipped rule is
two-sided (§5.9 R-S2 step 2): a negative net shift is an EXACT left shift, so
coarser blocks contribute no alignment error at all, and the saturation
sentinel covers the one case where that left shift would leave int32. Recorded
here rather than silently fixed, because the false invariant is the kind a
reader would re-derive: the fix is not a guard bolted onto a correct rule, it
is a different rule.

## 12. Decision register

The decisions the text and code comments cite by ID. Only IDs cited somewhere
in the repository are listed.

**Epic design (2026-07-29) — cited as `D<n>` here, `BFP D<n>` elsewhere.**

| ID | Decision | Applied in |
|---|---|---|
| D3 | First-class BFP: a new `qtype_t BFP` and a new `ARITH_BFP` (both appended), built on the group-quant group machinery; a constrained-SYM shortcut was rejected (4-byte float scales, no exponent-width axis). | `Quantization.h`, `ArithmeticType.h`, §5 |
| D4 | Native `ARITH_BFP` everywhere, Softmax and norms included — except by decision: Dropout is non-native (float bridge), Flatten is a passthrough, losses are fake-quant (native CE/MSE an optional stretch). | §5.7–§5.9; `Dropout.c`, `MSE.c`, `CrossEntropy.c` |
| D5 | Wires are fully blocked; the wire-block size is independent of the weight-block size; per-tensor (`numGroups = 1`) is the degenerate sweep point. | §5.5, wire allocators |
| D6 | Exponent saturation, not abort: the shared exponent clamps to its representable range; at the top mantissas saturate to the grid `[-2^(m-1), qMax]`, at the bottom they flush toward zero. Deliberate deviation from the `#227` abort discipline — saturation is what the `exponentBits` sweep axis measures. Doc aligned to the shipped clamp 2026-10-31 (positive cap `qMax`, negative floor `-2^(m-1)`; clamp code unchanged). | §1, §2; `deriveBfpStoredExponent`, BFP cells of `TensorConversion.c` |
| D7 | The raw intermediate under `ARITH_BFP` is FLOAT32: kernels MAC mantissa products in `int32` within a block, fold each block partial with an exact `·2^(Ea+Eb)` float multiply, and accumulate in float32 across blocks. No `int64`. **Amended 2026-11-04 (integer-only forward, §13):** a forward op that sets `opSpec_t.bfpRawOut` emits `BFP_RAW` instead (int32 mantissas + one unbiased int16 exponent per TARGET group) and the OUT_WRITE epilogue re-blocks in integers; ops without the flag — every backward op — keep the FLOAT32 raw. | §5.3, §13; every BFP kernel, `ExecuteOp.c`'s BFP arms, `reblockBfpRawTensor` |
| D8 | Op-local blocking: compute blocking belongs to the op, storage blocking to the tensor. Amended 2026-09-02 (PR3): an already BFP-stored operand is never re-blocked — op-local blocking realizes as the exact fold-segmentation contract (fold whenever either operand's group changes); fresh blocking happens only on the FLOAT32-staging path. | §5.3, §9; GEMM kernels, pools, Relu/Dropout |
| D9 | Every ConvT1d op under `ARITH_BFP` is gather-formulated (output-centric: each output is a dot product over its contributors), restoring the int32 block-partial contract; the SYM scatter core stays SYM-only. Amended 2026-11-05 (§14, experiment): the Conv1d dx (a ConvT1d op) scatters on integer accumulators instead. | §6, §14; `convTranspose1dKernelBfpGather`, `convTranspose1dTapsAt` |
| D12 | Sequencing: the epic starts after group-quant PR3; group-quant PR4 (ASYM groups) runs in parallel with a single coordinated ODTS version bump (v5); the group-quant HAR sweep is the vehicle the BFP sweep extends. | `Serialize.c` (v5) |
| deviations register 5 | Dropout handles mantissas through a float bridge (non-native by decision, D4). | §5.7 "Dropout bridge" |
| deviations register 6 | ReLU is block-transparent: exponents are kept, the mantissa-utilization drop is accepted. | §5.7 "Packed-domain transparency" |

**Epic-PR2 implementation decisions — cited as `Decision N` here, `PR2-Decision N` elsewhere.**

| ID | Decision | Applied in |
|---|---|---|
| Decision 1 | A FLOAT32-stored operand under `ARITH_BFP` stages per-tensor (`{1,0}`) at the weight operand's `(mantissaBits, exponentBits)`, rounded by the op's `arithmetic.roundingMode`. Marked revisable. | §5.4; `ExecuteOp.c` `bfpStage`, layer BFP arms |
| Decision 2 | Bias is staged uniformly like any other operand (FLOAT32 bias → per-tensor BFP scratch; BFP bias → unpacked and borrowed); no funnel special case. | §5.4 |
| Decision 3 | v1 operand-dtype scope: under `ARITH_BFP` only FLOAT32-stored (staged) and BFP-stored operands are legal; SYM/ASYM/INT32/BOOL-stored operands fail fast ("insert a Quantization layer"). | §5.4 |
| Decision 5 | Wire geometry: a per-tensor template gives a per-tensor wire; a grouped template gives `{N/groupSize, groupSize}` from the template's `groupSize`, ignoring its `numGroups`; `N % groupSize != 0` fails fast; exponents start zero-state. | §5.5; `initLayerOutputs`, `CalculateGradsSequential.c`, `InferenceApi.c`, norm-factory BFP params |
| Decision 7 | `ARITH_BFP` kernels are rounding-free: all rounding happens at staging (quantize) and at the OUT_WRITE epilogue (pack). Amended 2026-11-04 (§13): for an op with `bfpRawOut` the OUT_WRITE rounding is the integer re-block (shift + HALF_AWAY magnitude form or integer SR), one draw per element as before. | §5.3, §13 |
| Decision 8 | The `ARITH_BFP` derivation flip lands last: after it, `layerQuantInitUniform(bfp)` derives `ARITH_BFP` in all four math slots (backward slots had to be pinned to FLOAT32 until epic PR3). | §5.1, `ArithmeticType.c` |
| Decision 9 | The PR2 native end-to-end model keeps its final (loss-facing) wire FLOAT32, because losses had no BFP arms before PR4. | `UnitTestMultiLayerTraining.c` |
| Decision 11 | A BFP-stored operand under `ARITH_SYM_INT32` is denied (guided fail-fast in the funnel's SYM prologue): converting it would silently collapse its group structure to one scalar grid. Legal: BFP storage under `ARITH_FLOAT32` (fake-quant) or `ARITH_BFP` (native). | §5.4; `ExecuteOp.c`, `Softmax.c` |

**Epic-PR6 kickoff rulings — cited as `P6-N`.**

| ID | Ruling | Applied in |
|---|---|---|
| P6-1 | Softmax backward recomputes the softmax from the logits it actually receives (the layer input, per the training-loop contract) in every arm; the old code treated its input as the softmax output. | §11 Correction 1; `Softmax.c` |
| P6-2 | Pure-integer alignment onto the max element's block grid, with a shift-rounding knob (`BFP_SHIFT_TRUNC` default, `HALF_AWAY`, `SR`) as a `softmaxConfig_t` field; `layerQuant_t` does not grow. | §5.9 R-S2; `softmaxSetBfpExpShiftRounding` |
| P6-3 | Fixed dyadic work grid `2^-14`, so the i-exp constants are compile-time integers. | §5.9; `BfpSoftmaxExp.c` |
| P6-4 | The max search is an exact float compare of dequantized values (first max wins); no rounding introduced. Amended 2026-11-04: the exact integer `(code, E)` compare `bfpValueGreater`, same tie rule. | §5.9 |
| P6-5 | The cross-element sum and the normalizing divide run in float32 (D7); the only BFP rounding is the OUT_WRITE pack. Amended 2026-11-04 (integer-only forward): the FORWARD normalises in integers (HALF_AWAY, raw `BFP_RAW` at `2^-30`); the backward recompute keeps this row. | §5.9 R-S2 step 5, R-S3 |
| P6-6 | Backward is one funnel OUT_WRITE op anchored on `propLossQ`, recomputing the softmax in-kernel from the logits; `propLoss == NULL` is an early no-op. | §5.9 R-S4 |
| P6-7 | SYM_INT32 softmax stays fake-quant (float kernel); only `ARITH_BFP` gets native semantics. | `Softmax.c` dispatch |
| P6-8 | Silent behavioural flip, documented: a uniform-BFP profile now runs the native forward and backward; pin the math slots to `ARITH_FLOAT32` to opt out. | §5.9 R-S6 |

## 13. Integer raw intermediate (`BFP_RAW`) and the integer re-block rule (integer-only forward)

D7 made every BFP kernel emit a FLOAT32 raw that the OUT_WRITE epilogue
re-packed with float (per element `fdiv` + `roundf` / float SR, per group
`frexpf`) — on a core without an FPU that pack alone costs more than the
integer MACs it follows. This section records the integer replacement for the
FORWARD path (2026-11-04). The backward keeps D7's FLOAT32 raw, except the
Conv1d and Linear dx (§14, `BFP_RAW_BACKWARD_DX`, 2026-11-05) and the Conv1d
and Linear weightGrad / biasGrad into BFP grad storage (§14,
`BFP_RAW_BACKWARD_ACC`, 2026-11-05).

**The raw.** `qtype_t BFP_RAW` with `bfpRawQConfig_t {int16_t *exponents;
numGroups; groupSize}` (`Quantization.h`): int32 mantissas in the tensor data
plus ONE **unbiased** int16 exponent per TARGET group, `value = mant · 2^E`
(unbiased because an accumulation exponent is the sum of two unbiased operand
exponents plus a fold guard, which exceeds int8). It is transient by design:
it exists on the funnel's stack for one op, is never storage, never
serialized, copied or cloned, and every other dtype switch rejects it; the
enum value is appended last so stored tags stay stable. An op opts in per
call with `opSpec_t.bfpRawOut` (`ExecuteOp.h`, `bfpRawContract_t`):
`BFP_RAW_FORWARD` for the integer forward, `BFP_RAW_BACKWARD_DX` for a
backward op writing a dx wire (the Conv1d and Linear dx, §14). Both are `ARITH_BFP` and
`OUT_WRITE` only, raw geometry copied from the target (a BFP target's
`{numGroups, groupSize}`; a FLOAT32 target has no groups, so `{1,0}` — one
exponent over the whole output); any other target dtype, arithmetic type,
epilogue or contract value fails fast. The third value,
`BFP_RAW_BACKWARD_ACC`, is a grad op accumulating into BFP grad STORAGE (the
Conv1d and Linear weightGrad / biasGrad, §14): `ARITH_BFP`, `OUT_ACC_DYNAMIC_RESCALE`
and a BFP target only (raw on the grad's geometry); the epilogue is the
integer engine of §15 (`accumulateBfpRawIntoBfpRescale`), which rounds by
the grad's own storage mode — the ACC rule (#282) — drawing from the op's
`RNG_TAG_ACC` stream 0. A FLOAT32 grad or the fixed grid
(`OUT_ACC_FIXED_SCALE`) with this value fails fast; the layer keeps
`BFP_RAW_NONE` (the FLOAT32 raw) for both. The explicit per-op value is the raw
contract: a backward adapter that shares forward kernel code never switches
raw format by accident — Softmax's leaves it `BFP_RAW_NONE`, Linear's set
their own backward value (§14). A kernel that is given a `BFP_RAW` MUST write every group's
exponent.

**The re-block** (`conversionMatrix[BFP_RAW][BFP]`, `reblockBfpRawTensor`,
`TensorConversion.c`), per target group:

1. `absmax` of the int32 mantissas in `uint32` (`|INT32_MIN| = 2^31` is a
   legal magnitude).
2. Stored exponent by §3's rule in integer form — the smallest `E` with
   `absmax <= qMax · 2^E`: `s = bitlen(absmax) − (m − 1)`, plus one when
   `s >= 1` and `absmax > qMax << s` (`qMax = 2^(m−1) − 1` is one below a
   power of two, so the top of each octave needs one more; for `s <= 0` the
   correction never fires). `stored = rawE + s + bias`, then the D6 clamp to
   `[0, min(2^e − 1, bias + 127)]` — the same cap as `deriveBfpStoredExponent`,
   since a stored exponent is still decoded by `bfpGroupScale` on the float
   side. `absmax == 0` is the zero state (`stored = bias`); `n == 0` writes
   the zero state for every group and no payload (R7 parity).
3. ONE shift per group, `k = stored − bias − rawE` (right; negative = left).
   `k` in `[1, 31]`: HALF_AWAY in magnitude form (`t = mag >> k;
   t += (mag & (2^k − 1)) >= 2^(k−1)`) or integer SR (`t += (draw >> (32 − k))
   < rem`, `rem` = the `k` dropped bits) — the shipped law "lift iff u < frac"
   with `u` = the draw's top `k` bits on a `k`-bit grid; for `k <= 24` this is
   the SAME decision the float path takes from the same draw (`u` on the 24-bit
   grid), which the float-authority tests pin. `k >= 32` flushes to 0.
   `k <= 0` is an exact left shift that saturates at the sign's bound. Codes
   clamp to `[−2^(m−1), qMax]`, which binds only in the D6 cap regime. The
   sign is applied after the magnitude rounding, so SR stays odd per draw
   (`SR(−x) = −SR(x)`) as the law requires.
4. Under SR exactly ONE draw per element in element order, on every path
   (exact, flushed, zero group included) — the same stream position as the
   float re-pack it replaces (`roundByMode` draws for integer-valued quotients
   too), so pinned-draw tests and the Python mirrors keep their positions.
5. Packed through `packChunkGuarded`; the 32 → 8/16 byte/halfword fast path in
   `byteConversion` (`Tensor.c`) serves every `m = 8/16` pack site (SYM and
   BFP; Relu/Dropout/Comparison too). It is bit-identical to the bit walker by
   construction: the walker emits a code's low 8/16 bits LSB-first, which is
   the int32 word's first one or two bytes in memory order.

**The integer fold** (Conv1d forward, `conv1dKernelBfpRaw`, `Conv1dKernel.c`;
the helpers in `BfpKernelSupport.h`). The kernel keeps conv1dKernelBfp's walk
unchanged — loop skeleton, loop order, MAC count, `windowSlice1dAt`, index
math, the per-tap group tracking of BOTH operands — and replaces only the
float fold:

1. ONE `int32` accumulator per output element, in units of `2^E_acc`, with
   `E_acc` shared by every element of a raw TARGET group (the output wire's
   blocking; `{1,0}` for a FLOAT32 target).
2. `E_acc = E_top + g`. `E_top` of a target group is the max over the output
   rows `(b, oc)` it covers (a group may straddle channels and batches) of
   `max_ic(rowmax E_in) + rowmax E_w`, max'ed with the channel's unbiased
   bias exponent — the row maxima are per input row `(b, ic)` of the conv
   group and per weight row `oc`, computed once per call over the groups
   holding a NONZERO code of the row (cursors, no division; a zero bias code
   sets nothing — see "Zero groups never set `E_acc`" below). This is a cheap
   over-approximation of the max segment exponent (a row, not a window).
3. `g` is the static headroom proof (`bfpFoldGuardBits`), the
   **product-bound rule**: with `N = inChannels/groups · kernelSize`, in
   units of `2^E_acc` every term of the accumulator is bounded in absolute
   value, so every intermediate sum is, whatever the signs or the fold
   order:
   - product segments: `|partial_s| · 2^(E_s − E_acc) <= count_s · 2^(ma+mb−2)
     · 2^−g` (every `|code| <= 2^(m−1)`, `E_s <= E_top` for every NONZERO
     segment; a zero segment adds 0 at any alignment); run splitting only
     partitions the `N` products (`Σ count_s <= N`), so all product segments
     together contribute `<= P · 2^−g`, `P = N · 2^(ma+mb−2)`. Each UNALIGNED
     partial separately stays within int32 because a segment holds at most
     `bfpSegmentLimit(ma, mb)` products;
   - the bias segment: `|b| · 2^(E_b − E_acc) <= Bb · 2^−g`, `Bb = 2^(mBias−1)`
     (0 without bias), because `E_top` max'es the bias exponent in; when the
     bias exponent exceeds every product's, `E_top = E_b` and the products are
     shifted further right (they only shrink) while the bias is the term at
     `2^−g`;
   - rounding: a HALF_AWAY right shift adds at most one half, a left shift
     nothing; at most `N` product segments (a split never makes an empty one)
     plus the bias, so `<= (N + 1)/2`; the rule reserves one unit per fold,
     `N + 1`.
   `g` is the smallest integer in `[−30, 31]` with
   `(P + Bb) · 2^−g + (N + 1) <= INT32_MAX`. It is NEGATIVE whenever the
   reduction sits below the int32 limit (m = 8, N = 80: `g = −10`): the
   max-exponent partials are then shifted LEFT exactly (their `2^−g` images
   stay within the bound, so no overflow) and only segments more than `−g`
   binades below `E_top` are rounded; it is `>= 2` for m = 16 reductions
   beyond a few products (m16 × m16: 2 at N = 4, 6 at N = 80; m16 × m8: 3 at
   N = 2400). The design's `g = 1 + ceil(log2 S)` is the same proof with the
   int32 bound on a partial in place of the product bound; the product bound
   is tighter at every width (ruling 2026-11-04: kept, with negative `g`).
   A kernel that rescales or rounds partials BEFORE folding must add that
   error and its changed magnitude bound to this proof.
4. Alignment (`bfpAlignPartial`): a finished segment enters as
   `partial · 2^−k`, `k = E_acc − (E_in + E_w)`. `k < 0` is an exact left
   shift (no overflow, by 3.); `k > 0` is HALF_AWAY in the overflow-safe
   magnitude form `t = mag >> (k−1); r = (t >> 1) + (t & 1)`; `k >= 32`
   drops the partial (it is below one half). Never SR: the re-block owns the
   op's rounding mode, the fold rounds HALF_AWAY whatever the mode. This is
   the forward's exception to "kernels are rounding-free" (§5.3).
5. A segment closes on EITHER operand's group change (as before) AND when it
   holds `bfpSegmentLimit(ma, mb)` products: over-long same-exponent runs
   are SPLIT at the int32 limit, not rejected — the forward has no
   `bfpValidateBlockHeadroom` (m = 16 × 16 runs with one product per
   segment). The backward keeps the fail-fast.
6. The bias is one more segment: `bfpAlignPartial(code, E_acc − E_bias)`
   seeds the accumulator, hoisted per channel and target group.
7. Absolute fold error per output: `<= ½ (S + 1) · 2^E_acc`, `S` = segments
   folded (bias included). Relative to the output it is `2^−24`-class only
   near the static bound; a target group whose values sit far below the
   bound sees more — the price of one static exponent per group.

A FLOAT32 target gets ONE `E_acc` over the whole output (the `{1,0}` raw):
outputs more than `~31 + g` binades below the global row max lose mantissa
bits, and `(float)mant` rounds above `2^24`. The BFP-wire path (the proof
config) is per group and unaffected.

**Zero groups never set `E_acc` (2026-11-05).** Every pre-pass max of §13
and §14 — the row / line maxima of both operands, the operand-wide max of a
strided operand, the bias exponent, and the dx raw's max-`E_acc` re-grid
across rows — runs over the groups holding a NONZERO code of the run, never
over a group whose codes there are all zero: a zero code adds nothing to any
sum, so it bounds nothing, and the proof of item 3 only needs every nonzero
segment below `E_top`. An all-zero group stores the zero state (`stored =
bias`, `E = 0`), typically far ABOVE the live data next to it (a
ReLU-zeroed loss row beside gradients at `E ≈ −25`); counting it lifted
`E_acc` by up to ~20 binades and rounded live partials HALF_AWAY before the
final SR or re-block — a dead zone where the float fold had been exact.
Mechanics: `bfpRowMaxExponent` takes the operand's codes and returns the
marker `BFP_NO_LIVE_EXP` for a run without a nonzero code; a product pairs
to nothing if either factor's run is marked (`bfpLiveExpSum`); a unit (raw
target group, `E_acc` row) with nothing set takes the zero state's top 0,
i.e. `0 + g` — what an all-zero operand produced before. The pre-pass reads
each operand code at most once per call (O(elements), never per MAC; a group
whose exponent cannot raise the max is not read). A zero partial whose group
sits any distance above `E_acc` aligns to 0 (`bfpAlignPartial`: `−k >= 32`
returns 0 instead of the out-of-range shift). Measured, HAR 1 epoch seed 1,
bfp8 `BFP_GRADS=1 BFP_STATE=0`: live partials right-shifted inexactly
before → after the rule — Conv1d dx 203.8 M → 0 (829 M dropped bits, 3.0 M
partials flushed to 0), Conv1d weightGrad 6.02 M → 0.12 M, Conv1d biasGrad
23 → 0, Conv1d forward 100.1 M → 99.5 M (the rest is the live exponent
spread), Linear forward 345 → 344, Linear backward and AvgPool 0 both. The `*_zero_*` differential
cells mismatch under a kernel or a mirror that counts zero groups.

**Relation to the float authority.** On data where the float path is exact
(`|mant| < 2^24`, a power-of-two scale, `k <= 24`) the integer rule
reproduces `deriveBfpStoredExponent`'s stored exponents and the float
re-pack's codes bit for bit, under HALF_AWAY and under SR from the same key.
Above `2^24` the two legitimately diverge: `(float)mant` rounds the magnitude
(§7); the integer form is the exact one. Non-finite values cannot arise in
the integer domain, so R9's fail-fast has no integer twin.

**Loss-facing dequant** (`conversionMatrix[BFP_RAW][FLOAT32]`,
`convertBfpRawTensorToFloat32Tensor`): `ldexpf((float)mant, E)` per element,
not a hoisted `2^E` multiply — `E` is an uncapped int16, and `0 · 2^200` must
stay `0`, not `NaN`. One of the two permitted float boundaries of the
integer-only forward (allow-listed in the census).

**Cost shape.** No division anywhere: group runs use an incrementing cursor
(groups are contiguous), `bitlen` is a five-step binary search (no libgcc
`__clzsi2` on a core without CLZ). The census reports zero float / libm /
division helpers for the re-block; its one `mem` hit is `byteConversion`'s
`memset` on the non-8/16 width branch.

**The pool forwards** (`maxPool1dForwardKernelBfpRaw`, `MaxPool1d.c`;
`avgPool1dForwardKernelBfpRaw`, `AvgPool1d.c`). Both keep the FLOAT32 arm's
walk (windows, order, `windowSlice1dAt`, the per-element `bfpGroupAt`) and
emit `BFP_RAW` on the target's geometry.

1. MaxPool compare (`bfpValueGreater`, `BfpKernelSupport.h`): exact, no
   float. Same exponent: code compare. Gap `d = Ea − Eb` in `[1, 15]`: the
   code under the larger exponent is shifted left by `d` in uint32
   (`|code| · 2^15 <= 2^30`, needs `m <= 16`, checked). `d >= 16`: the
   shifted side's magnitude is `>= 2^16 > |other|` unless it is zero, so its
   sign decides, and a zero there beats only a negative other. First strict
   greater wins, ties across groups included, so the argmax is the FLOAT32
   arm's on the dequantized input, bit for bit (the old `ldexpf` compare was
   exact too).
2. MaxPool winners: `E_acc` of a raw target group = (max exponent over the
   group's NONZERO winners) `− 15`, unbiased; a group without one gets 0
   (all its mantissas are 0). A zero winner never sets `E_acc`: it aligns
   losslessly at any exponent and would only push the others down. Each
   winner enters as `bfpAlignPartial(code, E_acc − E_w)`: a left shift by at
   most 15 (exact, `<= 2^30`), or — for a winner more than 15 binades below
   its group's top — a HALF_AWAY right shift. That is the one lossy step:
   MaxPool is no longer a pure select when a raw group spans more than 15
   binades of winners. The re-block then rounds by the op's mode (SR
   included). The target group's absmax is at least `2^E_top`, so its last
   code bit is at least `2^(E_top − m_out + 1)` and the alignment rounding
   (at `2^(E_top − 15)`) sits `>= 16 − m_out` binades below it: 8 at
   `m_out = 8`, but 0 at `m_out = 16`, where the two roundings can stack
   (bounded by one unit of `2^E_acc`). With a FLOAT32 target the raw
   is per tensor (`{1,0}`), so the 15-binade window spans the whole output.
   Pass 1 (the select) collects the per-group max in the raw exponent array;
   pass 2 re-reads each winner's exponent through its argmax — O(outputs),
   no output-sized table.
3. AvgPool fold: the Conv1d fold with codes in place of products. `E_acc` of
   a raw target group = the max group exponent over the nonzero codes of the
   input ROWS its outputs read (row `r` of the output reads input row `r`;
   cursors, no division; none: 0) `+ g`, `g = bfpSumFoldGuardBits(K, m)` — item 3's proof with
   the second factor bounded by 1: the smallest `g` with
   `K · 2^(m−1) · 2^−g + (K + 1) <= INT32_MAX` (HAR m = 8, K = 32: `g = −18`).
   One int32 partial per same-group segment (`bfpValidateSumHeadroom` bounds
   each), aligned by `bfpAlignPartial` (left exact, right HALF_AWAY, never SR).
4. AvgPool mean: `−log2 K` moves into the raw exponent (exact, no
   division; HAR's K = 32). BFP math takes a power-of-two K only (ruling R1,
   §14: `avgPool1dBfpLog2Divisor` fails fast at the factory and at both
   integer kernels). Error per output `<= S/(2K) · 2^E_acc`, `S` = segments
   folded. `count_include_pad` stays true: the divisor is K. The remaining
   division sites of both roots are `bfpGroupAt`'s per-group-change
   recomputes and the window geometry.

**The Linear forward** (`matmulBfpRawTensors`, `Matmul.c`, behind
`linearForwardBfpRaw`'s transpose dance). `matmulBfpTensors`' walk verbatim —
row, column, reduction loops, `calcElementIndexByIndices` per MAC,
`bfpGroupAt` on both operands — with the Conv1d fold:

1. `E_top` of output `(r, c)` = max group exponent over the nonzero codes
   of the reduction LINE of input row `r` + over weight row `c` (the line
   output column `c` reduces over), max'ed with column `c`'s bias exponent
   when its code is nonzero (zero groups never set `E_acc`, above); `E_acc` of a raw
   target group = max `E_top` over its outputs (walked in output order,
   O(outputs)) `+ g`, `g = bfpFoldGuardBits(inFeatures, ...)` (HAR m8,
   N = 64, bias m8: `g = −10`). A line is one contiguous storage run when
   the reduction axis is the physically innermost dimension — the Linear
   layer's identity-ordered input and its transposed weight view — and one
   forward cursor walks all lines without a division. Any other view (a
   permuted operand handed to the kernel directly) gets the operand-wide
   max over all groups: coarser, still an upper bound.
2. Items 3-7 above unchanged; the bias segment is aligned per OUTPUT (the
   column changes every output, so there is no per-channel hoist).
3. The backward runs the same entry since 2026-11-05 (§14: the dx and the
   weightGrad into BFP grads, on the backward operands' views); only the
   weightGrad into FLOAT32 grads and under the fixed grid keeps
   `matmulBfpTensors`' FLOAT32 raw and `bfpValidateBlockHeadroom`'s
   fail-fast.
4. Cost: the index math stays per MAC (the same-algorithm rule), and
   `calcElementIndexByIndices` divides once per dimension per call, i.e.
   per MAC on the M0+ in BOTH arms (allow-listed in the census); every
   other division in the root is a `bfpGroupAt` recompute.

**The Softmax forward** (`softmaxForwardBfpRaw`, `Softmax.c`): §5.9 R-S2
steps 1-4 (`softmaxExpCodesBfp`, shared with the backward's recompute) and
the integer step 5 into a `BFP_RAW` at exponent `−30` in every raw group.
The one division left in the root besides `bfpGroupAt` is the i-exp's
`z = −qW / QLN2`, per ELEMENT (steps 2-4 unchanged; allow-listed).

**FLOAT32 targets of the two ops.** On the HAR proof path both head wires
(6 elements) are per-tensor BFP (`resolveWireShape` falls back to `{1,0}`
under `BFP_WIRE_BLOCK=16`), so no raw of these ops goes to a FLOAT32 target
there; under the bfp head the loss reads the packed Softmax wire (the float
head keeps `p` FLOAT32, §14 "The model head"). Under `BFP_WIRE_BLOCK=float`
(arm A) the Linear raw is `{1,0}` and the item-7 precision class above
applies (the differential spread cell rounds 65 folds per-tensor against 32
under raw groups of 3, float64 error `1.1e-2` vs `6e-6`, both inside the
bound). The Softmax raw loses nothing there: every element shares the
scale `2^-30` by construction, and `qe`'s own grid is already coarser.

**Status.** The Conv1d forward emits `BFP_RAW` (`conv1dKernelBfpRaw`, the
fold above); `conv1dKernelBfp` (FLOAT32 raw) remains for the ConvT1d dx
adjoint. The MaxPool, AvgPool, Linear and Softmax forwards emit `BFP_RAW`
too; their backwards keep the FLOAT32 raw (MaxPool's argmax meaning is
unchanged, Softmax's backward recomputes with the float32 step 5) except
Linear's, which runs the integer GEMM since 2026-11-05 (§14). Every other op keeps the
FLOAT32 raw. Numbers of native BFP configs changed with these forwards by
design; fake-quant and FLOAT32 paths are untouched.

**No float in the forward after the input layer (2026-11-04).** On BFP wires
the integer forward starts at ONE input `Quantization` layer (FLOAT32 → the
first BFP wire, rounded by the wire template's own mode); after it, every
operand of an integer forward op is BFP-stored. The funnel enforces it: an
op with `bfpRawOut` writing a BFP target fails fast on a FLOAT32-stored
operand instead of staging it (`ExecuteOp.c`, the `ARITH_BFP` prologue).
`bfpRawOut == BFP_RAW_FORWARD` is the forward marker (backward ops leave it
`BFP_RAW_NONE` or set `BFP_RAW_BACKWARD_DX` / `BFP_RAW_BACKWARD_ACC`, which
keep staging), and the
target dtype is the wire marker, so the rule leaves two cases staging as
before: FLOAT32 wires (the HAR arm A, `BFP_WIRE_BLOCK=float`: forward targets
are FLOAT32) and every backward op. Decision 1's per-tensor staging is
therefore reachable in the forward only under FLOAT32 wires; under BFP wires
the first layer's input arrives grouped at the wire's block size instead of
per-tensor at the weight widths. Softmax's forward always writes its BFP
anchor wire, so its R-N1 input staging is unreachable now. The HAR trainer
(`train_c_bfp.c`) inserts the input layer iff the wires are BFP and the math
is native; `BFP_MATH=fq` keeps the 12-layer model (its GEMMs run
`ARITH_FLOAT32`, the rule does not reach them) so it stays the unchanged
fake-quant reference. The static side is the M0+ census
(`tools/bfp-census/`, `har_forward_roots.txt --strict`): zero float, double,
libm and int64 calls in the HAR forward kernels, the re-block and the operand
unpack, and only per-call, per-group-change, per-output and shared-index-math
divisions, each allow-listed for its one root.

## 14. Integer backward: Conv1d dx and weightGrad scatter (experiment), Linear GEMM, pools, loss boundary

Rule: an `ARITH_BFP` kernel runs the same algorithm as its FLOAT32 twin —
loops, loop order, MAC count, index math. The Conv1d dx (`propLoss`) and
weightGrad kernels broke it: FLOAT32 scatters, BFP gathered (§5.6, §6). Since
2026-11-05 they scatter, as an experiment measured on its own; it changes the
summation order and therefore the numbers.

- **Loops:** `convTranspose1dKernelBfpScatter` is `convTranspose1dKernelFloat32`
  line for line; `weightGradKernelBfp` (`Conv1d.c`) is `weightGradKernelFloat`
  line for line, including the memset of the raw.
- **Accumulator:** one `int32` per destination, in the raw buffer itself.
  Consecutive products land in different destinations, so every product is
  its own one-product segment; there is no run to split.
- **Accumulation exponent, fixed before the walk:** `E_acc = E_top + g` per
  destination row group. dx: per (batch, conv group), `E_top` = max over the
  group's loss channels of (loss-row max exponent + weight-row max exponent).
  weightGrad: per output channel, `E_top` = max over the batch of (loss-row
  max + max over the conv group's input rows). `E_top` bounds every nonzero
  product exponent that reaches the row (the row-maxima pairing of §13, each
  maximum over the row's nonzero codes; a (batch, conv group) whose products
  are all zero sets nothing and does not take part in the dx raw's re-grid).
- **Headroom:** `g = bfpFoldGuardBits(N, ma, mb, 0)` (§13), N = the most
  products one destination can receive from the actual geometry. dx:
  `Cout/groups` × the most taps that meet on one position (taps k, k' meet
  when `(k - k') · dilation` is a multiple of stride; with stride 1 that is
  K). weightGrad: batch × the most output positions one tap reaches. The
  proof of §13 then bounds every intermediate sum; nothing fails fast except
  a reduction no `g ≤ 31` can hold. The old segment-headroom refusal
  (`bfpValidateBlockHeadroom`) no longer applies to these two ops.
- **Product entry:** `acc += bfpAlignPartial(a · b, E_acc − Ea − Eb)` —
  exact left shift, or a HALF_AWAY right shift (independent of the op's
  `roundingMode`, as in the forward fold). Rounding error ≤ ½·2^E_acc per
  product.
- **dx output: the raw, integer end to end (2026-11-05).** The dx op sets
  `opSpec_t.bfpRawOut = BFP_RAW_BACKWARD_DX` (§13), so the kernel writes a
  `BFP_RAW` on the dx wire's geometry. The scatter keeps its per-(batch,
  conv group) `E_acc`; a post-pass (O(outputs), cursors, no division)
  replaces the float bridge:
  - each raw target group's exponent `E_t` = the max `E_acc` over the dx
    rows it covers (§13 item 2's rule: a group may straddle channels, conv
    groups and the batch);
  - a sum on a lower row is re-gridded once: `bfpAlignPartial(acc, E_t −
    E_acc)`, a HALF_AWAY right shift of the FINISHED sum (it only shrinks;
    the headroom proof is untouched). Error per element `≤ N/2 · 2^E_acc +
    ½ · 2^E_t` — never worse than aligning every product to `E_t` directly
    (`N/2 · 2^E_t`), and without a per-MAC destination-group lookup.
  At m = 8 the guard is negative (`g ≈ −9…−16`), so every sum carries `−g`
  zero low bits on its own grid and only gaps beyond `−g` binades round.
  HAR (micro-batch 1, conv groups 1) has ONE `E_acc` per call: every shift
  is 0. A FLOAT32 dx target gets the `{1,0}` raw (§13): one `E_t` over the
  whole dx, then the exact `ldexpf` dequant (arm A, `BFP_WIRE_BLOCK=float`;
  with batch or conv groups > 1 the lower rows lose low bits there).
- **dx wire: the integer re-block, by the op's mode.** The OUT_WRITE
  epilogue is `conversionMatrix[BFP_RAW][BFP]` (§13) under the #282 swap, so
  it rounds by `propLossMath.roundingMode` — SR for the HAR proof config
  (training-side seam: dx packs are SR, forward wires HALF_AWAY). Under SR it
  draws ONE `rngNextU32` per element in element order from the op's
  `RNG_TAG_OUT` stream (`rngFold(op key, RNG_TAG_OUT, 0)`, op key =
  `CONV1D_OP_PROP_LOSS` under `RNG_TAG_BACKWARD`) — the stream and position
  the float re-pack it replaces drew from, so pinned-draw tests keep their
  positions. Where the float path was exact (`|acc| < 2^24`, `k <= 24`) the
  codes are the float re-pack's bit for bit; above `2^24` the integer form
  is the exact one (§7) and native BFP numbers change by design.
- **weightGrad output into FLOAT32 grads (bridge), and under the fixed
  grid:** each weightGrad accumulator becomes `float(acc) · 2^E_acc` in
  place (`weightGradKernelBfp`, `bfpLdexpInt32`; the cast rounds to nearest
  even above 2^24, §7), accumulated by the float ACC arms.
- **Grad raw into BFP grads (2026-11-05, `BFP_RAW_BACKWARD_ACC`):** with a
  BFP grad under `OUT_ACC_DYNAMIC_RESCALE` (the HAR `BFP_GRADS=1` path) the
  layer picks `weightGradKernelBfpRaw`, which runs the same scatter
  (`weightGradBfpScatter`, shared with the bridge kernel) and emits a
  `BFP_RAW` on the grad's geometry (`bfpRawEmitFromRows`,
  `BfpKernelSupport.h`); the funnel adds it to the stored grad with §15's
  integer engine. No 8-bit temporary sits between the two reductions (the
  in-call sum and the across-call accumulate). The raw exponent comes from
  the DATA, not from the rows' `E_acc` bounds: per raw group `E_t` = max over
  its nonzero sums of `E_acc + bitlen(|sum|)`, minus 31. The sum that sets
  it has `bitlen <= 31` (the guard keeps every sum below `INT32_MAX`), so it
  moves left by `31 − bitlen <= 30`, exactly; every other left move has a
  lower top and fits; a right move ends below `2^30`, so the half-away lift
  cannot reach `2^31`. Only a sum more than 31 binades below its group's top
  loses bits (HALF_AWAY: at most `½ · 2^E_t`, i.e. `< 2^(m−33)` of the
  grad's code step when the group's largest sum sets the grid — `2^-25` at
  m = 8; a documented deterministic rounding before the SR, next to the
  engine's own `2^-32` SR cutoff and the in-loop product alignment, which
  is coarser). An all-zero group writes exponent 0 (the engine ignores the exponent
  of an all-zero raw); zero sums are never shifted. Why not dx's max-`E_acc`
  rule: `E_acc` is a bound, and a loose N (or, before zero groups were
  excluded from every pre-pass, §13, an all-zero loss group in the zero
  state) inflates a row's bound far above its data; under a per-tensor grad that inflated row would re-grid every other
  row HALF_AWAY to a few bits. Measured on HAR bfp8 `BFP_GRADS=1` (1 epoch,
  18,000 weightGrad calls): with the max-`E_acc` rule 12.6 % of calls kept
  fewer than 9 bits at the tensor top (a rounding at or above the 8-bit grad
  step BEFORE the SR) and 3.6 % of the nonzero sums flushed to 0; the
  differential cell `b4_conv_dw_grad_bfp_zero_rows_m8` shows 28 of 64 codes
  differ. The dx raw keeps its rule (HAR dx has one `E_acc` per call; for
  batch or conv groups > 1 the same change would apply there).
- **biasGrad:** walks like FLOAT32 (channel outer, batch, position). Into
  FLOAT32 grads (and under the fixed grid) it keeps its float fold
  (`biasGradKernelBfp`). Into BFP grads under DYNAMIC_RESCALE,
  `biasGradKernelBfpRaw` keeps the walk and the segments (consecutive codes
  of one loss group, an exact int32 partial, `bfpValidateSumHeadroom`) and
  folds each segment partial into an int32 destination:
  `sum += bfpAlignPartial(partial, E_acc − E_segment)`, `E_acc` per output
  channel = max over the batch of the loss row's max exponent over its
  nonzero codes (none: 0) + `g`, `g =
  bfpSumFoldGuardBits(batch · outputLength, m)` (the §13 proof for a pure
  SUM, `N · 2^(m−1)`): rounding `<= ½ · 2^E_acc` per segment, only for
  segments more than `−g` binades below the top. Then the same raw emit
  (rows of one sum).
- **Grad accumulate: rounding and RNG.** The engine rounds by the grad's
  STORAGE mode (SR for the HAR grad template, `trainingSideRounding()`), one
  `rngNextU32` per element in flat grad order (element order of the stored
  grad, every element on every path) from `rngFold(op key, RNG_TAG_ACC, 0)`
  — the stream and positions the float rescale walk it replaces drew from.
  The weightGrad and biasGrad ops have their own op keys. Fresh grads (zero
  codes, any stored exponent) take the increment's grid; the bound and the
  grid are re-derived on every micro-batch call; old exponents are published
  only after their group's last read (§15).
- **Unchanged:** the Conv1dTransposed forward and its weightGrad keep the
  gather (§6); Linear keeps its GEMM walk (integer since 2026-11-05, below).

**Linear dx / weightGrad / biasGrad (2026-11-05).** Linear's backward has no
scatter to adopt: its BFP kernels already walk like FLOAT32 (one GEMM per op,
output-major, reduction inner, both operands group-tracked). The integer
twin is the forward's §13 fold on the same walk — `matmulBfpRawTensors`,
called by the backward cores exactly where the float fold
(`matmulBfpTensors`) was called:

- **dx** (`linearCalcPropLossBfp`): `loss [B, outF] · W` with the weight AS
  STORED `[outF, inF]`; every `ARITH_BFP` dx sets `BFP_RAW_BACKWARD_DX`
  (§13), so the dx is the raw on the dx wire's geometry, then the integer
  re-block by the op's mode (SR for HAR, one draw per element in element
  order from `rngFold(op key, RNG_TAG_OUT, 0)`, op key `LINEAR_OP_PROP_LOSS`)
  or, onto a FLOAT32 dx, the `{1,0}` raw + exact dequant.
- **weightGrad** (`linearCalcWeightGradsBfpRaw`): `loss^T · x` through the
  zero-copy transpose view; into BFP grads under `OUT_ACC_DYNAMIC_RESCALE`
  (`BFP_RAW_BACKWARD_ACC`), the raw on the grad's geometry. FLOAT32 grads
  and the fixed grid keep `linearCalcWeightGradsBfp` (the float fold,
  unchanged).
- **The accumulation exponent of a backward GEMM.** §13's rule: `E_acc` of a
  raw target group = the max over its outputs of `rowTop + colTop`, plus `g =
  bfpFoldGuardBits(K, ma, mb, 0)` over the reduction length `K` (dx: `outF`;
  weightGrad: the batch). `rowTop` / `colTop` is the max group exponent over
  the nonzero codes of the output's reduction LINE when the reduction runs along
  the operand's physically innermost axis — the dx loss rows — and the
  operand-wide max otherwise: the dx weight columns (strided by `inF`) and
  both weightGrad operands (the batch is the outer axis of the loss and of
  `x`). The operand-wide max is coarser and still sound (it bounds every
  group a line can visit). Over-long same-exponent runs split at the segment
  limit (no fail-fast): the m16 weightGrad, whose float fold refused a batch
  of 2 at `bfpValidateBlockHeadroom`, now runs.
- **No data rule for the GEMM raw.** Conv1d's grad raw takes its exponent
  from the data (above) because the scatter keeps one `E_acc` per ROW and the
  emit re-grids every row of a raw group onto one exponent — an inflated row
  would round the others. The GEMM has no such step: `E_acc` is chosen per
  raw target group BEFORE the MAC loop and the finished sums are never
  re-gridded. A looser raw exponent only adds zero low bits, which neither
  consumer sees: the engine (§15) takes its working grid from `E_r +
  bitlen(A_r)`, the re-block (§13) from the mantissa absmax. The remaining
  deterministic loss is the in-loop alignment: a segment more than `−g`
  binades below its group's bound rounds HALF_AWAY (at most ½·2^E_acc per
  segment). A loose bound moves that line up: the operand-wide max of a
  strided operand counts live groups the line never visits (an all-zero
  group no longer counts anywhere, §13).
- **biasGrad** (`linearCalcBiasGradsBfpRaw`): the Conv1d bias rule on the
  `[B, outF]` loss (one position per row): feature outer, batch inner,
  consecutive codes of one loss group form a segment (exact int32 partial,
  `bfpValidateSumHeadroom`), `sum += bfpAlignPartial(partial, E_acc −
  E_segment)`, `E_acc` per feature = max over the batch of its codes' group
  exponents + `bfpSumFoldGuardBits(B, m)`, then `bfpRawEmitFromRows` (rows
  of one sum, exponent from the data: each feature has its own `E_acc`).
  FLOAT32 grads and the fixed grid keep the float fold.
- **Accumulate:** the engine above (§15) with the op's ACC(0) stream, op
  keys `LINEAR_OP_WEIGHT_GRAD` / `LINEAR_OP_BIAS_GRAD`. Linear's factory
  default bias mode is `OUT_ACC_FIXED_SCALE` (Conv1d's too): a BFP bias grad
  there keeps the float bridge and the float fixed grid; the HAR trainer
  sets `OUT_ACC_DYNAMIC_RESCALE` for both grads of every GEMM layer.

**Pools, ReLU, Flatten, input layer and the loss boundary (2026-11-05).**
After this step the only float of a BFP training step (BFP wires, native
math) is the float head's loss island, and under the bfp head there is none
on the gradient path (below); `tools/bfp-census/har_backward_roots.txt`
gates both.

- **MaxPool1d dx** (`maxPool1dBackwardKernelBfpRaw`): the FLOAT32 arm's loop
  (rows, outputs, the saved argmax, the `-1` sentinel) and scatter-add. One
  `E_acc` per dx row (one (batch, channel)) = the loss row's max exponent
  over its NONZERO codes (none: 0) + `g`, `g = bfpSumFoldGuardBits(C, m)`, `C
  = (effK − 1) / stride + 1` the windows whose span covers one cell. A cell
  can only win windows covering it, so a stale argmax inside the input but
  outside its window's span would break that bound: it fails fast (span
  check, no division). A code enters `acc += bfpAlignPartial(code, E_acc −
  E_group)` (left exact; HALF_AWAY right only for a code more than `−g`
  binades below its row's top, `−g = 23` at m = 8, C = 1). The raw is
  `bfpRawEmitFromRows` (the data rule above) on the dx wire's geometry, then
  the integer re-block by the op's mode (SR under SR, one draw per element in
  element order from `rngFold(op key, RNG_TAG_OUT, 0)`, op key
  `MAXPOOL1D_OP_PROP_LOSS`: the float re-pack's stream and positions).
- **AvgPool1d dx** (`avgPool1dBackwardKernelBfpRaw`): the FLOAT32 loop (one
  contribution per output, added to every valid window member; padded
  positions are not visited and the divisor stays `K`, as in FLOAT32). The
  same per-row grid and guard (C terms). `−log2 K` folds into the row's
  exponent (exact; HAR's K = 32), so the mean adds no rounding; the only
  rounding onto a wire grid is the re-block, SR under SR.
- **AvgPool1d under BFP math: power-of-two K only (ruling R1, 2026-11-05).**
  The mean is always an exponent shift. Any other kernel size fails fast
  (`avgPool1dBfpLog2Divisor`: at `avgPool1dLayerInit` /
  `avgPool1dLayerInitOwning` when either math slot is `ARITH_BFP`, and again
  at both integer kernels for a config built without the factory). Padding
  cannot make an edge divisor differ: the layer is `count_include_pad =
  true` in every arm (FLOAT32 divides by `K`, SYM folds `s / K`, BFP folds
  `−log2 K`), padded positions are only skipped in the sum, never counted
  out of the divisor, so `K` is the divisor of every output. The former
  non-power-of-two path (one `roundedDivHalfAwayInt32` per output) is gone;
  the FLOAT32 and SYM arms still take any `K`.
- **Granularity (both pools): the row-grid flush, precisely.** One `E_acc
  = top_r + g` per dx row `r` (`top_r` the max exponent over the loss row's
  nonzero codes; `g = bfpSumFoldGuardBits(C, m)`: `−23` at m = 8 and `−15`
  at m = 16 for `C = 1`, the HAR pools). A loss code `c` of a group at `E_q`,
  `d = top_r − E_q` binades below the row top, enters with the shift `k = d +
  g` (AvgPool: on the grid `E_acc + log2 K`, the same `k`):
  - `d <= −g`: exact (left shift); this is every code on HAR (identity);
  - `−g < d`: the low `k` bits of `c` are rounded HALF_AWAY on the row grid
    (error `<= ½ · 2^E_acc` per contribution), deterministically and BEFORE
    the re-block's SR -- a dead zone for those bits whatever the op's mode;
  - `d >= m + 1 − g` (32 at m = 8, C = 1): `|c| <= 2^(m−1) < 2^(k−1)`, the
    contribution flushes to 0.
  The lost unit is `2^E_acc = 2^(top_r + g)`. A dx wire group whose code
  step is `2^E_w` with `E_w >= top_r` (any group holding a cell near the row
  top: its absmax is `~2^(m−1) · 2^top_r`) drops those bits in its own
  re-block anyway; the deterministic pre-rounding then biases its SR by at
  most `2^(g − 1)` of a code step (`2^−24` at m = 8). It matters only for a
  wire group whose cells all come from loss groups more than `−g` binades
  below the row top (its step is then finer than the lost unit's `2^−g`
  margin): there the float scatter it replaced was exact. Per-target-group
  exponents would need a destination-group
  lookup per scatter (an open follow-up; no algorithm change here).
  The row table is one `int` per (batch, channel) on the stack. (The former
  AvgPool non-power-of-two quotient and its row-top-relative margin are gone
  with ruling R1.)
- **ReLU dx:** already integer (R-P2, above): the code-domain sign mask, exponents
  carried. **Flatten dx:** payload and exponent copy.
- **Input layer dx** (`quantizationBackward`): straight-through. Between two
  BFP grids it is integer (`quantizationBackwardKernelBfp`, one funnel op,
  `BFP_RAW_BACKWARD_DX`): each loss code is a row of its source group (the
  row exponents read straight from the loss's biased group bytes,
  `bfpRawEmitFromGroupExponents`: no per-group table on the stack), the raw
  is the data rule on the target geometry (exact unless a code sits more
  than 31 binades below its target group's top), the re-block rounds by the
  target template's mode (the conversion cell's rule) and draws (SR) from the
  op's OUT stream (op key `QUANTIZATION_OP_PROP_LOSS`) instead of the
  convert purpose stream. Every other pair (a FLOAT32 input-gradient export,
  SYM) keeps `executeConvert`. HAR never runs it (no dx below conv1).
- **The loss boundary** (`crossEntropySoftmaxBackward`): the combined
  Softmax + CrossEntropy gradient `p − y`, seeded onto the Softmax's dx wire.
  Which arithmetic computes it is the model head's (below); the arm follows
  `p`'s dtype, validated against the declared head at init. A FLOAT32 `p` is
  the float island -- `p − y` in float, then ONE conversion into the grad
  wire (`convertTensor` on the `RNG_TAG_LOSS` stream, by the wire template's
  mode). The float arm raw-casts its operands, so it runs only when output,
  label and grad wire are ALL FLOAT32; any other mix (a FLOAT32 output into a
  BFP grad wire, a quantized label) takes the converting arm. A BFP `p` is
  the integer head. The training loop skips the model's last layer exactly
  once under CrossEntropy (`rematBackwardRange`), and that layer must be the
  Softmax: any other last layer (or an empty model) fails fast, so no
  layer's backward is dropped silently -- checked when a backward runs (an
  all-frozen, eval-only model skips nothing and is not checked).

**The model head: an explicit choice (ruling R2, 2026-11-05).** The head is
the Softmax forward plus CrossEntropy's combined `p − y` (the Softmax
backward is skipped under CE). There is no default: the last Softmax's
config IS the declaration, and the mechanism adds no new field -- the slots
every layer declares arithmetic with are read as the head:
`forwardMath` (the Softmax forward), `propLossMath` (the combined gradient:
it replaces exactly this layer's backward), `outputQ` (`p`'s wire) and
`propLossQ` (the loss-grad wire: the seed GRAD n IS the Softmax's dx, so
the remat wire table gives it the Softmax's `propLossQ`, type-derived; under
MSE the seed still inherits the model output's config).
`requireCrossEntropyHead` (`LayerConfigAccess.c`, called by
`rematBackwardRange`) accepts a head without any BFP unchanged (FLOAT32 and
SYM models), and with any BFP exactly two tuples:

| head | forwardMath | propLossMath | outputQ (`p`) | propLossQ (loss grad) |
|---|---|---|---|---|
| bfp | `ARITH_BFP` | `ARITH_BFP` | BFP | BFP |
| float | `ARITH_FLOAT32` | `ARITH_FLOAT32` | FLOAT32 | any (BFP: one conversion) |

Any other mix fails fast at init -- in particular an integer Softmax with a
float CE gradient (the numerics before the ruling) and a float Softmax whose
`p` is packed onto a BFP wire (two conversions). The HAR trainer exposes it
as the required knob `BFP_HEAD=float|bfp`.

- **float head:** the Softmax's FLOAT32 arm dequantizes the logits; `p`
  stays FLOAT32; CE value and `p − y` in float; ONE float → BFP conversion
  onto the loss-grad wire. The loss island is the census's only allowed
  float on that path.
- **bfp head: the integer `p − y`** (`crossEntropySoftmaxBackwardKernelBfp`,
  one funnel op, `BFP_RAW_BACKWARD_DX`, op key `CROSS_ENTROPY_OP_GRAD = 0` at
  the layer position after the last layer). `p` is borrowed as unpacked
  codes; the label rides in the kernel context (not staged) and must be a
  FLOAT32 one-hot, read by BIT PATTERN (`1.0f` = the class, `±0.0f` not,
  anything else fails fast: soft labels need the float head) -- no float
  compare. Each element is an int32 term on its own grid: a non-label
  element is its code `c` on its `p` group's exponent `E_g`; the label
  subtracts `1.0 = 2^−E` on `E = clamp(E_g, −30, 0)`: exact for every `E_g`
  in `[−30, 0]`, also when 1.0 needs the code `2^(m−1)`, one beyond the
  m-bit maximum (`p` near 1 at `E = −(m−1)`); `|term| <= 2^30 + 2^15`. A `p`
  group below `2^−30` (a confidently wrong prediction in a grouped `p`)
  rounds the label's code half away onto `2^−30` first (`< 2^−31` absolute,
  next to a value near −1); `E_g > 0` with a nonzero code is `p > 1` and
  fails fast. Per raw target group (the loss-grad wire's geometry) the
  exponent is the data rule of `bfpRawEmitFromRows` (`E_t = max(E +
  bitlen|term|) − 31`, two passes recomputing the terms, no table, `p`'s
  groups by cursor, no division); a term more than 31 binades below its
  group's top rounds half away. Then the funnel's integer re-block onto the
  loss-grad wire by the WIRE's mode (its template; SR under SR, one draw per
  element in element order from the op's `RNG_TAG_OUT` stream) -- the
  training-side seam rule of every dx wire. A BFP `p` without a BFP grad
  wire or a FLOAT32 label fails fast.
- **The loss value** (`crossEntropyForward`) stays float on both heads: it
  is monitoring (the logged loss, the divergence check), computed off the
  gradient path.

**Standalone Softmax backward: the integer VJP (2026-11-05).** Not on the
HAR path (CrossEntropy skips the Softmax), implemented for every other loss.
`dx_i = p_i (g_i − d)`, `d = Σ_j p_j g_j`, in the FLOAT32 arm's order, from
the LOGITS: `p` is the integer forward's Q30 raw (`softmaxExpCodesBfp` +
`softmaxNormalizeRaw`, replaying the forward's shift draws), no float
recompute. Precision is cut explicitly before every multiply:
- `p'_j = p_j >> 15` half away (`<= 2^15 + 1`), so `|p'_j g_j| < 2^31` for any
  `m_g <= 16`; the dot runs on `E_d = E_top − 15 + G`, `E_top` the max
  exponent over `g`'s NONZERO codes, `G = bfpFoldGuardBits(n + 1, 17, m_g)`,
  each product entering by `bfpAlignPartial` in index order;
- `D_i = g_i` aligned to `E_d`, minus `d`: the `n + 1` terms of `G`'s proof
  bound every `|D_i|`;
- per raw target group (`BFP_RAW_BACKWARD_DX`) both factors are cut to 15
  bits relative to the group's maximum (`s_p`, `s_D = bitlen(max) − 15`, half
  away) and multiplied, raw exponent `E_d + s_D + s_p − 30`; an all-zero `g`
  (or group) is `dx = 0`.
The error is about `2^-15` of the group's largest `|dx|` (the two cuts, the
dot's alignments, `p`'s Q30 rounding): below the step of any `m <= 15` wire,
about one step at m = 16. The re-block rounds by the op's mode (SR under SR,
the op's OUT stream). On the test fixtures the m = 8 codes equal the former
float VJP's. Single row only (#152).

**The Softmax max grid and zero groups.** The forward's (and the backward's
recompute's) max compare is value-exact, but the winner's exponent is the
alignment grid. A code of an all-zero group never decides it (the zero-group
rule of §13): when such a group's 0 beats every live (negative) value, the max
is 0 on the TOP live group's grid -- otherwise the zero state's `E = 0` would
align live logits many binades lower onto `2^0` (the test fixture: logits
`−0.16 … −0.94` at `E = −6` lost every fraction bit). HAR's logits are one
group of 6, which cannot hold an all-zero group next to live codes: HAR is
unaffected (identity below).

## 15. Integer training primitives: scalar, add-and-round, integer rescale accumulate

The integer backward, gradient accumulation and optimizer (full-BFP training)
build on three primitives. They are added first, unused by any layer; the
rows below are the contract later steps rely on.

**Scalar** (`bfpScalar_t {int32_t q; int16_t e;}`, `BfpKernelSupport.h`):
value `q · 2^e` with a normalized significand `2^14 <= |q| <= 2^15 − 1`, or
`{0, 0}`. A floating exponent, not one global Q15 grid, so a small learning
rate keeps all 15 bits. `bfpScalarFromFloat` encodes once at configuration
time (`frexpf`, significand rounded half away from zero, a round-up to
`2^15` renormalizes; relative error `<= 2^-15`; exact for 0, 1, every power
of two including subnormals; non-finite fails fast). `bfpScalarMulMant`:
a 16-bit code times `q` is `<= 2^30 − 2^15` in magnitude, exact in `int32`.

**Quotient/remainder form** (`bfpQr_t {q, frac, sticky}`): a value on a grid
`2^G` as `q + (frac + t)/2^32`, `q` the floor quotient, `frac` the top 32
bits of the remainder, `sticky = (t != 0)`. `bfpQrOfTerm` aligns a term
(right shifts keep every dropped bit in `frac`/`sticky`, at any distance;
left shifts are exact, the caller bounds them), `bfpQrAdd` adds two (exact
when at most one is sticky; two sticky operands fail fast -- a checked rule,
not a comment), `bfpQrShr` re-grids coarser by any distance,
exactly. Floors use `~(~v >> k)` for negative `v`, never a signed right
shift.

**Rounding** (`bfpQrRound`, `bfpAddAndRound`): on the magnitude, sign after
(SR odd per draw, the `Rounding.h` law); a negative value converts exactly
via `sticky` (`floor((1 − f)·2^32)` is `~frac` with a tail, `2^32 − frac`
without). With `F` = the magnitude remainder's top 32 bits: HALF_AWAY lifts
iff `F >= 2^31`; SR lifts iff `u < F` (one draw). For a remainder of at most
32 bits this is exact and draw-for-draw the decision of `reblockBfpRawTensor`
(`(u >> (32 − k)) < rem` ⇔ `u < rem · 2^(32 − k)`). **Cutoff:** beyond 32
remainder bits the probability is truncated to `F/2^32` — low by less than
`2^-32` — so a magnitude remainder below `2^-32` of a code step never lifts.
That is the price of exactly one draw per element; an exact probability
would need more random words. Unlike the re-block's `k >= 32` flush, nothing
above the cutoff is lost, at any exponent gap. Note: the xorshift32 backend
never draws `u = 0`, so in practice `F = 1` (a remainder in
`[2^-32, 2^-31)`) does not lift either — the effective resolution is
`2^-31`, a property of the RNG backend shared by every integer SR site.

**Integer rescale accumulate** (`accumulateBfpRawIntoBfpRescale`,
`TensorConversion.c`): BFP target `+=` BFP_RAW increment on the target's
geometry, the integer replacement of `accumulateTensorIntoBfpRescale` for
gradient accumulation (wired in through `BFP_RAW_BACKWARD_ACC`, §13: today
the Conv1d weightGrad / biasGrad, §14). Per target group,
`E_o` the old stored exponent minus bias, `E_r` the raw exponent:

1. Working grid (addition bound): `A_o`, `A_r` = actual absmax of the old
   codes and raw mantissas; `E_c = max(E_o + bitlen A_o, E_r + bitlen A_r) − 30`
   (a zero side has no top). Every aligned term is then `< 2^30` on `2^E_c`,
   so the `int32` sum cannot overflow; the bound is re-derived from the data
   on every call (independent of the number of micro-batches). A fresh
   accumulator (all old codes zero) never lets its stored exponent dominate.
   The term that sets `E_c` drops at most 2 bits, so at most one term is
   sticky and the sum is exact.
2. Exponent from the SUM: the per-element sum rounded UP to `2^E_c`, group
   absmax, the integer exponent rule (§13), floored at `E_c`, D6 clamp. As
   `qMax · 2^E` is a grid multiple for `E >= E_c`, this is the exact rule's
   exponent of the true sum, floored at `E_c`: cancellation down to
   `~31 − m` binades below the operands gets its exact fresh exponent, a
   sum cancelled further keeps `E_c` (codes below `2^(m−1)`, still SR-rounded
   unbiased). An all-zero sum (zero sides or exact cancellation) is the zero
   state; `n == 0` too (R7).
3. Emit: the same exact sum re-gridded `E_d − E_c >= 0` steps (any
   distance: the D6 flush regime SRs values far below the smallest step with
   their probability), rounded once by the target config's `roundingMode`,
   clamped to `[−2^(m−1), qMax]`; one draw per element in element order on
   every path (zero, exact, saturated) under SR.
4. Cap regime: `E_c` is capped at the largest storable exponent. The old
   term then stays `<= 2^15` on the grid, so a raw term reaching `2^30`
   there — the only one that could overflow — saturates its element with the
   raw's sign, exactly (`|sum| > 2^30 − 2^15`); neighbours compute exactly.
5. In place: passes 1–2 run one group boundary ahead of the emit (the #421 U6
   window), the old exponent stays in the config until the group's last
   element is read, per-group state lives in two fixed rings of
   `ODT_RESCALE_FRESH_WINDOW` slots (≈1.3 KB). Three reads of the data per
   call (absmax, sum absmax, emit); no float, no division in the element
   loops, no `int64`.

Fixed-grid accumulation stays a separate mode (it needs a whole-macro-batch
bound, not a per-call one) and has no integer twin yet.

### 15.6 Integer macro-batch mean scale

The training loop scales the accumulated gradients by `1/D` exactly once per
macro-batch — after the last micro-batch's accumulation, before
`optimizerStep`, never folded into the learning rate (that would change how
momentum treats its history) — and `optimizerZeroGrad` after the step makes the
next batch start from zero. `D` comes from the loss family as an INTEGER
(`computeMeanDivisor`: CE `batch->size`, MSE `batch->size × F`; `batch->size`
is the batch's actual sample count), and `scaleOptimizerGradientsMean` takes it:
non-BFP grads keep the float fold with `1.0f / (float)D` (bit-identical to
`computeMeanScale`), BFP grads take the integer path.

**`bfpScalarFromCount(D)`**: a power of two is exact (`{2^14, −14 − log2 D}`);
otherwise `q = round(2^(14+L) / D)`, `L = bitlen(D)`, by a 16-step restoring
division (no divide helper; no ties for a non-power-of-two `D`), relative error
`<= 2^-15`. `1 <= D < 2^31`, else fail fast.

**`scaleBfpTensorByScalarInPlace(t, s)`**, `s = q · 2^e` in `(0, 1]`: per group
`A` = absmax code; `A == 0` → zero state (stored = bias). Otherwise `code · q`
(int32-exact, `< 2^30`) is a raw mantissa on `2^(E+e)`; the fresh exponent is
the integer exponent rule on `q · A` (the float walker's rule), clamped to
`[0, cap]`; the emit shift `d = E' − E − e >= 0` (`q · A` has at least 15
bits). Paths:
1. Exponent only: `q == 2^14` and `d == 14` — a power-of-two `D` on a tight
   grid above the floor. Codes untouched, no draws. This is HAR's path
   (`D = 64`).
2. Re-emit: everything else (non-power-of-two `D`, a non-tight grid that
   renormalizes, the exponent floor): `code' = round(code · q / 2^d)` by the
   config's `roundingMode` through `bfpQrRound` (the whole remainder reaches
   the rounding, so a code driven below one step at the floor still lifts
   with its exact probability), one draw per element of the group in element
   order under SR. The cap is reachable only by the negative code bound at
   `s ≈ 1`, and clamps.

The draw count depends on the path, which is safe because each parameter's
scale draws come from its own `(PARAM, p) → (SCALE, 0)` stream that nothing
else reads. Same window structure as §15's accumulate (one group boundary
ahead, two fixed 257-slot rings, ≈2.3 KB frame); a chunk of unchanged groups is
neither unpacked nor packed, so path 1 costs one read of the codes. Power-of-two
results equal the float `scaleBfpTensorInPlace` (exact there too) wherever the
float walker does not overflow `mant · 2^E` near the top of the float range.

### 15.7 Integer SGD step: momentum store and parameter update

With a BFP gradient AND a BFP momentum state, `sgdStepM`'s state op is
`momentumStoreBfpInPlace(state, grad, mu, rm, rs)`: `v' = mu · v + g`, the float
op's formula and order (no dampening, no Nesterov); for a BFP parameter its
param op is then `paramUpdateBfpInPlace(param, state, lr, rm, rs)`: `w' = w −
lr · v'`, reading the STORED `v'` (after the state op's own rounding), the float
op's order. Without momentum (`momentumFactor == 0`, the stateless path) a BFP
parameter on a BFP gradient is `paramUpdateBfpInPlace(param, grad, lr, …)`: `w'
= w − lr · g`. With BFP_GRADS=1 BFP_STATE=1 and BFP parameters the whole step is
integer. Nonzero weight decay, a momentum outside `(0, 1]` (momentum store) and
a non-finite learning rate (parameter update) fail fast, at `sgdMCreateOptim`
and again at the step; there is no float fallback. Every other dtype
combination (a FLOAT32 state or gradient, a non-BFP parameter) keeps the float
ops unchanged. `mu` is encoded once at `sgdInit`, `lr` at `sgdInit` and at
every `sgdSetLr` (`bfpScalarFromFloat`, §15; float allowed at configuration
only) into `sgd_t.momentumQ / momentumE` and `learningRateQ / learningRateE`;
the step's own branches test bools decided at `sgdInit` (no soft-float
compare). The integer path is selected by the STORAGE dtypes; `updateMath`
stays `ARITH_FLOAT32` and names the arithmetic of the funnel ops that remain.

Both ops are one engine, the in-place scaled sum `t' = a · t + b · s`
(`a ∈ [0, 1]` on the in-place target, `b` any normalized signed scalar on the
read-only source): momentum `a = mu, b = 1`; parameter update `a = 1, b = −lr`.
Per TARGET group (the target's own geometry: per-channel weights keep one
exponent per channel), each operand under its OWN exponent: `t_i · a.q` on
`2^(E_t + a.e)` and `s_i · b.q` on `2^(E_s + b.e)` (both int32, `< 2^30`); the
source is per-tensor (its one exponent for every target group) or in the
target's geometry. `E_c = max(E_t + a.e + bitlen(a.q · A_t), E_s + b.e +
bitlen(|b.q| · A_s)) − 30`, capped at the target's top exponent: the term that
sets `E_c` lands with a shift `<= 0`, exact, so at most one operand of
`bfpQrAdd` carries a tail. The exact sum's absmax gives the fresh exponent by
the integer rule, floored at `E_c`, D6-clamped; the exact sum is then
re-gridded and rounded ONCE by `rm` (the optimizer's write-back mode, #282 --
not the storage mode), the discarded tail carried to that rounding, so a small
update is never erased before SR (dead-zone tests: `v = 64Δ`, `g = −Δ/256`
reaches `63Δ` with probability `1/256`; `w = 64Δ`, `lr · u = Δ/256` the same).
Cap regime: only the source can push `E_c` past the cap (`E_t <= capE` and `a
<= 1`, so `a.e <= −14`, keep the target's term `< 2^16` there); a source term
reaching `2^30` on the capped grid (only for a left shift) saturates its
element with that term's sign (for the parameter update: the sign of `−lr ·
u`). Widths and exponent biases of the two operands may differ.

**RNG.** The OUT stream of each op's key, `(PARAM, p) → (OP, k) → (OUT, 0)`
(state op `k = 0`, param op `k = 1`, on both paths), one draw per element in
element order on every path (zero group, exact, saturated) -- the stream and
draw count of the float op's OUT_WRITE re-pack it replaces, so every other
stream is unchanged. Same window structure as the accumulate engine (scan one
group boundary ahead, two fixed 257-slot rings, two fixed 256-element chunk
buffers, ≈3.3 KB frame, no VLA); the per-element emit runs in its own small
out-of-line frame. Differs from the float ops only by the 15-bit `mu` / `lr`
(relative error `<= 2^-15`) and the exact tails; on HAR bfp8 the float staging
of the state op (whole tensors on the stack) and of the param op (the BFP state
as float in `executeOp`'s prologue scratch) is gone.
