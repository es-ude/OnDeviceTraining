# Block-Floating-Point (BFP) arithmetic — PR1–PR5 conventions + deviations register

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
Path-scoped for Claude via
`.claude/rules/arithmetic-bfp.md`. Spec:
`docs/superpowers/specs/2026-07-29-block-floating-point-design.md` (decisions
D1–D12, deviations register §10; D8 amended 2026-09-02 at PR3 kickoff — §9
below); PR2 implementation plan:
`docs/superpowers/plans/2026-08-11-bfp-pr2-arith-bfp-gemm-forward.md` (its own
Decisions 1–11, cited below as "Decision N" to keep them distinct from the
spec's D1–D12). §§1–4 track where the shipped PR1 dtype-core deliberately
deviates from the cited literature and from ODT's own `#227` discipline — the
Deutel-note format used by `docs/conventions/arithmetic-sym.md`'s attribution
notes, applied to the BFP anchors (HBFP, MSFP, MX, FAST) instead. §5
documents the compute contract itself (PR2 forward + PR3 backward + PR4's
weight-less layers in §5.7 + PR5's norm layers in §5.8, all shipped, not a
forward pointer); §§6–8 extend
the deviations register with three deviations the PR2 kernels introduced; §9
amends spec decision D8 with the PR3 backward's own deviation (exact fold
segmentation instead of op-local re-blocking); §10 records the PR5 norms'
own register entries (the 3× stats recompute, and the float32 stats bridges
that bound what "native" means for a norm).

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
sign-magnitude's redundant `-0`); the quantizer never emits the extra
negative code anyway — every BFP quantize path clamps mantissas to `±qMax`
with `qMax = 2^(mantissaBits-1) - 1`, so the representable-but-unreached
`-2^(m-1)` code is inert. The reason to deviate is infrastructure reuse, not a
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
deviates from that discipline by design (spec D6): when the derived exponent
would fall outside `[0, 2^exponentBits - 1]`, it **clamps** instead of
aborting — high side clamps the stored exponent to its max, so the block's
mantissas saturate to `±qMax`; low side clamps the stored exponent to `0`, so
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
epic's first-class HAR sweep axes (spec §1) — the sweep exists specifically to
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

`getQLike`'s BFP arm mirrors the `SYM` precedent exactly (spec §2, refined
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

## 5. Compute contract (epic PR2 forward + epic PR3 backward + epic PR4 weight-less layers + epic PR5 norms) — shipped

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
staging rule" retired, in the codebase's own naming — see the comment at this
seam; plan Decision 8): a `layerQuant_t` profile built with
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
`quantization_t`, which the flip never touches.

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
  multiply is exact (spec D7) — the only rounding this step introduces is
  ordinary float32 addition across blocks, plus the two exceptions in §§7–8.
- **Kernels are rounding-free** (plan Decision 7). Every `roundByMode` call in
  the BFP path lives at exactly two seams: staging (quantizing a FLOAT32
  operand into scratch) and the `OUT_WRITE` epilogue (packing the raw FLOAT32
  kernel output back into a BFP target). Nothing inside a kernel's reduction
  loop rounds.
- **Headroom, exact form.** A same-exponent `int32` partial is sound only up
  to `bfpSegmentLimit(ma, mb) = INT32_MAX >> (ma+mb−2)`
  (`src/arithmetic/include/BfpKernelSupport.h`; `shift >= 31` clamps to `1` —
  two `2^30` products already overflow `int32`). `bfpValidateBlockHeadroom`
  fail-fasts at kernel entry over `min(runA, runB, reductionLen)` vs. that
  limit, where `runA`/`runB` are each operand's `groupSize` (or the full
  reduction length for a per-tensor operand) — the bound holds for strided
  walks too, since distinct storage indices inside one group number at most
  `groupSize` regardless of walk order. The spec's SYMBOLIC closed form
  `g ≤ 2^(33−2m)` (equal-width case, `ma = mb = m`) reads one product looser
  than what shipped if evaluated literally (`2^(33−2m)` is `131072` at m=8,
  `512` at m=12) — but the spec's own worked examples already state the
  TIGHT values that match the shipped formula exactly: "m=8 → 131 071;
  m=12 → 511" (spec §4). The shipped `INT32_MAX >> (ma+mb−2)` IS that tight
  form, expressed exactly rather than symbolically: for equal widths it
  reduces algebraically to `2^(33−2m) − 1` — one below the naive power of
  two, because `INT32_MAX = 2^31−1`, not `2^31` — which is why the spec's
  worked examples already land on the shipped values rather than the
  symbolic form's literal evaluation. Pinned by
  `testBfpSegmentLimitTableValues`, `UnitTestMatmul.c`, which also covers
  **m=16 → 1** (shipped-only; the spec's own worked examples stop at m=12).
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
  axis (e.g. a transposed reduction), which is exactly the situation spec
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
  without touching the kernel contract.
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

The four wire allocators (`initLayerOutputs`/`initGradTensor`,
`src/userApi/training_loop/calculate_grads/CalculateGradsSequential.c`;
`initBufferOutput`/`initBufferInput`, `src/userApi/InferenceApi.c`) each carry
a BFP arm keyed off the layer's `layerQuant_t` template (a
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
  one. Linear reuses `matmulBfpTensors` directly — the kernel is
  orientation-agnostic (per-element group lookup honors `orderOfDimensions`),
  so `linearCalcWeightGradsBfp` is a thin transpose-view wrapper
  (`transposeTensor(loss, 0, 1)` around the same GEMM entry the forward and
  `dx` both call). Conv1d/Conv1dTransposed instead walk `(outChannel,
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
  `conv1dTransposedCalcBiasGradsBfp`).
- **`dx` (`propLoss`) reuses the OTHER family's own forward kernel — the D9
  gather adjoint (§6), not a dedicated backward kernel.** Conv1d's `dx` is
  mathematically a Conv1dTransposed op, so its `propLossKernelBfp` funnel
  adapter dispatches straight to the forward's own
  `convTranspose1dKernelBfpGather`; Conv1dTransposed's `dx` is mathematically
  a Conv1d op, so ITS `propLossKernelBfp` dispatches to `conv1dKernelBfp`.
  Linear's `dx` (`linearCalcPropLossBfp`) is
  `matmulBfpTensors(loss, weights, NULL, propLoss)` — the same
  orientation-agnostic GEMM the forward and `weightGrad` both call, with
  `weights` as the second operand instead of the forward input.
- **BFP-stored operands entering any backward op go through the EXACT SAME
  fold contract as the forward (§5.3), never a re-quantize.** Every operand
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
    Two codes rather than one is conservative headroom, not a requirement:
    `SR_HALF_AWAY` dithers by `[-0.5, 0.5)` before rounding, but C's `round()`
    is half-AWAY, so a value clamped to `qMax+1` dithers into
    `[qMax+0.5, qMax+1.5)` and rounds to `qMax+1` at worst — it can never land
    back on `qMax`, and a `±1` band would already preserve the abort. The
    extra code buys margin against any future rounding mode with a wider
    dither.
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
  the TARGET's storage mode, spec D4, while the OP's
  `arithmetic.roundingMode` governs staging and the OUT_WRITE epilogue
  only). A power-of-two `factor` is exact end to end
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
  The saturation BOUNDS are each site's own, unchanged: the negative floor is
  `−2^(m−1)` as shipped, not spec D6's `±qMax` — an open decision in #420,
  and after the unification a one-site change.
- **A non-finite INPUT VALUE fails fast at the three value-domain pack
  entries** (`packFloatBufferAsBfp`, `quantizeFloatBufferToBfpCodes`,
  `packStreamAsBfp`) — `#421`, ruling R9, the value-side twin of the
  accumulate engines' R6 increment guard, with the same rationale: BFP has no
  NaN/inf code, so mapping the value to `0` drops the caller's data and
  saturating it to `±qMax` invents data, while the FLOAT32 wire keeps
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
- Only Softmax (PR6) still lacks an `ARITH_BFP` arm. Softmax lacks one on
  BOTH sides: its FORWARD has no BFP arm
  either — the arithmetic is hardcoded `ARITH_FLOAT32`, so a BFP wire merely
  CROSSES it as a funnel fake-quant bridge — and its backward rejects a BFP
  wire outright before dispatching. (`docs/FEATURES.md`'s carrier-gate list
  states it the same way.) The norms (LayerNorm/GroupNorm) SHIPPED with PR5
  — see §5.8. Pools,
  Relu, Flatten and Dropout SHIPPED with PR4 — see §5.7. Losses got their
  fake-quant `ARITH_BFP` arm in the same PR (`case BFP:` joining
  `case SYM_INT32:` on the dtype-generic helpers), which closes
  `docs/FEATURES.md`'s "no loss function has a BFP arm" gap; NATIVE BFP
  losses (integer CE/MSE) are explicitly OUT of this epic's committed scope
  — spec §9 files them as an optional stretch goal once PR6 lands, not a PR6
  deliverable.
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

**Pool numerics (R-P4).** MaxPool compares DEQUANTIZED values
(`ldexpf(mantissa, E − bias)`, exact) — raw mantissas are not comparable
across groups; argmax/auxOut semantics and the −1 empty-window sentinel are
unchanged. AvgPool sums int32 partials within same-group segments, folds on
every group crossing plus the tail, and divides by K in the FLOAT32 raw: the
SYM `s/K` scale fold has NO BFP analog, because a BFP scale is `2^E` and K is
not a power of two in general. AdaptiveAvgPool does the same with each
window's own count (the SYM arm's rounded integer division has no role — the
BFP raw is float). Forward window sums carry `bfpValidateSumHeadroom`; the
backwards do NOT, because contributions accumulate directly in the FLOAT32 raw
with no int32 partials.

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
FOURTH outside-funnel site (`Softmax.c`'s own wording) — it raw-casts all three
wires to `float*`/`int32*` — but it does not handle BFP at all: it rejects a
BFP wire on any of the three and waits for PR6, so it walks no packed payload.

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
softmax model training natively with no pins, both norms' dx EXECUTING —
is `testBfpUniformNormModelTrainsAndGridsMove`
(`test/unit/userAPI/UnitTestMultiLayerTraining.c`).

**R-N1 — wire-anchored staging.** Norms have no reduction-weight operand, so
GEMM rule 1 ("stage at the weights' widths", §§5.2/5.4) has nothing to anchor
on; the anchor is the layer's OWN produced-wire config — the R-P1 pool rule
at the norm layer. `outputQ` anchors the forward op; `propLossQ` anchors ALL
THREE backward ops, **including at `propLoss == NULL`** (a grads-only call
still stages `forwardInput`/`loss`/gamma at the `propLossQ` widths — the
anchor is a width source, not a write target). The check is EAGER at op
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
  no `int32` partials, no headroom guard. A mantissa-domain variance does
  not EXIST for BFP: centering subtracts an arbitrary float mean, so the
  centered values sit on no shared `2^E` grid and no same-exponent integer
  partial can represent `(x−μ)²`. This is the SYM_INT32 variance's own shape
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
the target's mode — spec D4; a FLOAT32 grad target is a plain float add,
exact), while staging and dx's OUT_WRITE round by
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

**Proof ladder: no §8c bit-identity twin — and what replaces it.** The
spec's proof-ladder rung (c) (spec §8: a BFP config expressible as grouped
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

**Error analysis (spec §10 item 4 — the `0.5·C·s_acc`-style deliverable,
mirroring `docs/conventions/arithmetic-sym.md` §"Grouped backward").**

1. *Mechanism.* Every norm op computes its stats, normalization and affine
   in float32 FROM EXACT DEQUANTS: `mantissa · 2^E` is a power-of-two
   multiply (`ldexpf`), exact in float32, and the mean's/dbeta's `int32`
   segment partials and folds are exact under the sum-headroom guard. The
   ONLY BFP-specific rounding on any norm path is the OUT_WRITE pack of
   the produced wire (forward y, backward dx) — one `roundByMode` per
   element — plus, when a grad tensor is STORED BFP, the ACC landing's own
   requant (§5.6's engines, a storage property, not a kernel one).
   Everything else is ordinary float32 arithmetic noise.
2. *Per-element pack bound.* The pack rounds each element to its group's
   grid: |err| ≤ `0.5·s_out` per element under HALF_AWAY, where
   `s_out = 2^{E_g}` is the scale of the element's group in the DERIVED
   wire grid (`E_g` the unbiased exponent `deriveBfpStoredExponent` snaps
   to from the raw's per-group absmax). `SR_HALF_AWAY` dithers within the
   same band — unbiased in expectation, per-draw error < `1·s_out`.
   Because `s_out ≈ absmax_g/qMax`, the bound is RELATIVE ≈ `2^{−(m−1)}`
   of the group's largest magnitude — the mantissa-width sweep axis,
   directly.
3. *Aggregate float32 noise.* An N-term float32 sum carries worst-case
   relative error ≤ `(N−1)·2^{−24}`, RMS ≈ `√N·2^{−24}` under the
   independent-rounding assumption. The BFP MEAN does strictly better than
   the FLOAT32 path's own mean: its segment folds are EXACT (`ldexpf`
   power-of-two shifts of exact `int32` partials), so its float roundings
   number `C ≈ ⌈N/groupSize⌉` fold-adds (plus one divide) instead of the
   FLOAT32 reduction's N adds. The variance and the backward reductions
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
   Practical consequence: sweeping `mantissaBits` moves norm accuracy;
   sweeping `groupSize`/N barely does.
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
   in higher precision; spec §10 item 4 (native LN/GN beyond the
   literature, each with its own error analysis) is realized HERE. The
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

`convTranspose1dKernelBfpGather` (spec D9; plan
`docs/superpowers/plans/2026-08-11-bfp-pr2-arith-bfp-gemm-forward.md`)
deviates from ODT's own scatter precedent by computing **output-centric**:
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
point beyond the spec's stated claim that "the only added rounding point is
float32 addition across blocks" (spec §4, D7).

This is unreachable at the `(mantissaBits, groupSize)` pairs the epic's sweep
axes exercise in practice: `bfpSegmentLimit` already caps a segment's product
COUNT well below where the partial's magnitude could reach `2^24` for any
headroom-legal segment at typical mantissa widths. It is real, not
hypothetical, at the wide-`m` end of the sweep, combined with long
same-exponent runs near the headroom ceiling — documented here rather than
guarded against, because guarding it would mean a second fail-fast stacked on
top of the headroom guard for a case the sweep may legitimately want to
explore (mantissa-width sensitivity is exactly what the epic measures).

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

**Amended 2026-09-02 (Leo, PR3 kickoff).** The spec's original D8 ("op-local
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
path (the shipped proof-ladder mechanism, spec §8c): a re-block changes which
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
implemented here and is gated on a real MCU measurement motivating it (spec
§9), not filed speculatively.

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

**Float32 stats bridges bound what "native" means.** Spec §10 item 4
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
