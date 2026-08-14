# Block-Floating-Point (BFP) arithmetic — PR1/PR2 conventions + deviations register

Conventions for the `BFP` qtype's dtype-core scope (`src/tensor/Quantization*`,
`src/tensor/TensorConversion.c`'s BFP cells) and, since epic PR2, the native
`ARITH_BFP` compute contract (`src/arithmetic/include/BfpKernelSupport.h`,
`Matmul.c`, `Conv1dKernel.c`, `ConvTranspose1dKernel.c`, `ExecuteOp.c`'s funnel
arms, and the `ARITH_BFP` arms in `src/layer/Linear.c`/`Conv1d.c`/
`Conv1dTransposed.c`). Path-scoped for Claude via
`.claude/rules/arithmetic-bfp.md`. Spec:
`docs/superpowers/specs/2026-07-29-block-floating-point-design.md` (decisions
D1–D12, deviations register §10); PR2 implementation plan:
`docs/superpowers/plans/2026-08-11-bfp-pr2-arith-bfp-gemm-forward.md` (its own
Decisions 1–11, cited below as "Decision N" to keep them distinct from the
spec's D1–D12). §§1–4 track where the shipped PR1 dtype-core deliberately
deviates from the cited literature and from ODT's own `#227` discipline — the
Deutel-note format used by `docs/conventions/arithmetic-sym.md`'s attribution
notes, applied to the BFP anchors (HBFP, MSFP, MX, FAST) instead. §5
documents the PR2 compute contract itself (shipped, not a forward pointer);
§§6–8 extend the deviations register with three deviations the PR2 kernels
introduced.

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

## 5. Compute contract (epic PR2) — shipped

### 5.1 The flip is done

`arithmeticFromQuantization`'s `BFP` arm (`src/arithmetic/ArithmeticType.c`)
now derives native `ARITH_BFP`, not the PR1 float bridge:

```c
case BFP:
    a.type = ARITH_BFP;
    a.roundingMode = ((bfpQConfig_t *)q->qConfig)->roundingMode;
    break;
```

This is the documented breaking change of epic PR2 (the PR1 "D5 float-bridge
staging rule" retired, in the codebase's own naming — see the comment at this
seam; plan Decision 8): a `layerQuant_t` profile built with
`layerQuantInitUniform(bfpConfig)` now derives `ARITH_BFP` in **all four**
math slots (`forwardMath`/`weightGradMath`/`biasGradMath`/`propLossMath`), not
just `forwardMath`. PR2 ships the forward only — a uniform-BFP model that
leaves a backward slot derived dies at the funnel's first backward
`executeOp` call (missing `bfpStage`, see §5.3) until epic PR3 lands native
BFP backward (`testBfpUniformModelDiesOnBackwardUntilPr3`,
`test/unit/userAPI/UnitTestMultiLayerTraining.c`, kept as a permanent
regression pin — delete it when PR3 lands). Pin the backward slots explicitly
to keep training natively on the forward while staying correct on the
backward:

```c
lq0.weightGradMath = (arithmetic_t){.type = ARITH_FLOAT32, .roundingMode = HALF_AWAY};
lq0.biasGradMath   = (arithmetic_t){.type = ARITH_FLOAT32, .roundingMode = HALF_AWAY};
lq0.propLossMath   = (arithmetic_t){.type = ARITH_FLOAT32, .roundingMode = HALF_AWAY};
lq0.propLossQ      = f->floatQ;
```

(the three-slot-plus-wire pin from
`testBfpNativeForwardTrainingLossDecreasesAndGridMoves`,
`test/unit/userAPI/UnitTestMultiLayerTraining.c` — `propLossQ` must move
together with `propLossMath`, since a derived-BFP `propLossQ` alone would not
by itself make the backward op die at allocation. Presented here as
unconditional for clarity: the shared fixture actually guards the
`weightGradMath` line behind a `pinWeightGradMath` flag — `true` for this
capstone, `false` for `testBfpUniformModelDiesOnBackwardUntilPr3`, which
deliberately leaves `weightGradMath` derived so it dies at the funnel; the
`biasGradMath`/`propLossMath`/`propLossQ` lines are unconditional in both.)
Fake-quant over BFP
storage is still available post-flip, but it is no longer free: pin the math
slot(s) to `ARITH_FLOAT32` explicitly and the funnel dequantizes the BFP
operand like any other storage-only dtype, exactly as PR1's derivation did
automatically. `testBfpFakeQuantTrainingLossDecreasesAndGridMoves` (the PR1
capstone) still passes unchanged post-flip: its math slots all derive from a
`FLOAT32` `quantization_t`, which the flip never touches.

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
- **Forward needs no re-blocking.** Storage groups are contiguous along
  exactly the axis every forward reduction walks (the row/column a GEMM
  reduces over, or a conv's input-channel × kernel-tap axis), so the forward
  kernels consume the tensors' STORAGE blocking directly, unmodified.
  Op-local re-blocking (spec D8 — recomputing a block's exponents immediately
  before a dot product whose reduction axis differs from storage order, e.g.
  a backward GEMM's transposed reduction) is exclusively a PR3/backward
  concern; no PR2 kernel re-blocks.

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
  count**, never taken from the template.
- The template's own `numGroups` is deliberately IGNORED: a `layerQuant_t`
  profile is shape-agnostic and routinely shared across layers/wires whose
  sizes differ, so a template `numGroups` can only be a guess — and honoring
  it would size `exponents[]` against a count that does not describe the
  wire while the packer's group index stays unbounded by it (heap overflow).
- `numberOfValues % groupSize != 0` fails fast with a guided message ("pick a
  divisor or a per-tensor `{1,0}` template") rather than silently flooring.
- A freshly allocated wire starts at the **zero-state** exponents (bias,
  scale `1.0`) — the forward's first `OUT_WRITE` derives the real grid.

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
