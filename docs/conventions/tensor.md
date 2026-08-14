# Tensor — quantization dtype semantics

Conventions for `src/tensor/**` — dtypes, quantization configs, and the
conversion matrix. Path-scoped for Claude via `.claude/rules/tensor.md`.

## SYM_INT32 is a compute format, not storage (#261)

`SYM_INT32` (int32 mantissa + one per-tensor float scale) is the framework's
**integer-compute** representation — the only integer-math path the kernels use.
It is **not** a storage format: it costs the same 4 bytes/element as `FLOAT32`
but is a single-scale fixed-point approximation, so as storage it is dominated by
both `FLOAT32` (same size, better fidelity — a per-value exponent keeps the small
magnitudes a single scale loses) and `SYM`/`ASYM` (which sub-byte-pack). The
integer math is a **transient**; nothing durable should be persisted `SYM_INT32`
to "save memory" — it saves nothing and adds error.

This bites hardest for **gradients**. Persistent parameter grads should be stored
`FLOAT32` (fidelity, same size) or `SYM`/`ASYM` (real compression); the integer
step stays transient `SYM_INT32`. The only legitimate `SYM_INT32` grads are the
transient dx/agrad operand-wires during backprop (int12, freed after the pass).

As of PR1c, the factory default for parameter grads (Linear/LayerNorm/Conv1d/
Conv1dTransposed) IS `FLOAT32` — the NULL-knob fallback that used to derive from
`propLossQ` (silently landing on `SYM_INT32` for a uniform-SYM profile) is now a
hard-pinned `FLOAT32`, closing the gap described above by default. `SYM_INT32`
parameter grads remain available and legitimate only via the explicit
`weightGradStorage`/`biasGradStorage` knob on `layerQuant_t` (#261).

## Packing / byte-count invariant (#172)

- Payload sizing: byte counts for tensor data always ceiling-divide;
  `calcNumberOfBytesForData(q, N)` is the single authority (allocation, copy,
  zeroing, serialization). `calcBytesPerElement` is an unpacked per-element
  stride — multiplying it by N over-counts packed sub-byte payloads (#172).

## Group-granular quantization (group-quant epic, #300 axis)

`symQConfig_t` is **always-array**: `scales[numGroups]`, `numGroups`, `groupSize`.
Exactly two shapes are valid — `{numGroups: 1, groupSize: 0}` (per-tensor, the
sentinel: `groupSize == 0` means "spans everything", N-agnostic) or
`{numGroups: >1, groupSize: >0}` with `numGroups * groupSize == N` (real
groups). `{1, N>0}` is never a valid alternate per-tensor spelling. The
divisibility identity is validated everywhere a config attaches to a concrete
element count — `initTensor` (`validateSymQConfigShape`), `requantizeTensorInPlace`,
and the ODTS/ODTR deserialize path (which additionally REALLOCATES a
skeleton's `scales[]` when the file's `numGroups` differs from its own,
rather than failing fast — see `docs/FEATURES.md`'s Serialization section).

**Storage-order binding.** A group is `groupSize` consecutive elements in
STORAGE order (flat index, not the logical/viewed shape): group id =
`flatIdx / groupSize`. Scales bind to storage order and are untouched by
`orderOfDimensions` view permutations — zero-copy transpose is a view, and
kernels compute flat storage offsets regardless of the permutation.

**Carriers** (spec §3 — YAGNI cuts, reversible later):

| Tensor class | Granularity |
|---|---|
| GEMM-family weights (Linear/Conv1d/ConvT1d) | groups allowed (any valid groupSize, SYM **and ASYM** since PR4); trainable end-to-end (forward + dx + optimizer updates; ASYM runs through the same symmetric kernels after the exact per-group zp shift, D5; error bounds in `docs/conventions/arithmetic-sym.md`) |
| Bias | per-tensor only |
| LayerNorm/GroupNorm gamma/beta | per-tensor only (the factories reject SYM/ASYM gamma/beta wholesale — the pre-existing dtype gate subsumes the group question) |
| Gradients (packed storage) | per-tensor only; grouped grads are a future #300 axis |
| Wires (`outputQ`/`propLossQ`), momentum | per-tensor only (`symInt32QConfig_t` stays scalar) |

For a row-major GEMM weight `[oc, ...]`, `groupSize == N/oc` IS the
per-output-channel special case — no separate axis field. Full design:
`docs/superpowers/specs/2026-07-28-group-quantization-design.md`.

## SYM ↔ * conversion bridge (#227)

`SYM` is the sub-byte bit-packed **storage** dtype; `SYM_INT32` is the int32-slot
**compute** dtype. The MCU lifecycle is store-packed (`SYM`) → unpack to int32
(`SYM_INT32`) → compute → repack. `conversionMatrix`
(`src/tensor/TensorConversion.c`) fills these cells: PR-B implements the **unpack
row** (`SYM → {SYM_INT32, FLOAT32, INT32, ASYM}`); the pack column (`* → SYM`) is
PR-C.

**Sign-extend on unpack.** `byteConversion` is a pure bit-copy that ZERO-FILLS on
widen, so a packed signed mantissa (e.g. `−3` at qBits=6 = `0b111101`) would read
back as `61`. Every `SYM →` cell routes through the shared
`unpackSignExtend(src, srcBits, srcStartBit, dst, n)` helper (public, declared in
`TensorConversion.h`), which widens then sign-extends the two's-complement payload
from `srcBits` (`(v ^ signBit) − signBit`); `srcStartBit` lets DeltaSym-style
decoders start mid-byte, byte-aligned callers pass 0. ASYM codes are
non-negative, so the ASYM **pack** path does not sign-extend. The same contract
applies to `byteConversionAppend` (the bit-offset entry point for mixed-width
streams, e.g. delta compression — bit-granular on BOTH sides: `dstStartBit` for
packing, `srcStartBit` for decoding): zero-fill on widen, low-bit truncation on
pack — signed read-back needs the `unpackSignExtend` idiom there too.

**`int_repr` vs `dequantize` (deliberate, documented asymmetry).** A conversion
whose destination is `INT32` emits the integer **codes** and drops the scale
(`int_repr`); a conversion whose destination is `FLOAT32` emits the **values** with
the scale applied (`dequantize`). This mirrors PyTorch `int_repr()` vs
`dequantize()` and is consistent across both source dtypes: `SYM → INT32` and
`SYM_INT32 → INT32` are both `int_repr`; `SYM → FLOAT32` and `SYM_INT32 → FLOAT32`
are both `dequantize`. No value-rounding `→INT32` variant exists (YAGNI;
near-useless for `scale ≪ 1`).

**Rescale on the symmetric↔asymmetric transition.** `SYM → ASYM` always rescales
(dequantize → derive a fresh asym `scale`+`zeroPoint` from min/max → requantize →
pack): a symmetric code grid cannot hold an off-center `+zeroPoint` band at the
carried scale, independent of width.

**Asymmetric quantization convention (group-quant PR4, supersedes #243's grid).**
Every `* → ASYM` cell builds a float buffer (from its own preamble) and routes
through one shared helper, `quantizeFloatToAsym` (`src/tensor/TensorConversion.c`)
— the single source of truth. **Nudged code-domain affine** (TFLite convention):
the band is first extended to include 0 (`mn = min(mn, 0)`, `mx = max(mx, 0)` —
guarantees 0 is exactly representable and bounds the zero-point into the code
domain by construction); `scale = (mx − mn) / (2^qBits − 1)`;
`zp = clamp(round(−mn/scale), 0, 2^qBits − 1)` stored as **uint16** per group;
`code = clamp(round(v/scale) + zp, 0, 2^qBits − 1)`. Dequant is
`(code − zp)·scale` — the same sign convention as PyTorch/TFLite (the old #243
value-domain grid used an additive signed `zeroPoint = round(min/scale)`; it is
gone, see the width contract below). An all-zero buffer uses `scale = 1` to avoid
divide-by-zero. The denominator is `2^qBits − 1`, **not** `2^qBits`. New
asym-producing converters MUST call this helper and never re-derive the grid
inline (#243's drift lesson). Grouped configs derive the grid **per group**
(`deriveAsymGridForGroup`). The float→SYM pack sibling is `packFloatBufferAsSym`.

**ASYM width/zeroPoint contract (D6, supersedes #246).** `zeroPoints[]` are
**uint16 code-domain** values in `[0, 2^qBits − 1]`; the nudge makes that range
an invariant, not a hope (the old int32 value-domain zp provably exceeded uint16
even at `qBits = 16` — the −72817 wide-band pin was the proof, and the reason
D6 mandates the code domain). ASYM `qBits` is therefore capped at **[1, 16]**
(uint16 code ceiling; was [1, 30] under the value-domain grid), enforced in
`initAsymQConfigGrouped`, re-checked in `deriveAsymGridForGroup`, and validated
against untrusted wire input in the ODTS v5 deserializer. On the wire, zeroPoints
ride as `u16` LE per group (ODTS v5; the old v4 `i32` slot died with the v5 bump).

**Grad-accumulate primitives (PR3, #261).** `accumulateFloatIntoSymTensorFixedGrid` /
`accumulateFloatIntoSymTensorRescale` / `accumulateFloatIntoAsymTensorRescale`
(`src/tensor/TensorConversion.c`) are the packed-grad accumulate primitives that back
the executeOp epilogue's `SYM`/`ASYM` accumulate arms: FixedGrid carries the target's
scale (fit-preserving; first store after a zero-fill derives the grid from the
increment) and **aborts** on grid overflow (#227 discipline — never clamps); Rescale
re-derives a fresh grid every store (absmax for SYM, affine min/max for ASYM). Both are
direct-call only, not `conversionMatrix` cells (there is no dtype-pair to key a matrix
cell on — the second operand is a raw float increment, not a tensor).

## BFP — block-floating-point storage (BFP epic PR1+PR2, spec `docs/superpowers/specs/2026-07-29-block-floating-point-design.md`)

`BFP` is qtype #7 (`qtype_t = {INT32, FLOAT32, SYM_INT32, SYM, ASYM, BOOL,
BFP}`, appended last — mid-enum insertion would corrupt old checkpoints): a
packed **two's-complement** `mantissaBits`-wide mantissa per element, plus one
**per-group** biased `exponentBits`-wide exponent (`bfpQConfig_t.exponents`, a
`uint8_t` SoA — 1 byte per GROUP, never interleaved into the payload, so
`packedByteOffset` chunk-streaming arithmetic stays untouched). `value =
mantissa · 2^(storedExponent − bias)`, `bias = 2^(exponentBits-1) - 1`. Group
geometry mirrors `symQConfig_t` exactly (storage-order-contiguous runs, the
`{1,0}` per-tensor / `{>1,>0}` grouped shapes, the `numGroups·groupSize == N`
identity validated at attach — see "Group-granular quantization" above); the
practical difference from grouped `SYM` is the shared-scale cost: **1
byte/group** (a biased exponent) vs. grouped SYM's **4 bytes/group** (a
`float` absmax scale) — the accuracy-per-byte comparison the BFP sweep epic
exists to measure (deviations from literature exponent conventions, the
saturation-not-abort exponent rule, and clone semantics are in
`docs/conventions/arithmetic-bfp.md`).

**`BFP` IS a storage dtype — "compute format, not storage" (above) does NOT
apply to it.** Unlike `SYM_INT32`, `BFP` costs `mantissaBits/8` bytes/element
plus the per-group exponent overhead and is meant to be persisted. Compute is
a separate, independent axis, and unlike `SYM`/`ASYM` it is not float-bridge
only: since epic PR2, `arithmeticFromQuantization` derives native `ARITH_BFP`
for a `BFP`-typed quantization — the documented breaking change over PR1's
`ARITH_FLOAT32` float bridge. `ARITH_BFP` runs the GEMM-family (Linear/
Conv1d/Conv1dTransposed) **forward** natively: both operands stay blocked,
`int32` mantissa products accumulate per same-exponent segment, and each
segment folds into a `float32` accumulator via an exact `ldexpf` power-of-two
shift at every group-boundary change (kernel contract, headroom guard and
deviations: `docs/conventions/arithmetic-bfp.md` §5). BFP **backward** is
still fake-quant-only until epic PR3: pin the backward math slots
(`weightGradMath`/`biasGradMath`/`propLossMath`) to `ARITH_FLOAT32` explicitly
— `layerQuantInitUniform` over one BFP template derives `ARITH_BFP` in all
four slots, and a model that leaves a backward slot derived dies at the
funnel on its first backward op.

**`int_repr`/`dequantize` convention holds.** `BFP → INT32` emits the packed
mantissa **codes** with the exponent dropped (`int_repr`, matching every other
`* → INT32` cell); `BFP → FLOAT32` **dequantizes** (`mantissa · 2^E` per
element, matching every other `* → FLOAT32` cell).

**Conversion-matrix coverage.** The 7×7 `conversionMatrix` is complete for
BFP: all 10 cross cells (`{FLOAT32, INT32, SYM_INT32, SYM, ASYM} × BFP`, both
directions) plus the `[BFP][BFP]` diagonal — the `SYM_INT32` "requant" analog:
a per-block width/geometry-restore that `writeOutConversion` routes
producer-restored BFP wires through explicitly, never a same-type memmove.
`BFP → SYM` has one directional gap: a **grouped** SYM target has no single
scalar compute image to land a BFP source on, so it fails fast with
dequant-first guidance (dequantize `BFP → FLOAT32`, then `FLOAT32 →
SYM`(grouped)); `BFP → SYM`(per-tensor) works directly (fresh absmax pack).
Conversions with a grouped-**ASYM** target stay denied the same way,
regardless of source dtype — including BFP sources: `convertBfpTensorToAsymTensor`
dies inside the shared per-tensor choke point (`requirePerTensorAsym`), the
same gate every other `* → ASYM` cell routes its grid derivation through;
route through `FLOAT32` instead. `BOOL` stays fail-fast in every direction,
as elsewhere. Same-dtype BFP pairs
with differing geometry/width are the one `QuantizationLayer` same-dtype
exception besides `SYM_INT32`: a same-type convert requires **identical**
geometry and widths (`numGroups`, `groupSize`, `mantissaBits`,
`exponentBits`) — a mismatch is a real re-block/width-change, routed through
the `[BFP][BFP]` diagonal via a Quantization layer, never a verbatim copy.
