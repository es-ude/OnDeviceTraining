# Block-Floating-Point (BFP) arithmetic — PR1 deviations register

Conventions for the `BFP` qtype's dtype-core scope (`src/tensor/Quantization*`,
`src/tensor/TensorConversion.c`'s BFP cells). Path-scoped for Claude via
`.claude/rules/arithmetic-bfp.md`. Spec:
`docs/superpowers/specs/2026-07-29-block-floating-point-design.md` (decisions
D1–D12, deviations register §10). This file tracks where the shipped PR1
dtype-core deliberately deviates from the cited literature and from ODT's own
`#227` discipline — the Deutel-note format used by
`docs/conventions/arithmetic-sym.md`'s attribution notes, applied to the BFP
anchors (HBFP, MSFP, MX, FAST) instead.

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

## 5. Forward pointer — compute contract arrives with epic PR2

This PR (epic PR1) ships the **dtype only**: `bfpQConfig_t`, the complete 7×7
conversion matrix, the owner chain, ODTS v5, and the `ARITH_FLOAT32`
float-bridge derivation (`arithmeticFromQuantization`'s `BFP` arm — see
`docs/conventions/tensor.md`). No `ARITH_BFP` compute kernel exists yet.

Epic PR2 adds the native compute contract this file will then document in
detail:

- **Headroom** — `int32` block-partial accumulation of `mantissaBits`-wide
  operand products is sound only for a block size `g ≤ 2^(33-2m)` (the int12
  table's generalization, `docs/conventions/arithmetic-sym.md`'s "Operand
  bit-width" section), fail-fast-checked at op entry over the
  `(mantissaBits, blockSize)` pair.
- **Op-local re-blocking** — compute blocking is a property of the *op*
  (exponents recomputed immediately before every dot product, the HBFP rule),
  independent of the tensor's *storage* blocking; a backward GEMM re-blocks
  along its own reduction axis rather than reusing the forward's group grid.
- **FLOAT32 raw intermediate** — cross-block partials combine via an exact
  `(float)partial · 2^(Ea+Eb)` power-of-two multiply (lossless) and accumulate
  in `float32` across blocks (no `int64` anywhere, per the framework's hard
  rule).

Until PR2 lands, treat every BFP-stored model as fake-quant: correct end to
end, no accuracy claim about native integer BFP compute.
