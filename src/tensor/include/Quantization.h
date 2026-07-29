#ifndef ENV5_RUNTIME_QUANTIZATION_H
#define ENV5_RUNTIME_QUANTIZATION_H

#include <stddef.h>

#include "Rounding.h"

typedef enum qtype { INT32, FLOAT32, SYM_INT32, SYM, ASYM, BOOL } qtype_t;

typedef struct symInt32QConfig {
    float scale;
    roundingMode_t roundingMode;
    uint8_t qMaxBits;
} symInt32QConfig_t;

/* SYM_INT32 operand bit-width contract (#227). Operands feeding product
 * accumulators are int12 so int12*int12 products stay within an int32
 * accumulator (no int64). Sound for reductions N <= 511 (512*2^22 > INT32_MAX);
 * narrow the knob for wider layers. Override with -DODT_SYM_OPERAND_QMAXBITS=N. */
#ifndef ODT_SYM_OPERAND_QMAXBITS
#define ODT_SYM_OPERAND_QMAXBITS 12
#endif
/* SYM grad ACCUMULATION-TARGET width contract — NOT a backward-compute knob.
 * Backward kernels always run int32-mantissa math at operand width (see
 * ODT_SYM_OPERAND_QMAXBITS above); this bound instead caps the qMaxBits/qBits
 * a SYM_INT32/SYM/ASYM grad TARGET may declare, enforced where increments are
 * accumulated into it (accumulateOut in ExecuteOp.c) and baked into
 * gradInitSymInt32. Rationale is value-sum soundness: int16-range mantissas
 * accumulate in an int32 buffer with headroom; wider widths break that
 * argument. It is NOT a footprint knob either — SYM_INT32 grads stay 4
 * B/element at any width (#261); sub-float memory saving comes only from
 * packed SYM/ASYM grad storage (<= this bound). The pool/reduce value-sum
 * guards reuse the same numeric value under a different contract (Reduce.c).
 * Override with -DODT_SYM_GRAD_QMAXBITS=N. */
#ifndef ODT_SYM_GRAD_QMAXBITS
#define ODT_SYM_GRAD_QMAXBITS 16
#endif

/* Group-quant PR1 (spec docs/superpowers/specs/2026-07-28-group-quantization-design.md
 * §2/D3): always-array representation, behavior-identical for PR1 (numGroups
 * is always 1; no grouping functionality yet -- PR2 introduces real groups).
 * groupSize == 0 is the "whole tensor" sentinel: standalone-built configs
 * (initSymQConfig, without a tensor in hand) cannot know N, so per-tensor
 * quantization keeps groupSize == 0 ("spans everything"), exactly as
 * N-agnostic as the old scalar scale. Ownership: scales is heap-allocated
 * (reserveMemory) by whichever call filled the struct -- initSymQConfig /
 * initSymQConfigGrouped always allocate a fresh `numGroups`-element array;
 * pair with freeReservedMemory (bare qConfig) or freeQuantization (config
 * wrapped in a quantization_t, e.g. via quantizationInitSym / getQLike).
 * Stack test fixtures that only need a fixed per-tensor scale should build
 * the struct directly with a local backing array instead of calling
 * initSymQConfig (see docs/conventions/testing.md) -- such fixtures are
 * never passed to freeQuantization or freeReservedMemory.
 *
 * Task-5 review fix (group-quant PR2): that "never passed to
 * freeQuantization or freeReservedMemory" rule is not just about explicit
 * free calls — it also rules out ANY use as a DESERIALIZE destination
 * (deserializeTensor / deserializeParameter / ppcaReplaySetDeserialize
 * skeletons, Deserialize.c / PpcaReplaySerialize.c). A file record whose
 * numGroups differs from the skeleton's own makes deserializeQConfig's SYM
 * arm freeReservedMemory() the skeleton's CURRENT scales pointer before
 * reserveMemory()ing a new one — implicitly, with no call-site free() to
 * spot. A stack-fixture symQConfig_t handed a mismatched grouped record
 * therefore has free() called on its stack-backed array: the same
 * undefined-behavior hazard the "never freed" rule already exists to avoid,
 * just reached through a deserialize call instead of an explicit free.
 * Deserialize destinations must be built via initSymQConfig /
 * initSymQConfigGrouped (or attached to a real tensor via initTensor).
 *
 * Group-quant PR2 (Task 1): the shape invariant is now a real constraint,
 * not just a PR1 sentinel pin. Exactly two shapes are valid:
 *   - per-tensor: numGroups == 1 && groupSize == 0 (unchanged PR1 sentinel).
 *   - grouped:    numGroups > 1 && groupSize > 0, with
 *                 numGroups * groupSize == numberOfElements at attach time.
 * {1, N} (N > 0) is NOT a valid alternate per-tensor spelling -- the
 * canonical per-tensor form is always {1, 0}. initSymQConfigGrouped enforces
 * the shape at construction (no tensor in hand yet, so it cannot check
 * against N); validateSymQConfigShape enforces the numGroups*groupSize == N
 * identity where a config attaches to a tensor (initTensor, for SYM types).
 *
 * getQLike's SYM arm (TensorApi.c) branches on numGroups: numGroups == 1
 * keeps the existing "Precedent A" fresh-reset clone (scale -> 1.f, pinned by
 * testGetQLikeSymPreservesWidthAndRoundingResetsScale) since a per-tensor
 * clone has no group SHAPE to lose; numGroups > 1 instead deep-copies
 * numGroups/groupSize AND the scales VALUES (matching deepCopyQuantization's
 * semantics, LayerQuant.c:71-82) since a grouped clone's group grid is an
 * attach-time fact the clone must retain, not an ungridded zero-state.
 *
 * Carrier gate: groups are legal ONLY on GEMM-family weight tensors. Grad
 * tensors stay per-tensor unconditionally -- gradInit fail-fasts if handed a
 * SYM template with numGroups > 1 (grouped grads are a future #300 axis). */
typedef struct symQConfig {
    float *scales;    /* [numGroups], owned by the qconfig (see ownership note above) */
    size_t numGroups; /* 1 = per-tensor sentinel; >1 = real groups (PR2+) */
    size_t groupSize; /* 0 = "whole tensor" sentinel (per-tensor, N-agnostic);
                        >0 only for real groups (PR2+), numGroups*groupSize == N */
    roundingMode_t roundingMode;
    uint8_t qBits;
} symQConfig_t;

/* Group-quant PR4 (Task 1, spec D6): always-array ASYM with a NUDGED
 * CODE-DOMAIN affine parametrization (TFLite-standard). Ownership and shape
 * grammar are exactly symQConfig_t's (see the block comment above): both
 * arrays are heap blocks owned by the config (reserveMemory, one block per
 * array), {1,0} is the per-tensor sentinel, {>1,>0} with
 * numGroups*groupSize == N the grouped form, {1,N>0} invalid; stack test
 * fixtures build the struct directly with local backing arrays and are never
 * freed / never used as deserialize destinations.
 *
 * zeroPoints are CODE-domain: dequant = (code - zp) * scale with
 * zp in [0, 2^qBits - 1]. The grid derivation nudges the band to include
 * zero (mn = min(mn, 0), mx = max(mx, 0)), which guarantees (a) 0.0 is
 * exactly representable (code == zp decodes to exactly 0.0f) and (b)
 * zpReal = -mn/scale lands in [0, 2^qBits - 1] BY CONSTRUCTION -- that
 * boundedness is what lets zp be uint16, and is also why qBits is now
 * capped at 16 (a 17-bit code domain would not fit uint16). This supersedes
 * the old value-domain int32 zeroPoint (dequant = (code + zp)*scale,
 * zp = round(min/scale)), whose #246 rationale -- wide all-negative bands
 * pushing zp past int16/int32 -- is void under the nudge. */
typedef struct asymQConfig {
    float *scales;        /* [numGroups], owned (symQConfig ownership rules) */
    uint16_t *zeroPoints; /* [numGroups], owned; CODE-domain zp in
                             [0, 2^qBits - 1] (see block comment above) */
    size_t numGroups;     /* 1 = per-tensor sentinel; >1 = real groups */
    size_t groupSize;     /* 0 = "whole tensor" sentinel (per-tensor);
                            >0 only for real groups, numGroups*groupSize == N */
    uint8_t qBits;        /* ASYM range [1, 16] (was [1, 30] pre-D6) */
    roundingMode_t roundingMode;
} asymQConfig_t;

typedef struct quantization {
    qtype_t type;
    void *qConfig;
} quantization_t;

// Important: This sets qMaxBits to ODT_SYM_OPERAND_QMAXBITS (12)
void initSymInt32QConfig(roundingMode_t roundingMode, symInt32QConfig_t *symInt32QConfig);
void initSymInt32QConfigWithQMaxBits(roundingMode_t roundingMode,
                                     symInt32QConfig_t *symInt32QConfig, uint8_t qMaxBits);
void initSymQConfig(uint8_t qBits, roundingMode_t roundingMode, symQConfig_t *symQConfig);
/*! Group-quant PR2: general-shape SYM config init. initSymQConfig delegates
 * here with (numGroups=1, groupSize=0). Allocates scales[numGroups] (each
 * 1.f); fail-fasts on numGroups == 0, on (numGroups == 1) != (groupSize == 0),
 * and on numGroups > 1 && groupSize == 0 -- i.e. only {1,0} (per-tensor) or
 * {>1,>0} (grouped) are constructible; {1,N>0} is rejected here even before
 * any tensor exists to validate group*groupSize against. */
void initSymQConfigGrouped(uint8_t qBits, roundingMode_t roundingMode, size_t numGroups,
                           size_t groupSize, symQConfig_t *qC);
/*! Group-quant PR2: attach-time shape check for a SYM config against a
 * concrete element count. Fail-fasts unless (numGroups==1 && groupSize==0)
 * or (numGroups>1 && groupSize>0 && numGroups*groupSize==numberOfElements).
 * Called by initTensor for SYM tensors; also the choke point the ODTS
 * deserialize path (Deserialize.c's deserializeQConfig, Task 5) re-validates
 * against after reallocating a skeleton's scales[] to a file's numGroups --
 * the serial read-path relax this shape check underwrites is SHIPPED, not a
 * future concern. */
void validateSymQConfigShape(const symQConfig_t *qC, size_t numberOfElements);
void initAsymQConfig(uint8_t qBits, roundingMode_t roundingMode, asymQConfig_t *asymQConfig);
/*! Group-quant PR4: general-shape ASYM config init, the exact ASYM twin of
 * initSymQConfigGrouped (same shape grammar, same fail-fasts) plus the D6
 * qBits ceiling [1, 16]. Allocates scales[numGroups] (each 1.f) AND
 * zeroPoints[numGroups] (each 0) as two separate owned blocks.
 * initAsymQConfig delegates here with (numGroups=1, groupSize=0). */
void initAsymQConfigGrouped(uint8_t qBits, roundingMode_t rm, size_t numGroups, size_t groupSize,
                            asymQConfig_t *qC);
/*! Group-quant PR4: attach-time shape check for an ASYM config against a
 * concrete element count (validateSymQConfigShape twin), PLUS the D6
 * qBits-in-[1,16] re-check for field-assigned configs. Called by initTensor
 * for ASYM tensors. */
void validateAsymQConfigShape(const asymQConfig_t *qC, size_t numberOfElements);

void initInt32Quantization(quantization_t *quantization);
void initFloat32Quantization(quantization_t *quantization);
void initBoolQuantization(quantization_t *quantization);

void initSymInt32Quantization(symInt32QConfig_t *symInt32QConfig, quantization_t *quantization);
void initSymQuantization(symQConfig_t *symQConfig, quantization_t *quantization);
void initAsymQuantization(asymQConfig_t *asymQConfig, quantization_t *quantization);

#endif // ENV5_RUNTIME_QUANTIZATION_H
