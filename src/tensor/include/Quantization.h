#ifndef ENV5_RUNTIME_QUANTIZATION_H
#define ENV5_RUNTIME_QUANTIZATION_H

#include <stddef.h>

#include "DyadicScale.h"
#include "Rounding.h"

/* BFP_RAW is appended last: stored tags (Serialize.c writes the enum value)
 * stay stable, and it is never a stored tag itself (see bfpRawQConfig_t). */
typedef enum qtype { INT32, FLOAT32, SYM_INT32, SYM, ASYM, BOOL, BFP, BFP_RAW } qtype_t;

typedef struct symInt32QConfig {
    odtScale_t scale; /* canonical dyadic (DyadicScale.h); read/write via ScaleCanon.h */
    roundingMode_t roundingMode;
    uint8_t qMaxBits;
} symInt32QConfig_t;

/* Default SYM_INT32 operand staging width (initSymInt32QConfig, FLOAT32 absmax
 * encode). A default, not a limit: the accumulator guard (SymAccumGuard.h)
 * picks the unchecked or the checked loop per op from the declared widths and
 * the reduction length (int12*int12 is proven safe up to N = 511). Narrow it
 * for wider layers. LayerNorm/GroupNorm/Reduce cap operand width for their
 * own sums. Override with -DODT_SYM_OPERAND_QMAXBITS=N. */
#ifndef ODT_SYM_OPERAND_QMAXBITS
#define ODT_SYM_OPERAND_QMAXBITS 12
#endif
/* Width cap for a SYM_INT32/SYM grad accumulation TARGET, enforced in
 * accumulateOut (ExecuteOp.c) and used by gradInitSymInt32. Not a backward
 * precision knob (backward runs at operand width) and not a footprint knob
 * (SYM_INT32 grads stay 4 B/element; only packed SYM/ASYM grads save memory).
 * Reason: int16-range codes accumulate in int32 with headroom. ASYM grads are
 * capped at 16 by validateAsymQBits at init, so a lower N does not cap them.
 * Override with -DODT_SYM_GRAD_QMAXBITS=N. */
#ifndef ODT_SYM_GRAD_QMAXBITS
#define ODT_SYM_GRAD_QMAXBITS 16
#endif

/* Shape: per-tensor {numGroups = 1, groupSize = 0} (groupSize 0 = "whole
 * tensor", so a config built without a tensor needs no element count) or
 * grouped {numGroups > 1, groupSize > 0} with numGroups * groupSize == N,
 * checked where the config attaches to a tensor (validateSymQConfigShape,
 * called by initTensor). {1, N > 0} is not a per-tensor spelling.
 *
 * Ownership: the allocating inits reserve scales[numGroups]; free a bare
 * config's array with freeReservedMemory, a wrapped one with freeQuantization.
 * A stack fixture with a local backing array must never be freed AND never be
 * a deserialize destination: deserializeQConfig frees and reallocates scales
 * when the file's numGroups differs, which would free stack memory.
 *
 * Groups are legal only on GEMM-family weights; gradInit rejects a grouped
 * template, grads stay per-tensor. getQLike resets a per-tensor clone's scale
 * to 1 but copies a grouped clone's geometry and scales. */
typedef struct symQConfig {
    odtScale_t *scales; /* [numGroups], owned by the qconfig (see ownership note above) */
    size_t numGroups;   /* 1 = per-tensor; >1 = grouped */
    size_t groupSize;   /* 0 = whole tensor; >0 grouped, numGroups*groupSize == N */
    roundingMode_t roundingMode;
    uint8_t qBits;
} symQConfig_t;

/* Block floating point: one packed mantissaBits-wide mantissa per element and
 * one biased exponentBits-wide exponent per group; value = mantissa *
 * 2^(storedExponent - bias), bias = 2^(exponentBits-1) - 1 (see
 * docs/conventions/arithmetic-bfp.md). Shape and ownership follow symQConfig_t;
 * the ...Into inits borrow the caller's exponent storage instead, under the
 * same never-free / never-deserialize-into rules. Value quantization (exponent
 * derived from float magnitudes) saturates; raw-code packing into mantissas
 * aborts on overflow. */
typedef struct bfpQConfig {
    uint8_t *exponents; /* [numGroups], biased; heap (reserveMemory) and owned by the qconfig,
                         * or borrowed caller storage via the ...Into inits */
    size_t numGroups;   /* 1 = per-tensor */
    size_t groupSize;   /* 0 = whole-tensor sentinel; >0 grouped, numGroups*groupSize == N */
    roundingMode_t roundingMode;
    uint8_t mantissaBits; /* [2, 16]; also the ARITH_BFP operand width */
    uint8_t exponentBits; /* [2, 8]; bias = 2^(exponentBits-1) - 1 */
} bfpQConfig_t;

/* Integer-only BFP: the executeOp funnel's RAW intermediate under ARITH_BFP
 * when an op opts in (opSpec_t.bfpRawOut: the integer forward, the Conv1d
 * dx, the Conv1d weight / bias grads into BFP grads) -- int32 mantissas in
 * the tensor data plus ONE exponent per TARGET group, so a BFP kernel folds
 * integer partials and the OUT_WRITE epilogue re-blocks with shifts
 * (conversionMatrix[BFP_RAW][BFP]) or dequantizes into a FLOAT32 target
 * ([BFP_RAW][FLOAT32]); the grad ACC epilogue adds it to the stored grad
 * (accumulateBfpRawIntoBfpRescale). Transient by design: it lives on the funnel's stack
 * for one op, is never storage, never serialized, copied or cloned, and every
 * other dtype switch rejects it.
 * Exponents are UNBIASED (value = mantissa * 2^exponents[g]), unlike the
 * biased bytes of bfpQConfig_t: an accumulation exponent is the sum of two
 * unbiased operand exponents plus the fold guard, which exceeds int8.
 * Geometry grammar is bfpQConfig_t's ({1,0} per-tensor or {>1,>0} with
 * numGroups*groupSize == n); the funnel COPIES it from the target wire (a
 * FLOAT32 target has no groups: {1,0}), and the re-block cell rejects any
 * other pairing. The exponent array is the funnel's transient slice. */
typedef struct bfpRawQConfig {
    int16_t *exponents; /* [numGroups], UNBIASED; written by the kernel, one per group */
    size_t numGroups;   /* 1 = per-tensor */
    size_t groupSize;   /* 0 = whole-tensor sentinel; >0 grouped */
} bfpRawQConfig_t;

/* Code-domain affine ASYM (TFLite style): value = (code - zeroPoint) * scale.
 * Shape and ownership follow symQConfig_t (two owned arrays). The grid
 * derivation widens the band to include zero, so 0.0 is exact and zeroPoint
 * lies in [0, 2^qBits - 1]; uint16 zeroPoints cap qBits at 16. */
typedef struct asymQConfig {
    odtScale_t *scales;   /* [numGroups], owned (symQConfig ownership rules) */
    uint16_t *zeroPoints; /* [numGroups], owned; CODE-domain zp in
                             [0, 2^qBits - 1] (see block comment above) */
    size_t numGroups;     /* 1 = per-tensor sentinel; >1 = real groups */
    size_t groupSize;     /* 0 = "whole tensor" sentinel (per-tensor);
                            >0 only for real groups, numGroups*groupSize == N */
    uint8_t qBits;        /* [1, 16] */
    roundingMode_t roundingMode;
} asymQConfig_t;

typedef struct quantization {
    qtype_t type;
    void *qConfig;
} quantization_t;

/*! Sets qMaxBits to ODT_SYM_OPERAND_QMAXBITS. */
void initSymInt32QConfig(roundingMode_t roundingMode, symInt32QConfig_t *symInt32QConfig);
void initSymInt32QConfigWithQMaxBits(roundingMode_t roundingMode,
                                     symInt32QConfig_t *symInt32QConfig, uint8_t qMaxBits);
/*! SYM storage width is [2, 31]; the narrower operand and grad-target caps
 * apply where those contracts apply. */
void initSymQConfig(uint8_t qBits, roundingMode_t roundingMode, symQConfig_t *symQConfig);
/*! Allocates scales[numGroups], each 1. Rejects a width outside [2, 31] and
 * any shape other than {1, 0} or {>1, >0}. */
void initSymQConfigGrouped(uint8_t qBits, roundingMode_t roundingMode, size_t numGroups,
                           size_t groupSize, symQConfig_t *qC);
/*! Use where a config attaches to a concrete element count (initTensor, the
 * deserializer, requantize targets): checks width, shape, and the overflow-
 * checked numGroups * groupSize == numberOfElements. */
void validateSymQConfigShape(const symQConfig_t *qC, size_t numberOfElements);
void initAsymQConfig(uint8_t qBits, roundingMode_t roundingMode, asymQConfig_t *asymQConfig);
/*! Allocates scales[numGroups] = 1 and zeroPoints[numGroups] = 0. Rejects a
 * width outside [1, 16] and any shape other than {1, 0} or {>1, >0}. */
void initAsymQConfigGrouped(uint8_t qBits, roundingMode_t rm, size_t numGroups, size_t groupSize,
                            asymQConfig_t *qC);
/*! The ASYM twin of validateSymQConfigShape, width [1, 16]. */
void validateAsymQConfigShape(const asymQConfig_t *qC, size_t numberOfElements);

void initBfpQConfig(uint8_t mantissaBits, uint8_t exponentBits, roundingMode_t roundingMode,
                    bfpQConfig_t *qC);
/*! Allocates exponents[numGroups], each = bias (scale 1). Rejects mantissaBits
 * outside [2, 16], exponentBits outside [2, 8], and any shape other than {1, 0}
 * or {>1, >0}. initBfpQConfig is the {1, 0} case. */
void initBfpQConfigGrouped(uint8_t mantissaBits, uint8_t exponentBits, roundingMode_t roundingMode,
                           size_t numGroups, size_t groupSize, bfpQConfig_t *qC);
/*! For callers that must not allocate (the remat bind writes into its slab):
 * same checks and zero state, written into caller storage of numGroups bytes,
 * which the config borrows. Rejects exponents == NULL. */
void initBfpQConfigGroupedInto(uint8_t mantissaBits, uint8_t exponentBits,
                               roundingMode_t roundingMode, size_t numGroups, size_t groupSize,
                               uint8_t *exponents, bfpQConfig_t *qC);
/*! Per-tensor ({1,0}) initBfpQConfigGroupedInto; exponents needs 1 byte. */
void initBfpQConfigInto(uint8_t mantissaBits, uint8_t exponentBits, roundingMode_t roundingMode,
                        uint8_t *exponents, bfpQConfig_t *qC);
/*! Shape check where a BFP config attaches to a concrete element count. Does
 * not re-check the widths. */
void validateBfpQConfigShape(const bfpQConfig_t *qC, size_t numberOfElements);
int32_t bfpExponentBias(const bfpQConfig_t *qC);
float bfpGroupScale(const bfpQConfig_t *qC, size_t group);

void initInt32Quantization(quantization_t *quantization);
void initFloat32Quantization(quantization_t *quantization);
void initBoolQuantization(quantization_t *quantization);

void initSymInt32Quantization(symInt32QConfig_t *symInt32QConfig, quantization_t *quantization);
void initSymQuantization(symQConfig_t *symQConfig, quantization_t *quantization);
void initAsymQuantization(asymQConfig_t *asymQConfig, quantization_t *quantization);
void initBfpQuantization(bfpQConfig_t *bfpQConfig, quantization_t *quantization);
void initBfpRawQuantization(bfpRawQConfig_t *bfpRawQConfig, quantization_t *quantization);
/*! The BFP_RAW geometry check every raw producer and consumer runs before
 * indexing exponents[]: fail-fast (banner prefixed by `what`) unless
 * {numGroups=1, groupSize=0} or {numGroups>1, groupSize>0,
 * numGroups*groupSize == numberOfElements}. */
void validateBfpRawQConfigShape(const bfpRawQConfig_t *qC, size_t numberOfElements,
                                const char *what);

#endif // ENV5_RUNTIME_QUANTIZATION_H
