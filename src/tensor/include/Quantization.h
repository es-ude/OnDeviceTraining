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
 * N-agnostic as the old scalar scale. Invariant, enforced where checkable:
 * numGroups == 1 <=> groupSize == 0. Ownership: scales is heap-allocated
 * (reserveMemory) by whichever call filled the struct -- initSymQConfig
 * (PR1's only producer) always allocates a fresh 1-element array; pair with
 * freeReservedMemory (bare qConfig) or freeQuantization (config wrapped in a
 * quantization_t, e.g. via quantizationInitSym / getQLike). Stack test
 * fixtures that only need a fixed per-tensor scale should build the struct
 * directly with a local backing array instead of calling initSymQConfig (see
 * docs/conventions/testing.md) -- such fixtures are never passed to
 * freeQuantization or freeReservedMemory. */
typedef struct symQConfig {
    float *scales;    /* [numGroups], owned by the qconfig (see ownership note above) */
    size_t numGroups; /* PR1: always 1 */
    size_t groupSize; /* 0 = "whole tensor" sentinel (per-tensor, N-agnostic);
                        >0 only for real groups (PR2+). PR1: always 0. */
    roundingMode_t roundingMode;
    uint8_t qBits;
} symQConfig_t;

typedef struct asymQConfig {
    float scale;
    /* int32: zeroPoint = round(min/scale) reaches -(2^qBits - 1) for negative
     * bands and exceeds it by min/(min - max) for all-negative ones -- far
     * outside int16 already at qBits=16 (#246). */
    int32_t zeroPoint;
    uint8_t qBits;
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
void initAsymQConfig(uint8_t qBits, roundingMode_t roundingMode, asymQConfig_t *asymQConfig);

void initInt32Quantization(quantization_t *quantization);
void initFloat32Quantization(quantization_t *quantization);
void initBoolQuantization(quantization_t *quantization);

void initSymInt32Quantization(symInt32QConfig_t *symInt32QConfig, quantization_t *quantization);
void initSymQuantization(symQConfig_t *symQConfig, quantization_t *quantization);
void initAsymQuantization(asymQConfig_t *asymQConfig, quantization_t *quantization);

#endif // ENV5_RUNTIME_QUANTIZATION_H
