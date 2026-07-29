#define SOURCE_FILE "QUANTIZATION"

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#include "Common.h"
#include "Quantization.h"
#include "Rounding.h"
#include "StorageApi.h"

void initSymInt32QConfig(roundingMode_t roundingMode, symInt32QConfig_t *symInt32QConfig) {
    symInt32QConfig->roundingMode = roundingMode;
    symInt32QConfig->scale = 1.f;
    symInt32QConfig->qMaxBits = ODT_SYM_OPERAND_QMAXBITS; /* was 16 — #227 int12 operands */
}

void initSymInt32QConfigWithQMaxBits(roundingMode_t roundingMode,
                                     symInt32QConfig_t *symInt32QConfig, uint8_t qMaxBits) {
    /* #202: qMaxBits > 31 makes the float32 clamp bound powf(2, qMaxBits - 1) - 1
     * round up past INT32_MAX, so the (int32_t) cast in the SYM_INT32 converters is
     * out of range (UB). 31 stays valid (raw-int/scale=1 regime, #227). This init is
     * the single chokepoint every SYM_INT32 qConfig passes through. */
    if (qMaxBits > 31) {
        PRINT_ERROR("qMaxBits (%u) exceeds the cast-safe SYM_INT32 ceiling of 31 (#202)",
                    (unsigned)qMaxBits);
        exit(1);
    }
    symInt32QConfig->roundingMode = roundingMode;
    symInt32QConfig->scale = 1.f;
    symInt32QConfig->qMaxBits = qMaxBits;
}

void initSymQConfig(uint8_t qBits, roundingMode_t roundingMode, symQConfig_t *symQConfig) {
    /* Group-quant PR2: delegates to the general-shape init with the PR1
     * per-tensor sentinel (numGroups=1, groupSize=0) -- keeps this producer's
     * behavior bit-identical to PR1 (the PR1 sentinel tests in
     * UnitTestQuantization.c are the regression net for that claim). */
    initSymQConfigGrouped(qBits, roundingMode, 1, 0, symQConfig);
}

void validateSymQConfigShape(const symQConfig_t *qC, size_t numberOfElements) {
    bool perTensor = qC->numGroups == 1 && qC->groupSize == 0;
    bool grouped =
        qC->numGroups > 1 && qC->groupSize > 0 && qC->numGroups * qC->groupSize == numberOfElements;
    if (!perTensor && !grouped) {
        PRINT_ERROR("validateSymQConfigShape: invalid SYM group shape numGroups=%zu "
                    "groupSize=%zu for %zu elements (per-tensor is {1,0}; grouped needs "
                    "numGroups*groupSize == elements)",
                    qC->numGroups, qC->groupSize, numberOfElements);
        exit(1);
    }
}

void initSymQConfigGrouped(uint8_t qBits, roundingMode_t roundingMode, size_t numGroups,
                           size_t groupSize, symQConfig_t *qC) {
    /* Caller allocates the outer struct; this init allocates the
     * numGroups-element scales array (contract per the header doc) -- the
     * filled config owns a heap scales array, pair with freeReservedMemory
     * (bare qConfig) or freeQuantization (qConfig wrapped in a
     * quantization_t). Only {1,0} (per-tensor) or {>1,>0} (grouped) are
     * constructible -- {1,N>0} and {0,*} are rejected here. */
    if (numGroups == 0 || (numGroups == 1) != (groupSize == 0)) {
        PRINT_ERROR("initSymQConfigGrouped: invalid group shape numGroups=%zu groupSize=%zu "
                    "(per-tensor is {1,0}; grouped needs numGroups>1 and groupSize>0)",
                    numGroups, groupSize);
        exit(1);
    }
    qC->scales = reserveMemory(numGroups * sizeof(float));
    for (size_t g = 0; g < numGroups; g++) {
        qC->scales[g] = 1.f;
    }
    qC->numGroups = numGroups;
    qC->groupSize = groupSize;
    qC->roundingMode = roundingMode;
    qC->qBits = qBits;
}

void initAsymQConfig(uint8_t qBits, roundingMode_t roundingMode, asymQConfig_t *asymQConfig) {
    /* Group-quant PR4: delegates to the general-shape init with the
     * per-tensor sentinel (numGroups=1, groupSize=0), mirroring
     * initSymQConfig's delegation. */
    initAsymQConfigGrouped(qBits, roundingMode, 1, 0, asymQConfig);
}

/* D6 width ceiling shared by the init and attach-time funnels: the
 * code-domain zeroPoint is uint16, so qBits > 16 has codes/zp with no uint16
 * representation (supersedes the old [1, 30] #246 ceiling, whose wide-band
 * int32-zp rationale is void under the zero-inclusion nudge -- see
 * Quantization.h). qBits == 0 would underflow the sub-byte packer. */
static void validateAsymQBits(uint8_t qBits, const char *what) {
    if (qBits == 0 || qBits > 16) {
        PRINT_ERROR("%s: qBits (%u) outside the ASYM range [1, 16] (D6)", what, (unsigned)qBits);
        exit(1);
    }
}

void initAsymQConfigGrouped(uint8_t qBits, roundingMode_t rm, size_t numGroups, size_t groupSize,
                            asymQConfig_t *qC) {
    /* Caller allocates the outer struct; this init allocates the two
     * numGroups-element arrays (scales all 1.f, zeroPoints all 0 -- the
     * zp==0 zero-state makes code 0 decode to exactly 0.0f). Shape grammar
     * identical to initSymQConfigGrouped: only {1,0} (per-tensor) or
     * {>1,>0} (grouped) are constructible. */
    validateAsymQBits(qBits, "initAsymQConfigGrouped");
    if (numGroups == 0 || (numGroups == 1) != (groupSize == 0)) {
        PRINT_ERROR("initAsymQConfigGrouped: invalid group shape numGroups=%zu groupSize=%zu "
                    "(per-tensor is {1,0}; grouped needs numGroups>1 and groupSize>0)",
                    numGroups, groupSize);
        exit(1);
    }
    qC->scales = reserveMemory(numGroups * sizeof(float));
    qC->zeroPoints = reserveMemory(numGroups * sizeof(uint16_t));
    for (size_t g = 0; g < numGroups; g++) {
        qC->scales[g] = 1.f;
        qC->zeroPoints[g] = 0;
    }
    qC->numGroups = numGroups;
    qC->groupSize = groupSize;
    qC->roundingMode = rm;
    qC->qBits = qBits;
}

void validateAsymQConfigShape(const asymQConfig_t *qC, size_t numberOfElements) {
    validateAsymQBits(qC->qBits, "validateAsymQConfigShape");
    bool perTensor = qC->numGroups == 1 && qC->groupSize == 0;
    bool grouped =
        qC->numGroups > 1 && qC->groupSize > 0 && qC->numGroups * qC->groupSize == numberOfElements;
    if (!perTensor && !grouped) {
        PRINT_ERROR("validateAsymQConfigShape: invalid ASYM group shape numGroups=%zu "
                    "groupSize=%zu for %zu elements (per-tensor is {1,0}; grouped needs "
                    "numGroups*groupSize == elements)",
                    qC->numGroups, qC->groupSize, numberOfElements);
        exit(1);
    }
}

void initBfpQConfigGrouped(uint8_t mantissaBits, uint8_t exponentBits, roundingMode_t roundingMode,
                           size_t numGroups, size_t groupSize, bfpQConfig_t *qC) {
    if (mantissaBits < 2 || mantissaBits > 16) {
        PRINT_ERROR("initBfpQConfigGrouped: mantissaBits (%u) outside [2, 16]",
                    (unsigned)mantissaBits);
        exit(1);
    }
    if (exponentBits < 2 || exponentBits > 8) {
        PRINT_ERROR("initBfpQConfigGrouped: exponentBits (%u) outside [2, 8]",
                    (unsigned)exponentBits);
        exit(1);
    }
    if (numGroups == 0 || (numGroups == 1) != (groupSize == 0)) {
        PRINT_ERROR("initBfpQConfigGrouped: invalid group shape numGroups=%zu groupSize=%zu "
                    "(per-tensor is {1,0}; grouped needs numGroups>1 and groupSize>0)",
                    numGroups, groupSize);
        exit(1);
    }
    qC->mantissaBits = mantissaBits;
    qC->exponentBits = exponentBits;
    qC->roundingMode = roundingMode;
    qC->numGroups = numGroups;
    qC->groupSize = groupSize;
    qC->exponents = reserveMemory(numGroups * sizeof(uint8_t));
    uint8_t bias = (uint8_t)((1 << (exponentBits - 1)) - 1);
    for (size_t g = 0; g < numGroups; g++) {
        qC->exponents[g] = bias; /* zero-state: E = 0, scale 1.0 (SYM's scales=1.f parity) */
    }
}

void initBfpQConfig(uint8_t mantissaBits, uint8_t exponentBits, roundingMode_t roundingMode,
                    bfpQConfig_t *qC) {
    initBfpQConfigGrouped(mantissaBits, exponentBits, roundingMode, 1, 0, qC);
}

void validateBfpQConfigShape(const bfpQConfig_t *qC, size_t numberOfElements) {
    bool perTensor = qC->numGroups == 1 && qC->groupSize == 0;
    bool grouped =
        qC->numGroups > 1 && qC->groupSize > 0 && qC->numGroups * qC->groupSize == numberOfElements;
    if (!perTensor && !grouped) {
        PRINT_ERROR("validateBfpQConfigShape: invalid BFP group shape numGroups=%zu "
                    "groupSize=%zu for %zu elements",
                    qC->numGroups, qC->groupSize, numberOfElements);
        exit(1);
    }
}

int32_t bfpExponentBias(const bfpQConfig_t *qC) {
    return (1 << (qC->exponentBits - 1)) - 1;
}

float bfpGroupScale(const bfpQConfig_t *qC, size_t group) {
    return ldexpf(1.f, (int)qC->exponents[group] - bfpExponentBias(qC));
}

void initInt32Quantization(quantization_t *quantization) {
    quantization->type = INT32;
    quantization->qConfig = NULL;
}

void initFloat32Quantization(quantization_t *quantization) {
    quantization->type = FLOAT32;
    quantization->qConfig = NULL;
}

void initBoolQuantization(quantization_t *quantization) {
    quantization->type = BOOL;
    quantization->qConfig = NULL;
}

void initSymInt32Quantization(symInt32QConfig_t *symInt32QConfig, quantization_t *quantization) {
    quantization->type = SYM_INT32;
    quantization->qConfig = symInt32QConfig;
}

void initSymQuantization(symQConfig_t *symQConfig, quantization_t *quantization) {
    quantization->type = SYM;
    quantization->qConfig = symQConfig;
}

void initAsymQuantization(asymQConfig_t *asymQConfig, quantization_t *quantization) {
    quantization->type = ASYM;
    quantization->qConfig = asymQConfig;
}

void initBfpQuantization(bfpQConfig_t *bfpQConfig, quantization_t *quantization) {
    quantization->type = BFP;
    quantization->qConfig = bfpQConfig;
}
