#define SOURCE_FILE "QUANTIZATION"

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#include "CheckedSize.h"
#include "Common.h"
#include "DyadicScale.h"
#include "Quantization.h"
#include "Rounding.h"
#include "StorageApi.h"

void initSymInt32QConfig(roundingMode_t roundingMode, symInt32QConfig_t *symInt32QConfig) {
    symInt32QConfig->roundingMode = roundingMode;
    odtScaleStore(&symInt32QConfig->scale, ODT_SCALE_ONE);
    symInt32QConfig->qMaxBits = ODT_SYM_OPERAND_QMAXBITS;
}

void initSymInt32QConfigWithQMaxBits(roundingMode_t roundingMode,
                                     symInt32QConfig_t *symInt32QConfig, uint8_t qMaxBits) {
    /* 31 is the widest width whose rounded float bounds stay representable in
     * int32 (SymGridBounds.h); 32 marks a raw kernel output. */
    if (qMaxBits > 31) {
        PRINT_ERROR("qMaxBits (%u) exceeds the cast-safe SYM_INT32 ceiling of 31; 32 marks "
                    "a raw kernel output -- requantize it into a <= 31-bit tensor before cloning "
                    "it (getQLike) or requantizing into it",
                    (unsigned)qMaxBits);
        exit(1);
    }
    symInt32QConfig->roundingMode = roundingMode;
    odtScaleStore(&symInt32QConfig->scale, ODT_SCALE_ONE);
    symInt32QConfig->qMaxBits = qMaxBits;
}

void initSymQConfig(uint8_t qBits, roundingMode_t roundingMode, symQConfig_t *symQConfig) {
    initSymQConfigGrouped(qBits, roundingMode, 1, 0, symQConfig);
}

/* The shape validators run on field-assigned configs too, so the group
 * product is untrusted: a wrapping numGroups*groupSize could otherwise match
 * the element count and pass a grouping that indexes far past the tensor. */
static size_t groupProductOrExit(size_t numGroups, size_t groupSize, const char *what) {
    size_t product;
    if (!checkedMulSize(numGroups, groupSize, &product)) {
        PRINT_ERROR("%s: numGroups*groupSize overflows size_t (numGroups=%zu groupSize=%zu)", what,
                    numGroups, groupSize);
        exit(1);
    }
    return product;
}

/* 31 is the widest width the packer and a SYM->SYM_INT32 copy carry. qBits == 0
 * underflows the packer; qBits == 1 has qMax = 0, so the derived scale is inf. */
static void validateSymQBits(uint8_t qBits, const char *what) {
    if (qBits < 2 || qBits > 31) {
        PRINT_ERROR("%s: qBits (%u) outside the SYM range [2, 31]", what, (unsigned)qBits);
        exit(1);
    }
}

void validateSymQConfigShape(const symQConfig_t *qC, size_t numberOfElements) {
    validateSymQBits(qC->qBits, "validateSymQConfigShape");
    bool perTensor = qC->numGroups == 1 && qC->groupSize == 0;
    size_t groupedElements =
        groupProductOrExit(qC->numGroups, qC->groupSize, "validateSymQConfigShape");
    bool grouped = qC->numGroups > 1 && qC->groupSize > 0 && groupedElements == numberOfElements;
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
    validateSymQBits(qBits, "initSymQConfigGrouped");
    if (numGroups == 0 || (numGroups == 1) != (groupSize == 0)) {
        PRINT_ERROR("initSymQConfigGrouped: invalid group shape numGroups=%zu groupSize=%zu "
                    "(per-tensor is {1,0}; grouped needs numGroups>1 and groupSize>0)",
                    numGroups, groupSize);
        exit(1);
    }
    qC->scales = reserveMemory(numGroups * sizeof(*qC->scales));
    for (size_t g = 0; g < numGroups; g++) {
        odtScaleStore(&qC->scales[g], ODT_SCALE_ONE);
    }
    qC->numGroups = numGroups;
    qC->groupSize = groupSize;
    qC->roundingMode = roundingMode;
    qC->qBits = qBits;
}

void initAsymQConfig(uint8_t qBits, roundingMode_t roundingMode, asymQConfig_t *asymQConfig) {
    initAsymQConfigGrouped(qBits, roundingMode, 1, 0, asymQConfig);
}

/* zeroPoints are uint16, so codes wider than 16 bits have no zero point;
 * qBits == 0 underflows the packer. */
static void validateAsymQBits(uint8_t qBits, const char *what) {
    if (qBits == 0 || qBits > 16) {
        PRINT_ERROR("%s: qBits (%u) outside the ASYM range [1, 16]", what, (unsigned)qBits);
        exit(1);
    }
}

void initAsymQConfigGrouped(uint8_t qBits, roundingMode_t rm, size_t numGroups, size_t groupSize,
                            asymQConfig_t *qC) {
    validateAsymQBits(qBits, "initAsymQConfigGrouped");
    if (numGroups == 0 || (numGroups == 1) != (groupSize == 0)) {
        PRINT_ERROR("initAsymQConfigGrouped: invalid group shape numGroups=%zu groupSize=%zu "
                    "(per-tensor is {1,0}; grouped needs numGroups>1 and groupSize>0)",
                    numGroups, groupSize);
        exit(1);
    }
    qC->scales = reserveMemory(numGroups * sizeof(*qC->scales));
    qC->zeroPoints = reserveMemory(numGroups * sizeof(uint16_t));
    for (size_t g = 0; g < numGroups; g++) {
        odtScaleStore(&qC->scales[g], ODT_SCALE_ONE);
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
    size_t groupedElements =
        groupProductOrExit(qC->numGroups, qC->groupSize, "validateAsymQConfigShape");
    bool grouped = qC->numGroups > 1 && qC->groupSize > 0 && groupedElements == numberOfElements;
    if (!perTensor && !grouped) {
        PRINT_ERROR("validateAsymQConfigShape: invalid ASYM group shape numGroups=%zu "
                    "groupSize=%zu for %zu elements (per-tensor is {1,0}; grouped needs "
                    "numGroups*groupSize == elements)",
                    qC->numGroups, qC->groupSize, numberOfElements);
        exit(1);
    }
}

/* Shared by the allocating and the borrowed-storage BFP inits; `what` names
 * the entry in the diagnostic. */
static void validateBfpInitArgs(uint8_t mantissaBits, uint8_t exponentBits, size_t numGroups,
                                size_t groupSize, const char *what) {
    if (mantissaBits < 2 || mantissaBits > 16) {
        PRINT_ERROR("%s: mantissaBits (%u) outside [2, 16]", what, (unsigned)mantissaBits);
        exit(1);
    }
    if (exponentBits < 2 || exponentBits > 8) {
        PRINT_ERROR("%s: exponentBits (%u) outside [2, 8]", what, (unsigned)exponentBits);
        exit(1);
    }
    if (numGroups == 0 || (numGroups == 1) != (groupSize == 0)) {
        PRINT_ERROR("%s: invalid group shape numGroups=%zu groupSize=%zu "
                    "(per-tensor is {1,0}; grouped needs numGroups>1 and groupSize>0)",
                    what, numGroups, groupSize);
        exit(1);
    }
}

void initBfpQConfigGroupedInto(uint8_t mantissaBits, uint8_t exponentBits,
                               roundingMode_t roundingMode, size_t numGroups, size_t groupSize,
                               uint8_t *exponents, bfpQConfig_t *qC) {
    validateBfpInitArgs(mantissaBits, exponentBits, numGroups, groupSize,
                        "initBfpQConfigGroupedInto");
    if (exponents == NULL) {
        PRINT_ERROR("initBfpQConfigGroupedInto: exponents is NULL (needs storage for %zu groups)",
                    numGroups);
        exit(1);
    }
    qC->mantissaBits = mantissaBits;
    qC->exponentBits = exponentBits;
    qC->roundingMode = roundingMode;
    qC->numGroups = numGroups;
    qC->groupSize = groupSize;
    qC->exponents = exponents;
    uint8_t bias = (uint8_t)((1 << (exponentBits - 1)) - 1);
    for (size_t g = 0; g < numGroups; g++) {
        qC->exponents[g] = bias; /* zero state: scale 1 */
    }
}

void initBfpQConfigGrouped(uint8_t mantissaBits, uint8_t exponentBits, roundingMode_t roundingMode,
                           size_t numGroups, size_t groupSize, bfpQConfig_t *qC) {
    /* Validate before reserving, so a rejected shape allocates nothing; the Into
     * call re-validates (init-time, cheap) and rejects a failed reservation. */
    validateBfpInitArgs(mantissaBits, exponentBits, numGroups, groupSize, "initBfpQConfigGrouped");
    uint8_t *exponents = reserveMemory(numGroups * sizeof(uint8_t));
    initBfpQConfigGroupedInto(mantissaBits, exponentBits, roundingMode, numGroups, groupSize,
                              exponents, qC);
}

void initBfpQConfig(uint8_t mantissaBits, uint8_t exponentBits, roundingMode_t roundingMode,
                    bfpQConfig_t *qC) {
    initBfpQConfigGrouped(mantissaBits, exponentBits, roundingMode, 1, 0, qC);
}

void initBfpQConfigInto(uint8_t mantissaBits, uint8_t exponentBits, roundingMode_t roundingMode,
                        uint8_t *exponents, bfpQConfig_t *qC) {
    initBfpQConfigGroupedInto(mantissaBits, exponentBits, roundingMode, 1, 0, exponents, qC);
}

void validateBfpQConfigShape(const bfpQConfig_t *qC, size_t numberOfElements) {
    bool perTensor = qC->numGroups == 1 && qC->groupSize == 0;
    size_t groupedElements =
        groupProductOrExit(qC->numGroups, qC->groupSize, "validateBfpQConfigShape");
    bool grouped = qC->numGroups > 1 && qC->groupSize > 0 && groupedElements == numberOfElements;
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

void initBfpRawQuantization(bfpRawQConfig_t *bfpRawQConfig, quantization_t *quantization) {
    quantization->type = BFP_RAW;
    quantization->qConfig = bfpRawQConfig;
}

void validateBfpRawQConfigShape(const bfpRawQConfig_t *qC, size_t n, const char *what) {
    bool perTensor = qC->numGroups == 1 && qC->groupSize == 0;
    bool grouped = qC->numGroups > 1 && qC->groupSize > 0 && qC->numGroups * qC->groupSize == n;
    if (!perTensor && !grouped) {
        PRINT_ERROR("%s: invalid BFP_RAW group shape numGroups=%zu groupSize=%zu for %zu elements "
                    "(valid: {1,0} or {>1,>0} with numGroups*groupSize == n)",
                    what, qC->numGroups, qC->groupSize, n);
        exit(1);
    }
}
