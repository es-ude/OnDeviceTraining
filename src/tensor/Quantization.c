#define SOURCE_FILE "QUANTIZATION"

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
    /* #246: qBits > 30 makes the unsigned code ceiling powf(2, qBits) - 1 round
     * to 2^31 in float, so the (int32_t) cast in the ASYM emit path is out of
     * range (UB) -- the unsigned twin of the #202 SYM_INT32 ceiling at 31. 0
     * would underflow the sub-byte packer. deriveAsymGridFromMinMax re-checks
     * for configs built without this init. */
    if (qBits == 0 || qBits > 30) {
        PRINT_ERROR("qBits (%u) outside the ASYM range [1, 30] (#246)", (unsigned)qBits);
        exit(1);
    }
    asymQConfig->qBits = qBits;
    asymQConfig->roundingMode = roundingMode;
    asymQConfig->scale = 1.f;
    asymQConfig->zeroPoint = 0;
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
