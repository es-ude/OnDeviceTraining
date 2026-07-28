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
    /* Group-quant PR1: caller allocates the outer struct; this init now also
     * allocates the 1-element scales array (contract change, see the header
     * doc) -- the filled config owns a heap scales array, pair with
     * freeReservedMemory (bare qConfig) or freeQuantization (qConfig wrapped
     * in a quantization_t). PR1 is always per-tensor: numGroups=1,
     * groupSize=0 (the "whole tensor" sentinel). */
    float *scales = reserveMemory(sizeof(float));
    scales[0] = 1.f;
    symQConfig->scales = scales;
    symQConfig->numGroups = 1;
    symQConfig->groupSize = 0;
    symQConfig->qBits = qBits;
    symQConfig->roundingMode = roundingMode;
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
