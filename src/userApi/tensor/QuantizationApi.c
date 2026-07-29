#define SOURCE_FILE "QUANTIZATION_API"

#include "QuantizationApi.h"
#include "QuantizationApiInternal.h"
#include "StorageApi.h"

quantization_t *quantizationInitFloat() {
    quantization_t *q = reserveMemory(sizeof(quantization_t));
    initFloat32Quantization(q);
    return q;
}

quantization_t *quantizationInitInt32() {
    quantization_t *q = reserveMemory(sizeof(quantization_t));
    initInt32Quantization(q);
    return q;
}

quantization_t *quantizationInitSymInt32(roundingMode_t roundingMode) {
    quantization_t *q = reserveMemory(sizeof(quantization_t));
    symInt32QConfig_t *qC = reserveMemory(sizeof(symInt32QConfig_t));
    initSymInt32QConfig(roundingMode, qC);
    initSymInt32Quantization(qC, q);
    return q;
}

quantization_t *quantizationInitSymInt32WithBits(roundingMode_t roundingMode, uint8_t qMaxBits) {
    quantization_t *q = reserveMemory(sizeof(quantization_t));
    symInt32QConfig_t *qC = reserveMemory(sizeof(symInt32QConfig_t));
    initSymInt32QConfigWithQMaxBits(roundingMode, qC, qMaxBits);
    initSymInt32Quantization(qC, q);
    return q;
}

quantization_t *quantizationInitSym(uint8_t qBits, roundingMode_t roundingMode) {
    quantization_t *q = reserveMemory(sizeof(quantization_t));
    symQConfig_t *qC = reserveMemory(sizeof(symQConfig_t));
    initSymQConfig(qBits, roundingMode, qC);
    initSymQuantization(qC, q);
    return q;
}

quantization_t *quantizationInitSymGrouped(uint8_t qBits, roundingMode_t roundingMode,
                                           size_t numGroups, size_t groupSize) {
    quantization_t *q = reserveMemory(sizeof(quantization_t));
    symQConfig_t *qC = reserveMemory(sizeof(symQConfig_t));
    initSymQConfigGrouped(qBits, roundingMode, numGroups, groupSize, qC);
    initSymQuantization(qC, q);
    return q;
}

quantization_t *quantizationInitBfp(uint8_t mantissaBits, uint8_t exponentBits,
                                    roundingMode_t roundingMode) {
    quantization_t *q = reserveMemory(sizeof(quantization_t));
    bfpQConfig_t *qC = reserveMemory(sizeof(bfpQConfig_t));
    initBfpQConfig(mantissaBits, exponentBits, roundingMode, qC);
    initBfpQuantization(qC, q);
    return q;
}

quantization_t *quantizationInitBfpGrouped(uint8_t mantissaBits, uint8_t exponentBits,
                                           roundingMode_t roundingMode, size_t numGroups,
                                           size_t groupSize) {
    quantization_t *q = reserveMemory(sizeof(quantization_t));
    bfpQConfig_t *qC = reserveMemory(sizeof(bfpQConfig_t));
    initBfpQConfigGrouped(mantissaBits, exponentBits, roundingMode, numGroups, groupSize, qC);
    initBfpQuantization(qC, q);
    return q;
}

quantization_t *quantizationInitAsym(uint8_t qBits, roundingMode_t roundingMode) {
    quantization_t *q = reserveMemory(sizeof(quantization_t));
    asymQConfig_t *qC = reserveMemory(sizeof(asymQConfig_t));
    initAsymQConfig(qBits, roundingMode, qC);
    initAsymQuantization(qC, q);
    return q;
}

quantization_t *quantizationInitAsymGrouped(uint8_t qBits, roundingMode_t rm, size_t numGroups,
                                            size_t groupSize) {
    quantization_t *q = reserveMemory(sizeof(quantization_t));
    asymQConfig_t *qC = reserveMemory(sizeof(asymQConfig_t));
    initAsymQConfigGrouped(qBits, rm, numGroups, groupSize, qC);
    initAsymQuantization(qC, q);
    return q;
}

quantization_t *quantizationInitBool(void) {
    quantization_t *q = reserveMemory(sizeof(quantization_t));
    initBoolQuantization(q);
    return q;
}
