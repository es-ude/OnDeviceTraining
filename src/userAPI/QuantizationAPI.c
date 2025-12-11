#include <stdlib.h>

#include "QuantizationAPI.h"

quantization_t *quantizationInitFloat() {
    quantization_t *q = calloc(1, sizeof(quantization_t));
    initFloat32Quantization(q);
    return q;
}

quantization_t *quantizationInitInt32() {
    quantization_t *q = calloc(1, sizeof(quantization_t));
    initInt32Quantization(q);
    return q;
}

quantization_t *quantizationInitSymInt32(roundingMode_t roundingMode) {
    quantization_t *q = calloc(1, sizeof(quantization_t));
    symInt32QConfig_t *qC = calloc(1, sizeof(symInt32QConfig_t));
    initSymInt32QConfig(roundingMode, qC);
    initSymInt32Quantization(qC, q);
    return q;
}

quantization_t *quantizationInitAsym(uint8_t qBits, roundingMode_t roundingMode) {
    quantization_t *q = calloc(1, sizeof(quantization_t));
    asymQConfig_t *qC = calloc(1, sizeof(asymQConfig_t));
    initAsymQConfig(qBits, roundingMode, qC);
    initAsymQuantization(qC, q);
    return q;
}