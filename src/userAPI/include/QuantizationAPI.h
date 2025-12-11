#ifndef QUANTIZATIONAPI_H
#define QUANTIZATIONAPI_H

#include "Quantization.h"
#include "Rounding.h"

quantization_t* quantizationInitFloat();

quantization_t* quantizationInitInt32();

quantization_t* quantizationInitSymInt32(roundingMode_t roundingMode);

quantization_t* quantizationInitAsym(uint8_t qBits, roundingMode_t roundingMode);

void freeQuantization(quantization_t* quantization);

#endif //QUANTIZATIONAPI_H
