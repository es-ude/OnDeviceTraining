#define SOURCE_FILE "MINMAX"

#include <math.h>

#include "DTypes.h"
#include "MinMax.h"

float absFloat32(float a) {
    return fabsf(a);
}

float maxFloat32s(float a, float b) {
    return a > b ? a : b;
}

float findAbsMaxFloat(uint8_t *bytes, size_t numberOfElements) {
    float *values = (float *)bytes;
    /* Empty tensors exist; 0.f is what the BFP exponent derivation reads as "no data". */
    if (numberOfElements == 0) {
        return 0.f;
    }
    float max = fabsf(values[0]);

    for (size_t i = 1; i < numberOfElements; i++) {
        if (fabsf(values[i]) > max) {
            max = fabsf(values[i]);
        }
    }
    return max;
}

float findMaxFloat(uint8_t *bytes, size_t numberOfElements) {
    size_t bytesPerElement = sizeof(float);
    float max = readBytesAsFloat(&bytes[0]);
    for (size_t i = 1; i < numberOfElements; i++) {
        size_t byteIndex = i * bytesPerElement;
        float current = readBytesAsFloat(&bytes[byteIndex]);
        if (current > max) {
            max = current;
        }
    }
    return max;
}

float findMinFloat(uint8_t *bytes, size_t numberOfElements) {
    size_t bytesPerElement = sizeof(float);
    float min = readBytesAsFloat(&bytes[0]);
    for (size_t i = 1; i < numberOfElements; i++) {
        size_t byteIndex = i * bytesPerElement;
        float current = readBytesAsFloat(&bytes[byteIndex]);
        if (current < min) {
            min = current;
        }
    }
    return min;
}
