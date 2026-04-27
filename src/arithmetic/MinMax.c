#define SOURCE_FILE "MINMAX"

#include <math.h>

#include "DTypes.h"
#include "MinMax.h"

float findAbsMaxFloat(uint8_t *bytes, size_t numberOfElements) {
    float *values = (float *)bytes;
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

int32_t findMaxInt32(uint8_t *bytes, size_t numberOfElements) {
    size_t bytesPerElement = sizeof(int32_t);
    int32_t max = readBytesAsInt32(&bytes[0]);
    for (size_t i = 1; i < numberOfElements; i++) {
        size_t byteIndex = i * bytesPerElement;
        int32_t current = readBytesAsInt32(&bytes[byteIndex]);
        if (current > max) {
            max = current;
        }
    }
    return max;
}

int32_t findMinInt32(uint8_t *bytes, size_t numberOfElements) {
    size_t bytesPerElement = sizeof(int32_t);
    int32_t min = readBytesAsInt32(&bytes[0]);
    for (size_t i = 1; i < numberOfElements; i++) {
        size_t byteIndex = i * bytesPerElement;
        int32_t current = readBytesAsInt32(&bytes[byteIndex]);
        if (current < min) {
            min = current;
        }
    }
    return min;
}
