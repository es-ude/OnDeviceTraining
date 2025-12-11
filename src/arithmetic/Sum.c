#include "Add.h"
#include "Sum.h"

int32_t sumint32(int32_t *values, size_t numberOfValues) {
    int32_t sum = 0;
    for(size_t i = 0; i < numberOfValues; i++) {
        sum += values[i];
    }
    return sum;
}

float sumFloat(float *values, size_t numberOfValues) {
    float sum = 0;
    for(size_t i = 0; i < numberOfValues; i++) {
        sum += values[i];
    }
    return sum;
}
