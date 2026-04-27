#define SOURCE_FILE "DISTRIBUTIONS"

#include <math.h>

#include "Distributions.h"
#include "RNG.h"

float randomNormal(float mean, float standardDeviation) {
    float u1 = rngNextFloat();
    float u2 = rngNextFloat();

    // Avoid log(0)
    while (u1 <= 1e-7f) {
        u1 = rngNextFloat();
    }

    float z = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * (float)M_PI * u2);
    return mean + standardDeviation * z;
}

float randomUniform(float min, float max) {
    float r = rngNextFloat();
    return min + r * (max - min);
}

float kaimingNormal(float gain, size_t fanMode) {
    float std = gain / sqrtf((float)fanMode);
    return randomNormal(0.0f, std);
}

float kaimingUniform(float gain, size_t fanMode) {
    float limit = gain * sqrtf(3.0f / (float)fanMode);
    return randomUniform(-limit, limit);
}

float xavierNormal(float gain, size_t fanIn, size_t fanOut) {
    float std = gain * sqrtf(2.0f / (float)(fanIn + fanOut));
    return randomNormal(0.0f, std);
}

float xavierUniform(float gain, size_t fanIn, size_t fanOut) {
    float limit = gain * sqrtf(6.0f / (float)(fanIn + fanOut));
    return randomUniform(-limit, limit);
}
