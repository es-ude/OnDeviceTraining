#define SOURCE_FILE "DISTRIBUTIONS"

#include <math.h>

#include "Distributions.h"

/* Box-Muller, cosine branch only (the second variate is discarded). */
static float boxMullerFromUniforms(float u1, float u2, float mean, float standardDeviation) {
    float r = sqrtf(-2.0f * logf(u1));
    return mean + standardDeviation * r * cosf(2.0f * (float)M_PI * u2);
}

float randomNormalCtx(rngStream_t *rs, float mean, float standardDeviation) {
    float u1 = rngNextUnitFloat(rs);
    float u2 = rngNextUnitFloat(rs);

    // Avoid log(0)
    while (u1 <= 1e-7f) {
        u1 = rngNextUnitFloat(rs);
    }

    return boxMullerFromUniforms(u1, u2, mean, standardDeviation);
}

float randomUniformCtx(rngStream_t *rs, float min, float max) {
    float r = rngNextUnitFloat(rs);
    return min + r * (max - min);
}

float kaimingNormal(float gain, size_t fanMode, rngStream_t *rs) {
    float std = gain / sqrtf((float)fanMode);
    return randomNormalCtx(rs, 0.0f, std);
}

float kaimingUniform(float gain, size_t fanMode, rngStream_t *rs) {
    float limit = gain * sqrtf(3.0f / (float)fanMode);
    return randomUniformCtx(rs, -limit, limit);
}

float xavierNormal(float gain, size_t fanIn, size_t fanOut, rngStream_t *rs) {
    float std = gain * sqrtf(2.0f / (float)(fanIn + fanOut));
    return randomNormalCtx(rs, 0.0f, std);
}

float xavierUniform(float gain, size_t fanIn, size_t fanOut, rngStream_t *rs) {
    float limit = gain * sqrtf(6.0f / (float)(fanIn + fanOut));
    return randomUniformCtx(rs, -limit, limit);
}
