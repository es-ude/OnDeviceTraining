#define SOURCE_FILE "DISTRIBUTIONS"

#include <math.h>
#include <stdlib.h>

#include "Common.h"
#include "Distributions.h"
#include "Quantization.h"
#include "Tensor.h"

/* Core Box-Muller on an abstract uniform source. u1 floored at 1e-7f to
 * avoid log(0); cosine branch only (second variate discarded) — preserves
 * the existing randomNormal() stream exactly. */
static float boxMullerFromUniforms(float u1, float u2, float mean, float standardDeviation) {
    float r = sqrtf(-2.0f * logf(u1));
    return mean + standardDeviation * r * cosf(2.0f * (float)M_PI * u2);
}

float randomNormalCtx(rng32_t *rng, float mean, float standardDeviation) {
    float u1 = rngNextFloatCtx(rng);
    float u2 = rngNextFloatCtx(rng);

    // Avoid log(0)
    while (u1 <= 1e-7f) {
        u1 = rngNextFloatCtx(rng);
    }

    return boxMullerFromUniforms(u1, u2, mean, standardDeviation);
}

float randomNormal(float mean, float standardDeviation) {
    float u1 = rngNextFloat();
    float u2 = rngNextFloat();

    // Avoid log(0)
    while (u1 <= 1e-7f) {
        u1 = rngNextFloat();
    }

    return boxMullerFromUniforms(u1, u2, mean, standardDeviation);
}

void fillNormalFloat32Tensor(tensor_t *out, rng32_t *rng, float mean, float stddev) {
    if (out->quantization->type != FLOAT32) {
        PRINT_ERROR("fillNormalFloat32Tensor: FLOAT32 tensors only");
        exit(1);
    }
    size_t n = calcNumberOfElementsByTensor(out);
    float *data = (float *)out->data;
    for (size_t i = 0; i < n; i++) {
        data[i] = randomNormalCtx(rng, mean, stddev);
    }
}

float randomUniform(float min, float max) {
    float r = rngNextFloat();
    return min + r * (max - min);
}

float randomUniformCtx(rngStream_t *rs, float min, float max) {
    if (rs == NULL) {
        PRINT_ERROR("randomUniformCtx: draw without a stream");
        exit(1);
    }
    float r = rngNextUnitFloat(rs);
    return min + r * (max - min);
}

float kaimingNormal(float gain, size_t fanMode) {
    float std = gain / sqrtf((float)fanMode);
    return randomNormal(0.0f, std);
}

float kaimingUniform(float gain, size_t fanMode, rngStream_t *rs) {
    float limit = gain * sqrtf(3.0f / (float)fanMode);
    return randomUniform(-limit, limit);
}

float xavierNormal(float gain, size_t fanIn, size_t fanOut) {
    float std = gain * sqrtf(2.0f / (float)(fanIn + fanOut));
    return randomNormal(0.0f, std);
}

float xavierUniform(float gain, size_t fanIn, size_t fanOut, rngStream_t *rs) {
    float limit = gain * sqrtf(6.0f / (float)(fanIn + fanOut));
    return randomUniform(-limit, limit);
}
