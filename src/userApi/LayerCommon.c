#define SOURCE_FILE "LAYER_COMMON"

#include <stdlib.h>

#include "Common.h"
#include "Distributions.h"
#include "LayerCommon.h"
#include "TensorApi.h"

/* PyTorch's default weight/bias init draws from uniform(+/- 1/sqrt(fan_in)).
 * kaimingUniform(gain, fan) = uniform(+/- gain*sqrt(3/fan)), so the gain that
 * reproduces the 1/sqrt(fan_in) bound is sqrt(1/3): gain*sqrt(3/fan) =
 * sqrt(1/3)*sqrt(3/fan) = sqrt(1/fan) = 1/sqrt(fan). This is exactly PyTorch's
 * kaiming_uniform_(a=sqrt(5)) default for Linear/Conv weights. */
#define INIT_DEFAULT_GAIN 0.57735026919f         /* sqrt(1/3) */
#define INIT_KAIMING_DEFAULT_GAIN 1.41421356237f /* sqrt(2), He */
#define INIT_XAVIER_DEFAULT_GAIN 1.0f

static void requireFloat32(const tensor_t *t, const char *what) {
    if (t->quantization->type != FLOAT32) {
        PRINT_ERROR("%s: tensor init currently requires FLOAT32 storage (got type %d)", what,
                    (int)t->quantization->type);
        exit(1);
    }
}

void initWeightTensor(tensor_t *weight, weightInit_t cfg, size_t fanIn, size_t fanOut) {
    requireFloat32(weight, "initWeightTensor");

    distribution_t dist;
    switch (cfg.scheme) {
    case INIT_DEFAULT:
        dist = (distribution_t){
            .type = KAIMING_UNIFORM,
            .params.kaiming = {.gain = INIT_DEFAULT_GAIN, .fanMode = fanIn},
        };
        break;
    case INIT_KAIMING_UNIFORM:
        dist = (distribution_t){
            .type = KAIMING_UNIFORM,
            .params.kaiming = {.gain = cfg.gain != 0.0f ? cfg.gain : INIT_KAIMING_DEFAULT_GAIN,
                               .fanMode = fanIn},
        };
        break;
    case INIT_XAVIER_UNIFORM:
        dist = (distribution_t){
            .type = XAVIER_UNIFORM,
            .params.xavier = {.gain = cfg.gain != 0.0f ? cfg.gain : INIT_XAVIER_DEFAULT_GAIN,
                              .fanIn = fanIn,
                              .fanOut = fanOut},
        };
        break;
    default:
        PRINT_ERROR("initWeightTensor: invalid init scheme (got %d)", (int)cfg.scheme);
        exit(1);
    }

    initDistribution(weight, &dist);
}

void initBiasTensor(tensor_t *bias, size_t fanIn) {
    requireFloat32(bias, "initBiasTensor");

    /* PyTorch bias default: uniform(+/- 1/sqrt(fan_in)), independent of the
     * weight scheme. Reuse kaimingUniform(sqrt(1/3), fan_in) = that bound. */
    distribution_t dist = {
        .type = KAIMING_UNIFORM,
        .params.kaiming = {.gain = INIT_DEFAULT_GAIN, .fanMode = fanIn},
    };
    initDistribution(bias, &dist);
}
