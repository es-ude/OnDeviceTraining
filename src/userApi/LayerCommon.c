#define SOURCE_FILE "LAYER_COMMON"

#include <stdlib.h>

#include "Common.h"
#include "Distributions.h"
#include "LayerCommon.h"
#include "RNG.h"
#include "TensorApi.h"

/* sqrt(1/3) gives PyTorch's default uniform bound +/-1/sqrt(fanIn). */
#define INIT_DEFAULT_GAIN 0.57735026919f         /* sqrt(1/3) */
#define INIT_KAIMING_DEFAULT_GAIN 1.41421356237f /* sqrt(2), He */
#define INIT_XAVIER_DEFAULT_GAIN 1.0f

static void requireFloat32RandomInit(const quantization_t *storage, const char *slot,
                                     const char *factoryName) {
    if (storage->type != FLOAT32) {
        PRINT_ERROR("%s: random init requires FLOAT32 layerQuant.%s (got type %d) -- "
                    "FLOAT32-init, then requantizeTensorInPlace",
                    factoryName, slot, (int)storage->type);
        exit(1);
    }
}

void validateRandomInit(weightInit_t init, const quantization_t *weightStorage,
                        const quantization_t *biasStorage, const char *factoryName) {
    if (init.scheme != INIT_DEFAULT && init.scheme != INIT_KAIMING_UNIFORM &&
        init.scheme != INIT_XAVIER_UNIFORM) {
        PRINT_ERROR("%s: invalid weightInit.scheme (got %d)", factoryName, (int)init.scheme);
        exit(1);
    }
    requireFloat32RandomInit(weightStorage, "weightStorage", factoryName);
    if (biasStorage != NULL) {
        requireFloat32RandomInit(biasStorage, "biasStorage", factoryName);
    }
}

/* The draws of one parameter: the stream of (PARAM, slot) under the layer's
 * init key; slot 0 is the weight, 1 the bias. Without a key there is no stream,
 * and initDistribution stops at the first draw. */
static rngStream_t *openParamStream(const rngKey_t *initKey, uint32_t slot, rngStream_t *storage) {
    if (initKey == NULL) {
        return NULL;
    }
    *storage = rngOpen(rngFold(*initKey, RNG_TAG_PARAM, slot));
    return storage;
}

void initWeightTensor(tensor_t *weight, weightInit_t cfg, size_t fanIn, size_t fanOut) {
    distribution_t dist = {
        .type = KAIMING_UNIFORM,
        .params.kaiming = {.gain = INIT_DEFAULT_GAIN, .fanMode = fanIn},
    };
    if (cfg.scheme == INIT_KAIMING_UNIFORM) {
        dist.params.kaiming.gain = cfg.gain != 0.0f ? cfg.gain : INIT_KAIMING_DEFAULT_GAIN;
    } else if (cfg.scheme == INIT_XAVIER_UNIFORM) {
        dist = (distribution_t){
            .type = XAVIER_UNIFORM,
            .params.xavier = {.gain = cfg.gain != 0.0f ? cfg.gain : INIT_XAVIER_DEFAULT_GAIN,
                              .fanIn = fanIn,
                              .fanOut = fanOut},
        };
    }

    rngStream_t rs;
    initDistribution(weight, &dist, openParamStream(cfg.key, 0u, &rs));
}

bool resolveTrainable(trainable_t t, const char *factoryName) {
    switch (t) {
    case TRAINABLE_DEFAULT:
    case TRAINABLE_TRUE:
        return true;
    case TRAINABLE_FALSE:
        return false;
    default:
        PRINT_ERROR("%s: invalid trainable value (got %d)", factoryName, (int)t);
        exit(1);
    }
}

void initBiasTensor(tensor_t *bias, size_t fanIn, const rngKey_t *initKey) {
    distribution_t dist = {
        .type = KAIMING_UNIFORM,
        .params.kaiming = {.gain = INIT_DEFAULT_GAIN, .fanMode = fanIn},
    };
    rngStream_t rs;
    initDistribution(bias, &dist, openParamStream(initKey, 1u, &rs));
}
