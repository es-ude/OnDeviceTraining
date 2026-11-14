#define SOURCE_FILE "GROUPNORM_API"

#include <stdlib.h>

#include "GroupNormApi.h"

#include "ArithmeticType.h"
#include "Common.h"
#include "GroupNorm.h"
#include "Layer.h"
#include "LayerQuant.h"
#include "NormParamFactory.h"
#include "QuantizationApi.h"
#include "StorageApi.h"
#include "Tensor.h"
#include "TensorApi.h"

#define GROUPNORM_DEFAULT_EPS 1e-5f

static void validateGroupNormInit(groupNormInit_t *init) {
    if (init == NULL) {
        PRINT_ERROR("groupNormLayerInit: init pointer is NULL");
        exit(1);
    }
    if (init->numChannels == 0) {
        PRINT_ERROR("groupNormLayerInit: numChannels must be > 0 (got 0)");
        exit(1);
    }
    if (init->eps < 0.0f) {
        PRINT_ERROR("groupNormLayerInit: eps must be >= 0 (got %f)", (double)init->eps);
        exit(1);
    }
}

/* The part both factories share; each then wires the maths, the wire configs and the
 * grad modes into `*outCfg` and validates the result. */
static layer_t *groupNormLayerInitCommon(groupNormInit_t *init, layerQuant_t *lq,
                                         groupNormConfig_t **outCfg) {
    validateGroupNormInit(init);
    normValidateLayerQuant(lq, "groupNormLayerInit");
    bool trainable = resolveTrainable(init->trainable, "groupNormLayerInit");

    float eps = (init->eps == 0.0f) ? GROUPNORM_DEFAULT_EPS : init->eps;

    layer_t *layer = reserveMemory(sizeof(layer_t));
    layer->type = GROUPNORM;

    layerConfig_t *layerCfg = reserveMemory(sizeof(layerConfig_t));
    groupNormConfig_t *cfg = reserveMemory(sizeof(groupNormConfig_t));
    layerCfg->groupNorm = cfg;
    layer->config = layerCfg;

    parameter_t *gamma;
    parameter_t *beta;
    normAllocGammaBeta(lq, (size_t[]){init->numChannels}, 1, trainable, "groupNormLayerInit",
                       &gamma, &beta);

    initGroupNormConfig(cfg, gamma, beta, init->numGroups, init->numChannels, eps, NULL, NULL);
    cfg->frozen = !trainable;

    *outCfg = cfg;
    return layer;
}

layer_t *groupNormLayerInit(groupNormInit_t *init, layerQuant_t *lq) {
    groupNormConfig_t *cfg;
    layer_t *layer = groupNormLayerInitCommon(init, lq, &cfg);

    cfg->forwardMath = lq->forwardMath;
    cfg->propLossMath = lq->propLossMath;
    cfg->outputQ = lq->outputQ;
    cfg->propLossQ = lq->propLossQ;
    cfg->weightGradAccMode = lq->weightGradAccMode;
    cfg->biasGradAccMode = lq->biasGradAccMode;
    cfg->ownsQuantizations = false;
    groupNormValidateConfig(cfg, "groupNormLayerInit");

    return layer;
}

layer_t *groupNormLayerInitOwning(groupNormInit_t *init, layerQuant_t *lq) {
    groupNormConfig_t *cfg;
    layer_t *layer = groupNormLayerInitCommon(init, lq, &cfg);

    cfg->forwardMath = lq->forwardMath;
    cfg->propLossMath = lq->propLossMath;
    cfg->outputQ = deepCopyQuantization(lq->outputQ);
    cfg->propLossQ = deepCopyQuantization(lq->propLossQ);
    cfg->weightGradAccMode = lq->weightGradAccMode;
    cfg->biasGradAccMode = lq->biasGradAccMode;
    cfg->ownsQuantizations = true;
    groupNormValidateConfig(cfg, "groupNormLayerInitOwning");

    return layer;
}

void freeGroupNormLayer(layer_t *layer) {
    if (layer == NULL) {
        return;
    }
    groupNormConfig_t *cfg = layer->config->groupNorm;

    /* The param tensors own their storage quantization. */
    freeParameter(cfg->gamma);
    freeParameter(cfg->beta);
    /* The owning factory deep-copies outputQ and propLossQ into two distinct allocations. */
    if (cfg->ownsQuantizations) {
        freeQuantization(cfg->outputQ);
        freeQuantization(cfg->propLossQ);
    }

    freeReservedMemory(cfg);
    freeReservedMemory(layer->config);
    freeReservedMemory(layer);
}
