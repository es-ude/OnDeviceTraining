#define SOURCE_FILE "LAYERNORM_API"

#include <stdlib.h>

#include "LayerNormApi.h"

#include "ArithmeticType.h"
#include "Common.h"
#include "Layer.h"
#include "LayerNorm.h"
#include "LayerQuant.h"
#include "NormParamFactory.h"
#include "QuantizationApi.h"
#include "StorageApi.h"
#include "Tensor.h"
#include "TensorApi.h"

#define LAYERNORM_DEFAULT_EPS 1e-5f

static void validateLayerNormInit(layerNormInit_t *init) {
    if (init == NULL) {
        PRINT_ERROR("layerNormLayerInit: init pointer is NULL");
        exit(1);
    }
    if (init->normalizedShape == NULL) {
        PRINT_ERROR("layerNormLayerInit: normalizedShape is NULL");
        exit(1);
    }
    if (init->numNormDims == 0) {
        PRINT_ERROR("layerNormLayerInit: numNormDims must be > 0 (got 0)");
        exit(1);
    }
    for (size_t d = 0; d < init->numNormDims; d++) {
        if (init->normalizedShape[d] == 0) {
            PRINT_ERROR("layerNormLayerInit: normalizedShape[%zu] must be > 0", d);
            exit(1);
        }
    }
    if (init->eps < 0.0f) {
        PRINT_ERROR("layerNormLayerInit: eps must be >= 0 (got %f)", (double)init->eps);
        exit(1);
    }
}

/* The part both factories share; each then wires the maths, the wire configs and the
 * grad modes into `*outCfg` and validates the result. */
static layer_t *layerNormLayerInitCommon(layerNormInit_t *init, layerQuant_t *lq,
                                         layerNormConfig_t **outCfg) {
    validateLayerNormInit(init);
    normValidateLayerQuant(lq, "layerNormLayerInit");
    bool trainable = resolveTrainable(init->trainable, "layerNormLayerInit");

    float eps = (init->eps == 0.0f) ? LAYERNORM_DEFAULT_EPS : init->eps;

    layer_t *layer = reserveMemory(sizeof(layer_t));
    layer->type = LAYERNORM;

    layerConfig_t *layerCfg = reserveMemory(sizeof(layerConfig_t));
    layerNormConfig_t *cfg = reserveMemory(sizeof(layerNormConfig_t));
    layerCfg->layerNorm = cfg;
    layer->config = layerCfg;

    parameter_t *gamma;
    parameter_t *beta;
    normAllocGammaBeta(lq, init->normalizedShape, init->numNormDims, trainable,
                       "layerNormLayerInit", &gamma, &beta);

    /* The caller may free its normalizedShape array. */
    size_t *normShapeCopy = reserveMemory(init->numNormDims * sizeof(size_t));
    for (size_t d = 0; d < init->numNormDims; d++) {
        normShapeCopy[d] = init->normalizedShape[d];
    }

    initLayerNormConfig(cfg, gamma, beta, normShapeCopy, init->numNormDims, eps, NULL, NULL);
    cfg->frozen = !trainable;

    *outCfg = cfg;
    return layer;
}

layer_t *layerNormLayerInit(layerNormInit_t *init, layerQuant_t *lq) {
    layerNormConfig_t *cfg;
    layer_t *layer = layerNormLayerInitCommon(init, lq, &cfg);

    cfg->forwardMath = lq->forwardMath;
    cfg->propLossMath = lq->propLossMath;
    cfg->outputQ = lq->outputQ;
    cfg->propLossQ = lq->propLossQ;
    cfg->weightGradAccMode = lq->weightGradAccMode;
    cfg->biasGradAccMode = lq->biasGradAccMode;
    cfg->ownsQuantizations = false;
    layerNormValidateConfig(cfg, "layerNormLayerInit");

    return layer;
}

layer_t *layerNormLayerInitOwning(layerNormInit_t *init, layerQuant_t *lq) {
    layerNormConfig_t *cfg;
    layer_t *layer = layerNormLayerInitCommon(init, lq, &cfg);

    cfg->forwardMath = lq->forwardMath;
    cfg->propLossMath = lq->propLossMath;
    cfg->outputQ = deepCopyQuantization(lq->outputQ);
    cfg->propLossQ = deepCopyQuantization(lq->propLossQ);
    cfg->weightGradAccMode = lq->weightGradAccMode;
    cfg->biasGradAccMode = lq->biasGradAccMode;
    cfg->ownsQuantizations = true;
    layerNormValidateConfig(cfg, "layerNormLayerInitOwning");

    return layer;
}

void freeLayerNormLayer(layer_t *layer) {
    if (layer == NULL) {
        return;
    }
    layerNormConfig_t *cfg = layer->config->layerNorm;

    /* The param tensors own their storage quantization. */
    freeParameter(cfg->gamma);
    freeParameter(cfg->beta);
    freeReservedMemory(cfg->normalizedShape);
    /* The owning factory deep-copies outputQ and propLossQ into two distinct allocations. */
    if (cfg->ownsQuantizations) {
        freeQuantization(cfg->outputQ);
        freeQuantization(cfg->propLossQ);
    }

    freeReservedMemory(cfg);
    freeReservedMemory(layer->config);
    freeReservedMemory(layer);
}
