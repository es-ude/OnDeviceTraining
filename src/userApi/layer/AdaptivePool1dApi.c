#define SOURCE_FILE "ADAPTIVE_POOL1D_API"

#include <stdbool.h>
#include <stdlib.h>

#include "AdaptiveAvgPool1d.h"
#include "AdaptivePool1dApi.h"
#include "ArithmeticType.h"
#include "Common.h"
#include "Layer.h"
#include "LayerQuant.h"
#include "QuantizationApi.h"
#include "StorageApi.h"
#include "TensorApi.h"

static void validateInit(adaptiveAvgPool1dInit_t *init) {
    if (init == NULL) {
        PRINT_ERROR("adaptiveAvgPool1dLayerInit: init pointer is NULL");
        exit(1);
    }
    if (init->outputSize == 0) {
        PRINT_ERROR("adaptiveAvgPool1dLayerInit: outputSize must be > 0");
        exit(1);
    }
}

static void validateLayerQuant(layerQuant_t *lq) {
    if (lq == NULL) {
        PRINT_ERROR("adaptiveAvgPool1dLayerInit: lq pointer is NULL");
        exit(1);
    }
    if (lq->outputQ == NULL) {
        PRINT_ERROR("adaptiveAvgPool1dLayerInit: layerQuant.outputQ must be set");
        exit(1);
    }
    if (lq->propLossQ == NULL) {
        PRINT_ERROR("adaptiveAvgPool1dLayerInit: layerQuant.propLossQ must be set");
        exit(1);
    }
}

static layer_t *buildSkeleton(adaptiveAvgPool1dInit_t *init) {
    layer_t *layer = reserveMemory(sizeof(layer_t));
    layer->type = ADAPTIVE_AVGPOOL1D;
    layerConfig_t *layerCfg = reserveMemory(sizeof(layerConfig_t));
    adaptiveAvgPool1dConfig_t *cfg = reserveMemory(sizeof(adaptiveAvgPool1dConfig_t));
    layerCfg->adaptiveAvgPool1d = cfg;
    layer->config = layerCfg;

    cfg->outputSize = init->outputSize;
    return layer;
}

layer_t *adaptiveAvgPool1dLayerInit(adaptiveAvgPool1dInit_t *init, layerQuant_t *lq) {
    validateInit(init);
    validateLayerQuant(lq);

    layer_t *layer = buildSkeleton(init);
    adaptiveAvgPool1dConfig_t *cfg = layer->config->adaptiveAvgPool1d;
    cfg->forwardMath = lq->forwardMath;
    cfg->propLossMath = lq->propLossMath;
    cfg->outputQ = lq->outputQ;
    cfg->propLossQ = lq->propLossQ;
    cfg->ownsQuantizations = false;
    return layer;
}

layer_t *adaptiveAvgPool1dLayerInitOwning(adaptiveAvgPool1dInit_t *init, layerQuant_t *lq) {
    validateInit(init);
    validateLayerQuant(lq);

    layer_t *layer = buildSkeleton(init);
    adaptiveAvgPool1dConfig_t *cfg = layer->config->adaptiveAvgPool1d;
    cfg->forwardMath = lq->forwardMath;
    cfg->propLossMath = lq->propLossMath;
    cfg->outputQ = deepCopyQuantization(lq->outputQ);
    cfg->propLossQ = deepCopyQuantization(lq->propLossQ);
    cfg->ownsQuantizations = true;
    return layer;
}

void freeAdaptiveAvgPool1dLayer(layer_t *layer) {
    if (layer == NULL) {
        return;
    }
    adaptiveAvgPool1dConfig_t *cfg = layer->config->adaptiveAvgPool1d;

    if (cfg->ownsQuantizations) {
        if (cfg->outputQ != NULL) {
            freeQuantization(cfg->outputQ);
        }
        if (cfg->propLossQ != NULL && cfg->propLossQ != cfg->outputQ) {
            freeQuantization(cfg->propLossQ);
        }
    }

    freeReservedMemory(cfg);
    freeReservedMemory(layer->config);
    freeReservedMemory(layer);
}
