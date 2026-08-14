#define SOURCE_FILE "RELU_API"

#include <stdbool.h>
#include <stdlib.h> /* exit */

#include "ArithmeticType.h"
#include "Common.h" /* PRINT_ERROR */
#include "LayerQuant.h"
#include "Relu.h"
#include "ReluApi.h"
#include "StorageApi.h"
#include "TensorApi.h"

/* ============================================================================
 * New factory API — layerQuant_t profile (PR 1).
 * ========================================================================== */

static void validateLayerQuantForRelu(layerQuant_t *lq) {
    if (lq == NULL) {
        PRINT_ERROR("reluLayerInit: lq pointer is NULL");
        exit(1);
    }
    if (lq->outputQ == NULL) {
        PRINT_ERROR("reluLayerInit: layerQuant.outputQ must be set");
        exit(1);
    }
    if (lq->propLossQ == NULL) {
        PRINT_ERROR("reluLayerInit: layerQuant.propLossQ must be set");
        exit(1);
    }
}

layer_t *reluLayerInit(layerQuant_t *lq) {
    validateLayerQuantForRelu(lq);

    layer_t *layer = reserveMemory(sizeof(layer_t));
    layer->type = RELU;

    layerConfig_t *layerCfg = reserveMemory(sizeof(layerConfig_t));
    reluConfig_t *cfg = reserveMemory(sizeof(reluConfig_t));
    layerCfg->relu = cfg;
    layer->config = layerCfg;

    cfg->forwardMath = lq->forwardMath;
    cfg->propLossMath = lq->propLossMath;
    cfg->outputQ = lq->outputQ;
    cfg->propLossQ = lq->propLossQ;
    cfg->ownsQuantizations = false;

    return layer;
}

layer_t *reluLayerInitOwning(layerQuant_t *lq) {
    validateLayerQuantForRelu(lq);

    layer_t *layer = reserveMemory(sizeof(layer_t));
    layer->type = RELU;

    layerConfig_t *layerCfg = reserveMemory(sizeof(layerConfig_t));
    reluConfig_t *cfg = reserveMemory(sizeof(reluConfig_t));
    layerCfg->relu = cfg;
    layer->config = layerCfg;

    cfg->forwardMath = lq->forwardMath;
    cfg->propLossMath = lq->propLossMath;
    cfg->outputQ = deepCopyQuantization(lq->outputQ);
    cfg->propLossQ = deepCopyQuantization(lq->propLossQ);
    cfg->ownsQuantizations = true;

    return layer;
}

void freeReluLayer(layer_t *reluLayer) {
    if (reluLayer == NULL) {
        return;
    }
    reluConfig_t *cfg = reluLayer->config->relu;

    if (cfg->ownsQuantizations) {
        if (cfg->outputQ != NULL) {
            freeQuantization(cfg->outputQ);
        }
        if (cfg->propLossQ != NULL && cfg->propLossQ != cfg->outputQ) {
            freeQuantization(cfg->propLossQ);
        }
    }
    freeReservedMemory(cfg);
    freeReservedMemory(reluLayer->config);
    freeReservedMemory(reluLayer);
}
