#define SOURCE_FILE "SOFTMAX_API"

#include <stdbool.h>
#include <stdlib.h>

#include "ArithmeticType.h"
#include "Common.h"
#include "LayerQuant.h"
#include "Softmax.h"
#include "SoftmaxApi.h"
#include "StorageApi.h"
#include "TensorApi.h"

/* ============================================================================
 * New factory API — layerQuant_t profile (PR 2).
 * ========================================================================== */

static void validateLayerQuantForSoftmax(layerQuant_t *lq) {
    if (lq == NULL) {
        PRINT_ERROR("softmaxLayerInit: lq pointer is NULL");
        exit(1);
    }
    if (lq->outputQ == NULL) {
        PRINT_ERROR("softmaxLayerInit: layerQuant.outputQ must be set");
        exit(1);
    }
    if (lq->propLossQ == NULL) {
        PRINT_ERROR("softmaxLayerInit: layerQuant.propLossQ must be set");
        exit(1);
    }
}

layer_t *softmaxLayerInit(layerQuant_t *lq) {
    validateLayerQuantForSoftmax(lq);

    layer_t *layer = reserveMemory(sizeof(layer_t));
    layer->type = SOFTMAX;

    layerConfig_t *layerCfg = reserveMemory(sizeof(layerConfig_t));
    softmaxConfig_t *cfg = reserveMemory(sizeof(softmaxConfig_t));
    layerCfg->softmax = cfg;
    layer->config = layerCfg;

    cfg->forwardMath = lq->forwardMath;
    cfg->propLossMath = lq->propLossMath;
    cfg->outputQ = lq->outputQ;
    cfg->propLossQ = lq->propLossQ;
    cfg->ownsQuantizations = false;

    return layer;
}

layer_t *softmaxLayerInitOwning(layerQuant_t *lq) {
    validateLayerQuantForSoftmax(lq);

    layer_t *layer = reserveMemory(sizeof(layer_t));
    layer->type = SOFTMAX;

    layerConfig_t *layerCfg = reserveMemory(sizeof(layerConfig_t));
    softmaxConfig_t *cfg = reserveMemory(sizeof(softmaxConfig_t));
    layerCfg->softmax = cfg;
    layer->config = layerCfg;

    cfg->forwardMath = lq->forwardMath;
    cfg->propLossMath = lq->propLossMath;
    cfg->outputQ = deepCopyQuantization(lq->outputQ);
    cfg->propLossQ = deepCopyQuantization(lq->propLossQ);
    cfg->ownsQuantizations = true;

    return layer;
}

void freeSoftmaxLayer(layer_t *softmaxLayer) {
    if (softmaxLayer == NULL) {
        return;
    }
    softmaxConfig_t *cfg = softmaxLayer->config->softmax;

    if (cfg->ownsQuantizations) {
        if (cfg->outputQ != NULL) {
            freeQuantization(cfg->outputQ);
        }
        if (cfg->propLossQ != NULL && cfg->propLossQ != cfg->outputQ) {
            freeQuantization(cfg->propLossQ);
        }
    }
    freeReservedMemory(cfg);
    freeReservedMemory(softmaxLayer->config);
    freeReservedMemory(softmaxLayer);
}
