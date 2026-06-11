#define SOURCE_FILE "QUANT_LAYER_API"

#include <stdbool.h>
#include <stdlib.h>

#include "Common.h"
#include "Layer.h"
#include "LayerQuant.h"
#include "QuantLayerApi.h"
#include "QuantizationLayer.h"
#include "StorageApi.h"

static void validateLayerQuantForQuantLayer(layerQuant_t *lq) {
    if (lq == NULL) {
        PRINT_ERROR("quantLayerInit: lq pointer is NULL");
        exit(1);
    }
    if (lq->forwardMath == NULL) {
        PRINT_ERROR("quantLayerInit: layerQuant.forwardMath must be set");
        exit(1);
    }
    if (lq->backwardMath == NULL) {
        PRINT_ERROR("quantLayerInit: layerQuant.backwardMath must be set");
        exit(1);
    }
}

static layer_t *quantLayerInitCommon(void) {
    layer_t *layer = reserveMemory(sizeof(layer_t));
    layer->type = QUANTIZATION;
    layerConfig_t *layerCfg = reserveMemory(sizeof(layerConfig_t));
    quantizationConfig_t *cfg = reserveMemory(sizeof(quantizationConfig_t));
    layerCfg->quantization = cfg;
    layer->config = layerCfg;
    return layer;
}

layer_t *quantLayerInit(layerQuant_t *lq) {
    validateLayerQuantForQuantLayer(lq);
    layer_t *layer = quantLayerInitCommon();
    quantizationConfig_t *cfg = layer->config->quantization;

    /* Borrowing: store the two math quant pointers verbatim. The caller owns
     * forwardMath/backwardMath and frees them; freeQuantLayer leaves them
     * untouched (ownsQuantizations=false). */
    cfg->forwardQ = lq->forwardMath;
    cfg->backwardQ = lq->backwardMath;
    cfg->ownsQuantizations = false;
    return layer;
}

layer_t *quantLayerInitOwning(layerQuant_t *lq) {
    validateLayerQuantForQuantLayer(lq);
    layer_t *layer = quantLayerInitCommon();
    quantizationConfig_t *cfg = layer->config->quantization;

    /* Owning: deep-copy each math quantization so the caller can drop its
     * pointers immediately. Always two separate copies, even if both lq slots
     * point at the same instance — keeps freeQuantLayer simple. Mirrors
     * linearLayerInitOwning / layerNormLayerInitOwning. */
    cfg->forwardQ = deepCopyQuantization(lq->forwardMath);
    cfg->backwardQ = deepCopyQuantization(lq->backwardMath);
    cfg->ownsQuantizations = true;
    return layer;
}

void freeQuantLayer(layer_t *quantLayer) {
    if (quantLayer == NULL) {
        return;
    }
    quantizationConfig_t *cfg = quantLayer->config->quantization;

    /* Owning-variant only — tear down the two math quantization_t (qConfig +
     * struct). Dedup guard exactly like freeLayerNormLayer: free backwardQ only
     * if it is a distinct allocation from forwardQ. */
    if (cfg->ownsQuantizations) {
        if (cfg->forwardQ != NULL) {
            freeReservedMemory(cfg->forwardQ->qConfig);
            freeReservedMemory(cfg->forwardQ);
        }
        if (cfg->backwardQ != NULL && cfg->backwardQ != cfg->forwardQ) {
            freeReservedMemory(cfg->backwardQ->qConfig);
            freeReservedMemory(cfg->backwardQ);
        }
    }

    freeReservedMemory(cfg);
    freeReservedMemory(quantLayer->config);
    freeReservedMemory(quantLayer);
}
