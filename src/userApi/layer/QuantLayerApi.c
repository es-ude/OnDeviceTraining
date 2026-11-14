#define SOURCE_FILE "QUANT_LAYER_API"

#include <stdbool.h>
#include <stdlib.h>

#include "Common.h"
#include "Layer.h"
#include "LayerQuant.h"
#include "QuantLayerApi.h"
#include "QuantizationLayer.h"
#include "StorageApi.h"
#include "TensorApi.h"

static void validateLayerQuantForQuantLayer(layerQuant_t *lq) {
    if (lq == NULL) {
        PRINT_ERROR("quantLayerInit: lq pointer is NULL");
        exit(1);
    }
    if (lq->outputQ == NULL) {
        PRINT_ERROR("quantLayerInit: layerQuant.outputQ must be set");
        exit(1);
    }
    if (lq->propLossQ == NULL) {
        PRINT_ERROR("quantLayerInit: layerQuant.propLossQ must be set");
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

    cfg->outputQ = lq->outputQ;
    cfg->propLossQ = lq->propLossQ;
    cfg->ownsQuantizations = false;
    return layer;
}

layer_t *quantLayerInitOwning(layerQuant_t *lq) {
    validateLayerQuantForQuantLayer(lq);
    layer_t *layer = quantLayerInitCommon();
    quantizationConfig_t *cfg = layer->config->quantization;

    /* Owns two independent quantization copies, even when the source pointers alias. */
    cfg->outputQ = deepCopyQuantization(lq->outputQ);
    cfg->propLossQ = deepCopyQuantization(lq->propLossQ);
    cfg->ownsQuantizations = true;
    return layer;
}

void freeQuantLayer(layer_t *quantLayer) {
    if (quantLayer == NULL) {
        return;
    }
    quantizationConfig_t *cfg = quantLayer->config->quantization;

    if (cfg->ownsQuantizations) {
        freeQuantization(cfg->outputQ);
        freeQuantization(cfg->propLossQ);
    }

    freeReservedMemory(cfg);
    freeReservedMemory(quantLayer->config);
    freeReservedMemory(quantLayer);
}
