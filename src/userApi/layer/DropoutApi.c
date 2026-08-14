#define SOURCE_FILE "DROPOUT_API"

#include "DropoutApi.h"

#include "Dropout.h"
#include "Layer.h"
#include "StorageApi.h"
#include "Tensor.h"
#include "TensorApi.h"

layer_t *dropoutLayerInit(float p, tensor_t *mask, quantization_t *forwardQ,
                          quantization_t *backwardQ) {
    layer_t *layer = reserveMemory(sizeof(layer_t));
    layer->type = DROPOUT;

    layerConfig_t *layerCfg = reserveMemory(sizeof(layerConfig_t));
    dropoutConfig_t *cfg = reserveMemory(sizeof(dropoutConfig_t));
    layerCfg->dropout = cfg;
    layer->config = layerCfg;

    initDropoutConfig(cfg, p, mask, forwardQ, backwardQ);

    return layer;
}

void freeDropoutLayer(layer_t *dropoutLayer) {
    if (dropoutLayer == NULL) {
        return;
    }
    dropoutConfig_t *cfg = dropoutLayer->config->dropout;
    if (cfg->ownsQuantizations) {
        if (cfg->outputQ != NULL) {
            freeQuantization(cfg->outputQ);
        }
        if (cfg->propLossQ != NULL && cfg->propLossQ != cfg->outputQ) {
            freeQuantization(cfg->propLossQ);
        }
    }
    freeReservedMemory(cfg);
    freeReservedMemory(dropoutLayer->config);
    freeReservedMemory(dropoutLayer);
}
