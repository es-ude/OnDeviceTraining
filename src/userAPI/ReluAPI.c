#include "StorageAPI.h"
#include "ReluAPI.h"
#include "Relu.h"

layer_t *reluLayerInit(quantization_t *forwardQ, quantization_t *backwardQ) {
    layer_t *reluLayer = *reserveMemory(sizeof(layer_t));

    reluLayer->type = RELU;

    layerConfig_t *reluConfig = *reserveMemory(sizeof(layerConfig_t));

    reluConfig_t *reluCfg = *reserveMemory(sizeof(reluConfig_t));
    reluConfig->relu = reluCfg;
    reluCfg->forwardQ = forwardQ;
    reluCfg->backwardQ = backwardQ;

    reluLayer->config = reluConfig;

    return reluLayer;
}

void freeReluLayer(layer_t *reluLayer) {
    freeReservedMemory(reluLayer->config);
    freeReservedMemory(reluLayer);
}
