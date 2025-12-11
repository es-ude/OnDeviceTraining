#include <stdlib.h>

#include "ReluAPI.h"

layer_t *reluLayerInit(layerQType_t layerQType, qtype_t inputQType, quantization_t* outputQ) {
    layer_t *reluLayer = calloc(1, sizeof(layer_t));

    reluLayer->type = RELU;
    reluLayer->qType = layerQType;
    reluLayer->config = NULL;
    reluLayer->inputQType = inputQType;
    reluLayer->outputQ = outputQ;

    return reluLayer;
}
