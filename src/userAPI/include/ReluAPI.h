#ifndef ODT_RELU_H
#define ODT_RELU_H

#include "Layer.h"

layer_t* reluLayerInit(layerQType_t layerQType, qtype_t inputQType, quantization_t* outputQ);

void freeReluLayer(layer_t* reluLayer);

#endif //ODT_RELU_H
