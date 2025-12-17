#ifndef ODT_RELU_H
#define ODT_RELU_H

#include "Layer.h"

layer_t *reluLayerInit(quantization_t *forwardQ, quantization_t *backwardQ);

void freeReluLayer(layer_t* reluLayer);

#endif //ODT_RELU_H
