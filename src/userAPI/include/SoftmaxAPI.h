#ifndef SOFTMAXAPI_H
#define SOFTMAXAPI_H

#include "Tensor.h"
#include "Layer.h"

layer_t *softmaxLayerInit(quantization_t *forwardQ, quantization_t *backwardQ);

#endif //SOFTMAXAPI_H
