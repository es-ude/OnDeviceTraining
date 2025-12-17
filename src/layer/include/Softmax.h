#ifndef ENV5_RUNTIME_SOFTMAX_H
#define ENV5_RUNTIME_SOFTMAX_H

#include "Layer.h"

typedef struct softmaxConfig
{
    quantization_t* forwardQ;
    quantization_t* backwardQ;
} softmaxConfig_t;

void softmaxInitConfig(softmaxConfig_t* softmaxConfig, quantization_t* forwardQ, quantization_t* backwardQ);

void softmaxInitLayer(layerConfig_t* softmaxConfig, layer_t* softmaxLayer);

void softmaxForward(layer_t* softmaxLayer, tensor_t* input, tensor_t* output);

void softmaxBackward(layer_t* softmaxLayer, tensor_t* input, tensor_t* loss, tensor_t* propLoss);

void softmaxCalcOutputShape(layer_t *softmaxLayer, shape_t *inputShape, shape_t *outputShape);

#endif // ENV5_RUNTIME_SOFTMAX_H
