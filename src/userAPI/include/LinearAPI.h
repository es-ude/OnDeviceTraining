#ifndef ODT_LINEAR_H
#define ODT_LINEAR_H

#include "Layer.h"

layer_t* linearLayerInit(parameter_t* weights, parameter_t* bias, quantization_t* forwardQ,
                         quantization_t* weightGradsQ, quantization_t* biasGradsQ, quantization_t* propLossQ);

layer_t* linearLayerInitNonTrainable(tensor_t* weights, tensor_t* bias, quantization_t* forwardQ);

void freeLinearLayer(layer_t* linearLayer);

#endif //ODT_LINEAR_H
