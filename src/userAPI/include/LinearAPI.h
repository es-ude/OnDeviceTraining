#ifndef ODT_LINEAR_H
#define ODT_LINEAR_H

#include "Layer.h"

layer_t* linearLayerInit(parameter_t* weights, parameter_t* bias, layerQType_t layerQType, qtype_t inputQType,
                         quantization_t* outputQ);

void freeLinearLayer(layer_t* linearLayer);

#endif //ODT_LINEAR_H
