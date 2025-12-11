#include <math.h>

#include "Layer.h"
#include "Linear.h"
#include "Relu.h"

layerFunctions_t layerFunctions[] = {
    [LINEAR] = {linearForward, linearBackward, linearCalcOutputShape},
    [RELU] = {reluForward, reluBackward, reluCalcOutputShape},
    [CONV1D] = {NULL, NULL, NULL},
    [SOFTMAX] = {NULL, NULL, NULL}
};

void initLayer(layer_t *layer, layerType_t type, layerConfig_t* config, layerQType_t qType, qtype_t inputQType, quantization_t* outputQ) {
    layer->type = type;
    layer->config = config;
    layer->qType = qType;
    layer->inputQType = inputQType;
    layer->outputQ = outputQ;
}

size_t calcBytesOutputData(quantization_t *outputQ, size_t numberOfOutputs) {
    switch (outputQ->type) {
    case FLOAT32:
        return numberOfOutputs * sizeof(float);
    case ASYM:
        size_t bitsPerElement = calcBitsPerElement(outputQ);
        return ceilf((float)(bitsPerElement * numberOfOutputs / 8));
    default:
        return 0;
    }
}