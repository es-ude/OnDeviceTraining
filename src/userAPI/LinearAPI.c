#include <stdlib.h>
#include <stdio.h>

#include "Layer.h"
#include "Tensor.h"
#include "StorageAPI.h"
#include "LinearAPI.h"
#include "Linear.h"

layer_t *linearLayerInit(parameter_t *weights, parameter_t *bias, quantization_t *forwardQ, quantization_t *weightGradsQ, quantization_t *biasGradsQ, quantization_t *propLossQ) {
    layer_t *linearLayer = *reserveMemory(sizeof(layer_t));

    linearLayer->type = LINEAR;

    layerConfig_t *layerConfig = *reserveMemory(sizeof(layerConfig_t));
    linearConfig_t *linearConfig = *reserveMemory(sizeof(linearConfig_t));
    layerConfig->linear = linearConfig;

    linearConfig->weights = weights;
    linearConfig->bias = bias;
    linearConfig->forwardQ = forwardQ;
    linearConfig->weightGradQ = weightGradsQ;
    linearConfig->biasGradQ = biasGradsQ;
    linearConfig->propLossQ = propLossQ;

    linearLayer->config = layerConfig;

    return linearLayer;
}

layer_t *linearLayerInitNonTrainable(tensor_t *weights, tensor_t *bias, quantization_t *forwardQ) {
    layer_t *linearLayer = *reserveMemory(sizeof(layer_t));

    linearLayer->type = LINEAR;

    layerConfig_t *layerConfig = *reserveMemory(sizeof(layerConfig_t));
    linearConfig_t *linearConfig = *reserveMemory(sizeof(linearConfig_t));
    layerConfig->linear = linearConfig;

    linearConfig->weights->param = weights;
    linearConfig->weights->grad = NULL;
    linearConfig->bias->param = bias;
    linearConfig->bias->grad = NULL;
    linearConfig->forwardQ = forwardQ;

    linearLayer->config = layerConfig;

    return linearLayer;
}

void freeLinearLayer(layer_t *linearLayer) {
    linearConfig_t *linearConfig = linearLayer->config->linear;
    freeReservedMemory(linearConfig);
    freeReservedMemory(linearLayer);
}
