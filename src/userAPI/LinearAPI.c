#include <stdlib.h>

#include "Layer.h"
#include "Tensor.h"
#include "LinearAPI.h"

#include <stdio.h>

layer_t *linearLayerInit(parameter_t *weights, parameter_t *bias, layerQType_t layerQType, qtype_t inputQType, quantization_t *outputQ) {
    layer_t *linearLayer = calloc(1, sizeof(layer_t));

    linearLayer->type = LINEAR;
    linearLayer->qType = layerQType;
    linearLayer->outputQ = outputQ;
    linearLayer->inputQType = inputQType;

    layerConfig_t *layerConfig = calloc(1, sizeof(layerConfig_t));
    linearConfig_t *linearConfig = calloc(1, sizeof(linearConfig_t));
    layerConfig->linear = linearConfig;

    linearConfig->weights = weights;
    linearConfig->bias = bias;

    linearLayer->config = layerConfig;

    return  linearLayer;
}

layer_t *linearLayerInitNonTrainable(tensor_t *weights, tensor_t *bias, layerQType_t layerQType, qtype_t inputQType, quantization_t *outputQ) {
    layer_t *linearLayer = calloc(1, sizeof(layer_t));

    linearLayer->type = LINEAR;
    linearLayer->qType = layerQType;
    linearLayer->outputQ = outputQ;
    linearLayer->inputQType = inputQType;

    layerConfig_t *layerConfig = calloc(1, sizeof(layerConfig_t));
    linearConfig_t *linearConfig = calloc(1, sizeof(linearConfig_t));
    layerConfig->linear = linearConfig;

    linearConfig->weights->param = weights;
    linearConfig->weights->grad = NULL;
    linearConfig->bias->param = bias;
    linearConfig->bias->grad = NULL;

    linearLayer->config = layerConfig;

    return  linearLayer;
}

