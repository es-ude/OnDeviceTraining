#define SOURCE_FILE "CONV1D_API"

#include "Conv1dApi.h"
#include "Conv1d.h"
#include "Layer.h"
#include "StorageApi.h"
#include "TensorApi.h"

#include <stdio.h>

layer_t *conv1dLayerInit(parameter_t *weights, parameter_t *bias, kernel_t *kernel,
                         quantization_t *forwardQ, quantization_t *weightGradQ,
                         quantization_t *biasGradQ, quantization_t *propLossQ) {
    layer_t *conv1dLayer = reserveMemory(sizeof(layer_t));
    layerConfig_t *layerConfig = reserveMemory(sizeof(layerConfig_t));
    conv1dConfig_t *conv1dConfig = reserveMemory(sizeof(conv1dConfig_t));

    initConv1dConfigWithWeightsAndBias(conv1dConfig, kernel, weights, bias, forwardQ, weightGradQ,
                                       biasGradQ, propLossQ);

    conv1dLayer->type = CONV1D;
    layerConfig->conv1d = conv1dConfig;
    conv1dLayer->config = layerConfig;

    return conv1dLayer;
}

void freeConv1dLayer(layer_t *conv1dLayer) {
    conv1dConfig_t *conv1dConfig = conv1dLayer->config->conv1d;

    freeParameter(conv1dConfig->weights);
    if (conv1dConfig->bias) {
        freeParameter(conv1dConfig->bias);
    }

    freeQuantization(conv1dConfig->forwardQ);
    freeQuantization(conv1dConfig->weightGradQ);
    freeQuantization(conv1dConfig->biasGradQ);
    freeQuantization(conv1dConfig->propLossQ);
    freeReservedMemory(conv1dConfig);
    freeReservedMemory(conv1dLayer);
}
