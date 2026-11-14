#define SOURCE_FILE "LAYER_WEIGHTS_API"

#include "LayerWeightsApi.h"
#include "BatchNorm1d.h"
#include "Common.h"
#include "Conv1d.h"
#include "Conv1dTransposed.h"
#include "GroupNorm.h"
#include "LayerNorm.h"
#include "LayerWeightsInternal.h"
#include "Linear.h"
#include "Tensor.h"
#include "TensorApi.h"
#include <stdlib.h>

layerLoadTargets_t layerLoadTargets(layer_t *layer) {
    layerLoadTargets_t targets = {.weights = NULL, .bias = NULL};
    switch (layer->type) {
    case LINEAR:
        targets.weights = layer->config->linear->weights;
        targets.bias = layer->config->linear->bias;
        break;
    case CONV1D:
        targets.weights = layer->config->conv1d->weights;
        targets.bias = layer->config->conv1d->bias;
        break;
    case CONV1D_TRANSPOSED:
        targets.weights = layer->config->conv1dTransposed->weights;
        targets.bias = layer->config->conv1dTransposed->bias;
        break;
    case LAYERNORM:
        targets.weights = layer->config->layerNorm->gamma;
        targets.bias = layer->config->layerNorm->beta;
        break;
    case GROUPNORM:
        targets.weights = layer->config->groupNorm->gamma;
        targets.bias = layer->config->groupNorm->beta;
        break;
    case BATCHNORM1D:
        if (layer->config->batchNorm1d->affine) {
            targets.weights = layer->config->batchNorm1d->gamma;
            targets.bias = layer->config->batchNorm1d->beta;
        }
        break;
    case RELU:
    case SOFTMAX:
    case FLATTEN:
    case MAXPOOL1D:
    case AVGPOOL1D:
    case ADAPTIVE_AVGPOOL1D:
    case DROPOUT:
    case QUANTIZATION:
    case ADD:
    case LAYER_TYPE_COUNT:
        break;
    }
    return targets;
}

const char *layerLoadTargetsProblem(layerLoadTargets_t targets, const float *weightData,
                                    const float *biasData) {
    if (weightData == NULL) {
        return "weightData is NULL";
    }
    if (targets.bias != NULL && biasData == NULL) {
        return "layer has a bias/beta but biasData is NULL";
    }
    if (targets.bias == NULL && biasData != NULL) {
        return "layer has no bias but biasData is non-NULL";
    }
    return NULL;
}

static void fillParameter(parameter_t *parameter, const float *data) {
    tensorFillFromFloatBuffer(parameter->param, data,
                              calcNumberOfElementsByTensor(parameter->param));
}

void layerLoadTargetsFill(layerLoadTargets_t targets, const float *weightData,
                          const float *biasData) {
    fillParameter(targets.weights, weightData);
    if (targets.bias != NULL) {
        fillParameter(targets.bias, biasData);
    }
}

void layerLoadWeights(layer_t *layer, float *weightData, float *biasData) {
    if (layer == NULL) {
        PRINT_ERROR("layerLoadWeights: layer is NULL");
        exit(1);
    }
    layerLoadTargets_t targets = layerLoadTargets(layer);
    if (targets.weights == NULL) {
        PRINT_ERROR("layerLoadWeights: layer type %d has no parameters to load", (int)layer->type);
        exit(1);
    }
    const char *problem = layerLoadTargetsProblem(targets, weightData, biasData);
    if (problem != NULL) {
        PRINT_ERROR("layerLoadWeights: layer type %d: %s", (int)layer->type, problem);
        exit(1);
    }
    layerLoadTargetsFill(targets, weightData, biasData);
}
