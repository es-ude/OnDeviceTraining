#define SOURCE_FILE "LAYER"

#include "Layer.h"
#include "AdaptiveAvgPool1d.h"
#include "AvgPool1d.h"
#include "BatchNorm1d.h"
#include "Conv1d.h"
#include "Conv1dTransposed.h"
#include "Dropout.h"
#include "Flatten.h"
#include "GroupNorm.h"
#include "LayerNorm.h"
#include "Linear.h"
#include "MaxPool1d.h"
#include "QuantizationLayer.h"
#include "Relu.h"
#include "Softmax.h"

layerFunctions_t layerFunctions[] = {
    [LINEAR] = {linearForward, linearBackward, linearCalcOutputShape},
    [RELU] = {reluForward, reluBackward, reluCalcOutputShape},
    [CONV1D] = {conv1dForward, conv1dBackward, conv1dCalcOutputShape},
    [CONV1D_TRANSPOSED] = {conv1dTransposedForward, conv1dTransposedBackward,
                           conv1dTransposedCalcOutputShape},
    [MAXPOOL1D] = {maxPool1dForward, maxPool1dBackward, maxPool1dCalcOutputShape},
    [AVGPOOL1D] = {avgPool1dForward, avgPool1dBackward, avgPool1dCalcOutputShape},
    [SOFTMAX] = {softmaxForward, softmaxBackward, softmaxCalcOutputShape},
    [FLATTEN] = {flattenForward, flattenBackward, flattenCalcOutputShape},
    [QUANTIZATION] = {quantizationForward, quantizationBackward, quantizationCalcOutputShape},
    [ADAPTIVE_AVGPOOL1D] = {adaptiveAvgPool1dForward, adaptiveAvgPool1dBackward,
                            adaptiveAvgPool1dCalcOutputShape},
    [DROPOUT] = {dropoutForward, dropoutBackward, dropoutCalcOutputShape},
    [LAYERNORM] = {layerNormForward, layerNormBackward, layerNormCalcOutputShape},
    [GROUPNORM] = {groupNormForward, groupNormBackward, groupNormCalcOutputShape},
    [BATCHNORM1D] = {batchNorm1dForward, batchNorm1dBackward, batchNorm1dCalcOutputShape}};

void initLayer(layer_t *layer, layerType_t type, layerConfig_t *config) {
    layer->type = type;
    layer->config = config;
}

bool layerIsFrozen(const layer_t *layer) {
    switch (layer->type) {
    case LINEAR:
        return layer->config->linear->frozen;
    case CONV1D:
        return layer->config->conv1d->frozen;
    case CONV1D_TRANSPOSED:
        return layer->config->conv1dTransposed->frozen;
    case LAYERNORM:
        return layer->config->layerNorm->frozen;
    case GROUPNORM:
        return layer->config->groupNorm->frozen;
    case BATCHNORM1D:
        return layer->config->batchNorm1d->frozen;
    default:
        return false;
    }
}

bool layerParameters(const layer_t *layer, parameter_t **weightOut, parameter_t **biasOut) {
    switch (layer->type) {
    case LINEAR:
        *weightOut = layer->config->linear->weights;
        *biasOut = layer->config->linear->bias;
        return true;
    case CONV1D:
        *weightOut = layer->config->conv1d->weights;
        *biasOut = layer->config->conv1d->bias; /* may be NULL */
        return true;
    case CONV1D_TRANSPOSED:
        *weightOut = layer->config->conv1dTransposed->weights;
        *biasOut = layer->config->conv1dTransposed->bias;
        return true;
    case LAYERNORM:
        *weightOut = layer->config->layerNorm->gamma;
        *biasOut = layer->config->layerNorm->beta;
        return true;
    case GROUPNORM:
        *weightOut = layer->config->groupNorm->gamma;
        *biasOut = layer->config->groupNorm->beta;
        return true;
    case BATCHNORM1D:
        if (!layer->config->batchNorm1d->affine) {
            return false; /* no gamma/beta: never trainable, never the deepest */
        }
        *weightOut = layer->config->batchNorm1d->gamma;
        *biasOut = layer->config->batchNorm1d->beta;
        return true;
    default:
        return false;
    }
}

size_t deepestTrainableIndex(layer_t **model, size_t modelSize) {
    for (size_t i = 0; i < modelSize; i++) {
        parameter_t *w = NULL;
        parameter_t *b = NULL;
        if (layerParameters(model[i], &w, &b) && !layerIsFrozen(model[i])) {
            return i;
        }
    }
    return modelSize;
}
