#define SOURCE_FILE "OPTIMIZER"

#include <stdlib.h>

#include "Common.h"
#include "Conv1d.h"
#include "Conv1dTransposed.h"
#include "Optimizer.h"
#include "Sgd.h"

optimizerFunctions_t optimizerFunctions[] = {
    [SGD] = {sgdStep, sgdZeroGrad}, [SGD_M] = {sgdStepM, sgdZeroGrad}};

/* Conv1d/Conv1dTransposed are bias-optional (BIAS_FALSE, header-sanctioned):
 * a bias-less conv contributes only its weight state, not a weight+bias
 * pair. Every other trainable layer type still has a fixed contribution. */
static size_t calcNumberOfStatesByLayer(const layer_t *layer) {
    switch (layer->type) {
    case LINEAR:
    case LAYERNORM:
    case GROUPNORM:
        return 2;
    case RELU:
    case SOFTMAX:
    case FLATTEN:
    case MAXPOOL1D:
    case AVGPOOL1D:
    case ADAPTIVE_AVGPOOL1D:
    case DROPOUT:
    case QUANTIZATION:
        return 0;
    case CONV1D:
        return layer->config->conv1d->bias != NULL ? 2 : 1;
    case CONV1D_TRANSPOSED:
        return layer->config->conv1dTransposed->bias != NULL ? 2 : 1;
    default:
        PRINT_ERROR("Unknown Layer Type!");
        exit(1);
    }
}

size_t calcTotalNumberOfStates(layer_t **model, size_t sizeModel) {
    size_t number = 0;
    for (size_t i = 0; i < sizeModel; i++) {
        number += calcNumberOfStatesByLayer(model[i]);
    }
    return number;
}
