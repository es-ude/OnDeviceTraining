#define SOURCE_FILE "LAYER"

#include <stdlib.h>

#include "AdaptiveAvgPool1d.h"
#include "AvgPool1d.h"
#include "BatchNorm1d.h"
#include "Common.h"
#include "Conv1d.h"
#include "Conv1dTransposed.h"
#include "Dropout.h"
#include "Flatten.h"
#include "GroupNorm.h"
#include "Layer.h"
#include "LayerNorm.h"
#include "Linear.h"
#include "MaxPool1d.h"
#include "QuantizationLayer.h"
#include "Relu.h"
#include "Softmax.h"

/* ADD is a merge node: only the graph driver runs it. A chain driver reaching it is a wiring
 * bug, so fail loudly instead of running a unary op on one input. */
#define ADD_CHAIN_DRIVER_ERROR                                                                     \
    "merge node reached a chain driver "                                                           \
    "(layer type ADD has no unary forward/backward/shape)"

static void addChainForwardStub(layer_t *layer, tensor_t *in, tensor_t *out,
                                const layerExec_t *exec) {
    (void)layer;
    (void)in;
    (void)out;
    (void)exec;
    PRINT_ERROR(ADD_CHAIN_DRIVER_ERROR);
    exit(1);
}

static void addChainBackwardStub(layer_t *layer, tensor_t *in, tensor_t *loss, tensor_t *propLoss,
                                 const layerExec_t *exec) {
    (void)layer;
    (void)in;
    (void)loss;
    (void)propLoss;
    (void)exec;
    PRINT_ERROR(ADD_CHAIN_DRIVER_ERROR);
    exit(1);
}

static void addChainShapeStub(layer_t *layer, shape_t *inShape, shape_t *outShape) {
    (void)layer;
    (void)inShape;
    (void)outShape;
    PRINT_ERROR(ADD_CHAIN_DRIVER_ERROR);
    exit(1);
}

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
    [BATCHNORM1D] = {batchNorm1dForward, batchNorm1dBackward, batchNorm1dCalcOutputShape},
    [ADD] = {addChainForwardStub, addChainBackwardStub, addChainShapeStub}};

_Static_assert(sizeof layerFunctions / sizeof layerFunctions[0] == LAYER_TYPE_COUNT,
               "one layerFunctions row per layerType_t member");

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
            return false; /* no gamma/beta: never trainable */
        }
        *weightOut = layer->config->batchNorm1d->gamma;
        *biasOut = layer->config->batchNorm1d->beta;
        return true;
    default:
        return false;
    }
}
