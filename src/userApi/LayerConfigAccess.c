#define SOURCE_FILE "LAYER_CONFIG_ACCESS"

#include <stdlib.h>

#include "AdaptiveAvgPool1d.h"
#include "AvgPool1d.h"
#include "Common.h"
#include "Conv1d.h"
#include "Conv1dTransposed.h"
#include "Dropout.h"
#include "GroupNorm.h"
#include "LayerConfigAccess.h"
#include "LayerNorm.h"
#include "Linear.h"
#include "MaxPool1d.h"
#include "QuantizationLayer.h"
#include "Relu.h"
#include "Softmax.h"

/* Flatten/Quantization have no consumed arithmetic (D4) — the universal
 * float bridge, matching arithmeticFromQuantizationOrDefault(NULL). */
static const arithmetic_t NO_ARITHMETIC = {.type = ARITH_FLOAT32, .roundingMode = HALF_AWAY};

quantization_t *layerOutputQ(layer_t *layer) {
    switch (layer->type) {
    case LINEAR:
        return layer->config->linear->outputQ;
    case RELU:
        return layer->config->relu->outputQ;
    case SOFTMAX:
        return layer->config->softmax->outputQ;
    case FLATTEN:
        // Flatten has no per-layer quantization; output dtype equals input dtype.
        return NULL;
    case CONV1D:
        return layer->config->conv1d->outputQ;
    case CONV1D_TRANSPOSED:
        return layer->config->conv1dTransposed->outputQ;
    case MAXPOOL1D:
        return layer->config->maxPool1d->outputQ;
    case AVGPOOL1D:
        return layer->config->avgPool1d->outputQ;
    case ADAPTIVE_AVGPOOL1D:
        return layer->config->adaptiveAvgPool1d->outputQ;
    case DROPOUT:
        return layer->config->dropout->outputQ;
    case LAYERNORM:
        return layer->config->layerNorm->outputQ;
    case GROUPNORM:
        return layer->config->groupNorm->outputQ;
    case QUANTIZATION:
        return layer->config->quantization->outputQ;
    default:
        PRINT_ERROR("Unknown Layer Type!");
        exit(1);
    }
}

/* Producer's declared backward config for the dx wire it emits (design spec
 * 2026-07-02 §5, #221). NULL = no declared config (Flatten) -> passthrough of
 * the upstream dtype. The loss-grad seed also passes NULL (lossConfig_t has
 * no quantization field -> model-output dtype, as before). */
quantization_t *backwardWireQ(layer_t *layer) {
    switch (layer->type) {
    case LINEAR:
        return layer->config->linear->propLossQ;
    case CONV1D:
        return layer->config->conv1d->propLossQ;
    case CONV1D_TRANSPOSED:
        return layer->config->conv1dTransposed->propLossQ;
    case MAXPOOL1D:
        return layer->config->maxPool1d->propLossQ;
    case AVGPOOL1D:
        return layer->config->avgPool1d->propLossQ;
    case ADAPTIVE_AVGPOOL1D:
        return layer->config->adaptiveAvgPool1d->propLossQ;
    case RELU:
        return layer->config->relu->propLossQ;
    case SOFTMAX:
        return layer->config->softmax->propLossQ;
    case DROPOUT:
        return layer->config->dropout->propLossQ;
    case LAYERNORM:
        return layer->config->layerNorm->propLossQ;
    case GROUPNORM:
        return layer->config->groupNorm->propLossQ;
    case QUANTIZATION:
        return layer->config->quantization->propLossQ;
    case FLATTEN:
        return NULL;
    default:
        PRINT_ERROR("Unknown Layer Type!");
        exit(1);
    }
}

arithmetic_t layerForwardMath(layer_t *layer) {
    switch (layer->type) {
    case LINEAR:
        return layer->config->linear->forwardMath;
    case RELU:
        return layer->config->relu->forwardMath;
    case SOFTMAX:
        return layer->config->softmax->forwardMath;
    case FLATTEN:
        return NO_ARITHMETIC;
    case CONV1D:
        return layer->config->conv1d->forwardMath;
    case CONV1D_TRANSPOSED:
        return layer->config->conv1dTransposed->forwardMath;
    case MAXPOOL1D:
        return layer->config->maxPool1d->forwardMath;
    case AVGPOOL1D:
        return layer->config->avgPool1d->forwardMath;
    case ADAPTIVE_AVGPOOL1D:
        return layer->config->adaptiveAvgPool1d->forwardMath;
    case DROPOUT:
        return layer->config->dropout->forwardMath;
    case LAYERNORM:
        return layer->config->layerNorm->forwardMath;
    case GROUPNORM:
        return layer->config->groupNorm->forwardMath;
    case QUANTIZATION:
        // Pure conversion node (D4): no consumed arithmetic.
        return NO_ARITHMETIC;
    default:
        PRINT_ERROR("Unknown Layer Type!");
        exit(1);
    }
}
