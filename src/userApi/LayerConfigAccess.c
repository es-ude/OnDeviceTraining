#define SOURCE_FILE "LAYER_CONFIG_ACCESS"

#include <stdlib.h>

#include "AdaptiveAvgPool1d.h"
#include "AvgPool1d.h"
#include "BatchNorm1d.h"
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

/* Flatten/Quantization have no consumed arithmetic (arithmetic-type-split D4) — the universal
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
    case BATCHNORM1D:
        return layer->config->batchNorm1d->outputQ;
    case QUANTIZATION:
        return layer->config->quantization->outputQ;
    default:
        PRINT_ERROR("Unknown Layer Type!");
        exit(1);
    }
}

/* Producer's declared backward config for the dx wire it emits (#221). NULL = no declared config
 * (Flatten) -> passthrough of the upstream dtype. The loss-grad seed also passes NULL (lossConfig_t
 * has no quantization field -> model-output dtype, as before). */
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
    case BATCHNORM1D:
        return layer->config->batchNorm1d->propLossQ;
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
    case BATCHNORM1D:
        return layer->config->batchNorm1d->forwardMath;
    case QUANTIZATION:
        // Pure conversion node (arithmetic-type-split D4): no consumed arithmetic.
        return NO_ARITHMETIC;
    default:
        PRINT_ERROR("Unknown Layer Type!");
        exit(1);
    }
}

/* ---- FLOAT32-only gate (#152 PR3b; forward-only variant #468) ------------
 * Stacked training (microBatchSize > 1) is FLOAT32-only (#152): every declared
 * arithmetic, every storage config and every parameter/grad tensor must be
 * FLOAT32. Stacked evaluation (#468) only runs forwards, so its variant
 * (forwardOnly) skips the backward-only slots: propLossMath, propLossQ, the
 * grad arithmetics and grad storage. NULL means "not declared" and never fails
 * the gate: a NULL wire config is the upstream-dtype passthrough
 * (initLayerOutputs), a NULL bias is a bias-less layer, a NULL grad is a
 * frozen layer (#380). */

static bool mathIsFloat32(arithmetic_t a) {
    return a.type == ARITH_FLOAT32;
}

static bool storageIsFloat32(const quantization_t *q) {
    return q == NULL || q->type == FLOAT32;
}

/* The four slots every layer except Flatten/Quantization declares. */
static const char *wireNonFloat32(arithmetic_t forwardMath, arithmetic_t propLossMath,
                                  const quantization_t *outputQ, const quantization_t *propLossQ,
                                  bool forwardOnly) {
    if (!mathIsFloat32(forwardMath)) {
        return "forwardMath";
    }
    if (!forwardOnly && !mathIsFloat32(propLossMath)) {
        return "propLossMath";
    }
    if (!storageIsFloat32(outputQ)) {
        return "outputQ";
    }
    if (!forwardOnly && !storageIsFloat32(propLossQ)) {
        return "propLossQ";
    }
    return NULL;
}

static const char *paramNonFloat32(const parameter_t *p, const char *paramField,
                                   const char *gradField, bool forwardOnly) {
    if (p == NULL) {
        return NULL;
    }
    if (!storageIsFloat32(p->param->quantization)) {
        return paramField;
    }
    if (!forwardOnly && p->grad != NULL && !storageIsFloat32(p->grad->quantization)) {
        return gradField;
    }
    return NULL;
}

/* Linear / Conv1d / Conv1dTransposed: the GEMM family declares per-op grad
 * arithmetic on top of the wire slots. */
static const char *gemmNonFloat32(arithmetic_t forwardMath, arithmetic_t weightGradMath,
                                  arithmetic_t biasGradMath, arithmetic_t propLossMath,
                                  const quantization_t *outputQ, const quantization_t *propLossQ,
                                  const parameter_t *weights, const parameter_t *bias,
                                  bool forwardOnly) {
    if (!forwardOnly && !mathIsFloat32(weightGradMath)) {
        return "weightGradMath";
    }
    if (!forwardOnly && !mathIsFloat32(biasGradMath)) {
        return "biasGradMath";
    }
    const char *field = wireNonFloat32(forwardMath, propLossMath, outputQ, propLossQ, forwardOnly);
    if (field == NULL) {
        field = paramNonFloat32(weights, "weights.param", "weights.grad", forwardOnly);
    }
    if (field == NULL) {
        field = paramNonFloat32(bias, "bias.param", "bias.grad", forwardOnly);
    }
    return field;
}

/* LayerNorm / GroupNorm: no separate grad arithmetic -- propLossMath also
 * drives the dgamma/dbeta ops (LayerNorm.c backward, the executeOp calls with
 * .arithmetic = cfg->propLossMath). */
static const char *normNonFloat32(arithmetic_t forwardMath, arithmetic_t propLossMath,
                                  const quantization_t *outputQ, const quantization_t *propLossQ,
                                  const parameter_t *gamma, const parameter_t *beta,
                                  bool forwardOnly) {
    const char *field = wireNonFloat32(forwardMath, propLossMath, outputQ, propLossQ, forwardOnly);
    if (field == NULL) {
        field = paramNonFloat32(gamma, "gamma.param", "gamma.grad", forwardOnly);
    }
    if (field == NULL) {
        field = paramNonFloat32(beta, "beta.param", "beta.grad", forwardOnly);
    }
    return field;
}

static const char *nonFloat32Field(layer_t *layer, bool forwardOnly) {
    switch (layer->type) {
    case LINEAR: {
        const linearConfig_t *c = layer->config->linear;
        return gemmNonFloat32(c->forwardMath, c->weightGradMath, c->biasGradMath, c->propLossMath,
                              c->outputQ, c->propLossQ, c->weights, c->bias, forwardOnly);
    }
    case CONV1D: {
        const conv1dConfig_t *c = layer->config->conv1d;
        return gemmNonFloat32(c->forwardMath, c->weightGradMath, c->biasGradMath, c->propLossMath,
                              c->outputQ, c->propLossQ, c->weights, c->bias, forwardOnly);
    }
    case CONV1D_TRANSPOSED: {
        const conv1dTransposedConfig_t *c = layer->config->conv1dTransposed;
        return gemmNonFloat32(c->forwardMath, c->weightGradMath, c->biasGradMath, c->propLossMath,
                              c->outputQ, c->propLossQ, c->weights, c->bias, forwardOnly);
    }
    case LAYERNORM: {
        const layerNormConfig_t *c = layer->config->layerNorm;
        return normNonFloat32(c->forwardMath, c->propLossMath, c->outputQ, c->propLossQ, c->gamma,
                              c->beta, forwardOnly);
    }
    case GROUPNORM: {
        const groupNormConfig_t *c = layer->config->groupNorm;
        return normNonFloat32(c->forwardMath, c->propLossMath, c->outputQ, c->propLossQ, c->gamma,
                              c->beta, forwardOnly);
    }
    case BATCHNORM1D: {
        /* gamma/beta are NULL without affine (paramNonFloat32 passes NULL);
         * the running buffers are value state read as float* -- gated too. */
        const batchNorm1dConfig_t *c = layer->config->batchNorm1d;
        const char *field = normNonFloat32(c->forwardMath, c->propLossMath, c->outputQ,
                                           c->propLossQ, c->gamma, c->beta, forwardOnly);
        if (field == NULL && c->runningMean != NULL &&
            !storageIsFloat32(c->runningMean->quantization)) {
            field = "runningMean";
        }
        if (field == NULL && c->runningVar != NULL &&
            !storageIsFloat32(c->runningVar->quantization)) {
            field = "runningVar";
        }
        return field;
    }
    case RELU: {
        const reluConfig_t *c = layer->config->relu;
        return wireNonFloat32(c->forwardMath, c->propLossMath, c->outputQ, c->propLossQ,
                              forwardOnly);
    }
    case SOFTMAX: {
        const softmaxConfig_t *c = layer->config->softmax;
        return wireNonFloat32(c->forwardMath, c->propLossMath, c->outputQ, c->propLossQ,
                              forwardOnly);
    }
    case MAXPOOL1D: {
        /* argmaxIndices is INT32 index state, not a value wire -- not gated. */
        const maxPool1dConfig_t *c = layer->config->maxPool1d;
        return wireNonFloat32(c->forwardMath, c->propLossMath, c->outputQ, c->propLossQ,
                              forwardOnly);
    }
    case AVGPOOL1D: {
        const avgPool1dConfig_t *c = layer->config->avgPool1d;
        return wireNonFloat32(c->forwardMath, c->propLossMath, c->outputQ, c->propLossQ,
                              forwardOnly);
    }
    case ADAPTIVE_AVGPOOL1D: {
        const adaptiveAvgPool1dConfig_t *c = layer->config->adaptiveAvgPool1d;
        return wireNonFloat32(c->forwardMath, c->propLossMath, c->outputQ, c->propLossQ,
                              forwardOnly);
    }
    case DROPOUT: {
        /* The BOOL mask is not a value wire -- not gated (its element count
         * is Dropout's own guard, which fails fast at m > 1). */
        const dropoutConfig_t *c = layer->config->dropout;
        return wireNonFloat32(c->forwardMath, c->propLossMath, c->outputQ, c->propLossQ,
                              forwardOnly);
    }
    case FLATTEN:
        return NULL; /* no per-layer config: pure passthrough */
    case QUANTIZATION:
        return "layerType QUANTIZATION (a conversion node)";
    default:
        PRINT_ERROR("Unknown Layer Type!");
        exit(1);
    }
}

const char *layerNonFloat32Field(layer_t *layer) {
    return nonFloat32Field(layer, false);
}

const char *layerForwardNonFloat32Field(layer_t *layer) {
    return nonFloat32Field(layer, true);
}

bool layerIsFloat32Only(layer_t *layer) {
    return layerNonFloat32Field(layer) == NULL;
}

/* One reason per group below. An
 * unknown type exits rather than defaulting: a wrong `false` lets the remat
 * plan free an input the backward still reads. */
bool layerBackwardReadsInput(const layer_t *layer) {
    switch (layer->type) {
    /* x feeds only the weight grad (dW from dy and x); a frozen layer skips it. */
    case LINEAR:
    case CONV1D:
    case CONV1D_TRANSPOSED:
        return !layerIsFrozen(layer);
    /* Norms (LayerNorm/GroupNorm/BatchNorm1d) recompute x-hat from x, frozen
     * or not; ReLU masks dy by the sign of x; Softmax recomputes softmax(x)
     * for its Jacobian. */
    case LAYERNORM:
    case GROUPNORM:
    case BATCHNORM1D:
    case RELU:
    case SOFTMAX:
        return true;
    /* Pools route dy by argmax indices or window geometry; Flatten reshapes;
     * Quantization is straight-through; Dropout uses its stored mask and p. */
    case MAXPOOL1D:
    case AVGPOOL1D:
    case ADAPTIVE_AVGPOOL1D:
    case FLATTEN:
    case QUANTIZATION:
    case DROPOUT:
        return false;
    default:
        PRINT_ERROR("layerBackwardReadsInput: unknown layerType_t %d", (int)layer->type);
        exit(1);
    }
}
