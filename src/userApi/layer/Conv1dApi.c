#define SOURCE_FILE "CONV1D_API"

#include <stdbool.h>
#include <stdlib.h>

#include "ArithmeticType.h"
#include "Common.h"
#include "Conv1d.h"
#include "Conv1dApi.h"
#include "ExecuteOp.h"
#include "Kernel.h"
#include "Layer.h"
#include "LayerCommon.h"
#include "LayerQuant.h"
#include "QuantizationApi.h"
#include "StorageApi.h"
#include "Tensor.h"
#include "TensorApi.h"

static bool resolveConv1dBias(bias_t b) {
    switch (b) {
    case BIAS_DEFAULT:
        return true; /* PyTorch parity for Conv1d */
    case BIAS_TRUE:
        return true;
    case BIAS_FALSE:
        return false;
    default:
        PRINT_ERROR("conv1dLayerInit: invalid bias value (got %d)", (int)b);
        exit(1);
    }
}

static parameter_t *allocateConv1dWeights(size_t outChannels, size_t inChannels, size_t groups,
                                          size_t kernelSize, weightInit_t weightInit,
                                          quantization_t *storageQ, quantization_t *gradQ,
                                          bool trainable) {
    /* Weight shape [outChannels, inChannels/groups, kernelSize]. */
    if (inChannels % groups != 0) {
        PRINT_ERROR("conv1dLayerInit: inChannels (%zu) must be divisible by groups (%zu)",
                    inChannels, groups);
        exit(1);
    }
    if (outChannels % groups != 0) {
        PRINT_ERROR("conv1dLayerInit: outChannels (%zu) must be divisible by groups (%zu)",
                    outChannels, groups);
        exit(1);
    }
    size_t inPerGroup = inChannels / groups;

    shape_t *shape = buildOwnedShape((size_t[]){outChannels, inPerGroup, kernelSize}, 3);
    tensor_t *paramTensor = initTensor(shape, getQLike(storageQ), NULL);

    /* fan_in = inPerGroup*kernelSize; fan_out = outPerGroup*kernelSize
     * (PyTorch _calculate_fan_in_and_fan_out for the Conv1d weight layout). */
    size_t fanIn = inPerGroup * kernelSize;
    size_t fanOut = (outChannels / groups) * kernelSize;
    initWeightTensor(paramTensor, weightInit, fanIn, fanOut);

    tensor_t *gradTensor = trainable ? gradInit(paramTensor, gradQ, NULL) : NULL;
    return parameterInit(paramTensor, gradTensor);
}

static parameter_t *allocateConv1dBias(size_t outChannels, size_t fanIn, quantization_t *storageQ,
                                       quantization_t *gradQ, bool trainable,
                                       const rngKey_t *initKey) {
    /* Bias tensor: shape [outChannels]. PyTorch draws bias from
     * uniform(+/- 1/sqrt(fan_in)) using the WEIGHT's fan_in. */
    shape_t *shape = buildOwnedShape((size_t[]){outChannels}, 1);
    tensor_t *paramTensor = initTensor(shape, getQLike(storageQ), NULL);
    initBiasTensor(paramTensor, fanIn, initKey);

    tensor_t *gradTensor = trainable ? gradInit(paramTensor, gradQ, NULL) : NULL;
    return parameterInit(paramTensor, gradTensor);
}

static void validateConv1dInit(conv1dInit_t *init) {
    if (init == NULL) {
        PRINT_ERROR("conv1dLayerInit: init pointer is NULL");
        exit(1);
    }
    if (init->inChannels == 0) {
        PRINT_ERROR("conv1dLayerInit: inChannels must be > 0");
        exit(1);
    }
    if (init->outChannels == 0) {
        PRINT_ERROR("conv1dLayerInit: outChannels must be > 0");
        exit(1);
    }
    if (init->kernelSize == 0) {
        PRINT_ERROR("conv1dLayerInit: kernelSize must be > 0");
        exit(1);
    }
}

static void validateLayerQuantForConv1d(layerQuant_t *lq, bool hasBias, bool trainable) {
    if (lq == NULL) {
        PRINT_ERROR("conv1dLayerInit: lq pointer is NULL");
        exit(1);
    }
    /* A frozen layer never accumulates grads, so its modes may stay unset. */
    if (trainable) {
        executeOpValidateAccMode(lq->weightGradAccMode,
                                 "conv1dLayerInit: layerQuant.weightGradAccMode");
        if (hasBias) {
            executeOpValidateAccMode(lq->biasGradAccMode,
                                     "conv1dLayerInit: layerQuant.biasGradAccMode");
        }
    }
    if (lq->outputQ == NULL) {
        PRINT_ERROR("conv1dLayerInit: layerQuant.outputQ must be set");
        exit(1);
    }
    if (lq->propLossQ == NULL) {
        PRINT_ERROR("conv1dLayerInit: layerQuant.propLossQ must be set");
        exit(1);
    }
    if (lq->weightStorage == NULL) {
        PRINT_ERROR("conv1dLayerInit: layerQuant.weightStorage must be set");
        exit(1);
    }
    if (hasBias && lq->biasStorage == NULL) {
        PRINT_ERROR("conv1dLayerInit: layerQuant.biasStorage must be set when bias is enabled");
        exit(1);
    }
}

/*! Build a heap-owned kernel_t from the conv1dInit_t fields, applying
 *  zero-init defaults (stride=1, dilation=1, padding=VALID). */
static kernel_t *buildConv1dKernel(conv1dInit_t *init) {
    kernel_t *kernel = reserveMemory(sizeof(kernel_t));
    size_t stride = init->stride == 0 ? 1 : init->stride;
    size_t dilation = init->dilation == 0 ? 1 : init->dilation;
    if (init->padding == EXPLICIT) {
        /* Explicit padding is symmetric (PyTorch padding=N); SAME may pad
         * asymmetrically, which diverges from such a reference under stride > 1. */
        initKernelExplicit(kernel, init->kernelSize, init->paddingAmount, dilation, stride);
    } else {
        initKernel(kernel, init->kernelSize, init->padding, dilation, stride);
    }
    return kernel;
}

static layer_t *buildConv1dLayerSkeleton(conv1dInit_t *init, layerQuant_t *lq, bool hasBias,
                                         size_t groups, bool trainable) {
    validateRandomInit(init->weightInit, lq->weightStorage, hasBias ? lq->biasStorage : NULL,
                       "conv1dLayerInit");
    layer_t *layer = reserveMemory(sizeof(layer_t));
    layer->type = CONV1D;

    layerConfig_t *layerCfg = reserveMemory(sizeof(layerConfig_t));
    conv1dConfig_t *cfg = reserveMemory(sizeof(conv1dConfig_t));
    layerCfg->conv1d = cfg;
    layer->config = layerCfg;

    cfg->kernel = buildConv1dKernel(init);
    size_t fanIn = (init->inChannels / groups) * init->kernelSize;
    /* Parameter gradients default to FLOAT32; explicit storage overrides take
     * precedence. */
    quantization_t *floatGradQ = quantizationInitFloat();
    quantization_t *weightGradQ =
        lq->weightGradStorage != NULL ? lq->weightGradStorage : floatGradQ;
    quantization_t *biasGradQ = lq->biasGradStorage != NULL ? lq->biasGradStorage : floatGradQ;
    cfg->weights =
        allocateConv1dWeights(init->outChannels, init->inChannels, groups, init->kernelSize,
                              init->weightInit, lq->weightStorage, weightGradQ, trainable);
    cfg->bias = hasBias ? allocateConv1dBias(init->outChannels, fanIn, lq->biasStorage, biasGradQ,
                                             trainable, init->weightInit.key)
                        : NULL;
    freeQuantization(floatGradQ);
    cfg->groups = groups;
    cfg->frozen = !trainable;
    return layer;
}

layer_t *conv1dLayerInit(conv1dInit_t *init, layerQuant_t *lq) {
    validateConv1dInit(init);
    bool hasBias = resolveConv1dBias(init->bias);
    bool trainable = resolveTrainable(init->trainable, "conv1dLayerInit");
    validateLayerQuantForConv1d(lq, hasBias, trainable);

    size_t groups = init->groups == 0 ? 1 : init->groups;

    layer_t *layer = buildConv1dLayerSkeleton(init, lq, hasBias, groups, trainable);
    conv1dConfig_t *cfg = layer->config->conv1d;

    cfg->forwardMath = lq->forwardMath;
    cfg->weightGradMath = lq->weightGradMath;
    cfg->biasGradMath = lq->biasGradMath;
    cfg->propLossMath = lq->propLossMath;
    cfg->outputQ = lq->outputQ;
    cfg->propLossQ = lq->propLossQ;
    cfg->weightGradAccMode = lq->weightGradAccMode;
    cfg->biasGradAccMode = lq->biasGradAccMode;
    cfg->ownsQuantizations = false;
    return layer;
}

layer_t *conv1dLayerInitOwning(conv1dInit_t *init, layerQuant_t *lq) {
    validateConv1dInit(init);
    bool hasBias = resolveConv1dBias(init->bias);
    bool trainable = resolveTrainable(init->trainable, "conv1dLayerInitOwning");
    validateLayerQuantForConv1d(lq, hasBias, trainable);

    size_t groups = init->groups == 0 ? 1 : init->groups;

    layer_t *layer = buildConv1dLayerSkeleton(init, lq, hasBias, groups, trainable);
    conv1dConfig_t *cfg = layer->config->conv1d;

    cfg->forwardMath = lq->forwardMath;
    cfg->weightGradMath = lq->weightGradMath;
    cfg->biasGradMath = lq->biasGradMath;
    cfg->propLossMath = lq->propLossMath;
    cfg->outputQ = deepCopyQuantization(lq->outputQ);
    cfg->propLossQ = deepCopyQuantization(lq->propLossQ);
    cfg->weightGradAccMode = lq->weightGradAccMode;
    cfg->biasGradAccMode = lq->biasGradAccMode;
    cfg->ownsQuantizations = true;
    return layer;
}

void freeConv1dLayer(layer_t *conv1dLayer) {
    if (conv1dLayer == NULL) {
        return;
    }
    conv1dConfig_t *cfg = conv1dLayer->config->conv1d;

    /* Always factory-owned: parameters + kernel. */
    if (cfg->weights != NULL) {
        freeParameter(cfg->weights);
    }
    if (cfg->bias != NULL) {
        freeParameter(cfg->bias);
    }
    freeReservedMemory(cfg->kernel);

    /* Conditionally factory-owned: the two storage configs (Owning variant only). */
    if (cfg->ownsQuantizations) {
        if (cfg->outputQ != NULL) {
            freeQuantization(cfg->outputQ);
        }
        if (cfg->propLossQ != NULL && cfg->propLossQ != cfg->outputQ) {
            freeQuantization(cfg->propLossQ);
        }
    }

    freeReservedMemory(cfg);
    freeReservedMemory(conv1dLayer->config);
    freeReservedMemory(conv1dLayer);
}
