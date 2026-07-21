#define SOURCE_FILE "CONV1D_API"

#include <stdbool.h>
#include <stdlib.h>

#include "ArithmeticType.h"
#include "Common.h"
#include "Conv1d.h"
#include "Conv1dApi.h"
#include "Kernel.h"
#include "Layer.h"
#include "LayerCommon.h"
#include "LayerQuant.h"
#include "QuantizationApi.h"
#include "StorageApi.h"
#include "Tensor.h"
#include "TensorApi.h"

/* ============================================================================
 * New factory API — conv1dInit_t struct + layerQuant_t profile (PR 2).
 * ========================================================================== */

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

/*! Build a heap-owned shape_t with the given dims; the tensor that this shape
 *  is passed to takes ownership and freeTensor cascades into freeShape. */
static shape_t *buildOwnedShape(const size_t *srcDims, size_t numberOfDims) {
    size_t *dims = reserveMemory(numberOfDims * sizeof(size_t));
    for (size_t i = 0; i < numberOfDims; i++) {
        dims[i] = srcDims[i];
    }
    size_t *order = reserveMemory(numberOfDims * sizeof(size_t));
    setOrderOfDimsForNewTensor(numberOfDims, order);
    shape_t *shape = reserveMemory(sizeof(shape_t));
    setShape(shape, dims, numberOfDims, order);
    return shape;
}

static parameter_t *allocateConv1dWeights(size_t outChannels, size_t inChannels, size_t groups,
                                          size_t kernelSize, weightInit_t weightInit,
                                          quantization_t *storageQ, quantization_t *gradQ,
                                          bool trainable) {
    /* Conv1d weight shape: [outChannels, inChannels/groups, kernelSize].
     * Per Conv1d.h:11. */
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
                                       quantization_t *gradQ, bool trainable) {
    /* Bias tensor: shape [outChannels]. PyTorch draws bias from
     * uniform(+/- 1/sqrt(fan_in)) using the WEIGHT's fan_in. */
    shape_t *shape = buildOwnedShape((size_t[]){outChannels}, 1);
    tensor_t *paramTensor = initTensor(shape, getQLike(storageQ), NULL);
    initBiasTensor(paramTensor, fanIn);

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

static void validateLayerQuantForConv1d(layerQuant_t *lq, bool hasBias) {
    if (lq == NULL) {
        PRINT_ERROR("conv1dLayerInit: lq pointer is NULL");
        exit(1);
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
        /* Symmetric integer padding (PyTorch padding=N). Required to match a
         * reference conv trained with explicit padding under stride>1, where
         * SAME's minimal/asymmetric padding would diverge (issue #177 enc1). */
        initKernelExplicit(kernel, init->kernelSize, init->paddingAmount, dilation, stride);
    } else {
        initKernel(kernel, init->kernelSize, init->padding, dilation, stride);
    }
    return kernel;
}

layer_t *conv1dLayerInit(conv1dInit_t *init, layerQuant_t *lq) {
    validateConv1dInit(init);
    bool hasBias = resolveConv1dBias(init->bias);
    validateLayerQuantForConv1d(lq, hasBias);
    bool trainable = resolveTrainable(init->trainable, "conv1dLayerInit");

    size_t groups = init->groups == 0 ? 1 : init->groups;

    layer_t *layer = reserveMemory(sizeof(layer_t));
    layer->type = CONV1D;

    layerConfig_t *layerCfg = reserveMemory(sizeof(layerConfig_t));
    conv1dConfig_t *cfg = reserveMemory(sizeof(conv1dConfig_t));
    layerCfg->conv1d = cfg;
    layer->config = layerCfg;

    cfg->kernel = buildConv1dKernel(init);
    size_t fanIn = (init->inChannels / groups) * init->kernelSize;
    /* Grad storage knob (#261): Conv1d backward is FLOAT32-only, so a NULL knob
     * keeps the hard-pinned FLOAT32 default; a non-NULL weightGradStorage /
     * biasGradStorage overrides it explicitly. */
    quantization_t *floatGradQ = quantizationInitFloat();
    quantization_t *weightGradQ =
        lq->weightGradStorage != NULL ? lq->weightGradStorage : floatGradQ;
    quantization_t *biasGradQ = lq->biasGradStorage != NULL ? lq->biasGradStorage : floatGradQ;
    cfg->weights =
        allocateConv1dWeights(init->outChannels, init->inChannels, groups, init->kernelSize,
                              init->weightInit, lq->weightStorage, weightGradQ, trainable);
    cfg->bias = hasBias ? allocateConv1dBias(init->outChannels, fanIn, lq->biasStorage, biasGradQ,
                                             trainable)
                        : NULL;
    freeQuantization(floatGradQ);
    cfg->groups = groups;

    /* Borrowing: store the forward-wire/dx-wire storage configs verbatim, no copy.
     * Arithmetic slots are plain by-value copies of the profile's declared math. */
    cfg->forwardMath = lq->forwardMath;
    cfg->weightGradMath = lq->weightGradMath;
    cfg->biasGradMath = lq->biasGradMath;
    cfg->propLossMath = lq->propLossMath;
    cfg->outputQ = lq->outputQ;
    cfg->propLossQ = lq->propLossQ;
    cfg->weightGradAccMode = lq->weightGradAccMode;
    cfg->biasGradAccMode = lq->biasGradAccMode;
    cfg->ownsQuantizations = false;
    cfg->frozen = !trainable;

    return layer;
}

layer_t *conv1dLayerInitOwning(conv1dInit_t *init, layerQuant_t *lq) {
    validateConv1dInit(init);
    bool hasBias = resolveConv1dBias(init->bias);
    validateLayerQuantForConv1d(lq, hasBias);
    bool trainable = resolveTrainable(init->trainable, "conv1dLayerInitOwning");

    size_t groups = init->groups == 0 ? 1 : init->groups;

    layer_t *layer = reserveMemory(sizeof(layer_t));
    layer->type = CONV1D;

    layerConfig_t *layerCfg = reserveMemory(sizeof(layerConfig_t));
    conv1dConfig_t *cfg = reserveMemory(sizeof(conv1dConfig_t));
    layerCfg->conv1d = cfg;
    layer->config = layerCfg;

    cfg->kernel = buildConv1dKernel(init);
    /* allocateConv1dWeights / allocateConv1dBias internally clone via getQLike,
     * so the parameter tensors own their quantization_t — caller can drop
     * lq->weightStorage / lq->biasStorage immediately. */
    size_t fanIn = (init->inChannels / groups) * init->kernelSize;
    /* Grad storage knob (#261): Conv1d backward is FLOAT32-only, so a NULL knob
     * keeps the hard-pinned FLOAT32 default; a non-NULL weightGradStorage /
     * biasGradStorage overrides it explicitly. */
    quantization_t *floatGradQ = quantizationInitFloat();
    quantization_t *weightGradQ =
        lq->weightGradStorage != NULL ? lq->weightGradStorage : floatGradQ;
    quantization_t *biasGradQ = lq->biasGradStorage != NULL ? lq->biasGradStorage : floatGradQ;
    cfg->weights =
        allocateConv1dWeights(init->outChannels, init->inChannels, groups, init->kernelSize,
                              init->weightInit, lq->weightStorage, weightGradQ, trainable);
    cfg->bias = hasBias ? allocateConv1dBias(init->outChannels, fanIn, lq->biasStorage, biasGradQ,
                                             trainable)
                        : NULL;
    freeQuantization(floatGradQ);
    cfg->groups = groups;

    /* Owning: same arithmetic as Borrowing; deep-copy the two storage configs
     * (outputQ, propLossQ) into fresh allocations — 2 allocs, not 4. */
    cfg->forwardMath = lq->forwardMath;
    cfg->weightGradMath = lq->weightGradMath;
    cfg->biasGradMath = lq->biasGradMath;
    cfg->propLossMath = lq->propLossMath;
    cfg->outputQ = deepCopyQuantization(lq->outputQ);
    cfg->propLossQ = deepCopyQuantization(lq->propLossQ);
    cfg->weightGradAccMode = lq->weightGradAccMode;
    cfg->biasGradAccMode = lq->biasGradAccMode;
    cfg->ownsQuantizations = true;
    cfg->frozen = !trainable;

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
            freeReservedMemory(cfg->outputQ->qConfig);
            freeReservedMemory(cfg->outputQ);
        }
        if (cfg->propLossQ != NULL && cfg->propLossQ != cfg->outputQ) {
            freeReservedMemory(cfg->propLossQ->qConfig);
            freeReservedMemory(cfg->propLossQ);
        }
    }

    freeReservedMemory(cfg);
    freeReservedMemory(conv1dLayer->config);
    freeReservedMemory(conv1dLayer);
}
