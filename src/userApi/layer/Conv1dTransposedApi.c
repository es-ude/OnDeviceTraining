#define SOURCE_FILE "CONV1D_TRANSPOSED_API"

#include <stdbool.h>
#include <stdlib.h>

#include "ArithmeticType.h"
#include "Common.h"
#include "Conv1dTransposed.h"
#include "Conv1dTransposedApi.h"
#include "Kernel.h"
#include "Layer.h"
#include "LayerCommon.h"
#include "LayerQuant.h"
#include "QuantizationApi.h"
#include "StorageApi.h"
#include "Tensor.h"
#include "TensorApi.h"

static bool resolveConv1dTransposedBias(bias_t b) {
    switch (b) {
    case BIAS_DEFAULT:
        return true;
    case BIAS_TRUE:
        return true;
    case BIAS_FALSE:
        return false;
    default:
        PRINT_ERROR("conv1dTransposedLayerInit: invalid bias value (got %d)", (int)b);
        exit(1);
    }
}

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

static parameter_t *allocateConv1dTransposedWeights(size_t inChannels, size_t outChannels,
                                                    size_t groups, size_t kernelSize,
                                                    weightInit_t weightInit,
                                                    quantization_t *storageQ, quantization_t *gradQ,
                                                    bool trainable) {
    /* Conv1dTransposed weight shape: [inChannels, outChannels/groups, kernelSize].
     * Note SWAP relative to Conv1d. Per Conv1dTransposed.h:12. */
    if (outChannels % groups != 0) {
        PRINT_ERROR("conv1dTransposedLayerInit: outChannels (%zu) must be divisible by "
                    "groups (%zu)",
                    outChannels, groups);
        exit(1);
    }
    if (inChannels % groups != 0) {
        PRINT_ERROR("conv1dTransposedLayerInit: inChannels (%zu) must be divisible by "
                    "groups (%zu)",
                    inChannels, groups);
        exit(1);
    }
    size_t outPerGroup = outChannels / groups;

    shape_t *shape = buildOwnedShape((size_t[]){inChannels, outPerGroup, kernelSize}, 3);
    tensor_t *paramTensor = initTensor(shape, getQLike(storageQ), NULL);

    /* ConvTranspose weight layout [inChannels, outPerGroup, kernelSize]:
     * PyTorch fan_in = weight.size(1)*k = outPerGroup*kernelSize,
     *         fan_out = weight.size(0)*k = inChannels*kernelSize. */
    size_t fanIn = outPerGroup * kernelSize;
    size_t fanOut = inChannels * kernelSize;
    initWeightTensor(paramTensor, weightInit, fanIn, fanOut);

    tensor_t *gradTensor = trainable ? gradInit(paramTensor, gradQ, NULL) : NULL;
    return parameterInit(paramTensor, gradTensor);
}

static parameter_t *allocateConv1dTransposedBias(size_t outChannels, size_t fanIn,
                                                 quantization_t *storageQ, quantization_t *gradQ,
                                                 bool trainable) {
    /* PyTorch draws bias from uniform(+/- 1/sqrt(fan_in)) using the WEIGHT's
     * fan_in (= outPerGroup*kernelSize). */
    shape_t *shape = buildOwnedShape((size_t[]){outChannels}, 1);
    tensor_t *paramTensor = initTensor(shape, getQLike(storageQ), NULL);
    initBiasTensor(paramTensor, fanIn);
    tensor_t *gradTensor = trainable ? gradInit(paramTensor, gradQ, NULL) : NULL;
    return parameterInit(paramTensor, gradTensor);
}

static void validateConv1dTransposedInit(conv1dTransposedInit_t *init) {
    if (init == NULL) {
        PRINT_ERROR("conv1dTransposedLayerInit: init pointer is NULL");
        exit(1);
    }
    if (init->inChannels == 0) {
        PRINT_ERROR("conv1dTransposedLayerInit: inChannels must be > 0");
        exit(1);
    }
    if (init->outChannels == 0) {
        PRINT_ERROR("conv1dTransposedLayerInit: outChannels must be > 0");
        exit(1);
    }
    if (init->kernelSize == 0) {
        PRINT_ERROR("conv1dTransposedLayerInit: kernelSize must be > 0");
        exit(1);
    }
}

static void validateLayerQuantForConv1dTransposed(layerQuant_t *lq, bool hasBias) {
    if (lq == NULL) {
        PRINT_ERROR("conv1dTransposedLayerInit: lq pointer is NULL");
        exit(1);
    }
    if (lq->outputQ == NULL) {
        PRINT_ERROR("conv1dTransposedLayerInit: layerQuant.outputQ must be set");
        exit(1);
    }
    if (lq->propLossQ == NULL) {
        PRINT_ERROR("conv1dTransposedLayerInit: layerQuant.propLossQ must be set");
        exit(1);
    }
    if (lq->weightStorage == NULL) {
        PRINT_ERROR("conv1dTransposedLayerInit: layerQuant.weightStorage must be set");
        exit(1);
    }
    if (hasBias && lq->biasStorage == NULL) {
        PRINT_ERROR("conv1dTransposedLayerInit: layerQuant.biasStorage must be set when bias "
                    "is enabled");
        exit(1);
    }
}

static kernel_t *buildConv1dTransposedKernel(conv1dTransposedInit_t *init) {
    kernel_t *kernel = reserveMemory(sizeof(kernel_t));
    size_t stride = init->stride == 0 ? 1 : init->stride;
    size_t dilation = init->dilation == 0 ? 1 : init->dilation;
    initKernel(kernel, init->kernelSize, init->padding, dilation, stride);
    return kernel;
}

static layer_t *buildConv1dTransposedLayerSkeleton(conv1dTransposedInit_t *init, layerQuant_t *lq,
                                                   bool hasBias, size_t groups, bool trainable) {
    layer_t *layer = reserveMemory(sizeof(layer_t));
    layer->type = CONV1D_TRANSPOSED;

    layerConfig_t *layerCfg = reserveMemory(sizeof(layerConfig_t));
    conv1dTransposedConfig_t *cfg = reserveMemory(sizeof(conv1dTransposedConfig_t));
    layerCfg->conv1dTransposed = cfg;
    layer->config = layerCfg;

    cfg->kernel = buildConv1dTransposedKernel(init);
    size_t fanIn = (init->outChannels / groups) * init->kernelSize;
    /* Grad storage knob (#261): Conv1dTransposed backward is FLOAT32-only, so a
     * NULL knob keeps the hard-pinned FLOAT32 default; a non-NULL
     * weightGradStorage/biasGradStorage overrides it explicitly. */
    quantization_t *floatGradQ = quantizationInitFloat();
    quantization_t *weightGradQ =
        lq->weightGradStorage != NULL ? lq->weightGradStorage : floatGradQ;
    quantization_t *biasGradQ = lq->biasGradStorage != NULL ? lq->biasGradStorage : floatGradQ;
    cfg->weights = allocateConv1dTransposedWeights(init->inChannels, init->outChannels, groups,
                                                   init->kernelSize, init->weightInit,
                                                   lq->weightStorage, weightGradQ, trainable);
    cfg->bias = hasBias ? allocateConv1dTransposedBias(init->outChannels, fanIn, lq->biasStorage,
                                                       biasGradQ, trainable)
                        : NULL;
    freeQuantization(floatGradQ);
    cfg->groups = groups;
    cfg->outputPadding = init->outputPadding;
    cfg->frozen = !trainable;
    return layer;
}

layer_t *conv1dTransposedLayerInit(conv1dTransposedInit_t *init, layerQuant_t *lq) {
    validateConv1dTransposedInit(init);
    bool hasBias = resolveConv1dTransposedBias(init->bias);
    validateLayerQuantForConv1dTransposed(lq, hasBias);
    bool trainable = resolveTrainable(init->trainable, "conv1dTransposedLayerInit");

    size_t groups = init->groups == 0 ? 1 : init->groups;

    layer_t *layer = buildConv1dTransposedLayerSkeleton(init, lq, hasBias, groups, trainable);
    conv1dTransposedConfig_t *cfg = layer->config->conv1dTransposed;

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
    return layer;
}

layer_t *conv1dTransposedLayerInitOwning(conv1dTransposedInit_t *init, layerQuant_t *lq) {
    validateConv1dTransposedInit(init);
    bool hasBias = resolveConv1dTransposedBias(init->bias);
    validateLayerQuantForConv1dTransposed(lq, hasBias);
    bool trainable = resolveTrainable(init->trainable, "conv1dTransposedLayerInitOwning");

    size_t groups = init->groups == 0 ? 1 : init->groups;

    layer_t *layer = buildConv1dTransposedLayerSkeleton(init, lq, hasBias, groups, trainable);
    conv1dTransposedConfig_t *cfg = layer->config->conv1dTransposed;

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
    return layer;
}

void freeConv1dTransposedLayer(layer_t *layer) {
    if (layer == NULL) {
        return;
    }
    conv1dTransposedConfig_t *cfg = layer->config->conv1dTransposed;

    if (cfg->weights != NULL) {
        freeParameter(cfg->weights);
    }
    if (cfg->bias != NULL) {
        freeParameter(cfg->bias);
    }
    freeReservedMemory(cfg->kernel);

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
    freeReservedMemory(layer->config);
    freeReservedMemory(layer);
}
