#define SOURCE_FILE "POOL1D_API"

#include <stdbool.h>
#include <stdlib.h>

#include "ArithmeticType.h"
#include "AvgPool1d.h"
#include "Common.h"
#include "Kernel.h"
#include "Layer.h"
#include "LayerQuant.h"
#include "MaxPool1d.h"
#include "Pool1dApi.h"
#include "QuantizationApi.h"
#include "StorageApi.h"
#include "Tensor.h"
#include "TensorApi.h"

/* ============================================================================
 * Shared helpers
 * ========================================================================== */

/*! Compute output length per the geometry rule used by the internal
 *  windowGeometry1dCalc, replicated here for factory pre-allocation
 *  without bringing in SlidingWindow1d.
 *
 *  VALID: outputLength = (inputLength - dilation*(kernelSize - 1) - 1) / stride + 1
 *  SAME:  outputLength = ceil(inputLength / stride)
 *
 *  Matches the runtime windowGeometry1dCalc result used by both pool
 *  layers' forward paths. */
static size_t computePool1dOutputLength(paddingType_t padding, size_t inputLength,
                                        size_t kernelSize, size_t dilation, size_t stride) {
    if (padding == SAME) {
        return (inputLength + stride - 1) / stride;
    }
    /* VALID */
    size_t effectiveK = dilation * (kernelSize - 1) + 1;
    if (effectiveK > inputLength) {
        PRINT_ERROR("Pool1d: effective kernel %zu exceeds inputLength %zu", effectiveK,
                    inputLength);
        exit(1);
    }
    return (inputLength - effectiveK) / stride + 1;
}

/* ============================================================================
 * MaxPool1d
 * ========================================================================== */

static void validateMaxPool1dInit(maxPool1dInit_t *init) {
    if (init == NULL) {
        PRINT_ERROR("maxPool1dLayerInit: init pointer is NULL");
        exit(1);
    }
    if (init->kernelSize == 0) {
        PRINT_ERROR("maxPool1dLayerInit: kernelSize must be > 0");
        exit(1);
    }
    if (init->inputChannels == 0) {
        PRINT_ERROR("maxPool1dLayerInit: inputChannels must be > 0");
        exit(1);
    }
    if (init->inputLength == 0) {
        PRINT_ERROR("maxPool1dLayerInit: inputLength must be > 0");
        exit(1);
    }
}

static void validateLayerQuantForMaxPool1d(layerQuant_t *lq) {
    if (lq == NULL) {
        PRINT_ERROR("maxPool1dLayerInit: lq pointer is NULL");
        exit(1);
    }
    if (lq->outputQ == NULL) {
        PRINT_ERROR("maxPool1dLayerInit: layerQuant.outputQ must be set");
        exit(1);
    }
    if (lq->propLossQ == NULL) {
        PRINT_ERROR("maxPool1dLayerInit: layerQuant.propLossQ must be set");
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

static tensor_t *buildMaxPool1dArgmax(size_t inputChannels, size_t outputLength) {
    /* Argmax buffer is sized for batch=1 (training_batch iterates microbatch-
     * by-microbatch in this framework). Shape: [1, inputChannels, outputLength]. */
    shape_t *shape = buildOwnedShape((size_t[]){1, inputChannels, outputLength}, 3);
    quantization_t *q = quantizationInitInt32();
    return initTensor(shape, q, NULL);
}

static layer_t *buildMaxPool1dLayerSkeleton(maxPool1dInit_t *init) {
    size_t stride = init->stride == 0 ? init->kernelSize : init->stride;
    size_t dilation = init->dilation == 0 ? 1 : init->dilation;

    kernel_t *kernel = reserveMemory(sizeof(kernel_t));
    initKernel(kernel, init->kernelSize, init->padding, dilation, stride);

    size_t outputLength = computePool1dOutputLength(init->padding, init->inputLength,
                                                    init->kernelSize, dilation, stride);
    tensor_t *argmax = buildMaxPool1dArgmax(init->inputChannels, outputLength);

    layer_t *layer = reserveMemory(sizeof(layer_t));
    layer->type = MAXPOOL1D;
    layerConfig_t *layerCfg = reserveMemory(sizeof(layerConfig_t));
    maxPool1dConfig_t *cfg = reserveMemory(sizeof(maxPool1dConfig_t));
    layerCfg->maxPool1d = cfg;
    layer->config = layerCfg;

    cfg->kernel = kernel;
    cfg->argmaxIndices = argmax;
    return layer;
}

layer_t *maxPool1dLayerInit(maxPool1dInit_t *init, layerQuant_t *lq) {
    validateMaxPool1dInit(init);
    validateLayerQuantForMaxPool1d(lq);

    layer_t *layer = buildMaxPool1dLayerSkeleton(init);
    maxPool1dConfig_t *cfg = layer->config->maxPool1d;
    cfg->forwardMath = lq->forwardMath;
    cfg->propLossMath = lq->propLossMath;
    cfg->outputQ = lq->outputQ;
    cfg->propLossQ = lq->propLossQ;
    cfg->ownsQuantizations = false;
    return layer;
}

layer_t *maxPool1dLayerInitOwning(maxPool1dInit_t *init, layerQuant_t *lq) {
    validateMaxPool1dInit(init);
    validateLayerQuantForMaxPool1d(lq);

    layer_t *layer = buildMaxPool1dLayerSkeleton(init);
    maxPool1dConfig_t *cfg = layer->config->maxPool1d;
    cfg->forwardMath = lq->forwardMath;
    cfg->propLossMath = lq->propLossMath;
    cfg->outputQ = deepCopyQuantization(lq->outputQ);
    cfg->propLossQ = deepCopyQuantization(lq->propLossQ);
    cfg->ownsQuantizations = true;
    return layer;
}

void freeMaxPool1dLayer(layer_t *layer) {
    if (layer == NULL) {
        return;
    }
    maxPool1dConfig_t *cfg = layer->config->maxPool1d;

    freeReservedMemory(cfg->kernel);
    if (cfg->argmaxIndices != NULL) {
        freeTensor(cfg->argmaxIndices);
    }

    if (cfg->ownsQuantizations) {
        if (cfg->outputQ != NULL) {
            freeQuantization(cfg->outputQ);
        }
        if (cfg->propLossQ != NULL && cfg->propLossQ != cfg->outputQ) {
            freeQuantization(cfg->propLossQ);
        }
    }

    freeReservedMemory(cfg);
    freeReservedMemory(layer->config);
    freeReservedMemory(layer);
}

/* ============================================================================
 * AvgPool1d
 * ========================================================================== */

static void validateAvgPool1dInit(avgPool1dInit_t *init) {
    if (init == NULL) {
        PRINT_ERROR("avgPool1dLayerInit: init pointer is NULL");
        exit(1);
    }
    if (init->kernelSize == 0) {
        PRINT_ERROR("avgPool1dLayerInit: kernelSize must be > 0");
        exit(1);
    }
}

static void validateLayerQuantForAvgPool1d(layerQuant_t *lq) {
    if (lq == NULL) {
        PRINT_ERROR("avgPool1dLayerInit: lq pointer is NULL");
        exit(1);
    }
    if (lq->outputQ == NULL) {
        PRINT_ERROR("avgPool1dLayerInit: layerQuant.outputQ must be set");
        exit(1);
    }
    if (lq->propLossQ == NULL) {
        PRINT_ERROR("avgPool1dLayerInit: layerQuant.propLossQ must be set");
        exit(1);
    }
}

static layer_t *buildAvgPool1dLayerSkeleton(avgPool1dInit_t *init) {
    size_t stride = init->stride == 0 ? init->kernelSize : init->stride;

    kernel_t *kernel = reserveMemory(sizeof(kernel_t));
    /* AvgPool1d has no dilation (kernel doesn't support it); pass 1. */
    initKernel(kernel, init->kernelSize, init->padding, /*dilation*/ 1, stride);

    layer_t *layer = reserveMemory(sizeof(layer_t));
    layer->type = AVGPOOL1D;
    layerConfig_t *layerCfg = reserveMemory(sizeof(layerConfig_t));
    avgPool1dConfig_t *cfg = reserveMemory(sizeof(avgPool1dConfig_t));
    layerCfg->avgPool1d = cfg;
    layer->config = layerCfg;

    cfg->kernel = kernel;
    return layer;
}

layer_t *avgPool1dLayerInit(avgPool1dInit_t *init, layerQuant_t *lq) {
    validateAvgPool1dInit(init);
    validateLayerQuantForAvgPool1d(lq);

    layer_t *layer = buildAvgPool1dLayerSkeleton(init);
    avgPool1dConfig_t *cfg = layer->config->avgPool1d;
    cfg->forwardMath = lq->forwardMath;
    cfg->propLossMath = lq->propLossMath;
    cfg->outputQ = lq->outputQ;
    cfg->propLossQ = lq->propLossQ;
    cfg->ownsQuantizations = false;
    return layer;
}

layer_t *avgPool1dLayerInitOwning(avgPool1dInit_t *init, layerQuant_t *lq) {
    validateAvgPool1dInit(init);
    validateLayerQuantForAvgPool1d(lq);

    layer_t *layer = buildAvgPool1dLayerSkeleton(init);
    avgPool1dConfig_t *cfg = layer->config->avgPool1d;
    cfg->forwardMath = lq->forwardMath;
    cfg->propLossMath = lq->propLossMath;
    cfg->outputQ = deepCopyQuantization(lq->outputQ);
    cfg->propLossQ = deepCopyQuantization(lq->propLossQ);
    cfg->ownsQuantizations = true;
    return layer;
}

void freeAvgPool1dLayer(layer_t *layer) {
    if (layer == NULL) {
        return;
    }
    avgPool1dConfig_t *cfg = layer->config->avgPool1d;

    freeReservedMemory(cfg->kernel);

    if (cfg->ownsQuantizations) {
        if (cfg->outputQ != NULL) {
            freeQuantization(cfg->outputQ);
        }
        if (cfg->propLossQ != NULL && cfg->propLossQ != cfg->outputQ) {
            freeQuantization(cfg->propLossQ);
        }
    }

    freeReservedMemory(cfg);
    freeReservedMemory(layer->config);
    freeReservedMemory(layer);
}
