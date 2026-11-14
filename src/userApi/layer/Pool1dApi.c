#define SOURCE_FILE "POOL1D_API"

#include <stdbool.h>
#include <stdlib.h>

#include "ArithmeticType.h"
#include "AvgPool1d.h"
#include "CheckedSize.h"
#include "Common.h"
#include "Kernel.h"
#include "Layer.h"
#include "LayerQuant.h"
#include "MaxPool1d.h"
#include "Pool1dApi.h"
#include "QuantizationApi.h"
#include "SlidingWindow1d.h"
#include "StorageApi.h"
#include "Tensor.h"
#include "TensorApi.h"

/* ============================================================================
 * Shared helpers
 * ========================================================================== */

/*! Derive the initial argmax length from the pooling geometry. Unlike the
 *  runtime geometry, which yields zero outputs, the factory rejects a non-SAME
 *  effective kernel longer than the declared input. kernel->size must be > 0. */
static size_t computePool1dOutputLength(const kernel_t *kernel, size_t inputLength) {
    size_t span = 0;
    size_t effectiveK = 0;
    if (!checkedMulSize(kernel->dilation, kernel->size - 1, &span) ||
        !checkedAddSize(span, 1, &effectiveK)) {
        PRINT_ERROR("Pool1d: effective kernel dilation (%zu) * (kernelSize (%zu) - 1) + 1 "
                    "overflows size_t",
                    kernel->dilation, kernel->size);
        exit(1);
    }
    if (kernel->paddingType != SAME && effectiveK > inputLength) {
        PRINT_ERROR("Pool1d: effective kernel %zu exceeds inputLength %zu", effectiveK,
                    inputLength);
        exit(1);
    }
    return windowGeometry1dCalc(inputLength, kernel).outputLength;
}

static bool isPoolArithmetic(arithmeticType_t type) {
    return type == ARITH_FLOAT32 || type == ARITH_SYM_INT32 || type == ARITH_BFP;
}

/* The ARITH_BFP kernels take their FLOAT32-operand staging widths from the
 * produced-wire config, so that config must be BFP. */
static void validatePoolMath(const layerQuant_t *lq, const char *where) {
    if (!isPoolArithmetic(lq->forwardMath.type) || !isPoolArithmetic(lq->propLossMath.type)) {
        PRINT_ERROR("%s: layerQuant.forwardMath and propLossMath must be ARITH_FLOAT32, "
                    "ARITH_SYM_INT32 or ARITH_BFP",
                    where);
        exit(1);
    }
    if (lq->forwardMath.type == ARITH_BFP && lq->outputQ->type != BFP) {
        PRINT_ERROR("%s: ARITH_BFP forwardMath requires a BFP-typed outputQ", where);
        exit(1);
    }
    if (lq->propLossMath.type == ARITH_BFP && lq->propLossQ->type != BFP) {
        PRINT_ERROR("%s: ARITH_BFP propLossMath requires a BFP-typed propLossQ", where);
        exit(1);
    }
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
    validatePoolMath(lq, "maxPool1dLayerInit");
}

static tensor_t *buildMaxPool1dArgmax(size_t inputChannels, size_t outputLength) {
    /* Created for batch=1: [1, inputChannels, outputLength]. maxPool1dForward
     * grows it on demand to the largest batch it is called with, e.g. a
     * stacked micro-batch or a batched inference() call. */
    shape_t *shape = buildOwnedShape((size_t[]){1, inputChannels, outputLength}, 3);
    quantization_t *q = quantizationInitInt32();
    return initTensor(shape, q, NULL);
}

static layer_t *buildMaxPool1dLayerSkeleton(maxPool1dInit_t *init) {
    size_t stride = init->stride == 0 ? init->kernelSize : init->stride;
    size_t dilation = init->dilation == 0 ? 1 : init->dilation;

    kernel_t *kernel = reserveMemory(sizeof(kernel_t));
    initKernel(kernel, init->kernelSize, init->padding, dilation, stride);

    size_t outputLength = computePool1dOutputLength(kernel, init->inputLength);
    tensor_t *argmax = buildMaxPool1dArgmax(init->inputChannels, outputLength);

    layer_t *layer = reserveMemory(sizeof(layer_t));
    layer->type = MAXPOOL1D;
    layerConfig_t *layerCfg = reserveMemory(sizeof(layerConfig_t));
    maxPool1dConfig_t *cfg = reserveMemory(sizeof(maxPool1dConfig_t));
    layerCfg->maxPool1d = cfg;
    layer->config = layerCfg;

    cfg->kernel = kernel;
    cfg->argmaxIndices = argmax;
    cfg->argmaxCapacity = calcNumberOfElementsByTensor(argmax);
    cfg->argmaxCapacityData = argmax->data;
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
    validatePoolMath(lq, "avgPool1dLayerInit");
}

/* Ruling R1: under BFP math (either slot) the mean is an exponent shift, so
 * the kernel size -- the divisor of every output, count_include_pad -- must
 * be a power of two. Checked at creation; the integer kernels check again
 * (a config built without the factory). */
static void validateAvgPool1dBfpDivisor(avgPool1dInit_t *init, layerQuant_t *lq) {
    if (lq->forwardMath.type == ARITH_BFP || lq->propLossMath.type == ARITH_BFP) {
        (void)avgPool1dBfpLog2Divisor(init->kernelSize, "avgPool1dLayerInit");
    }
}

static layer_t *buildAvgPool1dLayerSkeleton(avgPool1dInit_t *init) {
    size_t stride = init->stride == 0 ? init->kernelSize : init->stride;

    kernel_t *kernel = reserveMemory(sizeof(kernel_t));
    /* The public AvgPool1d factory fixes dilation to 1. */
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
    validateAvgPool1dBfpDivisor(init, lq);

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
    validateAvgPool1dBfpDivisor(init, lq);

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
