#define SOURCE_FILE "LINEAR_API"

#include <stdbool.h>
#include <stdlib.h>

#include "ArithmeticType.h"
#include "Common.h"
#include "Layer.h"
#include "LayerCommon.h"
#include "LayerQuant.h"
#include "Linear.h"
#include "LinearApi.h"
#include "QuantizationApi.h"
#include "StorageApi.h"
#include "Tensor.h"
#include "TensorApi.h"

/* ============================================================================
 * New factory API — layerQuant_t profile + linearInit_t struct (PR 1).
 * ========================================================================== */

static bool resolveLinearBias(bias_t b) {
    switch (b) {
    case BIAS_DEFAULT:
        return true; /* PyTorch parity for Linear */
    case BIAS_TRUE:
        return true;
    case BIAS_FALSE:
        return false;
    default:
        PRINT_ERROR("linearLayerInit: invalid bias value (got %d)", (int)b);
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

static parameter_t *allocateLinearWeights(size_t inFeatures, size_t outFeatures,
                                          weightInit_t weightInit, quantization_t *storageQ,
                                          quantization_t *gradQ, bool trainable) {
    /* Weight tensor: shape [outFeatures, inFeatures]. The tensor takes ownership
     * of `shape` and `quantization`, so we clone the borrowed storageQ via
     * getQLike to avoid tying the tensor's lifetime to the caller's quant. */
    shape_t *shape = buildOwnedShape((size_t[]){outFeatures, inFeatures}, 2);
    tensor_t *paramTensor = initTensor(shape, getQLike(storageQ), NULL);

    /* Linear: fan_in = inFeatures, fan_out = outFeatures (PyTorch
     * _calculate_fan_in_and_fan_out for a 2-D weight). Default scheme is
     * PyTorch parity: uniform(+/- 1/sqrt(fan_in)). */
    initWeightTensor(paramTensor, weightInit, inFeatures, outFeatures);

    tensor_t *gradTensor = trainable ? gradInit(paramTensor, gradQ, NULL) : NULL;
    return parameterInit(paramTensor, gradTensor);
}

static parameter_t *allocateLinearBias(size_t outFeatures, size_t fanIn, quantization_t *storageQ,
                                       quantization_t *gradQ, bool trainable) {
    /* Bias tensor: shape [outFeatures]. PyTorch draws bias from
     * uniform(+/- 1/sqrt(fan_in)) using the WEIGHT's fan_in (= inFeatures). */
    shape_t *shape = buildOwnedShape((size_t[]){outFeatures}, 1);
    tensor_t *paramTensor = initTensor(shape, getQLike(storageQ), NULL);
    initBiasTensor(paramTensor, fanIn);

    tensor_t *gradTensor = trainable ? gradInit(paramTensor, gradQ, NULL) : NULL;
    return parameterInit(paramTensor, gradTensor);
}

static void validateLinearInit(linearInit_t *init) {
    if (init == NULL) {
        PRINT_ERROR("linearLayerInit: init pointer is NULL");
        exit(1);
    }
    if (init->inFeatures == 0) {
        PRINT_ERROR("linearLayerInit: inFeatures must be > 0 (got 0)");
        exit(1);
    }
    if (init->outFeatures == 0) {
        PRINT_ERROR("linearLayerInit: outFeatures must be > 0 (got 0)");
        exit(1);
    }
}

static void validateLayerQuantForLinear(layerQuant_t *lq, bool hasBias) {
    if (lq == NULL) {
        PRINT_ERROR("linearLayerInit: lq pointer is NULL");
        exit(1);
    }
    if (lq->outputQ == NULL) {
        PRINT_ERROR("linearLayerInit: layerQuant.outputQ must be set");
        exit(1);
    }
    if (lq->propLossQ == NULL) {
        PRINT_ERROR("linearLayerInit: layerQuant.propLossQ must be set");
        exit(1);
    }
    if (lq->weightStorage == NULL) {
        PRINT_ERROR("linearLayerInit: layerQuant.weightStorage must be set");
        exit(1);
    }
    if (hasBias && lq->biasStorage == NULL) {
        PRINT_ERROR("linearLayerInit: layerQuant.biasStorage must be set when bias is enabled");
        exit(1);
    }
}

layer_t *linearLayerInit(linearInit_t *init, layerQuant_t *lq) {
    validateLinearInit(init);
    bool hasBias = resolveLinearBias(init->bias);
    validateLayerQuantForLinear(lq, hasBias);
    bool trainable = resolveTrainable(init->trainable, "linearLayerInit");

    layer_t *layer = reserveMemory(sizeof(layer_t));
    layer->type = LINEAR;

    layerConfig_t *layerCfg = reserveMemory(sizeof(layerConfig_t));
    linearConfig_t *cfg = reserveMemory(sizeof(linearConfig_t));
    layerCfg->linear = cfg;
    layer->config = layerCfg;

    /* Grad storage knob (#261, PR1c): NULL falls back to a hard-pinned FLOAT32
     * default (parameter grads are persistent state — SYM_INT32 is a compute
     * format, not storage); a non-NULL weightGradStorage/biasGradStorage
     * overrides it explicitly to opt back into SYM_INT32 (or another dtype). */
    quantization_t *floatGradQ = quantizationInitFloat();
    quantization_t *weightGradQ =
        lq->weightGradStorage != NULL ? lq->weightGradStorage : floatGradQ;
    quantization_t *biasGradQ = lq->biasGradStorage != NULL ? lq->biasGradStorage : floatGradQ;
    cfg->weights = allocateLinearWeights(init->inFeatures, init->outFeatures, init->weightInit,
                                         lq->weightStorage, weightGradQ, trainable);
    cfg->bias = hasBias ? allocateLinearBias(init->outFeatures, init->inFeatures, lq->biasStorage,
                                             biasGradQ, trainable)
                        : NULL;
    freeQuantization(floatGradQ);

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

layer_t *linearLayerInitOwning(linearInit_t *init, layerQuant_t *lq) {
    validateLinearInit(init);
    bool hasBias = resolveLinearBias(init->bias);
    validateLayerQuantForLinear(lq, hasBias);
    bool trainable = resolveTrainable(init->trainable, "linearLayerInitOwning");

    layer_t *layer = reserveMemory(sizeof(layer_t));
    layer->type = LINEAR;

    layerConfig_t *layerCfg = reserveMemory(sizeof(layerConfig_t));
    linearConfig_t *cfg = reserveMemory(sizeof(linearConfig_t));
    layerCfg->linear = cfg;
    layer->config = layerCfg;

    /* Allocate parameters using lq->weightStorage / lq->biasStorage as the
     * SOURCE of the tensor's owned quantization.  The allocator helpers (from
     * T12) internally clone via getQLike, so the parameter tensors hold their
     * own quantization_t copies — the caller can immediately drop the lq's
     * weightStorage/biasStorage pointers without breaking the parameters.
     * Grad storage knob (#261, PR1c): NULL falls back to a hard-pinned FLOAT32
     * default (parameter grads are persistent state — SYM_INT32 is a compute
     * format, not storage); a non-NULL weightGradStorage/biasGradStorage
     * overrides it explicitly to opt back into SYM_INT32 (or another dtype). */
    quantization_t *floatGradQ = quantizationInitFloat();
    quantization_t *weightGradQ =
        lq->weightGradStorage != NULL ? lq->weightGradStorage : floatGradQ;
    quantization_t *biasGradQ = lq->biasGradStorage != NULL ? lq->biasGradStorage : floatGradQ;
    cfg->weights = allocateLinearWeights(init->inFeatures, init->outFeatures, init->weightInit,
                                         lq->weightStorage, weightGradQ, trainable);
    cfg->bias = hasBias ? allocateLinearBias(init->outFeatures, init->inFeatures, lq->biasStorage,
                                             biasGradQ, trainable)
                        : NULL;
    freeQuantization(floatGradQ);

    /* Owning: same arithmetic as Borrowing; deep-copy the two storage configs
     * (outputQ, propLossQ) into fresh allocations — 2 allocs, not 3. */
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

void freeLinearLayer(layer_t *linearLayer) {
    if (linearLayer == NULL) {
        return;
    }
    linearConfig_t *cfg = linearLayer->config->linear;

    /* Tear down weight parameter (param tensor + grad tensor + their data + shape).
     * freeParameter handles the NULL-grad case post-#106. */
    if (cfg->weights != NULL) {
        freeParameter(cfg->weights);
    }
    if (cfg->bias != NULL) {
        freeParameter(cfg->bias);
    }

    if (cfg->ownsQuantizations) {
        /* freeQuantization, not freeReservedMemory(qConfig) + freeReservedMemory(q):
         * deepCopyQuantization gives SYM/ASYM/BFP copies their own inner heap
         * blocks (scales / scales+zeroPoints / exponents), and only
         * freeQuantization frees those (BFP epic PR2 Task 8). The same swap is
         * applied at every Owning teardown across src/userApi/layer/. */
        if (cfg->outputQ != NULL) {
            freeQuantization(cfg->outputQ);
        }
        if (cfg->propLossQ != NULL && cfg->propLossQ != cfg->outputQ) {
            freeQuantization(cfg->propLossQ);
        }
    }

    freeReservedMemory(cfg);
    freeReservedMemory(linearLayer->config);
    freeReservedMemory(linearLayer);
}
