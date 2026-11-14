#define SOURCE_FILE "LINEAR_API"

#include <stdbool.h>
#include <stdlib.h>

#include "ArithmeticType.h"
#include "Common.h"
#include "ExecuteOp.h"
#include "Layer.h"
#include "LayerCommon.h"
#include "LayerQuant.h"
#include "Linear.h"
#include "LinearApi.h"
#include "QuantizationApi.h"
#include "StorageApi.h"
#include "Tensor.h"
#include "TensorApi.h"

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

static parameter_t *allocateLinearWeights(size_t inFeatures, size_t outFeatures,
                                          weightInit_t weightInit, quantization_t *storageQ,
                                          quantization_t *gradQ, bool trainable) {
    shape_t *shape = buildOwnedShape((size_t[]){outFeatures, inFeatures}, 2);
    tensor_t *paramTensor = initTensor(shape, getQLike(storageQ), NULL);

    /* PyTorch's fan_in/fan_out for a 2-D weight. */
    initWeightTensor(paramTensor, weightInit, inFeatures, outFeatures);

    tensor_t *gradTensor = trainable ? gradInit(paramTensor, gradQ, NULL) : NULL;
    return parameterInit(paramTensor, gradTensor);
}

static parameter_t *allocateLinearBias(size_t outFeatures, size_t fanIn, quantization_t *storageQ,
                                       quantization_t *gradQ, bool trainable,
                                       const rngKey_t *initKey) {
    /* PyTorch draws the bias bound from the weight's fan_in. */
    shape_t *shape = buildOwnedShape((size_t[]){outFeatures}, 1);
    tensor_t *paramTensor = initTensor(shape, getQLike(storageQ), NULL);
    initBiasTensor(paramTensor, fanIn, initKey);

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

static void validateLinearArithmetic(arithmetic_t math, const char *slot) {
    if (math.type != ARITH_FLOAT32 && math.type != ARITH_SYM_INT32 && math.type != ARITH_BFP) {
        PRINT_ERROR("linearLayerInit: layerQuant.%s must be ARITH_FLOAT32, ARITH_SYM_INT32 or "
                    "ARITH_BFP (got %d)",
                    slot, (int)math.type);
        exit(1);
    }
}

static void validateLayerQuantForLinear(layerQuant_t *lq, bool hasBias, bool trainable) {
    if (lq == NULL) {
        PRINT_ERROR("linearLayerInit: lq pointer is NULL");
        exit(1);
    }
    validateLinearArithmetic(lq->forwardMath, "forwardMath");
    validateLinearArithmetic(lq->weightGradMath, "weightGradMath");
    validateLinearArithmetic(lq->biasGradMath, "biasGradMath");
    validateLinearArithmetic(lq->propLossMath, "propLossMath");
    /* A frozen layer never accumulates grads, so its modes may stay unset. */
    if (trainable) {
        executeOpValidateAccMode(lq->weightGradAccMode,
                                 "linearLayerInit: layerQuant.weightGradAccMode");
        if (hasBias) {
            executeOpValidateAccMode(lq->biasGradAccMode,
                                     "linearLayerInit: layerQuant.biasGradAccMode");
        }
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

/* The one construction path; the public factories differ only in whether outputQ and
 * propLossQ are borrowed or copied. `who` names the public factory. */
static layer_t *linearLayerBuild(linearInit_t *init, layerQuant_t *lq, bool owning,
                                 const char *who) {
    validateLinearInit(init);
    bool hasBias = resolveLinearBias(init->bias);
    bool trainable = resolveTrainable(init->trainable, who);
    validateLayerQuantForLinear(lq, hasBias, trainable);
    validateRandomInit(init->weightInit, lq->weightStorage, hasBias ? lq->biasStorage : NULL,
                       "linearLayerInit");

    layer_t *layer = reserveMemory(sizeof(layer_t));
    layer->type = LINEAR;

    layerConfig_t *layerCfg = reserveMemory(sizeof(layerConfig_t));
    linearConfig_t *cfg = reserveMemory(sizeof(linearConfig_t));
    layerCfg->linear = cfg;
    layer->config = layerCfg;

    /* Parameter tensors own cloned storage configs. Gradient storage defaults to FLOAT32. */
    quantization_t *floatGradQ = quantizationInitFloat();
    quantization_t *weightGradQ =
        lq->weightGradStorage != NULL ? lq->weightGradStorage : floatGradQ;
    quantization_t *biasGradQ = lq->biasGradStorage != NULL ? lq->biasGradStorage : floatGradQ;
    cfg->weights = allocateLinearWeights(init->inFeatures, init->outFeatures, init->weightInit,
                                         lq->weightStorage, weightGradQ, trainable);
    cfg->bias = hasBias ? allocateLinearBias(init->outFeatures, init->inFeatures, lq->biasStorage,
                                             biasGradQ, trainable, init->weightInit.key)
                        : NULL;
    freeQuantization(floatGradQ);

    cfg->forwardMath = lq->forwardMath;
    cfg->weightGradMath = lq->weightGradMath;
    cfg->biasGradMath = lq->biasGradMath;
    cfg->propLossMath = lq->propLossMath;
    cfg->outputQ = owning ? deepCopyQuantization(lq->outputQ) : lq->outputQ;
    cfg->propLossQ = owning ? deepCopyQuantization(lq->propLossQ) : lq->propLossQ;
    cfg->weightGradAccMode = lq->weightGradAccMode;
    cfg->biasGradAccMode = lq->biasGradAccMode;
    cfg->ownsQuantizations = owning;
    cfg->frozen = !trainable;
    return layer;
}

layer_t *linearLayerInit(linearInit_t *init, layerQuant_t *lq) {
    return linearLayerBuild(init, lq, false, "linearLayerInit");
}

layer_t *linearLayerInitOwning(linearInit_t *init, layerQuant_t *lq) {
    return linearLayerBuild(init, lq, true, "linearLayerInitOwning");
}

void freeLinearLayer(layer_t *linearLayer) {
    if (linearLayer == NULL) {
        return;
    }
    linearConfig_t *cfg = linearLayer->config->linear;

    freeParameter(cfg->weights);
    if (cfg->bias != NULL) {
        freeParameter(cfg->bias);
    }
    /* freeQuantization, not a plain free: grouped and BFP copies own backing arrays. */
    if (cfg->ownsQuantizations) {
        freeQuantization(cfg->outputQ);
        freeQuantization(cfg->propLossQ);
    }

    freeReservedMemory(cfg);
    freeReservedMemory(linearLayer->config);
    freeReservedMemory(linearLayer);
}
