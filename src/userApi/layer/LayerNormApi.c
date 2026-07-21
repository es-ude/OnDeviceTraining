#define SOURCE_FILE "LAYERNORM_API"

#include <stdlib.h>

#include "LayerNormApi.h"

#include "ArithmeticType.h"
#include "Common.h"
#include "Layer.h"
#include "LayerNorm.h"
#include "LayerQuant.h"
#include "QuantizationApi.h"
#include "StorageApi.h"
#include "Tensor.h"
#include "TensorApi.h"

#define LAYERNORM_DEFAULT_EPS 1e-5f

/* Heap-owned shape_t with default orderOfDimensions; ownership transfers to the
 * tensor (freeTensor cascades into freeShape). */
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

/* Constant fill via tensorFillFromFloatBuffer: plain memcpy for FLOAT32; for
 * SYM_INT32 it routes through convertFloatTensorToSymInt32Tensor, which IS
 * the spec's parameter quantization (all-ones -> mantissa 2047, scale
 * 1/2047 (#227 int12 operand default); all-zeros -> mantissa 0, scale 1.0
 * via the absMax==0 guard).
 * initDistribution cannot be used here: it is FLOAT32-only by guard, and
 * extending it is Issue-C scope. */
static void fillParamTensorWithConstant(tensor_t *paramTensor, float value) {
    size_t count = calcNumberOfElementsByTensor(paramTensor);
    float *buf = reserveMemory(count * sizeof(float));
    for (size_t i = 0; i < count; i++) {
        buf[i] = value;
    }
    tensorFillFromFloatBuffer(paramTensor, buf, count);
    freeReservedMemory(buf);
}

/* gamma: shape normalizedShape, init all-ones (FLOAT32: 1.0f each; SYM_INT32:
 * mantissa 2047, scale 1/2047 (#227 int12 operand default)); grad dtype from
 * gradQ (= the profile's backwardMath). */
static parameter_t *allocateLayerNormGamma(const size_t *normalizedShape, size_t numNormDims,
                                           quantization_t *storageQ, quantization_t *gradQ,
                                           bool trainable) {
    shape_t *shape = buildOwnedShape(normalizedShape, numNormDims);
    tensor_t *paramTensor = initTensor(shape, getQLike(storageQ), NULL);
    fillParamTensorWithConstant(paramTensor, 1.0f);
    tensor_t *gradTensor = trainable ? gradInit(paramTensor, gradQ, NULL) : NULL;
    return parameterInit(paramTensor, gradTensor);
}

/* beta: shape normalizedShape, init all-zeros (FLOAT32: 0.0f each; SYM_INT32:
 * mantissa 0, scale 1.0 — the explicit fill exercises the absMax==0 constant
 * guard instead of relying on calloc zeros + the default scale). */
static parameter_t *allocateLayerNormBeta(const size_t *normalizedShape, size_t numNormDims,
                                          quantization_t *storageQ, quantization_t *gradQ,
                                          bool trainable) {
    shape_t *shape = buildOwnedShape(normalizedShape, numNormDims);
    tensor_t *paramTensor = initTensor(shape, getQLike(storageQ), NULL);
    fillParamTensorWithConstant(paramTensor, 0.0f);
    tensor_t *gradTensor = trainable ? gradInit(paramTensor, gradQ, NULL) : NULL;
    return parameterInit(paramTensor, gradTensor);
}

static void validateLayerNormInit(layerNormInit_t *init) {
    if (init == NULL) {
        PRINT_ERROR("layerNormLayerInit: init pointer is NULL");
        exit(1);
    }
    if (init->normalizedShape == NULL) {
        PRINT_ERROR("layerNormLayerInit: normalizedShape is NULL");
        exit(1);
    }
    if (init->numNormDims == 0) {
        PRINT_ERROR("layerNormLayerInit: numNormDims must be > 0 (got 0)");
        exit(1);
    }
    for (size_t d = 0; d < init->numNormDims; d++) {
        if (init->normalizedShape[d] == 0) {
            PRINT_ERROR("layerNormLayerInit: normalizedShape[%zu] must be > 0", d);
            exit(1);
        }
    }
    if (init->eps < 0.0f) {
        PRINT_ERROR("layerNormLayerInit: eps must be >= 0 (got %f)", (double)init->eps);
        exit(1);
    }
}

static void validateLayerQuantForLayerNorm(layerQuant_t *lq) {
    if (lq == NULL) {
        PRINT_ERROR("layerNormLayerInit: lq pointer is NULL");
        exit(1);
    }
    if (lq->outputQ == NULL) {
        PRINT_ERROR("layerNormLayerInit: layerQuant.outputQ must be set");
        exit(1);
    }
    if (lq->propLossQ == NULL) {
        PRINT_ERROR("layerNormLayerInit: layerQuant.propLossQ must be set");
        exit(1);
    }
    if (lq->weightStorage == NULL) {
        PRINT_ERROR("layerNormLayerInit: layerQuant.weightStorage must be set (gamma storage)");
        exit(1);
    }
    if (lq->biasStorage == NULL) {
        PRINT_ERROR("layerNormLayerInit: layerQuant.biasStorage must be set (beta storage)");
        exit(1);
    }
    if (lq->weightStorage->type != FLOAT32 && lq->weightStorage->type != SYM_INT32) {
        PRINT_ERROR("layerNormLayerInit: gamma storage must be FLOAT32 or SYM_INT32");
        exit(1);
    }
    if (lq->biasStorage->type != FLOAT32 && lq->biasStorage->type != SYM_INT32) {
        PRINT_ERROR("layerNormLayerInit: beta storage must be FLOAT32 or SYM_INT32");
        exit(1);
    }
    /* The kernels read gamma/beta in the forward dtype: a FLOAT32 kernel over
     * SYM mantissas (or vice versa) is silent garbage. Fail at construction.
     * lq->forwardMath is now the declared arithmetic directly (by value); the
     * storage dtype is bridged through the same derivation the runtime uses
     * so a storage-only dtype (ASYM/SYM/BOOL/INT32) compares against its
     * ARITH_FLOAT32 bridge, not its raw qtype_t. */
    if (arithmeticFromQuantization(lq->weightStorage).type != lq->forwardMath.type ||
        arithmeticFromQuantization(lq->biasStorage).type != lq->forwardMath.type) {
        PRINT_ERROR("layerNormLayerInit: gamma/beta storage type must match forwardMath");
        exit(1);
    }
    /* The SYM_INT32 backward recomputes group stats from forwardInput's int32
     * mantissas and reads gamma mantissas; with a FLOAT32 forward those
     * buffers hold float bits — silent garbage. The REVERSE (SYM forwardMath +
     * FLOAT32 backwardMath) stays constructible: it is the inference-only
     * profile, and the runtime backward guard rejects training it. */
    if (lq->propLossMath.type == ARITH_SYM_INT32 && lq->forwardMath.type != ARITH_SYM_INT32) {
        PRINT_ERROR("layerNormLayerInit: SYM_INT32 backwardMath requires SYM_INT32 forwardMath");
        exit(1);
    }
}

/* Shared scaffolding for both factories: validate, allocate the layer/config
 * wrappers, allocate gamma(=1)/beta(=0) + grads, and copy normalizedShape into
 * factory-owned memory. Leaves forwardQ/backwardQ/ownsQuantizations for the
 * caller (the specific factory variant) to set. Returns the layer; `*outCfg`
 * receives the config so the variant can finish wiring the math quant slots. */
static layer_t *layerNormLayerInitCommon(layerNormInit_t *init, layerQuant_t *lq,
                                         layerNormConfig_t **outCfg) {
    validateLayerNormInit(init);
    validateLayerQuantForLayerNorm(lq);
    bool trainable = resolveTrainable(init->trainable, "layerNormLayerInit");

    float eps = (init->eps == 0.0f) ? LAYERNORM_DEFAULT_EPS : init->eps;

    layer_t *layer = reserveMemory(sizeof(layer_t));
    layer->type = LAYERNORM;

    layerConfig_t *layerCfg = reserveMemory(sizeof(layerConfig_t));
    layerNormConfig_t *cfg = reserveMemory(sizeof(layerNormConfig_t));
    layerCfg->layerNorm = cfg;
    layer->config = layerCfg;

    /* Grad storage knob (#261, PR1c): NULL falls back to a hard-pinned FLOAT32
     * default (parameter grads are persistent state — SYM_INT32 is a compute
     * format, not storage); a non-NULL weightGradStorage/biasGradStorage
     * overrides it explicitly to opt back into SYM_INT32 (or another dtype). */
    quantization_t *floatGradQ = quantizationInitFloat();
    quantization_t *gammaGradQ = lq->weightGradStorage != NULL ? lq->weightGradStorage : floatGradQ;
    quantization_t *betaGradQ = lq->biasGradStorage != NULL ? lq->biasGradStorage : floatGradQ;
    parameter_t *gamma = allocateLayerNormGamma(init->normalizedShape, init->numNormDims,
                                                lq->weightStorage, gammaGradQ, trainable);
    parameter_t *beta = allocateLayerNormBeta(init->normalizedShape, init->numNormDims,
                                              lq->biasStorage, betaGradQ, trainable);
    freeQuantization(floatGradQ);

    /* Factory-owned copy of normalizedShape (caller may free its own array). */
    size_t *normShapeCopy = reserveMemory(init->numNormDims * sizeof(size_t));
    for (size_t d = 0; d < init->numNormDims; d++) {
        normShapeCopy[d] = init->normalizedShape[d];
    }

    /* forwardQ/backwardQ filled by the variant below; pass NULL here. */
    initLayerNormConfig(cfg, gamma, beta, normShapeCopy, init->numNormDims, eps, NULL, NULL);
    cfg->frozen = !trainable;

    *outCfg = cfg;
    return layer;
}

layer_t *layerNormLayerInit(layerNormInit_t *init, layerQuant_t *lq) {
    layerNormConfig_t *cfg;
    layer_t *layer = layerNormLayerInitCommon(init, lq, &cfg);

    /* Borrowing: store the storage pointers verbatim; the arithmetic slots are
     * plain by-value copies of lq's declared math. The caller owns
     * outputQ/propLossQ and frees them; freeLayerNormLayer leaves them
     * untouched (ownsQuantizations=false). */
    cfg->forwardMath = lq->forwardMath;
    cfg->propLossMath = lq->propLossMath;
    cfg->outputQ = lq->outputQ;
    cfg->propLossQ = lq->propLossQ;
    cfg->weightGradAccMode = lq->weightGradAccMode;
    cfg->biasGradAccMode = lq->biasGradAccMode;
    cfg->ownsQuantizations = false;

    return layer;
}

layer_t *layerNormLayerInitOwning(layerNormInit_t *init, layerQuant_t *lq) {
    layerNormConfig_t *cfg;
    layer_t *layer = layerNormLayerInitCommon(init, lq, &cfg);

    /* Owning: deep-copy each storage quantization so the caller can drop its
     * outputQ/propLossQ pointers immediately. freeLayerNormLayer tears
     * the copies down (ownsQuantizations=true). Mirrors linearLayerInitOwning. */
    cfg->forwardMath = lq->forwardMath;
    cfg->propLossMath = lq->propLossMath;
    cfg->outputQ = deepCopyQuantization(lq->outputQ);
    cfg->propLossQ = deepCopyQuantization(lq->propLossQ);
    cfg->weightGradAccMode = lq->weightGradAccMode;
    cfg->biasGradAccMode = lq->biasGradAccMode;
    cfg->ownsQuantizations = true;

    return layer;
}

void freeLayerNormLayer(layer_t *layer) {
    if (layer == NULL) {
        return;
    }
    layerNormConfig_t *cfg = layer->config->layerNorm;

    /* ALWAYS tear down the factory-allocated parameters (gamma/beta + their grad
     * tensors + data + shapes). The param tensors own their storage quant via
     * getQLike, so storage quant is freed here regardless of ownsQuantizations. */
    if (cfg->gamma != NULL) {
        freeParameter(cfg->gamma);
    }
    if (cfg->beta != NULL) {
        freeParameter(cfg->beta);
    }
    freeReservedMemory(cfg->normalizedShape);

    /* Owning-variant only — tear down the two storage quantization_t (qConfig +
     * struct). Dedup guard exactly like freeLinearLayer: free propLossQ only if
     * it is a distinct allocation from outputQ. The Owning factory always
     * deep-copies into two separate instances, so the guard is a defensive
     * measure; the Borrowing variant has ownsQuantizations=false and skips this
     * branch (caller frees them). */
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
