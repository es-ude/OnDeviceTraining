#define SOURCE_FILE "GROUPNORM_API"

#include <stdlib.h>

#include "GroupNormApi.h"

#include "ArithmeticType.h"
#include "Common.h"
#include "GroupNorm.h"
#include "Layer.h"
#include "LayerQuant.h"
#include "QuantizationApi.h"
#include "StorageApi.h"
#include "Tensor.h"
#include "TensorApi.h"

#define GROUPNORM_DEFAULT_EPS 1e-5f

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

/* gamma: shape [numChannels], init all-ones (FLOAT32: 1.0f each; SYM_INT32:
 * mantissa 2047, scale 1/2047 (#227 int12 operand default)); grad dtype from
 * gradQ (= the profile's backwardMath). */
static parameter_t *allocateGroupNormGamma(size_t numChannels, quantization_t *storageQ,
                                           quantization_t *gradQ) {
    shape_t *shape = buildOwnedShape((size_t[]){numChannels}, 1);
    tensor_t *paramTensor = initTensor(shape, getQLike(storageQ), NULL);
    fillParamTensorWithConstant(paramTensor, 1.0f);
    tensor_t *gradTensor = gradInit(paramTensor, gradQ, NULL);
    return parameterInit(paramTensor, gradTensor);
}

/* beta: shape [numChannels], init all-zeros (FLOAT32: 0.0f each; SYM_INT32:
 * mantissa 0, scale 1.0 — the explicit fill exercises the absMax==0 constant
 * guard instead of relying on calloc zeros + the default scale). */
static parameter_t *allocateGroupNormBeta(size_t numChannels, quantization_t *storageQ,
                                          quantization_t *gradQ) {
    shape_t *shape = buildOwnedShape((size_t[]){numChannels}, 1);
    tensor_t *paramTensor = initTensor(shape, getQLike(storageQ), NULL);
    fillParamTensorWithConstant(paramTensor, 0.0f);
    tensor_t *gradTensor = gradInit(paramTensor, gradQ, NULL);
    return parameterInit(paramTensor, gradTensor);
}

static void validateGroupNormInit(groupNormInit_t *init) {
    if (init == NULL) {
        PRINT_ERROR("groupNormLayerInit: init pointer is NULL");
        exit(1);
    }
    if (init->numGroups == 0) {
        PRINT_ERROR("groupNormLayerInit: numGroups must be > 0 (got 0)");
        exit(1);
    }
    if (init->numChannels == 0) {
        PRINT_ERROR("groupNormLayerInit: numChannels must be > 0 (got 0)");
        exit(1);
    }
    if (init->numChannels % init->numGroups != 0) {
        PRINT_ERROR("groupNormLayerInit: numChannels (%zu) must be divisible by numGroups (%zu)",
                    init->numChannels, init->numGroups);
        exit(1);
    }
    if (init->eps < 0.0f) {
        PRINT_ERROR("groupNormLayerInit: eps must be >= 0 (got %f)", (double)init->eps);
        exit(1);
    }
}

static void validateLayerQuantForGroupNorm(layerQuant_t *lq) {
    if (lq == NULL) {
        PRINT_ERROR("groupNormLayerInit: lq pointer is NULL");
        exit(1);
    }
    if (lq->outputQ == NULL) {
        PRINT_ERROR("groupNormLayerInit: layerQuant.outputQ must be set");
        exit(1);
    }
    if (lq->propLossQ == NULL) {
        PRINT_ERROR("groupNormLayerInit: layerQuant.propLossQ must be set");
        exit(1);
    }
    if (lq->weightStorage == NULL) {
        PRINT_ERROR("groupNormLayerInit: layerQuant.weightStorage must be set (gamma storage)");
        exit(1);
    }
    if (lq->biasStorage == NULL) {
        PRINT_ERROR("groupNormLayerInit: layerQuant.biasStorage must be set (beta storage)");
        exit(1);
    }
    if (lq->weightStorage->type != FLOAT32 && lq->weightStorage->type != SYM_INT32) {
        PRINT_ERROR("groupNormLayerInit: gamma storage must be FLOAT32 or SYM_INT32");
        exit(1);
    }
    if (lq->biasStorage->type != FLOAT32 && lq->biasStorage->type != SYM_INT32) {
        PRINT_ERROR("groupNormLayerInit: beta storage must be FLOAT32 or SYM_INT32");
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
        PRINT_ERROR("groupNormLayerInit: gamma/beta storage type must match forwardMath");
        exit(1);
    }
    /* The SYM_INT32 backward recomputes group stats from forwardInput's int32
     * mantissas and reads gamma mantissas; with a FLOAT32 forward those
     * buffers hold float bits — silent garbage. The REVERSE (SYM forwardMath +
     * FLOAT32 backwardMath) stays constructible: it is the inference-only
     * profile, and the runtime backward guard rejects training it. */
    if (lq->propLossMath.type == ARITH_SYM_INT32 && lq->forwardMath.type != ARITH_SYM_INT32) {
        PRINT_ERROR("groupNormLayerInit: SYM_INT32 backwardMath requires SYM_INT32 forwardMath");
        exit(1);
    }
}

/* Shared scaffolding for both factories: validate, allocate the layer/config
 * wrappers, allocate gamma(=1)/beta(=0) + grads. Leaves forwardQ/backwardQ/
 * ownsQuantizations for the caller (the specific factory variant) to set.
 * Returns the layer; `*outCfg` receives the config so the variant can finish
 * wiring the math quant slots. */
static layer_t *groupNormLayerInitCommon(groupNormInit_t *init, layerQuant_t *lq,
                                         groupNormConfig_t **outCfg) {
    validateGroupNormInit(init);
    validateLayerQuantForGroupNorm(lq);

    float eps = (init->eps == 0.0f) ? GROUPNORM_DEFAULT_EPS : init->eps;

    layer_t *layer = reserveMemory(sizeof(layer_t));
    layer->type = GROUPNORM;

    layerConfig_t *layerCfg = reserveMemory(sizeof(layerConfig_t));
    groupNormConfig_t *cfg = reserveMemory(sizeof(groupNormConfig_t));
    layerCfg->groupNorm = cfg;
    layer->config = layerCfg;

    /* Grad storage knob (#261, PR1c): NULL falls back to a hard-pinned FLOAT32
     * default (parameter grads are persistent state — SYM_INT32 is a compute
     * format, not storage); a non-NULL weightGradStorage/biasGradStorage
     * overrides it explicitly to opt back into SYM_INT32 (or another dtype). */
    quantization_t *floatGradQ = quantizationInitFloat();
    quantization_t *gammaGradQ = lq->weightGradStorage != NULL ? lq->weightGradStorage : floatGradQ;
    quantization_t *betaGradQ = lq->biasGradStorage != NULL ? lq->biasGradStorage : floatGradQ;
    parameter_t *gamma = allocateGroupNormGamma(init->numChannels, lq->weightStorage, gammaGradQ);
    parameter_t *beta = allocateGroupNormBeta(init->numChannels, lq->biasStorage, betaGradQ);
    freeQuantization(floatGradQ);

    /* forwardQ/backwardQ filled by the variant below; pass NULL here. */
    initGroupNormConfig(cfg, gamma, beta, init->numGroups, init->numChannels, eps, NULL, NULL);

    *outCfg = cfg;
    return layer;
}

layer_t *groupNormLayerInit(groupNormInit_t *init, layerQuant_t *lq) {
    groupNormConfig_t *cfg;
    layer_t *layer = groupNormLayerInitCommon(init, lq, &cfg);

    /* Borrowing: store the storage pointers verbatim; the arithmetic slots are
     * plain by-value copies of lq's declared math. The caller owns
     * outputQ/propLossQ and frees them; freeGroupNormLayer leaves them
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

layer_t *groupNormLayerInitOwning(groupNormInit_t *init, layerQuant_t *lq) {
    groupNormConfig_t *cfg;
    layer_t *layer = groupNormLayerInitCommon(init, lq, &cfg);

    /* Owning: deep-copy each storage quantization so the caller can drop its
     * outputQ/propLossQ pointers immediately. freeGroupNormLayer tears
     * the copies down (ownsQuantizations=true). Mirrors layerNormLayerInitOwning. */
    cfg->forwardMath = lq->forwardMath;
    cfg->propLossMath = lq->propLossMath;
    cfg->outputQ = deepCopyQuantization(lq->outputQ);
    cfg->propLossQ = deepCopyQuantization(lq->propLossQ);
    cfg->weightGradAccMode = lq->weightGradAccMode;
    cfg->biasGradAccMode = lq->biasGradAccMode;
    cfg->ownsQuantizations = true;

    return layer;
}

void freeGroupNormLayer(layer_t *layer) {
    if (layer == NULL) {
        return;
    }
    groupNormConfig_t *cfg = layer->config->groupNorm;

    /* ALWAYS tear down the factory-allocated parameters (gamma/beta + their grad
     * tensors + data + shapes). The param tensors own their storage quant via
     * getQLike, so storage quant is freed here regardless of ownsQuantizations. */
    if (cfg->gamma != NULL) {
        freeParameter(cfg->gamma);
    }
    if (cfg->beta != NULL) {
        freeParameter(cfg->beta);
    }

    /* Owning-variant only — tear down the two storage quantization_t (qConfig +
     * struct). Dedup guard exactly like freeLayerNormLayer: free propLossQ only
     * if it is a distinct allocation from outputQ. The Owning factory always
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
