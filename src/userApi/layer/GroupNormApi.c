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

/* R-N6 (BFP epic PR5): a grouped BFP storage template's numGroups is a
 * shape-agnostic guess -- one layerQuant_t profile is shared across every layer
 * of a model -- so honor the template's groupSize ONLY and derive numGroups
 * from THIS parameter's element count (the wire-allocator BFP PR2-Decision-5 rule
 * applied to params). getQLike would preserve the guess verbatim and die at
 * initTensor's attach validation (validateBfpQConfigShape) with a message that
 * says nothing about where the geometry came from. */
static quantization_t *groupNormParamQLike(quantization_t *storageQ, size_t numberOfValues) {
    if (storageQ->type != BFP) {
        return getQLike(storageQ);
    }
    bfpQConfig_t *src = storageQ->qConfig;
    size_t groupSize = src->groupSize;
    if (groupSize == 0 || groupSize == numberOfValues) {
        /* {1, N} is not a constructible BFP shape -- normalize to per-tensor. */
        return quantizationInitBfp(src->mantissaBits, src->exponentBits, src->roundingMode);
    }
    if (numberOfValues % groupSize != 0) {
        PRINT_ERROR("groupNormLayerInit: BFP param storage groupSize %zu does not divide the "
                    "parameter's element count %zu -- pick a divisor or a per-tensor {1, 0} "
                    "template",
                    groupSize, numberOfValues);
        exit(1);
    }
    return quantizationInitBfpGrouped(src->mantissaBits, src->exponentBits, src->roundingMode,
                                      numberOfValues / groupSize, groupSize);
}

/* gamma: shape [numChannels], init all-ones (FLOAT32: 1.0f each; SYM_INT32:
 * mantissa 2047, scale 1/2047 (#227 int12 operand default); BFP: code 64 at
 * stored exponent bias-6 for m=8, i.e. exactly 1.0 -- all-ones is a grid
 * point); grad dtype from gradQ (= the profile's backwardMath). */
static parameter_t *allocateGroupNormGamma(size_t numChannels, quantization_t *storageQ,
                                           quantization_t *gradQ, bool trainable) {
    shape_t *shape = buildOwnedShape((size_t[]){numChannels}, 1);
    tensor_t *paramTensor = initTensor(shape, groupNormParamQLike(storageQ, numChannels), NULL);
    fillParamTensorWithConstant(paramTensor, 1.0f);
    tensor_t *gradTensor = trainable ? gradInit(paramTensor, gradQ, NULL) : NULL;
    return parameterInit(paramTensor, gradTensor);
}

/* beta: shape [numChannels], init all-zeros (FLOAT32: 0.0f each; SYM_INT32:
 * mantissa 0, scale 1.0; BFP: code 0 at the bias exponent — the explicit fill
 * exercises the absMax==0 constant guard of each quantizer instead of relying
 * on calloc zeros + the default grid). */
static parameter_t *allocateGroupNormBeta(size_t numChannels, quantization_t *storageQ,
                                          quantization_t *gradQ, bool trainable) {
    shape_t *shape = buildOwnedShape((size_t[]){numChannels}, 1);
    tensor_t *paramTensor = initTensor(shape, groupNormParamQLike(storageQ, numChannels), NULL);
    fillParamTensorWithConstant(paramTensor, 0.0f);
    tensor_t *gradTensor = trainable ? gradInit(paramTensor, gradQ, NULL) : NULL;
    return parameterInit(paramTensor, gradTensor);
}

static void validateGroupNormInit(groupNormInit_t *init) {
    if (init == NULL) {
        PRINT_ERROR("groupNormLayerInit: init pointer is NULL");
        exit(1);
    }
    /* groupNormValidateConfig checks the group geometry. */
    if (init->numChannels == 0) {
        PRINT_ERROR("groupNormLayerInit: numChannels must be > 0 (got 0)");
        exit(1);
    }
    if (init->eps < 0.0f) {
        PRINT_ERROR("groupNormLayerInit: eps must be >= 0 (got %f)", (double)init->eps);
        exit(1);
    }
}

static bool isGroupNormParamStorage(const quantization_t *q) {
    return q->type == FLOAT32 || q->type == SYM_INT32 || q->type == BFP;
}

/* Checks what the allocation needs; groupNormValidateConfig checks the built config. */
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
    if (!isGroupNormParamStorage(lq->weightStorage)) {
        PRINT_ERROR("groupNormLayerInit: gamma storage must be FLOAT32, SYM_INT32 or BFP");
        exit(1);
    }
    if (!isGroupNormParamStorage(lq->biasStorage)) {
        PRINT_ERROR("groupNormLayerInit: beta storage must be FLOAT32, SYM_INT32 or BFP");
        exit(1);
    }
    /* A grouped BFP grad template is rejected by gradInit. */
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
    bool trainable = resolveTrainable(init->trainable, "groupNormLayerInit");

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
    parameter_t *gamma =
        allocateGroupNormGamma(init->numChannels, lq->weightStorage, gammaGradQ, trainable);
    parameter_t *beta =
        allocateGroupNormBeta(init->numChannels, lq->biasStorage, betaGradQ, trainable);
    freeQuantization(floatGradQ);

    /* forwardQ/backwardQ filled by the variant below; pass NULL here. */
    initGroupNormConfig(cfg, gamma, beta, init->numGroups, init->numChannels, eps, NULL, NULL);
    cfg->frozen = !trainable;

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
    groupNormValidateConfig(cfg, "groupNormLayerInit");

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
    groupNormValidateConfig(cfg, "groupNormLayerInitOwning");

    return layer;
}

void freeGroupNormLayer(layer_t *layer) {
    if (layer == NULL) {
        return;
    }
    groupNormConfig_t *cfg = layer->config->groupNorm;

    /* The param tensors own their storage quantization (getQLike). */
    freeParameter(cfg->gamma);
    freeParameter(cfg->beta);
    /* The owning factory deep-copies outputQ and propLossQ into two distinct allocations. */
    if (cfg->ownsQuantizations) {
        freeQuantization(cfg->outputQ);
        freeQuantization(cfg->propLossQ);
    }

    freeReservedMemory(cfg);
    freeReservedMemory(layer->config);
    freeReservedMemory(layer);
}
