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
 * from THIS parameter's element count (the wire-allocator Decision-5 rule
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

/* NULL-guarded wire-config predicate: a hand-built layerQuant_t may carry NULL
 * wire configs, and the coherence rules below read them unconditionally. */
static bool isBfpTyped(const quantization_t *q) {
    return q != NULL && q->type == BFP;
}

static bool isGroupNormParamStorage(const quantization_t *q) {
    return q->type == FLOAT32 || q->type == SYM_INT32 || q->type == BFP;
}

/* Storage the ARITH_BFP arms can take as an operand: BFP is borrowed zero-copy,
 * FLOAT32 is staged into BFP scratch at the wire anchor. */
static bool isGroupNormBfpOperandStorage(const quantization_t *q) {
    return q->type == FLOAT32 || q->type == BFP;
}

/* Coherence rule set (BFP epic PR5 Task 6): R1 param dtypes, R2-R4 forwardMath
 * vs. param storage, R5-R7 propLossMath, R8 grad storage (delegated to
 * gradInit). Each rule is a straight-line check with one guided message. */
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
    /* R1: the three param dtypes a norm kernel can read at all. */
    if (!isGroupNormParamStorage(lq->weightStorage)) {
        PRINT_ERROR("groupNormLayerInit: gamma storage must be FLOAT32, SYM_INT32 or BFP");
        exit(1);
    }
    if (!isGroupNormParamStorage(lq->biasStorage)) {
        PRINT_ERROR("groupNormLayerInit: beta storage must be FLOAT32, SYM_INT32 or BFP");
        exit(1);
    }
    /* R2: the SYM_INT32 kernels read gamma/beta as int32 mantissas; a float
     * buffer read that way is silent garbage. */
    if (lq->forwardMath.type == ARITH_SYM_INT32 &&
        (lq->weightStorage->type != SYM_INT32 || lq->biasStorage->type != SYM_INT32)) {
        PRINT_ERROR("groupNormLayerInit: SYM_INT32 forwardMath requires SYM_INT32 gamma AND beta "
                    "storage");
        exit(1);
    }
    /* R3: the ARITH_BFP forward takes gamma/beta in the funnel's unpacked-BFP
     * scratch form — borrowed from BFP storage or staged from FLOAT32 — and
     * anchors that staging on outputQ (norms have no reduction-weight operand,
     * R-N1). Checked eagerly here so a mis-wired profile dies at construction
     * with this message instead of at the first forward. */
    if (lq->forwardMath.type == ARITH_BFP) {
        if (!isGroupNormBfpOperandStorage(lq->weightStorage) ||
            !isGroupNormBfpOperandStorage(lq->biasStorage)) {
            PRINT_ERROR("groupNormLayerInit: ARITH_BFP forwardMath requires FLOAT32 (staged) or "
                        "BFP (borrowed) gamma AND beta storage");
            exit(1);
        }
        if (!isBfpTyped(lq->outputQ)) {
            PRINT_ERROR("groupNormLayerInit: ARITH_BFP forwardMath requires a BFP-typed outputQ — "
                        "it is the staging width anchor (docs/conventions/arithmetic-bfp.md)");
            exit(1);
        }
    }
    /* R4: FLOAT32 math over BFP/SYM params stays UNCONSTRUCTIBLE for the norms,
     * deliberately unlike the GEMM family's fake-quant profile: the norms'
     * FLOAT32 BACKWARD raw-casts gamma and rejects any non-FLOAT32 operand, so
     * such a layer would forward fine and die at the first backward — a trap,
     * not a feature. The GEMM layers support it because their backward is
     * funnel-routed; the norms' is not. Fake-quant-forward experiments pin
     * cfg->forwardMath directly on a hand-wired config. */
    if (lq->forwardMath.type == ARITH_FLOAT32 &&
        (lq->weightStorage->type != FLOAT32 || lq->biasStorage->type != FLOAT32)) {
        PRINT_ERROR("groupNormLayerInit: ARITH_FLOAT32 forwardMath requires FLOAT32 gamma AND beta "
                    "storage — the FLOAT32 norm backward raw-casts gamma; declare ARITH_BFP / "
                    "ARITH_SYM_INT32 math, or insert a Quantization layer");
        exit(1);
    }
    /* R5: the SYM_INT32 backward recomputes group stats from forwardInput's
     * int32 mantissas and reads gamma mantissas; with a FLOAT32 forward those
     * buffers hold float bits — silent garbage. The REVERSE (SYM forwardMath +
     * FLOAT32 backwardMath) stays constructible: it is the inference-only
     * profile, and the runtime backward guard rejects training it. */
    if (lq->propLossMath.type == ARITH_SYM_INT32 && lq->forwardMath.type != ARITH_SYM_INT32) {
        PRINT_ERROR("groupNormLayerInit: SYM_INT32 backwardMath requires SYM_INT32 forwardMath");
        exit(1);
    }
    /* R6: R3's backward twin — all three backward ops stage at propLossQ. A
     * FLOAT32 forward with an ARITH_BFP backward is mechanically legal and
     * stays allowed (the operands stage at the propLossQ anchor). */
    if (lq->propLossMath.type == ARITH_BFP) {
        if (!isGroupNormBfpOperandStorage(lq->weightStorage) ||
            !isGroupNormBfpOperandStorage(lq->biasStorage)) {
            PRINT_ERROR("groupNormLayerInit: ARITH_BFP propLossMath requires FLOAT32 (staged) or "
                        "BFP (borrowed) gamma AND beta storage");
            exit(1);
        }
        if (!isBfpTyped(lq->propLossQ)) {
            PRINT_ERROR("groupNormLayerInit: ARITH_BFP propLossMath requires a BFP-typed propLossQ "
                        "— it is the staging width anchor (docs/conventions/arithmetic-bfp.md)");
            exit(1);
        }
    }
    /* R7: the factory-level mirror of the backward's op-entry guards. The
     * FLOAT32 norm backward raw-casts EVERY operand, so a single BFP among the
     * params or the wire configs makes it read packed payloads as floats. */
    if (lq->propLossMath.type == ARITH_FLOAT32 &&
        (lq->weightStorage->type == BFP || lq->biasStorage->type == BFP ||
         isBfpTyped(lq->outputQ) || isBfpTyped(lq->propLossQ))) {
        PRINT_ERROR("groupNormLayerInit: ARITH_FLOAT32 propLossMath rejects BFP gamma/beta storage "
                    "and BFP wire configs — the FLOAT32 norm backward raw-casts; keep wires and "
                    "params FLOAT32, declare ARITH_BFP, or insert a Quantization layer");
        exit(1);
    }
    /* R8 (weightGradStorage/biasGradStorage): NULL falls back to FLOAT32 (#261)
     * and a non-NULL template flows to gradInit, whose per-tensor-only BFP
     * carrier gate rejects grouped templates. No duplicate gate here. */
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
