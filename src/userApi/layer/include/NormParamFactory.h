#ifndef ODT_NORM_PARAM_FACTORY_H
#define ODT_NORM_PARAM_FACTORY_H

#include <stdbool.h>
#include <stdlib.h>

#include "Common.h"
#include "LayerQuant.h"
#include "QuantizationApi.h"
#include "StorageApi.h"
#include "Tensor.h"
#include "TensorApi.h"

/* Factory-side helpers GroupNormApi and LayerNormApi share. `who` names the factory in
 * every message. */

/* Heap-owned shape with default order; ownership transfers to the tensor built on it. */
static inline shape_t *normBuildOwnedShape(const size_t *srcDims, size_t numberOfDims) {
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

/* tensorFillFromFloatBuffer quantizes the constant into the param's own storage dtype;
 * initDistribution would not do, it is FLOAT32-only. */
static inline void normFillParamWithConstant(tensor_t *paramTensor, float value) {
    size_t count = calcNumberOfElementsByTensor(paramTensor);
    float *buf = reserveMemory(count * sizeof(float));
    for (size_t i = 0; i < count; i++) {
        buf[i] = value;
    }
    tensorFillFromFloatBuffer(paramTensor, buf, count);
    freeReservedMemory(buf);
}

/* One layerQuant_t profile is shared across every layer of a model, so a grouped BFP
 * storage template's numGroups is a guess: keep its groupSize and derive numGroups from
 * this parameter's element count. getQLike would keep the guess and die later at
 * initTensor with a message that does not name the factory. */
static inline quantization_t *normParamQLike(quantization_t *storageQ, size_t numberOfValues,
                                             const char *who) {
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
        PRINT_ERROR("%s: BFP param storage groupSize %zu does not divide the "
                    "parameter's element count %zu -- pick a divisor or a per-tensor {1, 0} "
                    "template",
                    who, groupSize, numberOfValues);
        exit(1);
    }
    return quantizationInitBfpGrouped(src->mantissaBits, src->exponentBits, src->roundingMode,
                                      numberOfValues / groupSize, groupSize);
}

/* A constant-filled param of shape dims in storageQ's dtype, with a gradQ grad unless
 * !trainable. */
static inline parameter_t *normAllocAffineParam(const size_t *dims, size_t numberOfDims,
                                                float value, quantization_t *storageQ,
                                                quantization_t *gradQ, bool trainable,
                                                const char *who) {
    shape_t *shape = normBuildOwnedShape(dims, numberOfDims);
    size_t count = calcNumberOfElementsByShape(shape);
    tensor_t *paramTensor = initTensor(shape, normParamQLike(storageQ, count, who), NULL);
    normFillParamWithConstant(paramTensor, value);
    tensor_t *gradTensor = trainable ? gradInit(paramTensor, gradQ, NULL) : NULL;
    return parameterInit(paramTensor, gradTensor);
}

static inline bool normIsParamStorage(const quantization_t *q) {
    return q->type == FLOAT32 || q->type == SYM_INT32 || q->type == BFP;
}

/* Checks what the allocation needs; the layer's ValidateConfig checks the built config. */
static inline void normValidateLayerQuant(const layerQuant_t *lq, const char *who) {
    if (lq == NULL) {
        PRINT_ERROR("%s: lq pointer is NULL", who);
        exit(1);
    }
    if (lq->outputQ == NULL) {
        PRINT_ERROR("%s: layerQuant.outputQ must be set", who);
        exit(1);
    }
    if (lq->propLossQ == NULL) {
        PRINT_ERROR("%s: layerQuant.propLossQ must be set", who);
        exit(1);
    }
    if (lq->weightStorage == NULL) {
        PRINT_ERROR("%s: layerQuant.weightStorage must be set (gamma storage)", who);
        exit(1);
    }
    if (lq->biasStorage == NULL) {
        PRINT_ERROR("%s: layerQuant.biasStorage must be set (beta storage)", who);
        exit(1);
    }
    if (!normIsParamStorage(lq->weightStorage)) {
        PRINT_ERROR("%s: gamma storage must be FLOAT32, SYM_INT32 or BFP", who);
        exit(1);
    }
    if (!normIsParamStorage(lq->biasStorage)) {
        PRINT_ERROR("%s: beta storage must be FLOAT32, SYM_INT32 or BFP", who);
        exit(1);
    }
    /* A grouped BFP grad template is rejected by gradInit. */
}

/* gamma = 1 and beta = 0 of shape dims. A NULL weightGradStorage / biasGradStorage means
 * FLOAT32 grads: grads are persistent state, and SYM_INT32 is a compute format. */
static inline void normAllocGammaBeta(const layerQuant_t *lq, const size_t *dims,
                                      size_t numberOfDims, bool trainable, const char *who,
                                      parameter_t **gamma, parameter_t **beta) {
    quantization_t *floatGradQ = quantizationInitFloat();
    quantization_t *gammaGradQ = lq->weightGradStorage != NULL ? lq->weightGradStorage : floatGradQ;
    quantization_t *betaGradQ = lq->biasGradStorage != NULL ? lq->biasGradStorage : floatGradQ;
    *gamma = normAllocAffineParam(dims, numberOfDims, 1.0f, lq->weightStorage, gammaGradQ,
                                  trainable, who);
    *beta =
        normAllocAffineParam(dims, numberOfDims, 0.0f, lq->biasStorage, betaGradQ, trainable, who);
    freeQuantization(floatGradQ);
}

#endif // ODT_NORM_PARAM_FACTORY_H
