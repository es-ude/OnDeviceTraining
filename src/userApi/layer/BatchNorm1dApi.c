#define SOURCE_FILE "BATCHNORM1D_API"

#include <math.h>
#include <stdlib.h>

#include "BatchNorm1dApi.h"

#include "ArithmeticType.h"
#include "BatchNorm1d.h"
#include "Common.h"
#include "Layer.h"
#include "LayerQuant.h"
#include "QuantizationApi.h"
#include "StorageApi.h"
#include "Tensor.h"
#include "TensorApi.h"

#define BATCHNORM1D_DEFAULT_EPS 1e-5f

static tensor_t *allocFloatVector(size_t numChannels, float value) {
    size_t *dims = reserveMemory(sizeof(size_t));
    dims[0] = numChannels;
    size_t *order = reserveMemory(sizeof(size_t));
    setOrderOfDimsForNewTensor(1, order);
    shape_t *shape = reserveMemory(sizeof(shape_t));
    setShape(shape, dims, 1, order);
    tensor_t *t = initTensor(shape, quantizationInitFloat(), NULL);
    float *d = (float *)t->data;
    for (size_t c = 0; c < numChannels; c++) {
        d[c] = value;
    }
    return t;
}

static parameter_t *allocFloatParam(size_t numChannels, float value, bool withGrad) {
    tensor_t *p = allocFloatVector(numChannels, value);
    tensor_t *g = withGrad ? gradInitFloat(p, NULL) : NULL;
    return parameterInit(p, g);
}

static void validateBatchNorm1dInit(const batchNorm1dInit_t *init) {
    if (init == NULL) {
        PRINT_ERROR("batchNorm1dLayerInit: init pointer is NULL");
        exit(1);
    }
    if (init->numChannels == 0) {
        PRINT_ERROR("batchNorm1dLayerInit: numChannels must be > 0 (got 0)");
        exit(1);
    }
    if (!isfinite(init->eps) || init->eps < 0.0f) {
        PRINT_ERROR("batchNorm1dLayerInit: eps must be finite and >= 0 (got %f)",
                    (double)init->eps);
        exit(1);
    }
    if (init->momentumMode != BN_MOMENTUM_DEFAULT && init->momentumMode != BN_MOMENTUM_VALUE &&
        init->momentumMode != BN_MOMENTUM_CUMULATIVE) {
        PRINT_ERROR("batchNorm1dLayerInit: unknown momentumMode %d", (int)init->momentumMode);
        exit(1);
    }
    if (init->momentumMode == BN_MOMENTUM_VALUE &&
        (!isfinite(init->momentum) || init->momentum < 0.0f || init->momentum > 1.0f)) {
        PRINT_ERROR("batchNorm1dLayerInit: momentum must be finite and in [0, 1] (got %f)",
                    (double)init->momentum);
        exit(1);
    }
    if (init->noAffine && init->trainable == TRAINABLE_TRUE) {
        PRINT_ERROR("batchNorm1dLayerInit: noAffine has no gamma/beta to train -- "
                    "TRAINABLE_TRUE is contradictory (use TRAINABLE_DEFAULT or TRAINABLE_FALSE)");
        exit(1);
    }
}

static void requireFloat32Storage(const quantization_t *q, const char *slot, bool required) {
    if (q == NULL) {
        if (required) {
            PRINT_ERROR("batchNorm1dLayerInit: layerQuant.%s must be set", slot);
            exit(1);
        }
        return;
    }
    if (q->type != FLOAT32) {
        PRINT_ERROR("batchNorm1dLayerInit: layerQuant.%s must be FLOAT32 (got dtype %d) -- "
                    "BatchNorm1d is FLOAT32-only (#152 D3)",
                    slot, (int)q->type);
        exit(1);
    }
}

static void validateLayerQuantForBatchNorm1d(const layerQuant_t *lq, bool affine) {
    if (lq == NULL) {
        PRINT_ERROR("batchNorm1dLayerInit: lq pointer is NULL");
        exit(1);
    }
    if (lq->forwardMath.type != ARITH_FLOAT32 || lq->propLossMath.type != ARITH_FLOAT32) {
        PRINT_ERROR("batchNorm1dLayerInit: forwardMath and propLossMath must be ARITH_FLOAT32 -- "
                    "BatchNorm1d is FLOAT32-only (#152 D3)");
        exit(1);
    }
    requireFloat32Storage(lq->outputQ, "outputQ", true);
    requireFloat32Storage(lq->propLossQ, "propLossQ", true);
    if (affine) {
        requireFloat32Storage(lq->weightStorage, "weightStorage (gamma)", true);
        requireFloat32Storage(lq->biasStorage, "biasStorage (beta)", true);
        requireFloat32Storage(lq->weightGradStorage, "weightGradStorage", false);
        requireFloat32Storage(lq->biasGradStorage, "biasGradStorage", false);
    }
}

static layer_t *batchNorm1dLayerInitCommon(batchNorm1dInit_t *init, layerQuant_t *lq,
                                           batchNorm1dConfig_t **outCfg) {
    validateBatchNorm1dInit(init);
    bool affine = !init->noAffine;
    validateLayerQuantForBatchNorm1d(lq, affine);
    bool trainable = resolveTrainable(init->trainable, "batchNorm1dLayerInit");
    float eps = init->eps == 0.0f ? BATCHNORM1D_DEFAULT_EPS : init->eps;
    size_t C = init->numChannels;

    layer_t *layer = reserveMemory(sizeof(layer_t));
    layer->type = BATCHNORM1D;
    layerConfig_t *layerCfg = reserveMemory(sizeof(layerConfig_t));
    batchNorm1dConfig_t *cfg = reserveMemory(sizeof(batchNorm1dConfig_t));
    layerCfg->batchNorm1d = cfg;
    layer->config = layerCfg;

    parameter_t *gamma = affine ? allocFloatParam(C, 1.0f, trainable) : NULL;
    parameter_t *beta = affine ? allocFloatParam(C, 0.0f, trainable) : NULL;
    tensor_t *runningMean = init->noRunningStats ? NULL : allocFloatVector(C, 0.0f);
    tensor_t *runningVar = init->noRunningStats ? NULL : allocFloatVector(C, 1.0f);
    initBatchNorm1dConfig(cfg, gamma, beta, runningMean, runningVar, C, eps, init->momentumMode,
                          init->momentum, NULL, NULL);
    cfg->frozen = !trainable;
    cfg->forwardMath = lq->forwardMath;
    cfg->propLossMath = lq->propLossMath;
    cfg->weightGradAccMode = lq->weightGradAccMode;
    cfg->biasGradAccMode = lq->biasGradAccMode;
    *outCfg = cfg;
    return layer;
}

layer_t *batchNorm1dLayerInit(batchNorm1dInit_t *init, layerQuant_t *lq) {
    batchNorm1dConfig_t *cfg;
    layer_t *layer = batchNorm1dLayerInitCommon(init, lq, &cfg);
    cfg->outputQ = lq->outputQ;
    cfg->propLossQ = lq->propLossQ;
    cfg->ownsQuantizations = false;
    return layer;
}

layer_t *batchNorm1dLayerInitOwning(batchNorm1dInit_t *init, layerQuant_t *lq) {
    batchNorm1dConfig_t *cfg;
    layer_t *layer = batchNorm1dLayerInitCommon(init, lq, &cfg);
    cfg->outputQ = deepCopyQuantization(lq->outputQ);
    cfg->propLossQ = deepCopyQuantization(lq->propLossQ);
    cfg->ownsQuantizations = true;
    return layer;
}

void freeBatchNorm1dLayer(layer_t *layer) {
    if (layer == NULL) {
        return;
    }
    batchNorm1dConfig_t *cfg = layer->config->batchNorm1d;
    if (cfg->gamma != NULL) {
        freeParameter(cfg->gamma);
    }
    if (cfg->beta != NULL) {
        freeParameter(cfg->beta);
    }
    if (cfg->runningMean != NULL) {
        freeTensor(cfg->runningMean);
    }
    if (cfg->runningVar != NULL) {
        freeTensor(cfg->runningVar);
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
