#define SOURCE_FILE "BATCHNORM1D"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "BatchNorm1d.h"

#include "Add.h"
#include "ArithmeticType.h"
#include "Common.h"
#include "Div.h"
#include "ExecuteOp.h"
#include "Layer.h"
#include "LayerExec.h"
#include "LayerWireGuards.h"
#include "Mul.h"
#include "Quantization.h"
#include "Reduce.h"
#include "Sub.h"
#include "Tensor.h"

#define BATCHNORM1D_DEFAULT_MOMENTUM 0.1f

void initBatchNorm1dConfig(batchNorm1dConfig_t *cfg, parameter_t *gamma, parameter_t *beta,
                           tensor_t *runningMean, tensor_t *runningVar, size_t numChannels,
                           float eps, bnMomentumMode_t momentumMode, float momentum,
                           quantization_t *forwardQ, quantization_t *backwardQ) {
    cfg->gamma = gamma;
    cfg->beta = beta;
    cfg->runningMean = runningMean;
    cfg->runningVar = runningVar;
    cfg->numBatchesTracked = 0;
    cfg->numChannels = numChannels;
    cfg->eps = eps;
    if (momentumMode == BN_MOMENTUM_DEFAULT) {
        momentumMode = BN_MOMENTUM_VALUE;
        momentum = BATCHNORM1D_DEFAULT_MOMENTUM;
    }
    cfg->momentumMode = momentumMode;
    cfg->momentum = momentum;
    cfg->affine = gamma != NULL;
    cfg->trackRunningStats = runningMean != NULL;
    cfg->training = false;
    cfg->frozen = false;
    cfg->forwardMath = arithmeticFromQuantizationOrDefault(forwardQ);
    cfg->propLossMath = arithmeticFromQuantizationOrDefault(backwardQ);
    cfg->outputQ = forwardQ;
    cfg->propLossQ = backwardQ;
    cfg->ownsQuantizations = false;
}

/* The kernels cast every wire to float *. */
static void bnRequireFloat32(const tensor_t *t, const char *where, const char *what) {
    if (t->quantization->type != FLOAT32) {
        PRINT_ERROR("%s: %s must be FLOAT32 (got dtype %d) -- BatchNorm1d is FLOAT32-only", where,
                    what, (int)t->quantization->type);
        exit(1);
    }
}

static bool bnUsesBatchStats(const batchNorm1dConfig_t *cfg) {
    return !cfg->trackRunningStats || (cfg->training && !cfg->frozen);
}

static size_t bnInner(const tensor_t *t) {
    return t->shape->numberOfDimensions == 3 ? t->shape->dimensions[2] : 1;
}

/* Batch statistics need n >= 2 values per channel: n = 1 has no variance
 * (and n/(n-1) divides by zero), n = 0 would write 0/0 into the running
 * stats. PyTorch rejects n = 1 only; n = 0 is stricter by design.
 *
 * An untracked (noRunningStats) BN always uses batch statistics, even in
 * eval, over its evaluation chunk: n is rows * T of that chunk, so a
 * single-row rank-2 [1, C] chunk has n = 1. Untracked BNs get their own
 * messages: the generic "(a frozen or eval-mode BN uses running statistics
 * instead)" is false without running statistics, and in eval the knob to
 * raise is the evaluation micro-batch size, not microBatchSize. */
static void bnDieUntrackedEval(const char *what, size_t n) {
    PRINT_ERROR("BatchNorm1d %s: no running statistics (noRunningStats), so evaluation normalizes "
                "with batch statistics over its chunk (n = rows x T values per channel) -- got "
                "n = %zu -- track running statistics, or raise the evaluation micro-batch size "
                "(trainingRunOptions_t.evalMicroBatchSize, or the evaluationEpoch* "
                "microBatchSize) so every chunk, including the last N mod m rows, has >= 2 values "
                "per channel",
                what, n);
    exit(1);
}

static void bnRequireBatchStatsSize(const batchNorm1dConfig_t *cfg, const tensor_t *t, size_t n,
                                    const char *what) {
    if (n >= 2) {
        return;
    }
    bool untrackedEval = !cfg->trackRunningStats && !cfg->training;
    const shape_t *s = t->shape;
    if (untrackedEval) {
        bnDieUntrackedEval(what, n);
    } else if (s->numberOfDimensions == 2) {
        if (n == 0) {
            PRINT_ERROR("BatchNorm1d %s: batch statistics need >= 2 values per channel, got an "
                        "empty batch (n = 0) for a [%zu, %zu] batch",
                        what, s->dimensions[0], s->dimensions[1]);
        } else if (!cfg->trackRunningStats) {
            PRINT_ERROR(
                "BatchNorm1d %s: no running statistics (noRunningStats), so this BN normalizes "
                "with batch statistics even when frozen; they need >= 2 values per channel, got "
                "n = %zu for a [%zu, %zu] batch -- track running statistics, or (without "
                "evaluation) set trainingRunOptions_t.microBatchSize >= 2",
                what, n, s->dimensions[0], s->dimensions[1]);
            exit(1);
        } else {
            PRINT_ERROR(
                "BatchNorm1d %s: batch statistics need >= 2 values per channel, got n = %zu "
                "for a [%zu, %zu] batch -- set trainingRunOptions_t.microBatchSize >= 2 "
                "(a frozen or eval-mode BN uses running statistics instead)",
                what, n, s->dimensions[0], s->dimensions[1]);
        }
    } else {
        PRINT_ERROR("BatchNorm1d %s: batch statistics need >= 2 values per channel, got n = %zu "
                    "for a [%zu, %zu, %zu] batch",
                    what, n, s->dimensions[0], s->dimensions[1], s->dimensions[2]);
    }
    exit(1);
}

/* An untracked BN normalizes with batch statistics even in evaluation
 * (bnUsesBatchStats), over its evaluation chunk -- [rows, C] or [rows, C, T] -- so
 * n = rows * prod(dims[2:]). */
void batchNorm1dRequireEvaluable(const layer_t *layer, const shape_t *evalInputShape,
                                 const char *what) {
    if (layer->config->batchNorm1d->trackRunningStats) {
        return;
    }
    size_t n = evalInputShape->dimensions[0];
    for (size_t d = 2; d < evalInputShape->numberOfDimensions; d++) {
        n *= evalInputShape->dimensions[d];
    }
    if (n < 2) {
        bnDieUntrackedEval(what, n);
    }
}

/* Per-channel mean / biased variance over (b, t) through Reduce: a stack
 * [m, C, T] alias of the identity-order input, transposeTensor(0,1) ->
 * logical [C, m, T] (physical bytes unchanged; Reduce reads honor
 * orderOfDimensions), k = 2. Callers guarantee n >= 2 (so m*T > 0). */
static void bnBatchStats(const batchNorm1dConfig_t *cfg, tensor_t *input, float *mean, float *var) {
    size_t viewDims[3] = {input->shape->dimensions[0], cfg->numChannels, bnInner(input)};
    size_t viewOrder[3] = {0, 1, 2};
    shape_t viewShape;
    setShape(&viewShape, viewDims, 3, viewOrder);
    tensor_t view;
    setTensorValues(&view, input->data, &viewShape, input->quantization, NULL);
    transposeTensor(&view, 0, 1);

    size_t statsDims[1] = {cfg->numChannels};
    size_t statsOrder[1] = {0};
    shape_t statsShape;
    setShape(&statsShape, statsDims, 1, statsOrder);
    quantization_t statsQ;
    initFloat32Quantization(&statsQ);
    tensor_t meanT;
    setTensorValues(&meanT, (uint8_t *)mean, &statsShape, &statsQ, NULL);
    tensor_t varT;
    setTensorValues(&varT, (uint8_t *)var, &statsShape, &statsQ, NULL);
    meanOverTrailingAxesFloat32(&view, 2, &meanT);
    varianceBiasedOverTrailingAxesFloat32(&view, 2, &meanT, &varT);
}

/* mean/var/invStd for the selected mode. Running mode copies the buffers. */
static void bnResolveStats(const batchNorm1dConfig_t *cfg, tensor_t *input, bool batchStats,
                           float *mean, float *var, float *invStd) {
    size_t C = cfg->numChannels;
    if (batchStats) {
        bnBatchStats(cfg, input, mean, var);
    } else {
        memcpy(mean, cfg->runningMean->data, C * sizeof(float));
        memcpy(var, cfg->runningVar->data, C * sizeof(float));
    }
    for (size_t c = 0; c < C; c++) {
        invStd[c] = rsqrtFloat32(var[c], cfg->eps); /* eps INSIDE the sqrt */
    }
}

/* PyTorch _BatchNorm.forward: count first, then the EMA with the UNBIASED
 * variance var*n/(n-1). Saturating counter: a wrap to 0 would make the
 * CUMULATIVE factor 1/0. */
static void bnUpdateRunningStats(batchNorm1dConfig_t *cfg, const float *mean, const float *var,
                                 size_t n) {
    if (cfg->numBatchesTracked < UINT64_MAX) {
        cfg->numBatchesTracked += 1;
    }
    float f = cfg->momentumMode == BN_MOMENTUM_CUMULATIVE
                  ? divFloat32s(1.0f, (float)cfg->numBatchesTracked)
                  : cfg->momentum;
    float keep = subFloat32s(1.0f, f);
    float unbias = divFloat32s((float)n, (float)(n - 1));
    float *rm = (float *)cfg->runningMean->data;
    float *rv = (float *)cfg->runningVar->data;
    for (size_t c = 0; c < cfg->numChannels; c++) {
        rm[c] = addFloat32s(mulFloat32s(keep, rm[c]), mulFloat32s(f, mean[c]));
        rv[c] = addFloat32s(mulFloat32s(keep, rv[c]), mulFloat32s(f, mulFloat32s(var[c], unbias)));
    }
}

typedef struct {
    const float *mean;   /* [C] */
    const float *invStd; /* [C] */
    size_t numChannels;
    size_t inner; /* T; 1 for rank 2 */
} bnAffineCtx_t;

/* executeOp kernel: operands {input[, gamma, beta]}; y = gamma*xhat + beta
 * (xhat alone when nOperands == 1, i.e. !affine). */
static void bnForwardKernelFloat(tensor_t **ops, size_t n, tensor_t *rawOut, tensor_t *auxOut,
                                 const void *ctx, rngStream_t *rs) {
    (void)auxOut;
    const bnAffineCtx_t *c = ctx;
    const float *x = (const float *)ops[0]->data;
    const float *gamma = n == 3 ? (const float *)ops[1]->data : NULL;
    const float *beta = n == 3 ? (const float *)ops[2]->data : NULL;
    float *y = (float *)rawOut->data;
    size_t total = calcNumberOfElementsByTensor(ops[0]);
    for (size_t i = 0; i < total; i++) {
        size_t ch = (i / c->inner) % c->numChannels;
        float xhat = mulFloat32s(subFloat32s(x[i], c->mean[ch]), c->invStd[ch]);
        y[i] = gamma != NULL ? addFloat32s(mulFloat32s(gamma[ch], xhat), beta[ch]) : xhat;
    }
}

/* OP(k) of each funnel op of this layer, for layerExecOpKey. Part of the
 * value contract: append, never renumber. */
enum {
    BATCHNORM1D_OP_FORWARD = 0,
};

void batchNorm1dForward(layer_t *layer, tensor_t *input, tensor_t *output,
                        const layerExec_t *exec) {
    rngKey_t opKey;
    batchNorm1dConfig_t *cfg = layer->config->batchNorm1d;
    size_t outDims[3];
    size_t outOrder[3];
    shape_t outExpected = {.dimensions = outDims, .orderOfDimensions = outOrder};
    batchNorm1dCalcOutputShape(layer, input->shape, &outExpected);
    bnRequireFloat32(input, "BatchNorm1d forward", "input");
    layerRequireDenseWire(output, &outExpected, "BatchNorm1d forward", "output");
    bnRequireFloat32(output, "BatchNorm1d forward", "output");
    size_t total = calcNumberOfElementsByTensor(input);
    size_t n = total / cfg->numChannels;
    bool batchStats = bnUsesBatchStats(cfg);
    if (batchStats) {
        bnRequireBatchStatsSize(cfg, input, n, "forward");
    } else if (total == 0) {
        return; /* running statistics, nothing to normalize, nothing written */
    }

    size_t C = cfg->numChannels;
    float mean[C];
    float var[C];
    float invStd[C];
    bnResolveStats(cfg, input, batchStats, mean, var, invStd);
    if (batchStats && cfg->trackRunningStats && cfg->training && !cfg->frozen) {
        bnUpdateRunningStats(cfg, mean, var, n);
    }

    bnAffineCtx_t ctx = {.mean = mean, .invStd = invStd, .numChannels = C, .inner = bnInner(input)};
    tensor_t *ops[3] = {input, NULL, NULL};
    size_t nOps = 1;
    if (cfg->affine) {
        ops[1] = getParamFromParameter(cfg->gamma);
        ops[2] = getParamFromParameter(cfg->beta);
        nOps = 3;
    }
    executeOp(
        &(opSpec_t){.kernel = bnForwardKernelFloat,
                    .ctx = &ctx,
                    .inputs = ops,
                    .nInputs = nOps,
                    .arithmetic = cfg->forwardMath,
                    .key = layerExecOpKey(exec, RNG_TAG_FORWARD, BATCHNORM1D_OP_FORWARD, &opKey),
                    .mode = OUT_WRITE},
        output);
}

void batchNorm1dBackward(layer_t *layer, tensor_t *forwardInput, tensor_t *loss, tensor_t *propLoss,
                         const layerExec_t *exec) {
    batchNorm1dConfig_t *cfg = layer->config->batchNorm1d;
    size_t wireDims[3];
    size_t wireOrder[3];
    shape_t wireExpected = {.dimensions = wireDims, .orderOfDimensions = wireOrder};
    batchNorm1dCalcOutputShape(layer, forwardInput->shape, &wireExpected);
    bnRequireFloat32(forwardInput, "BatchNorm1d backward", "forwardInput");
    layerRequireDenseWire(loss, &wireExpected, "BatchNorm1d backward", "loss");
    bnRequireFloat32(loss, "BatchNorm1d backward", "loss");
    if (propLoss != NULL) {
        layerRequireDenseWire(propLoss, &wireExpected, "BatchNorm1d backward", "propLoss");
        bnRequireFloat32(propLoss, "BatchNorm1d backward", "propLoss");
    }
    bool wantGrads = cfg->affine && !cfg->frozen;
    size_t total = calcNumberOfElementsByTensor(forwardInput);
    size_t C = cfg->numChannels;
    size_t n = total / C;
    bool batchStats = bnUsesBatchStats(cfg);
    if (batchStats) {
        bnRequireBatchStatsSize(cfg, forwardInput, n, "backward");
    } else if (total == 0) {
        return;
    }

    float mean[C];
    float var[C];
    float invStd[C];
    bnResolveStats(cfg, forwardInput, batchStats, mean, var, invStd);

    const float *x = (const float *)forwardInput->data;
    const float *dy = (const float *)loss->data;
    size_t T = bnInner(forwardInput);
    float sumDy[C];
    float sumDyXhat[C];
    for (size_t c = 0; c < C; c++) {
        sumDy[c] = 0.0f;
        sumDyXhat[c] = 0.0f;
    }
    for (size_t i = 0; i < total; i++) {
        size_t ch = (i / T) % C;
        float xhat = mulFloat32s(subFloat32s(x[i], mean[ch]), invStd[ch]);
        sumDy[ch] = addFloat32s(sumDy[ch], dy[i]);
        sumDyXhat[ch] = addFloat32s(sumDyXhat[ch], mulFloat32s(dy[i], xhat));
    }
    if (wantGrads) {
        float *dgamma = (float *)cfg->gamma->grad->data;
        float *dbeta = (float *)cfg->beta->grad->data;
        for (size_t c = 0; c < C; c++) {
            dgamma[c] = addFloat32s(dgamma[c], sumDyXhat[c]); /* SUM over (b, t) */
            dbeta[c] = addFloat32s(dbeta[c], sumDy[c]);
        }
    }
    if (propLoss == NULL) {
        return; /* grads-only */
    }
    const float *gamma = cfg->affine ? (const float *)cfg->gamma->param->data : NULL;
    float *dx = (float *)propLoss->data;
    float invN = batchStats ? divFloat32s(1.0f, (float)n) : 0.0f;
    for (size_t i = 0; i < total; i++) {
        size_t ch = (i / T) % C;
        float scale = gamma != NULL ? mulFloat32s(gamma[ch], invStd[ch]) : invStd[ch];
        if (batchStats) {
            float xhat = mulFloat32s(subFloat32s(x[i], mean[ch]), invStd[ch]);
            float centered = subFloat32s(subFloat32s(dy[i], mulFloat32s(sumDy[ch], invN)),
                                         mulFloat32s(xhat, mulFloat32s(sumDyXhat[ch], invN)));
            dx[i] = mulFloat32s(scale, centered);
        } else {
            dx[i] = mulFloat32s(scale, dy[i]);
        }
    }
}

/* Element (b, c, t) of an identity-order input sits at (b*C + c)*T + t, which
 * every flat loop above relies on. */
void batchNorm1dCalcOutputShape(layer_t *layer, shape_t *inputShape, shape_t *outputShape) {
    const batchNorm1dConfig_t *cfg = layer->config->batchNorm1d;
    size_t rank = inputShape->numberOfDimensions;
    if (rank != 2 && rank != 3) {
        PRINT_ERROR("BatchNorm1d: input must be rank 2 [m,C] or rank 3 [m,C,T] (got rank %zu)",
                    rank);
        exit(1);
    }
    for (size_t d = 0; d < rank; d++) {
        if (inputShape->orderOfDimensions[d] != d) {
            PRINT_ERROR("BatchNorm1d: input must be identity-order (dim %zu is order %zu)", d,
                        inputShape->orderOfDimensions[d]);
            exit(1);
        }
    }
    if (inputShape->dimensions[1] != cfg->numChannels) {
        PRINT_ERROR("BatchNorm1d: input channel dim is %zu but numChannels is %zu",
                    inputShape->dimensions[1], cfg->numChannels);
        exit(1);
    }
    memcpy(outputShape->dimensions, inputShape->dimensions, rank * sizeof(size_t));
    setOrderOfDimsForNewTensor(rank, outputShape->orderOfDimensions);
    outputShape->numberOfDimensions = rank;
}
