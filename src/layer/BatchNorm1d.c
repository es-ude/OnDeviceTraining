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
    /* Codex plan review: affine/track are derived from ONE pointer each, so a
     * half-pair would dereference NULL later. Both halves or neither. */
    if ((gamma == NULL) != (beta == NULL)) {
        PRINT_ERROR("initBatchNorm1dConfig: gamma and beta must both be set (affine) or both "
                    "be NULL (got gamma %s, beta %s)",
                    gamma ? "set" : "NULL", beta ? "set" : "NULL");
        exit(1);
    }
    if ((runningMean == NULL) != (runningVar == NULL)) {
        PRINT_ERROR("initBatchNorm1dConfig: runningMean and runningVar must both be set "
                    "(tracked) or both be NULL (got runningMean %s, runningVar %s)",
                    runningMean ? "set" : "NULL", runningVar ? "set" : "NULL");
        exit(1);
    }
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
    cfg->weightGradAccMode = OUT_ACC_DYNAMIC_RESCALE;
    cfg->biasGradAccMode = OUT_ACC_DYNAMIC_RESCALE;
    cfg->ownsQuantizations = false;
}

/* Fail fast unless `t` is a FLOAT32, identity-order [m, C] / [m, C, T] with
 * dims[1] == numChannels. Identity order makes element (b, c, t) sit at
 * (b*C + c)*T + t, which every flat loop below relies on. */
static void bnValidateInput(const batchNorm1dConfig_t *cfg, const tensor_t *t, const char *what) {
    const shape_t *s = t->shape;
    size_t rank = s->numberOfDimensions;
    if (rank != 2 && rank != 3) {
        PRINT_ERROR("BatchNorm1d %s: must be rank 2 [m,C] or rank 3 [m,C,T] (got rank %zu)", what,
                    rank);
        exit(1);
    }
    for (size_t d = 0; d < rank; d++) {
        if (s->orderOfDimensions[d] != d) {
            PRINT_ERROR("BatchNorm1d %s: must be identity-order (dim %zu is order %zu)", what, d,
                        s->orderOfDimensions[d]);
            exit(1);
        }
    }
    if (cfg->numChannels == 0 || s->dimensions[1] != cfg->numChannels) {
        PRINT_ERROR("BatchNorm1d %s: channel dim is %zu but numChannels is %zu", what,
                    s->dimensions[1], cfg->numChannels);
        exit(1);
    }
    if (t->quantization->type != FLOAT32) {
        PRINT_ERROR("BatchNorm1d %s: must be FLOAT32 (got dtype %d) -- BatchNorm1d is FLOAT32-only",
                    what, (int)t->quantization->type);
        exit(1);
    }
}

/* Every per-channel state tensor the kernels index as float[C]: FLOAT32 and
 * exactly numChannels elements (Codex plan review: dtype AND capacity). */
static void bnRequireChannelVector(const batchNorm1dConfig_t *cfg, const tensor_t *t,
                                   const char *what) {
    if (t->quantization->type != FLOAT32) {
        PRINT_ERROR("BatchNorm1d: %s must be FLOAT32 (got dtype %d) -- BatchNorm1d is "
                    "FLOAT32-only",
                    what, (int)t->quantization->type);
        exit(1);
    }
    size_t count = calcNumberOfElementsByTensor((tensor_t *)t);
    if (count != cfg->numChannels) {
        PRINT_ERROR("BatchNorm1d: %s has %zu elements but numChannels is %zu", what, count,
                    cfg->numChannels);
        exit(1);
    }
}

static void bnValidateAffineParams(const batchNorm1dConfig_t *cfg) {
    if (!cfg->affine) {
        return;
    }
    bnRequireChannelVector(cfg, cfg->gamma->param, "gamma");
    bnRequireChannelVector(cfg, cfg->beta->param, "beta");
}

static void bnValidateRunningBuffers(const batchNorm1dConfig_t *cfg) {
    if (!cfg->trackRunningStats) {
        return;
    }
    bnRequireChannelVector(cfg, cfg->runningMean, "runningMean");
    bnRequireChannelVector(cfg, cfg->runningVar, "runningVar");
}

/* executeOp sizes its raw target from `output` (ExecuteOp.c:107) while the
 * kernel loops over the input's element count: an output that does not
 * match the input's rank and dims would be written out of bounds. */
static void bnValidateOutputMatchesInput(const tensor_t *input, const tensor_t *output) {
    const shape_t *in = input->shape;
    const shape_t *out = output->shape;
    bool same = in->numberOfDimensions == out->numberOfDimensions;
    for (size_t d = 0; same && d < in->numberOfDimensions; d++) {
        same = in->dimensions[d] == out->dimensions[d];
    }
    if (!same) {
        PRINT_ERROR("BatchNorm1d forward: output must have the input's rank and dims "
                    "(input rank %zu, output rank %zu)",
                    in->numberOfDimensions, out->numberOfDimensions);
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
 * stats. PyTorch rejects n = 1 only; n = 0 is stricter by design (#460). */
static void bnRequireBatchStatsSize(const tensor_t *t, size_t n, const char *what) {
    if (n >= 2) {
        return;
    }
    const shape_t *s = t->shape;
    if (s->numberOfDimensions == 2) {
        PRINT_ERROR("BatchNorm1d %s: batch statistics need >= 2 values per channel, got n = %zu "
                    "for a [%zu, %zu] batch -- set trainingRunOptions_t.microBatchSize >= 2 "
                    "(a frozen or eval-mode BN uses running statistics instead)",
                    what, n, s->dimensions[0], s->dimensions[1]);
    } else {
        PRINT_ERROR("BatchNorm1d %s: batch statistics need >= 2 values per channel, got n = %zu "
                    "for a [%zu, %zu, %zu] batch",
                    what, n, s->dimensions[0], s->dimensions[1], s->dimensions[2]);
    }
    exit(1);
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
                                 const void *ctx) {
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

void batchNorm1dForward(layer_t *layer, tensor_t *input, tensor_t *output) {
    batchNorm1dConfig_t *cfg = layer->config->batchNorm1d;
    bnValidateInput(cfg, input, "forward input");
    bnValidateOutputMatchesInput(input, output);
    bnValidateAffineParams(cfg);
    bnValidateRunningBuffers(cfg);
    if (cfg->forwardMath.type != ARITH_FLOAT32) {
        PRINT_ERROR("BatchNorm1d forward: forwardMath %d not supported -- FLOAT32 only",
                    (int)cfg->forwardMath.type);
        exit(1);
    }
    size_t total = calcNumberOfElementsByTensor(input);
    size_t n = total / cfg->numChannels;
    bool batchStats = bnUsesBatchStats(cfg);
    if (batchStats) {
        bnRequireBatchStatsSize(input, n, "forward");
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
    executeOp(&(opSpec_t){.kernel = bnForwardKernelFloat,
                          .ctx = &ctx,
                          .inputs = ops,
                          .nInputs = nOps,
                          .arithmetic = cfg->forwardMath,
                          .mode = OUT_WRITE},
              output);
}

static void bnRequireSameShape(const tensor_t *ref, const tensor_t *t, const char *what) {
    const shape_t *a = ref->shape;
    const shape_t *b = t->shape;
    bool ok = a->numberOfDimensions == b->numberOfDimensions && t->quantization->type == FLOAT32;
    for (size_t d = 0; ok && d < a->numberOfDimensions; d++) {
        ok = a->dimensions[d] == b->dimensions[d] && b->orderOfDimensions[d] == d;
    }
    if (!ok) {
        PRINT_ERROR("BatchNorm1d backward: %s must be FLOAT32, identity-order and shaped like the "
                    "forward input",
                    what);
        exit(1);
    }
}

void batchNorm1dBackward(layer_t *layer, tensor_t *forwardInput, tensor_t *loss,
                         tensor_t *propLoss) {
    batchNorm1dConfig_t *cfg = layer->config->batchNorm1d;
    bnValidateInput(cfg, forwardInput, "backward input");
    bnRequireSameShape(forwardInput, loss, "loss");
    if (propLoss != NULL) {
        bnRequireSameShape(forwardInput, propLoss, "propLoss");
    }
    bnValidateAffineParams(cfg);
    bnValidateRunningBuffers(cfg);
    if (cfg->propLossMath.type != ARITH_FLOAT32) {
        PRINT_ERROR("BatchNorm1d backward: propLossMath %d not supported -- FLOAT32 only",
                    (int)cfg->propLossMath.type);
        exit(1);
    }
    bool wantGrads = cfg->affine && !cfg->frozen;
    if (wantGrads && (cfg->gamma->grad == NULL || cfg->beta->grad == NULL ||
                      cfg->gamma->grad->quantization->type != FLOAT32 ||
                      cfg->beta->grad->quantization->type != FLOAT32)) {
        PRINT_ERROR("BatchNorm1d backward: a trainable BN needs FLOAT32 gamma/beta grads");
        exit(1);
    }
    size_t total = calcNumberOfElementsByTensor(forwardInput);
    size_t C = cfg->numChannels;
    size_t n = total / C;
    bool batchStats = bnUsesBatchStats(cfg);
    if (batchStats) {
        bnRequireBatchStatsSize(forwardInput, n, "backward");
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

void batchNorm1dCalcOutputShape(layer_t *layer, shape_t *inputShape, shape_t *outputShape) {
    (void)layer;
    memcpy(outputShape->dimensions, inputShape->dimensions,
           inputShape->numberOfDimensions * sizeof(size_t));
    memcpy(outputShape->orderOfDimensions, inputShape->orderOfDimensions,
           inputShape->numberOfDimensions * sizeof(size_t));
    outputShape->numberOfDimensions = inputShape->numberOfDimensions;
}
