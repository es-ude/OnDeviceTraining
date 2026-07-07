#define SOURCE_FILE "GROUPNORM"

#include <math.h> /* powf: one-time config-derived range constants only (orchestration) */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "GroupNorm.h"

#include "Add.h"
#include "Arithmetic.h"
#include "ArithmeticType.h"
#include "Common.h"
#include "Div.h"
#include "ExecuteOp.h"
#include "Layer.h"
#include "MinMax.h"
#include "Mul.h"
#include "Quantization.h"
#include "Reduce.h"
#include "Rounding.h"
#include "Sub.h"
#include "Tensor.h"

void initGroupNormConfig(groupNormConfig_t *cfg, parameter_t *gamma, parameter_t *beta,
                         size_t numGroups, size_t numChannels, float eps, quantization_t *forwardQ,
                         quantization_t *backwardQ) {
    cfg->gamma = gamma;
    cfg->beta = beta;
    cfg->numGroups = numGroups;
    cfg->numChannels = numChannels;
    cfg->eps = eps;
    cfg->forwardMath = arithmeticFromQuantizationOrDefault(forwardQ);
    cfg->propLossMath = arithmeticFromQuantizationOrDefault(backwardQ);
    cfg->outputQ = forwardQ;
    cfg->propLossQ = backwardQ;
    cfg->ownsQuantizations = false;

    /* OUT_ACC_DYNAMIC_RESCALE for BOTH grads, the LayerNorm scheme: dgamma and
     * dbeta accumulate via the identity-kernel executeOp + Strategy-A requant —
     * GroupNorm's beta grad has no FIXED_SCALE bias history to preserve.
     * Carried on the config so hand-wired callers (UnitTestGroupNorm.c) get
     * the canonical behavior; a layerQuant_t-driven factory (Task 4) overrides
     * these right after this call if the caller opted into a different mode. */
    cfg->weightGradAccMode = OUT_ACC_DYNAMIC_RESCALE;
    cfg->biasGradAccMode = OUT_ACC_DYNAMIC_RESCALE;
}

/* Fail fast unless `t` is rank-3 [B,C,T], identity-order, with dims[1] ==
 * numChannels. This single guard is what makes every flat offset and the
 * [B,G,cpg,T] alias view below valid: identity order ⇒ physical layout ==
 * logical row-major, so element (b,c,t) sits at ((b*C)+c)*T + t. A mismatched
 * channel dim would read gamma/beta out of bounds in forward and WRITE the
 * grad tensors out of bounds in backward — silently. Non-identity order /
 * rank-4 is a documented follow-up (spec R4). */
static void groupNormValidateInputShape(groupNormConfig_t *cfg, tensor_t *input) {
    shape_t *s = input->shape;
    if (s->numberOfDimensions != 3) {
        PRINT_ERROR("GroupNorm: input must be rank-3 [B,C,T] (got rank %zu)",
                    s->numberOfDimensions);
        exit(1);
    }
    for (size_t d = 0; d < 3; d++) {
        if (s->orderOfDimensions[d] != d) {
            PRINT_ERROR("GroupNorm: input must be identity-order (dim %zu is order %zu)", d,
                        s->orderOfDimensions[d]);
            exit(1);
        }
    }
    if (s->dimensions[1] != cfg->numChannels) {
        PRINT_ERROR("GroupNorm: input channel dim is %zu but numChannels is %zu", s->dimensions[1],
                    cfg->numChannels);
        exit(1);
    }
    /* Config-time validation (numGroups > 0, numGroups divides numChannels)
     * belongs to the factory (Task 4); hand-wired configs bypass it, and the
     * kernels divide by these — cheap defensive re-check instead of an ugly
     * div-by-zero / mis-grouped OOB crash. */
    if (cfg->numGroups == 0 || cfg->numChannels % cfg->numGroups != 0) {
        PRINT_ERROR("GroupNorm: numGroups (%zu) must be > 0 and divide numChannels (%zu)",
                    cfg->numGroups, cfg->numChannels);
        exit(1);
    }
}

/* Group geometry from a validated [B,C,T] input: cpg = C/G channels per group,
 * K = B*G blocks, N = cpg*T elements per block. */
static void groupNormGroupGeom(tensor_t *input, groupNormConfig_t *cfg, size_t *K, size_t *N,
                               size_t *cpg, size_t *B, size_t *T) {
    *B = input->shape->dimensions[0];
    *T = input->shape->dimensions[2];
    *cpg = cfg->numChannels / cfg->numGroups;
    *K = *B * cfg->numGroups;
    *N = *cpg * *T;
}

/* All-blocks stats via the Reduce arithmetic module: fills mean[K] and
 * invSigma[K] (caller stack scratch, K = B*G) with one pass each of
 * meanOverTrailingAxes* + varianceBiasedOverTrailingAxes*, then invSigma[k] =
 * 1/sqrt(var[k]+eps). Variance is BIASED (÷N, NOT N-1); eps is INSIDE the sqrt.
 *
 * The reduction input is a stack [B,G,cpg,T] ALIAS view over t's data: the
 * rank-3/identity-order guard makes the physical layout logical row-major, so
 * splitting C into (G,cpg) is a pure relabeling of the same bytes and k=2
 * collapses (cpg,T) per (b,g) block. Reduce writes blocks row-major over the
 * leading [B,G] dims — the same k = b*G + g order the kernels iterate, so
 * mean[k]/invSigma[k] index the block the layer is processing.
 *
 * ONE helper for BOTH FLOAT32 and SYM_INT32, forward AND backward, so a
 * layer's passes can never desync on the stats definition (the Task-2
 * LayerNorm precedent). Callers MUST guarantee K > 0 && N > 0 (the stats VLAs
 * and Reduce's block loop are undefined at 0); every caller early-outs before
 * calling here. */
static void groupNormAllGroupStats(tensor_t *t, groupNormConfig_t *cfg, size_t B, size_t cpg,
                                   size_t T, size_t K, float eps, float *mean, float *invSigma) {
    size_t viewDims[4] = {B, cfg->numGroups, cpg, T};
    size_t viewOrder[4] = {0, 1, 2, 3};
    shape_t viewShape;
    setShape(&viewShape, viewDims, 4, viewOrder);
    tensor_t view;
    setTensorValues(&view, t->data, &viewShape, t->quantization, NULL);

    size_t statsDims[1] = {K};
    size_t statsOrder[1] = {0};
    shape_t statsShape;
    setShape(&statsShape, statsDims, 1, statsOrder);
    quantization_t statsQ;
    initFloat32Quantization(&statsQ);

    float var[K];
    tensor_t meanT;
    setTensorValues(&meanT, (uint8_t *)mean, &statsShape, &statsQ, NULL);
    tensor_t varT;
    setTensorValues(&varT, (uint8_t *)var, &statsShape, &statsQ, NULL);

    if (t->quantization->type == SYM_INT32) {
        meanOverTrailingAxesSymInt32(&view, 2, &meanT);
        varianceBiasedOverTrailingAxesSymInt32(&view, 2, &meanT, &varT);
    } else {
        meanOverTrailingAxesFloat32(&view, 2, &meanT);
        varianceBiasedOverTrailingAxesFloat32(&view, 2, &meanT, &varT);
    }
    for (size_t k = 0; k < K; k++) {
        invSigma[k] = rsqrtFloat32(var[k], eps); /* eps INSIDE sqrt */
    }
}

static void groupNormForwardFloat(groupNormConfig_t *cfg, tensor_t *gamma, tensor_t *beta,
                                  tensor_t *input, tensor_t *output) {
    float *in = (float *)input->data;
    float *out = (float *)output->data;
    float *g = (float *)gamma->data;
    float *bt = (float *)beta->data;

    size_t K;
    size_t N;
    size_t cpg;
    size_t B;
    size_t T;
    groupNormGroupGeom(input, cfg, &K, &N, &cpg, &B, &T);
    if (K == 0 || N == 0) {
        return; /* nothing to normalize (empty group geometry); cf. #160 */
    }

    float mean[K];
    float invSigma[K];
    groupNormAllGroupStats(input, cfg, B, cpg, T, K, cfg->eps, mean, invSigma);

    size_t G = cfg->numGroups;
    for (size_t b = 0; b < B; b++) {
        for (size_t grp = 0; grp < G; grp++) {
            size_t k = b * G + grp;
            size_t base = (b * cfg->numChannels + grp * cpg) * T;
            for (size_t j = 0; j < N; j++) {
                /* Per-CHANNEL affine: c indexes gamma/beta[C], NOT the inner
                 * j (LayerNorm's per-element indexing) — channel of inner
                 * index j is g*cpg + j/T (j spans cpg channels × T steps). */
                size_t c = grp * cpg + j / T;
                size_t off = base + j;
                float nval = mulFloat32s(subFloat32s(in[off], mean[k]), invSigma[k]);
                out[off] = addFloat32s(mulFloat32s(g[c], nval), bt[c]);
            }
        }
    }
}

/* The SYM_INT32 path reinterprets tensor data as int32 mantissas; a FLOAT32
 * buffer read that way is silent garbage, so fail fast. The int12 bound
 * (ODT_SYM_OPERAND_QMAXBITS) is required by the affine product q*gamma_q
 * (out[off]*gammaQ[c]); the mantissa-SUM behind the stats is a value-sum and
 * is sound at any qMaxBits <= 16 (it lives in the Reduce module, which
 * enforces that looser <= 16 bound itself). (#227) */
static void groupNormValidateSymTensor(tensor_t *t, const char *what) {
    if (t->quantization->type != SYM_INT32) {
        PRINT_ERROR("GroupNorm SYM_INT32: %s must be SYM_INT32", what);
        exit(1);
    }
    symInt32QConfig_t *qc = t->quantization->qConfig;
    if (qc->qMaxBits > ODT_SYM_OPERAND_QMAXBITS) {
        PRINT_ERROR("GroupNorm SYM_INT32: %s qMaxBits (%u) exceeds operand contract (%u)", what,
                    (unsigned)qc->qMaxBits, (unsigned)ODT_SYM_OPERAND_QMAXBITS);
        exit(1);
    }
}

/* Affine y = gamma*n + beta as a SEPARATE quantized elementwise stage, applied
 * in-place over the freshly written normalized mantissas (it destroys the
 * abs-max=qMax / var~1 invariants and has its own requantization + output
 * scale). Per-CHANNEL: gamma_c/beta_c are indexed by c = g*cpg + j/T (NOT
 * flat j, LayerNorm's per-element scheme) and broadcast over T and the batch.
 * Scale bookkeeping (the verified LayerNorm folding):
 *   s_y    = s_norm * s_gamma                  (product idiom)
 *   seed_c = round(beta_q,c * s_beta / s_y)    (bias-rescale idiom — a raw
 *            beta_q add would silently drop beta under dynamic scales)
 *   y_q    = q * gamma_q,c + seed_c            (the product q*gamma_q <= qMax^2
 *            fits int32, but the rescaled seed is DATA-DEPENDENT and unbounded:
 *            the shared rescaleIntoAccumulatorScale helper (#189) fails fast
 *            (under -DODT_SEED_GUARD) outside the safe envelope instead of
 *            casting an out-of-range float to int32 (UB); safe while
 *            |beta| <~ absmax_n * absmax_gamma, #227.)
 * This writes a RAW, unrestored y_q (accumulator-range, same class as
 * Linear/Conv's matmul output) into the raw output's own scale field — the
 * executeOp OUT_WRITE epilogue (caller, groupNormForward) restores width at
 * the producer via the SYM->SYM diagonal requant. */
static void groupNormAffineSymInt32(groupNormConfig_t *cfg, tensor_t *gamma, tensor_t *beta,
                                    tensor_t *output, float sNorm) {
    int32_t *out = (int32_t *)output->data;
    int32_t *gammaQ = (int32_t *)gamma->data;
    int32_t *betaQ = (int32_t *)beta->data;
    symInt32QConfig_t *outQC = output->quantization->qConfig;
    symInt32QConfig_t *gammaQC = gamma->quantization->qConfig;
    symInt32QConfig_t *betaQC = beta->quantization->qConfig;

    float sY = mulFloat32s(sNorm, gammaQC->scale);

    size_t K;
    size_t N;
    size_t cpg;
    size_t B;
    size_t T;
    groupNormGroupGeom(output, cfg, &K, &N, &cpg, &B, &T);

    /* seed_c = round(beta_q,c * s_beta / s_y) depends ONLY on the channel c
     * (betaQ[c], betaQC->scale and sY are all per-channel or scalar), so hoist
     * it out of the (b,grp,t) element loop and compute it once per channel into
     * stack scratch — the element loop then just indexes seed[c]. Numerics are
     * identical (same rescaleIntoAccumulatorScale inputs per channel). Guarded
     * K > 0 && N > 0 by the caller's early-out, so C = numChannels >= 1 here. */
    size_t C = cfg->numChannels;
    int32_t seed[C];
    for (size_t c = 0; c < C; c++) {
        seed[c] = rescaleIntoAccumulatorScale(betaQ[c], betaQC->scale, sY, outQC->roundingMode);
    }

    size_t G = cfg->numGroups;
    for (size_t b = 0; b < B; b++) {
        for (size_t grp = 0; grp < G; grp++) {
            size_t base = (b * cfg->numChannels + grp * cpg) * T;
            for (size_t j = 0; j < N; j++) {
                size_t c = grp * cpg + j / T;
                size_t off = base + j;
                out[off] = addInt32s(mulInt32s(out[off], gammaQ[c]), seed[c]);
            }
        }
    }
    outQC->scale = sY;
}

/* SYM_INT32 forward (the verified LayerNorm scale-folding scheme):
 * pass 1: per-block float stats + GLOBAL absmax of the normalized values.
 *         Multi-block REQUIRES the per-block 1/sigma_k to hit the DATA — one
 *         per-tensor scale cannot encode K different sigmas; only the global
 *         stretch lives in the scale.
 * pass 2: normalize from the same stored stats, stretch by qMax/absmax,
 *         round-clamp. Stats come ONCE from the Reduce module into K-float
 *         stack scratch (groupNormAllGroupStats) and both passes read them.
 * Output scale s_norm = 1/stretch, then the affine stage folds in gamma/beta
 * and writes the (raw, unrestored) producer scale. gamma/beta are funnel
 * operands (inputs = {input, gamma, beta}), not read via cfg — cfg carries
 * only eps/numGroups/numChannels geometry. */
static void groupNormForwardSymInt32(groupNormConfig_t *cfg, tensor_t *gamma, tensor_t *beta,
                                     tensor_t *input, tensor_t *output) {
    groupNormValidateSymTensor(input, "input");
    groupNormValidateSymTensor(output, "output");
    groupNormValidateSymTensor(gamma, "gamma");
    groupNormValidateSymTensor(beta, "beta");

    symInt32QConfig_t *inQC = input->quantization->qConfig;
    symInt32QConfig_t *outQC = output->quantization->qConfig;
    int32_t *in = (int32_t *)input->data;
    int32_t *out = (int32_t *)output->data;
    float inScale = inQC->scale;
    /* One-time range constants from the layer's static qMaxBits knob — powf,
     * the int decrement and the negation are ORCHESTRATION on config, not
     * runtime-data math; the -1 on the max is still routed through
     * subFloat32s (trivially wrappable). */
    const float qHalfRange = powf(2, (float)(outQC->qMaxBits - 1));
    const float qMax = subFloat32s(qHalfRange, 1.0f);
    const float qMin = -qHalfRange;

    size_t K;
    size_t N;
    size_t cpg;
    size_t B;
    size_t T;
    groupNormGroupGeom(input, cfg, &K, &N, &cpg, &B, &T);

    if (K == 0 || N == 0) {
        outQC->scale = 1.0f; /* nothing to normalize; neutral scale (cf. #160) */
        return;
    }

    /* Identity-order guard (groupNormValidateInputShape, forward entry) makes
     * whole-tensor flat scans valid: total == B*C*T contiguous elements. */
    size_t total = K * N;

    /* Integer range pre-check: if every mantissa is identical (global int32 min
     * == max), every block's centered value is exactly zero in integer space.
     * A float absMax==0.0f check alone is fragile here: gcc's default
     * -ffp-contract=fast may fuse (float)in[off]*inScale - mean into an fma,
     * so the product is not rounded before the subtraction and the cancellation
     * x*s - mean is NOT guaranteed to be exactly 0 even when all mantissas are
     * equal. The integer check is portable and catches this common case. */
    int32_t intMin = in[0];
    int32_t intMax = intMin;
    bool allConstant = true;
    for (size_t i = 0; i < total; i++) {
        int32_t v = in[i];
        if (v < intMin) {
            intMin = v;
            allConstant = false;
        }
        if (v > intMax) {
            intMax = v;
            allConstant = false;
        }
    }

    /* Pass 1: global absmax of n = (x_q*s_x - mu)/sigma over all blocks.
     * Skipped when allConstant — absMax stays 0.0f, caught by the unified
     * guard below; stats are only materialized on this branch (mean/invSigma
     * are dead in the constant/zero-absmax branch). Recompute-over-store scan
     * via absFloat32/maxFloat32s — no stored buffer exists for the
     * buffer-based findAbsMaxFloat. */
    float mean[K];
    float invSigma[K];
    float absMax = 0.0f;
    size_t G = cfg->numGroups;
    if (!allConstant) {
        groupNormAllGroupStats(input, cfg, B, cpg, T, K, cfg->eps, mean, invSigma);
        for (size_t b = 0; b < B; b++) {
            for (size_t grp = 0; grp < G; grp++) {
                size_t k = b * G + grp;
                size_t base = (b * cfg->numChannels + grp * cpg) * T;
                for (size_t j = 0; j < N; j++) {
                    size_t off = base + j;
                    float n = mulFloat32s(
                        subFloat32s(mulFloat32s((float)in[off], inScale), mean[k]), invSigma[k]);
                    absMax = maxFloat32s(absMax, absFloat32(n));
                }
            }
        }
    }

    float sNorm;
    if (allConstant || absMax == 0.0f) {
        /* Constant input (integer fast-path) OR exact float cancellation
         * (blocks internally constant, products exactly representable):
         * emit all-zero mantissas with scale 1.0 — mirrors the absMax==0
         * idiom in convertFloatTensorToSymInt32Tensor. */
        sNorm = 1.0f;
        for (size_t i = 0; i < total; i++) {
            out[i] = 0;
        }
    } else {
        float stretch = divFloat32s(qMax, absMax);
        sNorm = divFloat32s(1.0f, stretch);

        /* Pass 2: normalize from the same stored stats, quantize. The n
         * expression is IDENTICAL to pass 1's absmax expression (and both are
         * scalar-op calls, so no -ffp-contract divergence between passes);
         * the clamp absorbs any residual boundary case. */
        for (size_t b = 0; b < B; b++) {
            for (size_t grp = 0; grp < G; grp++) {
                size_t k = b * G + grp;
                size_t base = (b * cfg->numChannels + grp * cpg) * T;
                for (size_t j = 0; j < N; j++) {
                    size_t off = base + j;
                    float n = mulFloat32s(
                        subFloat32s(mulFloat32s((float)in[off], inScale), mean[k]), invSigma[k]);
                    out[off] = roundByMode(clamp(mulFloat32s(n, stretch), qMin, qMax),
                                           outQC->roundingMode);
                }
            }
        }
    }

    groupNormAffineSymInt32(cfg, gamma, beta, output, sNorm);
}

/* executeOp forward kernel adapters — operands {input, gamma, beta}; ctx =
 * cfg (eps/numGroups/numChannels geometry, not a tensor so it cannot travel
 * through the funnel's operand array). The SYM kernel emits a RAW, unrestored
 * producer scale: the OUT_WRITE epilogue (groupNormForward) restores width
 * via the SYM->SYM diagonal requant, same as Linear/Conv1d's matmul-family
 * forwards. */
static void groupNormForwardKernelFloat(tensor_t **ops, size_t n, tensor_t *rawOut,
                                        tensor_t *auxOut, const void *ctx) {
    (void)n;
    (void)auxOut;
    groupNormForwardFloat((groupNormConfig_t *)ctx, ops[1], ops[2], ops[0], rawOut);
}
static void groupNormForwardKernelSym(tensor_t **ops, size_t n, tensor_t *rawOut, tensor_t *auxOut,
                                      const void *ctx) {
    (void)n;
    (void)auxOut;
    groupNormForwardSymInt32((groupNormConfig_t *)ctx, ops[1], ops[2], ops[0], rawOut);
}

void groupNormForward(layer_t *layer, tensor_t *input, tensor_t *output) {
    groupNormConfig_t *cfg = layer->config->groupNorm;
    groupNormValidateInputShape(cfg, input);

    executeOp(
        &(opSpec_t){
            .kernel = cfg->forwardMath.type == ARITH_SYM_INT32 ? groupNormForwardKernelSym
                                                               : groupNormForwardKernelFloat,
            .ctx = cfg,
            .inputs = (tensor_t *[]){input, getParamFromParameter(cfg->gamma),
                                     getParamFromParameter(cfg->beta)},
            .nInputs = 3,
            .arithmetic = cfg->forwardMath,
            .mode = OUT_WRITE,
        },
        output);
}

static void groupNormBackwardFloat(groupNormConfig_t *cfg, tensor_t *forwardInput, tensor_t *loss,
                                   tensor_t *propLoss) {
    float *x = (float *)forwardInput->data;
    float *dy = (float *)loss->data;
    float *dx = (float *)propLoss->data;
    float *gamma = (float *)cfg->gamma->param->data;
    float *dgamma = (float *)cfg->gamma->grad->data; /* accumulated += */
    float *dbeta = (float *)cfg->beta->grad->data;   /* accumulated += */

    size_t K;
    size_t N;
    size_t cpg;
    size_t B;
    size_t T;
    groupNormGroupGeom(forwardInput, cfg, &K, &N, &cpg, &B, &T);
    if (K == 0 || N == 0) {
        return; /* empty group geometry: no grad increments, nothing to scatter */
    }

    /* Stats recomputed from forwardInput (no cache) through the SAME shared
     * Reduce helper the forward uses, so backward can never desync on the
     * stats definition; mean[k]/invSigma[k] feed both the grad pass and the
     * dx scatter. */
    float mean[K];
    float invSigma[K];
    groupNormAllGroupStats(forwardInput, cfg, B, cpg, T, K, cfg->eps, mean, invSigma);

    size_t G = cfg->numGroups;
    for (size_t b = 0; b < B; b++) {
        for (size_t grp = 0; grp < G; grp++) {
            size_t k = b * G + grp;
            size_t base = (b * cfg->numChannels + grp * cpg) * T;

            /* Pass over the block: accumulate dgamma/dbeta per CHANNEL (c, not
             * j — summed over batch and spatial steps) and the two block
             * reductions meanDn, meanDnN. */
            float meanDn = 0.0f;
            float meanDnN = 0.0f;
            for (size_t j = 0; j < N; j++) {
                size_t c = grp * cpg + j / T;
                size_t off = base + j;
                float nval = mulFloat32s(subFloat32s(x[off], mean[k]), invSigma[k]);
                float dyv = dy[off];
                dbeta[c] = addFloat32s(dbeta[c], dyv);                      /* SUM over b,t */
                dgamma[c] = addFloat32s(dgamma[c], mulFloat32s(dyv, nval)); /* SUM over b,t */
                float dn = mulFloat32s(dyv, gamma[c]);
                meanDn = addFloat32s(meanDn, dn);
                meanDnN = addFloat32s(meanDnN, mulFloat32s(dn, nval));
            }
            meanDn = divFloat32s(meanDn, (float)N);
            meanDnN = divFloat32s(meanDnN, (float)N);

            /* dx scattered back to the same physical offset its x came from
             * (overwrite, not accumulate). */
            for (size_t j = 0; j < N; j++) {
                size_t c = grp * cpg + j / T;
                size_t off = base + j;
                float nval = mulFloat32s(subFloat32s(x[off], mean[k]), invSigma[k]);
                float dn = mulFloat32s(dy[off], gamma[c]);
                dx[off] = mulFloat32s(
                    invSigma[k], subFloat32s(subFloat32s(dn, meanDn), mulFloat32s(nval, meanDnN)));
            }
        }
    }
}

/* SYM_INT32 backward (the verified LayerNorm scheme, per-channel deltas).
 * mu/sigma are computed ONCE from forwardInput through groupNormAllGroupStats
 * — the SAME shared Reduce helper the forward uses, so backward can never
 * desync from the forward definition — into K-float stack scratch that both
 * passes read. dy and gamma are dequantized per element via their own scales
 * (float math; dy/gamma mantissas are never integer-summed — only
 * forwardInput is subject to the int32 mantissa-sum bound).
 * pass A: per-CHANNEL grad increments (dgammaInc/dbetaInc[C], summed over
 *         batch, in-group channels and T) + global |dx| absmax
 *         (recompute-over-store via absFloat32/maxFloat32s — no stored
 *         buffer exists for the buffer-based findAbsMaxFloat).
 * The increments then funnel through executeOpIdentityKernel under
 * weightGradAccMode/biasGradAccMode into the grad tensors (any storage dtype).
 * pass B: recompute dx from the same stored stats and quantize into propLoss
 *         via the convertFloatTensorToSymInt32Tensor idiom (scale =
 *         absmax/qMax, round-clamp; absmax==0 -> zeros, scale 1.0). The
 *         propLoss scale is data-dependent and REFRESHED ON EVERY CALL. */
static void groupNormBackwardSymInt32(groupNormConfig_t *cfg, tensor_t *forwardInput,
                                      tensor_t *loss, tensor_t *propLoss) {
    groupNormValidateSymTensor(forwardInput, "forwardInput");
    groupNormValidateSymTensor(loss, "loss");
    groupNormValidateSymTensor(propLoss, "propLoss");
    groupNormValidateSymTensor(cfg->gamma->param, "gamma");
    /* beta->param is never read here (beta does not enter dx; dbeta needs only
     * dy) — deliberately not validated. */

    int32_t *xq = (int32_t *)forwardInput->data;
    int32_t *dyq = (int32_t *)loss->data;
    int32_t *gammaQ = (int32_t *)cfg->gamma->param->data;
    int32_t *dxq = (int32_t *)propLoss->data;
    float inScale = ((symInt32QConfig_t *)forwardInput->quantization->qConfig)->scale;
    float dyScale = ((symInt32QConfig_t *)loss->quantization->qConfig)->scale;
    float gammaScale = ((symInt32QConfig_t *)cfg->gamma->param->quantization->qConfig)->scale;
    symInt32QConfig_t *plQC = propLoss->quantization->qConfig;
    /* One-time config-derived range constants — orchestration (see forward). */
    const float qHalfRange = powf(2, (float)(plQC->qMaxBits - 1));
    const float qMax = subFloat32s(qHalfRange, 1.0f);
    const float qMin = -qHalfRange;

    size_t K;
    size_t N;
    size_t cpg;
    size_t B;
    size_t T;
    groupNormGroupGeom(forwardInput, cfg, &K, &N, &cpg, &B, &T);
    if (K == 0 || N == 0) {
        plQC->scale = 1.0f; /* nothing to do; neutral scale (cf. #160, forward) */
        return;
    }

    size_t C = cfg->numChannels;
    float dgammaInc[C]; /* per-CHANNEL (length C, NOT N): summed over b, in- */
    float dbetaInc[C];  /* group channel position and t before the funnel add */
    for (size_t c = 0; c < C; c++) {
        dgammaInc[c] = 0.0f;
        dbetaInc[c] = 0.0f;
    }

    float mean[K];
    float invSigma[K];
    groupNormAllGroupStats(forwardInput, cfg, B, cpg, T, K, cfg->eps, mean, invSigma);

    size_t G = cfg->numGroups;
    float absMax = 0.0f;
    /* Pass A: grad increments (SUM into the per-channel scratch) + global
     * |dx| absmax. */
    for (size_t b = 0; b < B; b++) {
        for (size_t grp = 0; grp < G; grp++) {
            size_t k = b * G + grp;
            size_t base = (b * C + grp * cpg) * T;
            float meanDn = 0.0f;
            float meanDnN = 0.0f;
            for (size_t j = 0; j < N; j++) {
                size_t c = grp * cpg + j / T;
                size_t off = base + j;
                float nval = mulFloat32s(subFloat32s(mulFloat32s((float)xq[off], inScale), mean[k]),
                                         invSigma[k]);
                float dyv = mulFloat32s((float)dyq[off], dyScale);
                dbetaInc[c] = addFloat32s(dbetaInc[c], dyv);
                dgammaInc[c] = addFloat32s(dgammaInc[c], mulFloat32s(dyv, nval));
                float dn = mulFloat32s(dyv, mulFloat32s((float)gammaQ[c], gammaScale));
                meanDn = addFloat32s(meanDn, dn);
                meanDnN = addFloat32s(meanDnN, mulFloat32s(dn, nval));
            }
            meanDn = divFloat32s(meanDn, (float)N);
            meanDnN = divFloat32s(meanDnN, (float)N);
            for (size_t j = 0; j < N; j++) {
                size_t c = grp * cpg + j / T;
                size_t off = base + j;
                float nval = mulFloat32s(subFloat32s(mulFloat32s((float)xq[off], inScale), mean[k]),
                                         invSigma[k]);
                float dn = mulFloat32s(mulFloat32s((float)dyq[off], dyScale),
                                       mulFloat32s((float)gammaQ[c], gammaScale));
                float a = absFloat32(mulFloat32s(
                    invSigma[k], subFloat32s(subFloat32s(dn, meanDn), mulFloat32s(nval, meanDnN))));
                absMax = maxFloat32s(absMax, a);
            }
        }
    }

    quantization_t incQ;
    initFloat32Quantization(&incQ);
    tensor_t dgammaT;
    setTensorValues(&dgammaT, (uint8_t *)dgammaInc, cfg->gamma->grad->shape, &incQ,
                    cfg->gamma->grad->sparsity);
    tensor_t dbetaT;
    setTensorValues(&dbetaT, (uint8_t *)dbetaInc, cfg->beta->grad->shape, &incQ,
                    cfg->beta->grad->sparsity);
    executeOpValidateAccMode(cfg->weightGradAccMode, "GroupNorm weightGradAccMode");
    executeOp(
        &(opSpec_t){
            .kernel = executeOpIdentityKernel,
            .inputs = (tensor_t *[]){&dgammaT},
            .nInputs = 1,
            .arithmetic = (arithmetic_t){.type = ARITH_FLOAT32, .roundingMode = HALF_AWAY},
            .mode = cfg->weightGradAccMode,
        },
        cfg->gamma->grad);
    executeOpValidateAccMode(cfg->biasGradAccMode, "GroupNorm biasGradAccMode");
    executeOp(
        &(opSpec_t){
            .kernel = executeOpIdentityKernel,
            .inputs = (tensor_t *[]){&dbetaT},
            .nInputs = 1,
            .arithmetic = (arithmetic_t){.type = ARITH_FLOAT32, .roundingMode = HALF_AWAY},
            .mode = cfg->biasGradAccMode,
        },
        cfg->beta->grad);

    /* dx requant: convertFloatTensorToSymInt32Tensor idiom (whole-tensor
     * absmax -> scale -> round-clamp). NO integer dy==0 pre-check is needed
     * (unlike the forward's constant-input case): the only realistic
     * absmax==0 source is dy == 0, and zero PROPAGATES exactly through
     * products and sums — the float check is reliable here. */
    if (absMax == 0.0f) {
        size_t total = K * N;
        for (size_t i = 0; i < total; i++) {
            dxq[i] = 0;
        }
        plQC->scale = 1.0f;
        return;
    }

    float dxScale = divFloat32s(absMax, qMax);
    /* Pass B: recompute dx from the stored stats and quantize. The dx
     * expression is IDENTICAL to pass A's absmax expression (all scalar-op
     * calls, no contraction divergence); the clamp absorbs any residual
     * boundary case. The propLoss scale is data-dependent and REFRESHED ON
     * EVERY CALL — a stale scale silently corrupts the downstream layer. */
    for (size_t b = 0; b < B; b++) {
        for (size_t grp = 0; grp < G; grp++) {
            size_t k = b * G + grp;
            size_t base = (b * C + grp * cpg) * T;
            float meanDn = 0.0f;
            float meanDnN = 0.0f;
            for (size_t j = 0; j < N; j++) {
                size_t c = grp * cpg + j / T;
                size_t off = base + j;
                float nval = mulFloat32s(subFloat32s(mulFloat32s((float)xq[off], inScale), mean[k]),
                                         invSigma[k]);
                float dyv = mulFloat32s((float)dyq[off], dyScale);
                float dn = mulFloat32s(dyv, mulFloat32s((float)gammaQ[c], gammaScale));
                meanDn = addFloat32s(meanDn, dn);
                meanDnN = addFloat32s(meanDnN, mulFloat32s(dn, nval));
            }
            meanDn = divFloat32s(meanDn, (float)N);
            meanDnN = divFloat32s(meanDnN, (float)N);
            for (size_t j = 0; j < N; j++) {
                size_t c = grp * cpg + j / T;
                size_t off = base + j;
                float nval = mulFloat32s(subFloat32s(mulFloat32s((float)xq[off], inScale), mean[k]),
                                         invSigma[k]);
                float dn = mulFloat32s(mulFloat32s((float)dyq[off], dyScale),
                                       mulFloat32s((float)gammaQ[c], gammaScale));
                float dxv = mulFloat32s(
                    invSigma[k], subFloat32s(subFloat32s(dn, meanDn), mulFloat32s(nval, meanDnN)));
                dxq[off] =
                    roundByMode(clamp(divFloat32s(dxv, dxScale), qMin, qMax), plQC->roundingMode);
            }
        }
    }
    plQC->scale = dxScale;
}

void groupNormBackward(layer_t *layer, tensor_t *forwardInput, tensor_t *loss, tensor_t *propLoss) {
    groupNormConfig_t *cfg = layer->config->groupNorm;
    groupNormValidateInputShape(cfg, forwardInput);
    switch (cfg->propLossMath.type) {
    case ARITH_FLOAT32:
        /* SYM_INT32 forwardMath + FLOAT32 backwardMath is an inference-only
         * profile: reading a SYM_INT32 forwardInput / loss / gamma as float*
         * here would be silent garbage — fail fast. Training a SYM forward
         * requires backwardMath = SYM_INT32 (factory rule, PR-3). */
        if (forwardInput->quantization->type != FLOAT32 || loss->quantization->type != FLOAT32 ||
            cfg->gamma->param->quantization->type != FLOAT32) {
            PRINT_ERROR("GroupNorm backward: FLOAT32 backward requires FLOAT32 tensors "
                        "(SYM_INT32 forwardMath + FLOAT32 backwardMath is inference-only; "
                        "use SYM_INT32 backwardMath to train)");
            exit(1);
        }
        /* groupNormBackwardFloat raw-casts cfg->gamma->grad->data /
         * cfg->beta->grad->data to float* (it bypasses the executeOp funnel,
         * unlike the SYM_INT32 path's dgamma/dbeta identity-kernel executeOp
         * calls). A packed (SYM/ASYM) grad tensor read/written that way is
         * silent memory corruption, not garbage values — fail fast instead.
         * PR3 (#261): routing float dgamma/dbeta through the funnel like the
         * SYM_INT32 path is a follow-up issue; this guard only closes the gap
         * until then (the LayerNorm precedent). */
        if (cfg->gamma->grad->quantization->type != FLOAT32 ||
            cfg->beta->grad->quantization->type != FLOAT32) {
            PRINT_ERROR("GroupNorm backward: FLOAT32 backward writes gamma/beta grads via a raw "
                        "float* cast — packed grad storage requires the funnel route (follow-up "
                        "issue, #261) — got gamma grad dtype %d, beta grad dtype %d",
                        (int)cfg->gamma->grad->quantization->type,
                        (int)cfg->beta->grad->quantization->type);
            exit(1);
        }
        /* groupNormBackwardFloat also writes propLoss->data (dx) via a raw
         * float* cast. A SYM-storage propLossQ (SYM_INT32 fixed-point, or packed
         * sub-byte SYM) paired with FLOAT32 propLossMath is factory-constructible,
         * and that raw write silently corrupts the mantissa/packed buffer — fail
         * fast instead (same #261 gap the gamma/beta grad guard closes). */
        if (propLoss->quantization->type != FLOAT32) {
            PRINT_ERROR("GroupNorm backward: FLOAT32 backward writes propLoss (dx) via a raw "
                        "float* cast — SYM/packed propLoss storage requires the funnel route "
                        "(follow-up issue, #261) — got propLoss dtype %d",
                        (int)propLoss->quantization->type);
            exit(1);
        }
        groupNormBackwardFloat(cfg, forwardInput, loss, propLoss);
        break;
    case ARITH_SYM_INT32:
        groupNormBackwardSymInt32(cfg, forwardInput, loss, propLoss);
        break;
    default:
        PRINT_ERROR(
            "GroupNorm backward: quantization type not implemented (FLOAT32/SYM_INT32 only)");
        exit(1);
    }
}

void groupNormCalcOutputShape(layer_t *layer, shape_t *inputShape, shape_t *outputShape) {
    (void)layer;
    memcpy(outputShape->dimensions, inputShape->dimensions,
           inputShape->numberOfDimensions * sizeof(size_t));
    memcpy(outputShape->orderOfDimensions, inputShape->orderOfDimensions,
           inputShape->numberOfDimensions * sizeof(size_t));
    outputShape->numberOfDimensions = inputShape->numberOfDimensions;
}
