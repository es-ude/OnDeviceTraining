#define SOURCE_FILE "GROUPNORM"

#include <math.h> /* powf: one-time config-derived range constants only (orchestration);
                   * ldexpf: exact BFP mantissa*2^E dequant (no rounding) */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "GroupNorm.h"

#include "Add.h"
#include "Arithmetic.h"
#include "ArithmeticType.h"
#include "BfpKernelSupport.h"
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
    cfg->frozen = false;
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
static void groupNormGroupGeom(tensor_t *input, const groupNormConfig_t *cfg, size_t *K, size_t *N,
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
static void groupNormAllGroupStats(tensor_t *t, const groupNormConfig_t *cfg, size_t B, size_t cpg,
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

    switch (t->quantization->type) {
    case SYM_INT32:
        meanOverTrailingAxesSymInt32(&view, 2, &meanT);
        varianceBiasedOverTrailingAxesSymInt32(&view, 2, &meanT, &varT);
        break;
    case BFP:
        /* The view ALIASES both t's data and t's quantization (setTensorValues
         * above), so the Reduce BFP arms fold on the operand's own grid. That
         * is sound here because the view only RELABELS dims: the layer's
         * rank-3/identity-order gate makes the storage layout row-major, and
         * the [B,G,cpg,T] split of C keeps that, so a view element's flat
         * index EQUALS its storage index and bfpGroupOf(qC, off) resolves the
         * exponent group the code was actually packed into. */
        meanOverTrailingAxesBfp(&view, 2, &meanT);
        varianceBiasedOverTrailingAxesBfp(&view, 2, &meanT, &varT);
        break;
    case FLOAT32:
        meanOverTrailingAxesFloat32(&view, 2, &meanT);
        varianceBiasedOverTrailingAxesFloat32(&view, 2, &meanT, &varT);
        break;
    default:
        /* PR2 Task 9 ruling (LayerNorm twin): a fall-through would hand int32
         * mantissa scratch to the float reducer through a float* cast --
         * silent wrong arithmetic, not a crash. Explicit switch, fail-fast
         * default. */
        PRINT_ERROR("GroupNorm stats: operand dtype %d has no stats path "
                    "(FLOAT32/SYM_INT32/BFP)",
                    (int)t->quantization->type);
        exit(1);
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

/* R-N1 (the R-P1 weight-less anchor at the norm layer): norms have no
 * reduction-weight operand, so the staging width anchor for FLOAT32-stored
 * operands is the layer's OWN produced-wire config -- outputQ for the forward
 * op, propLossQ for the backward ops. Eager at op entry: without a BFP-typed
 * wire config there is no width source at all. NULL-checked because userApi
 * factories copy layerQuant_t slots by value (a pinned ARITH_BFP slot can
 * arrive with a NULL or non-BFP wire config). */
static const bfpQConfig_t *groupNormBfpWireAnchor(const quantization_t *wireQ, const char *what) {
    if (wireQ == NULL || wireQ->type != BFP) {
        PRINT_ERROR("%s: ARITH_BFP requires a BFP-typed produced-wire config as the staging "
                    "width anchor (outputQ forward / propLossQ backward) -- see "
                    "docs/conventions/arithmetic-bfp.md",
                    what);
        exit(1);
    }
    return wireQ->qConfig;
}

/* F5-style count gate for the BFP kernels' flat gamma/beta/raw indexing. */
static void groupNormBfpRequireCount(tensor_t *t, size_t expected, const char *what) {
    size_t n = calcNumberOfElementsByTensor(t);
    if (n != expected) {
        PRINT_ERROR("%s: element count %zu != expected %zu", what, n, expected);
        exit(1);
    }
}

/* ARITH_BFP forward (R-N2/R-N3): BFP stats via the [B,G,cpg,T] alias view,
 * per-channel float affine, FLOAT32 raw (D7); OUT_WRITE packs the wire. The
 * SYM path's integer-affine/beta-seed bookkeeping has no BFP analog (a BFP
 * scale is 2^E; like AvgPool's /K fold, R-P4). Operands arrive in the funnel's
 * unpacked-BFP scratch form (borrowed or staged). */
static void groupNormForwardBfp(const groupNormConfig_t *cfg, tensor_t *gamma, tensor_t *beta,
                                tensor_t *input, tensor_t *rawOut) {
    size_t K;
    size_t N;
    size_t cpg;
    size_t B;
    size_t T;
    groupNormGroupGeom(input, cfg, &K, &N, &cpg, &B, &T);
    if (K == 0 || N == 0) {
        return;
    }
    const bfpQConfig_t *xQC = input->quantization->qConfig;
    const bfpQConfig_t *gQC = gamma->quantization->qConfig;
    const bfpQConfig_t *bQC = beta->quantization->qConfig;
    validateBfpQConfigShape(xQC, calcNumberOfElementsByTensor(input));
    groupNormBfpRequireCount(gamma, cfg->numChannels, "GroupNorm forward BFP gamma");
    groupNormBfpRequireCount(beta, cfg->numChannels, "GroupNorm forward BFP beta");
    validateBfpQConfigShape(gQC, cfg->numChannels); /* grid check per operand -- the count gate
                                                     * alone cannot catch a malformed grid
                                                     * (LayerNorm forward precedent) */
    validateBfpQConfigShape(bQC, cfg->numChannels);

    float mean[K];
    float invSigma[K];
    groupNormAllGroupStats(input, cfg, B, cpg, T, K, cfg->eps, mean, invSigma);

    const int32_t xBias = bfpExponentBias(xQC);
    const int32_t gBias = bfpExponentBias(gQC);
    const int32_t bBias = bfpExponentBias(bQC);
    int32_t const *xArr = (int32_t const *)input->data;
    int32_t const *gArr = (int32_t const *)gamma->data;
    int32_t const *bArr = (int32_t const *)beta->data;
    float *yArr = (float *)rawOut->data;
    /* Flat contiguous walk (identity order enforced by the layer's shape
     * gate): block k = (b, grp), base = (b*C + grp*cpg)*T, element
     * off = base + j, channel c = grp*cpg + j/T. cfg->numGroups is the LAYER's
     * channel-group count -- unrelated to any bfpQConfig_t's numGroups, which
     * is the exponent-block count and is only ever reached through
     * bfpGroupOf(). */
    for (size_t b = 0; b < B; b++) {
        for (size_t grp = 0; grp < cfg->numGroups; grp++) {
            size_t k = b * cfg->numGroups + grp;
            size_t base = (b * cfg->numChannels + grp * cpg) * T;
            for (size_t j = 0; j < N; j++) {
                size_t off = base + j;
                size_t c = grp * cpg + j / T;
                float x =
                    ldexpf((float)xArr[off], (int)xQC->exponents[bfpGroupOf(xQC, off)] - xBias);
                float nval = mulFloat32s(subFloat32s(x, mean[k]), invSigma[k]);
                float gv = ldexpf((float)gArr[c], (int)gQC->exponents[bfpGroupOf(gQC, c)] - gBias);
                float bv = ldexpf((float)bArr[c], (int)bQC->exponents[bfpGroupOf(bQC, c)] - bBias);
                yArr[off] = addFloat32s(mulFloat32s(gv, nval), bv);
            }
        }
    }
}

static void groupNormForwardKernelBfp(tensor_t **operands, size_t nOperands, tensor_t *rawOut,
                                      tensor_t *auxOut, const void *ctx) {
    (void)nOperands;
    (void)auxOut;
    const groupNormConfig_t *cfg = ctx;
    groupNormForwardBfp(cfg, operands[1], operands[2], operands[0], rawOut);
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

/* Explicit kernel dispatch, NOT a ternary — the LayerNorm twin (BFP epic PR2
 * Task 9): a ternary hands every non-SYM arithmetic to the FLOAT kernel, and
 * since the derivation flip a BFP profile arrives here as ARITH_BFP, whose
 * funnel-unpacked int32 mantissa scratch groupNormForwardFloat would read
 * through a float* cast — silent wrong arithmetic, not a crash. ARITH_BFP never
 * reaches this select since epic PR5 (groupNormForward early-returns through
 * its own arm), so the default covers only genuinely unimplemented
 * arithmetic. */
static opKernelFn_t groupNormSelectForwardKernel(const groupNormConfig_t *cfg) {
    switch (cfg->forwardMath.type) {
    case ARITH_FLOAT32:
        return groupNormForwardKernelFloat;
    case ARITH_SYM_INT32:
        return groupNormForwardKernelSym;
    default:
        PRINT_ERROR("GroupNorm forward: declared forwardMath %d not implemented",
                    (int)cfg->forwardMath.type);
        exit(1);
    }
}

void groupNormForward(layer_t *layer, tensor_t *input, tensor_t *output) {
    groupNormConfig_t *cfg = layer->config->groupNorm;
    groupNormValidateInputShape(cfg, input);

    if (cfg->forwardMath.type == ARITH_BFP) {
        const bfpQConfig_t *anchor = groupNormBfpWireAnchor(cfg->outputQ, "GroupNorm forward");
        bfpQConfig_t stage = {.exponents = NULL,
                              .numGroups = 1,
                              .groupSize = 0,
                              .roundingMode = cfg->forwardMath.roundingMode,
                              .mantissaBits = anchor->mantissaBits,
                              .exponentBits = anchor->exponentBits};
        tensor_t *gammaT = getParamFromParameter(cfg->gamma);
        tensor_t *betaT = getParamFromParameter(cfg->beta);
        executeOp(
            &(opSpec_t){
                .kernel = groupNormForwardKernelBfp,
                .ctx = cfg,
                .inputs = (tensor_t *[]){input, gammaT, betaT},
                .nInputs = 3,
                .arithmetic = cfg->forwardMath,
                .mode = OUT_WRITE,
                .bfpStage = {input->quantization->type == FLOAT32 ? &stage : NULL,
                             gammaT->quantization->type == FLOAT32 ? &stage : NULL,
                             betaT->quantization->type == FLOAT32 ? &stage : NULL},
            },
            output);
        return;
    }

    executeOp(
        &(opSpec_t){
            .kernel = groupNormSelectForwardKernel(cfg),
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
    /* propLoss == NULL (#380 PR2): grads-only call -- fetch NULL instead of
     * dereferencing the absent buffer; the scatter loop below is guarded on
     * dx and never touches propLoss in that case. */
    float *dx = (propLoss != NULL) ? (float *)propLoss->data : NULL;
    float *gamma = (float *)cfg->gamma->param->data;
    const bool frozen = cfg->frozen;
    /* Frozen: no grad tensors exist (Task 1 elides them) -- fetch NULL instead
     * of dereferencing cfg->gamma->grad / cfg->beta->grad. */
    float *dgamma = frozen ? NULL : (float *)cfg->gamma->grad->data; /* accumulated += */
    float *dbeta = frozen ? NULL : (float *)cfg->beta->grad->data;   /* accumulated += */

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
                if (!frozen) {
                    dbeta[c] = addFloat32s(dbeta[c], dyv);                      /* SUM over b,t */
                    dgamma[c] = addFloat32s(dgamma[c], mulFloat32s(dyv, nval)); /* SUM over b,t */
                }
                float dn = mulFloat32s(dyv, gamma[c]);
                meanDn = addFloat32s(meanDn, dn);
                meanDnN = addFloat32s(meanDnN, mulFloat32s(dn, nval));
            }
            meanDn = divFloat32s(meanDn, (float)N);
            meanDnN = divFloat32s(meanDnN, (float)N);

            /* dx scattered back to the same physical offset its x came from
             * (overwrite, not accumulate). propLoss == NULL (#380 PR2):
             * grads-only call -- skip the scatter entirely (no propLoss->...
             * touch, no dx write). */
            if (dx != NULL) {
                for (size_t j = 0; j < N; j++) {
                    size_t c = grp * cpg + j / T;
                    size_t off = base + j;
                    float nval = mulFloat32s(subFloat32s(x[off], mean[k]), invSigma[k]);
                    float dn = mulFloat32s(dy[off], gamma[c]);
                    dx[off] = mulFloat32s(invSigma[k], subFloat32s(subFloat32s(dn, meanDn),
                                                                   mulFloat32s(nval, meanDnN)));
                }
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
    /* propLoss == NULL (#380 PR2): grads-only call -- skip validating and
     * fetching the absent buffer; pass B (below) never runs in that case. */
    if (propLoss != NULL) {
        groupNormValidateSymTensor(propLoss, "propLoss");
    }
    groupNormValidateSymTensor(cfg->gamma->param, "gamma");
    /* beta->param is never read here (beta does not enter dx; dbeta needs only
     * dy) — deliberately not validated. */

    int32_t *xq = (int32_t *)forwardInput->data;
    int32_t *dyq = (int32_t *)loss->data;
    int32_t *gammaQ = (int32_t *)cfg->gamma->param->data;
    int32_t *dxq = (propLoss != NULL) ? (int32_t *)propLoss->data : NULL;
    float inScale = ((symInt32QConfig_t *)forwardInput->quantization->qConfig)->scale;
    float dyScale = ((symInt32QConfig_t *)loss->quantization->qConfig)->scale;
    float gammaScale = ((symInt32QConfig_t *)cfg->gamma->param->quantization->qConfig)->scale;
    symInt32QConfig_t *plQC =
        (propLoss != NULL) ? (symInt32QConfig_t *)propLoss->quantization->qConfig : NULL;

    size_t K;
    size_t N;
    size_t cpg;
    size_t B;
    size_t T;
    groupNormGroupGeom(forwardInput, cfg, &K, &N, &cpg, &B, &T);
    if (K == 0 || N == 0) {
        if (propLoss != NULL) {
            plQC->scale = 1.0f; /* nothing to do; neutral scale (cf. #160, forward) */
        }
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

    if (!cfg->frozen) {
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
    }

    /* dx requant: convertFloatTensorToSymInt32Tensor idiom (whole-tensor
     * absmax -> scale -> round-clamp). NO integer dy==0 pre-check is needed
     * (unlike the forward's constant-input case): the only realistic
     * absmax==0 source is dy == 0, and zero PROPAGATES exactly through
     * products and sums — the float check is reliable here.
     * propLoss == NULL (#380 PR2): grads-only call -- skip pass B entirely
     * (no dx write, no propLoss scale refresh; stats/absmax bookkeeping in
     * pass A above already ran unconditionally). */
    if (propLoss != NULL) {
        /* One-time config-derived range constants — orchestration (see forward). */
        const float qHalfRange = powf(2, (float)(plQC->qMaxBits - 1));
        const float qMax = subFloat32s(qHalfRange, 1.0f);
        const float qMin = -qHalfRange;

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
         * expression is IDENTICAL to pass A's absmax expression (all
         * scalar-op calls, no contraction divergence); the clamp absorbs
         * any residual boundary case. The propLoss scale is data-dependent
         * and REFRESHED ON EVERY CALL — a stale scale silently corrupts the
         * downstream layer. */
        for (size_t b = 0; b < B; b++) {
            for (size_t grp = 0; grp < G; grp++) {
                size_t k = b * G + grp;
                size_t base = (b * C + grp * cpg) * T;
                float meanDn = 0.0f;
                float meanDnN = 0.0f;
                for (size_t j = 0; j < N; j++) {
                    size_t c = grp * cpg + j / T;
                    size_t off = base + j;
                    float nval = mulFloat32s(
                        subFloat32s(mulFloat32s((float)xq[off], inScale), mean[k]), invSigma[k]);
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
                    float nval = mulFloat32s(
                        subFloat32s(mulFloat32s((float)xq[off], inScale), mean[k]), invSigma[k]);
                    float dn = mulFloat32s(mulFloat32s((float)dyq[off], dyScale),
                                           mulFloat32s((float)gammaQ[c], gammaScale));
                    float dxv = mulFloat32s(invSigma[k], subFloat32s(subFloat32s(dn, meanDn),
                                                                     mulFloat32s(nval, meanDnN)));
                    dxq[off] = roundByMode(clamp(divFloat32s(dxv, dxScale), qMin, qMax),
                                           plQC->roundingMode);
                }
            }
        }
        plQC->scale = dxScale;
    }
}

/* dbeta_c = sum over (b, t) of dy[b,c,t]: a pure mantissa VALUE-sum. Within
 * one (b,c) row the T elements are storage-contiguous -- real same-exponent
 * segments, so the R-P4 int32 segment-fold does actual work here (unlike
 * LayerNorm's strided dbeta). Raw is [C] FLOAT32 and memset here (#427:
 * funnel Phase-2 raw is uninitialized scratch). */
static void groupNormCalcBetaGradsBfp(const groupNormConfig_t *cfg, tensor_t *loss,
                                      tensor_t *rawOut) {
    size_t K;
    size_t N;
    size_t cpg;
    size_t B;
    size_t T;
    groupNormGroupGeom(loss, cfg, &K, &N, &cpg, &B, &T);
    groupNormBfpRequireCount(rawOut, cfg->numChannels, "GroupNorm dbeta BFP raw");
    float *out = (float *)rawOut->data;
    memset(out, 0, cfg->numChannels * sizeof(float));
    if (K == 0 || N == 0) {
        return;
    }
    const bfpQConfig_t *dyQC = loss->quantization->qConfig;
    validateBfpQConfigShape(dyQC, calcNumberOfElementsByTensor(loss));
    bfpValidateSumHeadroom(dyQC, B * T, "GroupNorm dbeta BFP");
    const int32_t dyBias = bfpExponentBias(dyQC);
    int32_t const *dyArr = (int32_t const *)loss->data;
    for (size_t c = 0; c < cfg->numChannels; c++) {
        float acc = 0.0f;
        int32_t partial = 0;
        size_t currentGroup = 0;
        bool haveSeg = false;
        for (size_t b = 0; b < B; b++) {
            for (size_t t = 0; t < T; t++) {
                size_t off = (b * cfg->numChannels + c) * T + t;
                size_t grp = bfpGroupOf(dyQC, off);
                if (!haveSeg) {
                    currentGroup = grp;
                    haveSeg = true;
                } else if (grp != currentGroup) {
                    acc = addFloat32s(
                        acc, ldexpf((float)partial, (int)dyQC->exponents[currentGroup] - dyBias));
                    partial = 0;
                    currentGroup = grp;
                }
                partial = addInt32s(partial, dyArr[off]);
            }
        }
        if (haveSeg) {
            acc = addFloat32s(acc,
                              ldexpf((float)partial, (int)dyQC->exponents[currentGroup] - dyBias));
        }
        out[c] = acc;
    }
}

/* dgamma_c = sum over (b, t) of dy[b,c,t] * n[b,c,t]. n is float, so the
 * accumulation is per-element float32 (no int32 partial for a mixed
 * mantissa-x-float product). Stats recomputed here (R-N4 per-op cost). */
static void groupNormCalcGammaGradsBfp(const groupNormConfig_t *cfg, tensor_t *forwardInput,
                                       tensor_t *loss, tensor_t *rawOut) {
    size_t K;
    size_t N;
    size_t cpg;
    size_t B;
    size_t T;
    groupNormGroupGeom(forwardInput, cfg, &K, &N, &cpg, &B, &T);
    groupNormBfpRequireCount(rawOut, cfg->numChannels, "GroupNorm dgamma BFP raw");
    float *out = (float *)rawOut->data;
    memset(out, 0, cfg->numChannels * sizeof(float));
    if (K == 0 || N == 0) {
        return;
    }
    const bfpQConfig_t *xQC = forwardInput->quantization->qConfig;
    const bfpQConfig_t *dyQC = loss->quantization->qConfig;
    validateBfpQConfigShape(xQC, calcNumberOfElementsByTensor(forwardInput));
    validateBfpQConfigShape(dyQC, calcNumberOfElementsByTensor(loss));
    groupNormBfpRequireCount(loss, calcNumberOfElementsByTensor(forwardInput),
                             "GroupNorm dgamma BFP loss");

    float mean[K];
    float invSigma[K];
    groupNormAllGroupStats(forwardInput, cfg, B, cpg, T, K, cfg->eps, mean, invSigma);

    const int32_t xBias = bfpExponentBias(xQC);
    const int32_t dyBias = bfpExponentBias(dyQC);
    int32_t const *xArr = (int32_t const *)forwardInput->data;
    int32_t const *dyArr = (int32_t const *)loss->data;
    for (size_t b = 0; b < B; b++) {
        for (size_t grp = 0; grp < cfg->numGroups; grp++) {
            size_t k = b * cfg->numGroups + grp;
            size_t base = (b * cfg->numChannels + grp * cpg) * T;
            for (size_t j = 0; j < N; j++) {
                size_t off = base + j;
                size_t c = grp * cpg + j / T;
                float x =
                    ldexpf((float)xArr[off], (int)xQC->exponents[bfpGroupOf(xQC, off)] - xBias);
                float nval = mulFloat32s(subFloat32s(x, mean[k]), invSigma[k]);
                float dy =
                    ldexpf((float)dyArr[off], (int)dyQC->exponents[bfpGroupOf(dyQC, off)] - dyBias);
                out[c] = addFloat32s(out[c], mulFloat32s(dy, nval));
            }
        }
    }
}

/* dx = invSigma * (dn - meanDn - n*meanDnN) with dn = dy * gamma_c, all in
 * the FLOAT32 raw from exact dequants; OUT_WRITE packs the propLoss wire.
 * Every element is written -- no memset. */
static void groupNormCalcPropLossBfp(const groupNormConfig_t *cfg, tensor_t *forwardInput,
                                     tensor_t *loss, tensor_t *gamma, tensor_t *rawOut) {
    size_t K;
    size_t N;
    size_t cpg;
    size_t B;
    size_t T;
    groupNormGroupGeom(forwardInput, cfg, &K, &N, &cpg, &B, &T);
    if (K == 0 || N == 0) {
        return;
    }
    const bfpQConfig_t *xQC = forwardInput->quantization->qConfig;
    const bfpQConfig_t *dyQC = loss->quantization->qConfig;
    const bfpQConfig_t *gQC = gamma->quantization->qConfig;
    validateBfpQConfigShape(xQC, calcNumberOfElementsByTensor(forwardInput));
    validateBfpQConfigShape(dyQC, calcNumberOfElementsByTensor(loss));
    groupNormBfpRequireCount(gamma, cfg->numChannels, "GroupNorm dx BFP gamma");
    validateBfpQConfigShape(gQC, cfg->numChannels); /* count gate alone cannot catch a malformed
                                                     * grid; bfpGroupOf(gQC, c) would index
                                                     * exponents[] OOB (Task 3 review finding) */
    /* The walk below is forwardInput's ((b, grp) base + j), but loss is indexed
     * at those offsets: a shorter loss reads outside its scratch. The dgamma
     * twin's gate covers this incidentally when unfrozen -- frozen skips
     * dgamma, so this gate is the sole catcher there. */
    groupNormBfpRequireCount(loss, calcNumberOfElementsByTensor(forwardInput),
                             "GroupNorm dx BFP loss");
    groupNormBfpRequireCount(rawOut, calcNumberOfElementsByTensor(forwardInput),
                             "GroupNorm dx BFP raw");

    float mean[K];
    float invSigma[K];
    groupNormAllGroupStats(forwardInput, cfg, B, cpg, T, K, cfg->eps, mean, invSigma);

    const int32_t xBias = bfpExponentBias(xQC);
    const int32_t dyBias = bfpExponentBias(dyQC);
    const int32_t gBias = bfpExponentBias(gQC);
    int32_t const *xArr = (int32_t const *)forwardInput->data;
    int32_t const *dyArr = (int32_t const *)loss->data;
    int32_t const *gArr = (int32_t const *)gamma->data;
    float *dxArr = (float *)rawOut->data;
    for (size_t b = 0; b < B; b++) {
        for (size_t grp = 0; grp < cfg->numGroups; grp++) {
            size_t k = b * cfg->numGroups + grp;
            size_t base = (b * cfg->numChannels + grp * cpg) * T;
            float sumDn = 0.0f;
            float sumDnN = 0.0f;
            for (size_t j = 0; j < N; j++) {
                size_t off = base + j;
                size_t c = grp * cpg + j / T;
                float x =
                    ldexpf((float)xArr[off], (int)xQC->exponents[bfpGroupOf(xQC, off)] - xBias);
                float nval = mulFloat32s(subFloat32s(x, mean[k]), invSigma[k]);
                float dy =
                    ldexpf((float)dyArr[off], (int)dyQC->exponents[bfpGroupOf(dyQC, off)] - dyBias);
                float gv = ldexpf((float)gArr[c], (int)gQC->exponents[bfpGroupOf(gQC, c)] - gBias);
                float dn = mulFloat32s(dy, gv);
                sumDn = addFloat32s(sumDn, dn);
                sumDnN = addFloat32s(sumDnN, mulFloat32s(dn, nval));
            }
            float meanDn = divFloat32s(sumDn, (float)N);
            float meanDnN = divFloat32s(sumDnN, (float)N);
            for (size_t j = 0; j < N; j++) {
                size_t off = base + j;
                size_t c = grp * cpg + j / T;
                float x =
                    ldexpf((float)xArr[off], (int)xQC->exponents[bfpGroupOf(xQC, off)] - xBias);
                float nval = mulFloat32s(subFloat32s(x, mean[k]), invSigma[k]);
                float dy =
                    ldexpf((float)dyArr[off], (int)dyQC->exponents[bfpGroupOf(dyQC, off)] - dyBias);
                float gv = ldexpf((float)gArr[c], (int)gQC->exponents[bfpGroupOf(gQC, c)] - gBias);
                float dn = mulFloat32s(dy, gv);
                dxArr[off] = mulFloat32s(
                    invSigma[k], subFloat32s(subFloat32s(dn, meanDn), mulFloat32s(nval, meanDnN)));
            }
        }
    }
}

/* executeOp backward kernel adapters for the ARITH_BFP arm -- dgamma
 * {forwardInput, loss}, dbeta {loss}, dx {forwardInput, loss, gamma}. */
static void groupNormDgammaKernelBfp(tensor_t **operands, size_t nOperands, tensor_t *rawOut,
                                     tensor_t *auxOut, const void *ctx) {
    (void)nOperands;
    (void)auxOut;
    const groupNormConfig_t *cfg = ctx;
    groupNormCalcGammaGradsBfp(cfg, operands[0], operands[1], rawOut);
}
static void groupNormDbetaKernelBfp(tensor_t **operands, size_t nOperands, tensor_t *rawOut,
                                    tensor_t *auxOut, const void *ctx) {
    (void)nOperands;
    (void)auxOut;
    const groupNormConfig_t *cfg = ctx;
    groupNormCalcBetaGradsBfp(cfg, operands[0], rawOut);
}
static void groupNormDxKernelBfp(tensor_t **operands, size_t nOperands, tensor_t *rawOut,
                                 tensor_t *auxOut, const void *ctx) {
    (void)nOperands;
    (void)auxOut;
    const groupNormConfig_t *cfg = ctx;
    groupNormCalcPropLossBfp(cfg, operands[0], operands[1], operands[2], rawOut);
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
         * until then (the LayerNorm precedent). Frozen (#380): no grad
         * tensors exist (Task 1 elides them), so there is nothing to
         * validate -- skip the whole check. */
        if (!cfg->frozen && (cfg->gamma->grad->quantization->type != FLOAT32 ||
                             cfg->beta->grad->quantization->type != FLOAT32)) {
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
         * fast instead (same #261 gap the gamma/beta grad guard closes).
         * propLoss == NULL (#380 PR2): grads-only call -- nothing to
         * validate, skip the whole check. */
        if (propLoss != NULL && propLoss->quantization->type != FLOAT32) {
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
    case ARITH_BFP: {
        /* R-N1: propLossQ anchors ALL THREE backward ops' staging, even when
         * the propLoss TENSOR is NULL (the grad ops still stage at it). */
        const bfpQConfig_t *anchor = groupNormBfpWireAnchor(cfg->propLossQ, "GroupNorm backward");
        /* The dbeta kernel derives B/T from the LOSS tensor while offsetting
         * with cfg->numChannels, so a count-equal but shape-PERMUTED loss
         * (e.g. [2,2,4] against a [2,4,2] input) passes every count and grid
         * gate and then reads out of bounds; dgamma/dx read it at
         * forward-derived offsets, in bounds but silently wrong. The loss gets
         * the same shape gate the forward applies to its input. */
        groupNormValidateInputShape(cfg, loss);
        bfpQConfig_t stage = {.exponents = NULL,
                              .numGroups = 1,
                              .groupSize = 0,
                              .roundingMode = cfg->propLossMath.roundingMode,
                              .mantissaBits = anchor->mantissaBits,
                              .exponentBits = anchor->exponentBits};
        tensor_t *gammaT = getParamFromParameter(cfg->gamma);
        const bfpQConfig_t *fiStage = forwardInput->quantization->type == FLOAT32 ? &stage : NULL;
        const bfpQConfig_t *dyStage = loss->quantization->type == FLOAT32 ? &stage : NULL;
        const bfpQConfig_t *gStage = gammaT->quantization->type == FLOAT32 ? &stage : NULL;
        if (!cfg->frozen) {
            executeOpValidateAccMode(cfg->weightGradAccMode, "GroupNorm weightGradAccMode");
            executeOp(&(opSpec_t){.kernel = groupNormDgammaKernelBfp,
                                  .ctx = cfg,
                                  .inputs = (tensor_t *[]){forwardInput, loss},
                                  .nInputs = 2,
                                  .arithmetic = cfg->propLossMath,
                                  .mode = cfg->weightGradAccMode,
                                  .bfpStage = {fiStage, dyStage, NULL}},
                      cfg->gamma->grad);
            executeOpValidateAccMode(cfg->biasGradAccMode, "GroupNorm biasGradAccMode");
            executeOp(&(opSpec_t){.kernel = groupNormDbetaKernelBfp,
                                  .ctx = cfg,
                                  .inputs = (tensor_t *[]){loss},
                                  .nInputs = 1,
                                  .arithmetic = cfg->propLossMath,
                                  .mode = cfg->biasGradAccMode,
                                  .bfpStage = {dyStage, NULL, NULL}},
                      cfg->beta->grad);
        }
        if (propLoss != NULL) {
            executeOp(&(opSpec_t){.kernel = groupNormDxKernelBfp,
                                  .ctx = cfg,
                                  .inputs = (tensor_t *[]){forwardInput, loss, gammaT},
                                  .nInputs = 3,
                                  .arithmetic = cfg->propLossMath,
                                  .mode = OUT_WRITE,
                                  .bfpStage = {fiStage, dyStage, gStage}},
                      propLoss);
        }
        break;
    }
    default:
        PRINT_ERROR("GroupNorm backward: declared propLossMath %d not implemented",
                    (int)cfg->propLossMath.type);
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
