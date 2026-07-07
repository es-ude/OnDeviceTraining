#define SOURCE_FILE "LAYERNORM"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "LayerNorm.h"

#include "Add.h"
#include "Arithmetic.h"
#include "ArithmeticType.h"
#include "Common.h"
#include "ExecuteOp.h"
#include "Layer.h"
#include "Mul.h"
#include "Quantization.h"
#include "Reduce.h"
#include "Rounding.h"
#include "Sub.h"
#include "Tensor.h"
#include "TensorConversion.h"

void initLayerNormConfig(layerNormConfig_t *cfg, parameter_t *gamma, parameter_t *beta,
                         size_t *normalizedShape, size_t numNormDims, float eps,
                         quantization_t *forwardQ, quantization_t *backwardQ) {
    cfg->gamma = gamma;
    cfg->beta = beta;
    cfg->normalizedShape = normalizedShape;
    cfg->numNormDims = numNormDims;
    cfg->eps = eps;
    cfg->forwardMath = arithmeticFromQuantizationOrDefault(forwardQ);
    cfg->propLossMath = arithmeticFromQuantizationOrDefault(backwardQ);
    cfg->outputQ = forwardQ;
    cfg->propLossQ = backwardQ;
    cfg->ownsQuantizations = false;

    /* Today's hardcode at both dgamma/dbeta executeOp call sites
     * (layerNormBackwardSymInt32, below) is OUT_ACC_DYNAMIC_RESCALE for BOTH
     * -- unlike Linear/Conv1d/Conv1dTransposed, LayerNorm's beta grad has
     * never used the FIXED_SCALE bias scheme (dgamma/dbeta both accumulate
     * via the identity-kernel + DYNAMIC_RESCALE requant). Carried on the
     * config so every caller of this init function -- factory-built or
     * hand-wired directly (test/unit/layer/UnitTestLayerNorm.c) -- gets the
     * historical behavior without having to know about the PR3 knob. A
     * layerQuant_t-driven factory overrides these right after this call
     * (LayerNormApi.c) if the caller opted into a different mode. */
    cfg->weightGradAccMode = OUT_ACC_DYNAMIC_RESCALE;
    cfg->biasGradAccMode = OUT_ACC_DYNAMIC_RESCALE;
}

/* Compute G (groups) and N (per-group element count) from a logical shape:
 *   N = product of the last numNormDims logical dims
 *   G = total / N
 * Logical dim sizes honor orderOfDimensions via getDimensionsByIndex. */
static void layerNormGroupSizes(tensor_t *t, size_t numNormDims, size_t *outG, size_t *outN) {
    size_t rank = t->shape->numberOfDimensions;
    if (numNormDims > rank) {
        PRINT_ERROR("LayerNorm: numNormDims (%zu) exceeds input rank (%zu)", numNormDims, rank);
        exit(1);
    }
    size_t total = calcNumberOfElementsByTensor(t);
    size_t n = 1;
    for (size_t d = rank - numNormDims; d < rank; d++) {
        n *= getDimensionsByIndex(t, d);
    }
    *outN = n;
    *outG = (n == 0) ? 0 : total / n;
}

/* Fail fast if the input's trailing D logical dims differ from the configured
 * normalizedShape (mirrors Matmul's shape checks). gamma/beta hold exactly
 * prod(normalizedShape) elements, so a mismatched input would otherwise read
 * gamma/beta out of bounds in forward and WRITE the grad tensors out of
 * bounds in backward — silently. */
static void layerNormValidateInputShape(layerNormConfig_t *cfg, tensor_t *input) {
    size_t rank = input->shape->numberOfDimensions;
    if (cfg->numNormDims > rank) {
        PRINT_ERROR("LayerNorm: numNormDims (%zu) exceeds input rank (%zu)", cfg->numNormDims,
                    rank);
        exit(1);
    }
    for (size_t d = 0; d < cfg->numNormDims; d++) {
        size_t inputDim = getDimensionsByIndex(input, rank - cfg->numNormDims + d);
        if (inputDim != cfg->normalizedShape[d]) {
            PRINT_ERROR("LayerNorm: input trailing dim %zu is %zu but normalizedShape[%zu] is %zu",
                        rank - cfg->numNormDims + d, inputDim, d, cfg->normalizedShape[d]);
            exit(1);
        }
    }
}

/* Physical flat offset of logical element (group g, inner j) in tensor t.
 * The logical multi-index is built by decomposing g over the leading
 * (rank - D) dims and j over the last D dims, then mapped through
 * calcElementIndexByIndices (which applies orderOfDimensions). */
static size_t layerNormPhysOffset(tensor_t *t, size_t numNormDims, size_t g, size_t j) {
    size_t rank = t->shape->numberOfDimensions;
    size_t idx[rank];

    /* Decompose j over the last D logical dims (row-major within the group). */
    size_t rem = j;
    for (size_t d = rank; d-- > rank - numNormDims;) {
        size_t dimSize = getDimensionsByIndex(t, d);
        idx[d] = rem % dimSize;
        rem /= dimSize;
    }
    /* Decompose g over the leading (rank - D) logical dims. */
    rem = g;
    for (size_t d = rank - numNormDims; d-- > 0;) {
        size_t dimSize = getDimensionsByIndex(t, d);
        idx[d] = rem % dimSize;
        rem /= dimSize;
    }
    return calcElementIndexByIndices(rank, t->shape->dimensions, idx, t->shape->orderOfDimensions);
}

/* All-groups stats via the Reduce arithmetic module: fills mean[G] and
 * invSigma[G] (caller stack scratch) for the WHOLE tensor with one pass each of
 * meanOverTrailingAxes* + varianceBiasedOverTrailingAxes* (k = numNormDims
 * collapses the trailing D norm dims into the G leading blocks; Reduce's block
 * order == layerNormPhysOffset's group order, so mean[g]/invSigma[g] index the
 * same group g the layer iterates), then invSigma[g] = 1/sqrt(var[g]+eps).
 * Variance is BIASED (÷N, NOT N-1); eps is INSIDE the sqrt.
 *
 * ONE helper for BOTH FLOAT32 and SYM_INT32, forward AND backward, so a layer's
 * passes can never desync on the stats definition. The path is chosen by input
 * dtype: SYM_INT32 dequants+centers in float INSIDE Reduce (guarded
 * qMaxBits <= 16) — same math as the old inline mantissa loops (d = q*scale -
 * mean, acc += d*d, ÷N).
 *
 * Memory: this deliberately trades the old per-group recompute-over-store (no
 * scratch at all, not even G floats) for 3*G floats of stack at peak (the
 * caller's mean[G] + invSigma[G], plus var[G] local to this helper), computed
 * ONCE and SHARED across both passes — not 2*G per pass. The approved tradeoff
 * (spec §6) keeps the statistics math in the arithmetic module, not the layer.
 * Callers MUST guarantee G > 0 (the stats
 * VLAs and Reduce's block loop are undefined at G == 0); every caller early-outs
 * on G == 0 || N == 0 before calling here. */
static void layerNormAllGroupStats(tensor_t *t, size_t numNormDims, size_t G, float eps,
                                   float *mean, float *invSigma) {
    size_t statsDims[1] = {G};
    size_t statsOrder[1] = {0};
    shape_t statsShape;
    setShape(&statsShape, statsDims, 1, statsOrder);
    quantization_t statsQ;
    initFloat32Quantization(&statsQ);

    float var[G];
    tensor_t meanT;
    setTensorValues(&meanT, (uint8_t *)mean, &statsShape, &statsQ, NULL);
    tensor_t varT;
    setTensorValues(&varT, (uint8_t *)var, &statsShape, &statsQ, NULL);

    if (t->quantization->type == SYM_INT32) {
        meanOverTrailingAxesSymInt32(t, numNormDims, &meanT);
        varianceBiasedOverTrailingAxesSymInt32(t, numNormDims, &meanT, &varT);
    } else {
        meanOverTrailingAxesFloat32(t, numNormDims, &meanT);
        varianceBiasedOverTrailingAxesFloat32(t, numNormDims, &meanT, &varT);
    }
    for (size_t g = 0; g < G; g++) {
        invSigma[g] = rsqrtFloat32(var[g], eps); /* eps INSIDE sqrt */
    }
}

/* The SYM_INT32 path reinterprets tensor data as int32 mantissas; a FLOAT32
 * buffer read that way is silent garbage, so fail fast. The int12 bound
 * (ODT_SYM_OPERAND_QMAXBITS) is required by the affine product q*gamma_q
 * (out[off]*gammaQ[j]); the mantissa-SUM behind the stats is a value-sum and is
 * sound at any qMaxBits <= 16 (it now lives in the Reduce module, which enforces
 * that looser <= 16 bound itself). (#227) */
static void layerNormValidateSymTensor(tensor_t *t, const char *what) {
    if (t->quantization->type != SYM_INT32) {
        PRINT_ERROR("LayerNorm SYM_INT32: %s must be SYM_INT32", what);
        exit(1);
    }
    symInt32QConfig_t *qc = t->quantization->qConfig;
    if (qc->qMaxBits > ODT_SYM_OPERAND_QMAXBITS) {
        PRINT_ERROR("LayerNorm SYM_INT32: %s qMaxBits (%u) exceeds operand contract (%u)", what,
                    (unsigned)qc->qMaxBits, (unsigned)ODT_SYM_OPERAND_QMAXBITS);
        exit(1);
    }
}

/* Affine y = gamma*n + beta as a SEPARATE quantized elementwise stage, applied
 * in-place over the freshly written normalized mantissas (spec: it destroys
 * the abs-max=qMax / var~1 invariants and has its own requantization + output
 * scale). Lives in LayerNorm.c, NOT arithmetic/: it needs broadcast over
 * groups (gamma_j shared by all G groups) and the layout-agnostic logical
 * index map — the flat equal-count arithmetic elementwise ops can express
 * neither. Scale bookkeeping:
 *   s_y    = s_norm * s_gamma                  (product idiom, mulSymInt32Tensors)
 *   seed_j = round(beta_q,j * s_beta / s_y)    (bias-rescale idiom,
 *            matmulSymInt32TensorsWithBias — a raw beta_q add would silently
 *            drop beta under dynamic per-tensor scales)
 *   y_q    = q * gamma_q,j + seed_j            (the product q*gamma_q <= qMax^2
 *            fits int32, but the rescaled seed is DATA-DEPENDENT and unbounded:
 *            |seed| ~ |beta| * qMax^2 / (absmax_n * absmax_gamma). The shared
 *            rescaleIntoAccumulatorScale helper (#189) fails fast (under
 *            -DODT_SEED_GUARD) outside the safe envelope instead of casting an
 *            out-of-range float to int32 (UB). Spec errata: "int32 is safe
 *            throughout" holds only while the rescaled seed leaves one worst-case
 *            int12xint12 product (2047*2047 ≈ 4.2e6, #227) of int32 headroom for the
 *            gamma-product, i.e. |beta| <~ absmax_n * absmax_gamma.)
 * gamma/beta are contiguous default-order rank-D tensors -> flat index j. This
 * writes a RAW, unrestored y_q (accumulator-range, same class as Linear/Conv's
 * matmul output, spec D2/D3) into rawOut's own scale field — the executeOp
 * OUT_WRITE epilogue (caller, layerNormForward) restores width at the
 * producer via the SYM->SYM diagonal requant. */
static void layerNormAffineSymInt32(size_t numNormDims, tensor_t *gamma, tensor_t *beta,
                                    tensor_t *output, float sNorm) {
    int32_t *out = (int32_t *)output->data;
    int32_t *gammaQ = (int32_t *)gamma->data;
    int32_t *betaQ = (int32_t *)beta->data;
    symInt32QConfig_t *outQC = output->quantization->qConfig;
    symInt32QConfig_t *gammaQC = gamma->quantization->qConfig;
    symInt32QConfig_t *betaQC = beta->quantization->qConfig;

    float sY = sNorm * gammaQC->scale;

    size_t G, N;
    layerNormGroupSizes(output, numNormDims, &G, &N);

    /* Seed-rescale + flag-gated int32-overflow guard live in the shared #189 helper
     * (rescaleIntoAccumulatorScale): beta is refolded from its own scale into the
     * output (gamma-product) scale sY per element. */
    for (size_t g = 0; g < G; g++) {
        for (size_t j = 0; j < N; j++) {
            size_t off = layerNormPhysOffset(output, numNormDims, g, j);
            int32_t seed =
                rescaleIntoAccumulatorScale(betaQ[j], betaQC->scale, sY, outQC->roundingMode);
            out[off] = out[off] * gammaQ[j] + seed;
        }
    }
    outQC->scale = sY;
}

/* SYM_INT32 forward (spec 2026-06-05, verified scale-folding scheme):
 * pass 1: per-group float stats + GLOBAL absmax of the normalized values.
 *         Multi-group REQUIRES the per-group 1/sigma_g to hit the DATA — one
 *         per-tensor scale cannot encode G different sigmas; only the global
 *         stretch lives in the scale.
 * pass 2: normalize the stored per-group stats, stretch by K = qMax/absmax,
 *         round-clamp. Stats come ONCE from the Reduce module into G-float stack
 *         scratch (layerNormAllGroupStats) and both passes read them — the
 *         approved tradeoff (spec §6) of 3*G floats of stack at peak (mean[G] +
 *         invSigma[G] + the helper-local var[G]), computed once and shared
 *         across both passes, for keeping the stats math in the arithmetic
 *         module (was: per-group recompute-over-store, no scratch at all).
 *         Stats are computed only when
 *         !allConstant (the constant/zero-absmax branch never reads them).
 * Output scale s_norm = 1/K, then the affine stage folds in gamma/beta and
 * writes the (raw, unrestored) producer scale. gamma/beta are passed
 * explicitly (funnel operands, PR1b.2) rather than read via cfg, mirroring
 * Linear's weights/bias adapter pattern — cfg is retained only for
 * eps/numNormDims/qMaxBits-independent geometry. */
static void layerNormForwardSymInt32(layerNormConfig_t *cfg, tensor_t *gamma, tensor_t *beta,
                                     tensor_t *input, tensor_t *output) {
    layerNormValidateSymTensor(input, "input");
    layerNormValidateSymTensor(output, "output");
    layerNormValidateSymTensor(gamma, "gamma");
    layerNormValidateSymTensor(beta, "beta");

    symInt32QConfig_t *inQC = input->quantization->qConfig;
    symInt32QConfig_t *outQC = output->quantization->qConfig;
    int32_t *in = (int32_t *)input->data;
    int32_t *out = (int32_t *)output->data;
    float inScale = inQC->scale;
    const float qMax = powf(2, (float)outQC->qMaxBits - 1) - 1;
    const float qMin = -powf(2, (float)outQC->qMaxBits - 1);

    size_t G, N;
    layerNormGroupSizes(input, cfg->numNormDims, &G, &N);

    if (G == 0 || N == 0) {
        outQC->scale = 1.0f; /* nothing to normalize; neutral scale (cf. #160) */
        return;
    }

    /* Integer range pre-check: if every mantissa is identical (global int32 min
     * == max), every group's centered value is exactly zero in integer space.
     * A float absMax==0.0f check alone is fragile here: gcc's default
     * -ffp-contract=fast may fuse (float)in[off]*inScale - mean into an fma,
     * so the product is not rounded before the subtraction and the cancellation
     * x*s - mean is NOT guaranteed to be exactly 0 even when all mantissas are
     * equal. The integer check is portable and catches this common case. */
    int32_t intMin = in[layerNormPhysOffset(input, cfg->numNormDims, 0, 0)];
    int32_t intMax = intMin;
    bool allConstant = true;
    for (size_t g = 0; g < G; g++) {
        for (size_t j = 0; j < N; j++) {
            int32_t v = in[layerNormPhysOffset(input, cfg->numNormDims, g, j)];
            if (v < intMin) {
                intMin = v;
                allConstant = false;
            }
            if (v > intMax) {
                intMax = v;
                allConstant = false;
            }
        }
    }

    /* Pass 1: global absmax of n = (x_q*s_x - mu)/sigma over all groups.
     * Skipped when allConstant — absMax stays 0.0f, caught by the unified guard
     * below; stats are only materialized on this branch (mean/invSigma are dead
     * in the constant/zero-absmax branch). */
    float mean[G];
    float invSigma[G];
    float absMax = 0.0f;
    if (!allConstant) {
        layerNormAllGroupStats(input, cfg->numNormDims, G, cfg->eps, mean, invSigma);
        for (size_t g = 0; g < G; g++) {
            for (size_t j = 0; j < N; j++) {
                size_t off = layerNormPhysOffset(input, cfg->numNormDims, g, j);
                float a = fabsf(((float)in[off] * inScale - mean[g]) * invSigma[g]);
                if (a > absMax) {
                    absMax = a;
                }
            }
        }
    }

    float sNorm;
    if (allConstant || absMax == 0.0f) {
        /* Constant input (integer fast-path) OR exact float cancellation
         * (groups internally constant, products exactly representable):
         * emit all-zero mantissas with scale 1.0 — mirrors the absMax==0
         * idiom in convertFloatTensorToSymInt32Tensor. */
        sNorm = 1.0f;
        for (size_t g = 0; g < G; g++) {
            for (size_t j = 0; j < N; j++) {
                out[layerNormPhysOffset(output, cfg->numNormDims, g, j)] = 0;
            }
        }
    } else {
        float K = qMax / absMax;
        sNorm = 1.0f / K;

        /* Pass 2: normalize the stored stats, quantize. */
        for (size_t g = 0; g < G; g++) {
            for (size_t j = 0; j < N; j++) {
                size_t inOff = layerNormPhysOffset(input, cfg->numNormDims, g, j);
                float n = ((float)in[inOff] * inScale - mean[g]) * invSigma[g];
                size_t outOff = layerNormPhysOffset(output, cfg->numNormDims, g, j);
                out[outOff] = roundByMode(clamp(n * K, qMin, qMax), outQC->roundingMode);
            }
        }
    }

    layerNormAffineSymInt32(cfg->numNormDims, gamma, beta, output, sNorm);
}

static void layerNormForwardFloat(layerNormConfig_t *cfg, tensor_t *gamma, tensor_t *beta,
                                  tensor_t *input, tensor_t *output) {
    float *in = (float *)input->data;
    float *out = (float *)output->data;
    float *g = (float *)gamma->data;
    float *b = (float *)beta->data;

    size_t G, N;
    layerNormGroupSizes(input, cfg->numNormDims, &G, &N);
    if (G == 0 || N == 0) {
        return; /* nothing to normalize (empty group geometry); cf. #160 */
    }

    float mean[G];
    float invSigma[G];
    layerNormAllGroupStats(input, cfg->numNormDims, G, cfg->eps, mean, invSigma);

    for (size_t grp = 0; grp < G; grp++) {
        for (size_t j = 0; j < N; j++) {
            size_t off = layerNormPhysOffset(input, cfg->numNormDims, grp, j);
            float nval = mulFloat32s(subFloat32s(in[off], mean[grp]), invSigma[grp]);
            size_t outOff = layerNormPhysOffset(output, cfg->numNormDims, grp, j);
            out[outOff] = addFloat32s(mulFloat32s(g[j], nval), b[j]);
        }
    }
}

/* executeOp forward kernel adapters — operands {input, gamma, beta}; ctx =
 * cfg (eps/normalizedShape/numNormDims geometry, not a tensor so it cannot
 * travel through the funnel's operand array). The SYM kernel emits a RAW,
 * unrestored producer scale (Finding A/D2/D3): the OUT_WRITE epilogue
 * (layerNormForward) restores width via the SYM->SYM diagonal requant, same
 * as Linear/Conv1d's matmul-family forwards. */
static void layerNormForwardKernelFloat(tensor_t **ops, size_t n, tensor_t *rawOut,
                                        tensor_t *auxOut, const void *ctx) {
    (void)n;
    (void)auxOut;
    layerNormForwardFloat((layerNormConfig_t *)ctx, ops[1], ops[2], ops[0], rawOut);
}
static void layerNormForwardKernelSym(tensor_t **ops, size_t n, tensor_t *rawOut, tensor_t *auxOut,
                                      const void *ctx) {
    (void)n;
    (void)auxOut;
    layerNormForwardSymInt32((layerNormConfig_t *)ctx, ops[1], ops[2], ops[0], rawOut);
}

void layerNormForward(layer_t *layer, tensor_t *input, tensor_t *output) {
    layerNormConfig_t *cfg = layer->config->layerNorm;
    layerNormValidateInputShape(cfg, input);

    executeOp(
        &(opSpec_t){
            .kernel = cfg->forwardMath.type == ARITH_SYM_INT32 ? layerNormForwardKernelSym
                                                               : layerNormForwardKernelFloat,
            .ctx = cfg,
            .inputs = (tensor_t *[]){input, getParamFromParameter(cfg->gamma),
                                     getParamFromParameter(cfg->beta)},
            .nInputs = 3,
            .arithmetic = cfg->forwardMath,
            .mode = OUT_WRITE,
        },
        output);
}

static void layerNormBackwardFloat(layerNormConfig_t *cfg, tensor_t *forwardInput, tensor_t *loss,
                                   tensor_t *propLoss) {
    float *x = (float *)forwardInput->data;
    float *dy = (float *)loss->data;
    float *dx = (float *)propLoss->data;
    float *gamma = (float *)cfg->gamma->param->data;
    float *dgamma = (float *)cfg->gamma->grad->data; /* accumulated += */
    float *dbeta = (float *)cfg->beta->grad->data;   /* accumulated += */

    size_t G, N;
    layerNormGroupSizes(forwardInput, cfg->numNormDims, &G, &N);
    if (G == 0 || N == 0) {
        return; /* empty group geometry: no grad increments, nothing to scatter */
    }

    /* Stats computed ONCE up front through the shared Reduce helper (was:
     * per-group recompute-over-store); mean[g]/invSigma[g] feed both the grad
     * pass and the dx scatter, so they cannot desync from the forward. */
    float mean[G];
    float invSigma[G];
    layerNormAllGroupStats(forwardInput, cfg->numNormDims, G, cfg->eps, mean, invSigma);

    for (size_t g = 0; g < G; g++) {
        /* Pass over the group: build n, accumulate dgamma/dbeta, and the two
         * reductions meanDn, meanDnN. */
        float meanDn = 0.0f;
        float meanDnN = 0.0f;
        for (size_t j = 0; j < N; j++) {
            size_t xoff = layerNormPhysOffset(forwardInput, cfg->numNormDims, g, j);
            size_t dyoff = layerNormPhysOffset(loss, cfg->numNormDims, g, j);
            float nval = (x[xoff] - mean[g]) * invSigma[g];
            float dyv = dy[dyoff];
            dbeta[j] += dyv;         /* SUM over groups */
            dgamma[j] += dyv * nval; /* SUM over groups */
            float dn = dyv * gamma[j];
            meanDn += dn;
            meanDnN += dn * nval;
        }
        meanDn /= (float)N;
        meanDnN /= (float)N;

        /* dx scattered back to the same physical offset its x came from. */
        for (size_t j = 0; j < N; j++) {
            size_t xoff = layerNormPhysOffset(forwardInput, cfg->numNormDims, g, j);
            size_t dyoff = layerNormPhysOffset(loss, cfg->numNormDims, g, j);
            float nval = (x[xoff] - mean[g]) * invSigma[g];
            float dn = dy[dyoff] * gamma[j];
            float dxv = invSigma[g] * (dn - meanDn - nval * meanDnN);
            size_t dxoff = layerNormPhysOffset(propLoss, cfg->numNormDims, g, j);
            dx[dxoff] = dxv;
        }
    }
}

/* SYM_INT32 backward (spec 2026-06-05, verified scheme). mu/sigma are computed
 * ONCE from forwardInput through layerNormAllGroupStats — the SAME shared Reduce
 * helper the forward uses, so backward can never desync from the forward
 * definition — into G-float stack scratch that both passes read (was: per-group
 * recompute-over-store; approved 2*G-float tradeoff, spec §6). dy and gamma are
 * dequantized per element via their own scales (float math; dy/gamma mantissas
 * are never integer-summed — only forwardInput is subject to the int32
 * mantissa-sum bound).
 * pass A: grad increments (SUM over groups) + global |dx| absmax.
 * pass B: recompute dx from the same stored stats and quantize into propLoss via
 *         the convertFloatTensorToSymInt32Tensor idiom (scale = absmax/qMax,
 *         round-clamp; absmax==0 -> zeros, scale 1.0). The propLoss scale is
 *         data-dependent and REFRESHED ON EVERY CALL. */
static void layerNormBackwardSymInt32(layerNormConfig_t *cfg, tensor_t *forwardInput,
                                      tensor_t *loss, tensor_t *propLoss) {
    layerNormValidateSymTensor(forwardInput, "forwardInput");
    layerNormValidateSymTensor(loss, "loss");
    layerNormValidateSymTensor(propLoss, "propLoss");
    layerNormValidateSymTensor(cfg->gamma->param, "gamma");
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
    const float qMax = powf(2, (float)plQC->qMaxBits - 1) - 1;
    const float qMin = -powf(2, (float)plQC->qMaxBits - 1);

    size_t G, N;
    layerNormGroupSizes(forwardInput, cfg->numNormDims, &G, &N);
    if (G == 0 || N == 0) {
        plQC->scale = 1.0f; /* nothing to do; neutral scale (cf. #160, forward) */
        return;
    }

    float dgammaInc[N];
    float dbetaInc[N];
    for (size_t j = 0; j < N; j++) {
        dgammaInc[j] = 0.0f;
        dbetaInc[j] = 0.0f;
    }

    /* Stats once from forwardInput via the shared Reduce helper; mean[g]/
     * invSigma[g] drive both pass A and pass B (was: per-group recompute). */
    float mean[G];
    float invSigma[G];
    layerNormAllGroupStats(forwardInput, cfg->numNormDims, G, cfg->eps, mean, invSigma);

    float absMax = 0.0f;
    /* Pass A: grad increments (SUM over groups) + global |dx| absmax. */
    for (size_t g = 0; g < G; g++) {
        float meanDn = 0.0f;
        float meanDnN = 0.0f;
        for (size_t j = 0; j < N; j++) {
            size_t xoff = layerNormPhysOffset(forwardInput, cfg->numNormDims, g, j);
            size_t dyoff = layerNormPhysOffset(loss, cfg->numNormDims, g, j);
            float nval = ((float)xq[xoff] * inScale - mean[g]) * invSigma[g];
            float dyv = (float)dyq[dyoff] * dyScale;
            dbetaInc[j] += dyv;         /* SUM over groups */
            dgammaInc[j] += dyv * nval; /* SUM over groups */
            float dn = dyv * ((float)gammaQ[j] * gammaScale);
            meanDn += dn;
            meanDnN += dn * nval;
        }
        meanDn /= (float)N;
        meanDnN /= (float)N;
        for (size_t j = 0; j < N; j++) {
            size_t xoff = layerNormPhysOffset(forwardInput, cfg->numNormDims, g, j);
            size_t dyoff = layerNormPhysOffset(loss, cfg->numNormDims, g, j);
            float nval = ((float)xq[xoff] * inScale - mean[g]) * invSigma[g];
            float dn = ((float)dyq[dyoff] * dyScale) * ((float)gammaQ[j] * gammaScale);
            float a = fabsf(invSigma[g] * (dn - meanDn - nval * meanDnN));
            if (a > absMax) {
                absMax = a;
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
    executeOpValidateAccMode(cfg->weightGradAccMode, "LayerNorm weightGradAccMode");
    executeOp(
        &(opSpec_t){
            .kernel = executeOpIdentityKernel,
            .inputs = (tensor_t *[]){&dgammaT},
            .nInputs = 1,
            .arithmetic = (arithmetic_t){.type = ARITH_FLOAT32, .roundingMode = HALF_AWAY},
            .mode = cfg->weightGradAccMode,
        },
        cfg->gamma->grad);
    executeOpValidateAccMode(cfg->biasGradAccMode, "LayerNorm biasGradAccMode");
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
     * products and sums even under -ffp-contract=fast, so the float check is
     * reliable here. */
    if (absMax == 0.0f) {
        for (size_t g = 0; g < G; g++) {
            for (size_t j = 0; j < N; j++) {
                dxq[layerNormPhysOffset(propLoss, cfg->numNormDims, g, j)] = 0;
            }
        }
        plQC->scale = 1.0f;
        return;
    }

    float dxScale = absMax / qMax;
    /* Pass B: recompute dx from the stored stats and quantize. The dx expression
     * is textually IDENTICAL to pass A's absmax expression so gcc's
     * -ffp-contract=fast contracts both the same way; the clamp absorbs any
     * residual divergence. The propLoss scale is data-dependent and REFRESHED ON
     * EVERY CALL — a stale scale silently corrupts the downstream layer. */
    for (size_t g = 0; g < G; g++) {
        float meanDn = 0.0f;
        float meanDnN = 0.0f;
        for (size_t j = 0; j < N; j++) {
            size_t xoff = layerNormPhysOffset(forwardInput, cfg->numNormDims, g, j);
            size_t dyoff = layerNormPhysOffset(loss, cfg->numNormDims, g, j);
            float nval = ((float)xq[xoff] * inScale - mean[g]) * invSigma[g];
            float dyv = (float)dyq[dyoff] * dyScale;
            float dn = dyv * ((float)gammaQ[j] * gammaScale);
            meanDn += dn;
            meanDnN += dn * nval;
        }
        meanDn /= (float)N;
        meanDnN /= (float)N;
        for (size_t j = 0; j < N; j++) {
            size_t xoff = layerNormPhysOffset(forwardInput, cfg->numNormDims, g, j);
            size_t dyoff = layerNormPhysOffset(loss, cfg->numNormDims, g, j);
            float nval = ((float)xq[xoff] * inScale - mean[g]) * invSigma[g];
            float dn = ((float)dyq[dyoff] * dyScale) * ((float)gammaQ[j] * gammaScale);
            float dxv = invSigma[g] * (dn - meanDn - nval * meanDnN);
            size_t dxoff = layerNormPhysOffset(propLoss, cfg->numNormDims, g, j);
            dxq[dxoff] = roundByMode(clamp(dxv / dxScale, qMin, qMax), plQC->roundingMode);
        }
    }
    plQC->scale = dxScale;
}

void layerNormBackward(layer_t *layer, tensor_t *forwardInput, tensor_t *loss, tensor_t *propLoss) {
    layerNormConfig_t *cfg = layer->config->layerNorm;
    layerNormValidateInputShape(cfg, forwardInput);
    switch (cfg->propLossMath.type) {
    case ARITH_FLOAT32:
        /* SYM_INT32 forwardMath + FLOAT32 backwardMath is an inference-only
         * profile: reading a SYM_INT32 forwardInput / loss / gamma as float*
         * here would be silent garbage — fail fast. Training a SYM forward
         * requires backwardMath = SYM_INT32 (factory rule, PR-3). */
        if (forwardInput->quantization->type != FLOAT32 || loss->quantization->type != FLOAT32 ||
            cfg->gamma->param->quantization->type != FLOAT32) {
            PRINT_ERROR("LayerNorm backward: FLOAT32 backward requires FLOAT32 tensors "
                        "(SYM_INT32 forwardMath + FLOAT32 backwardMath is inference-only; "
                        "use SYM_INT32 backwardMath to train)");
            exit(1);
        }
        /* layerNormBackwardFloat raw-casts cfg->gamma->grad->data /
         * cfg->beta->grad->data to float* (it bypasses the executeOp funnel,
         * unlike the SYM_INT32 path's dgamma/dbeta identity-kernel executeOp
         * calls above). A packed (SYM/ASYM) grad tensor read/written that way
         * is silent memory corruption, not garbage values — fail fast instead.
         * PR3 (#261): routing float dgamma/dbeta through the funnel like the
         * SYM_INT32 path is a follow-up issue; this guard only closes the gap
         * until then. */
        if (cfg->gamma->grad->quantization->type != FLOAT32 ||
            cfg->beta->grad->quantization->type != FLOAT32) {
            PRINT_ERROR("LayerNorm backward: FLOAT32 backward writes gamma/beta grads via a raw "
                        "float* cast — packed grad storage requires the funnel route (follow-up "
                        "issue, #261) — got gamma grad dtype %d, beta grad dtype %d",
                        (int)cfg->gamma->grad->quantization->type,
                        (int)cfg->beta->grad->quantization->type);
            exit(1);
        }
        /* layerNormBackwardFloat also writes propLoss->data (dx) via a raw
         * float* cast. A SYM-storage propLossQ (SYM_INT32 fixed-point, or packed
         * sub-byte SYM) paired with FLOAT32 propLossMath is factory-constructible,
         * and that raw write silently corrupts the mantissa/packed buffer — fail
         * fast instead (same #261 gap the gamma/beta grad guard closes). */
        if (propLoss->quantization->type != FLOAT32) {
            PRINT_ERROR("LayerNorm backward: FLOAT32 backward writes propLoss (dx) via a raw "
                        "float* cast — SYM/packed propLoss storage requires the funnel route "
                        "(follow-up issue, #261) — got propLoss dtype %d",
                        (int)propLoss->quantization->type);
            exit(1);
        }
        layerNormBackwardFloat(cfg, forwardInput, loss, propLoss);
        break;
    case ARITH_SYM_INT32:
        layerNormBackwardSymInt32(cfg, forwardInput, loss, propLoss);
        break;
    default:
        PRINT_ERROR(
            "LayerNorm backward: quantization type not implemented (FLOAT32/SYM_INT32 only)");
        exit(1);
    }
}

void layerNormCalcOutputShape(layer_t *layer, shape_t *inputShape, shape_t *outputShape) {
    (void)layer;
    memcpy(outputShape->dimensions, inputShape->dimensions,
           inputShape->numberOfDimensions * sizeof(size_t));
    memcpy(outputShape->orderOfDimensions, inputShape->orderOfDimensions,
           inputShape->numberOfDimensions * sizeof(size_t));
    outputShape->numberOfDimensions = inputShape->numberOfDimensions;
}
