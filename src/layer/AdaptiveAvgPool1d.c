#define SOURCE_FILE "ODT_ADAPTIVE_AVG_POOL_1D"

#include <math.h>
#include <string.h>

#include "AdaptiveAvgPool1d.h"

#include "AdaptiveWindow1d.h"
#include "ArithmeticType.h"
#include "BfpKernelSupport.h"
#include "Common.h"
#include "ExecuteOp.h"
#include "Layer.h"
#include "Quantization.h"
#include "Tensor.h"

void initAdaptiveAvgPool1dConfig(adaptiveAvgPool1dConfig_t *cfg, size_t outputSize,
                                 quantization_t *forwardQ, quantization_t *propLossQ) {
    if (outputSize == 0) {
        PRINT_ERROR("AdaptiveAvgPool1d: outputSize must be >= 1");
        exit(1);
    }
    cfg->outputSize = outputSize;
    cfg->forwardMath = arithmeticFromQuantizationOrDefault(forwardQ);
    cfg->propLossMath = arithmeticFromQuantizationOrDefault(propLossQ);
    cfg->outputQ = forwardQ;
    cfg->propLossQ = propLossQ;
}

/* executeOp forward kernel adapter — ctx = adaptiveAvgPool1dConfig_t* (outputSize
 * geometry, mirrors AvgPool1d's ctx convention, AvgPool1d.c); 1 input, no
 * auxOut. The SYM_INT32 arm lives in adaptiveAvgPool1dForwardKernelSymInt32
 * below (#205). */
static void adaptiveAvgPool1dForwardKernel(tensor_t **ops, size_t n, tensor_t *rawOut,
                                           tensor_t *auxOut, const void *ctx) {
    (void)n;
    (void)auxOut;
    const adaptiveAvgPool1dConfig_t *cfg = ctx;
    tensor_t *input = ops[0];

    size_t batch = input->shape->dimensions[0];
    size_t channels = input->shape->dimensions[1];
    size_t inputLength = input->shape->dimensions[2];
    size_t outputLength = cfg->outputSize;

    if (rawOut->shape->dimensions[2] != outputLength) {
        PRINT_ERROR("AdaptiveAvgPool1d forward: output length (%zu) does not match "
                    "configured outputSize (%zu)",
                    rawOut->shape->dimensions[2], outputLength);
        exit(1);
    }

    float const *xArr = (float const *)input->data;
    float *yArr = (float *)rawOut->data;

    for (size_t b = 0; b < batch; b++) {
        for (size_t c = 0; c < channels; c++) {
            for (size_t outPos = 0; outPos < outputLength; outPos++) {
                adaptiveWindow1d_t w = adaptiveWindow1dAt(inputLength, outputLength, outPos);

                float sum = 0.0f;
                for (size_t i = 0; i < w.count; i++) {
                    sum += xArr[(b * channels + c) * inputLength + w.start + i];
                }

                size_t outIdx = (b * channels + c) * outputLength + outPos;
                yArr[outIdx] = sum / (float)w.count;
            }
        }
    }
}

/* Pool SYM value-sum guard (Reduce.c precedent: reduceValidateSymOperand +
 * meanOverTrailingAxesSymInt32's N-bound): the window sum / backward scatter
 * accumulates a VALUE-sum of operand-width mantissas in an int32 — sound only
 * for qMaxBits in [1,16] and worst-case terms < 2^(32-qMaxBits). qMaxBits == 0
 * is degenerate and would make the shift UB. No dtype check: the executeOp
 * prologue guarantees a SYM_INT32 operand inside an ARITH_SYM_INT32 kernel. */
#define POOL_SYM_VALUESUM_QMAXBITS 16u
static void poolValidateSymValueSum(const tensor_t *operand, size_t maxTerms, const char *op) {
    const symInt32QConfig_t *qc = operand->quantization->qConfig;
    if (qc->qMaxBits == 0 || qc->qMaxBits > POOL_SYM_VALUESUM_QMAXBITS) {
        PRINT_ERROR("%s: operand qMaxBits (%u) outside the value-sum bound [1,%u]", op,
                    (unsigned)qc->qMaxBits, (unsigned)POOL_SYM_VALUESUM_QMAXBITS);
        exit(1);
    }
    size_t bound = (size_t)1 << (32u - qc->qMaxBits);
    if (maxTerms >= bound) {
        PRINT_ERROR("%s: worst-case summed terms (%zu) reach the value-sum bound for "
                    "qMaxBits (%u) -- must be < 2^(32-qMaxBits) (%zu)",
                    op, maxTerms, (unsigned)qc->qMaxBits, bound);
        exit(1);
    }
}

/* Rounded integer division, half away from zero — roundByMode(HALF_AWAY)
 * semantics without leaving the integer domain: (|s| + k/2) / k truncates to
 * the correctly rounded magnitude (exact for odd k where no .5 tie exists,
 * tie-away for even k). Window sums are bounded by count * qMax, far from
 * int32 overflow for realistic lengths. */
static inline int32_t roundedDivHalfAwayInt32(int32_t sum, int32_t count) {
    int32_t mag = sum >= 0 ? sum : -sum;
    int32_t q = (mag + count / 2) / count;
    return sum >= 0 ? q : -q;
}

/* SYM_INT32 arm (#205): per-window element count varies, so the division
 * cannot fold into ONE scale (AvgPool1d's trick) — decided mechanics: rounded
 * integer division of the mantissa sum (half-away, roundByMode-consistent),
 * scale unchanged. At most 0.5 LSB rounding error per element — standard
 * fixed-point practice, not homegrown numerics. The quotient stays in operand
 * range (|mean| <= |max|); the OUT_WRITE epilogue still renormalizes to the
 * target's own absmax like every funnel SYM producer. */
static void adaptiveAvgPool1dForwardKernelSymInt32(tensor_t **ops, size_t n, tensor_t *rawOut,
                                                   tensor_t *auxOut, const void *ctx) {
    (void)n;
    (void)auxOut;
    const adaptiveAvgPool1dConfig_t *cfg = ctx;
    tensor_t *input = ops[0];

    size_t batch = input->shape->dimensions[0];
    size_t channels = input->shape->dimensions[1];
    size_t inputLength = input->shape->dimensions[2];
    size_t outputLength = cfg->outputSize;

    if (rawOut->shape->dimensions[2] != outputLength) {
        PRINT_ERROR("AdaptiveAvgPool1d forward: output length (%zu) does not match "
                    "configured outputSize (%zu)",
                    rawOut->shape->dimensions[2], outputLength);
        exit(1);
    }

    /* Max adaptive window count = ceil((o+1)L/O) - floor(oL/O) <= L/O + 1,
     * conservatively bounded by ceil(L/O) + 1. */
    poolValidateSymValueSum(input, (inputLength + outputLength - 1) / outputLength + 1,
                            "AdaptiveAvgPool1d forward SYM");

    int32_t const *xArr = (int32_t const *)input->data;
    int32_t *yArr = (int32_t *)rawOut->data;

    for (size_t b = 0; b < batch; b++) {
        for (size_t c = 0; c < channels; c++) {
            for (size_t outPos = 0; outPos < outputLength; outPos++) {
                adaptiveWindow1d_t w = adaptiveWindow1dAt(inputLength, outputLength, outPos);

                int32_t sum = 0;
                for (size_t i = 0; i < w.count; i++) {
                    sum += xArr[(b * channels + c) * inputLength + w.start + i];
                }

                yArr[(b * channels + c) * outputLength + outPos] =
                    roundedDivHalfAwayInt32(sum, (int32_t)w.count);
            }
        }
    }

    ((symInt32QConfig_t *)rawOut->quantization->qConfig)->scale =
        ((symInt32QConfig_t *)input->quantization->qConfig)->scale;
}

/* BFP epic PR4 (R-P1): GEMM's rule 1 is WEIGHT-anchored, but a pool has no
 * weight operand, so the width anchor for staging a FLOAT32-stored operand is
 * the layer's OWN produced-wire config — outputQ for the forward op,
 * propLossQ for the backward op. Validated EAGERLY at op entry: without a
 * BFP-typed produced-wire config there is no width source at all, so the arm
 * must not run even when the operand happens to be BFP-stored. The pointer may
 * be NULL — the userApi factories copy layerQuant_t slots BY VALUE and never
 * call initAdaptiveAvgPool1dConfig, so a pinned ARITH_BFP math slot can arrive
 * with a NULL or non-BFP wire config. NULL-check before ->type. (Per-file
 * static, like requireNoBfpWire and poolValidateSymValueSum above — the pool
 * layers duplicate these rather than sharing a header.) */
static const bfpQConfig_t *poolBfpWireAnchor(const quantization_t *wireQ, const char *what) {
    if (wireQ == NULL || wireQ->type != BFP) {
        PRINT_ERROR("%s: ARITH_BFP requires this layer's produced-wire config to be BFP-typed "
                    "(the width anchor for staging FLOAT32 operands -- pools have no weight "
                    "operand; see docs/conventions/arithmetic-bfp.md §5.7); got %s",
                    what, wireQ == NULL ? "NULL" : "a non-BFP config");
        exit(1);
    }
    return wireQ->qConfig;
}

/* BFP epic PR4 (F5): the new BFP kernels index rawOut as a DENSE
 * [batch][channels][length] array whose batch and channels come from the
 * OPERAND's dims, so a rawOut that disagrees on dim 0 or dim 1 is written past
 * its end even when its length is right — checking dimensions[2] alone is not
 * enough. Rank is checked first: dimensions[0..2] on a rank-2 shape is itself
 * an over-read. NOTE the deliberate scope: the FLOAT32/SYM arms of this layer
 * have the SAME gap and are NOT touched here. */
static void poolBfpRequireDims3(const tensor_t *t, size_t d0, size_t d1, size_t d2,
                                const char *what) {
    if (t->shape->numberOfDimensions != 3) {
        PRINT_ERROR("%s: expected a rank-3 [batch, channels, length] tensor, got rank %zu", what,
                    t->shape->numberOfDimensions);
        exit(1);
    }
    if (t->shape->dimensions[0] != d0 || t->shape->dimensions[1] != d1 ||
        t->shape->dimensions[2] != d2) {
        PRINT_ERROR("%s: expected shape [%zu, %zu, %zu], got [%zu, %zu, %zu]", what, d0, d1, d2,
                    t->shape->dimensions[0], t->shape->dimensions[1], t->shape->dimensions[2]);
        exit(1);
    }
}

/* BFP epic PR4 (R-P4): the PR2 fold contract per adaptive window (int32
 * partial per same-group segment, lossless ldexpf fold on group change and at
 * the tail), then a float32 divide by THAT window's own count in the FLOAT32
 * raw (D7). The SYM arm's roundedDivHalfAwayInt32 has no role here: it exists
 * only because a SYM raw is int32; the BFP raw is float, so the exact divide
 * is both simpler AND more accurate. Sum headroom uses the SAME conservative
 * bound as the SYM arm, ceil(L/O) + 1. */
static void adaptiveAvgPool1dForwardKernelBfp(tensor_t **ops, size_t n, tensor_t *rawOut,
                                              tensor_t *auxOut, const void *ctx) {
    (void)n;
    (void)auxOut;
    const adaptiveAvgPool1dConfig_t *cfg = ctx;
    tensor_t *input = ops[0];

    if (input->shape->numberOfDimensions != 3) {
        PRINT_ERROR("AdaptiveAvgPool1d forward BFP: input must be rank-3 [batch, channels, "
                    "length], got rank %zu",
                    input->shape->numberOfDimensions);
        exit(1);
    }
    size_t batch = input->shape->dimensions[0];
    size_t channels = input->shape->dimensions[1];
    size_t inputLength = input->shape->dimensions[2];
    size_t outputLength = cfg->outputSize;

    /* All three dims (F5), not just the length — see poolBfpRequireDims3. */
    poolBfpRequireDims3(rawOut, batch, channels, outputLength,
                        "AdaptiveAvgPool1d forward BFP (rawOut)");

    const bfpQConfig_t *qC = input->quantization->qConfig;
    validateBfpQConfigShape(qC, calcNumberOfElementsByTensor(input));
    bfpValidateSumHeadroom(qC, (inputLength + outputLength - 1) / outputLength + 1,
                           "AdaptiveAvgPool1d forward BFP");
    const int32_t expBias = bfpExponentBias(qC);

    int32_t const *xArr = (int32_t const *)input->data;
    float *yArr = (float *)rawOut->data;

    for (size_t b = 0; b < batch; b++) {
        for (size_t c = 0; c < channels; c++) {
            for (size_t outPos = 0; outPos < outputLength; outPos++) {
                adaptiveWindow1d_t w = adaptiveWindow1dAt(inputLength, outputLength, outPos);

                float acc = 0.f;
                int32_t partial = 0;
                size_t currentGroup = 0;
                for (size_t i = 0; i < w.count; i++) {
                    size_t storageIdx = (b * channels + c) * inputLength + w.start + i;
                    size_t g = bfpGroupOf(qC, storageIdx);
                    if (i == 0) {
                        currentGroup = g;
                    } else if (g != currentGroup) {
                        acc += ldexpf((float)partial, (int)qC->exponents[currentGroup] - expBias);
                        partial = 0;
                        currentGroup = g;
                    }
                    partial += xArr[storageIdx];
                }
                if (w.count > 0) {
                    acc += ldexpf((float)partial, (int)qC->exponents[currentGroup] - expBias);
                }

                yArr[(b * channels + c) * outputLength + outPos] = acc / (float)w.count;
            }
        }
    }
}

void adaptiveAvgPool1dForward(layer_t *layer, tensor_t *input, tensor_t *output) {
    adaptiveAvgPool1dConfig_t *cfg = layer->config->adaptiveAvgPool1d;
    if (cfg->forwardMath.type == ARITH_BFP) {
        const bfpQConfig_t *anchor = poolBfpWireAnchor(cfg->outputQ, "AdaptiveAvgPool1d forward");
        /* Stack template: lifetime covers the executeOp call (same frame).
         * Always per-tensor {1,0} — the anchor supplies WIDTHS only; the
         * funnel owns exponent backing and rounds by the OP (#282). A
         * BFP-stored operand gets NULL: borrowed zero-copy, never re-blocked. */
        bfpQConfig_t stage = {.exponents = NULL,
                              .numGroups = 1,
                              .groupSize = 0,
                              .roundingMode = cfg->forwardMath.roundingMode,
                              .mantissaBits = anchor->mantissaBits,
                              .exponentBits = anchor->exponentBits};
        executeOp(
            &(opSpec_t){
                .kernel = adaptiveAvgPool1dForwardKernelBfp,
                .ctx = cfg,
                .inputs = (tensor_t *[]){input},
                .nInputs = 1,
                .arithmetic = cfg->forwardMath,
                .mode = OUT_WRITE,
                .bfpStage = {input->quantization->type == FLOAT32 ? &stage : NULL, NULL, NULL},
            },
            output);
        return;
    }

    opKernelFn_t kernel;
    switch (cfg->forwardMath.type) {
    case ARITH_FLOAT32:
        kernel = adaptiveAvgPool1dForwardKernel;
        break;
    case ARITH_SYM_INT32:
        kernel = adaptiveAvgPool1dForwardKernelSymInt32;
        break;
    default:
        PRINT_ERROR("AdaptiveAvgPool1d forward: quantization type not implemented");
        exit(1);
    }
    executeOp(
        &(opSpec_t){
            .kernel = kernel,
            .ctx = cfg,
            .inputs = (tensor_t *[]){input},
            .nInputs = 1,
            .arithmetic = cfg->forwardMath,
            .mode = OUT_WRITE,
        },
        output);
}

/* BFP epic PR4 (R-P4): the ARITH_BFP arm below IS the native pooling path, so
 * this guard is NARROWED to the two arms that raw-view ->data in their own
 * storage format — a packed BFP wire read as float* / int32_t* is a 4x heap
 * over-read on lossGrad and an over-write into the packed propLoss buffer.
 * Keyed on the wire's STORAGE dtype, not the declared arithmetic (#315
 * parity). forwardInput is NOT guarded: this layer never dereferences it. */
static void requireNoBfpWire(const tensor_t *t, const char *what) {
    if (t->quantization->type == BFP) {
        PRINT_ERROR("%s: this arm raw-views the wire in its own storage format and cannot read "
                    "packed BFP mantissas -- derive ARITH_BFP from a BFP wire config, or keep "
                    "BFP off this wire",
                    what);
        exit(1);
    }
}

void adaptiveAvgPool1dBackwardFloat(layer_t *layer, tensor_t *forwardInput, tensor_t *lossGrad,
                                    tensor_t *propLoss) {
    adaptiveAvgPool1dConfig_t *cfg = layer->config->adaptiveAvgPool1d;
    (void)forwardInput; // not needed: window geometry is determined by shapes

    size_t batch = lossGrad->shape->dimensions[0];
    size_t channels = lossGrad->shape->dimensions[1];
    size_t outputLength = lossGrad->shape->dimensions[2];
    size_t inputLength = propLoss->shape->dimensions[2];

    if (outputLength != cfg->outputSize) {
        PRINT_ERROR("AdaptiveAvgPool1d backward: lossGrad outputLength (%zu) does not match "
                    "configured outputSize (%zu)",
                    outputLength, cfg->outputSize);
        exit(1);
    }

    float const *gyArr = (float const *)lossGrad->data;
    float *gxArr = (float *)propLoss->data;
    // propLoss arrives calloc-zeroed (StorageApi reserveMemory); overlapping
    // windows accumulate correctly via +=.

    for (size_t b = 0; b < batch; b++) {
        for (size_t c = 0; c < channels; c++) {
            for (size_t outPos = 0; outPos < outputLength; outPos++) {
                adaptiveWindow1d_t w = adaptiveWindow1dAt(inputLength, outputLength, outPos);
                size_t outIdx = (b * channels + c) * outputLength + outPos;
                float contribution = gyArr[outIdx] / (float)w.count;

                for (size_t i = 0; i < w.count; i++) {
                    gxArr[(b * channels + c) * inputLength + w.start + i] += contribution;
                }
            }
        }
    }
}

/* SYM_INT32 dx kernel (#205): per-window rounded quotient of the loss-grad
 * mantissa (same half-away integer division as the forward — the transpose of
 * mean is spread-by-1/count), scattered += into every window member; scale
 * unchanged. Funnel-routed like Conv1d's dx wire; rawOut is the funnel's
 * uninitialized Phase-2 scratch, so the kernel memsets it before the scatter
 * `+=` (adjacent adaptive windows overlap when L % outputSize != 0). */
static void adaptiveAvgPool1dBackwardKernelSymInt32(tensor_t **ops, size_t n, tensor_t *rawOut,
                                                    tensor_t *auxOut, const void *ctx) {
    (void)n;
    (void)auxOut;
    const adaptiveAvgPool1dConfig_t *cfg = ctx;
    tensor_t *lossGrad = ops[0];

    size_t batch = lossGrad->shape->dimensions[0];
    size_t channels = lossGrad->shape->dimensions[1];
    size_t outputLength = lossGrad->shape->dimensions[2];
    size_t inputLength = rawOut->shape->dimensions[2];

    if (outputLength != cfg->outputSize) {
        PRINT_ERROR("AdaptiveAvgPool1d backward: lossGrad outputLength (%zu) does not match "
                    "configured outputSize (%zu)",
                    outputLength, cfg->outputSize);
        exit(1);
    }

    /* Worst-case scatter collisions per input cell = covering windows
     * <= O/L + 1 (window o covers idx iff oL/O < idx+1 and (o+1)L/O > idx),
     * conservatively bounded by ceil(O/L) + 1 — reaches ceil(O/L)+1 only in
     * the upsample regime (O > L). */
    poolValidateSymValueSum(lossGrad, (outputLength + inputLength - 1) / inputLength + 1,
                            "AdaptiveAvgPool1d backward SYM");

    int32_t const *gyArr = (int32_t const *)lossGrad->data;
    int32_t *gxArr = (int32_t *)rawOut->data;

    memset(gxArr, 0, batch * channels * inputLength * sizeof(int32_t));

    for (size_t b = 0; b < batch; b++) {
        for (size_t c = 0; c < channels; c++) {
            for (size_t outPos = 0; outPos < outputLength; outPos++) {
                adaptiveWindow1d_t w = adaptiveWindow1dAt(inputLength, outputLength, outPos);
                int32_t contribution = roundedDivHalfAwayInt32(
                    gyArr[(b * channels + c) * outputLength + outPos], (int32_t)w.count);

                for (size_t i = 0; i < w.count; i++) {
                    gxArr[(b * channels + c) * inputLength + w.start + i] += contribution;
                }
            }
        }
    }

    ((symInt32QConfig_t *)rawOut->quantization->qConfig)->scale =
        ((symInt32QConfig_t *)lossGrad->quantization->qConfig)->scale;
}

/* dx twin: exact dequant of each output cell divided by ITS window's count,
 * accumulated directly in the FLOAT32 raw. No int32 partials across the
 * scattered writes -> no sum-headroom guard (R-P4). rawOut is the funnel's
 * uninitialized Phase-2 scratch, so it is memset before the `+=` (adjacent
 * adaptive windows overlap when L % outputSize != 0). */
static void adaptiveAvgPool1dBackwardKernelBfp(tensor_t **ops, size_t n, tensor_t *rawOut,
                                               tensor_t *auxOut, const void *ctx) {
    (void)n;
    (void)auxOut;
    const adaptiveAvgPool1dConfig_t *cfg = ctx;
    tensor_t *lossGrad = ops[0];

    if (lossGrad->shape->numberOfDimensions != 3 || rawOut->shape->numberOfDimensions != 3) {
        PRINT_ERROR("AdaptiveAvgPool1d backward BFP: lossGrad and rawOut must both be rank-3, "
                    "got ranks %zu and %zu",
                    lossGrad->shape->numberOfDimensions, rawOut->shape->numberOfDimensions);
        exit(1);
    }
    size_t batch = lossGrad->shape->dimensions[0];
    size_t channels = lossGrad->shape->dimensions[1];
    size_t outputLength = lossGrad->shape->dimensions[2];
    size_t inputLength = rawOut->shape->dimensions[2];

    if (outputLength != cfg->outputSize) {
        PRINT_ERROR("AdaptiveAvgPool1d backward: lossGrad outputLength (%zu) does not match "
                    "configured outputSize (%zu)",
                    outputLength, cfg->outputSize);
        exit(1);
    }
    /* All three dims (F5): the memset sizes batch * channels * inputLength
     * floats and the scatter uses batch/channels off the LOSSGRAD. */
    poolBfpRequireDims3(rawOut, batch, channels, inputLength,
                        "AdaptiveAvgPool1d backward BFP (rawOut)");

    const bfpQConfig_t *qC = lossGrad->quantization->qConfig;
    validateBfpQConfigShape(qC, calcNumberOfElementsByTensor(lossGrad));
    const int32_t expBias = bfpExponentBias(qC);

    int32_t const *gyArr = (int32_t const *)lossGrad->data;
    float *gxArr = (float *)rawOut->data;

    memset(gxArr, 0, batch * channels * inputLength * sizeof(float));

    for (size_t b = 0; b < batch; b++) {
        for (size_t c = 0; c < channels; c++) {
            for (size_t outPos = 0; outPos < outputLength; outPos++) {
                adaptiveWindow1d_t w = adaptiveWindow1dAt(inputLength, outputLength, outPos);
                size_t outIdx = (b * channels + c) * outputLength + outPos;
                size_t g = bfpGroupOf(qC, outIdx);
                float contribution =
                    ldexpf((float)gyArr[outIdx], (int)qC->exponents[g] - expBias) / (float)w.count;

                for (size_t i = 0; i < w.count; i++) {
                    gxArr[(b * channels + c) * inputLength + w.start + i] += contribution;
                }
            }
        }
    }
}

void adaptiveAvgPool1dBackward(layer_t *layer, tensor_t *forwardInput, tensor_t *lossGrad,
                               tensor_t *propLoss) {
    adaptiveAvgPool1dConfig_t *cfg = layer->config->adaptiveAvgPool1d;
    switch (cfg->propLossMath.type) {
    case ARITH_FLOAT32:
        /* Runs OUTSIDE executeOp and raw-casts both wires to float*; a packed
         * BFP wire would be read as wide scalars (4x heap over-read on
         * lossGrad, over-write into the packed propLoss buffer). #315 parity. */
        requireNoBfpWire(lossGrad, "AdaptiveAvgPool1d backward (lossGrad)");
        requireNoBfpWire(propLoss, "AdaptiveAvgPool1d backward (propLoss)");
        adaptiveAvgPool1dBackwardFloat(layer, forwardInput, lossGrad, propLoss);
        break;
    case ARITH_SYM_INT32:
        requireNoBfpWire(lossGrad, "AdaptiveAvgPool1d backward (lossGrad)");
        requireNoBfpWire(propLoss, "AdaptiveAvgPool1d backward (propLoss)");
        (void)forwardInput; // not needed: window geometry comes from shapes
        executeOp(
            &(opSpec_t){
                .kernel = adaptiveAvgPool1dBackwardKernelSymInt32,
                .ctx = cfg,
                .inputs = (tensor_t *[]){lossGrad},
                .nInputs = 1,
                .arithmetic = cfg->propLossMath,
                .mode = OUT_WRITE,
            },
            propLoss);
        break;
    case ARITH_BFP: {
        const bfpQConfig_t *anchor =
            poolBfpWireAnchor(cfg->propLossQ, "AdaptiveAvgPool1d backward");
        bfpQConfig_t stage = {.exponents = NULL,
                              .numGroups = 1,
                              .groupSize = 0,
                              .roundingMode = cfg->propLossMath.roundingMode,
                              .mantissaBits = anchor->mantissaBits,
                              .exponentBits = anchor->exponentBits};
        (void)forwardInput; // not needed: window geometry comes from shapes
        executeOp(
            &(opSpec_t){
                .kernel = adaptiveAvgPool1dBackwardKernelBfp,
                .ctx = cfg,
                .inputs = (tensor_t *[]){lossGrad},
                .nInputs = 1,
                .arithmetic = cfg->propLossMath,
                .mode = OUT_WRITE,
                .bfpStage = {lossGrad->quantization->type == FLOAT32 ? &stage : NULL, NULL, NULL},
            },
            propLoss);
        break;
    }
    default:
        PRINT_ERROR("AdaptiveAvgPool1d backward: quantization type not implemented");
        exit(1);
    }
}

void adaptiveAvgPool1dCalcOutputShape(layer_t *layer, shape_t *inputShape, shape_t *outputShape) {
    if (inputShape->numberOfDimensions != 3) {
        PRINT_ERROR("AdaptiveAvgPool1d expects 3D input [batch, channel, length], got %luD",
                    inputShape->numberOfDimensions);
        exit(1);
    }
    adaptiveAvgPool1dConfig_t *cfg = layer->config->adaptiveAvgPool1d;
    outputShape->numberOfDimensions = 3;
    outputShape->dimensions[0] = inputShape->dimensions[0]; // B
    outputShape->dimensions[1] = inputShape->dimensions[1]; // C
    outputShape->dimensions[2] = cfg->outputSize;
    setOrderOfDimsForNewTensor(inputShape->numberOfDimensions, outputShape->orderOfDimensions);
}
