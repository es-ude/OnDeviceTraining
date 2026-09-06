#define SOURCE_FILE "ODT_AVG_POOL_1D"

#include <math.h>
#include <string.h>

#include "AvgPool1d.h"

#include "ArithmeticType.h"
#include "BfpKernelSupport.h"
#include "Common.h"
#include "ExecuteOp.h"
#include "Layer.h"
#include "Quantization.h"
#include "SlidingWindow1d.h"
#include "Tensor.h"

void initAvgPool1dConfig(avgPool1dConfig_t *cfg, kernel_t *kernel, quantization_t *forwardQ,
                         quantization_t *propLossQ) {
    if (kernel->size == 0) {
        PRINT_ERROR("AvgPool1d: kernel size must be >= 1");
        exit(1);
    }
    cfg->kernel = kernel;
    cfg->forwardMath = arithmeticFromQuantizationOrDefault(forwardQ);
    cfg->propLossMath = arithmeticFromQuantizationOrDefault(propLossQ);
    cfg->outputQ = forwardQ;
    cfg->propLossQ = propLossQ;
}

/* executeOp forward kernel adapter — ctx = avgPool1dConfig_t* for kernel_t
 * geometry (mirrors Conv1d's ctx convention, Conv1d.c); 1 input, no auxOut.
 * The SYM_INT32 arm lives in avgPool1dForwardKernelSymInt32 below (#205). */
static void avgPool1dForwardKernel(tensor_t **ops, size_t n, tensor_t *rawOut, tensor_t *auxOut,
                                   const void *ctx) {
    (void)n;
    (void)auxOut;
    const avgPool1dConfig_t *cfg = ctx;
    tensor_t *input = ops[0];

    size_t batch = input->shape->dimensions[0];
    size_t channels = input->shape->dimensions[1];
    size_t inputLength = input->shape->dimensions[2];

    windowGeometry1d_t geom = windowGeometry1dCalc(inputLength, cfg->kernel);
    size_t outputLength = geom.outputLength;

    if (rawOut->shape->dimensions[2] != outputLength) {
        PRINT_ERROR("AvgPool1d forward: output length (%zu) does not match "
                    "geometry-derived (%zu)",
                    rawOut->shape->dimensions[2], outputLength);
        exit(1);
    }

    float const *xArr = (float const *)input->data;
    float *yArr = (float *)rawOut->data;
    float divisor = (float)cfg->kernel->size; // count_include_pad=true: divisor = K always

    for (size_t b = 0; b < batch; b++) {
        for (size_t c = 0; c < channels; c++) {
            for (size_t outPos = 0; outPos < outputLength; outPos++) {
                windowSlice1d_t slice = windowSlice1dAt(&geom, outPos);

                float sum = 0.0f;
                for (size_t i = 0; i < slice.validCount; i++) {
                    size_t inputIdx = slice.firstValidInputIdx + i * geom.dilation;
                    sum += xArr[(b * channels + c) * inputLength + inputIdx];
                }

                size_t outIdx = (b * channels + c) * outputLength + outPos;
                yArr[outIdx] = sum / divisor;
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

/* SYM_INT32 arm (#205): exact int32 mantissa window sum + exact scale fold
 * s_out = s_in / K (Dropout idiom) — the /K division never touches the
 * mantissas, so the kernel carries ZERO rounding. count_include_pad=true
 * falls out for free: padded positions contribute 0 to the sum while the
 * fold keeps the divisor at K. The window sum is a value-sum of at most K
 * operand-width mantissas (no products — the no-int64 rule is trivially
 * satisfied), guarded by poolValidateSymValueSum above. The OUT_WRITE
 * epilogue width-restores the accumulator-range sums at the producer. */
static void avgPool1dForwardKernelSymInt32(tensor_t **ops, size_t n, tensor_t *rawOut,
                                           tensor_t *auxOut, const void *ctx) {
    (void)n;
    (void)auxOut;
    const avgPool1dConfig_t *cfg = ctx;
    tensor_t *input = ops[0];

    size_t batch = input->shape->dimensions[0];
    size_t channels = input->shape->dimensions[1];
    size_t inputLength = input->shape->dimensions[2];

    windowGeometry1d_t geom = windowGeometry1dCalc(inputLength, cfg->kernel);
    size_t outputLength = geom.outputLength;

    if (rawOut->shape->dimensions[2] != outputLength) {
        PRINT_ERROR("AvgPool1d forward: output length (%zu) does not match "
                    "geometry-derived (%zu)",
                    rawOut->shape->dimensions[2], outputLength);
        exit(1);
    }

    poolValidateSymValueSum(input, cfg->kernel->size, "AvgPool1d forward SYM");

    int32_t const *xArr = (int32_t const *)input->data;
    int32_t *yArr = (int32_t *)rawOut->data;

    for (size_t b = 0; b < batch; b++) {
        for (size_t c = 0; c < channels; c++) {
            for (size_t outPos = 0; outPos < outputLength; outPos++) {
                windowSlice1d_t slice = windowSlice1dAt(&geom, outPos);

                int32_t sum = 0;
                for (size_t i = 0; i < slice.validCount; i++) {
                    size_t inputIdx = slice.firstValidInputIdx + i * geom.dilation;
                    sum += xArr[(b * channels + c) * inputLength + inputIdx];
                }

                yArr[(b * channels + c) * outputLength + outPos] = sum;
            }
        }
    }

    ((symInt32QConfig_t *)rawOut->quantization->qConfig)->scale =
        ((symInt32QConfig_t *)input->quantization->qConfig)->scale /
        (float)cfg->kernel->size; // count_include_pad=true: divisor = K always
}

/* BFP epic PR4 (R-P1): GEMM's rule 1 is WEIGHT-anchored, but a pool has no
 * weight operand, so the width anchor for staging a FLOAT32-stored operand is
 * the layer's OWN produced-wire config — outputQ for the forward op,
 * propLossQ for the backward op (spec §4: "geometry comes from the op's ctx;
 * layers pass their bfpQConfigs via opSpec_t.ctx"). Validated EAGERLY at op
 * entry, the idiom ratified for the GEMM layers: without a BFP-typed
 * produced-wire config there is no width source at all, so the arm must not
 * run even when the operand happens to be BFP-stored. The pointer may be NULL
 * — the userApi factories copy layerQuant_t slots BY VALUE and never call
 * initAvgPool1dConfig, so a pinned ARITH_BFP math slot can arrive with a NULL
 * or non-BFP wire config. NULL-check before ->type. */
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
 * [batch][channels][outputLength] array whose batch and channels come from the
 * INPUT's dims, so a rawOut that disagrees on dim 0 or dim 1 is written past
 * its end even when its length is right — checking dimensions[2] alone is not
 * enough. Rank is checked first: dimensions[0..2] on a rank-2 shape is itself
 * an over-read. NOTE the deliberate scope: the FLOAT32/SYM arms of these three
 * pool layers have the SAME gap and are NOT touched here — this plan hardens
 * only the code it adds. */
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

/* BFP epic PR4 (R-P4), ARITH_BFP arm: operands arrive as the funnel's
 * unpacked-BFP scratch — int32 mantissa codes under a live bfpQConfig_t, with
 * exponents BORROWED zero-copy for a BFP-stored input and funnel-staged for a
 * FLOAT32 one. The window sum rides the PR2 fold contract: one int32 partial
 * per same-group visited segment, a lossless ldexpf fold on every group change
 * and at the tail (D8 — the operand is NEVER re-blocked; segments just fold
 * more often). Group lookup is PER ELEMENT, not a run precompute: dilation and
 * stride skip storage indices. The SYM arm's exact s/K fold has NO BFP analog
 * (a BFP scale is 2^E; K is not a power of two in general), so the divide
 * happens in the FLOAT32 raw intermediate (D7), which costs nothing extra.
 * count_include_pad=true still falls out free: padded positions are simply not
 * visited and the divisor stays K. Guarded by bfpValidateSumHeadroom — the
 * SUM limit (INT32_MAX >> (m-1)), not the product limit. */
static void avgPool1dForwardKernelBfp(tensor_t **ops, size_t n, tensor_t *rawOut, tensor_t *auxOut,
                                      const void *ctx) {
    (void)n;
    (void)auxOut;
    const avgPool1dConfig_t *cfg = ctx;
    tensor_t *input = ops[0];

    if (input->shape->numberOfDimensions != 3) {
        PRINT_ERROR("AvgPool1d forward BFP: input must be rank-3 [batch, channels, length], got "
                    "rank %zu",
                    input->shape->numberOfDimensions);
        exit(1);
    }
    size_t batch = input->shape->dimensions[0];
    size_t channels = input->shape->dimensions[1];
    size_t inputLength = input->shape->dimensions[2];

    windowGeometry1d_t geom = windowGeometry1dCalc(inputLength, cfg->kernel);
    size_t outputLength = geom.outputLength;

    /* All THREE dims, not just the length (F5): the write index below is
     * (b * channels + c) * outputLength + outPos with batch/channels taken
     * from the INPUT, so a rawOut that agrees on length but not on batch or
     * channels is written past its end. */
    poolBfpRequireDims3(rawOut, batch, channels, outputLength, "AvgPool1d forward BFP (rawOut)");

    const bfpQConfig_t *qC = input->quantization->qConfig;
    validateBfpQConfigShape(qC, calcNumberOfElementsByTensor(input));
    bfpValidateSumHeadroom(qC, cfg->kernel->size, "AvgPool1d forward BFP");
    const int32_t expBias = bfpExponentBias(qC);

    int32_t const *xArr = (int32_t const *)input->data;
    float *yArr = (float *)rawOut->data;
    const float divisor = (float)cfg->kernel->size;

    for (size_t b = 0; b < batch; b++) {
        for (size_t c = 0; c < channels; c++) {
            for (size_t outPos = 0; outPos < outputLength; outPos++) {
                windowSlice1d_t slice = windowSlice1dAt(&geom, outPos);

                float acc = 0.f;
                int32_t partial = 0;
                size_t currentGroup = 0;
                for (size_t i = 0; i < slice.validCount; i++) {
                    size_t inputIdx = slice.firstValidInputIdx + i * geom.dilation;
                    size_t storageIdx = (b * channels + c) * inputLength + inputIdx;
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
                if (slice.validCount > 0) {
                    acc += ldexpf((float)partial, (int)qC->exponents[currentGroup] - expBias);
                }

                yArr[(b * channels + c) * outputLength + outPos] = acc / divisor;
            }
        }
    }
}

void avgPool1dForward(layer_t *layer, tensor_t *input, tensor_t *output) {
    avgPool1dConfig_t *cfg = layer->config->avgPool1d;
    if (cfg->forwardMath.type == ARITH_BFP) {
        const bfpQConfig_t *anchor = poolBfpWireAnchor(cfg->outputQ, "AvgPool1d forward");
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
                .kernel = avgPool1dForwardKernelBfp,
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
        kernel = avgPool1dForwardKernel;
        break;
    case ARITH_SYM_INT32:
        kernel = avgPool1dForwardKernelSymInt32;
        break;
    default:
        PRINT_ERROR("AvgPool1d forward: quantization type not implemented");
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

/* BFP epic PR2 Task 8: same outside-funnel hole as Softmax backward. The
 * ARITH_FLOAT32 arm below runs OUTSIDE executeOp and raw-casts lossGrad/propLoss
 * to float*, selected by the layer's DECLARED propLossMath -- so a BFP dx wire
 * whose math slot is pinned to (or, before the Task 9 flip, derived as)
 * ARITH_FLOAT32 lands straight in those casts (4x heap over-read on lossGrad, over-write into the
 * packed propLoss buffer). Reachable only since this task's initGradTensor BFP
 * arm; before it a BFP propLossQ died in the allocator's default arm. Forward
 * needs no guard (it runs inside executeOp). BFP pooling is epic PR4 (spec
 * section 7), named in the message so the failure points at the roadmap.
 * forwardInput is NOT guarded: this layer never dereferences it. */
static void requireNoBfpWire(const tensor_t *t, const char *what) {
    if (t->quantization->type == BFP) {
        PRINT_ERROR("%s: BFP pooling semantics arrive with epic PR4 -- keep BFP off this wire "
                    "or use FLOAT32 wires",
                    what);
        exit(1);
    }
}

void avgPool1dBackwardFloat(layer_t *layer, tensor_t *forwardInput, tensor_t *lossGrad,
                            tensor_t *propLoss) {
    avgPool1dConfig_t *cfg = layer->config->avgPool1d;
    (void)forwardInput; // not needed: window geometry is determined by kernel + input shape

    size_t batch = lossGrad->shape->dimensions[0];
    size_t channels = lossGrad->shape->dimensions[1];
    size_t outputLength = lossGrad->shape->dimensions[2];
    size_t inputLength = propLoss->shape->dimensions[2];

    windowGeometry1d_t geom = windowGeometry1dCalc(inputLength, cfg->kernel);
    if (geom.outputLength != outputLength) {
        PRINT_ERROR("AvgPool1d backward: lossGrad outputLength (%zu) does not match "
                    "geometry-derived (%zu)",
                    outputLength, geom.outputLength);
        exit(1);
    }

    float const *gyArr = (float const *)lossGrad->data;
    float *gxArr = (float *)propLoss->data;
    float divisor = (float)cfg->kernel->size;

    for (size_t b = 0; b < batch; b++) {
        for (size_t c = 0; c < channels; c++) {
            for (size_t outPos = 0; outPos < outputLength; outPos++) {
                windowSlice1d_t slice = windowSlice1dAt(&geom, outPos);
                size_t outIdx = (b * channels + c) * outputLength + outPos;
                float contribution = gyArr[outIdx] / divisor;

                for (size_t i = 0; i < slice.validCount; i++) {
                    size_t inputIdx = slice.firstValidInputIdx + i * geom.dilation;
                    gxArr[(b * channels + c) * inputLength + inputIdx] += contribution;
                }
            }
        }
    }
}

/* SYM_INT32 dx kernel (#205): zero + scatter loss-grad mantissas into every
 * valid window member; the 1/K contribution folds EXACTLY into the raw scale
 * (s_dx = s_gy / K) — the transpose of the forward's fold, zero kernel
 * rounding. Funnel-routed like Conv1d's dx wire (prologue converts a
 * mismatched-dtype lossGrad, OUT_WRITE epilogue width-restores propLoss at
 * the producer). rawOut is the funnel's uninitialized Phase-2 scratch, so
 * the kernel memsets it before the scatter `+=` (overlapping windows). */
static void avgPool1dBackwardKernelSymInt32(tensor_t **ops, size_t n, tensor_t *rawOut,
                                            tensor_t *auxOut, const void *ctx) {
    (void)n;
    (void)auxOut;
    const avgPool1dConfig_t *cfg = ctx;
    tensor_t *lossGrad = ops[0];

    size_t batch = lossGrad->shape->dimensions[0];
    size_t channels = lossGrad->shape->dimensions[1];
    size_t outputLength = lossGrad->shape->dimensions[2];
    size_t inputLength = rawOut->shape->dimensions[2];

    windowGeometry1d_t geom = windowGeometry1dCalc(inputLength, cfg->kernel);
    if (geom.outputLength != outputLength) {
        PRINT_ERROR("AvgPool1d backward: lossGrad outputLength (%zu) does not match "
                    "geometry-derived (%zu)",
                    outputLength, geom.outputLength);
        exit(1);
    }

    /* Worst-case scatter collisions per input cell = covering windows =
     * (effK-1)/stride + 1. */
    size_t effectiveKernel = geom.dilation * (geom.kernelSize - 1) + 1;
    poolValidateSymValueSum(lossGrad, (effectiveKernel - 1) / geom.stride + 1,
                            "AvgPool1d backward SYM");

    int32_t const *gyArr = (int32_t const *)lossGrad->data;
    int32_t *gxArr = (int32_t *)rawOut->data;

    memset(gxArr, 0, batch * channels * inputLength * sizeof(int32_t));

    for (size_t b = 0; b < batch; b++) {
        for (size_t c = 0; c < channels; c++) {
            for (size_t outPos = 0; outPos < outputLength; outPos++) {
                windowSlice1d_t slice = windowSlice1dAt(&geom, outPos);
                int32_t gy = gyArr[(b * channels + c) * outputLength + outPos];

                for (size_t i = 0; i < slice.validCount; i++) {
                    size_t inputIdx = slice.firstValidInputIdx + i * geom.dilation;
                    gxArr[(b * channels + c) * inputLength + inputIdx] += gy;
                }
            }
        }
    }

    ((symInt32QConfig_t *)rawOut->quantization->qConfig)->scale =
        ((symInt32QConfig_t *)lossGrad->quantization->qConfig)->scale / (float)cfg->kernel->size;
}

void avgPool1dBackward(layer_t *layer, tensor_t *forwardInput, tensor_t *lossGrad,
                       tensor_t *propLoss) {
    requireNoBfpWire(lossGrad, "AvgPool1d backward (lossGrad)");
    requireNoBfpWire(propLoss, "AvgPool1d backward (propLoss)");
    avgPool1dConfig_t *cfg = layer->config->avgPool1d;
    switch (cfg->propLossMath.type) {
    case ARITH_FLOAT32:
        avgPool1dBackwardFloat(layer, forwardInput, lossGrad, propLoss);
        break;
    case ARITH_SYM_INT32:
        (void)forwardInput; // not needed: window geometry comes from kernel + shapes
        executeOp(
            &(opSpec_t){
                .kernel = avgPool1dBackwardKernelSymInt32,
                .ctx = cfg,
                .inputs = (tensor_t *[]){lossGrad},
                .nInputs = 1,
                .arithmetic = cfg->propLossMath,
                .mode = OUT_WRITE,
            },
            propLoss);
        break;
    default:
        PRINT_ERROR("AvgPool1d backward: quantization type not implemented");
        exit(1);
    }
}

void avgPool1dCalcOutputShape(layer_t *layer, shape_t *inputShape, shape_t *outputShape) {
    if (inputShape->numberOfDimensions != 3) {
        PRINT_ERROR("AvgPool1d expects 3D input [batch, channel, length], got %luD",
                    inputShape->numberOfDimensions);
        exit(1);
    }

    avgPool1dConfig_t *cfg = layer->config->avgPool1d;
    size_t inputLength = inputShape->dimensions[2];
    windowGeometry1d_t geom = windowGeometry1dCalc(inputLength, cfg->kernel);

    outputShape->numberOfDimensions = 3;
    outputShape->dimensions[0] = inputShape->dimensions[0]; // B
    outputShape->dimensions[1] = inputShape->dimensions[1]; // C
    outputShape->dimensions[2] = geom.outputLength;
    setOrderOfDimsForNewTensor(inputShape->numberOfDimensions, outputShape->orderOfDimensions);
}
