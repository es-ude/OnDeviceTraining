#define SOURCE_FILE "ODT_MAX_POOL_1D"

#include <math.h>
#include <string.h>

#include "MaxPool1d.h"

#include "ArithmeticType.h"
#include "BfpKernelSupport.h"
#include "Common.h"
#include "ExecuteOp.h"
#include "Layer.h"
#include "Quantization.h"
#include "SlidingWindow1d.h"
#include "Tensor.h"

void initMaxPool1dConfig(maxPool1dConfig_t *cfg, kernel_t *kernel, tensor_t *argmaxIndices,
                         quantization_t *forwardQ, quantization_t *propLossQ) {
    if (argmaxIndices == NULL) {
        PRINT_ERROR("MaxPool1d: argmaxIndices must not be NULL — caller must pre-allocate");
        exit(1);
    }
    cfg->kernel = kernel;
    cfg->argmaxIndices = argmaxIndices;
    cfg->forwardMath = arithmeticFromQuantizationOrDefault(forwardQ);
    cfg->propLossMath = arithmeticFromQuantizationOrDefault(propLossQ);
    cfg->outputQ = forwardQ;
    cfg->propLossQ = propLossQ;
}

/* executeOp forward kernel adapter — ctx = maxPool1dConfig_t* for kernel_t
 * geometry (mirrors AvgPool1d/Conv1d's ctx convention). auxOut = the layer's
 * pre-allocated argmaxIndices tensor (opSpec_t.auxOut, spec D1): the funnel
 * never converts it (kernel-written verbatim, in ITS OWN storage format,
 * INT32) — this is exactly the dual-output shape auxOut was added for (D1's
 * "MaxPool argmaxIndices lives here"). The SYM_INT32 arm lives in
 * maxPool1dForwardKernelSymInt32 below (#205). */
static void maxPool1dForwardKernel(tensor_t **ops, size_t n, tensor_t *rawOut, tensor_t *auxOut,
                                   const void *ctx) {
    (void)n;
    const maxPool1dConfig_t *cfg = ctx;
    tensor_t *input = ops[0];

    size_t batch = input->shape->dimensions[0];
    size_t channels = input->shape->dimensions[1];
    size_t inputLength = input->shape->dimensions[2];

    windowGeometry1d_t geom = windowGeometry1dCalc(inputLength, cfg->kernel);
    size_t outputLength = geom.outputLength;

    if (rawOut->shape->dimensions[2] != outputLength) {
        PRINT_ERROR("MaxPool1d forward: output length (%zu) does not match "
                    "geometry-derived (%zu)",
                    rawOut->shape->dimensions[2], outputLength);
        exit(1);
    }
    if (auxOut->shape->dimensions[2] != outputLength) {
        PRINT_ERROR("MaxPool1d forward: argmaxIndices length (%zu) does not match "
                    "geometry-derived (%zu)",
                    auxOut->shape->dimensions[2], outputLength);
        exit(1);
    }

    float const *xArr = (float const *)input->data;
    float *yArr = (float *)rawOut->data;
    int32_t *argmaxArr = (int32_t *)auxOut->data;

    for (size_t b = 0; b < batch; b++) {
        for (size_t c = 0; c < channels; c++) {
            for (size_t outPos = 0; outPos < outputLength; outPos++) {
                windowSlice1d_t slice = windowSlice1dAt(&geom, outPos);

                float bestVal = -INFINITY;
                int32_t bestInputIdx = -1;
                for (size_t i = 0; i < slice.validCount; i++) {
                    size_t inputIdx = slice.firstValidInputIdx + i * geom.dilation;
                    float v = xArr[(b * channels + c) * inputLength + inputIdx];
                    if (v > bestVal) {
                        bestVal = v;
                        bestInputIdx = (int32_t)inputIdx;
                    }
                }

                size_t outIdx = (b * channels + c) * outputLength + outPos;
                if (slice.validCount > 0) {
                    yArr[outIdx] = bestVal;
                    argmaxArr[outIdx] = bestInputIdx;
                } else {
                    // spec §6.3: empty window is theoretically possible but in
                    // practice unreachable; log + sentinel-encode rather than exit
                    yArr[outIdx] = 0.0f;
                    argmaxArr[outIdx] = -1;
                    PRINT_ERROR("MaxPool1d: empty window at outPos=%zu — likely user misconfig",
                                outPos);
                }
            }
        }
    }
}

/* SYM_INT32 arm (#205): pure mantissa select — argmax over int32 mantissas is
 * argmax over values (scale > 0 preserves order), scale copied to the raw
 * intermediate (ReLU idiom); the OUT_WRITE epilogue width-restores at the
 * producer like every funnel SYM path. Tie-break matches the FLOAT32 arm:
 * strict >, first occurrence wins — but ties happen in the QUANTIZED domain,
 * so inputs that quantize to the same mantissa tie here even when their float
 * values differ. */
static void maxPool1dForwardKernelSymInt32(tensor_t **ops, size_t n, tensor_t *rawOut,
                                           tensor_t *auxOut, const void *ctx) {
    (void)n;
    const maxPool1dConfig_t *cfg = ctx;
    tensor_t *input = ops[0];

    size_t batch = input->shape->dimensions[0];
    size_t channels = input->shape->dimensions[1];
    size_t inputLength = input->shape->dimensions[2];

    windowGeometry1d_t geom = windowGeometry1dCalc(inputLength, cfg->kernel);
    size_t outputLength = geom.outputLength;

    if (rawOut->shape->dimensions[2] != outputLength) {
        PRINT_ERROR("MaxPool1d forward: output length (%zu) does not match "
                    "geometry-derived (%zu)",
                    rawOut->shape->dimensions[2], outputLength);
        exit(1);
    }
    if (auxOut->shape->dimensions[2] != outputLength) {
        PRINT_ERROR("MaxPool1d forward: argmaxIndices length (%zu) does not match "
                    "geometry-derived (%zu)",
                    auxOut->shape->dimensions[2], outputLength);
        exit(1);
    }

    int32_t const *xArr = (int32_t const *)input->data;
    int32_t *yArr = (int32_t *)rawOut->data;
    int32_t *argmaxArr = (int32_t *)auxOut->data;

    for (size_t b = 0; b < batch; b++) {
        for (size_t c = 0; c < channels; c++) {
            for (size_t outPos = 0; outPos < outputLength; outPos++) {
                windowSlice1d_t slice = windowSlice1dAt(&geom, outPos);

                int32_t bestVal = 0;
                int32_t bestInputIdx = -1;
                for (size_t i = 0; i < slice.validCount; i++) {
                    size_t inputIdx = slice.firstValidInputIdx + i * geom.dilation;
                    int32_t v = xArr[(b * channels + c) * inputLength + inputIdx];
                    if (bestInputIdx < 0 || v > bestVal) {
                        bestVal = v;
                        bestInputIdx = (int32_t)inputIdx;
                    }
                }

                size_t outIdx = (b * channels + c) * outputLength + outPos;
                if (slice.validCount > 0) {
                    yArr[outIdx] = bestVal;
                    argmaxArr[outIdx] = bestInputIdx;
                } else {
                    // spec §6.3: empty window is theoretically possible but in
                    // practice unreachable; log + sentinel-encode rather than exit
                    yArr[outIdx] = 0;
                    argmaxArr[outIdx] = -1;
                    PRINT_ERROR("MaxPool1d: empty window at outPos=%zu — likely user misconfig",
                                outPos);
                }
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
 * call initMaxPool1dConfig, so a pinned ARITH_BFP math slot can arrive with a
 * NULL or non-BFP wire config. NULL-check before ->type. (Per-file static,
 * like requireNoBfpWire and poolValidateSymValueSum — the pool layers
 * duplicate these rather than sharing a header.) */
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

/* BFP epic PR4 (F5): the new BFP kernels index their outputs as DENSE
 * [batch][channels][length] arrays whose batch and channels come from the
 * OPERAND's dims, so an output that disagrees on dim 0 or dim 1 is written
 * past its end even when its length is right — checking dimensions[2] alone is
 * not enough. Rank is checked first: dimensions[0..2] on a rank-2 shape is
 * itself an over-read. NOTE the deliberate scope: the FLOAT32/SYM arms of this
 * layer have the SAME gap and are NOT touched here. */
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

/* BFP epic PR4 (R-P4), ARITH_BFP arm: unlike the SYM arm's free ride (scale > 0
 * preserves order, so mantissa select IS value select), raw BFP mantissas are
 * NOT comparable across groups — a smaller code under a larger group exponent
 * can be the true maximum. Every candidate is therefore DEQUANTIZED with
 * ldexpf((float)mant, E_g - bias), which is exact (a float32 multiply by a
 * power of two), and the comparison runs on values. The winner's dequant goes
 * straight into the FLOAT32 raw (D7) — no scale copy, because the raw has no
 * scale. Tie-break matches the FLOAT32 arm: strict >, seeded at -INFINITY,
 * first occurrence wins. argmax semantics (auxOut, INT32, never
 * funnel-converted; -1 empty-window sentinel) are unchanged. No headroom
 * guard: nothing is summed. */
static void maxPool1dForwardKernelBfp(tensor_t **ops, size_t n, tensor_t *rawOut, tensor_t *auxOut,
                                      const void *ctx) {
    (void)n;
    const maxPool1dConfig_t *cfg = ctx;
    tensor_t *input = ops[0];

    if (input->shape->numberOfDimensions != 3) {
        PRINT_ERROR("MaxPool1d forward BFP: input must be rank-3 [batch, channels, length], got "
                    "rank %zu",
                    input->shape->numberOfDimensions);
        exit(1);
    }
    size_t batch = input->shape->dimensions[0];
    size_t channels = input->shape->dimensions[1];
    size_t inputLength = input->shape->dimensions[2];

    windowGeometry1d_t geom = windowGeometry1dCalc(inputLength, cfg->kernel);
    size_t outputLength = geom.outputLength;

    /* Full shape on BOTH outputs (F5): rawOut and auxOut are indexed with the
     * SAME (b * channels + c) * outputLength + outPos, batch/channels taken
     * from the input, so both need all three dims validated — auxOut most of
     * all, because it is never funnel-converted and is written raw. */
    poolBfpRequireDims3(rawOut, batch, channels, outputLength, "MaxPool1d forward BFP (rawOut)");
    poolBfpRequireDims3(auxOut, batch, channels, outputLength,
                        "MaxPool1d forward BFP (argmaxIndices)");

    const bfpQConfig_t *qC = input->quantization->qConfig;
    validateBfpQConfigShape(qC, calcNumberOfElementsByTensor(input));
    const int32_t expBias = bfpExponentBias(qC);

    int32_t const *xArr = (int32_t const *)input->data;
    float *yArr = (float *)rawOut->data;
    int32_t *argmaxArr = (int32_t *)auxOut->data;

    for (size_t b = 0; b < batch; b++) {
        for (size_t c = 0; c < channels; c++) {
            for (size_t outPos = 0; outPos < outputLength; outPos++) {
                windowSlice1d_t slice = windowSlice1dAt(&geom, outPos);

                float bestVal = -INFINITY;
                int32_t bestInputIdx = -1;
                for (size_t i = 0; i < slice.validCount; i++) {
                    size_t inputIdx = slice.firstValidInputIdx + i * geom.dilation;
                    size_t storageIdx = (b * channels + c) * inputLength + inputIdx;
                    size_t g = bfpGroupOf(qC, storageIdx);
                    float v = ldexpf((float)xArr[storageIdx], (int)qC->exponents[g] - expBias);
                    if (v > bestVal) {
                        bestVal = v;
                        bestInputIdx = (int32_t)inputIdx;
                    }
                }

                size_t outIdx = (b * channels + c) * outputLength + outPos;
                if (slice.validCount > 0) {
                    yArr[outIdx] = bestVal;
                    argmaxArr[outIdx] = bestInputIdx;
                } else {
                    // spec §6.3: empty window is theoretically possible but in
                    // practice unreachable; log + sentinel-encode rather than exit
                    yArr[outIdx] = 0.0f;
                    argmaxArr[outIdx] = -1;
                    PRINT_ERROR("MaxPool1d: empty window at outPos=%zu — likely user misconfig",
                                outPos);
                }
            }
        }
    }
}

void maxPool1dForward(layer_t *layer, tensor_t *input, tensor_t *output) {
    maxPool1dConfig_t *cfg = layer->config->maxPool1d;
    if (cfg->forwardMath.type == ARITH_BFP) {
        const bfpQConfig_t *anchor = poolBfpWireAnchor(cfg->outputQ, "MaxPool1d forward");
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
                .kernel = maxPool1dForwardKernelBfp,
                .ctx = cfg,
                .inputs = (tensor_t *[]){input},
                .nInputs = 1,
                .arithmetic = cfg->forwardMath,
                .mode = OUT_WRITE,
                .auxOut = cfg->argmaxIndices,
                .bfpStage = {input->quantization->type == FLOAT32 ? &stage : NULL, NULL, NULL},
            },
            output);
        return;
    }

    opKernelFn_t kernel;
    switch (cfg->forwardMath.type) {
    case ARITH_FLOAT32:
        kernel = maxPool1dForwardKernel;
        break;
    case ARITH_SYM_INT32:
        kernel = maxPool1dForwardKernelSymInt32;
        break;
    default:
        PRINT_ERROR("MaxPool1d forward: quantization type not implemented");
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
            .auxOut = cfg->argmaxIndices,
        },
        output);
}

/* BFP epic PR4 (R-P4): the ARITH_BFP arm below IS the native pooling path, so
 * this guard is NARROWED to the two arms that raw-view ->data in their own
 * storage format — a packed BFP wire read as float* / int32_t* is a 4x heap
 * over-read on lossGrad and an over-write into the packed propLoss buffer.
 * Keyed on the wire's STORAGE dtype, not the declared arithmetic (#315
 * parity). forwardInput is NOT guarded: this layer never dereferences it
 * (the argmax indices recorded by the forward carry all the routing). */
static void requireNoBfpWire(const tensor_t *t, const char *what) {
    if (t->quantization->type == BFP) {
        PRINT_ERROR("%s: this arm raw-views the wire in its own storage format and cannot read "
                    "packed BFP mantissas -- derive ARITH_BFP from a BFP wire config, or keep "
                    "BFP off this wire",
                    what);
        exit(1);
    }
}

void maxPool1dBackwardFloat(layer_t *layer, tensor_t *forwardInput, tensor_t *lossGrad,
                            tensor_t *propLoss) {
    maxPool1dConfig_t *cfg = layer->config->maxPool1d;
    (void)forwardInput; // not needed: argmax already encodes which input position to update.

    size_t batch = lossGrad->shape->dimensions[0];
    size_t channels = lossGrad->shape->dimensions[1];
    size_t outputLength = lossGrad->shape->dimensions[2];
    size_t inputLength = propLoss->shape->dimensions[2];

    // Defensive: argmax shape must match lossGrad shape.
    if (cfg->argmaxIndices->shape->dimensions[2] != outputLength) {
        PRINT_ERROR("MaxPool1d backward: argmaxIndices length (%zu) does not match "
                    "lossGrad outputLength (%zu)",
                    cfg->argmaxIndices->shape->dimensions[2], outputLength);
        exit(1);
    }

    float const *gyArr = (float const *)lossGrad->data;
    int32_t const *argmaxArr = (int32_t const *)cfg->argmaxIndices->data;
    float *gxArr = (float *)propLoss->data;

    for (size_t b = 0; b < batch; b++) {
        for (size_t c = 0; c < channels; c++) {
            for (size_t outPos = 0; outPos < outputLength; outPos++) {
                size_t outIdx = (b * channels + c) * outputLength + outPos;
                int32_t inputIdx = argmaxArr[outIdx];
                if (inputIdx < 0) {
                    continue; // sentinel: empty window, no gradient flows
                }
                gxArr[(b * channels + c) * inputLength + (size_t)inputIdx] += gyArr[outIdx];
            }
        }
    }
}

/* Pool SYM value-sum guard (Reduce.c precedent: reduceValidateSymOperand +
 * meanOverTrailingAxesSymInt32's N-bound): the scatter accumulates a VALUE-sum
 * of operand-width mantissas in an int32 — sound only for qMaxBits in [1,16]
 * and worst-case terms < 2^(32-qMaxBits). qMaxBits == 0 is degenerate and
 * would make the shift UB. No dtype check: the executeOp prologue guarantees
 * a SYM_INT32 operand inside an ARITH_SYM_INT32 kernel. */
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

/* SYM_INT32 dx kernel (#205): zero + scatter loss-grad mantissas to the argmax
 * positions recorded by the forward (scale copied from the loss grad — pure
 * select transpose). Funnel-routed like Conv1d's dx wire: the prologue
 * converts a mismatched-dtype lossGrad into SYM scratch, the OUT_WRITE
 * epilogue width-restores propLoss at the producer. ops = {lossGrad}; the
 * argmax tensor arrives via ctx (kernel-written by forward, never
 * funnel-converted), auxOut is unused. rawOut is the funnel's uninitialized
 * Phase-2 scratch, so the kernel memsets it before the scatter `+=`
 * (overlapping windows may hit the same input index — Conv1d weight-grad
 * kernel precedent). */
static void maxPool1dBackwardKernelSymInt32(tensor_t **ops, size_t n, tensor_t *rawOut,
                                            tensor_t *auxOut, const void *ctx) {
    (void)n;
    (void)auxOut;
    const maxPool1dConfig_t *cfg = ctx;
    tensor_t *lossGrad = ops[0];

    size_t batch = lossGrad->shape->dimensions[0];
    size_t channels = lossGrad->shape->dimensions[1];
    size_t outputLength = lossGrad->shape->dimensions[2];
    size_t inputLength = rawOut->shape->dimensions[2];

    // Defensive: argmax shape must match lossGrad shape (FLOAT32 arm parity).
    if (cfg->argmaxIndices->shape->dimensions[2] != outputLength) {
        PRINT_ERROR("MaxPool1d backward: argmaxIndices length (%zu) does not match "
                    "lossGrad outputLength (%zu)",
                    cfg->argmaxIndices->shape->dimensions[2], outputLength);
        exit(1);
    }

    /* An input position can be argmax only of windows CONTAINING it, so the
     * worst-case scatter collisions per cell = covering windows =
     * (effK-1)/stride + 1. */
    size_t effectiveKernel = cfg->kernel->dilation * (cfg->kernel->size - 1) + 1;
    poolValidateSymValueSum(lossGrad, (effectiveKernel - 1) / cfg->kernel->stride + 1,
                            "MaxPool1d backward SYM");

    int32_t const *gyArr = (int32_t const *)lossGrad->data;
    int32_t const *argmaxArr = (int32_t const *)cfg->argmaxIndices->data;
    int32_t *gxArr = (int32_t *)rawOut->data;

    memset(gxArr, 0, batch * channels * inputLength * sizeof(int32_t));

    for (size_t b = 0; b < batch; b++) {
        for (size_t c = 0; c < channels; c++) {
            for (size_t outPos = 0; outPos < outputLength; outPos++) {
                size_t outIdx = (b * channels + c) * outputLength + outPos;
                int32_t inputIdx = argmaxArr[outIdx];
                if (inputIdx < 0) {
                    continue; // sentinel: empty window, no gradient flows
                }
                gxArr[(b * channels + c) * inputLength + (size_t)inputIdx] += gyArr[outIdx];
            }
        }
    }

    ((symInt32QConfig_t *)rawOut->quantization->qConfig)->scale =
        ((symInt32QConfig_t *)lossGrad->quantization->qConfig)->scale;
}

/* BFP epic PR4 (R-P4 backward): funnel-routed dx, the exact shape of the
 * ARITH_SYM_INT32 arm above. Each output cell's EXACT dequant
 * (mantissa * 2^(E-bias)) is routed to the input position the forward recorded
 * in argmax and accumulated in the FLOAT32 raw (D7) — no divide (unlike
 * AvgPool: max is a SELECT, its transpose is a scatter of the untouched
 * gradient) and no int32 partials, hence no sum-headroom guard. ops =
 * {lossGrad}; the argmax tensor arrives via ctx (kernel-written by the
 * forward, never funnel-converted), auxOut is unused. rawOut is the funnel's
 * uninitialized Phase-2 scratch, so it is memset before the `+=` (one input
 * cell can be the argmax of several overlapping windows). */
static void maxPool1dBackwardKernelBfp(tensor_t **ops, size_t n, tensor_t *rawOut, tensor_t *auxOut,
                                       const void *ctx) {
    (void)n;
    (void)auxOut;
    const maxPool1dConfig_t *cfg = ctx;
    tensor_t *lossGrad = ops[0];

    if (lossGrad->shape->numberOfDimensions != 3 || rawOut->shape->numberOfDimensions != 3) {
        PRINT_ERROR("MaxPool1d backward BFP: lossGrad and rawOut must both be rank-3, got ranks "
                    "%zu and %zu",
                    lossGrad->shape->numberOfDimensions, rawOut->shape->numberOfDimensions);
        exit(1);
    }
    size_t batch = lossGrad->shape->dimensions[0];
    size_t channels = lossGrad->shape->dimensions[1];
    size_t outputLength = lossGrad->shape->dimensions[2];
    size_t inputLength = rawOut->shape->dimensions[2];

    /* All three dims on BOTH (F5): the memset sizes batch * channels *
     * inputLength floats, and argmaxArr is read at the SAME flat index as
     * lossGrad — an argmax tensor that matches only on outputLength is read
     * past its end for every b, c beyond its own. */
    poolBfpRequireDims3(rawOut, batch, channels, inputLength, "MaxPool1d backward BFP (rawOut)");
    poolBfpRequireDims3(cfg->argmaxIndices, batch, channels, outputLength,
                        "MaxPool1d backward BFP (argmaxIndices)");

    const bfpQConfig_t *qC = lossGrad->quantization->qConfig;
    validateBfpQConfigShape(qC, calcNumberOfElementsByTensor(lossGrad));
    const int32_t expBias = bfpExponentBias(qC);

    int32_t const *gyArr = (int32_t const *)lossGrad->data;
    int32_t const *argmaxArr = (int32_t const *)cfg->argmaxIndices->data;
    float *gxArr = (float *)rawOut->data;

    memset(gxArr, 0, batch * channels * inputLength * sizeof(float));

    for (size_t b = 0; b < batch; b++) {
        for (size_t c = 0; c < channels; c++) {
            for (size_t outPos = 0; outPos < outputLength; outPos++) {
                size_t outIdx = (b * channels + c) * outputLength + outPos;
                int32_t inputIdx = argmaxArr[outIdx];
                if (inputIdx < 0) {
                    continue; // sentinel: empty window, no gradient flows
                }
                /* F7: -1 is the ONLY legal out-of-range value. Any other index
                 * outside [0, inputLength) is a scatter past the end of the
                 * raw grad buffer — a silent heap write, not a wrong number.
                 * The argmax tensor is kernel-written and never funnel-
                 * converted, so nothing upstream has validated its CONTENT;
                 * a stale argmax left over from a differently-shaped forward
                 * is exactly how a too-large index arrives. Bounds-check on
                 * read rather than trusting the producer. */
                if ((size_t)inputIdx >= inputLength) {
                    PRINT_ERROR("MaxPool1d backward BFP: argmax index %d at output position %zu "
                                "is outside [0, %zu) -- the recorded argmax does not belong to "
                                "this input shape (stale forward?)",
                                inputIdx, outIdx, inputLength);
                    exit(1);
                }
                size_t g = bfpGroupOf(qC, outIdx);
                gxArr[(b * channels + c) * inputLength + (size_t)inputIdx] +=
                    ldexpf((float)gyArr[outIdx], (int)qC->exponents[g] - expBias);
            }
        }
    }
}

void maxPool1dBackward(layer_t *layer, tensor_t *forwardInput, tensor_t *lossGrad,
                       tensor_t *propLoss) {
    maxPool1dConfig_t *cfg = layer->config->maxPool1d;
    switch (cfg->propLossMath.type) {
    case ARITH_FLOAT32:
        /* Runs OUTSIDE executeOp and raw-casts both wires to float*; a packed
         * BFP wire would be read as wide scalars (4x heap over-read on
         * lossGrad, over-write into the packed propLoss buffer). #315 parity. */
        requireNoBfpWire(lossGrad, "MaxPool1d backward (lossGrad)");
        requireNoBfpWire(propLoss, "MaxPool1d backward (propLoss)");
        maxPool1dBackwardFloat(layer, forwardInput, lossGrad, propLoss);
        break;
    case ARITH_SYM_INT32:
        requireNoBfpWire(lossGrad, "MaxPool1d backward (lossGrad)");
        requireNoBfpWire(propLoss, "MaxPool1d backward (propLoss)");
        (void)forwardInput; // not needed: argmax already encodes the update position
        executeOp(
            &(opSpec_t){
                .kernel = maxPool1dBackwardKernelSymInt32,
                .ctx = cfg,
                .inputs = (tensor_t *[]){lossGrad},
                .nInputs = 1,
                .arithmetic = cfg->propLossMath,
                .mode = OUT_WRITE,
            },
            propLoss);
        break;
    case ARITH_BFP: {
        const bfpQConfig_t *anchor = poolBfpWireAnchor(cfg->propLossQ, "MaxPool1d backward");
        bfpQConfig_t stage = {.exponents = NULL,
                              .numGroups = 1,
                              .groupSize = 0,
                              .roundingMode = cfg->propLossMath.roundingMode,
                              .mantissaBits = anchor->mantissaBits,
                              .exponentBits = anchor->exponentBits};
        (void)forwardInput; // not needed: argmax already encodes the update position
        executeOp(
            &(opSpec_t){
                .kernel = maxPool1dBackwardKernelBfp,
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
        PRINT_ERROR("MaxPool1d backward: quantization type not implemented");
        exit(1);
    }
}

void maxPool1dCalcOutputShape(layer_t *layer, shape_t *inputShape, shape_t *outputShape) {
    if (inputShape->numberOfDimensions != 3) {
        PRINT_ERROR("MaxPool1d expects 3D input [batch, channel, length], got %luD",
                    inputShape->numberOfDimensions);
        exit(1);
    }

    maxPool1dConfig_t *cfg = layer->config->maxPool1d;
    size_t inputLength = inputShape->dimensions[2];
    windowGeometry1d_t geom = windowGeometry1dCalc(inputLength, cfg->kernel);

    outputShape->numberOfDimensions = 3;
    outputShape->dimensions[0] = inputShape->dimensions[0]; // B
    outputShape->dimensions[1] = inputShape->dimensions[1]; // C
    outputShape->dimensions[2] = geom.outputLength;
    setOrderOfDimsForNewTensor(inputShape->numberOfDimensions, outputShape->orderOfDimensions);
}
