#define SOURCE_FILE "ODT_MAX_POOL_1D"

#include <math.h>
#include <string.h>

#include "MaxPool1d.h"

#include "ArithmeticType.h"
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

void maxPool1dForward(layer_t *layer, tensor_t *input, tensor_t *output) {
    maxPool1dConfig_t *cfg = layer->config->maxPool1d;
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

/* BFP epic PR2 Task 8: same outside-funnel hole as Softmax backward. The
 * ARITH_FLOAT32 arm below runs OUTSIDE executeOp and raw-casts lossGrad/propLoss
 * to float*, selected by the layer's DECLARED propLossMath -- so pre-flip, when
 * arithmeticFromQuantization(BFP) is still ARITH_FLOAT32, a BFP dx wire lands
 * straight in those casts (4x heap over-read on lossGrad, over-write into the
 * packed propLoss buffer). Reachable only since this task's initGradTensor BFP
 * arm; before it a BFP propLossQ died in the allocator's default arm. Forward
 * needs no guard (it runs inside executeOp). No epic PR is assigned to BFP
 * pooling yet -- the message states the gap rather than promising a milestone.
 * forwardInput is NOT guarded: this layer never dereferences it. */
static void requireNoBfpWire(const tensor_t *t, const char *what) {
    if (t->quantization->type == BFP) {
        PRINT_ERROR("%s: BFP pooling semantics are not implemented -- keep BFP off this wire "
                    "or use FLOAT32 wires",
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

void maxPool1dBackward(layer_t *layer, tensor_t *forwardInput, tensor_t *lossGrad,
                       tensor_t *propLoss) {
    requireNoBfpWire(lossGrad, "MaxPool1d backward (lossGrad)");
    requireNoBfpWire(propLoss, "MaxPool1d backward (propLoss)");
    maxPool1dConfig_t *cfg = layer->config->maxPool1d;
    switch (cfg->propLossMath.type) {
    case ARITH_FLOAT32:
        maxPool1dBackwardFloat(layer, forwardInput, lossGrad, propLoss);
        break;
    case ARITH_SYM_INT32:
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
