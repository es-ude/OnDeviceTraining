#define SOURCE_FILE "ODT_ADAPTIVE_AVG_POOL_1D"

#include <string.h>

#include "AdaptiveAvgPool1d.h"

#include "AdaptiveWindow1d.h"
#include "ArithmeticType.h"
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

void adaptiveAvgPool1dForward(layer_t *layer, tensor_t *input, tensor_t *output) {
    adaptiveAvgPool1dConfig_t *cfg = layer->config->adaptiveAvgPool1d;
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

void adaptiveAvgPool1dBackward(layer_t *layer, tensor_t *forwardInput, tensor_t *lossGrad,
                               tensor_t *propLoss) {
    requireNoBfpWire(lossGrad, "AdaptiveAvgPool1d backward (lossGrad)");
    requireNoBfpWire(propLoss, "AdaptiveAvgPool1d backward (propLoss)");
    adaptiveAvgPool1dConfig_t *cfg = layer->config->adaptiveAvgPool1d;
    switch (cfg->propLossMath.type) {
    case ARITH_FLOAT32:
        adaptiveAvgPool1dBackwardFloat(layer, forwardInput, lossGrad, propLoss);
        break;
    case ARITH_SYM_INT32:
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
