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
#include "LayerExec.h"
#include "LayerWireGuards.h"
#include "PoolSymGuard.h"
#include "Quantization.h"
#include "ScaleCanon.h"
#include "Tensor.h"

void initAdaptiveAvgPool1dConfig(adaptiveAvgPool1dConfig_t *cfg, size_t outputSize,
                                 quantization_t *forwardQ, quantization_t *propLossQ) {
    cfg->outputSize = outputSize;
    cfg->forwardMath = arithmeticFromQuantizationOrDefault(forwardQ);
    cfg->propLossMath = arithmeticFromQuantizationOrDefault(propLossQ);
    cfg->outputQ = forwardQ;
    cfg->propLossQ = propLossQ;
    cfg->ownsQuantizations = false;
}

static void adaptiveAvgPool1dForwardKernel(tensor_t **ops, size_t n, tensor_t *rawOut,
                                           tensor_t *auxOut, const void *ctx, rngStream_t *rs) {
    (void)n;
    (void)auxOut;
    const adaptiveAvgPool1dConfig_t *cfg = ctx;
    tensor_t *input = ops[0];

    size_t batch = input->shape->dimensions[0];
    size_t channels = input->shape->dimensions[1];
    size_t inputLength = input->shape->dimensions[2];
    size_t outputLength = cfg->outputSize;

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

/* Variable window counts cannot fold into one scale, so divide each mantissa
 * sum with half-away integer division and preserve the operand scale (at most
 * 0.5 LSB error). The quotient stays in operand range (|mean| <= |max|). */
static void adaptiveAvgPool1dForwardKernelSymInt32(tensor_t **ops, size_t n, tensor_t *rawOut,
                                                   tensor_t *auxOut, const void *ctx,
                                                   rngStream_t *rs) {
    (void)n;
    (void)auxOut;
    const adaptiveAvgPool1dConfig_t *cfg = ctx;
    tensor_t *input = ops[0];

    size_t batch = input->shape->dimensions[0];
    size_t channels = input->shape->dimensions[1];
    size_t inputLength = input->shape->dimensions[2];
    size_t outputLength = cfg->outputSize;

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

    storeCanonScale(&((symInt32QConfig_t *)rawOut->quantization->qConfig)->scale,
                    loadCanonScale(&((symInt32QConfig_t *)input->quantization->qConfig)->scale));
}

/* Fold same-group int32 partials into float (on each group change and at the
 * tail), then divide by this adaptive window's count in the FLOAT32 raw. Sum
 * headroom uses the SYM arm's conservative bound, ceil(L/O) + 1. */
static void adaptiveAvgPool1dForwardKernelBfp(tensor_t **ops, size_t n, tensor_t *rawOut,
                                              tensor_t *auxOut, const void *ctx, rngStream_t *rs) {
    (void)n;
    (void)auxOut;
    const adaptiveAvgPool1dConfig_t *cfg = ctx;
    tensor_t *input = ops[0];

    size_t batch = input->shape->dimensions[0];
    size_t channels = input->shape->dimensions[1];
    size_t inputLength = input->shape->dimensions[2];
    size_t outputLength = cfg->outputSize;

    const bfpQConfig_t *qC = input->quantization->qConfig;
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

/* OP(k) of each funnel op of this layer, for layerExecOpKey. Part of the
 * value contract: append, never renumber. */
enum {
    ADAPTIVE_AVGPOOL1D_OP_FORWARD = 0,
    ADAPTIVE_AVGPOOL1D_OP_PROP_LOSS = 1,
};

void adaptiveAvgPool1dForward(layer_t *layer, tensor_t *input, tensor_t *output,
                              const layerExec_t *exec) {
    rngKey_t opKey;
    adaptiveAvgPool1dConfig_t *cfg = layer->config->adaptiveAvgPool1d;

    size_t outDims[3];
    size_t outOrder[3];
    shape_t outExpected = {.dimensions = outDims, .orderOfDimensions = outOrder};
    adaptiveAvgPool1dCalcOutputShape(layer, input->shape, &outExpected);
    layerRequireDenseWire(output, &outExpected, "AdaptiveAvgPool1d forward", "output");

    if (cfg->forwardMath.type == ARITH_BFP) {
        const bfpQConfig_t *anchor = bfpWireAnchor(cfg->outputQ, "AdaptiveAvgPool1d forward");
        /* Stage FLOAT32 operands per-tensor at the produced-wire widths; borrow
         * BFP operands without reblocking. The stack template lives through
         * the executeOp call. */
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
                .key = layerExecOpKey(exec, RNG_TAG_FORWARD, ADAPTIVE_AVGPOOL1D_OP_FORWARD, &opKey),
                .mode = OUT_WRITE,
                .bfpStage = {input->quantization->type == FLOAT32 ? &stage : NULL, NULL, NULL},
            },
            output);
        return;
    }

    opKernelFn_t kernel = cfg->forwardMath.type == ARITH_SYM_INT32
                              ? adaptiveAvgPool1dForwardKernelSymInt32
                              : adaptiveAvgPool1dForwardKernel;
    executeOp(
        &(opSpec_t){
            .kernel = kernel,
            .ctx = cfg,
            .inputs = (tensor_t *[]){input},
            .nInputs = 1,
            .arithmetic = cfg->forwardMath,
            .key = layerExecOpKey(exec, RNG_TAG_FORWARD, ADAPTIVE_AVGPOOL1D_OP_FORWARD, &opKey),
            .mode = OUT_WRITE,
        },
        output);
}

static void adaptiveAvgPool1dBackwardFloat(tensor_t *lossGrad, tensor_t *propLoss) {
    size_t batch = lossGrad->shape->dimensions[0];
    size_t channels = lossGrad->shape->dimensions[1];
    size_t outputLength = lossGrad->shape->dimensions[2];
    size_t inputLength = propLoss->shape->dimensions[2];

    float const *gyArr = (float const *)lossGrad->data;
    float *gxArr = (float *)propLoss->data;

    // propLoss may hold stale bytes; clear before accumulating overlapping contributions.
    memset(gxArr, 0, batch * channels * inputLength * sizeof(float));

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

/* Variable window counts require half-away integer division of each loss-grad
 * mantissa; preserve the operand scale and scatter the quotient into every
 * window member. rawOut is uninitialized scratch: clear before accumulating,
 * since adjacent windows overlap when L % outputSize != 0. */
static void adaptiveAvgPool1dBackwardKernelSymInt32(tensor_t **ops, size_t n, tensor_t *rawOut,
                                                    tensor_t *auxOut, const void *ctx,
                                                    rngStream_t *rs) {
    (void)n;
    (void)auxOut;
    (void)ctx;
    tensor_t *lossGrad = ops[0];

    size_t batch = lossGrad->shape->dimensions[0];
    size_t channels = lossGrad->shape->dimensions[1];
    size_t outputLength = lossGrad->shape->dimensions[2];
    size_t inputLength = rawOut->shape->dimensions[2];

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

    storeCanonScale(&((symInt32QConfig_t *)rawOut->quantization->qConfig)->scale,
                    loadCanonScale(&((symInt32QConfig_t *)lossGrad->quantization->qConfig)->scale));
}

/* Divide each exact dequantized gradient by its window count before scattering
 * into the FLOAT32 raw; no int32 partials, so no sum-headroom guard. Clear the
 * raw first: adjacent windows overlap when L % outputSize != 0. */
static void adaptiveAvgPool1dBackwardKernelBfp(tensor_t **ops, size_t n, tensor_t *rawOut,
                                               tensor_t *auxOut, const void *ctx, rngStream_t *rs) {
    (void)n;
    (void)auxOut;
    (void)ctx;
    tensor_t *lossGrad = ops[0];

    size_t batch = lossGrad->shape->dimensions[0];
    size_t channels = lossGrad->shape->dimensions[1];
    size_t outputLength = lossGrad->shape->dimensions[2];
    size_t inputLength = rawOut->shape->dimensions[2];

    const bfpQConfig_t *qC = lossGrad->quantization->qConfig;
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

/* forwardInput is unused and may be NULL. */
void adaptiveAvgPool1dBackward(layer_t *layer, tensor_t *forwardInput, tensor_t *lossGrad,
                               tensor_t *propLoss, const layerExec_t *exec) {
    (void)forwardInput;
    rngKey_t opKey;
    adaptiveAvgPool1dConfig_t *cfg = layer->config->adaptiveAvgPool1d;

    size_t lossDims[3];
    size_t lossOrder[3];
    shape_t lossExpected = {.dimensions = lossDims, .orderOfDimensions = lossOrder};
    adaptiveAvgPool1dCalcOutputShape(layer, propLoss->shape, &lossExpected);
    layerRequireDenseWire(lossGrad, &lossExpected, "AdaptiveAvgPool1d backward", "loss");

    switch (cfg->propLossMath.type) {
    case ARITH_FLOAT32:
        /* This path casts lossGrad and propLoss directly to float. */
        bfpRequireNoBfpWire(lossGrad, "AdaptiveAvgPool1d backward (lossGrad)");
        bfpRequireNoBfpWire(propLoss, "AdaptiveAvgPool1d backward (propLoss)");
        adaptiveAvgPool1dBackwardFloat(lossGrad, propLoss);
        break;
    case ARITH_SYM_INT32:
        executeOp(
            &(opSpec_t){
                .kernel = adaptiveAvgPool1dBackwardKernelSymInt32,
                .ctx = cfg,
                .inputs = (tensor_t *[]){lossGrad},
                .nInputs = 1,
                .arithmetic = cfg->propLossMath,
                .key =
                    layerExecOpKey(exec, RNG_TAG_BACKWARD, ADAPTIVE_AVGPOOL1D_OP_PROP_LOSS, &opKey),
                .mode = OUT_WRITE,
            },
            propLoss);
        break;
    case ARITH_BFP: {
        const bfpQConfig_t *anchor = bfpWireAnchor(cfg->propLossQ, "AdaptiveAvgPool1d backward");
        bfpQConfig_t stage = {.exponents = NULL,
                              .numGroups = 1,
                              .groupSize = 0,
                              .roundingMode = cfg->propLossMath.roundingMode,
                              .mantissaBits = anchor->mantissaBits,
                              .exponentBits = anchor->exponentBits};
        executeOp(
            &(opSpec_t){
                .kernel = adaptiveAvgPool1dBackwardKernelBfp,
                .ctx = cfg,
                .inputs = (tensor_t *[]){lossGrad},
                .nInputs = 1,
                .arithmetic = cfg->propLossMath,
                .key =
                    layerExecOpKey(exec, RNG_TAG_BACKWARD, ADAPTIVE_AVGPOOL1D_OP_PROP_LOSS, &opKey),
                .mode = OUT_WRITE,
                .bfpStage = {lossGrad->quantization->type == FLOAT32 ? &stage : NULL, NULL, NULL},
            },
            propLoss);
        break;
    }
    }
}

void adaptiveAvgPool1dCalcOutputShape(layer_t *layer, shape_t *inputShape, shape_t *outputShape) {
    if (inputShape->numberOfDimensions != 3) {
        PRINT_ERROR("AdaptiveAvgPool1d expects 3D input [batch, channel, length], got %zuD",
                    inputShape->numberOfDimensions);
        exit(1);
    }
    if (inputShape->dimensions[2] == 0) {
        PRINT_ERROR("AdaptiveAvgPool1d: input length must be > 0");
        exit(1);
    }
    adaptiveAvgPool1dConfig_t *cfg = layer->config->adaptiveAvgPool1d;
    outputShape->numberOfDimensions = 3;
    outputShape->dimensions[0] = inputShape->dimensions[0]; // B
    outputShape->dimensions[1] = inputShape->dimensions[1]; // C
    outputShape->dimensions[2] = cfg->outputSize;
    setOrderOfDimsForNewTensor(inputShape->numberOfDimensions, outputShape->orderOfDimensions);
}
