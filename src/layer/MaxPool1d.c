#define SOURCE_FILE "ODT_MAX_POOL_1D"

#include <math.h>
#include <stdint.h>
#include <string.h>

#include "MaxPool1d.h"

#include "ArithmeticType.h"
#include "BfpKernelSupport.h"
#include "CheckedSize.h"
#include "Common.h"
#include "ExecuteOp.h"
#include "Layer.h"
#include "LayerExec.h"
#include "LayerWireGuards.h"
#include "PoolSymGuard.h"
#include "Quantization.h"
#include "ScaleCanon.h"
#include "SlidingWindow1d.h"
#include "StorageApi.h"
#include "Tensor.h"

void initMaxPool1dConfig(maxPool1dConfig_t *cfg, kernel_t *kernel, tensor_t *argmaxIndices,
                         quantization_t *forwardQ, quantization_t *propLossQ) {
    cfg->kernel = kernel;
    cfg->argmaxIndices = argmaxIndices;
    cfg->argmaxCapacity = calcNumberOfElementsByTensor(argmaxIndices);
    cfg->argmaxCapacityData = argmaxIndices->data;
    cfg->forwardMath = arithmeticFromQuantizationOrDefault(forwardQ);
    cfg->propLossMath = arithmeticFromQuantizationOrDefault(propLossQ);
    cfg->outputQ = forwardQ;
    cfg->propLossQ = propLossQ;
    cfg->ownsQuantizations = false;
}

/* auxOut is INT32 index state and is never converted. */
static void maxPool1dForwardKernel(tensor_t **ops, size_t n, tensor_t *rawOut, tensor_t *auxOut,
                                   const void *ctx, rngStream_t *rs) {
    (void)n;
    const maxPool1dConfig_t *cfg = ctx;
    tensor_t *input = ops[0];

    size_t batch = input->shape->dimensions[0];
    size_t channels = input->shape->dimensions[1];
    size_t inputLength = input->shape->dimensions[2];

    windowGeometry1d_t geom = windowGeometry1dCalc(inputLength, cfg->kernel);
    size_t outputLength = geom.outputLength;

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
                    // Empty dilated windows emit zero and argmax -1.
                    yArr[outIdx] = 0.0f;
                    argmaxArr[outIdx] = -1;
                    PRINT_ERROR("MaxPool1d: empty window at outPos=%zu — likely user misconfig",
                                outPos);
                }
            }
        }
    }
}

/* Mantissa select is value select (scale > 0 preserves order); the scale is
 * copied to the raw. Strict > keeps the first winner, but ties happen in the
 * QUANTIZED domain: inputs that quantize to the same mantissa tie even when
 * their float values differ. */
static void maxPool1dForwardKernelSymInt32(tensor_t **ops, size_t n, tensor_t *rawOut,
                                           tensor_t *auxOut, const void *ctx, rngStream_t *rs) {
    (void)n;
    const maxPool1dConfig_t *cfg = ctx;
    tensor_t *input = ops[0];

    size_t batch = input->shape->dimensions[0];
    size_t channels = input->shape->dimensions[1];
    size_t inputLength = input->shape->dimensions[2];

    windowGeometry1d_t geom = windowGeometry1dCalc(inputLength, cfg->kernel);
    size_t outputLength = geom.outputLength;

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
                    // Empty dilated windows emit zero and argmax -1.
                    yArr[outIdx] = 0;
                    argmaxArr[outIdx] = -1;
                    PRINT_ERROR("MaxPool1d: empty window at outPos=%zu — likely user misconfig",
                                outPos);
                }
            }
        }
    }

    storeCanonScale(&((symInt32QConfig_t *)rawOut->quantization->qConfig)->scale,
                    loadCanonScale(&((symInt32QConfig_t *)input->quantization->qConfig)->scale));
}

/* Integer-only forward (arithmetic-bfp.md section 13), ARITH_BFP arm. Raw BFP
 * mantissas are NOT comparable across groups -- a smaller code under a larger
 * group exponent can be the true maximum -- so candidates are compared as
 * exact (code, E) values (bfpValueGreater: no float, no rounding). The walk
 * is the FLOAT32 arm's: same windows, same order, first strict greater wins
 * (ties across groups included), so argmax (auxOut, INT32, never
 * funnel-converted; -1 empty-window sentinel) is exactly the float arm's on
 * the dequantized input. The winners then go to the BFP_RAW output aligned
 * to ONE exponent per raw target group, E_acc = (max E over the group's
 * NONZERO winners) - 15: a code shifted left by <= 15 stays below 2^31, and
 * a winner more than 15 binades below the group's top is rounded HALF_AWAY
 * (a zero aligns losslessly at any E, so it never sets E_acc). Pass 1
 * collects the per-group max in the raw exponent array; pass 2 re-reads each
 * winner's exponent through its argmax. No headroom guard: nothing is
 * summed. */
void maxPool1dForwardKernelBfpRaw(tensor_t **ops, size_t n, tensor_t *rawOut, tensor_t *auxOut,
                                  const void *ctx, struct rngStream *rs) {
    (void)n;
    (void)rs;
    const maxPool1dConfig_t *cfg = ctx;
    tensor_t *input = ops[0];

    if (input->quantization->type != BFP || rawOut->quantization->type != BFP_RAW) {
        PRINT_ERROR("MaxPool1d forward BFP: needs a BFP (unpacked scratch) input and a BFP_RAW "
                    "output (opSpec_t.bfpRawOut), got dtypes %d -> %d",
                    (int)input->quantization->type, (int)rawOut->quantization->type);
        exit(1);
    }
    size_t batch = input->shape->dimensions[0];
    size_t channels = input->shape->dimensions[1];
    size_t inputLength = input->shape->dimensions[2];

    windowGeometry1d_t geom = windowGeometry1dCalc(inputLength, cfg->kernel);
    size_t outputLength = geom.outputLength;

    const bfpQConfig_t *qC = input->quantization->qConfig;
    if (qC->mantissaBits > 16) {
        PRINT_ERROR("MaxPool1d forward BFP: mantissaBits %u > 16 breaks the exact compare",
                    (unsigned)qC->mantissaBits);
        exit(1);
    }
    const int32_t expBias = bfpExponentBias(qC);

    size_t outElems = batch * channels * outputLength;
    bfpRawQConfig_t *rawQC = rawOut->quantization->qConfig;
    size_t gsz = bfpRawRequireOutput(rawQC, outElems, "MaxPool1d forward BFP");
    int16_t *rawE = rawQC->exponents;
    if (outElems == 0) {
        return;
    }
    for (size_t t = 0; t < rawQC->numGroups; t++) {
        rawE[t] = INT16_MIN; /* pass 1: max STORED exponent of a nonzero winner */
    }

    int32_t const *xArr = (int32_t const *)input->data;
    int32_t *mant = (int32_t *)rawOut->data;
    int32_t *argmaxArr = (int32_t *)auxOut->data;
    bfpGroupRange_t qRange = {0};
    size_t targetGroup = 0;
    size_t targetEnd = gsz;

    for (size_t b = 0; b < batch; b++) {
        for (size_t c = 0; c < channels; c++) {
            for (size_t outPos = 0; outPos < outputLength; outPos++) {
                windowSlice1d_t slice = windowSlice1dAt(&geom, outPos);

                int32_t bestCode = 0;
                int bestE = 0;
                int32_t bestInputIdx = -1;
                for (size_t i = 0; i < slice.validCount; i++) {
                    size_t inputIdx = slice.firstValidInputIdx + i * geom.dilation;
                    size_t storageIdx = (b * channels + c) * inputLength + inputIdx;
                    size_t g = bfpGroupAt(&qRange, qC, storageIdx);
                    int32_t v = xArr[storageIdx];
                    int e = (int)qC->exponents[g];
                    if (bestInputIdx < 0 || bfpValueGreater(v, e, bestCode, bestE)) {
                        bestCode = v;
                        bestE = e;
                        bestInputIdx = (int32_t)inputIdx;
                    }
                }

                size_t outIdx = (b * channels + c) * outputLength + outPos;
                if (outIdx >= targetEnd) {
                    targetGroup++;
                    targetEnd += gsz;
                }
                if (slice.validCount > 0) {
                    mant[outIdx] = bestCode;
                    argmaxArr[outIdx] = bestInputIdx;
                    if (bestCode != 0 && bestE > rawE[targetGroup]) {
                        rawE[targetGroup] = (int16_t)bestE;
                    }
                } else {
                    // Empty dilated windows emit zero and argmax -1.
                    mant[outIdx] = 0;
                    argmaxArr[outIdx] = -1;
                    PRINT_ERROR("MaxPool1d: empty window at outPos=%zu — likely user misconfig",
                                outPos);
                }
            }
        }
    }

    /* E_acc per raw group, unbiased (0 for a group without a nonzero
     * winner: every mantissa there is 0). */
    for (size_t t = 0; t < rawQC->numGroups; t++) {
        rawE[t] = rawE[t] == INT16_MIN ? 0 : (int16_t)(rawE[t] - expBias - 15);
    }

    /* Pass 2: align each nonzero winner, k = E_acc - E_winner >= -15. */
    bfpGroupRange_t wRange = {0};
    targetGroup = 0;
    targetEnd = gsz;
    size_t outIdx = 0;
    for (size_t row = 0; row < batch * channels; row++) {
        for (size_t outPos = 0; outPos < outputLength; outPos++, outIdx++) {
            if (outIdx >= targetEnd) {
                targetGroup++;
                targetEnd += gsz;
            }
            int32_t a = argmaxArr[outIdx];
            if (a >= 0 && mant[outIdx] != 0) {
                size_t g = bfpGroupAt(&wRange, qC, row * inputLength + (size_t)a);
                int eWin = (int)qC->exponents[g] - expBias;
                mant[outIdx] = bfpAlignPartial(mant[outIdx], rawE[targetGroup] - eWin);
            }
        }
    }
}

/* Grow argmax storage for this batch before any arm runs, preserving capacity
 * on smaller calls. Check allocation-size overflow before reserving, and
 * reserve the new block before freeing the old one so a failed reservation
 * leaves the layer intact. Only dims[0] is rewritten, and only when B changes
 * (the forward gate rejects wrong channels or length afterwards).
 * Concurrency: see maxPool1dConfig_t. */
static void maxPoolEnsureArgmaxRows(maxPool1dConfig_t *cfg, const shape_t *outputShape) {
    tensor_t *argmax = cfg->argmaxIndices;
    size_t batch = outputShape->dimensions[0];
    size_t channels = outputShape->dimensions[1];
    size_t outputLength = outputShape->dimensions[2];

    if (cfg->argmaxCapacityData != argmax->data) {
        cfg->argmaxCapacityData = argmax->data;
        cfg->argmaxCapacity = calcNumberOfElementsByTensor(argmax);
    }

    size_t rowElements = 0;
    size_t needed = 0;
    size_t bytes = 0;
    if (!checkedMulSize(channels, outputLength, &rowElements) ||
        !checkedMulSize(batch, rowElements, &needed) ||
        !checkedMulSize(needed, sizeof(int32_t), &bytes)) {
        PRINT_ERROR("MaxPool1d forward: argmax size [%zu, %zu, %zu] overflows size_t", batch,
                    channels, outputLength);
        exit(1);
    }
    if (needed > cfg->argmaxCapacity) {
        uint8_t *grown = reserveMemory(bytes);
        if (grown == NULL) {
            PRINT_ERROR("MaxPool1d forward: reserving the [%zu, %zu, %zu] argmax (%zu bytes) "
                        "failed",
                        batch, channels, outputLength, bytes);
            exit(1);
        }
        freeReservedMemory(argmax->data);
        argmax->data = grown;
        cfg->argmaxCapacityData = grown;
        cfg->argmaxCapacity = needed;
    }
    if (argmax->shape->dimensions[0] != batch) {
        argmax->shape->dimensions[0] = batch;
    }
}

/* OP(k) of each funnel op of this layer, for layerExecOpKey. Part of the
 * value contract: append, never renumber. */
enum {
    MAXPOOL1D_OP_FORWARD = 0,
    MAXPOOL1D_OP_PROP_LOSS = 1,
};

void maxPool1dForward(layer_t *layer, tensor_t *input, tensor_t *output, const layerExec_t *exec) {
    rngKey_t opKey;
    maxPool1dConfig_t *cfg = layer->config->maxPool1d;

    size_t outDims[3];
    size_t outOrder[3];
    shape_t outExpected = {.dimensions = outDims, .orderOfDimensions = outOrder};
    maxPool1dCalcOutputShape(layer, input->shape, &outExpected);
    layerRequireDenseWire(output, &outExpected, "MaxPool1d forward", "output");
    maxPoolEnsureArgmaxRows(cfg, &outExpected);
    layerRequireDenseWire(cfg->argmaxIndices, &outExpected, "MaxPool1d forward", "argmaxIndices");

    if (cfg->forwardMath.type == ARITH_BFP) {
        const bfpQConfig_t *anchor = bfpWireAnchor(cfg->outputQ, "MaxPool1d forward");
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
                .kernel = maxPool1dForwardKernelBfpRaw,
                .ctx = cfg,
                .inputs = (tensor_t *[]){input},
                .nInputs = 1,
                .arithmetic = cfg->forwardMath,
                .key = layerExecOpKey(exec, RNG_TAG_FORWARD, MAXPOOL1D_OP_FORWARD, &opKey),
                .mode = OUT_WRITE,
                .auxOut = cfg->argmaxIndices,
                .bfpStage = {input->quantization->type == FLOAT32 ? &stage : NULL, NULL, NULL},
                .bfpRawOut = BFP_RAW_FORWARD, /* the integer forward raw */
            },
            output);
        return;
    }

    opKernelFn_t kernel = cfg->forwardMath.type == ARITH_SYM_INT32 ? maxPool1dForwardKernelSymInt32
                                                                   : maxPool1dForwardKernel;
    executeOp(
        &(opSpec_t){
            .kernel = kernel,
            .ctx = cfg,
            .inputs = (tensor_t *[]){input},
            .nInputs = 1,
            .arithmetic = cfg->forwardMath,
            .key = layerExecOpKey(exec, RNG_TAG_FORWARD, MAXPOOL1D_OP_FORWARD, &opKey),
            .mode = OUT_WRITE,
            .auxOut = cfg->argmaxIndices,
        },
        output);
}

static void maxPool1dBackwardFloat(maxPool1dConfig_t *cfg, tensor_t *lossGrad, tensor_t *propLoss) {
    size_t batch = lossGrad->shape->dimensions[0];
    size_t channels = lossGrad->shape->dimensions[1];
    size_t outputLength = lossGrad->shape->dimensions[2];
    size_t inputLength = propLoss->shape->dimensions[2];

    float const *gyArr = (float const *)lossGrad->data;
    int32_t const *argmaxArr = (int32_t const *)cfg->argmaxIndices->data;
    float *gxArr = (float *)propLoss->data;

    // propLoss may hold stale bytes; clear before accumulating overlapping contributions.
    memset(gxArr, 0, batch * channels * inputLength * sizeof(float));

    for (size_t b = 0; b < batch; b++) {
        for (size_t c = 0; c < channels; c++) {
            for (size_t outPos = 0; outPos < outputLength; outPos++) {
                size_t outIdx = (b * channels + c) * outputLength + outPos;
                int32_t inputIdx = argmaxArr[outIdx];
                if (inputIdx == -1) { // -1 marks an empty window
                    continue;
                }
                gxArr[(b * channels + c) * inputLength + (size_t)inputIdx] += gyArr[outIdx];
            }
        }
    }
}

/* Scatter loss-grad mantissas to the forward's argmax positions; the scale is
 * copied from lossGrad. argmax arrives via ctx; auxOut is unused. rawOut is
 * uninitialized scratch: clear before accumulating overlapping contributions. */
static void maxPool1dBackwardKernelSymInt32(tensor_t **ops, size_t n, tensor_t *rawOut,
                                            tensor_t *auxOut, const void *ctx, rngStream_t *rs) {
    (void)n;
    (void)auxOut;
    const maxPool1dConfig_t *cfg = ctx;
    tensor_t *lossGrad = ops[0];

    size_t batch = lossGrad->shape->dimensions[0];
    size_t channels = lossGrad->shape->dimensions[1];
    size_t outputLength = lossGrad->shape->dimensions[2];
    size_t inputLength = rawOut->shape->dimensions[2];

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
                if (inputIdx == -1) { // -1 marks an empty window
                    continue;
                }
                gxArr[(b * channels + c) * inputLength + (size_t)inputIdx] += gyArr[outIdx];
            }
        }
    }

    storeCanonScale(&((symInt32QConfig_t *)rawOut->quantization->qConfig)->scale,
                    loadCanonScale(&((symInt32QConfig_t *)lossGrad->quantization->qConfig)->scale));
}

/* An argmax inside the input but outside the span of its own window: a stale
 * argmax (another geometry's forward). Cold and out of line. */
static __attribute__((noinline, cold)) void maxPoolArgmaxOutsideWindow(int32_t idx, size_t outIdx) {
    PRINT_ERROR("MaxPool1d backward BFP: argmax index %d at output position %zu lies outside "
                "that window's span -- the recorded argmax does not belong to this geometry "
                "(stale forward?)",
                idx, outIdx);
    exit(1);
}

/* The scatter of the integer dx (BFP_WALK_NOINLINE: no VLA in this frame).
 * The FLOAT32 arm's loops, order and index math; each loss code enters its
 * dx cell aligned from its group's exponent to its row's E_acc. */
typedef struct maxPoolBwdWalk {
    const int32_t *gy;
    const bfpQConfig_t *qC;
    int32_t expBias;
    const int32_t *argmax;
    const int *eAccOfRow;
    int32_t *acc;
    size_t rows, outputLength, inputLength, stride, span, padLeft;
} maxPoolBwdWalk_t;

BFP_WALK_NOINLINE void maxPool1dBackwardBfpWalk(const maxPoolBwdWalk_t *p) {
    const bfpQConfig_t *qC = p->qC;
    bfpGroupRange_t qRange = {0};
    size_t outIdx = 0;
    for (size_t row = 0; row < p->rows; row++) {
        const int eAcc = p->eAccOfRow[row];
        int32_t *accRow = p->acc + row * p->inputLength;
        size_t winStart = 0; /* outPos * stride, in padded coordinates */
        for (size_t outPos = 0; outPos < p->outputLength;
             outPos++, outIdx++, winStart += p->stride) {
            int32_t inputIdx = p->argmax[outIdx];
            if (inputIdx == -1) { // -1 marks an empty window
                continue;
            }
            size_t padded = (size_t)inputIdx + p->padLeft;
            if (padded < winStart || padded - winStart >= p->span) {
                maxPoolArgmaxOutsideWindow(inputIdx, outIdx);
            }
            size_t g = bfpGroupAt(&qRange, qC, outIdx);
            accRow[inputIdx] +=
                bfpAlignPartial(p->gy[outIdx], eAcc - ((int)qC->exponents[g] - p->expBias));
        }
    }
}

/* Integer-only dx (arithmetic-bfp.md section 14), ARITH_BFP arm, ops =
 * {lossGrad} (unpacked-BFP scratch), the argmax via ctx (kernel-written by
 * the forward, never funnel-converted). Max is a SELECT, so its transpose
 * scatters the loss codes untouched to the recorded positions; a dx cell sums
 * the codes of every window whose argmax it is. Per dx row (one (batch,
 * channel)): E_acc = (max exponent over the loss row's NONZERO codes; 0 if
 * none) + g, g = bfpSumFoldGuardBits over C terms, C = (effK - 1) / stride +
 * 1 the windows whose span covers one cell -- the bound the argmax span check
 * keeps (a cell can win only windows covering it). Each code enters by
 * bfpAlignPartial (left exact; HALF_AWAY right for a code more than -g
 * binades below its row's top, -g = 23 at m 8, C 1); the raw is the data
 * rule of bfpRawEmitFromRows on the dx geometry; the funnel's re-block owns
 * the op's rounding (SR under SR). No float, no division in the walk. The
 * per-row table (one int per (batch, channel)) is the only stack beyond the
 * funnel's raw. */
void maxPool1dBackwardKernelBfpRaw(tensor_t **ops, size_t n, tensor_t *rawOut, tensor_t *auxOut,
                                   const void *ctx, rngStream_t *rs) {
    (void)n;
    (void)auxOut;
    (void)rs;
    const maxPool1dConfig_t *cfg = ctx;
    tensor_t *lossGrad = ops[0];

    if (lossGrad->quantization->type != BFP || rawOut->quantization->type != BFP_RAW) {
        PRINT_ERROR("MaxPool1d backward BFP: needs a BFP (unpacked scratch) lossGrad and a BFP_RAW "
                    "output (opSpec_t.bfpRawOut), got dtypes %d -> %d",
                    (int)lossGrad->quantization->type, (int)rawOut->quantization->type);
        exit(1);
    }
    size_t batch = lossGrad->shape->dimensions[0];
    size_t channels = lossGrad->shape->dimensions[1];
    size_t outputLength = lossGrad->shape->dimensions[2];
    size_t inputLength = rawOut->shape->dimensions[2];

    windowGeometry1d_t geom = windowGeometry1dCalc(inputLength, cfg->kernel);

    const bfpQConfig_t *qC = lossGrad->quantization->qConfig;
    const int32_t expBias = bfpExponentBias(qC);
    bfpRawQConfig_t *rawQC = rawOut->quantization->qConfig;
    size_t rows = batch * channels;
    size_t nOut = rows * inputLength;
    bfpRawRequireOutput(rawQC, nOut, "MaxPool1d backward BFP");
    if (nOut == 0) {
        return;
    }
    size_t span = geom.dilation * (geom.kernelSize - 1) + 1;
    int g = bfpSumFoldGuardBits((span - 1) / geom.stride + 1, qC->mantissaBits,
                                "MaxPool1d backward BFP");

    const int32_t *gyArr = (const int32_t *)lossGrad->data;
    int eAccOfRow[rows];
    bfpRowCursor_t cursor = {.group = 0, .groupEnd = qC->groupSize};
    for (size_t r = 0; r < rows; r++) {
        eAccOfRow[r] = bfpLiveTopOrZero(bfpRowMaxExponent(&cursor, qC, expBias, gyArr,
                                                          r * outputLength, outputLength)) +
                       g;
    }

    int32_t *acc = (int32_t *)rawOut->data;
    memset(acc, 0, nOut * sizeof(int32_t));
    maxPool1dBackwardBfpWalk(
        &(maxPoolBwdWalk_t){.gy = gyArr,
                            .qC = qC,
                            .expBias = expBias,
                            .argmax = (const int32_t *)cfg->argmaxIndices->data,
                            .eAccOfRow = eAccOfRow,
                            .acc = acc,
                            .rows = rows,
                            .outputLength = outputLength,
                            .inputLength = inputLength,
                            .stride = geom.stride,
                            .span = span,
                            .padLeft = geom.padLeft});
    bfpRawEmitFromRows(acc, nOut, eAccOfRow, inputLength, rawQC, "MaxPool1d backward BFP");
}

/* forwardInput is unused (the forward's argmax carries the routing) and may be
 * NULL. */
void maxPool1dBackward(layer_t *layer, tensor_t *forwardInput, tensor_t *lossGrad,
                       tensor_t *propLoss, const layerExec_t *exec) {
    (void)forwardInput;
    rngKey_t opKey;
    maxPool1dConfig_t *cfg = layer->config->maxPool1d;

    size_t lossDims[3];
    size_t lossOrder[3];
    shape_t lossExpected = {.dimensions = lossDims, .orderOfDimensions = lossOrder};
    maxPool1dCalcOutputShape(layer, propLoss->shape, &lossExpected);
    layerRequireDenseWire(lossGrad, &lossExpected, "MaxPool1d backward", "loss");
    layerRequireDenseWire(cfg->argmaxIndices, &lossExpected, "MaxPool1d backward", "argmaxIndices");

    switch (cfg->propLossMath.type) {
    case ARITH_FLOAT32:
        /* This path casts lossGrad and propLoss directly to float. */
        bfpRequireNoBfpWire(lossGrad, "MaxPool1d backward (lossGrad)");
        bfpRequireNoBfpWire(propLoss, "MaxPool1d backward (propLoss)");
        maxPool1dBackwardFloat(cfg, lossGrad, propLoss);
        break;
    case ARITH_SYM_INT32:
        executeOp(
            &(opSpec_t){
                .kernel = maxPool1dBackwardKernelSymInt32,
                .ctx = cfg,
                .inputs = (tensor_t *[]){lossGrad},
                .nInputs = 1,
                .arithmetic = cfg->propLossMath,
                .key = layerExecOpKey(exec, RNG_TAG_BACKWARD, MAXPOOL1D_OP_PROP_LOSS, &opKey),
                .mode = OUT_WRITE,
            },
            propLoss);
        break;
    case ARITH_BFP: {
        const bfpQConfig_t *anchor = bfpWireAnchor(cfg->propLossQ, "MaxPool1d backward");
        bfpQConfig_t stage = {.exponents = NULL,
                              .numGroups = 1,
                              .groupSize = 0,
                              .roundingMode = cfg->propLossMath.roundingMode,
                              .mantissaBits = anchor->mantissaBits,
                              .exponentBits = anchor->exponentBits};
        executeOp(
            &(opSpec_t){
                .kernel = maxPool1dBackwardKernelBfpRaw,
                .ctx = cfg,
                .inputs = (tensor_t *[]){lossGrad},
                .nInputs = 1,
                .arithmetic = cfg->propLossMath,
                .key = layerExecOpKey(exec, RNG_TAG_BACKWARD, MAXPOOL1D_OP_PROP_LOSS, &opKey),
                .mode = OUT_WRITE,
                .bfpStage = {lossGrad->quantization->type == FLOAT32 ? &stage : NULL, NULL, NULL},
                .bfpRawOut = BFP_RAW_BACKWARD_DX, /* the integer dx raw */
            },
            propLoss);
        break;
    }
    }
}

void maxPool1dCalcOutputShape(layer_t *layer, shape_t *inputShape, shape_t *outputShape) {
    if (inputShape->numberOfDimensions != 3) {
        PRINT_ERROR("MaxPool1d expects 3D input [batch, channel, length], got %zuD",
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
