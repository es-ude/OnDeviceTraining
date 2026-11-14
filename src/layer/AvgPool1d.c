#define SOURCE_FILE "ODT_AVG_POOL_1D"

#include <math.h>
#include <string.h>

#include "AvgPool1d.h"

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
#include "SlidingWindow1d.h"
#include "Tensor.h"

void initAvgPool1dConfig(avgPool1dConfig_t *cfg, kernel_t *kernel, quantization_t *forwardQ,
                         quantization_t *propLossQ) {
    cfg->kernel = kernel;
    cfg->forwardMath = arithmeticFromQuantizationOrDefault(forwardQ);
    cfg->propLossMath = arithmeticFromQuantizationOrDefault(propLossQ);
    cfg->outputQ = forwardQ;
    cfg->propLossQ = propLossQ;
    cfg->ownsQuantizations = false;
}

static void avgPool1dForwardKernel(tensor_t **ops, size_t n, tensor_t *rawOut, tensor_t *auxOut,
                                   const void *ctx, rngStream_t *rs) {
    (void)n;
    (void)auxOut;
    const avgPool1dConfig_t *cfg = ctx;
    tensor_t *input = ops[0];

    size_t batch = input->shape->dimensions[0];
    size_t channels = input->shape->dimensions[1];
    size_t inputLength = input->shape->dimensions[2];

    windowGeometry1d_t geom = windowGeometry1dCalc(inputLength, cfg->kernel);
    size_t outputLength = geom.outputLength;

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

/* Exact int32 window sum; fold 1/K into the scale (s_out = s_in / K) so the
 * mantissas carry no rounding. The divisor is K including padding: padded
 * positions add 0 to the sum. At most K terms per sum. */
static void avgPool1dForwardKernelSymInt32(tensor_t **ops, size_t n, tensor_t *rawOut,
                                           tensor_t *auxOut, const void *ctx, rngStream_t *rs) {
    (void)n;
    (void)auxOut;
    const avgPool1dConfig_t *cfg = ctx;
    tensor_t *input = ops[0];

    size_t batch = input->shape->dimensions[0];
    size_t channels = input->shape->dimensions[1];
    size_t inputLength = input->shape->dimensions[2];

    windowGeometry1d_t geom = windowGeometry1dCalc(inputLength, cfg->kernel);
    size_t outputLength = geom.outputLength;

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

    /* count_include_pad=true: divisor = K always; s / K folded once (NEAREST) */
    storeCanonScale(
        &((symInt32QConfig_t *)rawOut->quantization->qConfig)->scale,
        canonRatioScale(loadCanonScale(&((symInt32QConfig_t *)input->quantization->qConfig)->scale),
                        1u, (uint32_t)cfg->kernel->size));
}

int avgPool1dBfpLog2Divisor(size_t kSize, const char *what) {
    if (kSize == 0 || (kSize & (kSize - 1)) != 0) {
        PRINT_ERROR("%s: BFP math needs a power-of-two kernel size (the mean is an exponent "
                    "shift; count_include_pad keeps the divisor K at every output), got K = %zu",
                    what, kSize);
        exit(1);
    }
    int log2K = 0;
    while (((size_t)1 << log2K) < kSize) {
        log2K++;
    }
    return log2K;
}

/* Integer-only forward (arithmetic-bfp.md section 13), ARITH_BFP arm:
 * operands arrive as the funnel's unpacked-BFP scratch (exponents BORROWED
 * for a BFP-stored input, funnel-staged for a FLOAT32 one). The walk is the
 * FLOAT32 arm's; the window sum keeps one int32 partial per same-group
 * visited segment (per-ELEMENT group lookup: dilation and stride skip
 * storage indices) and folds it into ONE int32 accumulator per output in
 * units of 2^E_acc, E_acc shared by a raw target group: the max input-row
 * exponent over the rows its outputs read, plus the sum guard
 * (bfpSumFoldGuardBits, K terms). Alignment is bfpAlignPartial (left exact,
 * right HALF_AWAY, never SR -- the re-block owns the op's rounding). The
 * mean folds -log2 K into the raw exponent (exact; HAR's K = 32): BFP math
 * takes a power-of-two K only (avgPool1dBfpLog2Divisor fails fast
 * otherwise). count_include_pad=true falls out free: padded positions are
 * not visited and the divisor stays K. bfpValidateSumHeadroom bounds each
 * partial. */
void avgPool1dForwardKernelBfpRaw(tensor_t **ops, size_t n, tensor_t *rawOut, tensor_t *auxOut,
                                  const void *ctx, struct rngStream *rs) {
    (void)n;
    (void)auxOut;
    (void)rs;
    const avgPool1dConfig_t *cfg = ctx;
    tensor_t *input = ops[0];

    if (input->quantization->type != BFP || rawOut->quantization->type != BFP_RAW) {
        PRINT_ERROR("AvgPool1d forward BFP: needs a BFP (unpacked scratch) input and a BFP_RAW "
                    "output (opSpec_t.bfpRawOut), got dtypes %d -> %d",
                    (int)input->quantization->type, (int)rawOut->quantization->type);
        exit(1);
    }
    const int log2K = avgPool1dBfpLog2Divisor(cfg->kernel->size, "AvgPool1d forward BFP");
    size_t batch = input->shape->dimensions[0];
    size_t channels = input->shape->dimensions[1];
    size_t inputLength = input->shape->dimensions[2];

    windowGeometry1d_t geom = windowGeometry1dCalc(inputLength, cfg->kernel);
    size_t outputLength = geom.outputLength;

    const bfpQConfig_t *qC = input->quantization->qConfig;
    bfpValidateSumHeadroom(qC, cfg->kernel->size, "AvgPool1d forward BFP");
    const int32_t expBias = bfpExponentBias(qC);

    size_t outElems = batch * channels * outputLength;
    bfpRawQConfig_t *rawQC = rawOut->quantization->qConfig;
    size_t gsz = bfpRawRequireOutput(rawQC, outElems, "AvgPool1d forward BFP");
    int16_t *rawE = rawQC->exponents;
    if (outElems == 0) {
        return;
    }

    const size_t kSize = cfg->kernel->size;
    int guard = bfpSumFoldGuardBits(kSize, qC->mantissaBits, "AvgPool1d forward BFP");

    int32_t const *xArr = (int32_t const *)input->data;
    /* Pre-pass (one read per input code, cursors, no division): per raw group
     * the max exponent over the groups holding a nonzero code of the input
     * rows its outputs read (row r of the output reads input row r), plus g
     * (a group without one: the zero state's top 0); the raw exponent also
     * carries -log2 K. */
    {
        bfpRowCursor_t cursor = {.group = 0, .groupEnd = qC->groupSize};
        size_t row = 0;
        size_t rowEnd = outputLength;
        int rowTop = bfpRowMaxExponent(&cursor, qC, expBias, xArr, 0, inputLength);
        for (size_t t = 0; t < rawQC->numGroups; t++) {
            size_t tStart = t * gsz;
            size_t tEnd = tStart + gsz;
            while (tStart >= rowEnd) {
                row++;
                rowEnd += outputLength;
                rowTop =
                    bfpRowMaxExponent(&cursor, qC, expBias, xArr, row * inputLength, inputLength);
            }
            int eTop = rowTop;
            while (tEnd > rowEnd) {
                row++;
                rowEnd += outputLength;
                rowTop =
                    bfpRowMaxExponent(&cursor, qC, expBias, xArr, row * inputLength, inputLength);
                if (rowTop > eTop) {
                    eTop = rowTop;
                }
            }
            rawE[t] = (int16_t)(bfpLiveTopOrZero(eTop) + guard - log2K);
        }
    }

    int32_t *mant = (int32_t *)rawOut->data;
    bfpGroupRange_t qRange = {0};
    size_t targetGroup = 0;
    size_t targetEnd = gsz;
    int eAcc = rawE[0] + log2K;

    for (size_t b = 0; b < batch; b++) {
        for (size_t c = 0; c < channels; c++) {
            for (size_t outPos = 0; outPos < outputLength; outPos++) {
                windowSlice1d_t slice = windowSlice1dAt(&geom, outPos);
                size_t outIdx = (b * channels + c) * outputLength + outPos;
                if (outIdx >= targetEnd) {
                    targetGroup++;
                    targetEnd += gsz;
                    eAcc = rawE[targetGroup] + log2K;
                }

                int32_t acc = 0;
                int32_t partial = 0;
                size_t currentGroup = 0;
                for (size_t i = 0; i < slice.validCount; i++) {
                    size_t inputIdx = slice.firstValidInputIdx + i * geom.dilation;
                    size_t storageIdx = (b * channels + c) * inputLength + inputIdx;
                    size_t g = bfpGroupAt(&qRange, qC, storageIdx);
                    if (i == 0) {
                        currentGroup = g;
                    } else if (g != currentGroup) {
                        acc += bfpAlignPartial(partial,
                                               eAcc - ((int)qC->exponents[currentGroup] - expBias));
                        partial = 0;
                        currentGroup = g;
                    }
                    partial += xArr[storageIdx];
                }
                if (slice.validCount > 0) {
                    acc += bfpAlignPartial(partial,
                                           eAcc - ((int)qC->exponents[currentGroup] - expBias));
                }

                mant[outIdx] = acc;
            }
        }
    }
}

/* OP(k) of each funnel op of this layer, for layerExecOpKey. Part of the
 * value contract: append, never renumber. */
enum {
    AVGPOOL1D_OP_FORWARD = 0,
    AVGPOOL1D_OP_PROP_LOSS = 1,
};

void avgPool1dForward(layer_t *layer, tensor_t *input, tensor_t *output, const layerExec_t *exec) {
    rngKey_t opKey;
    avgPool1dConfig_t *cfg = layer->config->avgPool1d;

    size_t outDims[3];
    size_t outOrder[3];
    shape_t outExpected = {.dimensions = outDims, .orderOfDimensions = outOrder};
    avgPool1dCalcOutputShape(layer, input->shape, &outExpected);
    layerRequireDenseWire(output, &outExpected, "AvgPool1d forward", "output");

    if (cfg->forwardMath.type == ARITH_BFP) {
        const bfpQConfig_t *anchor = bfpWireAnchor(cfg->outputQ, "AvgPool1d forward");
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
                .kernel = avgPool1dForwardKernelBfpRaw,
                .ctx = cfg,
                .inputs = (tensor_t *[]){input},
                .nInputs = 1,
                .arithmetic = cfg->forwardMath,
                .key = layerExecOpKey(exec, RNG_TAG_FORWARD, AVGPOOL1D_OP_FORWARD, &opKey),
                .mode = OUT_WRITE,
                .bfpStage = {input->quantization->type == FLOAT32 ? &stage : NULL, NULL, NULL},
                .bfpRawOut = BFP_RAW_FORWARD, /* the integer forward raw */
            },
            output);
        return;
    }

    opKernelFn_t kernel = cfg->forwardMath.type == ARITH_SYM_INT32 ? avgPool1dForwardKernelSymInt32
                                                                   : avgPool1dForwardKernel;
    executeOp(
        &(opSpec_t){
            .kernel = kernel,
            .ctx = cfg,
            .inputs = (tensor_t *[]){input},
            .nInputs = 1,
            .arithmetic = cfg->forwardMath,
            .key = layerExecOpKey(exec, RNG_TAG_FORWARD, AVGPOOL1D_OP_FORWARD, &opKey),
            .mode = OUT_WRITE,
        },
        output);
}

static void avgPool1dBackwardFloat(const avgPool1dConfig_t *cfg, tensor_t *lossGrad,
                                   tensor_t *propLoss) {
    size_t batch = lossGrad->shape->dimensions[0];
    size_t channels = lossGrad->shape->dimensions[1];
    size_t outputLength = lossGrad->shape->dimensions[2];
    size_t inputLength = propLoss->shape->dimensions[2];

    windowGeometry1d_t geom = windowGeometry1dCalc(inputLength, cfg->kernel);

    float const *gyArr = (float const *)lossGrad->data;
    float *gxArr = (float *)propLoss->data;
    float divisor = (float)cfg->kernel->size;

    // propLoss may hold stale bytes; clear before accumulating overlapping contributions.
    memset(gxArr, 0, batch * channels * inputLength * sizeof(float));

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

/* Scatter loss-grad mantissas into every valid window member and fold 1/K into
 * the scale (s_dx = s_gy / K) to avoid mantissa rounding. rawOut is
 * uninitialized scratch: clear before accumulating overlapping contributions. */
static void avgPool1dBackwardKernelSymInt32(tensor_t **ops, size_t n, tensor_t *rawOut,
                                            tensor_t *auxOut, const void *ctx, rngStream_t *rs) {
    (void)n;
    (void)auxOut;
    const avgPool1dConfig_t *cfg = ctx;
    tensor_t *lossGrad = ops[0];

    size_t batch = lossGrad->shape->dimensions[0];
    size_t channels = lossGrad->shape->dimensions[1];
    size_t outputLength = lossGrad->shape->dimensions[2];
    size_t inputLength = rawOut->shape->dimensions[2];

    windowGeometry1d_t geom = windowGeometry1dCalc(inputLength, cfg->kernel);

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

    storeCanonScale(
        &((symInt32QConfig_t *)rawOut->quantization->qConfig)->scale,
        canonRatioScale(
            loadCanonScale(&((symInt32QConfig_t *)lossGrad->quantization->qConfig)->scale), 1u,
            (uint32_t)cfg->kernel->size));
}

/* The scatter of the integer dx (BFP_WALK_NOINLINE: no VLA in this frame).
 * The FLOAT32 arm's loops, order and index math: one contribution per
 * output, added to every valid member of its window. */
typedef struct avgPoolBwdWalk {
    const int32_t *gy;
    const bfpQConfig_t *qC;
    int32_t expBias;
    const int *eAccOfRow; /* the row's VALUE exponent (-log2 K folded in) */
    int32_t *acc;
    const windowGeometry1d_t *geom;
    size_t rows;
    int log2K;
} avgPoolBwdWalk_t;

BFP_WALK_NOINLINE void avgPool1dBackwardBfpWalk(const avgPoolBwdWalk_t *p) {
    const bfpQConfig_t *qC = p->qC;
    const windowGeometry1d_t *geom = p->geom;
    const size_t outputLength = geom->outputLength;
    const size_t inputLength = geom->inputLength;
    const size_t dilation = geom->dilation;
    bfpGroupRange_t qRange = {0};
    size_t outIdx = 0;
    for (size_t row = 0; row < p->rows; row++) {
        /* the alignment grid: the value exponent plus log2 K back */
        const int eAlign = p->eAccOfRow[row] + p->log2K;
        int32_t *accRow = p->acc + row * inputLength;
        for (size_t outPos = 0; outPos < outputLength; outPos++, outIdx++) {
            windowSlice1d_t slice = windowSlice1dAt(geom, outPos);
            size_t g = bfpGroupAt(&qRange, qC, outIdx);
            int32_t contribution =
                bfpAlignPartial(p->gy[outIdx], eAlign - ((int)qC->exponents[g] - p->expBias));

            for (size_t i = 0; i < slice.validCount; i++) {
                accRow[slice.firstValidInputIdx + i * dilation] += contribution;
            }
        }
    }
}

/* Integer-only dx (arithmetic-bfp.md section 14), ARITH_BFP arm, ops =
 * {lossGrad} (unpacked-BFP scratch). The FLOAT32 arm adds gy / K to every
 * valid member of the window (count_include_pad: padded positions are not
 * visited, the divisor stays K). Per dx row (one (batch, channel)): the code
 * grid E = (max exponent over the loss row's NONZERO codes; 0 if none) + g,
 * g = bfpSumFoldGuardBits over C terms, C = (effK - 1) / stride + 1 the
 * windows covering one cell. A loss code enters by bfpAlignPartial (left
 * exact; HALF_AWAY right for a code more than -g binades below its row's
 * top). K is a power of two (avgPool1dBfpLog2Divisor fails fast
 * otherwise, ruling R1): -log2 K folds into the row's exponent, exact (HAR's
 * K = 32). Rounding onto the wire grid (SR under SR) is the funnel's
 * re-block of the raw, the data rule of bfpRawEmitFromRows on the dx
 * geometry. No float, no division per output. The per-row table (one int
 * per (batch, channel)) is the only stack beyond the funnel's raw. */
void avgPool1dBackwardKernelBfpRaw(tensor_t **ops, size_t n, tensor_t *rawOut, tensor_t *auxOut,
                                   const void *ctx, rngStream_t *rs) {
    (void)n;
    (void)auxOut;
    (void)rs;
    const avgPool1dConfig_t *cfg = ctx;
    tensor_t *lossGrad = ops[0];

    if (lossGrad->quantization->type != BFP || rawOut->quantization->type != BFP_RAW) {
        PRINT_ERROR("AvgPool1d backward BFP: needs a BFP (unpacked scratch) lossGrad and a BFP_RAW "
                    "output (opSpec_t.bfpRawOut), got dtypes %d -> %d",
                    (int)lossGrad->quantization->type, (int)rawOut->quantization->type);
        exit(1);
    }
    const int log2K = avgPool1dBfpLog2Divisor(cfg->kernel->size, "AvgPool1d backward BFP");
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
    bfpRawRequireOutput(rawQC, nOut, "AvgPool1d backward BFP");
    if (nOut == 0) {
        return;
    }
    size_t span = geom.dilation * (geom.kernelSize - 1) + 1;
    int g = bfpSumFoldGuardBits((span - 1) / geom.stride + 1, qC->mantissaBits,
                                "AvgPool1d backward BFP");

    const int32_t *gyArr = (const int32_t *)lossGrad->data;
    int eAccOfRow[rows];
    bfpRowCursor_t cursor = {.group = 0, .groupEnd = qC->groupSize};
    for (size_t r = 0; r < rows; r++) {
        eAccOfRow[r] = bfpLiveTopOrZero(bfpRowMaxExponent(&cursor, qC, expBias, gyArr,
                                                          r * outputLength, outputLength)) +
                       g - log2K;
    }

    int32_t *acc = (int32_t *)rawOut->data;
    memset(acc, 0, nOut * sizeof(int32_t));
    avgPool1dBackwardBfpWalk(&(avgPoolBwdWalk_t){.gy = gyArr,
                                                 .qC = qC,
                                                 .expBias = expBias,
                                                 .eAccOfRow = eAccOfRow,
                                                 .acc = acc,
                                                 .geom = &geom,
                                                 .rows = rows,
                                                 .log2K = log2K});
    bfpRawEmitFromRows(acc, nOut, eAccOfRow, inputLength, rawQC, "AvgPool1d backward BFP");
}

/* forwardInput is unused and may be NULL. */
void avgPool1dBackward(layer_t *layer, tensor_t *forwardInput, tensor_t *lossGrad,
                       tensor_t *propLoss, const layerExec_t *exec) {
    (void)forwardInput;
    rngKey_t opKey;
    avgPool1dConfig_t *cfg = layer->config->avgPool1d;

    size_t lossDims[3];
    size_t lossOrder[3];
    shape_t lossExpected = {.dimensions = lossDims, .orderOfDimensions = lossOrder};
    avgPool1dCalcOutputShape(layer, propLoss->shape, &lossExpected);
    layerRequireDenseWire(lossGrad, &lossExpected, "AvgPool1d backward", "loss");

    switch (cfg->propLossMath.type) {
    case ARITH_FLOAT32:
        /* This path casts lossGrad and propLoss directly to float. */
        bfpRequireNoBfpWire(lossGrad, "AvgPool1d backward (lossGrad)");
        bfpRequireNoBfpWire(propLoss, "AvgPool1d backward (propLoss)");
        avgPool1dBackwardFloat(cfg, lossGrad, propLoss);
        break;
    case ARITH_SYM_INT32:
        executeOp(
            &(opSpec_t){
                .kernel = avgPool1dBackwardKernelSymInt32,
                .ctx = cfg,
                .inputs = (tensor_t *[]){lossGrad},
                .nInputs = 1,
                .arithmetic = cfg->propLossMath,
                .key = layerExecOpKey(exec, RNG_TAG_BACKWARD, AVGPOOL1D_OP_PROP_LOSS, &opKey),
                .mode = OUT_WRITE,
            },
            propLoss);
        break;
    case ARITH_BFP: {
        const bfpQConfig_t *anchor = bfpWireAnchor(cfg->propLossQ, "AvgPool1d backward");
        bfpQConfig_t stage = {.exponents = NULL,
                              .numGroups = 1,
                              .groupSize = 0,
                              .roundingMode = cfg->propLossMath.roundingMode,
                              .mantissaBits = anchor->mantissaBits,
                              .exponentBits = anchor->exponentBits};
        executeOp(
            &(opSpec_t){
                .kernel = avgPool1dBackwardKernelBfpRaw,
                .ctx = cfg,
                .inputs = (tensor_t *[]){lossGrad},
                .nInputs = 1,
                .arithmetic = cfg->propLossMath,
                .key = layerExecOpKey(exec, RNG_TAG_BACKWARD, AVGPOOL1D_OP_PROP_LOSS, &opKey),
                .mode = OUT_WRITE,
                .bfpStage = {lossGrad->quantization->type == FLOAT32 ? &stage : NULL, NULL, NULL},
                .bfpRawOut = BFP_RAW_BACKWARD_DX, /* the integer dx raw */
            },
            propLoss);
        break;
    }
    }
}

void avgPool1dCalcOutputShape(layer_t *layer, shape_t *inputShape, shape_t *outputShape) {
    if (inputShape->numberOfDimensions != 3) {
        PRINT_ERROR("AvgPool1d expects 3D input [batch, channel, length], got %zuD",
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
