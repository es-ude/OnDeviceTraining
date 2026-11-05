#define SOURCE_FILE "UNIT_TEST_AVG_POOL_1D"

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "AvgPool1d.h"
#include "BfpKernelSupport.h"
#include "DeathTest.h"
#include "KeyedDrawCheck.h"
#include "Layer.h"
#include "Quantization.h"
#include "QuantizationApi.h"
#include "StorageApi.h"
#include "TensorApi.h"
#include "TensorConversion.h"
#include "expected_avg_pool_1d.h"
#include "unity.h"

#include "BfpRawFixture.h"
#include "BfpScatterRef.h"
#include "LayerExec.h"
#include "RNG.h"
#include "SlidingWindow1d.h"

typedef struct avgPool1dRunResult {
    layer_t *layer;
    tensor_t *input;
    tensor_t *output;
    quantization_t *q;
} avgPool1dRunResult_t;

static tensor_t *makeFloatTensor(size_t const *dims, size_t numDims, float const *data) {
    size_t *ownedDims = reserveMemory(numDims * sizeof(size_t));
    memcpy(ownedDims, dims, numDims * sizeof(size_t));
    size_t *order = reserveMemory(numDims * sizeof(size_t));
    setOrderOfDimsForNewTensor(numDims, order);
    shape_t *shape = reserveMemory(sizeof(shape_t));
    setShape(shape, ownedDims, numDims, order);
    tensor_t *t = initTensor(shape, quantizationInitFloat(), NULL);
    if (data != NULL) {
        tensorFillFromFloatBuffer(t, data, calcNumberOfElementsByTensor(t));
    }
    return t;
}

static avgPool1dRunResult_t avgPool1dBuild(float const *inputData, size_t const *inputDims,
                                           size_t kSize, paddingType_t padding, size_t dilation,
                                           size_t stride, float *outputBuf,
                                           size_t const *outputDims) {
    static kernel_t kernelStore;
    static avgPool1dConfig_t cfgStore;
    static layer_t layerStore;
    static layerConfig_t lcStore;

    initKernel(&kernelStore, kSize, padding, dilation, stride);

    quantization_t *q = quantizationInitFloat();
    initAvgPool1dConfig(&cfgStore, &kernelStore, q, q);

    layerStore.type = AVGPOOL1D;
    lcStore.avgPool1d = &cfgStore;
    layerStore.config = &lcStore;

    avgPool1dRunResult_t r = {0};
    r.layer = &layerStore;
    r.input = makeFloatTensor(inputDims, 3, inputData);
    r.output = makeFloatTensor(outputDims, 3, NULL);
    (void)outputBuf;
    r.q = q;
    return r;
}

void testAvgPool1dForwardBasic(void) {
    size_t inputDims[] = {1, 1, 4};
    size_t outputDims[] = {1, 1, 3};
    float outputData[1 * 1 * 3] = {0};

    avgPool1dRunResult_t r =
        avgPool1dBuild(input_avgPool1d_basic, inputDims, 2, VALID, 1, 1, outputData, outputDims);

    avgPool1dForward(r.layer, r.input, r.output, NULL);

    for (size_t i = 0; i < expectedForward_avgPool1d_basic_len; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-5f, expectedForward_avgPool1d_basic[i],
                                 ((float *)r.output->data)[i]);
    }
}

void testAvgPool1dBackwardBasic(void) {
    size_t inputDims[] = {1, 1, 4};
    size_t outputDims[] = {1, 1, 3};
    float outputData[1 * 1 * 3] = {0};

    avgPool1dRunResult_t r =
        avgPool1dBuild(input_avgPool1d_basic, inputDims, 2, VALID, 1, 1, outputData, outputDims);

    avgPool1dForward(r.layer, r.input, r.output, NULL);

    float lossGradData[1 * 1 * 3];
    for (size_t i = 0; i < 3; i++) {
        lossGradData[i] = 1.0f;
    }
    tensor_t *lossGrad = makeFloatTensor(outputDims, 3, lossGradData);

    tensor_t *propLoss = makeFloatTensor(inputDims, 3, NULL);

    avgPool1dBackward(r.layer, r.input, lossGrad, propLoss, NULL);

    for (size_t i = 0; i < expectedPropLoss_avgPool1d_basic_len; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-5f, expectedPropLoss_avgPool1d_basic[i],
                                 ((float *)propLoss->data)[i]);
    }
}

void testAvgPool1dMultiChannel(void) {
    size_t inputDims[] = {1, 3, 5};
    size_t outputDims[] = {1, 3, 4};
    float outputData[1 * 3 * 4] = {0};

    avgPool1dRunResult_t r = avgPool1dBuild(input_avgPool1d_multiChannel, inputDims, 2, VALID, 1, 1,
                                            outputData, outputDims);

    avgPool1dForward(r.layer, r.input, r.output, NULL);
    for (size_t i = 0; i < expectedForward_avgPool1d_multiChannel_len; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-5f, expectedForward_avgPool1d_multiChannel[i],
                                 ((float *)r.output->data)[i]);
    }

    float lossGradData[1 * 3 * 4];
    for (size_t i = 0; i < 12; i++) {
        lossGradData[i] = 1.0f;
    }
    tensor_t *lossGrad = makeFloatTensor(outputDims, 3, lossGradData);

    tensor_t *propLoss = makeFloatTensor(inputDims, 3, NULL);

    avgPool1dBackward(r.layer, r.input, lossGrad, propLoss, NULL);
    for (size_t i = 0; i < expectedPropLoss_avgPool1d_multiChannel_len; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-5f, expectedPropLoss_avgPool1d_multiChannel[i],
                                 ((float *)propLoss->data)[i]);
    }
}

void testAvgPool1dMultiBatch(void) {
    size_t inputDims[] = {4, 2, 4};
    size_t outputDims[] = {4, 2, 3};
    float outputData[4 * 2 * 3] = {0};

    avgPool1dRunResult_t r = avgPool1dBuild(input_avgPool1d_multiBatch, inputDims, 2, VALID, 1, 1,
                                            outputData, outputDims);

    avgPool1dForward(r.layer, r.input, r.output, NULL);
    for (size_t i = 0; i < expectedForward_avgPool1d_multiBatch_len; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-5f, expectedForward_avgPool1d_multiBatch[i],
                                 ((float *)r.output->data)[i]);
    }

    float lossGradData[4 * 2 * 3];
    for (size_t i = 0; i < 24; i++) {
        lossGradData[i] = 1.0f;
    }
    tensor_t *lossGrad = makeFloatTensor(outputDims, 3, lossGradData);

    tensor_t *propLoss = makeFloatTensor(inputDims, 3, NULL);

    avgPool1dBackward(r.layer, r.input, lossGrad, propLoss, NULL);
    for (size_t i = 0; i < expectedPropLoss_avgPool1d_multiBatch_len; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-5f, expectedPropLoss_avgPool1d_multiBatch[i],
                                 ((float *)propLoss->data)[i]);
    }
}

void testAvgPool1dWithStrideAndDilation(void) {
    size_t inputDims[] = {1, 1, 9};
    size_t outputDims[] = {1, 1, 3};
    float outputData[1 * 1 * 3] = {0};

    avgPool1dRunResult_t r = avgPool1dBuild(input_avgPool1d_withStrideAndDilation, inputDims, 2,
                                            VALID, 2, 3, outputData, outputDims);

    avgPool1dForward(r.layer, r.input, r.output, NULL);
    for (size_t i = 0; i < expectedForward_avgPool1d_withStrideAndDilation_len; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-5f, expectedForward_avgPool1d_withStrideAndDilation[i],
                                 ((float *)r.output->data)[i]);
    }

    tensor_t *lossGrad = makeFloatTensor(outputDims, 3, lossGrad_avgPool1d_withStrideAndDilation);

    tensor_t *propLoss = makeFloatTensor(inputDims, 3, NULL);

    avgPool1dBackward(r.layer, r.input, lossGrad, propLoss, NULL);
    for (size_t i = 0; i < expectedPropLoss_avgPool1d_withStrideAndDilation_len; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-5f, expectedPropLoss_avgPool1d_withStrideAndDilation[i],
                                 ((float *)propLoss->data)[i]);
    }
}

void testAvgPool1dWithSamePadding(void) {
    size_t inputDims[] = {1, 1, 5};
    size_t outputDims[] = {1, 1, 5}; // SAME -> outLen = inLen
    float outputData[1 * 1 * 5] = {0};

    avgPool1dRunResult_t r = avgPool1dBuild(input_avgPool1d_withSamePadding, inputDims, 3, SAME, 1,
                                            1, outputData, outputDims);

    avgPool1dForward(r.layer, r.input, r.output, NULL);
    for (size_t i = 0; i < expectedForward_avgPool1d_withSamePadding_len; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-5f, expectedForward_avgPool1d_withSamePadding[i],
                                 ((float *)r.output->data)[i]);
    }

    float lossGradData[1 * 1 * 5];
    for (size_t i = 0; i < 5; i++) {
        lossGradData[i] = 1.0f;
    }
    tensor_t *lossGrad = makeFloatTensor(outputDims, 3, lossGradData);

    tensor_t *propLoss = makeFloatTensor(inputDims, 3, NULL);

    avgPool1dBackward(r.layer, r.input, lossGrad, propLoss, NULL);
    for (size_t i = 0; i < expectedPropLoss_avgPool1d_withSamePadding_len; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-5f, expectedPropLoss_avgPool1d_withSamePadding[i],
                                 ((float *)propLoss->data)[i]);
    }
}

void testAvgPool1dEdgeCases(void) {
    size_t inputDims[] = {1, 1, 4};
    size_t outputDims[] = {1, 1, 1}; // K=L=4, stride=1, VALID -> outLen = 4-4+1 = 1
    float outputData[1] = {0};

    avgPool1dRunResult_t r = avgPool1dBuild(input_avgPool1d_edgeCases, inputDims, 4, VALID, 1, 1,
                                            outputData, outputDims);

    avgPool1dForward(r.layer, r.input, r.output, NULL);
    // input [1,2,3,4] mean = 2.5
    for (size_t i = 0; i < expectedForward_avgPool1d_edgeCases_len; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-5f, expectedForward_avgPool1d_edgeCases[i],
                                 ((float *)r.output->data)[i]);
    }

    float lossGradData[1] = {1.0f};
    tensor_t *lossGrad = makeFloatTensor(outputDims, 3, lossGradData);

    tensor_t *propLoss = makeFloatTensor(inputDims, 3, NULL);

    avgPool1dBackward(r.layer, r.input, lossGrad, propLoss, NULL);
    // 1/K = 0.25 contribution to each of 4 input positions.
    for (size_t i = 0; i < expectedPropLoss_avgPool1d_edgeCases_len; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-5f, expectedPropLoss_avgPool1d_edgeCases[i],
                                 ((float *)propLoss->data)[i]);
    }
}

/* #4 PR0 (C13): the FLOAT32 dx arm must OUT_WRITE propLoss -- the remat arena
 * hands it bytes a previous step left behind. 0xA5A5A5A5 read as a float is
 * -2.87e-16, which `+=` of an O(1) gradient absorbs exactly, so the first call
 * exposes stale bytes only in cells no window covers: K=2, dilation=2,
 * stride=2 over L=8 covers {0,2} {2,4} {4,6} (2 and 4 twice) and leaves 1, 3,
 * 5, 7 uncovered in every (b, c) row. The second call catches an accumulating
 * arm on every touched cell. All values are dyadic, so the compare is
 * bit-exact. */
void testAvgPool1dBackwardFloatOverwritesStalePropLoss(void) {
    size_t inputDims[] = {2, 2, 8};
    size_t outputDims[] = {2, 2, 3}; // effective K=3, stride=2, VALID -> outLen = 3
    float const gyData[2 * 2 * 3] = {
        1.0f,  -0.5f, 3.0f,  // b0c0
        2.0f,  0.5f,  -1.0f, // b0c1
        -3.0f, 1.0f,  0.25f, // b1c0
        0.5f,  0.5f,  0.5f,  // b1c1
    };
    float const expectedDx[2 * 2 * 8] = {
        0.5f,  0.0f, 0.25f, 0.0f, 1.25f,  0.0f, 1.5f,   0.0f, // b0c0
        1.0f,  0.0f, 1.25f, 0.0f, -0.25f, 0.0f, -0.5f,  0.0f, // b0c1
        -1.5f, 0.0f, -1.0f, 0.0f, 0.625f, 0.0f, 0.125f, 0.0f, // b1c0
        0.25f, 0.0f, 0.5f,  0.0f, 0.5f,   0.0f, 0.25f,  0.0f, // b1c1
    };

    avgPool1dRunResult_t r = avgPool1dBuild(NULL, inputDims, 2, VALID, 2, 2, NULL, outputDims);
    tensor_t *lossGrad = makeFloatTensor(outputDims, 3, gyData);
    tensor_t *propLoss = makeFloatTensor(inputDims, 3, NULL);
    size_t propLossElements = calcNumberOfElementsByTensor(propLoss);
    float firstDx[2 * 2 * 8];
    float secondDx[2 * 2 * 8];

    memset(propLoss->data, 0xA5, propLossElements * sizeof(float));
    avgPool1dBackward(r.layer, r.input, lossGrad, propLoss, NULL);
    memcpy(firstDx, propLoss->data, sizeof(firstDx));
    avgPool1dBackward(r.layer, r.input, lossGrad, propLoss, NULL);
    memcpy(secondDx, propLoss->data, sizeof(secondDx));

    freeTensor(propLoss);
    freeTensor(lossGrad);
    freeTensor(r.output);
    freeTensor(r.input);
    freeQuantization(r.q);

    TEST_ASSERT_EQUAL_size_t_MESSAGE(2 * 2 * 8, propLossElements, "fixture: propLoss size");
    TEST_ASSERT_EQUAL_MEMORY_MESSAGE(expectedDx, firstDx, sizeof(expectedDx),
                                     "dx must not keep the prefilled 0xA5 bytes");
    TEST_ASSERT_EQUAL_MEMORY_MESSAGE(expectedDx, secondDx, sizeof(expectedDx),
                                     "a repeated backward must reproduce dx (OUT_WRITE)");
}

/* #4 PR0: the FLOAT32 dx arm takes batch and channels from
 * lossGrad but memsets and scatters into propLoss, so a propLoss with fewer
 * channels is written past its end -- it must fail fast instead. */
void testAvgPool1dBackwardFloatRejectsPropLossWithFewerChannels(void) {
    size_t inputDims[] = {2, 2, 8};
    size_t outputDims[] = {2, 2, 3};
    size_t smallPropLossDims[] = {2, 1, 8};

    avgPool1dRunResult_t r = avgPool1dBuild(NULL, inputDims, 2, VALID, 2, 2, NULL, outputDims);
    tensor_t *lossGrad = makeFloatTensor(outputDims, 3, NULL);
    tensor_t *propLoss = makeFloatTensor(smallPropLossDims, 3, NULL);

    ASSERT_EXITS_WITH_OUTPUT(
        1, "AvgPool1d backward FLOAT32 (propLoss): expected shape [2, 2, 8], got [2, 1, 8]",
        avgPool1dBackward(r.layer, r.input, lossGrad, propLoss, NULL));

    freeTensor(propLoss);
    freeTensor(lossGrad);
    freeTensor(r.output);
    freeTensor(r.input);
    freeQuantization(r.q);
}

/* #4 PR0: the propLoss rank is checked before its dimensions[2] is read -- on
 * a rank-2 shape that read is itself a heap over-read (visible under ASan). */
void testAvgPool1dBackwardFloatRejectsRank2PropLoss(void) {
    size_t inputDims[] = {2, 2, 8};
    size_t outputDims[] = {2, 2, 3};
    size_t rank2PropLossDims[] = {2, 16};

    avgPool1dRunResult_t r = avgPool1dBuild(NULL, inputDims, 2, VALID, 2, 2, NULL, outputDims);
    tensor_t *lossGrad = makeFloatTensor(outputDims, 3, NULL);
    tensor_t *propLoss = makeFloatTensor(rank2PropLossDims, 2, NULL);

    ASSERT_EXITS_WITH_OUTPUT(1,
                             "AvgPool1d backward FLOAT32: lossGrad and propLoss must both be "
                             "rank-3, got ranks 3 and 2",
                             avgPool1dBackward(r.layer, r.input, lossGrad, propLoss, NULL));

    freeTensor(propLoss);
    freeTensor(lossGrad);
    freeTensor(r.output);
    freeTensor(r.input);
    freeQuantization(r.q);
}

/* #4 PR0: the FLOAT32 forward takes batch and channels from the input but
 * writes them into rawOut, so an output with a smaller batch is written past
 * its end -- it must fail fast instead. */
void testAvgPool1dForwardFloatRejectsOutputWithSmallerBatch(void) {
    size_t inputDims[] = {2, 2, 8};
    size_t outputDims[] = {2, 2, 3};
    size_t smallOutputDims[] = {1, 2, 3};

    avgPool1dRunResult_t r = avgPool1dBuild(NULL, inputDims, 2, VALID, 2, 2, NULL, outputDims);
    tensor_t *smallOutput = makeFloatTensor(smallOutputDims, 3, NULL);

    ASSERT_EXITS_WITH_OUTPUT(
        1, "AvgPool1d forward FLOAT32 (rawOut): expected shape [2, 2, 3], got [1, 2, 3]",
        avgPool1dForward(r.layer, r.input, smallOutput, NULL));

    freeTensor(smallOutput);
    freeTensor(r.output);
    freeTensor(r.input);
    freeQuantization(r.q);
}

// Smoke test for the funnel's new SYM-input capability (spec Testing list):
// the layer's own forwardMath stays FLOAT32 (no SYM kernel body exists for
// this op), but a SYM_INT32-typed *producer* tensor feeding it must now be
// dequantized by the executeOp prologue rather than silently reinterpreted
// as raw float bits (the pre-migration direct-cast hazard). Tolerance is
// widened relative to the FLOAT32 fixture tests above: it must cover the
// SYM_INT32@12 quantization step (qMax=2047) on top of the exact average,
// not just float rounding noise. absMax(1,4,2,3)=4.0 -> scale=4/2047;
// per-element error <= 0.5*scale ~ 9.8e-4, unchanged by the K=2 average
// (linear in the elements) -> 5e-3 leaves a >5x margin, not vacuous.
void testAvgPool1dForwardWithSymInt32Input(void) {
    size_t inputDims[] = {1, 1, 4};
    size_t outputDims[] = {1, 1, 3};

    static kernel_t kernelStore;
    static avgPool1dConfig_t cfgStore;
    static layer_t layerStore;
    static layerConfig_t lcStore;

    initKernel(&kernelStore, 2, VALID, 1, 1);
    quantization_t *floatQ = quantizationInitFloat();
    initAvgPool1dConfig(&cfgStore, &kernelStore, floatQ, floatQ);
    layerStore.type = AVGPOOL1D;
    lcStore.avgPool1d = &cfgStore;
    layerStore.config = &lcStore;

    size_t *ownedDims = reserveMemory(3 * sizeof(size_t));
    memcpy(ownedDims, inputDims, 3 * sizeof(size_t));
    size_t *order = reserveMemory(3 * sizeof(size_t));
    setOrderOfDimsForNewTensor(3, order);
    shape_t *shape = reserveMemory(sizeof(shape_t));
    setShape(shape, ownedDims, 3, order);
    tensor_t *symInput = initTensor(shape, quantizationInitSymInt32WithBits(HALF_AWAY, 12), NULL);
    tensorFillFromFloatBuffer(symInput, input_avgPool1d_basic,
                              calcNumberOfElementsByTensor(symInput));

    tensor_t *output = makeFloatTensor(outputDims, 3, NULL);

    avgPool1dForward(&layerStore, symInput, output, NULL);

    for (size_t i = 0; i < expectedForward_avgPool1d_basic_len; i++) {
        TEST_ASSERT_FLOAT_WITHIN(5e-3f, expectedForward_avgPool1d_basic[i],
                                 ((float *)output->data)[i]);
    }
}

/* ---- SYM_INT32 arm (#205) ---- */

/* Build a SYM_INT32 (HALF_AWAY, qMaxBits=12) tensor from a float fixture —
 * UnitTestConv1d.c helper pattern; fixtures are dequant-round-trip-stable
 * (sym_gold.stable_dequant_i12) so the C side lands on exactly the gold
 * mantissas+scale. NULL vals -> zero mantissas, scale 1.0. */
static tensor_t *buildSymTensor(size_t const *dims, size_t numDims, float const *vals) {
    size_t *ownedDims = reserveMemory(numDims * sizeof(size_t));
    memcpy(ownedDims, dims, numDims * sizeof(size_t));
    size_t *order = reserveMemory(numDims * sizeof(size_t));
    setOrderOfDimsForNewTensor(numDims, order);
    shape_t *shape = reserveMemory(sizeof(shape_t));
    setShape(shape, ownedDims, numDims, order);
    tensor_t *t = initTensor(shape, quantizationInitSymInt32WithBits(HALF_AWAY, 12), NULL);
    if (vals != NULL) {
        tensorFillFromFloatBuffer(t, vals, calcNumberOfElementsByTensor(t));
    }
    return t;
}

static void assertSymTensorMatchesGold(tensor_t *t, int32_t const *mantissas, size_t len,
                                       float expectedScale, int32_t mantissaTol, float scaleTol,
                                       float const *dequant, float dequantTol) {
    int32_t const *m = (int32_t const *)t->data;
    float scale = ((symInt32QConfig_t *)t->quantization->qConfig)->scale;
    TEST_ASSERT_FLOAT_WITHIN(expectedScale * scaleTol, expectedScale, scale);
    for (size_t i = 0; i < len; i++) {
        TEST_ASSERT_INT_WITHIN(mantissaTol, mantissas[i], m[i]);
        TEST_ASSERT_FLOAT_WITHIN(dequantTol, dequant[i], (float)m[i] * scale);
    }
}

#define ASSERT_SYM_FORWARD_GOLD(t, fix)                                                            \
    assertSymTensorMatchesGold(t, expectedForwardMantissas_avgPool1dSym_##fix,                     \
                               expectedForwardMantissas_avgPool1dSym_##fix##_len,                  \
                               expectedForwardScale_avgPool1dSym_##fix,                            \
                               mantissaTol_avgPool1dSym_##fix, scaleTol_avgPool1dSym_##fix,        \
                               expectedForwardDequant_avgPool1dSym_##fix,                          \
                               forwardDequantTol_avgPool1dSym_##fix)

#define ASSERT_SYM_PROPLOSS_GOLD(t, fix)                                                           \
    assertSymTensorMatchesGold(t, expectedPropLossMantissas_avgPool1dSym_##fix,                    \
                               expectedPropLossMantissas_avgPool1dSym_##fix##_len,                 \
                               expectedPropLossScale_avgPool1dSym_##fix,                           \
                               mantissaTol_avgPool1dSym_##fix, scaleTol_avgPool1dSym_##fix,        \
                               expectedPropLossDequant_avgPool1dSym_##fix,                         \
                               propLossDequantTol_avgPool1dSym_##fix)

typedef struct avgPool1dSymRun {
    layer_t *layer;
    tensor_t *input;
    tensor_t *output;
} avgPool1dSymRun_t;

/* Mirrors avgPool1dBuild but on SYM_INT32 wires: forwardQ/propLossQ declare
 * ARITH_SYM_INT32 compute via arithmeticFromQuantizationOrDefault. */
static avgPool1dSymRun_t avgPool1dBuildSym(float const *inputData, size_t const *inputDims,
                                           size_t kSize, paddingType_t padding, size_t dilation,
                                           size_t stride, size_t const *outputDims) {
    static kernel_t kernelStore;
    static avgPool1dConfig_t cfgStore;
    static layer_t layerStore;
    static layerConfig_t lcStore;

    initKernel(&kernelStore, kSize, padding, dilation, stride);

    quantization_t *q = quantizationInitSymInt32(HALF_AWAY);
    initAvgPool1dConfig(&cfgStore, &kernelStore, q, q);

    layerStore.type = AVGPOOL1D;
    lcStore.avgPool1d = &cfgStore;
    layerStore.config = &lcStore;

    avgPool1dSymRun_t r = {0};
    r.layer = &layerStore;
    r.input = buildSymTensor(inputDims, 3, inputData);
    r.output = buildSymTensor(outputDims, 3, NULL);
    return r;
}

void testAvgPool1dForwardSymBasic(void) {
    size_t inputDims[] = {1, 1, 4};
    size_t outputDims[] = {1, 1, 3};

    avgPool1dSymRun_t r =
        avgPool1dBuildSym(input_avgPool1dSym_symBasic, inputDims, 2, VALID, 1, 1, outputDims);

    avgPool1dForward(r.layer, r.input, r.output, NULL);

    ASSERT_SYM_FORWARD_GOLD(r.output, symBasic);
}

void testAvgPool1dBackwardSymBasic(void) {
    size_t inputDims[] = {1, 1, 4};
    size_t outputDims[] = {1, 1, 3};

    avgPool1dSymRun_t r =
        avgPool1dBuildSym(input_avgPool1dSym_symBasic, inputDims, 2, VALID, 1, 1, outputDims);

    tensor_t *lossGrad = buildSymTensor(outputDims, 3, lossGrad_avgPool1dSym_symBasic);
    tensor_t *propLoss = buildSymTensor(inputDims, 3, NULL);

    avgPool1dBackward(r.layer, r.input, lossGrad, propLoss, NULL);

    ASSERT_SYM_PROPLOSS_GOLD(propLoss, symBasic);
}

void testAvgPool1dSymStrideDilationForwardBackward(void) {
    size_t inputDims[] = {1, 1, 9};
    size_t outputDims[] = {1, 1, 3};

    // K=2, stride=3, dilation=2 — random gold lossGrad so positional mutations
    // on the SYM scatter path are non-vacuous.
    avgPool1dSymRun_t r = avgPool1dBuildSym(input_avgPool1dSym_symStrideDilation, inputDims, 2,
                                            VALID, 2, 3, outputDims);

    avgPool1dForward(r.layer, r.input, r.output, NULL);
    ASSERT_SYM_FORWARD_GOLD(r.output, symStrideDilation);

    tensor_t *lossGrad = buildSymTensor(outputDims, 3, lossGrad_avgPool1dSym_symStrideDilation);
    tensor_t *propLoss = buildSymTensor(inputDims, 3, NULL);

    avgPool1dBackward(r.layer, r.input, lossGrad, propLoss, NULL);
    ASSERT_SYM_PROPLOSS_GOLD(propLoss, symStrideDilation);
}

void testAvgPool1dSymSamePadding(void) {
    size_t inputDims[] = {1, 1, 5};
    size_t outputDims[] = {1, 1, 5};

    // Edge windows have validCount=2 but the scale fold keeps the divisor at
    // K=3 — pins count_include_pad=true on the SYM path.
    avgPool1dSymRun_t r =
        avgPool1dBuildSym(input_avgPool1dSym_symSamePadding, inputDims, 3, SAME, 1, 1, outputDims);

    avgPool1dForward(r.layer, r.input, r.output, NULL);
    ASSERT_SYM_FORWARD_GOLD(r.output, symSamePadding);

    tensor_t *lossGrad = buildSymTensor(outputDims, 3, lossGrad_avgPool1dSym_symSamePadding);
    tensor_t *propLoss = buildSymTensor(inputDims, 3, NULL);

    avgPool1dBackward(r.layer, r.input, lossGrad, propLoss, NULL);
    ASSERT_SYM_PROPLOSS_GOLD(propLoss, symSamePadding);
}

static tensor_t *buildSymTensorWithBits(size_t const *dims, size_t numDims, uint8_t qMaxBits) {
    size_t *ownedDims = reserveMemory(numDims * sizeof(size_t));
    memcpy(ownedDims, dims, numDims * sizeof(size_t));
    size_t *order = reserveMemory(numDims * sizeof(size_t));
    setOrderOfDimsForNewTensor(numDims, order);
    shape_t *shape = reserveMemory(sizeof(shape_t));
    setShape(shape, ownedDims, numDims, order);
    return initTensor(shape, quantizationInitSymInt32WithBits(HALF_AWAY, qMaxBits), NULL);
}

/* Value-sum guard, width branch: a 31-bit input mantissa can overflow the
 * int32 window sum after 2 terms — the forward kernel must fail fast on
 * qMaxBits > 16 (Reduce.c value-sum contract). */
void testAvgPool1dForwardSymRejectsWideOperand(void) {
    size_t inputDims[] = {1, 1, 4};
    size_t outputDims[] = {1, 1, 3};

    avgPool1dSymRun_t r =
        avgPool1dBuildSym(input_avgPool1dSym_symBasic, inputDims, 2, VALID, 1, 1, outputDims);
    r.input = buildSymTensorWithBits(inputDims, 3, 31);

    ASSERT_EXITS_WITH_FAILURE(avgPool1dForward(r.layer, r.input, r.output, NULL));
}

/* Value-sum guard, N branch: at qMaxBits=16 the scatter bound is
 * 2^(32-16) = 65536 covering windows — K=65536/stride=1 reaches it
 * ((effK-1)/stride + 1 = 65536). */
void testAvgPool1dBackwardSymRejectsTermsOverBound(void) {
    size_t inputDims[] = {1, 1, 65536};
    size_t outputDims[] = {1, 1, 1};

    avgPool1dSymRun_t r = avgPool1dBuildSym(NULL, inputDims, 65536, VALID, 1, 1, outputDims);

    tensor_t *lossGrad = buildSymTensorWithBits(outputDims, 3, 16);
    tensor_t *propLoss = buildSymTensor(inputDims, 3, NULL);

    ASSERT_EXITS_WITH_FAILURE(avgPool1dBackward(r.layer, r.input, lossGrad, propLoss, NULL));
}

/* #4 PR0: the SYM_INT32 dx kernel takes batch and channels from lossGrad but
 * memsets and scatters into rawOut (propLoss's shape), so a propLoss with a
 * smaller batch is written past its end -- it must fail fast instead. */
void testAvgPool1dBackwardSymRejectsPropLossWithSmallerBatch(void) {
    size_t inputDims[] = {2, 2, 8};
    size_t outputDims[] = {2, 2, 3};
    size_t smallPropLossDims[] = {1, 2, 8};

    avgPool1dSymRun_t r = avgPool1dBuildSym(NULL, inputDims, 2, VALID, 2, 2, outputDims);
    tensor_t *lossGrad = buildSymTensor(outputDims, 3, NULL);
    tensor_t *propLoss = buildSymTensor(smallPropLossDims, 3, NULL);

    ASSERT_EXITS_WITH_OUTPUT(
        1, "AvgPool1d backward SYM_INT32 (rawOut): expected shape [2, 2, 8], got [1, 2, 8]",
        avgPool1dBackward(r.layer, r.input, lossGrad, propLoss, NULL));

    freeTensor(propLoss);
    freeTensor(lossGrad);
    freeTensor(r.output);
    freeTensor(r.input);
}

/* #4 PR0: the dx output's rank is checked before its dimensions[2] is read --
 * on a rank-2 shape that read is itself a heap over-read (visible under ASan). */
void testAvgPool1dBackwardSymRejectsRank2PropLoss(void) {
    size_t inputDims[] = {2, 2, 8};
    size_t outputDims[] = {2, 2, 3};
    size_t rank2PropLossDims[] = {2, 16};

    avgPool1dSymRun_t r = avgPool1dBuildSym(NULL, inputDims, 2, VALID, 2, 2, outputDims);
    tensor_t *lossGrad = buildSymTensor(outputDims, 3, NULL);
    tensor_t *propLoss = buildSymTensor(rank2PropLossDims, 2, NULL);

    ASSERT_EXITS_WITH_OUTPUT(1,
                             "AvgPool1d backward SYM_INT32: lossGrad and rawOut must both be "
                             "rank-3, got ranks 3 and 2",
                             avgPool1dBackward(r.layer, r.input, lossGrad, propLoss, NULL));

    freeTensor(propLoss);
    freeTensor(lossGrad);
    freeTensor(r.output);
    freeTensor(r.input);
}

/* #4 PR0: the SYM_INT32 forward takes batch and channels from the input but
 * writes them into rawOut (the output's shape), so an output with fewer
 * channels is written past its end -- it must fail fast instead. */
void testAvgPool1dForwardSymRejectsOutputWithFewerChannels(void) {
    size_t inputDims[] = {2, 2, 8};
    size_t outputDims[] = {2, 2, 3};
    size_t smallOutputDims[] = {2, 1, 3};

    avgPool1dSymRun_t r = avgPool1dBuildSym(NULL, inputDims, 2, VALID, 2, 2, outputDims);
    tensor_t *smallOutput = buildSymTensor(smallOutputDims, 3, NULL);

    ASSERT_EXITS_WITH_OUTPUT(
        1, "AvgPool1d forward SYM_INT32 (rawOut): expected shape [2, 2, 3], got [2, 1, 3]",
        avgPool1dForward(r.layer, r.input, smallOutput, NULL));

    freeTensor(smallOutput);
    freeTensor(r.output);
    freeTensor(r.input);
}

/* ---- BFP epic PR4 (R-P1/R-P4): native ARITH_BFP arms ---- */

/* BFP epic PR4: build a BFP wire with EXACT codes and per-group exponents.
 * Writing the packed payload directly (byteConversion) instead of quantizing
 * keeps the fixture independent of the quantizer and lets the test pin the
 * borrowed exponents the kernel folds with. */
static tensor_t *buildBfpWireWithCodes(size_t const *dims, size_t numDims, uint8_t mantissaBits,
                                       uint8_t exponentBits, size_t numGroups, size_t groupSize,
                                       int32_t *codes, uint8_t const *exponents) {
    size_t *ownedDims = reserveMemory(numDims * sizeof(size_t));
    memcpy(ownedDims, dims, numDims * sizeof(size_t));
    size_t *order = reserveMemory(numDims * sizeof(size_t));
    setOrderOfDimsForNewTensor(numDims, order);
    shape_t *shape = reserveMemory(sizeof(shape_t));
    setShape(shape, ownedDims, numDims, order);
    quantization_t *q = numGroups > 1 ? quantizationInitBfpGrouped(mantissaBits, exponentBits,
                                                                   HALF_AWAY, numGroups, groupSize)
                                      : quantizationInitBfp(mantissaBits, exponentBits, HALF_AWAY);
    tensor_t *t = initTensor(shape, q, NULL);
    size_t n = calcNumberOfElementsByTensor(t);
    if (codes != NULL) {
        byteConversion((uint8_t *)codes, 32, t->data, mantissaBits, n);
    }
    bfpQConfig_t *qc = q->qConfig;
    if (exponents != NULL) {
        memcpy(qc->exponents, exponents, numGroups);
    }
    return t;
}

/* BFP epic PR4 (R-P1): pools have no weight operand, so the width anchor for
 * staging is the layer's OWN produced-wire config — outputQ for the forward,
 * propLossQ for the backward. initAvgPool1dConfig derives both math slots
 * from it, so passing a BFP wireQ selects the native arms. */
static layer_t *avgPool1dBuildBfpLayer(avgPool1dConfig_t *cfgStore, kernel_t *kernelStore,
                                       layerConfig_t *lcStore, layer_t *layerStore, size_t kSize,
                                       paddingType_t padding, size_t dilation, size_t stride,
                                       quantization_t *wireQ) {
    initKernel(kernelStore, kSize, padding, dilation, stride);
    initAvgPool1dConfig(cfgStore, kernelStore, wireQ, wireQ);
    layerStore->type = AVGPOOL1D;
    lcStore->avgPool1d = cfgStore;
    layerStore->config = lcStore;
    return layerStore;
}

/* Borrowed BFP-stored grouped input (BFP D8: never re-blocked): the integer
 * kernel folds same-group segments into the per-tensor BFP_RAW (a FLOAT32
 * target has no groups) and folds -log2 K into the exponent; the epilogue
 * dequantizes ldexpf((float)mant, E). Bit-exact against the integer mirror. */
void testAvgPool1dForwardBfpBorrowedGroupedInput(void) {
    size_t inputDims[] = {1, 2, 8};
    size_t outputDims[] = {1, 2, 4};
    int32_t inCodes[16];
    for (size_t i = 0; i < kBfpAvgPoolInCodes_len; i++) {
        inCodes[i] = kBfpAvgPoolInCodes[i];
    }
    tensor_t *input = buildBfpWireWithCodes(
        inputDims, 3, (uint8_t)kBfpAvgPoolMantissaBits, (uint8_t)kBfpAvgPoolExponentBits,
        (size_t)kBfpAvgPoolInNumGroups, (size_t)kBfpAvgPoolInGroupSize, inCodes, kBfpAvgPoolInExps);
    tensor_t *output = makeFloatTensor(outputDims, 3, NULL);
    quantization_t *wireQ = quantizationInitBfpGrouped(
        (uint8_t)kBfpAvgPoolMantissaBits, (uint8_t)kBfpAvgPoolExponentBits, HALF_AWAY,
        (size_t)kBfpAvgPoolInNumGroups, (size_t)kBfpAvgPoolInGroupSize);
    kernel_t k;
    avgPool1dConfig_t cfg;
    layerConfig_t lc;
    layer_t layer;
    avgPool1dBuildBfpLayer(&cfg, &k, &lc, &layer, (size_t)kBfpAvgPoolKernelSize, SAME,
                           (size_t)kBfpAvgPoolDilation, (size_t)kBfpAvgPoolStride, wireQ);

    avgPool1dForward(&layer, input, output, NULL);

    TEST_ASSERT_EQUAL_MEMORY_MESSAGE(kBfpAvgPoolExpectedForward, output->data,
                                     kBfpAvgPoolExpectedForward_len * sizeof(float),
                                     "the FLOAT32 target is the exact dequant of the integer "
                                     "BFP_RAW");
    freeQuantization(wireQ);
    freeTensor(output);
    freeTensor(input);
}

/* R-P1 staging: a FLOAT32-stored operand is quantized into BFP scratch
 * per-tensor at the PRODUCED WIRE's widths (m=6 here). The gold's generator
 * asserts that staging at m=8 instead would change these numbers. */
void testAvgPool1dForwardBfpStagesFloat32InputAtWireWidths(void) {
    size_t inputDims[] = {1, 2, 8};
    size_t outputDims[] = {1, 2, 4};
    tensor_t *input = makeFloatTensor(inputDims, 3, kBfpAvgPoolStageInput);
    tensor_t *output = makeFloatTensor(outputDims, 3, NULL);
    quantization_t *wireQ = quantizationInitBfpGrouped(
        (uint8_t)kBfpAvgPoolMantissaBits, (uint8_t)kBfpAvgPoolExponentBits, HALF_AWAY,
        (size_t)kBfpAvgPoolInNumGroups, (size_t)kBfpAvgPoolInGroupSize);
    kernel_t k;
    avgPool1dConfig_t cfg;
    layerConfig_t lc;
    layer_t layer;
    avgPool1dBuildBfpLayer(&cfg, &k, &lc, &layer, (size_t)kBfpAvgPoolKernelSize, SAME,
                           (size_t)kBfpAvgPoolDilation, (size_t)kBfpAvgPoolStride, wireQ);

    avgPool1dForward(&layer, input, output, NULL);

    TEST_ASSERT_EQUAL_MEMORY_MESSAGE(kBfpAvgPoolExpectedStaged, output->data,
                                     kBfpAvgPoolExpectedStaged_len * sizeof(float),
                                     "the FLOAT32 operand must stage PER-TENSOR at the "
                                     "produced-wire widths (R-P1)");
    freeQuantization(wireQ);
    freeTensor(output);
    freeTensor(input);
}

/* OUT_WRITE onto a BFP wire: the raw takes the target's blocking and the
 * epilogue re-blocks it in integer with the OP's rounding (#282). */
void testAvgPool1dForwardBfpPacksOutputWire(void) {
    size_t inputDims[] = {1, 2, 8};
    size_t outputDims[] = {1, 2, 4};
    int32_t inCodes[16];
    for (size_t i = 0; i < kBfpAvgPoolInCodes_len; i++) {
        inCodes[i] = kBfpAvgPoolInCodes[i];
    }
    int32_t sentinel[8] = {-9, -9, -9, -9, -9, -9, -9, -9};
    uint8_t zeroState[2] = {127, 127};
    tensor_t *input = buildBfpWireWithCodes(
        inputDims, 3, (uint8_t)kBfpAvgPoolMantissaBits, (uint8_t)kBfpAvgPoolExponentBits,
        (size_t)kBfpAvgPoolInNumGroups, (size_t)kBfpAvgPoolInGroupSize, inCodes, kBfpAvgPoolInExps);
    tensor_t *output = buildBfpWireWithCodes(
        outputDims, 3, (uint8_t)kBfpAvgPoolMantissaBits, (uint8_t)kBfpAvgPoolExponentBits,
        (size_t)kBfpAvgPoolOutNumGroups, (size_t)kBfpAvgPoolOutGroupSize, sentinel, zeroState);
    quantization_t *wireQ = quantizationInitBfpGrouped(
        (uint8_t)kBfpAvgPoolMantissaBits, (uint8_t)kBfpAvgPoolExponentBits, HALF_AWAY,
        (size_t)kBfpAvgPoolInNumGroups, (size_t)kBfpAvgPoolInGroupSize);
    kernel_t k;
    avgPool1dConfig_t cfg;
    layerConfig_t lc;
    layer_t layer;
    avgPool1dBuildBfpLayer(&cfg, &k, &lc, &layer, (size_t)kBfpAvgPoolKernelSize, SAME,
                           (size_t)kBfpAvgPoolDilation, (size_t)kBfpAvgPoolStride, wireQ);

    avgPool1dForward(&layer, input, output, NULL);

    int32_t got[8];
    unpackSignExtend(output->data, (uint8_t)kBfpAvgPoolMantissaBits, 0, got,
                     kBfpAvgPoolPackedCodes_len);
    TEST_ASSERT_EQUAL_INT32_ARRAY(kBfpAvgPoolPackedCodes, got, kBfpAvgPoolPackedCodes_len);
    bfpQConfig_t *outQC = output->quantization->qConfig;
    TEST_ASSERT_EQUAL_UINT8_MESSAGE(kBfpAvgPoolPackedExps[0], outQC->exponents[0],
                                    "OUT_WRITE must derive the output wire's group exponents");
    TEST_ASSERT_EQUAL_UINT8_MESSAGE(kBfpAvgPoolPackedExps[1], outQC->exponents[1],
                                    "OUT_WRITE must derive the output wire's group exponents");
    freeQuantization(wireQ);
    freeTensor(output);
    freeTensor(input);
}

/* ---- Integer-only forward at the kernel (arithmetic-bfp.md section 13) ----
 * avgPool1dForwardKernelBfpRaw on stack operands (BfpRawFixture.h), checked
 * bit for bit against a reference written from the definition in int64:
 * guard g = the smallest g with K 2^(m-1) 2^-g + (K + 1) <= INT32_MAX;
 * E_acc of a raw group = max input exponent over the input rows its outputs
 * read, + g; per window one partial per same-group segment, aligned
 * (left exact, right HALF_AWAY); then a power-of-two K folds -log2 K into
 * the exponent, any other K is ONE half-away division per output. */

typedef struct bfpAvgCase {
    const int32_t *codes;
    const uint8_t *exps; /* biased, e = 8 */
    size_t numGroups;
    size_t groupSize;
    uint8_t mantissaBits;
    size_t rows; /* batch 1, `rows` channels */
    size_t len;
    size_t k;
    size_t dilation;
    size_t stride;
    size_t rawNumGroups;
    size_t rawGroupSize;
} bfpAvgCase_t;

static size_t bfpAvgOutLen(const bfpAvgCase_t *c) {
    return (c->len - c->dilation * (c->k - 1) - 1) / c->stride + 1;
}

static size_t bfpAvgGroup(const bfpAvgCase_t *c, size_t idx) {
    return c->groupSize == 0 ? 0 : idx / c->groupSize;
}

static int bfpAvgE(const bfpAvgCase_t *c, size_t idx) {
    return (int)c->exps[bfpAvgGroup(c, idx)] - 127;
}

static int64_t refDivHalfAway(int64_t v, int64_t d) {
    int64_t mag = v < 0 ? -v : v;
    int64_t q = mag / d;
    if (2 * (mag - q * d) >= d) {
        q++;
    }
    return v < 0 ? -q : q;
}

static int64_t refAlign(int64_t v, int k) {
    if (v == 0) {
        return 0; /* a zero group may sit any distance above E_acc */
    }
    if (k <= 0) {
        return v * ((int64_t)1 << -k);
    }
    return k > 62 ? 0 : refDivHalfAway(v, (int64_t)1 << k);
}

static int refSumGuard(size_t k, uint8_t m) {
    uint64_t total = (uint64_t)k << (m - 1u);
    uint64_t room = (uint64_t)INT32_MAX - (k + 1u);
    for (int g = -30; g <= 31; g++) {
        bool holds = g < 0 ? (total << (unsigned)(-g)) <= room : total <= (room << (unsigned)g);
        if (holds) {
            return g;
        }
    }
    TEST_FAIL_MESSAGE("refSumGuard: no g");
    return 0;
}

static void refAvgPool(const bfpAvgCase_t *c, int32_t *mant, int16_t *rawE) {
    size_t outLen = bfpAvgOutLen(c);
    size_t n = c->rows * outLen;
    size_t gsz = c->rawGroupSize == 0 ? n : c->rawGroupSize;
    int g = refSumGuard(c->k, c->mantissaBits);
    int log2K = -1;
    for (int b = 0; b < 31; b++) {
        if ((size_t)1 << b == c->k) {
            log2K = b;
        }
    }
    for (size_t t = 0; t < c->rawNumGroups; t++) {
        int top = INT32_MIN;
        for (size_t j = t * gsz; j < (t + 1) * gsz; j++) {
            size_t row = j / outLen;
            for (size_t i = 0; i < c->len; i++) {
                if (c->codes[row * c->len + i] == 0) {
                    continue; /* zero groups never set E_acc */
                }
                int e = bfpAvgE(c, row * c->len + i);
                top = e > top ? e : top;
            }
        }
        int eAcc = (top == INT32_MIN ? 0 : top) + g;
        rawE[t] = (int16_t)(log2K >= 0 ? eAcc - log2K : eAcc);
        for (size_t j = t * gsz; j < (t + 1) * gsz; j++) {
            size_t row = j / outLen, o = j % outLen;
            int64_t acc = 0, partial = 0;
            size_t segGroup = SIZE_MAX;
            size_t segments = 0;
            double exact = 0.0;
            for (size_t i = 0; i < c->k; i++) {
                size_t idx = row * c->len + o * c->stride + i * c->dilation;
                if (segGroup != SIZE_MAX && bfpAvgGroup(c, idx) != segGroup) {
                    acc += refAlign(partial, eAcc - ((int)c->exps[segGroup] - 127));
                    TEST_ASSERT_TRUE(acc >= INT32_MIN && acc <= INT32_MAX);
                    partial = 0;
                    segments++;
                }
                segGroup = bfpAvgGroup(c, idx);
                partial += c->codes[idx];
                exact += ldexp((double)c->codes[idx], bfpAvgE(c, idx));
            }
            acc += refAlign(partial, eAcc - ((int)c->exps[segGroup] - 127));
            segments++;
            TEST_ASSERT_TRUE(acc >= INT32_MIN && acc <= INT32_MAX);
            mant[j] = (int32_t)(log2K >= 0 ? acc : refDivHalfAway(acc, (int64_t)c->k));
            /* The documented error bound: S folds of at most 1/2 each, over K,
             * plus the division's 1/2 -- all in units of 2^E_acc. */
            double got = ldexp((double)mant[j], rawE[t]);
            double bound = ldexp(0.5 * (double)segments / (double)c->k + 0.5, eAcc);
            TEST_ASSERT_TRUE(fabs(got - exact / (double)c->k) <= bound + 1e-12 * fabs(exact));
        }
    }
}

static void checkAvgPoolBfpRaw(const bfpAvgCase_t *c) {
    size_t inDims[3] = {1, c->rows, c->len};
    size_t outLen = bfpAvgOutLen(c);
    size_t outDims[3] = {1, c->rows, outLen};
    bfpScratchFixture_t in;
    bfpScratchInit(&in, 3, inDims, c->codes, c->exps, c->numGroups, c->groupSize, c->mantissaBits,
                   8);
    bfpRawFixture_t raw;
    bfpRawInit(&raw, 3, outDims, c->rawNumGroups, c->rawGroupSize);
    kernel_t kern;
    initKernel(&kern, c->k, VALID, c->dilation, c->stride);
    avgPool1dConfig_t cfg = {0};
    cfg.kernel = &kern;
    tensor_t *ops[1] = {&in.tensor};

    avgPool1dForwardKernelBfpRaw(ops, 1, &raw.tensor, NULL, &cfg, NULL);

    int32_t mant[64];
    int16_t rawE[16];
    refAvgPool(c, mant, rawE);
    assertRawEquals(&raw, mant, c->rows * outLen, rawE, c->rawNumGroups);
}

/* K = 4 is a power of two: the mean is the aligned sum under E_acc - 2,
 * EXACT here (every window within -g binades of its row max), so the raw
 * equals sum / 4 exactly -- checked against hand values too. */
void testAvgPool1dBfpRawPow2KFoldsIntoTheExponent(void) {
    int32_t codes[16] = {12, -7, 0, 31, -128, 5, -1, 20, 3, -14, 22, -1, 8, 0, 127, 6};
    uint8_t exps[8] = {130, 127, 133, 125, 128, 128, 120, 127};
    bfpAvgCase_t c = {codes, exps, 8, 2, 8, 2, 8, 4, 1, 2, 1, 0};
    checkAvgPoolBfpRaw(&c);

    /* Row 0, output 0 = (12 * 2^3 - 7 * 2^3 + 0 + 31) / 4 = 71 / 4, exactly:
     * the raw holds it as mant * 2^E with no rounding. */
    size_t inDims[3] = {1, 2, 8};
    size_t outDims[3] = {1, 2, 3};
    bfpScratchFixture_t in;
    bfpScratchInit(&in, 3, inDims, codes, exps, 8, 2, 8, 8);
    bfpRawFixture_t raw;
    bfpRawInit(&raw, 3, outDims, 1, 0);
    kernel_t kern;
    initKernel(&kern, 4, VALID, 1, 2);
    avgPool1dConfig_t cfg = {0};
    cfg.kernel = &kern;
    tensor_t *ops[1] = {&in.tensor};
    avgPool1dForwardKernelBfpRaw(ops, 1, &raw.tensor, NULL, &cfg, NULL);
    TEST_ASSERT_EQUAL_FLOAT(71.0f / 4.0f, ldexpf((float)raw.mant[0], raw.exps[0]));
}

/* K = 2 with dilation 4 (dilated windows that cross groups), per-tensor raw
 * and a raw group straddling the rows. */
void testAvgPool1dBfpRawDilatedPow2KAcrossGroups(void) {
    int32_t codes[16] = {12, -7, 0, 31, -32, 5, -1, 20, 3, -14, 22, -1, 8, 0, -30, 6};
    uint8_t exps[4] = {130, 127, 133, 125};
    bfpAvgCase_t per = {codes, exps, 4, 4, 6, 2, 8, 2, 4, 1, 1, 0};
    checkAvgPoolBfpRaw(&per);
    bfpAvgCase_t grouped = {codes, exps, 4, 4, 6, 2, 8, 2, 4, 1, 2, 4};
    checkAvgPoolBfpRaw(&grouped);
    bfpAvgCase_t eight = {codes, exps, 4, 4, 6, 2, 8, 8, 1, 1, 1, 0};
    checkAvgPoolBfpRaw(&eight);
}

/* Ruling R1: under BFP math the divisor K must be a power of two, so the
 * mean is always an exponent shift. count_include_pad = true makes the
 * divisor K at every output, padded edges included, so the kernel size is
 * the only divisor to check. Both integer kernels and the factory refuse
 * K = 3 (and the FLOAT32 and SYM arms keep accepting it). */
static void apForwardNonPow2Body(size_t k) {
    int32_t codes[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    uint8_t exps[1] = {127};
    size_t inDims[3] = {1, 1, 8};
    kernel_t kern;
    initKernel(&kern, k, VALID, 1, 1);
    size_t outDims[3] = {1, 1, windowGeometry1dCalc(8, &kern).outputLength};
    bfpScratchFixture_t in;
    bfpScratchInit(&in, 3, inDims, codes, exps, 1, 0, 8, 8);
    bfpRawFixture_t raw;
    bfpRawInit(&raw, 3, outDims, 1, 0);
    avgPool1dConfig_t cfg = {0};
    cfg.kernel = &kern;
    tensor_t *ops[1] = {&in.tensor};
    avgPool1dForwardKernelBfpRaw(ops, 1, &raw.tensor, NULL, &cfg, NULL);
}

static void apBackwardNonPow2Body(size_t k) {
    int32_t gy[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    uint8_t exps[1] = {127};
    kernel_t kern;
    initKernel(&kern, k, SAME, 1, 1);
    size_t gyDims[3] = {1, 1, windowGeometry1dCalc(8, &kern).outputLength};
    size_t dxDims[3] = {1, 1, 8};
    bfpScratchFixture_t g;
    bfpScratchInit(&g, 3, gyDims, gy, exps, 1, 0, 8, 8);
    bfpRawFixture_t raw;
    bfpRawInit(&raw, 3, dxDims, 1, 0);
    avgPool1dConfig_t cfg = {0};
    cfg.kernel = &kern;
    tensor_t *ops[1] = {&g.tensor};
    avgPool1dBackwardKernelBfpRaw(ops, 1, &raw.tensor, NULL, &cfg, NULL);
}

void testAvgPool1dBfpRejectsANonPowerOfTwoKernel(void) {
    static const size_t bad[3] = {3, 6, 33};
    for (size_t i = 0; i < 3; i++) {
        ASSERT_EXITS_WITH_OUTPUT(1, "AvgPool1d forward BFP: BFP math needs a power-of-two kernel",
                                 apForwardNonPow2Body(bad[i]));
        ASSERT_EXITS_WITH_OUTPUT(1, "AvgPool1d backward BFP: BFP math needs a power-of-two kernel",
                                 apBackwardNonPow2Body(bad[i]));
    }
}

/* The power-of-two helper itself: log2 K for K = 1, 2, 32, 2^20. */
void testAvgPool1dBfpLog2DivisorOfPowersOfTwo(void) {
    TEST_ASSERT_EQUAL_INT(0, avgPool1dBfpLog2Divisor(1, "t"));
    TEST_ASSERT_EQUAL_INT(1, avgPool1dBfpLog2Divisor(2, "t"));
    TEST_ASSERT_EQUAL_INT(5, avgPool1dBfpLog2Divisor(32, "t"));
    TEST_ASSERT_EQUAL_INT(20, avgPool1dBfpLog2Divisor((size_t)1 << 20, "t"));
}

/* Worst cases: every code at the floor -2^(m-1), at the ceiling, or mixed,
 * over exponent gaps 0, 15, 16, 17 and 40 between neighbouring groups (the
 * low segments are rounded or flushed), K in {2, 4, 8} with m = 16 and 8,
 * per-tensor and grouped raws. The reference asserts int32 range at every
 * fold and the documented error bound. */
void testAvgPool1dBfpRawWorstCaseCodesAcrossExponentGaps(void) {
    uint8_t exps[12] = {127, 127, 142, 158, 175, 135, 120, 136, 176, 136, 121, 104};
    static const uint8_t widths[2] = {16, 8};
    static const size_t ks[3] = {2, 4, 8};
    for (size_t w = 0; w < 2; w++) {
        int32_t lo = -(1 << (widths[w] - 1)), hi = (1 << (widths[w] - 1)) - 1;
        int32_t floorCodes[24], ceilCodes[24], mixed[24];
        for (size_t i = 0; i < 24; i++) {
            floorCodes[i] = lo;
            ceilCodes[i] = hi;
            mixed[i] = i % 3 == 0 ? lo : (i % 3 == 1 ? hi : (int32_t)(i % 5) - 2);
        }
        const int32_t *sets[3] = {floorCodes, ceilCodes, mixed};
        for (size_t s = 0; s < 3; s++) {
            for (size_t ki = 0; ki < 3; ki++) {
                size_t outLen = 12 - ks[ki] + 1;
                bfpAvgCase_t per = {sets[s], exps, 12, 2, widths[w], 2, 12, ks[ki], 1, 1, 1, 0};
                checkAvgPoolBfpRaw(&per);
                bfpAvgCase_t byRow = {sets[s], exps,   12, 2, widths[w], 2,
                                      12,      ks[ki], 1,  1, 2,         outLen};
                checkAvgPoolBfpRaw(&byRow);
            }
        }
    }
}

static void avgPoolBfpRawIntoFloatRawBody(void) {
    int32_t codes[4] = {1, 2, 3, 4};
    uint8_t exps[1] = {127};
    size_t inDims[3] = {1, 1, 4};
    size_t outDims[3] = {1, 1, 3};
    bfpScratchFixture_t in;
    bfpScratchInit(&in, 3, inDims, codes, exps, 1, 0, 8, 8);
    tensor_t *notRaw = makeFloatTensor(outDims, 3, NULL);
    kernel_t kern;
    initKernel(&kern, 2, VALID, 1, 1);
    avgPool1dConfig_t cfg = {0};
    cfg.kernel = &kern;
    tensor_t *ops[1] = {&in.tensor};
    avgPool1dForwardKernelBfpRaw(ops, 1, notRaw, NULL, &cfg, NULL);
}

/* Zero groups never set E_acc: input [1, 1, 8] in groups of 4, group 0 all
 * zero in the zero state (E = 0), group 1 live at E = -28; K = 2, stride 2
 * (a power of two: the raw is the exact window sum under E - 1). Every
 * output equals its exact value. */
void testAvgPool1dBfpRawZeroGroupDoesNotSetTheAccumulator(void) {
    int32_t codes[8] = {0, 0, 0, 0, 113, -77, 91, -59};
    uint8_t exps[2] = {127, 127 - 28};
    size_t inDims[3] = {1, 1, 8};
    size_t outDims[3] = {1, 1, 4};
    bfpScratchFixture_t in;
    bfpScratchInit(&in, 3, inDims, codes, exps, 2, 4, 8, 8);
    bfpRawFixture_t raw;
    bfpRawInit(&raw, 3, outDims, 1, 0);
    kernel_t kern;
    initKernel(&kern, 2, VALID, 1, 2);
    avgPool1dConfig_t cfg = {0};
    cfg.kernel = &kern;
    tensor_t *ops[1] = {&in.tensor};

    avgPool1dForwardKernelBfpRaw(ops, 1, &raw.tensor, NULL, &cfg, NULL);

    for (size_t o = 0; o < 4; o++) {
        int64_t sum = (int64_t)codes[2 * o] + codes[2 * o + 1];
        assertBfpValueExact(raw.mant[o], raw.exps[0], sum, -28 - 1, o);
    }
}

void testAvgPool1dBfpRawRejectsANonRawOutput(void) {
    ASSERT_EXITS_WITH_OUTPUT(
        1, "AvgPool1d forward BFP: needs a BFP (unpacked scratch) input and a BFP_RAW",
        avgPoolBfpRawIntoFloatRawBody());
}

/* ---- Integer-only backward at the kernel (arithmetic-bfp.md section 14) ----
 * avgPool1dBackwardKernelBfpRaw against a reference written from the
 * definition (BfpScatterRef.h helpers, output-centric): the FLOAT32 arm
 * scatters gy / K into every valid member of the window (count_include_pad:
 * the divisor stays K). Per dx row (one (batch, channel)): E(r) = (max
 * exponent over the loss row's NONZERO codes, 0 if none) + g, g = the guard
 * of C terms of m-bit codes, C = the windows covering one cell. A loss code
 * enters aligned to E(r) (left exact, right half away); a power-of-two K
 * folds -log2 K into the row's exponent, any other K divides the aligned
 * code ONCE per output, half away. The BFP_RAW is the data rule over the
 * sums (bfpRefRawFromRows). */

typedef struct apBwdCase {
    const int32_t *gy;
    const uint8_t *gyExps;
    size_t gyNumGroups, gyGroupSize;
    uint8_t m;
    size_t rows, len, k, dilation, stride;
    paddingType_t padding;
    size_t rawNumGroups, rawGroupSize;
} apBwdCase_t;

static uint32_t apLcg(uint32_t *s) {
    *s = *s * 1664525u + 1013904223u;
    return *s >> 8;
}

static void apRandomCodes(uint32_t seed, int32_t *codes, size_t n, uint8_t m) {
    uint32_t s = seed;
    uint32_t span = 1u << m;
    for (size_t i = 0; i < n; i++) {
        codes[i] = (int32_t)(apLcg(&s) % span) - (int32_t)(span / 2); /* -2^(m-1) included */
    }
}

static bfpRefStats_t refAvgPoolBackwardRaw(const apBwdCase_t *c, const windowGeometry1d_t *geom,
                                           int32_t *mant, int16_t *rawE) {
    bfpRefStats_t st = {.fitsInt32 = true};
    bfpRefOperand_t gy = {c->gy, c->gyExps, c->gyGroupSize, c->m, 8};
    size_t outLen = geom->outputLength;
    size_t cover = (c->dilation * (c->k - 1)) / c->stride + 1;
    int g = bfpRefGuard(cover, c->m, 1);
    int log2K = -1;
    for (int b = 0; b < 31; b++) {
        if ((size_t)1 << b == c->k) {
            log2K = b;
        }
    }
    int64_t sum[BFP_RAW_FIXTURE_MAX_OUT];
    int eAcc[BFP_RAW_FIXTURE_MAX_OUT];
    for (size_t r = 0; r < c->rows; r++) {
        int eRow = bfpRefEAcc(bfpRefMaxExp(&gy, r * outLen, outLen), g);
        for (size_t i = 0; i < c->len; i++) {
            sum[r * c->len + i] = 0;
            eAcc[r * c->len + i] = log2K >= 0 ? eRow - log2K : eRow;
        }
        for (size_t o = 0; o < outLen; o++) {
            size_t oi = r * outLen + o;
            int64_t term = bfpRefAlign(c->gy[oi], eRow - bfpRefExp(&gy, oi), &st);
            if (log2K < 0) {
                term = refDivHalfAway(term, (int64_t)c->k);
            }
            long long start = (long long)(o * c->stride) - (long long)geom->padLeft;
            for (size_t j = 0; j < c->k; j++) {
                long long idx = start + (long long)(j * c->dilation);
                if (idx >= 0 && idx < (long long)c->len) {
                    sum[r * c->len + (size_t)idx] += term;
                }
            }
        }
    }
    bfpRefRawFromRows(sum, eAcc, c->rows * c->len, c->rawGroupSize, mant, rawE, &st);
    return st;
}

static bfpRefStats_t checkAvgPoolBackwardBfpRaw(const apBwdCase_t *c) {
    kernel_t kern;
    initKernel(&kern, c->k, c->padding, c->dilation, c->stride);
    windowGeometry1d_t geom = windowGeometry1dCalc(c->len, &kern);
    size_t outLen = geom.outputLength;
    size_t gyDims[3] = {1, c->rows, outLen};
    size_t dxDims[3] = {1, c->rows, c->len};
    TEST_ASSERT_EQUAL_size_t(
        c->rows * outLen, c->gyGroupSize == 0 ? c->rows * outLen : c->gyNumGroups * c->gyGroupSize);
    bfpScratchFixture_t gy;
    bfpScratchInit(&gy, 3, gyDims, c->gy, c->gyExps, c->gyNumGroups, c->gyGroupSize, c->m, 8);
    bfpRawFixture_t raw;
    bfpRawInit(&raw, 3, dxDims, c->rawNumGroups, c->rawGroupSize);
    avgPool1dConfig_t cfg = {0};
    cfg.kernel = &kern;
    tensor_t *ops[1] = {&gy.tensor};

    avgPool1dBackwardKernelBfpRaw(ops, 1, &raw.tensor, NULL, &cfg, NULL);

    int32_t mant[BFP_RAW_FIXTURE_MAX_OUT];
    int16_t rawE[BFP_RAW_FIXTURE_MAX_GROUPS];
    bfpRefStats_t st = refAvgPoolBackwardRaw(c, &geom, mant, rawE);
    TEST_ASSERT_TRUE(st.fitsInt32);
    assertRawEquals(&raw, mant, c->rows * c->len, rawE, c->rawNumGroups);
    return st;
}

/* K 2 dilation 2 with overlap (stride 1, up to 3 windows per cell), K 4
 * with SAME padding (the divisor stays K at the edges), HAR's global pool
 * (K = stride = 32: one output per row, 32 cells each), and K 2 dilation 4
 * at m 16 (5 windows cover a cell); loss groups spread over 21 binades; raw
 * groups inside rows, across rows and per tensor. */
void testAvgPool1dBackwardBfpRawMatchesTheIntegerReference(void) {
    int32_t gy[64];
    uint8_t exps[16];
    uint32_t s = 5150u;
    for (size_t i = 0; i < 16; i++) {
        exps[i] = (uint8_t)(127 - 10 + (int)(apLcg(&s) % 21u));
    }
    apRandomCodes(1u, gy, 30, 8);
    apBwdCase_t a = {gy, exps, 6, 5, 8, 3, 12, 2, 2, 1, VALID, 9, 4};
    bfpRefStats_t st = checkAvgPoolBackwardBfpRaw(&a);
    TEST_ASSERT_TRUE(st.leftShift > 0);
    a.rawNumGroups = 4;
    a.rawGroupSize = 9;
    checkAvgPoolBackwardBfpRaw(&a);
    a.rawNumGroups = 1;
    a.rawGroupSize = 0;
    checkAvgPoolBackwardBfpRaw(&a);

    apRandomCodes(2u, gy, 18, 8); /* K 4, stride 2, SAME: 3 rows of 6 */
    apBwdCase_t same = {gy, exps, 6, 3, 8, 3, 12, 4, 1, 2, SAME, 6, 6};
    checkAvgPoolBackwardBfpRaw(&same);

    apRandomCodes(3u, gy, 3, 8); /* HAR: 3 rows of 32, K = stride = 32 */
    apBwdCase_t har = {gy, exps, 3, 1, 8, 3, 32, 32, 1, 32, VALID, 6, 16};
    checkAvgPoolBackwardBfpRaw(&har);
    har.gyNumGroups = 1;
    har.gyGroupSize = 0;
    har.rawNumGroups = 1;
    har.rawGroupSize = 0;
    checkAvgPoolBackwardBfpRaw(&har);

    apRandomCodes(4u, gy, 32, 16); /* K 2, dilation 4, m 16: 4 rows of 8 */
    apBwdCase_t dil = {gy, exps, 8, 4, 16, 4, 12, 2, 4, 1, VALID, 8, 6};
    checkAvgPoolBackwardBfpRaw(&dil);
}

/* One E per dx row: row 0's second loss group 30 binades below its first
 * rounds half away on the row's grid (beyond the guard). Row 2's first loss
 * group is all zero in the zero state (stored = bias) 30 binades ABOVE its
 * live second group: it must not set E (that row stays exact). K 2, and K 2
 * dilated to span 3. */
void testAvgPool1dBackwardBfpRawRowGridAndZeroGroups(void) {
    int32_t gy[16];
    apRandomCodes(21u, gy, 16, 8);
    gy[8] = 0;
    gy[9] = 0;
    gy[2] = 1;
    uint8_t exps[8] = {130, 100, 127, 126, 127, 97, 125, 125};
    apBwdCase_t c = {gy, exps, 8, 2, 8, 4, 8, 2, 1, 2, VALID, 8, 4};
    bfpRefStats_t st = checkAvgPoolBackwardBfpRaw(&c);
    TEST_ASSERT_TRUE_MESSAGE(st.rounded > 0, "the 30-binade row rounds");
    apBwdCase_t z = {gy + 8, exps + 4, 4, 2, 8, 2, 8, 2, 1, 2, VALID, 4, 4};
    st = checkAvgPoolBackwardBfpRaw(&z);
    TEST_ASSERT_EQUAL_size_t(0, st.rounded);
    apBwdCase_t z3 = {gy + 8, exps + 4, 4, 2, 8, 2, 12, 2, 2, 3, VALID, 4, 6};
    checkAvgPoolBackwardBfpRaw(&z3);

    /* A power-of-two K folds -log2 K into the EXPONENT, not into the code
     * grid: a loss group 22 binades below its row's top is still exact (g =
     * -23); aligning on a grid log2 K coarser would round its odd codes. */
    int32_t odd[8] = {101, -77, 5, -3, 63, 1, -9, 7};
    uint8_t gap[4] = {130, 108, 127, 105};
    apBwdCase_t k4 = {odd, gap, 4, 2, 8, 2, 16, 4, 1, 4, VALID, 2, 16};
    st = checkAvgPoolBackwardBfpRaw(&k4);
    TEST_ASSERT_EQUAL_size_t(0, st.rounded);
}

/* Layer level into a BFP dx wire: the integer raw re-blocked by the funnel's
 * epilogue (BfpScatterRef.h's section-13 re-block) by the OP's rounding
 * mode; under SR one draw per element in element order from the op's
 * RNG_TAG_OUT stream (op key = AVGPOOL1D_OP_PROP_LOSS = 1 under
 * RNG_TAG_BACKWARD). K 2 and K 4. */
static void apLayerIntoBfpWire(size_t kSize, roundingMode_t opMode, uint64_t seed) {
    kernel_t probe;
    initKernel(&probe, kSize, VALID, 1, 1);
    size_t outLen = windowGeometry1dCalc(12, &probe).outputLength;
    size_t gyDims[3] = {1, 3, outLen};
    size_t dxDims[3] = {1, 3, 12};
    int32_t gy[36];
    apRandomCodes(31u + (uint32_t)kSize, gy, 3 * outLen, 8);
    uint8_t exps[3] = {120, 131, 127};
    tensor_t *lossGrad = buildBfpWireWithCodes(gyDims, 3, 8, 8, 3, outLen, gy, exps);
    tensor_t *dx = buildBfpWireWithCodes(dxDims, 3, 6, 8, 9, 4, NULL, NULL);
    quantization_t *wireQ = quantizationInitBfpGrouped(6, 8, HALF_AWAY, 9, 4);
    kernel_t k;
    avgPool1dConfig_t cfg;
    layerConfig_t lc;
    layer_t layer;
    avgPool1dBuildBfpLayer(&cfg, &k, &lc, &layer, kSize, VALID, 1, 1, wireQ);
    cfg.propLossMath.roundingMode = opMode;

    layerExec_t exec = {.key = rngKeyFromSeed(seed)};
    avgPool1dBackward(&layer, NULL, lossGrad, dx, opMode == SR_HALF_AWAY ? &exec : NULL);

    int32_t codes[36];
    unpackSignExtend(dx->data, 6, 0, codes, 36);
    windowGeometry1d_t geom = windowGeometry1dCalc(12, &k);
    apBwdCase_t c = {gy, exps, 3, outLen, 8, 3, 12, kSize, 1, 1, VALID, 9, 4};
    int32_t mant[BFP_RAW_FIXTURE_MAX_OUT];
    int16_t rawE[BFP_RAW_FIXTURE_MAX_GROUPS];
    refAvgPoolBackwardRaw(&c, &geom, mant, rawE);
    rngStream_t out = rngOpen(
        rngFold(rngFold(rngFold(exec.key, RNG_TAG_BACKWARD, 0u), RNG_TAG_OP, 1u), RNG_TAG_OUT, 0u));
    int32_t refCodes[36];
    uint8_t refStored[9];
    bfpRefReblockRaw(mant, rawE, 36, 4, 6, 8, opMode == SR_HALF_AWAY ? &out : NULL, refCodes,
                     refStored);
    TEST_ASSERT_EQUAL_INT32_ARRAY(refCodes, codes, 36);
    TEST_ASSERT_EQUAL_UINT8_ARRAY(refStored, ((bfpQConfig_t *)dx->quantization->qConfig)->exponents,
                                  9);
    freeQuantization(wireQ);
    freeTensor(dx);
    freeTensor(lossGrad);
}

void testAvgPool1dBackwardBfpIntoBfpWireReblocksTheIntegerRaw(void) {
    apLayerIntoBfpWire(2, HALF_AWAY, 0u);
    apLayerIntoBfpWire(2, SR_HALF_AWAY, 811u);
    apLayerIntoBfpWire(4, HALF_AWAY, 0u);
    apLayerIntoBfpWire(4, SR_HALF_AWAY, 812u);
}

static void apFloatRawBody(void) {
    int32_t gy[4] = {1, 2, 3, 4};
    uint8_t exps[1] = {127};
    kernel_t kern;
    initKernel(&kern, 2, VALID, 1, 2);
    size_t gyDims[3] = {1, 1, 4};
    size_t dxDims[3] = {1, 1, 8};
    bfpScratchFixture_t g;
    bfpScratchInit(&g, 3, gyDims, gy, exps, 1, 0, 8, 8);
    tensor_t *notRaw = makeFloatTensor(dxDims, 3, NULL);
    avgPool1dConfig_t cfg = {0};
    cfg.kernel = &kern;
    tensor_t *ops[1] = {&g.tensor};
    avgPool1dBackwardKernelBfpRaw(ops, 1, notRaw, NULL, &cfg, NULL);
}

void testAvgPool1dBackwardBfpRawRejectsAFloatRaw(void) {
    ASSERT_EXITS_WITH_OUTPUT(
        1, "AvgPool1d backward BFP: needs a BFP (unpacked scratch) lossGrad and a BFP_RAW",
        apFloatRawBody());
}

/* BFP epic PR4 (R-P4 backward): the dx arm is FUNNEL-ROUTED, mirroring the
 * existing ARITH_SYM_INT32 arm — the prologue unpacks the BFP-stored lossGrad
 * and the OUT_WRITE epilogue writes propLoss. Into a FLOAT32 propLoss the
 * integer raw is per tensor ({1, 0}) and the epilogue dequantizes
 * ldexpf((float)mant, E) -- compared bit for bit with the integer reference
 * dequantized the same way. */
void testAvgPool1dBackwardBfpScattersIntoFloat32Raw(void) {
    size_t inputDims[] = {1, 2, 8};
    size_t outputDims[] = {1, 2, 4};
    int32_t gyCodes[8];
    for (size_t i = 0; i < kBfpAvgPoolGyCodes_len; i++) {
        gyCodes[i] = kBfpAvgPoolGyCodes[i];
    }
    tensor_t *lossGrad =
        buildBfpWireWithCodes(outputDims, 3, (uint8_t)kBfpAvgPoolMantissaBits,
                              (uint8_t)kBfpAvgPoolExponentBits, (size_t)kBfpAvgPoolOutNumGroups,
                              (size_t)kBfpAvgPoolOutGroupSize, gyCodes, kBfpAvgPoolGyExps);
    tensor_t *propLoss = makeFloatTensor(inputDims, 3, NULL);
    quantization_t *wireQ = quantizationInitBfpGrouped(
        (uint8_t)kBfpAvgPoolMantissaBits, (uint8_t)kBfpAvgPoolExponentBits, HALF_AWAY,
        (size_t)kBfpAvgPoolOutNumGroups, (size_t)kBfpAvgPoolOutGroupSize);
    kernel_t k;
    avgPool1dConfig_t cfg;
    layerConfig_t lc;
    layer_t layer;
    avgPool1dBuildBfpLayer(&cfg, &k, &lc, &layer, (size_t)kBfpAvgPoolKernelSize, SAME,
                           (size_t)kBfpAvgPoolDilation, (size_t)kBfpAvgPoolStride, wireQ);

    /* Called TWICE on purpose. dx is OUT_WRITE, so a repeated backward must
     * reproduce the same numbers — that contract is what this test pins. The
     * second call is also the best available observability for the kernel's
     * memset of the funnel's uninitialized Phase-2 scratch: that stack VLA can
     * read as zero (it holds whatever the prologue's callee frames left at
     * this stack depth), so a single call may not notice a missing memset.
     * Not a deterministic kill either way — the durable fix is funnel-level
     * zeroing of the Phase-2 raw (PR4 follow-up FU-1). */
    avgPool1dBackward(&layer, NULL, lossGrad, propLoss, NULL);
    avgPool1dBackward(&layer, NULL, lossGrad, propLoss, NULL);

    apBwdCase_t c = {gyCodes,
                     kBfpAvgPoolGyExps,
                     (size_t)kBfpAvgPoolOutNumGroups,
                     (size_t)kBfpAvgPoolOutGroupSize,
                     (uint8_t)kBfpAvgPoolMantissaBits,
                     2,
                     8,
                     (size_t)kBfpAvgPoolKernelSize,
                     (size_t)kBfpAvgPoolDilation,
                     (size_t)kBfpAvgPoolStride,
                     SAME,
                     1,
                     0};
    windowGeometry1d_t geom = windowGeometry1dCalc(8, &k);
    int32_t mant[BFP_RAW_FIXTURE_MAX_OUT];
    int16_t rawE[BFP_RAW_FIXTURE_MAX_GROUPS];
    refAvgPoolBackwardRaw(&c, &geom, mant, rawE);
    const float *got = (const float *)propLoss->data;
    float maxGold = 0.0f;
    for (size_t i = 0; i < kBfpAvgPoolExpectedBackward_len; i++) {
        maxGold = fmaxf(maxGold, fabsf(kBfpAvgPoolExpectedBackward[i]));
    }
    for (size_t i = 0; i < 16; i++) {
        TEST_ASSERT_EQUAL_FLOAT_MESSAGE(ldexpf((float)mant[i], rawE[0]), got[i],
                                        "the integer raw, dequantized");
        TEST_ASSERT_TRUE_MESSAGE(fabsf(got[i] - kBfpAvgPoolExpectedBackward[i]) <=
                                     ldexpf(maxGold, -22),
                                 "within the raw grid of the float scatter");
    }
    freeQuantization(wireQ);
    freeTensor(propLoss);
    freeTensor(lossGrad);
}

/* BFP epic PR4 (R-P7d): the rule-1 mirror. A pinned ARITH_BFP forward with a
 * NULL or non-BFP outputQ has no width anchor and must die EAGERLY — the
 * userApi factories copy layerQuant_t slots by value and never call
 * initAvgPool1dConfig, so both spellings are reachable. */
void testAvgPool1dForwardBfpRequiresBfpOutputQ(void) {
    size_t inputDims[] = {1, 2, 8};
    size_t outputDims[] = {1, 2, 4};
    tensor_t *input = makeFloatTensor(inputDims, 3, kBfpAvgPoolStageInput);
    tensor_t *output = makeFloatTensor(outputDims, 3, NULL);
    quantization_t *floatQ = quantizationInitFloat();

    kernel_t k;
    initKernel(&k, (size_t)kBfpAvgPoolKernelSize, SAME, (size_t)kBfpAvgPoolDilation,
               (size_t)kBfpAvgPoolStride);
    avgPool1dConfig_t nullCfg = {0};
    nullCfg.kernel = &k;
    nullCfg.forwardMath = (arithmetic_t){.type = ARITH_BFP, .roundingMode = HALF_AWAY};
    nullCfg.outputQ = NULL;
    layerConfig_t nullLc = {.avgPool1d = &nullCfg};
    layer_t nullLayer = {.type = AVGPOOL1D, .config = &nullLc};
    ASSERT_EXITS_WITH_FAILURE(avgPool1dForward(&nullLayer, input, output, NULL));

    avgPool1dConfig_t floatCfg = nullCfg;
    floatCfg.outputQ = floatQ;
    layerConfig_t floatLc = {.avgPool1d = &floatCfg};
    layer_t floatLayer = {.type = AVGPOOL1D, .config = &floatLc};
    ASSERT_EXITS_WITH_FAILURE(avgPool1dForward(&floatLayer, input, output, NULL));

    freeQuantization(floatQ);
    freeTensor(output);
    freeTensor(input);
}

/* BFP epic PR4: the shape gate on the NEW kernel. The write index is
 * (b * channels + c) * outputLength + outPos with batch/channels read off the
 * INPUT, so an output that matches on LENGTH but not on batch or channels is
 * written past its end — a checker that looks only at dimensions[2] passes it.
 * Three cases: matching length + wrong channels, matching length + wrong
 * batch, and wrong rank. The first case is the one the length-only check
 * misses; it is listed first so a regression is unmistakable. */
void testAvgPool1dForwardBfpRejectsMismatchedOutputShape(void) {
    size_t inputDims[] = {1, 2, 8};
    size_t wrongChannels[] = {1, 1, 4}; /* right length, wrong channels */
    size_t wrongBatch[] = {2, 2, 4};    /* right length + channels, wrong batch */
    size_t wrongRank[] = {2, 4};        /* right element count, rank 2 */
    int32_t inCodes[16];
    for (size_t i = 0; i < kBfpAvgPoolInCodes_len; i++) {
        inCodes[i] = kBfpAvgPoolInCodes[i];
    }
    tensor_t *input = buildBfpWireWithCodes(
        inputDims, 3, (uint8_t)kBfpAvgPoolMantissaBits, (uint8_t)kBfpAvgPoolExponentBits,
        (size_t)kBfpAvgPoolInNumGroups, (size_t)kBfpAvgPoolInGroupSize, inCodes, kBfpAvgPoolInExps);
    tensor_t *badChannels = makeFloatTensor(wrongChannels, 3, NULL);
    tensor_t *badBatch = makeFloatTensor(wrongBatch, 3, NULL);
    tensor_t *badRank = makeFloatTensor(wrongRank, 2, NULL);
    quantization_t *wireQ = quantizationInitBfpGrouped(
        (uint8_t)kBfpAvgPoolMantissaBits, (uint8_t)kBfpAvgPoolExponentBits, HALF_AWAY,
        (size_t)kBfpAvgPoolInNumGroups, (size_t)kBfpAvgPoolInGroupSize);
    kernel_t k;
    avgPool1dConfig_t cfg;
    layerConfig_t lc;
    layer_t layer;
    avgPool1dBuildBfpLayer(&cfg, &k, &lc, &layer, (size_t)kBfpAvgPoolKernelSize, SAME,
                           (size_t)kBfpAvgPoolDilation, (size_t)kBfpAvgPoolStride, wireQ);

    ASSERT_EXITS_WITH_FAILURE(avgPool1dForward(&layer, input, badChannels, NULL));
    ASSERT_EXITS_WITH_FAILURE(avgPool1dForward(&layer, input, badBatch, NULL));
    ASSERT_EXITS_WITH_FAILURE(avgPool1dForward(&layer, input, badRank, NULL));

    freeQuantization(wireQ);
    freeTensor(badRank);
    freeTensor(badBatch);
    freeTensor(badChannels);
    freeTensor(input);
}

/* The SUM headroom guard (bfpValidateSumHeadroom -- the sum limit
 * INT32_MAX >> (m-1), NOT the product limit). It cannot be tripped through a
 * legal AvgPool fixture: mantissaBits is capped at 16 at construction, so the
 * limit is at least 65535 codes per same-exponent segment, and the segment is
 * bounded by the kernel size -- a 65536-tap pool is not constructible on any
 * realistic input length. That is exactly the spec's statement that "with
 * typical block sizes the contract only binds at very wide mantissas". So the
 * test pins BOTH halves honestly: the kernel's call site does NOT fire for a
 * normal fixture, and the guard itself dies when its own bound is exceeded
 * (called directly -- it is a public static inline in BfpKernelSupport.h). */
void testAvgPool1dForwardBfpSumHeadroomGuardIsWired(void) {
    size_t inputDims[] = {1, 1, 8};
    size_t outputDims[] = {1, 1, 1};
    int32_t codes[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    uint8_t exps[1] = {127};
    tensor_t *input = buildBfpWireWithCodes(inputDims, 3, 16, 8, 1, 0, codes, exps);
    tensor_t *output = makeFloatTensor(outputDims, 3, NULL);
    quantization_t *wireQ = quantizationInitBfp(16, 8, HALF_AWAY);
    kernel_t k;
    avgPool1dConfig_t cfg;
    layerConfig_t lc;
    layer_t layer;
    avgPool1dBuildBfpLayer(&cfg, &k, &lc, &layer, 8, VALID, 1, 1, wireQ);

    /* bfpSumSegmentLimit(16) = INT32_MAX >> 15 = 65535; 8 terms is far inside
     * it, so the guarded call must SUCCEED (36 / 8 = 4.5 at scale 1.0). */
    avgPool1dForward(&layer, input, output, NULL);
    TEST_ASSERT_EQUAL_FLOAT(4.5f, ((float *)output->data)[0]);

    /* The guard's own bound, exercised directly. */
    uint8_t wideExps[1] = {127};
    bfpQConfig_t wideQC = {.exponents = wideExps,
                           .numGroups = 1,
                           .groupSize = 0,
                           .roundingMode = HALF_AWAY,
                           .mantissaBits = 16,
                           .exponentBits = 8};
    ASSERT_EXITS_WITH_FAILURE(bfpValidateSumHeadroom(&wideQC, 70000, "test"));

    freeQuantization(wireQ);
    freeTensor(output);
    freeTensor(input);
}

/* R-P7: the "BFP wire under a PINNED ARITH_FLOAT32" fake-quant path, the
 * direct mirror of testAvgPool1dForwardWithSymInt32Input. Pinning the math
 * slot (instead of deriving it) sends the BFP-stored operand through the
 * funnel's ARITH_FLOAT32 prologue, which dequants it via
 * conversionMatrix[BFP][FLOAT32]. Because the fixture is grid-exact
 * (mantissa * 2^E is a lossless float32 multiply) the bridge is EXACT, so this
 * asserts EQUALITY with the native run's gold — not a tolerance. */
void testAvgPool1dForwardBfpWireUnderPinnedFloat32(void) {
    size_t inputDims[] = {1, 2, 8};
    size_t outputDims[] = {1, 2, 4};
    int32_t inCodes[16];
    for (size_t i = 0; i < kBfpAvgPoolInCodes_len; i++) {
        inCodes[i] = kBfpAvgPoolInCodes[i];
    }
    tensor_t *input = buildBfpWireWithCodes(
        inputDims, 3, (uint8_t)kBfpAvgPoolMantissaBits, (uint8_t)kBfpAvgPoolExponentBits,
        (size_t)kBfpAvgPoolInNumGroups, (size_t)kBfpAvgPoolInGroupSize, inCodes, kBfpAvgPoolInExps);
    tensor_t *output = makeFloatTensor(outputDims, 3, NULL);
    quantization_t *wireQ = quantizationInitBfpGrouped(
        (uint8_t)kBfpAvgPoolMantissaBits, (uint8_t)kBfpAvgPoolExponentBits, HALF_AWAY,
        (size_t)kBfpAvgPoolInNumGroups, (size_t)kBfpAvgPoolInGroupSize);
    kernel_t k;
    avgPool1dConfig_t cfg;
    layerConfig_t lc;
    layer_t layer;
    avgPool1dBuildBfpLayer(&cfg, &k, &lc, &layer, (size_t)kBfpAvgPoolKernelSize, SAME,
                           (size_t)kBfpAvgPoolDilation, (size_t)kBfpAvgPoolStride, wireQ);
    cfg.forwardMath = (arithmetic_t){.type = ARITH_FLOAT32, .roundingMode = HALF_AWAY};

    avgPool1dForward(&layer, input, output, NULL);

    /* The fake-quant arm sums the exact floats and divides once in float32;
     * the integer native kernel folds -log2 K into the exponent. K is a power
     * of two (the only K BFP math takes), so both are exact and agree bit for
     * bit. */
    TEST_ASSERT_EQUAL_MEMORY_MESSAGE(kBfpAvgPoolExpectedFloatBridge, output->data,
                                     kBfpAvgPoolExpectedFloatBridge_len * sizeof(float),
                                     "the fake-quant bridge is the float32 mean of the exact "
                                     "dequantized values");
    TEST_ASSERT_EQUAL_MEMORY_MESSAGE(kBfpAvgPoolExpectedForward, output->data,
                                     kBfpAvgPoolExpectedForward_len * sizeof(float),
                                     "power-of-two K: the bridge equals the integer native result");
    freeQuantization(wireQ);
    freeTensor(output);
    freeTensor(input);
}

void setUp(void) {}
void tearDown(void) {}

/* BFP epic PR4 (R-P7d): the guard is NARROWED, not removed — a BFP wire under
 * the raw-casting FLOAT32/SYM arms must still die, and the ARITH_BFP backward
 * needs a BFP-typed propLossQ anchor (R-P1). */
void testAvgPool1dBackwardBfpGuardsNarrowedNotRemoved(void) {
    size_t inputDims[] = {1, 2, 8};
    size_t outputDims[] = {1, 2, 4};
    tensor_t *bfpLossGrad = buildBfpWireWithCodes(
        outputDims, 3, (uint8_t)kBfpAvgPoolMantissaBits, (uint8_t)kBfpAvgPoolExponentBits,
        (size_t)kBfpAvgPoolOutNumGroups, (size_t)kBfpAvgPoolOutGroupSize, NULL, NULL);
    tensor_t *floatLossGrad = makeFloatTensor(outputDims, 3, NULL);
    tensor_t *floatPropLoss = makeFloatTensor(inputDims, 3, NULL);
    tensor_t *bfpPropLoss = buildBfpWireWithCodes(
        inputDims, 3, (uint8_t)kBfpAvgPoolMantissaBits, (uint8_t)kBfpAvgPoolExponentBits,
        (size_t)kBfpAvgPoolInNumGroups, (size_t)kBfpAvgPoolInGroupSize, NULL, NULL);

    quantization_t *floatQ = quantizationInitFloat();
    kernel_t kf;
    avgPool1dConfig_t floatCfg;
    layerConfig_t floatLc;
    layer_t floatLayer;
    avgPool1dBuildBfpLayer(&floatCfg, &kf, &floatLc, &floatLayer, (size_t)kBfpAvgPoolKernelSize,
                           SAME, (size_t)kBfpAvgPoolDilation, (size_t)kBfpAvgPoolStride, floatQ);
    ASSERT_EXITS_WITH_FAILURE(
        avgPool1dBackward(&floatLayer, NULL, bfpLossGrad, floatPropLoss, NULL));
    ASSERT_EXITS_WITH_FAILURE(
        avgPool1dBackward(&floatLayer, NULL, floatLossGrad, bfpPropLoss, NULL));

    kernel_t kb;
    initKernel(&kb, (size_t)kBfpAvgPoolKernelSize, SAME, (size_t)kBfpAvgPoolDilation,
               (size_t)kBfpAvgPoolStride);
    avgPool1dConfig_t bfpCfg = {0};
    bfpCfg.kernel = &kb;
    bfpCfg.propLossMath = (arithmetic_t){.type = ARITH_BFP, .roundingMode = HALF_AWAY};
    bfpCfg.propLossQ = NULL;
    layerConfig_t bfpLc = {.avgPool1d = &bfpCfg};
    layer_t bfpLayer = {.type = AVGPOOL1D, .config = &bfpLc};
    ASSERT_EXITS_WITH_FAILURE(avgPool1dBackward(&bfpLayer, NULL, bfpLossGrad, floatPropLoss, NULL));

    freeQuantization(floatQ);
    freeTensor(bfpPropLoss);
    freeTensor(floatPropLoss);
    freeTensor(floatLossGrad);
    freeTensor(bfpLossGrad);
}

/* BFP epic PR4, the OPERAND side of the rank gate. executeOp never
 * inspects operand rank (it sizes the raw from the TARGET only), so a rank-2
 * BFP operand reaches the kernel, which would read dimensions[0..2] off a
 * two-element dims array — an over-read of the shape itself. Both kernels
 * therefore open with their own rank check BEFORE poolRequireDims3, and
 * these two deaths are the only ones that can reach it: every other rank-2
 * case in this file passes a rank-2 OUTPUT, which dies in poolRequireDims3
 * first. Operands carry no codes — the guard fires before any data read.
 *
 * MUTATION STATUS, stated honestly: DELETING either rank guard does NOT redden
 * this test. The statements right after it feed the over-read dims[0..2] into
 * poolRequireDims3 (and, in the backward, the geometry check), which
 * exit(1) all the same — and the death harness compares exit codes only, with
 * no message capture, so it cannot tell the two deaths apart. That is defense
 * in depth, not coverage. What IS proven: giving each rank guard a distinct
 * exit code makes exactly the matching assertion below fail, so both deaths
 * really do originate in the rank guards and no rank-2 operand reaches the
 * indexing loops. */
void testAvgPool1dBfpRejectsRank2Operands(void) {
    size_t rank2Input[] = {2, 8};
    size_t rank2LossGrad[] = {2, 4};
    size_t inputDims[] = {1, 2, 8};
    size_t outputDims[] = {1, 2, 4};
    tensor_t *badInput = buildBfpWireWithCodes(
        rank2Input, 2, (uint8_t)kBfpAvgPoolMantissaBits, (uint8_t)kBfpAvgPoolExponentBits,
        (size_t)kBfpAvgPoolInNumGroups, (size_t)kBfpAvgPoolInGroupSize, NULL, NULL);
    tensor_t *badLossGrad = buildBfpWireWithCodes(
        rank2LossGrad, 2, (uint8_t)kBfpAvgPoolMantissaBits, (uint8_t)kBfpAvgPoolExponentBits,
        (size_t)kBfpAvgPoolOutNumGroups, (size_t)kBfpAvgPoolOutGroupSize, NULL, NULL);
    tensor_t *output = makeFloatTensor(outputDims, 3, NULL);
    tensor_t *propLoss = makeFloatTensor(inputDims, 3, NULL);
    quantization_t *wireQ = quantizationInitBfpGrouped(
        (uint8_t)kBfpAvgPoolMantissaBits, (uint8_t)kBfpAvgPoolExponentBits, HALF_AWAY,
        (size_t)kBfpAvgPoolInNumGroups, (size_t)kBfpAvgPoolInGroupSize);
    kernel_t k;
    avgPool1dConfig_t cfg;
    layerConfig_t lc;
    layer_t layer;
    avgPool1dBuildBfpLayer(&cfg, &k, &lc, &layer, (size_t)kBfpAvgPoolKernelSize, SAME,
                           (size_t)kBfpAvgPoolDilation, (size_t)kBfpAvgPoolStride, wireQ);

    ASSERT_EXITS_WITH_FAILURE(avgPool1dForward(&layer, badInput, output, NULL));
    ASSERT_EXITS_WITH_FAILURE(avgPool1dBackward(&layer, NULL, badLossGrad, propLoss, NULL));

    freeQuantization(wireQ);
    freeTensor(propLoss);
    freeTensor(output);
    freeTensor(badLossGrad);
    freeTensor(badInput);
}

/* The SR forward rounds in the OUT_WRITE epilogue of its forward op: OP 0
 * under the FORWARD phase of the layer key. The state from the Python mirror,
 * tags as numbers (FORWARD 4, OP 7, OUT 10):
 *   uv run python -c "from examples._shared.keyed_rng import fold, key_from_seed as kfs,
 *   open_stream as o; print(hex(o(fold(fold(fold(kfs(0xa002), 4, 0), 7, 0), 10, 0))))" */
#define AVGPOOL1D_KEYED_SEED 0xa002u
#define AVGPOOL1D_KEYED_OUT_STATE 0xbafc192fu

/* AvgPool1d(k = 2, stride 1) over 4 channels of length 5: 16 outputs, each
 * rounded SR into an 8-bit SYM_INT32 output by its own draw. */
void testAvgPool1dSrForwardIsAPureFunctionOfItsForwardOpKey(void) {
    size_t inputDims[] = {1, 4, 5};
    kernel_t kernel;
    initKernel(&kernel, 2, VALID, 1, 1);
    quantization_t *q = quantizationInitFloat();
    avgPool1dConfig_t cfg;
    initAvgPool1dConfig(&cfg, &kernel, q, q);
    layerConfig_t lc = {.avgPool1d = &cfg};
    layer_t layer = {.type = AVGPOOL1D, .config = &lc};
    float values[20];
    for (size_t i = 0; i < 20; i++) {
        values[i] = 0.37f * (float)((i * 7) % 20) - 3.1f;
    }
    tensor_t *input = makeFloatTensor(inputDims, 3, values);
    keyedSymForward_t f = {.layer = &layer,
                           .input = input,
                           .outDims = {1, 4, 4},
                           .outRank = 3,
                           .forwardRounding = &cfg.forwardMath.roundingMode};

    assertKeyedSymForwardDrawsFrom(&f, AVGPOOL1D_KEYED_SEED,
                                   keyedOpOutState(AVGPOOL1D_KEYED_SEED, RNG_TAG_FORWARD, 0u),
                                   AVGPOOL1D_KEYED_OUT_STATE);

    freeTensor(input);
    freeQuantization(q);
}

/* #440: see testLinearInitConfigClearsOwnsQuantizations. */
void testAvgPool1dInitConfigClearsOwnsQuantizations(void) {
    kernel_t kernel;
    initKernel(&kernel, 2, VALID, 1, 1);
    avgPool1dConfig_t cfg;
    memset(&cfg, 0xFF, sizeof cfg);
    initAvgPool1dConfig(&cfg, &kernel, NULL, NULL);
    TEST_ASSERT_EQUAL_UINT8(0, *(const unsigned char *)&cfg.ownsQuantizations);
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(testAvgPool1dForwardBasic);
    RUN_TEST(testAvgPool1dBackwardBasic);
    RUN_TEST(testAvgPool1dMultiChannel);
    RUN_TEST(testAvgPool1dMultiBatch);
    RUN_TEST(testAvgPool1dWithStrideAndDilation);
    RUN_TEST(testAvgPool1dWithSamePadding);
    RUN_TEST(testAvgPool1dEdgeCases);
    RUN_TEST(testAvgPool1dBackwardFloatOverwritesStalePropLoss);
    RUN_TEST(testAvgPool1dBackwardFloatRejectsPropLossWithFewerChannels);
    RUN_TEST(testAvgPool1dBackwardFloatRejectsRank2PropLoss);
    RUN_TEST(testAvgPool1dForwardFloatRejectsOutputWithSmallerBatch);
    RUN_TEST(testAvgPool1dBackwardSymRejectsPropLossWithSmallerBatch);
    RUN_TEST(testAvgPool1dBackwardSymRejectsRank2PropLoss);
    RUN_TEST(testAvgPool1dForwardSymRejectsOutputWithFewerChannels);
    RUN_TEST(testAvgPool1dForwardWithSymInt32Input);
    RUN_TEST(testAvgPool1dForwardSymBasic);
    RUN_TEST(testAvgPool1dBackwardSymBasic);
    RUN_TEST(testAvgPool1dSymStrideDilationForwardBackward);
    RUN_TEST(testAvgPool1dSymSamePadding);
    RUN_TEST(testAvgPool1dForwardSymRejectsWideOperand);
    RUN_TEST(testAvgPool1dBackwardSymRejectsTermsOverBound);
    RUN_TEST(testAvgPool1dForwardBfpBorrowedGroupedInput);
    RUN_TEST(testAvgPool1dForwardBfpStagesFloat32InputAtWireWidths);
    RUN_TEST(testAvgPool1dForwardBfpPacksOutputWire);
    RUN_TEST(testAvgPool1dBfpRawPow2KFoldsIntoTheExponent);
    RUN_TEST(testAvgPool1dBfpRawDilatedPow2KAcrossGroups);
    RUN_TEST(testAvgPool1dBfpRejectsANonPowerOfTwoKernel);
    RUN_TEST(testAvgPool1dBfpLog2DivisorOfPowersOfTwo);
    RUN_TEST(testAvgPool1dBfpRawWorstCaseCodesAcrossExponentGaps);
    RUN_TEST(testAvgPool1dBfpRawRejectsANonRawOutput);
    RUN_TEST(testAvgPool1dBackwardBfpRawMatchesTheIntegerReference);
    RUN_TEST(testAvgPool1dBackwardBfpRawRowGridAndZeroGroups);
    RUN_TEST(testAvgPool1dBackwardBfpIntoBfpWireReblocksTheIntegerRaw);
    RUN_TEST(testAvgPool1dBackwardBfpRawRejectsAFloatRaw);
    RUN_TEST(testAvgPool1dBfpRawZeroGroupDoesNotSetTheAccumulator);
    RUN_TEST(testAvgPool1dForwardBfpRequiresBfpOutputQ);
    RUN_TEST(testAvgPool1dForwardBfpRejectsMismatchedOutputShape);
    RUN_TEST(testAvgPool1dForwardBfpSumHeadroomGuardIsWired);
    RUN_TEST(testAvgPool1dForwardBfpWireUnderPinnedFloat32);
    RUN_TEST(testAvgPool1dBackwardBfpScattersIntoFloat32Raw);
    RUN_TEST(testAvgPool1dBackwardBfpGuardsNarrowedNotRemoved);
    RUN_TEST(testAvgPool1dBfpRejectsRank2Operands);
    RUN_TEST(testAvgPool1dSrForwardIsAPureFunctionOfItsForwardOpKey);
    RUN_TEST(testAvgPool1dInitConfigClearsOwnsQuantizations);
    return UNITY_END();
}
