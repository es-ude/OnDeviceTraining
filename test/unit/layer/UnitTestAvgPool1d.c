#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "AvgPool1d.h"
#include "DeathTest.h"
#include "Layer.h"
#include "QuantizationApi.h"
#include "StorageApi.h"
#include "TensorApi.h"
#include "expected_avg_pool_1d.h"
#include "unity.h"

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

/* BFP epic PR2 Task 8: BFP wire (per-tensor {1,0}, 8-bit mantissas). The buffer
 * is BFP-SIZED, so an unguarded float* access runs past it. */
static tensor_t *makeBfpTensor(size_t const *dims, size_t numDims) {
    size_t *ownedDims = reserveMemory(numDims * sizeof(size_t));
    memcpy(ownedDims, dims, numDims * sizeof(size_t));
    size_t *order = reserveMemory(numDims * sizeof(size_t));
    setOrderOfDimsForNewTensor(numDims, order);
    shape_t *shape = reserveMemory(sizeof(shape_t));
    setShape(shape, ownedDims, numDims, order);
    return initTensor(shape, quantizationInitBfp(8, 8, HALF_AWAY), NULL);
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

    avgPool1dForward(r.layer, r.input, r.output);

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

    avgPool1dForward(r.layer, r.input, r.output);

    float lossGradData[1 * 1 * 3];
    for (size_t i = 0; i < 3; i++) {
        lossGradData[i] = 1.0f;
    }
    tensor_t *lossGrad = makeFloatTensor(outputDims, 3, lossGradData);

    tensor_t *propLoss = makeFloatTensor(inputDims, 3, NULL);

    avgPool1dBackward(r.layer, r.input, lossGrad, propLoss);

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

    avgPool1dForward(r.layer, r.input, r.output);
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

    avgPool1dBackward(r.layer, r.input, lossGrad, propLoss);
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

    avgPool1dForward(r.layer, r.input, r.output);
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

    avgPool1dBackward(r.layer, r.input, lossGrad, propLoss);
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

    avgPool1dForward(r.layer, r.input, r.output);
    for (size_t i = 0; i < expectedForward_avgPool1d_withStrideAndDilation_len; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-5f, expectedForward_avgPool1d_withStrideAndDilation[i],
                                 ((float *)r.output->data)[i]);
    }

    tensor_t *lossGrad = makeFloatTensor(outputDims, 3, lossGrad_avgPool1d_withStrideAndDilation);

    tensor_t *propLoss = makeFloatTensor(inputDims, 3, NULL);

    avgPool1dBackward(r.layer, r.input, lossGrad, propLoss);
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

    avgPool1dForward(r.layer, r.input, r.output);
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

    avgPool1dBackward(r.layer, r.input, lossGrad, propLoss);
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

    avgPool1dForward(r.layer, r.input, r.output);
    // input [1,2,3,4] mean = 2.5
    for (size_t i = 0; i < expectedForward_avgPool1d_edgeCases_len; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-5f, expectedForward_avgPool1d_edgeCases[i],
                                 ((float *)r.output->data)[i]);
    }

    float lossGradData[1] = {1.0f};
    tensor_t *lossGrad = makeFloatTensor(outputDims, 3, lossGradData);

    tensor_t *propLoss = makeFloatTensor(inputDims, 3, NULL);

    avgPool1dBackward(r.layer, r.input, lossGrad, propLoss);
    // 1/K = 0.25 contribution to each of 4 input positions.
    for (size_t i = 0; i < expectedPropLoss_avgPool1d_edgeCases_len; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-5f, expectedPropLoss_avgPool1d_edgeCases[i],
                                 ((float *)propLoss->data)[i]);
    }
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

    avgPool1dForward(&layerStore, symInput, output);

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

    avgPool1dForward(r.layer, r.input, r.output);

    ASSERT_SYM_FORWARD_GOLD(r.output, symBasic);
}

void testAvgPool1dBackwardSymBasic(void) {
    size_t inputDims[] = {1, 1, 4};
    size_t outputDims[] = {1, 1, 3};

    avgPool1dSymRun_t r =
        avgPool1dBuildSym(input_avgPool1dSym_symBasic, inputDims, 2, VALID, 1, 1, outputDims);

    tensor_t *lossGrad = buildSymTensor(outputDims, 3, lossGrad_avgPool1dSym_symBasic);
    tensor_t *propLoss = buildSymTensor(inputDims, 3, NULL);

    avgPool1dBackward(r.layer, r.input, lossGrad, propLoss);

    ASSERT_SYM_PROPLOSS_GOLD(propLoss, symBasic);
}

void testAvgPool1dSymStrideDilationForwardBackward(void) {
    size_t inputDims[] = {1, 1, 9};
    size_t outputDims[] = {1, 1, 3};

    // K=2, stride=3, dilation=2 — random gold lossGrad so positional mutations
    // on the SYM scatter path are non-vacuous.
    avgPool1dSymRun_t r = avgPool1dBuildSym(input_avgPool1dSym_symStrideDilation, inputDims, 2,
                                            VALID, 2, 3, outputDims);

    avgPool1dForward(r.layer, r.input, r.output);
    ASSERT_SYM_FORWARD_GOLD(r.output, symStrideDilation);

    tensor_t *lossGrad = buildSymTensor(outputDims, 3, lossGrad_avgPool1dSym_symStrideDilation);
    tensor_t *propLoss = buildSymTensor(inputDims, 3, NULL);

    avgPool1dBackward(r.layer, r.input, lossGrad, propLoss);
    ASSERT_SYM_PROPLOSS_GOLD(propLoss, symStrideDilation);
}

void testAvgPool1dSymSamePadding(void) {
    size_t inputDims[] = {1, 1, 5};
    size_t outputDims[] = {1, 1, 5};

    // Edge windows have validCount=2 but the scale fold keeps the divisor at
    // K=3 — pins count_include_pad=true on the SYM path.
    avgPool1dSymRun_t r =
        avgPool1dBuildSym(input_avgPool1dSym_symSamePadding, inputDims, 3, SAME, 1, 1, outputDims);

    avgPool1dForward(r.layer, r.input, r.output);
    ASSERT_SYM_FORWARD_GOLD(r.output, symSamePadding);

    tensor_t *lossGrad = buildSymTensor(outputDims, 3, lossGrad_avgPool1dSym_symSamePadding);
    tensor_t *propLoss = buildSymTensor(inputDims, 3, NULL);

    avgPool1dBackward(r.layer, r.input, lossGrad, propLoss);
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

    ASSERT_EXITS_WITH_FAILURE(avgPool1dForward(r.layer, r.input, r.output));
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

    ASSERT_EXITS_WITH_FAILURE(avgPool1dBackward(r.layer, r.input, lossGrad, propLoss));
}

void setUp(void) {}
void tearDown(void) {}

/* BFP epic PR2 Task 8: avgPool1dBackward's ARITH_FLOAT32 arm runs outside
 * executeOp and raw-casts lossGrad/propLoss to float*. Task 8 made BFP dx wires
 * allocatable and pre-flip they select exactly that arm -- guard the storage
 * dtype. */
void testAvgPool1dBackwardRejectsBfpWire(void) {
    size_t inputDims[] = {1, 1, 4};
    size_t outputDims[] = {1, 1, 3};
    float outputData[1 * 1 * 3] = {0};
    avgPool1dRunResult_t r =
        avgPool1dBuild(input_avgPool1d_basic, inputDims, 2, VALID, 1, 1, outputData, outputDims);

    tensor_t *lossGrad = makeFloatTensor(outputDims, 3, NULL);
    tensor_t *propLoss = makeFloatTensor(inputDims, 3, NULL);
    tensor_t *bfpLossGrad = makeBfpTensor(outputDims, 3);
    tensor_t *bfpPropLoss = makeBfpTensor(inputDims, 3);

    ASSERT_EXITS_WITH_FAILURE(avgPool1dBackward(r.layer, r.input, bfpLossGrad, propLoss));
    ASSERT_EXITS_WITH_FAILURE(avgPool1dBackward(r.layer, r.input, lossGrad, bfpPropLoss));

    freeTensor(bfpPropLoss);
    freeTensor(bfpLossGrad);
    freeTensor(propLoss);
    freeTensor(lossGrad);
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(testAvgPool1dBackwardRejectsBfpWire);
    RUN_TEST(testAvgPool1dForwardBasic);
    RUN_TEST(testAvgPool1dBackwardBasic);
    RUN_TEST(testAvgPool1dMultiChannel);
    RUN_TEST(testAvgPool1dMultiBatch);
    RUN_TEST(testAvgPool1dWithStrideAndDilation);
    RUN_TEST(testAvgPool1dWithSamePadding);
    RUN_TEST(testAvgPool1dEdgeCases);
    RUN_TEST(testAvgPool1dForwardWithSymInt32Input);
    RUN_TEST(testAvgPool1dForwardSymBasic);
    RUN_TEST(testAvgPool1dBackwardSymBasic);
    RUN_TEST(testAvgPool1dSymStrideDilationForwardBackward);
    RUN_TEST(testAvgPool1dSymSamePadding);
    RUN_TEST(testAvgPool1dForwardSymRejectsWideOperand);
    RUN_TEST(testAvgPool1dBackwardSymRejectsTermsOverBound);
    return UNITY_END();
}
