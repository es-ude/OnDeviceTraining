#define SOURCE_FILE "UNIT_TEST_AVG_POOL_1D"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "AvgPool1d.h"
#include "BfpKernelSupport.h"
#include "DeathTest.h"
#include "Layer.h"
#include "Quantization.h"
#include "QuantizationApi.h"
#include "StorageApi.h"
#include "TensorApi.h"
#include "TensorConversion.h"
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

/* Borrowed BFP-stored grouped input (D8: never re-blocked): the kernel folds
 * same-group segments with ldexpf into the FLOAT32 raw (D7) and divides by K
 * there. A FLOAT32 output wire makes the OUT_WRITE epilogue a memmove, so the
 * comparison is BIT-exact against the np.float32 reference. */
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
    avgPool1dBuildBfpLayer(&cfg, &k, &lc, &layer, (size_t)kBfpAvgPoolKernelSize, VALID,
                           (size_t)kBfpAvgPoolDilation, (size_t)kBfpAvgPoolStride, wireQ);

    avgPool1dForward(&layer, input, output);

    TEST_ASSERT_EQUAL_MEMORY_MESSAGE(kBfpAvgPoolExpectedForward, output->data,
                                     kBfpAvgPoolExpectedForward_len * sizeof(float),
                                     "ARITH_BFP raw is FLOAT32 (D7) and the FLOAT32 OUT_WRITE "
                                     "is a memmove -- the output must be bit-exact");
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
    avgPool1dBuildBfpLayer(&cfg, &k, &lc, &layer, (size_t)kBfpAvgPoolKernelSize, VALID,
                           (size_t)kBfpAvgPoolDilation, (size_t)kBfpAvgPoolStride, wireQ);

    avgPool1dForward(&layer, input, output);

    TEST_ASSERT_EQUAL_MEMORY_MESSAGE(kBfpAvgPoolExpectedStaged, output->data,
                                     kBfpAvgPoolExpectedStaged_len * sizeof(float),
                                     "the FLOAT32 operand must stage PER-TENSOR at the "
                                     "produced-wire widths (R-P1)");
    freeQuantization(wireQ);
    freeTensor(output);
    freeTensor(input);
}

/* OUT_WRITE onto a BFP wire: the epilogue derives the target's per-group
 * exponents (packFloatBufferAsBfp) with the OP's rounding (#282). */
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
    avgPool1dBuildBfpLayer(&cfg, &k, &lc, &layer, (size_t)kBfpAvgPoolKernelSize, VALID,
                           (size_t)kBfpAvgPoolDilation, (size_t)kBfpAvgPoolStride, wireQ);

    avgPool1dForward(&layer, input, output);

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
    initKernel(&k, (size_t)kBfpAvgPoolKernelSize, VALID, (size_t)kBfpAvgPoolDilation,
               (size_t)kBfpAvgPoolStride);
    avgPool1dConfig_t nullCfg = {0};
    nullCfg.kernel = &k;
    nullCfg.forwardMath = (arithmetic_t){.type = ARITH_BFP, .roundingMode = HALF_AWAY};
    nullCfg.outputQ = NULL;
    layerConfig_t nullLc = {.avgPool1d = &nullCfg};
    layer_t nullLayer = {.type = AVGPOOL1D, .config = &nullLc};
    ASSERT_EXITS_WITH_FAILURE(avgPool1dForward(&nullLayer, input, output));

    avgPool1dConfig_t floatCfg = nullCfg;
    floatCfg.outputQ = floatQ;
    layerConfig_t floatLc = {.avgPool1d = &floatCfg};
    layer_t floatLayer = {.type = AVGPOOL1D, .config = &floatLc};
    ASSERT_EXITS_WITH_FAILURE(avgPool1dForward(&floatLayer, input, output));

    freeQuantization(floatQ);
    freeTensor(output);
    freeTensor(input);
}

/* BFP epic PR4 (F5): the shape gate on the NEW kernel. The write index is
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
    avgPool1dBuildBfpLayer(&cfg, &k, &lc, &layer, (size_t)kBfpAvgPoolKernelSize, VALID,
                           (size_t)kBfpAvgPoolDilation, (size_t)kBfpAvgPoolStride, wireQ);

    ASSERT_EXITS_WITH_FAILURE(avgPool1dForward(&layer, input, badChannels));
    ASSERT_EXITS_WITH_FAILURE(avgPool1dForward(&layer, input, badBatch));
    ASSERT_EXITS_WITH_FAILURE(avgPool1dForward(&layer, input, badRank));

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
    avgPool1dForward(&layer, input, output);
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
    avgPool1dBuildBfpLayer(&cfg, &k, &lc, &layer, (size_t)kBfpAvgPoolKernelSize, VALID,
                           (size_t)kBfpAvgPoolDilation, (size_t)kBfpAvgPoolStride, wireQ);
    cfg.forwardMath = (arithmetic_t){.type = ARITH_FLOAT32, .roundingMode = HALF_AWAY};

    avgPool1dForward(&layer, input, output);

    /* The native kernel folds int32 partials then divides; the fake-quant arm
     * sums exact floats then divides. Both reduce the SAME exact values in the
     * SAME order, so on this grid-exact fixture the results are identical. */
    TEST_ASSERT_EQUAL_MEMORY_MESSAGE(kBfpAvgPoolExpectedForward, output->data,
                                     kBfpAvgPoolExpectedForward_len * sizeof(float),
                                     "the fake-quant bridge must reproduce the native result "
                                     "on a grid-exact fixture");
    freeQuantization(wireQ);
    freeTensor(output);
    freeTensor(input);
}

void setUp(void) {}
void tearDown(void) {}

/* BFP epic PR2 Task 8: avgPool1dBackward's ARITH_FLOAT32 arm runs outside
 * executeOp and raw-casts lossGrad/propLoss to float*. Task 8 made BFP dx wires
 * allocatable, and an ARITH_FLOAT32 propLossMath -- pinned, or derived as such
 * before the Task 9 flip -- selects exactly that arm; guard the storage dtype. */
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
    RUN_TEST(testAvgPool1dForwardBfpBorrowedGroupedInput);
    RUN_TEST(testAvgPool1dForwardBfpStagesFloat32InputAtWireWidths);
    RUN_TEST(testAvgPool1dForwardBfpPacksOutputWire);
    RUN_TEST(testAvgPool1dForwardBfpRequiresBfpOutputQ);
    RUN_TEST(testAvgPool1dForwardBfpRejectsMismatchedOutputShape);
    RUN_TEST(testAvgPool1dForwardBfpSumHeadroomGuardIsWired);
    RUN_TEST(testAvgPool1dForwardBfpWireUnderPinnedFloat32);
    return UNITY_END();
}
