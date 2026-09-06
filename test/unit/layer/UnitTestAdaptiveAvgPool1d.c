#define SOURCE_FILE "UNIT_TEST_ADAPTIVE_AVG_POOL_1D"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "AdaptiveAvgPool1d.h"
#include "DeathTest.h"
#include "Layer.h"
#include "Quantization.h"
#include "QuantizationApi.h"
#include "StorageApi.h"
#include "TensorApi.h"
#include "TensorConversion.h"
#include "expected_adaptive_avg_pool_1d.h"
#include "unity.h"

void setUp(void) {}
void tearDown(void) {}

typedef struct adaptivePoolRun {
    layer_t *layer;
    tensor_t *input;
    tensor_t *output;
    quantization_t *q;
} adaptivePoolRun_t;

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

static adaptivePoolRun_t build(float const *inputData, size_t const *inputDims, size_t outputSize,
                               float *outputBuf, size_t const *outputDims) {
    static adaptiveAvgPool1dConfig_t cfgStore;
    static layer_t layerStore;
    static layerConfig_t lcStore;

    quantization_t *q = quantizationInitFloat();
    initAdaptiveAvgPool1dConfig(&cfgStore, outputSize, q, q);

    lcStore.adaptiveAvgPool1d = &cfgStore;
    layerStore.config = &lcStore;

    adaptivePoolRun_t r = {0};
    r.layer = &layerStore;
    r.input = makeFloatTensor(inputDims, 3, inputData);
    r.output = makeFloatTensor(outputDims, 3, NULL);
    (void)outputBuf;
    r.q = q;
    return r;
}

void testForwardBasic(void) {
    size_t inDims[] = {1, 1, 4};
    size_t outDims[] = {1, 1, 2};
    float outData[1 * 1 * 2] = {0};
    adaptivePoolRun_t r = build(input_adaptiveAvgPool1d_basic, inDims, 2, outData, outDims);

    adaptiveAvgPool1dForward(r.layer, r.input, r.output);

    for (size_t i = 0; i < expectedForward_adaptiveAvgPool1d_basic_len; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-5f, expectedForward_adaptiveAvgPool1d_basic[i],
                                 ((float *)r.output->data)[i]);
    }
}

void testForwardMultiChannelOverlap(void) {
    size_t inDims[] = {1, 3, 5};
    size_t outDims[] = {1, 3, 2};
    float outData[1 * 3 * 2] = {0};
    adaptivePoolRun_t r = build(input_adaptiveAvgPool1d_multiChannel, inDims, 2, outData, outDims);

    adaptiveAvgPool1dForward(r.layer, r.input, r.output);

    for (size_t i = 0; i < expectedForward_adaptiveAvgPool1d_multiChannel_len; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-5f, expectedForward_adaptiveAvgPool1d_multiChannel[i],
                                 ((float *)r.output->data)[i]);
    }
}

void testForwardMultiBatch(void) {
    size_t inDims[] = {4, 2, 6};
    size_t outDims[] = {4, 2, 4};
    float outData[4 * 2 * 4] = {0};
    adaptivePoolRun_t r = build(input_adaptiveAvgPool1d_multiBatch, inDims, 4, outData, outDims);

    adaptiveAvgPool1dForward(r.layer, r.input, r.output);

    for (size_t i = 0; i < expectedForward_adaptiveAvgPool1d_multiBatch_len; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-5f, expectedForward_adaptiveAvgPool1d_multiBatch[i],
                                 ((float *)r.output->data)[i]);
    }
}

void testForwardGlobal(void) {
    size_t inDims[] = {1, 2, 7};
    size_t outDims[] = {1, 2, 1};
    float outData[1 * 2 * 1] = {0};
    adaptivePoolRun_t r = build(input_adaptiveAvgPool1d_global, inDims, 1, outData, outDims);

    adaptiveAvgPool1dForward(r.layer, r.input, r.output);

    for (size_t i = 0; i < expectedForward_adaptiveAvgPool1d_global_len; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-5f, expectedForward_adaptiveAvgPool1d_global[i],
                                 ((float *)r.output->data)[i]);
    }
}

void testForwardIdentity(void) {
    size_t inDims[] = {1, 1, 4};
    size_t outDims[] = {1, 1, 4};
    float outData[1 * 1 * 4] = {0};
    adaptivePoolRun_t r = build(input_adaptiveAvgPool1d_identity, inDims, 4, outData, outDims);

    adaptiveAvgPool1dForward(r.layer, r.input, r.output);

    for (size_t i = 0; i < expectedForward_adaptiveAvgPool1d_identity_len; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-5f, expectedForward_adaptiveAvgPool1d_identity[i],
                                 ((float *)r.output->data)[i]);
    }
}

void testForwardUpsample(void) {
    size_t inDims[] = {1, 1, 3};
    size_t outDims[] = {1, 1, 5};
    float outData[1 * 1 * 5] = {0};
    adaptivePoolRun_t r = build(input_adaptiveAvgPool1d_upsample, inDims, 5, outData, outDims);

    adaptiveAvgPool1dForward(r.layer, r.input, r.output);

    for (size_t i = 0; i < expectedForward_adaptiveAvgPool1d_upsample_len; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-5f, expectedForward_adaptiveAvgPool1d_upsample[i],
                                 ((float *)r.output->data)[i]);
    }
}

void testBackwardBasic(void) {
    size_t inDims[] = {1, 1, 4};
    size_t outDims[] = {1, 1, 2};
    float outData[1 * 1 * 2] = {0};
    adaptivePoolRun_t r = build(input_adaptiveAvgPool1d_basic, inDims, 2, outData, outDims);
    adaptiveAvgPool1dForward(r.layer, r.input, r.output);

    float gyData[1 * 1 * 2] = {1.0f, 1.0f};
    tensor_t *lossGrad = makeFloatTensor(outDims, 3, gyData);
    tensor_t *propLoss = makeFloatTensor(inDims, 3, NULL);

    adaptiveAvgPool1dBackward(r.layer, r.input, lossGrad, propLoss);

    for (size_t i = 0; i < expectedPropLoss_adaptiveAvgPool1d_basic_len; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-5f, expectedPropLoss_adaptiveAvgPool1d_basic[i],
                                 ((float *)propLoss->data)[i]);
    }
}

void testBackwardMultiChannelOverlap(void) {
    size_t inDims[] = {1, 3, 5};
    size_t outDims[] = {1, 3, 2};
    float outData[1 * 3 * 2] = {0};
    adaptivePoolRun_t r = build(input_adaptiveAvgPool1d_multiChannel, inDims, 2, outData, outDims);
    adaptiveAvgPool1dForward(r.layer, r.input, r.output);

    tensor_t *lossGrad = makeFloatTensor(outDims, 3, lossGrad_adaptiveAvgPool1d_multiChannel);
    tensor_t *propLoss = makeFloatTensor(inDims, 3, NULL);

    adaptiveAvgPool1dBackward(r.layer, r.input, lossGrad, propLoss);

    for (size_t i = 0; i < expectedPropLoss_adaptiveAvgPool1d_multiChannel_len; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-5f, expectedPropLoss_adaptiveAvgPool1d_multiChannel[i],
                                 ((float *)propLoss->data)[i]);
    }
}

void testBackwardMultiBatch(void) {
    size_t inDims[] = {4, 2, 6};
    size_t outDims[] = {4, 2, 4};
    float outData[4 * 2 * 4] = {0};
    adaptivePoolRun_t r = build(input_adaptiveAvgPool1d_multiBatch, inDims, 4, outData, outDims);
    adaptiveAvgPool1dForward(r.layer, r.input, r.output);

    tensor_t *lossGrad = makeFloatTensor(outDims, 3, lossGrad_adaptiveAvgPool1d_multiBatch);
    tensor_t *propLoss = makeFloatTensor(inDims, 3, NULL);

    adaptiveAvgPool1dBackward(r.layer, r.input, lossGrad, propLoss);

    for (size_t i = 0; i < expectedPropLoss_adaptiveAvgPool1d_multiBatch_len; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-5f, expectedPropLoss_adaptiveAvgPool1d_multiBatch[i],
                                 ((float *)propLoss->data)[i]);
    }
}

void testBackwardGlobal(void) {
    size_t inDims[] = {1, 2, 7};
    size_t outDims[] = {1, 2, 1};
    float outData[1 * 2 * 1] = {0};
    adaptivePoolRun_t r = build(input_adaptiveAvgPool1d_global, inDims, 1, outData, outDims);
    adaptiveAvgPool1dForward(r.layer, r.input, r.output);

    float gyData[1 * 2 * 1] = {1.0f, 1.0f};
    tensor_t *lossGrad = makeFloatTensor(outDims, 3, gyData);
    tensor_t *propLoss = makeFloatTensor(inDims, 3, NULL);

    adaptiveAvgPool1dBackward(r.layer, r.input, lossGrad, propLoss);

    for (size_t i = 0; i < expectedPropLoss_adaptiveAvgPool1d_global_len; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-5f, expectedPropLoss_adaptiveAvgPool1d_global[i],
                                 ((float *)propLoss->data)[i]);
    }
}

void testBackwardUpsample(void) {
    size_t inDims[] = {1, 1, 3};
    size_t outDims[] = {1, 1, 5};
    float outData[1 * 1 * 5] = {0};
    adaptivePoolRun_t r = build(input_adaptiveAvgPool1d_upsample, inDims, 5, outData, outDims);
    adaptiveAvgPool1dForward(r.layer, r.input, r.output);

    tensor_t *lossGrad = makeFloatTensor(outDims, 3, lossGrad_adaptiveAvgPool1d_upsample);
    tensor_t *propLoss = makeFloatTensor(inDims, 3, NULL);

    adaptiveAvgPool1dBackward(r.layer, r.input, lossGrad, propLoss);

    for (size_t i = 0; i < expectedPropLoss_adaptiveAvgPool1d_upsample_len; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-5f, expectedPropLoss_adaptiveAvgPool1d_upsample[i],
                                 ((float *)propLoss->data)[i]);
    }
}

void testCalcOutputShapeFixedRegardlessOfInput(void) {
    static adaptiveAvgPool1dConfig_t cfgStore;
    static layer_t layerStore;
    static layerConfig_t lcStore;
    quantization_t *q = quantizationInitFloat();
    initAdaptiveAvgPool1dConfig(&cfgStore, 3, q, q);
    lcStore.adaptiveAvgPool1d = &cfgStore;
    layerStore.config = &lcStore;

    size_t inDims[3] = {1, 5, 20};
    size_t inOrder[3];
    setOrderOfDimsForNewTensor(3, inOrder);
    shape_t inShape;
    setShape(&inShape, inDims, 3, inOrder);

    size_t outDims[3] = {0, 0, 0};
    size_t outOrder[3] = {0, 0, 0};
    shape_t outShape;
    outShape.dimensions = outDims;
    outShape.numberOfDimensions = 3;
    outShape.orderOfDimensions = outOrder;

    adaptiveAvgPool1dCalcOutputShape(&layerStore, &inShape, &outShape);

    TEST_ASSERT_EQUAL_UINT(3, outShape.numberOfDimensions);
    TEST_ASSERT_EQUAL_UINT(1, outShape.dimensions[0]);
    TEST_ASSERT_EQUAL_UINT(5, outShape.dimensions[1]);
    TEST_ASSERT_EQUAL_UINT(3, outShape.dimensions[2]);
    freeQuantization(q);
}

// Smoke test for the funnel's new SYM-input capability (spec Testing list) —
// same rationale as AvgPool1d's twin test (UnitTestAvgPool1d.c): forwardMath
// stays FLOAT32 (no SYM kernel body), but a SYM_INT32 producer tensor must
// now be dequantized by the executeOp prologue rather than reinterpreted as
// raw float bits. absMax(1,2,3,4)=4.0 -> scale=4/2047; per-element error
// <= 0.5*scale ~ 9.8e-4 -> 5e-3 leaves a >5x margin.
void testAdaptiveAvgPool1dForwardWithSymInt32Input(void) {
    size_t inDims[] = {1, 1, 4};
    size_t outDims[] = {1, 1, 2};

    static adaptiveAvgPool1dConfig_t cfgStore;
    static layer_t layerStore;
    static layerConfig_t lcStore;

    quantization_t *floatQ = quantizationInitFloat();
    initAdaptiveAvgPool1dConfig(&cfgStore, 2, floatQ, floatQ);
    lcStore.adaptiveAvgPool1d = &cfgStore;
    layerStore.config = &lcStore;

    size_t *ownedDims = reserveMemory(3 * sizeof(size_t));
    memcpy(ownedDims, inDims, 3 * sizeof(size_t));
    size_t *order = reserveMemory(3 * sizeof(size_t));
    setOrderOfDimsForNewTensor(3, order);
    shape_t *shape = reserveMemory(sizeof(shape_t));
    setShape(shape, ownedDims, 3, order);
    tensor_t *symInput = initTensor(shape, quantizationInitSymInt32WithBits(HALF_AWAY, 12), NULL);
    tensorFillFromFloatBuffer(symInput, input_adaptiveAvgPool1d_basic,
                              calcNumberOfElementsByTensor(symInput));

    tensor_t *output = makeFloatTensor(outDims, 3, NULL);

    adaptiveAvgPool1dForward(&layerStore, symInput, output);

    for (size_t i = 0; i < expectedForward_adaptiveAvgPool1d_basic_len; i++) {
        TEST_ASSERT_FLOAT_WITHIN(5e-3f, expectedForward_adaptiveAvgPool1d_basic[i],
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
    assertSymTensorMatchesGold(                                                                    \
        t, expectedForwardMantissas_adaptiveAvgPool1dSym_##fix,                                    \
        expectedForwardMantissas_adaptiveAvgPool1dSym_##fix##_len,                                 \
        expectedForwardScale_adaptiveAvgPool1dSym_##fix, mantissaTol_adaptiveAvgPool1dSym_##fix,   \
        scaleTol_adaptiveAvgPool1dSym_##fix, expectedForwardDequant_adaptiveAvgPool1dSym_##fix,    \
        forwardDequantTol_adaptiveAvgPool1dSym_##fix)

#define ASSERT_SYM_PROPLOSS_GOLD(t, fix)                                                           \
    assertSymTensorMatchesGold(                                                                    \
        t, expectedPropLossMantissas_adaptiveAvgPool1dSym_##fix,                                   \
        expectedPropLossMantissas_adaptiveAvgPool1dSym_##fix##_len,                                \
        expectedPropLossScale_adaptiveAvgPool1dSym_##fix, mantissaTol_adaptiveAvgPool1dSym_##fix,  \
        scaleTol_adaptiveAvgPool1dSym_##fix, expectedPropLossDequant_adaptiveAvgPool1dSym_##fix,   \
        propLossDequantTol_adaptiveAvgPool1dSym_##fix)

/* Mirrors build() but on SYM_INT32 wires: forwardQ/propLossQ declare
 * ARITH_SYM_INT32 compute via arithmeticFromQuantizationOrDefault. */
static adaptivePoolRun_t buildSym(float const *inputData, size_t const *inputDims,
                                  size_t outputSize, size_t const *outputDims) {
    static adaptiveAvgPool1dConfig_t cfgStore;
    static layer_t layerStore;
    static layerConfig_t lcStore;

    quantization_t *q = quantizationInitSymInt32(HALF_AWAY);
    initAdaptiveAvgPool1dConfig(&cfgStore, outputSize, q, q);

    lcStore.adaptiveAvgPool1d = &cfgStore;
    layerStore.config = &lcStore;

    adaptivePoolRun_t r = {0};
    r.layer = &layerStore;
    r.input = buildSymTensor(inputDims, 3, inputData);
    r.output = buildSymTensor(outputDims, 3, NULL);
    r.q = q;
    return r;
}

void testForwardBackwardSymUneven(void) {
    size_t inDims[] = {1, 1, 5};
    size_t outDims[] = {1, 1, 3};

    // L=5 -> O=3: window counts 2/3/2 with overlaps — varying divisor is the
    // core adaptive case; the symUneven gold set contains an exact .5-tie
    // (sum 981, count 2), pinning half-away-from-zero integer division.
    adaptivePoolRun_t r = buildSym(input_adaptiveAvgPool1dSym_symUneven, inDims, 3, outDims);

    adaptiveAvgPool1dForward(r.layer, r.input, r.output);
    ASSERT_SYM_FORWARD_GOLD(r.output, symUneven);

    tensor_t *lossGrad = buildSymTensor(outDims, 3, lossGrad_adaptiveAvgPool1dSym_symUneven);
    tensor_t *propLoss = buildSymTensor(inDims, 3, NULL);

    adaptiveAvgPool1dBackward(r.layer, r.input, lossGrad, propLoss);
    ASSERT_SYM_PROPLOSS_GOLD(propLoss, symUneven);
}

void testForwardBackwardSymGlobal(void) {
    size_t inDims[] = {1, 2, 6};
    size_t outDims[] = {1, 2, 1};

    // Global average: count = L, the largest divisor.
    adaptivePoolRun_t r = buildSym(input_adaptiveAvgPool1dSym_symGlobal, inDims, 1, outDims);

    adaptiveAvgPool1dForward(r.layer, r.input, r.output);
    ASSERT_SYM_FORWARD_GOLD(r.output, symGlobal);

    tensor_t *lossGrad = buildSymTensor(outDims, 3, lossGrad_adaptiveAvgPool1dSym_symGlobal);
    tensor_t *propLoss = buildSymTensor(inDims, 3, NULL);

    adaptiveAvgPool1dBackward(r.layer, r.input, lossGrad, propLoss);
    ASSERT_SYM_PROPLOSS_GOLD(propLoss, symGlobal);
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

/* Value-sum guard, N branch: the global-pool window (outputSize=1) sums
 * inputLength terms — at qMaxBits=16 the bound is 2^(32-16) = 65536, so
 * L=65536 must fail fast instead of silently overflowing the int32 sum. */
void testForwardSymRejectsTermsOverBound(void) {
    size_t inDims[] = {1, 1, 65536};
    size_t outDims[] = {1, 1, 1};

    adaptivePoolRun_t r = buildSym(NULL, inDims, 1, outDims);
    r.input = buildSymTensorWithBits(inDims, 3, 16);

    ASSERT_EXITS_WITH_FAILURE(adaptiveAvgPool1dForward(r.layer, r.input, r.output));
}

/* Value-sum guard, width branch: a 31-bit loss-grad mantissa can overflow the
 * int32 scatter accumulator after 2 covering windows — the backward kernel
 * must fail fast on qMaxBits > 16. */
void testBackwardSymRejectsWideLossGrad(void) {
    size_t inDims[] = {1, 1, 5};
    size_t outDims[] = {1, 1, 3};

    adaptivePoolRun_t r = buildSym(input_adaptiveAvgPool1dSym_symUneven, inDims, 3, outDims);

    tensor_t *lossGrad = buildSymTensorWithBits(outDims, 3, 31);
    tensor_t *propLoss = buildSymTensor(inDims, 3, NULL);

    ASSERT_EXITS_WITH_FAILURE(adaptiveAvgPool1dBackward(r.layer, r.input, lossGrad, propLoss));
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
 * propLossQ for the backward. initAdaptiveAvgPool1dConfig derives both math
 * slots from it, so passing a BFP wireQ selects the native arms. */
static layer_t *adaptiveBuildBfpLayer(adaptiveAvgPool1dConfig_t *cfgStore, layerConfig_t *lcStore,
                                      layer_t *layerStore, size_t outputSize,
                                      quantization_t *wireQ) {
    initAdaptiveAvgPool1dConfig(cfgStore, outputSize, wireQ, wireQ);
    layerStore->type = ADAPTIVE_AVGPOOL1D;
    lcStore->adaptiveAvgPool1d = cfgStore;
    layerStore->config = lcStore;
    return layerStore;
}

/* Borrowed BFP-stored grouped input (D8: never re-blocked): the kernel folds
 * same-group segments with ldexpf into the FLOAT32 raw (D7) and divides by
 * EACH window's own count there — L=8, O=3 gives counts 3/4/3, so a constant
 * divisor is observably wrong. A FLOAT32 output wire makes the OUT_WRITE
 * epilogue a memmove, so the comparison is bit-exact. */
void testAdaptiveAvgPool1dForwardBfpBorrowedGroupedInput(void) {
    size_t inputDims[] = {1, 2, 8};
    size_t outputDims[] = {1, 2, 3};
    int32_t inCodes[16];
    for (size_t i = 0; i < kBfpAdaptiveInCodes_len; i++) {
        inCodes[i] = kBfpAdaptiveInCodes[i];
    }
    tensor_t *input =
        buildBfpWireWithCodes(inputDims, 3, (uint8_t)kBfpAdaptiveMantissaBits,
                              (uint8_t)kBfpAdaptiveExponentBits, (size_t)kBfpAdaptiveInNumGroups,
                              (size_t)kBfpAdaptiveInGroupSize, inCodes, kBfpAdaptiveInExps);
    tensor_t *output = makeFloatTensor(outputDims, 3, NULL);
    quantization_t *wireQ = quantizationInitBfpGrouped(
        (uint8_t)kBfpAdaptiveMantissaBits, (uint8_t)kBfpAdaptiveExponentBits, HALF_AWAY,
        (size_t)kBfpAdaptiveInNumGroups, (size_t)kBfpAdaptiveInGroupSize);
    adaptiveAvgPool1dConfig_t cfg;
    layerConfig_t lc;
    layer_t layer;
    adaptiveBuildBfpLayer(&cfg, &lc, &layer, (size_t)kBfpAdaptiveOutputSize, wireQ);

    adaptiveAvgPool1dForward(&layer, input, output);

    TEST_ASSERT_EQUAL_MEMORY_MESSAGE(kBfpAdaptiveExpectedForward, output->data,
                                     kBfpAdaptiveExpectedForward_len * sizeof(float),
                                     "each adaptive window divides by ITS OWN count in float32");
    freeQuantization(wireQ);
    freeTensor(output);
    freeTensor(input);
}

void testAdaptiveAvgPool1dBackwardBfpScattersIntoFloat32Raw(void) {
    size_t inputDims[] = {1, 2, 8};
    size_t outputDims[] = {1, 2, 3};
    int32_t gyCodes[6];
    for (size_t i = 0; i < kBfpAdaptiveGyCodes_len; i++) {
        gyCodes[i] = kBfpAdaptiveGyCodes[i];
    }
    tensor_t *lossGrad =
        buildBfpWireWithCodes(outputDims, 3, (uint8_t)kBfpAdaptiveMantissaBits,
                              (uint8_t)kBfpAdaptiveExponentBits, (size_t)kBfpAdaptiveGyNumGroups,
                              (size_t)kBfpAdaptiveGyGroupSize, gyCodes, kBfpAdaptiveGyExps);
    tensor_t *propLoss = makeFloatTensor(inputDims, 3, NULL);
    quantization_t *wireQ = quantizationInitBfpGrouped(
        (uint8_t)kBfpAdaptiveMantissaBits, (uint8_t)kBfpAdaptiveExponentBits, HALF_AWAY,
        (size_t)kBfpAdaptiveGyNumGroups, (size_t)kBfpAdaptiveGyGroupSize);
    adaptiveAvgPool1dConfig_t cfg;
    layerConfig_t lc;
    layer_t layer;
    adaptiveBuildBfpLayer(&cfg, &lc, &layer, (size_t)kBfpAdaptiveOutputSize, wireQ);

    /* Called TWICE on purpose: dx is OUT_WRITE, so a repeated backward must
     * reproduce the same numbers — that contract is what this test pins. The
     * second call is also the best available observability for the kernel's
     * memset of the funnel's uninitialized Phase-2 scratch: that stack VLA can
     * read as zero (it holds whatever the prologue's callee frames left at
     * this stack depth), so a single call may not notice a missing memset. For
     * THIS layer the doubled call did not kill the dropped-memset mutant in
     * one build — the observability is stack-layout dependent. The durable fix
     * is funnel-level zeroing of the Phase-2 raw (PR4 follow-up FU-1). */
    adaptiveAvgPool1dBackward(&layer, NULL, lossGrad, propLoss);
    adaptiveAvgPool1dBackward(&layer, NULL, lossGrad, propLoss);

    TEST_ASSERT_EQUAL_MEMORY_MESSAGE(kBfpAdaptiveExpectedBackward, propLoss->data,
                                     kBfpAdaptiveExpectedBackward_len * sizeof(float),
                                     "dx spreads each output cell by 1/count into the float raw");
    freeQuantization(wireQ);
    freeTensor(propLoss);
    freeTensor(lossGrad);
}

/* BFP epic PR4 (F5): both new kernels index their raw output as a DENSE
 * [batch][channels][length] array with batch/channels read off the OPERAND, so
 * a raw output that matches on LENGTH but not on batch or channels is written
 * past its end — a checker that looks only at dimensions[2] passes it. Covers
 * both directions and both spellings, plus a rank-2 tensor on either entry. */
void testAdaptiveAvgPool1dBfpRejectsMismatchedShapes(void) {
    size_t inputDims[] = {1, 2, 8};
    size_t outputDims[] = {1, 2, 3};
    size_t wrongChannels[] = {1, 1, 3}; /* right length, wrong channels */
    size_t wrongBatch[] = {2, 2, 3};    /* right length + channels, wrong batch */
    size_t wrongPropChannels[] = {1, 1, 8};
    size_t wrongRank[] = {2, 3};
    int32_t inCodes[16];
    for (size_t i = 0; i < kBfpAdaptiveInCodes_len; i++) {
        inCodes[i] = kBfpAdaptiveInCodes[i];
    }
    int32_t gyCodes[6];
    for (size_t i = 0; i < kBfpAdaptiveGyCodes_len; i++) {
        gyCodes[i] = kBfpAdaptiveGyCodes[i];
    }
    tensor_t *input =
        buildBfpWireWithCodes(inputDims, 3, (uint8_t)kBfpAdaptiveMantissaBits,
                              (uint8_t)kBfpAdaptiveExponentBits, (size_t)kBfpAdaptiveInNumGroups,
                              (size_t)kBfpAdaptiveInGroupSize, inCodes, kBfpAdaptiveInExps);
    tensor_t *lossGrad =
        buildBfpWireWithCodes(outputDims, 3, (uint8_t)kBfpAdaptiveMantissaBits,
                              (uint8_t)kBfpAdaptiveExponentBits, (size_t)kBfpAdaptiveGyNumGroups,
                              (size_t)kBfpAdaptiveGyGroupSize, gyCodes, kBfpAdaptiveGyExps);
    tensor_t *badChannels = makeFloatTensor(wrongChannels, 3, NULL);
    tensor_t *badBatch = makeFloatTensor(wrongBatch, 3, NULL);
    tensor_t *badPropLoss = makeFloatTensor(wrongPropChannels, 3, NULL);
    tensor_t *badRank = makeFloatTensor(wrongRank, 2, NULL);
    quantization_t *wireQ = quantizationInitBfpGrouped(
        (uint8_t)kBfpAdaptiveMantissaBits, (uint8_t)kBfpAdaptiveExponentBits, HALF_AWAY,
        (size_t)kBfpAdaptiveInNumGroups, (size_t)kBfpAdaptiveInGroupSize);
    adaptiveAvgPool1dConfig_t cfg;
    layerConfig_t lc;
    layer_t layer;
    adaptiveBuildBfpLayer(&cfg, &lc, &layer, (size_t)kBfpAdaptiveOutputSize, wireQ);

    ASSERT_EXITS_WITH_FAILURE(adaptiveAvgPool1dForward(&layer, input, badChannels));
    ASSERT_EXITS_WITH_FAILURE(adaptiveAvgPool1dForward(&layer, input, badBatch));
    ASSERT_EXITS_WITH_FAILURE(adaptiveAvgPool1dBackward(&layer, NULL, lossGrad, badPropLoss));
    ASSERT_EXITS_WITH_FAILURE(adaptiveAvgPool1dForward(&layer, input, badRank));

    freeQuantization(wireQ);
    freeTensor(badRank);
    freeTensor(badPropLoss);
    freeTensor(badBatch);
    freeTensor(badChannels);
    freeTensor(lossGrad);
    freeTensor(input);
}

/* BFP epic PR4 (R-P7d): the guard is NARROWED, not removed — a BFP wire under
 * the raw-casting FLOAT32/SYM arms must still die, and both ARITH_BFP arms need
 * a BFP-typed produced-wire anchor (R-P1: outputQ forward, propLossQ dx). */
void testAdaptiveAvgPool1dBackwardBfpGuardsNarrowedNotRemoved(void) {
    size_t inputDims[] = {1, 2, 8};
    size_t outputDims[] = {1, 2, 3};
    tensor_t *bfpLossGrad = buildBfpWireWithCodes(
        outputDims, 3, (uint8_t)kBfpAdaptiveMantissaBits, (uint8_t)kBfpAdaptiveExponentBits,
        (size_t)kBfpAdaptiveGyNumGroups, (size_t)kBfpAdaptiveGyGroupSize, NULL, NULL);
    tensor_t *floatLossGrad = makeFloatTensor(outputDims, 3, NULL);
    tensor_t *floatPropLoss = makeFloatTensor(inputDims, 3, NULL);
    tensor_t *floatOutput = makeFloatTensor(outputDims, 3, NULL);
    tensor_t *bfpPropLoss = buildBfpWireWithCodes(
        inputDims, 3, (uint8_t)kBfpAdaptiveMantissaBits, (uint8_t)kBfpAdaptiveExponentBits,
        (size_t)kBfpAdaptiveInNumGroups, (size_t)kBfpAdaptiveInGroupSize, NULL, NULL);
    tensor_t *floatInput = makeFloatTensor(inputDims, 3, NULL);

    quantization_t *floatQ = quantizationInitFloat();
    adaptiveAvgPool1dConfig_t floatCfg;
    layerConfig_t floatLc;
    layer_t floatLayer;
    adaptiveBuildBfpLayer(&floatCfg, &floatLc, &floatLayer, (size_t)kBfpAdaptiveOutputSize, floatQ);
    ASSERT_EXITS_WITH_FAILURE(
        adaptiveAvgPool1dBackward(&floatLayer, NULL, bfpLossGrad, floatPropLoss));
    ASSERT_EXITS_WITH_FAILURE(
        adaptiveAvgPool1dBackward(&floatLayer, NULL, floatLossGrad, bfpPropLoss));

    adaptiveAvgPool1dConfig_t bfpCfg = {0};
    bfpCfg.outputSize = (size_t)kBfpAdaptiveOutputSize;
    bfpCfg.forwardMath = (arithmetic_t){.type = ARITH_BFP, .roundingMode = HALF_AWAY};
    bfpCfg.propLossMath = (arithmetic_t){.type = ARITH_BFP, .roundingMode = HALF_AWAY};
    bfpCfg.outputQ = NULL;
    bfpCfg.propLossQ = NULL;
    layerConfig_t bfpLc = {.adaptiveAvgPool1d = &bfpCfg};
    layer_t bfpLayer = {.type = ADAPTIVE_AVGPOOL1D, .config = &bfpLc};
    ASSERT_EXITS_WITH_FAILURE(
        adaptiveAvgPool1dBackward(&bfpLayer, NULL, bfpLossGrad, floatPropLoss));
    ASSERT_EXITS_WITH_FAILURE(adaptiveAvgPool1dForward(&bfpLayer, floatInput, floatOutput));

    /* The NON-NULL, non-BFP spelling of the same hole (the AvgPool/MaxPool
     * twins pin it too): a FLOAT32 produced-wire config is just as anchor-less
     * as a NULL one — it carries no mantissa/exponent widths — so a guard that
     * only NULL-checks would walk into staging with a bogus anchor. */
    adaptiveAvgPool1dConfig_t floatWireCfg = bfpCfg;
    floatWireCfg.outputQ = floatQ;
    floatWireCfg.propLossQ = floatQ;
    layerConfig_t floatWireLc = {.adaptiveAvgPool1d = &floatWireCfg};
    layer_t floatWireLayer = {.type = ADAPTIVE_AVGPOOL1D, .config = &floatWireLc};
    ASSERT_EXITS_WITH_FAILURE(
        adaptiveAvgPool1dBackward(&floatWireLayer, NULL, bfpLossGrad, floatPropLoss));
    ASSERT_EXITS_WITH_FAILURE(adaptiveAvgPool1dForward(&floatWireLayer, floatInput, floatOutput));

    freeQuantization(floatQ);
    freeTensor(floatInput);
    freeTensor(bfpPropLoss);
    freeTensor(floatOutput);
    freeTensor(floatPropLoss);
    freeTensor(floatLossGrad);
    freeTensor(bfpLossGrad);
}

/* BFP epic PR4 (F5), the OPERAND side of the rank gate. executeOp never
 * inspects operand rank (it sizes the raw from the TARGET only), so a rank-2
 * BFP operand reaches the kernel, which would read dimensions[0..2] off a
 * two-element dims array — an over-read of the shape itself. Both kernels
 * therefore open with their own rank check BEFORE poolBfpRequireDims3, and
 * these two deaths are the only ones that can reach it: the rank-2 case in
 * testAdaptiveAvgPool1dBfpRejectsMismatchedShapes passes a rank-2 OUTPUT,
 * which dies in poolBfpRequireDims3 first.
 *
 * MUTATION STATUS, stated honestly: DELETING either rank guard does NOT redden
 * this test — the over-read dims[0..2] then reach poolBfpRequireDims3, which
 * exit(1)s all the same, and the death harness compares exit codes only.
 * Defense in depth, not coverage. What IS proven: giving each rank guard a
 * distinct exit code makes exactly the matching assertion below fail, so both
 * deaths really do originate in the rank guards. */
void testAdaptiveAvgPool1dBfpRejectsRank2Operands(void) {
    size_t rank2Input[] = {2, 8};
    size_t rank2LossGrad[] = {2, 3};
    size_t inputDims[] = {1, 2, 8};
    size_t outputDims[] = {1, 2, 3};
    tensor_t *badInput = buildBfpWireWithCodes(
        rank2Input, 2, (uint8_t)kBfpAdaptiveMantissaBits, (uint8_t)kBfpAdaptiveExponentBits,
        (size_t)kBfpAdaptiveInNumGroups, (size_t)kBfpAdaptiveInGroupSize, NULL, NULL);
    tensor_t *badLossGrad = buildBfpWireWithCodes(
        rank2LossGrad, 2, (uint8_t)kBfpAdaptiveMantissaBits, (uint8_t)kBfpAdaptiveExponentBits,
        (size_t)kBfpAdaptiveGyNumGroups, (size_t)kBfpAdaptiveGyGroupSize, NULL, NULL);
    tensor_t *output = makeFloatTensor(outputDims, 3, NULL);
    tensor_t *propLoss = makeFloatTensor(inputDims, 3, NULL);
    quantization_t *wireQ = quantizationInitBfpGrouped(
        (uint8_t)kBfpAdaptiveMantissaBits, (uint8_t)kBfpAdaptiveExponentBits, HALF_AWAY,
        (size_t)kBfpAdaptiveInNumGroups, (size_t)kBfpAdaptiveInGroupSize);
    adaptiveAvgPool1dConfig_t cfg;
    layerConfig_t lc;
    layer_t layer;
    adaptiveBuildBfpLayer(&cfg, &lc, &layer, (size_t)kBfpAdaptiveOutputSize, wireQ);

    ASSERT_EXITS_WITH_FAILURE(adaptiveAvgPool1dForward(&layer, badInput, output));
    ASSERT_EXITS_WITH_FAILURE(adaptiveAvgPool1dBackward(&layer, NULL, badLossGrad, propLoss));

    freeQuantization(wireQ);
    freeTensor(propLoss);
    freeTensor(output);
    freeTensor(badLossGrad);
    freeTensor(badInput);
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(testForwardBasic);
    RUN_TEST(testForwardMultiChannelOverlap);
    RUN_TEST(testForwardMultiBatch);
    RUN_TEST(testForwardGlobal);
    RUN_TEST(testForwardIdentity);
    RUN_TEST(testForwardUpsample);
    RUN_TEST(testCalcOutputShapeFixedRegardlessOfInput);
    RUN_TEST(testBackwardBasic);
    RUN_TEST(testBackwardMultiChannelOverlap);
    RUN_TEST(testBackwardMultiBatch);
    RUN_TEST(testBackwardGlobal);
    RUN_TEST(testBackwardUpsample);
    RUN_TEST(testAdaptiveAvgPool1dForwardWithSymInt32Input);
    RUN_TEST(testForwardBackwardSymUneven);
    RUN_TEST(testForwardBackwardSymGlobal);
    RUN_TEST(testForwardSymRejectsTermsOverBound);
    RUN_TEST(testBackwardSymRejectsWideLossGrad);
    RUN_TEST(testAdaptiveAvgPool1dForwardBfpBorrowedGroupedInput);
    RUN_TEST(testAdaptiveAvgPool1dBackwardBfpScattersIntoFloat32Raw);
    RUN_TEST(testAdaptiveAvgPool1dBfpRejectsMismatchedShapes);
    RUN_TEST(testAdaptiveAvgPool1dBackwardBfpGuardsNarrowedNotRemoved);
    RUN_TEST(testAdaptiveAvgPool1dBfpRejectsRank2Operands);
    return UNITY_END();
}
