#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "AdaptiveAvgPool1d.h"
#include "DeathTest.h"
#include "Layer.h"
#include "QuantizationApi.h"
#include "StorageApi.h"
#include "TensorApi.h"
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

/* BFP epic PR2 Task 8: adaptiveAvgPool1dBackward's ARITH_FLOAT32 arm runs
 * outside executeOp and raw-casts lossGrad/propLoss to float*. Task 8 made BFP
 * dx wires allocatable and pre-flip they select exactly that arm -- guard the
 * storage dtype. */
void testAdaptiveAvgPool1dBackwardRejectsBfpWire(void) {
    size_t inDims[] = {1, 1, 4};
    size_t outDims[] = {1, 1, 2};
    float outData[1 * 1 * 2] = {0};
    adaptivePoolRun_t r = build(input_adaptiveAvgPool1d_basic, inDims, 2, outData, outDims);

    tensor_t *lossGrad = makeFloatTensor(outDims, 3, NULL);
    tensor_t *propLoss = makeFloatTensor(inDims, 3, NULL);
    tensor_t *bfpLossGrad = makeBfpTensor(outDims, 3);
    tensor_t *bfpPropLoss = makeBfpTensor(inDims, 3);

    ASSERT_EXITS_WITH_FAILURE(adaptiveAvgPool1dBackward(r.layer, r.input, bfpLossGrad, propLoss));
    ASSERT_EXITS_WITH_FAILURE(adaptiveAvgPool1dBackward(r.layer, r.input, lossGrad, bfpPropLoss));

    freeTensor(bfpPropLoss);
    freeTensor(bfpLossGrad);
    freeTensor(propLoss);
    freeTensor(lossGrad);
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(testAdaptiveAvgPool1dBackwardRejectsBfpWire);
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
    return UNITY_END();
}
