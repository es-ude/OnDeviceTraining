#define SOURCE_FILE "UNIT_TEST_MAX_POOL_1D"

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "DeathTest.h"
#include "KeyedDrawCheck.h"
#include "Layer.h"
#include "MaxPool1d.h"
#include "Quantization.h"
#include "QuantizationApi.h"
#include "StorageApi.h"
#include "TensorApi.h"
#include "TensorConversion.h"
#include "expected_max_pool_1d.h"
#include "unity.h"

#include "BfpRawFixture.h"
#include "BfpScatterRef.h"
#include "LayerExec.h"
#include "RNG.h"
#include "SlidingWindow1d.h"

// Builds a FLOAT32 MaxPool1d layer without the factory. Static fixture storage
// outlives the returned handles.
typedef struct maxPool1dRunResult {
    layer_t *layer;
    tensor_t *input;
    tensor_t *output;
    tensor_t *argmax;
    quantization_t *q;
} maxPool1dRunResult_t;

static size_t *ownedDims(size_t const *dims, size_t numDims) {
    size_t *owned = reserveMemory(numDims * sizeof(size_t));
    memcpy(owned, dims, numDims * sizeof(size_t));
    return owned;
}

static shape_t *makeShape(size_t const *dims, size_t numDims) {
    size_t *order = reserveMemory(numDims * sizeof(size_t));
    setOrderOfDimsForNewTensor(numDims, order);
    shape_t *shape = reserveMemory(sizeof(shape_t));
    setShape(shape, ownedDims(dims, numDims), numDims, order);
    return shape;
}

static tensor_t *makeFloatTensor(size_t const *dims, size_t numDims, float const *data) {
    tensor_t *t = initTensor(makeShape(dims, numDims), quantizationInitFloat(), NULL);
    if (data != NULL) {
        tensorFillFromFloatBuffer(t, data, calcNumberOfElementsByTensor(t));
    }
    return t;
}

static tensor_t *makeInt32Tensor(size_t const *dims, size_t numDims) {
    return initTensor(makeShape(dims, numDims), quantizationInitInt32(), NULL);
}

static maxPool1dRunResult_t maxPool1dBuild(float const *inputData, size_t const *inputDims,
                                           size_t kSize, paddingType_t padding, size_t dilation,
                                           size_t stride, size_t const *outputDims) {
    static kernel_t kernelStore;
    static maxPool1dConfig_t cfgStore;
    static layer_t layerStore;
    static layerConfig_t lcStore;

    initKernel(&kernelStore, kSize, padding, dilation, stride);

    quantization_t *q = quantizationInitFloat();

    tensor_t *argmax = makeInt32Tensor(outputDims, 3);
    initMaxPool1dConfig(&cfgStore, &kernelStore, argmax, q, q);

    layerStore.type = MAXPOOL1D;
    lcStore.maxPool1d = &cfgStore;
    layerStore.config = &lcStore;

    maxPool1dRunResult_t r = {0};
    r.layer = &layerStore;
    r.input = makeFloatTensor(inputDims, 3, inputData);
    r.output = makeFloatTensor(outputDims, 3, NULL);
    r.argmax = argmax;
    r.q = q;
    return r;
}

void testMaxPool1dForwardBasic(void) {
    size_t inputDims[] = {1, 1, 4};
    size_t outputDims[] = {1, 1, 3};

    maxPool1dRunResult_t r =
        maxPool1dBuild(input_maxPool1d_basic, inputDims, 2, VALID, 1, 1, outputDims);

    maxPool1dForward(r.layer, r.input, r.output, NULL);

    for (size_t i = 0; i < expectedForward_maxPool1d_basic_len; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-5f, expectedForward_maxPool1d_basic[i],
                                 ((float *)r.output->data)[i]);
    }
    int32_t const *argmaxActual = (int32_t const *)r.argmax->data;
    for (size_t i = 0; i < expectedArgmax_maxPool1d_basic_len; i++) {
        TEST_ASSERT_EQUAL_INT32(expectedArgmax_maxPool1d_basic[i], argmaxActual[i]);
    }
}

void testMaxPool1dCalcOutputShapeValidAndSame(void) {
    quantization_t *q = quantizationInitFloat();

    // VALID: K=3, stride=1, dilation=1 -> outLen = inLen - K + 1 = 5
    {
        kernel_t kernel;
        initKernel(&kernel, 3, VALID, 1, 1);
        maxPool1dConfig_t cfg = {0};
        // argmax tensor not used by calcOutputShape — pass a dummy via minimal init.
        size_t dummyDims[] = {1, 1, 1};
        tensor_t *dummyArgmax = makeInt32Tensor(dummyDims, 3);
        initMaxPool1dConfig(&cfg, &kernel, dummyArgmax, q, q);

        layer_t layer;
        layerConfig_t lc;
        layer.type = MAXPOOL1D;
        lc.maxPool1d = &cfg;
        layer.config = &lc;

        // shape_t uses pointer fields — must point to valid stack arrays.
        size_t inDimsBacking[] = {2, 4, 7};
        size_t inOrderBacking[] = {0, 1, 2};
        size_t outDimsBacking[] = {0, 0, 0};
        size_t outOrderBacking[] = {0, 0, 0};
        shape_t inShape = {.dimensions = inDimsBacking,
                           .orderOfDimensions = inOrderBacking,
                           .numberOfDimensions = 3};
        shape_t outShape = {.dimensions = outDimsBacking,
                            .orderOfDimensions = outOrderBacking,
                            .numberOfDimensions = 0};

        maxPool1dCalcOutputShape(&layer, &inShape, &outShape);

        TEST_ASSERT_EQUAL_size_t(3, outShape.numberOfDimensions);
        TEST_ASSERT_EQUAL_size_t(2, outShape.dimensions[0]);
        TEST_ASSERT_EQUAL_size_t(4, outShape.dimensions[1]);
        TEST_ASSERT_EQUAL_size_t(5, outShape.dimensions[2]);
    }

    // SAME: K=3, stride=1, dilation=1 -> outLen = inLen
    {
        kernel_t kernel;
        initKernel(&kernel, 3, SAME, 1, 1);
        maxPool1dConfig_t cfg = {0};
        size_t dummyDims[] = {1, 1, 1};
        tensor_t *dummyArgmax = makeInt32Tensor(dummyDims, 3);
        initMaxPool1dConfig(&cfg, &kernel, dummyArgmax, q, q);

        layer_t layer;
        layerConfig_t lc;
        layer.type = MAXPOOL1D;
        lc.maxPool1d = &cfg;
        layer.config = &lc;

        size_t inDimsBacking[] = {1, 1, 7};
        size_t inOrderBacking[] = {0, 1, 2};
        size_t outDimsBacking[] = {0, 0, 0};
        size_t outOrderBacking[] = {0, 0, 0};
        shape_t inShape = {.dimensions = inDimsBacking,
                           .orderOfDimensions = inOrderBacking,
                           .numberOfDimensions = 3};
        shape_t outShape = {.dimensions = outDimsBacking,
                            .orderOfDimensions = outOrderBacking,
                            .numberOfDimensions = 0};

        maxPool1dCalcOutputShape(&layer, &inShape, &outShape);

        TEST_ASSERT_EQUAL_size_t(7, outShape.dimensions[2]);
    }
}

void testMaxPool1dBackwardBasic(void) {
    size_t inputDims[] = {1, 1, 4};
    size_t outputDims[] = {1, 1, 3};

    maxPool1dRunResult_t r =
        maxPool1dBuild(input_maxPool1d_basic, inputDims, 2, VALID, 1, 1, outputDims);

    // Forward populates argmax — required precondition for backward.
    maxPool1dForward(r.layer, r.input, r.output, NULL);

    // lossGrad = ones (matches what the generator used for autograd on `basic`).
    float lossGradData[1 * 1 * 3];
    for (size_t i = 0; i < 3; i++) {
        lossGradData[i] = 1.0f;
    }
    tensor_t *lossGrad = makeFloatTensor(outputDims, 3, lossGradData);

    tensor_t *propLoss = makeFloatTensor(inputDims, 3, NULL);

    maxPool1dBackward(r.layer, r.input, lossGrad, propLoss, NULL);

    for (size_t i = 0; i < expectedPropLoss_maxPool1d_basic_len; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-5f, expectedPropLoss_maxPool1d_basic[i],
                                 ((float *)propLoss->data)[i]);
    }
}

void testMaxPool1dMultiChannel(void) {
    size_t inputDims[] = {1, 3, 5};  // B=1, C=3, L=5
    size_t outputDims[] = {1, 3, 4}; // outLen = (5-2)/1 + 1 = 4

    maxPool1dRunResult_t r =
        maxPool1dBuild(input_maxPool1d_multiChannel, inputDims, 2, VALID, 1, 1, outputDims);

    maxPool1dForward(r.layer, r.input, r.output, NULL);
    for (size_t i = 0; i < expectedForward_maxPool1d_multiChannel_len; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-5f, expectedForward_maxPool1d_multiChannel[i],
                                 ((float *)r.output->data)[i]);
    }

    int32_t const *argmaxActual = (int32_t const *)r.argmax->data;
    for (size_t i = 0; i < expectedArgmax_maxPool1d_multiChannel_len; i++) {
        TEST_ASSERT_EQUAL_INT32(expectedArgmax_maxPool1d_multiChannel[i], argmaxActual[i]);
    }

    float lossGradData[1 * 3 * 4];
    for (size_t i = 0; i < 12; i++) {
        lossGradData[i] = 1.0f;
    }
    tensor_t *lossGrad = makeFloatTensor(outputDims, 3, lossGradData);

    tensor_t *propLoss = makeFloatTensor(inputDims, 3, NULL);

    maxPool1dBackward(r.layer, r.input, lossGrad, propLoss, NULL);
    for (size_t i = 0; i < expectedPropLoss_maxPool1d_multiChannel_len; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-5f, expectedPropLoss_maxPool1d_multiChannel[i],
                                 ((float *)propLoss->data)[i]);
    }
}

void testMaxPool1dMultiBatch(void) {
    size_t inputDims[] = {4, 2, 4};  // B=4, C=2, L=4
    size_t outputDims[] = {4, 2, 3}; // outLen = (4-2)/1 + 1 = 3

    maxPool1dRunResult_t r =
        maxPool1dBuild(input_maxPool1d_multiBatch, inputDims, 2, VALID, 1, 1, outputDims);

    maxPool1dForward(r.layer, r.input, r.output, NULL);
    for (size_t i = 0; i < expectedForward_maxPool1d_multiBatch_len; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-5f, expectedForward_maxPool1d_multiBatch[i],
                                 ((float *)r.output->data)[i]);
    }

    float lossGradData[4 * 2 * 3];
    for (size_t i = 0; i < 24; i++) {
        lossGradData[i] = 1.0f;
    }
    tensor_t *lossGrad = makeFloatTensor(outputDims, 3, lossGradData);

    tensor_t *propLoss = makeFloatTensor(inputDims, 3, NULL);

    maxPool1dBackward(r.layer, r.input, lossGrad, propLoss, NULL);
    for (size_t i = 0; i < expectedPropLoss_maxPool1d_multiBatch_len; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-5f, expectedPropLoss_maxPool1d_multiBatch[i],
                                 ((float *)propLoss->data)[i]);
    }
}

void testMaxPool1dWithStrideAndDilation(void) {
    size_t inputDims[] = {1, 1, 9}; // B=1, C=1, L=9
    // K=2, stride=3, dilation=2 -> effective_K = (2-1)*2+1 = 3, outLen = (9-3)/3+1 = 3
    size_t outputDims[] = {1, 1, 3};

    maxPool1dRunResult_t r = maxPool1dBuild(input_maxPool1d_withStrideAndDilation, inputDims, 2,
                                            VALID, 2, 3, outputDims);

    maxPool1dForward(r.layer, r.input, r.output, NULL);
    for (size_t i = 0; i < expectedForward_maxPool1d_withStrideAndDilation_len; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-5f, expectedForward_maxPool1d_withStrideAndDilation[i],
                                 ((float *)r.output->data)[i]);
    }

    // auxOut is INT32 index state and is never converted; this fixture pins a
    // non-trivial dilation/stride index pattern.
    int32_t const *argmaxActual = (int32_t const *)r.argmax->data;
    for (size_t i = 0; i < expectedArgmax_maxPool1d_withStrideAndDilation_len; i++) {
        TEST_ASSERT_EQUAL_INT32(expectedArgmax_maxPool1d_withStrideAndDilation[i], argmaxActual[i]);
    }

    // Random gold lossGrad (not ones), so a misplaced scatter changes the result.
    tensor_t *lossGrad = makeFloatTensor(outputDims, 3, lossGrad_maxPool1d_withStrideAndDilation);

    tensor_t *propLoss = makeFloatTensor(inputDims, 3, NULL);

    maxPool1dBackward(r.layer, r.input, lossGrad, propLoss, NULL);
    for (size_t i = 0; i < expectedPropLoss_maxPool1d_withStrideAndDilation_len; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-5f, expectedPropLoss_maxPool1d_withStrideAndDilation[i],
                                 ((float *)propLoss->data)[i]);
    }
}

void testMaxPool1dWithSamePadding(void) {
    size_t inputDims[] = {1, 1, 5};
    size_t outputDims[] = {1, 1, 5}; // SAME -> outLen = inLen

    maxPool1dRunResult_t r =
        maxPool1dBuild(input_maxPool1d_withSamePadding, inputDims, 3, SAME, 1, 1, outputDims);

    maxPool1dForward(r.layer, r.input, r.output, NULL);
    for (size_t i = 0; i < expectedForward_maxPool1d_withSamePadding_len; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-5f, expectedForward_maxPool1d_withSamePadding[i],
                                 ((float *)r.output->data)[i]);
    }

    // Verify argmax content for both edge windows (outPos=0 and outPos=4).
    int32_t const *argmaxActual = (int32_t const *)r.argmax->data;
    for (size_t i = 0; i < expectedArgmax_maxPool1d_withSamePadding_len; i++) {
        TEST_ASSERT_EQUAL_INT32(expectedArgmax_maxPool1d_withSamePadding[i], argmaxActual[i]);
    }

    float lossGradData[1 * 1 * 5];
    for (size_t i = 0; i < 5; i++) {
        lossGradData[i] = 1.0f;
    }
    tensor_t *lossGrad = makeFloatTensor(outputDims, 3, lossGradData);

    tensor_t *propLoss = makeFloatTensor(inputDims, 3, NULL);

    maxPool1dBackward(r.layer, r.input, lossGrad, propLoss, NULL);
    for (size_t i = 0; i < expectedPropLoss_maxPool1d_withSamePadding_len; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-5f, expectedPropLoss_maxPool1d_withSamePadding[i],
                                 ((float *)propLoss->data)[i]);
    }
}

/* ---- SYM_INT32 arm ---- */

/* Build a SYM_INT32 (HALF_AWAY, qMaxBits=12) tensor from a float fixture: values
 * are quantized via tensorFillFromFloatBuffer (absmax->scale, round-clamp). The
 * fixtures are dequant-round-trip-stable (sym_gold.stable_dequant_i12) so the C
 * side lands on exactly the gold mantissas+scale. NULL vals -> zero mantissas,
 * scale 1.0. */
static tensor_t *buildSymTensor(size_t const *dims, size_t numDims, float const *vals) {
    tensor_t *t =
        initTensor(makeShape(dims, numDims), quantizationInitSymInt32WithBits(HALF_AWAY, 12), NULL);
    if (vals != NULL) {
        tensorFillFromFloatBuffer(t, vals, calcNumberOfElementsByTensor(t));
    }
    return t;
}

static void assertSymTensorMatchesGold(tensor_t *t, int32_t const *mantissas, size_t len,
                                       float expectedScale, int32_t mantissaTol, float scaleTol,
                                       float const *dequant, float dequantTol) {
    int32_t const *m = (int32_t const *)t->data;
    float scale = odtScaleToFloat(((symInt32QConfig_t *)t->quantization->qConfig)->scale);
    TEST_ASSERT_FLOAT_WITHIN(expectedScale * scaleTol, expectedScale, scale);
    for (size_t i = 0; i < len; i++) {
        TEST_ASSERT_INT_WITHIN(mantissaTol, mantissas[i], m[i]);
        TEST_ASSERT_FLOAT_WITHIN(dequantTol, dequant[i], (float)m[i] * scale);
    }
}

#define ASSERT_SYM_FORWARD_GOLD(t, fix)                                                            \
    assertSymTensorMatchesGold(t, expectedForwardMantissas_maxPool1dSym_##fix,                     \
                               expectedForwardMantissas_maxPool1dSym_##fix##_len,                  \
                               expectedForwardScale_maxPool1dSym_##fix,                            \
                               mantissaTol_maxPool1dSym_##fix, scaleTol_maxPool1dSym_##fix,        \
                               expectedForwardDequant_maxPool1dSym_##fix,                          \
                               forwardDequantTol_maxPool1dSym_##fix)

#define ASSERT_SYM_PROPLOSS_GOLD(t, fix)                                                           \
    assertSymTensorMatchesGold(t, expectedPropLossMantissas_maxPool1dSym_##fix,                    \
                               expectedPropLossMantissas_maxPool1dSym_##fix##_len,                 \
                               expectedPropLossScale_maxPool1dSym_##fix,                           \
                               mantissaTol_maxPool1dSym_##fix, scaleTol_maxPool1dSym_##fix,        \
                               expectedPropLossDequant_maxPool1dSym_##fix,                         \
                               propLossDequantTol_maxPool1dSym_##fix)

#define ASSERT_SYM_ARGMAX_GOLD(argmaxTensor, fix)                                                  \
    do {                                                                                           \
        int32_t const *am_ = (int32_t const *)(argmaxTensor)->data;                                \
        for (size_t i_ = 0; i_ < expectedArgmax_maxPool1dSym_##fix##_len; i_++) {                  \
            TEST_ASSERT_EQUAL_INT32(expectedArgmax_maxPool1dSym_##fix[i_], am_[i_]);               \
        }                                                                                          \
    } while (0)

typedef struct maxPool1dSymRun {
    layer_t *layer;
    tensor_t *input;
    tensor_t *output;
    tensor_t *argmax;
} maxPool1dSymRun_t;

/* Mirrors maxPool1dBuild but on SYM_INT32 wires: forwardQ/propLossQ declare
 * ARITH_SYM_INT32 compute via arithmeticFromQuantizationOrDefault. */
static maxPool1dSymRun_t maxPool1dBuildSym(float const *inputData, size_t const *inputDims,
                                           size_t kSize, paddingType_t padding, size_t dilation,
                                           size_t stride, size_t const *outputDims) {
    static kernel_t kernelStore;
    static maxPool1dConfig_t cfgStore;
    static layer_t layerStore;
    static layerConfig_t lcStore;

    initKernel(&kernelStore, kSize, padding, dilation, stride);

    quantization_t *q = quantizationInitSymInt32(HALF_AWAY);
    tensor_t *argmax = makeInt32Tensor(outputDims, 3);
    initMaxPool1dConfig(&cfgStore, &kernelStore, argmax, q, q);

    layerStore.type = MAXPOOL1D;
    lcStore.maxPool1d = &cfgStore;
    layerStore.config = &lcStore;

    maxPool1dSymRun_t r = {0};
    r.layer = &layerStore;
    r.input = buildSymTensor(inputDims, 3, inputData);
    r.output = buildSymTensor(outputDims, 3, NULL);
    r.argmax = argmax;
    return r;
}

void testMaxPool1dForwardSymBasic(void) {
    size_t inputDims[] = {1, 1, 4};
    size_t outputDims[] = {1, 1, 3};

    maxPool1dSymRun_t r =
        maxPool1dBuildSym(input_maxPool1dSym_symBasic, inputDims, 2, VALID, 1, 1, outputDims);

    maxPool1dForward(r.layer, r.input, r.output, NULL);

    ASSERT_SYM_FORWARD_GOLD(r.output, symBasic);
    ASSERT_SYM_ARGMAX_GOLD(r.argmax, symBasic);
}

void testMaxPool1dBackwardSymBasic(void) {
    size_t inputDims[] = {1, 1, 4};
    size_t outputDims[] = {1, 1, 3};

    maxPool1dSymRun_t r =
        maxPool1dBuildSym(input_maxPool1dSym_symBasic, inputDims, 2, VALID, 1, 1, outputDims);

    // Forward populates argmax — required precondition for backward.
    maxPool1dForward(r.layer, r.input, r.output, NULL);

    tensor_t *lossGrad = buildSymTensor(outputDims, 3, lossGrad_maxPool1dSym_symBasic);
    tensor_t *propLoss = buildSymTensor(inputDims, 3, NULL);

    maxPool1dBackward(r.layer, r.input, lossGrad, propLoss, NULL);

    ASSERT_SYM_PROPLOSS_GOLD(propLoss, symBasic);
}

void testMaxPool1dSymStrideDilationForwardBackward(void) {
    size_t inputDims[] = {1, 1, 9};
    size_t outputDims[] = {1, 1, 3};

    // K=2, stride=3, dilation=2 — random gold lossGrad so a misplaced SYM scatter
    // changes the result.
    maxPool1dSymRun_t r = maxPool1dBuildSym(input_maxPool1dSym_symStrideDilation, inputDims, 2,
                                            VALID, 2, 3, outputDims);

    maxPool1dForward(r.layer, r.input, r.output, NULL);
    ASSERT_SYM_FORWARD_GOLD(r.output, symStrideDilation);
    ASSERT_SYM_ARGMAX_GOLD(r.argmax, symStrideDilation);

    tensor_t *lossGrad = buildSymTensor(outputDims, 3, lossGrad_maxPool1dSym_symStrideDilation);
    tensor_t *propLoss = buildSymTensor(inputDims, 3, NULL);

    maxPool1dBackward(r.layer, r.input, lossGrad, propLoss, NULL);
    ASSERT_SYM_PROPLOSS_GOLD(propLoss, symStrideDilation);
}

void testMaxPool1dSymTieSamePadding(void) {
    size_t inputDims[] = {1, 1, 5};
    size_t outputDims[] = {1, 1, 5};

    // x[0] and x[1] quantize to the SAME mantissa: the argmax gold pins the
    // first-occurrence tie-break (`>=` would pick x[1]). SAME edge
    // windows exercise validCount < K on the SYM path.
    maxPool1dSymRun_t r = maxPool1dBuildSym(input_maxPool1dSym_symTieSamePadding, inputDims, 3,
                                            SAME, 1, 1, outputDims);

    maxPool1dForward(r.layer, r.input, r.output, NULL);
    ASSERT_SYM_FORWARD_GOLD(r.output, symTieSamePadding);
    ASSERT_SYM_ARGMAX_GOLD(r.argmax, symTieSamePadding);

    tensor_t *lossGrad = buildSymTensor(outputDims, 3, lossGrad_maxPool1dSym_symTieSamePadding);
    tensor_t *propLoss = buildSymTensor(inputDims, 3, NULL);

    maxPool1dBackward(r.layer, r.input, lossGrad, propLoss, NULL);
    ASSERT_SYM_PROPLOSS_GOLD(propLoss, symTieSamePadding);
}

static tensor_t *buildSymTensorWithBits(size_t const *dims, size_t numDims, uint8_t qMaxBits) {
    return initTensor(makeShape(dims, numDims),
                      quantizationInitSymInt32WithBits(HALF_AWAY, qMaxBits), NULL);
}

/* Value-sum guard, width branch: a 31-bit loss-grad mantissa can overflow the
 * int32 scatter accumulator after 2 argmax collisions — the backward kernel
 * must fail fast on qMaxBits > 16 (Reduce.c value-sum contract). */
void testMaxPool1dBackwardSymRejectsWideLossGrad(void) {
    size_t inputDims[] = {1, 1, 4};
    size_t outputDims[] = {1, 1, 3};

    maxPool1dSymRun_t r =
        maxPool1dBuildSym(input_maxPool1dSym_symBasic, inputDims, 2, VALID, 1, 1, outputDims);
    maxPool1dForward(r.layer, r.input, r.output, NULL);

    tensor_t *lossGrad = buildSymTensorWithBits(outputDims, 3, 31);
    tensor_t *propLoss = buildSymTensor(inputDims, 3, NULL);

    ASSERT_EXITS_WITH_FAILURE(maxPool1dBackward(r.layer, r.input, lossGrad, propLoss, NULL));
}

/* Value-sum guard, N branch: at qMaxBits=16 the scatter bound is
 * 2^(32-16) = 65536 worst-case argmax collisions — K=65536/stride=1 reaches
 * it ((effK-1)/stride + 1 = 65536). */
void testMaxPool1dBackwardSymRejectsTermsOverBound(void) {
    size_t inputDims[] = {1, 1, 65536};
    size_t outputDims[] = {1, 1, 1};

    maxPool1dSymRun_t r = maxPool1dBuildSym(NULL, inputDims, 65536, VALID, 1, 1, outputDims);
    maxPool1dForward(r.layer, r.input, r.output, NULL);

    tensor_t *lossGrad = buildSymTensorWithBits(outputDims, 3, 16);
    tensor_t *propLoss = buildSymTensor(inputDims, 3, NULL);

    ASSERT_EXITS_WITH_FAILURE(maxPool1dBackward(r.layer, r.input, lossGrad, propLoss, NULL));
}

void testMaxPool1dEdgeCases(void) {
    size_t inputDims[] = {1, 1, 4};
    size_t outputDims[] = {1, 1, 4}; // K=1 stride=1 -> outLen = inLen

    maxPool1dRunResult_t r =
        maxPool1dBuild(input_maxPool1d_edgeCases, inputDims, 1, VALID, 1, 1, outputDims);

    maxPool1dForward(r.layer, r.input, r.output, NULL);
    for (size_t i = 0; i < expectedForward_maxPool1d_edgeCases_len; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-5f, expectedForward_maxPool1d_edgeCases[i],
                                 ((float *)r.output->data)[i]);
    }

    // K=1 -> argmax[i] == i for every output position.
    int32_t const *argmaxActual = (int32_t const *)r.argmax->data;
    for (size_t i = 0; i < expectedArgmax_maxPool1d_edgeCases_len; i++) {
        TEST_ASSERT_EQUAL_INT32(expectedArgmax_maxPool1d_edgeCases[i], argmaxActual[i]);
    }

    float lossGradData[1 * 1 * 4];
    for (size_t i = 0; i < 4; i++) {
        lossGradData[i] = 1.0f;
    }
    tensor_t *lossGrad = makeFloatTensor(outputDims, 3, lossGradData);

    tensor_t *propLoss = makeFloatTensor(inputDims, 3, NULL);

    maxPool1dBackward(r.layer, r.input, lossGrad, propLoss, NULL);
    for (size_t i = 0; i < expectedPropLoss_maxPool1d_edgeCases_len; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-5f, expectedPropLoss_maxPool1d_edgeCases[i],
                                 ((float *)propLoss->data)[i]);
    }
}

/* The FLOAT32 dx arm must OUT_WRITE propLoss -- the remat arena
 * hands it bytes a previous step left behind. 0xA5A5A5A5 read as a float is
 * -2.87e-16, which `+=` of an O(1) gradient absorbs exactly, so the first call
 * exposes stale bytes only in cells that receive no gradient: every (b, c) row
 * here has non-argmax cells. The second call catches an accumulating arm on
 * every touched cell. All values are dyadic, so the compare is bit-exact. */
void testMaxPool1dBackwardFloatOverwritesStalePropLoss(void) {
    size_t inputDims[] = {2, 2, 5};
    size_t outputDims[] = {2, 2, 4}; // K=2, stride=1, VALID -> outLen = 4
    float const inputData[2 * 2 * 5] = {
        1.0f,  5.0f,  2.0f,  0.0f,  3.0f,  // b0c0: argmax {1, 1, 2, 4}
        4.0f,  0.0f,  -1.0f, 6.0f,  2.0f,  // b0c1: argmax {0, 1, 3, 3}
        0.0f,  2.0f,  7.0f,  1.0f,  -3.0f, // b1c0: argmax {1, 2, 2, 3}
        -2.0f, -1.0f, -4.0f, -5.0f, 8.0f,  // b1c1: argmax {1, 1, 2, 4}
    };
    int32_t const expectedArgmax[2 * 2 * 4] = {1, 1, 2, 4, 0, 1, 3, 3, 1, 2, 2, 3, 1, 1, 2, 4};
    float const gyData[2 * 2 * 4] = {
        0.5f,  0.25f,  1.0f,   -2.0f,   // b0c0
        0.75f, -0.5f,  2.0f,   0.125f,  // b0c1
        1.5f,  -0.25f, 0.5f,   3.0f,    // b1c0
        -1.0f, 0.25f,  0.625f, -0.375f, // b1c1
    };
    float const expectedDx[2 * 2 * 5] = {
        0.0f,  0.75f,  1.0f,   0.0f,   -2.0f,   // b0c0
        0.75f, -0.5f,  0.0f,   2.125f, 0.0f,    // b0c1
        0.0f,  1.5f,   0.25f,  3.0f,   0.0f,    // b1c0
        0.0f,  -0.75f, 0.625f, 0.0f,   -0.375f, // b1c1
    };

    maxPool1dRunResult_t r = maxPool1dBuild(inputData, inputDims, 2, VALID, 1, 1, outputDims);
    tensor_t *lossGrad = makeFloatTensor(outputDims, 3, gyData);
    tensor_t *propLoss = makeFloatTensor(inputDims, 3, NULL);
    size_t propLossElements = calcNumberOfElementsByTensor(propLoss);
    int32_t argmax[2 * 2 * 4];
    float firstDx[2 * 2 * 5];
    float secondDx[2 * 2 * 5];

    maxPool1dForward(r.layer, r.input, r.output, NULL);
    memcpy(argmax, r.argmax->data, sizeof(argmax));
    memset(propLoss->data, 0xA5, propLossElements * sizeof(float));
    maxPool1dBackward(r.layer, r.input, lossGrad, propLoss, NULL);
    memcpy(firstDx, propLoss->data, sizeof(firstDx));
    maxPool1dBackward(r.layer, r.input, lossGrad, propLoss, NULL);
    memcpy(secondDx, propLoss->data, sizeof(secondDx));

    freeTensor(propLoss);
    freeTensor(lossGrad);
    freeTensor(r.argmax);
    freeTensor(r.output);
    freeTensor(r.input);
    freeQuantization(r.q);

    TEST_ASSERT_EQUAL_size_t_MESSAGE(2 * 2 * 5, propLossElements, "fixture: propLoss size");
    TEST_ASSERT_EQUAL_INT32_ARRAY_MESSAGE(expectedArgmax, argmax, 2 * 2 * 4,
                                          "fixture: forward argmax");
    TEST_ASSERT_EQUAL_MEMORY_MESSAGE(expectedDx, firstDx, sizeof(expectedDx),
                                     "dx must not keep the prefilled 0xA5 bytes");
    TEST_ASSERT_EQUAL_MEMORY_MESSAGE(expectedDx, secondDx, sizeof(expectedDx),
                                     "a repeated backward must reproduce dx (OUT_WRITE)");
}

void setUp(void) {}
void tearDown(void) {}

/* ---- native ARITH_BFP arms ---- */

/* Build a BFP wire with EXACT codes and per-group exponents.
 * Writing the packed payload directly (byteConversion) instead of quantizing
 * keeps the fixture independent of the quantizer and lets the test pin the
 * borrowed exponents the kernel compares with. */
static tensor_t *buildBfpWireWithCodes(size_t const *dims, size_t numDims, uint8_t mantissaBits,
                                       uint8_t exponentBits, size_t numGroups, size_t groupSize,
                                       int32_t *codes, uint8_t const *exponents) {
    quantization_t *q = numGroups > 1 ? quantizationInitBfpGrouped(mantissaBits, exponentBits,
                                                                   HALF_AWAY, numGroups, groupSize)
                                      : quantizationInitBfp(mantissaBits, exponentBits, HALF_AWAY);
    tensor_t *t = initTensor(makeShape(dims, numDims), q, NULL);
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

/* Pools have no weight operand, so the width anchor for
 * staging is the layer's OWN produced-wire config — outputQ for the forward,
 * propLossQ for the backward. initMaxPool1dConfig derives both math slots from
 * it, so passing a BFP wireQ selects the native arms. */
static layer_t *maxPool1dBuildBfpLayer(maxPool1dConfig_t *cfgStore, kernel_t *kernelStore,
                                       layerConfig_t *lcStore, layer_t *layerStore,
                                       tensor_t *argmax, size_t kSize, paddingType_t padding,
                                       size_t dilation, size_t stride, quantization_t *wireQ) {
    initKernel(kernelStore, kSize, padding, dilation, stride);
    initMaxPool1dConfig(cfgStore, kernelStore, argmax, wireQ, wireQ);
    layerStore->type = MAXPOOL1D;
    lcStore->maxPool1d = cfgStore;
    layerStore->config = lcStore;
    return layerStore;
}

/* Argmax over BFP values is NOT argmax over mantissas —
 * a smaller code in a larger-exponent block can be the true maximum. The
 * fixture is built so exactly that happens; a mantissa-comparing kernel gets
 * both the value AND the index wrong. */
void testMaxPool1dForwardBfpComparesDequantizedValues(void) {
    size_t inputDims[] = {1, 2, 8};
    size_t outputDims[] = {1, 2, 4};
    int32_t inCodes[16];
    for (size_t i = 0; i < kBfpMaxPoolInCodes_len; i++) {
        inCodes[i] = kBfpMaxPoolInCodes[i];
    }
    tensor_t *input = buildBfpWireWithCodes(
        inputDims, 3, (uint8_t)kBfpMaxPoolMantissaBits, (uint8_t)kBfpMaxPoolExponentBits,
        (size_t)kBfpMaxPoolInNumGroups, (size_t)kBfpMaxPoolInGroupSize, inCodes, kBfpMaxPoolInExps);
    tensor_t *output = makeFloatTensor(outputDims, 3, NULL);
    tensor_t *argmax = makeInt32Tensor(outputDims, 3);
    quantization_t *wireQ = quantizationInitBfpGrouped(
        (uint8_t)kBfpMaxPoolMantissaBits, (uint8_t)kBfpMaxPoolExponentBits, HALF_AWAY,
        (size_t)kBfpMaxPoolInNumGroups, (size_t)kBfpMaxPoolInGroupSize);
    kernel_t k;
    maxPool1dConfig_t cfg;
    layerConfig_t lc;
    layer_t layer;
    maxPool1dBuildBfpLayer(&cfg, &k, &lc, &layer, argmax, (size_t)kBfpMaxPoolKernelSize, VALID,
                           (size_t)kBfpMaxPoolDilation, (size_t)kBfpMaxPoolStride, wireQ);

    maxPool1dForward(&layer, input, output, NULL);

    TEST_ASSERT_EQUAL_MEMORY_MESSAGE(
        kBfpMaxPoolExpectedForward, output->data, kBfpMaxPoolExpectedForward_len * sizeof(float),
        "the winner aligns exactly into the BFP_RAW (within 15 binades of the max)");
    TEST_ASSERT_EQUAL_INT32_ARRAY_MESSAGE(kBfpMaxPoolExpectedArgmax, (int32_t *)argmax->data,
                                          kBfpMaxPoolExpectedArgmax_len,
                                          "argmax must follow the DEQUANTIZED comparison, and "
                                          "the funnel must leave auxOut in its INT32 storage");
    freeQuantization(wireQ);
    freeTensor(argmax);
    freeTensor(output);
    freeTensor(input);
}

/* Integer-only forward (arithmetic-bfp.md section 13): a FLOAT32 target gets
 * ONE raw exponent over the whole output (the {1,0} raw), so every winner is
 * aligned to the tensor's max nonzero winner E minus 15 before the dequant.
 * A winner more than 15 binades below it is right-shifted with HALF_AWAY
 * rounding: 100 at E=0 under a 5 at E=20 becomes 3 * 2^5 = 96, where the
 * float raw used to carry the exact 100. The argmax is unaffected. */
void testMaxPool1dForwardBfpFloat32TargetAlignsWinnersToTheTensorMax(void) {
    size_t inputDims[] = {1, 1, 4};
    size_t outputDims[] = {1, 1, 2};
    int32_t inCodes[4] = {3, 5, 100, 7};
    uint8_t inExps[2] = {147, 127}; /* unbiased 20 and 0 */
    tensor_t *input = buildBfpWireWithCodes(inputDims, 3, 8, 8, 2, 2, inCodes, inExps);
    tensor_t *output = makeFloatTensor(outputDims, 3, NULL);
    tensor_t *argmax = makeInt32Tensor(outputDims, 3);
    quantization_t *wireQ = quantizationInitBfpGrouped(8, 8, HALF_AWAY, 2, 2);
    kernel_t k;
    maxPool1dConfig_t cfg;
    layerConfig_t lc;
    layer_t layer;
    maxPool1dBuildBfpLayer(&cfg, &k, &lc, &layer, argmax, 2, VALID, 1, 2, wireQ);

    maxPool1dForward(&layer, input, output, NULL);

    float expected[2] = {5242880.0f, 96.0f};
    int32_t expectedArgmax[2] = {1, 2};
    TEST_ASSERT_EQUAL_FLOAT_ARRAY_MESSAGE(expected, (float *)output->data, 2,
                                          "a FLOAT32 target dequantizes the per-tensor raw: "
                                          "the winner 20 binades below the max is aligned "
                                          "with HALF_AWAY rounding (100 -> 3 * 2^5)");
    TEST_ASSERT_EQUAL_INT32_ARRAY_MESSAGE(expectedArgmax, (int32_t *)argmax->data, 2,
                                          "argmax is the exact comparison's winner");
    freeQuantization(wireQ);
    freeTensor(argmax);
    freeTensor(output);
    freeTensor(input);
}

/* Integer-only forward: a BFP output wire is re-blocked from the raw by the
 * funnel. Codes and stored exponents are the exponent rule applied to the
 * exact winner values (hand-derived: group 0 = {20, 40} -> E = -1, codes
 * {40, 80}; group 1 = {5 * 4, 7 * 4} -> E = -2, codes {80, 112}) -- the same
 * numbers the float re-pack produced, so this pins "no change on the wire". */
void testMaxPool1dForwardBfpPacksOutputWire(void) {
    size_t inputDims[] = {1, 1, 8};
    size_t outputDims[] = {1, 1, 4};
    int32_t inCodes[8] = {10, 20, -30, 40, 5, -128, 7, 0};
    uint8_t inExps[2] = {127, 129}; /* unbiased 0 and 2 */
    int32_t sentinel[4] = {-9, -9, -9, -9};
    uint8_t zeroState[2] = {127, 127};
    tensor_t *input = buildBfpWireWithCodes(inputDims, 3, 8, 8, 2, 4, inCodes, inExps);
    tensor_t *output = buildBfpWireWithCodes(outputDims, 3, 8, 8, 2, 2, sentinel, zeroState);
    tensor_t *argmax = makeInt32Tensor(outputDims, 3);
    quantization_t *wireQ = quantizationInitBfpGrouped(8, 8, HALF_AWAY, 2, 2);
    kernel_t k;
    maxPool1dConfig_t cfg;
    layerConfig_t lc;
    layer_t layer;
    maxPool1dBuildBfpLayer(&cfg, &k, &lc, &layer, argmax, 2, VALID, 1, 2, wireQ);

    maxPool1dForward(&layer, input, output, NULL);

    int32_t got[4];
    unpackSignExtend(output->data, 8, 0, got, 4);
    int32_t expectedCodes[4] = {40, 80, 80, 112};
    uint8_t expectedExps[2] = {126, 125};
    int32_t expectedArgmax[4] = {1, 3, 4, 6};
    TEST_ASSERT_EQUAL_INT32_ARRAY(expectedCodes, got, 4);
    TEST_ASSERT_EQUAL_UINT8_ARRAY_MESSAGE(
        expectedExps, ((bfpQConfig_t *)output->quantization->qConfig)->exponents, 2,
        "OUT_WRITE derives the wire's group exponents");
    TEST_ASSERT_EQUAL_INT32_ARRAY(expectedArgmax, (int32_t *)argmax->data, 4);
    freeQuantization(wireQ);
    freeTensor(argmax);
    freeTensor(output);
    freeTensor(input);
}

/* ---- Integer-only forward at the kernel (arithmetic-bfp.md section 13) ----
 * maxPool1dForwardKernelBfpRaw on stack operands (BfpRawFixture.h), checked
 * bit for bit against a reference written from the definition: exact value
 * compare in int64, first strict greater, winners aligned to (max nonzero
 * winner E of their raw group) - 15, HALF_AWAY below that. */

typedef struct bfpPoolCase {
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
} bfpPoolCase_t;

static size_t bfpPoolOutLen(const bfpPoolCase_t *c) {
    return (c->len - c->dilation * (c->k - 1) - 1) / c->stride + 1;
}

static int bfpPoolE(const bfpPoolCase_t *c, size_t idx) {
    return (int)c->exps[c->groupSize == 0 ? 0 : idx / c->groupSize] - 127;
}

/* value(ma, ea) > value(mb, eb) in int64 on the smaller exponent's grid; a
 * gap beyond 47 decides like 47 (|code| <= 2^15). */
static bool refValueGreater(int32_t ma, int ea, int32_t mb, int eb) {
    int lo = ea < eb ? ea : eb;
    int da = ea - lo > 47 ? 47 : ea - lo;
    int db = eb - lo > 47 ? 47 : eb - lo;
    return (int64_t)ma * ((int64_t)1 << da) > (int64_t)mb * ((int64_t)1 << db);
}

/* v * 2^-k rounded half away from zero; k < 0 is exact. */
static int64_t refShiftHalfAway(int64_t v, int k) {
    if (k <= 0) {
        return v * ((int64_t)1 << -k);
    }
    if (k > 62) {
        return 0;
    }
    int64_t d = (int64_t)1 << k;
    int64_t mag = v < 0 ? -v : v;
    int64_t q = mag / d;
    if (2 * (mag - q * d) >= d) {
        q++;
    }
    return v < 0 ? -q : q;
}

static size_t refMaxPool(const bfpPoolCase_t *c, int32_t *mant, int16_t *rawE, int32_t *argmax) {
    size_t outLen = bfpPoolOutLen(c);
    size_t n = c->rows * outLen;
    int winE[64];
    size_t crossTies = 0;
    for (size_t r = 0; r < c->rows; r++) {
        for (size_t o = 0; o < outLen; o++) {
            size_t best = r * c->len + o * c->stride;
            for (size_t i = 1; i < c->k; i++) {
                size_t idx = r * c->len + o * c->stride + i * c->dilation;
                int32_t ma = c->codes[idx], mb = c->codes[best];
                int ea = bfpPoolE(c, idx), eb = bfpPoolE(c, best);
                if (refValueGreater(ma, ea, mb, eb)) {
                    best = idx;
                } else if (!refValueGreater(mb, eb, ma, ea) && ea != eb) {
                    crossTies++;
                }
            }
            mant[r * outLen + o] = c->codes[best];
            winE[r * outLen + o] = bfpPoolE(c, best);
            argmax[r * outLen + o] = (int32_t)(best - r * c->len);
        }
    }
    size_t gsz = c->rawGroupSize == 0 ? n : c->rawGroupSize;
    for (size_t t = 0; t < c->rawNumGroups; t++) {
        bool any = false;
        int top = 0;
        for (size_t j = t * gsz; j < (t + 1) * gsz; j++) {
            if (mant[j] != 0 && (!any || winE[j] > top)) {
                top = winE[j];
                any = true;
            }
        }
        rawE[t] = (int16_t)(any ? top - 15 : 0);
        for (size_t j = t * gsz; j < (t + 1) * gsz; j++) {
            int64_t v = refShiftHalfAway(mant[j], rawE[t] - winE[j]);
            TEST_ASSERT_TRUE(v >= INT32_MIN && v <= INT32_MAX);
            mant[j] = (int32_t)v;
        }
    }
    return crossTies;
}

/* Runs the kernel on the case and asserts mantissas, raw exponents (both
 * sentinel-guarded) and argmax equal the reference. Returns the reference's
 * count of cross-group ties. */
static size_t checkMaxPoolBfpRaw(const bfpPoolCase_t *c) {
    size_t inDims[3] = {1, c->rows, c->len};
    size_t outLen = bfpPoolOutLen(c);
    size_t outDims[3] = {1, c->rows, outLen};
    size_t n = c->rows * outLen;
    bfpScratchFixture_t in;
    bfpScratchInit(&in, 3, inDims, c->codes, c->exps, c->numGroups, c->groupSize, c->mantissaBits,
                   8);
    bfpRawFixture_t raw;
    bfpRawInit(&raw, 3, outDims, c->rawNumGroups, c->rawGroupSize);
    tensor_t *argmax = makeInt32Tensor(outDims, 3);
    kernel_t kern;
    initKernel(&kern, c->k, VALID, c->dilation, c->stride);
    maxPool1dConfig_t cfg = {0};
    cfg.kernel = &kern;
    tensor_t *ops[1] = {&in.tensor};

    maxPool1dForwardKernelBfpRaw(ops, 1, &raw.tensor, argmax, &cfg, NULL);

    int32_t mant[64];
    int16_t rawE[16];
    int32_t am[64];
    size_t ties = refMaxPool(c, mant, rawE, am);
    assertRawEquals(&raw, mant, n, rawE, c->rawNumGroups);
    TEST_ASSERT_EQUAL_INT32_ARRAY_MESSAGE(am, (int32_t *)argmax->data, n,
                                          "argmax = the exact compare's first strict greater");
    freeTensor(argmax);
    return ties;
}

/* Ties across groups keep the FIRST candidate (4 * 2^1 vs 8 * 2^0 in both
 * orders), like the float arm's strict >; a same-group tie too. */
void testMaxPool1dBfpRawCrossGroupTiesKeepTheFirst(void) {
    int32_t codes[8] = {4, 4, 8, 3, 5, 8, 4, 0};
    uint8_t exps[4] = {128, 127, 127, 128}; /* unbiased 1, 0, 0, 1 */
    bfpPoolCase_t c = {codes, exps, 4, 2, 8, 2, 4, 2, 1, 1, 1, 0};

    TEST_ASSERT_EQUAL_size_t(2, checkMaxPoolBfpRaw(&c));
}

/* A zero winner aligns losslessly at any E, so it must not set its raw
 * group's exponent: 0 * 2^30 next to 100 * 2^0 keeps 100 exact (E_acc = -15,
 * not 15). Zeros tie across groups too (first kept); an all-zero output
 * writes exponent 0 for every raw group. */
void testMaxPool1dBfpRawZeroWinnersDoNotSetTheExponent(void) {
    int32_t codes[4] = {0, 0, 0, 100};
    uint8_t exps[2] = {157, 127}; /* unbiased 30, 0 */
    bfpPoolCase_t c = {codes, exps, 2, 2, 8, 1, 4, 2, 1, 1, 1, 0};
    TEST_ASSERT_EQUAL_size_t(1, checkMaxPoolBfpRaw(&c));

    int32_t zeros[8] = {0};
    uint8_t zExps[4] = {140, 100, 127, 160};
    bfpPoolCase_t z = {zeros, zExps, 4, 2, 8, 2, 4, 2, 1, 1, 2, 3};
    (void)checkMaxPoolBfpRaw(&z);
}

/* Worst cases: every code at the floor -2^(m-1), at the ceiling, or mixed
 * with zeros and +-1, over exponent gaps 0, 15, 16, 17 and 40 between
 * neighbouring groups (d >= 16 is where the sign alone decides), dilated
 * windows crossing up to three groups, per-tensor and grouped raws (one
 * raw group straddles the two rows). m = 16 and m = 8. */
void testMaxPool1dBfpRawWorstCaseCodesAcrossExponentGaps(void) {
    uint8_t exps[12] = {127, 127, 142, 158, 175, 135, 120, 136, 176, 136, 121, 104};
    static const uint8_t widths[2] = {16, 8};
    for (size_t w = 0; w < 2; w++) {
        int32_t lo = -(1 << (widths[w] - 1)), hi = (1 << (widths[w] - 1)) - 1;
        int32_t floorCodes[24], ceilCodes[24], mixed[24];
        for (size_t i = 0; i < 24; i++) {
            floorCodes[i] = lo;
            ceilCodes[i] = hi;
        }
        static const int32_t pattern[8] = {0, -1, 1, 0, 0, 0, -1, 1};
        for (size_t i = 0; i < 24; i++) {
            mixed[i] = i % 5 == 0 ? lo : (i % 7 == 0 ? hi : pattern[i % 8]);
        }
        const int32_t *sets[3] = {floorCodes, ceilCodes, mixed};
        for (size_t s = 0; s < 3; s++) {
            bfpPoolCase_t per = {sets[s], exps, 12, 2, widths[w], 2, 12, 3, 2, 1, 1, 0};
            (void)checkMaxPoolBfpRaw(&per);
            bfpPoolCase_t byRow = {sets[s], exps, 12, 2, widths[w], 2, 12, 3, 2, 1, 2, 8};
            (void)checkMaxPoolBfpRaw(&byRow);
            bfpPoolCase_t straddle = {sets[s], exps, 12, 2, widths[w], 2, 12, 3, 2, 1, 4, 4};
            (void)checkMaxPoolBfpRaw(&straddle);
        }
    }
}

/* The integer compare picks the SAME argmax as the FLOAT32 arm on the
 * dequantized input (ldexpf is exact for every legal code), ties included:
 * pseudo-random m8 data, groups of 3 straddling windows, a wide exponent
 * spread (no ties) and a narrow one (cross-group ties, asserted present). */
void testMaxPool1dBfpRawArgmaxMatchesTheFloatPath(void) {
    uint32_t lcg = 12345u;
    for (size_t pass = 0; pass < 2; pass++) {
        int32_t codes[36];
        uint8_t exps[12];
        for (size_t i = 0; i < 36; i++) {
            lcg = lcg * 1664525u + 1013904223u;
            codes[i] = pass == 0 ? (int32_t)(lcg >> 24) - 128 : (int32_t)((lcg >> 24) % 9) - 4;
        }
        for (size_t g = 0; g < 12; g++) {
            lcg = lcg * 1664525u + 1013904223u;
            exps[g] = (uint8_t)(pass == 0 ? 107 + (lcg >> 24) % 41 : 126 + (lcg >> 24) % 3);
        }
        bfpPoolCase_t c = {codes, exps, 12, 3, 8, 3, 12, 3, 2, 1, 1, 0};
        size_t ties = checkMaxPoolBfpRaw(&c);
        if (pass == 1) {
            TEST_ASSERT_TRUE_MESSAGE(ties > 0, "the narrow pass must contain cross-group ties");
        }

        float values[36];
        for (size_t i = 0; i < 36; i++) {
            values[i] = ldexpf((float)codes[i], bfpPoolE(&c, i));
        }
        size_t inDims[3] = {1, 3, 12};
        size_t outDims[3] = {1, 3, 8};
        maxPool1dRunResult_t f = maxPool1dBuild(values, inDims, 3, VALID, 2, 1, outDims);
        maxPool1dForward(f.layer, f.input, f.output, NULL);
        int32_t mant[64];
        int16_t rawE[16];
        int32_t am[64];
        (void)refMaxPool(&c, mant, rawE, am);
        TEST_ASSERT_EQUAL_INT32_ARRAY_MESSAGE((int32_t *)f.argmax->data, am, 24,
                                              "BFP argmax must equal the FLOAT32 arm's");
    }
}

static void maxPoolBfpRawBadGeometryBody(void) {
    int32_t codes[4] = {1, 2, 3, 4};
    uint8_t exps[1] = {127};
    size_t inDims[3] = {1, 1, 4};
    size_t outDims[3] = {1, 1, 3};
    bfpScratchFixture_t in;
    bfpScratchInit(&in, 3, inDims, codes, exps, 1, 0, 8, 8);
    bfpRawFixture_t raw;
    bfpRawInit(&raw, 3, outDims, 2, 2); /* 2 * 2 != 3 outputs */
    tensor_t *argmax = makeInt32Tensor(outDims, 3);
    kernel_t kern;
    initKernel(&kern, 2, VALID, 1, 1);
    maxPool1dConfig_t cfg = {0};
    cfg.kernel = &kern;
    tensor_t *ops[1] = {&in.tensor};
    maxPool1dForwardKernelBfpRaw(ops, 1, &raw.tensor, argmax, &cfg, NULL);
}

void testMaxPool1dBfpRawRejectsBadRawGeometry(void) {
    ASSERT_EXITS_WITH_FAILURE(maxPoolBfpRawBadGeometryBody());
}

/* ---- Integer-only backward at the kernel (arithmetic-bfp.md section 14) ----
 * maxPool1dBackwardKernelBfpRaw against a reference written from the
 * definition (BfpScatterRef.h helpers, output-centric): dx row r (one
 * (batch, channel)) sums, for each dx cell, the loss codes of the outputs
 * whose argmax is that cell, each aligned to E_acc(r) = (max exponent over
 * the loss row's NONZERO codes, 0 if none) + g, g = the guard of C terms of
 * m-bit codes, C = the windows covering one cell; the BFP_RAW is the data
 * rule over those sums (bfpRefRawFromRows). */

typedef struct mpBwdCase {
    const int32_t *gy;
    const uint8_t *gyExps;
    size_t gyNumGroups, gyGroupSize;
    uint8_t m;
    size_t rows, len, k, dilation, stride;
    paddingType_t padding;
    size_t rawNumGroups, rawGroupSize;
} mpBwdCase_t;

static uint32_t mpLcg(uint32_t *s) {
    *s = *s * 1664525u + 1013904223u;
    return *s >> 8;
}

/* One random valid argmax per window (a tap inside the input). */
static void mpBwdArgmax(const mpBwdCase_t *c, const windowGeometry1d_t *geom, uint32_t seed,
                        int32_t *argmax) {
    uint32_t s = seed;
    for (size_t r = 0; r < c->rows; r++) {
        for (size_t o = 0; o < geom->outputLength; o++) {
            windowSlice1d_t sl = windowSlice1dAt(geom, o);
            TEST_ASSERT_TRUE(sl.validCount > 0);
            size_t j = mpLcg(&s) % sl.validCount;
            argmax[r * geom->outputLength + o] = (int32_t)(sl.firstValidInputIdx + j * c->dilation);
        }
    }
}

static bfpRefStats_t refMaxPoolBackwardRaw(const mpBwdCase_t *c, size_t outLen,
                                           const int32_t *argmax, int32_t *mant, int16_t *rawE) {
    bfpRefStats_t st = {.fitsInt32 = true};
    bfpRefOperand_t gy = {c->gy, c->gyExps, c->gyGroupSize, c->m, 8};
    size_t cover = (c->dilation * (c->k - 1)) / c->stride + 1;
    int g = bfpRefGuard(cover, c->m, 1);
    int64_t sum[BFP_RAW_FIXTURE_MAX_OUT];
    int eAcc[BFP_RAW_FIXTURE_MAX_OUT];
    for (size_t r = 0; r < c->rows; r++) {
        int eRow = bfpRefEAcc(bfpRefMaxExp(&gy, r * outLen, outLen), g);
        for (size_t i = 0; i < c->len; i++) {
            int64_t s = 0;
            for (size_t o = 0; o < outLen; o++) {
                size_t oi = r * outLen + o;
                if (argmax[oi] == (int32_t)i) {
                    s += bfpRefAlign(c->gy[oi], eRow - bfpRefExp(&gy, oi), &st);
                }
            }
            sum[r * c->len + i] = s;
            eAcc[r * c->len + i] = eRow;
        }
    }
    bfpRefRawFromRows(sum, eAcc, c->rows * c->len, c->rawGroupSize, mant, rawE, &st);
    return st;
}

static bfpRefStats_t checkMaxPoolBackwardBfpRaw(const mpBwdCase_t *c, uint32_t seed) {
    kernel_t kern;
    initKernel(&kern, c->k, c->padding, c->dilation, c->stride);
    windowGeometry1d_t geom = windowGeometry1dCalc(c->len, &kern);
    size_t outLen = geom.outputLength;
    size_t gyDims[3] = {1, c->rows, outLen};
    size_t dxDims[3] = {1, c->rows, c->len};
    size_t n = c->rows * c->len;
    TEST_ASSERT_EQUAL_size_t(
        c->rows * outLen, c->gyGroupSize == 0 ? c->rows * outLen : c->gyNumGroups * c->gyGroupSize);
    tensor_t *argmax = makeInt32Tensor(gyDims, 3);
    mpBwdArgmax(c, &geom, seed, (int32_t *)argmax->data);
    bfpScratchFixture_t gy;
    bfpScratchInit(&gy, 3, gyDims, c->gy, c->gyExps, c->gyNumGroups, c->gyGroupSize, c->m, 8);
    bfpRawFixture_t raw;
    bfpRawInit(&raw, 3, dxDims, c->rawNumGroups, c->rawGroupSize);
    maxPool1dConfig_t cfg = {0};
    cfg.kernel = &kern;
    cfg.argmaxIndices = argmax;
    tensor_t *ops[1] = {&gy.tensor};

    maxPool1dBackwardKernelBfpRaw(ops, 1, &raw.tensor, NULL, &cfg, NULL);

    int32_t mant[BFP_RAW_FIXTURE_MAX_OUT];
    int16_t rawE[BFP_RAW_FIXTURE_MAX_GROUPS];
    bfpRefStats_t st = refMaxPoolBackwardRaw(c, outLen, (int32_t *)argmax->data, mant, rawE);
    TEST_ASSERT_TRUE(st.fitsInt32);
    assertRawEquals(&raw, mant, n, rawE, c->rawNumGroups);
    freeTensor(argmax);
    return st;
}

static void mpRandomCodes(uint32_t seed, int32_t *codes, size_t n, uint8_t m) {
    uint32_t s = seed;
    uint32_t span = 1u << m;
    for (size_t i = 0; i < n; i++) {
        codes[i] = (int32_t)(mpLcg(&s) % span) - (int32_t)(span / 2); /* -2^(m-1) included */
    }
}

/* Overlapping windows (k 3, stride 1: up to 3 winners per cell), SAME padding
 * and dilation 2 / stride 2, m 8 and m 16, loss groups spread over 21
 * binades, raw groups of 4 (inside rows), 9 (across rows) and per tensor. */
void testMaxPool1dBackwardBfpRawMatchesTheIntegerReference(void) {
    int32_t gy[60];
    uint8_t exps[12];
    uint32_t s = 4242u;
    for (size_t i = 0; i < 12; i++) {
        exps[i] = (uint8_t)(127 - 10 + (int)(mpLcg(&s) % 21u));
    }
    mpRandomCodes(11u, gy, 30, 8);
    mpBwdCase_t a = {gy, exps, 6, 5, 8, 3, 12, 3, 1, 1, VALID, 9, 4};
    bfpRefStats_t st = checkMaxPoolBackwardBfpRaw(&a, 1u);
    TEST_ASSERT_TRUE(st.leftShift > 0);
    a.rawNumGroups = 4;
    a.rawGroupSize = 9;
    checkMaxPoolBackwardBfpRaw(&a, 2u);
    a.rawNumGroups = 1;
    a.rawGroupSize = 0;
    checkMaxPoolBackwardBfpRaw(&a, 3u);

    mpRandomCodes(12u, gy, 36, 8); /* SAME: 3 rows of 12 outputs */
    mpBwdCase_t same = {gy, exps, 12, 3, 8, 3, 12, 3, 1, 1, SAME, 9, 4};
    checkMaxPoolBackwardBfpRaw(&same, 4u);

    mpRandomCodes(13u, gy, 16, 16); /* dilation 2, stride 2, m 16: 4 rows of 4 */
    mpBwdCase_t dil = {gy, exps, 4, 4, 16, 4, 12, 3, 2, 2, VALID, 6, 8};
    checkMaxPoolBackwardBfpRaw(&dil, 5u);
}

/* HAR's pool shape (k 2, stride 2: one winner per cell). Row 0's second loss
 * group sits 30 binades below its first: beyond the guard (g = -23 at m 8),
 * so the low codes round half away on the row's grid -- the documented
 * granularity (one E_acc per dx row). Row 2's first loss group is all zero
 * in the zero state (stored = bias) 30 binades ABOVE its live second group:
 * it must not set E_acc, so that row stays exact. */
void testMaxPool1dBackwardBfpRawRowGridAndZeroGroups(void) {
    int32_t gy[16];
    mpRandomCodes(21u, gy, 16, 8);
    for (size_t i = 8; i < 10; i++) {
        gy[i] = 0;
    }
    gy[2] = 1; /* a code the 30-binade gap must round away */
    uint8_t exps[8] = {130, 100, 127, 126, 127, 97, 125, 125};
    mpBwdCase_t c = {gy, exps, 8, 2, 8, 4, 8, 2, 1, 2, VALID, 8, 4};
    bfpRefStats_t st = checkMaxPoolBackwardBfpRaw(&c, 6u);
    TEST_ASSERT_TRUE_MESSAGE(st.rounded > 0, "the 30-binade row rounds");
    c.rawNumGroups = 1;
    c.rawGroupSize = 0;
    checkMaxPoolBackwardBfpRaw(&c, 7u);

    /* rows 2 and 3 alone: exact (the zero group does not lift E_acc) */
    mpBwdCase_t z = {gy + 8, exps + 4, 4, 2, 8, 2, 8, 2, 1, 2, VALID, 4, 4};
    st = checkMaxPoolBackwardBfpRaw(&z, 8u);
    TEST_ASSERT_EQUAL_size_t(0, st.rounded);
}

/* Layer level into a BFP dx wire: the integer raw re-blocked by the
 * funnel's epilogue in integers (BfpScatterRef.h's section-13 re-block), by
 * the OP's rounding mode; under SR one draw per element in element order
 * from the op's RNG_TAG_OUT stream (op key = MaxPool1d's
 * MAXPOOL1D_OP_PROP_LOSS = 1 under RNG_TAG_BACKWARD). */
static void mpLayerIntoBfpWire(roundingMode_t opMode, uint64_t seed) {
    size_t gyDims[3] = {1, 3, 10};
    size_t dxDims[3] = {1, 3, 12};
    int32_t gy[30];
    mpRandomCodes(31u, gy, 30, 8);
    uint8_t exps[6] = {120, 131, 127, 124, 129, 118};
    tensor_t *lossGrad = buildBfpWireWithCodes(gyDims, 3, 8, 8, 6, 5, gy, exps);
    tensor_t *dx = buildBfpWireWithCodes(dxDims, 3, 6, 8, 9, 4, NULL, NULL);
    tensor_t *argmax = makeInt32Tensor(gyDims, 3);
    quantization_t *wireQ = quantizationInitBfpGrouped(6, 8, HALF_AWAY, 9, 4);
    kernel_t k;
    maxPool1dConfig_t cfg;
    layerConfig_t lc;
    layer_t layer;
    maxPool1dBuildBfpLayer(&cfg, &k, &lc, &layer, argmax, 3, VALID, 1, 1, wireQ);
    cfg.propLossMath.roundingMode = opMode;
    windowGeometry1d_t geom = windowGeometry1dCalc(12, &k);
    mpBwdCase_t c = {gy, exps, 6, 5, 8, 3, 12, 3, 1, 1, VALID, 9, 4};
    mpBwdArgmax(&c, &geom, 77u, (int32_t *)argmax->data);

    layerExec_t exec = {.key = rngKeyFromSeed(seed)};
    maxPool1dBackward(&layer, NULL, lossGrad, dx, opMode == SR_HALF_AWAY ? &exec : NULL);

    int32_t codes[36];
    unpackSignExtend(dx->data, 6, 0, codes, 36);
    int32_t mant[BFP_RAW_FIXTURE_MAX_OUT];
    int16_t rawE[BFP_RAW_FIXTURE_MAX_GROUPS];
    bfpRefStats_t st = refMaxPoolBackwardRaw(&c, 10, (int32_t *)argmax->data, mant, rawE);
    TEST_ASSERT_EQUAL_size_t(0, st.rounded);
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
    freeTensor(argmax);
    freeTensor(dx);
    freeTensor(lossGrad);
}

void testMaxPool1dBackwardBfpIntoBfpWireReblocksTheIntegerRaw(void) {
    mpLayerIntoBfpWire(HALF_AWAY, 0u);
    mpLayerIntoBfpWire(SR_HALF_AWAY, 901u);
    mpLayerIntoBfpWire(SR_HALF_AWAY, 902u);
}

/* An argmax inside the input but outside its own window's span would break
 * the C-term headroom proof (more winners per cell than windows cover it):
 * a stale argmax, it dies like an out-of-range one. */
static void mpStaleArgmaxBody(void) {
    int32_t gy[10];
    mpRandomCodes(41u, gy, 10, 8);
    uint8_t exps[2] = {127, 127};
    kernel_t kern;
    initKernel(&kern, 2, VALID, 1, 1);
    size_t gyDims[3] = {1, 1, 10};
    size_t dxDims[3] = {1, 1, 11};
    tensor_t *argmax = makeInt32Tensor(gyDims, 3);
    for (size_t o = 0; o < 10; o++) {
        ((int32_t *)argmax->data)[o] = (int32_t)o;
    }
    ((int32_t *)argmax->data)[9] = 3; /* window 9 spans [9, 10] */
    bfpScratchFixture_t g;
    bfpScratchInit(&g, 3, gyDims, gy, exps, 2, 5, 8, 8);
    bfpRawFixture_t raw;
    bfpRawInit(&raw, 3, dxDims, 1, 0);
    maxPool1dConfig_t cfg = {0};
    cfg.kernel = &kern;
    cfg.argmaxIndices = argmax;
    tensor_t *ops[1] = {&g.tensor};
    maxPool1dBackwardKernelBfpRaw(ops, 1, &raw.tensor, NULL, &cfg, NULL);
}

void testMaxPool1dBackwardBfpRawRejectsStaleArgmax(void) {
    ASSERT_EXITS_WITH_OUTPUT(1, "that window's span", mpStaleArgmaxBody());
}
/* Funnel-routed dx mirroring the ARITH_SYM_INT32
 * arm — gradient flows only to the argmax positions the forward recorded, and
 * an input cell that wins several overlapping windows accumulates. Running the
 * forward first is what fills cfg->argmaxIndices (consumed via ctx). */
void testMaxPool1dBackwardBfpScattersToArgmaxPositions(void) {
    size_t inputDims[] = {1, 2, 8};
    size_t outputDims[] = {1, 2, 4};
    int32_t inCodes[16];
    for (size_t i = 0; i < kBfpMaxPoolInCodes_len; i++) {
        inCodes[i] = kBfpMaxPoolInCodes[i];
    }
    int32_t gyCodes[8];
    for (size_t i = 0; i < kBfpMaxPoolGyCodes_len; i++) {
        gyCodes[i] = kBfpMaxPoolGyCodes[i];
    }
    tensor_t *input = buildBfpWireWithCodes(
        inputDims, 3, (uint8_t)kBfpMaxPoolMantissaBits, (uint8_t)kBfpMaxPoolExponentBits,
        (size_t)kBfpMaxPoolInNumGroups, (size_t)kBfpMaxPoolInGroupSize, inCodes, kBfpMaxPoolInExps);
    tensor_t *output = makeFloatTensor(outputDims, 3, NULL);
    tensor_t *argmax = makeInt32Tensor(outputDims, 3);
    tensor_t *lossGrad = buildBfpWireWithCodes(
        outputDims, 3, (uint8_t)kBfpMaxPoolMantissaBits, (uint8_t)kBfpMaxPoolExponentBits,
        (size_t)kBfpMaxPoolGyNumGroups, (size_t)kBfpMaxPoolGyGroupSize, gyCodes, kBfpMaxPoolGyExps);
    tensor_t *propLoss = makeFloatTensor(inputDims, 3, NULL);
    quantization_t *wireQ = quantizationInitBfpGrouped(
        (uint8_t)kBfpMaxPoolMantissaBits, (uint8_t)kBfpMaxPoolExponentBits, HALF_AWAY,
        (size_t)kBfpMaxPoolInNumGroups, (size_t)kBfpMaxPoolInGroupSize);
    kernel_t k;
    maxPool1dConfig_t cfg;
    layerConfig_t lc;
    layer_t layer;
    maxPool1dBuildBfpLayer(&cfg, &k, &lc, &layer, argmax, (size_t)kBfpMaxPoolKernelSize, VALID,
                           (size_t)kBfpMaxPoolDilation, (size_t)kBfpMaxPoolStride, wireQ);

    maxPool1dForward(&layer, input, output, NULL);
    /* Backward TWICE on purpose: dx is OUT_WRITE, so a repeated backward must
     * reproduce the same numbers — and that is also what makes the kernel's
     * memset of the funnel's uninitialized Phase-2 scratch observable (that
     * stack VLA can read as zero on a first call). */
    maxPool1dBackward(&layer, NULL, lossGrad, propLoss, NULL);
    maxPool1dBackward(&layer, NULL, lossGrad, propLoss, NULL);

    TEST_ASSERT_EQUAL_MEMORY_MESSAGE(kBfpMaxPoolExpectedBackward, propLoss->data,
                                     kBfpMaxPoolExpectedBackward_len * sizeof(float),
                                     "gradient flows only to the recorded argmax positions");
    freeQuantization(wireQ);
    freeTensor(propLoss);
    freeTensor(lossGrad);
    freeTensor(argmax);
    freeTensor(output);
    freeTensor(input);
}

/* -1 marks an empty window: the backward must skip it. In the BFP walk a missing skip
 * exits at the window-span check (VALID padding puts index -1 outside every window). */
void testMaxPool1dBackwardBfpSkipsMinusOneSentinel(void) {
    size_t lossDims[] = {1, 2, 4};
    size_t propDims[] = {1, 2, 8};
    int32_t gyCodes[8];
    for (size_t i = 0; i < kBfpMaxPoolGyCodes_len; i++) {
        gyCodes[i] = kBfpMaxPoolGyCodes[i];
    }
    tensor_t *lossGrad = buildBfpWireWithCodes(
        lossDims, 3, (uint8_t)kBfpMaxPoolMantissaBits, (uint8_t)kBfpMaxPoolExponentBits,
        (size_t)kBfpMaxPoolGyNumGroups, (size_t)kBfpMaxPoolGyGroupSize, gyCodes, kBfpMaxPoolGyExps);
    tensor_t *propLoss = makeFloatTensor(propDims, 3, NULL);
    tensor_t *argmax = makeInt32Tensor(lossDims, 3);
    int32_t *argmaxArr = (int32_t *)argmax->data;
    for (size_t i = 0; i < kBfpMaxPoolExpectedArgmax_len; i++) {
        argmaxArr[i] = kBfpMaxPoolExpectedArgmax[i];
    }
    argmaxArr[4] = -1;

    quantization_t *wireQ = quantizationInitBfpGrouped(
        (uint8_t)kBfpMaxPoolMantissaBits, (uint8_t)kBfpMaxPoolExponentBits, HALF_AWAY,
        (size_t)kBfpMaxPoolGyNumGroups, (size_t)kBfpMaxPoolGyGroupSize);
    kernel_t k;
    maxPool1dConfig_t cfg;
    layerConfig_t lc;
    layer_t layer;
    maxPool1dBuildBfpLayer(&cfg, &k, &lc, &layer, argmax, (size_t)kBfpMaxPoolKernelSize, VALID,
                           (size_t)kBfpMaxPoolDilation, (size_t)kBfpMaxPoolStride, wireQ);

    maxPool1dBackward(&layer, NULL, lossGrad, propLoss, NULL);
    float channel0[8];
    memcpy(channel0, propLoss->data, sizeof channel0);

    freeQuantization(wireQ);
    freeTensor(argmax);
    freeTensor(propLoss);
    freeTensor(lossGrad);

    TEST_ASSERT_EQUAL_MEMORY_MESSAGE(kBfpMaxPoolExpectedBackward, channel0, sizeof channel0,
                                     "the sentinel must route no gradient");
}

/* The FLOAT32 backward casts both wires to float, so a BFP wire must die. */
void testMaxPool1dBackwardFloatRejectsBfpWires(void) {
    size_t inputDims[] = {1, 2, 8};
    size_t outputDims[] = {1, 2, 4};
    tensor_t *bfpLossGrad = buildBfpWireWithCodes(
        outputDims, 3, (uint8_t)kBfpMaxPoolMantissaBits, (uint8_t)kBfpMaxPoolExponentBits,
        (size_t)kBfpMaxPoolGyNumGroups, (size_t)kBfpMaxPoolGyGroupSize, NULL, NULL);
    tensor_t *floatLossGrad = makeFloatTensor(outputDims, 3, NULL);
    tensor_t *floatPropLoss = makeFloatTensor(inputDims, 3, NULL);
    tensor_t *bfpPropLoss = buildBfpWireWithCodes(
        inputDims, 3, (uint8_t)kBfpMaxPoolMantissaBits, (uint8_t)kBfpMaxPoolExponentBits,
        (size_t)kBfpMaxPoolInNumGroups, (size_t)kBfpMaxPoolInGroupSize, NULL, NULL);
    tensor_t *argmax = makeInt32Tensor(outputDims, 3);

    quantization_t *floatQ = quantizationInitFloat();
    kernel_t kf;
    maxPool1dConfig_t floatCfg;
    layerConfig_t floatLc;
    layer_t floatLayer;
    maxPool1dBuildBfpLayer(&floatCfg, &kf, &floatLc, &floatLayer, argmax,
                           (size_t)kBfpMaxPoolKernelSize, VALID, (size_t)kBfpMaxPoolDilation,
                           (size_t)kBfpMaxPoolStride, floatQ);
    ASSERT_EXITS_WITH_OUTPUT(
        1, "MaxPool1d backward (lossGrad): no BFP semantics",
        maxPool1dBackward(&floatLayer, NULL, bfpLossGrad, floatPropLoss, NULL));
    ASSERT_EXITS_WITH_OUTPUT(
        1, "MaxPool1d backward (propLoss): no BFP semantics",
        maxPool1dBackward(&floatLayer, NULL, floatLossGrad, bfpPropLoss, NULL));

    freeQuantization(floatQ);
    freeTensor(argmax);
    freeTensor(bfpPropLoss);
    freeTensor(floatPropLoss);
    freeTensor(floatLossGrad);
    freeTensor(bfpLossGrad);
}

/* The forward writes [B, C, Lout] output cells from the input's dims, so an
 * input of the wrong rank or an output of another shape must die first. */
void testMaxPool1dForwardRejectsMismatchedShapes(void) {
    size_t inputDims[] = {2, 2, 5};
    size_t outputDims[] = {2, 2, 4};
    size_t rank2Dims[] = {2, 10};
    size_t smallBatch[] = {1, 2, 4};
    size_t fewerChannels[] = {2, 1, 4};
    size_t shorter[] = {2, 2, 3};
    maxPool1dRunResult_t r = maxPool1dBuild(NULL, inputDims, 2, VALID, 1, 1, outputDims);
    tensor_t *rank2Input = makeFloatTensor(rank2Dims, 2, NULL);
    tensor_t *batchOut = makeFloatTensor(smallBatch, 3, NULL);
    tensor_t *channelOut = makeFloatTensor(fewerChannels, 3, NULL);
    tensor_t *lengthOut = makeFloatTensor(shorter, 3, NULL);

    ASSERT_EXITS_WITH_OUTPUT(1, "MaxPool1d expects 3D input [batch, channel, length], got 2D",
                             maxPool1dForward(r.layer, rank2Input, r.output, NULL));
    ASSERT_EXITS_WITH_OUTPUT(1, "MaxPool1d forward: output dim 0 must be 2, got 1",
                             maxPool1dForward(r.layer, r.input, batchOut, NULL));
    ASSERT_EXITS_WITH_OUTPUT(1, "MaxPool1d forward: output dim 1 must be 2, got 1",
                             maxPool1dForward(r.layer, r.input, channelOut, NULL));
    ASSERT_EXITS_WITH_OUTPUT(1, "MaxPool1d forward: output dim 2 must be 4, got 3",
                             maxPool1dForward(r.layer, r.input, lengthOut, NULL));

    freeTensor(lengthOut);
    freeTensor(channelOut);
    freeTensor(batchOut);
    freeTensor(rank2Input);
    freeTensor(r.argmax);
    freeTensor(r.output);
    freeTensor(r.input);
    freeQuantization(r.q);
}

/* The argmax keeps the channels and length the layer was built for; an input
 * with other channels must die instead of writing indices past them. */
void testMaxPool1dForwardRejectsInputChannelsOtherThanConfigured(void) {
    size_t configuredDims[] = {1, 2, 4};
    size_t argmaxDims[] = {1, 2, 3};
    size_t inputDims[] = {1, 3, 4};
    size_t outputDims[] = {1, 3, 3};
    maxPool1dRunResult_t r = maxPool1dBuild(NULL, configuredDims, 2, VALID, 1, 1, argmaxDims);
    tensor_t *input = makeFloatTensor(inputDims, 3, NULL);
    tensor_t *output = makeFloatTensor(outputDims, 3, NULL);

    ASSERT_EXITS_WITH_OUTPUT(1, "MaxPool1d forward: argmaxIndices dim 1 must be 3, got 2",
                             maxPool1dForward(r.layer, input, output, NULL));

    freeTensor(output);
    freeTensor(input);
    freeTensor(r.argmax);
    freeTensor(r.output);
    freeTensor(r.input);
    freeQuantization(r.q);
}

/* The backward reads lossGrad as [B, C, Lout] of propLoss's shape and
 * scatters into propLoss, so a propLoss of the wrong rank or a loss of another
 * shape must die first. */
void testMaxPool1dBackwardRejectsMismatchedShapes(void) {
    size_t inputDims[] = {2, 2, 5};
    size_t outputDims[] = {2, 2, 4};
    size_t rank2Dims[] = {2, 10};
    size_t smallBatch[] = {1, 2, 4};
    size_t fewerChannels[] = {2, 1, 4};
    size_t shorter[] = {2, 2, 3};
    maxPool1dRunResult_t r = maxPool1dBuild(NULL, inputDims, 2, VALID, 1, 1, outputDims);
    tensor_t *lossGrad = makeFloatTensor(outputDims, 3, NULL);
    tensor_t *propLoss = makeFloatTensor(inputDims, 3, NULL);
    tensor_t *rank2PropLoss = makeFloatTensor(rank2Dims, 2, NULL);
    tensor_t *batchLoss = makeFloatTensor(smallBatch, 3, NULL);
    tensor_t *channelLoss = makeFloatTensor(fewerChannels, 3, NULL);
    tensor_t *lengthLoss = makeFloatTensor(shorter, 3, NULL);
    maxPool1dForward(r.layer, r.input, r.output, NULL);

    ASSERT_EXITS_WITH_OUTPUT(1, "MaxPool1d expects 3D input [batch, channel, length], got 2D",
                             maxPool1dBackward(r.layer, NULL, lossGrad, rank2PropLoss, NULL));
    ASSERT_EXITS_WITH_OUTPUT(1, "MaxPool1d backward: loss dim 0 must be 2, got 1",
                             maxPool1dBackward(r.layer, NULL, batchLoss, propLoss, NULL));
    ASSERT_EXITS_WITH_OUTPUT(1, "MaxPool1d backward: loss dim 1 must be 2, got 1",
                             maxPool1dBackward(r.layer, NULL, channelLoss, propLoss, NULL));
    ASSERT_EXITS_WITH_OUTPUT(1, "MaxPool1d backward: loss dim 2 must be 4, got 3",
                             maxPool1dBackward(r.layer, NULL, lengthLoss, propLoss, NULL));

    freeTensor(lengthLoss);
    freeTensor(channelLoss);
    freeTensor(batchLoss);
    freeTensor(rank2PropLoss);
    freeTensor(propLoss);
    freeTensor(lossGrad);
    freeTensor(r.argmax);
    freeTensor(r.output);
    freeTensor(r.input);
    freeQuantization(r.q);
}

/* The backward consumes the argmax of the matching forward: after a batch-1
 * forward, a batch-2 backward would read rows that forward never wrote. */
void testMaxPool1dBackwardFloatRejectsBatch1ArgmaxAtBatch2(void) {
    size_t batch1InputDims[] = {1, 3, 5};
    size_t batch1OutputDims[] = {1, 3, 4};
    size_t inputDims[] = {2, 3, 5};
    size_t outputDims[] = {2, 3, 4};
    maxPool1dRunResult_t r =
        maxPool1dBuild(NULL, batch1InputDims, 2, VALID, 1, 1, batch1OutputDims);
    tensor_t *batch1LossGrad = makeFloatTensor(batch1OutputDims, 3, NULL);
    tensor_t *batch1PropLoss = makeFloatTensor(batch1InputDims, 3, NULL);
    tensor_t *lossGrad = makeFloatTensor(outputDims, 3, NULL);
    tensor_t *propLoss = makeFloatTensor(inputDims, 3, NULL);

    maxPool1dForward(r.layer, r.input, r.output, NULL);
    maxPool1dBackward(r.layer, NULL, batch1LossGrad, batch1PropLoss, NULL);

    ASSERT_EXITS_WITH_OUTPUT(1, "MaxPool1d backward: argmaxIndices dim 0 must be 2, got 1",
                             maxPool1dBackward(r.layer, NULL, lossGrad, propLoss, NULL));

    freeTensor(propLoss);
    freeTensor(lossGrad);
    freeTensor(batch1PropLoss);
    freeTensor(batch1LossGrad);
    freeTensor(r.argmax);
    freeTensor(r.output);
    freeTensor(r.input);
    freeQuantization(r.q);
}

/* The SR forward rounds in the OUT_WRITE epilogue of its forward op: OP 0
 * under the FORWARD phase of the layer key. The state from the Python mirror,
 * tags as numbers (FORWARD 4, OP 7, OUT 10):
 *   uv run python -c "from examples._shared.keyed_rng import fold, key_from_seed as kfs,
 *   open_stream as o; print(hex(o(fold(fold(fold(kfs(0xa001), 4, 0), 7, 0), 10, 0))))" */
#define MAXPOOL1D_KEYED_SEED 0xa001u
#define MAXPOOL1D_KEYED_OUT_STATE 0xcf3d22e6u

/* MaxPool1d(k = 2, stride 1) over 4 channels of length 5: 16 outputs, each
 * rounded SR into an 8-bit SYM_INT32 output by its own draw. */
void testMaxPool1dSrForwardIsAPureFunctionOfItsForwardOpKey(void) {
    size_t inputDims[] = {1, 4, 5};
    size_t outputDims[] = {1, 4, 4};
    kernel_t kernel;
    initKernel(&kernel, 2, VALID, 1, 1);
    quantization_t *q = quantizationInitFloat();
    tensor_t *argmax = makeInt32Tensor(outputDims, 3);
    maxPool1dConfig_t cfg;
    initMaxPool1dConfig(&cfg, &kernel, argmax, q, q);
    layerConfig_t lc = {.maxPool1d = &cfg};
    layer_t layer = {.type = MAXPOOL1D, .config = &lc};
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

    assertKeyedSymForwardDrawsFrom(&f, MAXPOOL1D_KEYED_SEED,
                                   keyedOpOutState(MAXPOOL1D_KEYED_SEED, RNG_TAG_FORWARD, 0u),
                                   MAXPOOL1D_KEYED_OUT_STATE);

    freeTensor(input);
    freeTensor(argmax);
    freeQuantization(q);
}

void testMaxPool1dInitConfigClearsOwnsQuantizations(void) {
    kernel_t kernel;
    initKernel(&kernel, 2, VALID, 1, 1);
    size_t argmaxDims[] = {1, 1, 1};
    tensor_t *argmax = makeInt32Tensor(argmaxDims, 3);
    maxPool1dConfig_t cfg;
    memset(&cfg, 0xFF, sizeof cfg);
    initMaxPool1dConfig(&cfg, &kernel, argmax, NULL, NULL);
    TEST_ASSERT_EQUAL_UINT8(0, *(const unsigned char *)&cfg.ownsQuantizations);
    freeTensor(argmax);
}

/* k=2 s=2 VALID on [1,2,6] -> Lout 3. Every slot is filled so a zero in dx is
 * never an untouched calloc index 0. The sentinel sits in channel 1, so a
 * missing skip would land in channel 0's last cell. */
static const float kArgmaxGuardLossGrad[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
static const float kArgmaxGuardDxWithSentinel[] = {0.0f, 1.0f, 2.0f, 0.0f, 3.0f, 0.0f,
                                                   0.0f, 0.0f, 0.0f, 5.0f, 0.0f, 6.0f};

static void fillArgmaxGuardIndices(tensor_t *argmax) {
    static const int32_t withSentinel[] = {1, 2, 4, -1, 3, 5};
    memcpy(argmax->data, withSentinel, sizeof withSentinel);
}

void testMaxPool1dBackwardFloatSkipsMinusOneSentinel(void) {
    size_t inputDims[] = {1, 2, 6};
    size_t outputDims[] = {1, 2, 3};
    maxPool1dRunResult_t r = maxPool1dBuild(NULL, inputDims, 2, VALID, 1, 2, outputDims);
    tensor_t *lossGrad = makeFloatTensor(outputDims, 3, kArgmaxGuardLossGrad);
    tensor_t *propLoss = makeFloatTensor(inputDims, 3, NULL);
    fillArgmaxGuardIndices(r.argmax);

    maxPool1dBackward(r.layer, NULL, lossGrad, propLoss, NULL);

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(kArgmaxGuardDxWithSentinel, (float *)propLoss->data, 12);

    freeTensor(propLoss);
    freeTensor(lossGrad);
    freeTensor(r.argmax);
    freeTensor(r.output);
    freeTensor(r.input);
    freeQuantization(r.q);
}

/* propLoss is the width-restored wire, so scattered cells are checked as
 * dequant within one step; zeros are exact under any scale. */
void testMaxPool1dBackwardSymSkipsMinusOneSentinel(void) {
    size_t inputDims[] = {1, 2, 6};
    size_t outputDims[] = {1, 2, 3};
    maxPool1dSymRun_t r = maxPool1dBuildSym(NULL, inputDims, 2, VALID, 1, 2, outputDims);
    tensor_t *lossGrad = buildSymTensor(outputDims, 3, kArgmaxGuardLossGrad);
    tensor_t *propLoss = buildSymTensor(inputDims, 3, NULL);
    fillArgmaxGuardIndices(r.argmax);

    maxPool1dBackward(r.layer, NULL, lossGrad, propLoss, NULL);

    int32_t const *m = (int32_t const *)propLoss->data;
    float scale = odtScaleToFloat(((symInt32QConfig_t *)propLoss->quantization->qConfig)->scale);
    for (size_t i = 0; i < 12; i++) {
        if (kArgmaxGuardDxWithSentinel[i] == 0.0f) {
            TEST_ASSERT_EQUAL_INT32(0, m[i]);
        } else {
            TEST_ASSERT_FLOAT_WITHIN(scale, kArgmaxGuardDxWithSentinel[i], (float)m[i] * scale);
        }
    }

    freeTensor(propLoss);
    freeTensor(lossGrad);
    freeTensor(r.argmax);
    freeTensor(r.output);
    freeTensor(r.input);
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(testMaxPool1dForwardBasic);
    RUN_TEST(testMaxPool1dCalcOutputShapeValidAndSame);
    RUN_TEST(testMaxPool1dBackwardBasic);
    RUN_TEST(testMaxPool1dMultiChannel);
    RUN_TEST(testMaxPool1dMultiBatch);
    RUN_TEST(testMaxPool1dWithStrideAndDilation);
    RUN_TEST(testMaxPool1dWithSamePadding);
    RUN_TEST(testMaxPool1dEdgeCases);
    RUN_TEST(testMaxPool1dBackwardFloatOverwritesStalePropLoss);
    RUN_TEST(testMaxPool1dForwardSymBasic);
    RUN_TEST(testMaxPool1dBackwardSymBasic);
    RUN_TEST(testMaxPool1dSymStrideDilationForwardBackward);
    RUN_TEST(testMaxPool1dSymTieSamePadding);
    RUN_TEST(testMaxPool1dBackwardSymRejectsWideLossGrad);
    RUN_TEST(testMaxPool1dBackwardSymRejectsTermsOverBound);
    RUN_TEST(testMaxPool1dForwardBfpComparesDequantizedValues);
    RUN_TEST(testMaxPool1dForwardBfpFloat32TargetAlignsWinnersToTheTensorMax);
    RUN_TEST(testMaxPool1dForwardBfpPacksOutputWire);
    RUN_TEST(testMaxPool1dBfpRawCrossGroupTiesKeepTheFirst);
    RUN_TEST(testMaxPool1dBfpRawZeroWinnersDoNotSetTheExponent);
    RUN_TEST(testMaxPool1dBfpRawWorstCaseCodesAcrossExponentGaps);
    RUN_TEST(testMaxPool1dBfpRawArgmaxMatchesTheFloatPath);
    RUN_TEST(testMaxPool1dBfpRawRejectsBadRawGeometry);
    RUN_TEST(testMaxPool1dBackwardBfpRawMatchesTheIntegerReference);
    RUN_TEST(testMaxPool1dBackwardBfpRawRowGridAndZeroGroups);
    RUN_TEST(testMaxPool1dBackwardBfpIntoBfpWireReblocksTheIntegerRaw);
    RUN_TEST(testMaxPool1dBackwardBfpRawRejectsStaleArgmax);
    RUN_TEST(testMaxPool1dBackwardBfpScattersToArgmaxPositions);
    RUN_TEST(testMaxPool1dBackwardBfpSkipsMinusOneSentinel);
    RUN_TEST(testMaxPool1dBackwardFloatRejectsBfpWires);
    RUN_TEST(testMaxPool1dForwardRejectsMismatchedShapes);
    RUN_TEST(testMaxPool1dForwardRejectsInputChannelsOtherThanConfigured);
    RUN_TEST(testMaxPool1dBackwardRejectsMismatchedShapes);
    RUN_TEST(testMaxPool1dBackwardFloatRejectsBatch1ArgmaxAtBatch2);
    RUN_TEST(testMaxPool1dSrForwardIsAPureFunctionOfItsForwardOpKey);
    RUN_TEST(testMaxPool1dInitConfigClearsOwnsQuantizations);
    RUN_TEST(testMaxPool1dBackwardFloatSkipsMinusOneSentinel);
    RUN_TEST(testMaxPool1dBackwardSymSkipsMinusOneSentinel);
    return UNITY_END();
}
