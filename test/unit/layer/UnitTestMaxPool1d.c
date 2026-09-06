#define SOURCE_FILE "UNIT_TEST_MAX_POOL_1D"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "DeathTest.h"
#include "Layer.h"
#include "MaxPool1d.h"
#include "Quantization.h"
#include "QuantizationApi.h"
#include "StorageApi.h"
#include "TensorApi.h"
#include "TensorConversion.h"
#include "expected_max_pool_1d.h"
#include "unity.h"

// Helper: build a MaxPool1d layer manually (no UserAPI in Phase 1).
// Uses function-local statics for kernel/cfg/layer storage so addresses
// survive the return-by-value (per PR-2 plan-bug helper-pattern dangling pointers).
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
                                           size_t stride, float *outputBuf, int32_t *argmaxBuf,
                                           size_t const *outputDims) {
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
    (void)outputBuf;
    (void)argmaxBuf;
    r.argmax = argmax;
    r.q = q;
    return r;
}

void testMaxPool1dForwardBasic(void) {
    size_t inputDims[] = {1, 1, 4};
    size_t outputDims[] = {1, 1, 3};
    float outputData[1 * 1 * 3] = {0};
    int32_t argmaxData[1 * 1 * 3] = {0};

    maxPool1dRunResult_t r = maxPool1dBuild(input_maxPool1d_basic, inputDims, 2, VALID, 1, 1,
                                            outputData, argmaxData, outputDims);

    maxPool1dForward(r.layer, r.input, r.output);

    for (size_t i = 0; i < expectedForward_maxPool1d_basic_len; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-5f, expectedForward_maxPool1d_basic[i],
                                 ((float *)r.output->data)[i]);
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
    float outputData[1 * 1 * 3] = {0};
    int32_t argmaxData[1 * 1 * 3] = {0};

    maxPool1dRunResult_t r = maxPool1dBuild(input_maxPool1d_basic, inputDims, 2, VALID, 1, 1,
                                            outputData, argmaxData, outputDims);

    // Forward populates argmax — required precondition for backward.
    maxPool1dForward(r.layer, r.input, r.output);

    // lossGrad = ones (matches what the generator used for autograd on `basic`).
    float lossGradData[1 * 1 * 3];
    for (size_t i = 0; i < 3; i++) {
        lossGradData[i] = 1.0f;
    }
    tensor_t *lossGrad = makeFloatTensor(outputDims, 3, lossGradData);

    tensor_t *propLoss = makeFloatTensor(inputDims, 3, NULL);

    maxPool1dBackward(r.layer, r.input, lossGrad, propLoss);

    // (Mutation: sentinel-skip removal is vacuous in basic fixture; no empty-window
    //  fixture is in this PR's test set per spec §6.3 / Q3.)
    for (size_t i = 0; i < expectedPropLoss_maxPool1d_basic_len; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-5f, expectedPropLoss_maxPool1d_basic[i],
                                 ((float *)propLoss->data)[i]);
    }
}

void testMaxPool1dArgmaxIndicesContent(void) {
    size_t inputDims[] = {1, 1, 4};
    size_t outputDims[] = {1, 1, 3};
    float outputData[1 * 1 * 3] = {0};
    int32_t argmaxData[1 * 1 * 3] = {0};

    maxPool1dRunResult_t r = maxPool1dBuild(input_maxPool1d_basic, inputDims, 2, VALID, 1, 1,
                                            outputData, argmaxData, outputDims);

    maxPool1dForward(r.layer, r.input, r.output);

    // Compare argmax tensor content against generator-emitted gold values.
    int32_t const *actual = (int32_t const *)r.argmax->data;
    for (size_t i = 0; i < expectedArgmax_maxPool1d_basic_len; i++) {
        TEST_ASSERT_EQUAL_INT32(expectedArgmax_maxPool1d_basic[i], actual[i]);
    }
}

void testMaxPool1dMultiChannel(void) {
    size_t inputDims[] = {1, 3, 5};  // B=1, C=3, L=5
    size_t outputDims[] = {1, 3, 4}; // outLen = (5-2)/1 + 1 = 4
    float outputData[1 * 3 * 4] = {0};
    int32_t argmaxData[1 * 3 * 4] = {0};

    maxPool1dRunResult_t r = maxPool1dBuild(input_maxPool1d_multiChannel, inputDims, 2, VALID, 1, 1,
                                            outputData, argmaxData, outputDims);

    maxPool1dForward(r.layer, r.input, r.output);
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

    maxPool1dBackward(r.layer, r.input, lossGrad, propLoss);
    for (size_t i = 0; i < expectedPropLoss_maxPool1d_multiChannel_len; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-5f, expectedPropLoss_maxPool1d_multiChannel[i],
                                 ((float *)propLoss->data)[i]);
    }
}

void testMaxPool1dMultiBatch(void) {
    size_t inputDims[] = {4, 2, 4};  // B=4, C=2, L=4
    size_t outputDims[] = {4, 2, 3}; // outLen = (4-2)/1 + 1 = 3
    float outputData[4 * 2 * 3] = {0};
    int32_t argmaxData[4 * 2 * 3] = {0};

    maxPool1dRunResult_t r = maxPool1dBuild(input_maxPool1d_multiBatch, inputDims, 2, VALID, 1, 1,
                                            outputData, argmaxData, outputDims);

    maxPool1dForward(r.layer, r.input, r.output);
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

    maxPool1dBackward(r.layer, r.input, lossGrad, propLoss);
    for (size_t i = 0; i < expectedPropLoss_maxPool1d_multiBatch_len; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-5f, expectedPropLoss_maxPool1d_multiBatch[i],
                                 ((float *)propLoss->data)[i]);
    }
}

void testMaxPool1dWithStrideAndDilation(void) {
    size_t inputDims[] = {1, 1, 9}; // B=1, C=1, L=9
    // K=2, stride=3, dilation=2 -> effective_K = (2-1)*2+1 = 3, outLen = (9-3)/3+1 = 3
    size_t outputDims[] = {1, 1, 3};
    float outputData[1 * 1 * 3] = {0};
    int32_t argmaxData[1 * 1 * 3] = {0};

    maxPool1dRunResult_t r = maxPool1dBuild(input_maxPool1d_withStrideAndDilation, inputDims, 2,
                                            VALID, 2, 3, outputData, argmaxData, outputDims);

    maxPool1dForward(r.layer, r.input, r.output);
    for (size_t i = 0; i < expectedForward_maxPool1d_withStrideAndDilation_len; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-5f, expectedForward_maxPool1d_withStrideAndDilation[i],
                                 ((float *)r.output->data)[i]);
    }

    // auxOut integration (spec Testing list): argmaxIndices now flows through
    // opSpec_t.auxOut (kernel-written verbatim, never funnel-converted) —
    // assert it is byte-identical to this pre-migration, unregenerated
    // fixture (a non-trivial dilation/stride pattern, unlike the other
    // argmax-checking tests' simpler geometries).
    int32_t const *argmaxActual = (int32_t const *)r.argmax->data;
    for (size_t i = 0; i < expectedArgmax_maxPool1d_withStrideAndDilation_len; i++) {
        TEST_ASSERT_EQUAL_INT32(expectedArgmax_maxPool1d_withStrideAndDilation[i], argmaxActual[i]);
    }

    // Use the gold-emitted random lossGrad (NOT ones), so positional mutations
    // on the backward path are non-vacuous (codebase_uniform_lossgrad_mutation_vacuity).
    tensor_t *lossGrad = makeFloatTensor(outputDims, 3, lossGrad_maxPool1d_withStrideAndDilation);

    tensor_t *propLoss = makeFloatTensor(inputDims, 3, NULL);

    maxPool1dBackward(r.layer, r.input, lossGrad, propLoss);
    for (size_t i = 0; i < expectedPropLoss_maxPool1d_withStrideAndDilation_len; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-5f, expectedPropLoss_maxPool1d_withStrideAndDilation[i],
                                 ((float *)propLoss->data)[i]);
    }
}

void testMaxPool1dWithSamePadding(void) {
    size_t inputDims[] = {1, 1, 5};
    size_t outputDims[] = {1, 1, 5}; // SAME -> outLen = inLen
    float outputData[1 * 1 * 5] = {0};
    int32_t argmaxData[1 * 1 * 5] = {0};

    maxPool1dRunResult_t r = maxPool1dBuild(input_maxPool1d_withSamePadding, inputDims, 3, SAME, 1,
                                            1, outputData, argmaxData, outputDims);

    maxPool1dForward(r.layer, r.input, r.output);
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

    maxPool1dBackward(r.layer, r.input, lossGrad, propLoss);
    for (size_t i = 0; i < expectedPropLoss_maxPool1d_withSamePadding_len; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-5f, expectedPropLoss_maxPool1d_withSamePadding[i],
                                 ((float *)propLoss->data)[i]);
    }
}

/* ---- SYM_INT32 arm (#205) ---- */

/* Build a SYM_INT32 (HALF_AWAY, qMaxBits=12) tensor from a float fixture: values
 * are quantized via tensorFillFromFloatBuffer (absmax->scale, round-clamp). The
 * fixtures are dequant-round-trip-stable (sym_gold.stable_dequant_i12) so the C
 * side lands on exactly the gold mantissas+scale. NULL vals -> zero mantissas,
 * scale 1.0. (UnitTestConv1d.c helper pattern.) */
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
    float scale = ((symInt32QConfig_t *)t->quantization->qConfig)->scale;
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

    maxPool1dForward(r.layer, r.input, r.output);

    ASSERT_SYM_FORWARD_GOLD(r.output, symBasic);
    ASSERT_SYM_ARGMAX_GOLD(r.argmax, symBasic);
}

void testMaxPool1dBackwardSymBasic(void) {
    size_t inputDims[] = {1, 1, 4};
    size_t outputDims[] = {1, 1, 3};

    maxPool1dSymRun_t r =
        maxPool1dBuildSym(input_maxPool1dSym_symBasic, inputDims, 2, VALID, 1, 1, outputDims);

    // Forward populates argmax — required precondition for backward.
    maxPool1dForward(r.layer, r.input, r.output);

    tensor_t *lossGrad = buildSymTensor(outputDims, 3, lossGrad_maxPool1dSym_symBasic);
    tensor_t *propLoss = buildSymTensor(inputDims, 3, NULL);

    maxPool1dBackward(r.layer, r.input, lossGrad, propLoss);

    ASSERT_SYM_PROPLOSS_GOLD(propLoss, symBasic);
}

void testMaxPool1dSymStrideDilationForwardBackward(void) {
    size_t inputDims[] = {1, 1, 9};
    size_t outputDims[] = {1, 1, 3};

    // K=2, stride=3, dilation=2 — random gold lossGrad so positional mutations
    // on the SYM scatter path are non-vacuous.
    maxPool1dSymRun_t r = maxPool1dBuildSym(input_maxPool1dSym_symStrideDilation, inputDims, 2,
                                            VALID, 2, 3, outputDims);

    maxPool1dForward(r.layer, r.input, r.output);
    ASSERT_SYM_FORWARD_GOLD(r.output, symStrideDilation);
    ASSERT_SYM_ARGMAX_GOLD(r.argmax, symStrideDilation);

    tensor_t *lossGrad = buildSymTensor(outputDims, 3, lossGrad_maxPool1dSym_symStrideDilation);
    tensor_t *propLoss = buildSymTensor(inputDims, 3, NULL);

    maxPool1dBackward(r.layer, r.input, lossGrad, propLoss);
    ASSERT_SYM_PROPLOSS_GOLD(propLoss, symStrideDilation);
}

void testMaxPool1dSymTieSamePadding(void) {
    size_t inputDims[] = {1, 1, 5};
    size_t outputDims[] = {1, 1, 5};

    // x[0] and x[1] quantize to the SAME mantissa: the argmax gold pins the
    // first-occurrence tie-break (a `>` -> `>=` mutation flips it). SAME edge
    // windows exercise validCount < K on the SYM path.
    maxPool1dSymRun_t r = maxPool1dBuildSym(input_maxPool1dSym_symTieSamePadding, inputDims, 3,
                                            SAME, 1, 1, outputDims);

    maxPool1dForward(r.layer, r.input, r.output);
    ASSERT_SYM_FORWARD_GOLD(r.output, symTieSamePadding);
    ASSERT_SYM_ARGMAX_GOLD(r.argmax, symTieSamePadding);

    tensor_t *lossGrad = buildSymTensor(outputDims, 3, lossGrad_maxPool1dSym_symTieSamePadding);
    tensor_t *propLoss = buildSymTensor(inputDims, 3, NULL);

    maxPool1dBackward(r.layer, r.input, lossGrad, propLoss);
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
    maxPool1dForward(r.layer, r.input, r.output);

    tensor_t *lossGrad = buildSymTensorWithBits(outputDims, 3, 31);
    tensor_t *propLoss = buildSymTensor(inputDims, 3, NULL);

    ASSERT_EXITS_WITH_FAILURE(maxPool1dBackward(r.layer, r.input, lossGrad, propLoss));
}

/* Value-sum guard, N branch: at qMaxBits=16 the scatter bound is
 * 2^(32-16) = 65536 worst-case argmax collisions — K=65536/stride=1 reaches
 * it ((effK-1)/stride + 1 = 65536). */
void testMaxPool1dBackwardSymRejectsTermsOverBound(void) {
    size_t inputDims[] = {1, 1, 65536};
    size_t outputDims[] = {1, 1, 1};

    maxPool1dSymRun_t r = maxPool1dBuildSym(NULL, inputDims, 65536, VALID, 1, 1, outputDims);
    maxPool1dForward(r.layer, r.input, r.output);

    tensor_t *lossGrad = buildSymTensorWithBits(outputDims, 3, 16);
    tensor_t *propLoss = buildSymTensor(inputDims, 3, NULL);

    ASSERT_EXITS_WITH_FAILURE(maxPool1dBackward(r.layer, r.input, lossGrad, propLoss));
}

void testMaxPool1dEdgeCases(void) {
    size_t inputDims[] = {1, 1, 4};
    size_t outputDims[] = {1, 1, 4}; // K=1 stride=1 -> outLen = inLen
    float outputData[1 * 1 * 4] = {0};
    int32_t argmaxData[1 * 1 * 4] = {0};

    maxPool1dRunResult_t r = maxPool1dBuild(input_maxPool1d_edgeCases, inputDims, 1, VALID, 1, 1,
                                            outputData, argmaxData, outputDims);

    maxPool1dForward(r.layer, r.input, r.output);
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

    maxPool1dBackward(r.layer, r.input, lossGrad, propLoss);
    for (size_t i = 0; i < expectedPropLoss_maxPool1d_edgeCases_len; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-5f, expectedPropLoss_maxPool1d_edgeCases[i],
                                 ((float *)propLoss->data)[i]);
    }
}

void setUp(void) {}
void tearDown(void) {}

/* ---- BFP epic PR4 (R-P1/R-P4): native ARITH_BFP arms ---- */

/* BFP epic PR4: build a BFP wire with EXACT codes and per-group exponents.
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

/* BFP epic PR4 (R-P1): pools have no weight operand, so the width anchor for
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

/* BFP epic PR4 (R-P4): argmax over BFP values is NOT argmax over mantissas —
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

    maxPool1dForward(&layer, input, output);

    TEST_ASSERT_EQUAL_MEMORY_MESSAGE(kBfpMaxPoolExpectedForward, output->data,
                                     kBfpMaxPoolExpectedForward_len * sizeof(float),
                                     "the winner's EXACT dequant goes to the FLOAT32 raw (D7)");
    TEST_ASSERT_EQUAL_INT32_ARRAY_MESSAGE(kBfpMaxPoolExpectedArgmax, (int32_t *)argmax->data,
                                          kBfpMaxPoolExpectedArgmax_len,
                                          "argmax must follow the DEQUANTIZED comparison, and "
                                          "the funnel must leave auxOut in its INT32 storage");
    freeQuantization(wireQ);
    freeTensor(argmax);
    freeTensor(output);
    freeTensor(input);
}

/* BFP epic PR4 (R-P4 backward): funnel-routed dx mirroring the ARITH_SYM_INT32
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

    maxPool1dForward(&layer, input, output);
    /* Backward TWICE on purpose: dx is OUT_WRITE, so a repeated backward must
     * reproduce the same numbers — and that is also what makes the kernel's
     * memset of the funnel's uninitialized Phase-2 scratch observable (that
     * stack VLA can read as zero on a first call). */
    maxPool1dBackward(&layer, NULL, lossGrad, propLoss);
    maxPool1dBackward(&layer, NULL, lossGrad, propLoss);

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

/* BFP epic PR4 (R-P7d): the rule-1 mirror. A pinned ARITH_BFP forward with a
 * NULL or non-BFP outputQ has no width anchor and must die EAGERLY — the
 * userApi factories copy layerQuant_t slots by value and never call
 * initMaxPool1dConfig, so both spellings are reachable (which is also why the
 * config is built by field assignment here: initMaxPool1dConfig would reject
 * the NULL argmax and derive the math slots for us). */
void testMaxPool1dForwardBfpRequiresBfpOutputQ(void) {
    size_t inputDims[] = {1, 2, 8};
    size_t outputDims[] = {1, 2, 4};
    tensor_t *input = makeFloatTensor(inputDims, 3, NULL);
    tensor_t *output = makeFloatTensor(outputDims, 3, NULL);
    tensor_t *argmax = makeInt32Tensor(outputDims, 3);
    quantization_t *floatQ = quantizationInitFloat();

    kernel_t k;
    initKernel(&k, (size_t)kBfpMaxPoolKernelSize, VALID, (size_t)kBfpMaxPoolDilation,
               (size_t)kBfpMaxPoolStride);
    maxPool1dConfig_t nullCfg = {0};
    nullCfg.kernel = &k;
    nullCfg.argmaxIndices = argmax;
    nullCfg.forwardMath = (arithmetic_t){.type = ARITH_BFP, .roundingMode = HALF_AWAY};
    nullCfg.outputQ = NULL;
    layerConfig_t nullLc = {.maxPool1d = &nullCfg};
    layer_t nullLayer = {.type = MAXPOOL1D, .config = &nullLc};
    ASSERT_EXITS_WITH_FAILURE(maxPool1dForward(&nullLayer, input, output));

    maxPool1dConfig_t floatCfg = nullCfg;
    floatCfg.outputQ = floatQ;
    layerConfig_t floatLc = {.maxPool1d = &floatCfg};
    layer_t floatLayer = {.type = MAXPOOL1D, .config = &floatLc};
    ASSERT_EXITS_WITH_FAILURE(maxPool1dForward(&floatLayer, input, output));

    freeQuantization(floatQ);
    freeTensor(argmax);
    freeTensor(output);
    freeTensor(input);
}

/* BFP epic PR4 (F5): the forward writes rawOut AND auxOut at the same flat
 * index (b * channels + c) * outputLength + outPos, with batch/channels taken
 * from the INPUT — so both need all three dims validated, and auxOut most of
 * all: it is never funnel-converted, so nothing else looks at its shape before
 * the kernel writes it raw. Assertion 1 is the case a length-only check
 * misses; assertion 2 is the same hole on the argmax side. */
void testMaxPool1dForwardBfpRejectsMismatchedOutputShapes(void) {
    size_t inputDims[] = {1, 2, 8};
    size_t goodDims[] = {1, 2, 4};
    size_t wrongChannels[] = {1, 1, 4}; /* right length, wrong channels */
    size_t wrongRank[] = {2, 4};
    int32_t inCodes[16];
    for (size_t i = 0; i < kBfpMaxPoolInCodes_len; i++) {
        inCodes[i] = kBfpMaxPoolInCodes[i];
    }
    tensor_t *input = buildBfpWireWithCodes(
        inputDims, 3, (uint8_t)kBfpMaxPoolMantissaBits, (uint8_t)kBfpMaxPoolExponentBits,
        (size_t)kBfpMaxPoolInNumGroups, (size_t)kBfpMaxPoolInGroupSize, inCodes, kBfpMaxPoolInExps);
    tensor_t *goodOut = makeFloatTensor(goodDims, 3, NULL);
    tensor_t *badOut = makeFloatTensor(wrongChannels, 3, NULL);
    tensor_t *badRankOut = makeFloatTensor(wrongRank, 2, NULL);
    tensor_t *goodArgmax = makeInt32Tensor(goodDims, 3);
    tensor_t *badArgmax = makeInt32Tensor(wrongChannels, 3);
    quantization_t *wireQ = quantizationInitBfpGrouped(
        (uint8_t)kBfpMaxPoolMantissaBits, (uint8_t)kBfpMaxPoolExponentBits, HALF_AWAY,
        (size_t)kBfpMaxPoolInNumGroups, (size_t)kBfpMaxPoolInGroupSize);
    kernel_t k;
    maxPool1dConfig_t cfg;
    layerConfig_t lc;
    layer_t layer;
    maxPool1dBuildBfpLayer(&cfg, &k, &lc, &layer, goodArgmax, (size_t)kBfpMaxPoolKernelSize, VALID,
                           (size_t)kBfpMaxPoolDilation, (size_t)kBfpMaxPoolStride, wireQ);

    ASSERT_EXITS_WITH_FAILURE(maxPool1dForward(&layer, input, badOut));
    cfg.argmaxIndices = badArgmax;
    ASSERT_EXITS_WITH_FAILURE(maxPool1dForward(&layer, input, goodOut));
    cfg.argmaxIndices = goodArgmax;
    ASSERT_EXITS_WITH_FAILURE(maxPool1dForward(&layer, input, badRankOut));

    freeQuantization(wireQ);
    freeTensor(badArgmax);
    freeTensor(goodArgmax);
    freeTensor(badRankOut);
    freeTensor(badOut);
    freeTensor(goodOut);
    freeTensor(input);
}

/* BFP epic PR4 (F7): the argmax array is kernel-written and never
 * funnel-converted, so NOTHING upstream validates its CONTENT — only -1 (the
 * empty-window sentinel) is a legal out-of-range value, and any other index
 * >= inputLength scatters a float straight past the end of the raw gradient
 * buffer. The fixture writes the poison index by hand, which is precisely how
 * a stale argmax from a differently-shaped forward would arrive. The second
 * half is the F5 twin: an argmax tensor matching only on outputLength is read
 * past its end for every b, c beyond its own. */
void testMaxPool1dBackwardBfpRejectsOutOfRangeArgmax(void) {
    size_t lossDims[] = {1, 2, 4};
    size_t propDims[] = {1, 2, 8};
    size_t argmaxDims[] = {1, 2, 4};
    /* Right length, wrong channels — and deliberately WIDER (3 channels, not
     * 1): a wider argmax is fully in bounds for the kernel's reads and its
     * calloc-zeroed indices are all legal, so nothing else can reject it. The
     * three-dim shape gate is the ONLY thing standing between this call and a
     * silently-wrong dx. A narrower argmax would be caught by the F7 content
     * check reading heap garbage, which is luck, not coverage. */
    size_t wideArgmaxDims[] = {1, 3, 4};
    int32_t gyCodes[8];
    for (size_t i = 0; i < kBfpMaxPoolGyCodes_len; i++) {
        gyCodes[i] = kBfpMaxPoolGyCodes[i];
    }
    tensor_t *lossGrad = buildBfpWireWithCodes(
        lossDims, 3, (uint8_t)kBfpMaxPoolMantissaBits, (uint8_t)kBfpMaxPoolExponentBits,
        (size_t)kBfpMaxPoolGyNumGroups, (size_t)kBfpMaxPoolGyGroupSize, gyCodes, kBfpMaxPoolGyExps);
    tensor_t *propLoss = makeFloatTensor(propDims, 3, NULL);
    tensor_t *argmax = makeInt32Tensor(argmaxDims, 3);
    tensor_t *wideArgmax = makeInt32Tensor(wideArgmaxDims, 3);
    int32_t *argmaxArr = (int32_t *)argmax->data;
    for (size_t i = 0; i < kBfpMaxPoolExpectedArgmax_len; i++) {
        argmaxArr[i] = kBfpMaxPoolExpectedArgmax[i];
    }
    argmaxArr[3] = 8; /* inputLength is 8, so 8 is one past the last cell */

    quantization_t *wireQ = quantizationInitBfpGrouped(
        (uint8_t)kBfpMaxPoolMantissaBits, (uint8_t)kBfpMaxPoolExponentBits, HALF_AWAY,
        (size_t)kBfpMaxPoolGyNumGroups, (size_t)kBfpMaxPoolGyGroupSize);
    kernel_t k;
    maxPool1dConfig_t cfg;
    layerConfig_t lc;
    layer_t layer;
    maxPool1dBuildBfpLayer(&cfg, &k, &lc, &layer, argmax, (size_t)kBfpMaxPoolKernelSize, VALID,
                           (size_t)kBfpMaxPoolDilation, (size_t)kBfpMaxPoolStride, wireQ);

    ASSERT_EXITS_WITH_FAILURE(maxPool1dBackward(&layer, NULL, lossGrad, propLoss));

    /* -1 in the SAME slot must NOT die: the sentinel stays legal. */
    argmaxArr[3] = -1;
    maxPool1dBackward(&layer, NULL, lossGrad, propLoss);

    cfg.argmaxIndices = wideArgmax;
    ASSERT_EXITS_WITH_FAILURE(maxPool1dBackward(&layer, NULL, lossGrad, propLoss));

    freeQuantization(wireQ);
    freeTensor(wideArgmax);
    freeTensor(argmax);
    freeTensor(propLoss);
    freeTensor(lossGrad);
}

/* BFP epic PR4 (R-P7d): the guard is NARROWED, not removed — a BFP wire under
 * the raw-casting FLOAT32/SYM arms must still die, and the ARITH_BFP backward
 * needs a BFP-typed propLossQ anchor (R-P1). */
void testMaxPool1dBackwardBfpGuardsNarrowedNotRemoved(void) {
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
    ASSERT_EXITS_WITH_FAILURE(maxPool1dBackward(&floatLayer, NULL, bfpLossGrad, floatPropLoss));
    ASSERT_EXITS_WITH_FAILURE(maxPool1dBackward(&floatLayer, NULL, floatLossGrad, bfpPropLoss));

    kernel_t kb;
    initKernel(&kb, (size_t)kBfpMaxPoolKernelSize, VALID, (size_t)kBfpMaxPoolDilation,
               (size_t)kBfpMaxPoolStride);
    maxPool1dConfig_t bfpCfg = {0};
    bfpCfg.kernel = &kb;
    bfpCfg.argmaxIndices = argmax;
    bfpCfg.propLossMath = (arithmetic_t){.type = ARITH_BFP, .roundingMode = HALF_AWAY};
    bfpCfg.propLossQ = NULL;
    layerConfig_t bfpLc = {.maxPool1d = &bfpCfg};
    layer_t bfpLayer = {.type = MAXPOOL1D, .config = &bfpLc};
    ASSERT_EXITS_WITH_FAILURE(maxPool1dBackward(&bfpLayer, NULL, bfpLossGrad, floatPropLoss));

    freeQuantization(floatQ);
    freeTensor(argmax);
    freeTensor(bfpPropLoss);
    freeTensor(floatPropLoss);
    freeTensor(floatLossGrad);
    freeTensor(bfpLossGrad);
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(testMaxPool1dForwardBasic);
    RUN_TEST(testMaxPool1dCalcOutputShapeValidAndSame);
    RUN_TEST(testMaxPool1dBackwardBasic);
    RUN_TEST(testMaxPool1dArgmaxIndicesContent);
    RUN_TEST(testMaxPool1dMultiChannel);
    RUN_TEST(testMaxPool1dMultiBatch);
    RUN_TEST(testMaxPool1dWithStrideAndDilation);
    RUN_TEST(testMaxPool1dWithSamePadding);
    RUN_TEST(testMaxPool1dEdgeCases);
    RUN_TEST(testMaxPool1dForwardSymBasic);
    RUN_TEST(testMaxPool1dBackwardSymBasic);
    RUN_TEST(testMaxPool1dSymStrideDilationForwardBackward);
    RUN_TEST(testMaxPool1dSymTieSamePadding);
    RUN_TEST(testMaxPool1dBackwardSymRejectsWideLossGrad);
    RUN_TEST(testMaxPool1dBackwardSymRejectsTermsOverBound);
    RUN_TEST(testMaxPool1dForwardBfpComparesDequantizedValues);
    RUN_TEST(testMaxPool1dForwardBfpRequiresBfpOutputQ);
    RUN_TEST(testMaxPool1dForwardBfpRejectsMismatchedOutputShapes);
    RUN_TEST(testMaxPool1dBackwardBfpScattersToArgmaxPositions);
    RUN_TEST(testMaxPool1dBackwardBfpRejectsOutOfRangeArgmax);
    RUN_TEST(testMaxPool1dBackwardBfpGuardsNarrowedNotRemoved);
    return UNITY_END();
}
