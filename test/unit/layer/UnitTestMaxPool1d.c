#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "DeathTest.h"
#include "Layer.h"
#include "MaxPool1d.h"
#include "QuantizationApi.h"
#include "StorageApi.h"
#include "TensorApi.h"
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

/* BFP epic PR2 Task 8: 1-D/N-D BFP wire (per-tensor {1,0}, 8-bit mantissas). The
 * buffer is BFP-SIZED, so an unguarded float* access runs past it. */
static tensor_t *makeBfpTensor(size_t const *dims, size_t numDims) {
    return initTensor(makeShape(dims, numDims), quantizationInitBfp(8, 8, HALF_AWAY), NULL);
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

/* BFP epic PR2 Task 8: maxPool1dBackward's ARITH_FLOAT32 arm runs outside
 * executeOp and raw-casts lossGrad/propLoss to float*. Task 8 made BFP dx wires
 * allocatable, and an ARITH_FLOAT32 propLossMath -- pinned, or derived as such
 * before the Task 9 flip -- selects exactly that arm; guard the storage dtype. */
void testMaxPool1dBackwardRejectsBfpWire(void) {
    size_t inputDims[] = {1, 1, 4};
    size_t outputDims[] = {1, 1, 3};
    float outputData[1 * 1 * 3] = {0};
    int32_t argmaxData[1 * 1 * 3] = {0};
    maxPool1dRunResult_t r = maxPool1dBuild(input_maxPool1d_basic, inputDims, 2, VALID, 1, 1,
                                            outputData, argmaxData, outputDims);
    maxPool1dForward(r.layer, r.input, r.output);
    tensor_t *lossGrad = makeFloatTensor(outputDims, 3, NULL);
    tensor_t *propLoss = makeFloatTensor(inputDims, 3, NULL);
    tensor_t *bfpLossGrad = makeBfpTensor(outputDims, 3);
    tensor_t *bfpPropLoss = makeBfpTensor(inputDims, 3);

    ASSERT_EXITS_WITH_FAILURE(maxPool1dBackward(r.layer, r.input, bfpLossGrad, propLoss));
    ASSERT_EXITS_WITH_FAILURE(maxPool1dBackward(r.layer, r.input, lossGrad, bfpPropLoss));

    freeTensor(bfpPropLoss);
    freeTensor(bfpLossGrad);
    freeTensor(propLoss);
    freeTensor(lossGrad);
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(testMaxPool1dBackwardRejectsBfpWire);
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
    return UNITY_END();
}
