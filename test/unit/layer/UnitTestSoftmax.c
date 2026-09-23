#include <stdlib.h>
#include <string.h>

#include "ArithmeticType.h"
#include "DeathTest.h"
#include "LayerQuant.h"
#include "Quantization.h"
#include "QuantizationApi.h"
#include "Softmax.h"
#include "SoftmaxApi.h"
#include "StorageApi.h"
#include "Tensor.h"
#include "TensorApi.h"
#include "TensorConversion.h"
#include "expected_bfp_softmax.h"
#include "expected_softmax.h"
#include "unity.h"

void unitTestSoftmaxForwardFloat() {
    size_t inputSize = 6;

    /* 1. Build heap input tensor (shape 2x3). */
    size_t *inputDims = reserveMemory(2 * sizeof(size_t));
    inputDims[0] = 2;
    inputDims[1] = 3;
    size_t *inputOrder = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, inputOrder);
    shape_t *inputShape = reserveMemory(sizeof(shape_t));
    setShape(inputShape, inputDims, 2, inputOrder);
    tensor_t *input = initTensor(inputShape, quantizationInitFloat(), NULL);
    tensorFillFromFloatBuffer(input, softmaxForwardX, softmaxForwardX_len);

    /* 2. Build heap output tensor (shape 2x3). */
    size_t *outputDims = reserveMemory(2 * sizeof(size_t));
    outputDims[0] = 2;
    outputDims[1] = 3;
    size_t *outputOrder = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, outputOrder);
    shape_t *outputShape = reserveMemory(sizeof(shape_t));
    setShape(outputShape, outputDims, 2, outputOrder);
    tensor_t *output = initTensor(outputShape, quantizationInitFloat(), NULL);

    /* 3. Build the layer with shared float quantization. */
    quantization_t *floatQ = quantizationInitFloat();
    layerQuant_t lq;
    layerQuantInitUniform(&lq, floatQ);
    layer_t *softmaxLayer = softmaxLayerInit(&lq);
    layerFunctions_t softmaxFns = layerFunctions[SOFTMAX];
    softmaxFns.forward(softmaxLayer, input, output);

    /* 4. CAPTURE. */
    float captured[6];
    for (size_t i = 0; i < inputSize; i++) {
        captured[i] = ((float *)output->data)[i];
    }

    /* 5. FREE. */
    freeSoftmaxLayer(softmaxLayer);
    freeTensor(output);
    freeTensor(input);
    freeQuantization(floatQ);

    /* 6. ASSERT: per-row gold (#152, generate_expected_softmax.py section 0)
     * -- each row of the [2,3] input normalizes over its own 3 logits. */
    for (size_t i = 0; i < inputSize; i++) {
        TEST_ASSERT_FLOAT_WITHIN(0.0001f, softmaxForwardExpected[i], captured[i]);
    }
}

void unitTestSoftmaxForwardSymInt32() {
    size_t inputSize = 6;

    /* 1. Build heap input tensor (SymInt32, shape 2x3). */
    size_t *inputDims = reserveMemory(2 * sizeof(size_t));
    inputDims[0] = 2;
    inputDims[1] = 3;
    size_t *inputOrder = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, inputOrder);
    shape_t *inputShape = reserveMemory(sizeof(shape_t));
    setShape(inputShape, inputDims, 2, inputOrder);
    tensor_t *input = initTensor(inputShape, quantizationInitSymInt32(HALF_AWAY), NULL);
    tensorFillFromFloatBuffer(input, softmaxForwardX, softmaxForwardX_len);

    /* 2. Build heap output tensor (SymInt32, shape 2x3). */
    size_t *outputDims = reserveMemory(2 * sizeof(size_t));
    outputDims[0] = 2;
    outputDims[1] = 3;
    size_t *outputOrder = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, outputOrder);
    shape_t *outputShape = reserveMemory(sizeof(shape_t));
    setShape(outputShape, outputDims, 2, outputOrder);
    tensor_t *output = initTensor(outputShape, quantizationInitSymInt32(HALF_AWAY), NULL);

    /* 3. Shared SymInt32 quantization for the layer. */
    quantization_t *symIntQ = quantizationInitSymInt32(HALF_AWAY);
    layerQuant_t lq;
    layerQuantInitUniform(&lq, symIntQ);
    layer_t *softmaxLayer = softmaxLayerInit(&lq);
    layerFunctions_t softmaxFns = layerFunctions[SOFTMAX];
    softmaxFns.forward(softmaxLayer, input, output);

    /* 4. Convert SymInt32 output back to Float for comparison. */
    size_t *outFloatDims = reserveMemory(2 * sizeof(size_t));
    outFloatDims[0] = 2;
    outFloatDims[1] = 3;
    size_t *outFloatOrder = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, outFloatOrder);
    shape_t *outFloatShape = reserveMemory(sizeof(shape_t));
    setShape(outFloatShape, outFloatDims, 2, outFloatOrder);
    tensor_t *outputFloat = initTensor(outFloatShape, quantizationInitFloat(), NULL);
    convertTensor(output, outputFloat);

    /* 5. CAPTURE. */
    float captured[6];
    for (size_t i = 0; i < inputSize; i++) {
        captured[i] = ((float *)outputFloat->data)[i];
    }

    /* 6. FREE. */
    freeTensor(outputFloat);
    freeSoftmaxLayer(softmaxLayer);
    freeTensor(output);
    freeTensor(input);
    freeQuantization(symIntQ);

    /* 7. ASSERT: the same per-row gold (#152) at 0.01, the SYM backward
     * twin's tolerance. The int12 in/out quantization error stays below
     * 1e-3; 0.01 lets four of the six elements (not just two, as at the old
     * 0.1) catch a whole-tensor partition sum. */
    for (size_t i = 0; i < inputSize; i++) {
        TEST_ASSERT_FLOAT_WITHIN(0.01f, softmaxForwardExpected[i], captured[i]);
    }
}

/* P6-1 root fix: backward consumes LOGITS -- see docs/conventions/
 * arithmetic-bfp.md §5.9/§11 (Correction 1, P6-1). Fixture X/DLDS/EXPECTED_DX
 * are goldgen'd per row (#152: each row of the [2,3] input is its own softmax;
 * generate_expected_softmax.py section 1, self-checked against torch.autograd
 * on the per-row softmax with upstream grad DLDS). */
void unitTestSoftmaxBackwardFloat() {
    size_t inputSize = 6;

    /* 1. Build heap input tensor. */
    size_t *inputDims = reserveMemory(2 * sizeof(size_t));
    inputDims[0] = 2;
    inputDims[1] = 3;
    size_t *inputOrder = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, inputOrder);
    shape_t *inputShape = reserveMemory(sizeof(shape_t));
    setShape(inputShape, inputDims, 2, inputOrder);
    tensor_t *input = initTensor(inputShape, quantizationInitFloat(), NULL);
    tensorFillFromFloatBuffer(input, (float *)softmaxBackwardX, softmaxBackwardX_len);

    /* 2. Build heap loss tensor. */
    size_t *lossDims = reserveMemory(2 * sizeof(size_t));
    lossDims[0] = 2;
    lossDims[1] = 3;
    size_t *lossOrder = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, lossOrder);
    shape_t *lossShape = reserveMemory(sizeof(shape_t));
    setShape(lossShape, lossDims, 2, lossOrder);
    tensor_t *loss = initTensor(lossShape, quantizationInitFloat(), NULL);
    tensorFillFromFloatBuffer(loss, (float *)softmaxBackwardDLds, softmaxBackwardDLds_len);

    /* 3. Build heap propLoss tensor. */
    size_t *propLossDims = reserveMemory(2 * sizeof(size_t));
    propLossDims[0] = 2;
    propLossDims[1] = 3;
    size_t *propLossOrder = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, propLossOrder);
    shape_t *propLossShape = reserveMemory(sizeof(shape_t));
    setShape(propLossShape, propLossDims, 2, propLossOrder);
    tensor_t *propLoss = initTensor(propLossShape, quantizationInitFloat(), NULL);

    /* 4. Build layer. */
    quantization_t *floatQ = quantizationInitFloat();
    layerQuant_t lq;
    layerQuantInitUniform(&lq, floatQ);
    layer_t *softmaxLayer = softmaxLayerInit(&lq);
    layerFunctions_t softmaxFns = layerFunctions[SOFTMAX];
    softmaxFns.backward(softmaxLayer, input, loss, propLoss);

    /* 5. CAPTURE. */
    float captured[6];
    for (size_t i = 0; i < inputSize; i++) {
        captured[i] = ((float *)propLoss->data)[i];
    }

    /* 6. FREE. */
    freeSoftmaxLayer(softmaxLayer);
    freeTensor(propLoss);
    freeTensor(loss);
    freeTensor(input);
    freeQuantization(floatQ);

    /* 7. ASSERT. */
    for (size_t i = 0; i < inputSize; i++) {
        TEST_ASSERT_FLOAT_WITHIN(0.0001f, softmaxBackwardExpectedDx[i], captured[i]);
    }
}

/* P6-1 SYM_INT32 arm: same X/DLDS as unitTestSoftmaxBackwardFloat, but
 * quantized through the layer's SymInt32 wires. softmaxBackwardSymExpectedDx
 * is goldgen'd from X requantized/dequantized at the fixture's int12
 * per-tensor absmax grid (the SAME grid tensorFillFromFloatBuffer derives
 * here), then the same per-row (#152) float Jacobian formula -- the (loose)
 * tolerance below absorbs the rest of the quantization noise (DLDS and
 * propLoss also round-trip through SymInt32). */
void unitTestSoftmaxBackwardSymInt32() {
    size_t inputSize = 6;

    /* 1. Build heap input tensor (SymInt32). */
    size_t *inputDims = reserveMemory(2 * sizeof(size_t));
    inputDims[0] = 2;
    inputDims[1] = 3;
    size_t *inputOrder = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, inputOrder);
    shape_t *inputShape = reserveMemory(sizeof(shape_t));
    setShape(inputShape, inputDims, 2, inputOrder);
    tensor_t *input = initTensor(inputShape, quantizationInitSymInt32(HALF_AWAY), NULL);
    tensorFillFromFloatBuffer(input, (float *)softmaxBackwardX, softmaxBackwardX_len);

    /* 2. Build heap loss tensor (SymInt32). */
    size_t *lossDims = reserveMemory(2 * sizeof(size_t));
    lossDims[0] = 2;
    lossDims[1] = 3;
    size_t *lossOrder = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, lossOrder);
    shape_t *lossShape = reserveMemory(sizeof(shape_t));
    setShape(lossShape, lossDims, 2, lossOrder);
    tensor_t *loss = initTensor(lossShape, quantizationInitSymInt32(HALF_AWAY), NULL);
    tensorFillFromFloatBuffer(loss, (float *)softmaxBackwardDLds, softmaxBackwardDLds_len);

    /* 3. Build heap propLoss tensor (SymInt32). */
    size_t *propLossDims = reserveMemory(2 * sizeof(size_t));
    propLossDims[0] = 2;
    propLossDims[1] = 3;
    size_t *propLossOrder = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, propLossOrder);
    shape_t *propLossShape = reserveMemory(sizeof(shape_t));
    setShape(propLossShape, propLossDims, 2, propLossOrder);
    tensor_t *propLoss = initTensor(propLossShape, quantizationInitSymInt32(HALF_AWAY), NULL);

    /* 4. Build layer. */
    quantization_t *symIntQ = quantizationInitSymInt32(HALF_AWAY);
    layerQuant_t lq;
    layerQuantInitUniform(&lq, symIntQ);
    layer_t *softmaxLayer = softmaxLayerInit(&lq);
    layerFunctions_t softmaxFns = layerFunctions[SOFTMAX];
    softmaxFns.backward(softmaxLayer, input, loss, propLoss);

    /* 5. Convert SymInt32 propLoss back to Float for comparison. */
    size_t *propLossFloatDims = reserveMemory(2 * sizeof(size_t));
    propLossFloatDims[0] = 2;
    propLossFloatDims[1] = 3;
    size_t *propLossFloatOrder = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, propLossFloatOrder);
    shape_t *propLossFloatShape = reserveMemory(sizeof(shape_t));
    setShape(propLossFloatShape, propLossFloatDims, 2, propLossFloatOrder);
    tensor_t *propLossFloat = initTensor(propLossFloatShape, quantizationInitFloat(), NULL);
    convertTensor(propLoss, propLossFloat);

    /* 6. CAPTURE. */
    float captured[6];
    for (size_t i = 0; i < inputSize; i++) {
        captured[i] = ((float *)propLossFloat->data)[i];
    }

    /* 7. FREE. */
    freeTensor(propLossFloat);
    freeSoftmaxLayer(softmaxLayer);
    freeTensor(propLoss);
    freeTensor(loss);
    freeTensor(input);
    freeQuantization(symIntQ);

    /* 8. ASSERT. */
    for (size_t i = 0; i < inputSize; i++) {
        TEST_ASSERT_FLOAT_WITHIN(0.01f, softmaxBackwardSymExpectedDx[i], captured[i]);
    }
}

/* Large-logit regression fixture (#201 closure residual, folded into #206):
 * pins the shared funnel kernel's max-subtraction. UnitTestSoftmax's other
 * fixtures keep logits in [-6, 5], where deleting the stabilization still
 * passes every assertion; at logits ~90 an unstabilized expf(90) overflows to
 * inf and the outputs collapse to NaN — the WITHIN asserts below fail on NaN.
 * Analytic reference: max=90 -> exps [1, e^-1, e^-89], p = [0.73106, 0.26894,
 * ~1.6e-39]. */
void testSoftmaxForwardLargeLogitsStaysFinite(void) {
    size_t *inputDims = reserveMemory(2 * sizeof(size_t));
    inputDims[0] = 1;
    inputDims[1] = 3;
    size_t *inputOrder = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, inputOrder);
    shape_t *inputShape = reserveMemory(sizeof(shape_t));
    setShape(inputShape, inputDims, 2, inputOrder);
    tensor_t *input = initTensor(inputShape, quantizationInitFloat(), NULL);
    tensorFillFromFloatBuffer(input, (float[]){90.f, 89.f, 1.f}, 3);

    size_t *outputDims = reserveMemory(2 * sizeof(size_t));
    outputDims[0] = 1;
    outputDims[1] = 3;
    size_t *outputOrder = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, outputOrder);
    shape_t *outputShape = reserveMemory(sizeof(shape_t));
    setShape(outputShape, outputDims, 2, outputOrder);
    tensor_t *output = initTensor(outputShape, quantizationInitFloat(), NULL);

    quantization_t *floatQ = quantizationInitFloat();
    layerQuant_t lq;
    layerQuantInitUniform(&lq, floatQ);
    layer_t *softmaxLayer = softmaxLayerInit(&lq);
    layerFunctions[SOFTMAX].forward(softmaxLayer, input, output);

    float captured[3];
    for (size_t i = 0; i < 3; i++) {
        captured[i] = ((float *)output->data)[i];
    }

    freeSoftmaxLayer(softmaxLayer);
    freeTensor(output);
    freeTensor(input);
    freeQuantization(floatQ);

    float expected[] = {0.73106f, 0.26894f, 0.0f};
    for (size_t i = 0; i < 3; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-4f, expected[i], captured[i]);
    }
}

/* SYM_INT32 twin: the int12 scale (90/2047 ~ 0.044) accommodates the logit;
 * the funnel prologue dequantizes, the stabilized kernel runs, the epilogue
 * requantizes. Tolerance 0.02 covers the input-quantization shift of the
 * logit gap (<= ~0.009 on p0) — an unstabilized kernel still lands at
 * NaN/garbage, far outside it. */
void testSoftmaxForwardSymLargeLogitsStaysFinite(void) {
    size_t *inputDims = reserveMemory(2 * sizeof(size_t));
    inputDims[0] = 1;
    inputDims[1] = 3;
    size_t *inputOrder = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, inputOrder);
    shape_t *inputShape = reserveMemory(sizeof(shape_t));
    setShape(inputShape, inputDims, 2, inputOrder);
    tensor_t *input = initTensor(inputShape, quantizationInitSymInt32(HALF_AWAY), NULL);
    tensorFillFromFloatBuffer(input, (float[]){90.f, 89.f, 1.f}, 3);

    size_t *outputDims = reserveMemory(2 * sizeof(size_t));
    outputDims[0] = 1;
    outputDims[1] = 3;
    size_t *outputOrder = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, outputOrder);
    shape_t *outputShape = reserveMemory(sizeof(shape_t));
    setShape(outputShape, outputDims, 2, outputOrder);
    tensor_t *output = initTensor(outputShape, quantizationInitSymInt32(HALF_AWAY), NULL);

    quantization_t *symIntQ = quantizationInitSymInt32(HALF_AWAY);
    layerQuant_t lq;
    layerQuantInitUniform(&lq, symIntQ);
    layer_t *softmaxLayer = softmaxLayerInit(&lq);
    layerFunctions[SOFTMAX].forward(softmaxLayer, input, output);

    size_t *outFloatDims = reserveMemory(2 * sizeof(size_t));
    outFloatDims[0] = 1;
    outFloatDims[1] = 3;
    size_t *outFloatOrder = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, outFloatOrder);
    shape_t *outFloatShape = reserveMemory(sizeof(shape_t));
    setShape(outFloatShape, outFloatDims, 2, outFloatOrder);
    tensor_t *outputFloat = initTensor(outFloatShape, quantizationInitFloat(), NULL);
    convertTensor(output, outputFloat);

    float captured[3];
    for (size_t i = 0; i < 3; i++) {
        captured[i] = ((float *)outputFloat->data)[i];
    }

    freeTensor(outputFloat);
    freeSoftmaxLayer(softmaxLayer);
    freeTensor(output);
    freeTensor(input);
    freeQuantization(symIntQ);

    float expected[] = {0.73106f, 0.26894f, 0.0f};
    for (size_t i = 0; i < 3; i++) {
        TEST_ASSERT_FLOAT_WITHIN(0.02f, expected[i], captured[i]);
    }
}

void testSoftmaxLayerInitAndFreeRoundTrip(void) {
    /* Roundtrip: softmaxLayerInit allocates layer + outer layerConfig +
     * inner softmaxConfig (3 reserveMemory calls). freeSoftmaxLayer must
     * release all three. Leak verification is delegated to the LSan
     * sweep — this test asserts only that the round-trip completes
     * without a crash and that the layer was wired correctly. */
    quantization_t *floatQ = quantizationInitFloat();
    layerQuant_t lq;
    layerQuantInitUniform(&lq, floatQ);
    layer_t *softmaxLayer = softmaxLayerInit(&lq);
    TEST_ASSERT_NOT_NULL(softmaxLayer);
    TEST_ASSERT_EQUAL_INT(SOFTMAX, softmaxLayer->type);
    TEST_ASSERT_NOT_NULL(softmaxLayer->config);
    TEST_ASSERT_NOT_NULL(softmaxLayer->config->softmax);

    freeSoftmaxLayer(softmaxLayer);

    /* floatQ is owned by the test; freeSoftmaxLayer must not have freed
     * it (quantization configs are externally owned and shared). */
    freeQuantization(floatQ);
}

/* ============================================================================
 * Tests for the new layerQuant_t-based Softmax factory (PR 2).
 * ========================================================================== */

void testSoftmaxLayerInitBorrowingStoresLqPointers(void) {
    quantization_t *qFwd = quantizationInitFloat();
    quantization_t *qBwd = quantizationInitFloat();
    layerQuant_t lq = {
        .forwardMath = arithmeticFromQuantization(qFwd),
        .propLossMath = arithmeticFromQuantization(qBwd),
        .outputQ = qFwd,
        .propLossQ = qBwd,
    };

    layer_t *layer = softmaxLayerInit(&lq);

    TEST_ASSERT_NOT_NULL(layer);
    TEST_ASSERT_EQUAL_INT(SOFTMAX, layer->type);

    softmaxConfig_t *cfg = layer->config->softmax;
    TEST_ASSERT_EQUAL_PTR(qFwd, cfg->outputQ);
    TEST_ASSERT_EQUAL_PTR(qBwd, cfg->propLossQ);
    TEST_ASSERT_EQUAL_INT(ARITH_FLOAT32, cfg->forwardMath.type);
    TEST_ASSERT_EQUAL_INT(ARITH_FLOAT32, cfg->propLossMath.type);
    TEST_ASSERT_FALSE(cfg->ownsQuantizations);
    /* P6-2: the shift-rounding knob is an ORTHOGONAL config field (never
     * derived from any roundingMode_t) and factories default it to TRUNC. */
    TEST_ASSERT_EQUAL_INT(BFP_SHIFT_TRUNC, cfg->bfpExpShiftRounding);

    freeSoftmaxLayer(layer);
    freeQuantization(qFwd);
    freeQuantization(qBwd);
}

void testSoftmaxLayerInitOwningDeepCopiesLqPointers(void) {
    quantization_t *qFwd = quantizationInitFloat();
    quantization_t *qBwd = quantizationInitFloat();
    layerQuant_t lq = {
        .forwardMath = arithmeticFromQuantization(qFwd),
        .propLossMath = arithmeticFromQuantization(qBwd),
        .outputQ = qFwd,
        .propLossQ = qBwd,
    };

    layer_t *layer = softmaxLayerInitOwning(&lq);

    softmaxConfig_t *cfg = layer->config->softmax;
    TEST_ASSERT_NOT_EQUAL(qFwd, cfg->outputQ);
    TEST_ASSERT_NOT_EQUAL(qBwd, cfg->propLossQ);
    TEST_ASSERT_EQUAL_INT(qFwd->type, cfg->outputQ->type);
    TEST_ASSERT_EQUAL_INT(ARITH_FLOAT32, cfg->forwardMath.type);
    TEST_ASSERT_TRUE(cfg->ownsQuantizations);
    TEST_ASSERT_EQUAL_INT(BFP_SHIFT_TRUNC, cfg->bfpExpShiftRounding);

    freeSoftmaxLayer(layer);
    freeQuantization(qFwd);
    freeQuantization(qBwd);
}

void setUp() {}
void tearDown() {}

/* 1-D wire builder for the BFP fixtures (n elements, caller-chosen storage).
 * The PR2-Task-8 blanket "no BFP wires on softmaxBackward" guard this file
 * once pinned is RETIRED with PR6 Task 5: the backward now carries a native
 * ARITH_BFP funnel arm; the FLOAT32/SYM arms keep per-arm guards (the
 * funnel's Decision-11 twin -- see softmaxBackward). */
static tensor_t *buildSoftmaxWire1D(size_t n, quantization_t *q) {
    size_t *dims = reserveMemory(sizeof(size_t));
    dims[0] = n;
    size_t *order = reserveMemory(sizeof(size_t));
    setOrderOfDimsForNewTensor(1, order);
    shape_t *shape = reserveMemory(sizeof(shape_t));
    setShape(shape, dims, 1, order);
    return initTensor(shape, q, NULL);
}

/* N-d FLOAT32 wire builder for the per-row (#152) fixtures: identity order,
 * dims copied from the caller, filled from `values` when non-NULL. */
static tensor_t *buildSoftmaxWireNd(const size_t *dimsIn, size_t rank, const float *values) {
    size_t *dims = reserveMemory(rank * sizeof(size_t));
    memcpy(dims, dimsIn, rank * sizeof(size_t));
    size_t *order = reserveMemory(rank * sizeof(size_t));
    setOrderOfDimsForNewTensor(rank, order);
    shape_t *shape = reserveMemory(sizeof(shape_t));
    setShape(shape, dims, rank, order);
    tensor_t *t = initTensor(shape, quantizationInitFloat(), NULL);
    if (values != NULL) {
        tensorFillFromFloatBuffer(t, values, calcNumberOfElementsByTensor(t));
    }
    return t;
}

/* #152 invariant: every row of a multi-row input is its own distribution. A
 * whole-tensor partition sum makes the rows share one denominator, so no row
 * sums to 1. The rows are deliberately unlike each other: a wide spread, a
 * uniform row, a dominant large logit, a small-magnitude row. */
void testSoftmaxForwardRowsEachSumToOne(void) {
    const size_t dims[2] = {4, 5};
    const float x[20] = {3.1f,   -2.4f, 0.7f, 8.9f,  -5.5f, 0.0f, 0.0f,  0.0f,    0.0f, 0.0f,
                         -30.0f, 25.0f, 1.0f, -1.0f, 12.0f, 0.5f, 0.25f, -0.125f, 2.0f, -4.0f};
    tensor_t *input = buildSoftmaxWireNd(dims, 2, x);
    tensor_t *output = buildSoftmaxWireNd(dims, 2, NULL);
    quantization_t *floatQ = quantizationInitFloat();
    layerQuant_t lq;
    layerQuantInitUniform(&lq, floatQ);
    layer_t *softmaxLayer = softmaxLayerInit(&lq);

    layerFunctions[SOFTMAX].forward(softmaxLayer, input, output);

    float rowSums[4];
    for (size_t r = 0; r < 4; r++) {
        rowSums[r] = 0.0f;
        for (size_t i = 0; i < 5; i++) {
            rowSums[r] += ((float *)output->data)[r * 5 + i];
        }
    }

    freeSoftmaxLayer(softmaxLayer);
    freeTensor(output);
    freeTensor(input);
    freeQuantization(floatQ);

    for (size_t r = 0; r < 4; r++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-5f, 1.0f, rowSums[r]);
    }
}

/* A rank-1 input is ONE row (#152): the whole vector normalizes together,
 * exactly the pre-#152 behaviour (softmaxForwardRank1Expected is the old
 * whole-vector gold). Pins that the row geometry never reads a rank-1 dims[0]
 * as a row count, which would make six one-element rows (every output 1.0). */
void testSoftmaxForwardRank1IsOneRow(void) {
    const size_t dims[1] = {6};
    tensor_t *input = buildSoftmaxWireNd(dims, 1, softmaxForwardX);
    tensor_t *output = buildSoftmaxWireNd(dims, 1, NULL);
    quantization_t *floatQ = quantizationInitFloat();
    layerQuant_t lq;
    layerQuantInitUniform(&lq, floatQ);
    layer_t *softmaxLayer = softmaxLayerInit(&lq);

    layerFunctions[SOFTMAX].forward(softmaxLayer, input, output);

    float captured[6];
    for (size_t i = 0; i < 6; i++) {
        captured[i] = ((float *)output->data)[i];
    }

    freeSoftmaxLayer(softmaxLayer);
    freeTensor(output);
    freeTensor(input);
    freeQuantization(floatQ);

    for (size_t i = 0; i < 6; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-4f, softmaxForwardRank1Expected[i], captured[i]);
    }
}

/* Rank 3 [2,2,3]: a row is EVERYTHING after axis 0 (6 elements), not the last
 * axis -- PyTorch's softmax(dim=-1) would normalize each 3-element vector.
 * The generator asserts the two differ, so this pins the row geometry. */
void testSoftmaxForwardRank3RowSpansTrailingAxes(void) {
    const size_t dims[3] = {2, 2, 3};
    tensor_t *input = buildSoftmaxWireNd(dims, 3, softmaxRank3X);
    tensor_t *output = buildSoftmaxWireNd(dims, 3, NULL);
    quantization_t *floatQ = quantizationInitFloat();
    layerQuant_t lq;
    layerQuantInitUniform(&lq, floatQ);
    layer_t *softmaxLayer = softmaxLayerInit(&lq);

    layerFunctions[SOFTMAX].forward(softmaxLayer, input, output);

    float captured[12];
    for (size_t i = 0; i < 12; i++) {
        captured[i] = ((float *)output->data)[i];
    }

    freeSoftmaxLayer(softmaxLayer);
    freeTensor(output);
    freeTensor(input);
    freeQuantization(floatQ);

    for (size_t i = 0; i < 12; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-4f, softmaxRank3ExpectedS[i], captured[i]);
    }
}

/* Rows at very different logit scales ([120, 119, 1] next to [-1, 0, 1]):
 * each row subtracts ITS OWN max. One shared max (120) underflows every exp
 * of row 1 to 0 in float32, and its 0/0 divide yields NaN, which fails the
 * WITHIN asserts. */
void testSoftmaxForwardMixedScaleRowsStayFinite(void) {
    const size_t dims[2] = {2, 3};
    tensor_t *input = buildSoftmaxWireNd(dims, 2, softmaxMixedScaleX);
    tensor_t *output = buildSoftmaxWireNd(dims, 2, NULL);
    quantization_t *floatQ = quantizationInitFloat();
    layerQuant_t lq;
    layerQuantInitUniform(&lq, floatQ);
    layer_t *softmaxLayer = softmaxLayerInit(&lq);

    layerFunctions[SOFTMAX].forward(softmaxLayer, input, output);

    float captured[6];
    for (size_t i = 0; i < 6; i++) {
        captured[i] = ((float *)output->data)[i];
    }

    freeSoftmaxLayer(softmaxLayer);
    freeTensor(output);
    freeTensor(input);
    freeQuantization(floatQ);

    for (size_t i = 0; i < 6; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-6f, softmaxMixedScaleExpected[i], captured[i]);
    }
}

/* A zero-copy transposed view (transposeTensor swaps orderOfDimensions only)
 * with more than one STORAGE row: physical [2, 3] transposed is logically
 * [3, 2], but the per-row walk runs over physical storage, so it would
 * normalize a partition that is not the logical rows. Fail fast instead
 * (GroupNorm.c:67's identity-order rule). */
void testSoftmaxForwardRejectsTransposedMultiRow(void) {
    const size_t dims[2] = {2, 3};
    tensor_t *input = buildSoftmaxWireNd(dims, 2, softmaxForwardX);
    tensor_t *output = buildSoftmaxWireNd(dims, 2, NULL);
    transposeTensor(input, 0, 1);
    quantization_t *floatQ = quantizationInitFloat();
    layerQuant_t lq;
    layerQuantInitUniform(&lq, floatQ);
    layer_t *softmaxLayer = softmaxLayerInit(&lq);

    ASSERT_EXITS_WITH_FAILURE(layerFunctions[SOFTMAX].forward(softmaxLayer, input, output));

    freeSoftmaxLayer(softmaxLayer);
    freeTensor(output);
    freeTensor(input);
    freeQuantization(floatQ);
}

/* The row count is the STORAGE dims[0] (the field CrossEntropy.c:42 reads),
 * not the logical axis 0: physical [1, 6] transposed is logically [6, 1], yet
 * it has ONE storage row, so the whole six-element vector normalizes together
 * (the whole-vector gold), exactly as before #152 and as CE's MEAN divisor
 * counts it. Pins that the identity-order rule binds only when storage
 * dims[0] > 1. */
void testSoftmaxForwardTransposedSingleRowIsOneRow(void) {
    const size_t dims[2] = {1, 6};
    tensor_t *input = buildSoftmaxWireNd(dims, 2, softmaxForwardX);
    tensor_t *output = buildSoftmaxWireNd(dims, 2, NULL);
    transposeTensor(input, 0, 1);
    quantization_t *floatQ = quantizationInitFloat();
    layerQuant_t lq;
    layerQuantInitUniform(&lq, floatQ);
    layer_t *softmaxLayer = softmaxLayerInit(&lq);

    layerFunctions[SOFTMAX].forward(softmaxLayer, input, output);

    float captured[6];
    for (size_t i = 0; i < 6; i++) {
        captured[i] = ((float *)output->data)[i];
    }

    freeSoftmaxLayer(softmaxLayer);
    freeTensor(output);
    freeTensor(input);
    freeQuantization(floatQ);

    for (size_t i = 0; i < 6; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-4f, softmaxForwardRank1Expected[i], captured[i]);
    }
}

/* Empty rows ([2, 0]: two rows of zero elements) are a no-op -- no max read
 * from an empty row. The RED of this pin is sanitizer-only: under
 * unit_test_asan the unguarded kernel's x[0] read is a heap-buffer-overflow;
 * the plain presets read a stray float and carry on. */
void testSoftmaxForwardEmptyRowsAreNoOp(void) {
    const size_t dims[2] = {2, 0};
    tensor_t *input = buildSoftmaxWireNd(dims, 2, NULL);
    tensor_t *output = buildSoftmaxWireNd(dims, 2, NULL);
    quantization_t *floatQ = quantizationInitFloat();
    layerQuant_t lq;
    layerQuantInitUniform(&lq, floatQ);
    layer_t *softmaxLayer = softmaxLayerInit(&lq);

    layerFunctions[SOFTMAX].forward(softmaxLayer, input, output);
    size_t capturedCount = calcNumberOfElementsByTensor(output);

    freeSoftmaxLayer(softmaxLayer);
    freeTensor(output);
    freeTensor(input);
    freeQuantization(floatQ);

    TEST_ASSERT_EQUAL_size_t(0, capturedCount);
}

/* Backward twin of testSoftmaxForwardRank1IsOneRow: a rank-1 input is ONE row,
 * so the dx is the whole-vector Jacobian-VJP -- softmaxBackwardRank1ExpectedDx
 * is byte-identical to the pre-#152 softmaxBackwardExpectedDx gold. */
void testSoftmaxBackwardRank1IsOneRow(void) {
    const size_t dims[1] = {6};
    tensor_t *input = buildSoftmaxWireNd(dims, 1, softmaxBackwardX);
    tensor_t *loss = buildSoftmaxWireNd(dims, 1, softmaxBackwardDLds);
    tensor_t *propLoss = buildSoftmaxWireNd(dims, 1, NULL);
    quantization_t *floatQ = quantizationInitFloat();
    layerQuant_t lq;
    layerQuantInitUniform(&lq, floatQ);
    layer_t *softmaxLayer = softmaxLayerInit(&lq);

    layerFunctions[SOFTMAX].backward(softmaxLayer, input, loss, propLoss);

    float captured[6];
    for (size_t i = 0; i < 6; i++) {
        captured[i] = ((float *)propLoss->data)[i];
    }

    freeSoftmaxLayer(softmaxLayer);
    freeTensor(propLoss);
    freeTensor(loss);
    freeTensor(input);
    freeQuantization(floatQ);

    for (size_t i = 0; i < 6; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-4f, softmaxBackwardRank1ExpectedDx[i], captured[i]);
    }
}

/* Backward twin of testSoftmaxForwardRank3RowSpansTrailingAxes: the Jacobian
 * is block-diagonal over the 2 rows of 6 trailing elements -- each row
 * recomputes its own s and takes its own dot. */
void testSoftmaxBackwardRank3RowSpansTrailingAxes(void) {
    const size_t dims[3] = {2, 2, 3};
    tensor_t *input = buildSoftmaxWireNd(dims, 3, softmaxRank3X);
    tensor_t *loss = buildSoftmaxWireNd(dims, 3, softmaxRank3DLds);
    tensor_t *propLoss = buildSoftmaxWireNd(dims, 3, NULL);
    quantization_t *floatQ = quantizationInitFloat();
    layerQuant_t lq;
    layerQuantInitUniform(&lq, floatQ);
    layer_t *softmaxLayer = softmaxLayerInit(&lq);

    layerFunctions[SOFTMAX].backward(softmaxLayer, input, loss, propLoss);

    float captured[12];
    for (size_t i = 0; i < 12; i++) {
        captured[i] = ((float *)propLoss->data)[i];
    }

    freeSoftmaxLayer(softmaxLayer);
    freeTensor(propLoss);
    freeTensor(loss);
    freeTensor(input);
    freeQuantization(floatQ);

    for (size_t i = 0; i < 12; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-4f, softmaxRank3ExpectedDx[i], captured[i]);
    }
}

/* Backward twin of testSoftmaxForwardRejectsTransposedMultiRow. */
void testSoftmaxBackwardRejectsTransposedMultiRow(void) {
    const size_t dims[2] = {2, 3};
    tensor_t *input = buildSoftmaxWireNd(dims, 2, softmaxBackwardX);
    tensor_t *loss = buildSoftmaxWireNd(dims, 2, softmaxBackwardDLds);
    tensor_t *propLoss = buildSoftmaxWireNd(dims, 2, NULL);
    transposeTensor(input, 0, 1);
    quantization_t *floatQ = quantizationInitFloat();
    layerQuant_t lq;
    layerQuantInitUniform(&lq, floatQ);
    layer_t *softmaxLayer = softmaxLayerInit(&lq);

    ASSERT_EXITS_WITH_FAILURE(
        layerFunctions[SOFTMAX].backward(softmaxLayer, input, loss, propLoss));

    freeSoftmaxLayer(softmaxLayer);
    freeTensor(propLoss);
    freeTensor(loss);
    freeTensor(input);
    freeQuantization(floatQ);
}

/* Backward twin of testSoftmaxForwardEmptyRowsAreNoOp on the FLOAT32 arm: no
 * max read from an empty row and no zero-length VLA. Sanitizer-only RED
 * (unit_test_asan: UBSan vla-bound on the unguarded row scratch). */
void testSoftmaxBackwardEmptyRowsAreNoOp(void) {
    const size_t dims[2] = {2, 0};
    tensor_t *input = buildSoftmaxWireNd(dims, 2, NULL);
    tensor_t *loss = buildSoftmaxWireNd(dims, 2, NULL);
    tensor_t *propLoss = buildSoftmaxWireNd(dims, 2, NULL);
    quantization_t *floatQ = quantizationInitFloat();
    layerQuant_t lq;
    layerQuantInitUniform(&lq, floatQ);
    layer_t *softmaxLayer = softmaxLayerInit(&lq);

    layerFunctions[SOFTMAX].backward(softmaxLayer, input, loss, propLoss);
    size_t capturedCount = calcNumberOfElementsByTensor(propLoss);

    freeSoftmaxLayer(softmaxLayer);
    freeTensor(propLoss);
    freeTensor(loss);
    freeTensor(input);
    freeQuantization(floatQ);

    TEST_ASSERT_EQUAL_size_t(0, capturedCount);
}

/* ---- BFP epic PR6 Task 4: native ARITH_BFP forward (P6-2..P6-5) ----
 *
 * Gold: expected_bfp_softmax.h (generate_expected_bfp_softmax.py mirrors the
 * numerics-spec steps 1-5 bit-exactly; the script asserts the TRUNC and
 * HALF_AWAY wires differ, so the two knob tests double as the knob
 * discriminator). Every gold test asserts the ARITH_BFP derivation through
 * the ordinary config path before running. */

/* Packed BFP wire from explicit codes + per-group exponents (the sanctioned
 * fixture route, mirrored from UnitTestLayerNorm.c's buildBfpWireWithCodesLn:
 * byteConversion pack + exponent memcpy, arithmetic-bfp.md §5.7 inventory).
 * Writing the payload directly keeps the fixture independent of the quantizer
 * and pins the exponents the kernel borrows. */
static tensor_t *buildSmBfpWireWithCodes(size_t n, uint8_t mantissaBits, uint8_t exponentBits,
                                         size_t numGroups, size_t groupSize, int32_t const *codes,
                                         uint8_t const *exponents) {
    quantization_t *q = (groupSize == 0)
                            ? quantizationInitBfp(mantissaBits, exponentBits, HALF_AWAY)
                            : quantizationInitBfpGrouped(mantissaBits, exponentBits, HALF_AWAY,
                                                         numGroups, groupSize);
    size_t *dims = reserveMemory(sizeof(size_t));
    dims[0] = n;
    size_t *order = reserveMemory(sizeof(size_t));
    setOrderOfDimsForNewTensor(1, order);
    shape_t *shape = reserveMemory(sizeof(shape_t));
    setShape(shape, dims, 1, order);
    tensor_t *t = initTensor(shape, q, NULL);
    int32_t packSrc[n]; /* byteConversion takes a mutable source */
    memcpy(packSrc, codes, n * sizeof(int32_t));
    byteConversion((uint8_t *)packSrc, 32, t->data, mantissaBits, n);
    bfpQConfig_t *qc = t->quantization->qConfig;
    memcpy(qc->exponents, exponents, qc->numGroups);
    return t;
}

/* SM-A's input wire (grouped {2, 4}, two DIFFERENT stored exponents, so the
 * block-B alignment shift is 4 bits with nonzero remainders). */
static tensor_t *buildSmBfpAInput(void) {
    return buildSmBfpWireWithCodes(8, (uint8_t)kSmBfpXMantissaBits, (uint8_t)kSmBfpXExponentBits,
                                   (size_t)kSmBfpXNumGroups, (size_t)kSmBfpXGroupSize, kSmBfpXCodes,
                                   kSmBfpXExponents);
}

/* A freshly zero-seeded produced wire (all-zero codes, all-bias exponents),
 * so every emitted code/exponent comes from the OUT_WRITE epilogue. */
static tensor_t *buildSmBfpOutputWire(uint8_t mantissaBits) {
    return buildSmBfpWireWithCodes(8, mantissaBits, (uint8_t)kSmBfpOutExponentBits,
                                   (size_t)kSmBfpOutNumGroups, (size_t)kSmBfpOutGroupSize,
                                   kSmBfpOutZeroCodes, kSmBfpOutZeroExponents);
}

/* The primary oracle at the default knob (TRUNC, I-BERT-faithful): BFP-stored
 * input borrowed zero-copy, integer alignment + i-exp on the block mantissas,
 * float boundary, OUT_WRITE pack at the grouped {2, 4} m=8 outputQ. */
void unitTestSoftmaxForwardBfpNativeTrunc(void) {
    tensor_t *in = buildSmBfpAInput();
    tensor_t *out = buildSmBfpOutputWire((uint8_t)kSmBfpOutMantissaBits);

    layerQuant_t lq;
    layerQuantInitUniform(&lq, out->quantization);
    layer_t *softmaxLayer = softmaxLayerInit(&lq);
    /* Derived through the ordinary config path -- pins that the flip holds. */
    TEST_ASSERT_EQUAL_INT(ARITH_BFP, softmaxLayer->config->softmax->forwardMath.type);

    layerFunctions[SOFTMAX].forward(softmaxLayer, in, out);

    /* 8 = kSmBfpN, 2 = kSmBfpOutNumGroups (literal sizes: a static-const
     * bound would make these VLAs). */
    int32_t got[8];
    uint8_t gotExps[2];
    bfpQConfig_t *outQC = out->quantization->qConfig;
    unpackSignExtend(out->data, outQC->mantissaBits, 0, got, 8);
    memcpy(gotExps, outQC->exponents, outQC->numGroups);

    freeSoftmaxLayer(softmaxLayer);
    freeTensor(out);
    freeTensor(in);

    TEST_ASSERT_EQUAL_INT32_ARRAY(kSmBfpOutCodesTrunc, got, 8);
    TEST_ASSERT_EQUAL_UINT8_ARRAY(kSmBfpOutExponentsTrunc, gotExps, 2);
}

/* Same fixture through the knob's other deterministic position -- the gold
 * wires differ (script-asserted), so PASSING BOTH tests proves the knob
 * actually reaches the kernel's shift sites. */
void unitTestSoftmaxForwardBfpNativeHalfAway(void) {
    tensor_t *in = buildSmBfpAInput();
    tensor_t *out = buildSmBfpOutputWire((uint8_t)kSmBfpOutMantissaBits);

    layerQuant_t lq;
    layerQuantInitUniform(&lq, out->quantization);
    layer_t *softmaxLayer = softmaxLayerInit(&lq);
    TEST_ASSERT_EQUAL_INT(ARITH_BFP, softmaxLayer->config->softmax->forwardMath.type);
    softmaxSetBfpExpShiftRounding(softmaxLayer, BFP_SHIFT_HALF_AWAY);

    layerFunctions[SOFTMAX].forward(softmaxLayer, in, out);

    int32_t got[8];
    uint8_t gotExps[2];
    bfpQConfig_t *outQC = out->quantization->qConfig;
    unpackSignExtend(out->data, outQC->mantissaBits, 0, got, 8);
    memcpy(gotExps, outQC->exponents, outQC->numGroups);

    freeSoftmaxLayer(softmaxLayer);
    freeTensor(out);
    freeTensor(in);

    TEST_ASSERT_EQUAL_INT32_ARRAY(kSmBfpOutCodesHalfAway, got, 8);
    TEST_ASSERT_EQUAL_UINT8_ARRAY(kSmBfpOutExponentsHalfAway, gotExps, 2);
}

/* Fix round 1 (amended spec step 2): the coarse-negative-block regime. Block
 * A holds the SIGNED max (2.0 at E=-5); block B {-100, -80, +1.0, -50} is
 * negative-dominated, so its absmax-minimal grid (E=0) is legitimately
 * COARSER than the argmax block's -- E_i > EMax, the case the old kernel's
 * uint32-wrap/clamp-31 path got wrong by tens of percent (the +1.0 element
 * received ~exp(-x_max) mass instead of ~exp(1-2)/sum ~ 0.2). The amended
 * kernel takes an EXACT saturating left shift; the -100/-80/-50 elements
 * pack to code 0 exactly. Runs BOTH knob positions against per-knob gold --
 * the alignment here is exact, so only bfpIExpQ's >>z differs between them. */
void unitTestSoftmaxForwardBfpCoarseNegativeBlock(void) {
    tensor_t *in = buildSmBfpWireWithCodes(
        8, (uint8_t)kSmBfpXMantissaBits, (uint8_t)kSmBfpXExponentBits, (size_t)kSmBfpXNumGroups,
        (size_t)kSmBfpXGroupSize, kSmBfpCnXCodes, kSmBfpCnXExponents);
    tensor_t *outTrunc = buildSmBfpOutputWire((uint8_t)kSmBfpOutMantissaBits);
    tensor_t *outHalfAway = buildSmBfpOutputWire((uint8_t)kSmBfpOutMantissaBits);

    layerQuant_t lq;
    layerQuantInitUniform(&lq, outTrunc->quantization);
    layer_t *softmaxLayer = softmaxLayerInit(&lq);
    TEST_ASSERT_EQUAL_INT(ARITH_BFP, softmaxLayer->config->softmax->forwardMath.type);

    layerFunctions[SOFTMAX].forward(softmaxLayer, in, outTrunc);
    softmaxSetBfpExpShiftRounding(softmaxLayer, BFP_SHIFT_HALF_AWAY);
    layerFunctions[SOFTMAX].forward(softmaxLayer, in, outHalfAway);

    int32_t gotTrunc[8];
    uint8_t gotTruncExps[2];
    bfpQConfig_t *truncQC = outTrunc->quantization->qConfig;
    unpackSignExtend(outTrunc->data, truncQC->mantissaBits, 0, gotTrunc, 8);
    memcpy(gotTruncExps, truncQC->exponents, truncQC->numGroups);
    int32_t gotHalfAway[8];
    uint8_t gotHalfAwayExps[2];
    bfpQConfig_t *haQC = outHalfAway->quantization->qConfig;
    unpackSignExtend(outHalfAway->data, haQC->mantissaBits, 0, gotHalfAway, 8);
    memcpy(gotHalfAwayExps, haQC->exponents, haQC->numGroups);

    freeSoftmaxLayer(softmaxLayer);
    freeTensor(outHalfAway);
    freeTensor(outTrunc);
    freeTensor(in);

    TEST_ASSERT_EQUAL_INT32_ARRAY(kSmBfpCnOutCodesTrunc, gotTrunc, 8);
    TEST_ASSERT_EQUAL_UINT8_ARRAY(kSmBfpCnOutExponentsTrunc, gotTruncExps, 2);
    TEST_ASSERT_EQUAL_INT32_ARRAY(kSmBfpCnOutCodesHalfAway, gotHalfAway, 8);
    TEST_ASSERT_EQUAL_UINT8_ARRAY(kSmBfpCnOutExponentsHalfAway, gotHalfAwayExps, 2);
}

/* Fix round 2: the saturation sub-branches and the zero-code up>=31 clause.
 * Grouped {4, 2}: g1 (up=24) saturates code -128 via the MAGNITUDE disjunct
 * (|m| > INT32_MAX >> 24), g2 (up=32) via the up>=31 disjunct; each coarse
 * block also holds a ZERO code -- g2's reaches the kernel's up>=31 clause,
 * whose absence is formal UB (0 << 32, C11 6.5.7p3). Masked shifts return
 * the correct 0 on real targets, so there is NO behavioral RED for the
 * zero clause on this preset -- the pin is behavioral-under-sanitizer (the
 * unit_test_asan preset's UBSan aborts on the unfixed kernel) and the gold
 * vector equality is the oracle: the saturated elements pack to 0 while the
 * zero-code elements KEEP their true mass exp(-x_max)/sum (nonzero codes,
 * script-asserted), which any corrupted alignment would move. */
void unitTestSoftmaxForwardBfpCoarseSaturation(void) {
    tensor_t *in = buildSmBfpWireWithCodes(
        8, (uint8_t)kSmBfpXMantissaBits, (uint8_t)kSmBfpXExponentBits, (size_t)kSmBfpCsXNumGroups,
        (size_t)kSmBfpCsXGroupSize, kSmBfpCsXCodes, kSmBfpCsXExponents);
    tensor_t *out = buildSmBfpOutputWire((uint8_t)kSmBfpOutMantissaBits);

    layerQuant_t lq;
    layerQuantInitUniform(&lq, out->quantization);
    layer_t *softmaxLayer = softmaxLayerInit(&lq);
    TEST_ASSERT_EQUAL_INT(ARITH_BFP, softmaxLayer->config->softmax->forwardMath.type);

    layerFunctions[SOFTMAX].forward(softmaxLayer, in, out);

    int32_t got[8];
    uint8_t gotExps[2];
    bfpQConfig_t *outQC = out->quantization->qConfig;
    unpackSignExtend(out->data, outQC->mantissaBits, 0, got, 8);
    memcpy(gotExps, outQC->exponents, outQC->numGroups);

    freeSoftmaxLayer(softmaxLayer);
    freeTensor(out);
    freeTensor(in);

    TEST_ASSERT_EQUAL_INT32_ARRAY(kSmBfpCsOutCodesTrunc, got, 8);
    TEST_ASSERT_EQUAL_UINT8_ARRAY(kSmBfpCsOutExponentsTrunc, gotExps, 2);
}

/* R-N1 staging: a FLOAT32-stored input is staged per-tensor at the ANCHOR
 * widths -- the layer's own produced-wire config (outputQ, m = 6 here), not
 * the operand's width and not a hardcoded 8. The generator asserts an m=8
 * staging changes these codes, so this test pins BOTH the .bfpStage wiring
 * AND the anchor widths. */
void unitTestSoftmaxForwardBfpStagedWidths(void) {
    tensor_t *in = buildSoftmaxWire1D(8, quantizationInitFloat());
    tensorFillFromFloatBuffer(in, (float *)kSmBfpBXValues, kSmBfpBXValues_len);
    tensor_t *out = buildSmBfpOutputWire((uint8_t)kSmBfpBOutMantissaBits);

    layerQuant_t lq;
    layerQuantInitUniform(&lq, out->quantization);
    layer_t *softmaxLayer = softmaxLayerInit(&lq);
    TEST_ASSERT_EQUAL_INT(ARITH_BFP, softmaxLayer->config->softmax->forwardMath.type);

    layerFunctions[SOFTMAX].forward(softmaxLayer, in, out);

    int32_t got[8];
    uint8_t gotExps[2];
    bfpQConfig_t *outQC = out->quantization->qConfig;
    unpackSignExtend(out->data, outQC->mantissaBits, 0, got, 8);
    memcpy(gotExps, outQC->exponents, outQC->numGroups);

    freeSoftmaxLayer(softmaxLayer);
    freeTensor(out);
    freeTensor(in);

    TEST_ASSERT_EQUAL_INT32_ARRAY(kSmBfpBOutCodes, got, 8);
    TEST_ASSERT_EQUAL_UINT8_ARRAY(kSmBfpBOutExponents, gotExps, 2);
}

/* The forward's staging width anchor is the produced-wire config: a pinned
 * ARITH_BFP forwardMath next to a non-BFP outputQ leaves the arm with no
 * width source at all -- fail fast at op entry (bfpWireAnchor), never a
 * silent fallback width. Reachable because userApi factories copy
 * layerQuant_t slots by value. */
void testSoftmaxForwardBfpRequiresBfpOutputQ(void) {
    quantization_t *floatQ = quantizationInitFloat();
    layerQuant_t lq;
    layerQuantInitUniform(&lq, floatQ);
    layer_t *softmaxLayer = softmaxLayerInit(&lq);
    softmaxLayer->config->softmax->forwardMath =
        (arithmetic_t){.type = ARITH_BFP, .roundingMode = HALF_AWAY};

    tensor_t *in = buildSoftmaxWire1D(8, quantizationInitFloat());
    tensorFillFromFloatBuffer(in, (float *)kSmBfpBXValues, kSmBfpBXValues_len);
    tensor_t *out = buildSoftmaxWire1D(8, quantizationInitFloat());

    ASSERT_EXITS_WITH_FAILURE(layerFunctions[SOFTMAX].forward(softmaxLayer, in, out));

    freeTensor(out);
    freeTensor(in);
    freeSoftmaxLayer(softmaxLayer);
    freeQuantization(floatQ);
}

/* ---- BFP epic PR6 Task 5: native ARITH_BFP backward (P6-6) ----
 *
 * ONE funnel op: OUT_WRITE of dx into the propLoss wire, anchored on
 * propLossQ. The kernel recomputes s from the LOGITS via the forward
 * pipeline (same knob), dequantizes dLds exactly, dots in float32 index
 * order, and emits raw = s * (dLds - dot); the epilogue packs at propLossQ.
 * Gold: kSmBfpBwd* (generate_expected_bfp_softmax.py mirrors the kernel
 * statement for statement). */

/* The primary backward oracle: x = SM-A's logits, dLds a per-tensor {1, 0}
 * m=8/e=8 wire on its OWN grid with NON-uniform values (script-asserted
 * dot != 0, so the -dot term is load-bearing -- the uniform-lossGrad
 * lesson), dx packed at the grouped {2, 4} m=8 propLossQ. Knob TRUNC. */
void unitTestSoftmaxBackwardBfpNative(void) {
    tensor_t *in = buildSmBfpAInput();
    tensor_t *loss =
        buildSmBfpWireWithCodes(8, (uint8_t)kSmBfpXMantissaBits, (uint8_t)kSmBfpXExponentBits, 1, 0,
                                kSmBfpBwdDLdsCodes, kSmBfpBwdDLdsExponents);
    tensor_t *propLoss = buildSmBfpOutputWire((uint8_t)kSmBfpOutMantissaBits);

    layerQuant_t lq;
    layerQuantInitUniform(&lq, propLoss->quantization);
    layer_t *softmaxLayer = softmaxLayerInit(&lq);
    /* Derived through the ordinary config path -- pins that the flip holds. */
    TEST_ASSERT_EQUAL_INT(ARITH_BFP, softmaxLayer->config->softmax->propLossMath.type);

    layerFunctions[SOFTMAX].backward(softmaxLayer, in, loss, propLoss);

    int32_t got[8];
    uint8_t gotExps[2];
    bfpQConfig_t *plQC = propLoss->quantization->qConfig;
    unpackSignExtend(propLoss->data, plQC->mantissaBits, 0, got, 8);
    memcpy(gotExps, plQC->exponents, plQC->numGroups);

    freeSoftmaxLayer(softmaxLayer);
    freeTensor(propLoss);
    freeTensor(loss);
    freeTensor(in);

    TEST_ASSERT_EQUAL_INT32_ARRAY(kSmBfpBwdOutCodes, got, 8);
    TEST_ASSERT_EQUAL_UINT8_ARRAY(kSmBfpBwdOutExponents, gotExps, 2);
}

/* The loss-side .bfpStage entry: a FLOAT32-stored dLds is staged per-tensor
 * at the ANCHOR widths (propLossQ, m=8/e=8) with the op's storage-derived
 * rounding. The generator asserts the staging is LOSSY (skipping it moves
 * the packed wire) and that this gold differs from the native one, so the
 * test pins the staging step itself, not just the funnel plumbing. */
void unitTestSoftmaxBackwardBfpStagedLoss(void) {
    tensor_t *in = buildSmBfpAInput();
    tensor_t *loss = buildSoftmaxWire1D(8, quantizationInitFloat());
    tensorFillFromFloatBuffer(loss, (float *)kSmBfpBwdStagedDLdsValues,
                              kSmBfpBwdStagedDLdsValues_len);
    tensor_t *propLoss = buildSmBfpOutputWire((uint8_t)kSmBfpOutMantissaBits);

    layerQuant_t lq;
    layerQuantInitUniform(&lq, propLoss->quantization);
    layer_t *softmaxLayer = softmaxLayerInit(&lq);
    TEST_ASSERT_EQUAL_INT(ARITH_BFP, softmaxLayer->config->softmax->propLossMath.type);

    layerFunctions[SOFTMAX].backward(softmaxLayer, in, loss, propLoss);

    int32_t got[8];
    uint8_t gotExps[2];
    bfpQConfig_t *plQC = propLoss->quantization->qConfig;
    unpackSignExtend(propLoss->data, plQC->mantissaBits, 0, got, 8);
    memcpy(gotExps, plQC->exponents, plQC->numGroups);

    freeSoftmaxLayer(softmaxLayer);
    freeTensor(propLoss);
    freeTensor(loss);
    freeTensor(in);

    TEST_ASSERT_EQUAL_INT32_ARRAY(kSmBfpBwdStagedOutCodes, got, 8);
    TEST_ASSERT_EQUAL_UINT8_ARRAY(kSmBfpBwdStagedOutExponents, gotExps, 2);
}

/* The backward's staging width anchor is the produced-wire config: a pinned
 * ARITH_BFP propLossMath next to a non-BFP propLossQ leaves the arm with no
 * width source at all -- fail fast at op entry (bfpWireAnchor), never a
 * silent fallback width. Reachable because userApi factories copy
 * layerQuant_t slots by value. propLoss is non-NULL here: the anchor binds
 * only when an op actually runs (the NULL ordering is pinned below). */
void testSoftmaxBackwardBfpRequiresBfpPropLossQ(void) {
    quantization_t *floatQ = quantizationInitFloat();
    layerQuant_t lq;
    layerQuantInitUniform(&lq, floatQ);
    layer_t *softmaxLayer = softmaxLayerInit(&lq);
    softmaxLayer->config->softmax->propLossMath =
        (arithmetic_t){.type = ARITH_BFP, .roundingMode = HALF_AWAY};

    tensor_t *in = buildSoftmaxWire1D(8, quantizationInitFloat());
    tensorFillFromFloatBuffer(in, (float *)kSmBfpBXValues, kSmBfpBXValues_len);
    tensor_t *loss = buildSoftmaxWire1D(8, quantizationInitFloat());
    tensorFillFromFloatBuffer(loss, (float *)kSmBfpBwdStagedDLdsValues,
                              kSmBfpBwdStagedDLdsValues_len);
    tensor_t *propLoss = buildSoftmaxWire1D(8, quantizationInitFloat());

    ASSERT_EXITS_WITH_FAILURE(layerFunctions[SOFTMAX].backward(softmaxLayer, in, loss, propLoss));

    freeTensor(propLoss);
    freeTensor(loss);
    freeTensor(in);
    freeSoftmaxLayer(softmaxLayer);
    freeQuantization(floatQ);
}

/* The kernel's loss-count gate (the #436 OOB class): both kernel walks run
 * x's flat n, and a SHORTER per-tensor {1, 0} loss passes every grid check
 * (that sentinel is legal for ANY element count) while lArr[i] reads outside
 * its funnel scratch. Added RED-first against the gate-less kernel (mutation
 * (c)): without the gate the walk completes on garbage and the child exits
 * 0, so this death test fails -- the gate is the sole catcher. */
void testSoftmaxBackwardBfpRejectsCountMismatchLoss(void) {
    static const int32_t xCodes[8] = {10, 20, -30, 40, 5, -6, 7, 8};
    static const int32_t lossCodes[4] = {1, 2, 3, -4};
    static const uint8_t perTensorExp[1] = {127}; /* E = 0 */
    tensor_t *in = buildSmBfpWireWithCodes(
        8, (uint8_t)kSmBfpXMantissaBits, (uint8_t)kSmBfpXExponentBits, 1, 0, xCodes, perTensorExp);
    tensor_t *loss =
        buildSmBfpWireWithCodes(4, (uint8_t)kSmBfpXMantissaBits, (uint8_t)kSmBfpXExponentBits, 1, 0,
                                lossCodes, perTensorExp);
    tensor_t *propLoss = buildSmBfpOutputWire((uint8_t)kSmBfpOutMantissaBits);

    layerQuant_t lq;
    layerQuantInitUniform(&lq, propLoss->quantization);
    layer_t *softmaxLayer = softmaxLayerInit(&lq);

    ASSERT_EXITS_WITH_FAILURE(layerFunctions[SOFTMAX].backward(softmaxLayer, in, loss, propLoss));

    freeSoftmaxLayer(softmaxLayer);
    freeTensor(propLoss);
    freeTensor(loss);
    freeTensor(in);
}

/* P6-6: softmax has NO param grads, so propLoss == NULL means no op runs at
 * all -- the ARITH_BFP arm returns BEFORE the anchor gate binds (documented
 * deviation from the norms' anchor-binds-even-when-NULL rule). The layer
 * here carries the SAME broken config as the death test above (pinned
 * ARITH_BFP propLossMath, FLOAT32 propLossQ): with a non-NULL propLoss it
 * dies at the anchor, so a plain return here pins the gate ORDERING, not
 * just the no-op. The input/loss wires must come back untouched. */
void unitTestSoftmaxBackwardBfpNullPropLossIsNoOp(void) {
    quantization_t *floatQ = quantizationInitFloat();
    layerQuant_t lq;
    layerQuantInitUniform(&lq, floatQ);
    layer_t *softmaxLayer = softmaxLayerInit(&lq);
    softmaxLayer->config->softmax->propLossMath =
        (arithmetic_t){.type = ARITH_BFP, .roundingMode = HALF_AWAY};

    tensor_t *in = buildSmBfpAInput();
    tensor_t *loss =
        buildSmBfpWireWithCodes(8, (uint8_t)kSmBfpXMantissaBits, (uint8_t)kSmBfpXExponentBits, 1, 0,
                                kSmBfpBwdDLdsCodes, kSmBfpBwdDLdsExponents);

    layerFunctions[SOFTMAX].backward(softmaxLayer, in, loss, NULL);

    int32_t gotIn[8];
    int32_t gotLoss[8];
    unpackSignExtend(in->data, (uint8_t)kSmBfpXMantissaBits, 0, gotIn, 8);
    unpackSignExtend(loss->data, (uint8_t)kSmBfpXMantissaBits, 0, gotLoss, 8);

    freeTensor(loss);
    freeTensor(in);
    freeSoftmaxLayer(softmaxLayer);
    freeQuantization(floatQ);

    TEST_ASSERT_EQUAL_INT32_ARRAY(kSmBfpXCodes, gotIn, 8);
    TEST_ASSERT_EQUAL_INT32_ARRAY(kSmBfpBwdDLdsCodes, gotLoss, 8);
}

/* Per-arm guard pins (fix round 1 -- restoring the coverage the deleted
 * blanket-guard test carried): a BFP wire under DECLARED FLOAT32 math must
 * die at the arm's bfpRequireNoBfpWire, on every position -- the FLOAT32
 * arm raw-casts all three wires to float* (a ~4x heap over-read on
 * input/loss, an over-write into the packed propLoss buffer), so a missing
 * guard is silent memory corruption, not garbage values. */
void testSoftmaxBackwardFloatArmRejectsBfpWire(void) {
    quantization_t *floatQ = quantizationInitFloat();
    layerQuant_t lq;
    layerQuantInitUniform(&lq, floatQ);
    layer_t *softmaxLayer = softmaxLayerInit(&lq);
    TEST_ASSERT_EQUAL_INT(ARITH_FLOAT32, softmaxLayer->config->softmax->propLossMath.type);

    /* propLoss BFP: the DESTINATION of the raw float* writes -- 6 packed
     * bytes receiving 24 bytes of float. */
    tensor_t *input = buildSoftmaxWire1D(6, quantizationInitFloat());
    tensor_t *loss = buildSoftmaxWire1D(6, quantizationInitFloat());
    tensor_t *bfpPropLoss = buildSoftmaxWire1D(6, quantizationInitBfp(8, 8, HALF_AWAY));
    ASSERT_EXITS_WITH_FAILURE(
        layerFunctions[SOFTMAX].backward(softmaxLayer, input, loss, bfpPropLoss));

    /* input BFP: the logits the recompute reads. */
    tensor_t *bfpInput = buildSoftmaxWire1D(6, quantizationInitBfp(8, 8, HALF_AWAY));
    tensor_t *propLoss = buildSoftmaxWire1D(6, quantizationInitFloat());
    ASSERT_EXITS_WITH_FAILURE(
        layerFunctions[SOFTMAX].backward(softmaxLayer, bfpInput, loss, propLoss));

    /* loss BFP: the incoming gradient. */
    tensor_t *bfpLoss = buildSoftmaxWire1D(6, quantizationInitBfp(8, 8, HALF_AWAY));
    ASSERT_EXITS_WITH_FAILURE(
        layerFunctions[SOFTMAX].backward(softmaxLayer, input, bfpLoss, propLoss));

    freeTensor(bfpLoss);
    freeTensor(propLoss);
    freeTensor(bfpInput);
    freeTensor(bfpPropLoss);
    freeTensor(loss);
    freeTensor(input);
    freeSoftmaxLayer(softmaxLayer);
    freeQuantization(floatQ);
}

/* The SYM arm's guards -- never exercised before this pin. These are NOT
 * memory-safety guards: convertTensor's BFP->FLOAT32 cell exists, so the
 * arm would silently "work". They are the outside-funnel twin of the
 * funnel's Decision-11 deny (legal BFP-storage arithmetics are FLOAT32
 * fake-quant and BFP native only), so the pin is on POLICY. propLossMath
 * is overridden the way the anchor death test above does it (userApi
 * factories copy layerQuant_t slots by value). */
void testSoftmaxBackwardSymArmRejectsBfpWire(void) {
    quantization_t *floatQ = quantizationInitFloat();
    layerQuant_t lq;
    layerQuantInitUniform(&lq, floatQ);
    layer_t *softmaxLayer = softmaxLayerInit(&lq);
    softmaxLayer->config->softmax->propLossMath =
        (arithmetic_t){.type = ARITH_SYM_INT32, .roundingMode = HALF_AWAY};

    tensor_t *input = buildSoftmaxWire1D(6, quantizationInitFloat());
    tensor_t *loss = buildSoftmaxWire1D(6, quantizationInitFloat());
    tensor_t *bfpPropLoss = buildSoftmaxWire1D(6, quantizationInitBfp(8, 8, HALF_AWAY));
    ASSERT_EXITS_WITH_FAILURE(
        layerFunctions[SOFTMAX].backward(softmaxLayer, input, loss, bfpPropLoss));

    tensor_t *bfpInput = buildSoftmaxWire1D(6, quantizationInitBfp(8, 8, HALF_AWAY));
    tensor_t *propLoss = buildSoftmaxWire1D(6, quantizationInitFloat());
    ASSERT_EXITS_WITH_FAILURE(
        layerFunctions[SOFTMAX].backward(softmaxLayer, bfpInput, loss, propLoss));

    tensor_t *bfpLoss = buildSoftmaxWire1D(6, quantizationInitBfp(8, 8, HALF_AWAY));
    ASSERT_EXITS_WITH_FAILURE(
        layerFunctions[SOFTMAX].backward(softmaxLayer, input, bfpLoss, propLoss));

    freeTensor(bfpLoss);
    freeTensor(propLoss);
    freeTensor(bfpInput);
    freeTensor(bfpPropLoss);
    freeTensor(loss);
    freeTensor(input);
    freeSoftmaxLayer(softmaxLayer);
    freeQuantization(floatQ);
}

/* ---- #152: the BFP arms stay single-row ----
 *
 * The native BFP pipeline runs ONE max/alignment grid and ONE partition sum
 * over the whole wire, so a multi-row BFP call must fail fast (per-row BFP is
 * out of scope, spec D3) while a single row -- rank 1 [N] or rank 2 [1, N] --
 * runs bit-identically to the rank-1 gold. */

/* Re-views a rank-1 fixture wire as [d0, d1] over the same storage (d0 * d1
 * must equal its element count). The dims/order blocks are swapped for
 * reserveMemory'd rank-2 ones, so freeTensor's cascade stays valid. */
static void softmaxWireAs2D(tensor_t *t, size_t d0, size_t d1) {
    freeReservedMemory(t->shape->dimensions);
    freeReservedMemory(t->shape->orderOfDimensions);
    size_t *dims = reserveMemory(2 * sizeof(size_t));
    dims[0] = d0;
    dims[1] = d1;
    size_t *order = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, order);
    setShape(t->shape, dims, 2, order);
}

void testSoftmaxForwardBfpRejectsMultiRow(void) {
    tensor_t *in = buildSmBfpAInput();
    tensor_t *out = buildSmBfpOutputWire((uint8_t)kSmBfpOutMantissaBits);
    softmaxWireAs2D(in, 2, 4);
    softmaxWireAs2D(out, 2, 4);

    layerQuant_t lq;
    layerQuantInitUniform(&lq, out->quantization);
    layer_t *softmaxLayer = softmaxLayerInit(&lq);
    TEST_ASSERT_EQUAL_INT(ARITH_BFP, softmaxLayer->config->softmax->forwardMath.type);

    ASSERT_EXITS_WITH_FAILURE(layerFunctions[SOFTMAX].forward(softmaxLayer, in, out));

    freeSoftmaxLayer(softmaxLayer);
    freeTensor(out);
    freeTensor(in);
}

void testSoftmaxBackwardBfpRejectsMultiRow(void) {
    tensor_t *in = buildSmBfpAInput();
    tensor_t *loss =
        buildSmBfpWireWithCodes(8, (uint8_t)kSmBfpXMantissaBits, (uint8_t)kSmBfpXExponentBits, 1, 0,
                                kSmBfpBwdDLdsCodes, kSmBfpBwdDLdsExponents);
    tensor_t *propLoss = buildSmBfpOutputWire((uint8_t)kSmBfpOutMantissaBits);
    softmaxWireAs2D(in, 2, 4);
    softmaxWireAs2D(loss, 2, 4);
    softmaxWireAs2D(propLoss, 2, 4);

    layerQuant_t lq;
    layerQuantInitUniform(&lq, propLoss->quantization);
    layer_t *softmaxLayer = softmaxLayerInit(&lq);
    TEST_ASSERT_EQUAL_INT(ARITH_BFP, softmaxLayer->config->softmax->propLossMath.type);

    ASSERT_EXITS_WITH_FAILURE(layerFunctions[SOFTMAX].backward(softmaxLayer, in, loss, propLoss));

    freeSoftmaxLayer(softmaxLayer);
    freeTensor(propLoss);
    freeTensor(loss);
    freeTensor(in);
}

/* A rank-2 SINGLE row [1, 8] is still one row: the native forward must emit
 * exactly the rank-1 TRUNC gold. Pins that the guard keys on the row count,
 * not on the rank (the HAR BFP trainer feeds its softmax [1, 6]). */
void unitTestSoftmaxForwardBfpSingleRowRank2MatchesRank1(void) {
    tensor_t *in = buildSmBfpAInput();
    tensor_t *out = buildSmBfpOutputWire((uint8_t)kSmBfpOutMantissaBits);
    softmaxWireAs2D(in, 1, 8);
    softmaxWireAs2D(out, 1, 8);

    layerQuant_t lq;
    layerQuantInitUniform(&lq, out->quantization);
    layer_t *softmaxLayer = softmaxLayerInit(&lq);
    TEST_ASSERT_EQUAL_INT(ARITH_BFP, softmaxLayer->config->softmax->forwardMath.type);

    layerFunctions[SOFTMAX].forward(softmaxLayer, in, out);

    int32_t got[8];
    uint8_t gotExps[2];
    bfpQConfig_t *outQC = out->quantization->qConfig;
    unpackSignExtend(out->data, outQC->mantissaBits, 0, got, 8);
    memcpy(gotExps, outQC->exponents, outQC->numGroups);

    freeSoftmaxLayer(softmaxLayer);
    freeTensor(out);
    freeTensor(in);

    TEST_ASSERT_EQUAL_INT32_ARRAY(kSmBfpOutCodesTrunc, got, 8);
    TEST_ASSERT_EQUAL_UINT8_ARRAY(kSmBfpOutExponentsTrunc, gotExps, 2);
}

/* Backward twin: [1, 8] logits/loss/propLoss reproduce the rank-1 native
 * backward gold (unitTestSoftmaxBackwardBfpNative's fixture). */
void unitTestSoftmaxBackwardBfpSingleRowRank2MatchesRank1(void) {
    tensor_t *in = buildSmBfpAInput();
    tensor_t *loss =
        buildSmBfpWireWithCodes(8, (uint8_t)kSmBfpXMantissaBits, (uint8_t)kSmBfpXExponentBits, 1, 0,
                                kSmBfpBwdDLdsCodes, kSmBfpBwdDLdsExponents);
    tensor_t *propLoss = buildSmBfpOutputWire((uint8_t)kSmBfpOutMantissaBits);
    softmaxWireAs2D(in, 1, 8);
    softmaxWireAs2D(loss, 1, 8);
    softmaxWireAs2D(propLoss, 1, 8);

    layerQuant_t lq;
    layerQuantInitUniform(&lq, propLoss->quantization);
    layer_t *softmaxLayer = softmaxLayerInit(&lq);
    TEST_ASSERT_EQUAL_INT(ARITH_BFP, softmaxLayer->config->softmax->propLossMath.type);

    layerFunctions[SOFTMAX].backward(softmaxLayer, in, loss, propLoss);

    int32_t got[8];
    uint8_t gotExps[2];
    bfpQConfig_t *plQC = propLoss->quantization->qConfig;
    unpackSignExtend(propLoss->data, plQC->mantissaBits, 0, got, 8);
    memcpy(gotExps, plQC->exponents, plQC->numGroups);

    freeSoftmaxLayer(softmaxLayer);
    freeTensor(propLoss);
    freeTensor(loss);
    freeTensor(in);

    TEST_ASSERT_EQUAL_INT32_ARRAY(kSmBfpBwdOutCodes, got, 8);
    TEST_ASSERT_EQUAL_UINT8_ARRAY(kSmBfpBwdOutExponents, gotExps, 2);
}

int main() {
    UNITY_BEGIN();
    RUN_TEST(unitTestSoftmaxForwardFloat);
    RUN_TEST(unitTestSoftmaxForwardSymInt32);

    RUN_TEST(unitTestSoftmaxBackwardFloat);
    RUN_TEST(unitTestSoftmaxBackwardSymInt32);

    RUN_TEST(testSoftmaxForwardLargeLogitsStaysFinite);
    RUN_TEST(testSoftmaxForwardSymLargeLogitsStaysFinite);

    RUN_TEST(testSoftmaxForwardRowsEachSumToOne);
    RUN_TEST(testSoftmaxForwardRank1IsOneRow);
    RUN_TEST(testSoftmaxForwardRank3RowSpansTrailingAxes);
    RUN_TEST(testSoftmaxForwardMixedScaleRowsStayFinite);
    RUN_TEST(testSoftmaxForwardRejectsTransposedMultiRow);
    RUN_TEST(testSoftmaxForwardTransposedSingleRowIsOneRow);
    RUN_TEST(testSoftmaxForwardEmptyRowsAreNoOp);
    RUN_TEST(testSoftmaxBackwardRank1IsOneRow);
    RUN_TEST(testSoftmaxBackwardRank3RowSpansTrailingAxes);
    RUN_TEST(testSoftmaxBackwardRejectsTransposedMultiRow);
    RUN_TEST(testSoftmaxBackwardEmptyRowsAreNoOp);

    RUN_TEST(unitTestSoftmaxForwardBfpNativeTrunc);
    RUN_TEST(unitTestSoftmaxForwardBfpNativeHalfAway);
    RUN_TEST(unitTestSoftmaxForwardBfpCoarseNegativeBlock);
    RUN_TEST(unitTestSoftmaxForwardBfpCoarseSaturation);
    RUN_TEST(unitTestSoftmaxForwardBfpStagedWidths);
    RUN_TEST(testSoftmaxForwardBfpRequiresBfpOutputQ);

    RUN_TEST(unitTestSoftmaxBackwardBfpNative);
    RUN_TEST(unitTestSoftmaxBackwardBfpStagedLoss);
    RUN_TEST(testSoftmaxBackwardBfpRequiresBfpPropLossQ);
    RUN_TEST(testSoftmaxBackwardBfpRejectsCountMismatchLoss);
    RUN_TEST(unitTestSoftmaxBackwardBfpNullPropLossIsNoOp);
    RUN_TEST(testSoftmaxBackwardFloatArmRejectsBfpWire);
    RUN_TEST(testSoftmaxBackwardSymArmRejectsBfpWire);
    RUN_TEST(testSoftmaxForwardBfpRejectsMultiRow);
    RUN_TEST(testSoftmaxBackwardBfpRejectsMultiRow);
    RUN_TEST(unitTestSoftmaxForwardBfpSingleRowRank2MatchesRank1);
    RUN_TEST(unitTestSoftmaxBackwardBfpSingleRowRank2MatchesRank1);

    RUN_TEST(testSoftmaxLayerInitAndFreeRoundTrip);
    RUN_TEST(testSoftmaxLayerInitBorrowingStoresLqPointers);
    RUN_TEST(testSoftmaxLayerInitOwningDeepCopiesLqPointers);
    return UNITY_END();
}
