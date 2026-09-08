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
    tensorFillFromFloatBuffer(input, (float[]){-1.f, 0.f, 1.f, 2.f, 5.f, -6.f}, 6);

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

    /* 6. ASSERT. */
    float expected[] = {2.3008e-03f, 6.2543e-03f, 1.7001e-02f,
                        4.6213e-02f, 9.2822e-01f, 1.5503e-05f};
    for (size_t i = 0; i < inputSize; i++) {
        TEST_ASSERT_FLOAT_WITHIN(0.0001f, expected[i], captured[i]);
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
    tensorFillFromFloatBuffer(input, (float[]){-1.f, 0.f, 1.f, 2.f, 5.f, -6.f}, 6);

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

    /* 7. ASSERT. */
    float expected[] = {2.3008e-03f, 6.2543e-03f, 1.7001e-02f,
                        4.6213e-02f, 9.2822e-01f, 1.5503e-05f};
    for (size_t i = 0; i < inputSize; i++) {
        TEST_ASSERT_FLOAT_WITHIN(0.1f, expected[i], captured[i]);
    }
}

/* P6-1 root fix: backward consumes LOGITS -- see docs/superpowers/sdd/
 * 2026-09-08-bfp-pr6-softmax/task-2-brief.md. Fixture X/DLDS/EXPECTED_DX
 * are goldgen'd (generate_expected_softmax.py, self-checked against
 * torch.autograd on softmax(x) with upstream grad DLDS). */
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
 * here), then the same float Jacobian formula -- the existing (loose)
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

/* BFP epic PR2 Task 8 (fourth outside-funnel site): softmaxBackward dispatches
 * on the layer's DECLARED propLossMath and the ARITH_FLOAT32 arm raw-casts all
 * three wires to float* with no dtype check at all -- unlike Relu/Dropout, which
 * carried #315-style arm guards already. Softmax FORWARD is safe (it runs inside
 * executeOp, whose prologue/epilogue convert), and the SYM_INT32 backward arm
 * converts via convertTensor; backward-FLOAT32 is the sole hole.
 *
 * This became REACHABLE with Task 8: before it, a BFP propLossQ died in
 * initGradTensor's default arm, so no BFP wire could ever arrive here. Now the
 * dx wire allocates, and an ARITH_FLOAT32 propLossMath -- pinned, or derived
 * as such before the Task 9 flip -- routes it straight into the raw casts: a
 * ~4x heap over-read on
 * the input/loss side and an over-WRITE into the (4x smaller at 8 mantissa bits)
 * packed propLoss buffer. Keyed on each wire's STORAGE dtype, checked before the
 * dispatch. */
static tensor_t *buildSoftmaxWire1D(size_t n, quantization_t *q) {
    size_t *dims = reserveMemory(sizeof(size_t));
    dims[0] = n;
    size_t *order = reserveMemory(sizeof(size_t));
    setOrderOfDimsForNewTensor(1, order);
    shape_t *shape = reserveMemory(sizeof(shape_t));
    setShape(shape, dims, 1, order);
    return initTensor(shape, q, NULL);
}

void testSoftmaxBackwardRejectsBfpWire(void) {
    quantization_t *floatQ = quantizationInitFloat();
    layerQuant_t lq;
    layerQuantInitUniform(&lq, floatQ);
    layer_t *softmaxLayer = softmaxLayerInit(&lq);
    layerFunctions_t softmaxFns = layerFunctions[SOFTMAX];

    /* propLoss BFP: the DESTINATION of the raw float* writes -- 6 packed bytes
     * receiving 24 bytes of float. */
    tensor_t *input = buildSoftmaxWire1D(6, quantizationInitFloat());
    tensor_t *loss = buildSoftmaxWire1D(6, quantizationInitFloat());
    tensor_t *bfpPropLoss = buildSoftmaxWire1D(6, quantizationInitBfp(8, 8, HALF_AWAY));
    ASSERT_EXITS_WITH_FAILURE(softmaxFns.backward(softmaxLayer, input, loss, bfpPropLoss));

    /* input BFP: the softmax activations the dot product reads. */
    tensor_t *bfpInput = buildSoftmaxWire1D(6, quantizationInitBfp(8, 8, HALF_AWAY));
    tensor_t *propLoss = buildSoftmaxWire1D(6, quantizationInitFloat());
    ASSERT_EXITS_WITH_FAILURE(softmaxFns.backward(softmaxLayer, bfpInput, loss, propLoss));

    /* loss BFP: the incoming gradient. */
    tensor_t *bfpLoss = buildSoftmaxWire1D(6, quantizationInitBfp(8, 8, HALF_AWAY));
    ASSERT_EXITS_WITH_FAILURE(softmaxFns.backward(softmaxLayer, input, bfpLoss, propLoss));

    freeTensor(bfpLoss);
    freeTensor(propLoss);
    freeTensor(bfpInput);
    freeTensor(bfpPropLoss);
    freeTensor(loss);
    freeTensor(input);
    freeSoftmaxLayer(softmaxLayer);
    freeQuantization(floatQ);
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

int main() {
    UNITY_BEGIN();
    RUN_TEST(testSoftmaxBackwardRejectsBfpWire);
    RUN_TEST(unitTestSoftmaxForwardFloat);
    RUN_TEST(unitTestSoftmaxForwardSymInt32);

    RUN_TEST(unitTestSoftmaxBackwardFloat);
    RUN_TEST(unitTestSoftmaxBackwardSymInt32);

    RUN_TEST(testSoftmaxForwardLargeLogitsStaysFinite);
    RUN_TEST(testSoftmaxForwardSymLargeLogitsStaysFinite);

    RUN_TEST(unitTestSoftmaxForwardBfpNativeTrunc);
    RUN_TEST(unitTestSoftmaxForwardBfpNativeHalfAway);
    RUN_TEST(unitTestSoftmaxForwardBfpCoarseNegativeBlock);
    RUN_TEST(unitTestSoftmaxForwardBfpCoarseSaturation);
    RUN_TEST(unitTestSoftmaxForwardBfpStagedWidths);
    RUN_TEST(testSoftmaxForwardBfpRequiresBfpOutputQ);
    RUN_TEST(testSoftmaxLayerInitAndFreeRoundTrip);
    RUN_TEST(testSoftmaxLayerInitBorrowingStoresLqPointers);
    RUN_TEST(testSoftmaxLayerInitOwningDeepCopiesLqPointers);
    return UNITY_END();
}
