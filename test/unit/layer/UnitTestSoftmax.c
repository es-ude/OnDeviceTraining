#include <stdlib.h>

#include "ArithmeticType.h"
#include "DeathTest.h"
#include "LayerQuant.h"
#include "QuantizationApi.h"
#include "Softmax.h"
#include "SoftmaxApi.h"
#include "StorageApi.h"
#include "Tensor.h"
#include "TensorApi.h"
#include "TensorConversion.h"
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
    tensorFillFromFloatBuffer(
        input,
        (float[]){2.3008e-03f, 6.2543e-03f, 1.7001e-02f, 4.6213e-02f, 9.2822e-01f, 1.5503e-05f}, 6);

    /* 2. Build heap loss tensor. */
    size_t *lossDims = reserveMemory(2 * sizeof(size_t));
    lossDims[0] = 2;
    lossDims[1] = 3;
    size_t *lossOrder = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, lossOrder);
    shape_t *lossShape = reserveMemory(sizeof(shape_t));
    setShape(lossShape, lossDims, 2, lossOrder);
    tensor_t *loss = initTensor(lossShape, quantizationInitFloat(), NULL);
    tensorFillFromFloatBuffer(loss, (float[]){0.f, 2.f, -4.f, 6.f, 3.f, 2.f}, 6);

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
    float expected[] = {-6.9173e-03f, -6.2947e-03f, -1.1912e-01f,
                        1.3834e-01f,  -5.9973e-03f, -1.5603e-05f};
    for (size_t i = 0; i < inputSize; i++) {
        TEST_ASSERT_FLOAT_WITHIN(0.0001f, expected[i], captured[i]);
    }
}

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
    tensorFillFromFloatBuffer(
        input,
        (float[]){2.3008e-03f, 6.2543e-03f, 1.7001e-02f, 4.6213e-02f, 9.2822e-01f, 1.5503e-05f}, 6);

    /* 2. Build heap loss tensor (SymInt32). */
    size_t *lossDims = reserveMemory(2 * sizeof(size_t));
    lossDims[0] = 2;
    lossDims[1] = 3;
    size_t *lossOrder = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, lossOrder);
    shape_t *lossShape = reserveMemory(sizeof(shape_t));
    setShape(lossShape, lossDims, 2, lossOrder);
    tensor_t *loss = initTensor(lossShape, quantizationInitSymInt32(HALF_AWAY), NULL);
    tensorFillFromFloatBuffer(loss, (float[]){0.f, 2.f, -4.f, 6.f, 3.f, 2.f}, 6);

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
    float expected[] = {-6.9173e-03f, -6.2947e-03f, -1.1912e-01f,
                        1.3834e-01f,  -5.9973e-03f, -1.5603e-05f};
    for (size_t i = 0; i < inputSize; i++) {
        TEST_ASSERT_FLOAT_WITHIN(0.01f, expected[i], captured[i]);
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
 * dx wire allocates, and pre-flip arithmeticFromQuantization(BFP) ==
 * ARITH_FLOAT32 routes it straight into the raw casts -- a ~4x heap over-read on
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

int main() {
    UNITY_BEGIN();
    RUN_TEST(testSoftmaxBackwardRejectsBfpWire);
    RUN_TEST(unitTestSoftmaxForwardFloat);
    RUN_TEST(unitTestSoftmaxForwardSymInt32);

    RUN_TEST(unitTestSoftmaxBackwardFloat);
    RUN_TEST(unitTestSoftmaxBackwardSymInt32);

    RUN_TEST(testSoftmaxForwardLargeLogitsStaysFinite);
    RUN_TEST(testSoftmaxForwardSymLargeLogitsStaysFinite);
    RUN_TEST(testSoftmaxLayerInitAndFreeRoundTrip);
    RUN_TEST(testSoftmaxLayerInitBorrowingStoresLqPointers);
    RUN_TEST(testSoftmaxLayerInitOwningDeepCopiesLqPointers);
    return UNITY_END();
}
