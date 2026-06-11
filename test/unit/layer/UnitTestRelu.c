#include "DTypes.h"
#include "LayerQuant.h"
#include "Quantization.h"
#include "QuantizationApi.h"
#include "Relu.h"
#include "ReluApi.h"
#include "StorageApi.h"
#include "Tensor.h"
#include "TensorApi.h"
#include "TensorConversion.h"
#include "unity.h"

void testReluForwardFloat() {
    size_t numberOfElements = 6;

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

    /* 3. Build shared float quantization for the layer. */
    quantization_t *floatQ = quantizationInitFloat();
    layer_t *reluLayer = reluLayerInitLegacy(floatQ, floatQ);

    /* 4. Exercise. */
    reluForward(reluLayer, input, output);

    /* 5. CAPTURE assertion values. */
    float captured[6];
    readBytesAsFloatArray(numberOfElements, output->data, captured);

    /* 6. FREE in reverse-init order. freeReluLayer releases only the layer
     *    config wrapper; the shared floatQ is freed exactly once at the end. */
    freeReluLayerLegacy(reluLayer);
    freeTensor(output);
    freeTensor(input);
    freeQuantization(floatQ);

    /* 7. ASSERT on captured. */
    float expected[] = {0.f, 0.f, 1.f, 2.f, 5.f, 0.f};
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(expected, captured, numberOfElements);
}

void testReluForwardSymInt32() {
    size_t numberOfValues = 6;

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
    layer_t *reluLayer = reluLayerInitLegacy(symIntQ, symIntQ);
    layerFunctions_t reluFns = layerFunctions[RELU];
    reluFns.forward(reluLayer, input, output);

    /* 4. Convert SymInt32 output back to Float for comparison; output buffer
     *    is heap-allocated to keep us in the heap-tier idiom. */
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
    for (size_t i = 0; i < numberOfValues; i++) {
        captured[i] = ((float *)outputFloat->data)[i];
    }

    /* 6. FREE. */
    freeTensor(outputFloat);
    freeReluLayerLegacy(reluLayer);
    freeTensor(output);
    freeTensor(input);
    freeQuantization(symIntQ);

    /* 7. ASSERT. */
    float expected[] = {0, 0, 1, 2, 5, 0};
    for (size_t i = 0; i < numberOfValues; i++) {
        TEST_ASSERT_FLOAT_WITHIN(0.1f, expected[i], captured[i]);
    }
}

void testReluBackwardFloat() {
    size_t numberOfElements = 6;

    /* 1. Build heap forwardInput tensor. */
    size_t *fwdDims = reserveMemory(1 * sizeof(size_t));
    fwdDims[0] = numberOfElements;
    size_t *fwdOrder = reserveMemory(1 * sizeof(size_t));
    setOrderOfDimsForNewTensor(1, fwdOrder);
    shape_t *fwdShape = reserveMemory(sizeof(shape_t));
    setShape(fwdShape, fwdDims, 1, fwdOrder);
    tensor_t *forwardInput = initTensor(fwdShape, quantizationInitFloat(), NULL);
    tensorFillFromFloatBuffer(forwardInput, (float[]){-1.f, 0.f, 1.f, 2.f, 5.f, -6.f}, 6);

    /* 2. Build heap loss tensor. */
    size_t *lossDims = reserveMemory(1 * sizeof(size_t));
    lossDims[0] = numberOfElements;
    size_t *lossOrder = reserveMemory(1 * sizeof(size_t));
    setOrderOfDimsForNewTensor(1, lossOrder);
    shape_t *lossShape = reserveMemory(sizeof(shape_t));
    setShape(lossShape, lossDims, 1, lossOrder);
    tensor_t *loss = initTensor(lossShape, quantizationInitFloat(), NULL);
    tensorFillFromFloatBuffer(loss, (float[]){0.f, 2.f, -4.f, 6.f, 3.f, 2.f}, 6);

    /* 3. Build heap propLoss tensor (output of backward). */
    size_t *propLossDims = reserveMemory(1 * sizeof(size_t));
    propLossDims[0] = numberOfElements;
    size_t *propLossOrder = reserveMemory(1 * sizeof(size_t));
    setOrderOfDimsForNewTensor(1, propLossOrder);
    shape_t *propLossShape = reserveMemory(sizeof(shape_t));
    setShape(propLossShape, propLossDims, 1, propLossOrder);
    tensor_t *propLoss = initTensor(propLossShape, quantizationInitFloat(), NULL);

    /* 4. Build the layer with shared float quantization. */
    quantization_t *floatQ = quantizationInitFloat();
    layer_t *reluLayer = reluLayerInitLegacy(floatQ, floatQ);
    layerFunctions_t reluFns = layerFunctions[RELU];
    reluFns.backward(reluLayer, forwardInput, loss, propLoss);

    /* 5. CAPTURE. */
    float captured[6];
    for (size_t i = 0; i < numberOfElements; i++) {
        captured[i] = ((float *)propLoss->data)[i];
    }

    /* 6. FREE. */
    freeReluLayerLegacy(reluLayer);
    freeTensor(propLoss);
    freeTensor(loss);
    freeTensor(forwardInput);
    freeQuantization(floatQ);

    /* 7. ASSERT. */
    float expected[] = {0.f, 0.f, -4.f, 6.f, 3.f, 0.f};
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(expected, captured, numberOfElements);
}

void testReluBackwardSymInt32() {
    size_t numberOfValues = 6;

    /* 1. Build heap forwardInput tensor (SymInt32). */
    size_t *fwdDims = reserveMemory(1 * sizeof(size_t));
    fwdDims[0] = numberOfValues;
    size_t *fwdOrder = reserveMemory(1 * sizeof(size_t));
    setOrderOfDimsForNewTensor(1, fwdOrder);
    shape_t *fwdShape = reserveMemory(sizeof(shape_t));
    setShape(fwdShape, fwdDims, 1, fwdOrder);
    tensor_t *forwardInput = initTensor(fwdShape, quantizationInitSymInt32(HALF_AWAY), NULL);
    tensorFillFromFloatBuffer(forwardInput, (float[]){-1.f, 0.f, 1.f, 2.f, 5.f, -6.f}, 6);

    /* 2. Build heap loss tensor (SymInt32). */
    size_t *lossDims = reserveMemory(1 * sizeof(size_t));
    lossDims[0] = numberOfValues;
    size_t *lossOrder = reserveMemory(1 * sizeof(size_t));
    setOrderOfDimsForNewTensor(1, lossOrder);
    shape_t *lossShape = reserveMemory(sizeof(shape_t));
    setShape(lossShape, lossDims, 1, lossOrder);
    tensor_t *loss = initTensor(lossShape, quantizationInitSymInt32(HALF_AWAY), NULL);
    tensorFillFromFloatBuffer(loss, (float[]){0.f, 2.f, -4.f, 6.f, 3.f, 2.f}, 6);

    /* 3. Build heap propLoss tensor (SymInt32). */
    size_t *propLossDims = reserveMemory(1 * sizeof(size_t));
    propLossDims[0] = numberOfValues;
    size_t *propLossOrder = reserveMemory(1 * sizeof(size_t));
    setOrderOfDimsForNewTensor(1, propLossOrder);
    shape_t *propLossShape = reserveMemory(sizeof(shape_t));
    setShape(propLossShape, propLossDims, 1, propLossOrder);
    tensor_t *propLoss = initTensor(propLossShape, quantizationInitSymInt32(HALF_AWAY), NULL);

    /* 4. Build layer with shared SymInt32 quantization. */
    quantization_t *symIntQ = quantizationInitSymInt32(HALF_AWAY);
    layer_t *reluLayer = reluLayerInitLegacy(symIntQ, symIntQ);
    layerFunctions_t reluFns = layerFunctions[RELU];
    reluFns.backward(reluLayer, forwardInput, loss, propLoss);

    /* 5. Convert SymInt32 propLoss back to Float for comparison. */
    size_t *propLossFloatDims = reserveMemory(1 * sizeof(size_t));
    propLossFloatDims[0] = numberOfValues;
    size_t *propLossFloatOrder = reserveMemory(1 * sizeof(size_t));
    setOrderOfDimsForNewTensor(1, propLossFloatOrder);
    shape_t *propLossFloatShape = reserveMemory(sizeof(shape_t));
    setShape(propLossFloatShape, propLossFloatDims, 1, propLossFloatOrder);
    tensor_t *propLossFloat = initTensor(propLossFloatShape, quantizationInitFloat(), NULL);
    convertTensor(propLoss, propLossFloat);

    /* 6. CAPTURE. */
    float captured[6];
    for (size_t i = 0; i < numberOfValues; i++) {
        captured[i] = ((float *)propLossFloat->data)[i];
    }

    /* 7. FREE. */
    freeTensor(propLossFloat);
    freeReluLayerLegacy(reluLayer);
    freeTensor(propLoss);
    freeTensor(loss);
    freeTensor(forwardInput);
    freeQuantization(symIntQ);

    /* 8. ASSERT. */
    float expected[] = {0, 0, -4, 6, 3, 0};
    for (size_t i = 0; i < numberOfValues; i++) {
        TEST_ASSERT_FLOAT_WITHIN(0.1f, expected[i], captured[i]);
    }
}

void testReluLayerInitAndFreeRoundTrip(void) {
    /* Roundtrip: reluLayerInit allocates layer + outer layerConfig +
     * inner reluConfig (3 reserveMemory calls). freeReluLayer must
     * release all three. Pre-fix this test runs to completion but leaks
     * the inner reluConfig; post-fix it is leak-clean (verified via the
     * LSan sweep). NULL is acceptable for the quantization arguments —
     * reluLayerInit only stores them, freeReluLayer doesn't touch them. */
    layer_t *reluLayer = reluLayerInitLegacy(NULL, NULL);
    TEST_ASSERT_NOT_NULL(reluLayer);
    TEST_ASSERT_EQUAL_INT(RELU, reluLayer->type);
    TEST_ASSERT_NOT_NULL(reluLayer->config);
    TEST_ASSERT_NOT_NULL(reluLayer->config->relu);

    freeReluLayerLegacy(reluLayer);
}

/* ============================================================================
 * Tests for the new layerQuant_t-based factory API (PR 1).
 * ========================================================================== */

void testReluLayerInitBorrowingStoresLqPointers(void) {
    quantization_t *qFwd = quantizationInitFloat();
    quantization_t *qBwd = quantizationInitFloat();
    layerQuant_t lq = {
        .forwardMath = qFwd,
        .backwardMath = qBwd,
        /* weightStorage / biasStorage ignored by ReLU */
    };

    layer_t *layer = reluLayerInit(&lq);

    TEST_ASSERT_NOT_NULL(layer);
    TEST_ASSERT_EQUAL_INT(RELU, layer->type);

    reluConfig_t *cfg = layer->config->relu;
    TEST_ASSERT_EQUAL_PTR(qFwd, cfg->forwardQ);
    TEST_ASSERT_EQUAL_PTR(qBwd, cfg->backwardQ);
    TEST_ASSERT_FALSE(cfg->ownsQuantizations);

    freeReluLayer(layer);
}

void testReluLayerInitOwningDeepCopiesLqPointers(void) {
    quantization_t *qFwd = quantizationInitFloat();
    quantization_t *qBwd = quantizationInitFloat();
    layerQuant_t lq = {
        .forwardMath = qFwd,
        .backwardMath = qBwd,
    };

    layer_t *layer = reluLayerInitOwning(&lq);

    reluConfig_t *cfg = layer->config->relu;
    TEST_ASSERT_NOT_EQUAL(qFwd, cfg->forwardQ);
    TEST_ASSERT_NOT_EQUAL(qBwd, cfg->backwardQ);
    TEST_ASSERT_EQUAL_INT(qFwd->type, cfg->forwardQ->type);
    TEST_ASSERT_TRUE(cfg->ownsQuantizations);

    freeReluLayer(layer);
}

void setUp() {}
void tearDown() {}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(testReluForwardFloat);
    RUN_TEST(testReluForwardSymInt32);

    RUN_TEST(testReluBackwardFloat);
    RUN_TEST(testReluBackwardSymInt32);

    RUN_TEST(testReluLayerInitAndFreeRoundTrip);

    RUN_TEST(testReluLayerInitBorrowingStoresLqPointers);
    RUN_TEST(testReluLayerInitOwningDeepCopiesLqPointers);
    return UNITY_END();
}
