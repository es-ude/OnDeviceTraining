#define SOURCE_FILE "UNIT_TEST_QUANTIZATION_LAYER"

#include <math.h>
#include <stdint.h>
#include <string.h>

#include "Layer.h"
#include "Quantization.h"
#include "QuantizationApi.h"
#include "QuantizationLayer.h"
#include "Rounding.h"
#include "StorageApi.h"
#include "Tensor.h"
#include "TensorApi.h"
#include "TensorConversion.h"
#include "unity.h"

void setUp(void) {}
void tearDown(void) {}

/* 2D tensor builders (UnitTestLayerNormIntegration pattern). shape_t backing
 * arrays come from reserveMemory because shape_t.dimensions is size_t*. */
static tensor_t *buildSym2D(size_t rows, size_t cols) {
    size_t *dims = reserveMemory(2 * sizeof(size_t));
    dims[0] = rows;
    dims[1] = cols;
    size_t *order = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, order);
    shape_t *shape = reserveMemory(sizeof(shape_t));
    setShape(shape, dims, 2, order);
    return initTensor(shape, quantizationInitSymInt32(HALF_AWAY), NULL);
}

static tensor_t *buildFloat2D(size_t rows, size_t cols) {
    size_t *dims = reserveMemory(2 * sizeof(size_t));
    dims[0] = rows;
    dims[1] = cols;
    size_t *order = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, order);
    shape_t *shape = reserveMemory(sizeof(shape_t));
    setShape(shape, dims, 2, order);
    return initTensor(shape, quantizationInitFloat(), NULL);
}

void testQuantizationForwardSymToSymMatchesDirectRequant(void) {
    /* Linear-SYM-style accumulator-range input: |mantissa| >> 32767. */
    const int32_t mantissas[6] = {1500000, -730000, 42, 0, 999999, -1500000};
    tensor_t *input = buildSym2D(2, 3);
    memcpy(input->data, mantissas, sizeof(mantissas));
    ((symInt32QConfig_t *)input->quantization->qConfig)->scale = 1e-4f;

    tensor_t *expected = buildSym2D(2, 3);
    requantSymInt32Tensor(input, expected);

    /* Fixture sanity: the requant must actually change mantissas + scale,
     * otherwise the differential assert below is vacuous (scale-freeze class). */
    TEST_ASSERT_TRUE(((int32_t *)expected->data)[0] != 1500000);
    TEST_ASSERT_TRUE(((symInt32QConfig_t *)expected->quantization->qConfig)->scale != 1e-4f);

    tensor_t *actual = buildSym2D(2, 3);
    layer_t qLayer = {.type = QUANTIZATION, .config = NULL}; /* kernel reads only tensors */
    quantizationForward(&qLayer, input, actual);

    int32_t expectedMantissas[6];
    int32_t actualMantissas[6];
    memcpy(expectedMantissas, expected->data, sizeof(expectedMantissas));
    memcpy(actualMantissas, actual->data, sizeof(actualMantissas));
    float expectedScale = ((symInt32QConfig_t *)expected->quantization->qConfig)->scale;
    float actualScale = ((symInt32QConfig_t *)actual->quantization->qConfig)->scale;

    freeTensor(actual);
    freeTensor(expected);
    freeTensor(input);

    TEST_ASSERT_EQUAL_INT32_ARRAY(expectedMantissas, actualMantissas, 6);
    TEST_ASSERT_EQUAL_FLOAT(expectedScale, actualScale);
}

void testQuantizationForwardSymToFloatMatchesConvertTensor(void) {
    const int32_t mantissas[4] = {12000, -32767, 5, 32767};
    tensor_t *input = buildSym2D(2, 2);
    memcpy(input->data, mantissas, sizeof(mantissas));
    ((symInt32QConfig_t *)input->quantization->qConfig)->scale = 0.5f;

    tensor_t *expected = buildFloat2D(2, 2);
    convertTensor(input, expected);

    tensor_t *actual = buildFloat2D(2, 2);
    layer_t qLayer = {.type = QUANTIZATION, .config = NULL};
    quantizationForward(&qLayer, input, actual);

    float expectedVals[4];
    float actualVals[4];
    memcpy(expectedVals, expected->data, sizeof(expectedVals));
    memcpy(actualVals, actual->data, sizeof(actualVals));

    freeTensor(actual);
    freeTensor(expected);
    freeTensor(input);

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(expectedVals, actualVals, 4);
    TEST_ASSERT_EQUAL_FLOAT(6000.0f, actualVals[0]); /* anti-vacuity: 12000 * 0.5 */
}

void testQuantizationBackwardRequantsDyMantissasLikeDirectRequant(void) {
    const int32_t dyMantissas[6] = {800000, -123456, 7, -800000, 65536, 0};
    tensor_t *loss = buildSym2D(2, 3);
    memcpy(loss->data, dyMantissas, sizeof(dyMantissas));
    ((symInt32QConfig_t *)loss->quantization->qConfig)->scale = 2e-5f;

    tensor_t *expected = buildSym2D(2, 3);
    requantSymInt32Tensor(loss, expected);

    tensor_t *forwardInput = buildSym2D(2, 3); /* unused: straight-through backward */
    tensor_t *propLoss = buildSym2D(2, 3);
    layer_t qLayer = {.type = QUANTIZATION, .config = NULL};
    quantizationBackward(&qLayer, forwardInput, loss, propLoss);

    int32_t expectedMantissas[6];
    int32_t actualMantissas[6];
    memcpy(expectedMantissas, expected->data, sizeof(expectedMantissas));
    memcpy(actualMantissas, propLoss->data, sizeof(actualMantissas));
    float expectedScale = ((symInt32QConfig_t *)expected->quantization->qConfig)->scale;
    float actualScale = ((symInt32QConfig_t *)propLoss->quantization->qConfig)->scale;

    freeTensor(propLoss);
    freeTensor(forwardInput);
    freeTensor(expected);
    freeTensor(loss);

    TEST_ASSERT_EQUAL_INT32_ARRAY(expectedMantissas, actualMantissas, 6);
    TEST_ASSERT_EQUAL_FLOAT(expectedScale, actualScale);
}

void testQuantizationCalcOutputShapeIdentityIncludingPermutedOrder(void) {
    size_t inDims[2] = {2, 3};
    size_t inOrder[2] = {1, 0}; /* transposed view */
    shape_t inputShape = {
        .dimensions = inDims, .numberOfDimensions = 2, .orderOfDimensions = inOrder};

    size_t outDims[2] = {0, 0};
    size_t outOrder[2] = {0, 0};
    shape_t outputShape = {
        .dimensions = outDims, .numberOfDimensions = 0, .orderOfDimensions = outOrder};

    layer_t qLayer = {.type = QUANTIZATION, .config = NULL};
    quantizationCalcOutputShape(&qLayer, &inputShape, &outputShape);

    TEST_ASSERT_EQUAL_UINT(2, outputShape.numberOfDimensions);
    TEST_ASSERT_EQUAL_UINT(2, outputShape.dimensions[0]);
    TEST_ASSERT_EQUAL_UINT(3, outputShape.dimensions[1]);
    TEST_ASSERT_EQUAL_UINT(1, outputShape.orderOfDimensions[0]); /* permutation preserved */
    TEST_ASSERT_EQUAL_UINT(0, outputShape.orderOfDimensions[1]);
}

void testLayerFunctionsVtableHasQuantizationRow(void) {
    TEST_ASSERT_EQUAL_PTR((void *)quantizationForward,
                          (void *)layerFunctions[QUANTIZATION].forward);
    TEST_ASSERT_EQUAL_PTR((void *)quantizationBackward,
                          (void *)layerFunctions[QUANTIZATION].backward);
    TEST_ASSERT_EQUAL_PTR((void *)quantizationCalcOutputShape,
                          (void *)layerFunctions[QUANTIZATION].calcOutputShape);
}

void testQuantizationConfigUnionMemberRoundTrip(void) {
    quantization_t *fq = quantizationInitSymInt32(HALF_AWAY);
    quantization_t *bq = quantizationInitSymInt32(HALF_AWAY);
    quantizationConfig_t cfg = {.forwardQ = fq, .backwardQ = bq, .ownsQuantizations = false};
    layerConfig_t lc = {.quantization = &cfg};
    layer_t layer;
    initLayer(&layer, QUANTIZATION, &lc);

    TEST_ASSERT_EQUAL_INT(QUANTIZATION, layer.type);
    TEST_ASSERT_EQUAL_PTR(fq, layer.config->quantization->forwardQ);
    TEST_ASSERT_EQUAL_PTR(bq, layer.config->quantization->backwardQ);

    freeQuantization(bq);
    freeQuantization(fq);
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(testQuantizationForwardSymToSymMatchesDirectRequant);
    RUN_TEST(testQuantizationForwardSymToFloatMatchesConvertTensor);
    RUN_TEST(testQuantizationBackwardRequantsDyMantissasLikeDirectRequant);
    RUN_TEST(testQuantizationCalcOutputShapeIdentityIncludingPermutedOrder);
    RUN_TEST(testLayerFunctionsVtableHasQuantizationRow);
    RUN_TEST(testQuantizationConfigUnionMemberRoundTrip);
    return UNITY_END();
}
