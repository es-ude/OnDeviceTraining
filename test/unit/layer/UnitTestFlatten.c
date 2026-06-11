#define SOURCE_FILE "UNIT_TEST_FLATTEN"

#include <stdbool.h>

#include "DTypes.h"
#include "Flatten.h"
#include "FlattenApi.h"
#include "Layer.h"
#include "Quantization.h"
#include "QuantizationApi.h"
#include "StorageApi.h"
#include "Tensor.h"
#include "TensorApi.h"
#include "TensorConversion.h"
#include "unity.h"

void testFlattenLayerInit_ReturnsFlattenTypedLayer(void) {
    layer_t *flattenLayer = flattenLayerInit();

    /* CAPTURE before frees. */
    bool capturedNotNull = (flattenLayer != NULL);
    int capturedType = flattenLayer ? (int)flattenLayer->type : -1;
    bool capturedConfigNull = flattenLayer ? (flattenLayer->config == NULL) : false;

    /* FREE. */
    freeFlattenLayer(flattenLayer);

    /* ASSERT. */
    TEST_ASSERT_TRUE(capturedNotNull);
    TEST_ASSERT_EQUAL_INT(FLATTEN, capturedType);
    TEST_ASSERT_TRUE_MESSAGE(capturedConfigNull,
                             "FLATTEN carries no per-layer config; layer->config must be NULL");
}

void testFlattenCalcOutputShape_NonSquareInput(void) {
    // Regression: product of all trailing dims, not the squared last dim.
    // Input [1, 3, 4, 5] must flatten to [1, 60], NOT [1, 25] or similar.
    size_t inputDims[] = {1, 3, 4, 5};
    size_t inputOrder[] = {0, 1, 2, 3};
    shape_t inputShape = {
        .dimensions = inputDims, .orderOfDimensions = inputOrder, .numberOfDimensions = 4};

    size_t outputDimsBuf[4] = {0};
    size_t outputOrderBuf[4] = {0};
    shape_t outputShape = {
        .dimensions = outputDimsBuf, .orderOfDimensions = outputOrderBuf, .numberOfDimensions = 0};

    layer_t *flatten = flattenLayerInit();
    flattenCalcOutputShape(flatten, &inputShape, &outputShape);

    /* CAPTURE before frees. */
    size_t capturedNumDims = outputShape.numberOfDimensions;
    size_t capturedDim0 = outputShape.dimensions[0];
    size_t capturedDim1 = outputShape.dimensions[1];
    size_t capturedOrder0 = outputShape.orderOfDimensions[0];
    size_t capturedOrder1 = outputShape.orderOfDimensions[1];

    /* FREE. */
    freeFlattenLayer(flatten);

    /* ASSERT. */
    TEST_ASSERT_EQUAL_UINT(2, capturedNumDims);
    TEST_ASSERT_EQUAL_UINT(1, capturedDim0);
    TEST_ASSERT_EQUAL_UINT(60, capturedDim1);
    TEST_ASSERT_EQUAL_UINT(0, capturedOrder0);
    TEST_ASSERT_EQUAL_UINT(1, capturedOrder1);
}

void testFlattenForwardFloat_PreservesBytesAndReshapes(void) {
    size_t n = 12; // 1 * 2 * 2 * 3
    float inputData[] = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f, 12.f};

    /* 1. Build heap input tensor (shape 1x2x2x3). */
    size_t *inputDims = reserveMemory(4 * sizeof(size_t));
    inputDims[0] = 1;
    inputDims[1] = 2;
    inputDims[2] = 2;
    inputDims[3] = 3;
    size_t *inputOrder = reserveMemory(4 * sizeof(size_t));
    setOrderOfDimsForNewTensor(4, inputOrder);
    shape_t *inputShape = reserveMemory(sizeof(shape_t));
    setShape(inputShape, inputDims, 4, inputOrder);
    tensor_t *input = initTensor(inputShape, quantizationInitFloat(), NULL);
    tensorFillFromFloatBuffer(input, inputData, n);

    /* 2. Build heap output tensor (shape 1x12). */
    size_t *outputDims = reserveMemory(2 * sizeof(size_t));
    outputDims[0] = 1;
    outputDims[1] = 12;
    size_t *outputOrder = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, outputOrder);
    shape_t *outputShape = reserveMemory(sizeof(shape_t));
    setShape(outputShape, outputDims, 2, outputOrder);
    tensor_t *output = initTensor(outputShape, quantizationInitFloat(), NULL);

    /* 3. Build flatten layer. */
    layer_t *flatten = flattenLayerInit();
    flattenForward(flatten, input, output);

    /* 4. CAPTURE assertion values before frees. */
    float captured[12];
    for (size_t i = 0; i < n; i++) {
        captured[i] = ((float *)output->data)[i];
    }

    /* 5. FREE in reverse-init order. */
    freeFlattenLayer(flatten);
    freeTensor(output);
    freeTensor(input);

    /* 6. ASSERT on captured. */
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(inputData, captured, n);
}

void testFlattenForwardSymInt32_PropagatesScaleAndValues(void) {
    size_t n = 6;
    float inputFloatData[] = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f};

    /* 1. Build heap input tensor (SymInt32, shape 1x2x3). */
    size_t *inputDims = reserveMemory(3 * sizeof(size_t));
    inputDims[0] = 1;
    inputDims[1] = 2;
    inputDims[2] = 3;
    size_t *inputOrder = reserveMemory(3 * sizeof(size_t));
    setOrderOfDimsForNewTensor(3, inputOrder);
    shape_t *inputShape = reserveMemory(sizeof(shape_t));
    setShape(inputShape, inputDims, 3, inputOrder);
    tensor_t *input = initTensor(inputShape, quantizationInitSymInt32(HALF_AWAY), NULL);
    tensorFillFromFloatBuffer(input, inputFloatData, n);

    /* 2. Build heap output tensor (SymInt32, shape 1x6). */
    size_t *outputDims = reserveMemory(2 * sizeof(size_t));
    outputDims[0] = 1;
    outputDims[1] = 6;
    size_t *outputOrder = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, outputOrder);
    shape_t *outputShape = reserveMemory(sizeof(shape_t));
    setShape(outputShape, outputDims, 2, outputOrder);
    tensor_t *output = initTensor(outputShape, quantizationInitSymInt32(HALF_AWAY), NULL);

    /* Clobber the output's scale so we can verify flattenForward writes it. */
    symInt32QConfig_t *outputQC = output->quantization->qConfig;
    outputQC->scale = 0.0f;

    layer_t *flatten = flattenLayerInit();
    flattenForward(flatten, input, output);

    /* 3. Capture scale-propagation observation before frees. */
    symInt32QConfig_t *inputQC = input->quantization->qConfig;
    float capturedInputScale = inputQC->scale;
    float capturedOutputScale = outputQC->scale;

    /* 4. Build heap roundtrip Float tensor (shape 1x6) and convert. */
    size_t *roundTripDims = reserveMemory(2 * sizeof(size_t));
    roundTripDims[0] = 1;
    roundTripDims[1] = 6;
    size_t *roundTripOrder = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, roundTripOrder);
    shape_t *roundTripShape = reserveMemory(sizeof(shape_t));
    setShape(roundTripShape, roundTripDims, 2, roundTripOrder);
    tensor_t *roundTrip = initTensor(roundTripShape, quantizationInitFloat(), NULL);
    convertTensor(output, roundTrip);

    /* 5. CAPTURE roundtrip values before frees. */
    float captured[6];
    for (size_t i = 0; i < n; i++) {
        captured[i] = ((float *)roundTrip->data)[i];
    }

    /* 6. FREE in reverse-init order. */
    freeTensor(roundTrip);
    freeFlattenLayer(flatten);
    freeTensor(output);
    freeTensor(input);

    /* 7. ASSERT. */
    TEST_ASSERT_EQUAL_FLOAT(capturedInputScale, capturedOutputScale);
    for (size_t i = 0; i < n; i++) {
        TEST_ASSERT_FLOAT_WITHIN(0.1f, inputFloatData[i], captured[i]);
    }
}

void testFlattenBackwardFloat_CopiesGradsUnchanged(void) {
    size_t n = 6;
    float lossData[] = {0.1f, 0.2f, -0.3f, 0.4f, 0.5f, 0.6f};

    /* 1. Build heap forwardInput tensor (shape 1x2x3). */
    size_t *forwardDims = reserveMemory(3 * sizeof(size_t));
    forwardDims[0] = 1;
    forwardDims[1] = 2;
    forwardDims[2] = 3;
    size_t *forwardOrder = reserveMemory(3 * sizeof(size_t));
    setOrderOfDimsForNewTensor(3, forwardOrder);
    shape_t *forwardShape = reserveMemory(sizeof(shape_t));
    setShape(forwardShape, forwardDims, 3, forwardOrder);
    tensor_t *forwardInput = initTensor(forwardShape, quantizationInitFloat(), NULL);
    tensorFillFromFloatBuffer(forwardInput, (float[]){-1.f, 0.f, 1.f, 2.f, 5.f, -6.f}, 6);

    /* 2. Build heap loss tensor (shape 1x6). */
    size_t *lossDims = reserveMemory(2 * sizeof(size_t));
    lossDims[0] = 1;
    lossDims[1] = 6;
    size_t *lossOrder = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, lossOrder);
    shape_t *lossShape = reserveMemory(sizeof(shape_t));
    setShape(lossShape, lossDims, 2, lossOrder);
    tensor_t *loss = initTensor(lossShape, quantizationInitFloat(), NULL);
    tensorFillFromFloatBuffer(loss, lossData, n);

    /* 3. Build heap propLoss tensor (shape 1x2x3 — same as forwardInput). */
    size_t *propLossDims = reserveMemory(3 * sizeof(size_t));
    propLossDims[0] = 1;
    propLossDims[1] = 2;
    propLossDims[2] = 3;
    size_t *propLossOrder = reserveMemory(3 * sizeof(size_t));
    setOrderOfDimsForNewTensor(3, propLossOrder);
    shape_t *propLossShape = reserveMemory(sizeof(shape_t));
    setShape(propLossShape, propLossDims, 3, propLossOrder);
    tensor_t *propLoss = initTensor(propLossShape, quantizationInitFloat(), NULL);

    layer_t *flatten = flattenLayerInit();
    flattenBackward(flatten, forwardInput, loss, propLoss);

    /* 4. CAPTURE before frees. */
    float captured[6];
    for (size_t i = 0; i < n; i++) {
        captured[i] = ((float *)propLoss->data)[i];
    }

    /* 5. FREE. */
    freeFlattenLayer(flatten);
    freeTensor(propLoss);
    freeTensor(loss);
    freeTensor(forwardInput);

    /* 6. ASSERT. */
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(lossData, captured, n);
}

void testFlattenBackwardSymInt32_PropagatesScale(void) {
    size_t n = 6;
    float lossFloatData[] = {0.1f, 0.2f, -0.3f, 0.4f, 0.5f, 0.6f};

    /* 1. Build heap forwardInput tensor (SymInt32, shape 1x2x3). */
    size_t *forwardDims = reserveMemory(3 * sizeof(size_t));
    forwardDims[0] = 1;
    forwardDims[1] = 2;
    forwardDims[2] = 3;
    size_t *forwardOrder = reserveMemory(3 * sizeof(size_t));
    setOrderOfDimsForNewTensor(3, forwardOrder);
    shape_t *forwardShape = reserveMemory(sizeof(shape_t));
    setShape(forwardShape, forwardDims, 3, forwardOrder);
    tensor_t *forwardInput = initTensor(forwardShape, quantizationInitSymInt32(HALF_AWAY), NULL);
    tensorFillFromFloatBuffer(forwardInput, (float[]){-1.f, 0.f, 1.f, 2.f, 5.f, -6.f}, 6);

    /* 2. Build heap loss tensor (SymInt32, shape 1x6). */
    size_t *lossDims = reserveMemory(2 * sizeof(size_t));
    lossDims[0] = 1;
    lossDims[1] = 6;
    size_t *lossOrder = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, lossOrder);
    shape_t *lossShape = reserveMemory(sizeof(shape_t));
    setShape(lossShape, lossDims, 2, lossOrder);
    tensor_t *loss = initTensor(lossShape, quantizationInitSymInt32(HALF_AWAY), NULL);
    tensorFillFromFloatBuffer(loss, lossFloatData, n);

    /* 3. Build heap propLoss tensor (SymInt32, shape 1x2x3). */
    size_t *propLossDims = reserveMemory(3 * sizeof(size_t));
    propLossDims[0] = 1;
    propLossDims[1] = 2;
    propLossDims[2] = 3;
    size_t *propLossOrder = reserveMemory(3 * sizeof(size_t));
    setOrderOfDimsForNewTensor(3, propLossOrder);
    shape_t *propLossShape = reserveMemory(sizeof(shape_t));
    setShape(propLossShape, propLossDims, 3, propLossOrder);
    tensor_t *propLoss = initTensor(propLossShape, quantizationInitSymInt32(HALF_AWAY), NULL);

    /* Clobber propLoss scale so we can verify flattenBackward writes it. */
    symInt32QConfig_t *propLossQC = propLoss->quantization->qConfig;
    propLossQC->scale = 0.0f;

    layer_t *flatten = flattenLayerInit();
    flattenBackward(flatten, forwardInput, loss, propLoss);

    /* 4. Capture scale-propagation observation before frees. */
    symInt32QConfig_t *lossQC = loss->quantization->qConfig;
    float capturedLossScale = lossQC->scale;
    float capturedPropLossScale = propLossQC->scale;

    /* 5. Build heap roundtrip Float tensor (shape 1x2x3) and convert. */
    size_t *roundTripDims = reserveMemory(3 * sizeof(size_t));
    roundTripDims[0] = 1;
    roundTripDims[1] = 2;
    roundTripDims[2] = 3;
    size_t *roundTripOrder = reserveMemory(3 * sizeof(size_t));
    setOrderOfDimsForNewTensor(3, roundTripOrder);
    shape_t *roundTripShape = reserveMemory(sizeof(shape_t));
    setShape(roundTripShape, roundTripDims, 3, roundTripOrder);
    tensor_t *roundTrip = initTensor(roundTripShape, quantizationInitFloat(), NULL);
    convertTensor(propLoss, roundTrip);

    /* 6. CAPTURE roundtrip values before frees. */
    float captured[6];
    for (size_t i = 0; i < n; i++) {
        captured[i] = ((float *)roundTrip->data)[i];
    }

    /* 7. FREE in reverse-init order. */
    freeTensor(roundTrip);
    freeFlattenLayer(flatten);
    freeTensor(propLoss);
    freeTensor(loss);
    freeTensor(forwardInput);

    /* 8. ASSERT. */
    TEST_ASSERT_EQUAL_FLOAT(capturedLossScale, capturedPropLossScale);
    for (size_t i = 0; i < n; i++) {
        TEST_ASSERT_FLOAT_WITHIN(0.1f, lossFloatData[i], captured[i]);
    }
}

void setUp(void) {}
void tearDown(void) {}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(testFlattenLayerInit_ReturnsFlattenTypedLayer);
    RUN_TEST(testFlattenCalcOutputShape_NonSquareInput);
    RUN_TEST(testFlattenForwardFloat_PreservesBytesAndReshapes);
    RUN_TEST(testFlattenForwardSymInt32_PropagatesScaleAndValues);
    RUN_TEST(testFlattenBackwardFloat_CopiesGradsUnchanged);
    RUN_TEST(testFlattenBackwardSymInt32_PropagatesScale);
    return UNITY_END();
}
