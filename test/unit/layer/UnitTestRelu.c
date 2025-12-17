#include "Relu.h"
#include "Quantization.h"
#include "unity.h"
#include "DTypes.h"
#include "Tensor.h"
#include "TensorConversion.h"
#include "ReluAPI.h"
#include "TensorAPI.h"
#include "QuantizationAPI.h"

void testReluForwardFloat() {
    size_t numberOfElements = 6;

    tensor_t input;
    float inputData[] = {-1.f, 0.f, 1.f, 2.f, 5.f, -6.f};
    size_t inputDims[] = {2, 3};
    size_t inputNumberOfDims = 2;
    size_t inputOrderOfDims[] = {0, 1};
    shape_t inputShape = {
        .dimensions = inputDims,
        .orderOfDimensions = inputOrderOfDims,
        .numberOfDimensions = inputNumberOfDims
    };
    quantization_t inputQ;
    initFloat32Quantization(&inputQ);
    setTensorValues(&input, (uint8_t *)inputData, &inputShape,
                    &inputQ, NULL);

    tensor_t output;
    float outputData[numberOfElements];
    size_t outputDims[] = {2, 3};
    size_t outputNumberOfDims = 2;
    size_t outputOrderOfDims[] = {0, 1};
    shape_t outputShape = {
        .dimensions = outputDims,
        .orderOfDimensions = outputOrderOfDims,
        .numberOfDimensions = outputNumberOfDims
    };
    quantization_t outputQ;
    initFloat32Quantization(&outputQ);
    setTensorValues(&output, (uint8_t *)outputData, &outputShape, &outputQ, NULL);

    quantization_t floatQ;
    initFloat32Quantization(&floatQ);
    layer_t *reluLayer = reluLayerInit(&floatQ, &floatQ);

    reluForward(reluLayer, &input, &output);

    float expected[] = {0.f, 0.f, 1.f, 2.f, 5.f, 0.f};

    float actual[numberOfElements];
    readBytesAsFloatArray(numberOfElements, output.data, actual);

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(expected, actual, numberOfElements);
}

void testReluForwardSymInt32() {
    float inputData[] = {-1.f, 0.f, 1.f, 2.f, 5.f, -6.f};
    size_t inputDims[] = {2, 3};
    size_t inputNumberOfDims = 2;
    tensor_t *input = tensorInitSymInt32(inputData, inputDims, inputNumberOfDims, HTE, NULL);

    float outputData[6];
    size_t outputDims[] = {2, 3};
    size_t outputNumberOfDims = 2;
    tensor_t *output = tensorInitSymInt32(outputData, outputDims, outputNumberOfDims, HTE, NULL);

    quantization_t *symIntQ = quantizationInitSymInt32(HTE);
    layer_t *reluLayer = reluLayerInit(symIntQ, symIntQ);
    layerFunctions_t reluFns = layerFunctions[RELU];
    reluFns.forward(reluLayer, input, output);

    int32_t expected[] = {0, 0, 1, 2, 5, 0};
    int32_t *actual = (int32_t *)output->data;

    TEST_ASSERT_EQUAL_INT32_ARRAY(expected, actual, 6);
}

void testReluBackwardFloat() {
    size_t numberOfElements = 6;

    size_t dims[] = {numberOfElements};
    size_t numberOfDims = 1;

    float forwardInputData[] = {-1.f, 0.f, 1.f, 2.f, 5.f, -6.f};
    tensor_t *forwardInput = tensorInitFloat(forwardInputData, dims, numberOfDims, NULL);

    float lossData[] = {0.f, 2.f, -4.f, 6.f, 3.f, 2.f};
    tensor_t *loss = tensorInitFloat(lossData, dims, numberOfDims, NULL);

    float propLossData[numberOfElements];
    tensor_t *propLoss = tensorInitFloat(propLossData, dims, numberOfDims, NULL);

    quantization_t *floatQ = quantizationInitFloat();
    layer_t *reluLayer = reluLayerInit(floatQ, floatQ);
    layerFunctions_t reluFns = layerFunctions[RELU];
    reluFns.backward(reluLayer, forwardInput, loss, propLoss);

    float expected[] = {0.f, 0.f, -4.f, 6.f, 3.f, 0.f};

    float *actual = (float *)propLoss->data;

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(expected, actual, numberOfElements);
}

void testReluBackwardSymInt32() {
    size_t numberOfElements = 6;

    size_t dims[] = {6};
    size_t numberOfDims = 1;

    float forwardInputData[] = {-1.f, 0.f, 1.f, 2.f, 5.f, -6.f};
    tensor_t *forwardInput = tensorInitSymInt32(forwardInputData, dims, numberOfDims, HTE, NULL);

    float lossData[] = {0.f, 2.f, -4.f, 6.f, 3.f, 2.f};
    tensor_t *loss = tensorInitSymInt32(lossData, dims, numberOfDims, HTE, NULL);

    float propLossData[numberOfElements];
    tensor_t *propLoss = tensorInitSymInt32(propLossData, dims, numberOfDims, HTE, NULL);

    quantization_t *symIntQ = quantizationInitSymInt32(HTE);
    layer_t *reluLayer = reluLayerInit(symIntQ, symIntQ);
    layerFunctions_t reluFns = layerFunctions[RELU];
    reluFns.backward(reluLayer, forwardInput, loss, propLoss);

    int32_t expected[] = {0, 0, -4, 6, 3, 0};
    int32_t *actual = (int32_t *)propLoss->data;

    TEST_ASSERT_EQUAL_INT32_ARRAY(expected, actual, numberOfElements);
}


void setUp() {}
void tearDown() {}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(testReluForwardFloat);
    RUN_TEST(testReluForwardSymInt32);

    RUN_TEST(testReluBackwardFloat);
    RUN_TEST(testReluBackwardSymInt32);
    return UNITY_END();
}
