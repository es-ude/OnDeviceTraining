#include <stdlib.h>
#include <string.h>

#include "QuantizationApi.h"
#include "Softmax.h"
#include "SoftmaxApi.h"
#include "TensorApi.h"
#include "TensorConversion.h"
#include "unity.h"

void unitTestSoftmaxForwardFloat() {
    size_t inputSize = 6;

    float inputData[] = {-1.f, 0.f, 1.f, 2.f, 5.f, -6.f};
    size_t inputDims[] = {2, 3};
    size_t inputNumberOfDims = 2;
    tensor_t *input = tensorInitFloat(inputData, inputDims, inputNumberOfDims, NULL);

    float outputData[inputSize];
    size_t outputDims[] = {2, 3};
    size_t outputNumberOfDims = 2;
    tensor_t *output = tensorInitFloat(outputData, outputDims, outputNumberOfDims, NULL);

    quantization_t *floatQ = quantizationInitFloat();
    layer_t *softmaxLayer = softmaxLayerInit(floatQ, floatQ);
    layerFunctions_t softmaxFns = layerFunctions[SOFTMAX];
    softmaxFns.forward(softmaxLayer, input, output);

    float expected[] = {2.3008e-03f, 6.2543e-03f, 1.7001e-02f,
                        4.6213e-02f, 9.2822e-01f, 1.5503e-05f};

    float *actual = (float *)output->data;

    for (size_t i = 0; i < inputSize; i++) {
        TEST_ASSERT_FLOAT_WITHIN(0.0001f, expected[i], actual[i]);
    }
}

void unitTestSoftmaxForwardSymInt32() {
    size_t inputSize = 6;

    size_t dims[] = {2, 3};
    size_t numberOfDims = 2;

    float inputData[] = {-1.f, 0.f, 1.f, 2.f, 5.f, -6.f};

    tensor_t *input = tensorInitSymInt32(inputData, dims, numberOfDims, HTE, NULL);

    float outputData[inputSize];
    tensor_t *output = tensorInitSymInt32(outputData, dims, numberOfDims, HTE, NULL);

    quantization_t *symIntQ = quantizationInitSymInt32(HTE);
    layer_t *softmaxLayer = softmaxLayerInit(symIntQ, symIntQ);
    layerFunctions_t softmaxFns = layerFunctions[SOFTMAX];
    softmaxFns.forward(softmaxLayer, input, output);

    float outputFloatData[inputSize];
    tensor_t *outputFloat = tensorInitFloat(outputFloatData, dims, numberOfDims, NULL);
    convertTensor(output, outputFloat);

    float expected[] = {2.3008e-03f, 6.2543e-03f, 1.7001e-02f,
                        4.6213e-02f, 9.2822e-01f, 1.5503e-05f};
    float *actual = (float *)outputFloat->data;

    for (size_t i = 0; i < inputSize; i++) {
        TEST_ASSERT_FLOAT_WITHIN(0.1f, expected[i], actual[i]);
    }
}

void unitTestSoftmaxBackwardFloat() {
    size_t inputSize = 6;

    float inputData[] = {2.3008e-03, 6.2543e-03, 1.7001e-02, 4.6213e-02, 9.2822e-01, 1.5503e-05};
    size_t inputDims[] = {2, 3};
    size_t inputNumberOfDims = 2;
    tensor_t *input = tensorInitFloat(inputData, inputDims, inputNumberOfDims, NULL);

    float lossData[] = {0.f, 2.f, -4.f, 6.f, 3.f, 2.f};
    size_t lossDims[] = {2, 3};
    size_t lossNumberOfDims = 2;
    tensor_t *loss = tensorInitFloat(lossData, lossDims, lossNumberOfDims, NULL);

    float propLossData[inputSize];
    size_t propLossDims[] = {2, 3};
    size_t propLossNumberOfDims = 2;
    tensor_t *propLoss = tensorInitFloat(propLossData, propLossDims, propLossNumberOfDims, NULL);

    quantization_t *floatQ = quantizationInitFloat();
    layer_t *softmaxLayer = softmaxLayerInit(floatQ, floatQ);
    layerFunctions_t softmaxFns = layerFunctions[SOFTMAX];
    softmaxFns.backward(softmaxLayer, input, loss, propLoss);

    float expected[] = {-6.9173e-03f, -6.2947e-03f, -1.1912e-01f,
                        1.3834e-01f,  -5.9973e-03f, -1.5603e-05f};

    float *actual = (float *)propLoss->data;

    for (size_t i = 0; i < inputSize; i++) {
        TEST_ASSERT_FLOAT_WITHIN(0.0001f, expected[i], actual[i]);
    }
}

void unitTestSoftmaxBackwardSymInt32() {
    size_t inputSize = 6;

    float inputData[] = {2.3008e-03f, 6.2543e-03f, 1.7001e-02f,
                         4.6213e-02f, 9.2822e-01f, 1.5503e-05f};
    size_t inputDims[] = {2, 3};
    size_t inputNumberOfDims = 2;
    tensor_t *input = tensorInitSymInt32(inputData, inputDims, inputNumberOfDims, HTE, NULL);

    float lossData[] = {0.f, 2.f, -4.f, 6.f, 3.f, 2.f};
    size_t lossDims[] = {2, 3};
    size_t lossNumberOfDims = 2;
    tensor_t *loss = tensorInitSymInt32(lossData, lossDims, lossNumberOfDims, HTE, NULL);

    float propLossData[inputSize];
    memset(propLossData, 0, sizeof(propLossData));
    size_t propLossDims[] = {2, 3};
    size_t propLossNumberOfDims = 2;
    tensor_t *propLoss =
        tensorInitSymInt32(propLossData, propLossDims, propLossNumberOfDims, HTE, NULL);

    quantization_t *symIntQ = quantizationInitSymInt32(HTE);
    layer_t *softmaxLayer = softmaxLayerInit(symIntQ, symIntQ);
    layerFunctions_t softmaxFns = layerFunctions[SOFTMAX];
    softmaxFns.backward(softmaxLayer, input, loss, propLoss);

    float expected[] = {-6.9173e-03f, -6.2947e-03f, -1.1912e-01f,
                        1.3834e-01f,  -5.9973e-03f, -1.5603e-05f};

    float propLossDataFloat[inputSize];
    tensor_t *propLossFloat =
        tensorInitFloat(propLossDataFloat, propLossDims, propLossNumberOfDims, NULL);
    convertTensor(propLoss, propLossFloat);
    float *actual = (float *)propLossFloat->data;

    for (size_t i = 0; i < inputSize; i++) {
        TEST_ASSERT_FLOAT_WITHIN(0.01f, expected[i], actual[i]);
    }
}

void setUp() {}
void tearDown() {}

int main() {
    UNITY_BEGIN();
    RUN_TEST(unitTestSoftmaxForwardFloat);
    RUN_TEST(unitTestSoftmaxForwardSymInt32);

    RUN_TEST(unitTestSoftmaxBackwardFloat);
    RUN_TEST(unitTestSoftmaxBackwardSymInt32);
    return UNITY_END();
}
