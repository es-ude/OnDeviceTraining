#include <stdlib.h>

#include "Conv1d.h"
#include "Conv1dApi.h"
#include "Layer.h"
#include "QuantizationApi.h"
#include "TensorApi.h"
#include "unity.h"

void testCalcOutputSizePerChannel() {
    kernel_t kernel = {.dilation = 2, .paddingType = VALID, .size = 2, .stride = 3};

    size_t inputLengthPerChannel = 9;

    size_t actual = calcOutputLengthPerChannel(inputLengthPerChannel, &kernel);
    size_t expected = 3;

    TEST_ASSERT_EQUAL_size_t(expected, actual);
}

void testConv1dForwardFloat() {
    float weightData[] = {2, 4};
    // [outChannels, inChannels, kernelSize]
    size_t weightDims[] = {2, 2, 2};
    size_t weightNumberOfDims = 3;
    tensor_t *weightParam = tensorInitFloat(weightData, weightDims, weightNumberOfDims, NULL);
    tensor_t *weightGrad = gradInitFloat(weightParam, NULL);
    parameter_t *weights = parameterInit(weightParam, weightGrad);

    float biasData[] = {1, 2};
    // [outChannels]
    size_t biasDims[] = {2};
    size_t biasNumberOfDims = 1;
    tensor_t *biasParam = tensorInitFloat(biasData, biasDims, biasNumberOfDims, NULL);
    tensor_t *biasGrad = gradInitFloat(biasParam, NULL);
    parameter_t *bias = parameterInit(biasParam, biasGrad);

    size_t size = 2, dilation = 1, stride = 1;

    kernel_t kernel;
    initKernel(&kernel, size, VALID, dilation, stride);

    quantization_t *q = quantizationInitFloat();
    layer_t *conv1d = conv1dLayerInit(weights, bias, &kernel, q, q, q, q);

    float inputData[] = {1, 2, 3, 4, 1, 2, 3, 4};
    // [batch, channel, length]
    size_t inputDims[] = {1, 2, 4};
    size_t inputNumberOfDims = 3;
    tensor_t *input = tensorInitFloat(inputData, inputDims, inputNumberOfDims, NULL);

    float outputData[6];
    size_t outputDims[] = {1, 2, 3};
    size_t outputNumberOfDims = 3;
    tensor_t *output = tensorInitFloat(outputData, outputDims, outputNumberOfDims, NULL);

    conv1dForwardFloat(conv1d, input, output);

    float expected[] = {11, 17, 23, 12, 18, 24};
    float *actual = (float *)output->data;

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(expected, actual, 6);
}

void testConv1dForwardFloatWithStride() {
    float weightData[] = {2, 4};
    // [outChannels, inChannels, kernelSize]
    size_t weightDims[] = {1, 1, 2};
    size_t weightNumberOfDims = 3;
    tensor_t *weightParam = tensorInitFloat(weightData, weightDims, weightNumberOfDims, NULL);
    tensor_t *weightGrad = gradInitFloat(weightParam, NULL);
    parameter_t *weights = parameterInit(weightParam, weightGrad);

    float biasData[] = {1};
    // [outChannels]
    size_t biasDims[] = {1};
    size_t biasNumberOfDims = 1;
    tensor_t *biasParam = tensorInitFloat(biasData, biasDims, biasNumberOfDims, NULL);
    tensor_t *biasGrad = gradInitFloat(biasParam, NULL);
    parameter_t *bias = parameterInit(biasParam, biasGrad);

    size_t size = 2, dilation = 1, stride = 2;

    kernel_t kernel;
    initKernel(&kernel, size, VALID, dilation, stride);

    quantization_t *q = quantizationInitFloat();

    layer_t *conv1d = conv1dLayerInit(weights, bias, &kernel, q, q, q, q);

    float inputData[] = {1, 2, 0, 0, 3, 4};
    // [batch, channel, length]
    size_t inputDims[] = {1, 1, 6};
    size_t inputNumberOfDims = 3;
    tensor_t *input = tensorInitFloat(inputData, inputDims, inputNumberOfDims, NULL);

    float outputData[3];
    size_t outputDims[] = {1, 1, 3};
    size_t outputNumberOfDims = 3;
    tensor_t *output = tensorInitFloat(outputData, outputDims, outputNumberOfDims, NULL);

    conv1dForwardFloat(conv1d, input, output);

    float expected[] = {11, 1, 23};
    float *actual = (float *)output->data;

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(expected, actual, 3);
}

void testConv1dForwardFloatWithStrideAndDilation() {
    float weightData[] = {2, 4};
    // [outChannels, inChannels, kernelSize]
    size_t weightDims[] = {1, 1, 2};
    size_t weightNumberOfDims = 3;
    tensor_t *weightParam = tensorInitFloat(weightData, weightDims, weightNumberOfDims, NULL);
    tensor_t *weightGrad = gradInitFloat(weightParam, NULL);
    parameter_t *weights = parameterInit(weightParam, weightGrad);

    float biasData[] = {1};
    // [outChannels]
    size_t biasDims[] = {1};
    size_t biasNumberOfDims = 1;
    tensor_t *biasParam = tensorInitFloat(biasData, biasDims, biasNumberOfDims, NULL);
    tensor_t *biasGrad = gradInitFloat(biasParam, NULL);
    parameter_t *bias = parameterInit(biasParam, biasGrad);

    size_t size = 2, dilation = 2, stride = 3;

    kernel_t kernel;
    initKernel(&kernel, size, VALID, dilation, stride);

    quantization_t *q = quantizationInitFloat();

    layer_t *conv1d = conv1dLayerInit(weights, bias, &kernel, q, q, q, q);

    float inputData[] = {1, 0, 2, 0, 0, 0, 3, 0, 4};
    // [batch, channel, length]
    size_t inputDims[] = {1, 1, 9};
    size_t inputNumberOfDims = 3;
    tensor_t *input = tensorInitFloat(inputData, inputDims, inputNumberOfDims, NULL);

    float outputData[3];
    size_t outputDims[] = {1, 1, 3};
    size_t outputNumberOfDims = 3;
    tensor_t *output = tensorInitFloat(outputData, outputDims, outputNumberOfDims, NULL);

    conv1dForwardFloat(conv1d, input, output);

    float expected[] = {11, 1, 23};
    float *actual = (float *)output->data;

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(expected, actual, 3);
}

void testConv1dForwardFloatWithStrideDilationAndPadding() {
    float weightData[] = {2, 4, 3};
    // [outChannels, inChannels, kernelSize]
    size_t weightDims[] = {1, 1, 2};
    size_t weightNumberOfDims = 3;
    tensor_t *weightParam = tensorInitFloat(weightData, weightDims, weightNumberOfDims, NULL);
    tensor_t *weightGrad = gradInitFloat(weightParam, NULL);
    parameter_t *weights = parameterInit(weightParam, weightGrad);

    size_t size = 3, dilation = 2, stride = 2;

    kernel_t kernel;
    initKernel(&kernel, size, SAME, dilation, stride);

    quantization_t *q = quantizationInitFloat();

    layer_t *conv1d = conv1dLayerInit(weights, NULL, &kernel, q, q, q, q);

    float inputData[] = {1, 2, 3, 4, 5, 6};
    // [batch, channel, length]
    size_t inputDims[] = {1, 1, 6};
    size_t inputNumberOfDims = 3;
    tensor_t *input = tensorInitFloat(inputData, inputDims, inputNumberOfDims, NULL);

    float outputData[6] = {0};
    tensor_t *output = tensorInitFloat(outputData, inputDims, inputNumberOfDims, NULL);

    conv1dForwardFloat(conv1d, input, output);

    float expected[] = {3, 13, 29, 26, 10, 0};
    float *actual = (float *)output->data;

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(expected, actual, 6);
}

void testConv1dBackwardFloat() {
    float weightData[] = {2, 4};
    // [outChannels, inChannels, kernelSize]
    size_t weightDims[] = {1, 1, 2};
    tensor_t *weightParam = tensorInitFloat(weightData, weightDims, 3, NULL);
    tensor_t *weightGrad = gradInitFloat(weightParam, NULL);
    parameter_t *weights = parameterInit(weightParam, weightGrad);

    float biasData[] = {1};
    // [outChannels]
    size_t biasDims[] = {1};
    tensor_t *biasParam = tensorInitFloat(biasData, biasDims, 1, NULL);
    tensor_t *biasGrad = gradInitFloat(biasParam, NULL);
    parameter_t *bias = parameterInit(biasParam, biasGrad);

    kernel_t kernel;
    initKernel(&kernel, 2, VALID, 1, 1);

    quantization_t *q = quantizationInitFloat();

    layer_t *conv1d = conv1dLayerInit(weights, bias, &kernel, q, q, q, q);

    float inputData[] = {1, 2, 3, 4};
    // [batch, channel, length]
    size_t inputDims[] = {1, 1, 4};
    tensor_t *input = tensorInitFloat(inputData, inputDims, 3, NULL);

    float outputData[3] = {0};
    size_t outputDims[] = {1, 1, 3};
    tensor_t *output = tensorInitFloat(outputData, outputDims, 3, NULL);

    conv1dForwardFloat(conv1d, input, output);

    float lossGradData[] = {1, 1, 1};
    tensor_t *lossGrad = tensorInitFloat(lossGradData, outputDims, 3, NULL);

    float propLossData[4] = {0};
    tensor_t *propLoss = tensorInitFloat(propLossData, inputDims, 3, NULL);

    conv1dBackwardFloat(input, propLoss, lossGrad, &kernel, weights, bias);

    float expectedOutput[] = {11, 17, 23};
    float expectedPropLoss[] = {2, 6, 6, 4};
    float expectedWeightGrad[] = {6, 9};
    float expectedBiasGrad[] = {3};

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(expectedOutput, output->data, 2);
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(expectedPropLoss, propLoss->data, 4);
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(expectedWeightGrad, weights->grad->data, 2);
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(expectedBiasGrad, bias->grad->data, 1);
}

void testConv1dBackwardFloatWithPadding() {
    float weightData[] = {2, 4, 3};
    // [outChannels, inChannels, kernelSize]
    size_t weightDims[] = {1, 1, 3};
    tensor_t *weightParam = tensorInitFloat(weightData, weightDims, 3, NULL);
    tensor_t *weightGrad = gradInitFloat(weightParam, NULL);
    parameter_t *weights = parameterInit(weightParam, weightGrad);

    float biasData[] = {1};
    // [outChannels]
    size_t biasDims[] = {1};
    tensor_t *biasParam = tensorInitFloat(biasData, biasDims, 1, NULL);
    tensor_t *biasGrad = gradInitFloat(biasParam, NULL);
    parameter_t *bias = parameterInit(biasParam, biasGrad);

    kernel_t kernel;
    initKernel(&kernel, 3, SAME, 2, 1);

    float inputData[] = {1, 2, 3, 4, 5, 6};
    // [batch, channel, length]
    size_t inputDims[] = {1, 1, 6};
    tensor_t *input = tensorInitFloat(inputData, inputDims, 3, NULL);

    float outputData[6];
    tensor_t *output = tensorInitFloat(outputData, inputDims, 3, NULL);

    quantization_t *q = quantizationInitFloat();
    layer_t *conv1d = conv1dLayerInit(weights, bias, &kernel, q, q, q, q);

    conv1dForwardFloat(conv1d, input, output);

    float lossGradData[] = {1, 1, 1, 1, 1, 1};
    tensor_t *lossGrad = tensorInitFloat(lossGradData, inputDims, 3, NULL);

    float propLossData[6] = {0};
    tensor_t *propLoss = tensorInitFloat(propLossData, inputDims, 3, NULL);

    conv1dBackwardFloat(input, propLoss, lossGrad, &kernel, weights, bias);

    // Values from pytorch (jupyter notebook)
    float expectedOutput[] = {14, 21, 30, 39, 27, 33};
    float expectedWeightGrad[] = {10, 21, 18};
    float expectedPropLoss[] = {6, 6, 9, 9, 7, 7};
    float expectedBiasGrad[] = {6};

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(expectedOutput, output->data, 2);
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(expectedWeightGrad, weights->grad->data, 2);
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(expectedPropLoss, propLoss->data, 6);
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(expectedBiasGrad, bias->grad->data, 1);
}

void setUp() {}
void tearDown() {}

int main() {
    UNITY_BEGIN();

    RUN_TEST(testCalcOutputSizePerChannel);

    RUN_TEST(testConv1dForwardFloat);
    RUN_TEST(testConv1dForwardFloatWithStride);
    RUN_TEST(testConv1dForwardFloatWithStrideAndDilation);
    RUN_TEST(testConv1dForwardFloatWithStrideDilationAndPadding);

    RUN_TEST(testConv1dBackwardFloat);
    RUN_TEST(testConv1dBackwardFloatWithPadding);

    return UNITY_END();
}
