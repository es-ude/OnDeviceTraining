#include "Tensor.h"
#include "Rounding.h"
#include "MSE.h"
#include "unity.h"
#include "TensorConversion.h"
#include "TensorAPI.h"

void testMSEForward() {
    float outputData[] = {1.f, 2.f, 3.f};
    size_t outputDims[] = {3};
    size_t outputNumberOfDims = 1;
    tensor_t *output = tensorInitFloat(outputData, outputDims, outputNumberOfDims, NULL);

    float labelData[] = {2.f, 4.f, 6.f};
    size_t labelDims[] = {3};
    size_t labelNumberOfDims = 1;
    tensor_t *label = tensorInitFloat(labelData, labelDims, labelNumberOfDims, NULL);

    float loss = mseLossForward(output, label);

    float expected = 4.67f;

    TEST_ASSERT_FLOAT_WITHIN(0.01f, expected, loss);
}

void testMSELossBackwardFloat() {

    size_t numberOfElements = 3;
    size_t dims[] = {numberOfElements};
    size_t numberOfDims = 1;
    size_t orderOfDims[] = {0};
    shape_t shape = {
        .dimensions = dims,
        .orderOfDimensions = orderOfDims,
        .numberOfDimensions = numberOfDims
    };

    tensor_t modelOutput;
    quantization_t modelOutputQ;
    initFloat32Quantization(&modelOutputQ);
    float modelOutputData[] = {1.f, 2.f, -3.f};
    setTensorValues(&modelOutput, (uint8_t *)modelOutputData, &shape, &modelOutputQ, NULL);

    tensor_t label;
    quantization_t labelQ;
    initFloat32Quantization(&labelQ);
    float labelData[] = {-5.f, -4.f, 2.f};
    setTensorValues(&label, (uint8_t *)labelData, &shape, &labelQ, NULL);

    tensor_t result;
    quantization_t resultQ;
    initFloat32Quantization(&resultQ);
    float resultData[numberOfElements];
    setTensorValues(&result, (uint8_t *)resultData, &shape, &resultQ, NULL);

    mseLossBackwardFloat(&modelOutput, &label, &result);

    float expected[] = {4.f, 4.f, -3.3333f};

    float *actual = (float *)result.data;

    for(size_t i = 0; i < 3; i++) {
        TEST_ASSERT_FLOAT_WITHIN(0.0001f, expected[i], actual[i]);
    }
}

void testMSELossBackwardSymInt32() {
    size_t numberOfElements = 3;

    size_t dims[] = {numberOfElements};
    size_t numberOfDims = 1;
    size_t orderOfDims[] = {0};
    shape_t shape = {
        .dimensions = dims,
        .orderOfDimensions = orderOfDims,
        .numberOfDimensions = numberOfDims
    };

    tensor_t modelOutput;
    quantization_t modelOutputQ;
    initFloat32Quantization(&modelOutputQ);
    float modelOutputData[] = {1.f, 2.f, -3.f};
    setTensorValues(&modelOutput, (uint8_t *)modelOutputData, &shape, &modelOutputQ, NULL);

    tensor_t modelOutputSymInt32;
    symInt32QConfig_t modelOutputSymInt32QC;
    initSymInt32QConfig(HTE, &modelOutputSymInt32QC);
    quantization_t modelOutputSymInt32Q;
    initSymInt32Quantization(&modelOutputSymInt32QC, &modelOutputSymInt32Q);
    uint8_t modelOutputSymInt32Data[numberOfElements];
    setTensorValuesForConversion(modelOutputSymInt32Data, &modelOutputSymInt32Q, &modelOutput, &modelOutputSymInt32);
    convertTensor(&modelOutput, &modelOutputSymInt32);

    tensor_t label;
    quantization_t labelQ;
    initFloat32Quantization(&labelQ);
    float labelData[] = {-5.f, -4.f, 2.f};
    setTensorValues(&label, (uint8_t *)labelData, &shape, &labelQ, NULL);

    tensor_t labelSymInt32;
    symInt32QConfig_t labelSymInt32QC;
    initSymInt32QConfig(HTE, &labelSymInt32QC);
    quantization_t labelSymInt32Q;
    initSymInt32Quantization(&labelSymInt32QC, &labelSymInt32Q);
    uint8_t labelSymInt32Data[numberOfElements];
    setTensorValuesForConversion(labelSymInt32Data, &labelSymInt32Q, &label, &labelSymInt32);
    convertTensor(&label, &labelSymInt32);


    tensor_t result;
    quantization_t resultQ;
    initFloat32Quantization(&resultQ);
    float resultData[numberOfElements];
    setTensorValues(&result, (uint8_t *)resultData, &shape, &resultQ, NULL);

    tensor_t resultSymInt32;
    symInt32QConfig_t resultSymInt32QC;
    initSymInt32QConfig(HTE, &resultSymInt32QC);
    quantization_t resultSymInt32Q;
    initSymInt32Quantization(&resultSymInt32QC, &resultSymInt32Q);
    uint8_t resultSymInt32Data[numberOfElements];
    setTensorValuesForConversion(resultSymInt32Data, &resultSymInt32Q, &result, &resultSymInt32);
    convertTensor(&result, &resultSymInt32);


    mseLossBackward(&modelOutputSymInt32, &labelSymInt32, &resultSymInt32);

    convertTensor(&resultSymInt32, &result);

    float expected[] = {4.f, 4.f, -3.f};

    float *actual = (float *)result.data;

    for(size_t i = 0; i < numberOfElements; i++) {
        TEST_ASSERT_FLOAT_WITHIN(0.1f, expected[i], actual[i]);
    }
}

void setUp(){}
void tearDown(){}

int main(void) {
    UNITY_BEGIN();

    RUN_TEST(testMSEForward);

    RUN_TEST(testMSELossBackwardFloat);
    RUN_TEST(testMSELossBackwardSymInt32);

    return UNITY_END();
}