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

void testMSELossBackwardAsym() {
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

    tensor_t modelOutputAsym;
    asymQConfig_t modelOutputAsymQC;
    initAsymQConfig(8, HTE, &modelOutputAsymQC);
    quantization_t modelOutputAsymQ;
    initAsymQuantization(&modelOutputAsymQC, &modelOutputAsymQ);
    uint8_t modelOutputAsymData[numberOfElements];
    setTensorValuesForConversion(modelOutputAsymData, &modelOutputAsymQ, &modelOutput, &modelOutputAsym);
    convertTensor(&modelOutput, &modelOutputAsym);

    tensor_t label;
    quantization_t labelQ;
    initFloat32Quantization(&labelQ);
    float labelData[] = {-5.f, -4.f, 2.f};
    setTensorValues(&label, (uint8_t *)labelData, &shape, &labelQ, NULL);

    tensor_t labelAsym;
    asymQConfig_t labelAsymQC;
    initAsymQConfig(8, HTE, &labelAsymQC);
    quantization_t labelAsymQ;
    initAsymQuantization(&labelAsymQC, &labelAsymQ);
    uint8_t labelAsymData[numberOfElements];
    setTensorValuesForConversion(labelAsymData, &labelAsymQ, &label, &labelAsym);
    convertTensor(&label, &labelAsym);


    tensor_t result;
    quantization_t resultQ;
    initFloat32Quantization(&resultQ);
    float resultData[numberOfElements];
    setTensorValues(&result, (uint8_t *)resultData, &shape, &resultQ, NULL);

    tensor_t resultAsym;
    asymQConfig_t resultAsymQC;
    initAsymQConfig(8, HTE, &resultAsymQC);
    quantization_t resultAsymQ;
    initAsymQuantization(&resultAsymQC, &resultAsymQ);
    uint8_t resultAsymData[numberOfElements];
    setTensorValuesForConversion(resultAsymData, &resultAsymQ, &result, &resultAsym);
    convertTensor(&result, &resultAsym);


    mseLossBackward(&modelOutputAsym, &labelAsym, &resultAsym);

    convertTensor(&resultAsym, &result);

    float expected[] = {4.f, 4.f, -4.f};

    float *actual = (float *)result.data;

    for(size_t i = 0; i < numberOfElements; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1.f, expected[i], actual[i]);
    }
}

void setUp(){}
void tearDown(){}

int main(void) {
    UNITY_BEGIN();

    RUN_TEST(testMSEForward);

    RUN_TEST(testMSELossBackwardFloat);
    RUN_TEST(testMSELossBackwardAsym);

    UNITY_END();
}