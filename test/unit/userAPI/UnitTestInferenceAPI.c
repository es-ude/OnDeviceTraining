#include "Layer.h"
#include "TensorAPI.h"
#include "InferenceAPI.h"
#include "unity.h"
#include "LinearAPI.h"
#include "ReluAPI.h"
#include "QuantizationAPI.h"
#include "TensorConversion.h"
#include "LossFunction.h"

void testInferenceLinearReluFloat() {

    float weightData[] = {-1.f, 2.f, -3.f, 4.f, 5.f, 6.f};
    size_t weightDims[] = {2, 3};
    size_t weightNumberOfDims = 2;

    tensor_t *weightParam = tensorInitFloat(weightData, weightDims, weightNumberOfDims, NULL);

    tensor_t *weightGrad = gradInitFloat(weightParam, NULL);
    parameter_t *weights = parameterInit(weightParam, weightGrad);

    float biasData[] = {-1.f, 3.f};
    size_t biasDims[] = {2, 1};
    size_t biasNumberOfDims = 2;

    tensor_t *biasParam = tensorInitFloat(biasData, biasDims, biasNumberOfDims, NULL);
    tensor_t *biasGrad = gradInitFloat(biasParam, NULL);
    parameter_t *bias = parameterInit(biasParam, biasGrad);

    float inputData[] = {0.f, 1.f, 2.f};
    size_t inputDims[] = {1, 3};
    size_t inputNumberOfDims = 2;
    tensor_t *input = tensorInitFloat(inputData, inputDims, inputNumberOfDims, NULL);

    quantization_t *test = quantizationInitFloat();

    layer_t *linear = linearLayerInit(weights, bias, test, test, test, test);

    layer_t *relu = reluLayerInit(test, test);

    layer_t *model[] = {linear, relu};

    tensor_t *output = inference(model, 2, input);

    float expected[] = {0.f, 20.f};

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(expected, output->data, 2);
}

void testInferenceLinearReluSymInt32() {
    size_t numberOfOutputs = 2;

    float weightDataFloat[] = {-1.f, 2.f, -3.f, 4.f, 5.f, 6.f};
    size_t weightDims[] = {2, 3};
    size_t weightNumberOfDims = 2;
    tensor_t *weightsParam = tensorInitSymInt32(weightDataFloat, weightDims, weightNumberOfDims,
                                                HTE, NULL);
    tensor_t *weightGrad = gradInitSymInt32(weightsParam, HTE, NULL);
    parameter_t *weights = parameterInit(weightsParam, weightGrad);

    float biasData[] = {-1, 3};
    size_t biasDims[] = {2, 1};
    size_t biasNumberOfDims = 2;
    tensor_t *biasParam = tensorInitSymInt32(biasData, biasDims, biasNumberOfDims, HTE, NULL);
    tensor_t *biasGrad = gradInitSymInt32(biasParam, HTE, NULL);
    parameter_t *bias = parameterInit(biasParam, biasGrad);

    float inputDataFloat[] = {0.f, 1.f, 2.f};
    size_t inputDims[] = {1, 3};
    size_t inputNumberOfDims = 2;
    tensor_t *input = tensorInitSymInt32(inputDataFloat, inputDims, inputNumberOfDims, HTE, NULL);

    quantization_t *test = quantizationInitSymInt32(HTE);
    layer_t *linear = linearLayerInit(weights, bias, test, test, test, test);

    layer_t *relu = reluLayerInit(test, test);

    layer_t *model[] = {linear, relu};

    tensor_t *outputSymInt32 = inference(model, 2, input);

    float expected[] = {0.f, 20.f};
    float outputData[numberOfOutputs];
    tensor_t *outputFloat = tensorInitFloat(outputData, biasDims, biasNumberOfDims, NULL);
    convertTensor(outputSymInt32, outputFloat);
    float *actual = (float *)outputFloat->data;

    for (size_t i = 0; i < numberOfOutputs; i++) {
        TEST_ASSERT_FLOAT_WITHIN(0.1f, expected[i], actual[i]);
    }
}

void testInferenceWithLossLinearReluFloat() {

    float weightData[] = {-1.f, 2.f, -3.f, 4.f, 5.f, 6.f};
    size_t weightDims[] = {2, 3};
    size_t weightNumberOfDims = 2;

    tensor_t *weightParam = tensorInitFloat(weightData, weightDims, weightNumberOfDims, NULL);
    tensor_t *weightGrad = gradInitFloat(weightParam, NULL);
    parameter_t *weights = parameterInit(weightParam, weightGrad);

    float biasData[] = {-1.f, 3.f};
    size_t biasDims[] = {2, 1};
    size_t biasNumberOfDims = 2;

    tensor_t *biasParam = tensorInitFloat(biasData, biasDims, biasNumberOfDims, NULL);
    tensor_t *biasGrad = gradInitFloat(biasParam, NULL);
    parameter_t *bias = parameterInit(biasParam, biasGrad);

    float inputData[] = {0.f, 1.f, 2.f};
    size_t inputDims[] = {1, 3};
    size_t inputNumberOfDims = 2;
    tensor_t *input = tensorInitFloat(inputData, inputDims, inputNumberOfDims, NULL);

    quantization_t *test = quantizationInitFloat();

    layer_t *linear = linearLayerInit(weights, bias, test, test, test, test);

    layer_t *relu = reluLayerInit(test, test);

    layer_t *model[] = {linear, relu};

    float label0Data[] = {59.f, -23.f};
    size_t label0Dims[] = {2, 1};
    size_t label0NumberOfDims = 2;
    tensor_t *label0 = tensorInitFloat(label0Data, label0Dims, label0NumberOfDims, NULL);

    inferenceStats_t *inferenceStats = inferenceWithLoss(model, 2, input, label0, MSE);

    float expectedOutput[] = {0.f, 20.f};
    float expectedLoss = 2665.f;

    TEST_ASSERT_EQUAL_FLOAT(expectedLoss, inferenceStats->loss);
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(expectedOutput, inferenceStats->output->data, 2);

    freeInferenceStats(inferenceStats);
}

void setUp() {}
void tearDown() {}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(testInferenceLinearReluFloat);
    RUN_TEST(testInferenceLinearReluSymInt32);

    RUN_TEST(testInferenceWithLossLinearReluFloat);
    return UNITY_END();
}
