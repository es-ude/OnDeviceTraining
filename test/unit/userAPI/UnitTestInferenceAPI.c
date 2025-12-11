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

    quantization_t *outputQ = quantizationInitFloat();

    layer_t *linear = linearLayerInit(weights, bias, FLOAT_LAYER, input->quantization->type,
                                      outputQ);

    layer_t *relu = reluLayerInit(FLOAT_LAYER, linear->outputQ->type, outputQ);

    layer_t *model[] = {linear, relu};

    tensor_t *output = inference(model, 2, input);

    float expected[] = {0.f, 20.f};

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(expected, output->data, 2);
}

void testInferenceLinearReluAsym() {
    size_t numberOfWeights = 6;
    size_t numberOfBiases = 2;
    size_t numberOfInputs = 3;
    size_t numberOfOutputs = 2;

    float weightDataFloat[] = {-1.f, 2.f, -3.f, 4.f, 5.f, 6.f};
    size_t weightDims[] = {2, 3};
    size_t weightNumberOfDims = 2;
    tensor_t *weightsParamFloat = tensorInitFloat(weightDataFloat, weightDims, weightNumberOfDims,
                                                  NULL);

    float weightDataAsym[numberOfWeights];
    quantization_t *weightAsymQ = quantizationInitAsym(8, HTE);
    tensor_t *weightsParamAsym = tensorInit(weightDataAsym, weightDims, weightNumberOfDims,
                                            weightAsymQ, NULL);

    convertTensor(weightsParamFloat, weightsParamAsym);

    tensor_t *weightGradAsym = gradInitAsym(weightsParamAsym, 8, HTE, NULL);
    parameter_t *weights = parameterInit(weightsParamAsym, weightGradAsym);

    // IMPORTANT: WHEN ASYM --> BIASQ = INT32!
    float biasData[] = {-1, 3};
    size_t biasDims[] = {2, 1};
    size_t biasNumberOfDims = 2;
    quantization_t *biasQ = quantizationInitInt32();
    tensor_t *biasParamInt = tensorInit(biasData, biasDims, biasNumberOfDims, biasQ, NULL);

    tensor_t *biasGradAsym = gradInitAsym(biasParamInt, 8, HTE, NULL);
    parameter_t *bias = parameterInit(biasParamInt, biasGradAsym);

    float inputDataFloat[] = {0.f, 1.f, 2.f};
    size_t inputDims[] = {1, 3};
    size_t inputNumberOfDims = 2;
    tensor_t *inputFloat = tensorInitFloat(inputDataFloat, inputDims, inputNumberOfDims, NULL);

    float inputDataAsym[numberOfInputs];
    quantization_t *inputAsymQ = quantizationInitAsym(8, HTE);
    tensor_t *inputAsym = tensorInit(inputDataAsym, inputDims, inputNumberOfDims, inputAsymQ, NULL);
    convertTensor(inputFloat, inputAsym);

    quantization_t *outputQ = quantizationInitAsym(8, HTE);
    layer_t *linear = linearLayerInit(weights, bias, ASYM_LAYER, ASYM, outputQ);

    layer_t *relu = reluLayerInit(ASYM_LAYER, ASYM, outputQ);

    layer_t *model[] = {linear, relu};

    tensor_t *outputAsym = inference(model, 2, inputAsym);

    float expected[] = {0.f, 20.f - (float)biasData[1]};
    float outputData[numberOfOutputs];
    tensor_t *outputFloat = tensorInitFloat(outputData, biasDims, biasNumberOfDims, NULL);
    convertTensor(outputAsym, outputFloat);
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

    quantization_t *outputQ = quantizationInitFloat();

    layer_t *linear = linearLayerInit(weights, bias, FLOAT_LAYER, input->quantization->type,
                                      outputQ);

    layer_t *relu = reluLayerInit(FLOAT_LAYER, linear->outputQ->type, outputQ);

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
    RUN_TEST(testInferenceLinearReluAsym);

    RUN_TEST(testInferenceWithLossLinearReluFloat);
    UNITY_END();
}
