#include "Layer.h"
#include "TensorAPI.h"
#include "InferenceAPI.h"
#include "unity.h"
#include "LinearAPI.h"
#include "ReluAPI.h"

#include <QuantizationAPI.h>

void testInferenceLinearReluFloat() {

    float weightData[] = {-1.f, 2.f, -3.f, 4.f, 5.f, 6.f};
    size_t weightDims[] = {2, 3};
    size_t weightNumberOfDims = 2;

    tensor_t *weightParam = tensorInitFloat(weightData, weightDims, weightNumberOfDims, false);

    tensor_t *weightGrad = gradInitFloat(weightParam);
    parameter_t *weights = parameterInit(weightParam, weightGrad);


    float biasData[] = {-1.f, 3.f};
    size_t biasDims[] = {2, 1};
    size_t biasNumberOfDims = 2;

    tensor_t *biasParam = tensorInitFloat(biasData, biasDims, biasNumberOfDims, false);
    tensor_t *biasGrad = gradInitFloat(biasParam);
    parameter_t *bias = parameterInit(biasParam, biasGrad);


    float inputData[] = {0.f, 1.f, 2.f};
    size_t inputDims[] = {1, 3};
    size_t inputNumberOfDims = 2;
    tensor_t *input = tensorInitFloat(inputData, inputDims, inputNumberOfDims, false);


    quantization_t *outputQ =quantizationInitFloat();

    layer_t *linear = linearLayerInit(weights, bias, FLOAT_LAYER, input->quantization->type, outputQ);

    layer_t *relu = reluLayerInit(FLOAT_LAYER, linear->outputQ->type, outputQ);

    layer_t *model[] = {linear, relu};

    tensor_t *output = inference(model, 2, input);

    float expected[] = {0.f, 20.f};

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(expected, output->data, 2);
}

void setUp() {}
void tearDown() {}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(testInferenceLinearReluFloat);
    UNITY_END();
}