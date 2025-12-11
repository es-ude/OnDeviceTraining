#include <stddef.h>

#include "TensorAPI.h"
#include "LinearAPI.h"
#include "SgdAPI.h"
#include "unity.h"
#include "TrainingAPI.h"
#include "StorageAPI.h"

void testCalcGradsLinearFloat() {
    float weightData[] = {1.f, 1.f, 1.f, 1.f, 1.f, 1.f};
    size_t weightDims[] = {2, 3};
    size_t weightNumberOfDims = 2;
    quantization_t *weightParamQ = initFloat32Quantization();
    tensor_t *weightsParam = tensorInitFloat(weightData, weightDims, weightNumberOfDims);

    float weightGradData[] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    quantization_t *weightGradQ = initFloat32Quantization();
    tensor_t *weightsGrad = tensorInitFloat(weightGradData, weightDims, weightNumberOfDims);

    parameter_t *weights = parameterInit(weightsParam, weightsGrad);

    float biasData[] = {-1.f, 3.f};
    size_t biasDims[] = {2, 1};
    size_t biasNumberOfDims = 2;
    tensor_t *biasParam = tensorInitFloat(biasData, biasDims, biasNumberOfDims);

    float biasGradData[] = {0.f, 0.f};
    tensor_t *biasGrad = tensorInitFloat(biasGradData, biasDims, biasNumberOfDims);

    parameter_t *bias = parameterInit(biasParam, biasGrad);

    float input0Data[] = {-4.f, 1.f, 9.f,};
    size_t input0Dims[] = {1, 3};
    size_t input0NumberOfDims = 2;
    tensor_t *input0 = tensorInitFloat(input0Data, input0Dims, input0NumberOfDims);

    float input1Data[] = {5.f, -1.f, 2.f};
    size_t input1Dims[] = {1, 3};
    size_t input1NumberOfDims = 2;
    tensor_t *input1 = tensorInitFloat(input1Data, input1Dims, input1NumberOfDims);

    float input2Data[] = {-7.f, -5.f, 6.f};
    size_t input2Dims[] = {1, 3};
    size_t input2NumberOfDims = 2;
    tensor_t *input2 = tensorInitFloat(input2Data, input2Dims, input2NumberOfDims);

    // TODO check API for outputQ
    quantization_t outputQ;
    initFloat32Quantization(&outputQ);
    layer_t *linear = linearLayerInit(weights, bias, FLOAT_LAYER, input0->quantization->type, outputQ);

    layer_t *model[] = {linear};
    size_t sizeModel = 1;

    float label0Data[] = {59.f, -23.f};
    size_t label0Dims[] = {2, 1};
    size_t label0NumberOfDims = 2;
    tensor_t *label0 = tensorInitFloat(label0Data, label0Dims, label0NumberOfDims);

    float label1Data[] = {43.f, 249.f};
    size_t label1Dims[] = {2, 1};
    size_t label1NumberOfDims = 2;
    tensor_t *label1 = tensorInitFloat(label1Data, label1Dims, label1NumberOfDims);

    float label2Data[] = {23.f, 457.f};
    size_t label2Dims[] = {2, 1};
    size_t label2NumberOfDims = 2;
    tensor_t *label2 = tensorInitFloat(label2Data, label2Dims, label2NumberOfDims);

    // TODO check if trainable bias is a problem for the test result
    sgd_t *sgd = sgdInit(model, sizeModel, 0.01f, 0.f, 0.1f);

    for (size_t i = 0; i < 1000; i++) {
        trainingStats_t *trainingStats0 = calculateGrads(model, sizeModel, MSE, input0, label0);
        trainingStats_t *trainingStats1 = calculateGrads(model, sizeModel, MSE, input1, label1);
        trainingStats_t *trainingStats2 = calculateGrads(model, sizeModel, MSE, input2, label2);

        SGDStepFloat(sgd);
        SGDZeroGrad(sgd);

        freeTrainingStats(trainingStats0);
        freeTrainingStats(trainingStats1);
        freeTrainingStats(trainingStats2);
    }

    float expectedWeights[] = {5.f, -1.f, 9.f, 22.f, -100.f, 18.f};
    linearConfig_t *linearConfig = linear->config->linear;
    float *actualWeights = (float *)linearConfig->weights->param->data;
    for (size_t i = 0; i < 6; i++) {
        TEST_ASSERT_FLOAT_WITHIN(0.01f, expectedWeights[i], actualWeights[i]);
    }
}

void setUp(){}
void tearDown(){}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(testCalcGradsLinearFloat);
    UNITY_END();
}