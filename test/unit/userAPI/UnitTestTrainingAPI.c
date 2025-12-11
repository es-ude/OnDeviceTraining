#include <stddef.h>

#include "LossFunction.h"
#include "TensorAPI.h"
#include "LinearAPI.h"
#include "SgdAPI.h"
#include "unity.h"
#include "TrainingAPI.h"
#include "StorageAPI.h"
#include "TensorConversion.h"
#include "QuantizationAPI.h"

void testCalcGradsLinearFloat_WeightsMatchPyTorch() {
    float weightData[] = {1.f, 1.f, 1.f, 1.f, 1.f, 1.f};
    size_t weightDims[] = {2, 3};
    size_t weightNumberOfDims = 2;
    tensor_t *weightsParam = tensorInitFloat(weightData, weightDims, weightNumberOfDims, NULL);

    float weightGradData[] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    tensor_t *weightsGrad = tensorInitFloat(weightGradData, weightDims, weightNumberOfDims, NULL);

    parameter_t *weights = parameterInit(weightsParam, weightsGrad);

    float biasData[] = {-1.f, 3.f};
    size_t biasDims[] = {2, 1};
    size_t biasNumberOfDims = 2;
    tensor_t *biasParam = tensorInitFloat(biasData, biasDims, biasNumberOfDims, NULL);

    float biasGradData[] = {0.f, 0.f};
    tensor_t *biasGrad = tensorInitFloat(biasGradData, biasDims, biasNumberOfDims, NULL);

    parameter_t *bias = parameterInit(biasParam, biasGrad);

    float input0Data[] = {-4.f, 1.f, 9.f,};
    size_t input0Dims[] = {1, 3};
    size_t input0NumberOfDims = 2;
    tensor_t *input0 = tensorInitFloat(input0Data, input0Dims, input0NumberOfDims, NULL);

    float input1Data[] = {5.f, -1.f, 2.f};
    size_t input1Dims[] = {1, 3};
    size_t input1NumberOfDims = 2;
    tensor_t *input1 = tensorInitFloat(input1Data, input1Dims, input1NumberOfDims, NULL);

    float input2Data[] = {-7.f, -5.f, 6.f};
    size_t input2Dims[] = {1, 3};
    size_t input2NumberOfDims = 2;
    tensor_t *input2 = tensorInitFloat(input2Data, input2Dims, input2NumberOfDims, NULL);

    quantization_t outputQ;
    initFloat32Quantization(&outputQ);
    layer_t *linear = linearLayerInit(weights, bias, FLOAT_LAYER, input0->quantization->type,
                                      &outputQ);

    layer_t *model[] = {linear};
    size_t sizeModel = 1;

    float label0Data[] = {59.f, -23.f};
    size_t label0Dims[] = {2, 1};
    size_t label0NumberOfDims = 2;
    tensor_t *label0 = tensorInitFloat(label0Data, label0Dims, label0NumberOfDims, NULL);

    float label1Data[] = {43.f, 249.f};
    size_t label1Dims[] = {2, 1};
    size_t label1NumberOfDims = 2;
    tensor_t *label1 = tensorInitFloat(label1Data, label1Dims, label1NumberOfDims, NULL);

    float label2Data[] = {23.f, 457.f};
    size_t label2Dims[] = {2, 1};
    size_t label2NumberOfDims = 2;
    tensor_t *label2 = tensorInitFloat(label2Data, label2Dims, label2NumberOfDims, NULL);

    optimizer_t *sgd = sgdMCreateOptim(0.01f, 0.f, 0.f, model, sizeModel, FLOAT32);
    optimizerFunctions_t sgdFns = optimizerFunctions[SGD_M];

    // IMPORTANT: we want only the weights to be trainable, to check against pytorch learned weights!
    // That's why we set sizeStates to 1 (bias states are ignored).
    sgd->sizeStates = 1;

    for (size_t i = 0; i < 100; i++) {
        trainingStats_t *trainingStats0 = calculateGrads(model, sizeModel, MSE, input0, label0);
        trainingStats_t *trainingStats1 = calculateGrads(model, sizeModel, MSE, input1, label1);
        trainingStats_t *trainingStats2 = calculateGrads(model, sizeModel, MSE, input2, label2);

        sgdFns.step(sgd);
        sgdFns.zero(sgd);

        freeTrainingStats(trainingStats0);
        freeTrainingStats(trainingStats1);
        freeTrainingStats(trainingStats2);
    }

    float expectedWeights[] = {5.f, -1.f, 9.f, 22.f, -100.f, 18.f};
    linearConfig_t *linearConfig = linear->config->linear;
    float *actualWeights = (float *)linearConfig->weights->param->data;

    const float errorPercent = 0.03f;
    for (size_t i = 0; i < 6; i++) {
        float currentThreshold = actualWeights[i] * errorPercent;
        TEST_ASSERT_FLOAT_WITHIN(currentThreshold, expectedWeights[i], actualWeights[i]);
    }
}


void testCalcGradsLinearAsym_WeightsMatchPyTorch() {
    size_t numberOfWeights = 6;
    size_t numberOfInputs = 3;

    float weightDataFloat[] = {1.f, 1.f, 1.f, 1.f, 1.f, 1.f};
    size_t weightDims[] = {2, 3};
    size_t weightNumberOfDims = 2;
    tensor_t *weightsParamFloat = tensorInitFloat(weightDataFloat, weightDims, weightNumberOfDims,
                                                  NULL);

    float weightDataAsym[numberOfWeights];
    tensor_t *weightsParamAsym = tensorInitAsym(weightDataAsym, weightDims, weightNumberOfDims, 8,
                                                HTE, NULL);
    convertTensor(weightsParamFloat, weightsParamAsym);

    float weightGradAsymData[] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    tensor_t *weightsGradAsym = tensorInitAsym(weightGradAsymData, weightDims, weightNumberOfDims,
                                               8, HTE, NULL);

    parameter_t *weights = parameterInit(weightsParamAsym, weightsGradAsym);


    // IMPORTANT: WHEN ASYM --> BIASQ = INT32
    int32_t biasData[] = {-1, 3};
    size_t biasDims[] = {2, 1};
    size_t biasNumberOfDims = 2;
    tensor_t *biasParam = tensorInitInt32(biasData, biasDims, biasNumberOfDims, NULL);

    tensor_t *biasGradAsym = gradInitAsym(biasParam, 8, HTE, NULL);

    parameter_t *bias = parameterInit(biasParam, biasGradAsym);


    float input0Data[] = {-4.f, 1.f, 9.f,};
    size_t input0Dims[] = {1, 3};
    size_t input0NumberOfDims = 2;
    tensor_t *input0Float = tensorInitFloat(input0Data, input0Dims, input0NumberOfDims, NULL);

    float input0DataAsym[numberOfInputs];
    tensor_t *input0Asym = tensorInitAsym(input0DataAsym, input0Dims, input0NumberOfDims, 8, HTE,
                                          NULL);
    convertTensor(input0Float, input0Asym);

    float input1Data[] = {5.f, -1.f, 2.f};
    size_t input1Dims[] = {1, 3};
    size_t input1NumberOfDims = 2;
    tensor_t *input1Float = tensorInitFloat(input1Data, input1Dims, input1NumberOfDims, NULL);

    float input1DataAsym[numberOfInputs];
    tensor_t *input1Asym = tensorInitAsym(input1DataAsym, input1Dims, input1NumberOfDims, 8, HTE,
                                          NULL);
    convertTensor(input1Float, input1Asym);

    float input2Data[] = {-7.f, -5.f, 6.f};
    size_t input2Dims[] = {1, 3};
    size_t input2NumberOfDims = 2;
    tensor_t *input2Float = tensorInitFloat(input2Data, input2Dims, input2NumberOfDims, NULL);

    float input2DataAsym[numberOfInputs];
    tensor_t *input2Asym = tensorInitAsym(input2DataAsym, input2Dims, input2NumberOfDims, 8, HTE,
                                          NULL);
    convertTensor(input2Float, input2Asym);

    quantization_t *outputQ = quantizationInitAsym(8, HTE);
    layer_t *linear = linearLayerInit(weights, bias, ASYM_LAYER, ASYM, outputQ);

    layer_t *model[] = {linear};
    size_t modelSize = 1;


    float label0Data[] = {59.f, -23.f};
    size_t label0Dims[] = {2, 1};
    size_t label0NumberOfDims = 2;
    tensor_t *label0Float = tensorInitFloat(label0Data, label0Dims, label0NumberOfDims, NULL);

    float label0DataAsym[numberOfInputs];
    tensor_t *label0Asym = tensorInitAsym(label0DataAsym, label0Dims, label0NumberOfDims, 8, HTE, NULL);
    convertTensor(label0Float, label0Asym);


    float label1Data[] = {43.f, 249.f};
    size_t label1Dims[] = {2, 1};
    size_t label1NumberOfDims = 2;
    tensor_t *label1Float = tensorInitFloat(label1Data, label1Dims, label1NumberOfDims, NULL);

    float label1DataAsym[numberOfInputs];
    tensor_t *label1Asym = tensorInitAsym(label1DataAsym, label1Dims, label1NumberOfDims, 8, HTE, NULL);
    convertTensor(label1Float, label1Asym);


    float label2Data[] = {23.f, 457.f};
    size_t label2Dims[] = {2, 1};
    size_t label2NumberOfDims = 2;
    tensor_t *label2Float = tensorInitFloat(label2Data, label2Dims, label2NumberOfDims, NULL);

    float label2DataAsym[numberOfInputs];
    tensor_t *label2Asym = tensorInitAsym(label2DataAsym, label2Dims, label2NumberOfDims, 8, HTE, NULL);
    convertTensor(label2Float, label2Asym);

    optimizer_t *sgd = sgdMCreateOptim(0.01f, 0.f, 0.f, model, modelSize, ASYM);
    optimizerFunctions_t sgdFns = optimizerFunctions[SGD_M];

    // IMPORTANT: we want only the weights to be trainable, to check against pytorch learned weights!
    // That's why we set sizeStates to 1 (bias states are ignored).
    sgd->sizeStates = 1;

    for (size_t i = 0; i < 100; i++) {
        trainingStats_t *trainingStats0 = calculateGrads(model, modelSize, MSE, input0Asym, label0Asym);
        trainingStats_t *trainingStats1 = calculateGrads(model, modelSize, MSE, input1Asym, label1Asym);
        trainingStats_t *trainingStats2 = calculateGrads(model, modelSize, MSE, input2Asym, label2Asym);

        sgdFns.step(sgd);
        sgdFns.zero(sgd);

        freeTrainingStats(trainingStats0);
        freeTrainingStats(trainingStats1);
        freeTrainingStats(trainingStats2);
    }


    float expectedWeights[] = {5.f, -1.f, 9.f, 22.f, -100.f, 18.f};

    tensor_t actualWeights;
    float data[numberOfWeights];
    quantization_t q;
    initFloat32Quantization(&q);
    setTensorValuesForConversion((uint8_t *)data, &q, linear->config->linear->weights->param,
                                 &actualWeights);
    convertTensor(linear->config->linear->weights->param, &actualWeights);

    float *actualWeightsArr = (float *)actualWeights.data;
    const float errorPercent = 0.06f;
    for (size_t i = 0; i < 6; i++) {
        float currentThreshold = actualWeightsArr[i] * errorPercent;
        TEST_ASSERT_FLOAT_WITHIN(currentThreshold, expectedWeights[i], actualWeightsArr[i]);
    }
}

void setUp() {}
void tearDown() {}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(testCalcGradsLinearFloat_WeightsMatchPyTorch);
    RUN_TEST(testCalcGradsLinearAsym_WeightsMatchPyTorch);
    UNITY_END();
}
