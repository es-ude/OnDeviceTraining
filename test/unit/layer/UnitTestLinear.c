#include "DTypes.h"
#include "Layer.h"
#include "Linear.h"
#include "Rounding.h"
#include "Tensor.h"
#include "TensorConversion.h"
#include "unity.h"
#include "TensorAPI.h"
#include "QuantizationAPI.h"
#include "LinearAPI.h"

void setUp() {}
void tearDown() {}

void testLinearForwardFloat() {
    parameter_t weights;
    tensor_t weightsParam;
    float weightData[] = {-1.f, 2.f, -3.f, 4.f, 5.f, -6.f};
    size_t weightDims[] = {2, 3};
    size_t weightNumberOfDims = 2;
    size_t weightOrderOfDims[] = {0, 1};
    shape_t weightShape = {.dimensions = weightDims,
                           .orderOfDimensions = weightOrderOfDims,
                           .numberOfDimensions = weightNumberOfDims};
    quantization_t weightQ;
    initFloat32Quantization(&weightQ);
    setTensorValues(&weightsParam, (uint8_t *)weightData, &weightShape, &weightQ, NULL);

    setParameterValues(&weights, &weightsParam, NULL);

    parameter_t bias;
    tensor_t biasParam;
    float biasData[] = {-1.f, 3.f};
    size_t biasDims[] = {2};
    size_t biasNumberOfDims = 1;
    size_t biasOrderOfDims[] = {0};
    shape_t biasShape = {.dimensions = biasDims,
                         .orderOfDimensions = biasOrderOfDims,
                         .numberOfDimensions = biasNumberOfDims};
    quantization_t biasQ;
    initFloat32Quantization(&biasQ);
    setTensorValues(&biasParam, (uint8_t *)biasData, &biasShape, &biasQ, NULL);
    setParameterValues(&bias, &biasParam, NULL);

    tensor_t input;
    float inputData[] = {0.f, 1.f, 2.f};
    size_t inputDims[] = {1, 3};
    size_t inputNumberOfDims = 2;
    size_t inputOrderOfDims[] = {0, 1};
    shape_t inputShape = {.dimensions = inputDims,
                          .orderOfDimensions = inputOrderOfDims,
                          .numberOfDimensions = inputNumberOfDims};
    quantization_t inputQ;
    initFloat32Quantization(&inputQ);
    setTensorValues(&input, (uint8_t *)inputData, &inputShape, &inputQ, NULL);

    float outputData[2] = {0, 0};
    size_t outputDims[] = {2};
    size_t outputNumberOfDims = 1;
    size_t outputOrderOfDims[] = {0};
    shape_t outputShape = {.dimensions = outputDims,
                           .orderOfDimensions = outputOrderOfDims,
                           .numberOfDimensions = outputNumberOfDims};
    quantization_t outputQ;
    initFloat32Quantization(&outputQ);
    tensor_t output;
    setTensorValues(&output, (uint8_t *)outputData, &outputShape, &outputQ, NULL);

    layer_t linearLayer;
    layerConfig_t linearConfig;
    linearConfig_t linCfg;
    linearConfig.linear = &linCfg;
    quantization_t testQ;
    initFloat32Quantization(&testQ);
    linearInitConfig(linearConfig.linear, &weights, &bias, &testQ, &testQ, &testQ, &testQ);

    initLayer(&linearLayer, LINEAR, &linearConfig);

    linearForward(&linearLayer, &input, &output);

    float expected[] = {-5.f, -4.f};

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(expected, output.data, 2);
}

void testLinearForwardSymInt32() {
    size_t numberOfInputs = 3;
    size_t numberOfOutputs = 2;

    float weightData[] = {-1.f, 2.f, -3.f, 4.f, 5.f, -6.f};
    size_t weightDims[] = {2, 3};
    size_t weightNumberOfDims = 2;
    tensor_t *weightsParam = tensorInitSymInt32(weightData, weightDims, weightNumberOfDims, HTE, NULL);
    tensor_t *weightsGrad = gradInitSymInt32(weightsParam, HTE, NULL);
    parameter_t *weights = parameterInit(weightsParam, weightsGrad);

    float biasData[] = {-1, 3};
    size_t biasDims[] = {2, 1};
    size_t biasNumberOfDims = 2;
    tensor_t *biasParam = tensorInitSymInt32(biasData, biasDims, biasNumberOfDims, HTE, NULL);
    tensor_t *biasGrad = gradInitSymInt32(biasParam, HTE, NULL);
    parameter_t *bias = parameterInit(biasParam, biasGrad);

    float inputData[] = {0.f, 1.f, 2.f};
    size_t inputDims[] = {3};
    size_t inputNumberOfDims = 1;
    tensor_t *input = tensorInitSymInt32(inputData, inputDims, inputNumberOfDims, HTE, NULL);

    float outputData[numberOfOutputs];
    size_t outputDims[] = {2, 1};
    size_t outputNumberOfDims = 2;
    tensor_t *output = tensorInitSymInt32(outputData, outputDims, outputNumberOfDims, HTE, NULL);

    quantization_t *test = quantizationInitSymInt32(HTE);
    layer_t *linearLayer = linearLayerInit(weights, bias, test, test, test, test);

    linearForward(linearLayer, input, output);

    float outputFloatData[numberOfOutputs];
    quantization_t outputFloatQ;
    initFloat32Quantization(&outputFloatQ);
    tensor_t outputFloat;
    setTensorValuesForConversion((uint8_t *)outputFloatData, &outputFloatQ, output, &outputFloat);
    convertTensor(output, &outputFloat);

    float *actual = (float *)outputFloat.data;

    float expected[] = {-5, -4};

    for (size_t i = 0; i < numberOfOutputs; i++) {
        TEST_ASSERT_FLOAT_WITHIN(0.1f, expected[i], actual[i]);
    }
}

void testLinearBackwardFloat() {
    parameter_t weights;
    tensor_t weightsParam;
    tensor_t weightsGrad;

    float weightData[] = {-1.f, 2.f, -3.f, 4.f, 5.f, -6.f};
    size_t weightDims[] = {2, 3};
    size_t weightNumberOfDims = 2;
    size_t weightOrderOfDims[] = {0, 1};
    shape_t weightShape = {.dimensions = weightDims,
                           .orderOfDimensions = weightOrderOfDims,
                           .numberOfDimensions = weightNumberOfDims};
    quantization_t weightQ;
    initFloat32Quantization(&weightQ);
    setTensorValues(&weightsParam, (uint8_t *)weightData, &weightShape, &weightQ, NULL);
    float weightGradData[] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    quantization_t weightGradQ;
    initFloat32Quantization(&weightGradQ);
    setTensorValues(&weightsGrad, (uint8_t *)weightGradData, &weightShape, &weightGradQ, NULL);
    setParameterValues(&weights, &weightsParam, &weightsGrad);

    parameter_t bias;
    tensor_t biasParam;
    tensor_t biasGrad;

    float biasData[] = {-1.f, 3.f};
    size_t biasDims[] = {2, 1};
    size_t biasNumberOfDims = 2;
    size_t biasOrderOfDims[] = {0, 1};
    shape_t biasShape = {.dimensions = biasDims,
                         .orderOfDimensions = biasOrderOfDims,
                         .numberOfDimensions = biasNumberOfDims};
    quantization_t biasQ;
    initFloat32Quantization(&biasQ);
    setTensorValues(&biasParam, (uint8_t *)biasData, &biasShape, &biasQ, NULL);
    float biasGradData[] = {0.f, 0.f};
    quantization_t biasGradQ;
    initFloat32Quantization(&biasGradQ);
    setTensorValues(&biasGrad, (uint8_t *)biasGradData, &biasShape, &biasGradQ, NULL);
    setParameterValues(&bias, &biasParam, &biasGrad);

    tensor_t forwardInput;
    float forwardInputData[] = {0.f, 1.f, 2.f};
    size_t forwardInputDims[] = {1, 3};
    size_t forwardInputNumberOfDims = 2;
    size_t forwardInputOrderOfDims[] = {0, 1};
    shape_t forwardInputShape = {.dimensions = forwardInputDims,
                                 .orderOfDimensions = forwardInputOrderOfDims,
                                 .numberOfDimensions = forwardInputNumberOfDims};
    quantization_t forwardInputQ;
    initFloat32Quantization(&forwardInputQ);
    setTensorValues(&forwardInput, (uint8_t *)forwardInputData, &forwardInputShape, &forwardInputQ, NULL);

    tensor_t loss;
    float lossData[] = {-4.f, -3.f};
    size_t lossDims[] = {2, 1};
    size_t lossNumberOfDims = 2;
    size_t lossOrderOfDims[] = {0, 1};
    shape_t lossShape = {.dimensions = lossDims,
                         .orderOfDimensions = lossOrderOfDims,
                         .numberOfDimensions = lossNumberOfDims};
    quantization_t lossQ;
    initFloat32Quantization(&lossQ);
    setTensorValues(&loss, (uint8_t *)lossData, &lossShape, &lossQ, NULL);

    tensor_t propLoss;
    float propLossData[3];
    size_t propLossDims[] = {1, 3};
    size_t propLossNumberOfDims = 2;
    size_t propLossOrderOfDims[] = {0, 1};
    shape_t propLossShape = {.dimensions = propLossDims,
                             .orderOfDimensions = propLossOrderOfDims,
                             .numberOfDimensions = propLossNumberOfDims};
    quantization_t propLossQ;
    initFloat32Quantization(&propLossQ);
    setTensorValues(&propLoss, (uint8_t *)propLossData, &propLossShape, &propLossQ, NULL);

    layer_t linearLayer;
    layerConfig_t linearConfig;
    linearConfig_t linCfg;
    linearConfig.linear = &linCfg;
    quantization_t testQ;
    initFloat32Quantization(&testQ);
    linearInitConfig(linearConfig.linear, &weights, &bias, &testQ, &testQ, &testQ, &testQ);
    initLayer(&linearLayer, LINEAR, &linearConfig);

    linearBackward(&linearLayer, &forwardInput, &loss, &propLoss);

    float expected_propagated_loss[] = {-8.f, -23.f, 30.f};
    float expected_weight_grad[] = {0.f, -4.f, -8.f, 0.f, -3.f, -6.f};
    float expected_bias_grad[] = {-4.f, -3.f};

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(
        expected_weight_grad, linearConfig.linear->weights->grad->data,
        calcNumberOfElementsByShape(linearConfig.linear->weights->param->shape));

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(expected_propagated_loss, propLoss.data,
                                  sizeof(expected_propagated_loss) / sizeof(float));

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(
        expected_bias_grad, linearConfig.linear->bias->grad->data,
        calcNumberOfElementsByShape(linearConfig.linear->bias->param->shape));
}


void testLinearBackwardFloatWithMismatchedQuantizations() {
    parameter_t weights;
    tensor_t weightsParam;
    tensor_t weightsGrad;

    float weightData[] = {-1.f, 2.f, -3.f, 4.f, 5.f, -6.f};
    size_t weightDims[] = {2, 3};
    size_t weightNumberOfDims = 2;
    size_t weightOrderOfDims[] = {0, 1};
    shape_t weightShape = {.dimensions = weightDims,
                           .orderOfDimensions = weightOrderOfDims,
                           .numberOfDimensions = weightNumberOfDims};
    quantization_t weightQ;
    initFloat32Quantization(&weightQ);
    setTensorValues(&weightsParam, (uint8_t *)weightData, &weightShape, &weightQ, NULL);
    float weightGradData[] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    quantization_t weightGradQ;
    initFloat32Quantization(&weightGradQ);
    setTensorValues(&weightsGrad, (uint8_t *)weightGradData, &weightShape, &weightGradQ, NULL);
    setParameterValues(&weights, &weightsParam, &weightsGrad);

    parameter_t bias;
    tensor_t biasParam;
    tensor_t biasGrad;

    float biasData[] = {-1.f, 3.f};
    size_t biasDims[] = {2, 1};
    size_t biasNumberOfDims = 2;
    size_t biasOrderOfDims[] = {0, 1};
    shape_t biasShape = {.dimensions = biasDims,
                         .orderOfDimensions = biasOrderOfDims,
                         .numberOfDimensions = biasNumberOfDims};
    quantization_t biasQ;
    initFloat32Quantization(&biasQ);
    setTensorValues(&biasParam, (uint8_t *)biasData, &biasShape, &biasQ, NULL);
    float biasGradData[] = {0.f, 0.f};
    quantization_t biasGradQ;
    initFloat32Quantization(&biasGradQ);
    setTensorValues(&biasGrad, (uint8_t *)biasGradData, &biasShape, &biasGradQ, NULL);
    setParameterValues(&bias, &biasParam, &biasGrad);

    tensor_t forwardInput;
    float forwardInputData[] = {0.f, 1.f, 2.f};
    size_t forwardInputDims[] = {1, 3};
    size_t forwardInputNumberOfDims = 2;
    size_t forwardInputOrderOfDims[] = {0, 1};
    shape_t forwardInputShape = {.dimensions = forwardInputDims,
                                 .orderOfDimensions = forwardInputOrderOfDims,
                                 .numberOfDimensions = forwardInputNumberOfDims};
    quantization_t forwardInputQ;
    initFloat32Quantization(&forwardInputQ);
    setTensorValues(&forwardInput, (uint8_t *)forwardInputData, &forwardInputShape, &forwardInputQ, NULL);

    tensor_t loss;
    float lossData[] = {-4.f, -3.f};
    size_t lossDims[] = {2, 1};
    size_t lossNumberOfDims = 2;
    size_t lossOrderOfDims[] = {0, 1};
    shape_t lossShape = {.dimensions = lossDims,
                         .orderOfDimensions = lossOrderOfDims,
                         .numberOfDimensions = lossNumberOfDims};
    quantization_t lossQ;
    initFloat32Quantization(&lossQ);
    setTensorValues(&loss, (uint8_t *)lossData, &lossShape, &lossQ, NULL);

    tensor_t lossAsym;
    uint8_t lossAsymData[2];
    quantization_t lossAsymQ;
    asymQConfig_t lossAsymQC;
    initAsymQConfig(8, HTE, &lossAsymQC);
    initAsymQuantization(&lossAsymQC, &lossAsymQ);
    setTensorValuesForConversion(lossAsymData, &lossAsymQ, &loss, &lossAsym);
    convertTensor(&loss, &lossAsym);

    tensor_t propLoss;
    float propLossData[3];
    size_t propLossDims[] = {1, 3};
    size_t propLossNumberOfDims = 2;
    size_t propLossOrderOfDims[] = {0, 1};
    shape_t propLossShape = {.dimensions = propLossDims,
                             .orderOfDimensions = propLossOrderOfDims,
                             .numberOfDimensions = propLossNumberOfDims};
    quantization_t propLossQ;
    initFloat32Quantization(&propLossQ);
    setTensorValues(&propLoss, (uint8_t *)propLossData, &propLossShape, &propLossQ, NULL);

    layer_t linearLayer;
    layerConfig_t linearConfig;
    linearConfig_t linCfg;
    linearConfig.linear = &linCfg;
    quantization_t testQ;
    initFloat32Quantization(&testQ);
    linearInitConfig(linearConfig.linear, &weights, &bias, &testQ, &testQ, &testQ, &testQ);
    initLayer(&linearLayer, LINEAR, &linearConfig);

    linearBackward(&linearLayer, &forwardInput, &lossAsym, &propLoss);

    float expectedPropagatedLoss[] = {-8.f, -23.f, 30.f};
    float expectedWeightGrad[] = {0.f, -4.f, -8.f, 0.f, -3.f, -6.f};
    float expectedBiasGrad[] = {-4.f, -3.f};


    size_t sizeWeights = calcNumberOfElementsByParameter(&weights);
    float *actualWeightGrad = (float *)linearConfig.linear->weights->grad->data;
    for(size_t i = 0; i < sizeWeights; i++) {
        TEST_ASSERT_FLOAT_WITHIN(0.1f, expectedWeightGrad[i], actualWeightGrad[i]);
    }

    size_t sizeBias = calcNumberOfElementsByParameter(&bias);
    float *actualBiasGrad = (float *)linearConfig.linear->bias->grad->data;
    for(size_t i = 0; i < sizeBias; i++) {
        TEST_ASSERT_FLOAT_WITHIN(0.1f, expectedBiasGrad[i], actualBiasGrad[i]);
    }

    size_t sizePropLoss = calcNumberOfElementsByTensor(&propLoss);
    float *actualPropLoss = (float *)propLoss.data;
    for(size_t i = 0; i < sizePropLoss; i++) {
        TEST_ASSERT_FLOAT_WITHIN(0.1f, expectedPropagatedLoss[i], actualPropLoss[i]);
    }
}

void testLinearBackwardSymInt32() {

    size_t numberOfWeights = 6;
    size_t numberOfBiases = 2;
    size_t numberOfForwardInputs = 3;

    float weightFloatData[] = {-1.f, 2.f, -3.f, 4.f, 5.f, -6.f};
    size_t weightDims[] = {2, 3};
    size_t weightNumberOfDims = 2;
    tensor_t *weightsParam = tensorInitSymInt32(weightFloatData, weightDims, weightNumberOfDims, HTE, NULL);
    tensor_t *weightsGrad = gradInitSymInt32(weightsParam, HTE, NULL);
    parameter_t *weights = parameterInit(weightsParam, weightsGrad);


    int32_t biasData[] = {-1, 3};
    size_t biasDims[] = {2, 1};
    size_t biasNumberOfDims = 2;
    tensor_t *biasParam = tensorInitInt32(biasData, biasDims, biasNumberOfDims, NULL);
    tensor_t *biasGrad = gradInitInt32(biasParam, NULL);
    parameter_t *bias = parameterInit(biasParam, biasGrad);

    float forwardInputData[] = {0.f, 1.f, 2.f};
    size_t forwardInputDims[] = {1, 3};
    size_t forwardInputNumberOfDims = 2;
    tensor_t *forwardInput = tensorInitSymInt32(forwardInputData, forwardInputDims, forwardInputNumberOfDims, HTE, NULL);

    float lossData[] = {-4.f, -3.f};
    size_t lossDims[] = {2, 1};
    size_t lossNumberOfDims = 2;
    tensor_t *loss = tensorInitSymInt32(lossData, lossDims, lossNumberOfDims, HTE, NULL);

    float propLossData[numberOfForwardInputs];
    size_t propLossDims[] = {numberOfForwardInputs};
    size_t propLossNumberOfDims = 1;
    tensor_t *propLoss = tensorInitSymInt32(propLossData, propLossDims, propLossNumberOfDims, HTE, NULL);

    quantization_t *test = quantizationInitSymInt32(HTE);
    layer_t *linearLayer = linearLayerInit(weights, bias, test, test, test, test);

    linearBackward(linearLayer, forwardInput, loss, propLoss);

    linearConfig_t *linearConfig = linearLayer->config->linear;
    tensor_t weightGradsFloat;
    float weightGradFloatData[numberOfWeights];
    quantization_t weightGradFloatQ;
    initFloat32Quantization(&weightGradFloatQ);
    setTensorValuesForConversion((uint8_t *)weightGradFloatData, &weightGradFloatQ, linearConfig->weights->grad,
                                 &weightGradsFloat);
    convertTensor(linearConfig->weights->grad, &weightGradsFloat);

    float expectedWeightGrads[] = {0.f, -4.f, -8.f, 0.f, -3.f, -6.f};
    float *actualWeightGrads = (float *)weightGradsFloat.data;

    for (size_t i = 0; i < numberOfWeights; i++) {
        TEST_ASSERT_FLOAT_WITHIN(0.1f, expectedWeightGrads[i], actualWeightGrads[i]);
    }

    tensor_t biasGradsFloat;
    float biasGradFloatData[numberOfBiases];
    quantization_t biasGradFloatQ;
    initFloat32Quantization(&biasGradFloatQ);
    setTensorValuesForConversion((uint8_t *)biasGradFloatData, &biasGradFloatQ, linearConfig->bias->grad,
                                 &biasGradsFloat);
    convertTensor(linearConfig->bias->grad, &biasGradsFloat);

    float expectedBiasGrads[] = {-4.f, -3.f};
    float *actualBiasGrads = (float *)biasGradsFloat.data;
    for(size_t i = 0; i < numberOfBiases; i++) {
        TEST_ASSERT_FLOAT_WITHIN(0.1f, expectedBiasGrads[i], actualBiasGrads[i]);
    }

    tensor_t propLossFloat;
    float propLossFloatData[numberOfForwardInputs];
    quantization_t propLossFloatQ;
    initFloat32Quantization(&propLossFloatQ);
    setTensorValuesForConversion((uint8_t *)propLossFloatData, &propLossFloatQ, propLoss, &propLossFloat);
    convertTensor(propLoss, &propLossFloat);

    float *propLossFloatArr = (float *)propLossFloat.data;

    float expectedPropagatedLoss[] = {-8.f, -23.f, 30.f};

    for (size_t i = 0; i < numberOfForwardInputs; i++) {
        TEST_ASSERT_FLOAT_WITHIN(.2f, expectedPropagatedLoss[i], propLossFloatArr[i]);
    }
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(testLinearForwardFloat);
    RUN_TEST(testLinearBackwardFloat);

    RUN_TEST(testLinearBackwardFloatWithMismatchedQuantizations);

    RUN_TEST(testLinearForwardSymInt32);
    RUN_TEST(testLinearBackwardSymInt32);
    return UNITY_END();
}
