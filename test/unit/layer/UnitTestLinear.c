#include "DTypes.h"
#include "Layer.h"
#include "Linear.h"
#include "Rounding.h"
#include "Tensor.h"
#include "TensorConversion.h"
#include "unity.h"

#include <string.h>

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
    linearInitConfig(linearConfig.linear, &weights, &bias);

    initLayer(&linearLayer, LINEAR, &linearConfig, FLOAT_LAYER, inputQ.type, &outputQ);

    linearForward(&linearLayer, &input, &output);

    float expected[] = {-5.f, -4.f};

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(expected, output.data, 2);
}

void testLinearForwardAsym() {
    size_t numberOfWeights = 6;
    size_t numberOfBiases = 2;
    size_t numberOfInputs = 3;
    size_t numberOfOutputs = 2;

    parameter_t weights;

    tensor_t weightsFloat;
    float weightFloatData[] = {-1.f, 2.f, -3.f, 4.f, 5.f, -6.f};
    size_t weightDims[] = {2, 3};
    size_t weightNumberOfDims = 2;
    size_t weightOrderOfDims[] = {0, 1};
    shape_t weightShape = {.dimensions = weightDims,
                           .orderOfDimensions = weightOrderOfDims,
                           .numberOfDimensions = weightNumberOfDims};
    quantization_t weightFloatQ;
    initFloat32Quantization(&weightFloatQ);
    setTensorValues(&weightsFloat, (uint8_t *)weightFloatData, &weightShape, &weightFloatQ, NULL);

    tensor_t weightsParamAsym;
    uint8_t weightAsymData[numberOfWeights];
    asymQConfig_t weightsAsymQConfig;
    initAsymQConfig(8, HTE, &weightsAsymQConfig);
    quantization_t weightAsymQ;
    initAsymQuantization(&weightsAsymQConfig, &weightAsymQ);
    setTensorValuesForConversion(weightAsymData, &weightAsymQ, &weightsFloat, &weightsParamAsym);
    convertTensor(&weightsFloat, &weightsParamAsym);
    setParameterValues(&weights, &weightsParamAsym, NULL);

    parameter_t bias;
    int32_t biasIntData[] = {-1, 3};
    size_t biasDims[] = {2, 1};
    size_t biasNumberOfDims = 2;
    size_t biasOrderOfDims[] = {0, 1};
    shape_t biasShape = {.dimensions = biasDims,
                         .orderOfDimensions = biasOrderOfDims,
                         .numberOfDimensions = biasNumberOfDims};
    quantization_t biasIntQ;
    initInt32Quantization(&biasIntQ);
    tensor_t biasIntParam;
    setTensorValues(&biasIntParam, (uint8_t *)biasIntData, &biasShape, &biasIntQ, NULL);
    setParameterValues(&bias, &biasIntParam, NULL);

    float inputFloatData[] = {0.f, 1.f, 2.f};
    size_t inputDims[] = {3};
    size_t inputNumberOfDims = 1;
    size_t inputOrderOfDims[] = {0};
    shape_t inputShape = {.dimensions = inputDims,
                          .orderOfDimensions = inputOrderOfDims,
                          .numberOfDimensions = inputNumberOfDims};

    quantization_t inputFloatQ;
    initFloat32Quantization(&inputFloatQ);

    tensor_t inputFloat;
    setTensorValues(&inputFloat, (uint8_t *)inputFloatData, &inputShape, &inputFloatQ, NULL);

    asymQConfig_t inputAsymQConfig;
    initAsymQConfig(8, HTE, &inputAsymQConfig);
    quantization_t inputAsymQ;
    initAsymQuantization(&inputAsymQConfig, &inputAsymQ);
    uint8_t inputAsymData[numberOfInputs * calcBytesPerElement(&inputAsymQ)];

    tensor_t inputAsym;
    setTensorValuesForConversion(inputAsymData, &inputAsymQ, &inputFloat, &inputAsym);
    convertTensor(&inputFloat, &inputAsym);

    size_t outputDims[] = {2, 1};
    size_t outputNumberOfDims = 2;
    size_t outputOrderOfDims[] = {0, 1};
    shape_t outputShape = {.dimensions = outputDims,
                           .orderOfDimensions = outputOrderOfDims,
                           .numberOfDimensions = outputNumberOfDims};

    asymQConfig_t outputAsymQConfig;
    initAsymQConfig(8, HTE, &outputAsymQConfig);
    quantization_t outputAsymQ;
    initAsymQuantization(&outputAsymQConfig, &outputAsymQ);
    uint8_t outputAsymData[numberOfOutputs * calcBytesPerElement(&outputAsymQ)];

    tensor_t outputAsym;
    setTensorValues(&outputAsym, outputAsymData, &outputShape, &outputAsymQ, NULL);

    layer_t linearLayer;
    layerConfig_t linearConfig;
    linearConfig_t linCfg;
    linearConfig.linear = &linCfg;
    linearInitConfig(linearConfig.linear, &weights, &bias);
    initLayer(&linearLayer, LINEAR, &linearConfig, ASYM_LAYER, inputAsymQ.type, &outputAsymQ);

    linearForward(&linearLayer, &inputAsym, &outputAsym);

    float outputFloatData[numberOfOutputs];
    quantization_t outputFloatQ;
    initFloat32Quantization(&outputFloatQ);
    tensor_t outputFloat;
    setTensorValuesForConversion((uint8_t *)outputFloatData, &outputFloatQ, &outputAsym, &outputFloat);
    convertTensor(&outputAsym, &outputFloat);

    float *actual = (float *)outputFloat.data;

    // values are off, because scale is ignored, when adding bias
    // float expected[] = {-5.f, -4.f};
    float expected[] = {-4, -7};

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
    linearInitConfig(linearConfig.linear, &weights, &bias);
    initLayer(&linearLayer, LINEAR, &linearConfig, FLOAT_LAYER, forwardInputQ.type, &lossQ);

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

void testLinearBackwardAsym() {

    size_t numberOfWeights = 6;
    size_t numberOfBiases = 2;
    size_t numberOfLosses = 2;
    size_t numberOfForwardInputs = 3;

    parameter_t weights;
    tensor_t weightsParamFloat;
    tensor_t weightsParamAsym;
    tensor_t weightsGradAsym;

    float weightFloatData[] = {-1.f, 2.f, -3.f, 4.f, 5.f, -6.f};
    /*
    -1,  4
     2,  5
    -3, -6
    */
    size_t weightDims[] = {2, 3};
    size_t weightNumberOfDims = 2;
    size_t weightOrderOfDims[] = {0, 1};
    shape_t weightShape = {.dimensions = weightDims,
                           .orderOfDimensions = weightOrderOfDims,
                           .numberOfDimensions = weightNumberOfDims};
    quantization_t weightFloatQ;
    initFloat32Quantization(&weightFloatQ);
    setTensorValues(&weightsParamFloat, (uint8_t *)weightFloatData, &weightShape, &weightFloatQ, NULL);

    asymQConfig_t weightsAsymQConfig;
    initAsymQConfig(8, HTE, &weightsAsymQConfig);
    quantization_t weightsAsymQ;
    initAsymQuantization(&weightsAsymQConfig, &weightsAsymQ);
    uint8_t weightsAsymData[numberOfWeights * calcBytesPerElement(&weightsAsymQ)];
    setTensorValuesForConversion(weightsAsymData, &weightsAsymQ, &weightsParamFloat, &weightsParamAsym);
    convertTensor(&weightsParamFloat, &weightsParamAsym);

    asymQConfig_t weightGradsAsymQConfig;
    initAsymQConfig(8, HTE, &weightGradsAsymQConfig);
    quantization_t weightGradsAsymQ;
    initAsymQuantization(&weightGradsAsymQConfig, &weightGradsAsymQ);
    uint8_t weightGradsData[numberOfWeights * calcBytesPerElement(&weightGradsAsymQ)];
    setTensorValues(&weightsGradAsym, weightGradsData, &weightShape, &weightGradsAsymQ, NULL);
    memset(weightGradsData, 0, numberOfWeights * calcBytesPerElement(&weightGradsAsymQ));
    setParameterValues(&weights, &weightsParamAsym, &weightsGradAsym);

    parameter_t bias;
    tensor_t biasParam;
    tensor_t biasGradAsym;

    int32_t biasData[] = {-1, 3};
    size_t biasDims[] = {2, 1};
    size_t biasNumberOfDims = 2;
    size_t biasOrderOfDims[] = {0, 1};
    shape_t biasShape = {.dimensions = biasDims,
                         .orderOfDimensions = biasOrderOfDims,
                         .numberOfDimensions = biasNumberOfDims};
    quantization_t biasQ;
    initInt32Quantization(&biasQ);
    setTensorValues(&biasParam, (uint8_t *)biasData, &biasShape, &biasQ, NULL);

    asymQConfig_t biasGradAsymQConfig;
    initAsymQConfig(8, HTE, &biasGradAsymQConfig);
    quantization_t biasGradAsymQ;
    initAsymQuantization(&biasGradAsymQConfig, &biasGradAsymQ);
    uint8_t biasGradAsymData[numberOfBiases * calcBytesPerElement(&biasGradAsymQ)];
    // IMPORTANT, OTHERWISE BIAS GRADS IS FILLED WITH RANDOM DATA
    memset(biasGradAsymData, 0, numberOfBiases * calcBytesPerElement(&biasGradAsymQ));
    setTensorValues(&biasGradAsym, biasGradAsymData, &biasShape, &biasGradAsymQ, NULL);

    setParameterValues(&bias, &biasParam, &biasGradAsym);

    float forwardInputFloatData[] = {0.f, 1.f, 2.f};
    size_t forwardInputDims[] = {1, 3};
    size_t forwardInputNumberOfDims = 2;
    size_t forwardInputOrderOfDims[] = {0, 1};
    shape_t forwardInputShape = {.dimensions = forwardInputDims,
                                 .orderOfDimensions = forwardInputOrderOfDims,
                                 .numberOfDimensions = forwardInputNumberOfDims};

    quantization_t forwardInputFloatQ;
    initFloat32Quantization(&forwardInputFloatQ);

    tensor_t forwardInputFloat;
    setTensorValues(&forwardInputFloat, (uint8_t *)forwardInputFloatData, &forwardInputShape,
                    &forwardInputFloatQ, NULL);

    asymQConfig_t forwardInputAsymQConfig;
    initAsymQConfig(8, HTE, &forwardInputAsymQConfig);
    quantization_t forwardInputAsymQ;
    initAsymQuantization(&forwardInputAsymQConfig, &forwardInputAsymQ);
    uint8_t forwardInputAsymData[numberOfForwardInputs * calcBytesPerElement(&forwardInputAsymQ)];

    tensor_t forwardInputAsym;
    setTensorValuesForConversion(forwardInputAsymData, &forwardInputAsymQ, &forwardInputFloat,
                                 &forwardInputAsym);
    convertTensor(&forwardInputFloat, &forwardInputAsym);

    float lossFloatData[] = {-4.f, -3.f};
    size_t lossDims[] = {2, 1};
    size_t lossNumberOfDims = 2;
    size_t lossOrderOfDims[] = {0, 1};
    shape_t lossShape = {.dimensions = lossDims,
                         .orderOfDimensions = lossOrderOfDims,
                         .numberOfDimensions = lossNumberOfDims};

    quantization_t lossFloatQ;
    initFloat32Quantization(&lossFloatQ);

    tensor_t lossFloat;
    setTensorValues(&lossFloat, (uint8_t *)lossFloatData, &lossShape, &lossFloatQ, NULL);

    asymQConfig_t lossAsymQConfig;
    initAsymQConfig(8, HTE, &lossAsymQConfig);
    quantization_t lossAsymQ;
    initAsymQuantization(&lossAsymQConfig, &lossAsymQ);
    uint8_t lossAsymData[numberOfLosses * calcBytesPerElement(&lossAsymQ)];

    tensor_t lossAsym;
    setTensorValuesForConversion(lossAsymData, &lossAsymQ, &lossFloat, &lossAsym);
    convertTensor(&lossFloat, &lossAsym);

    size_t propLossDims[] = {numberOfForwardInputs};
    size_t propLossNumberOfDims = 1;
    size_t propLossOrderOfDims[] = {0};
    shape_t propLossShape = {.dimensions = propLossDims,
                             .orderOfDimensions = propLossOrderOfDims,
                             .numberOfDimensions = propLossNumberOfDims};

    asymQConfig_t propLossAsymQConfig;
    initAsymQConfig(8, HTE, &propLossAsymQConfig);
    quantization_t propLossAsymQ;
    initAsymQuantization(&propLossAsymQConfig, &propLossAsymQ);
    uint8_t propLossAsymData[numberOfForwardInputs * calcBytesPerElement(&propLossAsymQ)];

    tensor_t propLossAsym;
    setTensorValues(&propLossAsym, propLossAsymData, &propLossShape, &propLossAsymQ, NULL);

    layer_t linearLayer;
    layerConfig_t linearConfig;
    linearConfig_t linCfg;
    linearConfig.linear = &linCfg;
    linearInitConfig(linearConfig.linear, &weights, &bias);
    initLayer(&linearLayer, LINEAR, &linearConfig, ASYM_LAYER, forwardInputAsymQ.type, &lossAsymQ);

    linearBackward(&linearLayer, &forwardInputAsym, &lossAsym, &propLossAsym);

    tensor_t weightGradsFloat;
    float weightGradFloatData[numberOfWeights];
    quantization_t weightGradFloatQ;
    initFloat32Quantization(&weightGradFloatQ);
    setTensorValuesForConversion((uint8_t *)weightGradFloatData, &weightGradFloatQ, linearConfig.linear->weights->grad,
                                 &weightGradsFloat);
    convertTensor(linearConfig.linear->weights->grad, &weightGradsFloat);

    float expectedWeightGrads[] = {0.f, -4.f, -8.f, 0.f, -3.f, -6.f};
    float *actualWeightGrads = (float *)weightGradsFloat.data;

    for (size_t i = 0; i < numberOfWeights; i++) {
        TEST_ASSERT_FLOAT_WITHIN(0.1f, expectedWeightGrads[i], actualWeightGrads[i]);
    }

    tensor_t biasGradsFloat;
    float biasGradFloatData[numberOfBiases];
    quantization_t biasGradFloatQ;
    initFloat32Quantization(&biasGradFloatQ);
    setTensorValuesForConversion((uint8_t *)biasGradFloatData, &biasGradFloatQ, linearConfig.linear->bias->grad,
                                 &biasGradsFloat);
    convertTensor(linearConfig.linear->bias->grad, &biasGradsFloat);

    float expectedBiasGrads[] = {-0.125f, -0.09375f};
    float *actualBiasGrads = (float *)biasGradsFloat.data;
    for(size_t i = 0; i < numberOfBiases; i++) {
        TEST_ASSERT_FLOAT_WITHIN(0.1f, expectedBiasGrads[i], actualBiasGrads[i]);
    }

    tensor_t propLossFloat;
    float propLossFloatData[numberOfForwardInputs];
    quantization_t propLossFloatQ;
    initFloat32Quantization(&propLossFloatQ);
    setTensorValuesForConversion((uint8_t *)propLossFloatData, &propLossFloatQ, &propLossAsym, &propLossFloat);
    convertTensor(&propLossAsym, &propLossFloat);

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

    RUN_TEST(testLinearForwardAsym);
    RUN_TEST(testLinearBackwardAsym);
    UNITY_END();
}
