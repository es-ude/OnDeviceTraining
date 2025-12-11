#define SOURCE_FILE "SGD-UTEST"
#include <stdlib.h>

#include "Linear.h"
#include "unity.h"
#include "Layer.h"
#include "Tensor.h"
#include "Sgd.h"
#include "SgdAPI.h"
#include "LinearAPI.h"
#include "TensorAPI.h"
#include "OptimizerAPI.h"

void setUp() {}
void tearDown() {}

void testSgdMCreateOptim() {

    float weightData[] = {0.f, 1.f, 2.f};
    size_t weightDims[] = {1, 3};
    size_t weightNumberOfDims = 2;
    tensor_t *weightParam = tensorInitFloat(weightData, weightDims, weightNumberOfDims, NULL);
    float weightGradData[] = {0.f, 0.f, 0.f};
    tensor_t *weightGrad = tensorInitFloat(weightGradData, weightDims, weightNumberOfDims, NULL);
    parameter_t *weights = parameterInit(weightParam, weightGrad);

    float biasData[] = {0.f, 1.f, -1.f};
    size_t biasDims[] = {1, 3};
    size_t biasNumberOfDims = 1;
    tensor_t *biasParam = tensorInitFloat(biasData, biasDims, biasNumberOfDims, NULL);
    float biasGradData[] = {0.f, 0.f, 0.f};
    tensor_t *biasGrad = tensorInitFloat(biasGradData, biasDims, biasNumberOfDims, NULL);
    parameter_t *bias = parameterInit(biasParam, biasGrad);

    quantization_t outputQ;
    initFloat32Quantization(&outputQ);
    layer_t *linear0 = linearLayerInit(weights, bias, FLOAT32, FLOAT32, &outputQ);

    layer_t relu0;
    initLayer(&relu0, RELU, NULL, FLOAT_LAYER, FLOAT32, NULL);

    layer_t *linear1 = linearLayerInit(weights, bias, FLOAT32, FLOAT32, &outputQ);

    layer_t *model[] = {linear0, &relu0, linear1};
    size_t sizeModel = sizeof(model) / sizeof(model[0]);
    float lr = 0.1f;
    float momentumFactor = 0.9f;
    float weightDecay = 0.5f;

    optimizer_t *optim = sgdMCreateOptim(lr, momentumFactor, weightDecay, model, sizeModel, FLOAT32);
    sgd_t *sgd = optim->impl->sgd;

    linearConfig_t *linear0Conf = linear0->config->linear;
    linearConfig_t *linear1Conf = linear1->config->linear;

    TEST_ASSERT_EQUAL_FLOAT(lr, sgd->learningRate);
    TEST_ASSERT_EQUAL_FLOAT(momentumFactor, sgd->momentumFactor);
    TEST_ASSERT_EQUAL_FLOAT(weightDecay, sgd->weightDecay);
    TEST_ASSERT_EQUAL_size_t(4, optim->sizeStates);

    TEST_ASSERT_EQUAL_PTR(linear0Conf->weights, optim->parameter[0]);
    TEST_ASSERT_EQUAL_PTR(linear0Conf->bias, optim->parameter[1]);
    TEST_ASSERT_EQUAL_PTR(linear1Conf->weights, optim->parameter[2]);
    TEST_ASSERT_EQUAL_PTR(linear1Conf->bias, optim->parameter[3]);

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(linear0Conf->weights->grad->data,
                                  optim->states[0]->stateBuffers[0]->data,
                                  calcNumberOfElementsByParameter(linear0Conf->weights));

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(linear0Conf->bias->grad->data,
                                  optim->states[1]->stateBuffers[0]->data,
                                  calcNumberOfElementsByParameter(linear0Conf->bias));

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(linear1Conf->weights->grad->data,
                                  optim->states[2]->stateBuffers[0]->data,
                                  calcNumberOfElementsByParameter(linear1Conf->weights));
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(linear1Conf->bias->grad->data,
                                  optim->states[3]->stateBuffers[0]->data,
                                  calcNumberOfElementsByParameter(linear1Conf->bias));
}

void testSGDStep() {
    float weightData[] = {1.f, 2.f, -3.f};
    size_t weightDims[] = {3, 1};
    size_t weightNumberOfDims = 1;
    tensor_t *weightParam = tensorInitFloat(weightData, weightDims, weightNumberOfDims, NULL);
    tensor_t *weightGrad = gradInitFloat(weightParam, NULL);
    float weightGradData[] = {1.f, -1.f, 2.f};
    weightGrad->data = (uint8_t *)weightGradData;
    parameter_t *weights = parameterInit(weightParam, weightGrad);

    float biasData[] = {-1.f, 3.f};
    size_t biasDims[] = {2, 1};
    size_t biasNumberOfDims = 1;
    tensor_t *biasParam = tensorInitFloat(biasData, biasDims, biasNumberOfDims, NULL);
    tensor_t *biasGrad = gradInitFloat(biasParam, NULL);
    float biasGradData[] = {1.f, 3.f};
    biasGrad->data = (uint8_t *) biasGradData;
    parameter_t *bias = parameterInit(biasParam, biasGrad);

    quantization_t outputQ;
    initFloat32Quantization(&outputQ);
    layer_t *linear = linearLayerInit(weights, bias, FLOAT_LAYER, FLOAT32, &outputQ);

    layer_t *model[] = {linear};
    size_t modelSize = 1;
    float lr = 0.1f;
    float momentumFactor = 0.9f;
    float weightDecay = 0.01f;

    optimizer_t *sgd = sgdMCreateOptim(lr, momentumFactor, weightDecay, model, modelSize, FLOAT32);

    optimizerFunctions_t sgdFns = optimizerFunctions[sgd->type];
    sgdFns.step(sgd);

    float wPExpected[] = {0.899f, 2.098f, -3.197f};
    float bPExpected[] = {-1.099f, 2.697f};

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(wPExpected, linear->config->linear->weights->param->data,
                                  sizeof(wPExpected)/sizeof(float));
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(bPExpected, linear->config->linear->bias->param->data,
                                  sizeof(bPExpected)/sizeof(float));

    sgdFns.step(sgd);

    float wPExpected2[] = {0.707201f, 2.284102f, -3.571103f};
    float bPExpected2[] = {-1.287001f, 2.121603f};

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(wPExpected2, linear->config->linear->weights->param->data,
                                  sizeof(wPExpected2)/sizeof(float));
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(bPExpected2, linear->config->linear->bias->param->data,
                                  sizeof(bPExpected2)/sizeof(float));
}

void testSGDZeroGrad() {
    float weightData[] = {1.f, 2.f, -3.f};
    size_t weightDims[] = {3, 1};
    size_t weightNumberOfDims = 2;
    tensor_t *weightParam = tensorInitFloat(weightData, weightDims, weightNumberOfDims, NULL);
    tensor_t *weightGrad = gradInitFloat(weightParam, NULL);
    float weightGradData[] = {1.f, -1.f, 2.f};
    weightGrad->data = (uint8_t *)weightGradData;
    parameter_t *weights = parameterInit(weightParam, weightGrad);


    float biasData[] = {-1.f, 3.f};
    size_t biasDims[] = {2, 1};
    size_t biasNumberOfDims = 2;
    tensor_t *biasParam = tensorInitFloat(biasData, biasDims, biasNumberOfDims, NULL);
    tensor_t *biasGrad = gradInitFloat(weightParam, NULL);
    float biasGradData[] = {1.f, 3.f};
    biasGrad->data = (uint8_t *)biasGradData;
    parameter_t *bias = parameterInit(biasParam, biasGrad);

    quantization_t outputQ;
    initFloat32Quantization(&outputQ);
    layer_t *linear = linearLayerInit(weights, bias, FLOAT_LAYER, FLOAT32, &outputQ);

    layer_t *model[] = {linear};
    size_t modelSize = 1;
    float lr = 0.1f;
    float momentumFactor = 0.9f;
    float weightDecay = 0.01f;

    optimizer_t *sgd = sgdMCreateOptim(lr, momentumFactor, weightDecay, model, modelSize, FLOAT32);

    sgdZeroGrad(sgd);
    float wGradExpected[] = {0.f, 0.f, 0.f};
    float bGradExpected[] = {0.f, 0.f};

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(wGradExpected, weights->grad->data,
                                  sizeof(wGradExpected)/sizeof(float));
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(bGradExpected, bias->grad->data,
                                  sizeof(bGradExpected)/sizeof(float));

}

int main() {
    UNITY_BEGIN();
    RUN_TEST(testSgdMCreateOptim);
    RUN_TEST(testSGDStep);
    RUN_TEST(testSGDZeroGrad);
    UNITY_END();
}
