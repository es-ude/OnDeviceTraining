#define SOURCE_FILE "UNIT_TEST_LAYER_WEIGHTS_API"

#include "Conv1d.h"
#include "Conv1dApi.h"
#include "Conv1dTransposed.h"
#include "Conv1dTransposedApi.h"
#include "LayerCommon.h"
#include "LayerNorm.h"
#include "LayerNormApi.h"
#include "LayerQuant.h"
#include "LayerWeightsApi.h"
#include "Linear.h"
#include "LinearApi.h"
#include "QuantizationApi.h"
#include "Tensor.h"
#include "TensorApi.h"
#include "unity.h"

void setUp() {}
void tearDown() {}

void testLayerLoadWeightsLinearOverwritesWeightAndBiasTensors(void) {
    quantization_t *q = quantizationInitFloat();
    layerQuant_t lq;
    layerQuantInitUniform(&lq, q);

    layer_t *layer = linearLayerInit(
        &(linearInit_t){
            .inFeatures = 3,
            .outFeatures = 2,
            .bias = BIAS_TRUE,
        },
        &lq);

    float weightData[6] = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f};
    float biasData[2] = {-1.f, -2.f};

    layerLoadWeights(layer, weightData, biasData);

    linearConfig_t *cfg = layer->config->linear;
    float *loadedWeights = (float *)cfg->weights->param->data;
    float *loadedBias = (float *)cfg->bias->param->data;

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(weightData, loadedWeights, 6);
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(biasData, loadedBias, 2);

    freeLinearLayer(layer);
}

void testLayerLoadWeightsLinearNoBiasAcceptsNullBiasData(void) {
    quantization_t *q = quantizationInitFloat();
    layerQuant_t lq;
    layerQuantInitUniform(&lq, q);

    layer_t *layer = linearLayerInit(
        &(linearInit_t){
            .inFeatures = 3,
            .outFeatures = 2,
            .bias = BIAS_FALSE,
        },
        &lq);

    float weightData[6] = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f};
    layerLoadWeights(layer, weightData, NULL);

    linearConfig_t *cfg = layer->config->linear;
    float *loadedWeights = (float *)cfg->weights->param->data;
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(weightData, loadedWeights, 6);
    TEST_ASSERT_NULL(cfg->bias);

    freeLinearLayer(layer);
}

void testLayerLoadWeightsConv1dOverwritesWeightAndBiasTensors(void) {
    quantization_t *q = quantizationInitFloat();
    layerQuant_t lq;
    layerQuantInitUniform(&lq, q);

    layer_t *layer = conv1dLayerInit(
        &(conv1dInit_t){
            .inChannels = 2,
            .outChannels = 3,
            .kernelSize = 4,
            .bias = BIAS_TRUE,
        },
        &lq);

    /* Weight tensor: [outChannels=3, inChannels/groups=2, K=4] → 24 elems
     * Bias tensor:   [outChannels=3] → 3 elems */
    float weightData[24] = {
        1.f,  2.f,  3.f,  4.f,  5.f,  6.f,  7.f,  8.f,  9.f,  10.f, 11.f, 12.f,
        13.f, 14.f, 15.f, 16.f, 17.f, 18.f, 19.f, 20.f, 21.f, 22.f, 23.f, 24.f,
    };
    float biasData[3] = {-1.f, -2.f, -3.f};

    layerLoadWeights(layer, weightData, biasData);

    conv1dConfig_t *cfg = layer->config->conv1d;
    float *loadedWeights = (float *)cfg->weights->param->data;
    float *loadedBias = (float *)cfg->bias->param->data;

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(weightData, loadedWeights, 24);
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(biasData, loadedBias, 3);

    freeConv1dLayer(layer);
    freeQuantization(q);
}

void testLayerLoadWeightsConv1dNoBiasAcceptsNullBiasData(void) {
    quantization_t *q = quantizationInitFloat();
    layerQuant_t lq;
    layerQuantInitUniform(&lq, q);

    layer_t *layer = conv1dLayerInit(
        &(conv1dInit_t){
            .inChannels = 1,
            .outChannels = 1,
            .kernelSize = 3,
            .bias = BIAS_FALSE,
        },
        &lq);

    float weightData[3] = {0.5f, 0.25f, 0.125f};
    layerLoadWeights(layer, weightData, NULL);

    conv1dConfig_t *cfg = layer->config->conv1d;
    float *loadedWeights = (float *)cfg->weights->param->data;
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(weightData, loadedWeights, 3);
    TEST_ASSERT_NULL(cfg->bias);

    freeConv1dLayer(layer);
    freeQuantization(q);
}

void testLayerLoadWeightsConv1dTransposedOverwritesWeightAndBiasTensors(void) {
    quantization_t *q = quantizationInitFloat();
    layerQuant_t lq;
    layerQuantInitUniform(&lq, q);

    layer_t *layer = conv1dTransposedLayerInit(
        &(conv1dTransposedInit_t){
            .inChannels = 4,
            .outChannels = 2,
            .kernelSize = 3,
            .bias = BIAS_TRUE,
        },
        &lq);

    /* Weight tensor: [inChannels=4, outChannels/groups=2, K=3] → 24 elems.
     * NOTE the SWAP relative to Conv1d. */
    float weightData[24] = {0};
    for (size_t i = 0; i < 24; i++) {
        weightData[i] = (float)(i + 100);
    }
    float biasData[2] = {-10.f, -20.f};

    layerLoadWeights(layer, weightData, biasData);

    conv1dTransposedConfig_t *cfg = layer->config->conv1dTransposed;
    float *loadedWeights = (float *)cfg->weights->param->data;
    float *loadedBias = (float *)cfg->bias->param->data;

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(weightData, loadedWeights, 24);
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(biasData, loadedBias, 2);

    freeConv1dTransposedLayer(layer);
    freeQuantization(q);
}

void testLayerLoadWeightsLayerNormOverwritesGammaAndBeta(void) {
    quantization_t *q = quantizationInitFloat();
    layerQuant_t lq;
    layerQuantInitUniform(&lq, q);

    size_t normShape[] = {3};
    layer_t *layer = layerNormLayerInit(
        &(layerNormInit_t){.normalizedShape = normShape, .numNormDims = 1, .eps = 1e-5f}, &lq);

    float gammaData[3] = {2.f, 3.f, 4.f};
    float betaData[3] = {-1.f, -2.f, -3.f};
    layerLoadWeights(layer, gammaData, betaData);

    layerNormConfig_t *cfg = layer->config->layerNorm;
    float *loadedGamma = (float *)cfg->gamma->param->data;
    float *loadedBeta = (float *)cfg->beta->param->data;

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(gammaData, loadedGamma, 3);
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(betaData, loadedBeta, 3);

    freeLayerNormLayer(layer);
    freeQuantization(q);
}

void testLayerLoadWeightsLayerNormQuantizesForSymStorage(void) {
    size_t normShape[] = {3};
    layerNormInit_t init = {.normalizedShape = normShape, .numNormDims = 1, .eps = 1e-5f};

    quantization_t *symQ = quantizationInitSymInt32(HALF_AWAY);
    quantization_t *bwd = quantizationInitFloat();
    layerQuant_t lq = {
        .forwardMath = symQ, .backwardMath = bwd, .weightStorage = symQ, .biasStorage = symQ};
    layer_t *layer = layerNormLayerInit(&init, &lq);

    float gammaData[3] = {2.f, 3.f, 4.f};
    float betaData[3] = {-1.f, -2.f, -3.f};
    layerLoadWeights(layer, gammaData, betaData);

    layerNormConfig_t *cfg = layer->config->layerNorm;
    int32_t g[3];
    int32_t b[3];
    for (size_t i = 0; i < 3; i++) {
        g[i] = ((int32_t *)cfg->gamma->param->data)[i];
        b[i] = ((int32_t *)cfg->beta->param->data)[i];
    }
    float gScale = ((symInt32QConfig_t *)cfg->gamma->param->quantization->qConfig)->scale;
    float bScale = ((symInt32QConfig_t *)cfg->beta->param->quantization->qConfig)->scale;

    freeLayerNormLayer(layer);
    freeQuantization(bwd);
    freeQuantization(symQ);

    /* gamma absmax 4 -> scale 4/32767; mantissas round(v*32767/4):
     * 16383.5 -> 16384 (half-away-from-zero: the framework's roundHalfAway is C
     * round() — see #188), 24575.25 -> 24575, 32767 exact. INT_WITHIN(1)
     * on the non-absmax elements (float division may land a hair off the
     * exact midpoint); exact on the absmax element. */
    TEST_ASSERT_FLOAT_WITHIN(1e-9f, 4.0f / 32767.0f, gScale);
    TEST_ASSERT_INT_WITHIN(1, 16384, g[0]);
    TEST_ASSERT_INT_WITHIN(1, 24575, g[1]);
    TEST_ASSERT_EQUAL_INT(32767, g[2]);
    /* beta absmax 3 -> scale 3/32767; round(-10922.33) = -10922,
     * round(-21844.67) = -21845, -32767 exact. */
    TEST_ASSERT_FLOAT_WITHIN(1e-9f, 3.0f / 32767.0f, bScale);
    TEST_ASSERT_INT_WITHIN(1, -10922, b[0]);
    TEST_ASSERT_INT_WITHIN(1, -21845, b[1]);
    TEST_ASSERT_EQUAL_INT(-32767, b[2]);
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(testLayerLoadWeightsLinearOverwritesWeightAndBiasTensors);
    RUN_TEST(testLayerLoadWeightsLinearNoBiasAcceptsNullBiasData);
    RUN_TEST(testLayerLoadWeightsConv1dOverwritesWeightAndBiasTensors);
    RUN_TEST(testLayerLoadWeightsConv1dNoBiasAcceptsNullBiasData);
    RUN_TEST(testLayerLoadWeightsConv1dTransposedOverwritesWeightAndBiasTensors);
    RUN_TEST(testLayerLoadWeightsLayerNormOverwritesGammaAndBeta);
    RUN_TEST(testLayerLoadWeightsLayerNormQuantizesForSymStorage);
    return UNITY_END();
}
