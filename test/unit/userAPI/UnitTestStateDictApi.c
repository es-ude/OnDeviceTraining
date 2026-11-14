#define SOURCE_FILE "UNIT_TEST_STATE_DICT_API"

#include "DeathTest.h"
#include "LayerCommon.h"
#include "LayerNorm.h"
#include "LayerNormApi.h"
#include "LayerQuant.h"
#include "Linear.h"
#include "LinearApi.h"
#include "QuantizationApi.h"
#include "RNG.h"
#include "ReluApi.h"
#include "StateDictApi.h"
#include "TensorApi.h"
#include "unity.h"

void setUp() {}
void tearDown() {}

void testModelLoadStateDictLoadsTwoLinearLayersSkippingRelu(void) {
    quantization_t *q = quantizationInitFloat();
    layerQuant_t lq;
    layerQuantInitUniform(&lq, q);

    layer_t *model[3];
    const rngKey_t initKey0 = rngFold(rngKeyFromSeed(1u), RNG_TAG_INIT, 0u);
    const rngKey_t initKey2 = rngFold(rngKeyFromSeed(1u), RNG_TAG_INIT, 2u);
    model[0] = linearLayerInit(
        &(linearInit_t){
            .inFeatures = 3, .outFeatures = 2, .bias = BIAS_TRUE, .weightInit.key = &initKey0},
        &lq);
    model[1] = reluLayerInit(&lq);
    model[2] = linearLayerInit(
        &(linearInit_t){
            .inFeatures = 2, .outFeatures = 1, .bias = BIAS_TRUE, .weightInit.key = &initKey2},
        &lq);

    float w0[6] = {1, 2, 3, 4, 5, 6};
    float b0[2] = {10, 20};
    float w1[2] = {7, 8};
    float b1[1] = {99};

    modelLoadStateDict(&MODEL_CHAIN(model, 3),
                       (stateDictEntry_t[]){
                           {.name = "fc1", .weightData = w0, .biasData = b0},
                           {.name = "fc2", .weightData = w1, .biasData = b1},
                       },
                       2);

    linearConfig_t *cfg0 = model[0]->config->linear;
    linearConfig_t *cfg2 = model[2]->config->linear;
    float *fc1Weights = (float *)cfg0->weights->param->data;
    float *fc1Bias = (float *)cfg0->bias->param->data;
    float *fc2Weights = (float *)cfg2->weights->param->data;
    float *fc2Bias = (float *)cfg2->bias->param->data;

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(w0, fc1Weights, 6);
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(b0, fc1Bias, 2);
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(w1, fc2Weights, 2);
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(b1, fc2Bias, 1);

    freeLinearLayer(model[0]);
    freeReluLayer(model[1]);
    freeLinearLayer(model[2]);
}

static layer_t *makeLinear(size_t in, size_t out, uint32_t keyIndex, layerQuant_t *lq) {
    const rngKey_t initKey = rngFold(rngKeyFromSeed(1u), RNG_TAG_INIT, keyIndex);
    return linearLayerInit(
        &(linearInit_t){
            .inFeatures = in, .outFeatures = out, .bias = BIAS_TRUE, .weightInit.key = &initKey},
        lq);
}

void testModelLoadStateDictRejectsEntryCountMismatch(void) {
    quantization_t *q = quantizationInitFloat();
    layerQuant_t lq;
    layerQuantInitUniform(&lq, q);
    layer_t *model[] = {makeLinear(3, 2, 0u, &lq), reluLayerInit(&lq), makeLinear(2, 1, 2u, &lq)};
    float w[6] = {0};
    float b[2] = {0};
    stateDictEntry_t entries[] = {{.name = "fc1", .weightData = w, .biasData = b}};

    ASSERT_EXITS_WITH_OUTPUT(1, "model has 2 param layers but 1 entries provided",
                             modelLoadStateDict(&MODEL_CHAIN(model, 3), entries, 1));

    freeLinearLayer(model[0]);
    freeReluLayer(model[1]);
    freeLinearLayer(model[2]);
    freeQuantization(q);
}

/* A bad entry is reported by its name. */
void testModelLoadStateDictRejectsEntryWithoutWeightData(void) {
    quantization_t *q = quantizationInitFloat();
    layerQuant_t lq;
    layerQuantInitUniform(&lq, q);
    layer_t *model[] = {makeLinear(3, 2, 0u, &lq), makeLinear(2, 1, 2u, &lq)};
    float w0[6] = {0};
    float b0[2] = {0};
    float b1[1] = {0};
    stateDictEntry_t entries[] = {{.name = "fc1", .weightData = w0, .biasData = b0},
                                  {.name = "fc2", .weightData = NULL, .biasData = b1}};

    ASSERT_EXITS_WITH_OUTPUT(1, "entry 'fc2' (#1): weightData is NULL",
                             modelLoadStateDict(&MODEL_CHAIN(model, 2), entries, 2));

    freeLinearLayer(model[0]);
    freeLinearLayer(model[1]);
    freeQuantization(q);
}

void testModelLoadStateDictTreatsLayerNormAsParameterLayer(void) {
    quantization_t *q = quantizationInitFloat();
    layerQuant_t lq;
    layerQuantInitUniform(&lq, q);

    size_t normShape[] = {2};
    layer_t *ln = layerNormLayerInit(
        &(layerNormInit_t){.normalizedShape = normShape, .numNormDims = 1, .eps = 1e-5f}, &lq);
    layer_t *model[] = {ln};

    float gammaData[2] = {5.f, 6.f};
    float betaData[2] = {7.f, 8.f};
    stateDictEntry_t entries[] = {{.name = "ln", .weightData = gammaData, .biasData = betaData}};

    /* If LAYERNORM were param-less, numEntries(1) != paramLayers(0) -> exit(1). */
    modelLoadStateDict(&MODEL_CHAIN(model, 1), entries, 1);

    layerNormConfig_t *cfg = ln->config->layerNorm;
    float *loadedGamma = (float *)cfg->gamma->param->data;
    float *loadedBeta = (float *)cfg->beta->param->data;

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(gammaData, loadedGamma, 2);
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(betaData, loadedBeta, 2);

    freeLayerNormLayer(ln);
    freeQuantization(q);
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(testModelLoadStateDictLoadsTwoLinearLayersSkippingRelu);
    RUN_TEST(testModelLoadStateDictRejectsEntryCountMismatch);
    RUN_TEST(testModelLoadStateDictRejectsEntryWithoutWeightData);
    RUN_TEST(testModelLoadStateDictTreatsLayerNormAsParameterLayer);
    return UNITY_END();
}
