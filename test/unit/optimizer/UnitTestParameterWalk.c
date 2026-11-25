#define SOURCE_FILE "UNIT_TEST_PARAMETER_WALK"

#include <stddef.h>

#include "BatchNorm1dApi.h"
#include "Conv1d.h"
#include "Conv1dApi.h"
#include "LayerNormApi.h"
#include "LayerQuant.h"
#include "Linear.h"
#include "LinearApi.h"
#include "Model.h"
#include "Optimizer.h"
#include "OptimizerApi.h"
#include "ParameterWalk.h"
#include "QuantizationApi.h"
#include "RNG.h"
#include "ReluApi.h"
#include "unity.h"

#define NUM_LAYERS 7u
#define MAX_VISITS 16u

static quantization_t *g_q;
static layerQuant_t g_lq;

void setUp(void) {}
void tearDown(void) {}

typedef struct visitLog {
    size_t count;
    paramRef_t refs[MAX_VISITS];
} visitLog_t;

static void recordVisit(const paramRef_t *ref, void *ctx) {
    visitLog_t *log = ctx;
    TEST_ASSERT_LESS_THAN_size_t(MAX_VISITS, log->count);
    log->refs[log->count++] = *ref;
}

/* Linear(bias), Conv1d(no bias), BN(affine), BN(non-affine), LayerNorm, frozen Linear, Relu:
 * one of every case the walk must tell apart. */
static void buildMixedModel(layer_t **layers) {
    const rngKey_t key = rngKeyFromSeed(7u);
    static size_t normShape[] = {4};
    layers[0] = linearLayerInit(
        &(linearInit_t){.inFeatures = 3, .outFeatures = 4, .weightInit.key = &key}, &g_lq);
    layers[1] = conv1dLayerInit(&(conv1dInit_t){.inChannels = 4,
                                                .outChannels = 4,
                                                .kernelSize = 3,
                                                .bias = BIAS_FALSE,
                                                .weightInit.key = &key},
                                &g_lq);
    layers[2] = batchNorm1dLayerInit(&(batchNorm1dInit_t){.numChannels = 4}, &g_lq);
    layers[3] =
        batchNorm1dLayerInit(&(batchNorm1dInit_t){.numChannels = 4, .noAffine = true}, &g_lq);
    layers[4] = layerNormLayerInit(
        &(layerNormInit_t){.normalizedShape = normShape, .numNormDims = 1}, &g_lq);
    layers[5] = linearLayerInit(&(linearInit_t){.inFeatures = 4,
                                                .outFeatures = 2,
                                                .weightInit.key = &key,
                                                .trainable = TRAINABLE_FALSE},
                                &g_lq);
    layers[6] = reluLayerInit(&g_lq);
}

static void freeMixedModel(layer_t **layers) {
    freeReluLayer(layers[6]);
    freeLinearLayer(layers[5]);
    freeLayerNormLayer(layers[4]);
    freeBatchNorm1dLayer(layers[3]);
    freeBatchNorm1dLayer(layers[2]);
    freeConv1dLayer(layers[1]);
    freeLinearLayer(layers[0]);
}

void testWalkVisitsInOptimizerOrderWithRoles(void) {
    layer_t *layers[NUM_LAYERS];
    buildMixedModel(layers);
    visitLog_t log = {0};

    forEachTrainableParameter(&MODEL_CHAIN(layers, NUM_LAYERS), recordVisit, &log);

    const struct {
        size_t layerIndex;
        paramRole_t role;
        parameter_t *parameter;
    } expected[] = {
        {0, PARAM_ROLE_WEIGHT, layers[0]->config->linear->weights},
        {0, PARAM_ROLE_BIAS, layers[0]->config->linear->bias},
        {1, PARAM_ROLE_WEIGHT, layers[1]->config->conv1d->weights},
        {2, PARAM_ROLE_NORM_SCALE, layers[2]->config->batchNorm1d->gamma},
        {2, PARAM_ROLE_NORM_SHIFT, layers[2]->config->batchNorm1d->beta},
        {4, PARAM_ROLE_NORM_SCALE, layers[4]->config->layerNorm->gamma},
        {4, PARAM_ROLE_NORM_SHIFT, layers[4]->config->layerNorm->beta},
    };
    const size_t numExpected = sizeof(expected) / sizeof(expected[0]);

    TEST_ASSERT_EQUAL_size_t(numExpected, log.count);
    for (size_t i = 0; i < numExpected; i++) {
        TEST_ASSERT_EQUAL_size_t(expected[i].layerIndex, log.refs[i].layerIndex);
        TEST_ASSERT_EQUAL_PTR(layers[expected[i].layerIndex], log.refs[i].layer);
        TEST_ASSERT_EQUAL_INT(expected[i].role, log.refs[i].role);
        TEST_ASSERT_NOT_NULL(log.refs[i].parameter);
        TEST_ASSERT_EQUAL_PTR(expected[i].parameter, log.refs[i].parameter);
    }
    freeMixedModel(layers);
}

void testWalkCountMatchesCalcTotalNumberOfStates(void) {
    layer_t *layers[NUM_LAYERS];
    buildMixedModel(layers);
    const model_t *model = &MODEL_CHAIN(layers, NUM_LAYERS);
    visitLog_t log = {0};

    forEachTrainableParameter(model, recordVisit, &log);
    size_t numStates = calcTotalNumberOfStates(model);
    parameter_t *slots[MAX_VISITS] = {0};
    collectTrainableParameters(model, slots);

    TEST_ASSERT_EQUAL_size_t(7, log.count);
    TEST_ASSERT_EQUAL_size_t(7, numStates);
    for (size_t i = 0; i < log.count; i++) {
        TEST_ASSERT_EQUAL_PTR(log.refs[i].parameter, slots[i]);
    }
    freeMixedModel(layers);
}

int main(void) {
    g_q = quantizationInitFloat();
    layerQuantInitUniform(&g_lq, g_q);
    UNITY_BEGIN();
    RUN_TEST(testWalkVisitsInOptimizerOrderWithRoles);
    RUN_TEST(testWalkCountMatchesCalcTotalNumberOfStates);
    return UNITY_END();
}
