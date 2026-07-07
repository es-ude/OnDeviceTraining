#define SOURCE_FILE "UNIT_TEST_BIASLESS_CONV_OPTIM"

#include <stdbool.h>

#include "Conv1d.h"
#include "Conv1dApi.h"
#include "Conv1dTransposed.h"
#include "Conv1dTransposedApi.h"
#include "Layer.h"
#include "LayerCommon.h"
#include "LayerQuant.h"
#include "Linear.h"
#include "LinearApi.h"
#include "Optimizer.h"
#include "QuantizationApi.h"
#include "SgdApi.h"
#include "StorageApi.h"
#include "TensorApi.h"
#include "unity.h"

void setUp(void) {}
void tearDown(void) {}

/* freeOptimSgdM cascades into freeParameter for every registered parameter_t
 * (here: the conv's weights -- its bias slot is NULL and never registered).
 * freeConv1dLayer would free cfg->weights again (double-free), so after
 * freeOptimSgdM the layer must be torn down shell-only: kernel + config +
 * layer wrapper, matching the pattern in UnitTestLayerNormIntegration.c
 * (freeLinearLayerShell / freeLayerNormLayerShell). This layer comes from the
 * Borrowing factory (ownsQuantizations=false), so outputQ/propLossQ are not
 * factory-owned either. */
static void freeConv1dLayerShell(layer_t *layer) {
    freeReservedMemory(layer->config->conv1d->kernel);
    freeReservedMemory(layer->config->conv1d);
    freeReservedMemory(layer->config);
    freeReservedMemory(layer);
}

/* Conv1dTransposed counterpart of freeConv1dLayerShell (same shell-only teardown
 * after freeOptimSgdM has already freed the registered weight parameter_t). */
static void freeConv1dTransposedLayerShell(layer_t *layer) {
    freeReservedMemory(layer->config->conv1dTransposed->kernel);
    freeReservedMemory(layer->config->conv1dTransposed);
    freeReservedMemory(layer->config);
    freeReservedMemory(layer);
}

/* Linear counterpart of freeConv1dLayerShell: freeOptimSgdM already freed the
 * linear's weights parameter_t (its bias slot is NULL and never registered),
 * so this only tears down the config/layer wrapper. Linear has no kernel
 * (unlike Conv1d/Conv1dTransposed), so there is one fewer level to free. */
static void freeLinearLayerShell(layer_t *layer) {
    freeReservedMemory(layer->config->linear);
    freeReservedMemory(layer->config);
    freeReservedMemory(layer);
}

/* A bias-less Conv1d (BIAS_FALSE, header-sanctioned, cfg->bias == NULL) used
 * to make calcNumberOfStatesByLayerType count 2 states unconditionally for
 * CONV1D, then sgdMCreateOptim NULL-derefed conv1dCfg->bias->param while
 * building the (nonexistent) bias momentum state. This test proves both the
 * COUNT (calcTotalNumberOfStates) and the COLLECTION (sgdMCreateOptim +
 * freeOptimSgdM) sites now handle bias == NULL correctly. */
void testOptimizer_OneStateForBiaslessConv1d(void) {
    quantization_t *q = quantizationInitFloat();
    layerQuant_t lq;
    layerQuantInitUniform(&lq, q);

    layer_t *conv = conv1dLayerInit(
        &(conv1dInit_t){
            .inChannels = 2,
            .outChannels = 4,
            .kernelSize = 3,
            .bias = BIAS_FALSE,
        },
        &lq);
    layer_t *model[1] = {conv};

    size_t numStates = calcTotalNumberOfStates(model, 1);

    quantization_t *momentumQuant = quantizationInitFloat();
    optimizer_t *optim = sgdMCreateOptim(0.1f, 0.0f, 0.0f, model, 1, FLOAT32, momentumQuant);
    bool optimNotNull = (optim != NULL);

    freeOptimSgdM(optim); /* frees the conv's weights parameter_t */
    freeConv1dLayerShell(conv);
    freeQuantization(momentumQuant);
    freeQuantization(q);

    TEST_ASSERT_EQUAL_UINT(1, numStates);
    TEST_ASSERT_TRUE(optimNotNull);
}

/* Companion coverage for the WEIGHT+BIAS path (bias != NULL): a bias-present
 * Conv1d must count 2 states, and sgdMCreateOptim must build BOTH the weight and
 * bias momentum states (the collection site the bias-less test cannot exercise).
 * freeOptimSgdM then frees both registered parameter_t (weights + bias), so the
 * layer is torn down shell-only afterwards. */
void testOptimizer_TwoStatesForBiasedConv1d(void) {
    quantization_t *q = quantizationInitFloat();
    layerQuant_t lq;
    layerQuantInitUniform(&lq, q); /* sets biasStorage so the bias tensor allocates */

    layer_t *conv = conv1dLayerInit(
        &(conv1dInit_t){
            .inChannels = 2,
            .outChannels = 4,
            .kernelSize = 3,
            .bias = BIAS_TRUE,
        },
        &lq);
    layer_t *model[1] = {conv};

    size_t numStates = calcTotalNumberOfStates(model, 1);

    quantization_t *momentumQuant = quantizationInitFloat();
    optimizer_t *optim = sgdMCreateOptim(0.1f, 0.0f, 0.0f, model, 1, FLOAT32, momentumQuant);
    bool optimNotNull = (optim != NULL);

    freeOptimSgdM(optim); /* frees the conv's weights AND bias parameter_t */
    freeConv1dLayerShell(conv);
    freeQuantization(momentumQuant);
    freeQuantization(q);

    TEST_ASSERT_EQUAL_UINT(2, numStates);
    TEST_ASSERT_TRUE(optimNotNull);
}

/* Same bias-less state-count/collection guard as the Conv1d test, but for a
 * CONV1D_TRANSPOSED layer (BIAS_FALSE): calcNumberOfStatesByLayer's separate
 * CONV1D_TRANSPOSED arm must count 1, and sgdMCreateOptim must collect only the
 * weight momentum state (no NULL bias deref). */
void testOptimizer_OneStateForBiaslessConv1dTransposed(void) {
    quantization_t *q = quantizationInitFloat();
    layerQuant_t lq;
    layerQuantInitUniform(&lq, q);

    layer_t *convT = conv1dTransposedLayerInit(
        &(conv1dTransposedInit_t){
            .inChannels = 2,
            .outChannels = 4,
            .kernelSize = 3,
            .bias = BIAS_FALSE,
        },
        &lq);
    layer_t *model[1] = {convT};

    size_t numStates = calcTotalNumberOfStates(model, 1);

    quantization_t *momentumQuant = quantizationInitFloat();
    optimizer_t *optim = sgdMCreateOptim(0.1f, 0.0f, 0.0f, model, 1, FLOAT32, momentumQuant);
    bool optimNotNull = (optim != NULL);

    freeOptimSgdM(optim); /* frees the convT's weights parameter_t (bias slot NULL) */
    freeConv1dTransposedLayerShell(convT);
    freeQuantization(momentumQuant);
    freeQuantization(q);

    TEST_ASSERT_EQUAL_UINT(1, numStates);
    TEST_ASSERT_TRUE(optimNotNull);
}

/* Same bias-less state-count/collection guard as the Conv1d test, but for a
 * LINEAR layer (BIAS_FALSE, header-sanctioned, cfg->bias == NULL): the
 * CASE LINEAR arm in calcNumberOfStatesByLayer must count 1 (not the fixed
 * 2), and sgdMCreateOptim's LINEAR collection block must skip the bias
 * momentum state instead of NULL-dereferencing linearConfig->bias->param
 * (issue #292, mirrors the CONV1D fix from PR #291). */
void testBiaslessLinearCountsOneStateAndOptimizerSurvives(void) {
    quantization_t *q = quantizationInitFloat();
    layerQuant_t lq;
    layerQuantInitUniform(&lq, q);

    layer_t *linear = linearLayerInit(
        &(linearInit_t){
            .inFeatures = 3,
            .outFeatures = 2,
            .bias = BIAS_FALSE,
        },
        &lq);
    layer_t *model[1] = {linear};

    size_t numStates = calcTotalNumberOfStates(model, 1);

    quantization_t *momentumQuant = quantizationInitFloat();
    optimizer_t *optim = sgdMCreateOptim(0.1f, 0.0f, 0.0f, model, 1, FLOAT32, momentumQuant);
    bool optimNotNull = (optim != NULL);

    freeOptimSgdM(optim); /* frees the linear's weights parameter_t */
    freeLinearLayerShell(linear);
    freeQuantization(momentumQuant);
    freeQuantization(q);

    TEST_ASSERT_EQUAL_UINT(1, numStates);
    TEST_ASSERT_TRUE(optimNotNull);
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(testOptimizer_OneStateForBiaslessConv1d);
    RUN_TEST(testOptimizer_TwoStatesForBiasedConv1d);
    RUN_TEST(testOptimizer_OneStateForBiaslessConv1dTransposed);
    RUN_TEST(testBiaslessLinearCountsOneStateAndOptimizerSurvives);
    return UNITY_END();
}
