#define SOURCE_FILE "UNIT_TEST_BATCHNORM1D_INTEGRATION"

#include <stdbool.h>
#include <stdint.h>
#include <string.h>

#include "BatchNorm1d.h"
#include "BatchNorm1dApi.h"
#include "DeathTest.h"
#include "LayerCommon.h"
#include "LayerConfigAccess.h"
#include "LayerQuant.h"
#include "LayerWeightsApi.h"
#include "OptimizerApi.h"
#include "QuantizationApi.h"
#include "StateDictApi.h"
#include "StorageApi.h"
#include "Tensor.h"
#include "TensorApi.h"
#include "unity.h"

static quantization_t *g_q;
static layerQuant_t g_lq;

void setUp(void) {}
void tearDown(void) {}

static layer_t *bnLayer(size_t C, bool noAffine, bool noRunningStats, trainable_t trainable) {
    return batchNorm1dLayerInit(&(batchNorm1dInit_t){.numChannels = C,
                                                     .noAffine = noAffine,
                                                     .noRunningStats = noRunningStats,
                                                     .trainable = trainable},
                                &g_lq);
}

void testOptimizerCountsAndCollectsGammaBetaOnlyWhenAffineAndTrainable(void) {
    layer_t *affine = bnLayer(3, false, false, TRAINABLE_DEFAULT);
    layer_t *noAffine = bnLayer(3, true, false, TRAINABLE_DEFAULT);
    layer_t *frozen = bnLayer(3, false, false, TRAINABLE_FALSE);
    layer_t *model[] = {affine, noAffine, frozen};
    size_t n = calcTotalNumberOfStates(model, 3);
    parameter_t *slots[2] = {NULL, NULL};
    collectTrainableParameters(model, 3, slots);
    bool order = slots[0] == affine->config->batchNorm1d->gamma &&
                 slots[1] == affine->config->batchNorm1d->beta;
    freeBatchNorm1dLayer(frozen);
    freeBatchNorm1dLayer(noAffine);
    freeBatchNorm1dLayer(affine);
    TEST_ASSERT_EQUAL_size_t(2, n);
    TEST_ASSERT_TRUE(order);
}

void testLayerIsFrozenFollowsTrainable(void) {
    layer_t *t = bnLayer(2, false, false, TRAINABLE_DEFAULT);
    layer_t *f = bnLayer(2, false, false, TRAINABLE_FALSE);
    bool tf = layerIsFrozen(t);
    bool ff = layerIsFrozen(f);
    freeBatchNorm1dLayer(f);
    freeBatchNorm1dLayer(t);
    TEST_ASSERT_FALSE(tf);
    TEST_ASSERT_TRUE(ff);
}

void testLayerConfigAccessArms(void) {
    layer_t *bn = bnLayer(2, false, false, TRAINABLE_DEFAULT);
    layer_t *bare = bnLayer(2, true, true, TRAINABLE_DEFAULT);
    bool slots = layerOutputQ(bn) == g_q && backwardWireQ(bn) == g_q &&
                 layerForwardMath(bn).type == ARITH_FLOAT32;
    const char *fieldBn = layerNonFloat32Field(bn);
    const char *fieldBare = layerNonFloat32Field(bare);
    bn->config->batchNorm1d->forwardMath =
        (arithmetic_t){.type = ARITH_SYM_INT32, .roundingMode = HALF_AWAY};
    const char *fieldSym = layerNonFloat32Field(bn);
    bn->config->batchNorm1d->forwardMath =
        (arithmetic_t){.type = ARITH_FLOAT32, .roundingMode = HALF_AWAY};
    freeBatchNorm1dLayer(bare);
    freeBatchNorm1dLayer(bn);
    TEST_ASSERT_TRUE(slots);
    TEST_ASSERT_NULL(fieldBn);
    TEST_ASSERT_NULL(fieldBare);
    TEST_ASSERT_EQUAL_STRING("forwardMath", fieldSym);
}

void testLayerLoadWeightsFillsGammaBeta(void) {
    layer_t *bn = bnLayer(2, false, false, TRAINABLE_DEFAULT);
    float g[2] = {1.5f, -2.0f};
    float b[2] = {0.25f, 3.0f};
    layerLoadWeights(bn, g, b);
    float g1 = ((float *)bn->config->batchNorm1d->gamma->param->data)[1];
    float b1 = ((float *)bn->config->batchNorm1d->beta->param->data)[1];
    freeBatchNorm1dLayer(bn);
    TEST_ASSERT_EQUAL_FLOAT(-2.0f, g1);
    TEST_ASSERT_EQUAL_FLOAT(3.0f, b1);
}

static void loadWeightsIntoNoAffine(void) {
    layer_t *bn = bnLayer(2, true, false, TRAINABLE_DEFAULT);
    float g[2] = {1.f, 1.f};
    float b[2] = {0.f, 0.f};
    layerLoadWeights(bn, g, b);
}

void testLayerLoadWeightsRejectsNoAffine(void) {
    ASSERT_EXITS_WITH_FAILURE(loadWeightsIntoNoAffine());
}

/* layerHasParameters == affine: a no-affine BN takes no StateDict entry. */
void testStateDictSkipsNoAffineBatchNorm(void) {
    layer_t *a = bnLayer(2, true, false, TRAINABLE_DEFAULT);
    layer_t *b = bnLayer(2, false, false, TRAINABLE_DEFAULT);
    layer_t *model[] = {a, b};
    float g[2] = {2.f, 3.f};
    float be[2] = {4.f, 5.f};
    modelLoadStateDict(model, 2, (stateDictEntry_t[]){{.weightData = g, .biasData = be}}, 1);
    float g0 = ((float *)b->config->batchNorm1d->gamma->param->data)[0];
    freeBatchNorm1dLayer(b);
    freeBatchNorm1dLayer(a);
    TEST_ASSERT_EQUAL_FLOAT(2.0f, g0);
}

int main(void) {
    g_q = quantizationInitFloat();
    layerQuantInitUniform(&g_lq, g_q);
    UNITY_BEGIN();
    RUN_TEST(testOptimizerCountsAndCollectsGammaBetaOnlyWhenAffineAndTrainable);
    RUN_TEST(testLayerIsFrozenFollowsTrainable);
    RUN_TEST(testLayerConfigAccessArms);
    RUN_TEST(testLayerLoadWeightsFillsGammaBeta);
    RUN_TEST(testLayerLoadWeightsRejectsNoAffine);
    RUN_TEST(testStateDictSkipsNoAffineBatchNorm);
    int rc = UNITY_END();
    freeQuantization(g_q);
    return rc;
}
