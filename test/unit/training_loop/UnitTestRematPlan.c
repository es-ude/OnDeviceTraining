#define SOURCE_FILE "UNIT_TEST_REMAT_PLAN"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "Common.h"
#include "DeathTest.h"
#include "Layer.h"
#include "LayerQuant.h"
#include "LinearApi.h"
#include "LossFunction.h"
#include "Quantization.h"
#include "ReluApi.h"
#include "RematPlan.h"
#include "SoftmaxApi.h"
#include "StorageApi.h"
#include "Tensor.h"
#include "unity.h"

void setUp(void) {}
void tearDown(void) {}

/* Fixture layers borrow their wire templates, so one FLOAT32 template outlives
 * every fixture model. Tests that edit a template use a local one. */
static quantization_t g_floatQ = {.type = FLOAT32, .qConfig = NULL};

static layer_t *makeLinear(size_t in, size_t out, bool frozen) {
    layerQuant_t lq;
    layerQuantInitUniform(&lq, &g_floatQ);
    return linearLayerInit(
        &(linearInit_t){.inFeatures = in,
                        .outFeatures = out,
                        .trainable = frozen ? TRAINABLE_FALSE : TRAINABLE_DEFAULT},
        &lq);
}

static layer_t *makeRelu(quantization_t *q) {
    return reluLayerInit(&(layerQuant_t){.outputQ = q, .propLossQ = q});
}

static layer_t *makeSoftmax(void) {
    layerQuant_t lq;
    layerQuantInitUniform(&lq, &g_floatQ);
    return softmaxLayerInit(&lq);
}

static void freeModel(layer_t **model, size_t n) {
    for (size_t i = 0; i < n; i++) {
        switch (model[i]->type) {
        case LINEAR:
            freeLinearLayer(model[i]);
            break;
        case RELU:
            freeReluLayer(model[i]);
            break;
        case SOFTMAX:
            freeSoftmaxLayer(model[i]);
            break;
        default:
            TEST_FAIL_MESSAGE("freeModel: extend the switch for this layer type");
        }
    }
}

/* ---- rematBackwardRange (spec §4.3, §12.1) ---- */

void testBackwardRangeMseRunsFromLastLayerToDeepest(void) {
    layer_t *model[3] = {makeLinear(2, 4, false), makeRelu(&g_floatQ), makeLinear(4, 2, false)};
    size_t deepest = 99;
    ptrdiff_t top = 99;
    rematBackwardRange(model, 3, MSE, &deepest, &top);
    TEST_ASSERT_EQUAL_size_t(0, deepest);
    TEST_ASSERT_EQUAL_INT(2, (int)top);
    freeModel(model, 3);
}

void testBackwardRangeCrossEntropySkipsTheLastLayerPositionally(void) {
    layer_t *model[2] = {makeLinear(2, 3, false), makeSoftmax()};
    size_t deepest = 99;
    ptrdiff_t top = 99;
    rematBackwardRange(model, 2, CROSS_ENTROPY, &deepest, &top);
    TEST_ASSERT_EQUAL_size_t(0, deepest);
    TEST_ASSERT_EQUAL_INT(0, (int)top);
    freeModel(model, 2);
}

void testBackwardRangeTruncatesAtDeepestTrainable(void) {
    layer_t *model[3] = {makeLinear(2, 4, true), makeRelu(&g_floatQ), makeLinear(4, 2, false)};
    size_t deepest = 99;
    ptrdiff_t top = 99;
    rematBackwardRange(model, 3, MSE, &deepest, &top);
    TEST_ASSERT_EQUAL_size_t(2, deepest);
    TEST_ASSERT_EQUAL_INT(2, (int)top);
    freeModel(model, 3);
}

void testBackwardRangeAllFrozenReturnsModelSize(void) {
    layer_t *model[2] = {makeLinear(2, 4, true), makeRelu(&g_floatQ)};
    size_t deepest = 99;
    ptrdiff_t top = 99;
    rematBackwardRange(model, 2, MSE, &deepest, &top);
    TEST_ASSERT_EQUAL_size_t(2, deepest);
    TEST_ASSERT_EQUAL_INT(1, (int)top);
    freeModel(model, 2);
}

/* D20: n = 1 under CE keeps today's signed top = -1. */
void testBackwardRangeSingleLayerUnderCrossEntropyIsMinusOne(void) {
    layer_t *model[1] = {makeLinear(2, 3, false)};
    size_t deepest = 99;
    ptrdiff_t top = 99;
    rematBackwardRange(model, 1, CROSS_ENTROPY, &deepest, &top);
    TEST_ASSERT_EQUAL_size_t(0, deepest);
    TEST_ASSERT_EQUAL_INT(-1, (int)top);
    freeModel(model, 1);
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(testBackwardRangeMseRunsFromLastLayerToDeepest);
    RUN_TEST(testBackwardRangeCrossEntropySkipsTheLastLayerPositionally);
    RUN_TEST(testBackwardRangeTruncatesAtDeepestTrainable);
    RUN_TEST(testBackwardRangeAllFrozenReturnsModelSize);
    RUN_TEST(testBackwardRangeSingleLayerUnderCrossEntropyIsMinusOne);
    return UNITY_END();
}
