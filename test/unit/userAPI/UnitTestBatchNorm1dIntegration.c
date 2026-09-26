#define SOURCE_FILE "UNIT_TEST_BATCHNORM1D_INTEGRATION"

#include <stdbool.h>
#include <stdint.h>
#include <string.h>

#include "BatchNorm1d.h"
#include "BatchNorm1dApi.h"
#include "BatchView.h"
#include "CalculateGradsSequential.h"
#include "DataLoaderApi.h"
#include "Dataset.h"
#include "DeathTest.h"
#include "InferenceApi.h"
#include "LayerCommon.h"
#include "LayerConfigAccess.h"
#include "LayerQuant.h"
#include "LayerWeightsApi.h"
#include "Linear.h"
#include "LinearApi.h"
#include "LossFunction.h"
#include "OptimizerApi.h"
#include "QuantizationApi.h"
#include "StateDictApi.h"
#include "StorageApi.h"
#include "Tensor.h"
#include "TensorApi.h"
#include "TrainingBatchDefault.h"
#include "TrainingLoopApi.h"
#include "expected_bn_cadence.h"
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

static tensor_t *buildFloatTensor(const size_t *dims, size_t rank, const float *src) {
    size_t *d = reserveMemory(rank * sizeof(size_t));
    for (size_t i = 0; i < rank; i++) {
        d[i] = dims[i];
    }
    size_t *order = reserveMemory(rank * sizeof(size_t));
    setOrderOfDimsForNewTensor(rank, order);
    shape_t *shape = reserveMemory(sizeof(shape_t));
    setShape(shape, d, rank, order);
    tensor_t *t = initTensor(shape, quantizationInitFloat(), NULL);
    if (src != NULL) {
        tensorFillFromFloatBuffer(t, (float *)src, calcNumberOfElementsByTensor(t));
    }
    return t;
}

static const float kX[8] = {0.3f, -1.2f, 0.8f, 2.1f, -0.4f, 1.7f, -0.9f, 0.05f}; /* [4, 2] */
static const float kY[8] = {0.1f, 0.4f, -0.3f, 0.9f, 0.2f, -0.7f, 0.6f, -0.2f};

/* [BN(2) -> Linear(2->2)] with deterministic Linear weights. */
static void buildBnLinear(layer_t **model, trainable_t bnTrainable) {
    model[0] = bnLayer(2, false, false, bnTrainable);
    model[1] = linearLayerInit(&(linearInit_t){.inFeatures = 2, .outFeatures = 2}, &g_lq);
    float w[4] = {0.5f, -0.25f, 0.75f, 1.0f};
    float b[2] = {0.1f, -0.1f};
    layerLoadWeights(model[1], w, b);
}

static void freeBnLinear(layer_t **model) {
    freeLinearLayer(model[1]);
    freeBatchNorm1dLayer(model[0]);
}

static void runGrads(layer_t **model, size_t n) {
    tensor_t *x = buildFloatTensor((size_t[]){4, 2}, 2, kX);
    tensor_t *y = buildFloatTensor((size_t[]){4, 2}, 2, kY);
    freeTrainingStats(
        calculateGradsSequential(model, n, defaultLossConfig(MSE), REDUCTION_MEAN, x, y));
    freeTensor(y);
    freeTensor(x);
}

void testGradsCallTrainsBatchNormThenLeavesEvalMode(void) {
    layer_t *model[2];
    buildBnLinear(model, TRAINABLE_DEFAULT);
    runGrads(model, 2);
    batchNorm1dConfig_t *c = model[0]->config->batchNorm1d;
    uint64_t nbt = c->numBatchesTracked;
    bool training = c->training;
    float rm0 = ((float *)c->runningMean->data)[0];
    float dg = ((float *)c->gamma->grad->data)[0];
    freeBnLinear(model);
    TEST_ASSERT_EQUAL_UINT64(1, nbt);
    TEST_ASSERT_FALSE(training);
    TEST_ASSERT_TRUE(rm0 != 0.0f);
    TEST_ASSERT_TRUE(dg != 0.0f); /* BN is the deepest trainable: grads-only call */
}

void testInferenceUsesRunningStatsAndWritesNothing(void) {
    layer_t *model[2];
    buildBnLinear(model, TRAINABLE_DEFAULT);
    runGrads(model, 2);
    batchNorm1dConfig_t *c = model[0]->config->batchNorm1d;
    float before[4];
    memcpy(before, c->runningMean->data, 2 * sizeof(float));
    memcpy(before + 2, c->runningVar->data, 2 * sizeof(float));
    tensor_t *x = buildFloatTensor((size_t[]){1, 2}, 2, kX); /* [1, 2]: fine in eval */
    tensor_t *out = inference(model, 2, x);
    float after[4];
    memcpy(after, c->runningMean->data, 2 * sizeof(float));
    memcpy(after + 2, c->runningVar->data, 2 * sizeof(float));
    uint64_t nbt = c->numBatchesTracked;
    freeTensor(out);
    freeTensor(x);
    freeBnLinear(model);
    TEST_ASSERT_EQUAL_MEMORY(before, after, sizeof before);
    TEST_ASSERT_EQUAL_UINT64(1, nbt);
}

void testFrozenBatchNormNeverMovesDuringTraining(void) {
    layer_t *model[2];
    buildBnLinear(model, TRAINABLE_FALSE);
    runGrads(model, 2);
    batchNorm1dConfig_t *c = model[0]->config->batchNorm1d;
    float rm0 = ((float *)c->runningMean->data)[0];
    float rv0 = ((float *)c->runningVar->data)[0];
    uint64_t nbt = c->numBatchesTracked;
    float linGrad = ((float *)model[1]->config->linear->weights->grad->data)[0];
    freeBnLinear(model);
    TEST_ASSERT_EQUAL_FLOAT(0.0f, rm0);
    TEST_ASSERT_EQUAL_FLOAT(1.0f, rv0);
    TEST_ASSERT_EQUAL_UINT64(0, nbt);
    TEST_ASSERT_TRUE(linGrad != 0.0f);
}

/* Running stats move even when nothing trains (deepest == modelSize):
 * PyTorch train() parity. */
void testNoAffineBatchNormAloneStillUpdatesRunningStats(void) {
    layer_t *model[1] = {bnLayer(2, true, false, TRAINABLE_DEFAULT)};
    runGrads(model, 1);
    uint64_t nbt = model[0]->config->batchNorm1d->numBatchesTracked;
    freeBatchNorm1dLayer(model[0]);
    TEST_ASSERT_EQUAL_UINT64(1, nbt);
}

/* Review Focus 1: a custom calculateGradsFn_t that does NOT route through
 * calculateGradsSequential/tracedGrads never flips the mode -> BN runs in
 * eval mode (documented contract, TrainingLoopApi.h). */
void testCustomGradsFnWithoutFlipRunsBatchNormInEvalMode(void) {
    layer_t *model[2];
    buildBnLinear(model, TRAINABLE_DEFAULT);
    tensor_t *x = buildFloatTensor((size_t[]){4, 2}, 2, kX);
    tensor_t *out = inference(model, 2, x); /* stands in for a hand-rolled forward */
    uint64_t nbt = model[0]->config->batchNorm1d->numBatchesTracked;
    freeTensor(out);
    freeTensor(x);
    freeBnLinear(model);
    TEST_ASSERT_EQUAL_UINT64(0, nbt);
}

/* Closes the known gap (task-6-brief mutation 2): if the loop's `false` flip
 * moved to before the backward pass instead of after, BN's backward would
 * silently see training=false (running-stats mode) instead of true
 * (batch-stats mode). Catch it without any hand-derived numbers: build a
 * second, identically-initialized BN layer and drive it directly through
 * the same kernels with `training` pinned true across forward AND backward
 * -- that reference must equal what the loop produces for a model whose only
 * layer is that BN (grads-only call, same input/label). */
void testGradsCallKeepsBatchNormInTrainingModeThroughBackward(void) {
    layer_t *loopModel[1] = {bnLayer(2, false, false, TRAINABLE_DEFAULT)};
    runGrads(loopModel, 1);
    float dgLoop = ((float *)loopModel[0]->config->batchNorm1d->gamma->grad->data)[0];
    freeBatchNorm1dLayer(loopModel[0]);

    layer_t *ref = bnLayer(2, false, false, TRAINABLE_DEFAULT);
    batchNorm1dConfig_t *rc = ref->config->batchNorm1d;
    tensor_t *x = buildFloatTensor((size_t[]){4, 2}, 2, kX);
    tensor_t *y = buildFloatTensor((size_t[]){4, 2}, 2, kY);
    tensor_t *out = buildFloatTensor((size_t[]){4, 2}, 2, NULL);
    rc->training = true;
    batchNorm1dForward(ref, x, out);
    tensor_t *dy = buildFloatTensor((size_t[]){4, 2}, 2, NULL);
    lossFunctions[MSE].backward(out, y, dy);
    rc->training = true; /* pinned true through backward: what the loop must match */
    batchNorm1dBackward(ref, x, dy, NULL);
    float dgRef = ((float *)rc->gamma->grad->data)[0];

    freeTensor(dy);
    freeTensor(out);
    freeTensor(y);
    freeTensor(x);
    freeBatchNorm1dLayer(ref);
    TEST_ASSERT_EQUAL_FLOAT(dgRef, dgLoop);
}

/* Ghost-BN cadence through the stacked loop (#460 spec §7 item 6): one
 * macro batch of b = 8 through BN(3) -> Linear(3->2), chunked by m into
 * b/m calculateGradsSequential calls via trainingBatchDefault. Compares
 * running stats + mean-scaled grads (before the optimizer step) against a
 * PyTorch chunked reference (generate_expected_bn_cadence.py). */
#define CAD_B 8
#define CAD_GRADS 14

static batch_t *buildCadBatch(tensor_t **items, tensor_t **labels) {
    batch_t *batch = reserveMemory(sizeof(batch_t));
    batch->samples = reserveMemory(CAD_B * sizeof(sample_t *));
    batch->size = CAD_B;
    for (size_t i = 0; i < CAD_B; i++) {
        sample_t *s = reserveMemory(sizeof(sample_t));
        s->item = items[i];
        s->label = labels[i];
        batch->samples[i] = s;
    }
    return batch;
}

static void runCadence(size_t m, float *rm, float *rv, uint64_t *nbt, float *grads) {
    layer_t *model[2];
    model[0] = bnLayer(3, false, false, TRAINABLE_DEFAULT);
    model[1] = linearLayerInit(&(linearInit_t){.inFeatures = 3, .outFeatures = 2}, &g_lq);
    float g[3], be[3], w[6], b[2];
    memcpy(g, bnCadGamma, sizeof g);
    memcpy(be, bnCadBeta, sizeof be);
    memcpy(w, bnCadLinW, sizeof w);
    memcpy(b, bnCadLinB, sizeof b);
    layerLoadWeights(model[0], g, be);
    layerLoadWeights(model[1], w, b);
    tensor_t *items[CAD_B];
    tensor_t *labels[CAD_B];
    for (size_t s = 0; s < CAD_B; s++) {
        items[s] = buildFloatTensor((size_t[]){3}, 1, bnCadItems + 3 * s); /* natural [3] */
        labels[s] = buildFloatTensor((size_t[]){2}, 1, bnCadLabels + 2 * s);
    }
    batchView_t view;
    float scale = lossFunctions[MSE].computeMeanScale(CAD_B, batchViewOf(&view, labels[0]));
    batch_t *batch = buildCadBatch(items, labels);
    (void)trainingBatchDefault(model, 2, defaultLossConfig(MSE), batch, calculateGradsSequential,
                               REDUCTION_MEAN, m);
    freeBatch(batch);
    batchNorm1dConfig_t *c = model[0]->config->batchNorm1d;
    memcpy(rm, c->runningMean->data, 3 * sizeof(float));
    memcpy(rv, c->runningVar->data, 3 * sizeof(float));
    *nbt = c->numBatchesTracked;
    parameter_t *slots[4];
    TEST_ASSERT_EQUAL_size_t(4, calcTotalNumberOfStates(model, 2));
    collectTrainableParameters(model, 2, slots);
    size_t k = 0;
    for (size_t p = 0; p < 4; p++) {
        size_t n = calcNumberOfElementsByTensor(slots[p]->grad);
        for (size_t i = 0; i < n; i++) {
            grads[k++] = ((float *)slots[p]->grad->data)[i] * scale;
        }
    }
    for (size_t s = 0; s < CAD_B; s++) {
        freeTensor(labels[s]);
        freeTensor(items[s]);
    }
    freeLinearLayer(model[1]);
    freeBatchNorm1dLayer(model[0]);
}

static void assertCadence(size_t m, const float *expRm, const float *expRv, uint64_t expNbt,
                          const float *expGrads) {
    float rm[3], rv[3], grads[CAD_GRADS];
    uint64_t nbt;
    runCadence(m, rm, rv, &nbt, grads);
    TEST_ASSERT_EQUAL_UINT64(expNbt, nbt);
    for (size_t c = 0; c < 3; c++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-5f, expRm[c], rm[c]);
        TEST_ASSERT_FLOAT_WITHIN(1e-5f, expRv[c], rv[c]);
    }
    for (size_t i = 0; i < CAD_GRADS; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-5f, expGrads[i], grads[i]);
    }
}

void testGhostBatchNormCadenceM2(void) {
    assertCadence(2, bnCadRunningMean_m2, bnCadRunningVar_m2, bnCadNbt_m2, bnCadGrads_m2);
}
void testGhostBatchNormCadenceM4(void) {
    assertCadence(4, bnCadRunningMean_m4, bnCadRunningVar_m4, bnCadNbt_m4, bnCadGrads_m4);
}
void testGhostBatchNormCadenceM8(void) {
    assertCadence(8, bnCadRunningMean_m8, bnCadRunningVar_m8, bnCadNbt_m8, bnCadGrads_m8);
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
    RUN_TEST(testGradsCallTrainsBatchNormThenLeavesEvalMode);
    RUN_TEST(testInferenceUsesRunningStatsAndWritesNothing);
    RUN_TEST(testFrozenBatchNormNeverMovesDuringTraining);
    RUN_TEST(testNoAffineBatchNormAloneStillUpdatesRunningStats);
    RUN_TEST(testCustomGradsFnWithoutFlipRunsBatchNormInEvalMode);
    RUN_TEST(testGradsCallKeepsBatchNormInTrainingModeThroughBackward);
    RUN_TEST(testGhostBatchNormCadenceM2);
    RUN_TEST(testGhostBatchNormCadenceM4);
    RUN_TEST(testGhostBatchNormCadenceM8);
    int rc = UNITY_END();
    freeQuantization(g_q);
    return rc;
}
