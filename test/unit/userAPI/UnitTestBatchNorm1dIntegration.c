#define SOURCE_FILE "UNIT_TEST_BATCHNORM1D_INTEGRATION"

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

#include "BatchNorm1d.h"
#include "BatchNorm1dApi.h"
#include "BatchView.h"
#include "BorrowedLayer.h"
#include "CalculateGradsSequential.h"
#include "DataLoaderApi.h"
#include "Dataset.h"
#include "DeathTest.h"
#include "FlattenApi.h"
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
#include "SgdApi.h"
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

void testLoadBuffersInModelOrderSkippingUntrackedBatchNorm(void) {
    layer_t *a = bnLayer(2, false, false, TRAINABLE_DEFAULT);
    layer_t *untracked = bnLayer(2, false, true, TRAINABLE_DEFAULT);
    layer_t *b = bnLayer(2, false, false, TRAINABLE_DEFAULT);
    layer_t *model[] = {a, untracked, b};
    const float rmA[2] = {0.5f, -0.5f}, rvA[2] = {2.f, 3.f};
    const float rmB[2] = {1.5f, 2.5f}, rvB[2] = {0.25f, 0.75f};
    modelLoadStateDictBuffers(
        model, 3,
        (stateDictBuffers_t[]){
            {.name = "bn1", .runningMean = rmA, .runningVar = rvA, .numBatchesTracked = 7},
            {.name = "bn2", .runningMean = rmB, .runningVar = rvB, .numBatchesTracked = 9},
        },
        2);
    float gotA = ((float *)a->config->batchNorm1d->runningVar->data)[1];
    float gotB = ((float *)b->config->batchNorm1d->runningMean->data)[1];
    uint64_t nA = a->config->batchNorm1d->numBatchesTracked;
    uint64_t nB = b->config->batchNorm1d->numBatchesTracked;
    freeBatchNorm1dLayer(b);
    freeBatchNorm1dLayer(untracked);
    freeBatchNorm1dLayer(a);
    TEST_ASSERT_EQUAL_FLOAT(3.f, gotA);
    TEST_ASSERT_EQUAL_FLOAT(2.5f, gotB);
    TEST_ASSERT_EQUAL_UINT64(7, nA);
    TEST_ASSERT_EQUAL_UINT64(9, nB);
}

static void loadBuffersOrDie(size_t numEntries, const float *rm, const float *rv) {
    layer_t *model[] = {bnLayer(2, false, false, TRAINABLE_DEFAULT)};
    stateDictBuffers_t e[2] = {{.runningMean = rm, .runningVar = rv},
                               {.runningMean = rm, .runningVar = rv}};
    modelLoadStateDictBuffers(model, 1, e, numEntries);
}

void testLoadBuffersRejectsBadInput(void) {
    const float ok[2] = {0.f, 1.f};
    const float nanMean[2] = {NAN, 0.f};
    const float negVar[2] = {1.f, -0.5f};
    const float infVar[2] = {1.f, INFINITY};
    ASSERT_EXITS_WITH_FAILURE(loadBuffersOrDie(2, ok, ok));      /* count mismatch */
    ASSERT_EXITS_WITH_FAILURE(loadBuffersOrDie(1, NULL, ok));    /* NULL mean */
    ASSERT_EXITS_WITH_FAILURE(loadBuffersOrDie(1, ok, NULL));    /* NULL var */
    ASSERT_EXITS_WITH_FAILURE(loadBuffersOrDie(1, nanMean, ok)); /* NaN mean */
    ASSERT_EXITS_WITH_FAILURE(loadBuffersOrDie(1, ok, negVar));  /* negative var */
    ASSERT_EXITS_WITH_FAILURE(loadBuffersOrDie(1, ok, infVar));  /* inf var */
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
        calculateGradsSequential(model, n, defaultLossConfig(MSE), REDUCTION_MEAN, x, y, NULL));
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

/* Loaded buffers + frozen BN, fine-tuned at [1, C]. */
void testFrozenBatchNormKeepsLoadedBuffersDuringTraining(void) {
    layer_t *model[2];
    buildBnLinear(model, TRAINABLE_FALSE);
    const float rm[2] = {0.2f, -0.3f}, rv[2] = {1.5f, 0.5f};
    modelLoadStateDictBuffers(model, 2,
                              (stateDictBuffers_t[]){{.runningMean = rm, .runningVar = rv}}, 1);
    tensor_t *x = buildFloatTensor((size_t[]){1, 2}, 2, kX);
    tensor_t *y = buildFloatTensor((size_t[]){1, 2}, 2, kY);
    freeTrainingStats(
        calculateGradsSequential(model, 2, defaultLossConfig(MSE), REDUCTION_MEAN, x, y, NULL));
    batchNorm1dConfig_t *c = model[0]->config->batchNorm1d;
    float got[4];
    memcpy(got, c->runningMean->data, 2 * sizeof(float));
    memcpy(got + 2, c->runningVar->data, 2 * sizeof(float));
    float linGrad = ((float *)model[1]->config->linear->weights->grad->data)[0];
    freeTensor(y);
    freeTensor(x);
    freeBnLinear(model);
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(rm, got, 2);
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(rv, got + 2, 2);
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

/* A custom calculateGradsFn_t that does NOT route through
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

/* Closes a known mutation gap: if the loop's `false` flip
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

/* Ghost-BN cadence through the stacked loop (#460): one
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
                               REDUCTION_MEAN, m, NULL);
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

/* ---- #467 item 1: trainingRun evaluation pre-flight ---- */

/* The train loader must never be read: if the pre-flight is missing, the run
 * reaches epoch 0's first getBatch and the child exits 2, not 1 -- that is
 * what proves the failure happens BEFORE training, not at the first eval. */
static batch_t *trainGetBatchMustNotRun(dataLoader_t *dl, size_t index) {
    (void)dl;
    (void)index;
    _exit(2);
}
static size_t twoSamples(void) {
    return 2;
}

/* Eval dataset: one sample, item shape settable per test, label [L]. */
static tensor_t *pfItem;
static tensor_t *pfLabel;
static sample_t *pfGetSample(size_t id) {
    (void)id;
    sample_t *s = reserveMemory(sizeof(sample_t));
    s->item = pfItem;
    s->label = pfLabel;
    return s;
}
static size_t g_pfDatasetSize = 2;
static size_t pfDatasetSize(void) {
    return g_pfDatasetSize;
}

/* exit 1: rejected before epoch 0; exit 2: every pre-flight passed and epoch
 * 0 read the (tripwire) train loader. */
static void runPreflightOnly(layer_t **model, size_t n, size_t evalMicroBatchSize) {
    dataLoader_t trainDl = {
        .getDatasetSize = twoSamples, .batchSize = 2, .getBatch = trainGetBatchMustNotRun};
    dataLoader_t *evalDl =
        dataLoaderInit(pfGetSample, pfDatasetSize, 1, NULL, NULL, false, 0, true);
    quantization_t *momentumQ = quantizationInitFloat();
    optimizer_t *sgd =
        sgdMCreateOptim(0.01f, 0.f, 0.f, model, n, momentumQ,
                        (arithmetic_t){.type = ARITH_FLOAT32, .roundingMode = HALF_AWAY});
    trainingRunOptions_t opts = {.microBatchSize = 2, .evalMicroBatchSize = evalMicroBatchSize};
    (void)trainingRun(model, n, defaultLossConfig(MSE), &trainDl, evalDl, sgd, 1,
                      calculateGradsSequential, inferenceWithLoss, &opts);
}

/* Rank-2 untracked BN: eval item [2] -> [1, 2] -> n = 1. */
void testTrainingRunRejectsUntrackedRank2BatchNormBeforeEpoch0(void) {
    g_pfDatasetSize = 2;
    pfItem = buildFloatTensor((size_t[]){2}, 1, (float[]){0.1f, 0.2f});
    pfLabel = buildFloatTensor((size_t[]){2}, 1, (float[]){0.f, 1.f});
    layer_t *model[2] = {
        bnLayer(2, false, true, TRAINABLE_DEFAULT),
        linearLayerInit(&(linearInit_t){.inFeatures = 2, .outFeatures = 2}, &g_lq)};
    ASSERT_EXITS_WITH_FAILURE(runPreflightOnly(model, 2, 1));
    freeLinearLayer(model[1]);
    freeBatchNorm1dLayer(model[0]);
    freeTensor(pfLabel);
    freeTensor(pfItem);
}

/* The sample is rank 2 ([2, 3] -> [1, 2, 3]) but Flatten
 * hands BN [1, 6]: the rank must be walked through the model. */
void testTrainingRunRejectsUntrackedBatchNormBehindFlatten(void) {
    g_pfDatasetSize = 2;
    pfItem = buildFloatTensor((size_t[]){2, 3}, 2, (float[]){0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f});
    pfLabel = buildFloatTensor((size_t[]){6}, 1, (float[]){0.f, 1.f, 0.f, 1.f, 0.f, 1.f});
    layer_t *model[2] = {flattenLayerInit(), bnLayer(6, false, true, TRAINABLE_DEFAULT)};
    ASSERT_EXITS_WITH_FAILURE(runPreflightOnly(model, 2, 1));
    freeBatchNorm1dLayer(model[1]);
    freeFlattenLayer(model[0]);
    freeTensor(pfLabel);
    freeTensor(pfItem);
}

/* Tracked BN(2) first, untracked BN(2) second; item [2]. */
void testTrainingRunRejectsUntrackedSecondBatchNorm(void) {
    g_pfDatasetSize = 2;
    pfItem = buildFloatTensor((size_t[]){2}, 1, (float[]){0.1f, 0.2f});
    pfLabel = buildFloatTensor((size_t[]){2}, 1, (float[]){0.f, 1.f});
    layer_t *model[2] = {bnLayer(2, false, false, TRAINABLE_DEFAULT),
                         bnLayer(2, false, true, TRAINABLE_DEFAULT)};
    ASSERT_EXITS_WITH_FAILURE(runPreflightOnly(model, 2, 1));
    freeBatchNorm1dLayer(model[1]);
    freeBatchNorm1dLayer(model[0]);
    freeTensor(pfLabel);
    freeTensor(pfItem);
}

/* Rank-3 untracked BN with T = 1: item [2, 1] -> [1, 2, 1] -> n = 1.
 * model {BN untracked(2), Flatten, Linear(2->2)}; label [2]. */
void testTrainingRunRejectsUntrackedRank3SingleStep(void) {
    g_pfDatasetSize = 2;
    pfItem = buildFloatTensor((size_t[]){2, 1}, 2, (float[]){0.1f, 0.2f});
    pfLabel = buildFloatTensor((size_t[]){2}, 1, (float[]){0.f, 1.f});
    layer_t *model[3] = {
        bnLayer(2, false, true, TRAINABLE_DEFAULT), flattenLayerInit(),
        linearLayerInit(&(linearInit_t){.inFeatures = 2, .outFeatures = 2}, &g_lq)};
    ASSERT_EXITS_WITH_FAILURE(runPreflightOnly(model, 3, 1));
    freeLinearLayer(model[2]);
    freeFlattenLayer(model[1]);
    freeBatchNorm1dLayer(model[0]);
    freeTensor(pfLabel);
    freeTensor(pfItem);
}

/* The optimizer built over `model` already freed gamma/beta (collected as
 * trainable parameters) and Linear's weights/bias -- freeBatchNorm1dLayer /
 * freeLinearLayer would double-free them. This frees only what freeOptim
 * does not own: BatchNorm1d's running buffers (never parameters) and the
 * layer/config shells (mirrors BorrowedLayer.h's freeLinearLayerShellOnly,
 * extended to BatchNorm1d, which has no such helper there yet). */
static void freeBatchNorm1dLayerAfterOptim(layer_t *layer) {
    batchNorm1dConfig_t *cfg = layer->config->batchNorm1d;
    if (cfg->runningMean != NULL) {
        freeTensor(cfg->runningMean);
    }
    if (cfg->runningVar != NULL) {
        freeTensor(cfg->runningVar);
    }
    freeReservedMemory(cfg);
    freeReservedMemory(layer->config);
    freeReservedMemory(layer);
}

/* Tracked rank-2 BN evaluates on running statistics: must not be rejected. */
void testTrainingRunAcceptsTrackedRank2BatchNorm(void) {
    g_pfDatasetSize = 2;
    pfItem = buildFloatTensor((size_t[]){2}, 1, (float[]){0.1f, 0.2f});
    pfLabel = buildFloatTensor((size_t[]){2}, 1, (float[]){0.f, 1.f});
    layer_t *model[2] = {
        bnLayer(2, false, false, TRAINABLE_DEFAULT),
        linearLayerInit(&(linearInit_t){.inFeatures = 2, .outFeatures = 2}, &g_lq)};

    dataLoader_t *trainDl =
        dataLoaderInit(pfGetSample, pfDatasetSize, 2, NULL, NULL, false, 0, true);
    dataLoader_t *evalDl =
        dataLoaderInit(pfGetSample, pfDatasetSize, 1, NULL, NULL, false, 0, true);
    quantization_t *momentumQ = quantizationInitFloat();
    optimizer_t *sgd =
        sgdMCreateOptim(0.01f, 0.f, 0.f, model, 2, momentumQ,
                        (arithmetic_t){.type = ARITH_FLOAT32, .roundingMode = HALF_AWAY});
    trainingRunOptions_t opts = {.microBatchSize = 2, .evalMicroBatchSize = 1};

    trainingRunResult_t result = trainingRun(model, 2, defaultLossConfig(MSE), trainDl, evalDl, sgd,
                                             1, calculateGradsSequential, inferenceWithLoss, &opts);

    size_t epochsCompleted = result.epochsCompleted;

    freeOptim(sgd);
    freeQuantization(momentumQ);
    freeDataLoader(evalDl);
    freeDataLoader(trainDl);
    freeLinearLayerShellOnly(model[1]);
    freeBatchNorm1dLayerAfterOptim(model[0]);
    freeTensor(pfLabel);
    freeTensor(pfItem);

    TEST_ASSERT_EQUAL_size_t(1, epochsCompleted);
}

/* Untracked rank-3 BN with T >= 2: per-sample evaluation (evalMicroBatchSize 1)
 * has n = T per eval call, stacked evaluation (2) has n = 2 x T. Neither may be
 * rejected. Returns epochsCompleted. */
static size_t acceptUntrackedRank3(size_t evalMicroBatchSize) {
    g_pfDatasetSize = 2;
    pfItem = buildFloatTensor((size_t[]){2, 3}, 2, (float[]){0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f});
    pfLabel = buildFloatTensor((size_t[]){2}, 1, (float[]){0.f, 1.f});
    layer_t *model[3] = {
        bnLayer(2, false, true, TRAINABLE_DEFAULT), flattenLayerInit(),
        linearLayerInit(&(linearInit_t){.inFeatures = 6, .outFeatures = 2}, &g_lq)};

    dataLoader_t *trainDl =
        dataLoaderInit(pfGetSample, pfDatasetSize, 2, NULL, NULL, false, 0, true);
    dataLoader_t *evalDl =
        dataLoaderInit(pfGetSample, pfDatasetSize, 1, NULL, NULL, false, 0, true);
    quantization_t *momentumQ = quantizationInitFloat();
    optimizer_t *sgd =
        sgdMCreateOptim(0.01f, 0.f, 0.f, model, 3, momentumQ,
                        (arithmetic_t){.type = ARITH_FLOAT32, .roundingMode = HALF_AWAY});
    trainingRunOptions_t opts = {.microBatchSize = 2, .evalMicroBatchSize = evalMicroBatchSize};

    trainingRunResult_t result = trainingRun(model, 3, defaultLossConfig(MSE), trainDl, evalDl, sgd,
                                             1, calculateGradsSequential, inferenceWithLoss, &opts);

    size_t epochsCompleted = result.epochsCompleted;

    freeOptim(sgd);
    freeQuantization(momentumQ);
    freeDataLoader(evalDl);
    freeDataLoader(trainDl);
    freeLinearLayerShellOnly(model[2]);
    freeFlattenLayer(model[1]);
    freeBatchNorm1dLayerAfterOptim(model[0]);
    freeTensor(pfLabel);
    freeTensor(pfItem);
    return epochsCompleted;
}

void testTrainingRunAcceptsUntrackedRank3BatchNorm(void) {
    TEST_ASSERT_EQUAL_size_t(1, acceptUntrackedRank3(1));
}

void testTrainingRunAcceptsUntrackedRank3BatchNormStacked(void) {
    TEST_ASSERT_EQUAL_size_t(1, acceptUntrackedRank3(2));
}

/* ---- #468: stacked evaluation pre-flight ---- */

/* #468: an untracked rank-2 BN is evaluable once evaluation stacks >= 2 rows.
 * Inherited (evalMicroBatchSize 0 -> microBatchSize 2) and explicit. */
void testTrainingRunPassesPreflightForUntrackedRank2WhenEvalStacks(void) {
    g_pfDatasetSize = 2;
    pfItem = buildFloatTensor((size_t[]){2}, 1, (float[]){0.1f, 0.2f});
    pfLabel = buildFloatTensor((size_t[]){2}, 1, (float[]){0.f, 1.f});
    layer_t *model[2] = {
        bnLayer(2, false, true, TRAINABLE_DEFAULT),
        linearLayerInit(&(linearInit_t){.inFeatures = 2, .outFeatures = 2}, &g_lq)};
    ASSERT_EXITS_WITH(2, runPreflightOnly(model, 2, 0));
    ASSERT_EXITS_WITH(2, runPreflightOnly(model, 2, 2));
    freeLinearLayer(model[1]);
    freeBatchNorm1dLayer(model[0]);
    freeTensor(pfLabel);
    freeTensor(pfItem);
}

/* The ragged tail chunk is judged too: N = 3, m = 2 -> chunks 2 + 1; the 1-row tail has n = 1. */
void testTrainingRunRejectsUntrackedRank2WithOneRowTail(void) {
    g_pfDatasetSize = 3;
    pfItem = buildFloatTensor((size_t[]){2}, 1, (float[]){0.1f, 0.2f});
    pfLabel = buildFloatTensor((size_t[]){2}, 1, (float[]){0.f, 1.f});
    layer_t *model[2] = {
        bnLayer(2, false, true, TRAINABLE_DEFAULT),
        linearLayerInit(&(linearInit_t){.inFeatures = 2, .outFeatures = 2}, &g_lq)};
    ASSERT_EXITS_WITH_FAILURE(runPreflightOnly(model, 2, 2));
    ASSERT_EXITS_WITH(2, runPreflightOnly(model, 2, 3)); /* one 3-row chunk */
    freeLinearLayer(model[1]);
    freeBatchNorm1dLayer(model[0]);
    freeTensor(pfLabel);
    freeTensor(pfItem);
}

/* m > N judges one N-row chunk, not an m-row one. N = 3,
 * m = 8 -> [3, C]: passes. N = 1, m = 8 -> [1, C]: fails. */
void testPreflightJudgesMinOfMAndN(void) {
    pfItem = buildFloatTensor((size_t[]){2}, 1, (float[]){0.1f, 0.2f});
    pfLabel = buildFloatTensor((size_t[]){2}, 1, (float[]){0.f, 1.f});
    layer_t *model[2] = {
        bnLayer(2, false, true, TRAINABLE_DEFAULT),
        linearLayerInit(&(linearInit_t){.inFeatures = 2, .outFeatures = 2}, &g_lq)};
    g_pfDatasetSize = 3;
    ASSERT_EXITS_WITH(2, runPreflightOnly(model, 2, 8));
    g_pfDatasetSize = 1;
    ASSERT_EXITS_WITH_FAILURE(runPreflightOnly(model, 2, 8));
    g_pfDatasetSize = 2;
    freeLinearLayer(model[1]);
    freeBatchNorm1dLayer(model[0]);
    freeTensor(pfLabel);
    freeTensor(pfItem);
}

/* #468: an explicit evalMicroBatchSize > 1 on a model with a non-FLOAT32
 * forward fails before epoch 0 (exit 1), even with per-sample training. */
static void runSymForwardPreflight(void) {
    g_pfDatasetSize = 2;
    pfItem = buildFloatTensor((size_t[]){2}, 1, (float[]){0.1f, 0.2f});
    pfLabel = buildFloatTensor((size_t[]){2}, 1, (float[]){0.f, 1.f});
    layer_t *model[1] = {
        linearLayerInit(&(linearInit_t){.inFeatures = 2, .outFeatures = 2}, &g_lq)};
    model[0]->config->linear->forwardMath.type = ARITH_SYM_INT32; /* child only */
    dataLoader_t trainDl = {
        .getDatasetSize = twoSamples, .batchSize = 2, .getBatch = trainGetBatchMustNotRun};
    dataLoader_t *evalDl =
        dataLoaderInit(pfGetSample, pfDatasetSize, 1, NULL, NULL, false, 0, true);
    quantization_t *momentumQ = quantizationInitFloat();
    optimizer_t *sgd =
        sgdMCreateOptim(0.01f, 0.f, 0.f, model, 1, momentumQ,
                        (arithmetic_t){.type = ARITH_FLOAT32, .roundingMode = HALF_AWAY});
    trainingRunOptions_t opts = {.microBatchSize = 1, .evalMicroBatchSize = 2};
    (void)trainingRun(model, 1, defaultLossConfig(MSE), &trainDl, evalDl, sgd, 1,
                      calculateGradsSequential, inferenceWithLoss, &opts);
}

void testTrainingRunRejectsNonFloat32ForwardAtEvalMicroBatchBeforeEpoch0(void) {
    ASSERT_EXITS_WITH_FAILURE(runSymForwardPreflight());
}

/* #468: the failure is the EVAL knob's, so it must name evalMicroBatchSize
 * (microBatchSize is 1 here). The child's stdout is a pipe; PRINT_ERROR writes
 * to stdout in every preset and exit() flushes it. */
void testTrainingRunEvalGateNamesEvalMicroBatchSize(void) {
    int fds[2];
    TEST_ASSERT_EQUAL_INT(0, pipe(fds));
    fflush(stdout);
    pid_t pid = fork();
    TEST_ASSERT_TRUE(pid >= 0);
    if (pid == 0) {
        close(fds[0]);
        dup2(fds[1], STDOUT_FILENO);
        close(fds[1]);
        runSymForwardPreflight();
        _exit(0);
    }
    close(fds[1]);
    char message[1024];
    size_t len = 0;
    ssize_t n;
    while ((n = read(fds[0], message + len, sizeof message - 1 - len)) > 0) {
        len += (size_t)n;
        if (len + 1 >= sizeof message) {
            break;
        }
    }
    message[len] = '\0';
    char drain[256];
    while (read(fds[0], drain, sizeof drain) > 0) {}
    close(fds[0]);
    int status = 0;
    (void)waitpid(pid, &status, 0);
    bool exitedWithFailure = WIFEXITED(status) && WEXITSTATUS(status) == 1;
    bool namesEvalKnob = strstr(message, "evalMicroBatchSize") != NULL;
    TEST_ASSERT_TRUE(exitedWithFailure);
    TEST_ASSERT_TRUE_MESSAGE(namesEvalKnob, message);
}

/* In trainingRun: an eval dataset smaller than its batchSize fails before the peek. */
static batch_t *evalGetBatchMustNotRun(dataLoader_t *dl, size_t index) {
    (void)dl;
    (void)index;
    _exit(3);
}
static size_t oneSample(void) {
    return 1;
}
static void runEmptyEvalLoader(void) {
    layer_t *model[1] = {
        linearLayerInit(&(linearInit_t){.inFeatures = 2, .outFeatures = 2}, &g_lq)};
    dataLoader_t trainDl = {
        .getDatasetSize = twoSamples, .batchSize = 2, .getBatch = trainGetBatchMustNotRun};
    dataLoader_t evalDl = {
        .getDatasetSize = oneSample, .batchSize = 2, .getBatch = evalGetBatchMustNotRun};
    quantization_t *momentumQ = quantizationInitFloat();
    optimizer_t *sgd =
        sgdMCreateOptim(0.01f, 0.f, 0.f, model, 1, momentumQ,
                        (arithmetic_t){.type = ARITH_FLOAT32, .roundingMode = HALF_AWAY});
    (void)trainingRun(model, 1, defaultLossConfig(MSE), &trainDl, &evalDl, sgd, 1,
                      calculateGradsSequential, inferenceWithLoss, NULL);
}

void testTrainingRunRejectsEmptyEvalLoaderBeforeThePeek(void) {
    ASSERT_EXITS_WITH_FAILURE(runEmptyEvalLoader());
}

static batch_t *emptyEvalGetBatch(dataLoader_t *dl, size_t index) {
    (void)dl;
    (void)index;
    batch_t *b = reserveMemory(sizeof(batch_t));
    b->size = 0;
    b->samples = reserveMemory(sizeof(sample_t *));
    return b;
}
static void runEmptyFirstEvalBatch(void) {
    layer_t *model[1] = {
        linearLayerInit(&(linearInit_t){.inFeatures = 2, .outFeatures = 2}, &g_lq)};
    dataLoader_t trainDl = {
        .getDatasetSize = twoSamples, .batchSize = 2, .getBatch = trainGetBatchMustNotRun};
    dataLoader_t evalDl = {
        .getDatasetSize = oneSample, .batchSize = 1, .getBatch = emptyEvalGetBatch};
    quantization_t *momentumQ = quantizationInitFloat();
    optimizer_t *sgd =
        sgdMCreateOptim(0.01f, 0.f, 0.f, model, 1, momentumQ,
                        (arithmetic_t){.type = ARITH_FLOAT32, .roundingMode = HALF_AWAY});
    (void)trainingRun(model, 1, defaultLossConfig(MSE), &trainDl, &evalDl, sgd, 1,
                      calculateGradsSequential, inferenceWithLoss, NULL);
}

/* exit 1 (not the tripwire's 2): the peek guard fires before epoch 0 reads
 * the train loader, instead of reading samples[0] of an empty batch. */
void testTrainingRunRejectsEmptyFirstEvalBatch(void) {
    ASSERT_EXITS_WITH_FAILURE(runEmptyFirstEvalBatch());
}

static tensor_t *e2eItems[4];
static tensor_t *e2eLabels[4];
static sample_t *e2eGetSample(size_t id) {
    sample_t *s = reserveMemory(sizeof(sample_t));
    s->item = e2eItems[id];
    s->label = e2eLabels[id];
    return s;
}
static size_t fourSamples(void) {
    return 4;
}

/* Records every output the evaluation hands out, per call. */
static float e2eOutputs[2][4];
static size_t e2eCalls;
static inferenceStats_t *recordingInference(layer_t **model, size_t n, tensor_t *in,
                                            tensor_t *label, lossFuncType_t f, reduction_t r,
                                            const trainingCall_t *call) {
    inferenceStats_t *s = inferenceWithLoss(model, n, in, label, f, r, call);
    if (e2eCalls < 2) {
        memcpy(e2eOutputs[e2eCalls], s->output->data, 4 * sizeof(float));
    }
    e2eCalls++;
    return s;
}

/* Spec 5.5: a chunk's eval output equals a manual inference() over the same
 * stacked rows -- the untracked BN normalizes over exactly that chunk. */
void testUntrackedRank2BatchNormEvaluatesPerChunk(void) {
    const float x[4][2] = {{0.1f, -0.4f}, {0.9f, 0.3f}, {-0.6f, 0.8f}, {0.2f, -0.1f}};
    for (size_t i = 0; i < 4; i++) {
        e2eItems[i] = buildFloatTensor((size_t[]){2}, 1, x[i]);
        e2eLabels[i] =
            buildFloatTensor((size_t[]){2}, 1, (float[]){(float)(i % 2), (float)((i + 1) % 2)});
    }
    layer_t *model[1] = {bnLayer(2, true /* noAffine */, true, TRAINABLE_DEFAULT)};
    dataLoader_t *evalDl = dataLoaderInit(e2eGetSample, fourSamples, 1, NULL, NULL, false, 0, true);
    e2eCalls = 0;
    epochStats_t stats =
        evaluationEpochWithMetrics(model, 1, MSE, evalDl, recordingInference, REDUCTION_MEAN, 2);
    size_t calls = e2eCalls;
    /* Manual reference: chunk 0 = rows {0, 1}, chunk 1 = rows {2, 3}. */
    float expected[2][4];
    for (size_t c = 0; c < 2; c++) {
        tensor_t *stack = buildFloatTensor((size_t[]){2, 2}, 2, NULL);
        float *d = (float *)stack->data;
        memcpy(d, x[2 * c], 2 * sizeof(float));
        memcpy(d + 2, x[2 * c + 1], 2 * sizeof(float));
        tensor_t *out = inference(model, 1, stack);
        memcpy(expected[c], out->data, 4 * sizeof(float));
        freeTensor(out);
        freeTensor(stack);
    }
    freeDataLoader(evalDl);
    freeBatchNorm1dLayer(model[0]);
    for (size_t i = 0; i < 4; i++) {
        freeTensor(e2eLabels[i]);
        freeTensor(e2eItems[i]);
    }
    TEST_ASSERT_EQUAL_size_t(2, calls);
    TEST_ASSERT_TRUE(isfinite(stats.loss));
    TEST_ASSERT_EQUAL_MEMORY(expected, e2eOutputs, sizeof(expected));
}

/* #468 "Done when": an untracked rank-2 BN TRAINS AND EVALUATES through
 * trainingRun (microBatchSize 2, evaluation inherits it). Distinct rows:
 * identical rows would give batch variance 0 in both phases. */
void testTrainingRunTrainsAndEvaluatesUntrackedRank2BatchNorm(void) {
    const float x[4][2] = {{0.1f, -0.4f}, {0.9f, 0.3f}, {-0.6f, 0.8f}, {0.2f, -0.1f}};
    for (size_t i = 0; i < 4; i++) {
        e2eItems[i] = buildFloatTensor((size_t[]){2}, 1, x[i]);
        e2eLabels[i] =
            buildFloatTensor((size_t[]){2}, 1, (float[]){(float)(i % 2), (float)((i + 1) % 2)});
    }
    layer_t *model[2] = {
        bnLayer(2, false, true, TRAINABLE_DEFAULT),
        linearLayerInit(&(linearInit_t){.inFeatures = 2, .outFeatures = 2}, &g_lq)};
    dataLoader_t *trainDl =
        dataLoaderInit(e2eGetSample, fourSamples, 2, NULL, NULL, false, 0, true);
    dataLoader_t *evalDl = dataLoaderInit(e2eGetSample, fourSamples, 1, NULL, NULL, false, 0, true);
    quantization_t *momentumQ = quantizationInitFloat();
    optimizer_t *sgd =
        sgdMCreateOptim(0.01f, 0.f, 0.f, model, 2, momentumQ,
                        (arithmetic_t){.type = ARITH_FLOAT32, .roundingMode = HALF_AWAY});
    trainingRunOptions_t opts = {.microBatchSize = 2};

    trainingRunResult_t result = trainingRun(model, 2, defaultLossConfig(MSE), trainDl, evalDl, sgd,
                                             1, calculateGradsSequential, inferenceWithLoss, &opts);
    size_t epochsCompleted = result.epochsCompleted;
    float evalLoss = result.finalEvalStats.loss;

    freeOptim(sgd);
    freeQuantization(momentumQ);
    freeDataLoader(evalDl);
    freeDataLoader(trainDl);
    freeLinearLayerShellOnly(model[1]);
    freeBatchNorm1dLayerAfterOptim(model[0]);
    for (size_t i = 0; i < 4; i++) {
        freeTensor(e2eLabels[i]);
        freeTensor(e2eItems[i]);
    }
    TEST_ASSERT_EQUAL_size_t(1, epochsCompleted);
    TEST_ASSERT_TRUE(isfinite(evalLoss));
}

/* The direct entry points pre-flight inline, on the first sample they
 * already fetch (no extra getBatch call): at m = 1 this model is
 * rejected BEFORE the first inferenceFn call (the tripwire would exit 2);
 * at m = 2 it evaluates. */
static inferenceStats_t *inferenceMustNotRun(layer_t **model, size_t n, tensor_t *in,
                                             tensor_t *label, lossFuncType_t f, reduction_t r,
                                             const trainingCall_t *call) {
    (void)model;
    (void)n;
    (void)in;
    (void)label;
    (void)f;
    (void)r;
    (void)call;
    _exit(2);
}

static void untrackedMetrics(inferenceWithLossFn_t fn, size_t m) {
    pfItem = buildFloatTensor((size_t[]){2}, 1, (float[]){0.1f, 0.2f});
    pfLabel = buildFloatTensor((size_t[]){2}, 1, (float[]){0.f, 1.f});
    g_pfDatasetSize = 2;
    layer_t *model[1] = {bnLayer(2, true, true, TRAINABLE_DEFAULT)};
    dataLoader_t *evalDl =
        dataLoaderInit(pfGetSample, pfDatasetSize, 1, NULL, NULL, false, 0, true);
    (void)evaluationEpochWithMetrics(model, 1, MSE, evalDl, fn, REDUCTION_MEAN, m);
}

void testEntryPointPreflightsUntrackedBatchNormInline(void) {
    ASSERT_EXITS_WITH(0, untrackedMetrics(inferenceWithLoss, 2));
    ASSERT_EXITS_WITH_FAILURE(untrackedMetrics(inferenceMustNotRun, 1));
}

/* At m = 1 the inline pre-flight runs on the first sample
 * of the first NON-EMPTY batch -- an empty batch 0 must not let the untracked
 * BN reach inferenceFn (the tripwire would exit 2). evaluationEpochWithMetrics
 * is left out: its numClasses peek reads batch 0's first sample. */
static getBatchFn_t g_realGetBatch;
static batch_t *emptyFirstGetBatch(dataLoader_t *dl, size_t index) {
    if (index == 0) {
        batch_t *b = reserveMemory(sizeof(batch_t));
        b->size = 0;
        b->samples = reserveMemory(sizeof(sample_t *));
        return b;
    }
    return g_realGetBatch(dl, index);
}

static void untrackedAfterEmptyFirstBatch(bool viaReport) {
    pfItem = buildFloatTensor((size_t[]){2}, 1, (float[]){0.1f, 0.2f});
    pfLabel = buildFloatTensor((size_t[]){2}, 1, (float[]){0.f, 1.f});
    g_pfDatasetSize = 2;
    layer_t *model[1] = {bnLayer(2, true, true, TRAINABLE_DEFAULT)};
    dataLoader_t *evalDl =
        dataLoaderInit(pfGetSample, pfDatasetSize, 1, NULL, NULL, false, 0, true);
    g_realGetBatch = evalDl->getBatch;
    evalDl->getBatch = emptyFirstGetBatch;
    if (viaReport) {
        size_t cm[4];
        (void)evaluationEpochWithReport(model, 1, MSE, evalDl, inferenceMustNotRun, cm, 2,
                                        REDUCTION_MEAN, 1);
    } else {
        (void)evaluationEpoch(model, 1, MSE, evalDl, inferenceMustNotRun, REDUCTION_MEAN, 1);
    }
}

void testPerSamplePreflightSkipsAnEmptyFirstBatch(void) {
    ASSERT_EXITS_WITH(1, untrackedAfterEmptyFirstBatch(false));
    ASSERT_EXITS_WITH(1, untrackedAfterEmptyFirstBatch(true));
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
    RUN_TEST(testLoadBuffersInModelOrderSkippingUntrackedBatchNorm);
    RUN_TEST(testLoadBuffersRejectsBadInput);
    RUN_TEST(testGradsCallTrainsBatchNormThenLeavesEvalMode);
    RUN_TEST(testInferenceUsesRunningStatsAndWritesNothing);
    RUN_TEST(testFrozenBatchNormNeverMovesDuringTraining);
    RUN_TEST(testFrozenBatchNormKeepsLoadedBuffersDuringTraining);
    RUN_TEST(testNoAffineBatchNormAloneStillUpdatesRunningStats);
    RUN_TEST(testCustomGradsFnWithoutFlipRunsBatchNormInEvalMode);
    RUN_TEST(testGradsCallKeepsBatchNormInTrainingModeThroughBackward);
    RUN_TEST(testGhostBatchNormCadenceM2);
    RUN_TEST(testGhostBatchNormCadenceM4);
    RUN_TEST(testGhostBatchNormCadenceM8);
    RUN_TEST(testTrainingRunRejectsUntrackedRank2BatchNormBeforeEpoch0);
    RUN_TEST(testTrainingRunRejectsUntrackedBatchNormBehindFlatten);
    RUN_TEST(testTrainingRunRejectsUntrackedSecondBatchNorm);
    RUN_TEST(testTrainingRunRejectsUntrackedRank3SingleStep);
    RUN_TEST(testTrainingRunAcceptsTrackedRank2BatchNorm);
    RUN_TEST(testTrainingRunAcceptsUntrackedRank3BatchNorm);
    RUN_TEST(testTrainingRunAcceptsUntrackedRank3BatchNormStacked);
    RUN_TEST(testTrainingRunPassesPreflightForUntrackedRank2WhenEvalStacks);
    RUN_TEST(testTrainingRunRejectsUntrackedRank2WithOneRowTail);
    RUN_TEST(testPreflightJudgesMinOfMAndN);
    RUN_TEST(testTrainingRunRejectsNonFloat32ForwardAtEvalMicroBatchBeforeEpoch0);
    RUN_TEST(testTrainingRunEvalGateNamesEvalMicroBatchSize);
    RUN_TEST(testTrainingRunRejectsEmptyEvalLoaderBeforeThePeek);
    RUN_TEST(testTrainingRunRejectsEmptyFirstEvalBatch);
    RUN_TEST(testUntrackedRank2BatchNormEvaluatesPerChunk);
    RUN_TEST(testEntryPointPreflightsUntrackedBatchNormInline);
    RUN_TEST(testPerSamplePreflightSkipsAnEmptyFirstBatch);
    RUN_TEST(testTrainingRunTrainsAndEvaluatesUntrackedRank2BatchNorm);
    int rc = UNITY_END();
    freeQuantization(g_q);
    return rc;
}
