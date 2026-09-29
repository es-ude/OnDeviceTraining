#define SOURCE_FILE "UNIT_TEST_BATCHNORM1D"

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "ArithmeticType.h"
#include "BatchNorm1d.h"
#include "Layer.h"
#include "Quantization.h"
#include "QuantizationApi.h"
#include "StorageApi.h"
#include "Tensor.h"
#include "TensorApi.h"
#include "unity.h"

#include "BatchNorm1dApi.h"
#include "DeathTest.h"
#include "LayerCommon.h"
#include "LayerQuant.h"
#include "expected_batchnorm1d.h"

void setUp(void) {}
void tearDown(void) {}

#define BN_MAX_C 16
#define BN_MAX_N 64

static tensor_t *buildFloatTensorND(size_t numDims, const size_t *dimsIn, const float *vals) {
    size_t *dims = reserveMemory(numDims * sizeof(size_t));
    for (size_t i = 0; i < numDims; i++) {
        dims[i] = dimsIn[i];
    }
    size_t *order = reserveMemory(numDims * sizeof(size_t));
    setOrderOfDimsForNewTensor(numDims, order);
    shape_t *shape = reserveMemory(sizeof(shape_t));
    setShape(shape, dims, numDims, order);
    tensor_t *t = initTensor(shape, quantizationInitFloat(), NULL);
    if (vals != NULL) {
        tensorFillFromFloatBuffer(t, (float *)vals, calcNumberOfElementsByShape(shape));
    }
    return t;
}

static parameter_t *buildFloatParam(size_t numChannels, const float *vals) {
    tensor_t *p = buildFloatTensorND(1, (size_t[]){numChannels}, vals);
    tensor_t *g = gradInitFloat(p, NULL);
    return parameterInit(p, g);
}

/* Hand-wired BN layer (no factory): the fixture owns every tensor. rmVals /
 * rvVals NULL -> PyTorch init (0 / 1). Initialize IN PLACE: layer.config
 * points into the struct. */
typedef struct {
    batchNorm1dConfig_t cfg;
    layerConfig_t lcfg;
    layer_t layer;
    parameter_t *gamma;
    parameter_t *beta;
    tensor_t *runningMean;
    tensor_t *runningVar;
    quantization_t *fq;
    quantization_t *bq;
} bnFixture_t;

static void bnFixtureInit(bnFixture_t *f, size_t C, bool affine, bool track, const float *gammaVals,
                          const float *betaVals, const float *rmVals, const float *rvVals,
                          bnMomentumMode_t mode, float momentum) {
    TEST_ASSERT_TRUE_MESSAGE(C <= BN_MAX_C, "fixture exceeds channel capture buffer");
    float ones[BN_MAX_C];
    for (size_t c = 0; c < C; c++) {
        ones[c] = 1.0f;
    }
    f->gamma = affine ? buildFloatParam(C, gammaVals) : NULL;
    f->beta = affine ? buildFloatParam(C, betaVals) : NULL;
    f->runningMean = track ? buildFloatTensorND(1, (size_t[]){C}, rmVals) : NULL;
    f->runningVar = track ? buildFloatTensorND(1, (size_t[]){C}, rvVals ? rvVals : ones) : NULL;
    f->fq = quantizationInitFloat();
    f->bq = quantizationInitFloat();
    initBatchNorm1dConfig(&f->cfg, f->gamma, f->beta, f->runningMean, f->runningVar, C, 1e-5f, mode,
                          momentum, f->fq, f->bq);
    f->lcfg.batchNorm1d = &f->cfg;
    f->layer = (layer_t){.type = BATCHNORM1D, .config = &f->lcfg};
}

static void bnFixtureFree(bnFixture_t *f) {
    freeQuantization(f->bq);
    freeQuantization(f->fq);
    if (f->runningVar != NULL) {
        freeTensor(f->runningVar);
    }
    if (f->runningMean != NULL) {
        freeTensor(f->runningMean);
    }
    if (f->beta != NULL) {
        freeParameter(f->beta);
    }
    if (f->gamma != NULL) {
        freeParameter(f->gamma);
    }
}

typedef struct {
    float y[BN_MAX_N];
    float rm[BN_MAX_C];
    float rv[BN_MAX_C];
    uint64_t nbt;
} bnForwardCapture_t;

/* One forward of `x` (dims/rank) through the fixture; captures y + buffers. */
static void bnRunForward(bnFixture_t *f, const size_t *dims, size_t rank, const float *x,
                         bnForwardCapture_t *cap) {
    tensor_t *in = buildFloatTensorND(rank, dims, x);
    tensor_t *out = buildFloatTensorND(rank, dims, NULL);
    size_t total = calcNumberOfElementsByTensor(in);
    TEST_ASSERT_TRUE_MESSAGE(total <= BN_MAX_N, "fixture exceeds capture buffer");
    layerFunctions[BATCHNORM1D].forward(&f->layer, in, out);
    for (size_t i = 0; i < total; i++) {
        cap->y[i] = ((float *)out->data)[i];
    }
    for (size_t c = 0; f->runningMean != NULL && c < f->cfg.numChannels; c++) {
        cap->rm[c] = ((float *)f->runningMean->data)[c];
        cap->rv[c] = ((float *)f->runningVar->data)[c];
    }
    cap->nbt = f->cfg.numBatchesTracked;
    freeTensor(out);
    freeTensor(in);
}

static void assertFloatsWithin(float tol, const float *exp, const float *got, size_t n) {
    for (size_t i = 0; i < n; i++) {
        TEST_ASSERT_FLOAT_WITHIN(tol, exp[i], got[i]);
    }
}

/* Training forward over a train_fixture: y + running buffers + counter. */
static void runTrainForwardGold(const size_t *dims, size_t rank, bool affine, bool track,
                                bnMomentumMode_t mode, float momentum, size_t steps,
                                const float *input, const float *gammaV, const float *betaV,
                                const float *expY, size_t n, const float *expRm, const float *expRv,
                                uint64_t expNbt) {
    bnFixture_t f;
    bnFixtureInit(&f, dims[1], affine, track, gammaV, betaV, NULL, NULL, mode, momentum);
    f.cfg.training = true;
    bnForwardCapture_t cap;
    for (size_t k = 0; k < steps; k++) {
        bnRunForward(&f, dims, rank, input + k * n, &cap);
    }
    size_t C = dims[1];
    bnFixtureFree(&f);
    assertFloatsWithin(1e-4f, expY, cap.y, n);
    if (track) {
        assertFloatsWithin(1e-5f, expRm, cap.rm, C);
        assertFloatsWithin(1e-5f, expRv, cap.rv, C);
        TEST_ASSERT_EQUAL_UINT64(expNbt, cap.nbt);
    }
}

void testGoldTrainForwardRank2(void) {
    runTrainForwardGold((size_t[]){4, 3}, 2, true, true, BN_MOMENTUM_DEFAULT, 0.0f, 1,
                        input_bn_trainRank2, gamma_bn_trainRank2, beta_bn_trainRank2,
                        expectedForward_bn_trainRank2, expectedForward_bn_trainRank2_len,
                        expectedRunningMean_bn_trainRank2, expectedRunningVar_bn_trainRank2,
                        numBatchesTracked_bn_trainRank2);
}

void testGoldTrainForwardRank3(void) {
    runTrainForwardGold((size_t[]){3, 2, 5}, 3, true, true, BN_MOMENTUM_DEFAULT, 0.0f, 1,
                        input_bn_trainRank3, gamma_bn_trainRank3, beta_bn_trainRank3,
                        expectedForward_bn_trainRank3, expectedForward_bn_trainRank3_len,
                        expectedRunningMean_bn_trainRank3, expectedRunningVar_bn_trainRank3,
                        numBatchesTracked_bn_trainRank3);
}

void testGoldTrainForwardMinimalBatchUsesUnbiasedRunningVar(void) {
    runTrainForwardGold((size_t[]){2, 3}, 2, true, true, BN_MOMENTUM_DEFAULT, 0.0f, 1,
                        input_bn_minimalRank2, gamma_bn_minimalRank2, beta_bn_minimalRank2,
                        expectedForward_bn_minimalRank2, expectedForward_bn_minimalRank2_len,
                        expectedRunningMean_bn_minimalRank2, expectedRunningVar_bn_minimalRank2,
                        numBatchesTracked_bn_minimalRank2);
}

void testGoldTrainForwardNoAffine(void) {
    runTrainForwardGold((size_t[]){3, 2, 5}, 3, false, true, BN_MOMENTUM_DEFAULT, 0.0f, 1,
                        input_bn_noAffineRank3, NULL, NULL, expectedForward_bn_noAffineRank3,
                        expectedForward_bn_noAffineRank3_len, expectedRunningMean_bn_noAffineRank3,
                        expectedRunningVar_bn_noAffineRank3, numBatchesTracked_bn_noAffineRank3);
}

void testGoldTrainForwardMomentum03(void) {
    runTrainForwardGold(
        (size_t[]){4, 3}, 2, true, true, BN_MOMENTUM_VALUE, 0.3f, 1, input_bn_momentum03Rank2,
        gamma_bn_momentum03Rank2, beta_bn_momentum03Rank2, expectedForward_bn_momentum03Rank2,
        expectedForward_bn_momentum03Rank2_len, expectedRunningMean_bn_momentum03Rank2,
        expectedRunningVar_bn_momentum03Rank2, numBatchesTracked_bn_momentum03Rank2);
}

/* Review Focus 5: momentum = 1 copies the batch mean / unbiased variance. */
void testMomentumOneCopiesBatchStatistics(void) {
    runTrainForwardGold(
        (size_t[]){4, 3}, 2, true, true, BN_MOMENTUM_VALUE, 1.0f, 1, input_bn_momentumOneRank2,
        gamma_bn_momentumOneRank2, beta_bn_momentumOneRank2, expectedForward_bn_momentumOneRank2,
        expectedForward_bn_momentumOneRank2_len, expectedRunningMean_bn_momentumOneRank2,
        expectedRunningVar_bn_momentumOneRank2, numBatchesTracked_bn_momentumOneRank2);
}

void testGoldTrainForwardCumulativeOverThreeSteps(void) {
    runTrainForwardGold((size_t[]){3, 2, 5}, 3, true, true, BN_MOMENTUM_CUMULATIVE, 0.0f, 3,
                        input_bn_cumulativeRank3, gamma_bn_cumulativeRank3, beta_bn_cumulativeRank3,
                        expectedForward_bn_cumulativeRank3, expectedForward_bn_cumulativeRank3_len,
                        expectedRunningMean_bn_cumulativeRank3,
                        expectedRunningVar_bn_cumulativeRank3,
                        numBatchesTracked_bn_cumulativeRank3);
}

/* Spec §4.3: eps INSIDE the sqrt. Batch variance ~1e-6 ~ eps/10, so dropping
 * eps or adding it outside the sqrt moves y far beyond the 1e-4 tolerance
 * (the O(1)-variance golds above cannot tell the placements apart). */
void testGoldTrainForwardTinyVarianceKeepsEpsInsideSqrt(void) {
    runTrainForwardGold((size_t[]){4, 3}, 2, true, true, BN_MOMENTUM_DEFAULT, 0.0f, 1,
                        input_bn_tinyVarRank2, gamma_bn_tinyVarRank2, beta_bn_tinyVarRank2,
                        expectedForward_bn_tinyVarRank2, expectedForward_bn_tinyVarRank2_len,
                        expectedRunningMean_bn_tinyVarRank2, expectedRunningVar_bn_tinyVarRank2,
                        numBatchesTracked_bn_tinyVarRank2);
}

/* track off: batch statistics in training, in eval, AND when frozen (D3 says
 * frozen normally means running-stat mode, but with track off there are no
 * running buffers to fall back to -- !trackRunningStats must keep winning
 * over frozen, per bnUsesBatchStats's `||`, not a `&&`). PyTorch parity. */
void testNoTrackUsesBatchStatisticsInTrainingAndEval(void) {
    for (int mode = 0; mode < 3; mode++) { /* 0: eval, 1: training, 2: frozen + training */
        bnFixture_t f;
        bnFixtureInit(&f, 3, true, false, gamma_bn_noTrackRank2, beta_bn_noTrackRank2, NULL, NULL,
                      BN_MOMENTUM_DEFAULT, 0.0f);
        f.cfg.training = mode != 0;
        f.cfg.frozen = mode == 2;
        bnForwardCapture_t cap;
        bnRunForward(&f, (size_t[]){4, 3}, 2, input_bn_noTrackRank2, &cap);
        bnFixtureFree(&f);
        assertFloatsWithin(1e-4f, expectedForward_bn_noTrackRank2, cap.y,
                           expectedForward_bn_noTrackRank2_len);
    }
}

/* Eval: running statistics, buffers + counter untouched (byte-identical). */
void testEvalForwardUsesRunningStatsAndWritesNothing(void) {
    bnFixture_t f;
    bnFixtureInit(&f, 2, true, true, gamma_bn_evalRank3, beta_bn_evalRank3,
                  runningMeanInit_bn_evalRank3, runningVarInit_bn_evalRank3, BN_MOMENTUM_DEFAULT,
                  0.0f);
    bnForwardCapture_t cap;
    bnRunForward(&f, (size_t[]){3, 2, 5}, 3, input_bn_evalRank3, &cap);
    bnFixtureFree(&f);
    assertFloatsWithin(1e-4f, expectedForward_bn_evalRank3, cap.y,
                       expectedForward_bn_evalRank3_len);
    TEST_ASSERT_EQUAL_MEMORY(runningMeanInit_bn_evalRank3, cap.rm, 2 * sizeof(float));
    TEST_ASSERT_EQUAL_MEMORY(runningVarInit_bn_evalRank3, cap.rv, 2 * sizeof(float));
    TEST_ASSERT_EQUAL_UINT64(0, cap.nbt);
}

/* Spec §4.3 on the running-statistics path: running_var in [1e-6, 1.1e-5]. */
void testEvalForwardTinyRunningVarKeepsEpsInsideSqrt(void) {
    bnFixture_t f;
    bnFixtureInit(&f, 2, true, true, gamma_bn_evalTinyVarRank3, beta_bn_evalTinyVarRank3,
                  runningMeanInit_bn_evalTinyVarRank3, runningVarInit_bn_evalTinyVarRank3,
                  BN_MOMENTUM_DEFAULT, 0.0f);
    bnForwardCapture_t cap;
    bnRunForward(&f, (size_t[]){3, 2, 5}, 3, input_bn_evalTinyVarRank3, &cap);
    bnFixtureFree(&f);
    assertFloatsWithin(1e-4f, expectedForward_bn_evalTinyVarRank3, cap.y,
                       expectedForward_bn_evalTinyVarRank3_len);
}

/* D3: frozen BN in a training call behaves exactly like eval. */
void testFrozenTrainingForwardUsesRunningStatsAndWritesNothing(void) {
    bnFixture_t f;
    bnFixtureInit(&f, 2, true, true, gamma_bn_evalRank3, beta_bn_evalRank3,
                  runningMeanInit_bn_evalRank3, runningVarInit_bn_evalRank3, BN_MOMENTUM_DEFAULT,
                  0.0f);
    f.cfg.training = true;
    f.cfg.frozen = true;
    bnForwardCapture_t cap;
    bnRunForward(&f, (size_t[]){3, 2, 5}, 3, input_bn_evalRank3, &cap);
    bnFixtureFree(&f);
    assertFloatsWithin(1e-4f, expectedForward_bn_evalRank3, cap.y,
                       expectedForward_bn_evalRank3_len);
    TEST_ASSERT_EQUAL_MEMORY(runningMeanInit_bn_evalRank3, cap.rm, 2 * sizeof(float));
    TEST_ASSERT_EQUAL_MEMORY(runningVarInit_bn_evalRank3, cap.rv, 2 * sizeof(float));
    TEST_ASSERT_EQUAL_UINT64(0, cap.nbt);
}

void testMomentumZeroKeepsRunningStatsButCounts(void) {
    bnFixture_t f;
    bnFixtureInit(&f, 3, true, true, gamma_bn_trainRank2, beta_bn_trainRank2, NULL, NULL,
                  BN_MOMENTUM_VALUE, 0.0f);
    f.cfg.training = true;
    bnForwardCapture_t cap;
    bnRunForward(&f, (size_t[]){4, 3}, 2, input_bn_trainRank2, &cap);
    bnFixtureFree(&f);
    for (size_t c = 0; c < 3; c++) {
        TEST_ASSERT_EQUAL_FLOAT(0.0f, cap.rm[c]);
        TEST_ASSERT_EQUAL_FLOAT(1.0f, cap.rv[c]);
    }
    TEST_ASSERT_EQUAL_UINT64(1, cap.nbt);
}

/* Spec §4.3: the counter saturates, so CUMULATIVE never divides by 0. */
void testCounterSaturatesAtMaxAndStaysFinite(void) {
    bnFixture_t f;
    bnFixtureInit(&f, 3, true, true, gamma_bn_trainRank2, beta_bn_trainRank2, NULL, NULL,
                  BN_MOMENTUM_CUMULATIVE, 0.0f);
    f.cfg.training = true;
    f.cfg.numBatchesTracked = UINT64_MAX;
    bnForwardCapture_t cap;
    bnRunForward(&f, (size_t[]){4, 3}, 2, input_bn_trainRank2, &cap);
    bnFixtureFree(&f);
    TEST_ASSERT_EQUAL_UINT64(UINT64_MAX, cap.nbt);
    for (size_t c = 0; c < 3; c++) {
        TEST_ASSERT_TRUE(isfinite(cap.rm[c]));
        TEST_ASSERT_TRUE(isfinite(cap.rv[c]));
    }
}

void testCalcOutputShapeIsIdentity(void) {
    size_t inDims[] = {3, 2, 5};
    size_t inOrder[] = {0, 1, 2};
    shape_t inShape;
    setShape(&inShape, inDims, 3, inOrder);
    size_t outDims[3] = {0};
    size_t outOrder[3] = {0};
    shape_t outShape;
    setShape(&outShape, outDims, 3, outOrder);
    bnFixture_t f;
    bnFixtureInit(&f, 2, true, true, NULL, NULL, NULL, NULL, BN_MOMENTUM_DEFAULT, 0.0f);
    layerFunctions[BATCHNORM1D].calcOutputShape(&f.layer, &inShape, &outShape);
    bnFixtureFree(&f);
    TEST_ASSERT_EQUAL_size_t(3, outShape.numberOfDimensions);
    TEST_ASSERT_EQUAL_size_t(3, outDims[0]);
    TEST_ASSERT_EQUAL_size_t(2, outDims[1]);
    TEST_ASSERT_EQUAL_size_t(5, outDims[2]);
}

/* ---- fail-fast (forward) ---- */

/* Stack-built input of any dims (heap N = 0 is implementation-defined, #160). */
static void bnForwardOnStackInput(bnFixture_t *f, size_t *dims, size_t rank, size_t *order) {
    shape_t shape;
    setShape(&shape, dims, rank, order);
    quantization_t q;
    initFloat32Quantization(&q);
    float buf[BN_MAX_N] = {0};
    tensor_t in;
    setTensorValues(&in, (uint8_t *)buf, &shape, &q, NULL);
    float obuf[BN_MAX_N] = {0};
    tensor_t out;
    setTensorValues(&out, (uint8_t *)obuf, &shape, &q, NULL);
    layerFunctions[BATCHNORM1D].forward(&f->layer, &in, &out);
}

static void expectTrainingForwardDies(size_t *dims, size_t rank) {
    size_t order[3] = {0, 1, 2};
    bnFixture_t f;
    bnFixtureInit(&f, dims[1], true, true, NULL, NULL, NULL, NULL, BN_MOMENTUM_DEFAULT, 0.0f);
    f.cfg.training = true;
    ASSERT_EXITS_WITH_FAILURE(bnForwardOnStackInput(&f, dims, rank, order));
    bnFixtureFree(&f);
}

void testForwardTrainingRejectsSingleRowRank2(void) {
    expectTrainingForwardDies((size_t[]){1, 3}, 2); /* n = 1 */
}

/* Review Focus 2. */
void testForwardTrainingRejectsRank3SingleValuePerChannel(void) {
    expectTrainingForwardDies((size_t[]){1, 3, 1}, 3); /* n = 1 */
}

void testForwardTrainingRejectsEmptyBatch(void) {
    expectTrainingForwardDies((size_t[]){0, 3}, 2); /* n = 0 */
}

void testForwardTrainingRejectsZeroLengthTime(void) {
    expectTrainingForwardDies((size_t[]){2, 3, 0}, 3); /* n = 0 (Codex pre-design finding) */
}

/* The same degenerate shapes are fine in eval (running statistics). */
void testForwardEvalAcceptsSingleRowAndEmpty(void) {
    size_t order[3] = {0, 1, 2};
    bnFixture_t f;
    bnFixtureInit(&f, 3, true, true, NULL, NULL, NULL, NULL, BN_MOMENTUM_DEFAULT, 0.0f);
    bnForwardOnStackInput(&f, (size_t[]){1, 3}, 2, order);
    bnForwardOnStackInput(&f, (size_t[]){0, 3}, 2, order);
    uint64_t nbt = f.cfg.numBatchesTracked;
    bnFixtureFree(&f);
    TEST_ASSERT_EQUAL_UINT64(0, nbt);
}

/* Adversarial-review fix #1: !trackRunningStats always uses batch statistics
 * (even in eval, D4), so an untracked rank-2 [1, C] evaluation sample still
 * has n = 1 and must die -- but with a DEDICATED message (not the training
 * one, which wrongly tells the caller to raise microBatchSize/says a frozen
 * or eval-mode BN would fall back to running statistics; an untracked BN has
 * none to fall back to). Distinct from testNoTrackUsesBatchStatisticsInTraining
 * AndEval, whose eval-mode case uses n = 4 and never dies. */
void testForwardEvalUntrackedRejectsSingleSample(void) {
    size_t order[2] = {0, 1};
    bnFixture_t f;
    bnFixtureInit(&f, 3, true, false /* track */, NULL, NULL, NULL, NULL, BN_MOMENTUM_DEFAULT,
                  0.0f);
    /* f.cfg.training stays false (factory default): eval mode. */
    ASSERT_EXITS_WITH_FAILURE(bnForwardOnStackInput(&f, (size_t[]){1, 3}, 2, order));
    bnFixtureFree(&f);
}

/* #467 item 2: a frozen, untracked BN still normalizes with batch statistics
 * in training (bnUsesBatchStats: !trackRunningStats), so [1, C] dies -- with
 * its own message, since the generic one claims a frozen/eval BN falls back
 * to running statistics, which an untracked BN does not have. Exit code only;
 * pinned by the return-early mutation on the untracked-training branch. */
void testForwardTrainingFrozenUntrackedRejectsSingleRow(void) {
    size_t order[2] = {0, 1};
    bnFixture_t f;
    bnFixtureInit(&f, 3, true, false /* track */, NULL, NULL, NULL, NULL, BN_MOMENTUM_DEFAULT,
                  0.0f);
    f.cfg.training = true;
    f.cfg.frozen = true;
    ASSERT_EXITS_WITH_FAILURE(bnForwardOnStackInput(&f, (size_t[]){1, 3}, 2, order));
    bnFixtureFree(&f);
}

void testForwardRejectsRank1(void) {
    size_t order[1] = {0};
    bnFixture_t f;
    bnFixtureInit(&f, 3, true, true, NULL, NULL, NULL, NULL, BN_MOMENTUM_DEFAULT, 0.0f);
    ASSERT_EXITS_WITH_FAILURE(bnForwardOnStackInput(&f, (size_t[]){3}, 1, order));
    bnFixtureFree(&f);
}

void testForwardRejectsRank4(void) {
    size_t order[4] = {0, 1, 2, 3};
    bnFixture_t f;
    bnFixtureInit(&f, 3, true, true, NULL, NULL, NULL, NULL, BN_MOMENTUM_DEFAULT, 0.0f);
    ASSERT_EXITS_WITH_FAILURE(bnForwardOnStackInput(&f, (size_t[]){2, 3, 2, 2}, 4, order));
    bnFixtureFree(&f);
}

void testForwardRejectsTransposedInput(void) {
    size_t order[2] = {1, 0};
    bnFixture_t f;
    bnFixtureInit(&f, 3, true, true, NULL, NULL, NULL, NULL, BN_MOMENTUM_DEFAULT, 0.0f);
    ASSERT_EXITS_WITH_FAILURE(bnForwardOnStackInput(&f, (size_t[]){3, 3}, 2, order));
    bnFixtureFree(&f);
}

void testForwardRejectsWrongChannelCount(void) {
    size_t order[2] = {0, 1};
    bnFixture_t f;
    bnFixtureInit(&f, 3, true, true, NULL, NULL, NULL, NULL, BN_MOMENTUM_DEFAULT, 0.0f);
    ASSERT_EXITS_WITH_FAILURE(bnForwardOnStackInput(&f, (size_t[]){4, 2}, 2, order));
    bnFixtureFree(&f);
}

static void forwardSymInput(bnFixture_t *f) {
    size_t dims[] = {4, 3};
    tensor_t *in = buildFloatTensorND(2, dims, input_bn_trainRank2);
    tensor_t *out = buildFloatTensorND(2, dims, NULL);
    /* relabel the input's dtype only: the guard fires before any data read */
    freeQuantization(in->quantization);
    in->quantization = quantizationInitSymInt32(HALF_AWAY);
    layerFunctions[BATCHNORM1D].forward(&f->layer, in, out);
}

void testForwardRejectsNonFloat32Input(void) {
    bnFixture_t f;
    bnFixtureInit(&f, 3, true, true, NULL, NULL, NULL, NULL, BN_MOMENTUM_DEFAULT, 0.0f);
    ASSERT_EXITS_WITH_FAILURE(forwardSymInput(&f));
    bnFixtureFree(&f);
}

void testForwardRejectsNonFloat32Math(void) {
    size_t order[2] = {0, 1};
    bnFixture_t f;
    bnFixtureInit(&f, 3, true, true, NULL, NULL, NULL, NULL, BN_MOMENTUM_DEFAULT, 0.0f);
    f.cfg.forwardMath = (arithmetic_t){.type = ARITH_SYM_INT32, .roundingMode = HALF_AWAY};
    ASSERT_EXITS_WITH_FAILURE(bnForwardOnStackInput(&f, (size_t[]){4, 3}, 2, order));
    bnFixtureFree(&f);
}

/* Codex plan review: half-pairs, running-buffer dtype/capacity, output shape. */
static void initWithGammaOnly(void) {
    batchNorm1dConfig_t cfg;
    parameter_t *g = buildFloatParam(3, NULL);
    quantization_t *q = quantizationInitFloat();
    initBatchNorm1dConfig(&cfg, g, NULL, NULL, NULL, 3, 1e-5f, BN_MOMENTUM_DEFAULT, 0.0f, q, q);
}

static void initWithRunningMeanOnly(void) {
    batchNorm1dConfig_t cfg;
    tensor_t *rm = buildFloatTensorND(1, (size_t[]){3}, NULL);
    quantization_t *q = quantizationInitFloat();
    initBatchNorm1dConfig(&cfg, NULL, NULL, rm, NULL, 3, 1e-5f, BN_MOMENTUM_DEFAULT, 0.0f, q, q);
}

void testInitRejectsGammaWithoutBeta(void) {
    ASSERT_EXITS_WITH_FAILURE(initWithGammaOnly());
}

void testInitRejectsRunningMeanWithoutRunningVar(void) {
    ASSERT_EXITS_WITH_FAILURE(initWithRunningMeanOnly());
}

static void evalForwardWithSymRunningVar(bnFixture_t *f) {
    size_t order[2] = {0, 1};
    /* relabel only: the guard fires before any data read */
    freeQuantization(f->runningVar->quantization);
    f->runningVar->quantization = quantizationInitSymInt32(HALF_AWAY);
    bnForwardOnStackInput(f, (size_t[]){4, 3}, 2, order);
}

void testForwardRejectsNonFloat32RunningBuffer(void) {
    bnFixture_t f;
    bnFixtureInit(&f, 3, true, true, NULL, NULL, NULL, NULL, BN_MOMENTUM_DEFAULT, 0.0f);
    ASSERT_EXITS_WITH_FAILURE(evalForwardWithSymRunningVar(&f));
    bnFixtureFree(&f);
}

static void evalForwardWithShortRunningMean(bnFixture_t *f) {
    size_t order[2] = {0, 1};
    f->cfg.runningMean = buildFloatTensorND(1, (size_t[]){2}, NULL); /* C = 3 */
    bnForwardOnStackInput(f, (size_t[]){4, 3}, 2, order);
}

void testForwardRejectsShortRunningBuffer(void) {
    bnFixture_t f;
    bnFixtureInit(&f, 3, true, true, NULL, NULL, NULL, NULL, BN_MOMENTUM_DEFAULT, 0.0f);
    ASSERT_EXITS_WITH_FAILURE(evalForwardWithShortRunningMean(&f));
    bnFixtureFree(&f);
}

static void forwardIntoSmallerOutput(bnFixture_t *f) {
    size_t order[2] = {0, 1};
    shape_t inShape;
    setShape(&inShape, (size_t[]){2, 3}, 2, order);
    shape_t outShape;
    setShape(&outShape, (size_t[]){1, 3}, 2, order);
    quantization_t q;
    initFloat32Quantization(&q);
    float buf[6] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    tensor_t in;
    setTensorValues(&in, (uint8_t *)buf, &inShape, &q, NULL);
    float obuf[3] = {0};
    tensor_t out;
    setTensorValues(&out, (uint8_t *)obuf, &outShape, &q, NULL);
    layerFunctions[BATCHNORM1D].forward(&f->layer, &in, &out);
}

void testForwardRejectsMismatchedOutputShape(void) {
    bnFixture_t f;
    bnFixtureInit(&f, 3, true, true, NULL, NULL, NULL, NULL, BN_MOMENTUM_DEFAULT, 0.0f);
    f.cfg.training = true; /* batch-stats path, n = 2 passes the size check */
    ASSERT_EXITS_WITH_FAILURE(forwardIntoSmallerOutput(&f));
    bnFixtureFree(&f);
}

/* Adversarial-review fix #3: bnValidateOutputMatchesInput only checked rank
 * and dims, not orderOfDimensions or dtype -- the kernel writes the output in
 * the input's flat IDENTITY order (bnForwardKernelFloat indexes `y[i]`
 * linearly), so a same-shape but transposed output would land values at the
 * wrong physical offsets, and a non-FLOAT32 output would misinterpret the
 * written bit pattern. m == C == 2 here so dims alone cannot tell {1,0} from
 * {0,1} apart -- only the order check can. */
static void forwardWithTransposedOutput(bnFixture_t *f) {
    size_t dims[] = {2, 2};
    size_t inOrder[] = {0, 1};
    size_t outOrder[] = {1, 0};
    shape_t inShape;
    setShape(&inShape, dims, 2, inOrder);
    shape_t outShape;
    setShape(&outShape, dims, 2, outOrder);
    quantization_t q;
    initFloat32Quantization(&q);
    float buf[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    tensor_t in;
    setTensorValues(&in, (uint8_t *)buf, &inShape, &q, NULL);
    float obuf[4] = {0};
    tensor_t out;
    setTensorValues(&out, (uint8_t *)obuf, &outShape, &q, NULL);
    layerFunctions[BATCHNORM1D].forward(&f->layer, &in, &out);
}

void testForwardRejectsTransposedOutput(void) {
    bnFixture_t f;
    bnFixtureInit(&f, 2, true, true, NULL, NULL, NULL, NULL, BN_MOMENTUM_DEFAULT, 0.0f);
    f.cfg.training = true; /* batch-stats path, n = 2 passes the size check */
    ASSERT_EXITS_WITH_FAILURE(forwardWithTransposedOutput(&f));
    bnFixtureFree(&f);
}

/* relabel only: the guard fires before any data read/write. */
static void forwardWithNonFloat32Output(bnFixture_t *f) {
    size_t dims[] = {4, 3};
    tensor_t *in = buildFloatTensorND(2, dims, input_bn_trainRank2);
    tensor_t *out = buildFloatTensorND(2, dims, NULL);
    freeQuantization(out->quantization);
    out->quantization = quantizationInitSymInt32(HALF_AWAY);
    layerFunctions[BATCHNORM1D].forward(&f->layer, in, out);
}

void testForwardRejectsNonFloat32Output(void) {
    bnFixture_t f;
    bnFixtureInit(&f, 3, true, true, NULL, NULL, NULL, NULL, BN_MOMENTUM_DEFAULT, 0.0f);
    f.cfg.training = true;
    ASSERT_EXITS_WITH_FAILURE(forwardWithNonFloat32Output(&f));
    bnFixtureFree(&f);
}

/* ---- backward ---- */

typedef struct {
    float dx[BN_MAX_N];
    float dg[BN_MAX_C];
    float db[BN_MAX_C];
    float rm[BN_MAX_C];
    float rv[BN_MAX_C];
    uint64_t nbt;
} bnBackwardCapture_t;

/* passes backward calls through the vtable; propLoss NULL -> grads-only. */
static void bnRunBackward(bnFixture_t *f, const size_t *dims, size_t rank, const float *x,
                          const float *gy, bool withPropLoss, size_t passes,
                          bnBackwardCapture_t *cap) {
    tensor_t *in = buildFloatTensorND(rank, dims, x);
    tensor_t *loss = buildFloatTensorND(rank, dims, gy);
    tensor_t *prop = withPropLoss ? buildFloatTensorND(rank, dims, NULL) : NULL;
    size_t total = calcNumberOfElementsByTensor(in);
    for (size_t p = 0; p < passes; p++) {
        layerFunctions[BATCHNORM1D].backward(&f->layer, in, loss, prop);
    }
    for (size_t i = 0; prop != NULL && i < total; i++) {
        cap->dx[i] = ((float *)prop->data)[i];
    }
    for (size_t c = 0; c < f->cfg.numChannels; c++) {
        if (f->gamma != NULL && f->gamma->grad != NULL) {
            cap->dg[c] = ((float *)f->gamma->grad->data)[c];
            cap->db[c] = ((float *)f->beta->grad->data)[c];
        }
        if (f->runningMean != NULL) {
            cap->rm[c] = ((float *)f->runningMean->data)[c];
            cap->rv[c] = ((float *)f->runningVar->data)[c];
        }
    }
    cap->nbt = f->cfg.numBatchesTracked;
    if (prop != NULL) {
        freeTensor(prop);
    }
    freeTensor(loss);
    freeTensor(in);
}

static void runTrainBackwardGold(const size_t *dims, size_t rank, bool affine, const float *x,
                                 const float *gammaV, const float *betaV, const float *gy,
                                 const float *expDx, size_t n, const float *expDg,
                                 const float *expDb) {
    bnFixture_t f;
    bnFixtureInit(&f, dims[1], affine, true, gammaV, betaV, NULL, NULL, BN_MOMENTUM_DEFAULT, 0.0f);
    f.cfg.training = true;
    bnBackwardCapture_t cap;
    bnRunBackward(&f, dims, rank, x, gy, true, 1, &cap);
    size_t C = dims[1];
    bnFixtureFree(&f);
    assertFloatsWithin(1e-4f, expDx, cap.dx, n);
    if (affine) {
        assertFloatsWithin(1e-4f, expDg, cap.dg, C);
        assertFloatsWithin(1e-4f, expDb, cap.db, C);
    }
    /* backward never touches the running stats (init 0 / 1, counter 0) */
    for (size_t c = 0; c < C; c++) {
        TEST_ASSERT_EQUAL_FLOAT(0.0f, cap.rm[c]);
        TEST_ASSERT_EQUAL_FLOAT(1.0f, cap.rv[c]);
    }
    TEST_ASSERT_EQUAL_UINT64(0, cap.nbt);
}

void testGoldTrainBackwardRank2(void) {
    runTrainBackwardGold((size_t[]){4, 3}, 2, true, input_bn_trainRank2, gamma_bn_trainRank2,
                         beta_bn_trainRank2, lossGrad_bn_trainRank2, expectedPropLoss_bn_trainRank2,
                         expectedPropLoss_bn_trainRank2_len, expectedDgamma_bn_trainRank2,
                         expectedDbeta_bn_trainRank2);
}

void testGoldTrainBackwardRank3(void) {
    runTrainBackwardGold((size_t[]){3, 2, 5}, 3, true, input_bn_trainRank3, gamma_bn_trainRank3,
                         beta_bn_trainRank3, lossGrad_bn_trainRank3, expectedPropLoss_bn_trainRank3,
                         expectedPropLoss_bn_trainRank3_len, expectedDgamma_bn_trainRank3,
                         expectedDbeta_bn_trainRank3);
}

void testGoldTrainBackwardMinimalBatch(void) {
    runTrainBackwardGold((size_t[]){2, 3}, 2, true, input_bn_minimalRank2, gamma_bn_minimalRank2,
                         beta_bn_minimalRank2, lossGrad_bn_minimalRank2,
                         expectedPropLoss_bn_minimalRank2, expectedPropLoss_bn_minimalRank2_len,
                         expectedDgamma_bn_minimalRank2, expectedDbeta_bn_minimalRank2);
}

void testGoldTrainBackwardNoAffine(void) {
    runTrainBackwardGold((size_t[]){3, 2, 5}, 3, false, input_bn_noAffineRank3, NULL, NULL,
                         lossGrad_bn_noAffineRank3, expectedPropLoss_bn_noAffineRank3,
                         expectedPropLoss_bn_noAffineRank3_len, NULL, NULL);
}

/* Running-statistics backward (PyTorch eval-mode parity): dx = dy*g*invStd,
 * dgamma = sum dy*xhat_running, dbeta = sum dy. */
void testGoldEvalBackward(void) {
    bnFixture_t f;
    bnFixtureInit(&f, 2, true, true, gamma_bn_evalRank3, beta_bn_evalRank3,
                  runningMeanInit_bn_evalRank3, runningVarInit_bn_evalRank3, BN_MOMENTUM_DEFAULT,
                  0.0f);
    bnBackwardCapture_t cap;
    bnRunBackward(&f, (size_t[]){3, 2, 5}, 3, input_bn_evalRank3, lossGrad_bn_evalRank3, true, 1,
                  &cap);
    bnFixtureFree(&f);
    assertFloatsWithin(1e-4f, expectedPropLoss_bn_evalRank3, cap.dx,
                       expectedPropLoss_bn_evalRank3_len);
    assertFloatsWithin(1e-4f, expectedDgamma_bn_evalRank3, cap.dg, 2);
    assertFloatsWithin(1e-4f, expectedDbeta_bn_evalRank3, cap.db, 2);
}

/* D3: frozen BN, training call -> running-stat dx, no grad buffers touched. */
void testFrozenBackwardUsesRunningStatsAndNoGrads(void) {
    bnFixture_t f;
    bnFixtureInit(&f, 2, true, true, gamma_bn_evalRank3, beta_bn_evalRank3,
                  runningMeanInit_bn_evalRank3, runningVarInit_bn_evalRank3, BN_MOMENTUM_DEFAULT,
                  0.0f);
    /* factory-frozen layers have NO grad tensors: free them to prove none is read */
    freeTensor(f.gamma->grad);
    f.gamma->grad = NULL;
    freeTensor(f.beta->grad);
    f.beta->grad = NULL;
    f.cfg.training = true;
    f.cfg.frozen = true;
    bnBackwardCapture_t cap;
    bnRunBackward(&f, (size_t[]){3, 2, 5}, 3, input_bn_evalRank3, lossGrad_bn_evalRank3, true, 1,
                  &cap);
    bnFixtureFree(&f);
    assertFloatsWithin(1e-4f, expectedPropLoss_bn_evalRank3, cap.dx,
                       expectedPropLoss_bn_evalRank3_len);
}

/* Grads-only (deepest trainable layer, propLoss NULL): same grads; two
 * passes accumulate 2x (grads +=), dx memory never touched. */
void testGradsOnlyBackwardMatchesFullAndAccumulates(void) {
    bnFixture_t f;
    bnFixtureInit(&f, 3, true, true, gamma_bn_trainRank2, beta_bn_trainRank2, NULL, NULL,
                  BN_MOMENTUM_DEFAULT, 0.0f);
    f.cfg.training = true;
    bnBackwardCapture_t cap;
    bnRunBackward(&f, (size_t[]){4, 3}, 2, input_bn_trainRank2, lossGrad_bn_trainRank2, false, 2,
                  &cap);
    bnFixtureFree(&f);
    for (size_t c = 0; c < 3; c++) {
        TEST_ASSERT_FLOAT_WITHIN(2e-4f, 2.0f * expectedDgamma_bn_trainRank2[c], cap.dg[c]);
        TEST_ASSERT_FLOAT_WITHIN(2e-4f, 2.0f * expectedDbeta_bn_trainRank2[c], cap.db[c]);
    }
}

static void backwardOnSingleRow(bnFixture_t *f) {
    size_t dims[] = {1, 3};
    tensor_t *in = buildFloatTensorND(2, dims, (float[]){1.f, 2.f, 3.f});
    tensor_t *loss = buildFloatTensorND(2, dims, (float[]){0.5f, -1.f, 2.f});
    tensor_t *prop = buildFloatTensorND(2, dims, NULL);
    layerFunctions[BATCHNORM1D].backward(&f->layer, in, loss, prop);
}

void testBackwardTrainingRejectsSingleRow(void) {
    bnFixture_t f;
    bnFixtureInit(&f, 3, true, true, NULL, NULL, NULL, NULL, BN_MOMENTUM_DEFAULT, 0.0f);
    f.cfg.training = true;
    ASSERT_EXITS_WITH_FAILURE(backwardOnSingleRow(&f));
    bnFixtureFree(&f);
}

static void backwardWithMismatchedLoss(bnFixture_t *f) {
    tensor_t *in = buildFloatTensorND(2, (size_t[]){4, 3}, input_bn_trainRank2);
    tensor_t *loss = buildFloatTensorND(2, (size_t[]){3, 4}, lossGrad_bn_trainRank2);
    tensor_t *prop = buildFloatTensorND(2, (size_t[]){4, 3}, NULL);
    layerFunctions[BATCHNORM1D].backward(&f->layer, in, loss, prop);
}

void testBackwardRejectsLossShapeMismatch(void) {
    bnFixture_t f;
    bnFixtureInit(&f, 3, true, true, NULL, NULL, NULL, NULL, BN_MOMENTUM_DEFAULT, 0.0f);
    f.cfg.training = true;
    ASSERT_EXITS_WITH_FAILURE(backwardWithMismatchedLoss(&f));
    bnFixtureFree(&f);
}

static void backwardWithShortGammaGrad(bnFixture_t *f) {
    freeTensor(f->gamma->grad);
    f->gamma->grad = buildFloatTensorND(1, (size_t[]){f->cfg.numChannels - 1}, NULL);
    tensor_t *in = buildFloatTensorND(2, (size_t[]){4, 3}, input_bn_trainRank2);
    tensor_t *loss = buildFloatTensorND(2, (size_t[]){4, 3}, lossGrad_bn_trainRank2);
    tensor_t *prop = buildFloatTensorND(2, (size_t[]){4, 3}, NULL);
    layerFunctions[BATCHNORM1D].backward(&f->layer, in, loss, prop);
}

/* Adversarial-review fix #2: the ad hoc grad check only verified dtype, not
 * element count -- a gamma grad with the wrong number of elements (here
 * C - 1) was accepted, and the backward's per-channel write loop would then
 * walk off the end of the buffer. bnRequireChannelVector (already used for
 * gamma/beta/running buffers) checks both dtype and capacity; reuse it here
 * instead of the bespoke predicate. */
void testBackwardRejectsShortGammaGrad(void) {
    bnFixture_t f;
    bnFixtureInit(&f, 3, true, true, gamma_bn_trainRank2, beta_bn_trainRank2, NULL, NULL,
                  BN_MOMENTUM_DEFAULT, 0.0f);
    f.cfg.training = true;
    ASSERT_EXITS_WITH_FAILURE(backwardWithShortGammaGrad(&f));
    bnFixtureFree(&f);
}

static layerQuant_t floatLq(quantization_t *q) {
    layerQuant_t lq;
    layerQuantInitUniform(&lq, q);
    return lq;
}

void testFactoryZeroInitGivesPyTorchDefaults(void) {
    quantization_t *q = quantizationInitFloat();
    layerQuant_t lq = floatLq(q);
    layer_t *layer = batchNorm1dLayerInit(&(batchNorm1dInit_t){.numChannels = 3}, &lq);
    batchNorm1dConfig_t *c = layer->config->batchNorm1d;
    bool typeOk = layer->type == BATCHNORM1D;
    float eps = c->eps;
    bnMomentumMode_t mode = c->momentumMode;
    float momentum = c->momentum;
    bool affine = c->affine, track = c->trackRunningStats, training = c->training,
         frozen = c->frozen;
    float g = ((float *)c->gamma->param->data)[2];
    float b = ((float *)c->beta->param->data)[2];
    float rm = ((float *)c->runningMean->data)[2];
    float rv = ((float *)c->runningVar->data)[2];
    bool gradsFloat = c->gamma->grad->quantization->type == FLOAT32 &&
                      c->beta->grad->quantization->type == FLOAT32;
    bool borrowed = c->outputQ == q && !c->ownsQuantizations;
    uint64_t nbt = c->numBatchesTracked;
    freeBatchNorm1dLayer(layer);
    freeQuantization(q);
    TEST_ASSERT_TRUE(typeOk);
    TEST_ASSERT_FLOAT_WITHIN(1e-9f, 1e-5f, eps);
    TEST_ASSERT_EQUAL_INT(BN_MOMENTUM_VALUE, mode);
    TEST_ASSERT_FLOAT_WITHIN(1e-9f, 0.1f, momentum);
    TEST_ASSERT_TRUE(affine && track && !training && !frozen && gradsFloat && borrowed);
    TEST_ASSERT_EQUAL_FLOAT(1.0f, g);
    TEST_ASSERT_EQUAL_FLOAT(0.0f, b);
    TEST_ASSERT_EQUAL_FLOAT(0.0f, rm);
    TEST_ASSERT_EQUAL_FLOAT(1.0f, rv);
    TEST_ASSERT_EQUAL_UINT64(0, nbt);
}

void testFactoryOptionsNoAffineNoStatsCumulativeFrozen(void) {
    quantization_t *q = quantizationInitFloat();
    layerQuant_t lq = floatLq(q);
    layer_t *a = batchNorm1dLayerInitOwning(
        &(batchNorm1dInit_t){.numChannels = 2, .noAffine = true, .noRunningStats = true}, &lq);
    layer_t *b = batchNorm1dLayerInitOwning(
        &(batchNorm1dInit_t){
            .numChannels = 2, .momentumMode = BN_MOMENTUM_CUMULATIVE, .trainable = TRAINABLE_FALSE},
        &lq);
    batchNorm1dConfig_t *ca = a->config->batchNorm1d;
    batchNorm1dConfig_t *cb = b->config->batchNorm1d;
    bool aOk = !ca->affine && ca->gamma == NULL && ca->beta == NULL && !ca->trackRunningStats &&
               ca->runningMean == NULL && ca->runningVar == NULL && !ca->frozen;
    bool bOk = cb->momentumMode == BN_MOMENTUM_CUMULATIVE && cb->frozen &&
               cb->gamma->grad == NULL && cb->beta->grad == NULL && cb->ownsQuantizations &&
               cb->outputQ != q;
    freeBatchNorm1dLayer(b);
    freeBatchNorm1dLayer(a);
    freeQuantization(q);
    TEST_ASSERT_TRUE(aOk);
    TEST_ASSERT_TRUE(bOk);
}

static void initBnOrDie(batchNorm1dInit_t init, qtype_t slotType, int slot) {
    quantization_t *q = quantizationInitFloat();
    quantization_t *sym = quantizationInitSymInt32(HALF_AWAY);
    layerQuant_t lq = floatLq(q);
    if (slot == 1) {
        lq.forwardMath = arithmeticFromQuantization(sym);
    } else if (slot == 2) {
        lq.outputQ = sym;
    } else if (slot == 3) {
        lq.weightStorage = sym;
    } else if (slot == 4) {
        lq.weightGradStorage = sym;
    }
    (void)slotType;
    (void)batchNorm1dLayerInit(&init, &lq);
}

void testFactoryRejectsInvalidInit(void) {
    ASSERT_EXITS_WITH_FAILURE(initBnOrDie((batchNorm1dInit_t){.numChannels = 0}, FLOAT32, 0));
    ASSERT_EXITS_WITH_FAILURE(
        initBnOrDie((batchNorm1dInit_t){.numChannels = 3, .eps = -1e-5f}, FLOAT32, 0));
    ASSERT_EXITS_WITH_FAILURE(
        initBnOrDie((batchNorm1dInit_t){.numChannels = 3, .eps = NAN}, FLOAT32, 0));
    ASSERT_EXITS_WITH_FAILURE(initBnOrDie(
        (batchNorm1dInit_t){.numChannels = 3, .momentumMode = (bnMomentumMode_t)7}, FLOAT32, 0));
    ASSERT_EXITS_WITH_FAILURE(initBnOrDie(
        (batchNorm1dInit_t){.numChannels = 3, .momentumMode = BN_MOMENTUM_VALUE, .momentum = 1.5f},
        FLOAT32, 0));
    ASSERT_EXITS_WITH_FAILURE(initBnOrDie(
        (batchNorm1dInit_t){.numChannels = 3, .momentumMode = BN_MOMENTUM_VALUE, .momentum = NAN},
        FLOAT32, 0));
    ASSERT_EXITS_WITH_FAILURE(initBnOrDie(
        (batchNorm1dInit_t){.numChannels = 3, .noAffine = true, .trainable = TRAINABLE_TRUE},
        FLOAT32, 0));
}

void testFactoryRejectsNonFloat32Slots(void) {
    for (int slot = 1; slot <= 4; slot++) {
        ASSERT_EXITS_WITH_FAILURE(
            initBnOrDie((batchNorm1dInit_t){.numChannels = 3}, SYM_INT32, slot));
    }
}

void testFactoryRejectsNullPointers(void) {
    quantization_t *q = quantizationInitFloat();
    layerQuant_t lq = floatLq(q);
    ASSERT_EXITS_WITH_FAILURE((void)batchNorm1dLayerInit(NULL, &lq));
    ASSERT_EXITS_WITH_FAILURE(
        (void)batchNorm1dLayerInit(&(batchNorm1dInit_t){.numChannels = 3}, NULL));
    freeQuantization(q);
}

/* Momentum 0 via VALUE is legal (stats frozen in place), unlike zero-init. */
void testFactoryAcceptsExplicitMomentumZeroAndOne(void) {
    quantization_t *q = quantizationInitFloat();
    layerQuant_t lq = floatLq(q);
    layer_t *z = batchNorm1dLayerInit(
        &(batchNorm1dInit_t){.numChannels = 2, .momentumMode = BN_MOMENTUM_VALUE, .momentum = 0.0f},
        &lq);
    layer_t *o = batchNorm1dLayerInit(
        &(batchNorm1dInit_t){.numChannels = 2, .momentumMode = BN_MOMENTUM_VALUE, .momentum = 1.0f},
        &lq);
    float mz = z->config->batchNorm1d->momentum;
    float mo = o->config->batchNorm1d->momentum;
    freeBatchNorm1dLayer(o);
    freeBatchNorm1dLayer(z);
    freeQuantization(q);
    TEST_ASSERT_EQUAL_FLOAT(0.0f, mz);
    TEST_ASSERT_EQUAL_FLOAT(1.0f, mo);
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(testGoldTrainForwardRank2);
    RUN_TEST(testGoldTrainForwardRank3);
    RUN_TEST(testGoldTrainForwardMinimalBatchUsesUnbiasedRunningVar);
    RUN_TEST(testGoldTrainForwardNoAffine);
    RUN_TEST(testGoldTrainForwardMomentum03);
    RUN_TEST(testMomentumOneCopiesBatchStatistics);
    RUN_TEST(testGoldTrainForwardCumulativeOverThreeSteps);
    RUN_TEST(testGoldTrainForwardTinyVarianceKeepsEpsInsideSqrt);
    RUN_TEST(testEvalForwardTinyRunningVarKeepsEpsInsideSqrt);
    RUN_TEST(testNoTrackUsesBatchStatisticsInTrainingAndEval);
    RUN_TEST(testEvalForwardUsesRunningStatsAndWritesNothing);
    RUN_TEST(testFrozenTrainingForwardUsesRunningStatsAndWritesNothing);
    RUN_TEST(testMomentumZeroKeepsRunningStatsButCounts);
    RUN_TEST(testCounterSaturatesAtMaxAndStaysFinite);
    RUN_TEST(testCalcOutputShapeIsIdentity);
    RUN_TEST(testForwardTrainingRejectsSingleRowRank2);
    RUN_TEST(testForwardTrainingRejectsRank3SingleValuePerChannel);
    RUN_TEST(testForwardTrainingRejectsEmptyBatch);
    RUN_TEST(testForwardTrainingRejectsZeroLengthTime);
    RUN_TEST(testForwardEvalAcceptsSingleRowAndEmpty);
    RUN_TEST(testForwardEvalUntrackedRejectsSingleSample);
    RUN_TEST(testForwardTrainingFrozenUntrackedRejectsSingleRow);
    RUN_TEST(testForwardRejectsRank1);
    RUN_TEST(testForwardRejectsRank4);
    RUN_TEST(testForwardRejectsTransposedInput);
    RUN_TEST(testForwardRejectsWrongChannelCount);
    RUN_TEST(testForwardRejectsNonFloat32Input);
    RUN_TEST(testForwardRejectsNonFloat32Math);
    RUN_TEST(testInitRejectsGammaWithoutBeta);
    RUN_TEST(testInitRejectsRunningMeanWithoutRunningVar);
    RUN_TEST(testForwardRejectsNonFloat32RunningBuffer);
    RUN_TEST(testForwardRejectsShortRunningBuffer);
    RUN_TEST(testForwardRejectsMismatchedOutputShape);
    RUN_TEST(testForwardRejectsTransposedOutput);
    RUN_TEST(testForwardRejectsNonFloat32Output);
    RUN_TEST(testGoldTrainBackwardRank2);
    RUN_TEST(testGoldTrainBackwardRank3);
    RUN_TEST(testGoldTrainBackwardMinimalBatch);
    RUN_TEST(testGoldTrainBackwardNoAffine);
    RUN_TEST(testGoldEvalBackward);
    RUN_TEST(testFrozenBackwardUsesRunningStatsAndNoGrads);
    RUN_TEST(testGradsOnlyBackwardMatchesFullAndAccumulates);
    RUN_TEST(testBackwardTrainingRejectsSingleRow);
    RUN_TEST(testBackwardRejectsLossShapeMismatch);
    RUN_TEST(testBackwardRejectsShortGammaGrad);
    RUN_TEST(testFactoryZeroInitGivesPyTorchDefaults);
    RUN_TEST(testFactoryOptionsNoAffineNoStatsCumulativeFrozen);
    RUN_TEST(testFactoryRejectsInvalidInit);
    RUN_TEST(testFactoryRejectsNonFloat32Slots);
    RUN_TEST(testFactoryRejectsNullPointers);
    RUN_TEST(testFactoryAcceptsExplicitMomentumZeroAndOne);
    return UNITY_END();
}
