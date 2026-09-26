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

#include "DeathTest.h"
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

/* track off: batch statistics in training AND in eval (PyTorch parity). */
void testNoTrackUsesBatchStatisticsInTrainingAndEval(void) {
    for (int training = 0; training <= 1; training++) {
        bnFixture_t f;
        bnFixtureInit(&f, 3, true, false, gamma_bn_noTrackRank2, beta_bn_noTrackRank2, NULL, NULL,
                      BN_MOMENTUM_DEFAULT, 0.0f);
        f.cfg.training = training != 0;
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

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(testGoldTrainForwardRank2);
    RUN_TEST(testGoldTrainForwardRank3);
    RUN_TEST(testGoldTrainForwardMinimalBatchUsesUnbiasedRunningVar);
    RUN_TEST(testGoldTrainForwardNoAffine);
    RUN_TEST(testGoldTrainForwardMomentum03);
    RUN_TEST(testMomentumOneCopiesBatchStatistics);
    RUN_TEST(testGoldTrainForwardCumulativeOverThreeSteps);
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
    return UNITY_END();
}
