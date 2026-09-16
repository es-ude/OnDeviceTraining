#define SOURCE_FILE "BS-SCHED-UTEST"

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include "ArithmeticType.h"
#include "BsScheduler.h"
#include "DataLoader.h"
#include "DeathTest.h"
#include "Optimizer.h"
#include "Sgd.h"
#include "unity.h"

/* Signature contracts (design A1): init fns take (sched, dataLoader,
 * optimizerOrNull, <params...>, maxBatchSize); step takes the scheduler only.
 * Compile-time pins. */
_Static_assert(_Generic(&stepBsInit,
                   void (*)(bsScheduler_t *, dataLoader_t *, optimizer_t *, size_t, float,
                            size_t): 1,
                   default: 0),
               "stepBsInit must be (sched, dataLoader, optimizerOrNull, stepSize, gamma, "
               "maxBatchSize)");
_Static_assert(_Generic(&exponentialBsInit,
                   void (*)(bsScheduler_t *, dataLoader_t *, optimizer_t *, float, size_t): 1,
                   default: 0),
               "exponentialBsInit must be (sched, dataLoader, optimizerOrNull, gamma, "
               "maxBatchSize)");
_Static_assert(_Generic(&bsSchedulerStep, void (*)(bsScheduler_t *): 1, default: 0),
               "bsSchedulerStep must be (sched)");

void setUp() {}
void tearDown() {}

/* Loader stub: the scheduler only reads batchSize at init and writes it on
 * every step; nothing else of dataLoader_t is touched. */
static size_t stubDatasetSize(void) {
    return 1000;
}

static dataLoader_t makeLoader(uint16_t batchSize) {
    dataLoader_t dl = {0};
    dl.batchSize = batchSize;
    dl.getDatasetSize = stubDatasetSize;
    return dl;
}

/* Hand-assembled SGD optimizer on the stack (UnitTestLrScheduler idiom): the
 * scheduler needs only the SGD_M vtable's getLr/setLr, no parameters/states. */
static sgd_t g_sgd;
static optimImpl_t g_impl;

static optimizer_t makeSgdOptimizer(float lr) {
    sgdInit(&g_sgd, lr, 0.9f, 0.0f,
            (arithmetic_t){.type = ARITH_FLOAT32, .roundingMode = HALF_AWAY});
    g_impl.sgd = &g_sgd;
    return (optimizer_t){
        .type = SGD_M, .impl = &g_impl, .parameter = NULL, .states = NULL, .sizeStates = 0};
}

static float currentLr(optimizer_t *optim) {
    return optimizerFunctions[optim->type].getLr(optim);
}

/* Steps `count` times, capturing dataLoader->batchSize after each step. */
static void runBatchSequence(bsScheduler_t *sched, const dataLoader_t *dl, size_t *out,
                             size_t count) {
    for (size_t i = 0; i < count; i++) {
        bsSchedulerStep(sched);
        out[i] = dl->batchSize;
    }
}

/* Per-element compare instead of TEST_ASSERT_EQUAL_size_t_ARRAY: Unity's
 * UINT array style reads UNITY_INT_WIDTH/8 == 4 bytes per element on an LP64
 * host, so on a size_t[] it only ever inspects the first ceil(N/2) elements —
 * the cap/clamp steps at the END of these sequences would go unchecked. The
 * scalar size_t assert compares full 64-bit UNITY_INT values. */
static void assertBatchSequence(const size_t *expected, const size_t *got, size_t count) {
    for (size_t i = 0; i < count; i++) {
        char msg[48];
        snprintf(msg, sizeof msg, "batch mismatch at step %zu", i + 1);
        TEST_ASSERT_EQUAL_size_t_MESSAGE(expected[i], got[i], msg);
    }
}

/* ---- closed forms (batch = clamp(round(baseBs / gamma^...), 1, max)) ---- */

void testExponentialBsGrowsAndCaps(void) {
    /* b0=16, gamma=0.5, max=100: 32, 64, 128->100, 256->100 */
    dataLoader_t dl = makeLoader(16);
    bsScheduler_t sched;
    exponentialBsInit(&sched, &dl, NULL, 0.5f, 100);
    size_t got[4];
    runBatchSequence(&sched, &dl, got, 4);
    const size_t expected[4] = {32, 64, 100, 100};
    assertBatchSequence(expected, got, 4);
}

void testStepBsHoldsWithinAStep(void) {
    /* b0=16, gamma=0.5, stepSize=2: floor(k/2) = 0,1,1,2,2 -> 16,32,32,64,64 */
    dataLoader_t dl = makeLoader(16);
    bsScheduler_t sched;
    stepBsInit(&sched, &dl, NULL, 2, 0.5f, 100);
    size_t got[5];
    runBatchSequence(&sched, &dl, got, 5);
    const size_t expected[5] = {16, 32, 32, 64, 64};
    assertBatchSequence(expected, got, 5);
}

void testGrowingTargetRoundsToNearest(void) {
    /* b0=1, gamma=0.8: targets 1.25, 1.5625, 1.953, 2.441, 3.052 -> 1,2,2,2,3 */
    dataLoader_t dl = makeLoader(1);
    bsScheduler_t sched;
    exponentialBsInit(&sched, &dl, NULL, 0.8f, 100);
    size_t got[5];
    runBatchSequence(&sched, &dl, got, 5);
    const size_t expected[5] = {1, 2, 2, 2, 3};
    assertBatchSequence(expected, got, 5);
}

void testTieRoundsHalfAwayFromZero(void) {
    /* b0=5, gamma=2, max=5: targets 2.5, 1.25, 0.625 -> 3 (C round, not
     * banker's 2), 1, 1 */
    dataLoader_t dl = makeLoader(5);
    bsScheduler_t sched;
    exponentialBsInit(&sched, &dl, NULL, 2.0f, 5);
    size_t got[3];
    runBatchSequence(&sched, &dl, got, 3);
    const size_t expected[3] = {3, 1, 1};
    assertBatchSequence(expected, got, 3);
}

void testGammaAboveOneShrinksAndClampsAtOne(void) {
    /* b0=4, gamma=2: targets 2, 1, 0.5 (rounds to 1), 0.25 (rounds to 0 ->
     * clamped to 1). The 4th step is the one that exercises the lower clamp. */
    dataLoader_t dl = makeLoader(4);
    bsScheduler_t sched;
    exponentialBsInit(&sched, &dl, NULL, 2.0f, 4);
    size_t got[4];
    runBatchSequence(&sched, &dl, got, 4);
    const size_t expected[4] = {2, 1, 1, 1};
    assertBatchSequence(expected, got, 4);
}

/* ---- LR compensation ---- */

void testCompensationKeepsLrOverBatchOnTheExactTrajectory(void) {
    /* b0=1, gamma=0.8, baseLr=0.1: lr_k = 0.1 * b_k / x_k
     *   k=1: x=1.25    b=1 -> 0.08
     *   k=2: x=1.5625  b=2 -> 0.128
     *   k=3: x=1.953.. b=2 -> 0.1024
     *   k=4: x=2.441.. b=2 -> 0.08192
     *   k=5: x=3.051.. b=3 -> 0.098304
     * and lr_k / b_k == 0.1 * 0.8^k regardless of rounding. */
    dataLoader_t dl = makeLoader(1);
    optimizer_t optim = makeSgdOptimizer(0.1f);
    bsScheduler_t sched;
    exponentialBsInit(&sched, &dl, &optim, 0.8f, 100);
    const float expectedLr[5] = {0.08f, 0.128f, 0.1024f, 0.08192f, 0.098304f};
    float gotLr[5];
    size_t gotBatch[5];
    for (size_t k = 0; k < 5; k++) {
        bsSchedulerStep(&sched);
        gotLr[k] = currentLr(&optim);
        gotBatch[k] = dl.batchSize;
    }
    for (size_t k = 0; k < 5; k++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-6f, expectedLr[k], gotLr[k]);
        float ratio = gotLr[k] / (float)gotBatch[k];
        float lrSchedulerRatio = 0.1f * powf(0.8f, (float)(k + 1));
        TEST_ASSERT_FLOAT_WITHIN(1e-6f, lrSchedulerRatio, ratio);
    }
}

void testCompensationAfterTheCapIsPlainLrDecay(void) {
    /* b0=16, gamma=0.5, max=32, baseLr=0.1: x = 32, 64, 128; b = 32, 32, 32
     * -> lr = 0.1, 0.05, 0.025 (power-of-two scaling: exact floats). */
    dataLoader_t dl = makeLoader(16);
    optimizer_t optim = makeSgdOptimizer(0.1f);
    bsScheduler_t sched;
    exponentialBsInit(&sched, &dl, &optim, 0.5f, 32);
    float gotLr[3];
    for (size_t k = 0; k < 3; k++) {
        bsSchedulerStep(&sched);
        gotLr[k] = currentLr(&optim);
    }
    TEST_ASSERT_EQUAL_FLOAT(0.1f, gotLr[0]);
    TEST_ASSERT_EQUAL_FLOAT(0.05f, gotLr[1]);
    TEST_ASSERT_EQUAL_FLOAT(0.025f, gotLr[2]);
    TEST_ASSERT_EQUAL_size_t(32, dl.batchSize);
}

void testBaseLrIsCapturedAtInitNotAtStep(void) {
    /* init at 0.1, then an external setLr(0.7); the step must write from the
     * captured 0.1 (-> 0.08), never from the mutated 0.7 (-> 0.56). */
    dataLoader_t dl = makeLoader(1);
    optimizer_t optim = makeSgdOptimizer(0.1f);
    bsScheduler_t sched;
    exponentialBsInit(&sched, &dl, &optim, 0.8f, 100);
    optimizerFunctions[SGD_M].setLr(&optim, 0.7f); /* sabotage */
    bsSchedulerStep(&sched);
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 0.08f, currentLr(&optim));
}

void testNullOptimizerNeverTouchesTheLr(void) {
    dataLoader_t dl = makeLoader(1);
    optimizer_t optim = makeSgdOptimizer(0.1f); /* exists, but NOT handed over */
    bsScheduler_t sched;
    exponentialBsInit(&sched, &dl, NULL, 0.8f, 100);
    for (size_t k = 0; k < 3; k++) {
        bsSchedulerStep(&sched);
    }
    TEST_ASSERT_EQUAL_FLOAT(0.1f, currentLr(&optim));
    TEST_ASSERT_EQUAL_FLOAT(0.0f, sched.baseLr);
    TEST_ASSERT_NULL(sched.optimizer);
}

/* ---- lastEpoch / init state ---- */

void testInitCapturesBaseBsAndDoesNotStep(void) {
    dataLoader_t dl = makeLoader(16);
    bsScheduler_t sched;
    stepBsInit(&sched, &dl, NULL, 3, 0.5f, 100);
    TEST_ASSERT_EQUAL_size_t(0, sched.lastEpoch);
    TEST_ASSERT_EQUAL_size_t(16, sched.baseBs);
    TEST_ASSERT_EQUAL_size_t(100, sched.maxBatchSize);
    TEST_ASSERT_EQUAL_size_t(16, dl.batchSize); /* unchanged until the first step */
    bsSchedulerStep(&sched);
    TEST_ASSERT_EQUAL_size_t(1, sched.lastEpoch);
    TEST_ASSERT_EQUAL_size_t(16, dl.batchSize); /* floor(1/3) == 0: still baseBs */
}

/* ---- init-time guards (PRINT_ERROR + exit(1)) ---- */

void testInitRejectsZeroGamma(void) {
    dataLoader_t dl = makeLoader(16);
    bsScheduler_t sched;
    ASSERT_EXITS_WITH_FAILURE(exponentialBsInit(&sched, &dl, NULL, 0.0f, 100));
}

void testInitRejectsNegativeGamma(void) {
    dataLoader_t dl = makeLoader(16);
    bsScheduler_t sched;
    ASSERT_EXITS_WITH_FAILURE(stepBsInit(&sched, &dl, NULL, 1, -0.5f, 100));
}

void testInitRejectsNanGamma(void) {
    dataLoader_t dl = makeLoader(16);
    bsScheduler_t sched;
    ASSERT_EXITS_WITH_FAILURE(exponentialBsInit(&sched, &dl, NULL, NAN, 100));
}

void testInitRejectsInfGamma(void) {
    dataLoader_t dl = makeLoader(16);
    bsScheduler_t sched;
    ASSERT_EXITS_WITH_FAILURE(stepBsInit(&sched, &dl, NULL, 1, INFINITY, 100));
}

void testStepBsInitRejectsZeroStepSize(void) {
    dataLoader_t dl = makeLoader(16);
    bsScheduler_t sched;
    ASSERT_EXITS_WITH_FAILURE(stepBsInit(&sched, &dl, NULL, 0, 0.5f, 100));
}

void testInitRejectsMaxBelowInitialBatch(void) {
    dataLoader_t dl = makeLoader(16);
    bsScheduler_t sched;
    ASSERT_EXITS_WITH_FAILURE(exponentialBsInit(&sched, &dl, NULL, 0.5f, 15));
}

void testInitRejectsMaxAboveUint16(void) {
    dataLoader_t dl = makeLoader(16);
    bsScheduler_t sched;
    ASSERT_EXITS_WITH_FAILURE(exponentialBsInit(&sched, &dl, NULL, 0.5f, (size_t)UINT16_MAX + 1));
}

void testInitRejectsNullDataLoader(void) {
    bsScheduler_t sched;
    ASSERT_EXITS_WITH_FAILURE(exponentialBsInit(&sched, NULL, NULL, 0.5f, 100));
}

int main() {
    UNITY_BEGIN();
    RUN_TEST(testExponentialBsGrowsAndCaps);
    RUN_TEST(testStepBsHoldsWithinAStep);
    RUN_TEST(testGrowingTargetRoundsToNearest);
    RUN_TEST(testTieRoundsHalfAwayFromZero);
    RUN_TEST(testGammaAboveOneShrinksAndClampsAtOne);
    RUN_TEST(testCompensationKeepsLrOverBatchOnTheExactTrajectory);
    RUN_TEST(testCompensationAfterTheCapIsPlainLrDecay);
    RUN_TEST(testBaseLrIsCapturedAtInitNotAtStep);
    RUN_TEST(testNullOptimizerNeverTouchesTheLr);
    RUN_TEST(testInitCapturesBaseBsAndDoesNotStep);
    RUN_TEST(testInitRejectsZeroGamma);
    RUN_TEST(testInitRejectsNegativeGamma);
    RUN_TEST(testInitRejectsNanGamma);
    RUN_TEST(testInitRejectsInfGamma);
    RUN_TEST(testStepBsInitRejectsZeroStepSize);
    RUN_TEST(testInitRejectsMaxBelowInitialBatch);
    RUN_TEST(testInitRejectsMaxAboveUint16);
    RUN_TEST(testInitRejectsNullDataLoader);
    return UNITY_END();
}
