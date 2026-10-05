#define SOURCE_FILE "UNIT_TEST_CALCULATE_GRADS_DEATHS"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "CalculateGradsSequential.h"
#include "DeathTest.h"
#include "Layer.h"
#include "LossFunction.h"
#include "RematPlan.h"
#include "RematScheduler.h"
#include "RematTestFixtures.h"
#include "Tensor.h"
#include "TrainingCall.h"
#include "TrainingLoopApi.h"
#include "unity.h"

void setUp(void) {}
void tearDown(void) {}

/* Every death here runs one training call through the real driver on a
 * caller's scheduler, and is asserted on both rows: the checker and the key
 * name the rule, the row only prefixes it. */

/* Linear 4 -> 3 -> ReLU -> Linear 3 -> 2 under MSE: n = 3, deepest 0, top 2.
 * Steps: FORWARD 0, 1, 2 (#0-#2), LOSS_FORWARD (#3), LOSS_BACKWARD (#4),
 * BACKWARD 2, 1, 0 (#5-#7). The input bytes are zeroed: a death fires
 * before any value matters. */
static void buildMlpModel(fixture_t *f) {
    memset(f, 0, sizeof *f);
    f->model[0] = makeLinear(4, 3, false);
    f->model[1] = makeRelu(&g_floatQ);
    f->model[2] = makeLinear(3, 2, false);
    f->n = 3;
    f->lt = MSE;
    f->x = makeResidentInput(&f->in, (size_t[]){1, 4}, 2);
}

/* The MSE label of the call; one per binary, re-made by every test. */
static inputLike_t g_labelIn;
static tensor_t *g_label;

static void makeLabel(size_t outFeatures) {
    memset(&g_labelIn, 0, sizeof g_labelIn);
    g_label = makeResidentInput(&g_labelIn, (size_t[]){1, outFeatures}, 2);
}

static void trainOn(fixture_t *f, rematScheduler_t *s) {
    freeTrainingStats(calculateGradsSequential(f->model, f->n, defaultLossConfig(f->lt),
                                               REDUCTION_MEAN, f->x, g_label,
                                               &(trainingCall_t){.remat = s}));
}

/* ---- tampered plans: the order and range rules fire first ---- */

/* Edited through RematPlan.h after init, which bypasses the init verifier,
 * so the run-time rule is the one that fires. */
typedef void (*tamperFn_t)(rematScheduler_t *s);

static void assertTamperedRunExits(rowInit_t init, tamperFn_t tamper, const char *violation) {
    fixture_t f;
    buildMlpModel(&f);
    makeLabel(2);
    rematScheduler_t s = init(&f, NULL);
    tamper(&s);
    ASSERT_EXITS_WITH_OUTPUT(1, violation, trainOn(&f, &s));
    freeFixture(&f, &s);
}

static void swapTheFirstTwoSteps(rematScheduler_t *s) {
    rematStep_t *steps = s->plan->train.steps;
    rematStep_t first = steps[0];
    steps[0] = steps[1];
    steps[1] = first;
}

void testASwappedStepExitsOnBothRows(void) {
    assertTamperedRunExits(
        initArena, swapTheFirstTwoSteps,
        "remat[arena]: step #0 FORWARD(layer 1) violates 'forward order: expected FORWARD(0)'");
    assertTamperedRunExits(
        initHeap, swapTheFirstTwoSteps,
        "remat[heap]: step #0 FORWARD(layer 1) violates 'forward order: expected FORWARD(0)'");
}

static void duplicateTheFirstStep(rematScheduler_t *s) {
    s->plan->train.steps[1] = s->plan->train.steps[0];
}

void testADuplicatedStepExitsOnBothRows(void) {
    assertTamperedRunExits(
        initArena, duplicateTheFirstStep,
        "remat[arena]: step #1 FORWARD(layer 0) violates 'forward order: expected FORWARD(1)'");
    assertTamperedRunExits(
        initHeap, duplicateTheFirstStep,
        "remat[heap]: step #1 FORWARD(layer 0) violates 'forward order: expected FORWARD(1)'");
}

/* The stream ends before BACKWARD(0); the driver checks it before rematEnd,
 * so the checker names the missing step before the row's own walk check. */
static void dropTheLastStep(rematScheduler_t *s) {
    s->plan->train.numSteps--;
}

void testADroppedLastStepExitsOnBothRows(void) {
    assertTamperedRunExits(
        initArena, dropTheLastStep,
        "remat[arena]: stream of 7 steps violates 'incomplete stream: missing BACKWARD(0)'");
    assertTamperedRunExits(
        initHeap, dropTheLastStep,
        "remat[heap]: stream of 7 steps violates 'incomplete stream: missing BACKWARD(0)'");
}

static void retargetTheFirstStepsKind(rematScheduler_t *s) {
    s->plan->train.steps[0].kind = REMAT_STEP_BACKWARD;
}

void testARetargetedKindExitsOnBothRows(void) {
    assertTamperedRunExits(
        initArena, retargetTheFirstStepsKind,
        "remat[arena]: step #0 BACKWARD(layer 0) violates 'backward before loss-backward'");
    assertTamperedRunExits(
        initHeap, retargetTheFirstStepsKind,
        "remat[heap]: step #0 BACKWARD(layer 0) violates 'backward before loss-backward'");
}

/* Step #5 is BACKWARD(2), the first backward. */
static void retargetTheFirstBackwardsLayer(rematScheduler_t *s) {
    s->plan->train.steps[5].layer = 1u;
}

void testARetargetedLayerExitsOnBothRows(void) {
    assertTamperedRunExits(
        initArena, retargetTheFirstBackwardsLayer,
        "remat[arena]: step #5 BACKWARD(layer 1) violates 'backward order: expected BACKWARD(2)'");
    assertTamperedRunExits(
        initHeap, retargetTheFirstBackwardsLayer,
        "remat[heap]: step #5 BACKWARD(layer 1) violates 'backward order: expected BACKWARD(2)'");
}

static void sendTheFirstStepOutOfRange(rematScheduler_t *s) {
    s->plan->train.steps[0].layer = 7u;
}

void testAnOutOfRangeLayerExitsOnBothRows(void) {
    assertTamperedRunExits(
        initArena, sendTheFirstStepOutOfRange,
        "remat[arena]: step #0 FORWARD(layer 7) violates 'layer out of range' (n = 3)");
    assertTamperedRunExits(
        initHeap, sendTheFirstStepOutOfRange,
        "remat[heap]: step #0 FORWARD(layer 7) violates 'layer out of range' (n = 3)");
}

static size_t rangeOf(const rematProgram_t *p, uint16_t wire) {
    for (size_t r = 0; r < p->numRanges; r++) {
        if (p->ranges[r].wire == wire) {
            return r;
        }
    }
    TEST_FAIL_MESSAGE("no range for the wire");
    return 0;
}

/* ARENA only: HEAP places each range in its own block. Under STORE_ALL both
 * wires are co-live at the step that touches them together. */
static void placeAct2OntoAct1(rematScheduler_t *s) {
    const rematProgram_t *p = &s->plan->train;
    s->row.arena.offsets[rangeOf(p, rematActId(s->wires, 2))] =
        s->row.arena.offsets[rangeOf(p, rematActId(s->wires, 1))];
}

void testArenaOffsetsOverlappingAForwardsInAndOutExit(void) {
    assertTamperedRunExits(initArena, placeAct2OntoAct1,
                           "remat[arena]: step #1 FORWARD(layer 1) violates 'operands share bytes: "
                           "in/out' (ACT 1 and ACT 2)");
}

/* BACKWARD(2) reads its input ACT 2 (Linear's weight grad) and writes its dx
 * GRAD 2: the roles print as in/out. */
static void placeGrad2OntoAct2(rematScheduler_t *s) {
    const rematProgram_t *p = &s->plan->train;
    s->row.arena.offsets[rangeOf(p, rematGradId(s->wires, 2))] =
        s->row.arena.offsets[rangeOf(p, rematActId(s->wires, 2))];
}

void testArenaOffsetsOverlappingABackwardsInAndDxExit(void) {
    assertTamperedRunExits(initArena, placeGrad2OntoAct2,
                           "remat[arena]: step #5 BACKWARD(layer 2) violates 'operands share "
                           "bytes: in/out' (ACT 2 and GRAD 2)");
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(testASwappedStepExitsOnBothRows);
    RUN_TEST(testADuplicatedStepExitsOnBothRows);
    RUN_TEST(testADroppedLastStepExitsOnBothRows);
    RUN_TEST(testARetargetedKindExitsOnBothRows);
    RUN_TEST(testARetargetedLayerExitsOnBothRows);
    RUN_TEST(testAnOutOfRangeLayerExitsOnBothRows);
    RUN_TEST(testArenaOffsetsOverlappingAForwardsInAndOutExit);
    RUN_TEST(testArenaOffsetsOverlappingABackwardsInAndDxExit);
    return UNITY_END();
}
