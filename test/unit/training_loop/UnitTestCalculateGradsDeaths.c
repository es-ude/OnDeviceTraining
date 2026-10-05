#define SOURCE_FILE "UNIT_TEST_CALCULATE_GRADS_DEATHS"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "AsanDeath.h"
#include "CalculateGradsSequential.h"
#include "DeathTest.h"
#include "Layer.h"
#include "LossFunction.h"
#include "Quantization.h"
#include "RematPlan.h"
#include "RematScheduler.h"
#include "RematTestDecorators.h"
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

/* ---- decorator rows that break the row contract (remat D24) ---- */

/* One decorator table serves both rows (RematTestDecorators.h), so each death
 * names the decorator, and its text is identical on ARENA and HEAP. The real
 * table is restored before the parent's deinit. */
static void assertDecoratedRunExits(rowInit_t init, const rematSchedulerFunctions_t *fns,
                                    const char *violation) {
    fixture_t f;
    buildMlpModel(&f);
    makeLabel(2);
    rematScheduler_t s = init(&f, NULL);
    s.fns = fns;
    ASSERT_EXITS_WITH_OUTPUT(1, violation, trainOn(&f, &s));
    s.fns = &rematSchedulerFunctions[s.type];
    freeFixture(&f, &s);
}

static void assertDecoratedRunExitsOnBothRows(const rematSchedulerFunctions_t *fns,
                                              const char *violation) {
    assertDecoratedRunExits(initArena, fns, violation);
    assertDecoratedRunExits(initHeap, fns, violation);
}

/* Releases ACT 1 once FORWARD(1) is done; the ReLU's BACKWARD(1) still reads
 * it. */
static void releasingDone(rematScheduler_t *s, const rematStep_t *st) {
    decoratedDone(s, st);
    if (st->kind == REMAT_STEP_FORWARD && st->layer == 1u) {
        rematWireRelease(s->wires, rematActId(s->wires, 1));
    }
}

static const rematSchedulerFunctions_t g_releasing = {.name = "releasing",
                                                      .begin = decoratedBegin,
                                                      .next = decoratedNext,
                                                      .done = releasingDone,
                                                      .end = decoratedEnd,
                                                      .deinit = decoratedDeinit};

void testARowThatReleasesAWireEarlyExitsOnBothRows(void) {
    assertDecoratedRunExitsOnBothRows(
        &g_releasing,
        "remat[releasing]: step #6 BACKWARD(layer 1) violates 'operand not resident: in ACT 1'");
}

/* Hands out BACKWARD(2) with ACT 2 released and bound again to the same bytes:
 * resident, but not under the binding FORWARD(1) produced it in. A bare second
 * bind would exit in rematWireBind ("is already bound"). */
static bool rebindingNext(rematScheduler_t *s, rematStep_t *st) {
    if (!decoratedNext(s, st)) {
        return false;
    }
    if (st->kind == REMAT_STEP_BACKWARD && st->layer == 2u) {
        uint16_t act2 = rematActId(s->wires, 2);
        uint8_t *bytes = rematWireHdr(s->wires, act2)->data;
        rematWireRelease(s->wires, act2);
        rematWireBind(s->wires, act2, bytes);
    }
    return true;
}

static const rematSchedulerFunctions_t g_rebinding = {.name = "rebinding",
                                                      .begin = decoratedBegin,
                                                      .next = rebindingNext,
                                                      .done = decoratedDone,
                                                      .end = decoratedEnd,
                                                      .deinit = decoratedDeinit};

void testARowThatRebindsWithoutProducingExitsOnBothRows(void) {
    assertDecoratedRunExitsOnBothRows(&g_rebinding,
                                      "remat[rebinding]: step #5 BACKWARD(layer 2) violates "
                                      "'operand stale: ACT 2 rebound since produced' (produced at "
                                      "bindGen 1, bound now at bindGen 2)");
}

/* Large enough for ACT 1 (3 floats), which VERIFY poisons at bind. */
static uint64_t g_strayBytes[2];

static void leakyEnd(rematScheduler_t *s) {
    decoratedEnd(s);
    rematWireBind(s->wires, rematActId(s->wires, 1), (uint8_t *)g_strayBytes);
}

static const rematSchedulerFunctions_t g_leaky = {.name = "leaky",
                                                  .begin = decoratedBegin,
                                                  .next = decoratedNext,
                                                  .done = decoratedDone,
                                                  .end = leakyEnd,
                                                  .deinit = decoratedDeinit};

void testARowThatLeavesAWireResidentAtEndExitsOnBothRows(void) {
    assertDecoratedRunExitsOnBothRows(
        &g_leaky, "remat[leaky]: after rematEnd violates 'wire left resident after end: ACT 1'");
}

/* Fetches the real last step and answers it through the real done(), then
 * reports the stream complete: the row's walk is complete, so rematEnd's own
 * check passes, and only the driver's rematCheckFinish sees the gap. */
static bool eatingNext(rematScheduler_t *s, rematStep_t *st) {
    if (s->walk.step + 1u == s->plan->train.numSteps) {
        rematStep_t eaten;
        (void)decoratedNext(s, &eaten);
        decoratedDone(s, &eaten);
        return false;
    }
    return decoratedNext(s, st);
}

static const rematSchedulerFunctions_t g_eating = {.name = "eating",
                                                   .begin = decoratedBegin,
                                                   .next = eatingNext,
                                                   .done = decoratedDone,
                                                   .end = decoratedEnd,
                                                   .deinit = decoratedDeinit};

void testARowThatEatsTheLastStepExitsOnBothRows(void) {
    assertDecoratedRunExitsOnBothRows(
        &g_eating, "remat[eating]: stream of 7 steps violates 'incomplete stream: missing "
                   "BACKWARD(0)'");
}

/* rematBegin's arguments for the re-entering row; set by its test. */
static fixture_t *g_reentered;

static bool reenteringNext(rematScheduler_t *s, rematStep_t *st) {
    rematBegin(s, g_reentered->model, g_reentered->n, defaultLossConfig(g_reentered->lt),
               g_reentered->x);
    return decoratedNext(s, st);
}

static const rematSchedulerFunctions_t g_reentering = {.name = "reentering",
                                                       .begin = decoratedBegin,
                                                       .next = reenteringNext,
                                                       .done = decoratedDone,
                                                       .end = decoratedEnd,
                                                       .deinit = decoratedDeinit};

static void assertReenteredRunExits(rowInit_t init) {
    fixture_t f;
    buildMlpModel(&f);
    makeLabel(2);
    g_reentered = &f;
    rematScheduler_t s = init(&f, NULL);
    s.fns = &g_reentering;
    ASSERT_EXITS_WITH_OUTPUT(1, "rematBegin: scheduler 'reentering' re-entered", trainOn(&f, &s));
    s.fns = &rematSchedulerFunctions[s.type];
    freeFixture(&f, &s);
}

void testARowThatReentersBeginExitsOnBothRows(void) {
    assertReenteredRunExits(initArena);
    assertReenteredRunExits(initHeap);
}

/* ---- the inherited seed checks before it writes (remat D54) ---- */

/* Linear 2 -> 8, then a Quantization to a per-tensor BFP wire, under MSE: the
 * last wire ACT 2 is BFP (8 B, expCapacity 1), so the seed GRAD 2 inherits
 * its config from the live ACT 2 when its range opens at LOSS_BACKWARD. */
static void buildBfpSeedModel(fixture_t *f) {
    memset(f, 0, sizeof *f);
    initBfpQConfigInto(8, 8, HALF_AWAY, f->bfpExponent, &f->bfpQc);
    f->bfpQ = (quantization_t){.type = BFP, .qConfig = &f->bfpQc};
    f->model[0] = makeLinear(2, 8, false);
    f->model[1] = makeQuant(&f->bfpQ, &g_floatQ);
    f->n = 2;
    f->lt = MSE;
    f->x = makeResidentInput(&f->in, (size_t[]){1, 2}, 2);
}

/* Regroups the live ACT 2 to 4 groups of 2 just before LOSS_BACKWARD's range
 * opens (fields only: no exponent byte is written). */
static bool regroupingNext(rematScheduler_t *s, rematStep_t *st) {
    const rematProgram_t *p = &s->plan->train;
    if (s->walk.step < p->numSteps && p->steps[s->walk.step].kind == REMAT_STEP_LOSS_BACKWARD) {
        bfpQConfig_t *act2 = rematActHdr(s->wires, 2)->quantization->qConfig;
        act2->numGroups = 4;
        act2->groupSize = 2;
    }
    return decoratedNext(s, st);
}

static const rematSchedulerFunctions_t g_regrouping = {.name = "regrouping",
                                                       .begin = decoratedBegin,
                                                       .next = regroupingNext,
                                                       .done = decoratedDone,
                                                       .end = decoratedEnd,
                                                       .deinit = decoratedDeinit};

/* The seed's exponent byte ends its slab block, so a write before the check
 * is an ASan report (exit 86 through the callback), never the named exit. */
static void trainUnderTheAsanCallback(fixture_t *f, rematScheduler_t *s) {
    odtInstallAsanDeathExit();
    trainOn(f, s);
}

static void assertRegroupedSeedExits(rowInit_t init) {
    fixture_t f;
    buildBfpSeedModel(&f);
    makeLabel(8);
    rematScheduler_t s = init(&f, NULL);
    s.fns = &g_regrouping;
    ASSERT_EXITS_WITH_OUTPUT(1,
                             "remat: wire GRAD 2 needs 4 BFP exponent groups, above its "
                             "expCapacity 1",
                             trainUnderTheAsanCallback(&f, &s));
    s.fns = &rematSchedulerFunctions[s.type];
    freeFixture(&f, &s);
}

void testWireBindInheritedGradChecksBeforeWrite(void) {
    assertRegroupedSeedExits(initArena);
    assertRegroupedSeedExits(initHeap);
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
    RUN_TEST(testARowThatReleasesAWireEarlyExitsOnBothRows);
    RUN_TEST(testARowThatRebindsWithoutProducingExitsOnBothRows);
    RUN_TEST(testARowThatLeavesAWireResidentAtEndExitsOnBothRows);
    RUN_TEST(testARowThatEatsTheLastStepExitsOnBothRows);
    RUN_TEST(testARowThatReentersBeginExitsOnBothRows);
    RUN_TEST(testWireBindInheritedGradChecksBeforeWrite);
    return UNITY_END();
}
