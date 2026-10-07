#define SOURCE_FILE "UNIT_TEST_REMAT_PLAN"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "AsanDeath.h"
#include "Common.h"
#include "Conv1d.h"
#include "Conv1dApi.h"
#include "Conv1dTransposed.h"
#include "DeathTest.h"
#include "Deserialize.h"
#include "FlattenApi.h"
#include "GroupNorm.h"
#include "Layer.h"
#include "LayerConfigAccess.h"
#include "LayerNorm.h"
#include "LayerNormApi.h"
#include "LayerQuant.h"
#include "Linear.h"
#include "LinearApi.h"
#include "LossFunction.h"
#include "Pool1dApi.h"
#include "QuantLayerApi.h"
#include "Quantization.h"
#include "ReluApi.h"
#include "RematPlan.h"
#include "RematTestFixtures.h"
#include "Serialize.h"
#include "SoftmaxApi.h"
#include "StorageApi.h"
#include "Tensor.h"
#include "unity.h"

void setUp(void) {}
void tearDown(void) {}

static rematWireTable_t *initTable(layer_t **model, size_t n, lossFuncType_t lt,
                                   const tensor_t *input) {
    rematWireTable_t *t = NULL;
    TEST_ASSERT_TRUE(rematWireTableInit(&t, model, n, defaultLossConfig(lt), input));
    TEST_ASSERT_NOT_NULL(t);
    return t;
}

static void assertDims(const shape_t *shape, const size_t *dims, size_t rank) {
    TEST_ASSERT_EQUAL_size_t(rank, shape->numberOfDimensions);
    for (size_t d = 0; d < rank; d++) {
        TEST_ASSERT_EQUAL_size_t(dims[d], shape->dimensions[d]);
    }
}

/* [Linear 2->8, Quant FLOAT32->BFP per-tensor] under MSE: ACT 1 (32 B), ACT 2
 * (BFP, 8 B, 1 exponent slot), the seed GRAD 2 (id 3, inherits ACT 2, 8 B,
 * its exponent byte the last object of the block) and GRAD 1 (id 4, 32 B). */
typedef struct seedFixture {
    uint8_t tmplExponent[1];
    bfpQConfig_t tmplQc;
    quantization_t tmplQ;
    layer_t *model[2];
    inputLike_t in;
    tensor_t *x;
    rematWireTable_t *t;
} seedFixture_t;

static void buildBfpSeedFixture(seedFixture_t *f) {
    initBfpQConfigInto(8, 8, HALF_AWAY, f->tmplExponent, &f->tmplQc);
    f->tmplQ = (quantization_t){.type = BFP, .qConfig = &f->tmplQc};
    f->model[0] = makeLinear(2, 8, false);
    f->model[1] = makeQuant(&f->tmplQ, &g_floatQ);
    f->x = makeInput(&f->in, (size_t[]){1, 2}, 2, &g_floatQ);
    f->t = initTable(f->model, 2, MSE, f->x);
    rematWireTableBind(f->t, f->model, 2, MSE, REMAT_MODE_TRAIN, f->x);
}

static void freeSeedFixture(seedFixture_t *f) {
    rematWireTableFree(f->t);
    freeModel(f->model, 2);
}

#define SEED_ACT1 1u
#define SEED_ACT2 2u
#define SEED_GRAD2 3u

static rematPlan_t *buildPlan(const rematWireTable_t *t, layer_t **model,
                              const rematPlanSpec_t *spec) {
    rematPlan_t *p = NULL;
    TEST_ASSERT_TRUE(rematPlanBuild(&p, t, model, spec));
    TEST_ASSERT_NOT_NULL(p);
    return p;
}

/* Builds table + plan for a fixture model and returns the plan's peak. */
static size_t peakOf(layer_t **model, size_t n, lossFuncType_t lt, const tensor_t *x,
                     const rematPlanSpec_t *spec) {
    rematWireTable_t *t = initTable(model, n, lt, x);
    rematPlan_t *p = buildPlan(t, model, spec);
    size_t peak = p->train.peakLiveBytes;
    rematPlanFree(p);
    rematWireTableFree(t);
    return peak;
}

static const rematRange_t *rangeOfWire(const rematProgram_t *p, uint16_t wire) {
    for (size_t r = 0; r < p->numRanges; r++) {
        if (p->ranges[r].wire == wire) {
            return &p->ranges[r];
        }
    }
    return NULL;
}

static void assertLiveAt(const rematProgram_t *p, uint16_t wire, size_t step, const char *what) {
    if (wire == 0) {
        return; /* ACT 0 is borrowed: always present, never ranged */
    }
    const rematRange_t *r = rangeOfWire(p, wire);
    TEST_ASSERT_NOT_NULL_MESSAGE(r, what);
    TEST_ASSERT_TRUE_MESSAGE(r->begin <= step && step <= r->end, what);
}

/* The operands of every step, derived here independently of the generator,
 * must all be live at that step: inclusive ranges are what keep
 * a FORWARD's input and output, GRAD l+1 and GRAD l, and ACT l and GRAD l at a
 * reading BACKWARD(l) out of each other's bytes. */
static void assertEveryStepsOperandsAreCoLive(const rematProgram_t *p, const rematWireTable_t *t,
                                              layer_t **model) {
    size_t n = t->modelSize;
    for (size_t s = 0; s < p->numSteps; s++) {
        size_t l = p->steps[s].layer;
        switch (p->steps[s].kind) {
        case REMAT_STEP_FORWARD:
            assertLiveAt(p, rematActId(t, l), s, "FORWARD input");
            assertLiveAt(p, rematActId(t, l + 1), s, "FORWARD output");
            break;
        case REMAT_STEP_LOSS_FORWARD:
            assertLiveAt(p, rematActId(t, n), s, "LOSS_FORWARD reads ACT n");
            break;
        case REMAT_STEP_LOSS_BACKWARD:
            assertLiveAt(p, rematActId(t, n), s, "LOSS_BACKWARD reads ACT n");
            assertLiveAt(p, rematGradId(t, n), s, "LOSS_BACKWARD writes the seed");
            break;
        default: { /* REMAT_STEP_BACKWARD */
            uint16_t gradIn =
                ((ptrdiff_t)l == t->backwardTop) ? rematGradId(t, n) : rematGradId(t, l + 1);
            assertLiveAt(p, gradIn, s, "BACKWARD reads gradIn");
            if (layerBackwardReadsInput(model[l])) {
                assertLiveAt(p, rematActId(t, l), s, "BACKWARD reads its input");
            }
            if (l > t->deepest) {
                assertLiveAt(p, rematGradId(t, l), s, "BACKWARD writes its dx");
            }
            break;
        }
        }
    }
}

static void assertCoLiveUnderBothPolicies(layer_t **model, size_t n, lossFuncType_t lt,
                                          const tensor_t *x) {
    rematWireTable_t *t = initTable(model, n, lt, x);
    rematPlan_t *all = buildPlan(t, model, NULL);
    rematPlan_t *live = buildPlan(t, model, &g_liveness);
    assertEveryStepsOperandsAreCoLive(&all->train, t, model);
    assertEveryStepsOperandsAreCoLive(&live->train, t, model);
    TEST_ASSERT_TRUE(live->train.peakLiveBytes <= all->train.peakLiveBytes);
    rematPlanFree(live);
    rematPlanFree(all);
    rematWireTableFree(t);
}

static layer_t *randomRank2Layer(uint32_t *state) {
    switch (nextRandom(state) % 4u) {
    case 0:
        return makeLinear(4, 4, nextRandom(state) % 2u == 0u);
    case 1:
        return makeRelu(&g_floatQ);
    case 2:
        return makeSoftmax();
    default:
        return makeLayerNorm(4, nextRandom(state) % 2u == 0u);
    }
}

/* A built plan whose program a test tampers with inside the death-test child. */
typedef struct grammarFixture {
    layer_t *model[HAR_N];
    size_t n;
    inputLike_t in;
    rematWireTable_t *t;
    rematPlan_t *p;
} grammarFixture_t;

static void buildHarLivenessFixture(grammarFixture_t *f) {
    buildHar(f->model, false);
    f->n = HAR_N;
    f->t = initTable(f->model, HAR_N, CROSS_ENTROPY, makeHarInput(&f->in));
    f->p = buildPlan(f->t, f->model, &g_liveness);
}

static void buildHarStoreAllFixture(grammarFixture_t *f) {
    buildHar(f->model, false);
    f->n = HAR_N;
    f->t = initTable(f->model, HAR_N, CROSS_ENTROPY, makeHarInput(&f->in));
    f->p = buildPlan(f->t, f->model, NULL);
}

static void buildAllFrozenFixture(grammarFixture_t *f) {
    f->model[0] = makeLinear(2, 4, true);
    f->model[1] = makeRelu(&g_floatQ);
    f->n = 2;
    f->t = initTable(f->model, 2, MSE, makeInput(&f->in, (size_t[]){1, 2}, 2, &g_floatQ));
    f->p = buildPlan(f->t, f->model, NULL);
}

static void freeGrammarFixture(grammarFixture_t *f) {
    rematPlanFree(f->p);
    rematWireTableFree(f->t);
    freeModel(f->model, f->n);
}

static void tamperAndValidate(grammarFixture_t *f, void (*tamper)(rematProgram_t *)) {
    tamper(&f->p->train);
    rematPlanValidateGrammar(&f->p->train, f->t, f->model, REMAT_MODE_TRAIN);
}

#define ASSERT_GRAMMAR_EXIT(buildFixture, tamper, rule)                                            \
    do {                                                                                           \
        grammarFixture_t _fixture;                                                                 \
        buildFixture(&_fixture);                                                                   \
        ASSERT_EXITS_WITH_OUTPUT(1, rule, tamperAndValidate(&_fixture, tamper));                   \
        freeGrammarFixture(&_fixture);                                                             \
    } while (0)

/* HAR TRAIN: F0..F11 = steps 0..11, LOSS_FORWARD 12, LOSS_BACKWARD 13, B10..B0 = 14..24. */
static void duplicateTheFirstForward(rematProgram_t *p) {
    p->steps[1] = p->steps[0];
}
static void swapLastForwardAndLossForward(rematProgram_t *p) {
    rematStep_t s = p->steps[11];
    p->steps[11] = p->steps[12];
    p->steps[12] = s;
}
static void forwardBeyondTheLastLayer(rematProgram_t *p) { /* LOSS_FORWARD -> FORWARD(12) */
    p->steps[12] = (rematStep_t){.kind = REMAT_STEP_FORWARD, .layer = 12};
}
static void swapTheFirstTwoBackwards(rematProgram_t *p) {
    rematStep_t s = p->steps[14];
    p->steps[14] = p->steps[15];
    p->steps[15] = s;
}
static void unknownStepKind(rematProgram_t *p) {
    p->steps[0].kind = 9u;
}
static void dropTheLastBackward(rematProgram_t *p) {
    p->numSteps--;
}
static void dropFromLossBackwardOn(rematProgram_t *p) {
    p->numSteps = 13;
}
static void dropTheLossForward(rematProgram_t *p) { /* all-frozen */
    p->numSteps = 2;
}
static void endAct1BeforeItsLastRead(rematProgram_t *p) { /* LIVENESS: ACT 1 ends at B1 = 23 */
    p->ranges[0].end--;
}
static void beginAct2AfterItsWrite(rematProgram_t *p) { /* FORWARD(1) = step 1 writes ACT 2 */
    p->ranges[1].begin++;
}

/* ACT 2's only reader is FORWARD(2) at step 2 (MaxPool's backward does not
 * read its input, LayerConfigAccess.c:302-311). */
static void endAct2BeforeItsOnlyForwardRead(rematProgram_t *p) {
    p->ranges[1].end--;
}

/* HAR STORE_ALL: an absolute end (not a decrement) forces ACT 12's range shut
 * right after its own write (step 11), so LOSS_FORWARD's read at step 12 is
 * the first violation regardless of ACT 12's natural (last-step) end. */
static void endAct12BeforeLossForwardReadsIt(rematProgram_t *p) {
    p->ranges[11].end = 11;
}

/* HAR LIVENESS: ACT 12 is read twice (LOSS_FORWARD at 12, LOSS_BACKWARD at
 * 13); shortening its end by one still covers the first read, isolating the
 * LOSS_BACKWARD check. */
static void endAct12BeforeLossBackwardReadsIt(rematProgram_t *p) {
    p->ranges[11].end--;
}

/* The seed (GRAD n) begins at LOSS_BACKWARD's write (step 13); moving begin
 * one step later makes that write itself the violation. */
static void beginSeedAfterLossBackwardWritesIt(rematProgram_t *p) {
    p->ranges[12].begin++;
}

/* GRAD 10 (wire 14) is read once, by BACKWARD(9)'s gradIn at step 15;
 * shortening its end by one isolates that read. */
static void endGrad10BeforeBackwardReadsIt(rematProgram_t *p) {
    p->ranges[13].end--;
}

/* ---- rematBackwardRange ---- */

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

/* n = 1 under CE keeps today's signed top = -1. */
void testBackwardRangeSingleLayerUnderCrossEntropyIsMinusOne(void) {
    layer_t *model[1] = {makeLinear(2, 3, false)};
    size_t deepest = 99;
    ptrdiff_t top = 99;
    rematBackwardRange(model, 1, CROSS_ENTROPY, &deepest, &top);
    TEST_ASSERT_EQUAL_size_t(0, deepest);
    TEST_ASSERT_EQUAL_INT(-1, (int)top);
    freeModel(model, 1);
}

/* ---- the one BFP wire-grouping rule ---- */

/* The helper must reproduce the pre-remat driver's rule (the Legacy oracle's
 * ACT and dx wire allocators) and initBufferOutput's copy (InferenceApi.c)
 * on its canonical cases. */
static void assertGrouping(size_t tmplGroups, size_t tmplGroupSize, size_t elements,
                           size_t numGroups, size_t groupSize) {
    uint8_t exponents[8];
    bfpQConfig_t tmpl;
    initBfpQConfigGroupedInto(8, 8, HALF_AWAY, tmplGroups, tmplGroupSize, exponents, &tmpl);
    rematBfpGroups_t g = rematBfpWireGrouping(&tmpl, elements, REMAT_WIRE_ACT, 1);
    TEST_ASSERT_EQUAL_size_t(numGroups, g.numGroups);
    TEST_ASSERT_EQUAL_size_t(groupSize, g.groupSize);
}

void testBfpWireGroupingMatchesTheDriverRule(void) {
    assertGrouping(1, 0, 16, 1, 0);  /* per-tensor template stays per-tensor */
    assertGrouping(2, 16, 16, 1, 0); /* groupSize == N: the {1,N} spelling becomes {1,0} */
    assertGrouping(2, 4, 16, 4, 4);  /* the template's numGroups is ignored: derived from N */
    assertGrouping(8, 2, 16, 8, 2);
}

void testBfpWireGroupingExitsNamingTheWireOnAnIndivisibleGroupSize(void) {
    uint8_t exponents[2];
    bfpQConfig_t tmpl;
    initBfpQConfigGroupedInto(8, 8, HALF_AWAY, 2, 3, exponents, &tmpl);
    ASSERT_EXITS_WITH_OUTPUT(1, "BFP groupSize 3 does not divide the 16 elements of wire GRAD 5",
                             (void)rematBfpWireGrouping(&tmpl, 16, REMAT_WIRE_GRAD, 5));
}

/* ---- wire numbering and records ---- */

void testHarTableNumbersWiresInProductionOrder(void) {
    layer_t *model[HAR_N];
    buildHar(model, false);
    inputLike_t in;
    rematWireTable_t *t = initTable(model, HAR_N, CROSS_ENTROPY, makeHarInput(&in));

    /* ACT 0..12, then GRAD {12, 10, 9, ..., 1}: 23 slab headers + the borrowed ACT 0. */
    TEST_ASSERT_EQUAL_size_t(24, t->numWires);
    for (uint16_t j = 0; j <= 12; j++) {
        TEST_ASSERT_EQUAL_UINT8(REMAT_WIRE_ACT, t->wires[j].kind);
        TEST_ASSERT_EQUAL_UINT16(j, t->wires[j].index);
        TEST_ASSERT_EQUAL_UINT16(REMAT_NONE, t->wires[j].inheritFrom);
    }
    const uint16_t gradIndex[11] = {12, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1};
    for (uint16_t k = 0; k < 11; k++) {
        uint16_t id = (uint16_t)(13 + k);
        TEST_ASSERT_EQUAL_UINT8(REMAT_WIRE_GRAD, t->wires[id].kind);
        TEST_ASSERT_EQUAL_UINT16(gradIndex[k], t->wires[id].index);
        TEST_ASSERT_EQUAL_UINT16(id, t->gradIdOf[gradIndex[k]]);
    }
    /* The seed inherits ACT 12; Flatten (layer 9) passes its dx through, so
     * GRAD 9 inherits ACT 9; every other GRAD has a type-derived template. */
    TEST_ASSERT_EQUAL_UINT16(12, t->wires[13].inheritFrom);
    TEST_ASSERT_EQUAL_UINT16(9, t->wires[15].inheritFrom);
    TEST_ASSERT_EQUAL_UINT16(REMAT_NONE, t->wires[14].inheritFrom);
    TEST_ASSERT_EQUAL_UINT16(REMAT_NONE, t->wires[16].inheritFrom);
    /* No GRAD 11 under CE (the positional skip), no GRAD 0 (deepest, grads-only). */
    TEST_ASSERT_EQUAL_UINT16(REMAT_NONE, t->gradIdOf[11]);
    TEST_ASSERT_EQUAL_UINT16(REMAT_NONE, t->gradIdOf[0]);

    rematWireTableFree(t);
    freeModel(model, HAR_N);
}

void testHarTableRecordsBytesRanksAndKey(void) {
    layer_t *model[HAR_N];
    buildHar(model, false);
    inputLike_t in;
    rematWireTable_t *t = initTable(model, HAR_N, CROSS_ENTROPY, makeHarInput(&in));

    /* Scan-model bytes: ACT 0..12, then the GRADs in id order. */
    const size_t bytes[24] = {4608, 8192, 8192, 4096, 8192, 8192, 4096, 8192,
                              8192, 256,  256,  24,   24,   24,   256,  256,
                              8192, 8192, 4096, 8192, 8192, 4096, 8192, 8192};
    const uint8_t rank[24] = {3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2,
                              2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3};
    for (uint16_t id = 0; id < 24; id++) {
        TEST_ASSERT_EQUAL_size_t_MESSAGE(bytes[id], t->wires[id].bytes, "bytes");
        TEST_ASSERT_EQUAL_UINT8_MESSAGE(rank[id], t->wires[id].rank, "rank");
        TEST_ASSERT_EQUAL_UINT8(FLOAT32, t->wires[id].dtype);
        TEST_ASSERT_EQUAL_UINT8(id == 0 ? 1 : 0, t->wires[id].borrowed);
        TEST_ASSERT_EQUAL_size_t(0, t->wires[id].expCapacity);
    }
    TEST_ASSERT_NULL(
        t->wires[0].hdr); /* ACT 0 is the caller's, bound only between bind and unbind */

    TEST_ASSERT_EQUAL_size_t(HAR_N, t->modelSize);
    TEST_ASSERT_EQUAL_INT(CROSS_ENTROPY, t->lossType);
    TEST_ASSERT_EQUAL_size_t(0, t->deepest);
    TEST_ASSERT_EQUAL_INT(10, (int)t->backwardTop);
    TEST_ASSERT_TRUE(t->hasBackward);
    for (size_t i = 0; i < HAR_N; i++) {
        TEST_ASSERT_EQUAL_UINT8(model[i]->type, t->layerType[i]);
        TEST_ASSERT_EQUAL_UINT8(0, t->frozen[i]);
    }
    TEST_ASSERT_EQUAL_size_t(3, t->inputRank);
    const size_t dims[3] = {1, 9, 128};
    for (size_t d = 0; d < 3; d++) {
        TEST_ASSERT_EQUAL_size_t(dims[d], t->inputDims[d]);
        TEST_ASSERT_EQUAL_size_t(d, t->inputOrder[d]);
    }
    TEST_ASSERT_EQUAL_UINT8(FLOAT32, t->inputType);
    TEST_ASSERT_EQUAL_UINT8(3, t->maxRank);

    rematWireTableFree(t);
    freeModel(model, HAR_N);
}

/* One grouped template shared by wires of different sizes groups each
 * wire by its own element count, as the pre-remat driver did. */
void testSharedGroupedBfpTemplateGroupsPerWire(void) {
    uint8_t tmplExponents[2];
    bfpQConfig_t tmplQc;
    initBfpQConfigGroupedInto(8, 8, HALF_AWAY, 2, 4, tmplExponents, &tmplQc);
    quantization_t bfpQ = {.type = BFP, .qConfig = &tmplQc};
    layerQuant_t lq;
    layerQuantInitUniform(&lq, &g_floatQ);
    lq.outputQ = &bfpQ;
    layer_t *model[2] = {makeQuant(&bfpQ, &g_floatQ),
                         linearLayerInit(&(linearInit_t){.inFeatures = 4, .outFeatures = 16}, &lq)};
    inputLike_t in;
    rematWireTable_t *t = initTable(model, 2, MSE, makeInput(&in, (size_t[]){1, 4}, 2, &g_floatQ));

    /* ACT 1: 4 elements, groupSize 4 == N -> per-tensor, 1 slot, 4 B.
     * ACT 2: 16 elements -> 4 groups of 4, 16 B. Seed GRAD 2 inherits ACT 2.
     * deepest == top == 1, so no other GRAD. */
    TEST_ASSERT_EQUAL_size_t(4, t->numWires);
    TEST_ASSERT_EQUAL_UINT8(BFP, t->wires[1].dtype);
    TEST_ASSERT_EQUAL_size_t(1, t->wires[1].expCapacity);
    TEST_ASSERT_EQUAL_size_t(4, t->wires[1].bytes);
    TEST_ASSERT_EQUAL_UINT8(BFP, t->wires[2].dtype);
    TEST_ASSERT_EQUAL_size_t(4, t->wires[2].expCapacity);
    TEST_ASSERT_EQUAL_size_t(16, t->wires[2].bytes);
    TEST_ASSERT_EQUAL_UINT16(2, t->wires[3].inheritFrom);
    TEST_ASSERT_EQUAL_size_t(4, t->wires[3].expCapacity);
    TEST_ASSERT_EQUAL_size_t(16, t->wires[3].bytes);

    /* Pins the exponent tail's per-wire spacing so a SLAB_PLACE(...,
     * f->numGroups, ...) regression to a fixed count cannot slip past this
     * test. In wire order, each BFP wire's exponent array sits exactly its
     * OWN expCapacity bytes before the next one's, and the last (wire 3, the
     * seed) abuts the block end. */
    bfpQConfig_t *qc1 = t->wires[1].hdr->quantization->qConfig;
    bfpQConfig_t *qc2 = t->wires[2].hdr->quantization->qConfig;
    bfpQConfig_t *qc3 = t->wires[3].hdr->quantization->qConfig;
    TEST_ASSERT_EQUAL_PTR(qc1->exponents + t->wires[1].expCapacity, qc2->exponents);
    TEST_ASSERT_EQUAL_PTR(qc2->exponents + t->wires[2].expCapacity, qc3->exponents);
    TEST_ASSERT_EQUAL_PTR((uint8_t *)t + t->slabBytes, qc3->exponents + t->wires[3].expCapacity);

    rematWireTableFree(t);
    freeModel(model, 2);
}

/* ---- one block, released by a single release call ---- */

#ifdef ODT_MEM_PROFILE
/* The live-byte counter is real only under ODT_MEM_PROFILE (unit_test_debug,
 * asan, ubsan); the plain unit_test preset compiles this out. */
void testTableInitReservesOneBlockOfSlabBytesAndFreeReturnsIt(void) {
    layer_t *model[HAR_N];
    buildHar(model, false);
    inputLike_t in;
    tensor_t *x = makeHarInput(&in);
    size_t before = memProfileCurrentBytes();
    rematWireTable_t *t = initTable(model, HAR_N, CROSS_ENTROPY, x);
    TEST_ASSERT_EQUAL_size_t(before + t->slabBytes, memProfileCurrentBytes());
    rematWireTableFree(t);
    TEST_ASSERT_EQUAL_size_t(before, memProfileCurrentBytes());
    freeModel(model, HAR_N);
}
#endif

/* Slab headers borrow their shape, quantization_t, qConfig and
 * BFP exponents from the one table block, so the block's single
 * freeReservedMemory is the only legal release. The free runs in a forked
 * child: releasing an interior slab pointer aborts there (or, under
 * ODT_MEM_PROFILE, corrupts the live-byte count), and the parent sees it. */
static size_t g_memBeforeTable;

static void freeTableAndExitWithBaselineVerdict(rematWireTable_t *t) {
    rematWireTableFree(t);
    _exit(memProfileCurrentBytes() == g_memBeforeTable ? 0 : 2);
}

void testTableFreeReleasesExactlyTheOneBlock(void) {
    uint8_t exponent[1];
    bfpQConfig_t bfpQc;
    initBfpQConfigInto(8, 8, HALF_AWAY, exponent, &bfpQc);
    quantization_t bfpQ = {.type = BFP, .qConfig = &bfpQc};
    layer_t *model[1] = {makeQuant(&bfpQ, &g_floatQ)};
    inputLike_t in;
    tensor_t *x = makeInput(&in, (size_t[]){1, 8}, 2, &g_floatQ);
    g_memBeforeTable = memProfileCurrentBytes();
    rematWireTable_t *t = initTable(model, 1, MSE, x);
    ASSERT_EXITS_WITH(0, freeTableAndExitWithBaselineVerdict(t));
    rematWireTableFree(t);
    freeModel(model, 1);
}

void testTableFreeIsNullSafe(void) {
    ASSERT_EXITS_WITH(0, rematWireTableFree(NULL));
}

/* ---- slab alignment ---- */

#define ASSERT_ALIGNED(ptr, T)                                                                     \
    TEST_ASSERT_EQUAL_UINT_MESSAGE(0u, (unsigned)((uintptr_t)(ptr) % _Alignof(T)),                 \
                                   #ptr " is misaligned for " #T)

/* Wire order SYM_INT32, BFP, FLOAT32: the 12-byte SYM qConfig and the odd-sized
 * uint8_t key arrays leave the cursor unaligned for what follows them, and an
 * exponent array placed inline would sit between the BFP and the FLOAT32
 * header instead of at the tail. */
void testSlabObjectsAlignedAndExponentsAtTheTail(void) {
    symInt32QConfig_t symQc;
    initSymInt32QConfigWithQMaxBits(HALF_AWAY, &symQc, 12);
    quantization_t symQ = {.type = SYM_INT32, .qConfig = &symQc};
    uint8_t exponent[1];
    bfpQConfig_t bfpQc;
    initBfpQConfigInto(8, 8, HALF_AWAY, exponent, &bfpQc);
    quantization_t bfpQ = {.type = BFP, .qConfig = &bfpQc};
    layer_t *model[3] = {makeQuant(&symQ, &g_floatQ), makeQuant(&bfpQ, &g_floatQ),
                         makeRelu(&g_floatQ)};
    inputLike_t in;
    rematWireTable_t *t = initTable(model, 3, MSE, makeInput(&in, (size_t[]){1, 3}, 2, &g_floatQ));

    ASSERT_ALIGNED(t->wires, rematWire_t);
    ASSERT_ALIGNED(t->gradIdOf, uint16_t);
    ASSERT_ALIGNED(t->inputDims, size_t);
    ASSERT_ALIGNED(t->inputOrder, size_t);
    uintptr_t headersEnd = 0;
    for (uint16_t id = 1; id < t->numWires; id++) {
        tensor_t *hdr = t->wires[id].hdr;
        ASSERT_ALIGNED(hdr, tensor_t);
        ASSERT_ALIGNED(hdr->shape, shape_t);
        ASSERT_ALIGNED(hdr->shape->dimensions, size_t);
        ASSERT_ALIGNED(hdr->shape->orderOfDimensions, size_t);
        ASSERT_ALIGNED(hdr->quantization, quantization_t);
        uintptr_t end = (uintptr_t)hdr->quantization + sizeof(quantization_t);
        if (hdr->quantization->type == SYM_INT32) {
            ASSERT_ALIGNED(hdr->quantization->qConfig, symInt32QConfig_t);
            end = (uintptr_t)hdr->quantization->qConfig + sizeof(symInt32QConfig_t);
        } else if (hdr->quantization->type == BFP) {
            ASSERT_ALIGNED(hdr->quantization->qConfig, bfpQConfig_t);
            end = (uintptr_t)hdr->quantization->qConfig + sizeof(bfpQConfig_t);
        } else {
            TEST_ASSERT_NULL(hdr->quantization->qConfig); /* FLOAT32 reserves no qConfig */
        }
        headersEnd = end > headersEnd ? end : headersEnd;
    }
    bfpQConfig_t *slabBfp = t->wires[2].hdr->quantization->qConfig;
    TEST_ASSERT_TRUE_MESSAGE((uintptr_t)slabBfp->exponents >= headersEnd,
                             "BFP exponents are not at the slab tail");
    /* The one BFP wire's exponent array is the last object and abuts the block end. */
    TEST_ASSERT_EQUAL_PTR((uint8_t *)t + t->slabBytes, slabBfp->exponents + 1);

    rematWireTableFree(t);
    freeModel(model, 3);
}

/* ---- read-only accessors and the linked, content-free headers ---- */

void testAccessorsReadTheTableAndHeadersAreLinked(void) {
    layer_t *model[HAR_N];
    buildHar(model, false);
    inputLike_t in;
    rematWireTable_t *t = initTable(model, HAR_N, CROSS_ENTROPY, makeHarInput(&in));

    TEST_ASSERT_EQUAL_UINT16(5, rematActId(t, 5));
    TEST_ASSERT_EQUAL_UINT16(13, rematGradId(t, 12));
    TEST_ASSERT_EQUAL_UINT16(14, rematGradId(t, 10));
    TEST_ASSERT_EQUAL_UINT16(REMAT_NONE, rematGradId(t, 11));
    TEST_ASSERT_NULL(rematGradHdr(t, 11));
    TEST_ASSERT_NULL(rematGradHdr(t, 0));
    TEST_ASSERT_EQUAL_PTR(t->wires[13].hdr, rematGradHdr(t, 12));
    TEST_ASSERT_EQUAL_PTR(t->wires[3].hdr, rematActHdr(t, 3));
    TEST_ASSERT_EQUAL_PTR(t->wires[20].hdr, rematWireHdr(t, 20));
    TEST_ASSERT_NULL(rematActHdr(t, 0));
    TEST_ASSERT_EQUAL_size_t(8192, rematWireBytes(t, 1));
    TEST_ASSERT_EQUAL_size_t(24, rematWireBytes(t, 13));

    for (uint16_t id = 1; id < t->numWires; id++) {
        tensor_t *hdr = rematWireHdr(t, id);
        TEST_ASSERT_NULL(hdr->data);
        TEST_ASSERT_NULL(hdr->sparsity);
        TEST_ASSERT_EQUAL_size_t(t->wires[id].rank, hdr->shape->numberOfDimensions);
        TEST_ASSERT_EQUAL_INT(t->wires[id].dtype, hdr->quantization->type);
    }

    rematWireTableFree(t);
    freeModel(model, HAR_N);
}

/* ---- table init's named exits ---- */

void testTableInitExitsOnAnEmptyModel(void) {
    layer_t *model[1] = {makeRelu(&g_floatQ)};
    inputLike_t in;
    tensor_t *x = makeInput(&in, (size_t[]){1, 4}, 2, &g_floatQ);
    rematWireTable_t *t = NULL;
    ASSERT_EXITS_WITH_OUTPUT(1, "modelSize == 0: nothing to schedule",
                             (void)rematWireTableInit(&t, model, 0, defaultLossConfig(MSE), x));
    freeModel(model, 1);
}

/* 65534 parameter-free layers under MSE: no GRAD, so n + 1 = 65535 wires, the
 * first count whose ids would include the REMAT_NONE sentinel. One ReLU is
 * reused for every slot; the model array itself is test-owned. */
void testTableInitExitsWhenWireIdsWouldReachRematNone(void) {
    const size_t n = 65534u;
    layer_t *relu = makeRelu(&g_floatQ);
    layer_t **model = reserveMemory(n * sizeof(layer_t *));
    TEST_ASSERT_NOT_NULL(model);
    for (size_t i = 0; i < n; i++) {
        model[i] = relu;
    }
    inputLike_t in;
    tensor_t *x = makeInput(&in, (size_t[]){1, 4}, 2, &g_floatQ);
    rematWireTable_t *t = NULL;
    ASSERT_EXITS_WITH_OUTPUT(1, "65535 wires reach REMAT_NONE (0xFFFF)",
                             (void)rematWireTableInit(&t, model, n, defaultLossConfig(MSE), x));
    freeReservedMemory(model);
    freeReluLayer(relu);
}

void testTableInitExitsOnAnInputRankAboveTheRankField(void) {
    size_t dims[256];
    size_t order[256];
    for (size_t d = 0; d < 256; d++) {
        dims[d] = 1;
        order[d] = d;
    }
    shape_t shape = {.numberOfDimensions = 256, .dimensions = dims, .orderOfDimensions = order};
    tensor_t x = {.data = NULL, .shape = &shape, .quantization = &g_floatQ, .sparsity = NULL};
    layer_t *model[1] = {makeRelu(&g_floatQ)};
    rematWireTable_t *t = NULL;
    ASSERT_EXITS_WITH_OUTPUT(1, "wire ACT 0 has rank 256, above the uint8_t rank field",
                             (void)rematWireTableInit(&t, model, 1, defaultLossConfig(MSE), &x));
    freeModel(model, 1);
}

void testTableInitExitsOnAnUnsupportedWireDtype(void) {
    float scale = 1.f;
    symQConfig_t symQc = {
        .scales = &scale, .numGroups = 1, .groupSize = 0, .roundingMode = HALF_AWAY, .qBits = 8};
    quantization_t symQ = {.type = SYM, .qConfig = &symQc};
    layer_t *model[1] = {makeRelu(&symQ)};
    inputLike_t in;
    tensor_t *x = makeInput(&in, (size_t[]){1, 4}, 2, &g_floatQ);
    rematWireTable_t *t = NULL;
    ASSERT_EXITS_WITH_OUTPUT(1, "wire ACT 1 has dtype 3; remat wires are FLOAT32, SYM_INT32 or BFP",
                             (void)rematWireTableInit(&t, model, 1, defaultLossConfig(MSE), x));
    freeModel(model, 1);
}

/* #160: a zero-size block is implementation-defined (a zero-byte request may return NULL). */
void testTableInitExitsOnAZeroByteWire(void) {
    layer_t *model[1] = {makeRelu(&g_floatQ)};
    inputLike_t in;
    tensor_t *x = makeInput(&in, (size_t[]){1, 0}, 2, &g_floatQ);
    rematWireTable_t *t = NULL;
    ASSERT_EXITS_WITH_OUTPUT(1, "wire ACT 1 has zero bytes (#160)",
                             (void)rematWireTableInit(&t, model, 1, defaultLossConfig(MSE), x));
    freeModel(model, 1);
}

/* The borrowed ACT 0 may be any dtype, packed ones included;
 * its bytes are exact for the checker's disjointness test. */
void testTableInitAcceptsAPackedBorrowedInputAndSizesItExactly(void) {
    float scale = 1.f;
    uint16_t zeroPoint = 0;
    symQConfig_t symQc = {
        .scales = &scale, .numGroups = 1, .groupSize = 0, .roundingMode = HALF_AWAY, .qBits = 4};
    asymQConfig_t asymQc = {.scales = &scale,
                            .zeroPoints = &zeroPoint,
                            .numGroups = 1,
                            .groupSize = 0,
                            .qBits = 6,
                            .roundingMode = HALF_AWAY};
    quantization_t inputQ[4] = {{.type = SYM, .qConfig = &symQc},
                                {.type = ASYM, .qConfig = &asymQc},
                                {.type = INT32, .qConfig = NULL},
                                {.type = BOOL, .qConfig = NULL}};
    const size_t expectedBytes[4] = {4, 6, 32, 1}; /* 8 elements: 4, 6, 32 and 1 bits each */
    layer_t *model[1] = {makeRelu(&g_floatQ)};
    for (size_t k = 0; k < 4; k++) {
        inputLike_t in;
        rematWireTable_t *t =
            initTable(model, 1, MSE, makeInput(&in, (size_t[]){1, 8}, 2, &inputQ[k]));
        TEST_ASSERT_EQUAL_size_t(expectedBytes[k], rematWireBytes(t, 0));
        rematWireTableFree(t);
    }
    freeModel(model, 1);
}

void testTableInitExitsOnAnUnknownInputQtype(void) {
    quantization_t unknownQ = {.type = (qtype_t)99, .qConfig = NULL};
    layer_t *model[1] = {makeRelu(&g_floatQ)};
    inputLike_t in;
    tensor_t *x = makeInput(&in, (size_t[]){1, 4}, 2, &unknownQ);
    rematWireTable_t *t = NULL;
    ASSERT_EXITS_WITH_OUTPUT(1, "wire ACT 0 has unknown qtype 99",
                             (void)rematWireTableInit(&t, model, 1, defaultLossConfig(MSE), x));
    freeModel(model, 1);
}

/* ---- checked size arithmetic ---- */

/* The child prints how many bytes are live when it exits, so the parent can
 * check "exits before the table block is reserved". Real only under
 * ODT_MEM_PROFILE; on the plain preset both counters read 0. */
static size_t g_memBeforeExit;

static void printReservedBeforeExit(void) {
    printf("reservedBeforeExit=%zu\n", memProfileCurrentBytes() - g_memBeforeExit);
}

static void initExpectingAnExit(layer_t **model, size_t n, const tensor_t *x) {
    rematWireTable_t *t = NULL;
    g_memBeforeExit = memProfileCurrentBytes();
    (void)atexit(printReservedBeforeExit);
    (void)rematWireTableInit(&t, model, n, defaultLossConfig(MSE), x);
}

/* A borrowed [1, SIZE_MAX/4 + 2] FLOAT32 input: 4 * N wraps to 4. */
void testTableInitExitsOnAByteCountOverflowBeforeReservingTheTable(void) {
    layer_t *model[1] = {makeRelu(&g_floatQ)};
    inputLike_t in;
    tensor_t *x = makeInput(&in, (size_t[]){1, SIZE_MAX / 4u + 2u}, 2, &g_floatQ);
    ASSERT_EXITS_WITH_OUTPUT(1, "size overflow computing bytes of wire ACT 0",
                             initExpectingAnExit(model, 1, x));
    /* Only the transient derivation scratch (2 wires) is live at the exit: the
     * table block is never reserved. */
#ifdef ODT_MEM_PROFILE
    size_t expectedReserved = 2u * sizeof(rematWireFact_t);
#else
    size_t expectedReserved = 0u; /* the counters are no-ops without ODT_MEM_PROFILE */
#endif
    char expected[64];
    (void)snprintf(expected, sizeof expected, "reservedBeforeExit=%zu", expectedReserved);
    ASSERT_EXITS_WITH_OUTPUT(1, expected, initExpectingAnExit(model, 1, x));
    freeModel(model, 1);
}

/* Four BFP wires of 2^62 elements with groupSize 1: each wire's bytes fit
 * (2^60), but their exponent tails sum to 2^64. */
void testTableInitExitsOnASlabSizeOverflow(void) {
    uint8_t inputExponent[1];
    bfpQConfig_t inputQc;
    initBfpQConfigInto(2, 8, HALF_AWAY, inputExponent, &inputQc);
    quantization_t inputQ = {.type = BFP, .qConfig = &inputQc};
    uint8_t tmplExponents[2];
    bfpQConfig_t tmplQc;
    initBfpQConfigGroupedInto(2, 8, HALF_AWAY, 2, 1, tmplExponents, &tmplQc);
    quantization_t tmplQ = {.type = BFP, .qConfig = &tmplQc};
    layer_t *relu = makeRelu(&tmplQ);
    layer_t *model[4] = {relu, relu, relu, relu};
    inputLike_t in;
    tensor_t *x = makeInput(&in, (size_t[]){1, (size_t)1 << 62}, 2, &inputQ);
    ASSERT_EXITS_WITH_OUTPUT(1, "size overflow computing slabBytes of wire ACT 4",
                             initExpectingAnExit(model, 4, x));
    freeReluLayer(relu);
}

/* Three FLOAT32 wires of SIZE_MAX/8 elements: each fits (just under 2^63
 * bytes), two sum to just under 2^64, the third overflows. The checked total
 * bounds every later sum over wires. */
void testTableInitExitsOnATotalWireBytesOverflow(void) {
    layer_t *relu = makeRelu(&g_floatQ);
    layer_t *model[3] = {relu, relu, relu};
    inputLike_t in;
    tensor_t *x = makeInput(&in, (size_t[]){1, SIZE_MAX / 8u}, 2, &g_floatQ);
    ASSERT_EXITS_WITH_OUTPUT(1, "size overflow computing total wire bytes of wire ACT 3",
                             initExpectingAnExit(model, 3, x));
    freeReluLayer(relu);
}

/* ---- per-bind re-derivation ---- */

void testBindWritesHarHeadersAndPointsAct0AtTheInput(void) {
    layer_t *model[HAR_N];
    buildHar(model, false);
    inputLike_t in;
    tensor_t *x = makeHarInput(&in);
    rematWireTable_t *t = initTable(model, HAR_N, CROSS_ENTROPY, x);
    rematWireTableBind(t, model, HAR_N, CROSS_ENTROPY, REMAT_MODE_TRAIN, x);

    TEST_ASSERT_EQUAL_PTR(x, rematActHdr(t, 0));
    assertDims(rematActHdr(t, 1)->shape, (size_t[]){1, 16, 128}, 3);
    assertDims(rematActHdr(t, 3)->shape, (size_t[]){1, 16, 64}, 3);
    assertDims(rematActHdr(t, 9)->shape, (size_t[]){1, 64, 1}, 3);
    assertDims(rematActHdr(t, 10)->shape, (size_t[]){1, 64}, 2);
    assertDims(rematActHdr(t, 12)->shape, (size_t[]){1, 6}, 2);
    assertDims(rematGradHdr(t, 3)->shape, (size_t[]){1, 16, 64}, 3);
    assertDims(rematGradHdr(t, 10)->shape, (size_t[]){1, 64}, 2);
    for (uint16_t id = 1; id < t->numWires; id++) {
        TEST_ASSERT_NULL(rematWireHdr(t, id)->data);
        TEST_ASSERT_EQUAL_INT(FLOAT32, rematWireHdr(t, id)->quantization->type);
    }
    /* Inherited headers (the seed, Flatten's dx) wait for rematWireBind. */
    TEST_ASSERT_EQUAL_size_t(0, rematGradHdr(t, 12)->shape->dimensions[0]);
    TEST_ASSERT_EQUAL_size_t(0, rematGradHdr(t, 9)->shape->dimensions[0]);

    rematWireTableFree(t);
    freeModel(model, HAR_N);
}

/* An EVAL bind re-derives the ACT headers but writes no GRAD header: eval
 * binds no GRAD wire. GRAD 10 (the Linear's dx, type-derived) is tampered
 * after a TRAIN bind; only the next TRAIN bind re-derives it. */
void testEvalBindLeavesTheGradHeadersAsTheyAre(void) {
    layer_t *model[HAR_N];
    buildHar(model, false);
    inputLike_t in;
    tensor_t *x = makeHarInput(&in);
    rematWireTable_t *t = initTable(model, HAR_N, CROSS_ENTROPY, x);
    rematWireTableBind(t, model, HAR_N, CROSS_ENTROPY, REMAT_MODE_TRAIN, x);
    rematGradHdr(t, 10)->shape->dimensions[1] = 7u;
    rematActHdr(t, 10)->shape->dimensions[1] = 7u;
    rematWireTableBind(t, model, HAR_N, CROSS_ENTROPY, REMAT_MODE_EVAL, x);
    TEST_ASSERT_EQUAL_size_t(7, rematGradHdr(t, 10)->shape->dimensions[1]);
    assertDims(rematActHdr(t, 10)->shape, (size_t[]){1, 64}, 2);
    rematWireTableBind(t, model, HAR_N, CROSS_ENTROPY, REMAT_MODE_TRAIN, x);
    assertDims(rematGradHdr(t, 10)->shape, (size_t[]){1, 64}, 2);
    rematWireTableFree(t);
    freeModel(model, HAR_N);
}

/* The bind generations and the observed peak restart at every bind, an EVAL
 * one included: the peak a caller reads after an eval call is that call's. */
void testEvalBindResetsTheBindGenerationsAndTheObservedPeak(void) {
    layer_t *model[HAR_N];
    buildHar(model, false);
    inputLike_t in;
    tensor_t *x = makeHarInput(&in);
    rematWireTable_t *t = initTable(model, HAR_N, CROSS_ENTROPY, x);
    rematWireTableBind(t, model, HAR_N, CROSS_ENTROPY, REMAT_MODE_TRAIN, x);
    t->wires[1].bindGen = 5u;
    t->observedPeakLiveBytes = 99u;
    rematWireTableBind(t, model, HAR_N, CROSS_ENTROPY, REMAT_MODE_EVAL, x);
    TEST_ASSERT_EQUAL_UINT32(0, t->wires[1].bindGen);
    TEST_ASSERT_EQUAL_size_t(0, t->observedPeakLiveBytes);
    rematWireTableFree(t);
    freeModel(model, HAR_N);
}

/* EVAL runs the same key check: a batch of two against a table keyed to one
 * exits naming ACT 0's field. */
void testEvalBindRunsTheFullKeyCheck(void) {
    layer_t *model[HAR_N];
    buildHar(model, false);
    inputLike_t in;
    rematWireTable_t *t = initTable(model, HAR_N, CROSS_ENTROPY, makeHarInput(&in));
    inputLike_t in2;
    tensor_t *x2 = makeInput(&in2, (size_t[]){2, 9, 128}, 3, &g_floatQ);
    ASSERT_EXITS_WITH_OUTPUT(
        1, "key mismatch on wire ACT 0, field 'dims[0]': built 1, live 2",
        rematWireTableBind(t, model, HAR_N, CROSS_ENTROPY, REMAT_MODE_EVAL, x2));
    rematWireTableFree(t);
    freeModel(model, HAR_N);
}

/* Forward wires copy the upstream order (ReLU, LayerNorm); a dx wire always
 * gets identity order, as the pre-remat driver's dx allocator did. */
void testBindCopiesForwardOrderAndGivesGradsIdentityOrder(void) {
    layer_t *model[2] = {makeLayerNorm(4, false), makeRelu(&g_floatQ)};
    inputLike_t in;
    tensor_t *x = makeInput(&in, (size_t[]){1, 4}, 2, &g_floatQ);
    in.order[0] = 1;
    in.order[1] = 0;
    rematWireTable_t *t = initTable(model, 2, MSE, x);
    rematWireTableBind(t, model, 2, MSE, REMAT_MODE_TRAIN, x);

    TEST_ASSERT_EQUAL_size_t(1, rematActHdr(t, 1)->shape->orderOfDimensions[0]);
    TEST_ASSERT_EQUAL_size_t(0, rematActHdr(t, 1)->shape->orderOfDimensions[1]);
    assertDims(rematGradHdr(t, 1)->shape, (size_t[]){1, 4}, 2);
    TEST_ASSERT_EQUAL_size_t(0, rematGradHdr(t, 1)->shape->orderOfDimensions[0]);
    TEST_ASSERT_EQUAL_size_t(1, rematGradHdr(t, 1)->shape->orderOfDimensions[1]);

    rematWireTableFree(t);
    freeModel(model, 2);
}

#ifdef ODT_MEM_PROFILE
/* G4: a bind derives into the slab and reserves nothing. */
void testBindAllocatesNothing(void) {
    uint8_t exponent[1];
    bfpQConfig_t bfpQc;
    initBfpQConfigInto(8, 8, HALF_AWAY, exponent, &bfpQc);
    quantization_t bfpQ = {.type = BFP, .qConfig = &bfpQc};
    layer_t *model[1] = {makeQuant(&bfpQ, &g_floatQ)};
    inputLike_t in;
    tensor_t *x = makeInput(&in, (size_t[]){1, 8}, 2, &g_floatQ);
    rematWireTable_t *t = initTable(model, 1, MSE, x);
    size_t before = memProfileCurrentBytes();
    rematWireTableBind(t, model, 1, MSE, REMAT_MODE_TRAIN, x);
    TEST_ASSERT_EQUAL_size_t(before, memProfileCurrentBytes());
    rematWireTableFree(t);
    freeModel(model, 1);
}
#endif

/* Table level: Quant outputQ @8 at build, @16 at the next bind;
 * the dynamic scale restarts at its init value every bind. */
void testBindRederivesSymQMaxBitsAndResetsScale(void) {
    symInt32QConfig_t symQc;
    initSymInt32QConfigWithQMaxBits(HALF_AWAY, &symQc, 8);
    quantization_t symQ = {.type = SYM_INT32, .qConfig = &symQc};
    layer_t *model[1] = {makeQuant(&symQ, &g_floatQ)};
    inputLike_t in;
    tensor_t *x = makeInput(&in, (size_t[]){1, 4}, 2, &g_floatQ);
    rematWireTable_t *t = initTable(model, 1, MSE, x);
    rematWireTableBind(t, model, 1, MSE, REMAT_MODE_TRAIN, x);
    symInt32QConfig_t *slabQc = rematActHdr(t, 1)->quantization->qConfig;
    TEST_ASSERT_EQUAL_UINT8(8, slabQc->qMaxBits);

    slabQc->scale = 0.25f; /* a producer's OUT_WRITE epilogue */
    symQc.qMaxBits = 16;   /* key-preserving template edit between calls */
    rematWireTableBind(t, model, 1, MSE, REMAT_MODE_TRAIN, x);
    TEST_ASSERT_EQUAL_UINT8(16, slabQc->qMaxBits);
    TEST_ASSERT_EQUAL_FLOAT(1.0f, slabQc->scale);

    rematWireTableFree(t);
    freeModel(model, 1);
}

/* A deserializeModel into a skeleton whose wire width
 * differs is a key-preserving config edit (ODTS writes the layer's outputQ in
 * place, Deserialize.c:773), adopted at the next bind. */
void testBindAfterDeserializeModel(void) {
    symInt32QConfig_t savedQc;
    initSymInt32QConfigWithQMaxBits(HALF_AWAY, &savedQc, 8);
    quantization_t savedQ = {.type = SYM_INT32, .qConfig = &savedQc};
    layer_t *saved[1] = {makeQuant(&savedQ, &g_floatQ)};
    FILE *file = tmpfile();
    TEST_ASSERT_NOT_NULL(file);
    serializeModel(saved, 1, file);
    rewind(file);

    symInt32QConfig_t skeletonQc;
    initSymInt32QConfigWithQMaxBits(HALF_AWAY, &skeletonQc, 16);
    quantization_t skeletonQ = {.type = SYM_INT32, .qConfig = &skeletonQc};
    layer_t *skeleton[1] = {makeQuant(&skeletonQ, &g_floatQ)};
    inputLike_t in;
    tensor_t *x = makeInput(&in, (size_t[]){1, 4}, 2, &g_floatQ);
    rematWireTable_t *t = initTable(skeleton, 1, MSE, x);
    rematWireTableBind(t, skeleton, 1, MSE, REMAT_MODE_TRAIN, x);
    symInt32QConfig_t *slabQc = rematActHdr(t, 1)->quantization->qConfig;
    TEST_ASSERT_EQUAL_UINT8(16, slabQc->qMaxBits);

    deserializeModel(skeleton, 1, file);
    (void)fclose(file);
    rematWireTableBind(t, skeleton, 1, MSE, REMAT_MODE_TRAIN, x);
    TEST_ASSERT_EQUAL_UINT8(8, slabQc->qMaxBits);

    rematWireTableFree(t);
    freeModel(skeleton, 1);
    freeModel(saved, 1);
}

/* The rounding-mode half of the bind's rounding-mode and draw-count
 * re-derivation; the draw-count half needs the driver (PR2). */
void testBindRederivesRoundingMode(void) {
    symInt32QConfig_t symQc;
    initSymInt32QConfigWithQMaxBits(HALF_AWAY, &symQc, 12);
    quantization_t symQ = {.type = SYM_INT32, .qConfig = &symQc};
    layer_t *model[1] = {makeQuant(&symQ, &g_floatQ)};
    inputLike_t in;
    tensor_t *x = makeInput(&in, (size_t[]){1, 4}, 2, &g_floatQ);
    rematWireTable_t *t = initTable(model, 1, MSE, x);
    rematWireTableBind(t, model, 1, MSE, REMAT_MODE_TRAIN, x);
    symQc.roundingMode = SR_HALF_AWAY;
    rematWireTableBind(t, model, 1, MSE, REMAT_MODE_TRAIN, x);
    symInt32QConfig_t *slabQc = rematActHdr(t, 1)->quantization->qConfig;
    TEST_ASSERT_EQUAL_INT(SR_HALF_AWAY, slabQc->roundingMode);
    rematWireTableFree(t);
    freeModel(model, 1);
}

/* Flatten-at-0 re-inherits the live input's BFP grouping at
 * every bind -- {4,4} -> {2,8} shrinks within the built capacity and succeeds,
 * with fresh zero-state exponents. */
void testBindRederivesFlattenBfpGroupingFromTheLiveInput(void) {
    uint8_t inputExponents[4];
    bfpQConfig_t inputQc;
    initBfpQConfigGroupedInto(8, 8, HALF_AWAY, 4, 4, inputExponents, &inputQc);
    quantization_t inputQ = {.type = BFP, .qConfig = &inputQc};
    layer_t *model[1] = {flattenLayerInit()};
    inputLike_t in;
    tensor_t *x = makeInput(&in, (size_t[]){1, 4, 4}, 3, &inputQ);
    rematWireTable_t *t = initTable(model, 1, MSE, x);
    TEST_ASSERT_EQUAL_size_t(4, t->wires[1].expCapacity);
    rematWireTableBind(t, model, 1, MSE, REMAT_MODE_TRAIN, x);
    bfpQConfig_t *slabQc = rematActHdr(t, 1)->quantization->qConfig;
    TEST_ASSERT_EQUAL_size_t(4, slabQc->numGroups);
    TEST_ASSERT_EQUAL_size_t(4, slabQc->groupSize);

    slabQc->exponents[0] = 3; /* a producer's exponent write */
    initBfpQConfigGroupedInto(8, 8, HALF_AWAY, 2, 8, inputExponents, &inputQc);
    rematWireTableBind(t, model, 1, MSE, REMAT_MODE_TRAIN, x);
    TEST_ASSERT_EQUAL_size_t(2, slabQc->numGroups);
    TEST_ASSERT_EQUAL_size_t(8, slabQc->groupSize);
    TEST_ASSERT_EQUAL_UINT8(127, slabQc->exponents[0]); /* zero state: bias 2^(8-1)-1 */
    TEST_ASSERT_EQUAL_UINT8(127, slabQc->exponents[1]);

    rematWireTableFree(t);
    freeModel(model, 1);
}

/* SYM@12 -> @8 on the input carries qMaxBits 8 onto the Flatten
 * wire, so a stale width cannot slip past the #227 operand guard. */
void testBindCarriesSymQMaxBitsOntoTheFlattenWire(void) {
    symInt32QConfig_t inputQc;
    initSymInt32QConfigWithQMaxBits(HALF_AWAY, &inputQc, 12);
    quantization_t inputQ = {.type = SYM_INT32, .qConfig = &inputQc};
    layer_t *model[1] = {flattenLayerInit()};
    inputLike_t in;
    tensor_t *x = makeInput(&in, (size_t[]){1, 2, 3}, 3, &inputQ);
    rematWireTable_t *t = initTable(model, 1, MSE, x);
    rematWireTableBind(t, model, 1, MSE, REMAT_MODE_TRAIN, x);
    symInt32QConfig_t *slabQc = rematActHdr(t, 1)->quantization->qConfig;
    TEST_ASSERT_EQUAL_UINT8(12, slabQc->qMaxBits);
    inputQc.qMaxBits = 8;
    rematWireTableBind(t, model, 1, MSE, REMAT_MODE_TRAIN, x);
    TEST_ASSERT_EQUAL_UINT8(8, slabQc->qMaxBits);
    rematWireTableFree(t);
    freeModel(model, 1);
}

/* A packed borrowed input re-quantized between calls (8 -> 4 bits) is a
 * config edit, not a key change; the checker sizes ACT 0 from the live input. */
void testBindFollowsTheLivePackedInputBytes(void) {
    float scale = 1.f;
    symQConfig_t inputQc = {
        .scales = &scale, .numGroups = 1, .groupSize = 0, .roundingMode = HALF_AWAY, .qBits = 8};
    quantization_t inputQ = {.type = SYM, .qConfig = &inputQc};
    layer_t *model[1] = {makeRelu(&g_floatQ)};
    inputLike_t in;
    tensor_t *x = makeInput(&in, (size_t[]){1, 8}, 2, &inputQ);
    rematWireTable_t *t = initTable(model, 1, MSE, x);
    TEST_ASSERT_EQUAL_size_t(8, rematWireBytes(t, 0));
    inputQc.qBits = 4;
    rematWireTableBind(t, model, 1, MSE, REMAT_MODE_TRAIN, x);
    TEST_ASSERT_EQUAL_size_t(4, rematWireBytes(t, 0));
    rematWireTableFree(t);
    freeModel(model, 1);
}

/* P9 (table level): ACT 0 is the caller's tensor verbatim -- a table built on
 * sample A binds sample B, and B's header is not written. */
void testBindRunsSampleBOnATableBuiltOnSampleA(void) {
    layer_t *model[1] = {makeRelu(&g_floatQ)};
    inputLike_t inA;
    inputLike_t inB;
    float bytesB[4] = {1.f, 2.f, 3.f, 4.f};
    tensor_t *a = makeInput(&inA, (size_t[]){1, 4}, 2, &g_floatQ);
    tensor_t *b = makeInput(&inB, (size_t[]){1, 4}, 2, &g_floatQ);
    b->data = (uint8_t *)bytesB;
    inputLike_t snapshot = inB;
    rematWireTable_t *t = initTable(model, 1, MSE, a);
    rematWireTableBind(t, model, 1, MSE, REMAT_MODE_TRAIN, a);
    rematWireTableBind(t, model, 1, MSE, REMAT_MODE_TRAIN, b);
    TEST_ASSERT_EQUAL_PTR(b, rematActHdr(t, 0));
    TEST_ASSERT_EQUAL_PTR((uint8_t *)bytesB, b->data);
    TEST_ASSERT_EQUAL_MEMORY(snapshot.dims, inB.dims, sizeof inB.dims);
    TEST_ASSERT_EQUAL_MEMORY(snapshot.order, inB.order, sizeof inB.order);
    TEST_ASSERT_EQUAL_PTR(&g_floatQ, b->quantization);
    rematWireTableFree(t);
    freeModel(model, 1);
}

/* The bind's per-wire derivation scratch lives inside the
 * table block. LP64 layout arithmetic (not a scan-model pin): table struct 144
 * + 24 records x 40 + gradIdOf 26 + layerType/frozen 24, rounded to 8, + the
 * input key 48 = 1208; the bind scratch 960; the headers 2680. Total 4848.
 * Every host preset is LP64. */
void testHarSlabHoldsTheBindScratch(void) {
    layer_t *model[HAR_N];
    buildHar(model, false);
    inputLike_t in;
    rematWireTable_t *t = initTable(model, HAR_N, CROSS_ENTROPY, makeHarInput(&in));
    TEST_ASSERT_EQUAL_size_t(4848, t->slabBytes);
    const uint8_t *scratch = (const uint8_t *)t->bindScratch;
    TEST_ASSERT_TRUE(scratch >= (const uint8_t *)(t->inputOrder + t->inputRank));
    TEST_ASSERT_TRUE(scratch + t->numWires * sizeof(rematWireFact_t) <=
                     (const uint8_t *)t->wires[1].hdr);
    ASSERT_ALIGNED(t->bindScratch, rematWireFact_t);
    rematWireTableFree(t);
    freeModel(model, HAR_N);
}

void testUnbindClearsTheBorrowedInputOnly(void) {
    layer_t *model[1] = {makeRelu(&g_floatQ)};
    inputLike_t in;
    tensor_t *x = makeInput(&in, (size_t[]){1, 4}, 2, &g_floatQ);
    rematWireTable_t *t = initTable(model, 1, MSE, x);
    rematWireTableBind(t, model, 1, MSE, REMAT_MODE_TRAIN, x);
    tensor_t *act1 = rematActHdr(t, 1);
    rematWireTableUnbind(t);
    TEST_ASSERT_NULL(rematActHdr(t, 0));
    TEST_ASSERT_EQUAL_PTR(act1, rematActHdr(t, 1));
    rematWireTableFree(t);
    freeModel(model, 1);
}

/* ---- the ASan death callback ---- */

#ifdef ODT_TEST_ASAN
static void overrunAHeapBlockUnderTheCallback(void) {
    odtInstallAsanDeathExit();
    uint8_t *block = reserveMemory(4);
    volatile size_t past = 4;
    block[past] = 1;
}

/* The live-RED run that decides 4e: an ASan report inside a death-test child
 * must surface as exit 86, not as the SIGABRT of abort_on_error=1. */
void testAsanDeathCallbackExitsWithADistinctCode(void) {
    ASSERT_EXITS_WITH(ODT_ASAN_DEATH_EXIT, overrunAHeapBlockUnderTheCallback());
}
#endif

/* ---- the schedule key at bind ---- */

void testBindExitsOnAChangedModelSize(void) {
    layer_t *model[2] = {makeRelu(&g_floatQ), makeRelu(&g_floatQ)};
    inputLike_t in;
    tensor_t *x = makeInput(&in, (size_t[]){1, 4}, 2, &g_floatQ);
    rematWireTable_t *t = initTable(model, 2, MSE, x);
    ASSERT_EXITS_WITH_OUTPUT(1, "key mismatch on 'modelSize': built 2, live 1",
                             rematWireTableBind(t, model, 1, MSE, REMAT_MODE_TRAIN, x));
    rematWireTableFree(t);
    freeModel(model, 2);
}

void testBindExitsOnAChangedLossType(void) {
    layer_t *model[2] = {makeLinear(4, 3, false), makeSoftmax()};
    inputLike_t in;
    tensor_t *x = makeInput(&in, (size_t[]){1, 4}, 2, &g_floatQ);
    rematWireTable_t *t = initTable(model, 2, CROSS_ENTROPY, x);
    ASSERT_EXITS_WITH_OUTPUT(1, "key mismatch on 'lossType': built 1, live 0",
                             rematWireTableBind(t, model, 2, MSE, REMAT_MODE_TRAIN, x));
    rematWireTableFree(t);
    freeModel(model, 2);
}

void testBindExitsOnALayerTypeSwap(void) {
    layer_t *relu = makeRelu(&g_floatQ);
    layer_t *softmax = makeSoftmax();
    layer_t *model[2] = {makeRelu(&g_floatQ), relu};
    inputLike_t in;
    tensor_t *x = makeInput(&in, (size_t[]){1, 4}, 2, &g_floatQ);
    rematWireTable_t *t = initTable(model, 2, MSE, x);
    model[1] = softmax; /* same shape, same dtype: only the type changes */
    ASSERT_EXITS_WITH_OUTPUT(1, "key mismatch on 'layerType[1]': built 1, live 6",
                             rematWireTableBind(t, model, 2, MSE, REMAT_MODE_TRAIN, x));
    rematWireTableFree(t);
    model[1] = relu;
    freeModel(model, 2);
    freeSoftmaxLayer(softmax);
}

/* Freezing the deepest trainable layer moves deepest (0 -> 3 on HAR). */
void testBindExitsWhenFreezingMovesDeepest(void) {
    layer_t *model[HAR_N];
    buildHar(model, false);
    inputLike_t in;
    tensor_t *x = makeHarInput(&in);
    rematWireTable_t *t = initTable(model, HAR_N, CROSS_ENTROPY, x);
    model[0]->config->conv1d->frozen = true;
    ASSERT_EXITS_WITH_OUTPUT(
        1, "key mismatch on 'deepest': built 0, live 3",
        rematWireTableBind(t, model, HAR_N, CROSS_ENTROPY, REMAT_MODE_TRAIN, x));
    model[0]->config->conv1d->frozen = false;
    rematWireTableFree(t);
    freeModel(model, HAR_N);
}

/* Freezing a layer above deepest keeps deepest; the frozen[] key catches it. */
void testBindExitsWhenFreezingALayerAboveDeepest(void) {
    layer_t *model[HAR_N];
    buildHar(model, false);
    inputLike_t in;
    tensor_t *x = makeHarInput(&in);
    rematWireTable_t *t = initTable(model, HAR_N, CROSS_ENTROPY, x);
    model[3]->config->conv1d->frozen = true;
    ASSERT_EXITS_WITH_OUTPUT(
        1, "key mismatch on 'frozen[3]': built 0, live 1",
        rematWireTableBind(t, model, HAR_N, CROSS_ENTROPY, REMAT_MODE_TRAIN, x));
    model[3]->config->conv1d->frozen = false;
    rematWireTableFree(t);
    freeModel(model, HAR_N);
}

void testBindExitsOnAChangedInputRank(void) {
    layer_t *model[1] = {makeRelu(&g_floatQ)};
    inputLike_t in;
    inputLike_t in3;
    rematWireTable_t *t = initTable(model, 1, MSE, makeInput(&in, (size_t[]){1, 4}, 2, &g_floatQ));
    tensor_t *x3 = makeInput(&in3, (size_t[]){1, 1, 4}, 3, &g_floatQ);
    ASSERT_EXITS_WITH_OUTPUT(1, "key mismatch on wire ACT 0, field 'rank': built 2, live 3",
                             rematWireTableBind(t, model, 1, MSE, REMAT_MODE_TRAIN, x3));
    rematWireTableFree(t);
    freeModel(model, 1);
}

/* A B change (#152: the key holds the exact B) dies on ACT 0's dims[0], which phase 1
 * step 1 compares before any wire. */
void testBindExitsOnAChangedBatch(void) {
    layer_t *model[1] = {makeRelu(&g_floatQ)};
    inputLike_t in;
    inputLike_t in2;
    rematWireTable_t *t = initTable(model, 1, MSE, makeInput(&in, (size_t[]){1, 4}, 2, &g_floatQ));
    tensor_t *x2 = makeInput(&in2, (size_t[]){2, 4}, 2, &g_floatQ);
    ASSERT_EXITS_WITH_OUTPUT(1, "key mismatch on wire ACT 0, field 'dims[0]': built 1, live 2",
                             rematWireTableBind(t, model, 1, MSE, REMAT_MODE_TRAIN, x2));
    rematWireTableFree(t);
    freeModel(model, 1);
}

void testBindExitsOnAChangedInputOrder(void) {
    layer_t *model[1] = {makeRelu(&g_floatQ)};
    inputLike_t in;
    inputLike_t inT;
    rematWireTable_t *t = initTable(model, 1, MSE, makeInput(&in, (size_t[]){1, 4}, 2, &g_floatQ));
    tensor_t *xT = makeInput(&inT, (size_t[]){1, 4}, 2, &g_floatQ);
    inT.order[0] = 1;
    inT.order[1] = 0;
    ASSERT_EXITS_WITH_OUTPUT(1, "key mismatch on wire ACT 0, field 'order[0]': built 0, live 1",
                             rematWireTableBind(t, model, 1, MSE, REMAT_MODE_TRAIN, xT));
    rematWireTableFree(t);
    freeModel(model, 1);
}

void testBindExitsOnAChangedInputDtype(void) {
    symInt32QConfig_t symQc;
    initSymInt32QConfigWithQMaxBits(HALF_AWAY, &symQc, 12);
    quantization_t symQ = {.type = SYM_INT32, .qConfig = &symQc};
    layer_t *model[1] = {makeRelu(&g_floatQ)};
    inputLike_t in;
    inputLike_t inS;
    rematWireTable_t *t = initTable(model, 1, MSE, makeInput(&in, (size_t[]){1, 4}, 2, &g_floatQ));
    tensor_t *xS = makeInput(&inS, (size_t[]){1, 4}, 2, &symQ);
    ASSERT_EXITS_WITH_OUTPUT(1, "key mismatch on wire ACT 0, field 'dtype': built 1, live 2",
                             rematWireTableBind(t, model, 1, MSE, REMAT_MODE_TRAIN, xS));
    rematWireTableFree(t);
    freeModel(model, 1);
}

/* A BFP width edit changes the wire's bytes (8 -> 4 for 8 elements). */
void testBindExitsOnAChangedWireByteCount(void) {
    uint8_t exponent[1];
    bfpQConfig_t tmplQc;
    initBfpQConfigInto(8, 8, HALF_AWAY, exponent, &tmplQc);
    quantization_t tmplQ = {.type = BFP, .qConfig = &tmplQc};
    layer_t *model[1] = {makeQuant(&tmplQ, &g_floatQ)};
    inputLike_t in;
    tensor_t *x = makeInput(&in, (size_t[]){1, 8}, 2, &g_floatQ);
    rematWireTable_t *t = initTable(model, 1, MSE, x);
    tmplQc.mantissaBits = 4;
    ASSERT_EXITS_WITH_OUTPUT(1, "key mismatch on wire ACT 1, field 'bytes': built 8, live 4",
                             rematWireTableBind(t, model, 1, MSE, REMAT_MODE_TRAIN, x));
    rematWireTableFree(t);
    freeModel(model, 1);
}

/* Check before write, dtype twin: a byte-neutral FLOAT32 -> SYM_INT32 template edit.
 * FLOAT32 reserved no qConfig, so a write before the check goes through a NULL
 * qConfig: a crash (or, under ASan, exit 86), never the named exit. */
static void bindUnderTheAsanCallback(rematWireTable_t *t, layer_t **model, size_t n,
                                     lossFuncType_t lt, tensor_t *x) {
    odtInstallAsanDeathExit();
    rematWireTableBind(t, model, n, lt, REMAT_MODE_TRAIN, x);
}

void testBindFloatToSymTemplateEditExitsBeforeSlabWrite(void) {
    quantization_t tmplQ = {.type = FLOAT32, .qConfig = NULL};
    symInt32QConfig_t symQc;
    initSymInt32QConfigWithQMaxBits(HALF_AWAY, &symQc, 12);
    layer_t *model[1] = {makeRelu(&tmplQ)};
    inputLike_t in;
    tensor_t *x = makeInput(&in, (size_t[]){1, 4}, 2, &g_floatQ);
    rematWireTable_t *t = initTable(model, 1, MSE, x);
    tmplQ.type = SYM_INT32;
    tmplQ.qConfig = &symQc;
    ASSERT_EXITS_WITH_OUTPUT(1, "key mismatch on wire ACT 1, field 'dtype': built 1, live 2",
                             bindUnderTheAsanCallback(t, model, 1, MSE, x));
    rematWireTableFree(t);
    freeModel(model, 1);
}

/* Check before write, BFP twin: exactly one BFP wire, per-tensor (expCapacity 1), so its
 * exponent byte is the last object of the table block. A
 * grouped edit derives 4 groups; a write before the check would put 3 bytes
 * past the block end, which ASan reports. */
void testBindGroupedBfpEditExitsBeforeSlabWrite(void) {
    uint8_t tmplExponents[4];
    bfpQConfig_t tmplQc;
    initBfpQConfigInto(8, 8, HALF_AWAY, tmplExponents, &tmplQc);
    quantization_t tmplQ = {.type = BFP, .qConfig = &tmplQc};
    layer_t *model[1] = {makeQuant(&tmplQ, &g_floatQ)};
    inputLike_t in;
    tensor_t *x = makeInput(&in, (size_t[]){1, 8}, 2, &g_floatQ);
    rematWireTable_t *t = initTable(model, 1, MSE, x);
    TEST_ASSERT_EQUAL_size_t(1, t->wires[1].expCapacity);
    initBfpQConfigGroupedInto(8, 8, HALF_AWAY, 4, 2, tmplExponents, &tmplQc);
    ASSERT_EXITS_WITH_OUTPUT(
        1, "key mismatch on wire ACT 1, field 'numGroups': 4 groups exceed expCapacity 1",
        bindUnderTheAsanCallback(t, model, 1, MSE, x));
    rematWireTableFree(t);
    freeModel(model, 1);
}

/* ---- the row SDK ---- */

void testWireBindSetsDataCountsBytesAndBumpsBindGen(void) {
    seedFixture_t f;
    buildBfpSeedFixture(&f);
    uint32_t act1[8];
    uint32_t act2[2];
    rematWireBind(f.t, SEED_ACT1, (uint8_t *)act1);
    TEST_ASSERT_EQUAL_PTR((uint8_t *)act1, rematWireHdr(f.t, SEED_ACT1)->data);
    TEST_ASSERT_EQUAL_UINT32(1, f.t->wires[SEED_ACT1].bindGen);
    TEST_ASSERT_EQUAL_size_t(32, f.t->liveBytes);
    rematWireBind(f.t, SEED_ACT2, (uint8_t *)act2);
    TEST_ASSERT_EQUAL_size_t(40, f.t->liveBytes);
    TEST_ASSERT_EQUAL_size_t(40, f.t->observedPeakLiveBytes);
    freeSeedFixture(&f);
}

void testWireReleaseClearsDataAndKeepsThePeak(void) {
    seedFixture_t f;
    buildBfpSeedFixture(&f);
    uint32_t act1[8];
    uint32_t act2[2];
    rematWireBind(f.t, SEED_ACT1, (uint8_t *)act1);
    rematWireBind(f.t, SEED_ACT2, (uint8_t *)act2);
    rematWireRelease(f.t, SEED_ACT1);
    TEST_ASSERT_NULL(rematWireHdr(f.t, SEED_ACT1)->data);
    TEST_ASSERT_EQUAL_size_t(8, f.t->liveBytes);
    TEST_ASSERT_EQUAL_size_t(40, f.t->observedPeakLiveBytes);
    TEST_ASSERT_EQUAL_UINT32(1, f.t->wires[SEED_ACT1].bindGen); /* Release keeps the generation */
    rematWireBind(f.t, SEED_ACT1, (uint8_t *)act1);
    TEST_ASSERT_EQUAL_UINT32(2, f.t->wires[SEED_ACT1].bindGen);
    freeSeedFixture(&f);
}

/* Every bind starts a call -- generations, live bytes and the
 * observed peak restart at 0 (the peak is "over the last call"). */
void testTableBindResetsBindGenLiveBytesAndThePeak(void) {
    seedFixture_t f;
    buildBfpSeedFixture(&f);
    uint32_t act1[8];
    uint32_t act2[2];
    rematWireBind(f.t, SEED_ACT1, (uint8_t *)act1);
    rematWireBind(f.t, SEED_ACT2, (uint8_t *)act2);
    rematWireRelease(f.t, SEED_ACT1);
    rematWireRelease(f.t, SEED_ACT2);
    rematWireTableBind(f.t, f.model, 2, MSE, REMAT_MODE_TRAIN, f.x);
    for (uint16_t id = 0; id < f.t->numWires; id++) {
        TEST_ASSERT_EQUAL_UINT32(0, f.t->wires[id].bindGen);
    }
    TEST_ASSERT_EQUAL_size_t(0, f.t->liveBytes);
    TEST_ASSERT_EQUAL_size_t(0, f.t->observedPeakLiveBytes);
    freeSeedFixture(&f);
}

/* The seed takes ACT 2's LIVE config fields when its range opens (after the
 * forward, the pre-remat driver's timing), never its exponents: a fresh zero
 * state. */
void testWireBindDerivesTheSeedFromTheLiveActHeader(void) {
    seedFixture_t f;
    buildBfpSeedFixture(&f);
    uint32_t act2[2];
    uint32_t seed[2];
    rematWireBind(f.t, SEED_ACT2, (uint8_t *)act2);
    bfpQConfig_t *act2Qc = rematWireHdr(f.t, SEED_ACT2)->quantization->qConfig;
    act2Qc->exponents[0] = 5; /* the producer's OUT_WRITE */
    rematWireBind(f.t, SEED_GRAD2, (uint8_t *)seed);
    tensor_t *seedHdr = rematGradHdr(f.t, 2);
    assertDims(seedHdr->shape, (size_t[]){1, 8}, 2);
    TEST_ASSERT_EQUAL_size_t(0, seedHdr->shape->orderOfDimensions[0]);
    TEST_ASSERT_EQUAL_size_t(1, seedHdr->shape->orderOfDimensions[1]);
    TEST_ASSERT_EQUAL_INT(BFP, seedHdr->quantization->type);
    bfpQConfig_t *seedQc = seedHdr->quantization->qConfig;
    TEST_ASSERT_EQUAL_size_t(1, seedQc->numGroups);
    TEST_ASSERT_EQUAL_size_t(0, seedQc->groupSize);
    TEST_ASSERT_EQUAL_UINT8(8, seedQc->mantissaBits);
    TEST_ASSERT_EQUAL_UINT8(127, seedQc->exponents[0]);
    TEST_ASSERT_EQUAL_PTR((uint8_t *)seed, seedHdr->data);
    freeSeedFixture(&f);
}

void testWireBindInheritedSymTakesConfigNotScale(void) {
    symInt32QConfig_t symQc;
    initSymInt32QConfigWithQMaxBits(SR_HALF_AWAY, &symQc, 10);
    quantization_t symQ = {.type = SYM_INT32, .qConfig = &symQc};
    layer_t *model[2] = {makeLinear(2, 4, false), makeQuant(&symQ, &g_floatQ)};
    inputLike_t in;
    tensor_t *x = makeInput(&in, (size_t[]){1, 2}, 2, &g_floatQ);
    rematWireTable_t *t = initTable(model, 2, MSE, x);
    rematWireTableBind(t, model, 2, MSE, REMAT_MODE_TRAIN, x);
    uint32_t act2[4];
    uint32_t seed[4];
    rematWireBind(t, 2, (uint8_t *)act2);
    symInt32QConfig_t *act2Qc = rematActHdr(t, 2)->quantization->qConfig;
    act2Qc->scale = 0.25f;
    rematWireBind(t, rematGradId(t, 2), (uint8_t *)seed);
    symInt32QConfig_t *seedQc = rematGradHdr(t, 2)->quantization->qConfig;
    TEST_ASSERT_EQUAL_UINT8(10, seedQc->qMaxBits);
    TEST_ASSERT_EQUAL_INT(SR_HALF_AWAY, seedQc->roundingMode);
    TEST_ASSERT_EQUAL_FLOAT(1.0f, seedQc->scale);
    rematWireTableFree(t);
    freeModel(model, 2);
}

/* FLOAT32 and SYM_INT32 both charge 4 B/element (wireBytes), so this dtype
 * edit is byte-neutral -- the bytes check cannot catch it, only the dtype
 * check can. Without it, initFloat32Quantization would silently orphan the
 * slab's reserved SYM_INT32 qConfig. */
void testWireBindInheritedGradChecksTheLiveDtypeWhenByteNeutral(void) {
    symInt32QConfig_t symQc;
    initSymInt32QConfigWithQMaxBits(SR_HALF_AWAY, &symQc, 10);
    quantization_t symQ = {.type = SYM_INT32, .qConfig = &symQc};
    layer_t *model[2] = {makeLinear(2, 4, false), makeQuant(&symQ, &g_floatQ)};
    inputLike_t in;
    tensor_t *x = makeInput(&in, (size_t[]){1, 2}, 2, &g_floatQ);
    rematWireTable_t *t = initTable(model, 2, MSE, x);
    rematWireTableBind(t, model, 2, MSE, REMAT_MODE_TRAIN, x);
    uint32_t act2[4];
    uint32_t seed[4];
    rematWireBind(t, 2, (uint8_t *)act2);
    rematActHdr(t, 2)->quantization->type = FLOAT32;
    ASSERT_EXITS_WITH_OUTPUT(1, "rematWireBind: wire GRAD 2 field 'dtype'",
                             rematWireBind(t, rematGradId(t, 2), (uint8_t *)seed));
    rematWireTableFree(t);
    freeModel(model, 2);
}

/* Check before write (table-level twin of PR2's decorator test): a live ACT header whose
 * grouping grew past the seed's slab capacity must exit by name before
 * initBfpQConfigGroupedInto writes 4 exponents into a 1-byte tail at the block
 * end. */
static void bindSeedUnderTheAsanCallback(seedFixture_t *f, uint8_t *bytes) {
    odtInstallAsanDeathExit();
    rematWireBind(f->t, SEED_GRAD2, bytes);
}

void testWireBindInheritedGradChecksCapacityBeforeWrite(void) {
    seedFixture_t f;
    buildBfpSeedFixture(&f);
    uint32_t act2[2];
    uint32_t seed[2];
    rematWireBind(f.t, SEED_ACT2, (uint8_t *)act2);
    bfpQConfig_t *act2Qc = rematWireHdr(f.t, SEED_ACT2)->quantization->qConfig;
    act2Qc->numGroups = 4; /* fields only: no exponent byte is written */
    act2Qc->groupSize = 2;
    ASSERT_EXITS_WITH_OUTPUT(1, "wire GRAD 2 needs 4 BFP exponent groups, above its expCapacity 1",
                             bindSeedUnderTheAsanCallback(&f, (uint8_t *)seed));
    freeSeedFixture(&f);
}

void testWireBindInheritedGradChecksTheLiveDtype(void) {
    seedFixture_t f;
    buildBfpSeedFixture(&f);
    uint32_t act2[2];
    uint32_t seed[2];
    rematWireBind(f.t, SEED_ACT2, (uint8_t *)act2);
    rematWireHdr(f.t, SEED_ACT2)->quantization->type = FLOAT32;
    ASSERT_EXITS_WITH_OUTPUT(1, "rematWireBind: wire GRAD 2 field 'dtype'",
                             rematWireBind(f.t, SEED_GRAD2, (uint8_t *)seed));
    rematWireHdr(f.t, SEED_ACT2)->quantization->type = BFP;
    freeSeedFixture(&f);
}

/* A live source whose payload size changed (BFP m8 -> m16 doubles 8 -> 16 B)
 * would overrun the seed's bytes; the check runs before any slab write. */
void testWireBindInheritedGradChecksTheLiveBytes(void) {
    seedFixture_t f;
    buildBfpSeedFixture(&f);
    uint32_t act2[4];
    uint32_t seed[4];
    rematWireBind(f.t, SEED_ACT2, (uint8_t *)act2);
    bfpQConfig_t *act2Qc = rematWireHdr(f.t, SEED_ACT2)->quantization->qConfig;
    act2Qc->mantissaBits = 16;
    ASSERT_EXITS_WITH_OUTPUT(1, "rematWireBind: wire GRAD 2 field 'bytes'",
                             rematWireBind(f.t, SEED_GRAD2, (uint8_t *)seed));
    act2Qc->mantissaBits = 8;
    freeSeedFixture(&f);
}

void testWireBindInheritedGradChecksTheLiveRank(void) {
    seedFixture_t f;
    buildBfpSeedFixture(&f);
    uint32_t act2[2];
    uint32_t seed[2];
    rematWireBind(f.t, SEED_ACT2, (uint8_t *)act2);
    rematWireHdr(f.t, SEED_ACT2)->shape->numberOfDimensions = 3;
    ASSERT_EXITS_WITH_OUTPUT(1, "rematWireBind: wire GRAD 2 field 'rank'",
                             rematWireBind(f.t, SEED_GRAD2, (uint8_t *)seed));
    rematWireHdr(f.t, SEED_ACT2)->shape->numberOfDimensions = 2;
    freeSeedFixture(&f);
}

/* The test above sets numberOfDimensions past the header's allocated
 * dims[2]/order[2] arrays, so it (coincidentally) also reaches the 'bytes'
 * exit through an out-of-bounds read of slab memory. This variant swaps in
 * fully-backed rank-3 arrays ([1,8] -> [1,8,1], a byte-count-preserving
 * reshape), so only the rank check can catch it. */
void testWireBindInheritedGradChecksTheLiveRankWhenByteNeutral(void) {
    seedFixture_t f;
    buildBfpSeedFixture(&f);
    uint32_t act2[2];
    uint32_t seed[2];
    rematWireBind(f.t, SEED_ACT2, (uint8_t *)act2);
    shape_t *act2Shape = rematWireHdr(f.t, SEED_ACT2)->shape;
    size_t *origDims = act2Shape->dimensions;
    size_t *origOrder = act2Shape->orderOfDimensions;
    size_t dims3[3] = {1, 8, 1};
    size_t order3[3] = {0, 1, 2};
    act2Shape->dimensions = dims3;
    act2Shape->orderOfDimensions = order3;
    act2Shape->numberOfDimensions = 3;
    ASSERT_EXITS_WITH_OUTPUT(1, "rematWireBind: wire GRAD 2 field 'rank'",
                             rematWireBind(f.t, SEED_GRAD2, (uint8_t *)seed));
    act2Shape->dimensions = origDims;
    act2Shape->orderOfDimensions = origOrder;
    act2Shape->numberOfDimensions = 2;
    freeSeedFixture(&f);
}

/* ACT 0 is the caller's tensor; a row that binds it would overwrite the
 * caller's ->data. */
void testWireBindRefusesTheBorrowedInput(void) {
    seedFixture_t f;
    buildBfpSeedFixture(&f);
    uint32_t bytes[2];
    ASSERT_EXITS_WITH_OUTPUT(1, "wire ACT 0 is the caller's borrowed input; it is never bound",
                             rematWireBind(f.t, 0, (uint8_t *)bytes));
    freeSeedFixture(&f);
}

void testWireReleaseRefusesTheBorrowedInput(void) {
    seedFixture_t f;
    buildBfpSeedFixture(&f);
    ASSERT_EXITS_WITH_OUTPUT(1, "wire ACT 0 is the caller's borrowed input; it is never released",
                             rematWireRelease(f.t, 0));
    freeSeedFixture(&f);
}

/* Unbalanced SDK calls would double-count or wrap liveBytes. */
void testWireBindRefusesAnAlreadyBoundWire(void) {
    seedFixture_t f;
    buildBfpSeedFixture(&f);
    uint32_t act1[8];
    rematWireBind(f.t, SEED_ACT1, (uint8_t *)act1);
    ASSERT_EXITS_WITH_OUTPUT(1, "rematWireBind: wire ACT 1 is already bound",
                             rematWireBind(f.t, SEED_ACT1, (uint8_t *)act1));
    freeSeedFixture(&f);
}

void testWireReleaseRefusesAnUnboundWire(void) {
    seedFixture_t f;
    buildBfpSeedFixture(&f);
    ASSERT_EXITS_WITH_OUTPUT(1, "rematWireRelease: wire ACT 1 is not bound",
                             rematWireRelease(f.t, SEED_ACT1));
    freeSeedFixture(&f);
}

/* I2: REMAT_NONE is handed out by rematGradId and by the walk functions, and
 * an unchecked bind/release would write or subtract through a garbage
 * header, corrupting memory silently on an MCU. */
void testWireBindRefusesAWireIdOutOfRange(void) {
    seedFixture_t f;
    buildBfpSeedFixture(&f);
    uint32_t bytes[2];
    ASSERT_EXITS_WITH_OUTPUT(1, "rematWireBind: wire id 65535 out of range",
                             rematWireBind(f.t, REMAT_NONE, (uint8_t *)bytes));
    freeSeedFixture(&f);
}

void testWireReleaseRefusesAWireIdOutOfRange(void) {
    seedFixture_t f;
    buildBfpSeedFixture(&f);
    ASSERT_EXITS_WITH_OUTPUT(1, "rematWireRelease: wire id 65535 out of range",
                             rematWireRelease(f.t, REMAT_NONE));
    freeSeedFixture(&f);
}

/* m2: bytes == NULL would still increment liveBytes and bindGen, and under
 * ODT_REMAT_VERIFY poison a NULL pointer. */
void testWireBindRefusesNullBytes(void) {
    seedFixture_t f;
    buildBfpSeedFixture(&f);
    ASSERT_EXITS_WITH_OUTPUT(1, "rematWireBind: bytes is NULL",
                             rematWireBind(f.t, SEED_ACT1, NULL));
    freeSeedFixture(&f);
}

/* rematWireTableBind resets liveBytes to 0 but never clears a still-bound
 * wire's ->data (only Bind/Release write it), so a wire left bound
 * across a rebind reads as still-bound. Releasing it then would subtract from
 * a liveBytes that no longer reflects it: checked, so it exits by name
 * instead of wrapping size_t. */
void testWireReleaseExitsWhenLiveBytesWouldUnderflow(void) {
    seedFixture_t f;
    buildBfpSeedFixture(&f);
    uint32_t act1[8];
    rematWireBind(f.t, SEED_ACT1, (uint8_t *)act1);
    rematWireTableBind(f.t, f.model, 2, MSE, REMAT_MODE_TRAIN, f.x);
    ASSERT_EXITS_WITH_OUTPUT(1, "rematWireRelease: wire ACT 1 live bytes would underflow",
                             rematWireRelease(f.t, SEED_ACT1));
    freeSeedFixture(&f);
}

/* ---- ODT_REMAT_VERIFY poison ---- */

#ifdef ODT_REMAT_VERIFY
/* A signalling NaN: exponent all ones, quiet bit clear, payload non-zero. */
static bool isSignallingNan(uint32_t bits) {
    return (bits & 0x7F800000u) == 0x7F800000u && (bits & 0x00400000u) == 0u &&
           (bits & 0x003FFFFFu) != 0u;
}

/* One wire per wire dtype: ACT 1 FLOAT32 (16 B), ACT 2 SYM_INT32 (16 B),
 * ACT 3 BFP m8 per-tensor (4 B). */
typedef struct dtypeFixture {
    symInt32QConfig_t symQc;
    quantization_t symQ;
    uint8_t bfpExponent[1];
    bfpQConfig_t bfpQc;
    quantization_t bfpQ;
    layer_t *model[3];
    inputLike_t in;
    tensor_t *x;
    rematWireTable_t *t;
} dtypeFixture_t;

static void buildDtypeFixture(dtypeFixture_t *f) {
    initSymInt32QConfigWithQMaxBits(HALF_AWAY, &f->symQc, 12);
    f->symQ = (quantization_t){.type = SYM_INT32, .qConfig = &f->symQc};
    initBfpQConfigInto(8, 8, HALF_AWAY, f->bfpExponent, &f->bfpQc);
    f->bfpQ = (quantization_t){.type = BFP, .qConfig = &f->bfpQc};
    f->model[0] = makeRelu(&g_floatQ);
    f->model[1] = makeQuant(&f->symQ, &g_floatQ);
    f->model[2] = makeQuant(&f->bfpQ, &g_floatQ);
    f->x = makeInput(&f->in, (size_t[]){1, 4}, 2, &g_floatQ);
    f->t = initTable(f->model, 3, MSE, f->x);
    rematWireTableBind(f->t, f->model, 3, MSE, REMAT_MODE_TRAIN, f->x);
}

static void assertPoisoned(const uint32_t *f32, const uint32_t *sym, const uint8_t *bfp) {
    for (size_t k = 0; k < 4; k++) {
        TEST_ASSERT_TRUE_MESSAGE(isSignallingNan(f32[k]), "FLOAT32 word is not an sNaN");
        TEST_ASSERT_EQUAL_HEX32(0x80000000u, sym[k]);
        TEST_ASSERT_EQUAL_HEX8(0xA5, bfp[k]);
    }
}

/* Poison at Bind stops calloc zeros from masking a read of never-written bytes. */
void testWireBindPoisonsFreshBytesByDtype(void) {
    dtypeFixture_t f;
    buildDtypeFixture(&f);
    uint32_t f32[4] = {0};
    uint32_t sym[4] = {0};
    uint8_t bfp[4] = {0};
    rematWireBind(f.t, 1, (uint8_t *)f32);
    rematWireBind(f.t, 2, (uint8_t *)sym);
    rematWireBind(f.t, 3, bfp);
    assertPoisoned(f32, sym, bfp);
    rematWireTableFree(f.t);
    freeModel(f.model, 3);
}

/* Poison at Release (while the bytes are still owned) makes a read of released
 * bytes loud. */
void testWireReleasePoisonsTheOldBytesByDtype(void) {
    dtypeFixture_t f;
    buildDtypeFixture(&f);
    uint32_t f32[4];
    uint32_t sym[4];
    uint8_t bfp[4];
    rematWireBind(f.t, 1, (uint8_t *)f32);
    rematWireBind(f.t, 2, (uint8_t *)sym);
    rematWireBind(f.t, 3, bfp);
    for (size_t k = 0; k < 4; k++) { /* the producers' writes */
        f32[k] = 0x3F800000u;        /* 1.0f */
        sym[k] = 7u;
        bfp[k] = 0x11u;
    }
    rematWireRelease(f.t, 1);
    rematWireRelease(f.t, 2);
    rematWireRelease(f.t, 3);
    assertPoisoned(f32, sym, bfp);
    rematWireTableFree(f.t);
    freeModel(f.model, 3);
}
#endif

/* ---- the static plan ---- */

/* FORWARD 0..n-1, LOSS_FORWARD, LOSS_BACKWARD, BACKWARD top..deepest.
 * HAR: 12 + 1 + 1 + 11 = 25 steps (a scan-model pin). */
void testStoreAllHarStepOrder(void) {
    layer_t *model[HAR_N];
    buildHar(model, false);
    inputLike_t in;
    rematWireTable_t *t = initTable(model, HAR_N, CROSS_ENTROPY, makeHarInput(&in));
    rematPlan_t *p = buildPlan(t, model, NULL);
    const rematProgram_t *train = &p->train;
    TEST_ASSERT_EQUAL_size_t(25, train->numSteps);
    for (uint16_t l = 0; l < 12; l++) {
        TEST_ASSERT_EQUAL_UINT8(REMAT_STEP_FORWARD, train->steps[l].kind);
        TEST_ASSERT_EQUAL_UINT16(l, train->steps[l].layer);
    }
    TEST_ASSERT_EQUAL_UINT8(REMAT_STEP_LOSS_FORWARD, train->steps[12].kind);
    TEST_ASSERT_EQUAL_UINT16(12, train->steps[12].layer);
    TEST_ASSERT_EQUAL_UINT8(REMAT_STEP_LOSS_BACKWARD, train->steps[13].kind);
    TEST_ASSERT_EQUAL_UINT16(12, train->steps[13].layer);
    for (uint16_t k = 0; k < 11; k++) { /* BACKWARD(l) = n + 2 + (top - l), top = 10 */
        TEST_ASSERT_EQUAL_UINT8(REMAT_STEP_BACKWARD, train->steps[14 + k].kind);
        TEST_ASSERT_EQUAL_UINT16(10 - k, train->steps[14 + k].layer);
    }
    rematPlanFree(p);
    rematWireTableFree(t);
    freeModel(model, HAR_N);
}

/* STORE_ALL: ACT j [FORWARD(j-1), last step]; seed [LOSS_BACKWARD,
 * BACKWARD(top)]; GRAD l [BACKWARD(l), BACKWARD(l-1)]. One range per slab
 * wire, in wire-id order, begins strictly ascending. */
void testStoreAllHarRanges(void) {
    layer_t *model[HAR_N];
    buildHar(model, false);
    inputLike_t in;
    rematWireTable_t *t = initTable(model, HAR_N, CROSS_ENTROPY, makeHarInput(&in));
    rematPlan_t *p = buildPlan(t, model, NULL);
    const rematProgram_t *train = &p->train;
    TEST_ASSERT_EQUAL_size_t(23, train->numRanges);
    for (uint16_t j = 1; j <= 12; j++) {
        TEST_ASSERT_EQUAL_UINT16(j, train->ranges[j - 1].wire);
        TEST_ASSERT_EQUAL_UINT16(j - 1, train->ranges[j - 1].begin);
        TEST_ASSERT_EQUAL_UINT16(24, train->ranges[j - 1].end);
    }
    TEST_ASSERT_EQUAL_UINT16(13, train->ranges[12].wire); /* the seed */
    TEST_ASSERT_EQUAL_UINT16(13, train->ranges[12].begin);
    TEST_ASSERT_EQUAL_UINT16(14, train->ranges[12].end);
    for (uint16_t id = 14; id < 24; id++) {
        uint16_t l = t->wires[id].index;
        TEST_ASSERT_EQUAL_UINT16(id, train->ranges[id - 1].wire);
        TEST_ASSERT_EQUAL_UINT16(14 + (10 - l), train->ranges[id - 1].begin);
        TEST_ASSERT_EQUAL_UINT16(14 + (10 - (l - 1)), train->ranges[id - 1].end);
    }
    for (size_t r = 1; r < train->numRanges; r++) {
        TEST_ASSERT_TRUE(train->ranges[r - 1].begin < train->ranges[r].begin);
    }
    rematPlanFree(p);
    rematWireTableFree(t);
    freeModel(model, HAR_N);
}

void testEndOrderSortsRangeIdsByEndThenWire(void) {
    layer_t *model[HAR_N];
    buildHar(model, false);
    inputLike_t in;
    rematWireTable_t *t = initTable(model, HAR_N, CROSS_ENTROPY, makeHarInput(&in));
    rematPlan_t *p = buildPlan(t, model, NULL);
    const rematProgram_t *train = &p->train;
    bool seen[23] = {false};
    for (size_t k = 0; k < train->numRanges; k++) {
        uint16_t r = train->endOrder[k];
        TEST_ASSERT_TRUE(r < train->numRanges);
        TEST_ASSERT_FALSE(seen[r]);
        seen[r] = true;
        if (k > 0) {
            const rematRange_t *prev = &train->ranges[train->endOrder[k - 1]];
            const rematRange_t *cur = &train->ranges[r];
            TEST_ASSERT_TRUE(prev->end < cur->end ||
                             (prev->end == cur->end && prev->wire < cur->wire));
        }
    }
    rematPlanFree(p);
    rematWireTableFree(t);
    freeModel(model, HAR_N);
}

/* Walking every step opens each range at its begin and closes it at its
 * end, exactly once. */
void testWalkOpensAndClosesEachRangeOnceAtItsEndpoints(void) {
    layer_t *model[HAR_N];
    buildHar(model, false);
    inputLike_t in;
    rematWireTable_t *t = initTable(model, HAR_N, CROSS_ENTROPY, makeHarInput(&in));
    rematPlan_t *p = buildPlan(t, model, NULL);
    const rematProgram_t *train = &p->train;
    uint8_t opened[23] = {0};
    uint8_t closed[23] = {0};
    rematWalk_t walk = {0};
    for (walk.step = 0; walk.step < train->numSteps; walk.step++) {
        for (size_t r; (r = rematWalkOpening(train, &walk)) != REMAT_NONE;) {
            TEST_ASSERT_EQUAL_size_t(walk.step, train->ranges[r].begin);
            opened[r]++;
        }
        for (size_t r; (r = rematWalkClosing(train, &walk)) != REMAT_NONE;) {
            TEST_ASSERT_EQUAL_size_t(walk.step, train->ranges[r].end);
            closed[r]++;
        }
    }
    for (size_t r = 0; r < train->numRanges; r++) {
        TEST_ASSERT_EQUAL_UINT8(1, opened[r]);
        TEST_ASSERT_EQUAL_UINT8(1, closed[r]);
    }
    rematPlanFree(p);
    rematWireTableFree(t);
    freeModel(model, HAR_N);
}

/* Scan-model pins, written once, never retyped. */
void testStoreAllPeakHarIs74288(void) {
    layer_t *model[HAR_N];
    buildHar(model, false);
    inputLike_t in;
    TEST_ASSERT_EQUAL_size_t(74288, peakOf(model, HAR_N, CROSS_ENTROPY, makeHarInput(&in), NULL));
    freeModel(model, HAR_N);
}

void testStoreAllPeakMnistCnnIs175824(void) {
    layer_t *model[MNIST_N];
    buildMnistCnn(model);
    inputLike_t in;
    tensor_t *x = makeInput(&in, (size_t[]){1, 1, 784}, 3, &g_floatQ);
    TEST_ASSERT_EQUAL_size_t(175824, peakOf(model, MNIST_N, CROSS_ENTROPY, x, NULL));
    freeModel(model, MNIST_N);
}

/* #380: three convs frozen leaves only Linear(10) trainable, so deepest ==
 * top == 10 and the backward phase truncates to a single BACKWARD step; the
 * seed is the only GRAD wire (its range ends at that one step, not after a
 * multi-layer descent). */
void testStoreAllPeakFinetuneStage2Is57928(void) {
    layer_t *model[HAR_N];
    buildHar(model, true);
    inputLike_t in;
    rematWireTable_t *t = initTable(model, HAR_N, CROSS_ENTROPY, makeHarInput(&in));
    rematPlan_t *p = buildPlan(t, model, NULL);
    const rematProgram_t *train = &p->train;
    TEST_ASSERT_EQUAL_size_t(57928, train->peakLiveBytes);
    TEST_ASSERT_EQUAL_size_t(15, train->numSteps);
    TEST_ASSERT_EQUAL_UINT8(REMAT_STEP_BACKWARD, train->steps[14].kind);
    TEST_ASSERT_EQUAL_UINT16(10, train->steps[14].layer);
    TEST_ASSERT_EQUAL_UINT16(14, train->ranges[12].end);
    rematPlanFree(p);
    rematWireTableFree(t);
    freeModel(model, HAR_N);
}

/* One block of steps, ranges and endOrder for both programs. HAR TRAIN arrays:
 * 25 * 4 + 23 * 6 + 23 * 2 = 284 B after the (even-sized) plan struct; EVAL
 * arrays: 13 * 4 + 12 * 6 + 12 * 2 = 148 B. Every array is 2-aligned, so the
 * two programs pack without padding. */
void testPlanBlockHoldsStepsRangesAndEndOrder(void) {
    layer_t *model[HAR_N];
    buildHar(model, false);
    inputLike_t in;
    rematWireTable_t *t = initTable(model, HAR_N, CROSS_ENTROPY, makeHarInput(&in));
    rematPlan_t *p = buildPlan(t, model, NULL);
    TEST_ASSERT_EQUAL_size_t(sizeof(rematPlan_t) + 284u + 148u, p->blockBytes);
    const uint8_t *begin = (const uint8_t *)p;
    const uint8_t *end = begin + p->blockBytes;
    const rematProgram_t *eval = rematPlanProgram(p, REMAT_MODE_EVAL);
    TEST_ASSERT_TRUE((const uint8_t *)p->train.steps >= begin + sizeof(rematPlan_t));
    TEST_ASSERT_TRUE((const uint8_t *)eval->steps >=
                     (const uint8_t *)(p->train.endOrder + p->train.numRanges));
    TEST_ASSERT_TRUE((const uint8_t *)(eval->endOrder + eval->numRanges) <= end);
    rematPlanFree(p);
    rematWireTableFree(t);
    freeModel(model, HAR_N);
}

void testPlanProgramOfTrainIsTheTrainProgram(void) {
    layer_t *model[HAR_N];
    buildHar(model, false);
    inputLike_t in;
    rematWireTable_t *t = initTable(model, HAR_N, CROSS_ENTROPY, makeHarInput(&in));
    rematPlan_t *p = buildPlan(t, model, NULL);
    TEST_ASSERT_EQUAL_PTR(&p->train, rematPlanProgram(p, REMAT_MODE_TRAIN));
    TEST_ASSERT_EQUAL_size_t(25, rematPlanProgram(p, REMAT_MODE_TRAIN)->numSteps);
    rematPlanFree(p);
    rematWireTableFree(t);
    freeModel(model, HAR_N);
}

/* EVAL: FORWARD 0..n-1, LOSS_FORWARD; ACT j lives [FORWARD(j-1), FORWARD(j)]
 * and ACT n [FORWARD(n-1), LOSS_FORWARD], so range j-1 is [j-1, j]; no GRAD
 * range. The same program under every policy: eval never recomputes. */
void testEvalHarStepsAndRangesUnderEveryPolicy(void) {
    const rematPlanSpec_t *specs[2] = {NULL, &g_liveness};
    for (size_t k = 0; k < 2u; k++) {
        layer_t *model[HAR_N];
        buildHar(model, false);
        inputLike_t in;
        rematWireTable_t *t = initTable(model, HAR_N, CROSS_ENTROPY, makeHarInput(&in));
        rematPlan_t *p = buildPlan(t, model, specs[k]);
        const rematProgram_t *eval = rematPlanProgram(p, REMAT_MODE_EVAL);
        TEST_ASSERT_EQUAL_size_t(13, eval->numSteps);
        for (uint16_t l = 0; l < 12; l++) {
            TEST_ASSERT_EQUAL_UINT8(REMAT_STEP_FORWARD, eval->steps[l].kind);
            TEST_ASSERT_EQUAL_UINT16(l, eval->steps[l].layer);
        }
        TEST_ASSERT_EQUAL_UINT8(REMAT_STEP_LOSS_FORWARD, eval->steps[12].kind);
        TEST_ASSERT_EQUAL_UINT16(12, eval->steps[12].layer);
        TEST_ASSERT_EQUAL_size_t(12, eval->numRanges);
        for (uint16_t j = 1; j <= 12; j++) {
            TEST_ASSERT_EQUAL_UINT16(j, eval->ranges[j - 1].wire);
            TEST_ASSERT_EQUAL_UINT16(j - 1, eval->ranges[j - 1].begin);
            TEST_ASSERT_EQUAL_UINT16(j, eval->ranges[j - 1].end);
            TEST_ASSERT_EQUAL_UINT16(j - 1, eval->endOrder[j - 1]);
        }
        rematPlanFree(p);
        rematWireTableFree(t);
        freeModel(model, HAR_N);
    }
}

/* ACT 0 is borrowed and has no range, so FORWARD(0) holds ACT 1 alone and
 * FORWARD(j) holds ACT j + ACT j+1. HAR: the conv/relu pairs of 16 x 128 and
 * 32 x 64 floats, 8192 + 8192 B. One layer: ACT 1 alone, [1, 3] floats. */
void testEvalPeakIsTheLargestAdjacentActPair(void) {
    layer_t *har[HAR_N];
    buildHar(har, false);
    inputLike_t in;
    rematWireTable_t *t = initTable(har, HAR_N, CROSS_ENTROPY, makeHarInput(&in));
    rematPlan_t *p = buildPlan(t, har, NULL);
    TEST_ASSERT_EQUAL_size_t(16384, rematPlanProgram(p, REMAT_MODE_EVAL)->peakLiveBytes);
    rematPlanFree(p);
    rematWireTableFree(t);
    freeModel(har, HAR_N);
    layer_t *one[1] = {makeLinear(2, 3, false)};
    t = initTable(one, 1, CROSS_ENTROPY, makeInput(&in, (size_t[]){1, 2}, 2, &g_floatQ));
    p = buildPlan(t, one, NULL);
    TEST_ASSERT_EQUAL_size_t(2, rematPlanProgram(p, REMAT_MODE_EVAL)->numSteps);
    TEST_ASSERT_EQUAL_size_t(12, rematPlanProgram(p, REMAT_MODE_EVAL)->peakLiveBytes);
    rematPlanFree(p);
    rematWireTableFree(t);
    freeModel(one, 1);
}

#ifdef ODT_MEM_PROFILE
void testPlanBuildReservesOneBlockAndFreeReturnsIt(void) {
    layer_t *model[HAR_N];
    buildHar(model, false);
    inputLike_t in;
    rematWireTable_t *t = initTable(model, HAR_N, CROSS_ENTROPY, makeHarInput(&in));
    size_t before = memProfileCurrentBytes();
    rematPlan_t *p = buildPlan(t, model, NULL);
    TEST_ASSERT_EQUAL_size_t(before + p->blockBytes, memProfileCurrentBytes());
    rematPlanFree(p);
    TEST_ASSERT_EQUAL_size_t(before, memProfileCurrentBytes());
    rematWireTableFree(t);
    freeModel(model, HAR_N);
}
#endif

/* NULL, or a zero-initialised spec, means STORE_ALL: the scheduler library's
 * default, not the training call's (calculateGradsDefaultPlanSpec). */
void testNullOrZeroedSpecMeansStoreAll(void) {
    layer_t *model[HAR_N];
    buildHar(model, false);
    inputLike_t in;
    rematWireTable_t *t = initTable(model, HAR_N, CROSS_ENTROPY, makeHarInput(&in));
    rematPlan_t *fromNull = buildPlan(t, model, NULL);
    rematPlan_t *fromZero = buildPlan(t, model, &(rematPlanSpec_t){0});
    TEST_ASSERT_EQUAL_INT(REMAT_PLAN_STORE_ALL, fromNull->policy);
    TEST_ASSERT_EQUAL_INT(REMAT_PLAN_STORE_ALL, fromZero->policy);
    TEST_ASSERT_EQUAL_MEMORY(fromNull->train.ranges, fromZero->train.ranges,
                             fromNull->train.numRanges * sizeof(rematRange_t));
    rematPlanFree(fromNull);
    rematPlanFree(fromZero);
    rematWireTableFree(t);
    freeModel(model, HAR_N);
}

/* An uninitialised stack spec must not become a garbage plan. */
void testPlanBuildExitsOnAnUnknownPolicy(void) {
    layer_t *model[1] = {makeRelu(&g_floatQ)};
    inputLike_t in;
    rematWireTable_t *t = initTable(model, 1, MSE, makeInput(&in, (size_t[]){1, 4}, 2, &g_floatQ));
    rematPlan_t *p = NULL;
    ASSERT_EXITS_WITH_OUTPUT(
        1, "rematPlanBuild: unknown policy 7",
        (void)rematPlanBuild(&p, t, model, &(rematPlanSpec_t){.policy = (rematPlanPolicy_t)7}));
    rematWireTableFree(t);
    freeModel(model, 1);
}

/* n = 1 under CE has LOSS_BACKWARD but no BACKWARD (top = -1 < deepest);
 * the seed lives [LOSS_BACKWARD, LOSS_BACKWARD]. */
void testSingleLayerUnderCrossEntropyHasASeedButNoBackwardStep(void) {
    layer_t *model[1] = {makeLinear(2, 3, false)};
    inputLike_t in;
    rematWireTable_t *t =
        initTable(model, 1, CROSS_ENTROPY, makeInput(&in, (size_t[]){1, 2}, 2, &g_floatQ));
    rematPlan_t *p = buildPlan(t, model, NULL);
    TEST_ASSERT_EQUAL_size_t(3, p->train.numSteps);
    TEST_ASSERT_EQUAL_UINT8(REMAT_STEP_LOSS_BACKWARD, p->train.steps[2].kind);
    TEST_ASSERT_EQUAL_size_t(2, p->train.numRanges);
    TEST_ASSERT_EQUAL_UINT16(2, p->train.ranges[1].wire);
    TEST_ASSERT_EQUAL_UINT16(2, p->train.ranges[1].begin);
    TEST_ASSERT_EQUAL_UINT16(2, p->train.ranges[1].end);
    rematPlanFree(p);
    rematWireTableFree(t);
    freeModel(model, 1);
}

/* All frozen: no LOSS_BACKWARD, no GRAD wire (hasBackward false). */
void testAllFrozenPlanHasNoBackwardPhase(void) {
    layer_t *model[2] = {makeLinear(2, 4, true), makeRelu(&g_floatQ)};
    inputLike_t in;
    rematWireTable_t *t = initTable(model, 2, MSE, makeInput(&in, (size_t[]){1, 2}, 2, &g_floatQ));
    rematPlan_t *p = buildPlan(t, model, NULL);
    TEST_ASSERT_EQUAL_size_t(3, p->train.numSteps);
    TEST_ASSERT_EQUAL_UINT8(REMAT_STEP_LOSS_FORWARD, p->train.steps[2].kind);
    TEST_ASSERT_EQUAL_size_t(2, p->train.numRanges);
    rematPlanFree(p);
    rematWireTableFree(t);
    freeModel(model, 2);
}

void testPlanFreeIsNullSafe(void) {
    ASSERT_EXITS_WITH(0, rematPlanFree(NULL));
}

/* ---- LIVENESS and the read-set ---- */

void testLivenessHarRangesEndAtTheLastReader(void) {
    layer_t *model[HAR_N];
    buildHar(model, false);
    inputLike_t in;
    rematWireTable_t *t = initTable(model, HAR_N, CROSS_ENTROPY, makeHarInput(&in));
    rematPlan_t *p = buildPlan(t, model, &g_liveness);
    TEST_ASSERT_EQUAL_INT(REMAT_PLAN_LIVENESS, p->policy);
    /* B(l) = 14 + (10 - l). Pool / Flatten inputs and the CE logits die at
     * their forward; conv (trainable), ReLU and Linear inputs live to their
     * BACKWARD; ACT 12 to LOSS_BACKWARD. */
    const uint16_t end[13] = {0, 23, 2, 21, 20, 5, 18, 17, 8, 9, 14, 11, 13};
    for (uint16_t j = 1; j <= 12; j++) {
        TEST_ASSERT_EQUAL_UINT16_MESSAGE(end[j], p->train.ranges[j - 1].end, "ACT end");
    }
    rematPlanFree(p);
    rematWireTableFree(t);
    freeModel(model, HAR_N);
}

/* Scan-model pins. */
void testLivenessPeakHarIs49152(void) {
    layer_t *model[HAR_N];
    buildHar(model, false);
    inputLike_t in;
    TEST_ASSERT_EQUAL_size_t(49152,
                             peakOf(model, HAR_N, CROSS_ENTROPY, makeHarInput(&in), &g_liveness));
    freeModel(model, HAR_N);
}

void testLivenessPeakMnistCnnIs112896(void) {
    layer_t *model[MNIST_N];
    buildMnistCnn(model);
    inputLike_t in;
    tensor_t *x = makeInput(&in, (size_t[]){1, 1, 784}, 3, &g_floatQ);
    TEST_ASSERT_EQUAL_size_t(112896, peakOf(model, MNIST_N, CROSS_ENTROPY, x, &g_liveness));
    freeModel(model, MNIST_N);
}

void testLivenessPeakFinetuneStage2Is16384(void) {
    layer_t *model[HAR_N];
    buildHar(model, true);
    inputLike_t in;
    TEST_ASSERT_EQUAL_size_t(16384,
                             peakOf(model, HAR_N, CROSS_ENTROPY, makeHarInput(&in), &g_liveness));
    freeModel(model, HAR_N);
}

static size_t actPeakOf(layer_t **model, size_t n, lossFuncType_t lt, const tensor_t *x,
                        const rematPlanSpec_t *spec) {
    rematWireTable_t *t = initTable(model, n, lt, x);
    rematPlan_t *p = buildPlan(t, model, spec);
    size_t peak = rematProgramActPeakBytes(&p->train, t);
    rematPlanFree(p);
    rematWireTableFree(t);
    return peak;
}

/* The activation-only peak, by hand from HAR's TRAIN ranges: STORE_ALL keeps
 * every ACT (6 x 8192 + 2 x 4096 + 2 x 256 + 2 x 24); LIVENESS peaks at step
 * 8, ACT 1, 3, 4, 6, 7, 8 plus ACT 9 opening before ACT 8 closes. */
void testActPeakOfHarIs57904StoreAllAnd41216Liveness(void) {
    layer_t *model[HAR_N];
    buildHar(model, false);
    inputLike_t in;
    TEST_ASSERT_EQUAL_size_t(57904,
                             actPeakOf(model, HAR_N, CROSS_ENTROPY, makeHarInput(&in), NULL));
    TEST_ASSERT_EQUAL_size_t(
        41216, actPeakOf(model, HAR_N, CROSS_ENTROPY, makeHarInput(&in), &g_liveness));
    freeModel(model, HAR_N);
}

/* Finetune stage 2: under LIVENESS each frozen layer's input dies at its
 * forward, so the peak is ACT 1 + ACT 2 at step 1. */
void testActPeakOfFinetuneStage2Is57904StoreAllAnd16384Liveness(void) {
    layer_t *model[HAR_N];
    buildHar(model, true);
    inputLike_t in;
    TEST_ASSERT_EQUAL_size_t(57904,
                             actPeakOf(model, HAR_N, CROSS_ENTROPY, makeHarInput(&in), NULL));
    TEST_ASSERT_EQUAL_size_t(
        16384, actPeakOf(model, HAR_N, CROSS_ENTROPY, makeHarInput(&in), &g_liveness));
    freeModel(model, HAR_N);
}

/* mnist_cnn: STORE_ALL 4 x 25088 + 2 x 12544 + 2 x 64 + 2 x 40; LIVENESS
 * peaks at step 5 with ACT 1, 3, 4, 5, 6. */
void testActPeakOfMnistCnnIs125648StoreAllAnd100352Liveness(void) {
    layer_t *model[MNIST_N];
    buildMnistCnn(model);
    inputLike_t in;
    tensor_t *x = makeInput(&in, (size_t[]){1, 1, 784}, 3, &g_floatQ);
    TEST_ASSERT_EQUAL_size_t(125648, actPeakOf(model, MNIST_N, CROSS_ENTROPY, x, NULL));
    TEST_ASSERT_EQUAL_size_t(100352, actPeakOf(model, MNIST_N, CROSS_ENTROPY, x, &g_liveness));
    freeModel(model, MNIST_N);
}

/* n = 1 under CE gives top = -1, so no BACKWARD step exists to read
 * ACT 1 or the seed; both end at LOSS_BACKWARD (step 2). Same fixture as
 * testSingleLayerUnderCrossEntropyHasASeedButNoBackwardStep, built under
 * LIVENESS instead of STORE_ALL. */
void testLivenessSingleLayerUnderCrossEntropyEndsBothWiresAtLossBackward(void) {
    layer_t *model[1] = {makeLinear(2, 3, false)};
    inputLike_t in;
    rematWireTable_t *t =
        initTable(model, 1, CROSS_ENTROPY, makeInput(&in, (size_t[]){1, 2}, 2, &g_floatQ));
    rematPlan_t *p = buildPlan(t, model, &g_liveness);
    TEST_ASSERT_EQUAL_UINT16_MESSAGE(2, p->train.ranges[0].end, "ACT 1");
    TEST_ASSERT_EQUAL_UINT16_MESSAGE(2, p->train.ranges[1].end, "the seed");
    rematPlanFree(p);
    rematWireTableFree(t);
    freeModel(model, 1);
}

/* A frozen norm's input stays live through its BACKWARD, a frozen
 * GEMM's input dies at its forward. [Linear T, LayerNorm frozen, Linear frozen,
 * Linear T] under MSE: n = 4, top = 3, B(l) = 6 + (3 - l). */
void testFrozenNormStillNeedsItsInputWhileAFrozenGemmDoesNot(void) {
    layer_t *model[4] = {makeLinear(4, 4, false), makeLayerNorm(4, true), makeLinear(4, 4, true),
                         makeLinear(4, 2, false)};
    inputLike_t in;
    rematWireTable_t *t = initTable(model, 4, MSE, makeInput(&in, (size_t[]){1, 4}, 2, &g_floatQ));
    rematPlan_t *p = buildPlan(t, model, &g_liveness);
    TEST_ASSERT_EQUAL_UINT16_MESSAGE(8, p->train.ranges[0].end, "frozen LayerNorm input");
    TEST_ASSERT_EQUAL_UINT16_MESSAGE(2, p->train.ranges[1].end, "frozen Linear input");
    rematPlanFree(p);
    rematWireTableFree(t);
    freeModel(model, 4);
}

/* Under CE the logits die at FORWARD(n-1) (the positional skip means no
 * BACKWARD reads them); under MSE after Softmax they live to BACKWARD(n-1). */
void testCeLogitsDieAtForwardWhileMseSoftmaxInputIsRetained(void) {
    layer_t *model[2] = {makeLinear(2, 3, false), makeSoftmax()};
    inputLike_t in;
    tensor_t *x = makeInput(&in, (size_t[]){1, 2}, 2, &g_floatQ);
    rematWireTable_t *ce = initTable(model, 2, CROSS_ENTROPY, x);
    rematPlan_t *ceLive = buildPlan(ce, model, &g_liveness);
    TEST_ASSERT_EQUAL_UINT16_MESSAGE(1, ceLive->train.ranges[0].end, "CE logits");
    rematWireTable_t *mse = initTable(model, 2, MSE, x);
    rematPlan_t *mseLive = buildPlan(mse, model, &g_liveness);
    TEST_ASSERT_EQUAL_UINT16_MESSAGE(4, mseLive->train.ranges[0].end, "MSE Softmax input");
    rematPlanFree(ceLive);
    rematPlanFree(mseLive);
    rematWireTableFree(ce);
    rematWireTableFree(mse);
    freeModel(model, 2);
}

/* The read-set rule is deliberately not refined by propLoss; that rests on this
 * cross-seam property -- the deepest trainable layer (the only grads-only
 * backward) always reads its input. A characterization of PR0 code: no live
 * RED is possible; the mutation proves its teeth. */
void testTrainableParameterLayersReadTheirInputInBackward(void) {
    linearConfig_t linear[2] = {{.frozen = false}, {.frozen = true}};
    conv1dConfig_t conv[2] = {{.frozen = false}, {.frozen = true}};
    conv1dTransposedConfig_t convT[2] = {{.frozen = false}, {.frozen = true}};
    layerNormConfig_t layerNorm[2] = {{.frozen = false}, {.frozen = true}};
    groupNormConfig_t groupNorm[2] = {{.frozen = false}, {.frozen = true}};
    size_t trainableParamLayers = 0;
    for (int type = LINEAR; type <= GROUPNORM; type++) {
        for (size_t f = 0; f < 2; f++) {
            layerConfig_t cfg = {.linear = NULL};
            switch (type) {
            case LINEAR:
                cfg.linear = &linear[f];
                break;
            case CONV1D:
                cfg.conv1d = &conv[f];
                break;
            case CONV1D_TRANSPOSED:
                cfg.conv1dTransposed = &convT[f];
                break;
            case LAYERNORM:
                cfg.layerNorm = &layerNorm[f];
                break;
            case GROUPNORM:
                cfg.groupNorm = &groupNorm[f];
                break;
            default:
                break;
            }
            layer_t layer = {.type = (layerType_t)type, .config = &cfg};
            parameter_t *weight = NULL;
            parameter_t *bias = NULL;
            if (layerParameters(&layer, &weight, &bias) && !layerIsFrozen(&layer)) {
                trainableParamLayers++;
                TEST_ASSERT_TRUE_MESSAGE(layerBackwardReadsInput(&layer),
                                         "a trainable parameter layer skips its input");
            }
        }
    }
    TEST_ASSERT_EQUAL_size_t(5, trainableParamLayers); /* the loop covered all five */
}

void testEveryStepsOperandsAreCoLiveOnTheZoo(void) {
    layer_t *har[HAR_N];
    buildHar(har, false);
    inputLike_t in;
    assertCoLiveUnderBothPolicies(har, HAR_N, CROSS_ENTROPY, makeHarInput(&in));
    freeModel(har, HAR_N);
    layer_t *finetune[HAR_N];
    buildHar(finetune, true);
    assertCoLiveUnderBothPolicies(finetune, HAR_N, CROSS_ENTROPY, makeHarInput(&in));
    freeModel(finetune, HAR_N);
    layer_t *mnist[MNIST_N];
    buildMnistCnn(mnist);
    assertCoLiveUnderBothPolicies(mnist, MNIST_N, CROSS_ENTROPY,
                                  makeInput(&in, (size_t[]){1, 1, 784}, 3, &g_floatQ));
    freeModel(mnist, MNIST_N);
}

/* The no-overlap property over random plans: 50 random rank-2
 * chains of Linear / ReLU / Softmax / LayerNorm with random freezing, under
 * both losses and both policies. */
void testEveryStepsOperandsAreCoLiveOnRandomChains(void) {
    uint32_t state = 0x2545F491u;
    for (int trial = 0; trial < 50; trial++) {
        size_t n = 1u + nextRandom(&state) % 6u;
        layer_t *model[6];
        for (size_t i = 0; i < n; i++) {
            model[i] = randomRank2Layer(&state);
        }
        lossFuncType_t lt = (nextRandom(&state) % 2u == 0u) ? MSE : CROSS_ENTROPY;
        inputLike_t in;
        assertCoLiveUnderBothPolicies(model, n, lt, makeInput(&in, (size_t[]){1, 4}, 2, &g_floatQ));
        freeModel(model, n);
    }
}

/* ---- grammar validation ---- */

void testGrammarRejectsADuplicateForward(void) {
    ASSERT_GRAMMAR_EXIT(buildHarLivenessFixture, duplicateTheFirstForward,
                        "step #1 (kind 0, layer 0) violates grammar rule 1: one FORWARD per layer");
}

void testGrammarRejectsAnEarlyLossForward(void) {
    ASSERT_GRAMMAR_EXIT(buildHarLivenessFixture, swapLastForwardAndLossForward,
                        "step #11 (kind 1, layer 12) violates grammar rule 1: one LOSS_FORWARD");
}

/* Rule 1 bounds a FORWARD's layer below n: a FORWARD(n) would "write" ACT n+1,
 * which is not an ACT wire at all (id n+1 is the seed). */
void testGrammarRejectsAForwardBeyondTheLastLayer(void) {
    ASSERT_GRAMMAR_EXIT(
        buildHarLivenessFixture, forwardBeyondTheLastLayer,
        "step #12 (kind 0, layer 12) violates grammar rule 1: one FORWARD per layer");
}

/* A table that says no backward runs, validating HAR's program, which has one:
 * the LOSS_BACKWARD at step 13 is the first step rule 2 rejects. */
static void clearHasBackwardAndValidate(grammarFixture_t *f) {
    f->t->hasBackward = false;
    rematPlanValidateGrammar(&f->p->train, f->t, f->model, REMAT_MODE_TRAIN);
}

void testGrammarRejectsALossBackwardWithoutABackwardPhase(void) {
    grammarFixture_t f;
    buildHarLivenessFixture(&f);
    ASSERT_EXITS_WITH_OUTPUT(1, "step #13 (kind 2, layer 12) violates grammar rule 2",
                             clearHasBackwardAndValidate(&f));
    freeGrammarFixture(&f);
}

void testGrammarRejectsOutOfOrderBackwards(void) {
    ASSERT_GRAMMAR_EXIT(buildHarLivenessFixture, swapTheFirstTwoBackwards,
                        "step #14 (kind 3, layer 9) violates grammar rule 3");
}

void testGrammarRejectsAnUnknownStepKind(void) {
    ASSERT_GRAMMAR_EXIT(buildHarLivenessFixture, unknownStepKind,
                        "step #0 (kind 9, layer 0) violates grammar: unknown step kind");
}

void testGrammarRejectsAnIncompleteBackwardSet(void) {
    ASSERT_GRAMMAR_EXIT(buildHarLivenessFixture, dropTheLastBackward,
                        "the stream of 24 steps violates grammar rule 3: BACKWARD set incomplete");
}

void testGrammarRejectsAMissingLossBackward(void) {
    ASSERT_GRAMMAR_EXIT(buildHarLivenessFixture, dropFromLossBackwardOn,
                        "the stream of 13 steps violates grammar rule 2: LOSS_BACKWARD missing");
}

void testGrammarRejectsAMissingLossForward(void) {
    ASSERT_GRAMMAR_EXIT(buildAllFrozenFixture, dropTheLossForward,
                        "the stream of 2 steps violates grammar rule 1: no LOSS_FORWARD");
}

void testGrammarRejectsAReadOutsideItsRange(void) {
    ASSERT_GRAMMAR_EXIT(buildHarLivenessFixture, endAct1BeforeItsLastRead,
                        "step #23 (kind 3, layer 1) violates grammar rule 4: it reads wire 1");
}

void testGrammarRejectsAWriteOutsideItsRange(void) {
    ASSERT_GRAMMAR_EXIT(buildHarLivenessFixture, beginAct2AfterItsWrite,
                        "step #1 (kind 0, layer 1) violates grammar rule 4: it writes wire 2");
}

void testGrammarRejectsAForwardReadingOutsideItsRange(void) {
    ASSERT_GRAMMAR_EXIT(buildHarLivenessFixture, endAct2BeforeItsOnlyForwardRead,
                        "step #2 (kind 0, layer 2) violates grammar rule 4: it reads wire 2");
}

void testGrammarRejectsALossForwardReadingOutsideItsRange(void) {
    ASSERT_GRAMMAR_EXIT(buildHarStoreAllFixture, endAct12BeforeLossForwardReadsIt,
                        "step #12 (kind 1, layer 12) violates grammar rule 4: it reads wire 12");
}

void testGrammarRejectsALossBackwardReadingActNOutsideItsRange(void) {
    ASSERT_GRAMMAR_EXIT(buildHarLivenessFixture, endAct12BeforeLossBackwardReadsIt,
                        "step #13 (kind 2, layer 12) violates grammar rule 4: it reads wire 12");
}

void testGrammarRejectsALossBackwardWritingTheSeedOutsideItsRange(void) {
    ASSERT_GRAMMAR_EXIT(buildHarLivenessFixture, beginSeedAfterLossBackwardWritesIt,
                        "step #13 (kind 2, layer 12) violates grammar rule 4: it writes wire 13");
}

void testGrammarRejectsABackwardReadingGradInOutsideItsRange(void) {
    ASSERT_GRAMMAR_EXIT(buildHarLivenessFixture, endGrad10BeforeBackwardReadsIt,
                        "step #15 (kind 3, layer 9) violates grammar rule 4: it reads wire 14");
}

/* HAR EVAL: FORWARD 0..11 are steps 0..11, LOSS_FORWARD is step 12. The
 * tamper puts a LOSS_BACKWARD at step 12, where TRAIN's rule 2 would accept
 * one only after a LOSS_FORWARD; in EVAL rule 2 rejects any backward step,
 * and the message names the EVAL rule. */
static void lossBackwardInEvalAndValidate(grammarFixture_t *f) {
    rematProgram_t *eval = &f->p->eval;
    eval->steps[12] = (rematStep_t){.kind = REMAT_STEP_LOSS_BACKWARD, .layer = 12};
    rematPlanValidateGrammar(eval, f->t, f->model, REMAT_MODE_EVAL);
}

void testGrammarRejectsABackwardStepInAnEvalProgram(void) {
    grammarFixture_t f;
    buildHarStoreAllFixture(&f);
    ASSERT_EXITS_WITH_OUTPUT(1,
                             "step #12 (kind 2, layer 12) violates grammar rule 2: no "
                             "LOSS_BACKWARD or BACKWARD in an EVAL program",
                             lossBackwardInEvalAndValidate(&f));
    freeGrammarFixture(&f);
}

/* A layer BACKWARD in EVAL would otherwise fall to rule 3 (no LOSS_BACKWARD
 * precedes it); the message pins that the EVAL rule rejects it first. */
static void layerBackwardInEvalAndValidate(grammarFixture_t *f) {
    rematProgram_t *eval = &f->p->eval;
    eval->steps[12] = (rematStep_t){.kind = REMAT_STEP_BACKWARD, .layer = 11};
    rematPlanValidateGrammar(eval, f->t, f->model, REMAT_MODE_EVAL);
}

void testGrammarRejectsALayerBackwardInAnEvalProgram(void) {
    grammarFixture_t f;
    buildHarStoreAllFixture(&f);
    ASSERT_EXITS_WITH_OUTPUT(1,
                             "step #12 (kind 3, layer 11) violates grammar rule 2: no "
                             "LOSS_BACKWARD or BACKWARD in an EVAL program",
                             layerBackwardInEvalAndValidate(&f));
    freeGrammarFixture(&f);
}

/* rematPlanBuild runs the grammar on what it generates: a table record that
 * claims GRAD 8 is GRAD 7 (wire 16) yields a range that opens one step after
 * BACKWARD(8) writes it. */
static void mislabelGrad8AndBuild(grammarFixture_t *f) {
    f->t->wires[16].index = 7;
    rematPlan_t *p = NULL;
    (void)rematPlanBuild(&p, f->t, f->model, &g_liveness);
}

void testPlanBuildRunsTheGrammarOnWhatItGenerates(void) {
    grammarFixture_t f;
    buildHarLivenessFixture(&f);
    ASSERT_EXITS_WITH_OUTPUT(
        1, "step #16 (kind 3, layer 8) violates grammar rule 4: it writes wire 16",
        mislabelGrad8AndBuild(&f));
    freeGrammarFixture(&f);
}

/* ---- rematPlanBuild takes the table's model (PR1b) ---- */

static void buildLivenessPlanOn(grammarFixture_t *f, layer_t **model) {
    rematPlan_t *p = NULL;
    (void)rematPlanBuild(&p, f->t, model, &g_liveness);
}

/* A frozen Linear reads no input in its backward, so under LIVENESS the
 * generator AND grammar rule 4 -- both reading the read-set off `model` --
 * would end ACT 10 at FORWARD(10) and accept the plan, while the table's
 * trainable BACKWARD(10) still reads ACT 10. */
void testPlanBuildExitsWhenTheModelFreezesALayerTheTableSawTrainable(void) {
    grammarFixture_t f;
    buildHarLivenessFixture(&f);
    layer_t *other[HAR_N];
    memcpy(other, f.model, sizeof other);
    other[10] = makeLinear(64, 6, true);
    ASSERT_EXITS_WITH_OUTPUT(
        1, "the model differs from the table's key at 'frozen[10]': table 0, model 1",
        buildLivenessPlanOn(&f, other));
    freeLinearLayer(other[10]);
    freeGrammarFixture(&f);
}

/* ReLU -> Softmax keeps the read-set (both read their input), so only an
 * explicit type compare notices the swap. */
void testPlanBuildExitsWhenTheModelSwapsALayerType(void) {
    grammarFixture_t f;
    buildHarLivenessFixture(&f);
    layer_t *other[HAR_N];
    memcpy(other, f.model, sizeof other);
    other[1] = makeSoftmax();
    ASSERT_EXITS_WITH_OUTPUT(1, "the model differs from the table's key at 'layerType[1]'",
                             buildLivenessPlanOn(&f, other));
    freeSoftmaxLayer(other[1]);
    freeGrammarFixture(&f);
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(testBackwardRangeMseRunsFromLastLayerToDeepest);
    RUN_TEST(testBackwardRangeCrossEntropySkipsTheLastLayerPositionally);
    RUN_TEST(testBackwardRangeTruncatesAtDeepestTrainable);
    RUN_TEST(testBackwardRangeAllFrozenReturnsModelSize);
    RUN_TEST(testBackwardRangeSingleLayerUnderCrossEntropyIsMinusOne);
    RUN_TEST(testBfpWireGroupingMatchesTheDriverRule);
    RUN_TEST(testBfpWireGroupingExitsNamingTheWireOnAnIndivisibleGroupSize);
    RUN_TEST(testHarTableNumbersWiresInProductionOrder);
    RUN_TEST(testHarTableRecordsBytesRanksAndKey);
    RUN_TEST(testSharedGroupedBfpTemplateGroupsPerWire);
#ifdef ODT_MEM_PROFILE
    RUN_TEST(testTableInitReservesOneBlockOfSlabBytesAndFreeReturnsIt);
#endif
    RUN_TEST(testTableFreeReleasesExactlyTheOneBlock);
    RUN_TEST(testTableFreeIsNullSafe);
    RUN_TEST(testSlabObjectsAlignedAndExponentsAtTheTail);
    RUN_TEST(testAccessorsReadTheTableAndHeadersAreLinked);
    RUN_TEST(testTableInitExitsOnAnEmptyModel);
    RUN_TEST(testTableInitExitsWhenWireIdsWouldReachRematNone);
    RUN_TEST(testTableInitExitsOnAnInputRankAboveTheRankField);
    RUN_TEST(testTableInitExitsOnAnUnsupportedWireDtype);
    RUN_TEST(testTableInitExitsOnAZeroByteWire);
    RUN_TEST(testTableInitAcceptsAPackedBorrowedInputAndSizesItExactly);
    RUN_TEST(testTableInitExitsOnAnUnknownInputQtype);
    RUN_TEST(testTableInitExitsOnAByteCountOverflowBeforeReservingTheTable);
    RUN_TEST(testTableInitExitsOnASlabSizeOverflow);
    RUN_TEST(testTableInitExitsOnATotalWireBytesOverflow);
    RUN_TEST(testBindWritesHarHeadersAndPointsAct0AtTheInput);
    RUN_TEST(testEvalBindLeavesTheGradHeadersAsTheyAre);
    RUN_TEST(testEvalBindResetsTheBindGenerationsAndTheObservedPeak);
    RUN_TEST(testEvalBindRunsTheFullKeyCheck);
    RUN_TEST(testBindCopiesForwardOrderAndGivesGradsIdentityOrder);
#ifdef ODT_MEM_PROFILE
    RUN_TEST(testBindAllocatesNothing);
#endif
    RUN_TEST(testBindRederivesSymQMaxBitsAndResetsScale);
    RUN_TEST(testBindAfterDeserializeModel);
    RUN_TEST(testBindRederivesRoundingMode);
    RUN_TEST(testBindRederivesFlattenBfpGroupingFromTheLiveInput);
    RUN_TEST(testBindCarriesSymQMaxBitsOntoTheFlattenWire);
    RUN_TEST(testBindFollowsTheLivePackedInputBytes);
    RUN_TEST(testBindRunsSampleBOnATableBuiltOnSampleA);
    RUN_TEST(testHarSlabHoldsTheBindScratch);
    RUN_TEST(testUnbindClearsTheBorrowedInputOnly);
#ifdef ODT_TEST_ASAN
    RUN_TEST(testAsanDeathCallbackExitsWithADistinctCode);
#endif
    RUN_TEST(testBindExitsOnAChangedModelSize);
    RUN_TEST(testBindExitsOnAChangedLossType);
    RUN_TEST(testBindExitsOnALayerTypeSwap);
    RUN_TEST(testBindExitsWhenFreezingMovesDeepest);
    RUN_TEST(testBindExitsWhenFreezingALayerAboveDeepest);
    RUN_TEST(testBindExitsOnAChangedInputRank);
    RUN_TEST(testBindExitsOnAChangedBatch);
    RUN_TEST(testBindExitsOnAChangedInputOrder);
    RUN_TEST(testBindExitsOnAChangedInputDtype);
    RUN_TEST(testBindExitsOnAChangedWireByteCount);
    RUN_TEST(testBindFloatToSymTemplateEditExitsBeforeSlabWrite);
    RUN_TEST(testBindGroupedBfpEditExitsBeforeSlabWrite);
    RUN_TEST(testWireBindSetsDataCountsBytesAndBumpsBindGen);
    RUN_TEST(testWireReleaseClearsDataAndKeepsThePeak);
    RUN_TEST(testTableBindResetsBindGenLiveBytesAndThePeak);
    RUN_TEST(testWireBindDerivesTheSeedFromTheLiveActHeader);
    RUN_TEST(testWireBindInheritedSymTakesConfigNotScale);
    RUN_TEST(testWireBindInheritedGradChecksTheLiveDtypeWhenByteNeutral);
    RUN_TEST(testWireBindInheritedGradChecksCapacityBeforeWrite);
    RUN_TEST(testWireBindInheritedGradChecksTheLiveDtype);
    RUN_TEST(testWireBindInheritedGradChecksTheLiveBytes);
    RUN_TEST(testWireBindInheritedGradChecksTheLiveRank);
    RUN_TEST(testWireBindInheritedGradChecksTheLiveRankWhenByteNeutral);
    RUN_TEST(testWireBindRefusesTheBorrowedInput);
    RUN_TEST(testWireReleaseRefusesTheBorrowedInput);
    RUN_TEST(testWireBindRefusesAnAlreadyBoundWire);
    RUN_TEST(testWireReleaseRefusesAnUnboundWire);
    RUN_TEST(testWireBindRefusesAWireIdOutOfRange);
    RUN_TEST(testWireReleaseRefusesAWireIdOutOfRange);
    RUN_TEST(testWireBindRefusesNullBytes);
    RUN_TEST(testWireReleaseExitsWhenLiveBytesWouldUnderflow);
#ifdef ODT_REMAT_VERIFY
    RUN_TEST(testWireBindPoisonsFreshBytesByDtype);
    RUN_TEST(testWireReleasePoisonsTheOldBytesByDtype);
#endif
    RUN_TEST(testStoreAllHarStepOrder);
    RUN_TEST(testStoreAllHarRanges);
    RUN_TEST(testEndOrderSortsRangeIdsByEndThenWire);
    RUN_TEST(testWalkOpensAndClosesEachRangeOnceAtItsEndpoints);
    RUN_TEST(testStoreAllPeakHarIs74288);
    RUN_TEST(testStoreAllPeakMnistCnnIs175824);
    RUN_TEST(testStoreAllPeakFinetuneStage2Is57928);
    RUN_TEST(testPlanBlockHoldsStepsRangesAndEndOrder);
    RUN_TEST(testPlanProgramOfTrainIsTheTrainProgram);
    RUN_TEST(testEvalHarStepsAndRangesUnderEveryPolicy);
    RUN_TEST(testEvalPeakIsTheLargestAdjacentActPair);
#ifdef ODT_MEM_PROFILE
    RUN_TEST(testPlanBuildReservesOneBlockAndFreeReturnsIt);
#endif
    RUN_TEST(testNullOrZeroedSpecMeansStoreAll);
    RUN_TEST(testPlanBuildExitsOnAnUnknownPolicy);
    RUN_TEST(testSingleLayerUnderCrossEntropyHasASeedButNoBackwardStep);
    RUN_TEST(testAllFrozenPlanHasNoBackwardPhase);
    RUN_TEST(testPlanFreeIsNullSafe);
    RUN_TEST(testLivenessHarRangesEndAtTheLastReader);
    RUN_TEST(testLivenessPeakHarIs49152);
    RUN_TEST(testLivenessPeakMnistCnnIs112896);
    RUN_TEST(testLivenessPeakFinetuneStage2Is16384);
    RUN_TEST(testActPeakOfHarIs57904StoreAllAnd41216Liveness);
    RUN_TEST(testActPeakOfFinetuneStage2Is57904StoreAllAnd16384Liveness);
    RUN_TEST(testActPeakOfMnistCnnIs125648StoreAllAnd100352Liveness);
    RUN_TEST(testLivenessSingleLayerUnderCrossEntropyEndsBothWiresAtLossBackward);
    RUN_TEST(testFrozenNormStillNeedsItsInputWhileAFrozenGemmDoesNot);
    RUN_TEST(testCeLogitsDieAtForwardWhileMseSoftmaxInputIsRetained);
    RUN_TEST(testTrainableParameterLayersReadTheirInputInBackward);
    RUN_TEST(testEveryStepsOperandsAreCoLiveOnTheZoo);
    RUN_TEST(testEveryStepsOperandsAreCoLiveOnRandomChains);
    RUN_TEST(testGrammarRejectsADuplicateForward);
    RUN_TEST(testGrammarRejectsAnEarlyLossForward);
    RUN_TEST(testGrammarRejectsAForwardBeyondTheLastLayer);
    RUN_TEST(testGrammarRejectsALossBackwardWithoutABackwardPhase);
    RUN_TEST(testGrammarRejectsOutOfOrderBackwards);
    RUN_TEST(testGrammarRejectsAnUnknownStepKind);
    RUN_TEST(testGrammarRejectsAnIncompleteBackwardSet);
    RUN_TEST(testGrammarRejectsAMissingLossBackward);
    RUN_TEST(testGrammarRejectsAMissingLossForward);
    RUN_TEST(testGrammarRejectsAReadOutsideItsRange);
    RUN_TEST(testGrammarRejectsAWriteOutsideItsRange);
    RUN_TEST(testGrammarRejectsAForwardReadingOutsideItsRange);
    RUN_TEST(testGrammarRejectsALossForwardReadingOutsideItsRange);
    RUN_TEST(testGrammarRejectsALossBackwardReadingActNOutsideItsRange);
    RUN_TEST(testGrammarRejectsALossBackwardWritingTheSeedOutsideItsRange);
    RUN_TEST(testGrammarRejectsABackwardReadingGradInOutsideItsRange);
    RUN_TEST(testGrammarRejectsABackwardStepInAnEvalProgram);
    RUN_TEST(testGrammarRejectsALayerBackwardInAnEvalProgram);
    RUN_TEST(testPlanBuildRunsTheGrammarOnWhatItGenerates);
    RUN_TEST(testPlanBuildExitsWhenTheModelFreezesALayerTheTableSawTrainable);
    RUN_TEST(testPlanBuildExitsWhenTheModelSwapsALayerType);
    return UNITY_END();
}
