#define SOURCE_FILE "UNIT_TEST_REMAT_PLAN"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "AdaptiveAvgPool1d.h"
#include "AsanDeath.h"
#include "BatchNorm1d.h"
#include "Common.h"
#include "Conv1d.h"
#include "Conv1dApi.h"
#include "Conv1dTransposed.h"
#include "DagFixtures.h"
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
#include "Model.h"
#include "OpTable.h"
#include "Pool1dApi.h"
#include "QuantLayerApi.h"
#include "Quantization.h"
#include "QuantizationApi.h"
#include "RNG.h"
#include "ReluApi.h"
#include "RematPlan.h"
#include "RematPlanPolicy.h"
#include "RematTestFixtures.h"
#include "Serialize.h"
#include "Softmax.h"
#include "SoftmaxApi.h"
#include "StorageApi.h"
#include "Tensor.h"
#include "TensorApi.h"
#include "TestScale.h"
#include "unity.h"

void setUp(void) {}
void tearDown(void) {}

static rematWireTable_t *initTable(layer_t **model, size_t n, lossFuncType_t lt,
                                   const tensor_t *input) {
    rematWireTable_t *t = NULL;
    TEST_ASSERT_TRUE(rematWireTableInit(&t, &MODEL_CHAIN(model, n), defaultLossConfig(lt), input));
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
    f->model[0] = makeLinear(2, 8, false, 0);
    f->model[1] = makeQuant(&f->tmplQ, &g_floatQ);
    f->x = makeInput(&f->in, (size_t[]){1, 2}, 2, &g_floatQ);
    f->t = initTable(f->model, 2, MSE, f->x);
    rematWireTableBind(f->t, &MODEL_CHAIN(f->model, 2), MSE, REMAT_MODE_TRAIN, f->x);
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
    TEST_ASSERT_TRUE(rematPlanBuild(&p, t, &MODEL_CHAIN(model, t->modelSize), spec));
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

/* inPlace: the plan's marks or NULL; a member is live inside its chain
 * root's range. */
static void assertLiveAt(const rematProgram_t *p, const uint8_t *inPlace, uint16_t wire,
                         size_t step, const char *what) {
    while (inPlace != NULL && inPlace[wire] != 0u) {
        wire--;
    }
    if (wire == 0) {
        return; /* ACT 0 is borrowed: always present, never ranged */
    }
    bool live = false; /* any range of the wire: a recomputed wire has several */
    for (size_t r = 0; r < p->numRanges && !live; r++) {
        live = p->ranges[r].wire == wire && p->ranges[r].begin <= step && step <= p->ranges[r].end;
    }
    TEST_ASSERT_TRUE_MESSAGE(live, what);
}

/* The operands of every step, derived here independently of the generator,
 * must all be live at that step: inclusive ranges are what keep
 * a FORWARD's input and output, GRAD l+1 and GRAD l, and ACT l and GRAD l at a
 * reading BACKWARD(l) out of each other's bytes. */
static void assertEveryStepsOperandsAreCoLive(const rematProgram_t *p, const uint8_t *inPlace,
                                              const rematWireTable_t *t, layer_t **model) {
    size_t n = t->modelSize;
    size_t deepest;
    ptrdiff_t top;
    positionalRange(model, n, t->lossType, &deepest, &top);
    for (size_t s = 0; s < p->numSteps; s++) {
        size_t l = p->steps[s].layer;
        switch (p->steps[s].kind) {
        case REMAT_STEP_FORWARD:
        case REMAT_STEP_REFORWARD:
            assertLiveAt(p, inPlace, rematActId(t, l), s, "FORWARD or REFORWARD input");
            assertLiveAt(p, inPlace, rematActId(t, l + 1), s, "FORWARD or REFORWARD output");
            break;
        case REMAT_STEP_LOSS_FORWARD:
            assertLiveAt(p, inPlace, rematActId(t, n), s, "LOSS_FORWARD reads ACT n");
            break;
        case REMAT_STEP_LOSS_BACKWARD:
            assertLiveAt(p, inPlace, rematActId(t, n), s, "LOSS_BACKWARD reads ACT n");
            assertLiveAt(p, inPlace, rematGradId(t, n), s, "LOSS_BACKWARD writes the seed");
            break;
        case REMAT_STEP_BACKWARD: {
            uint16_t gradIn = ((ptrdiff_t)l == top) ? rematGradId(t, n) : rematGradId(t, l + 1);
            assertLiveAt(p, inPlace, gradIn, s, "BACKWARD reads gradIn");
            if (layerBackwardReadsInput(model[l])) {
                assertLiveAt(p, inPlace, rematActId(t, l), s, "BACKWARD reads its input");
            }
            if (l > deepest) {
                assertLiveAt(p, inPlace, rematGradId(t, l), s, "BACKWARD writes its dx");
            }
            break;
        }
        default:
            TEST_FAIL_MESSAGE("a step of unknown kind in a built plan");
        }
    }
}

static void assertCoLiveUnderBothPolicies(layer_t **model, size_t n, lossFuncType_t lt,
                                          const tensor_t *x) {
    rematWireTable_t *t = initTable(model, n, lt, x);
    rematPlan_t *all = buildPlan(t, model, NULL);
    rematPlan_t *live = buildPlan(t, model, &g_liveness);
    assertEveryStepsOperandsAreCoLive(&all->train, NULL, t, model);
    assertEveryStepsOperandsAreCoLive(&live->train, NULL, t, model);
    TEST_ASSERT_TRUE(live->train.peakLiveBytes <= all->train.peakLiveBytes);
    rematPlan_t *shared = buildPlan(t, model, &g_livenessInPlace);
    assertEveryStepsOperandsAreCoLive(&shared->train, shared->inPlace, t, model);
    TEST_ASSERT_TRUE(shared->train.peakLiveBytes <= live->train.peakLiveBytes);
    rematPlanFree(shared);
    rematPlanFree(live);
    rematPlanFree(all);
    rematWireTableFree(t);
}

static layer_t *randomRank2Layer(uint32_t *state, size_t position) {
    switch (nextRandom(state) % 4u) {
    case 0:
        return makeLinear(4, 4, nextRandom(state) % 2u == 0u, position);
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
    f->model[0] = makeLinear(2, 4, true, 0);
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
    rematPlanValidateGrammar(&f->p->train, f->t, &MODEL_CHAIN(f->model, f->t->modelSize),
                             REMAT_MODE_TRAIN, NULL);
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

/* ---- the CrossEntropy head at table init ---- */

/* CrossEntropy's backward is the COMBINED Softmax + CE gradient p - y, so the
 * backward skips the model's last layer exactly once: that layer must be the
 * Softmax. Any other last layer would be skipped silently (its backward never
 * runs) -- the table init dies instead, naming the layer type. */
static void ceTableOn(layer_t **model, size_t n) {
    inputLike_t in;
    rematWireTable_t *t =
        initTable(model, n, CROSS_ENTROPY, makeInput(&in, (size_t[]){1, 2}, 2, &g_floatQ));
    rematWireTableFree(t);
    freeModel(model, n);
}
static void ceWithoutSoftmaxLinear(void) {
    layer_t *model[1] = {makeLinear(2, 3, false, 0)};
    ceTableOn(model, 1);
}
static void ceWithoutSoftmaxRelu(void) {
    layer_t *model[2] = {makeLinear(2, 3, false, 0), makeRelu(&g_floatQ)};
    ceTableOn(model, 2);
}
static void ceWithSoftmax(void) {
    layer_t *model[2] = {makeLinear(2, 3, false, 0), makeSoftmax()};
    ceTableOn(model, 2);
}

void testTableInitCrossEntropyRequiresALastSoftmax(void) {
    ASSERT_EXITS_WITH_OUTPUT(1, "CrossEntropy skips the last layer's backward",
                             ceWithoutSoftmaxRelu());
    ASSERT_EXITS_WITH(0, ceWithSoftmax());
    ASSERT_EXITS_WITH_OUTPUT(1, "rematWireTableInit: CrossEntropy skips the last layer's backward",
                             ceWithoutSoftmaxLinear());
}

/* ---- the model head under CrossEntropy ----
 * The head is the Softmax forward plus CE's combined gradient p - y, declared
 * by the Softmax's config: (forwardMath, propLossMath, outputQ, propLossQ).
 * With any BFP in it, exactly two tuples are heads:
 *   bfp:   BFP, BFP, BFP, BFP                  (integer Softmax, integer p - y)
 *   float: FLOAT32, FLOAT32, FLOAT32, any wire (float island, ONE conversion
 *          onto the loss-grad wire = the Softmax's propLossQ).
 * Everything else -- e.g. a BFP Softmax with a float CE -- fails fast at init,
 * and only when a backward runs (an eval-only model is not checked). */
static uint8_t g_headExp[2];
static bfpQConfig_t g_headQc[2];
static quantization_t g_headBfp[2];

static quantization_t *headBfpQ(size_t k) {
    initBfpQConfigInto(8, 8, HALF_AWAY, &g_headExp[k], &g_headQc[k]);
    g_headBfp[k] = (quantization_t){.type = BFP, .qConfig = &g_headQc[k]};
    return &g_headBfp[k];
}

static layer_t *makeHead(arithmeticType_t fwd, arithmeticType_t bwd, bool bfpOut, bool bfpDx) {
    layerQuant_t lq = {
        .forwardMath = {.type = fwd, .roundingMode = HALF_AWAY},
        .propLossMath = {.type = bwd, .roundingMode = HALF_AWAY},
        .outputQ = bfpOut ? headBfpQ(0) : &g_floatQ,
        .propLossQ = bfpDx ? headBfpQ(1) : &g_floatQ,
    };
    return softmaxLayerInit(&lq);
}

static void headTableOn(arithmeticType_t fwd, arithmeticType_t bwd, bool bfpOut, bool bfpDx,
                        bool frozen) {
    layer_t *model[2] = {makeLinear(2, 3, frozen, 0), makeHead(fwd, bwd, bfpOut, bfpDx)};
    ceTableOn(model, 2);
}

static void headBfp(void) {
    headTableOn(ARITH_BFP, ARITH_BFP, true, true, false);
}
static void headFloatIntoBfp(void) {
    headTableOn(ARITH_FLOAT32, ARITH_FLOAT32, false, true, false);
}
static void headFloat(void) {
    headTableOn(ARITH_FLOAT32, ARITH_FLOAT32, false, false, false);
}
static void headBfpSoftmaxFloatCe(void) { /* the numerics before the two heads were fixed */
    headTableOn(ARITH_BFP, ARITH_FLOAT32, true, true, false);
}
static void headFloatSoftmaxPackedP(void) { /* float compute, p packed to BFP */
    headTableOn(ARITH_FLOAT32, ARITH_FLOAT32, true, true, false);
}
static void headIntegerCeFloatP(void) {
    headTableOn(ARITH_BFP, ARITH_BFP, false, true, false);
}
static void headIntegerCeFloatDx(void) {
    headTableOn(ARITH_BFP, ARITH_BFP, true, false, false);
}
static void headFloatSoftmaxIntegerCe(void) {
    headTableOn(ARITH_FLOAT32, ARITH_BFP, true, true, false);
}
static void headMismatchButFrozen(void) {
    headTableOn(ARITH_BFP, ARITH_FLOAT32, true, true, true);
}

void testTableInitCrossEntropyHeadIsOneOfTwoDeclaredArms(void) {
    ASSERT_EXITS_WITH(0, headBfp());
    ASSERT_EXITS_WITH(0, headFloatIntoBfp());
    ASSERT_EXITS_WITH(0, headFloat());
    const char *msg = "is neither the bfp head";
    ASSERT_EXITS_WITH_OUTPUT(1, msg, headBfpSoftmaxFloatCe());
    ASSERT_EXITS_WITH_OUTPUT(1, msg, headFloatSoftmaxPackedP());
    ASSERT_EXITS_WITH_OUTPUT(1, msg, headIntegerCeFloatP());
    ASSERT_EXITS_WITH_OUTPUT(1, msg, headIntegerCeFloatDx());
    ASSERT_EXITS_WITH_OUTPUT(1, msg, headFloatSoftmaxIntegerCe());
    ASSERT_EXITS_WITH(0, headMismatchButFrozen());
}

/* With nothing to train no backward runs, so nothing is skipped: a CE model
 * without a last Softmax (eval only) is rejected neither at init nor at a
 * bind, which re-checks the head of the live model. */
static void ceWithoutSoftmaxAllFrozen(void) {
    layer_t *model[2] = {makeLinear(2, 3, true, 0), makeRelu(&g_floatQ)};
    inputLike_t in;
    tensor_t *x = makeInput(&in, (size_t[]){1, 2}, 2, &g_floatQ);
    rematWireTable_t *t = initTable(model, 2, CROSS_ENTROPY, x);
    rematWireTableBind(t, &MODEL_CHAIN(model, 2), CROSS_ENTROPY, REMAT_MODE_TRAIN, x);
    rematWireTableFree(t);
    freeModel(model, 2);
}

void testTableInitCrossEntropyChecksTheHeadOnlyWhenABackwardRuns(void) {
    ASSERT_EXITS_WITH(0, ceWithoutSoftmaxAllFrozen());
}

/* The bind checks the head of the live model as well: a head config edited
 * after the table init (here a BFP Softmax under a float CE) dies at the next
 * bind, before it can reach a training step. */
static void headEditedBeforeBind(void) {
    layer_t *model[2] = {makeLinear(2, 3, false, 0),
                         makeHead(ARITH_FLOAT32, ARITH_FLOAT32, false, false)};
    inputLike_t in;
    tensor_t *x = makeInput(&in, (size_t[]){1, 2}, 2, &g_floatQ);
    rematWireTable_t *t = initTable(model, 2, CROSS_ENTROPY, x);
    model[1]->config->softmax->forwardMath.type = ARITH_BFP;
    rematWireTableBind(t, &MODEL_CHAIN(model, 2), CROSS_ENTROPY, REMAT_MODE_TRAIN, x);
    rematWireTableFree(t);
    freeModel(model, 2);
}

void testTableBindChecksTheCrossEntropyHeadOfTheLiveModel(void) {
    ASSERT_EXITS_WITH_OUTPUT(1, "rematWireTableBind: the CrossEntropy head",
                             headEditedBeforeBind());
}

/* Under CrossEntropy with a BFP-declaring head the seed GRAD n IS the
 * Softmax's dx (the combined gradient replaces its backward), so it takes
 * the Softmax's declared dx wire -- propLossQ, type-derived -- not ACT n's:
 * the float head's FLOAT32 p feeds a BFP loss-grad wire. Under MSE the seed
 * still inherits ACT n. */
void testCrossEntropySeedTakesTheSoftmaxDxWire(void) {
    layer_t *model[2] = {makeLinear(2, 3, false, 0),
                         makeHead(ARITH_FLOAT32, ARITH_FLOAT32, false, true)};
    inputLike_t in;
    rematWireTable_t *t =
        initTable(model, 2, CROSS_ENTROPY, makeInput(&in, (size_t[]){1, 2}, 2, &g_floatQ));
    TEST_ASSERT_EQUAL_size_t(4, t->numWires);
    TEST_ASSERT_EQUAL_UINT8(FLOAT32, t->wires[2].dtype); /* ACT 2: p */
    TEST_ASSERT_EQUAL_UINT8(REMAT_WIRE_GRAD, t->wires[3].kind);
    TEST_ASSERT_EQUAL_UINT16(2, t->wires[3].index);
    TEST_ASSERT_EQUAL_UINT8(BFP, t->wires[3].dtype);
    TEST_ASSERT_EQUAL_UINT16(REMAT_NONE, t->wires[3].inheritFrom);
    TEST_ASSERT_EQUAL_size_t(3, t->wires[3].bytes);
    rematWireTableFree(t);

    rematWireTable_t *m = initTable(model, 2, MSE, makeInput(&in, (size_t[]){1, 2}, 2, &g_floatQ));
    TEST_ASSERT_EQUAL_UINT16(2, m->wires[3].inheritFrom);
    TEST_ASSERT_EQUAL_UINT8(FLOAT32, m->wires[3].dtype);
    rematWireTableFree(m);
    freeModel(model, 2);
}
/* The bfp head (integer Softmax, integer p - y) also seeds onto its propLossQ. */
void testCrossEntropySeedOfTheBfpHeadTakesItsPropLossQ(void) {
    layer_t *model[2] = {makeLinear(2, 3, false, 0), makeHead(ARITH_BFP, ARITH_BFP, true, true)};
    inputLike_t in;
    rematWireTable_t *t =
        initTable(model, 2, CROSS_ENTROPY, makeInput(&in, (size_t[]){1, 2}, 2, &g_floatQ));
    TEST_ASSERT_EQUAL_UINT8(BFP, t->wires[3].dtype);
    TEST_ASSERT_EQUAL_UINT16(REMAT_NONE, t->wires[3].inheritFrom);
    rematWireTableFree(t);
    freeModel(model, 2);
}

/* A head with no BFP keeps the rule it always had: the seed inherits ACT n
 * (p's own wire), and the Softmax's propLossQ stays unused -- a FLOAT32
 * Softmax with an 8-bit SYM_INT32 propLossQ seeds FLOAT32, a packed SYM
 * propLossQ does not reach the table at all. */
static void floatHeadWithUnusedDx(quantization_t *dx) {
    layerQuant_t lq = {
        .forwardMath = {.type = ARITH_FLOAT32, .roundingMode = HALF_AWAY},
        .propLossMath = {.type = ARITH_FLOAT32, .roundingMode = HALF_AWAY},
        .outputQ = &g_floatQ,
        .propLossQ = dx,
    };
    layer_t *model[2] = {makeLinear(2, 3, false, 0), softmaxLayerInit(&lq)};
    inputLike_t in;
    rematWireTable_t *t =
        initTable(model, 2, CROSS_ENTROPY, makeInput(&in, (size_t[]){1, 2}, 2, &g_floatQ));
    TEST_ASSERT_EQUAL_UINT8(REMAT_WIRE_GRAD, t->wires[3].kind);
    TEST_ASSERT_EQUAL_UINT16(2, t->wires[3].inheritFrom);
    TEST_ASSERT_EQUAL_UINT8(FLOAT32, t->wires[3].dtype);
    TEST_ASSERT_EQUAL_size_t(3 * sizeof(float), t->wires[3].bytes);
    rematWireTableFree(t);
    freeModel(model, 2);
}

static void floatHeadWithPackedSymDx(void) {
    quantization_t *sym4 = quantizationInitSym(4, HALF_AWAY);
    floatHeadWithUnusedDx(sym4);
    freeQuantization(sym4);
}

void testCrossEntropySeedOfANonBfpHeadInheritsActN(void) {
    quantization_t *sym8 = quantizationInitSymInt32WithBits(HALF_AWAY, 8);
    floatHeadWithUnusedDx(sym8);
    freeQuantization(sym8);
    ASSERT_EXITS_WITH(0, floatHeadWithPackedSymDx());
}

/* ---- the one BFP wire-grouping rule ---- */

/* The helper must reproduce the pre-remat driver's rule (its ACT and dx wire
 * allocators) and initBufferOutput's copy (InferenceApi.c) on its canonical
 * cases. */
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
    /* Under CE the seed is buffer 11's gradient; no GRAD 0 (the input needs none). */
    TEST_ASSERT_EQUAL_UINT16(13, t->gradIdOf[11]);
    TEST_ASSERT_EQUAL_UINT16(REMAT_NONE, t->gradIdOf[0]);

    rematWireTableFree(t);
    freeModel(model, HAR_N);
}

/* HAR under CE: everything below the Softmax runs; buffers 1..12 need a
 * gradient; the read-set is the per-type rule; the seed is buffer 11's
 * gradient (the Softmax's input). */
void testHarTableHoldsTheGradFlagsAndTheDxWires(void) {
    layer_t *model[HAR_N];
    buildHar(model, false);
    inputLike_t in;
    rematWireTable_t *t = initTable(model, HAR_N, CROSS_ENTROPY, makeHarInput(&in));
    const uint8_t reads[HAR_N] = {1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1};
    for (size_t j = 0; j < HAR_N; j++) {
        TEST_ASSERT_EQUAL_UINT8(j < 11 ? 1 : 0, t->runsBackward[j]);
        TEST_ASSERT_EQUAL_UINT8(reads[j], t->readsInput[j]);
        TEST_ASSERT_EQUAL_UINT16(j >= 1 && j <= 10 ? (uint16_t)(24 - j) : REMAT_NONE,
                                 t->dxWire[j * MODEL_MAX_ARITY]);
        TEST_ASSERT_EQUAL_UINT8(j <= 10 ? OP_MODE_WRITE : OP_MODE_READ,
                                t->dxMode[j * MODEL_MAX_ARITY]);
        TEST_ASSERT_EQUAL_UINT16(REMAT_NONE, t->dxWire[j * MODEL_MAX_ARITY + 1]);
        TEST_ASSERT_EQUAL_UINT8(OP_MODE_READ, t->dxMode[j * MODEL_MAX_ARITY + 1]);
    }
    for (size_t b = 0; b <= HAR_N; b++) {
        TEST_ASSERT_EQUAL_UINT8(b >= 1 ? 1 : 0, t->needsGrad[b]);
    }
    TEST_ASSERT_TRUE(t->hasBackward);
    TEST_ASSERT_EQUAL_UINT16(13, t->gradIdOf[11]); /* the seed is buffer 11's gradient */
    TEST_ASSERT_EQUAL_UINT16(13, t->gradIdOf[12]);
    rematWireTableFree(t);
    freeModel(model, HAR_N);
}

/* Frozen convs: only the Linear runs backward, grads-only (its input needs no
 * gradient, so its one slot writes nothing); a frozen conv reads no input. */
void testFrozenHarTableRunsOnlyTheLinearGradsOnly(void) {
    layer_t *model[HAR_N];
    buildHar(model, true);
    inputLike_t in;
    rematWireTable_t *t = initTable(model, HAR_N, CROSS_ENTROPY, makeHarInput(&in));
    TEST_ASSERT_EQUAL_size_t(14, t->numWires);
    TEST_ASSERT_TRUE(t->hasBackward);
    for (size_t j = 0; j < HAR_N; j++) {
        TEST_ASSERT_EQUAL_UINT8(j == 10 ? 1 : 0, t->runsBackward[j]);
        for (size_t i = 0; i < MODEL_MAX_ARITY; i++) {
            TEST_ASSERT_EQUAL_UINT16(REMAT_NONE, t->dxWire[j * MODEL_MAX_ARITY + i]);
            TEST_ASSERT_EQUAL_UINT8(j == 10 && i == 0 ? OP_MODE_WRITE : OP_MODE_READ,
                                    t->dxMode[j * MODEL_MAX_ARITY + i]);
        }
    }
    for (size_t b = 0; b <= HAR_N; b++) {
        TEST_ASSERT_EQUAL_UINT8(b >= 11 ? 1 : 0, t->needsGrad[b]);
    }
    TEST_ASSERT_EQUAL_UINT8(0, t->readsInput[0]);
    TEST_ASSERT_EQUAL_UINT8(0, t->readsInput[3]);
    TEST_ASSERT_EQUAL_UINT8(0, t->readsInput[6]);
    TEST_ASSERT_EQUAL_UINT16(13, t->gradIdOf[11]);
    TEST_ASSERT_EQUAL_UINT16(13, t->gradIdOf[12]);
    for (size_t b = 0; b <= 10; b++) {
        TEST_ASSERT_EQUAL_UINT16(REMAT_NONE, t->gradIdOf[b]);
    }
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

void testHarTableKeepsEveryActShape(void) {
    layer_t *model[HAR_N];
    buildHar(model, false);
    inputLike_t in;
    rematWireTable_t *t = initTable(model, HAR_N, CROSS_ENTROPY, makeHarInput(&in));
    const size_t dims0[3] = {1, 9, 128};
    for (size_t d = 0; d < 3; d++) {
        TEST_ASSERT_EQUAL_size_t(dims0[d], t->actShapes[0].dimensions[d]);
    }
    for (size_t j = 0; j <= HAR_N; j++) {
        const shape_t *s = &t->actShapes[j];
        TEST_ASSERT_EQUAL_size_t(t->wires[j].rank, s->numberOfDimensions);
        size_t elements = 1;
        for (size_t d = 0; d < s->numberOfDimensions; d++) {
            elements *= s->dimensions[d];
            TEST_ASSERT_EQUAL_size_t(d, s->orderOfDimensions[d]);
        }
        TEST_ASSERT_EQUAL_size_t(t->wires[j].bytes, elements * sizeof(float));
    }
    rematWireTableFree(t);
    freeModel(model, HAR_N);
}

/* Every bind re-derives the shapes (phase 1); a scribble does not survive it. */
void testBindRewritesTheActShapes(void) {
    layer_t *model[HAR_N];
    buildHar(model, false);
    inputLike_t in;
    tensor_t *x = makeHarInput(&in);
    rematWireTable_t *t = initTable(model, HAR_N, CROSS_ENTROPY, x);
    size_t kept = t->actShapes[10].dimensions[1];
    t->actShapes[10].dimensions[1] = 7u;
    rematWireTableBind(t, &MODEL_CHAIN(model, HAR_N), CROSS_ENTROPY, REMAT_MODE_TRAIN, x);
    TEST_ASSERT_EQUAL_size_t(kept, t->actShapes[10].dimensions[1]);
    TEST_ASSERT_EQUAL_size_t(kept, rematActHdr(t, 10)->shape->dimensions[1]);
    rematWireTableUnbind(t);
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
    const rngKey_t initKey1 = rngFold(rngKeyFromSeed(1u), RNG_TAG_INIT, 1u);
    layer_t *model[2] = {
        makeQuant(&bfpQ, &g_floatQ),
        linearLayerInit(
            &(linearInit_t){.inFeatures = 4, .outFeatures = 16, .weightInit.key = &initKey1}, &lq)};
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
    TEST_ASSERT_EQUAL_UINT16(13, rematGradId(t, 11));
    TEST_ASSERT_EQUAL_PTR(t->wires[13].hdr, rematGradHdr(t, 11));
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
    ASSERT_EXITS_WITH_OUTPUT(
        1, "modelSize == 0: nothing to schedule",
        (void)rematWireTableInit(&t, &MODEL_CHAIN(model, 0), defaultLossConfig(MSE), x));
    freeModel(model, 1);
}

/* A wiring the chain drivers would misread is refused before anything is reserved. */
void testTableInitValidatesTheWiring(void) {
    layer_t *model[2] = {makeRelu(&g_floatQ), makeRelu(&g_floatQ)};
    static const uint16_t start[3] = {0, 1, 2}, ids[2] = {0, 0};
    inputLike_t in;
    tensor_t *x = makeInput(&in, (size_t[]){1, 4}, 2, &g_floatQ);
    rematWireTable_t *t = NULL;
    ASSERT_EXITS_WITH_OUTPUT(
        1, "rematWireTableInit: wiring: buffer 1 has no consumer",
        (void)rematWireTableInit(&t, &(model_t){model, 2, start, ids}, defaultLossConfig(MSE), x));
    freeModel(model, 2);
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
    ASSERT_EXITS_WITH_OUTPUT(
        1, "65535 wires reach REMAT_NONE (0xFFFF)",
        (void)rematWireTableInit(&t, &MODEL_CHAIN(model, n), defaultLossConfig(MSE), x));
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
    ASSERT_EXITS_WITH_OUTPUT(
        1, "wire ACT 0 has rank 256, above the uint8_t rank field",
        (void)rematWireTableInit(&t, &MODEL_CHAIN(model, 1), defaultLossConfig(MSE), &x));
    freeModel(model, 1);
}

void testTableInitExitsOnAnUnsupportedWireDtype(void) {
    odtScale_t scale = odtTestScale(1.f);
    symQConfig_t symQc = {
        .scales = &scale, .numGroups = 1, .groupSize = 0, .roundingMode = HALF_AWAY, .qBits = 8};
    quantization_t symQ = {.type = SYM, .qConfig = &symQc};
    layer_t *model[1] = {makeRelu(&symQ)};
    inputLike_t in;
    tensor_t *x = makeInput(&in, (size_t[]){1, 4}, 2, &g_floatQ);
    rematWireTable_t *t = NULL;
    ASSERT_EXITS_WITH_OUTPUT(
        1, "wire ACT 1 has dtype 3; remat wires are FLOAT32, SYM_INT32 or BFP",
        (void)rematWireTableInit(&t, &MODEL_CHAIN(model, 1), defaultLossConfig(MSE), x));
    freeModel(model, 1);
}

/* #160: a zero-size block is implementation-defined (a zero-byte request may return NULL). */
void testTableInitExitsOnAZeroByteWire(void) {
    layer_t *model[1] = {makeRelu(&g_floatQ)};
    inputLike_t in;
    tensor_t *x = makeInput(&in, (size_t[]){1, 0}, 2, &g_floatQ);
    rematWireTable_t *t = NULL;
    ASSERT_EXITS_WITH_OUTPUT(
        1, "wire ACT 1 has zero bytes (#160)",
        (void)rematWireTableInit(&t, &MODEL_CHAIN(model, 1), defaultLossConfig(MSE), x));
    freeModel(model, 1);
}

/* The declared-width contract: declared widths are the caller's responsibility; setup rejects a
 * narrower destination once; a width changed after setup is trusted unchecked, a wrong one is the
 * caller's error. Forward half of the setup check: Relu copies its input's codes verbatim, so an
 * output declared narrower than its input exits at table init. No trainable layer: the forward
 * pair is checked even without a GRAD wire. */
void testTableInitExitsOnAReluOutputNarrowerThanItsInput(void) {
    symInt32QConfig_t wideQc;
    initSymInt32QConfigWithQMaxBits(HALF_AWAY, &wideQc, 16);
    quantization_t wideQ = {.type = SYM_INT32, .qConfig = &wideQc};
    symInt32QConfig_t narrowQc;
    initSymInt32QConfigWithQMaxBits(HALF_AWAY, &narrowQc, 8);
    quantization_t narrowQ = {.type = SYM_INT32, .qConfig = &narrowQc};
    layer_t *model[1] = {makeRelu(&narrowQ)};
    inputLike_t in;
    tensor_t *x = makeInput(&in, (size_t[]){1, 4}, 2, &wideQ);
    rematWireTable_t *t = NULL;
    ASSERT_EXITS_WITH_OUTPUT(
        1,
        "layer 0 (Relu) forward: its destination wire is declared SYM_INT32 "
        "qMaxBits 8, narrower than its source wire's 16",
        (void)rematWireTableInit(&t, &MODEL_CHAIN(model, 1), defaultLossConfig(MSE), x));
    freeModel(model, 1);
}

/* The declared-width contract (see the Relu test above), backward half of the setup check:
 * Flatten's dx inherits ACT 2 (@8) while the loss it copies verbatim is GRAD 3, Quant's
 * propLossQ (@16). Every other pair is equal-width or not SYM_INT32 on both sides. */
void testTableInitExitsOnAFlattenDxNarrowerThanTheLossItCopies(void) {
    symInt32QConfig_t wideQc;
    initSymInt32QConfigWithQMaxBits(HALF_AWAY, &wideQc, 16);
    quantization_t wideQ = {.type = SYM_INT32, .qConfig = &wideQc};
    symInt32QConfig_t narrowQc;
    initSymInt32QConfigWithQMaxBits(HALF_AWAY, &narrowQc, 8);
    quantization_t narrowQ = {.type = SYM_INT32, .qConfig = &narrowQc};
    layer_t *model[4] = {makeLinear(2, 4, false, 0), makeQuant(&narrowQ, &g_floatQ),
                         flattenLayerInit(), makeQuant(&g_floatQ, &wideQ)};
    inputLike_t in;
    tensor_t *x = makeInput(&in, (size_t[]){1, 2}, 2, &g_floatQ);
    rematWireTable_t *t = NULL;
    ASSERT_EXITS_WITH_OUTPUT(
        1,
        "layer 2 (Flatten) backward: its destination wire is declared "
        "SYM_INT32 qMaxBits 8, narrower than its source wire's 16",
        (void)rematWireTableInit(&t, &MODEL_CHAIN(model, 4), defaultLossConfig(MSE), x));
    freeModel(model, 4);
}

/* The declared-width contract (see the Relu test above), its "once" half, as wanted behaviour:
 * setup checks the widths once and a later bind does not. A valid Relu (input and wires @16)
 * inits; the programmer then narrows the Relu's wires to 8 bits, below its 16-bit input. The
 * next bind runs without an exit and the wire carries the narrowed width as declared -- the
 * framework trusts it, and the wrong size is the programmer's own error. */
void testAWidthNarrowedAfterSetupIsTrustedAsTheCallersResponsibility(void) {
    symInt32QConfig_t inputQc;
    initSymInt32QConfigWithQMaxBits(HALF_AWAY, &inputQc, 16);
    quantization_t inputQ = {.type = SYM_INT32, .qConfig = &inputQc};
    symInt32QConfig_t reluQc;
    initSymInt32QConfigWithQMaxBits(HALF_AWAY, &reluQc, 16);
    quantization_t reluQ = {.type = SYM_INT32, .qConfig = &reluQc};
    layer_t *model[1] = {makeRelu(&reluQ)};
    inputLike_t in;
    tensor_t *x = makeInput(&in, (size_t[]){1, 4}, 2, &inputQ);
    rematWireTable_t *t = initTable(model, 1, MSE, x);

    reluQc.qMaxBits = 8; /* the programmer's edit after setup */
    ASSERT_EXITS_WITH(0, rematWireTableBind(t, &MODEL_CHAIN(model, 1), MSE, REMAT_MODE_TRAIN, x));
    rematWireTableBind(t, &MODEL_CHAIN(model, 1), MSE, REMAT_MODE_TRAIN, x);
    const symInt32QConfig_t *act1Qc = rematActHdr(t, 1)->quantization->qConfig;
    TEST_ASSERT_EQUAL_UINT8(8, act1Qc->qMaxBits);

    rematWireTableFree(t);
    freeModel(model, 1);
}

/* The borrowed ACT 0 may be any dtype, packed ones included;
 * its bytes are exact for the checker's disjointness test. */
void testTableInitAcceptsAPackedBorrowedInputAndSizesItExactly(void) {
    odtScale_t scale = odtTestScale(1.f);
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
    ASSERT_EXITS_WITH_OUTPUT(
        1, "wire ACT 0 has unknown qtype 99",
        (void)rematWireTableInit(&t, &MODEL_CHAIN(model, 1), defaultLossConfig(MSE), x));
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
    (void)rematWireTableInit(&t, &MODEL_CHAIN(model, n), defaultLossConfig(MSE), x);
}

/* A borrowed [1, SIZE_MAX/4 + 2] FLOAT32 input: 4 * N wraps to 4. */
void testTableInitExitsOnAByteCountOverflowBeforeReservingTheTable(void) {
    layer_t *model[1] = {makeRelu(&g_floatQ)};
    inputLike_t in;
    tensor_t *x = makeInput(&in, (size_t[]){1, SIZE_MAX / 4u + 2u}, 2, &g_floatQ);
    ASSERT_EXITS_WITH_OUTPUT(1, "size overflow computing bytes of wire ACT 0",
                             initExpectingAnExit(model, 1, x));
    /* Only the transient init scratch is live at the exit: the gradient flags
     * (n = 1: 2 dx slots x (2 + 1) bytes, needsGrad 2, runsBackward 1), 2 wire
     * facts, and 2 shapes with dims and order of maxRank 2. The table block is
     * never reserved. */
#ifdef ODT_MEM_PROFILE
    size_t expectedReserved = 2u * (sizeof(uint16_t) + 1u) + 2u + 1u +
                              2u * sizeof(rematWireFact_t) +
                              2u * (sizeof(shape_t) + 2u * 2u * sizeof(size_t));
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
    rematWireTableBind(t, &MODEL_CHAIN(model, HAR_N), CROSS_ENTROPY, REMAT_MODE_TRAIN, x);

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
    rematWireTableBind(t, &MODEL_CHAIN(model, HAR_N), CROSS_ENTROPY, REMAT_MODE_TRAIN, x);
    rematGradHdr(t, 10)->shape->dimensions[1] = 7u;
    rematActHdr(t, 10)->shape->dimensions[1] = 7u;
    rematWireTableBind(t, &MODEL_CHAIN(model, HAR_N), CROSS_ENTROPY, REMAT_MODE_EVAL, x);
    TEST_ASSERT_EQUAL_size_t(7, rematGradHdr(t, 10)->shape->dimensions[1]);
    assertDims(rematActHdr(t, 10)->shape, (size_t[]){1, 64}, 2);
    rematWireTableBind(t, &MODEL_CHAIN(model, HAR_N), CROSS_ENTROPY, REMAT_MODE_TRAIN, x);
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
    rematWireTableBind(t, &MODEL_CHAIN(model, HAR_N), CROSS_ENTROPY, REMAT_MODE_TRAIN, x);
    t->wires[1].bindGen = 5u;
    t->observedPeakLiveBytes = 99u;
    rematWireTableBind(t, &MODEL_CHAIN(model, HAR_N), CROSS_ENTROPY, REMAT_MODE_EVAL, x);
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
        rematWireTableBind(t, &MODEL_CHAIN(model, HAR_N), CROSS_ENTROPY, REMAT_MODE_EVAL, x2));
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
    rematWireTableBind(t, &MODEL_CHAIN(model, 2), MSE, REMAT_MODE_TRAIN, x);

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
    rematWireTableBind(t, &MODEL_CHAIN(model, 1), MSE, REMAT_MODE_TRAIN, x);
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
    rematWireTableBind(t, &MODEL_CHAIN(model, 1), MSE, REMAT_MODE_TRAIN, x);
    symInt32QConfig_t *slabQc = rematActHdr(t, 1)->quantization->qConfig;
    TEST_ASSERT_EQUAL_UINT8(8, slabQc->qMaxBits);

    slabQc->scale = odtTestScale(0.25f); /* a producer's OUT_WRITE epilogue */
    symQc.qMaxBits = 16;                 /* key-preserving template edit between calls */
    rematWireTableBind(t, &MODEL_CHAIN(model, 1), MSE, REMAT_MODE_TRAIN, x);
    TEST_ASSERT_EQUAL_UINT8(16, slabQc->qMaxBits);
    TEST_ASSERT_EQUAL_FLOAT(1.0f, odtScaleToFloat(slabQc->scale));

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
    serializeModel(&MODEL_CHAIN(saved, 1), file);
    rewind(file);

    symInt32QConfig_t skeletonQc;
    initSymInt32QConfigWithQMaxBits(HALF_AWAY, &skeletonQc, 16);
    quantization_t skeletonQ = {.type = SYM_INT32, .qConfig = &skeletonQc};
    layer_t *skeleton[1] = {makeQuant(&skeletonQ, &g_floatQ)};
    inputLike_t in;
    tensor_t *x = makeInput(&in, (size_t[]){1, 4}, 2, &g_floatQ);
    rematWireTable_t *t = initTable(skeleton, 1, MSE, x);
    rematWireTableBind(t, &MODEL_CHAIN(skeleton, 1), MSE, REMAT_MODE_TRAIN, x);
    symInt32QConfig_t *slabQc = rematActHdr(t, 1)->quantization->qConfig;
    TEST_ASSERT_EQUAL_UINT8(16, slabQc->qMaxBits);

    deserializeModel(&MODEL_CHAIN(skeleton, 1), file);
    (void)fclose(file);
    rematWireTableBind(t, &MODEL_CHAIN(skeleton, 1), MSE, REMAT_MODE_TRAIN, x);
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
    rematWireTableBind(t, &MODEL_CHAIN(model, 1), MSE, REMAT_MODE_TRAIN, x);
    symQc.roundingMode = SR_HALF_AWAY;
    rematWireTableBind(t, &MODEL_CHAIN(model, 1), MSE, REMAT_MODE_TRAIN, x);
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
    rematWireTableBind(t, &MODEL_CHAIN(model, 1), MSE, REMAT_MODE_TRAIN, x);
    bfpQConfig_t *slabQc = rematActHdr(t, 1)->quantization->qConfig;
    TEST_ASSERT_EQUAL_size_t(4, slabQc->numGroups);
    TEST_ASSERT_EQUAL_size_t(4, slabQc->groupSize);

    slabQc->exponents[0] = 3; /* a producer's exponent write */
    initBfpQConfigGroupedInto(8, 8, HALF_AWAY, 2, 8, inputExponents, &inputQc);
    rematWireTableBind(t, &MODEL_CHAIN(model, 1), MSE, REMAT_MODE_TRAIN, x);
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
    rematWireTableBind(t, &MODEL_CHAIN(model, 1), MSE, REMAT_MODE_TRAIN, x);
    symInt32QConfig_t *slabQc = rematActHdr(t, 1)->quantization->qConfig;
    TEST_ASSERT_EQUAL_UINT8(12, slabQc->qMaxBits);
    inputQc.qMaxBits = 8;
    rematWireTableBind(t, &MODEL_CHAIN(model, 1), MSE, REMAT_MODE_TRAIN, x);
    TEST_ASSERT_EQUAL_UINT8(8, slabQc->qMaxBits);
    rematWireTableFree(t);
    freeModel(model, 1);
}

/* A packed borrowed input re-quantized between calls (8 -> 4 bits) is a
 * config edit, not a key change; the checker sizes ACT 0 from the live input. */
void testBindFollowsTheLivePackedInputBytes(void) {
    odtScale_t scale = odtTestScale(1.f);
    symQConfig_t inputQc = {
        .scales = &scale, .numGroups = 1, .groupSize = 0, .roundingMode = HALF_AWAY, .qBits = 8};
    quantization_t inputQ = {.type = SYM, .qConfig = &inputQc};
    layer_t *model[1] = {makeRelu(&g_floatQ)};
    inputLike_t in;
    tensor_t *x = makeInput(&in, (size_t[]){1, 8}, 2, &inputQ);
    rematWireTable_t *t = initTable(model, 1, MSE, x);
    TEST_ASSERT_EQUAL_size_t(8, rematWireBytes(t, 0));
    inputQc.qBits = 4;
    rematWireTableBind(t, &MODEL_CHAIN(model, 1), MSE, REMAT_MODE_TRAIN, x);
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
    rematWireTableBind(t, &MODEL_CHAIN(model, 1), MSE, REMAT_MODE_TRAIN, a);
    rematWireTableBind(t, &MODEL_CHAIN(model, 1), MSE, REMAT_MODE_TRAIN, b);
    TEST_ASSERT_EQUAL_PTR(b, rematActHdr(t, 0));
    TEST_ASSERT_EQUAL_PTR((uint8_t *)bytesB, b->data);
    TEST_ASSERT_EQUAL_MEMORY(snapshot.dims, inB.dims, sizeof inB.dims);
    TEST_ASSERT_EQUAL_MEMORY(snapshot.order, inB.order, sizeof inB.order);
    TEST_ASSERT_EQUAL_PTR(&g_floatQ, b->quantization);
    rematWireTableFree(t);
    freeModel(model, 1);
}

/* The bind's per-wire derivation scratch and the n + 1 ACT shapes live inside
 * the table block. LP64 layout arithmetic (not a scan-model pin): table struct
 * 200 (hasBackward and isChain pack beside the 4-byte lossType; the wiring's
 * two pointers and numEdges add 24) + 24 records x 40 = 1160; gradIdOf 26,
 * layerType/frozen 24, runsBackward 12, needsGrad 13, readsInput 12 = 1247,
 * rounded to 2 for dxWire 24 x 2 = 1296, dxMode 24 = 1320 (already a multiple
 * of 8); a chain places no wiring copy; + the input key 48 = 1368; the 13 ACT
 * shapes 13 x (24 + 2 x 3 x 8) = 936; the bind scratch 960; the headers 2680.
 * Total 5944 = 1368 + 936 + 960 + 2680. Every host preset is LP64. */
void testHarSlabHoldsTheBindScratch(void) {
    layer_t *model[HAR_N];
    buildHar(model, false);
    inputLike_t in;
    rematWireTable_t *t = initTable(model, HAR_N, CROSS_ENTROPY, makeHarInput(&in));
    TEST_ASSERT_EQUAL_size_t(5944, t->slabBytes);
    const uint8_t *scratch = (const uint8_t *)t->bindScratch;
    TEST_ASSERT_TRUE(scratch >= (const uint8_t *)(t->inputOrder + t->inputRank));
    TEST_ASSERT_TRUE(scratch + t->numWires * sizeof(rematWireFact_t) <=
                     (const uint8_t *)t->wires[1].hdr);
    ASSERT_ALIGNED(t->bindScratch, rematWireFact_t);
    rematWireTableFree(t);
    freeModel(model, HAR_N);
}

/* F3's CSR: n0 reads b0, n1 b1, n2 (b2, b1), n3 (b3, b1). The model's arrays belong to its
 * builder (or a MODEL_CHAIN temporary), so the table must own a copy. */
void testTableCopiesTheWiringOfADag(void) {
    dagFixture_t f;
    dagBuildF3(&f, false);
    dagTensor_t xs;
    rematWireTable_t *t;
    TEST_ASSERT_TRUE(rematWireTableInit(&t, &f.m, defaultLossConfig(MSE), dagInput(&xs, 1, 2)));
    TEST_ASSERT_EQUAL_UINT8(0, t->isChain);
    TEST_ASSERT_EQUAL_size_t(6, t->numEdges);
    TEST_ASSERT_TRUE(t->inputStart != f.m.inputStart);
    TEST_ASSERT_EQUAL_UINT16_ARRAY(((uint16_t[]){0, 1, 2, 4, 6}), t->inputStart, 5);
    TEST_ASSERT_EQUAL_UINT16_ARRAY(((uint16_t[]){0, 1, 2, 1, 3, 1}), t->inputIds, 6);
    rematWireTableFree(t);
    dagFree(&f);
}

/* A chain keeps no wiring, whether it comes as MODEL_CHAIN or as an explicit CSR that wires
 * every node to the buffer before it: same flags, same slab bytes. */
void testTableOfAChainKeepsNoWiring(void) {
    layer_t *model[HAR_N];
    buildHar(model, false);
    inputLike_t in;
    tensor_t *x = makeHarInput(&in);
    rematWireTable_t *chain = initTable(model, HAR_N, CROSS_ENTROPY, x);

    layer_t *nodes[HAR_N];
    uint16_t start[HAR_N + 1];
    uint16_t ids[HAR_N];
    graphBuilder_t g;
    graphBegin(&g, nodes, start, ids, HAR_N, HAR_N);
    uint16_t b = GRAPH_INPUT;
    for (size_t j = 0; j < HAR_N; j++) {
        b = graphNode(&g, model[j], b);
    }
    model_t built = graphFinish(&g);
    rematWireTable_t *csr = NULL;
    TEST_ASSERT_TRUE(rematWireTableInit(&csr, &built, defaultLossConfig(CROSS_ENTROPY), x));

    const rematWireTable_t *tables[2] = {chain, csr};
    for (size_t k = 0; k < 2u; k++) {
        TEST_ASSERT_EQUAL_UINT8(1, tables[k]->isChain);
        TEST_ASSERT_NULL(tables[k]->inputStart);
        TEST_ASSERT_NULL(tables[k]->inputIds);
        TEST_ASSERT_EQUAL_size_t(0, tables[k]->numEdges);
    }
    TEST_ASSERT_EQUAL_size_t(chain->slabBytes, csr->slabBytes);
    rematWireTableFree(csr);
    rematWireTableFree(chain);
    freeModel(model, HAR_N);
}

/* F3 under MSE, in walk order: the seed (GRAD 4) = 5; B3 opens GRAD 3 = 6 and GRAD 1 = 7; B2
 * opens GRAD 2 = 8 and accumulates into 7 (ADD); B1, a Linear and b1's third contributor, writes
 * PART 1 = 9. GRAD 1's WRITE contributor is the ADD n3, not node 1, so its template is n3's. */
void testF3TableNumbersGradsThenThePartInWalkOrder(void) {
    dagFixture_t f;
    dagBuildF3(&f, false);
    dagTensor_t xs;
    tensor_t *x = dagInput(&xs, 1, 2);
    uint8_t ng[5];
    uint8_t rb[4];
    modelGradFlags(&f.m, MSE, ng, rb);
    rematWireTable_t *t;
    TEST_ASSERT_TRUE(rematWireTableInit(&t, &f.m, defaultLossConfig(MSE), x));
    TEST_ASSERT_EQUAL_size_t(10u, t->numWires);
    const uint8_t kind[] = {REMAT_WIRE_GRAD, REMAT_WIRE_GRAD, REMAT_WIRE_GRAD, REMAT_WIRE_GRAD,
                            REMAT_WIRE_PART};
    const uint16_t index[] = {4, 3, 1, 2, 1};
    for (uint16_t w = 5; w < 10; w++) {
        TEST_ASSERT_EQUAL_UINT8(kind[w - 5], t->wires[w].kind);
        TEST_ASSERT_EQUAL_UINT16(index[w - 5], t->wires[w].index);
    }
    TEST_ASSERT_EQUAL_UINT16(4, t->wires[5].inheritFrom);
    for (uint16_t w = 6; w < 10; w++) {
        TEST_ASSERT_EQUAL_UINT16(REMAT_NONE, t->wires[w].inheritFrom);
    }
    TEST_ASSERT_EQUAL_UINT16_ARRAY(((uint16_t[]){REMAT_NONE, 7, 8, 6, 5}), t->gradIdOf, 5);
    const uint16_t dxW[] = {REMAT_NONE, REMAT_NONE, 9, REMAT_NONE, 8, 7, 6, 7};
    TEST_ASSERT_EQUAL_UINT16_ARRAY(dxW, t->dxWire, 8);
    TEST_ASSERT_EQUAL_UINT8(OP_MODE_WRITE, t->dxMode[2]);
    TEST_ASSERT_EQUAL_UINT8(OP_MODE_WRITE, t->dxMode[4]);
    TEST_ASSERT_EQUAL_UINT8(OP_MODE_ACC, t->dxMode[5]);
    TEST_ASSERT_EQUAL_UINT8(OP_MODE_WRITE, t->dxMode[6]);
    TEST_ASSERT_EQUAL_UINT8(OP_MODE_WRITE, t->dxMode[7]);
    uint16_t scratchDxWire[4 * MODEL_MAX_ARITY];
    uint8_t scratchDxMode[4 * MODEL_MAX_ARITY];
    TEST_ASSERT_EQUAL_size_t(
        5u, rematDeriveGradWires(&f.m, MSE, ng, rb, NULL, scratchDxWire, scratchDxMode));

    rematWireTableBind(t, &f.m, MSE, REMAT_MODE_TRAIN, x);
    TEST_ASSERT_TRUE(backwardWireQ(f.layers[3]) != backwardWireQ(f.layers[1]));
    TEST_ASSERT_TRUE(t->bindScratch[7].tmpl == backwardWireQ(f.layers[3]));
    TEST_ASSERT_TRUE(t->bindScratch[9].tmpl == backwardWireQ(f.layers[1]));
    rematWireTableUnbind(t);
    rematWireTableFree(t);
    dagFree(&f);
}

/* Y = ADD(b1, b1): one BACKWARD opens GRAD 1 through slot 0 and accumulates into it through
 * slot 1. ACT 0..2, the seed 3, GRAD 1 = 4. */
void testAddOfTheSameBufferWritesThenAccumulates(void) {
    dagFixture_t f;
    dagBuildY(&f);
    dagTensor_t xs;
    rematWireTable_t *t;
    TEST_ASSERT_TRUE(rematWireTableInit(&t, &f.m, defaultLossConfig(MSE), dagInput(&xs, 1, 2)));
    TEST_ASSERT_EQUAL_size_t(5u, t->numWires);
    TEST_ASSERT_EQUAL_UINT8(REMAT_WIRE_GRAD, t->wires[4].kind);
    TEST_ASSERT_EQUAL_UINT16(1, t->wires[4].index);
    TEST_ASSERT_EQUAL_UINT16(REMAT_NONE, t->dxWire[0]);
    TEST_ASSERT_EQUAL_UINT16_ARRAY(((uint16_t[]){4, 4}), &t->dxWire[2], 2);
    TEST_ASSERT_EQUAL_UINT8(OP_MODE_WRITE, t->dxMode[2]);
    TEST_ASSERT_EQUAL_UINT8(OP_MODE_ACC, t->dxMode[3]);
    rematWireTableFree(t);
    dagFree(&f);
}

/* K = ADD(b1, b0): the input needs no gradient, so the merge writes no dx for its skip slot.
 * ACT 0..2, the seed 3, GRAD 1 = 4. */
void testSkipFromTheInputGetsNoDx(void) {
    dagFixture_t f;
    dagBuildK(&f);
    dagTensor_t xs;
    rematWireTable_t *t;
    TEST_ASSERT_TRUE(rematWireTableInit(&t, &f.m, defaultLossConfig(MSE), dagInput(&xs, 1, 2)));
    TEST_ASSERT_EQUAL_size_t(5u, t->numWires);
    TEST_ASSERT_EQUAL_UINT16_ARRAY(((uint16_t[]){4, REMAT_NONE}), &t->dxWire[2], 2);
    TEST_ASSERT_EQUAL_UINT8(OP_MODE_WRITE, t->dxMode[2]);
    rematWireTableFree(t);
    dagFree(&f);
}

/* n0 = Linear(b0); n1 = Flatten(b0); n2 = ADD(b1, b2). A Flatten passes its input's config
 * through, and its input is the wired b0, not the buffer before it. */
void testFlattenActTakesTheTemplateOfItsWiredInput(void) {
    dagFixture_t f;
    dagBegin(&f);
    uint16_t b1 = graphNode(&f.g, dagLinear(0, 1.0f, 1.0f, false), GRAPH_INPUT);
    uint16_t b2 = graphNode(&f.g, flattenLayerInit(), GRAPH_INPUT);
    (void)graphMerge(&f.g, addLayerInit(NULL), b1, b2);
    f.m = graphFinish(&f.g);
    dagTensor_t xs;
    tensor_t *x = dagInput(&xs, 1, 2);
    rematWireTable_t *t;
    TEST_ASSERT_TRUE(rematWireTableInit(&t, &f.m, defaultLossConfig(MSE), x));
    rematWireTableBind(t, &f.m, MSE, REMAT_MODE_TRAIN, x);
    TEST_ASSERT_TRUE(t->bindScratch[1].tmpl != x->quantization);
    TEST_ASSERT_TRUE(t->bindScratch[2].tmpl == x->quantization);
    rematWireTableUnbind(t);
    rematWireTableFree(t);
    freeLinearLayer(f.layers[0]);
    freeFlattenLayer(f.layers[1]);
    freeAddLayer(f.layers[2]);
}

/* n0 = Linear(b0); n1 = Flatten(b1); n2 = ADD(b2, b1). The merge opens GRAD 2 = 5 and GRAD 1 = 6,
 * so the Flatten is b1's later, non-accumulating contributor: it writes PART 1 = 7, which passes
 * its dx through and so inherits ACT 1's template like a GRAD it would open. */
void testAPartAFlattenWritesInheritsItsInputsTemplate(void) {
    dagFixture_t f;
    dagBegin(&f);
    uint16_t b1 = graphNode(&f.g, dagLinear(0, 1.0f, 1.0f, false), GRAPH_INPUT);
    uint16_t b2 = graphNode(&f.g, flattenLayerInit(), b1);
    (void)graphMerge(&f.g, addLayerInit(NULL), b2, b1);
    f.m = graphFinish(&f.g);
    dagTensor_t xs;
    tensor_t *x = dagInput(&xs, 1, 2);
    rematWireTable_t *t;
    TEST_ASSERT_TRUE(rematWireTableInit(&t, &f.m, defaultLossConfig(MSE), x));
    TEST_ASSERT_EQUAL_size_t(8u, t->numWires);
    TEST_ASSERT_EQUAL_UINT8(REMAT_WIRE_GRAD, t->wires[6].kind);
    TEST_ASSERT_EQUAL_UINT16(1, t->wires[6].index);
    TEST_ASSERT_EQUAL_UINT8(REMAT_WIRE_PART, t->wires[7].kind);
    TEST_ASSERT_EQUAL_UINT16(1, t->wires[7].index);
    TEST_ASSERT_EQUAL_UINT16(1, t->wires[7].inheritFrom);
    rematWireTableBind(t, &f.m, MSE, REMAT_MODE_TRAIN, x);
    TEST_ASSERT_TRUE(t->bindScratch[7].tmpl == t->bindScratch[1].tmpl);
    rematWireTableUnbind(t);
    rematWireTableFree(t);
    freeLinearLayer(f.layers[0]);
    freeFlattenLayer(f.layers[1]);
    freeAddLayer(f.layers[2]);
}

/* A 2 -> 2 Linear with drawn weights (only table init reads it) whose dx wire is propLossQ,
 * borrowed: it must outlive the layer. */
static layer_t *dagLinearWithPropLossQ(size_t position, quantization_t *propLossQ) {
    layerQuant_t lq;
    layerQuantInitUniform(&lq, dagFloatQ());
    lq.propLossQ = propLossQ;
    const rngKey_t key = rngFold(rngKeyFromSeed(1u), RNG_TAG_INIT, (uint32_t)position);
    return linearLayerInit(&(linearInit_t){.inFeatures = 2,
                                           .outFeatures = 2,
                                           .trainable = TRAINABLE_DEFAULT,
                                           .weightInit.key = &key},
                           &lq);
}

/* x -> Quantization (SYM_INT32 12 bit) = b1; ADD(b1, b0): nothing trains, so the refusal rests on
 * the forward wire alone. */
void testTableInitRefusesASymInt32WireIntoAnAdd(void) {
    symInt32QConfig_t symQc;
    initSymInt32QConfigWithQMaxBits(HALF_AWAY, &symQc, 12);
    quantization_t symQ = {.type = SYM_INT32, .qConfig = &symQc};
    dagFixture_t f;
    dagBegin(&f);
    uint16_t b1 = graphNode(&f.g, makeQuant(&symQ, &g_floatQ), GRAPH_INPUT);
    (void)graphMerge(&f.g, addLayerInit(NULL), b1, GRAPH_INPUT);
    f.m = graphFinish(&f.g);
    dagTensor_t xs;
    tensor_t *x = dagInput(&xs, 1, 2);
    rematWireTable_t *t = NULL;
    ASSERT_EXITS_WITH_OUTPUT(1, "non-FLOAT32 merge or accumulate: wire ACT 1 has dtype 2",
                             (void)rematWireTableInit(&t, &f.m, defaultLossConfig(MSE), x));
    freeQuantLayer(f.layers[0]);
    freeAddLayer(f.layers[1]);
}

/* The same model through the eval-only table: the merge reads ACT 1 at inference too. */
void testEvalOnlyTableInitRefusesASymInt32WireIntoAnAdd(void) {
    symInt32QConfig_t symQc;
    initSymInt32QConfigWithQMaxBits(HALF_AWAY, &symQc, 12);
    quantization_t symQ = {.type = SYM_INT32, .qConfig = &symQc};
    dagFixture_t f;
    dagBegin(&f);
    uint16_t b1 = graphNode(&f.g, makeQuant(&symQ, &g_floatQ), GRAPH_INPUT);
    (void)graphMerge(&f.g, addLayerInit(NULL), b1, GRAPH_INPUT);
    f.m = graphFinish(&f.g);
    dagTensor_t xs;
    tensor_t *x = dagInput(&xs, 1, 2);
    rematWireTable_t *t = NULL;
    ASSERT_EXITS_WITH_OUTPUT(1, "non-FLOAT32 merge or accumulate: wire ACT 1 has dtype 2",
                             (void)rematWireTableInitEval(&t, &f.m, x));
    ASSERT_EXITS_WITH_OUTPUT(1, "rematWireTableInitEval: non-FLOAT32",
                             (void)rematWireTableInitEval(&t, &f.m, x));
    freeQuantLayer(f.layers[0]);
    freeAddLayer(f.layers[1]);
}

/* F3 with n1's dx wire SYM_INT32: n1 is b1's third contributor, so that wire is PART 1, which
 * the SUM step adds into GRAD 1. */
void testTableInitRefusesASymInt32Part(void) {
    symInt32QConfig_t symQc;
    initSymInt32QConfigWithQMaxBits(HALF_AWAY, &symQc, 12);
    quantization_t symQ = {.type = SYM_INT32, .qConfig = &symQc};
    dagFixture_t f;
    dagBegin(&f);
    uint16_t b1 = graphNode(&f.g, dagLinear(0, 1.0f, 1.0f, false), GRAPH_INPUT);
    uint16_t b2 = graphNode(&f.g, dagLinearWithPropLossQ(1, &symQ), b1);
    uint16_t b3 = graphMerge(&f.g, addLayerInit(NULL), b2, b1);
    (void)graphMerge(&f.g, addLayerInit(NULL), b3, b1);
    f.m = graphFinish(&f.g);
    dagTensor_t xs;
    tensor_t *x = dagInput(&xs, 1, 2);
    rematWireTable_t *t = NULL;
    ASSERT_EXITS_WITH_OUTPUT(1, "non-FLOAT32 merge or accumulate: wire PART 1 has dtype 2",
                             (void)rematWireTableInit(&t, &f.m, defaultLossConfig(MSE), x));
    dagFree(&f);
}

/* n0 = Linear(b0); n1 = Linear(b1); n2 = Linear(b1), dx SYM_INT32; n3 = ADD(b2, b3). n2 opens
 * GRAD 1 and n1 writes a FLOAT32 PART that the SUM step adds into it: no slot accumulates, yet
 * GRAD 1 is an accumulate target. */
void testTableInitRefusesASymInt32GradASumAddsInto(void) {
    symInt32QConfig_t symQc;
    initSymInt32QConfigWithQMaxBits(HALF_AWAY, &symQc, 12);
    quantization_t symQ = {.type = SYM_INT32, .qConfig = &symQc};
    dagFixture_t f;
    dagBegin(&f);
    uint16_t b1 = graphNode(&f.g, dagLinear(0, 1.0f, 1.0f, false), GRAPH_INPUT);
    uint16_t b2 = graphNode(&f.g, dagLinear(1, 2.0f, 3.0f, false), b1);
    uint16_t b3 = graphNode(&f.g, dagLinearWithPropLossQ(2, &symQ), b1);
    (void)graphMerge(&f.g, addLayerInit(NULL), b2, b3);
    f.m = graphFinish(&f.g);
    dagTensor_t xs;
    tensor_t *x = dagInput(&xs, 1, 2);
    rematWireTable_t *t = NULL;
    ASSERT_EXITS_WITH_OUTPUT(1, "non-FLOAT32 merge or accumulate: wire GRAD 1 has dtype 2",
                             (void)rematWireTableInit(&t, &f.m, defaultLossConfig(MSE), x));
    dagFree(&f);
}

/* n0 = Linear(b0); n1 = ADD(b1, b0); n2 = Linear(b1), dx SYM_INT32; n3 = ADD(b2, b3). n2 opens
 * GRAD 1, and the ADD n1 accumulates into it. */
void testTableInitRefusesASymInt32GradAnAddAccumulatesInto(void) {
    symInt32QConfig_t symQc;
    initSymInt32QConfigWithQMaxBits(HALF_AWAY, &symQc, 12);
    quantization_t symQ = {.type = SYM_INT32, .qConfig = &symQc};
    dagFixture_t f;
    dagBegin(&f);
    uint16_t b1 = graphNode(&f.g, dagLinear(0, 1.0f, 1.0f, false), GRAPH_INPUT);
    uint16_t b2 = graphMerge(&f.g, addLayerInit(NULL), b1, GRAPH_INPUT);
    uint16_t b3 = graphNode(&f.g, dagLinearWithPropLossQ(2, &symQ), b1);
    (void)graphMerge(&f.g, addLayerInit(NULL), b2, b3);
    f.m = graphFinish(&f.g);
    dagTensor_t xs;
    tensor_t *x = dagInput(&xs, 1, 2);
    rematWireTable_t *t = NULL;
    ASSERT_EXITS_WITH_OUTPUT(1, "non-FLOAT32 merge or accumulate: wire GRAD 1 has dtype 2",
                             (void)rematWireTableInit(&t, &f.m, defaultLossConfig(MSE), x));
    dagFree(&f);
}

/* n0 = Linear(b0); n1 = ADD(b1, b1); n2 = Linear(b2), dx SYM_INT32: GRAD 2, which n2 opens, is
 * the dy the ADD accumulates from. */
void testTableInitRefusesASymInt32DyOfAnAdd(void) {
    symInt32QConfig_t symQc;
    initSymInt32QConfigWithQMaxBits(HALF_AWAY, &symQc, 12);
    quantization_t symQ = {.type = SYM_INT32, .qConfig = &symQc};
    dagFixture_t f;
    dagBegin(&f);
    uint16_t b1 = graphNode(&f.g, dagLinear(0, 1.0f, 1.0f, false), GRAPH_INPUT);
    uint16_t b2 = graphMerge(&f.g, addLayerInit(NULL), b1, b1);
    (void)graphNode(&f.g, dagLinearWithPropLossQ(2, &symQ), b2);
    f.m = graphFinish(&f.g);
    dagTensor_t xs;
    tensor_t *x = dagInput(&xs, 1, 2);
    rematWireTable_t *t = NULL;
    ASSERT_EXITS_WITH_OUTPUT(1, "non-FLOAT32 merge or accumulate: wire GRAD 2 has dtype 2",
                             (void)rematWireTableInit(&t, &f.m, defaultLossConfig(MSE), x));
    dagFree(&f);
}

/* n0 = Linear(b0); n1 = ADD(b1, b1); n2 = Softmax, outputQ SYM_INT32, under CrossEntropy: the
 * seed GRAD 3 inherits the SYM_INT32 output and, the Softmax's backward being skipped, is the dy
 * the ADD accumulates from. */
void testTableInitRefusesASymInt32CrossEntropySeedIntoAnAdd(void) {
    symInt32QConfig_t symQc;
    initSymInt32QConfigWithQMaxBits(HALF_AWAY, &symQc, 12);
    quantization_t symQ = {.type = SYM_INT32, .qConfig = &symQc};
    layerQuant_t lq;
    layerQuantInitUniform(&lq, dagFloatQ());
    lq.outputQ = &symQ;
    dagFixture_t f;
    dagBegin(&f);
    uint16_t b1 = graphNode(&f.g, dagLinear(0, 1.0f, 1.0f, false), GRAPH_INPUT);
    uint16_t b2 = graphMerge(&f.g, addLayerInit(NULL), b1, b1);
    (void)graphNode(&f.g, softmaxLayerInit(&lq), b2);
    f.m = graphFinish(&f.g);
    dagTensor_t xs;
    tensor_t *x = dagInput(&xs, 1, 2);
    rematWireTable_t *t = NULL;
    ASSERT_EXITS_WITH_OUTPUT(
        1, "non-FLOAT32 merge or accumulate: wire GRAD 3 has dtype 2",
        (void)rematWireTableInit(&t, &f.m, defaultLossConfig(CROSS_ENTROPY), x));
    freeLinearLayer(f.layers[0]);
    freeAddLayer(f.layers[1]);
    freeSoftmaxLayer(f.layers[2]);
}

void testUnbindClearsTheBorrowedInputOnly(void) {
    layer_t *model[1] = {makeRelu(&g_floatQ)};
    inputLike_t in;
    tensor_t *x = makeInput(&in, (size_t[]){1, 4}, 2, &g_floatQ);
    rematWireTable_t *t = initTable(model, 1, MSE, x);
    rematWireTableBind(t, &MODEL_CHAIN(model, 1), MSE, REMAT_MODE_TRAIN, x);
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
    ASSERT_EXITS_WITH_OUTPUT(
        1, "key mismatch on 'modelSize': built 2, live 1",
        rematWireTableBind(t, &MODEL_CHAIN(model, 1), MSE, REMAT_MODE_TRAIN, x));
    rematWireTableFree(t);
    freeModel(model, 2);
}

void testBindExitsOnAChangedLossType(void) {
    layer_t *model[2] = {makeLinear(4, 3, false, 0), makeSoftmax()};
    inputLike_t in;
    tensor_t *x = makeInput(&in, (size_t[]){1, 4}, 2, &g_floatQ);
    rematWireTable_t *t = initTable(model, 2, CROSS_ENTROPY, x);
    ASSERT_EXITS_WITH_OUTPUT(
        1, "key mismatch on 'lossType': built 1, live 0",
        rematWireTableBind(t, &MODEL_CHAIN(model, 2), MSE, REMAT_MODE_TRAIN, x));
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
    ASSERT_EXITS_WITH_OUTPUT(
        1, "key mismatch on 'layerType[1]': built 1, live 6",
        rematWireTableBind(t, &MODEL_CHAIN(model, 2), MSE, REMAT_MODE_TRAIN, x));
    rematWireTableFree(t);
    model[1] = relu;
    freeModel(model, 2);
    freeSoftmaxLayer(softmax);
}

/* Freezing conv 0 makes conv 3 the first trainable layer, so nodes 0..2 stop
 * running backward; index 0 differs first. */
void testBindExitsWhenFreezingStopsALayersBackward(void) {
    layer_t *model[HAR_N];
    buildHar(model, false);
    inputLike_t in;
    tensor_t *x = makeHarInput(&in);
    rematWireTable_t *t = initTable(model, HAR_N, CROSS_ENTROPY, x);
    model[0]->config->conv1d->frozen = true;
    ASSERT_EXITS_WITH_OUTPUT(
        1, "key mismatch on 'runsBackward[0]': built 1, live 0",
        rematWireTableBind(t, &MODEL_CHAIN(model, HAR_N), CROSS_ENTROPY, REMAT_MODE_TRAIN, x));
    model[0]->config->conv1d->frozen = false;
    rematWireTableFree(t);
    freeModel(model, HAR_N);
}

/* Freezing a layer above the first trainable one changes no runsBackward flag; the frozen[]
 * key catches it. */
void testBindExitsWhenFreezingALayerAboveDeepest(void) {
    layer_t *model[HAR_N];
    buildHar(model, false);
    inputLike_t in;
    tensor_t *x = makeHarInput(&in);
    rematWireTable_t *t = initTable(model, HAR_N, CROSS_ENTROPY, x);
    model[3]->config->conv1d->frozen = true;
    ASSERT_EXITS_WITH_OUTPUT(
        1, "key mismatch on 'frozen[3]': built 0, live 1",
        rematWireTableBind(t, &MODEL_CHAIN(model, HAR_N), CROSS_ENTROPY, REMAT_MODE_TRAIN, x));
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
    ASSERT_EXITS_WITH_OUTPUT(
        1, "key mismatch on wire ACT 0, field 'rank': built 2, live 3",
        rematWireTableBind(t, &MODEL_CHAIN(model, 1), MSE, REMAT_MODE_TRAIN, x3));
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
    ASSERT_EXITS_WITH_OUTPUT(
        1, "key mismatch on wire ACT 0, field 'dims[0]': built 1, live 2",
        rematWireTableBind(t, &MODEL_CHAIN(model, 1), MSE, REMAT_MODE_TRAIN, x2));
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
    ASSERT_EXITS_WITH_OUTPUT(
        1, "key mismatch on wire ACT 0, field 'order[0]': built 0, live 1",
        rematWireTableBind(t, &MODEL_CHAIN(model, 1), MSE, REMAT_MODE_TRAIN, xT));
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
    ASSERT_EXITS_WITH_OUTPUT(
        1, "key mismatch on wire ACT 0, field 'dtype': built 1, live 2",
        rematWireTableBind(t, &MODEL_CHAIN(model, 1), MSE, REMAT_MODE_TRAIN, xS));
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
    ASSERT_EXITS_WITH_OUTPUT(
        1, "key mismatch on wire ACT 1, field 'bytes': built 8, live 4",
        rematWireTableBind(t, &MODEL_CHAIN(model, 1), MSE, REMAT_MODE_TRAIN, x));
    rematWireTableFree(t);
    freeModel(model, 1);
}

/* Check before write, dtype twin: a byte-neutral FLOAT32 -> SYM_INT32 template edit.
 * FLOAT32 reserved no qConfig, so a write before the check goes through a NULL
 * qConfig: a crash (or, under ASan, exit 86), never the named exit. */
static void bindUnderTheAsanCallback(rematWireTable_t *t, layer_t **model, size_t n,
                                     lossFuncType_t lt, tensor_t *x) {
    odtInstallAsanDeathExit();
    rematWireTableBind(t, &MODEL_CHAIN(model, n), lt, REMAT_MODE_TRAIN, x);
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
    rematWireTableBind(f.t, &MODEL_CHAIN(f.model, 2), MSE, REMAT_MODE_TRAIN, f.x);
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
    layer_t *model[2] = {makeLinear(2, 4, false, 0), makeQuant(&symQ, &g_floatQ)};
    inputLike_t in;
    tensor_t *x = makeInput(&in, (size_t[]){1, 2}, 2, &g_floatQ);
    rematWireTable_t *t = initTable(model, 2, MSE, x);
    rematWireTableBind(t, &MODEL_CHAIN(model, 2), MSE, REMAT_MODE_TRAIN, x);
    uint32_t act2[4];
    uint32_t seed[4];
    rematWireBind(t, 2, (uint8_t *)act2);
    symInt32QConfig_t *act2Qc = rematActHdr(t, 2)->quantization->qConfig;
    act2Qc->scale = odtTestScale(0.25f);
    rematWireBind(t, rematGradId(t, 2), (uint8_t *)seed);
    symInt32QConfig_t *seedQc = rematGradHdr(t, 2)->quantization->qConfig;
    TEST_ASSERT_EQUAL_UINT8(10, seedQc->qMaxBits);
    TEST_ASSERT_EQUAL_INT(SR_HALF_AWAY, seedQc->roundingMode);
    TEST_ASSERT_EQUAL_FLOAT(1.0f, odtScaleToFloat(seedQc->scale));
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
    layer_t *model[2] = {makeLinear(2, 4, false, 0), makeQuant(&symQ, &g_floatQ)};
    inputLike_t in;
    tensor_t *x = makeInput(&in, (size_t[]){1, 2}, 2, &g_floatQ);
    rematWireTable_t *t = initTable(model, 2, MSE, x);
    rematWireTableBind(t, &MODEL_CHAIN(model, 2), MSE, REMAT_MODE_TRAIN, x);
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
    rematWireTableBind(f.t, &MODEL_CHAIN(f.model, 2), MSE, REMAT_MODE_TRAIN, f.x);
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
    rematWireTableBind(f->t, &MODEL_CHAIN(f->model, 3), MSE, REMAT_MODE_TRAIN, f->x);
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

_Static_assert(sizeof(rematOperand_t) == 4, "an operand is a uint16_t wire and a uint8_t mode");

/* One block of steps, operands, nodeOps, ranges and endOrder for both
 * programs, after the (even-sized) plan struct. HAR TRAIN arrays:
 * 25 * 8 + 60 * 4 + 13 * 6 + 23 * 6 + 23 * 2 = 702 B; EVAL arrays:
 * 13 * 8 + 25 * 4 + 13 * 6 + 12 * 6 + 12 * 2 = 378 B. Every array is
 * 2-aligned, so the arrays pack without padding. */
void testPlanBlockHoldsStepsRangesAndEndOrder(void) {
    layer_t *model[HAR_N];
    buildHar(model, false);
    inputLike_t in;
    rematWireTable_t *t = initTable(model, HAR_N, CROSS_ENTROPY, makeHarInput(&in));
    rematPlan_t *p = buildPlan(t, model, NULL);
    TEST_ASSERT_EQUAL_size_t(sizeof(rematPlan_t) + 702u + 378u, p->blockBytes);
    const uint8_t *begin = (const uint8_t *)p;
    const uint8_t *end = begin + p->blockBytes;
    const rematProgram_t *eval = rematPlanProgram(p, REMAT_MODE_EVAL);
    TEST_ASSERT_TRUE((const uint8_t *)p->train.steps >= begin + sizeof(rematPlan_t));
    TEST_ASSERT_TRUE((const uint8_t *)p->train.operands >=
                     (const uint8_t *)(p->train.steps + p->train.numSteps));
    TEST_ASSERT_TRUE((const uint8_t *)p->train.nodeOps >=
                     (const uint8_t *)(p->train.operands + p->train.numOperands));
    TEST_ASSERT_TRUE((const uint8_t *)p->train.ranges >=
                     (const uint8_t *)(p->train.nodeOps + HAR_N + 1));
    TEST_ASSERT_TRUE((const uint8_t *)eval->steps >=
                     (const uint8_t *)(p->train.endOrder + p->train.numRanges));
    TEST_ASSERT_TRUE((const uint8_t *)(eval->endOrder + eval->numRanges) <= end);
    rematPlanFree(p);
    rematWireTableFree(t);
    freeModel(model, HAR_N);
}

/* ---- the operand pool ---- */

static void assertSlice(const rematProgram_t *p, const rematStep_t *st, uint8_t numReads,
                        uint8_t numWrites, const rematOperand_t *expected) {
    TEST_ASSERT_EQUAL_UINT8(numReads, st->numReads);
    TEST_ASSERT_EQUAL_UINT8(numWrites, st->numWrites);
    TEST_ASSERT_TRUE(st->operands + numReads + numWrites <= p->numOperands);
    for (size_t i = 0; i < (size_t)numReads + numWrites; i++) {
        TEST_ASSERT_EQUAL_UINT16(expected[i].wire, p->operands[st->operands + i].wire);
        TEST_ASSERT_EQUAL_UINT8(expected[i].mode, p->operands[st->operands + i].mode);
    }
}

/* HAR STORE_ALL (CE): wires ACT 0..12, seed 13, GRAD l at 24 - l. Pool:
 * FORWARD(j) at 2j, LOSS_FORWARD 24, LOSS_BACKWARD 25, BACKWARD(10..0) at 27 + 3(10 - l).
 * A pool, Flatten or Dropout BACKWARD does not read its input: META. */
void testHarTrainOperandsAreTheDerivedSlices(void) {
    layer_t *model[HAR_N];
    buildHar(model, false);
    inputLike_t in;
    rematWireTable_t *t = initTable(model, HAR_N, CROSS_ENTROPY, makeHarInput(&in));
    rematPlan_t *p = buildPlan(t, model, NULL);
    const rematProgram_t *tr = &p->train;
    TEST_ASSERT_EQUAL_size_t(60, tr->numOperands);
    for (uint16_t j = 0; j < 12; j++) {
        TEST_ASSERT_EQUAL_UINT16(2 * j, tr->nodeOps[j].forward);
        TEST_ASSERT_EQUAL_UINT16(REMAT_NONE, tr->nodeOps[j].sum);
        assertSlice(
            tr, &tr->steps[j], 1, 1,
            (const rematOperand_t[]){{j, OP_MODE_READ}, {(uint16_t)(j + 1), OP_MODE_WRITE}});
    }
    TEST_ASSERT_EQUAL_UINT16(24, tr->nodeOps[12].forward);
    TEST_ASSERT_EQUAL_UINT16(25, tr->nodeOps[12].backward);
    TEST_ASSERT_EQUAL_UINT16(REMAT_NONE, tr->nodeOps[12].sum);
    TEST_ASSERT_EQUAL_UINT16(REMAT_NONE, tr->nodeOps[11].backward); /* the CE-skipped Softmax */
    assertSlice(tr, &tr->steps[12], 1, 0, (const rematOperand_t[]){{12, OP_MODE_READ}});
    assertSlice(tr, &tr->steps[13], 1, 1,
                (const rematOperand_t[]){{12, OP_MODE_READ}, {13, OP_MODE_WRITE}});
    static const struct {
        uint16_t at, dy;
        uint8_t xMode;
        uint16_t dx;
    } b[11] = {
        {27, 13, OP_MODE_READ, 14}, {30, 14, OP_MODE_META, 15},        {33, 15, OP_MODE_META, 16},
        {36, 16, OP_MODE_READ, 17}, {39, 17, OP_MODE_READ, 18},        {42, 18, OP_MODE_META, 19},
        {45, 19, OP_MODE_READ, 20}, {48, 20, OP_MODE_READ, 21},        {51, 21, OP_MODE_META, 22},
        {54, 22, OP_MODE_READ, 23}, {57, 23, OP_MODE_READ, REMAT_NONE}};
    for (uint16_t k = 0; k < 11; k++) {
        uint16_t l = (uint16_t)(10 - k);
        TEST_ASSERT_EQUAL_UINT16(b[k].at, tr->nodeOps[l].backward);
        TEST_ASSERT_EQUAL_UINT16(b[k].at, tr->steps[14 + k].operands);
        assertSlice(tr, &tr->steps[14 + k], 2, 1,
                    (const rematOperand_t[]){
                        {b[k].dy, OP_MODE_READ}, {l, b[k].xMode}, {b[k].dx, OP_MODE_WRITE}});
    }
    rematPlanFree(p);
    rematWireTableFree(t);
    freeModel(model, HAR_N);
}

/* The first n + 1 TRAIN steps, their operands included, are the EVAL program,
 * under every policy. */
void testEvalProgramIsTheTrainPrefixOperandsIncluded(void) {
    const rematPlanSpec_t *specs[3] = {NULL, &g_liveness, &g_livenessInPlace};
    for (size_t k = 0; k < 3u; k++) {
        layer_t *model[HAR_N];
        buildHar(model, false);
        inputLike_t in;
        rematWireTable_t *t = initTable(model, HAR_N, CROSS_ENTROPY, makeHarInput(&in));
        rematPlan_t *p = buildPlan(t, model, specs[k]);
        const rematProgram_t *train = &p->train;
        const rematProgram_t *eval = rematPlanProgram(p, REMAT_MODE_EVAL);
        TEST_ASSERT_EQUAL_size_t(25, eval->numOperands);
        TEST_ASSERT_EQUAL_MEMORY(train->operands, eval->operands, 25 * sizeof(rematOperand_t));
        TEST_ASSERT_EQUAL_size_t(13, eval->numSteps);
        TEST_ASSERT_EQUAL_MEMORY(train->steps, eval->steps, 13 * sizeof(rematStep_t));
        for (size_t j = 0; j <= 12; j++) {
            TEST_ASSERT_EQUAL_UINT16(train->nodeOps[j].forward, eval->nodeOps[j].forward);
            TEST_ASSERT_EQUAL_UINT16(REMAT_NONE, eval->nodeOps[j].backward);
            TEST_ASSERT_EQUAL_UINT16(REMAT_NONE, eval->nodeOps[j].sum);
        }
        rematPlanFree(p);
        rematWireTableFree(t);
        freeModel(model, HAR_N);
    }
}

/* F3 EVAL: a merge node's FORWARD reads each of its input buffers in CSR order, and the shared
 * b1 is read by three FORWARDs. Derived directly: the chain-form grammar of a plan build
 * rejects a DAG. */
void testEvalSlicesOfADagReadTheWiring(void) {
    dagFixture_t f;
    dagBuildF3(&f, false);
    dagTensor_t xs;
    rematWireTable_t *t;
    TEST_ASSERT_TRUE(rematWireTableInit(&t, &f.m, defaultLossConfig(MSE), dagInput(&xs, 1, 2)));
    rematDeriveSize_t size = rematDeriveEvalSize(t);
    TEST_ASSERT_EQUAL_size_t(5, size.numSteps);
    TEST_ASSERT_EQUAL_size_t(11, size.numOperands);
    rematStep_t steps[5];
    rematOperand_t operands[11];
    rematNodeOps_t nodeOps[5];
    rematProgram_t p = {
        .numSteps = 5, .steps = steps, .numOperands = 11, .operands = operands, .nodeOps = nodeOps};
    rematDeriveEval(t, &p);
    TEST_ASSERT_EQUAL_UINT8(REMAT_STEP_FORWARD, steps[2].kind);
    TEST_ASSERT_EQUAL_UINT16(2, steps[2].layer);
    assertSlice(&p, &steps[2], 2, 1,
                (const rematOperand_t[]){{2, OP_MODE_READ}, {1, OP_MODE_READ}, {3, OP_MODE_WRITE}});
    TEST_ASSERT_EQUAL_UINT8(REMAT_STEP_FORWARD, steps[3].kind);
    TEST_ASSERT_EQUAL_UINT16(3, steps[3].layer);
    assertSlice(&p, &steps[3], 2, 1,
                (const rematOperand_t[]){{3, OP_MODE_READ}, {1, OP_MODE_READ}, {4, OP_MODE_WRITE}});
    rematWireTableFree(t);
    dagFree(&f);
}

static const rematPlanSpec_t g_storeAllSpec = {.policy = REMAT_PLAN_STORE_ALL};

/* Room for the DAG fixtures' programs, derived directly into test arrays, so a test can read or
 * tamper with every step before the grammar sees it. Zeroed, so the pool's padding bytes
 * compare. */
typedef struct dagProgram {
    rematStep_t steps[24];
    rematOperand_t operands[48];
    rematNodeOps_t nodeOps[6];
    rematRange_t ranges[16];
    rematProgram_t p;
} dagProgram_t;

static void deriveDagTrain(const rematWireTable_t *t, const rematPlanSpec_t *spec,
                           rematDeriveSize_t z, dagProgram_t *d) {
    TEST_ASSERT_TRUE(z.numSteps <= 24u && z.numOperands <= 48u && t->modelSize < 6u);
    memset(d, 0, sizeof *d);
    d->p = (rematProgram_t){.numSteps = z.numSteps,
                            .steps = d->steps,
                            .numOperands = z.numOperands,
                            .operands = d->operands,
                            .nodeOps = d->nodeOps};
    rematDeriveTrain(t, spec, &d->p);
}

static void assertStepSlice(const rematProgram_t *p, size_t s, rematStepKind_t kind, uint16_t layer,
                            const rematOperand_t *reads, uint8_t numReads,
                            const rematOperand_t *writes, uint8_t numWrites) {
    const rematStep_t *st = &p->steps[s];
    TEST_ASSERT_EQUAL_UINT8(kind, st->kind);
    TEST_ASSERT_EQUAL_UINT16(layer, st->layer);
    TEST_ASSERT_EQUAL_UINT8(numReads, st->numReads);
    TEST_ASSERT_EQUAL_UINT8(numWrites, st->numWrites);
    TEST_ASSERT_TRUE(st->operands + numReads + numWrites <= p->numOperands);
    const rematOperand_t *ops = &p->operands[st->operands];
    for (size_t k = 0; k < numReads; k++) {
        TEST_ASSERT_EQUAL_UINT16(reads[k].wire, ops[k].wire);
        TEST_ASSERT_EQUAL_UINT8(reads[k].mode, ops[k].mode);
    }
    for (size_t k = 0; k < numWrites; k++) {
        TEST_ASSERT_EQUAL_UINT16(writes[k].wire, ops[numReads + k].wire);
        TEST_ASSERT_EQUAL_UINT8(writes[k].mode, ops[numReads + k].mode);
    }
}

#define R(w) {(w), OP_MODE_READ}
#define M(w) {(w), OP_MODE_META}
#define W(w) {(w), OP_MODE_WRITE}
#define A(w) {(w), OP_MODE_ACC}
#define OPS(...) ((const rematOperand_t[]){__VA_ARGS__})

/* F3 under MSE: ACT 0..4, the seed 5, GRAD 3 = 6, GRAD 1 = 7, GRAD 2 = 8, PART 1 = 9. b1 has three
 * contributors: B3 opens GRAD 1, the ADD B2 accumulates natively, and the Linear B1 writes the PART
 * that the SUM right after it adds in. Pool: 11 forward operands, LOSS_BACKWARD 2, B3 and B2 5
 * each, B1 3, SUM 2, B0 3. */
void testDeriveF3TrainProgram(void) {
    dagFixture_t f;
    dagBuildF3(&f, false);
    dagTensor_t xs;
    rematWireTable_t *t;
    TEST_ASSERT_TRUE(rematWireTableInit(&t, &f.m, defaultLossConfig(MSE), dagInput(&xs, 1, 2)));
    const rematPlanSpec_t spec = {.policy = REMAT_PLAN_STORE_ALL};
    rematDeriveSize_t z = rematDeriveTrainSize(t, &spec);
    TEST_ASSERT_EQUAL_size_t(11u, z.numSteps);
    TEST_ASSERT_EQUAL_size_t(31u, z.numOperands);
    TEST_ASSERT_EQUAL_size_t(9u, z.numRanges);
    dagProgram_t d;
    deriveDagTrain(t, &g_storeAllSpec, z, &d);
    const rematProgram_t *p = &d.p;
    assertStepSlice(p, 0, REMAT_STEP_FORWARD, 0, OPS(R(0)), 1, OPS(W(1)), 1);
    assertStepSlice(p, 1, REMAT_STEP_FORWARD, 1, OPS(R(1)), 1, OPS(W(2)), 1);
    assertStepSlice(p, 2, REMAT_STEP_FORWARD, 2, OPS(R(2), R(1)), 2, OPS(W(3)), 1);
    assertStepSlice(p, 3, REMAT_STEP_FORWARD, 3, OPS(R(3), R(1)), 2, OPS(W(4)), 1);
    assertStepSlice(p, 4, REMAT_STEP_LOSS_FORWARD, 4, OPS(R(4)), 1, NULL, 0);
    assertStepSlice(p, 5, REMAT_STEP_LOSS_BACKWARD, 4, OPS(R(4)), 1, OPS(W(5)), 1);
    assertStepSlice(p, 6, REMAT_STEP_BACKWARD, 3, OPS(R(5), M(3), M(1)), 3, OPS(W(6), W(7)), 2);
    assertStepSlice(p, 7, REMAT_STEP_BACKWARD, 2, OPS(R(6), M(2), M(1)), 3, OPS(W(8), A(7)), 2);
    assertStepSlice(p, 8, REMAT_STEP_BACKWARD, 1, OPS(R(8), R(1)), 2, OPS(W(9)), 1);
    assertStepSlice(p, 9, REMAT_STEP_SUM, 1, OPS(R(9)), 1, OPS(A(7)), 1);
    assertStepSlice(p, 10, REMAT_STEP_BACKWARD, 0, OPS(R(7), R(0)), 2, OPS(W(REMAT_NONE)), 1);
    TEST_ASSERT_EQUAL_UINT16(p->steps[9].operands, p->nodeOps[1].sum);
    for (size_t j = 0; j <= 4u; j++) {
        if (j != 1u) {
            TEST_ASSERT_EQUAL_UINT16(REMAT_NONE, p->nodeOps[j].sum);
        }
    }
    TEST_ASSERT_EQUAL_UINT16(p->steps[6].operands, p->nodeOps[3].backward);
    TEST_ASSERT_EQUAL_UINT16(p->steps[2].operands, p->nodeOps[2].forward);
    rematWireTableFree(t);
    dagFree(&f);
}

/* The first n + 1 TRAIN steps of F3, operands included, are its EVAL program. */
void testDeriveF3EvalIsTheTrainPrefix(void) {
    dagFixture_t f;
    dagBuildF3(&f, false);
    dagTensor_t xs;
    rematWireTable_t *t;
    TEST_ASSERT_TRUE(rematWireTableInit(&t, &f.m, defaultLossConfig(MSE), dagInput(&xs, 1, 2)));
    dagProgram_t train;
    deriveDagTrain(t, &g_storeAllSpec, rematDeriveTrainSize(t, &g_storeAllSpec), &train);
    rematDeriveSize_t z = rematDeriveEvalSize(t);
    TEST_ASSERT_EQUAL_size_t(5u, z.numSteps);
    TEST_ASSERT_EQUAL_size_t(11u, z.numOperands);
    dagProgram_t eval;
    memset(&eval, 0, sizeof eval);
    eval.p = (rematProgram_t){.numSteps = z.numSteps,
                              .steps = eval.steps,
                              .numOperands = z.numOperands,
                              .operands = eval.operands,
                              .nodeOps = eval.nodeOps};
    rematDeriveEval(t, &eval.p);
    TEST_ASSERT_EQUAL_MEMORY(train.steps, eval.steps, 5u * sizeof(rematStep_t));
    TEST_ASSERT_EQUAL_MEMORY(train.operands, eval.operands, 11u * sizeof(rematOperand_t));
    for (size_t j = 0; j <= 4u; j++) {
        TEST_ASSERT_EQUAL_UINT16(train.nodeOps[j].forward, eval.nodeOps[j].forward);
        TEST_ASSERT_EQUAL_UINT16(REMAT_NONE, eval.nodeOps[j].backward);
        TEST_ASSERT_EQUAL_UINT16(REMAT_NONE, eval.nodeOps[j].sum);
    }
    rematWireTableFree(t);
    dagFree(&f);
}

/* S: n1 is frozen and reads only the input, so it runs no BACKWARD and b2 gets no dx; the backward
 * phase skips it (B2, then B0). ACT 0..3, the seed 4, GRAD 1 = 5. No PART, so no SUM. */
void testDeriveSkipsTheFrozenSkipNode(void) {
    dagFixture_t f;
    dagBuildS(&f);
    dagTensor_t xs;
    rematWireTable_t *t;
    TEST_ASSERT_TRUE(rematWireTableInit(&t, &f.m, defaultLossConfig(MSE), dagInput(&xs, 1, 2)));
    TEST_ASSERT_EQUAL_UINT16(5, t->gradIdOf[1]);
    rematDeriveSize_t z =
        rematDeriveTrainSize(t, &(rematPlanSpec_t){.policy = REMAT_PLAN_STORE_ALL});
    TEST_ASSERT_EQUAL_size_t(7u, z.numSteps);
    dagProgram_t d;
    deriveDagTrain(t, &g_storeAllSpec, z, &d);
    const rematProgram_t *p = &d.p;
    assertStepSlice(p, 0, REMAT_STEP_FORWARD, 0, OPS(R(0)), 1, OPS(W(1)), 1);
    assertStepSlice(p, 1, REMAT_STEP_FORWARD, 1, OPS(R(0)), 1, OPS(W(2)), 1);
    assertStepSlice(p, 2, REMAT_STEP_FORWARD, 2, OPS(R(1), R(2)), 2, OPS(W(3)), 1);
    assertStepSlice(p, 3, REMAT_STEP_LOSS_FORWARD, 3, OPS(R(3)), 1, NULL, 0);
    assertStepSlice(p, 4, REMAT_STEP_LOSS_BACKWARD, 3, OPS(R(3)), 1, OPS(W(4)), 1);
    assertStepSlice(p, 5, REMAT_STEP_BACKWARD, 2, OPS(R(4), M(1), M(2)), 3,
                    OPS(W(5), W(REMAT_NONE)), 2);
    assertStepSlice(p, 6, REMAT_STEP_BACKWARD, 0, OPS(R(5), R(0)), 2, OPS(W(REMAT_NONE)), 1);
    for (size_t j = 0; j <= 3u; j++) {
        TEST_ASSERT_EQUAL_UINT16(REMAT_NONE, p->nodeOps[j].sum);
    }
    TEST_ASSERT_EQUAL_UINT16(REMAT_NONE, p->nodeOps[1].backward);
    rematWireTableFree(t);
    dagFree(&f);
}

/* ---- the grammar on DAGs (derived programs, tampered inside the death-test child) ---- */

/* A DAG fixture's STORE_ALL TRAIN program under MSE with its scanned ranges. Built in place:
 * the model points into the fixture's own arrays. */
typedef struct dagGrammar {
    dagFixture_t f;
    dagTensor_t xs;
    rematWireTable_t *t;
    dagProgram_t d;
} dagGrammar_t;

/* The ranges of the program as it now stands under a policy the DAG gates admit; a tamper that
 * changes the steps calls it again. */
static void scanDagRanges(dagGrammar_t *g, rematPlanPolicy_t policy) {
    for (size_t r = 0; r < 16u; r++) {
        g->d.ranges[r] = (rematRange_t){.wire = REMAT_NONE};
    }
    rematScanRanges(g->t, &g->d.p, policy, NULL, g->d.ranges);
    size_t opened = 0;
    while (opened < 16u && g->d.ranges[opened].wire != REMAT_NONE) {
        opened++;
    }
    g->d.p.ranges = g->d.ranges;
    g->d.p.numRanges = opened;
}

static void buildDagGrammar(dagGrammar_t *g, void (*build)(dagFixture_t *)) {
    build(&g->f);
    TEST_ASSERT_TRUE(
        rematWireTableInit(&g->t, &g->f.m, defaultLossConfig(MSE), dagInput(&g->xs, 1, 2)));
    deriveDagTrain(g->t, &g_storeAllSpec, rematDeriveTrainSize(g->t, &g_storeAllSpec), &g->d);
    scanDagRanges(g, REMAT_PLAN_STORE_ALL);
}

static void freeDagGrammar(dagGrammar_t *g) {
    rematWireTableFree(g->t);
    dagFree(&g->f);
}

static void validateDag(dagGrammar_t *g) {
    rematPlanValidateGrammar(&g->d.p, g->t, &g->f.m, REMAT_MODE_TRAIN, NULL);
}

static void buildF3(dagFixture_t *f) {
    dagBuildF3(f, false);
}

static void tamperAndValidateDag(dagGrammar_t *g, void (*tamper)(dagGrammar_t *)) {
    tamper(g);
    validateDag(g);
}

#define ASSERT_DAG_GRAMMAR_EXIT(build, tamper, message)                                            \
    do {                                                                                           \
        dagGrammar_t _g;                                                                           \
        buildDagGrammar(&_g, build);                                                               \
        ASSERT_EXITS_WITH_OUTPUT(1, message, tamperAndValidateDag(&_g, tamper));                   \
        freeDagGrammar(&_g);                                                                       \
    } while (0)

static void assertTheGrammarAccepts(void (*build)(dagFixture_t *)) {
    dagGrammar_t g;
    buildDagGrammar(&g, build);
    ASSERT_EXITS_WITH(0, validateDag(&g));
    scanDagRanges(&g, REMAT_PLAN_LIVENESS);
    ASSERT_EXITS_WITH(0, validateDag(&g));
    freeDagGrammar(&g);
}

/* F3's program, under STORE_ALL and LIVENESS ranges (B3 opens GRAD 1 = 7, the ADD B2 accumulates
 * into it, the Linear B1 writes PART 9 and the SUM right after adds it) and S's, whose backward
 * skips the frozen n1. */
void testGrammarAcceptsF3AndS(void) {
    assertTheGrammarAccepts(buildF3);
    assertTheGrammarAccepts(dagBuildS);
}

/* Y opens and accumulates GRAD 1 in one step; K's skip from the input gets no dx; P2 writes two
 * PARTs into one buffer, each summed right after its BACKWARD. */
void testGrammarAcceptsYKAndP2(void) {
    assertTheGrammarAccepts(dagBuildY);
    assertTheGrammarAccepts(dagBuildK);
    assertTheGrammarAccepts(dagBuildP2);
}

/* F3 steps: F0..F3 = 0..3, LOSS_FORWARD 4, LOSS_BACKWARD 5, B3 6, B2 7, B1 8, SUM(1) 9, B0 10. A
 * BACKWARD's dx slots follow its dy and its k inputs in the pool. */
static rematOperand_t *dxSlot(dagGrammar_t *g, size_t node, size_t slot) {
    uint16_t ids[MODEL_MAX_ARITY];
    uint8_t k = modelNodeInputs(&g->f.m, node, ids);
    return &g->d.operands[g->d.nodeOps[node].backward + 1u + k + slot];
}

static void secondWriteOfGrad7(dagGrammar_t *g) {
    dxSlot(g, 2, 1)->mode = OP_MODE_WRITE;
}

static void accFromTheLinearB1(dagGrammar_t *g) {
    *dxSlot(g, 1, 0) = (rematOperand_t){.wire = 7, .mode = OP_MODE_ACC};
    g->d.steps[9] = g->d.steps[10];
    g->d.p.numSteps = 10;
    scanDagRanges(g, REMAT_PLAN_STORE_ALL);
}

static void swapTheSumAndB0(dagGrammar_t *g) {
    rematStep_t sum = g->d.steps[9];
    g->d.steps[9] = g->d.steps[10];
    g->d.steps[10] = sum;
}

static void dropTheAccOfB2(dagGrammar_t *g) {
    *dxSlot(g, 2, 1) = (rematOperand_t){.wire = REMAT_NONE, .mode = OP_MODE_WRITE};
}

/* Ten more SUM(1) after B0: 21 steps, one above F3's step bound 2n + 2 + n(n+1)/2 = 20, but only
 * 10 of them are not SUMs. */
static void tenSumsAfterTheLastBackward(dagGrammar_t *g) {
    for (size_t s = 11; s < 21u; s++) {
        g->d.steps[s] = g->d.steps[9];
    }
    g->d.p.numSteps = 21;
}

/* S steps: F0..F2 = 0..2, LOSS_FORWARD 3, LOSS_BACKWARD 4, B2 5, B0 6. */
static void b1OfTheFrozenSkipNode(dagGrammar_t *g) {
    g->d.steps[6].layer = 1;
}

/* P2 wires: ACT 0..5, the seed 6, GRAD 4 = 7, GRAD 1 = 8, GRAD 2 = 9, GRAD 3 = 10, PART 11 (B2),
 * PART 12 (B1). The ReLU B1 writes B2's PART again. */
static void b1WritesThePartOfB2(dagGrammar_t *g) {
    TEST_ASSERT_EQUAL_UINT16(12, dxSlot(g, 1, 0)->wire);
    dxSlot(g, 1, 0)->wire = 11;
}

void testGrammarRejectsASecondWriteOfAGrad(void) {
    ASSERT_DAG_GRAMMAR_EXIT(
        buildF3, secondWriteOfGrad7,
        "step #7 (kind 3, layer 2) violates grammar rule 7: GRAD 7 opened twice");
}

void testGrammarRejectsAnAccFromANonAccumulatingRow(void) {
    ASSERT_DAG_GRAMMAR_EXIT(buildF3, accFromTheLinearB1,
                            "violates grammar rule 7: ACC from a row without accumulatesDx");
}

void testGrammarRejectsAPartNotSummedNext(void) {
    ASSERT_DAG_GRAMMAR_EXIT(buildF3, swapTheSumAndB0,
                            "violates grammar rule 7: PART 9 not summed at the next step");
}

void testGrammarRejectsAGradReadBeforeItsLastContribution(void) {
    ASSERT_DAG_GRAMMAR_EXIT(buildF3, dropTheAccOfB2,
                            "violates grammar rule 7: GRAD 7 read with contributions missing");
}

void testGrammarRejectsABackwardOfTheFrozenSkipNode(void) {
    ASSERT_DAG_GRAMMAR_EXIT(dagBuildS, b1OfTheFrozenSkipNode,
                            "step #6 (kind 3, layer 1) violates grammar rule 3");
}

/* The step bound counts no SUM: rule 7 admits a SUM only right after the BACKWARD that wrote its
 * PART, so at most one per BACKWARD, and it is rule 7 that rejects the extra ones. */
void testGrammarRejectsASumWithoutItsPart(void) {
    ASSERT_DAG_GRAMMAR_EXIT(buildF3, tenSumsAfterTheLastBackward,
                            "step #11 (kind 5, layer 1) violates grammar rule 7: a SUM comes "
                            "right after the BACKWARD that wrote its PART");
}

void testGrammarRejectsAPartWrittenTwice(void) {
    ASSERT_DAG_GRAMMAR_EXIT(dagBuildP2, b1WritesThePartOfB2,
                            "step #11 (kind 3, layer 1) violates grammar rule 7: PART 11 written "
                            "twice");
}

/* Y steps: F0, F1, LOSS_FORWARD 2, LOSS_BACKWARD 3, B1 4, B0 5; GRAD 1 = 4. B1's two slots in
 * the other order: the ACC lands inside the range its own step opens, so no range rule sees it. */
static void accBeforeWriteInOneStep(dagGrammar_t *g) {
    *dxSlot(g, 1, 0) = (rematOperand_t){.wire = 4, .mode = OP_MODE_ACC};
    *dxSlot(g, 1, 1) = (rematOperand_t){.wire = 4, .mode = OP_MODE_WRITE};
}

void testGrammarRejectsAnAccBeforeTheOpeningWriteInOneStep(void) {
    ASSERT_DAG_GRAMMAR_EXIT(dagBuildY, accBeforeWriteInOneStep,
                            "step #4 (kind 3, layer 1) violates grammar rule 7: GRAD 4 opened "
                            "after an ACC into it");
}

/* F3 STORE_ALL ranges: ACT 1..4 = #0..#3, wire 5 #4, wire 6 #5, wire 7 [6, 10] #6, wire 8 #7,
 * wire 9 #8. Wire 7 split at B2's ACC: [6, 6] and [7, 10]. */
static void aRangeOpenedByAnAcc(dagGrammar_t *g) {
    rematRange_t *r = g->d.ranges;
    TEST_ASSERT_TRUE(r[6].wire == 7 && r[6].begin == 6 && r[7].wire == 8);
    memmove(&r[8], &r[7], (g->d.p.numRanges - 7u) * sizeof *r);
    r[6].end = 6;
    r[7] = (rematRange_t){.wire = 7, .begin = 7, .end = 10};
    g->d.p.numRanges++;
}

void testGrammarRejectsARangeAnAccOpens(void) {
    ASSERT_DAG_GRAMMAR_EXIT(buildF3, aRangeOpenedByAnAcc,
                            "range #7 (wire 7, steps 7..10) violates grammar rule 6: a range "
                            "opens where its wire is written");
}

/* Wire 7's range ends at B1, so the SUM's ACC at step 9 lies outside it. */
static void endGrad7BeforeTheSum(dagGrammar_t *g) {
    TEST_ASSERT_EQUAL_UINT16(7, g->d.ranges[6].wire);
    g->d.ranges[6].end = 8;
}

void testGrammarRejectsAnAccOutsideAnOpenRange(void) {
    ASSERT_DAG_GRAMMAR_EXIT(buildF3, endGrad7BeforeTheSum,
                            "step #9 (kind 5, layer 1) violates grammar rule 4: it writes wire 7 "
                            "outside any open range");
}

/* The alias rule reads chain positions, so a DAG program carries no in-place mark. */
static void validateDagWithAMark(dagGrammar_t *g) {
    uint8_t inPlace[16] = {0};
    inPlace[9] = 1u;
    rematPlanValidateGrammar(&g->d.p, g->t, &g->f.m, REMAT_MODE_TRAIN, inPlace);
}

void testGrammarRejectsInPlaceMarksOnADag(void) {
    dagGrammar_t g;
    buildDagGrammar(&g, buildF3);
    ASSERT_EXITS_WITH_OUTPUT(1,
                             "rematPlanBuild: in-place dx marks on a DAG model are not supported",
                             validateDagWithAMark(&g));
    freeDagGrammar(&g);
}

#undef R
#undef M
#undef W
#undef A
#undef OPS

/* A REFORWARD shares its FORWARD's slice; the pool does not depend on the policy. */
void testAReforwardCarriesItsForwardsSlice(void) {
    bool retain[HAR_N + 1] = {false};
    retain[3] = retain[7] = retain[12] = true;
    layer_t *model[HAR_N];
    buildHar(model, false);
    inputLike_t in;
    rematWireTable_t *t = initTable(model, HAR_N, CROSS_ENTROPY, makeHarInput(&in));
    rematPlanSpec_t spec = {.policy = REMAT_PLAN_RETAIN_LIST, .retain = retain};
    rematPlan_t *p = buildPlan(t, model, &spec);
    const rematProgram_t *train = &p->train;
    TEST_ASSERT_EQUAL_size_t(60, train->numOperands);
    TEST_ASSERT_EQUAL_UINT8(REMAT_STEP_REFORWARD, train->steps[14].kind);
    TEST_ASSERT_EQUAL_UINT16(7, train->steps[14].layer);
    TEST_ASSERT_EQUAL_UINT16(14, train->nodeOps[7].forward);
    assertSlice(train, &train->steps[14], 1, 1,
                (const rematOperand_t[]){{7, OP_MODE_READ}, {8, OP_MODE_WRITE}});
    TEST_ASSERT_EQUAL_UINT16(14, train->steps[14].operands);
    size_t reforwards = 0;
    bool sawLayer0 = false;
    for (size_t s = 0; s < train->numSteps; s++) {
        const rematStep_t *st = &train->steps[s];
        if (st->kind != REMAT_STEP_REFORWARD) {
            continue;
        }
        reforwards++;
        TEST_ASSERT_EQUAL_UINT16(train->nodeOps[st->layer].forward, st->operands);
        if (st->layer == 0) {
            sawLayer0 = true;
            TEST_ASSERT_EQUAL_UINT16(0, st->operands);
            assertSlice(train, st, 1, 1,
                        (const rematOperand_t[]){{0, OP_MODE_READ}, {1, OP_MODE_WRITE}});
        }
    }
    TEST_ASSERT_EQUAL_size_t(7, reforwards);
    TEST_ASSERT_TRUE(sawLayer0);
    rematPlanFree(p);
    rematWireTableFree(t);
    freeModel(model, HAR_N);
}

/* A caller's step list the grammar will reject still derives safely: the
 * Softmax's BACKWARD (CE skips it), an unknown kind, and a FORWARD and a
 * BACKWARD at layer n (whose nodeOps entry holds the loss slices) each get the
 * empty slice; so does a SUM after a BACKWARD that writes no PART (layer 10),
 * after the skipped Softmax (11), and at layer n. */
void testTheDeriverGivesARejectedSequenceStepAnEmptySlice(void) {
    layer_t *model[HAR_N];
    buildHar(model, false);
    inputLike_t in;
    rematWireTable_t *t = initTable(model, HAR_N, CROSS_ENTROPY, makeHarInput(&in));
    const rematStep_t sequence[7] = {STEP_B(11),
                                     {.kind = 9, .layer = 0},
                                     STEP_F(12),
                                     STEP_B(12),
                                     {.kind = REMAT_STEP_SUM, .layer = 10},
                                     {.kind = REMAT_STEP_SUM, .layer = 11},
                                     {.kind = REMAT_STEP_SUM, .layer = 12}};
    rematPlanSpec_t spec = {.policy = REMAT_PLAN_SEQUENCE, .sequence = sequence, .sequenceLen = 7};
    rematDeriveSize_t size = rematDeriveTrainSize(t, &spec);
    TEST_ASSERT_EQUAL_size_t(7, size.numSteps);
    TEST_ASSERT_EQUAL_size_t(60, size.numOperands);
    rematStep_t steps[7];
    rematOperand_t operands[60];
    rematNodeOps_t nodeOps[HAR_N + 1];
    memset(steps, 0xA5, sizeof steps);
    memset(operands, 0, sizeof operands);
    memset(nodeOps, 0, sizeof nodeOps);
    rematProgram_t p = {.numSteps = size.numSteps,
                        .steps = steps,
                        .numOperands = size.numOperands,
                        .operands = operands,
                        .nodeOps = nodeOps};
    rematDeriveTrain(t, &spec, &p);
    for (size_t s = 0; s < 7u; s++) {
        TEST_ASSERT_EQUAL_UINT8(sequence[s].kind, steps[s].kind);
        TEST_ASSERT_EQUAL_UINT16(sequence[s].layer, steps[s].layer);
        TEST_ASSERT_EQUAL_UINT8(0, steps[s].numReads);
        TEST_ASSERT_EQUAL_UINT8(0, steps[s].numWrites);
        TEST_ASSERT_EQUAL_UINT16(0, steps[s].operands);
    }
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
    layer_t *one[1] = {makeLinear(2, 3, false, 0)};
    t = initTable(one, 1, MSE, makeInput(&in, (size_t[]){1, 2}, 2, &g_floatQ));
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
        (void)rematPlanBuild(&p, t, &MODEL_CHAIN(model, t->modelSize),
                             &(rematPlanSpec_t){.policy = (rematPlanPolicy_t)7}));
    rematWireTableFree(t);
    freeModel(model, 1);
}

/* n = 1 under CE with a Linear: the positional skip would drop the Linear's
 * backward (its gradient would be taken as the combined Softmax + CE one), so
 * the table refuses the topology instead of planning a LOSS_BACKWARD without
 * any BACKWARD. */
static void singleLinearUnderCeTable(void) {
    layer_t *model[1] = {makeLinear(2, 3, false, 0)};
    inputLike_t in;
    (void)initTable(model, 1, CROSS_ENTROPY, makeInput(&in, (size_t[]){1, 2}, 2, &g_floatQ));
}

void testSingleLinearUnderCrossEntropyDiesAtTheTable(void) {
    ASSERT_EXITS_WITH_OUTPUT(1, "so the model must end in a SOFTMAX layer",
                             singleLinearUnderCeTable());
}

/* All frozen: no LOSS_BACKWARD, no GRAD wire (hasBackward false). */
void testAllFrozenPlanHasNoBackwardPhase(void) {
    layer_t *model[2] = {makeLinear(2, 4, true, 0), makeRelu(&g_floatQ)};
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

/* A frozen norm's input stays live through its BACKWARD, a frozen
 * GEMM's input dies at its forward. [Linear T, LayerNorm frozen, Linear frozen,
 * Linear T] under MSE: n = 4, top = 3, B(l) = 6 + (3 - l). */
void testFrozenNormStillNeedsItsInputWhileAFrozenGemmDoesNot(void) {
    layer_t *model[4] = {makeLinear(4, 4, false, 0), makeLayerNorm(4, true),
                         makeLinear(4, 4, true, 2), makeLinear(4, 2, false, 3)};
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
    layer_t *model[2] = {makeLinear(2, 3, false, 0), makeSoftmax()};
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
            model[i] = randomRank2Layer(&state, i);
        }
        /* CE only over a last Softmax (the skipped layer must be it); the
         * draw is taken either way, so the chains stay those of before */
        lossFuncType_t lt =
            (nextRandom(&state) % 2u == 0u || model[n - 1]->type != SOFTMAX) ? MSE : CROSS_ENTROPY;
        inputLike_t in;
        assertCoLiveUnderBothPolicies(model, n, lt, makeInput(&in, (size_t[]){1, 4}, 2, &g_floatQ));
        freeModel(model, n);
    }
}

typedef struct positionalOperands {
    rematOperand_t reads[2];
    size_t numReads;
    uint16_t write; /* REMAT_NONE: none */
} positionalOperands_t;

/* The operands of a chain step by position alone: a FORWARD reads ACT l and writes ACT l + 1,
 * a BACKWARD(l) reads the gradient of its output (the seed at top) and its input, and writes
 * GRAD l above deepest. A BACKWARD that does not read its input still names it, as META. */
static positionalOperands_t positionalOperands(const rematWireTable_t *t, layer_t **model,
                                               size_t deepest, ptrdiff_t top,
                                               const rematStep_t *st) {
    size_t n = t->modelSize;
    size_t l = st->layer;
    positionalOperands_t o = {.numReads = 0, .write = REMAT_NONE};
    switch (st->kind) {
    case REMAT_STEP_FORWARD:
    case REMAT_STEP_REFORWARD:
        o.reads[o.numReads++] = (rematOperand_t){rematActId(t, l), OP_MODE_READ};
        o.write = rematActId(t, l + 1u);
        break;
    case REMAT_STEP_LOSS_FORWARD:
        o.reads[o.numReads++] = (rematOperand_t){rematActId(t, n), OP_MODE_READ};
        break;
    case REMAT_STEP_LOSS_BACKWARD:
        o.reads[o.numReads++] = (rematOperand_t){rematActId(t, n), OP_MODE_READ};
        o.write = rematGradId(t, n);
        break;
    case REMAT_STEP_BACKWARD:
        o.reads[o.numReads++] = (rematOperand_t){
            ((ptrdiff_t)l == top) ? rematGradId(t, n) : rematGradId(t, l + 1u), OP_MODE_READ};
        o.reads[o.numReads++] = (rematOperand_t){
            rematActId(t, l), layerBackwardReadsInput(model[l]) ? OP_MODE_READ : OP_MODE_META};
        if (l > deepest) {
            o.write = rematGradId(t, l);
        }
        break;
    default:
        TEST_FAIL_MESSAGE("a step of unknown kind in a built plan");
    }
    return o;
}

static void assertStoredOperandsArePositional(const rematProgram_t *p, const rematWireTable_t *t,
                                              layer_t **model, lossFuncType_t lt) {
    size_t deepest;
    ptrdiff_t top;
    positionalRange(model, t->modelSize, lt, &deepest, &top);
    for (size_t s = 0; s < p->numSteps; s++) {
        const rematStep_t *st = &p->steps[s];
        positionalOperands_t want = positionalOperands(t, model, deepest, top, st);
        TEST_ASSERT_TRUE(st->operands + st->numReads + st->numWrites <= p->numOperands);
        const rematOperand_t *got = &p->operands[st->operands];
        TEST_ASSERT_EQUAL_size_t_MESSAGE(want.numReads, st->numReads, "read count");
        for (size_t i = 0; i < want.numReads; i++) {
            TEST_ASSERT_EQUAL_UINT16_MESSAGE(want.reads[i].wire, got[i].wire, "read wire");
            TEST_ASSERT_EQUAL_UINT8_MESSAGE(want.reads[i].mode, got[i].mode, "read mode");
        }
        TEST_ASSERT_EQUAL_UINT8_MESSAGE(st->kind == REMAT_STEP_LOSS_FORWARD ? 0u : 1u,
                                        st->numWrites, "write count");
        uint16_t write = REMAT_NONE;
        if (st->numWrites == 1u) {
            TEST_ASSERT_EQUAL_UINT8_MESSAGE(OP_MODE_WRITE, got[st->numReads].mode, "write mode");
            write = got[st->numReads].wire;
        }
        TEST_ASSERT_EQUAL_UINT16_MESSAGE(want.write, write, "write wire");
    }
}

/* The largest model of the operand-oracle tests (HAR), plus ACT 0. */
#define RETAIN_ALL_MAX (HAR_N + 1)

/* STORE_ALL, LIVENESS, LIVENESS + inPlaceDx and RETAIN_LIST keeping every ACT. */
static void assertStoredOperandsArePositionalUnderEveryPolicy(layer_t **model, size_t n,
                                                              lossFuncType_t lt,
                                                              const tensor_t *x) {
    bool retain[RETAIN_ALL_MAX];
    TEST_ASSERT_TRUE(n < RETAIN_ALL_MAX);
    for (size_t j = 0; j <= n; j++) {
        retain[j] = true;
    }
    const rematPlanSpec_t keepAll = {.policy = REMAT_PLAN_RETAIN_LIST, .retain = retain};
    const rematPlanSpec_t *specs[4] = {NULL, &g_liveness, &g_livenessInPlace, &keepAll};
    rematWireTable_t *t = initTable(model, n, lt, x);
    for (size_t k = 0; k < 4u; k++) {
        rematPlan_t *p = buildPlan(t, model, specs[k]);
        assertStoredOperandsArePositional(&p->train, t, model, lt);
        rematPlanFree(p);
    }
    rematWireTableFree(t);
}

/* The positional resolver the deriver replaced, frozen here as the chain
 * oracle: on the zoo, every stored step's reads, its META input and its write
 * equal it. */
void testStoredOperandsAreThePositionalOperandsOnTheZoo(void) {
    inputLike_t in;
    layer_t *har[HAR_N];
    buildHar(har, false);
    assertStoredOperandsArePositionalUnderEveryPolicy(har, HAR_N, CROSS_ENTROPY, makeHarInput(&in));
    freeModel(har, HAR_N);
    layer_t *finetune[HAR_N];
    buildHar(finetune, true);
    assertStoredOperandsArePositionalUnderEveryPolicy(finetune, HAR_N, CROSS_ENTROPY,
                                                      makeHarInput(&in));
    freeModel(finetune, HAR_N);
    layer_t *mnist[MNIST_N];
    buildMnistCnn(mnist);
    assertStoredOperandsArePositionalUnderEveryPolicy(
        mnist, MNIST_N, CROSS_ENTROPY, makeInput(&in, (size_t[]){1, 1, 784}, 3, &g_floatQ));
    freeModel(mnist, MNIST_N);
}

/* The 50 random chains of the co-live property, under both losses. */
void testStoredOperandsAreThePositionalOperandsOnRandomChains(void) {
    uint32_t state = 0x2545F491u;
    for (int trial = 0; trial < 50; trial++) {
        size_t n = 1u + nextRandom(&state) % 6u;
        layer_t *model[6];
        for (size_t i = 0; i < n; i++) {
            model[i] = randomRank2Layer(&state, i);
        }
        lossFuncType_t lt =
            (nextRandom(&state) % 2u == 0u || model[n - 1]->type != SOFTMAX) ? MSE : CROSS_ENTROPY;
        inputLike_t in;
        assertStoredOperandsArePositionalUnderEveryPolicy(
            model, n, lt, makeInput(&in, (size_t[]){1, 4}, 2, &g_floatQ));
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

/* HAR's program, which has a backward, validated against the same model with
 * every trainable layer frozen afterwards: the grammar reads needs-grad off
 * the model, so the LOSS_BACKWARD at step 13 is the first step rule 2
 * rejects. */
static void freezeEveryLayerAndValidate(grammarFixture_t *f) {
    f->model[0]->config->conv1d->frozen = true;
    f->model[3]->config->conv1d->frozen = true;
    f->model[6]->config->conv1d->frozen = true;
    f->model[10]->config->linear->frozen = true;
    rematPlanValidateGrammar(&f->p->train, f->t, &MODEL_CHAIN(f->model, f->t->modelSize),
                             REMAT_MODE_TRAIN, NULL);
}

void testGrammarRejectsALossBackwardWithoutABackwardPhase(void) {
    grammarFixture_t f;
    buildHarLivenessFixture(&f);
    ASSERT_EXITS_WITH_OUTPUT(1, "step #13 (kind 2, layer 12) violates grammar rule 2",
                             freezeEveryLayerAndValidate(&f));
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
    rematPlanValidateGrammar(eval, f->t, &MODEL_CHAIN(f->model, f->t->modelSize), REMAT_MODE_EVAL,
                             NULL);
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
    rematPlanValidateGrammar(eval, f->t, &MODEL_CHAIN(f->model, f->t->modelSize), REMAT_MODE_EVAL,
                             NULL);
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

/* HAR TRAIN pool: FORWARD(j) at 2j (its input, then ACT j+1), LOSS_FORWARD at
 * 24, LOSS_BACKWARD at 25..26, then BACKWARD(10 - i) at 27 + 3i: dy, input, dx.
 * 60 operands. */
static void wrongGradientIntoB10(rematProgram_t *p) {
    p->operands[27].wire = 14;
}
static void metaInputOfALinear(rematProgram_t *p) {
    p->operands[28].mode = OP_MODE_META;
}
static void readInputOfAFlatten(rematProgram_t *p) {
    p->operands[31].mode = OP_MODE_READ;
}
static void dxIntoTheInputOfB0(rematProgram_t *p) {
    p->operands[59].wire = 23;
}
static void forward3OffItsNodeSlice(rematProgram_t *p) {
    p->nodeOps[3].forward = 8;
}
static void sliceOfB0PastThePool(rematProgram_t *p) {
    p->steps[24].operands = 58;
}
static void b10ReadsOneOperand(rematProgram_t *p) {
    p->steps[14].numReads = 1;
}

void testGrammarRejectsABackwardReadingTheWrongGradient(void) {
    ASSERT_GRAMMAR_EXIT(buildHarStoreAllFixture, wrongGradientIntoB10,
                        "step #14 (kind 3, layer 10) violates grammar rule 8: operand 0 is wire 14 "
                        "(READ), expected wire 13 (READ)");
}

void testGrammarRejectsAMetaReadOfAnInputTheBackwardReads(void) {
    ASSERT_GRAMMAR_EXIT(buildHarStoreAllFixture, metaInputOfALinear,
                        "step #14 (kind 3, layer 10) violates grammar rule 8: operand 1 is wire 10 "
                        "(META), expected wire 10 (READ)");
}

/* A READ the read-set denies would keep a dead input alive. */
void testGrammarRejectsAReadOfAnInputTheBackwardNeverReads(void) {
    ASSERT_GRAMMAR_EXIT(buildHarStoreAllFixture, readInputOfAFlatten,
                        "step #15 (kind 3, layer 9) violates grammar rule 8: operand 1 is wire 9 "
                        "(READ), expected wire 9 (META)");
}

void testGrammarRejectsADxForAnInputThatNeedsNoGradient(void) {
    ASSERT_GRAMMAR_EXIT(buildHarStoreAllFixture, dxIntoTheInputOfB0,
                        "step #24 (kind 3, layer 0) violates grammar rule 8: operand 2 is wire 23 "
                        "(WRITE), expected wire 65535 (WRITE)");
}

void testGrammarRejectsAStepOffItsNodesSlice(void) {
    ASSERT_GRAMMAR_EXIT(buildHarStoreAllFixture, forward3OffItsNodeSlice,
                        "step #3 (kind 0, layer 3) violates grammar rule 8: its operand slice is "
                        "not its node's (nodeOps)");
}

void testGrammarRejectsASliceOutsideThePool(void) {
    ASSERT_GRAMMAR_EXIT(buildHarStoreAllFixture, sliceOfB0PastThePool,
                        "step #24 (kind 3, layer 0) violates grammar rule 8: its operand slice "
                        "lies outside the pool");
}

void testGrammarRejectsAStepWithTheWrongOperandCounts(void) {
    ASSERT_GRAMMAR_EXIT(buildHarStoreAllFixture, b10ReadsOneOperand,
                        "step #14 (kind 3, layer 10) violates grammar rule 8: 1 reads and 1 "
                        "writes, expected 2 and 1");
}

/* The grammar re-derives from the model, so flags tampered in the table do
 * not move it: the table is the deriver's input, and trusting it would share
 * the deriver's errors. */
static void tamperTheTablesFlagsAndValidate(grammarFixture_t *f) {
    f->t->needsGrad[1] = 0u;
    f->t->runsBackward[0] = 0u;
    f->t->readsInput[9] = 1u;
    f->t->dxWire[2] = REMAT_NONE;
    f->t->hasBackward = false;
    rematPlanValidateGrammar(&f->p->train, f->t, &MODEL_CHAIN(f->model, f->n), REMAT_MODE_TRAIN,
                             NULL);
}

void testGrammarDerivesFromTheModelNotFromTheTablesFlags(void) {
    grammarFixture_t f;
    buildHarStoreAllFixture(&f);
    ASSERT_EXITS_WITH(0, tamperTheTablesFlagsAndValidate(&f));
    freeGrammarFixture(&f);
}

/* gradIdOf is a lookup the grammar confirms against the wire records: a table
 * that maps buffer 11 to GRAD 10 (wire 14), and a slice that follows it, still
 * meet the seed (wire 13), the record CrossEntropy's dy at BACKWARD(10) is. */
static void mapBuffer11ToGrad10AndValidate(grammarFixture_t *f) {
    f->t->gradIdOf[11] = 14;
    f->p->train.operands[27].wire = 14;
    rematPlanValidateGrammar(&f->p->train, f->t, &MODEL_CHAIN(f->model, f->n), REMAT_MODE_TRAIN,
                             NULL);
}

void testGrammarConfirmsTheTablesGradientLookupAgainstTheRecords(void) {
    grammarFixture_t f;
    buildHarStoreAllFixture(&f);
    ASSERT_EXITS_WITH_OUTPUT(1,
                             "step #14 (kind 3, layer 10) violates grammar rule 8: operand 0 is "
                             "wire 14 (READ), expected wire 13 (READ)",
                             mapBuffer11ToGrad10AndValidate(&f));
    freeGrammarFixture(&f);
}

/* ---- REFORWARD and the rule-6 range checks (hand-written programs) ---- */

/* A program in test-owned arrays: hand-written steps and ranges, with the
 * operands the deriver gives those steps, so the tests reach the step and
 * range rules past rule 8. */
#define HAND_MAX 16
#define HAND_OPS 64
typedef struct handProgram {
    layer_t *model[MLP3_N];
    size_t n;
    inputLike_t in;
    rematWireTable_t *t;
    rematStep_t steps[HAND_MAX];
    rematOperand_t operands[HAND_OPS];
    rematNodeOps_t nodeOps[MLP3_N + 1];
    rematRange_t ranges[HAND_MAX];
    rematProgram_t p;
} handProgram_t;

static void buildMlp3Hand(handProgram_t *h) {
    buildMlp3(h->model);
    h->n = MLP3_N;
    h->t = initTable(h->model, MLP3_N, MSE, makeInput(&h->in, (size_t[]){1, 2}, 2, &g_floatQ));
}

/* Linear(2->4) frozen, ReLU under MSE: nothing trains (deepest = n = 2). */
static void buildAllFrozenHand(handProgram_t *h) {
    h->model[0] = makeLinear(2, 4, true, 0);
    h->model[1] = makeRelu(&g_floatQ);
    h->n = 2;
    h->t = initTable(h->model, 2, MSE, makeInput(&h->in, (size_t[]){1, 2}, 2, &g_floatQ));
}

/* One Linear(2->2) under MSE: the step bound 2n + 2 + n(n+1)/2 is 5. */
static void buildSingleLinearHand(handProgram_t *h) {
    h->model[0] = makeLinear(2, 2, false, 0);
    h->n = 1;
    h->t = initTable(h->model, 1, MSE, makeInput(&h->in, (size_t[]){1, 2}, 2, &g_floatQ));
}

static void setHandProgram(handProgram_t *h, const rematStep_t *steps, size_t numSteps,
                           const rematRange_t *ranges, size_t numRanges) {
    TEST_ASSERT_TRUE(numSteps <= HAND_MAX && numRanges <= HAND_MAX);
    const rematPlanSpec_t spec = {
        .policy = REMAT_PLAN_SEQUENCE, .sequence = steps, .sequenceLen = numSteps};
    rematDeriveSize_t size = rematDeriveTrainSize(h->t, &spec);
    TEST_ASSERT_TRUE(size.numOperands <= HAND_OPS);
    h->p = (rematProgram_t){.numSteps = numSteps,
                            .steps = h->steps,
                            .numOperands = size.numOperands,
                            .operands = h->operands,
                            .nodeOps = h->nodeOps,
                            .numRanges = numRanges,
                            .ranges = h->ranges};
    rematDeriveTrain(h->t, &spec, &h->p);
    memcpy(h->ranges, ranges, numRanges * sizeof *ranges);
}

static void freeHandProgram(handProgram_t *h) {
    rematWireTableFree(h->t);
    freeModel(h->model, h->n);
}

static void validateHandTrain(handProgram_t *h) {
    rematPlanValidateGrammar(&h->p, h->t, &MODEL_CHAIN(h->model, h->t->modelSize), REMAT_MODE_TRAIN,
                             NULL);
}

static void validateHandEval(handProgram_t *h) {
    rematPlanValidateGrammar(&h->p, h->t, &MODEL_CHAIN(h->model, h->t->modelSize), REMAT_MODE_EVAL,
                             NULL);
}

/* code 0: the grammar accepts the program; code 1 + message: it exits. */
#define ASSERT_HAND_EXITS_WITH(code, message, build, steps, ranges, validate)                      \
    do {                                                                                           \
        handProgram_t _h;                                                                          \
        build(&_h);                                                                                \
        setHandProgram(&_h, steps, COUNT_OF(steps), ranges, COUNT_OF(ranges));                     \
        ASSERT_EXITS_WITH_OUTPUT(code, message, validate(&_h));                                    \
        freeHandProgram(&_h);                                                                      \
    } while (0)

void testGrammarAcceptsAReforwardRunInsideTheWindow(void) {
    const rematStep_t steps[] = MLP3_RECOMPUTE_STEPS;
    const rematRange_t ranges[] = MLP3_RECOMPUTE_RANGES;
    handProgram_t h;
    buildMlp3Hand(&h);
    TEST_ASSERT_EQUAL_UINT16(4, rematGradId(h.t, 3));
    TEST_ASSERT_EQUAL_UINT16(5, rematGradId(h.t, 2));
    TEST_ASSERT_EQUAL_UINT16(6, rematGradId(h.t, 1));
    freeHandProgram(&h);
    ASSERT_HAND_EXITS_WITH(0, "", buildMlp3Hand, steps, ranges, validateHandTrain);
}

/* F0 LOSS_FORWARD REFORWARD(0) LOSS_BACKWARD B0: exactly 5 steps, the bound. */
void testGrammarAcceptsAProgramAtTheStepBound(void) {
    const rematStep_t steps[] = {STEP_F(0), STEP_LF(1), STEP_RF(0), STEP_LB(1), STEP_B(0)};
    const rematRange_t ranges[] = {RANGE_OF(1, 0, 1), RANGE_OF(1, 2, 3), RANGE_OF(2, 3, 4)};
    ASSERT_HAND_EXITS_WITH(0, "", buildSingleLinearHand, steps, ranges, validateHandTrain);
}

void testGrammarRejectsMoreStepsThanTheBound(void) {
    const rematStep_t steps[] = {STEP_F(0),  STEP_LF(1), STEP_RF(0),
                                 STEP_RF(0), STEP_LB(1), STEP_B(0)};
    const rematRange_t ranges[] = {RANGE_OF(1, 0, 1), RANGE_OF(1, 2, 2), RANGE_OF(1, 3, 4),
                                   RANGE_OF(2, 4, 5)};
    ASSERT_HAND_EXITS_WITH(
        1, "the stream of 6 steps violates grammar rule 6: more steps than 2n + 2 + n(n+1)/2",
        buildSingleLinearHand, steps, ranges, validateHandTrain);
}

void testGrammarRejectsAReforwardBeforeLossForward(void) {
    rematStep_t steps[] = MLP3_RECOMPUTE_STEPS;
    const rematRange_t ranges[] = MLP3_RECOMPUTE_RANGES;
    steps[3] = (rematStep_t)STEP_RF(0); /* in LOSS_FORWARD's place */
    ASSERT_HAND_EXITS_WITH(1,
                           "step #3 (kind 4, layer 0) violates grammar rule 6: a REFORWARD "
                           "comes after LOSS_FORWARD, below a pending backward step",
                           buildMlp3Hand, steps, ranges, validateHandTrain);
}

/* After LOSS_BACKWARD the next pending backward is BACKWARD(2): a REFORWARD
 * of layer 2 would regenerate ACT 3, which no pending step reads. */
void testGrammarRejectsAReforwardAtOrAboveThePendingBackward(void) {
    rematStep_t steps[] = MLP3_RECOMPUTE_STEPS;
    const rematRange_t ranges[] = MLP3_RECOMPUTE_RANGES;
    steps[6] = (rematStep_t)STEP_RF(2);
    ASSERT_HAND_EXITS_WITH(1,
                           "step #6 (kind 4, layer 2) violates grammar rule 6: a REFORWARD "
                           "comes after LOSS_FORWARD, below a pending backward step",
                           buildMlp3Hand, steps, ranges, validateHandTrain);
}

/* After BACKWARD(0) at deepest == 0 the backward cursor is -1: the window
 * compares it signed, so nothing is pending. */
void testGrammarRejectsAReforwardAfterTheFinalBackwardAtDeepestZero(void) {
    const rematStep_t steps[] = {STEP_F(0),  STEP_F(1),  STEP_F(2),  STEP_LF(3),
                                 STEP_LB(3), STEP_RF(0), STEP_RF(1), STEP_B(2),
                                 STEP_B(1),  STEP_B(0),  STEP_RF(0)};
    const rematRange_t ranges[] = MLP3_RECOMPUTE_RANGES;
    ASSERT_HAND_EXITS_WITH(1,
                           "step #10 (kind 4, layer 0) violates grammar rule 6: a REFORWARD "
                           "comes after LOSS_FORWARD, below a pending backward step",
                           buildMlp3Hand, steps, ranges, validateHandTrain);
}

void testGrammarRejectsAReforwardWhenNothingTrains(void) {
    const rematStep_t steps[] = {STEP_F(0), STEP_F(1), STEP_LF(2), STEP_RF(0)};
    const rematRange_t ranges[] = {RANGE_OF(1, 0, 1), RANGE_OF(2, 1, 2)};
    ASSERT_HAND_EXITS_WITH(1,
                           "step #3 (kind 4, layer 0) violates grammar rule 6: a REFORWARD "
                           "comes after LOSS_FORWARD, below a pending backward step",
                           buildAllFrozenHand, steps, ranges, validateHandTrain);
}

void testGrammarRejectsAReforwardInAnEvalProgram(void) {
    const rematStep_t steps[] = {STEP_F(0), STEP_F(1), STEP_F(2), STEP_LF(3), STEP_RF(0)};
    const rematRange_t ranges[] = {RANGE_OF(1, 0, 1), RANGE_OF(2, 1, 2), RANGE_OF(3, 2, 3)};
    ASSERT_HAND_EXITS_WITH(
        1, "step #4 (kind 4, layer 0) violates grammar rule 6: no REFORWARD in an EVAL program",
        buildMlp3Hand, steps, ranges, validateHandEval);
}

/* REFORWARD(1) reads ACT 1 at step 6, after ACT 1's second range (shortened
 * to step 5) has closed. */
void testGrammarRejectsAReforwardReadingOutsideItsRange(void) {
    const rematStep_t steps[] = MLP3_RECOMPUTE_STEPS;
    rematRange_t ranges[] = MLP3_RECOMPUTE_RANGES;
    ranges[4].end = 5;
    ASSERT_HAND_EXITS_WITH(
        1,
        "step #6 (kind 4, layer 1) violates grammar rule 4: it reads wire 1 outside any open range",
        buildMlp3Hand, steps, ranges, validateHandTrain);
}

/* ACT 1's first range stretched over the recompute, its second range gone:
 * REFORWARD(0) writes into a range FORWARD(0) opened. */
void testGrammarRejectsAWriteIntoARangeOpenedEarlier(void) {
    const rematStep_t steps[] = MLP3_RECOMPUTE_STEPS;
    const rematRange_t ranges[] = {RANGE_OF(1, 0, 8), RANGE_OF(2, 1, 2), RANGE_OF(3, 2, 4),
                                   RANGE_OF(4, 4, 7), RANGE_OF(2, 6, 7), RANGE_OF(5, 7, 8),
                                   RANGE_OF(6, 8, 9)};
    ASSERT_HAND_EXITS_WITH(1,
                           "step #5 (kind 4, layer 0) violates grammar rule 6: it writes wire 1 "
                           "into a range opened earlier",
                           buildMlp3Hand, steps, ranges, validateHandTrain);
}

/* Two ranges of ACT 1 overlap at steps 5..6. */
void testGrammarRejectsOverlappingRangesOfOneWire(void) {
    const rematStep_t steps[] = MLP3_RECOMPUTE_STEPS;
    rematRange_t ranges[] = MLP3_RECOMPUTE_RANGES;
    ranges[0].end = 6;
    ASSERT_HAND_EXITS_WITH(1,
                           "step #5 (kind 4, layer 0) violates grammar rule 6: it writes wire 1 "
                           "into a range opened earlier",
                           buildMlp3Hand, steps, ranges, validateHandTrain);
}

/* REFORWARD(0) runs twice in a row: the first one's output (range #4) is
 * overwritten before anything reads it. */
void testGrammarRejectsAReforwardWhoseOutputIsNeverRead(void) {
    const rematStep_t steps[] = {STEP_F(0),  STEP_F(1),  STEP_F(2),  STEP_LF(3),
                                 STEP_LB(3), STEP_RF(0), STEP_RF(0), STEP_RF(1),
                                 STEP_B(2),  STEP_B(1),  STEP_B(0)};
    const rematRange_t ranges[] = {RANGE_OF(1, 0, 1), RANGE_OF(2, 1, 2), RANGE_OF(3, 2, 4),
                                   RANGE_OF(4, 4, 8), RANGE_OF(1, 5, 5), RANGE_OF(1, 6, 9),
                                   RANGE_OF(2, 7, 8), RANGE_OF(5, 8, 9), RANGE_OF(6, 9, 10)};
    ASSERT_HAND_EXITS_WITH(1,
                           "range #4 (wire 1, steps 5..5) violates grammar rule 6: a "
                           "REFORWARD's output is read inside its range",
                           buildMlp3Hand, steps, ranges, validateHandTrain);
}

void testGrammarRejectsARangeEndingPastTheStream(void) {
    const rematStep_t steps[] = MLP3_RECOMPUTE_STEPS;
    rematRange_t ranges[] = MLP3_RECOMPUTE_RANGES;
    ranges[7].end = 10;
    ASSERT_HAND_EXITS_WITH(
        1, "range #7 (wire 6, steps 8..10) violates grammar rule 6: ranges lie inside the stream",
        buildMlp3Hand, steps, ranges, validateHandTrain);
}

/* The walk opens ranges in array order, so the array must be begin-sorted. */
void testGrammarRejectsRangesOutOfBeginOrder(void) {
    const rematStep_t steps[] = MLP3_RECOMPUTE_STEPS;
    rematRange_t ranges[] = MLP3_RECOMPUTE_RANGES;
    rematRange_t first = ranges[0];
    ranges[0] = ranges[1];
    ranges[1] = first;
    ASSERT_HAND_EXITS_WITH(1,
                           "range #1 (wire 1, steps 0..1) violates grammar rule 6: ranges "
                           "sorted by begin, ties by wire id",
                           buildMlp3Hand, steps, ranges, validateHandTrain);
}

/* ACT 1's first range twice: the row would bind the wire twice at step 0. */
void testGrammarRejectsTwoRangesOfOneWireOpeningTogether(void) {
    const rematStep_t steps[] = MLP3_RECOMPUTE_STEPS;
    const rematRange_t ranges[] = {RANGE_OF(1, 0, 1), RANGE_OF(1, 0, 1), RANGE_OF(2, 1, 2),
                                   RANGE_OF(3, 2, 4), RANGE_OF(4, 4, 7), RANGE_OF(1, 5, 8),
                                   RANGE_OF(2, 6, 7), RANGE_OF(5, 7, 8), RANGE_OF(6, 8, 9)};
    ASSERT_HAND_EXITS_WITH(1,
                           "range #1 (wire 1, steps 0..1) violates grammar rule 6: ranges "
                           "sorted by begin, ties by wire id",
                           buildMlp3Hand, steps, ranges, validateHandTrain);
}

/* An extra range of ACT 1 at LOSS_FORWARD (step 3), which writes nothing. */
void testGrammarRejectsARangeThatOpensWithoutAWrite(void) {
    const rematStep_t steps[] = MLP3_RECOMPUTE_STEPS;
    const rematRange_t ranges[] = {RANGE_OF(1, 0, 1), RANGE_OF(2, 1, 2), RANGE_OF(3, 2, 4),
                                   RANGE_OF(1, 3, 3), RANGE_OF(4, 4, 7), RANGE_OF(1, 5, 8),
                                   RANGE_OF(2, 6, 7), RANGE_OF(5, 7, 8), RANGE_OF(6, 8, 9)};
    ASSERT_HAND_EXITS_WITH(1,
                           "range #3 (wire 1, steps 3..3) violates grammar rule 6: a range "
                           "opens where its wire is written",
                           buildMlp3Hand, steps, ranges, validateHandTrain);
}

/* ---- REMAT_PLAN_SEQUENCE ---- */

static rematPlanSpec_t sequenceSpec(const rematStep_t *steps, size_t len) {
    return (rematPlanSpec_t){.policy = REMAT_PLAN_SEQUENCE, .sequence = steps, .sequenceLen = len};
}

/* The most steps a test feeds back from a built plan (HAR, nothing kept: 3n + 2 = 38). */
#define STEP_NAMES_MAX 64u

/* A built program's steps as a caller's SEQUENCE: kind and layer only, since
 * sequence records carry no operands. names holds STEP_NAMES_MAX records. */
static rematPlanSpec_t namedSequenceSpec(const rematProgram_t *p, rematStep_t *names) {
    TEST_ASSERT_TRUE(p->numSteps <= STEP_NAMES_MAX);
    for (size_t s = 0; s < p->numSteps; s++) {
        names[s] = (rematStep_t){.kind = p->steps[s].kind, .layer = p->steps[s].layer};
    }
    return sequenceSpec(names, p->numSteps);
}

static void assertSameSteps(const rematStep_t *want, size_t numSteps, const rematProgram_t *got) {
    TEST_ASSERT_EQUAL_size_t(numSteps, got->numSteps);
    for (size_t s = 0; s < numSteps; s++) {
        TEST_ASSERT_EQUAL_UINT8(want[s].kind, got->steps[s].kind);
        TEST_ASSERT_EQUAL_UINT16(want[s].layer, got->steps[s].layer);
    }
}

static void assertSameRanges(const rematRange_t *want, size_t numRanges,
                             const rematProgram_t *got) {
    TEST_ASSERT_EQUAL_size_t(numRanges, got->numRanges);
    for (size_t r = 0; r < numRanges; r++) {
        TEST_ASSERT_EQUAL_UINT16(want[r].wire, got->ranges[r].wire);
        TEST_ASSERT_EQUAL_UINT16(want[r].begin, got->ranges[r].begin);
        TEST_ASSERT_EQUAL_UINT16(want[r].end, got->ranges[r].end);
    }
}

static rematWireTable_t *initMlp3Table(layer_t **model, inputLike_t *in) {
    buildMlp3(model);
    return initTable(model, MLP3_N, MSE, makeInput(in, (size_t[]){1, 2}, 2, &g_floatQ));
}

/* The ranges by hand (each write opens a range that ends at the last read of
 * its wire before the wire's next write): ACT 1 [0,1] [5,6] [8,9], ACT 2
 * [1,2] [6,7], ACT 3 [2,4], GRAD 3 [4,7], GRAD 2 [7,9], GRAD 1 [9,10]; in
 * begin order, with ACT j = wire j, GRAD 3 = 4, GRAD 2 = 5, GRAD 1 = 6. */
void testSequencePlanCopiesTheStepsAndScansTheRanges(void) {
    const rematStep_t steps[] = MLP3_RECOMPUTE_TWICE_STEPS;
    const rematRange_t want[] = {RANGE_OF(1, 0, 1), RANGE_OF(2, 1, 2), RANGE_OF(3, 2, 4),
                                 RANGE_OF(4, 4, 7), RANGE_OF(1, 5, 6), RANGE_OF(2, 6, 7),
                                 RANGE_OF(5, 7, 9), RANGE_OF(1, 8, 9), RANGE_OF(6, 9, 10)};
    layer_t *model[MLP3_N];
    inputLike_t in;
    rematWireTable_t *t = initMlp3Table(model, &in);
    rematPlanSpec_t spec = sequenceSpec(steps, COUNT_OF(steps));
    rematPlan_t *p = buildPlan(t, model, &spec);
    TEST_ASSERT_EQUAL_INT(REMAT_PLAN_SEQUENCE, p->policy);
    assertSameSteps(steps, COUNT_OF(steps), &p->train);
    assertSameRanges(want, COUNT_OF(want), &p->train);
    assertEveryStepsOperandsAreCoLive(&p->train, NULL, t, model);
    rematPlanFree(p);
    rematWireTableFree(t);
    freeModel(model, MLP3_N);
}

void testSequencePlanKeepsItsOwnCopyOfTheSteps(void) {
    rematStep_t steps[] = MLP3_RECOMPUTE_TWICE_STEPS;
    const rematStep_t untouched[] = MLP3_RECOMPUTE_TWICE_STEPS;
    layer_t *model[MLP3_N];
    inputLike_t in;
    rematWireTable_t *t = initMlp3Table(model, &in);
    rematPlanSpec_t spec = sequenceSpec(steps, COUNT_OF(steps));
    rematPlan_t *p = buildPlan(t, model, &spec);
    memset(steps, 0, sizeof steps);
    TEST_ASSERT_TRUE(p->train.steps != steps);
    assertSameSteps(untouched, COUNT_OF(untouched), &p->train);
    rematPlanFree(p);
    rematWireTableFree(t);
    freeModel(model, MLP3_N);
}

/* Evaluation never recomputes: a SEQUENCE plan's EVAL program is STORE_ALL's. */
void testSequencePlanEvalProgramIsTheStoreAllEvalProgram(void) {
    const rematStep_t steps[] = MLP3_RECOMPUTE_TWICE_STEPS;
    layer_t *model[MLP3_N];
    inputLike_t in;
    rematWireTable_t *t = initMlp3Table(model, &in);
    rematPlanSpec_t spec = sequenceSpec(steps, COUNT_OF(steps));
    rematPlan_t *seq = buildPlan(t, model, &spec);
    rematPlan_t *all = buildPlan(t, model, NULL);
    assertSameSteps(all->eval.steps, all->eval.numSteps, &seq->eval);
    assertSameRanges(all->eval.ranges, all->eval.numRanges, &seq->eval);
    TEST_ASSERT_EQUAL_size_t(all->eval.peakLiveBytes, seq->eval.peakLiveBytes);
    rematPlanFree(all);
    rematPlanFree(seq);
    rematWireTableFree(t);
    freeModel(model, MLP3_N);
}

/* An independent check of the scan against the closed-form generator: the
 * LIVENESS step list fed back as a SEQUENCE yields LIVENESS's ranges and
 * both of its peaks, with and without inPlaceDx. */
static void assertLivenessStepsAsASequenceMatchLiveness(layer_t **model, size_t n,
                                                        lossFuncType_t lt, const tensor_t *x) {
    rematWireTable_t *t = initTable(model, n, lt, x);
    for (int inPlaceDx = 0; inPlaceDx <= 1; inPlaceDx++) {
        const rematPlanSpec_t liveSpec = {.policy = REMAT_PLAN_LIVENESS, .inPlaceDx = inPlaceDx};
        rematPlan_t *live = buildPlan(t, model, &liveSpec);
        rematStep_t names[STEP_NAMES_MAX];
        rematPlanSpec_t spec = namedSequenceSpec(&live->train, names);
        spec.inPlaceDx = inPlaceDx;
        rematPlan_t *seq = buildPlan(t, model, &spec);
        assertSameRanges(live->train.ranges, live->train.numRanges, &seq->train);
        TEST_ASSERT_EQUAL_size_t(live->train.peakLiveBytes, seq->train.peakLiveBytes);
        TEST_ASSERT_EQUAL_size_t(rematProgramActPeakBytes(&live->train, t),
                                 rematProgramActPeakBytes(&seq->train, t));
        rematPlanFree(seq);
        rematPlanFree(live);
    }
    rematWireTableFree(t);
}

void testLivenessStepsAsASequenceGiveTheLivenessPlanOnTheZoo(void) {
    inputLike_t in;
    layer_t *har[HAR_N];
    buildHar(har, false);
    assertLivenessStepsAsASequenceMatchLiveness(har, HAR_N, CROSS_ENTROPY, makeHarInput(&in));
    freeModel(har, HAR_N);
    layer_t *finetune[HAR_N];
    buildHar(finetune, true);
    assertLivenessStepsAsASequenceMatchLiveness(finetune, HAR_N, CROSS_ENTROPY, makeHarInput(&in));
    freeModel(finetune, HAR_N);
    layer_t *mnist[MNIST_N];
    buildMnistCnn(mnist);
    assertLivenessStepsAsASequenceMatchLiveness(
        mnist, MNIST_N, CROSS_ENTROPY, makeInput(&in, (size_t[]){1, 1, 784}, 3, &g_floatQ));
    freeModel(mnist, MNIST_N);
}

/* The 50 random chains of the co-live property, under both losses. */
void testLivenessStepsAsASequenceGiveTheLivenessPlanOnRandomChains(void) {
    uint32_t state = 0x2545F491u;
    for (int trial = 0; trial < 50; trial++) {
        size_t n = 1u + nextRandom(&state) % 6u;
        layer_t *model[6];
        for (size_t i = 0; i < n; i++) {
            model[i] = randomRank2Layer(&state, i);
        }
        /* CE only over a last Softmax (the skipped layer must be it); the
         * draw is taken either way, so the chains stay those of before */
        lossFuncType_t lt =
            (nextRandom(&state) % 2u == 0u || model[n - 1]->type != SOFTMAX) ? MSE : CROSS_ENTROPY;
        inputLike_t in;
        assertLivenessStepsAsASequenceMatchLiveness(model, n, lt,
                                                    makeInput(&in, (size_t[]){1, 4}, 2, &g_floatQ));
        freeModel(model, n);
    }
}

static void planBuildExpectingAnExit(const rematWireTable_t *t, layer_t **model,
                                     const rematPlanSpec_t *spec) {
    rematPlan_t *p = NULL;
    g_memBeforeExit = memProfileCurrentBytes();
    (void)atexit(printReservedBeforeExit);
    (void)rematPlanBuild(&p, t, &MODEL_CHAIN(model, t->modelSize), spec);
}

/* A step list written for the full HAR model, used after its convolutions
 * were frozen (deepest moves from 0 to 10): BACKWARD(9) at step 15 is one
 * backward step too many. */
void testSequencePlanRejectsAStepListOfAnotherFreezeState(void) {
    grammarFixture_t f;
    buildHarLivenessFixture(&f);
    layer_t *frozen[HAR_N];
    buildHar(frozen, true);
    inputLike_t in;
    rematWireTable_t *t = initTable(frozen, HAR_N, CROSS_ENTROPY, makeHarInput(&in));
    rematStep_t names[STEP_NAMES_MAX];
    rematPlanSpec_t spec = namedSequenceSpec(&f.p->train, names);
    ASSERT_EXITS_WITH_OUTPUT(1,
                             "step #15 (kind 3, layer 9) violates grammar rule 3: BACKWARD "
                             "over the running nodes, descending, after LOSS_BACKWARD",
                             planBuildExpectingAnExit(t, frozen, &spec));
    rematWireTableFree(t);
    freeModel(frozen, HAR_N);
    freeGrammarFixture(&f);
}

/* The pre-copy checks exit before the plan block exists: nothing is reserved. */
static void assertExitsBeforeThePlanBlock(const rematWireTable_t *t, layer_t **model,
                                          const rematPlanSpec_t *spec, const char *message) {
    ASSERT_EXITS_WITH_OUTPUT(1, message, planBuildExpectingAnExit(t, model, spec));
    ASSERT_EXITS_WITH_OUTPUT(1, "reservedBeforeExit=0\n", planBuildExpectingAnExit(t, model, spec));
}

void testSequencePlanExitsOnANullSequenceBeforeReserving(void) {
    layer_t *model[MLP3_N];
    inputLike_t in;
    rematWireTable_t *t = initMlp3Table(model, &in);
    rematPlanSpec_t spec = sequenceSpec(NULL, 3);
    assertExitsBeforeThePlanBlock(t, model, &spec,
                                  "REMAT_PLAN_SEQUENCE needs a sequence, got NULL");
    rematWireTableFree(t);
    freeModel(model, MLP3_N);
}

void testSequencePlanExitsOnAnEmptySequenceBeforeReserving(void) {
    const rematStep_t steps[] = MLP3_RECOMPUTE_TWICE_STEPS;
    layer_t *model[MLP3_N];
    inputLike_t in;
    rematWireTable_t *t = initMlp3Table(model, &in);
    rematPlanSpec_t spec = sequenceSpec(steps, 0);
    assertExitsBeforeThePlanBlock(t, model, &spec,
                                  "REMAT_PLAN_SEQUENCE needs at least one step, got sequenceLen 0");
    rematWireTableFree(t);
    freeModel(model, MLP3_N);
}

/* A sequence record names a step by kind and layer; the plan derives its
 * operand slice, so a caller's non-zero count, reserved byte or offset exits. */
static void sequenceWithOperandsOn(size_t at, rematStep_t carried) {
    layer_t *model[1] = {makeLinear(2, 2, false, 0)};
    inputLike_t in;
    rematWireTable_t *t = initTable(model, 1, MSE, makeInput(&in, (size_t[]){1, 2}, 2, &g_floatQ));
    rematStep_t seq[4] = {{.kind = REMAT_STEP_FORWARD, .layer = 0},
                          {.kind = REMAT_STEP_LOSS_FORWARD, .layer = 1},
                          {.kind = REMAT_STEP_LOSS_BACKWARD, .layer = 1},
                          {.kind = REMAT_STEP_BACKWARD, .layer = 0}};
    seq[at] = carried;
    rematPlan_t *p;
    (void)rematPlanBuild(
        &p, t, &MODEL_CHAIN(model, 1),
        &(rematPlanSpec_t){.policy = REMAT_PLAN_SEQUENCE, .sequence = seq, .sequenceLen = 4});
}

static void lossBackwardCarryingOperands(void) {
    sequenceWithOperandsOn(
        2, (rematStep_t){.kind = REMAT_STEP_LOSS_BACKWARD, .layer = 1, .operands = 7});
}

static void forwardCarryingAReservedByte(void) {
    sequenceWithOperandsOn(0, (rematStep_t){.kind = REMAT_STEP_FORWARD, .layer = 0, .reserved = 1});
}

static void backwardCarryingCounts(void) {
    sequenceWithOperandsOn(
        3, (rematStep_t){.kind = REMAT_STEP_BACKWARD, .layer = 0, .numReads = 2, .numWrites = 1});
}

static void lossForwardCarryingAReadCount(void) {
    sequenceWithOperandsOn(
        1, (rematStep_t){.kind = REMAT_STEP_LOSS_FORWARD, .layer = 1, .numReads = 1});
}

static void backwardCarryingAWriteCount(void) {
    sequenceWithOperandsOn(3,
                           (rematStep_t){.kind = REMAT_STEP_BACKWARD, .layer = 0, .numWrites = 1});
}

void testSequencePlanExitsOnARecordCarryingOperands(void) {
    ASSERT_EXITS_WITH_OUTPUT(1,
                             "sequence records carry no operands (step #2: numReads 0, "
                             "numWrites 0, reserved 0, operands 7)",
                             lossBackwardCarryingOperands());
    ASSERT_EXITS_WITH_OUTPUT(1, "(step #0: numReads 0, numWrites 0, reserved 1, operands 0)",
                             forwardCarryingAReservedByte());
    ASSERT_EXITS_WITH_OUTPUT(1, "(step #3: numReads 2, numWrites 1, reserved 0, operands 0)",
                             backwardCarryingCounts());
    ASSERT_EXITS_WITH_OUTPUT(1, "(step #1: numReads 1, numWrites 0, reserved 0, operands 0)",
                             lossForwardCarryingAReadCount());
    ASSERT_EXITS_WITH_OUTPUT(1, "(step #3: numReads 0, numWrites 1, reserved 0, operands 0)",
                             backwardCarryingAWriteCount());
}

/* HAR: 2n + 2 + n(n+1)/2 = 24 + 2 + 78 = 104. The buffer holds exactly 104
 * records, so a length check that ran after the counting pass or the copy
 * would read past it (ASan builds report that read; elsewhere the plan block
 * would exist at the exit and reservedBeforeExit would not be 0). */
void testSequencePlanExitsOneStepAboveTheBoundBeforeReserving(void) {
    rematStep_t steps[104] = {{0}};
    grammarFixture_t f;
    buildHarStoreAllFixture(&f);
    rematPlanSpec_t spec = sequenceSpec(steps, 105);
    assertExitsBeforeThePlanBlock(f.t, f.model, &spec,
                                  "sequenceLen 105 is above the step bound 104 "
                                  "(min(UINT16_MAX, 2n + 2 + n(n+1)/2), n = 12)");
    freeGrammarFixture(&f);
}

/* At the bound the length check passes and the grammar judges the steps: 104
 * copies of FORWARD(0) fail rule 1 at the second one. */
void testSequencePlanAcceptsALengthAtTheBound(void) {
    rematStep_t steps[104] = {{0}};
    grammarFixture_t f;
    buildHarStoreAllFixture(&f);
    rematPlanSpec_t spec = sequenceSpec(steps, 104);
    ASSERT_EXITS_WITH_OUTPUT(1, "step #1 (kind 0, layer 0) violates grammar rule 1",
                             planBuildExpectingAnExit(f.t, f.model, &spec));
    freeGrammarFixture(&f);
}

/* From n = 360 on, 2n + 2 + n(n+1)/2 (65702 at 360) passes UINT16_MAX, which
 * then bounds the length: step indexes and range ids are uint16_t. */
#define LONG_CHAIN_N 360u
static rematStep_t g_uint16MaxSteps[UINT16_MAX];

void testSequencePlanExitsAboveUint16MaxStepsBeforeReserving(void) {
    layer_t *relu = makeRelu(&g_floatQ);
    layer_t *model[LONG_CHAIN_N];
    for (size_t i = 0; i < LONG_CHAIN_N; i++) {
        model[i] = relu;
    }
    inputLike_t in;
    rematWireTable_t *t =
        initTable(model, LONG_CHAIN_N, MSE, makeInput(&in, (size_t[]){1, 1}, 2, &g_floatQ));
    rematPlanSpec_t spec = sequenceSpec(g_uint16MaxSteps, (size_t)UINT16_MAX + 1u);
    assertExitsBeforeThePlanBlock(t, model, &spec,
                                  "sequenceLen 65536 is above the step bound 65535 "
                                  "(min(UINT16_MAX, 2n + 2 + n(n+1)/2), n = 360)");
    rematWireTableFree(t);
    freeReluLayer(relu);
}

/* 4 ReLUs, Linear(2->2), 13104 ReLUs under MSE: the ReLUs below the Linear
 * need no gradient and skip their BACKWARD, so the pool holds 2n + 1 + 2 + 3b
 * = 26218 + 3 + 39315 = 65536 operands (n = 13109, b = 13105): one past
 * UINT16_MAX, which is REMAT_NONE and must never start a slice. The build exits
 * before it reserves the plan block. */
#define LONG_RELU_CHAIN_N 13109u
#define LONG_RELU_CHAIN_LINEAR 4u
static layer_t *g_longReluChain[LONG_RELU_CHAIN_N];

void testPlanBuildExitsOnAnOperandPoolAboveUint16Offsets(void) {
    layer_t *relu = makeRelu(&g_floatQ);
    for (size_t i = 0; i < LONG_RELU_CHAIN_N; i++) {
        g_longReluChain[i] = relu;
    }
    g_longReluChain[LONG_RELU_CHAIN_LINEAR] = makeLinear(2, 2, false, LONG_RELU_CHAIN_LINEAR);
    inputLike_t in;
    rematWireTable_t *t = initTable(g_longReluChain, LONG_RELU_CHAIN_N, MSE,
                                    makeInput(&in, (size_t[]){1, 2}, 2, &g_floatQ));
    assertExitsBeforeThePlanBlock(t, g_longReluChain, NULL,
                                  "rematPlanBuild: the TRAIN program needs 65536 operands, above "
                                  "UINT16_MAX (operand offsets are uint16_t, and UINT16_MAX is "
                                  "REMAT_NONE)");
    rematWireTableFree(t);
    freeModel(&g_longReluChain[LONG_RELU_CHAIN_LINEAR], 1);
    freeReluLayer(relu);
}

/* Fields that belong to another policy exit instead of being ignored. */
void testPlanBuildExitsOnASequenceForAnotherPolicy(void) {
    const rematStep_t steps[] = MLP3_RECOMPUTE_TWICE_STEPS;
    layer_t *model[MLP3_N];
    inputLike_t in;
    rematWireTable_t *t = initMlp3Table(model, &in);
    rematPlanSpec_t spec = {.policy = REMAT_PLAN_LIVENESS, .sequence = steps};
    ASSERT_EXITS_WITH_OUTPUT(
        1, "sequence or sequenceLen is set, but the policy 1 is not REMAT_PLAN_SEQUENCE",
        planBuildExpectingAnExit(t, model, &spec));
    rematWireTableFree(t);
    freeModel(model, MLP3_N);
}

void testPlanBuildExitsOnASequenceLengthForAnotherPolicy(void) {
    layer_t *model[MLP3_N];
    inputLike_t in;
    rematWireTable_t *t = initMlp3Table(model, &in);
    rematPlanSpec_t spec = {.policy = REMAT_PLAN_STORE_ALL, .sequenceLen = 2};
    ASSERT_EXITS_WITH_OUTPUT(
        1, "sequence or sequenceLen is set, but the policy 0 is not REMAT_PLAN_SEQUENCE",
        planBuildExpectingAnExit(t, model, &spec));
    rematWireTableFree(t);
    freeModel(model, MLP3_N);
}

/* ---- BatchNorm1d under recompute: a forward that updates state is never recomputed ---- */

/* BatchNorm1d(2) -> Linear(2->2) under MSE on a [4, 2] input (the zoo's
 * BatchNorm shape). A plan reads a layer only through its type and config, so
 * the BatchNorm1d is a stack-built config without buffers: nothing runs it. */
typedef struct bnPlanFixture {
    batchNorm1dConfig_t bn;
    layerConfig_t bnConfig;
    layer_t bnLayer;
    layer_t *model[2];
    inputLike_t in;
    rematWireTable_t *t;
} bnPlanFixture_t;

static void buildBnPlanFixture(bnPlanFixture_t *f, bool tracked, bool frozen) {
    f->bn = (batchNorm1dConfig_t){.numChannels = 2,
                                  .affine = true,
                                  .trackRunningStats = tracked,
                                  .frozen = frozen,
                                  .outputQ = &g_floatQ,
                                  .propLossQ = &g_floatQ};
    f->bnConfig = (layerConfig_t){.batchNorm1d = &f->bn};
    f->bnLayer = (layer_t){.type = BATCHNORM1D, .config = &f->bnConfig};
    f->model[0] = &f->bnLayer;
    f->model[1] = makeLinear(2, 2, false, 1);
    f->t = initTable(f->model, 2, MSE, makeInput(&f->in, (size_t[]){4, 2}, 2, &g_floatQ));
}

static void freeBnPlanFixture(bnPlanFixture_t *f) {
    rematWireTableFree(f->t);
    freeLinearLayer(f->model[1]);
}

/* deepest = 0 and top = 1: BACKWARD(1) reads ACT 1, so REFORWARD(0)
 * regenerates it after LOSS_BACKWARD. */
#define BN_RECOMPUTE_STEPS                                                                         \
    {STEP_F(0), STEP_F(1), STEP_LF(2), STEP_LB(2), STEP_RF(0), STEP_B(1), STEP_B(0)}

/* A tracked BatchNorm1d's training forward moves its running statistics: a
 * second forward would move them again. The build decides from the config
 * alone, whatever the momentary training flag says. */
void testSequencePlanRejectsRecomputingATrackedBatchNorm(void) {
    const rematStep_t steps[] = BN_RECOMPUTE_STEPS;
    rematPlanSpec_t spec = sequenceSpec(steps, COUNT_OF(steps));
    const char *message =
        "step #4 (kind 4, layer 0) violates grammar rule 6: layer 0 updates its state in every "
        "training forward (a tracked, unfrozen BatchNorm1d), so it is never recomputed; keep "
        "ACT 1";
    bnPlanFixture_t f;
    buildBnPlanFixture(&f, true, false);
    ASSERT_EXITS_WITH_OUTPUT(1, message, planBuildExpectingAnExit(f.t, f.model, &spec));
    f.bn.training = true;
    ASSERT_EXITS_WITH_OUTPUT(1, message, planBuildExpectingAnExit(f.t, f.model, &spec));
    freeBnPlanFixture(&f);
}

/* An untracked BatchNorm1d always normalizes over the batch and keeps no
 * running statistics: its forward is a pure function of its input. */
void testSequencePlanRecomputesAnUntrackedBatchNorm(void) {
    const rematStep_t steps[] = BN_RECOMPUTE_STEPS;
    rematPlanSpec_t spec = sequenceSpec(steps, COUNT_OF(steps));
    bnPlanFixture_t f;
    buildBnPlanFixture(&f, false, false);
    rematPlan_t *p = buildPlan(f.t, f.model, &spec);
    assertSameSteps(steps, COUNT_OF(steps), &p->train);
    rematPlanFree(p);
    freeBnPlanFixture(&f);
}

/* A frozen BatchNorm1d normalizes with its running statistics and never
 * moves them. It does not train, so deepest = 1 and BACKWARD(1) is the last
 * backward step. */
void testSequencePlanRecomputesAFrozenTrackedBatchNorm(void) {
    const rematStep_t steps[] = {STEP_F(0),  STEP_F(1),  STEP_LF(2),
                                 STEP_LB(2), STEP_RF(0), STEP_B(1)};
    rematPlanSpec_t spec = sequenceSpec(steps, COUNT_OF(steps));
    bnPlanFixture_t f;
    buildBnPlanFixture(&f, true, true);
    rematPlan_t *p = buildPlan(f.t, f.model, &spec);
    assertSameSteps(steps, COUNT_OF(steps), &p->train);
    rematPlanFree(p);
    freeBnPlanFixture(&f);
}

/* ---- REMAT_PLAN_RETAIN_LIST ---- */

/* The largest model of the plan-level retain tests (HAR), plus ACT 0. */
#define RETAIN_MAX (HAR_N + 1)

static rematPlanSpec_t retainSpec(const bool *retain) {
    return (rematPlanSpec_t){.policy = REMAT_PLAN_RETAIN_LIST, .retain = retain};
}

/* MLP3 keeping ACT 3 only: BACKWARD(2) reads ACT 2 and BACKWARD(1) reads
 * ACT 1, so the run 0..1 from ACT 0 recomputes both before BACKWARD(2), and
 * BACKWARD(1) reads the recomputed ACT 1. That is the hand-written recompute
 * program, with its hand-derived ranges. */
void testRetainListPlanOnTheMlpIsTheRecomputeProgram(void) {
    const bool retain[MLP3_N + 1] = {false, false, false, true};
    const rematStep_t steps[] = MLP3_RECOMPUTE_STEPS;
    const rematRange_t ranges[] = MLP3_RECOMPUTE_RANGES;
    layer_t *model[MLP3_N];
    inputLike_t in;
    rematWireTable_t *t = initMlp3Table(model, &in);
    rematPlanSpec_t spec = retainSpec(retain);
    rematPlan_t *p = buildPlan(t, model, &spec);
    TEST_ASSERT_EQUAL_INT(REMAT_PLAN_RETAIN_LIST, p->policy);
    assertSameSteps(steps, COUNT_OF(steps), &p->train);
    assertSameRanges(ranges, COUNT_OF(ranges), &p->train);
    rematPlanFree(p);
    rematWireTableFree(t);
    freeModel(model, MLP3_N);
}

/* Evaluation never recomputes: a RETAIN_LIST plan's EVAL program is
 * STORE_ALL's. */
void testRetainListPlanEvalProgramIsTheStoreAllEvalProgram(void) {
    const bool retain[MLP3_N + 1] = {false, false, false, true};
    layer_t *model[MLP3_N];
    inputLike_t in;
    rematWireTable_t *t = initMlp3Table(model, &in);
    rematPlanSpec_t spec = retainSpec(retain);
    rematPlan_t *kept = buildPlan(t, model, &spec);
    rematPlan_t *all = buildPlan(t, model, NULL);
    assertSameSteps(all->eval.steps, all->eval.numSteps, &kept->eval);
    assertSameRanges(all->eval.ranges, all->eval.numRanges, &kept->eval);
    TEST_ASSERT_EQUAL_size_t(all->eval.peakLiveBytes, kept->eval.peakLiveBytes);
    rematPlanFree(all);
    rematPlanFree(kept);
    rematWireTableFree(t);
    freeModel(model, MLP3_N);
}

/* Every ACT kept: nothing is recomputed, the generated list is the LIVENESS
 * step list, and its scan gives LIVENESS's ranges and both peaks. */
static void assertKeepingEveryActIsLiveness(layer_t **model, size_t n, lossFuncType_t lt,
                                            const tensor_t *x) {
    bool retain[RETAIN_MAX];
    TEST_ASSERT_TRUE(n < RETAIN_MAX);
    for (size_t j = 0; j <= n; j++) {
        retain[j] = true;
    }
    rematWireTable_t *t = initTable(model, n, lt, x);
    for (int inPlaceDx = 0; inPlaceDx <= 1; inPlaceDx++) {
        const rematPlanSpec_t liveSpec = {.policy = REMAT_PLAN_LIVENESS, .inPlaceDx = inPlaceDx};
        rematPlan_t *live = buildPlan(t, model, &liveSpec);
        rematPlanSpec_t spec = retainSpec(retain);
        spec.inPlaceDx = inPlaceDx;
        rematPlan_t *kept = buildPlan(t, model, &spec);
        assertSameSteps(live->train.steps, live->train.numSteps, &kept->train);
        assertSameRanges(live->train.ranges, live->train.numRanges, &kept->train);
        TEST_ASSERT_EQUAL_size_t(live->train.peakLiveBytes, kept->train.peakLiveBytes);
        TEST_ASSERT_EQUAL_size_t(rematProgramActPeakBytes(&live->train, t),
                                 rematProgramActPeakBytes(&kept->train, t));
        rematPlanFree(kept);
        rematPlanFree(live);
    }
    rematWireTableFree(t);
}

void testRetainListPlanKeepingEveryActIsTheLivenessPlanOnTheZoo(void) {
    inputLike_t in;
    layer_t *har[HAR_N];
    buildHar(har, false);
    assertKeepingEveryActIsLiveness(har, HAR_N, CROSS_ENTROPY, makeHarInput(&in));
    freeModel(har, HAR_N);
    layer_t *finetune[HAR_N];
    buildHar(finetune, true);
    assertKeepingEveryActIsLiveness(finetune, HAR_N, CROSS_ENTROPY, makeHarInput(&in));
    freeModel(finetune, HAR_N);
    layer_t *mnist[MNIST_N];
    buildMnistCnn(mnist);
    assertKeepingEveryActIsLiveness(mnist, MNIST_N, CROSS_ENTROPY,
                                    makeInput(&in, (size_t[]){1, 1, 784}, 3, &g_floatQ));
    freeModel(mnist, MNIST_N);
}

/* The 50 random chains of the co-live property, under both losses. */
void testRetainListPlanKeepingEveryActIsTheLivenessPlanOnRandomChains(void) {
    uint32_t state = 0x2545F491u;
    for (int trial = 0; trial < 50; trial++) {
        size_t n = 1u + nextRandom(&state) % 6u;
        layer_t *model[6];
        for (size_t i = 0; i < n; i++) {
            model[i] = randomRank2Layer(&state, i);
        }
        /* CE only over a last Softmax (the skipped layer must be it); the
         * draw is taken either way, so the chains stay those of before */
        lossFuncType_t lt =
            (nextRandom(&state) % 2u == 0u || model[n - 1]->type != SOFTMAX) ? MSE : CROSS_ENTROPY;
        inputLike_t in;
        assertKeepingEveryActIsLiveness(model, n, lt,
                                        makeInput(&in, (size_t[]){1, 4}, 2, &g_floatQ));
        freeModel(model, n);
    }
}

static void assertRetainListPlanIsSound(layer_t **model, size_t n, lossFuncType_t lt,
                                        const tensor_t *x, const bool *retain) {
    rematWireTable_t *t = initTable(model, n, lt, x);
    rematPlanSpec_t spec = retainSpec(retain);
    rematPlan_t *kept = buildPlan(t, model, &spec);
    assertEveryStepsOperandsAreCoLive(&kept->train, NULL, t, model);
    for (size_t j = 1; j <= n; j++) {
        size_t rangesOfAct = 0;
        for (size_t r = 0; r < kept->train.numRanges; r++) {
            rangesOfAct += (kept->train.ranges[r].wire == rematActId(t, j)) ? 1u : 0u;
        }
        TEST_ASSERT_TRUE_MESSAGE(rangesOfAct <= 2u, "an ACT recomputed more than once");
        if (retain[j]) {
            TEST_ASSERT_EQUAL_size_t_MESSAGE(1, rangesOfAct, "a kept ACT was recomputed");
        }
    }
    rematStep_t names[STEP_NAMES_MAX];
    rematPlanSpec_t again = namedSequenceSpec(&kept->train, names);
    rematPlan_t *seq = buildPlan(t, model, &again);
    assertSameSteps(kept->train.steps, kept->train.numSteps, &seq->train);
    assertSameRanges(kept->train.ranges, kept->train.numRanges, &seq->train);
    TEST_ASSERT_EQUAL_MEMORY(kept->train.endOrder, seq->train.endOrder,
                             kept->train.numRanges * sizeof(uint16_t));
    TEST_ASSERT_EQUAL_size_t(kept->train.peakLiveBytes, seq->train.peakLiveBytes);
    TEST_ASSERT_EQUAL_size_t(rematProgramActPeakBytes(&kept->train, t),
                             rematProgramActPeakBytes(&seq->train, t));
    TEST_ASSERT_EQUAL_size_t(kept->refwdSteps, seq->refwdSteps);
    TEST_ASSERT_EQUAL_size_t(kept->forwardCost, seq->forwardCost);
    TEST_ASSERT_EQUAL_size_t(kept->recomputeCost, seq->recomputeCost);
    rematPlanFree(seq);
    rematPlanFree(kept);
    rematWireTableFree(t);
}

/* Three masks per model: nothing kept, every other ACT kept, only ACT n
 * kept. HAR, its frozen-convolution stage 2 and mnist_cnn all have a
 * backward step under the loss backward, so every mask is valid. */
void testRetainListPlansAreSoundOnTheZoo(void) {
    for (int mask = 0; mask < 3; mask++) {
        bool retain[RETAIN_MAX];
        for (size_t j = 0; j < RETAIN_MAX; j++) {
            retain[j] = (mask == 1) && (j % 2u == 0u);
        }
        inputLike_t in;
        layer_t *har[HAR_N];
        buildHar(har, false);
        retain[HAR_N] = retain[HAR_N] || mask == 2;
        assertRetainListPlanIsSound(har, HAR_N, CROSS_ENTROPY, makeHarInput(&in), retain);
        freeModel(har, HAR_N);
        layer_t *finetune[HAR_N];
        buildHar(finetune, true);
        assertRetainListPlanIsSound(finetune, HAR_N, CROSS_ENTROPY, makeHarInput(&in), retain);
        freeModel(finetune, HAR_N);
        layer_t *mnist[MNIST_N];
        buildMnistCnn(mnist);
        retain[MNIST_N] = retain[MNIST_N] || mask == 2;
        assertRetainListPlanIsSound(mnist, MNIST_N, CROSS_ENTROPY,
                                    makeInput(&in, (size_t[]){1, 1, 784}, 3, &g_floatQ), retain);
        freeModel(mnist, MNIST_N);
    }
}

/* A random mask for a model of n layers. ACT n stays kept where no
 * REFORWARD window exists (the loss backward runs and no BACKWARD step
 * does): that mask exits by design (its own test below). */
static void randomRetainMask(uint32_t *state, layer_t **model, size_t n, lossFuncType_t lt,
                             bool *retain) {
    for (size_t j = 0; j <= n; j++) {
        retain[j] = nextRandom(state) % 2u == 0u;
    }
    size_t deepest;
    ptrdiff_t top;
    positionalRange(model, n, lt, &deepest, &top);
    if (deepest < n && top < (ptrdiff_t)deepest) {
        retain[n] = true;
    }
}

/* The 50 random chains of the co-live property, each under a random mask. */
void testRetainListPlansAreSoundOnRandomChains(void) {
    uint32_t state = 0x2545F491u;
    for (int trial = 0; trial < 50; trial++) {
        size_t n = 1u + nextRandom(&state) % 6u;
        layer_t *model[6];
        for (size_t i = 0; i < n; i++) {
            model[i] = randomRank2Layer(&state, i);
        }
        /* CE only over a last Softmax (the skipped layer must be it); the
         * draw is taken either way, so the chains stay those of before */
        lossFuncType_t lt =
            (nextRandom(&state) % 2u == 0u || model[n - 1]->type != SOFTMAX) ? MSE : CROSS_ENTROPY;
        bool retain[7];
        randomRetainMask(&state, model, n, lt, retain);
        inputLike_t in;
        assertRetainListPlanIsSound(model, n, lt, makeInput(&in, (size_t[]){1, 4}, 2, &g_floatQ),
                                    retain);
        freeModel(model, n);
    }
}

void testRetainListPlanExitsOnANullMaskBeforeReserving(void) {
    layer_t *model[MLP3_N];
    inputLike_t in;
    rematWireTable_t *t = initMlp3Table(model, &in);
    rematPlanSpec_t spec = retainSpec(NULL);
    assertExitsBeforeThePlanBlock(t, model, &spec,
                                  "REMAT_PLAN_RETAIN_LIST needs a retain mask, got NULL");
    rematWireTableFree(t);
    freeModel(model, MLP3_N);
}

/* A mask with another policy exits instead of being ignored, and so do the
 * SEQUENCE fields with RETAIN_LIST. */
void testPlanBuildExitsOnARetainMaskForAnotherPolicy(void) {
    const bool retain[MLP3_N + 1] = {false, false, false, true};
    layer_t *model[MLP3_N];
    inputLike_t in;
    rematWireTable_t *t = initMlp3Table(model, &in);
    rematPlanSpec_t spec = {.policy = REMAT_PLAN_LIVENESS, .retain = retain};
    ASSERT_EXITS_WITH_OUTPUT(1, "retain is set, but the policy 1 is not REMAT_PLAN_RETAIN_LIST",
                             planBuildExpectingAnExit(t, model, &spec));
    rematWireTableFree(t);
    freeModel(model, MLP3_N);
}

void testRetainListPlanExitsOnASequenceForItsPolicy(void) {
    const bool retain[MLP3_N + 1] = {false, false, false, true};
    const rematStep_t steps[] = MLP3_RECOMPUTE_STEPS;
    layer_t *model[MLP3_N];
    inputLike_t in;
    rematWireTable_t *t = initMlp3Table(model, &in);
    rematPlanSpec_t spec = {.policy = REMAT_PLAN_RETAIN_LIST,
                            .sequence = steps,
                            .sequenceLen = COUNT_OF(steps),
                            .retain = retain};
    ASSERT_EXITS_WITH_OUTPUT(
        1, "sequence or sequenceLen is set, but the policy 3 is not REMAT_PLAN_SEQUENCE",
        planBuildExpectingAnExit(t, model, &spec));
    rematWireTableFree(t);
    freeModel(model, MLP3_N);
}

/* The BatchNorm1d plan fixture with ACT 2 kept: BACKWARD(1) reads the
 * dropped ACT 1, so the run would recompute layer 0, a tracked, unfrozen
 * BatchNorm1d. The plan exits by name and says which ACT to keep; it never
 * keeps ACT 1 on its own. */
void testRetainListPlanRejectsRecomputingATrackedBatchNorm(void) {
    const bool retain[3] = {false, false, true};
    rematPlanSpec_t spec = retainSpec(retain);
    bnPlanFixture_t f;
    buildBnPlanFixture(&f, true, false);
    ASSERT_EXITS_WITH_OUTPUT(
        1,
        "step #4 (kind 4, layer 0) violates grammar rule 6: layer 0 updates its state in every "
        "training forward (a tracked, unfrozen BatchNorm1d), so it is never recomputed; keep "
        "ACT 1",
        planBuildExpectingAnExit(f.t, f.model, &spec));
    freeBnPlanFixture(&f);
}

/* The same mask over an untracked BatchNorm1d recomputes it. */
void testRetainListPlanRecomputesAnUntrackedBatchNorm(void) {
    const bool retain[3] = {false, false, true};
    const rematStep_t steps[] = BN_RECOMPUTE_STEPS;
    rematPlanSpec_t spec = retainSpec(retain);
    bnPlanFixture_t f;
    buildBnPlanFixture(&f, false, false);
    rematPlan_t *p = buildPlan(f.t, f.model, &spec);
    assertSameSteps(steps, COUNT_OF(steps), &p->train);
    rematPlanFree(p);
    freeBnPlanFixture(&f);
}

#define RETAIN_CHAIN_MAX 21845u
static layer_t *g_retainChain[RETAIN_CHAIN_MAX];
static bool g_retainMask[RETAIN_CHAIN_MAX + 1u];

/* model[0] = linear, the rest the one shared relu; g_retainMask cleared. */
static rematWireTable_t *initRetainChain(size_t n, layer_t *linear, layer_t *relu,
                                         inputLike_t *in) {
    g_retainChain[0] = linear;
    for (size_t i = 1; i < n; i++) {
        g_retainChain[i] = relu;
    }
    memset(g_retainMask, 0, sizeof g_retainMask);
    return initTable(g_retainChain, n, MSE, makeInput(in, (size_t[]){1, 1}, 2, &g_floatQ));
}

void testRetainListPlanExitsAboveUint16MaxStepsBeforeReserving(void) {
    layer_t *linear = makeLinear(1, 1, false, 0);
    layer_t *relu = makeRelu(&g_floatQ);
    inputLike_t in;
    rematWireTable_t *t = initRetainChain(21845u, linear, relu, &in);
    rematPlanSpec_t spec = retainSpec(g_retainMask);
    assertExitsBeforeThePlanBlock(t, g_retainChain, &spec,
                                  "the RETAIN_LIST program has 65537 steps, above the step bound "
                                  "65535 (min(UINT16_MAX, 2n + 2 + n(n+1)/2), n = 21845)");
    rematWireTableFree(t);
    freeReluLayer(relu);
    freeLinearLayer(linear);
}

void testRetainListPlanAtTheStepBoundStopsAtTheRangeCap(void) {
    layer_t *linear = makeLinear(1, 1, false, 0);
    layer_t *relu = makeRelu(&g_floatQ);
    inputLike_t in;
    rematWireTable_t *t = initRetainChain(21844u, linear, relu, &in);
    rematPlanSpec_t spec = retainSpec(g_retainMask);
    assertExitsBeforeThePlanBlock(t, g_retainChain, &spec,
                                  "plan has 65532 ranges, above ODT_REMAT_MAX_RANGES (1024)");
    rematWireTableFree(t);
    freeReluLayer(relu);
    freeLinearLayer(linear);
}

void testRetainListPlanAcceptsExactlyMaxRangesAndExitsAboveThem(void) {
    layer_t *linear = makeLinear(1, 1, false, 0);
    layer_t *relu = makeRelu(&g_floatQ);
    inputLike_t in;
    rematWireTable_t *t = initRetainChain(511u, linear, relu, &in);
    for (size_t j = 1; j <= 509u; j++) {
        g_retainMask[j] = true;
    }
    rematPlanSpec_t spec = retainSpec(g_retainMask);
    rematPlan_t *p = buildPlan(t, g_retainChain, &spec);
    TEST_ASSERT_EQUAL_size_t(ODT_REMAT_MAX_RANGES, p->train.numRanges);
    rematPlanFree(p);
    g_retainMask[509] = false;
    assertExitsBeforeThePlanBlock(t, g_retainChain, &spec,
                                  "plan has 1025 ranges, above ODT_REMAT_MAX_RANGES (1024)");
    rematWireTableFree(t);
    freeReluLayer(relu);
    freeLinearLayer(linear);
}

/* HAR with ACT 3, ACT 7 and ACT 12 kept: retain[3] = retain[7] = retain[12] =
 * true, every other entry false (retain[0] is ignored). BACKWARD(10) reads the
 * dropped ACT 10: the run 7..9 from the kept ACT 7. BACKWARD(6) reads the
 * dropped ACT 6: the run 3..5 from the kept ACT 3, which also makes ACT 4
 * present for BACKWARD(4). BACKWARD(1) reads the dropped ACT 1: REFORWARD(0)
 * from the input. 7 REFORWARDs, 32 steps, 30 ranges (wire ids: ACT j = j,
 * GRAD 12 = 13, GRAD l = 24 - l for l <= 10). */
#define HAR_RETAIN_3_7_12_STEPS                                                                    \
    {STEP_F(0),  STEP_F(1),  STEP_F(2),  STEP_F(3),  STEP_F(4),  STEP_F(5),   STEP_F(6),           \
     STEP_F(7),  STEP_F(8),  STEP_F(9),  STEP_F(10), STEP_F(11), STEP_LF(12), STEP_LB(12),         \
     STEP_RF(7), STEP_RF(8), STEP_RF(9), STEP_B(10), STEP_B(9),  STEP_B(8),   STEP_B(7),           \
     STEP_RF(3), STEP_RF(4), STEP_RF(5), STEP_B(6),  STEP_B(5),  STEP_B(4),   STEP_B(3),           \
     STEP_B(2),  STEP_RF(0), STEP_B(1),  STEP_B(0)}
#define HAR_RETAIN_3_7_12_RANGES                                                                   \
    {RANGE_OF(1, 0, 1),    RANGE_OF(2, 1, 2),    RANGE_OF(3, 2, 27),   RANGE_OF(4, 3, 4),          \
     RANGE_OF(5, 4, 5),    RANGE_OF(6, 5, 6),    RANGE_OF(7, 6, 20),   RANGE_OF(8, 7, 8),          \
     RANGE_OF(9, 8, 9),    RANGE_OF(10, 9, 10),  RANGE_OF(11, 10, 11), RANGE_OF(12, 11, 13),       \
     RANGE_OF(13, 13, 17), RANGE_OF(8, 14, 15),  RANGE_OF(9, 15, 16),  RANGE_OF(10, 16, 17),       \
     RANGE_OF(14, 17, 18), RANGE_OF(15, 18, 19), RANGE_OF(16, 19, 20), RANGE_OF(17, 20, 24),       \
     RANGE_OF(4, 21, 26),  RANGE_OF(5, 22, 23),  RANGE_OF(6, 23, 24),  RANGE_OF(18, 24, 25),       \
     RANGE_OF(19, 25, 26), RANGE_OF(20, 26, 27), RANGE_OF(21, 27, 28), RANGE_OF(22, 28, 30),       \
     RANGE_OF(1, 29, 30),  RANGE_OF(23, 30, 31)}
/* The same with ACT 12 dropped too (retain[3] = retain[7] = true, every other
 * entry false): LOSS_BACKWARD reads ACT 12, so the run 7..11 sits between
 * LOSS_FORWARD and LOSS_BACKWARD and already covers ACT 10 for BACKWARD(10):
 * 9 REFORWARDs, 34 steps, 32 ranges. */
#define HAR_RETAIN_3_7_STEPS                                                                       \
    {STEP_F(0),  STEP_F(1),  STEP_F(2),   STEP_F(3),   STEP_F(4),   STEP_F(5),   STEP_F(6),        \
     STEP_F(7),  STEP_F(8),  STEP_F(9),   STEP_F(10),  STEP_F(11),  STEP_LF(12), STEP_RF(7),       \
     STEP_RF(8), STEP_RF(9), STEP_RF(10), STEP_RF(11), STEP_LB(12), STEP_B(10),  STEP_B(9),        \
     STEP_B(8),  STEP_B(7),  STEP_RF(3),  STEP_RF(4),  STEP_RF(5),  STEP_B(6),   STEP_B(5),        \
     STEP_B(4),  STEP_B(3),  STEP_B(2),   STEP_RF(0),  STEP_B(1),   STEP_B(0)}

static size_t reforwardsOf(const rematProgram_t *p) {
    size_t count = 0;
    for (size_t s = 0; s < p->numSteps; s++) {
        count += (p->steps[s].kind == REMAT_STEP_REFORWARD) ? 1u : 0u;
    }
    return count;
}

/* HAR's forward costs at B = 1 (layerForwardCost on each layer's element
 * counts): Conv1d 9->16 K7 2048 outputs * 9 * 7 = 129024; ReLU 2048; MaxPool
 * 1024 * 2 = 2048; Conv1d 16->32 K5 2048 * 16 * 5 = 163840; ReLU 2048; MaxPool
 * 2048; Conv1d 32->64 K3 2048 * 32 * 3 = 196608; ReLU 2048; AvgPool 64 * 32 =
 * 2048; Flatten 0; Linear 64->6 6 * 64 = 384; Softmax 6. Every plan runs each
 * forward once: 502150. With ACT 3, 7 and 12 kept the plan recomputes layers
 * 7, 8, 9, 3, 4, 5 and 0: 2048 + 2048 + 0 + 163840 + 2048 + 2048 + 129024 =
 * 301056, 0.5995 of the forwards; with ACT 12 dropped too, also layers 10 and
 * 11: 301446. The two sums:
 *   uv run python -c "print(sum([
 *       2048 * 9 * 7, 2048, 1024 * 2, 2048 * 16 * 5, 2048, 1024 * 2,
 *       2048 * 32 * 3, 2048, 64 * 32, 0, 6 * 64, 6]),
 *       2048 + 64 * 32 + 0 + 2048 * 16 * 5 + 2048 + 1024 * 2 + 2048 * 9 * 7)"
 */
/* The peak is at REFORWARD(5), step 23: ACT 3 4096 + GRAD 7 8192 + ACT 4 8192
 * + ACT 5 8192 + ACT 6 4096 = 32768 bytes; the same step without GRAD 7 is
 * the activation peak, 24576 (LIVENESS: 49152 and 41216). Derived by hand
 * from the steps above and by an independent scan model. */
void testRetainListHarKeepingActs3And7And12(void) {
    bool retain[HAR_N + 1] = {false};
    retain[3] = retain[7] = retain[12] = true;
    const rematStep_t steps[] = HAR_RETAIN_3_7_12_STEPS;
    const rematRange_t ranges[] = HAR_RETAIN_3_7_12_RANGES;
    layer_t *model[HAR_N];
    buildHar(model, false);
    inputLike_t in;
    rematWireTable_t *t = initTable(model, HAR_N, CROSS_ENTROPY, makeHarInput(&in));
    rematPlanSpec_t spec = retainSpec(retain);
    rematPlan_t *p = buildPlan(t, model, &spec);
    assertSameSteps(steps, COUNT_OF(steps), &p->train);
    assertSameRanges(ranges, COUNT_OF(ranges), &p->train);
    TEST_ASSERT_EQUAL_size_t(7, reforwardsOf(&p->train));
    TEST_ASSERT_EQUAL_size_t(7, p->refwdSteps);
    TEST_ASSERT_EQUAL_size_t(502150, p->forwardCost);
    TEST_ASSERT_EQUAL_size_t(301056, p->recomputeCost);
    TEST_ASSERT_EQUAL_size_t(32768, p->train.peakLiveBytes);
    TEST_ASSERT_EQUAL_size_t(24576, rematProgramActPeakBytes(&p->train, t));
    rematPlanFree(p);
    rematWireTableFree(t);
    freeModel(model, HAR_N);
}

/* ACT 12 is 24 bytes and dies before the run 3..5, so the peak stays at
 * REFORWARD(5), now step 25, over the same wires: 32768 and 24576. */
void testRetainListHarKeepingActs3And7(void) {
    bool retain[HAR_N + 1] = {false};
    retain[3] = retain[7] = true;
    const rematStep_t steps[] = HAR_RETAIN_3_7_STEPS;
    layer_t *model[HAR_N];
    buildHar(model, false);
    inputLike_t in;
    rematWireTable_t *t = initTable(model, HAR_N, CROSS_ENTROPY, makeHarInput(&in));
    rematPlanSpec_t spec = retainSpec(retain);
    rematPlan_t *p = buildPlan(t, model, &spec);
    assertSameSteps(steps, COUNT_OF(steps), &p->train);
    TEST_ASSERT_EQUAL_size_t(32, p->train.numRanges);
    TEST_ASSERT_EQUAL_size_t(9, reforwardsOf(&p->train));
    TEST_ASSERT_EQUAL_size_t(9, p->refwdSteps);
    TEST_ASSERT_EQUAL_size_t(502150, p->forwardCost);
    TEST_ASSERT_EQUAL_size_t(301446, p->recomputeCost);
    TEST_ASSERT_EQUAL_size_t(32768, p->train.peakLiveBytes);
    TEST_ASSERT_EQUAL_size_t(24576, rematProgramActPeakBytes(&p->train, t));
    rematPlanFree(p);
    rematWireTableFree(t);
    freeModel(model, HAR_N);
}

/* ---- the recompute numbers of the report ---- */

/* MLP3 on [1, 2]: Linear 2->4 costs 4 outputs * 2 = 8, ReLU 4, Linear 4->2
 * costs 2 * 4 = 8, so every plan's forward cost is 20. The recompute program
 * recomputes layers 0 and 1 (8 + 4 = 12), the recompute-twice program layer 0
 * twice and layer 1 once (8 + 4 + 8 = 20). */
static void assertMlp3Costs(const rematPlanSpec_t *spec, size_t refwdSteps, size_t recomputeCost) {
    layer_t *model[MLP3_N];
    inputLike_t in;
    rematWireTable_t *t = initMlp3Table(model, &in);
    rematPlan_t *p = buildPlan(t, model, spec);
    TEST_ASSERT_EQUAL_size_t(refwdSteps, p->refwdSteps);
    TEST_ASSERT_EQUAL_size_t(20, p->forwardCost);
    TEST_ASSERT_EQUAL_size_t(recomputeCost, p->recomputeCost);
    rematPlanFree(p);
    rematWireTableFree(t);
    freeModel(model, MLP3_N);
}

void testPlanCostsEveryForwardAndEveryRecomputeOfMlp3(void) {
    const rematStep_t twice[] = MLP3_RECOMPUTE_TWICE_STEPS;
    rematPlanSpec_t twiceSpec = sequenceSpec(twice, COUNT_OF(twice));
    assertMlp3Costs(NULL, 0, 0);
    assertMlp3Costs(&g_liveness, 0, 0);
    assertMlp3Costs(&g_mlp3Recompute, 2, 12);
    assertMlp3Costs(&twiceSpec, 3, 20);
}

/* The plan hands each layer its input's element count too: an adaptive pool
 * over [1, 3, 8] to 2 outputs per channel costs its 24 inputs, not its 6
 * outputs. Nothing trains, so the plan is the forward and the loss. */
void testPlanCostsAnAdaptivePoolOnItsInput(void) {
    adaptiveAvgPool1dConfig_t poolConfig = {
        .outputSize = 2, .outputQ = &g_floatQ, .propLossQ = &g_floatQ};
    layer_t pool = {.type = ADAPTIVE_AVGPOOL1D,
                    .config = &(layerConfig_t){.adaptiveAvgPool1d = &poolConfig}};
    layer_t *model[1] = {&pool};
    inputLike_t in;
    rematWireTable_t *t =
        initTable(model, 1, MSE, makeInput(&in, (size_t[]){1, 3, 8}, 3, &g_floatQ));
    rematPlan_t *p = buildPlan(t, model, NULL);
    TEST_ASSERT_EQUAL_size_t(24, p->forwardCost);
    rematPlanFree(p);
    rematWireTableFree(t);
}

/* In a DAG a node's input is its wired buffer, not the buffer before it: n0 and n1 both pool the
 * [1, 3, 8] input (b0) to [1, 3, 4] and n2 = ADD(b1, b2), so n1 costs its 24 inputs, not b1's 12,
 * and the plan costs 24 + 24 + 12. */
void testPlanCostsADagNodeOnItsWiredInput(void) {
    adaptiveAvgPool1dConfig_t poolConfig = {
        .outputSize = 4, .outputQ = &g_floatQ, .propLossQ = &g_floatQ};
    layer_t pool0 = {.type = ADAPTIVE_AVGPOOL1D,
                     .config = &(layerConfig_t){.adaptiveAvgPool1d = &poolConfig}};
    layer_t pool1 = pool0;
    layer_t *model[3] = {&pool0, &pool1, addLayerInit(NULL)};
    static const uint16_t start[4] = {0, 1, 2, 4}, ids[4] = {0, 0, 1, 2};
    const model_t m = {model, 3, start, ids};
    inputLike_t in;
    tensor_t *x = makeInput(&in, (size_t[]){1, 3, 8}, 3, &g_floatQ);
    rematWireTable_t *t = NULL;
    TEST_ASSERT_TRUE(rematWireTableInit(&t, &m, defaultLossConfig(MSE), x));
    rematPlan_t *p = NULL;
    TEST_ASSERT_TRUE(rematPlanBuild(&p, t, &m, NULL));
    TEST_ASSERT_EQUAL_size_t(60, p->forwardCost);
    rematPlanFree(p);
    rematWireTableFree(t);
    freeAddLayer(model[2]);
}

/* Layer 0's input width raised in its weights' shape, which only the cost
 * reads (the table and the grammar read the output width): with SIZE_MAX / 4,
 * Linear 2->4 costs 4 * (SIZE_MAX / 4) = SIZE_MAX - 3, and ReLU's 4 more
 * cannot be added. */
static void buildMlp3PlanWithLayer0Width(const rematPlanSpec_t *spec, size_t inFeatures) {
    layer_t *model[MLP3_N];
    inputLike_t in;
    rematWireTable_t *t = initMlp3Table(model, &in);
    model[0]->config->linear->weights->param->shape->dimensions[1] = inFeatures;
    rematPlan_t *p = NULL;
    (void)rematPlanBuild(&p, t, &MODEL_CHAIN(model, t->modelSize), spec);
}

void testPlanBuildExitsOnAForwardCostAboveSizeMax(void) {
    ASSERT_EXITS_WITH_OUTPUT(1, "rematPlanBuild: size overflow computing forwardCost",
                             buildMlp3PlanWithLayer0Width(NULL, SIZE_MAX / 4u));
}

/* With SIZE_MAX / 6, layer 0 costs about two thirds of SIZE_MAX: the forward
 * cost still fits, but the recompute-twice program recomputes layer 0 twice. */
void testPlanBuildExitsOnARecomputeCostAboveSizeMax(void) {
    const rematStep_t twice[] = MLP3_RECOMPUTE_TWICE_STEPS;
    rematPlanSpec_t spec = sequenceSpec(twice, COUNT_OF(twice));
    ASSERT_EXITS_WITH_OUTPUT(1, "rematPlanBuild: size overflow computing recomputeCost",
                             buildMlp3PlanWithLayer0Width(&spec, SIZE_MAX / 6u));
}

/* rematPlanBuild runs the grammar on what it generates: a table that claims
 * conv 3's BACKWARD does not read its input derives a META operand there,
 * while the grammar's own reading of the model has BACKWARD(3) (step 21) read
 * it. */
static void clearConv3ReadsInputAndBuild(grammarFixture_t *f) {
    f->t->readsInput[3] = 0;
    rematPlan_t *p = NULL;
    (void)rematPlanBuild(&p, f->t, &MODEL_CHAIN(f->model, f->t->modelSize), &g_liveness);
}

void testPlanBuildRunsTheGrammarOnWhatItGenerates(void) {
    grammarFixture_t f;
    buildHarLivenessFixture(&f);
    ASSERT_EXITS_WITH_OUTPUT(1,
                             "step #21 (kind 3, layer 3) violates grammar rule 8: operand 1 is "
                             "wire 3 (META), expected wire 3 (READ)",
                             clearConv3ReadsInputAndBuild(&f));
    freeGrammarFixture(&f);
}

/* ---- rematPlanBuild takes the table's model (PR1b) ---- */

static void buildLivenessPlanOn(grammarFixture_t *f, layer_t **model) {
    rematPlan_t *p = NULL;
    (void)rematPlanBuild(&p, f->t, &MODEL_CHAIN(model, f->t->modelSize), &g_liveness);
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
    other[10] = makeLinear(64, 6, true, 10);
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

/* ---- in-place dx ---- */

/* The live bytes at one step, summed over the ranges that cover it: a sum of
 * its own, apart from the plan's peak walk. */
static size_t liveBytesAt(const rematProgram_t *p, const rematWireTable_t *t, size_t step) {
    size_t live = 0;
    for (size_t r = 0; r < p->numRanges; r++) {
        if (p->ranges[r].begin <= step && step <= p->ranges[r].end) {
            live += rematWireBytes(t, p->ranges[r].wire);
        }
    }
    return live;
}

/* members: the marked wire ids, ascending; every other wire is unmarked. */
static void assertMarks(const rematPlan_t *p, const rematWireTable_t *t, const uint16_t *members,
                        size_t numMembers) {
    TEST_ASSERT_NOT_NULL(p->inPlace);
    size_t k = 0;
    for (uint16_t w = 0; w < t->numWires; w++) {
        bool member = k < numMembers && members[k] == w;
        TEST_ASSERT_EQUAL_UINT8_MESSAGE(member ? 1u : 0u, p->inPlace[w], "in-place mark");
        k += member ? 1u : 0u;
    }
    TEST_ASSERT_EQUAL_size_t(numMembers, k);
}

/* The in-place ranges are the plain plan's with every member's range dropped
 * and its chain root's newest range ending where the member's ended. */
#define FOLD_MAX 64
static void assertFoldedFrom(const rematProgram_t *plain, const uint8_t *inPlace,
                             const rematProgram_t *folded) {
    rematRange_t want[FOLD_MAX];
    size_t k = 0;
    TEST_ASSERT_TRUE(plain->numRanges <= FOLD_MAX);
    for (size_t r = 0; r < plain->numRanges; r++) {
        uint16_t root = plain->ranges[r].wire;
        while (inPlace[root] != 0u) {
            root--;
        }
        if (root == plain->ranges[r].wire) {
            want[k++] = plain->ranges[r];
            continue;
        }
        size_t j = k;
        while (j > 0u && want[j - 1u].wire != root) {
            j--;
        }
        TEST_ASSERT_TRUE_MESSAGE(j > 0u, "a member's range before its root's");
        want[j - 1u].end = plain->ranges[r].end;
    }
    assertSameRanges(want, k, folded);
}

/* The one range of a wire that has exactly one. */
static void assertOnlyRangeOf(const rematProgram_t *p, uint16_t wire, uint16_t begin,
                              uint16_t end) {
    size_t found = 0;
    for (size_t r = 0; r < p->numRanges; r++) {
        if (p->ranges[r].wire == wire) {
            TEST_ASSERT_EQUAL_UINT16(begin, p->ranges[r].begin);
            TEST_ASSERT_EQUAL_UINT16(end, p->ranges[r].end);
            found++;
        }
    }
    TEST_ASSERT_EQUAL_size_t(1, found);
}

/* HAR's pairs: the backwards of Flatten 9 and ReLU 7, 4, 1 keep their wire's
 * size; every other backward changes it. With GRAD 12 = wire 13 and GRAD l =
 * wire 24 - l (l <= 10), the members are wires 15, 17, 20, 23, and their
 * roots 14, 16, 19, 22 run on to the members' readers (19 ranges, not 23).
 * The peak stays at FORWARD(8) and BACKWARD(8), 41216 B (49152 without); the
 * steps between show which wires share: BACKWARD(9) (step 15) holds ACT 1
 * 8192 + ACT 3 4096 + ACT 4 8192 + ACT 6 4096 + ACT 7 8192 + GRAD 10 256 =
 * 33024; BACKWARD(7) (17) the same ACTs + GRAD 8 8192 = 40960 (49152 with
 * GRAD 7's own bytes); BACKWARD(4) (20) ACT 1 + ACT 3 + ACT 4 + GRAD 5 =
 * 28672; BACKWARD(1) (23) ACT 1 + GRAD 2 = 16384. Derived by hand from the
 * ranges and by an independent scan model. */
void testInPlaceLivenessHarPeakLiveBytesIs41216(void) {
    layer_t *model[HAR_N];
    buildHar(model, false);
    inputLike_t in;
    rematWireTable_t *t = initTable(model, HAR_N, CROSS_ENTROPY, makeHarInput(&in));
    rematPlan_t *plain = buildPlan(t, model, &g_liveness);
    rematPlan_t *p = buildPlan(t, model, &g_livenessInPlace);
    const uint16_t members[] = {15, 17, 20, 23};
    assertMarks(p, t, members, COUNT_OF(members));
    assertFoldedFrom(&plain->train, p->inPlace, &p->train);
    TEST_ASSERT_EQUAL_size_t(19, p->train.numRanges);
    assertOnlyRangeOf(&p->train, 14, 14, 16);
    assertOnlyRangeOf(&p->train, 16, 16, 18);
    assertOnlyRangeOf(&p->train, 19, 19, 21);
    assertOnlyRangeOf(&p->train, 22, 22, 24);
    TEST_ASSERT_EQUAL_size_t(41216, p->train.peakLiveBytes);
    TEST_ASSERT_EQUAL_size_t(49152, liveBytesAt(&plain->train, t, 17));
    TEST_ASSERT_EQUAL_size_t(33024, liveBytesAt(&p->train, t, 15));
    TEST_ASSERT_EQUAL_size_t(40960, liveBytesAt(&p->train, t, 17));
    TEST_ASSERT_EQUAL_size_t(28672, liveBytesAt(&p->train, t, 20));
    TEST_ASSERT_EQUAL_size_t(16384, liveBytesAt(&p->train, t, 23));
    rematPlanFree(p);
    rematPlanFree(plain);
    rematWireTableFree(t);
    freeModel(model, HAR_N);
}

/* GRAD ranges do not depend on the policy: STORE_ALL folds the same chains.
 * Its peak, 70192 B (74288 without), is at BACKWARD(6) (step 18): every ACT,
 * 57904, + GRAD 8 8192 (GRAD 7 shares it) + GRAD 6 4096. */
void testInPlaceStoreAllHarPeakLiveBytesIs70192(void) {
    layer_t *model[HAR_N];
    buildHar(model, false);
    inputLike_t in;
    rematWireTable_t *t = initTable(model, HAR_N, CROSS_ENTROPY, makeHarInput(&in));
    rematPlan_t *plain = buildPlan(t, model, NULL);
    const rematPlanSpec_t spec = {.policy = REMAT_PLAN_STORE_ALL, .inPlaceDx = true};
    rematPlan_t *p = buildPlan(t, model, &spec);
    const uint16_t members[] = {15, 17, 20, 23};
    assertMarks(p, t, members, COUNT_OF(members));
    assertFoldedFrom(&plain->train, p->inPlace, &p->train);
    TEST_ASSERT_EQUAL_size_t(19, p->train.numRanges);
    TEST_ASSERT_EQUAL_size_t(70192, p->train.peakLiveBytes);
    TEST_ASSERT_EQUAL_size_t(70192, liveBytesAt(&p->train, t, 18));
    TEST_ASSERT_EQUAL_size_t(57904, rematProgramActPeakBytes(&p->train, t));
    rematPlanFree(p);
    rematPlanFree(plain);
    rematWireTableFree(t);
    freeModel(model, HAR_N);
}

/* Linear(2->4) -> ReLU -> Softmax under CE: top = n - 2 = 1, so BACKWARD(1)
 * reads the seed GRAD 3 (wire 4, 16 B, the Softmax output's shape) and writes
 * GRAD 1 (wire 5, 16 B) into it. One range for both, from LOSS_BACKWARD (step
 * 4) to BACKWARD(0) (step 6), which reads GRAD 1: 4 ranges. */
void testInPlaceReluAtTheCrossEntropyTopSharesTheSeed(void) {
    layer_t *model[3] = {makeLinear(2, 4, false, 0), makeRelu(&g_floatQ), makeSoftmax()};
    inputLike_t in;
    rematWireTable_t *t =
        initTable(model, 3, CROSS_ENTROPY, makeInput(&in, (size_t[]){1, 2}, 2, &g_floatQ));
    rematPlan_t *p = buildPlan(t, model, &g_livenessInPlace);
    const uint16_t members[] = {5};
    assertMarks(p, t, members, COUNT_OF(members));
    TEST_ASSERT_EQUAL_UINT16(4, rematGradId(t, 3));
    TEST_ASSERT_EQUAL_size_t(4, p->train.numRanges);
    assertOnlyRangeOf(&p->train, 4, 4, 6);
    TEST_ASSERT_EQUAL_size_t(48, p->train.peakLiveBytes);
    rematPlanFree(p);
    rematWireTableFree(t);
    freeModel(model, 3);
}

/* Linear(2->4) -> ReLU -> Flatten -> Linear(4->2) under MSE: BACKWARD(2)
 * (Flatten) writes GRAD 2 (wire 7) into GRAD 3 (wire 6), and BACKWARD(1)
 * (ReLU) writes GRAD 1 (wire 8) into GRAD 2's bytes, which are GRAD 3's: one
 * range, wire 6 from BACKWARD(3) (step 6) to BACKWARD(0) (step 9); 6 ranges.
 * The peak, 56 B, is at step 6: ACT 1 16 + ACT 3 16 + the seed 8 + GRAD 3 16. */
void testInPlaceChainOfTwoMembersHasOneRange(void) {
    layer_t *model[4] = {makeLinear(2, 4, false, 0), makeRelu(&g_floatQ), flattenLayerInit(),
                         makeLinear(4, 2, false, 3)};
    inputLike_t in;
    rematWireTable_t *t = initTable(model, 4, MSE, makeInput(&in, (size_t[]){1, 2}, 2, &g_floatQ));
    rematPlan_t *p = buildPlan(t, model, &g_livenessInPlace);
    const uint16_t members[] = {7, 8};
    assertMarks(p, t, members, COUNT_OF(members));
    TEST_ASSERT_EQUAL_size_t(6, p->train.numRanges);
    assertOnlyRangeOf(&p->train, 6, 6, 9);
    TEST_ASSERT_EQUAL_size_t(56, p->train.peakLiveBytes);
    rematPlanFree(p);
    rematWireTableFree(t);
    freeModel(model, 4);
}

/* HAR LIVENESS with in-place dx: GRAD 9, 7, 4, 1 (wires 15, 17, 20, 23) are
 * members. The scan reads only the stored operands: a member's WRITE lands in
 * its root's range, which then ends at the member's reader. */
void testScanRangesLandsAnInPlaceMembersWriteInItsRootsRange(void) {
    layer_t *model[HAR_N];
    buildHar(model, false);
    inputLike_t in;
    rematWireTable_t *t = initTable(model, HAR_N, CROSS_ENTROPY, makeHarInput(&in));
    rematPlan_t *p = buildPlan(t, model, &g_livenessInPlace);
    rematRange_t again[32];
    rematScanRanges(t, &p->train, REMAT_PLAN_LIVENESS, p->inPlace, again);
    TEST_ASSERT_EQUAL_size_t(19, p->train.numRanges);
    TEST_ASSERT_EQUAL_MEMORY(p->train.ranges, again, 19 * sizeof(rematRange_t));
    const uint16_t members[4] = {15, 17, 20, 23};
    for (size_t r = 0; r < 19; r++) {
        for (size_t m = 0; m < 4; m++) {
            TEST_ASSERT_NOT_EQUAL(members[m], p->train.ranges[r].wire);
        }
    }
    assertOnlyRangeOf(&p->train, 14, 14, 16); /* GRAD 10, root of GRAD 9: B10 writes, B8 reads */
    assertOnlyRangeOf(&p->train, 16, 16, 18); /* GRAD 8, root of GRAD 7 */
    assertOnlyRangeOf(&p->train, 19, 19, 21); /* GRAD 5, root of GRAD 4 */
    assertOnlyRangeOf(&p->train, 22, 22, 24); /* GRAD 2, root of GRAD 1 */
    rematPlanFree(p);
    rematWireTableFree(t);
    freeModel(model, HAR_N);
}

/* Linear(2->4) -> middle -> Linear(4->2) under MSE, layer 2's dx stored as
 * dxQ (NULL: FLOAT32): the middle layer's BACKWARD writes GRAD 1 (wire 6) and
 * reads GRAD 2 (wire 5), both of 4 elements. Returns wire 6's mark; no other
 * wire may carry one. Frees the model. */
static uint8_t middleLayerMark(layer_t *middle, quantization_t *dxQ, size_t grad1Bytes) {
    layer_t *model[3] = {makeLinear(2, 4, false, 0), middle, makeLinear(4, 2, false, 2)};
    if (dxQ != NULL) {
        model[2]->config->linear->propLossQ = dxQ;
    }
    inputLike_t in;
    rematWireTable_t *t = initTable(model, 3, MSE, makeInput(&in, (size_t[]){1, 2}, 2, &g_floatQ));
    if (grad1Bytes != 0u) {
        t->wires[6].bytes = grad1Bytes; /* a record no real layer produces */
    }
    rematPlan_t *p = buildPlan(t, model, &g_livenessInPlace);
    TEST_ASSERT_EQUAL_size_t(7, t->numWires);
    for (uint16_t w = 0; w < 6; w++) {
        TEST_ASSERT_EQUAL_UINT8(0, p->inPlace[w]);
    }
    uint8_t mark = p->inPlace[6];
    rematPlanFree(p);
    rematWireTableFree(t);
    freeModel(model, 3);
    return mark;
}

/* The rows a type-, dtype- or size-blind rule would pair: each keeps four
 * elements on both wires. SYM_INT32 widths are not compared (a 14-bit dx over
 * a 12-bit loss pairs; table init rejects the narrower dx): each wire keeps
 * its own qConfig. Dropout's row is in the
 * driver's conformance suite, whose binary links the Dropout layer. */
void testInPlaceEligibilityTruthTable(void) {
    uint8_t exponent[1];
    bfpQConfig_t bfpQc;
    initBfpQConfigInto(8, 8, HALF_AWAY, exponent, &bfpQc);
    quantization_t bfpQ = {.type = BFP, .qConfig = &bfpQc};
    symInt32QConfig_t sym12Qc;
    symInt32QConfig_t sym14Qc;
    initSymInt32QConfigWithQMaxBits(HALF_AWAY, &sym12Qc, 12);
    initSymInt32QConfigWithQMaxBits(HALF_AWAY, &sym14Qc, 14);
    quantization_t sym12Q = {.type = SYM_INT32, .qConfig = &sym12Qc};
    quantization_t sym14Q = {.type = SYM_INT32, .qConfig = &sym14Qc};
    TEST_ASSERT_EQUAL_UINT8_MESSAGE(1, middleLayerMark(makeRelu(&g_floatQ), NULL, 0), "ReLU");
    TEST_ASSERT_EQUAL_UINT8_MESSAGE(1, middleLayerMark(flattenLayerInit(), NULL, 0), "Flatten");
    TEST_ASSERT_EQUAL_UINT8_MESSAGE(0, middleLayerMark(makeLinear(4, 4, false, 1), NULL, 0),
                                    "square Linear");
    TEST_ASSERT_EQUAL_UINT8_MESSAGE(0, middleLayerMark(makeLayerNorm(4, false), NULL, 0),
                                    "LayerNorm");
    TEST_ASSERT_EQUAL_UINT8_MESSAGE(0, middleLayerMark(makeSoftmax(), NULL, 0), "Softmax");
    TEST_ASSERT_EQUAL_UINT8_MESSAGE(0, middleLayerMark(makeQuant(&g_floatQ, &g_floatQ), NULL, 0),
                                    "FLOAT32 Quantization");
    TEST_ASSERT_EQUAL_UINT8_MESSAGE(1, middleLayerMark(makeRelu(&sym14Q), &sym12Q, 0),
                                    "SYM_INT32 14 over 12 bits");
    TEST_ASSERT_EQUAL_UINT8_MESSAGE(0, middleLayerMark(makeRelu(&bfpQ), &bfpQ, 0), "BFP");
    TEST_ASSERT_EQUAL_UINT8_MESSAGE(0, middleLayerMark(makeRelu(&sym12Q), NULL, 0),
                                    "SYM_INT32 under FLOAT32");
    TEST_ASSERT_EQUAL_UINT8_MESSAGE(0, middleLayerMark(makeRelu(&g_floatQ), NULL, 32),
                                    "other bytes");
}

/* inPlaceDx on a model without a pair changes nothing but the marks at the
 * block's end, all zero: the steps, ranges and peak stay those of the plan
 * without it, under STORE_ALL and LIVENESS. */
static void assertNoPairKeepsThePlan(layer_t **model, size_t n, lossFuncType_t lt,
                                     const tensor_t *x) {
    rematWireTable_t *t = initTable(model, n, lt, x);
    for (int policy = REMAT_PLAN_STORE_ALL; policy <= REMAT_PLAN_LIVENESS; policy++) {
        const rematPlanSpec_t offSpec = {.policy = (rematPlanPolicy_t)policy};
        const rematPlanSpec_t onSpec = {.policy = (rematPlanPolicy_t)policy, .inPlaceDx = true};
        rematPlan_t *off = buildPlan(t, model, &offSpec);
        rematPlan_t *on = buildPlan(t, model, &onSpec);
        TEST_ASSERT_NULL(off->inPlace);
        assertMarks(on, t, NULL, 0);
        assertSameSteps(off->train.steps, off->train.numSteps, &on->train);
        assertSameRanges(off->train.ranges, off->train.numRanges, &on->train);
        TEST_ASSERT_EQUAL_size_t(off->train.peakLiveBytes, on->train.peakLiveBytes);
        TEST_ASSERT_EQUAL_size_t(off->blockBytes + t->numWires, on->blockBytes);
        rematPlanFree(on);
        rematPlanFree(off);
    }
    rematWireTableFree(t);
}

/* A ReLU below the deepest trainable layer has no backward step, HAR's
 * stage-2 backbone trains only its Linear, and BFP wires never pair. */
void testInPlaceDxWithoutAPairKeepsThePlan(void) {
    inputLike_t in;
    layer_t *belowDeepest[3] = {makeLinear(2, 4, true, 0), makeRelu(&g_floatQ),
                                makeLinear(4, 2, false, 2)};
    assertNoPairKeepsThePlan(belowDeepest, 3, MSE, makeInput(&in, (size_t[]){1, 2}, 2, &g_floatQ));
    freeModel(belowDeepest, 3);
    layer_t *finetune[HAR_N];
    buildHar(finetune, true);
    assertNoPairKeepsThePlan(finetune, HAR_N, CROSS_ENTROPY, makeHarInput(&in));
    freeModel(finetune, HAR_N);
    uint8_t exponent[1];
    bfpQConfig_t bfpQc;
    initBfpQConfigInto(8, 8, HALF_AWAY, exponent, &bfpQc);
    quantization_t bfpQ = {.type = BFP, .qConfig = &bfpQc};
    layer_t *bfp[3] = {makeLinear(2, 4, false, 0), makeRelu(&bfpQ), makeLinear(4, 2, false, 2)};
    bfp[2]->config->linear->propLossQ = &bfpQ;
    assertNoPairKeepsThePlan(bfp, 3, MSE, makeInput(&in, (size_t[]){1, 2}, 2, &g_floatQ));
    freeModel(bfp, 3);
}

/* HAR_RETAIN_3_7_12_RANGES folded: wires 15, 17, 20, 23 lose their ranges,
 * and 14, 16, 19, 22 end where those ended. */
#define HAR_RETAIN_3_7_12_IN_PLACE_RANGES                                                          \
    {RANGE_OF(1, 0, 1),    RANGE_OF(2, 1, 2),    RANGE_OF(3, 2, 27),   RANGE_OF(4, 3, 4),          \
     RANGE_OF(5, 4, 5),    RANGE_OF(6, 5, 6),    RANGE_OF(7, 6, 20),   RANGE_OF(8, 7, 8),          \
     RANGE_OF(9, 8, 9),    RANGE_OF(10, 9, 10),  RANGE_OF(11, 10, 11), RANGE_OF(12, 11, 13),       \
     RANGE_OF(13, 13, 17), RANGE_OF(8, 14, 15),  RANGE_OF(9, 15, 16),  RANGE_OF(10, 16, 17),       \
     RANGE_OF(14, 17, 19), RANGE_OF(16, 19, 24), RANGE_OF(4, 21, 26),  RANGE_OF(5, 22, 23),        \
     RANGE_OF(6, 23, 24),  RANGE_OF(18, 24, 25), RANGE_OF(19, 25, 27), RANGE_OF(21, 27, 28),       \
     RANGE_OF(22, 28, 31), RANGE_OF(1, 29, 30)}

/* The scan folds the chains over a recompute program: BACKWARD(7) (step 20)
 * writes GRAD 7 into GRAD 8 (wire 16) and opens no range; BACKWARD(6) reads
 * GRAD 7 only after the REFORWARD run 3..5 (steps 21..23), so the root's
 * range runs over the run to step 24. 26 ranges instead of 30. The peak stays
 * at REFORWARD(5) (step 23), 32768 B: ACT 3 4096 + ACT 4 8192 + ACT 5 8192 +
 * ACT 6 4096 + GRAD 8 8192, whose bytes GRAD 7 holds there; without the pair
 * GRAD 7 had its own 8192 and GRAD 8 had ended. Derived by hand from the
 * steps and by an independent scan model. */
void testInPlaceRetainListHarRunsARootOverAReforwardRun(void) {
    bool retain[HAR_N + 1] = {false};
    retain[3] = retain[7] = retain[12] = true;
    const rematStep_t steps[] = HAR_RETAIN_3_7_12_STEPS;
    const rematRange_t ranges[] = HAR_RETAIN_3_7_12_IN_PLACE_RANGES;
    layer_t *model[HAR_N];
    buildHar(model, false);
    inputLike_t in;
    rematWireTable_t *t = initTable(model, HAR_N, CROSS_ENTROPY, makeHarInput(&in));
    rematPlanSpec_t spec = retainSpec(retain);
    spec.inPlaceDx = true;
    rematPlan_t *p = buildPlan(t, model, &spec);
    const uint16_t members[] = {15, 17, 20, 23};
    assertMarks(p, t, members, COUNT_OF(members));
    assertSameSteps(steps, COUNT_OF(steps), &p->train);
    assertSameRanges(ranges, COUNT_OF(ranges), &p->train);
    TEST_ASSERT_EQUAL_size_t(32768, p->train.peakLiveBytes);
    TEST_ASSERT_EQUAL_size_t(24576, rematProgramActPeakBytes(&p->train, t));
    rematPlanFree(p);
    rematWireTableFree(t);
    freeModel(model, HAR_N);
}

/* MLP3's LIVENESS program with GRAD 1 (wire 6) in place over GRAD 2 (wire
 * 5): GRAD 2's range runs from BACKWARD(2) (step 5) to BACKWARD(0) (step 7),
 * which reads GRAD 1; no range of wire 6. */
#define MLP3_LIVENESS_STEPS                                                                        \
    {STEP_F(0), STEP_F(1), STEP_F(2), STEP_LF(3), STEP_LB(3), STEP_B(2), STEP_B(1), STEP_B(0)}
#define MLP3_IN_PLACE_RANGES                                                                       \
    {RANGE_OF(1, 0, 6), RANGE_OF(2, 1, 5), RANGE_OF(3, 2, 4), RANGE_OF(4, 4, 5), RANGE_OF(5, 5, 7)}

/* The grammar with one mark, on `marked`, of MLP3's seven wires. */
static void validateHandMarking(handProgram_t *h, uint16_t marked) {
    uint8_t marks[7] = {0};
    TEST_ASSERT_EQUAL_size_t(7, h->t->numWires);
    marks[marked] = 1u;
    rematPlanValidateGrammar(&h->p, h->t, &MODEL_CHAIN(h->model, h->t->modelSize), REMAT_MODE_TRAIN,
                             marks);
}

static void validateHandMarkingGrad1(handProgram_t *h) {
    validateHandMarking(h, 6);
}

static void validateHandMarkingTheSeed(handProgram_t *h) {
    validateHandMarking(h, 4);
}

static void validateHandMarkingAct2(handProgram_t *h) {
    validateHandMarking(h, 2);
}

/* Linear(2->4) -> Linear(4->4) -> Linear(4->2) under MSE: MLP3's wires, but
 * layer 1 never aliases, though its two dx wires are equal in size. */
static void buildSquareLinearHand(handProgram_t *h) {
    buildMlp3Hand(h);
    freeReluLayer(h->model[1]);
    h->model[1] = makeLinear(4, 4, false, 1);
}

/* Each of these edits a table record after MLP3's table is built: the
 * records swap GRAD 3 and GRAD 2, so BACKWARD(1) reads wire 4 and the marked
 * GRAD 1 (wire 6) no longer follows its gradient -- while the raw gradIdOf
 * lookup still says wire 5, so only a check that reads the records catches
 * it -- or GRAD 1's record changes its size or dtype. */
static void buildMlp3HandReadingTheSeedAtBackward1(handProgram_t *h) {
    buildMlp3Hand(h);
    h->t->wires[4].index = 2;
    h->t->wires[5].index = 3;
}

static void buildMlp3HandWithBfpDxWires(handProgram_t *h) {
    buildMlp3Hand(h);
    h->t->wires[5].dtype = BFP;
    h->t->wires[6].dtype = BFP;
}

static void buildMlp3HandWithAWiderGrad1(handProgram_t *h) {
    buildMlp3Hand(h);
    h->t->wires[6].bytes = 32;
}

static void buildMlp3HandWithASymInt32Grad1(handProgram_t *h) {
    buildMlp3Hand(h);
    h->t->wires[6].dtype = SYM_INT32;
}

static void buildMlp3HandWithAsymDxWires(handProgram_t *h) {
    buildMlp3Hand(h);
    h->t->wires[5].dtype = ASYM;
    h->t->wires[6].dtype = ASYM;
}

void testGrammarAcceptsAnInPlaceMemberInsideItsRootsRange(void) {
    const rematStep_t steps[] = MLP3_LIVENESS_STEPS;
    const rematRange_t ranges[] = MLP3_IN_PLACE_RANGES;
    ASSERT_HAND_EXITS_WITH(0, "", buildMlp3Hand, steps, ranges, validateHandMarkingGrad1);
}

/* The seed's wire - 1 is ACT n; ACT 2 is no dx wire at all. */
void testGrammarRejectsAMarkOnAWireThatIsNotALayersDx(void) {
    const rematStep_t steps[] = MLP3_LIVENESS_STEPS;
    const rematRange_t ranges[] = MLP3_IN_PLACE_RANGES;
    ASSERT_HAND_EXITS_WITH(1,
                           "rematPlanBuild: wire 4 carries an in-place mark, but it is not a "
                           "layer's dx wire (GRAD l, l < n)",
                           buildMlp3Hand, steps, ranges, validateHandMarkingTheSeed);
    ASSERT_HAND_EXITS_WITH(1,
                           "rematPlanBuild: wire 2 carries an in-place mark, but it is not a "
                           "layer's dx wire (GRAD l, l < n)",
                           buildMlp3Hand, steps, ranges, validateHandMarkingAct2);
}

void testGrammarRejectsAMarkOnALayerThatNeverAliases(void) {
    const rematStep_t steps[] = MLP3_LIVENESS_STEPS;
    const rematRange_t ranges[] = MLP3_IN_PLACE_RANGES;
    ASSERT_HAND_EXITS_WITH(1,
                           "rematPlanBuild: wire 6 carries an in-place mark, but layer 1 is not a "
                           "ReLU, Dropout or Flatten",
                           buildSquareLinearHand, steps, ranges, validateHandMarkingGrad1);
}

void testGrammarRejectsAMarkWhoseGradientIsNotTheWireBefore(void) {
    const rematStep_t steps[] = MLP3_LIVENESS_STEPS;
    const rematRange_t ranges[] = MLP3_IN_PLACE_RANGES;
    ASSERT_HAND_EXITS_WITH(1,
                           "rematPlanBuild: wire 6 carries an in-place mark, but BACKWARD(1) "
                           "reads wire 4, not wire 5",
                           buildMlp3HandReadingTheSeedAtBackward1, steps, ranges,
                           validateHandMarkingGrad1);
}

void testGrammarRejectsAMarkOnBfpWires(void) {
    const rematStep_t steps[] = MLP3_LIVENESS_STEPS;
    const rematRange_t ranges[] = MLP3_IN_PLACE_RANGES;
    ASSERT_HAND_EXITS_WITH(
        1, "rematPlanBuild: wire 6 carries an in-place mark, but BFP wires never alias",
        buildMlp3HandWithBfpDxWires, steps, ranges, validateHandMarkingGrad1);
}

void testGrammarRejectsAMarkOnWiresOfAnotherSizeOrDtype(void) {
    const rematStep_t steps[] = MLP3_LIVENESS_STEPS;
    const rematRange_t ranges[] = MLP3_IN_PLACE_RANGES;
    const char *message = "rematPlanBuild: wire 6 carries an in-place mark, but wires 5 and 6 "
                          "differ in dtype or bytes, or are neither FLOAT32 nor SYM_INT32";
    ASSERT_HAND_EXITS_WITH(1, message, buildMlp3HandWithAWiderGrad1, steps, ranges,
                           validateHandMarkingGrad1);
    ASSERT_HAND_EXITS_WITH(1, message, buildMlp3HandWithASymInt32Grad1, steps, ranges,
                           validateHandMarkingGrad1);
    ASSERT_HAND_EXITS_WITH(1, message, buildMlp3HandWithAsymDxWires, steps, ranges,
                           validateHandMarkingGrad1);
}

void testGrammarRejectsARangeOnAnInPlaceMember(void) {
    const rematStep_t steps[] = MLP3_LIVENESS_STEPS;
    const rematRange_t ranges[] = {RANGE_OF(1, 0, 6), RANGE_OF(2, 1, 5), RANGE_OF(3, 2, 4),
                                   RANGE_OF(4, 4, 5), RANGE_OF(5, 5, 7), RANGE_OF(6, 6, 7)};
    ASSERT_HAND_EXITS_WITH(1,
                           "range #5 (wire 6, steps 6..7) violates grammar rule 6: no range "
                           "names an in-place member",
                           buildMlp3Hand, steps, ranges, validateHandMarkingGrad1);
}

/* Wire 7 is the first id past MLP3's seven wires: the marks hold one byte
 * per wire, so the range check bounds the id before it reads a mark. */
void testGrammarRejectsARangeOnAWireOutsideTheTable(void) {
    const rematStep_t steps[] = MLP3_LIVENESS_STEPS;
    const rematRange_t ranges[] = {RANGE_OF(1, 0, 6), RANGE_OF(2, 1, 5), RANGE_OF(3, 2, 4),
                                   RANGE_OF(4, 4, 5), RANGE_OF(5, 5, 7), RANGE_OF(7, 7, 7)};
    ASSERT_HAND_EXITS_WITH(1,
                           "range #5 (wire 7, steps 7..7) violates grammar rule 6: ranges name "
                           "a wire of the table",
                           buildMlp3Hand, steps, ranges, validateHandMarkingGrad1);
}

/* GRAD 2's range ends at BACKWARD(1) (step 6), before BACKWARD(0) reads
 * GRAD 1 in GRAD 2's bytes. */
void testGrammarRejectsAMemberReadOutsideItsRootsRange(void) {
    const rematStep_t steps[] = MLP3_LIVENESS_STEPS;
    const rematRange_t ranges[] = {RANGE_OF(1, 0, 6), RANGE_OF(2, 1, 5), RANGE_OF(3, 2, 4),
                                   RANGE_OF(4, 4, 5), RANGE_OF(5, 5, 6)};
    ASSERT_HAND_EXITS_WITH(1,
                           "step #7 (kind 3, layer 0) violates grammar rule 4: it reads wire 6 "
                           "outside any open range",
                           buildMlp3Hand, steps, ranges, validateHandMarkingGrad1);
}

/* ---- rematPlanBuild on DAGs ---- */

/* A DAG fixture's table under MSE, the plan a policy builds from it, and the fixture both point
 * into. Built in place. */
typedef struct dagPlan {
    dagFixture_t f;
    dagTensor_t xs;
    rematWireTable_t *t;
    rematPlan_t *p;
} dagPlan_t;

static void dagTableOf(dagPlan_t *d, void (*build)(dagFixture_t *)) {
    build(&d->f);
    TEST_ASSERT_TRUE(
        rematWireTableInit(&d->t, &d->f.m, defaultLossConfig(MSE), dagInput(&d->xs, 1, 2)));
    d->p = NULL;
}

static void dagPlanOf(dagPlan_t *d, void (*build)(dagFixture_t *), const rematPlanSpec_t *spec) {
    dagTableOf(d, build);
    TEST_ASSERT_TRUE(rematPlanBuild(&d->p, d->t, &d->f.m, spec));
}

static void freeDagPlan(dagPlan_t *d) {
    rematPlanFree(d->p);
    rematWireTableFree(d->t);
    dagFree(&d->f);
}

/* F3 steps: F0..F3 = 0..3, LOSS_FORWARD 4, LOSS_BACKWARD 5, B3 6, B2 7, B1 8, SUM(1) 9, B0 10.
 * Wires: ACT 0..4, the seed 5, GRAD 3 = 6, GRAD 1 = 7, GRAD 2 = 8, PART 1 = 9. B3 and B2 read
 * ACT 3, ACT 2 and ACT 1 as META only, so those reads keep nothing alive: ACT 2 and ACT 3 end at
 * their FORWARD reader, ACT 1 at B1, which reads it as the Linear's input. */
void testPlanBuildF3LivenessRanges(void) {
    dagPlan_t d;
    dagPlanOf(&d, buildF3, &g_liveness);
    const rematRange_t want[] = {RANGE_OF(1, 0, 8),  RANGE_OF(2, 1, 2), RANGE_OF(3, 2, 3),
                                 RANGE_OF(4, 3, 5),  RANGE_OF(5, 5, 6), RANGE_OF(6, 6, 7),
                                 RANGE_OF(7, 6, 10), RANGE_OF(8, 7, 8), RANGE_OF(9, 8, 9)};
    assertSameRanges(want, COUNT_OF(want), &d.p->train);
    freeDagPlan(&d);
}

void testPlanBuildF3StoreAllExtendsEveryActToTheEnd(void) {
    dagPlan_t d;
    dagPlanOf(&d, buildF3, NULL);
    const rematRange_t want[] = {RANGE_OF(1, 0, 10), RANGE_OF(2, 1, 10), RANGE_OF(3, 2, 10),
                                 RANGE_OF(4, 3, 10), RANGE_OF(5, 5, 6),  RANGE_OF(6, 6, 7),
                                 RANGE_OF(7, 6, 10), RANGE_OF(8, 7, 8),  RANGE_OF(9, 8, 9)};
    assertSameRanges(want, COUNT_OF(want), &d.p->train);
    freeDagPlan(&d);
}

static void dagPlanBuildExpectingAnExit(dagPlan_t *d, const rematPlanSpec_t *spec) {
    (void)rematPlanBuild(&d->p, d->t, &d->f.m, spec);
}

/* Recompute and in-place read chain positions; on a DAG they exit, even for a mask or a step
 * list that is otherwise well-formed. */
void testPlanBuildRefusesRecomputeAndInPlaceOnADag(void) {
    dagPlan_t d;
    dagTableOf(&d, buildF3);
    const bool retain[5] = {true, true, true, true, false};
    const rematPlanSpec_t retainList = {.policy = REMAT_PLAN_RETAIN_LIST, .retain = retain};
    ASSERT_EXITS_WITH_OUTPUT(1,
                             "rematPlanBuild: REMAT_PLAN_RETAIN_LIST on a DAG model is not "
                             "supported yet (recompute / in-place on DAGs)",
                             dagPlanBuildExpectingAnExit(&d, &retainList));
    const rematStep_t steps[] = {
        STEP_F(0),  STEP_F(1), STEP_F(2), STEP_F(3), STEP_LF(4),
        STEP_LB(4), STEP_B(3), STEP_B(2), STEP_B(1), {.kind = REMAT_STEP_SUM, .layer = 1},
        STEP_B(0)};
    const rematPlanSpec_t sequence = sequenceSpec(steps, COUNT_OF(steps));
    ASSERT_EXITS_WITH_OUTPUT(1,
                             "rematPlanBuild: REMAT_PLAN_SEQUENCE on a DAG model is not "
                             "supported yet (recompute / in-place on DAGs)",
                             dagPlanBuildExpectingAnExit(&d, &sequence));
    ASSERT_EXITS_WITH_OUTPUT(1,
                             "rematPlanBuild: inPlaceDx on a DAG model is not supported yet "
                             "(recompute / in-place on DAGs)",
                             dagPlanBuildExpectingAnExit(&d, &g_livenessInPlace));
    rematWireTableFree(d.t);
    dagFree(&d.f);
}

/* P2 steps: F0..F4 = 0..4, LOSS_FORWARD 5, LOSS_BACKWARD 6, B4 7, B3 8, B2 9, SUM(2) 10, B1 11,
 * SUM(1) 12, B0 13. b1's later contributors, the Linear B2 and the ReLU B1, each write a PART
 * (wires 11 and 12) that lives from its BACKWARD to the SUM right after it, never until B0 reads
 * GRAD 1, so the two never share a step. */
void testTwoUnaryLaterContributorsGetTwoDisjointPartRanges(void) {
    dagPlan_t d;
    dagPlanOf(&d, dagBuildP2, &g_liveness);
    const rematProgram_t *train = &d.p->train;
    size_t parts = 0;
    for (size_t w = 0; w < d.t->numWires; w++) {
        parts += (d.t->wires[w].kind == REMAT_WIRE_PART) ? 1u : 0u;
    }
    TEST_ASSERT_EQUAL_size_t(2, parts);
    TEST_ASSERT_EQUAL_UINT8(REMAT_WIRE_PART, d.t->wires[11].kind);
    TEST_ASSERT_EQUAL_UINT8(REMAT_WIRE_PART, d.t->wires[12].kind);
    TEST_ASSERT_EQUAL_UINT8(REMAT_STEP_BACKWARD, train->steps[9].kind);
    TEST_ASSERT_EQUAL_UINT16(2, train->steps[9].layer);
    TEST_ASSERT_EQUAL_UINT8(REMAT_STEP_SUM, train->steps[10].kind);
    TEST_ASSERT_EQUAL_UINT16(2, train->steps[10].layer);
    TEST_ASSERT_EQUAL_UINT8(REMAT_STEP_BACKWARD, train->steps[11].kind);
    TEST_ASSERT_EQUAL_UINT16(1, train->steps[11].layer);
    TEST_ASSERT_EQUAL_UINT8(REMAT_STEP_SUM, train->steps[12].kind);
    TEST_ASSERT_EQUAL_UINT16(1, train->steps[12].layer);
    assertOnlyRangeOf(train, 11, 9, 10);
    assertOnlyRangeOf(train, 12, 11, 12);
    freeDagPlan(&d);
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(testTableInitCrossEntropyRequiresALastSoftmax);
    RUN_TEST(testTableInitCrossEntropyHeadIsOneOfTwoDeclaredArms);
    RUN_TEST(testTableInitCrossEntropyChecksTheHeadOnlyWhenABackwardRuns);
    RUN_TEST(testTableBindChecksTheCrossEntropyHeadOfTheLiveModel);
    RUN_TEST(testCrossEntropySeedTakesTheSoftmaxDxWire);
    RUN_TEST(testCrossEntropySeedOfTheBfpHeadTakesItsPropLossQ);
    RUN_TEST(testCrossEntropySeedOfANonBfpHeadInheritsActN);
    RUN_TEST(testBfpWireGroupingMatchesTheDriverRule);
    RUN_TEST(testBfpWireGroupingExitsNamingTheWireOnAnIndivisibleGroupSize);
    RUN_TEST(testHarTableNumbersWiresInProductionOrder);
    RUN_TEST(testHarTableHoldsTheGradFlagsAndTheDxWires);
    RUN_TEST(testFrozenHarTableRunsOnlyTheLinearGradsOnly);
    RUN_TEST(testHarTableRecordsBytesRanksAndKey);
    RUN_TEST(testHarTableKeepsEveryActShape);
    RUN_TEST(testBindRewritesTheActShapes);
    RUN_TEST(testSharedGroupedBfpTemplateGroupsPerWire);
#ifdef ODT_MEM_PROFILE
    RUN_TEST(testTableInitReservesOneBlockOfSlabBytesAndFreeReturnsIt);
#endif
    RUN_TEST(testTableFreeReleasesExactlyTheOneBlock);
    RUN_TEST(testTableFreeIsNullSafe);
    RUN_TEST(testSlabObjectsAlignedAndExponentsAtTheTail);
    RUN_TEST(testAccessorsReadTheTableAndHeadersAreLinked);
    RUN_TEST(testTableInitExitsOnAnEmptyModel);
    RUN_TEST(testTableInitValidatesTheWiring);
    RUN_TEST(testTableInitExitsWhenWireIdsWouldReachRematNone);
    RUN_TEST(testTableInitExitsOnAnInputRankAboveTheRankField);
    RUN_TEST(testTableInitExitsOnAnUnsupportedWireDtype);
    RUN_TEST(testTableInitExitsOnAZeroByteWire);
    RUN_TEST(testTableInitExitsOnAReluOutputNarrowerThanItsInput);
    RUN_TEST(testTableInitExitsOnAFlattenDxNarrowerThanTheLossItCopies);
    RUN_TEST(testAWidthNarrowedAfterSetupIsTrustedAsTheCallersResponsibility);
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
    RUN_TEST(testTableCopiesTheWiringOfADag);
    RUN_TEST(testTableOfAChainKeepsNoWiring);
    RUN_TEST(testF3TableNumbersGradsThenThePartInWalkOrder);
    RUN_TEST(testAddOfTheSameBufferWritesThenAccumulates);
    RUN_TEST(testSkipFromTheInputGetsNoDx);
    RUN_TEST(testFlattenActTakesTheTemplateOfItsWiredInput);
    RUN_TEST(testAPartAFlattenWritesInheritsItsInputsTemplate);
    RUN_TEST(testTableInitRefusesASymInt32WireIntoAnAdd);
    RUN_TEST(testEvalOnlyTableInitRefusesASymInt32WireIntoAnAdd);
    RUN_TEST(testTableInitRefusesASymInt32Part);
    RUN_TEST(testTableInitRefusesASymInt32GradASumAddsInto);
    RUN_TEST(testTableInitRefusesASymInt32GradAnAddAccumulatesInto);
    RUN_TEST(testTableInitRefusesASymInt32DyOfAnAdd);
    RUN_TEST(testTableInitRefusesASymInt32CrossEntropySeedIntoAnAdd);
    RUN_TEST(testUnbindClearsTheBorrowedInputOnly);
#ifdef ODT_TEST_ASAN
    RUN_TEST(testAsanDeathCallbackExitsWithADistinctCode);
#endif
    RUN_TEST(testBindExitsOnAChangedModelSize);
    RUN_TEST(testBindExitsOnAChangedLossType);
    RUN_TEST(testBindExitsOnALayerTypeSwap);
    RUN_TEST(testBindExitsWhenFreezingStopsALayersBackward);
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
    RUN_TEST(testHarTrainOperandsAreTheDerivedSlices);
    RUN_TEST(testEvalProgramIsTheTrainPrefixOperandsIncluded);
    RUN_TEST(testEvalSlicesOfADagReadTheWiring);
    RUN_TEST(testDeriveF3TrainProgram);
    RUN_TEST(testDeriveF3EvalIsTheTrainPrefix);
    RUN_TEST(testDeriveSkipsTheFrozenSkipNode);
    RUN_TEST(testGrammarAcceptsF3AndS);
    RUN_TEST(testGrammarAcceptsYKAndP2);
    RUN_TEST(testGrammarRejectsASecondWriteOfAGrad);
    RUN_TEST(testGrammarRejectsAnAccFromANonAccumulatingRow);
    RUN_TEST(testGrammarRejectsAPartNotSummedNext);
    RUN_TEST(testGrammarRejectsAGradReadBeforeItsLastContribution);
    RUN_TEST(testGrammarRejectsABackwardOfTheFrozenSkipNode);
    RUN_TEST(testGrammarRejectsASumWithoutItsPart);
    RUN_TEST(testGrammarRejectsAPartWrittenTwice);
    RUN_TEST(testGrammarRejectsAnAccBeforeTheOpeningWriteInOneStep);
    RUN_TEST(testGrammarRejectsARangeAnAccOpens);
    RUN_TEST(testGrammarRejectsAnAccOutsideAnOpenRange);
    RUN_TEST(testGrammarRejectsInPlaceMarksOnADag);
    RUN_TEST(testAReforwardCarriesItsForwardsSlice);
    RUN_TEST(testTheDeriverGivesARejectedSequenceStepAnEmptySlice);
    RUN_TEST(testPlanProgramOfTrainIsTheTrainProgram);
    RUN_TEST(testEvalHarStepsAndRangesUnderEveryPolicy);
    RUN_TEST(testEvalPeakIsTheLargestAdjacentActPair);
#ifdef ODT_MEM_PROFILE
    RUN_TEST(testPlanBuildReservesOneBlockAndFreeReturnsIt);
#endif
    RUN_TEST(testNullOrZeroedSpecMeansStoreAll);
    RUN_TEST(testPlanBuildExitsOnAnUnknownPolicy);
    RUN_TEST(testSingleLinearUnderCrossEntropyDiesAtTheTable);
    RUN_TEST(testAllFrozenPlanHasNoBackwardPhase);
    RUN_TEST(testPlanFreeIsNullSafe);
    RUN_TEST(testLivenessHarRangesEndAtTheLastReader);
    RUN_TEST(testLivenessPeakHarIs49152);
    RUN_TEST(testLivenessPeakMnistCnnIs112896);
    RUN_TEST(testLivenessPeakFinetuneStage2Is16384);
    RUN_TEST(testActPeakOfHarIs57904StoreAllAnd41216Liveness);
    RUN_TEST(testActPeakOfFinetuneStage2Is57904StoreAllAnd16384Liveness);
    RUN_TEST(testActPeakOfMnistCnnIs125648StoreAllAnd100352Liveness);
    RUN_TEST(testFrozenNormStillNeedsItsInputWhileAFrozenGemmDoesNot);
    RUN_TEST(testCeLogitsDieAtForwardWhileMseSoftmaxInputIsRetained);
    RUN_TEST(testTrainableParameterLayersReadTheirInputInBackward);
    RUN_TEST(testEveryStepsOperandsAreCoLiveOnTheZoo);
    RUN_TEST(testEveryStepsOperandsAreCoLiveOnRandomChains);
    RUN_TEST(testStoredOperandsAreThePositionalOperandsOnTheZoo);
    RUN_TEST(testStoredOperandsAreThePositionalOperandsOnRandomChains);
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
    RUN_TEST(testGrammarRejectsABackwardReadingTheWrongGradient);
    RUN_TEST(testGrammarRejectsAMetaReadOfAnInputTheBackwardReads);
    RUN_TEST(testGrammarRejectsAReadOfAnInputTheBackwardNeverReads);
    RUN_TEST(testGrammarRejectsADxForAnInputThatNeedsNoGradient);
    RUN_TEST(testGrammarRejectsAStepOffItsNodesSlice);
    RUN_TEST(testGrammarRejectsASliceOutsideThePool);
    RUN_TEST(testGrammarRejectsAStepWithTheWrongOperandCounts);
    RUN_TEST(testGrammarDerivesFromTheModelNotFromTheTablesFlags);
    RUN_TEST(testGrammarConfirmsTheTablesGradientLookupAgainstTheRecords);
    RUN_TEST(testGrammarAcceptsAReforwardRunInsideTheWindow);
    RUN_TEST(testGrammarAcceptsAProgramAtTheStepBound);
    RUN_TEST(testGrammarRejectsMoreStepsThanTheBound);
    RUN_TEST(testGrammarRejectsAReforwardBeforeLossForward);
    RUN_TEST(testGrammarRejectsAReforwardAtOrAboveThePendingBackward);
    RUN_TEST(testGrammarRejectsAReforwardAfterTheFinalBackwardAtDeepestZero);
    RUN_TEST(testGrammarRejectsAReforwardWhenNothingTrains);
    RUN_TEST(testGrammarRejectsAReforwardInAnEvalProgram);
    RUN_TEST(testGrammarRejectsAReforwardReadingOutsideItsRange);
    RUN_TEST(testGrammarRejectsAWriteIntoARangeOpenedEarlier);
    RUN_TEST(testGrammarRejectsOverlappingRangesOfOneWire);
    RUN_TEST(testGrammarRejectsAReforwardWhoseOutputIsNeverRead);
    RUN_TEST(testGrammarRejectsARangeEndingPastTheStream);
    RUN_TEST(testGrammarRejectsRangesOutOfBeginOrder);
    RUN_TEST(testGrammarRejectsTwoRangesOfOneWireOpeningTogether);
    RUN_TEST(testGrammarRejectsARangeThatOpensWithoutAWrite);
    RUN_TEST(testSequencePlanCopiesTheStepsAndScansTheRanges);
    RUN_TEST(testSequencePlanKeepsItsOwnCopyOfTheSteps);
    RUN_TEST(testSequencePlanEvalProgramIsTheStoreAllEvalProgram);
    RUN_TEST(testLivenessStepsAsASequenceGiveTheLivenessPlanOnTheZoo);
    RUN_TEST(testLivenessStepsAsASequenceGiveTheLivenessPlanOnRandomChains);
    RUN_TEST(testSequencePlanRejectsAStepListOfAnotherFreezeState);
    RUN_TEST(testSequencePlanExitsOnANullSequenceBeforeReserving);
    RUN_TEST(testSequencePlanExitsOnAnEmptySequenceBeforeReserving);
    RUN_TEST(testSequencePlanExitsOnARecordCarryingOperands);
    RUN_TEST(testSequencePlanExitsOneStepAboveTheBoundBeforeReserving);
    RUN_TEST(testSequencePlanAcceptsALengthAtTheBound);
    RUN_TEST(testSequencePlanExitsAboveUint16MaxStepsBeforeReserving);
    RUN_TEST(testPlanBuildExitsOnAnOperandPoolAboveUint16Offsets);
    RUN_TEST(testPlanBuildExitsOnASequenceForAnotherPolicy);
    RUN_TEST(testPlanBuildExitsOnASequenceLengthForAnotherPolicy);
    RUN_TEST(testSequencePlanRejectsRecomputingATrackedBatchNorm);
    RUN_TEST(testSequencePlanRecomputesAnUntrackedBatchNorm);
    RUN_TEST(testSequencePlanRecomputesAFrozenTrackedBatchNorm);
    RUN_TEST(testRetainListPlanOnTheMlpIsTheRecomputeProgram);
    RUN_TEST(testRetainListHarKeepingActs3And7And12);
    RUN_TEST(testRetainListHarKeepingActs3And7);
    RUN_TEST(testPlanCostsEveryForwardAndEveryRecomputeOfMlp3);
    RUN_TEST(testPlanCostsAnAdaptivePoolOnItsInput);
    RUN_TEST(testPlanCostsADagNodeOnItsWiredInput);
    RUN_TEST(testPlanBuildExitsOnAForwardCostAboveSizeMax);
    RUN_TEST(testPlanBuildExitsOnARecomputeCostAboveSizeMax);
    RUN_TEST(testRetainListPlanEvalProgramIsTheStoreAllEvalProgram);
    RUN_TEST(testRetainListPlanKeepingEveryActIsTheLivenessPlanOnTheZoo);
    RUN_TEST(testRetainListPlanKeepingEveryActIsTheLivenessPlanOnRandomChains);
    RUN_TEST(testRetainListPlansAreSoundOnTheZoo);
    RUN_TEST(testRetainListPlansAreSoundOnRandomChains);
    RUN_TEST(testRetainListPlanExitsOnANullMaskBeforeReserving);
    RUN_TEST(testPlanBuildExitsOnARetainMaskForAnotherPolicy);
    RUN_TEST(testRetainListPlanExitsOnASequenceForItsPolicy);
    RUN_TEST(testRetainListPlanRejectsRecomputingATrackedBatchNorm);
    RUN_TEST(testRetainListPlanRecomputesAnUntrackedBatchNorm);
    RUN_TEST(testRetainListPlanExitsAboveUint16MaxStepsBeforeReserving);
    RUN_TEST(testRetainListPlanAtTheStepBoundStopsAtTheRangeCap);
    RUN_TEST(testRetainListPlanAcceptsExactlyMaxRangesAndExitsAboveThem);
    RUN_TEST(testPlanBuildRunsTheGrammarOnWhatItGenerates);
    RUN_TEST(testPlanBuildExitsWhenTheModelFreezesALayerTheTableSawTrainable);
    RUN_TEST(testPlanBuildExitsWhenTheModelSwapsALayerType);
    RUN_TEST(testInPlaceLivenessHarPeakLiveBytesIs41216);
    RUN_TEST(testInPlaceStoreAllHarPeakLiveBytesIs70192);
    RUN_TEST(testInPlaceReluAtTheCrossEntropyTopSharesTheSeed);
    RUN_TEST(testInPlaceChainOfTwoMembersHasOneRange);
    RUN_TEST(testScanRangesLandsAnInPlaceMembersWriteInItsRootsRange);
    RUN_TEST(testInPlaceEligibilityTruthTable);
    RUN_TEST(testInPlaceDxWithoutAPairKeepsThePlan);
    RUN_TEST(testInPlaceRetainListHarRunsARootOverAReforwardRun);
    RUN_TEST(testGrammarAcceptsAnInPlaceMemberInsideItsRootsRange);
    RUN_TEST(testGrammarRejectsAMarkOnAWireThatIsNotALayersDx);
    RUN_TEST(testGrammarRejectsAMarkOnALayerThatNeverAliases);
    RUN_TEST(testGrammarRejectsAMarkWhoseGradientIsNotTheWireBefore);
    RUN_TEST(testGrammarRejectsAMarkOnBfpWires);
    RUN_TEST(testGrammarRejectsAMarkOnWiresOfAnotherSizeOrDtype);
    RUN_TEST(testGrammarRejectsARangeOnAnInPlaceMember);
    RUN_TEST(testGrammarRejectsARangeOnAWireOutsideTheTable);
    RUN_TEST(testGrammarRejectsAMemberReadOutsideItsRootsRange);
    RUN_TEST(testPlanBuildF3LivenessRanges);
    RUN_TEST(testPlanBuildF3StoreAllExtendsEveryActToTheEnd);
    RUN_TEST(testPlanBuildRefusesRecomputeAndInPlaceOnADag);
    RUN_TEST(testTwoUnaryLaterContributorsGetTwoDisjointPartRanges);
    return UNITY_END();
}
