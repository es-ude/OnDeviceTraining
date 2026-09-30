#define SOURCE_FILE "UNIT_TEST_REMAT_CHECK"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "Common.h"
#include "Conv1dApi.h"
#include "DeathTest.h"
#include "FlattenApi.h"
#include "Layer.h"
#include "LayerNormApi.h"
#include "LayerQuant.h"
#include "LinearApi.h"
#include "LossFunction.h"
#include "Pool1dApi.h"
#include "QuantLayerApi.h"
#include "Quantization.h"
#include "ReluApi.h"
#include "RematCheck.h"
#include "RematPlan.h"
#include "RematScheduler.h"
#include "SoftmaxApi.h"
#include "StorageApi.h"
#include "Tensor.h"
#include "unity.h"

void setUp(void) {}
void tearDown(void) {}

/* Fixture helpers copied from UnitTestRematScheduler.c, with one change: the
 * input carries real bytes, because the checker requires every operand it
 * reads to be resident, ACT 0 included. */
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

static layer_t *makeQuant(quantization_t *outputQ, quantization_t *propLossQ) {
    return quantLayerInit(&(layerQuant_t){.outputQ = outputQ, .propLossQ = propLossQ});
}

static layer_t *makeLayerNorm(size_t features, bool frozen) {
    layerQuant_t lq;
    layerQuantInitUniform(&lq, &g_floatQ);
    return layerNormLayerInit(
        &(layerNormInit_t){.normalizedShape = (size_t[]){features},
                           .numNormDims = 1,
                           .trainable = frozen ? TRAINABLE_FALSE : TRAINABLE_DEFAULT},
        &lq);
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
        case CONV1D:
            freeConv1dLayer(model[i]);
            break;
        case MAXPOOL1D:
            freeMaxPool1dLayer(model[i]);
            break;
        case AVGPOOL1D:
            freeAvgPool1dLayer(model[i]);
            break;
        case FLATTEN:
            freeFlattenLayer(model[i]);
            break;
        case QUANTIZATION:
            freeQuantLayer(model[i]);
            break;
        case LAYERNORM:
            freeLayerNormLayer(model[i]);
            break;
        default:
            TEST_FAIL_MESSAGE("freeModel: extend the switch for this layer type");
        }
    }
}

/* A borrowed input on the caller's stack, with FLOAT32 bytes for up to the
 * HAR sample. The checker never reads the bytes; it only needs them bound. */
#define TEST_MAX_RANK 4
#define TEST_MAX_INPUT_FLOATS 1152u
typedef struct inputLike {
    size_t dims[TEST_MAX_RANK];
    size_t order[TEST_MAX_RANK];
    shape_t shape;
    tensor_t tensor;
    float data[TEST_MAX_INPUT_FLOATS];
} inputLike_t;

static tensor_t *makeInput(inputLike_t *in, const size_t *dims, size_t rank) {
    TEST_ASSERT_TRUE(rank <= TEST_MAX_RANK);
    size_t elements = 1;
    for (size_t d = 0; d < rank; d++) {
        in->dims[d] = dims[d];
        in->order[d] = d;
        elements *= dims[d];
    }
    TEST_ASSERT_TRUE(elements <= TEST_MAX_INPUT_FLOATS);
    in->shape = (shape_t){
        .numberOfDimensions = rank, .dimensions = in->dims, .orderOfDimensions = in->order};
    in->tensor = (tensor_t){.data = (uint8_t *)in->data,
                            .shape = &in->shape,
                            .quantization = &g_floatQ,
                            .sparsity = NULL};
    return &in->tensor;
}

/* One fixture model and its borrowed input. The F1 model's BFP template lives
 * here too, because its layers borrow it. */
#define FIXTURE_MAX_LAYERS 12
typedef struct fixture {
    layer_t *model[FIXTURE_MAX_LAYERS];
    size_t n;
    lossFuncType_t lt;
    inputLike_t in;
    tensor_t *x;
    uint8_t bfpExponent[1];
    bfpQConfig_t bfpQc;
    quantization_t bfpQ;
} fixture_t;

/* examples/har_classifier/train_c.c:178-216 (B = 1): n = 12, CE, deepest 0,
 * top 10; 25 steps; ACT 0..12 plus GRAD {12, 10, 9, ..., 1} = 24 wires. */
#define HAR_N 12
static void buildHar(fixture_t *f, bool freezeConvs) {
    layerQuant_t lq;
    layerQuantInitUniform(&lq, &g_floatQ);
    trainable_t conv = freezeConvs ? TRAINABLE_FALSE : TRAINABLE_DEFAULT;
    layer_t **model = f->model;
    model[0] = conv1dLayerInit(&(conv1dInit_t){.inChannels = 9,
                                               .outChannels = 16,
                                               .kernelSize = 7,
                                               .padding = SAME,
                                               .trainable = conv},
                               &lq);
    model[1] = reluLayerInit(&lq);
    model[2] = maxPool1dLayerInit(
        &(maxPool1dInit_t){.kernelSize = 2, .stride = 2, .inputChannels = 16, .inputLength = 128},
        &lq);
    model[3] = conv1dLayerInit(&(conv1dInit_t){.inChannels = 16,
                                               .outChannels = 32,
                                               .kernelSize = 5,
                                               .padding = SAME,
                                               .trainable = conv},
                               &lq);
    model[4] = reluLayerInit(&lq);
    model[5] = maxPool1dLayerInit(
        &(maxPool1dInit_t){.kernelSize = 2, .stride = 2, .inputChannels = 32, .inputLength = 64},
        &lq);
    model[6] = conv1dLayerInit(&(conv1dInit_t){.inChannels = 32,
                                               .outChannels = 64,
                                               .kernelSize = 3,
                                               .padding = SAME,
                                               .trainable = conv},
                               &lq);
    model[7] = reluLayerInit(&lq);
    model[8] = avgPool1dLayerInit(&(avgPool1dInit_t){.kernelSize = 32, .stride = 32}, &lq);
    model[9] = flattenLayerInit();
    model[10] = linearLayerInit(&(linearInit_t){.inFeatures = 64, .outFeatures = 6}, &lq);
    model[11] = softmaxLayerInit(&lq);
    f->n = HAR_N;
    f->lt = CROSS_ENTROPY;
    f->x = makeInput(&f->in, (size_t[]){1, 9, 128}, 3);
}

static void buildHarModel(fixture_t *f) {
    buildHar(f, false);
}

/* The Codex F1 alignment model: FLOAT32 [1,5] -> Quantization to BFP m = 8 ->
 * Linear 5 -> 1 under MSE. n = 2, deepest 1, top 1. Wires: ACT 0, ACT 1 (BFP,
 * 5 B), ACT 2 (4 B), the seed GRAD 2 (id 3). Steps: FORWARD 0 (#0), FORWARD 1
 * (#1), LOSS_FORWARD (#2), LOSS_BACKWARD (#3), BACKWARD(1) (#4, grads-only). */
static void buildF1Model(fixture_t *f) {
    initBfpQConfigInto(8, 8, HALF_AWAY, f->bfpExponent, &f->bfpQc);
    f->bfpQ = (quantization_t){.type = BFP, .qConfig = &f->bfpQc};
    f->model[0] = makeQuant(&f->bfpQ, &g_floatQ);
    f->model[1] = makeLinear(5, 1, false);
    f->n = 2;
    f->lt = MSE;
    f->x = makeInput(&f->in, (size_t[]){1, 5}, 2);
}

static const rematPlanSpec_t g_liveness = {.policy = REMAT_PLAN_LIVENESS};

static rematScheduler_t initArena(fixture_t *f, const rematPlanSpec_t *spec) {
    rematScheduler_t s;
    TEST_ASSERT_TRUE(rematArenaInit(&s, f->model, f->n, defaultLossConfig(f->lt), f->x, spec));
    return s;
}

static rematScheduler_t initHeap(fixture_t *f, const rematPlanSpec_t *spec) {
    rematScheduler_t s;
    TEST_ASSERT_TRUE(rematHeapInit(&s, f->model, f->n, defaultLossConfig(f->lt), f->x, spec));
    return s;
}

static void freeFixture(fixture_t *f, rematScheduler_t *s) {
    rematSchedulerDeinit(s);
    freeModel(f->model, f->n);
}

/* ---- rematCheckNumWires and rematCheckInit (spec §7.2, §7.3) ---- */

void testNumWiresIsTheTablesWireCount(void) {
    fixture_t f;
    buildHarModel(&f);
    rematScheduler_t s = initHeap(&f, NULL);
    TEST_ASSERT_EQUAL_size_t(24, rematCheckNumWires(&s)); /* spec §3.1: 23 slab + ACT 0 */
    freeFixture(&f, &s);
    buildF1Model(&f);
    s = initArena(&f, NULL);
    TEST_ASSERT_EQUAL_size_t(4, rematCheckNumWires(&s));
    freeFixture(&f, &s);
}

static void numWiresOfAZeroedScheduler(void) {
    rematScheduler_t s = {0};
    (void)rematCheckNumWires(&s);
}

/* The state rematHeapInit leaves when its plan block fails: fns and table set,
 * plan NULL. */
static void numWiresOfAHeapWhosePlanFailed(fixture_t *f) {
    rematScheduler_t s = initHeap(f, NULL);
    rematPlanFree(s.plan);
    s.plan = NULL;
    (void)rematCheckNumWires(&s);
}

/* Spec §7.2: the guard runs before the VLA it sizes exists, so the VLA is
 * never built from a NULL table. */
void testNumWiresExitsOnASchedulerThatIsNotInitialised(void) {
    ASSERT_EXITS_WITH_OUTPUT(1, "rematCheckNumWires: scheduler not initialised",
                             numWiresOfAZeroedScheduler());
    fixture_t f;
    buildF1Model(&f);
    ASSERT_EXITS_WITH_OUTPUT(1, "rematCheckNumWires: scheduler not initialised",
                             numWiresOfAHeapWhosePlanFailed(&f));
    freeModel(f.model, f.n);
}

/* The cursors come from rematBackwardRange on the model passed in (the LIVE
 * one, spec §6.7), not from the table's built facts: this table was built on
 * the trainable HAR (deepest 0), the live model freezes the convs (deepest 10,
 * the #380 cut). producedGen is a VLA, so it starts as garbage. */
void testInitSetsTheCursorsFromTheLiveModelAndZeroesProducedGen(void) {
    fixture_t built;
    buildHarModel(&built);
    rematScheduler_t s = initArena(&built, NULL);
    fixture_t live;
    buildHar(&live, true);
    size_t numWires = rematCheckNumWires(&s);
    uint32_t producedGen[numWires];
    memset(producedGen, 0xA5, sizeof producedGen);
    rematCheck_t c;
    rematCheckInit(&c, &s, live.model, live.n, live.lt, producedGen);
    TEST_ASSERT_EQUAL_PTR(&s, c.sched);
    TEST_ASSERT_EQUAL_PTR(live.model, c.model);
    TEST_ASSERT_EQUAL_size_t(HAR_N, c.n);
    TEST_ASSERT_EQUAL_size_t(10, c.deepest);
    TEST_ASSERT_EQUAL_INT(10, (int)c.backwardTop); /* n - 2 under CE */
    TEST_ASSERT_TRUE(c.hasBackward);
    TEST_ASSERT_EQUAL_size_t(0, c.nextForward);
    TEST_ASSERT_EQUAL_INT(10, (int)c.nextBackward);
    TEST_ASSERT_FALSE(c.lossForwardSeen);
    TEST_ASSERT_FALSE(c.lossBackwardSeen);
    TEST_ASSERT_EQUAL_size_t(0, c.stepIndex);
    TEST_ASSERT_EQUAL_PTR(producedGen, c.producedGen);
    for (size_t w = 0; w < numWires; w++) {
        TEST_ASSERT_EQUAL_UINT32(0, producedGen[w]);
    }
    freeModel(live.model, live.n);
    freeFixture(&built, &s);
}

/* ---- rematCheckStep: positional resolution and commit (spec §7.4, §7.5) ---- */

/* The positional rule restated from the step (spec §3.1): what the driver must
 * be handed. BACKWARD(top) reads the seed (the CE skip), and the grads-only
 * BACKWARD(deepest) writes nothing. */
static void assertResolved(const rematCheck_t *c, const rematStep_t *st,
                           const rematOperands_t *op) {
    const rematWireTable_t *t = c->sched->wires;
    size_t n = c->n;
    size_t l = st->layer;
    switch (st->kind) {
    case REMAT_STEP_FORWARD:
        TEST_ASSERT_EQUAL_PTR(rematActHdr(t, l), op->in);
        TEST_ASSERT_NULL(op->gradIn);
        TEST_ASSERT_EQUAL_PTR(rematActHdr(t, l + 1u), op->out);
        break;
    case REMAT_STEP_LOSS_FORWARD:
        TEST_ASSERT_EQUAL_PTR(rematActHdr(t, n), op->in);
        TEST_ASSERT_NULL(op->gradIn);
        TEST_ASSERT_NULL(op->out);
        break;
    case REMAT_STEP_LOSS_BACKWARD:
        TEST_ASSERT_EQUAL_PTR(rematActHdr(t, n), op->in);
        TEST_ASSERT_NULL(op->gradIn);
        TEST_ASSERT_NOT_NULL(op->out);
        TEST_ASSERT_EQUAL_PTR(rematGradHdr(t, n), op->out);
        break;
    default: /* REMAT_STEP_BACKWARD */
        TEST_ASSERT_EQUAL_PTR(rematActHdr(t, l), op->in);
        TEST_ASSERT_NOT_NULL(op->gradIn);
        TEST_ASSERT_EQUAL_PTR(rematGradHdr(t, (ptrdiff_t)l == c->backwardTop ? n : l + 1u),
                              op->gradIn);
        if (l == c->deepest) {
            TEST_ASSERT_NULL(op->out);
        } else {
            TEST_ASSERT_NOT_NULL(op->out);
            TEST_ASSERT_EQUAL_PTR(rematGradHdr(t, l), op->out);
        }
        break;
    }
}

/* The spec §6.1 driver loop with the layer execution left out. Asserts what
 * the checker resolved at every step and what it committed by the end: every
 * cursor at its end, every slab wire produced under its current binding, ACT 0
 * never. */
static void checkedCall(fixture_t *f, rematScheduler_t *s) {
    const rematWireTable_t *t = s->wires;
    uint32_t producedGen[rematCheckNumWires(s)];
    rematCheck_t c;
    rematCheckInit(&c, s, f->model, f->n, f->lt, producedGen);
    rematBegin(s, f->model, f->n, defaultLossConfig(f->lt), f->x);
    rematStep_t st;
    while (rematNext(s, &st)) {
        rematOperands_t op;
        rematCheckStep(&c, &st, &op);
        assertResolved(&c, &st, &op);
        rematDone(s, &st);
    }
    TEST_ASSERT_EQUAL_size_t(s->plan->train.numSteps, c.stepIndex);
    TEST_ASSERT_EQUAL_size_t(f->n, c.nextForward);
    TEST_ASSERT_TRUE(c.lossForwardSeen);
    TEST_ASSERT_EQUAL(c.hasBackward, c.lossBackwardSeen);
    ptrdiff_t belowDeepest = (ptrdiff_t)c.deepest - 1;
    TEST_ASSERT_EQUAL_INT((int)(c.backwardTop < belowDeepest ? c.backwardTop : belowDeepest),
                          (int)c.nextBackward);
    TEST_ASSERT_EQUAL_UINT32(0, producedGen[0]);
    for (uint16_t w = 1; w < t->numWires; w++) {
        TEST_ASSERT_NOT_EQUAL_UINT32(0, producedGen[w]);
        TEST_ASSERT_EQUAL_UINT32(t->wires[w].bindGen, producedGen[w]);
    }
    rematCheckFinish(&c);
    rematEnd(s);
    rematCheckReleased(&c);
}

typedef rematScheduler_t (*rowInit_t)(fixture_t *f, const rematPlanSpec_t *spec);

/* Two calls per plan: the second binds a persistent scheduler again. */
static void assertEveryStepAccepted(rowInit_t init, void (*build)(fixture_t *)) {
    const rematPlanSpec_t *specs[2] = {NULL, &g_liveness};
    for (size_t k = 0; k < 2u; k++) {
        fixture_t f;
        build(&f);
        rematScheduler_t s = init(&f, specs[k]);
        checkedCall(&f, &s);
        checkedCall(&f, &s);
        freeFixture(&f, &s);
    }
}

void testCheckAcceptsEveryStepOnArenaHar(void) {
    assertEveryStepAccepted(initArena, buildHarModel);
}

void testCheckAcceptsEveryStepOnArenaF1(void) {
    assertEveryStepAccepted(initArena, buildF1Model);
}

void testCheckAcceptsEveryStepOnHeapHar(void) {
    assertEveryStepAccepted(initHeap, buildHarModel);
}

void testCheckAcceptsEveryStepOnHeapF1(void) {
    assertEveryStepAccepted(initHeap, buildF1Model);
}

/* Literal positions on HAR (steps: FORWARD l = l, LOSS_FORWARD 12,
 * LOSS_BACKWARD 13, BACKWARD(l) = 24 - l): the CE skip hands BACKWARD(10) the
 * seed GRAD 12, BACKWARD(9) gets GRAD 10, and BACKWARD(0) is grads-only.
 * Under LIVENESS, Flatten's input ACT 9 is dead at BACKWARD(9) and is handed
 * over anyway: its data is NULL (W_dead). */
void testOperandsAreResolvedPositionallyOnHar(void) {
    fixture_t f;
    buildHarModel(&f);
    rematScheduler_t s = initArena(&f, &g_liveness);
    const rematWireTable_t *t = s.wires;
    uint32_t producedGen[rematCheckNumWires(&s)];
    rematCheck_t c;
    rematCheckInit(&c, &s, f.model, f.n, f.lt, producedGen);
    rematBegin(&s, f.model, f.n, defaultLossConfig(f.lt), f.x);
    rematOperands_t at[25];
    rematStep_t st;
    size_t i = 0;
    while (rematNext(&s, &st)) {
        TEST_ASSERT_TRUE(i < 25u);
        rematCheckStep(&c, &st, &at[i]);
        if (i == 15u) {
            TEST_ASSERT_NULL(at[i].in->data);
        }
        rematDone(&s, &st);
        i++;
    }
    TEST_ASSERT_EQUAL_PTR(f.x, at[0].in);
    TEST_ASSERT_EQUAL_PTR(rematActHdr(t, 4), at[3].out);
    TEST_ASSERT_EQUAL_PTR(rematActHdr(t, 12), at[12].in);
    TEST_ASSERT_EQUAL_PTR(rematGradHdr(t, 12), at[13].out);
    TEST_ASSERT_EQUAL_PTR(rematActHdr(t, 10), at[14].in);
    TEST_ASSERT_EQUAL_PTR(rematGradHdr(t, 12), at[14].gradIn);
    TEST_ASSERT_EQUAL_PTR(rematGradHdr(t, 10), at[14].out);
    TEST_ASSERT_EQUAL_PTR(rematActHdr(t, 9), at[15].in);
    TEST_ASSERT_EQUAL_PTR(rematGradHdr(t, 10), at[15].gradIn);
    TEST_ASSERT_EQUAL_PTR(rematGradHdr(t, 9), at[15].out);
    TEST_ASSERT_EQUAL_PTR(rematGradHdr(t, 1), at[24].gradIn);
    TEST_ASSERT_NULL(at[24].out);
    rematEnd(&s);
    freeFixture(&f, &s);
}

/* ---- violations: a desynchronised step (spec §7.4 rule 1) ---- */

typedef void (*tamperFn_t)(rematCheck_t *c, rematScheduler_t *s);

/* Death-test children only. A checked call through step k-1; then next()
 * hands out step k (binding its ranges, if the stream has one), `tamper`
 * edits the scheduler or the checker (NULL: none), and the checker is offered
 * `forged` in place of step k (NULL: the step next() handed out). A checker
 * that accepts it returns here, and the child exits 0. */
static void offerStep(fixture_t *f, rematScheduler_t *s, size_t k, const rematStep_t *forged,
                      tamperFn_t tamper) {
    uint32_t producedGen[rematCheckNumWires(s)];
    rematCheck_t c;
    rematCheckInit(&c, s, f->model, f->n, f->lt, producedGen);
    rematBegin(s, f->model, f->n, defaultLossConfig(f->lt), f->x);
    rematStep_t st;
    rematOperands_t op;
    for (size_t i = 0; i < k; i++) {
        if (!rematNext(s, &st)) {
            return;
        }
        rematCheckStep(&c, &st, &op);
        rematDone(s, &st);
    }
    bool handedOut = rematNext(s, &st);
    if (tamper != NULL) {
        tamper(&c, s);
    }
    if (forged != NULL) {
        st = *forged;
    } else if (!handedOut) {
        return;
    }
    rematCheckStep(&c, &st, &op);
}

/* The full §7.7 message once: row, step index, kind, layer, rule. */
void testStepExitsOnAnUnknownStepKind(void) {
    fixture_t f;
    buildF1Model(&f);
    rematScheduler_t s = initArena(&f, NULL);
    rematStep_t forged = {.kind = 7u, .layer = 0u};
    ASSERT_EXITS_WITH_OUTPUT(
        1, "remat[arena]: step #0 UNKNOWN(layer 0) violates 'unknown step kind' (kind 7)",
        offerStep(&f, &s, 0, &forged, NULL));
    freeFixture(&f, &s);
}

/* Rule 1 runs before resolution, so a bad layer never indexes the model or
 * the table. FORWARD(n) at LOSS_FORWARD's slot passes the forward cursor
 * (nextForward == n) and would resolve ACT n+1, the seed's id. */
void testStepExitsOnALayerOutOfRangeBeforeResolvingIt(void) {
    fixture_t f;
    buildF1Model(&f);
    rematScheduler_t s = initArena(&f, NULL);
    rematStep_t forwardPastTheModel = {.kind = REMAT_STEP_FORWARD, .layer = 2u};
    ASSERT_EXITS_WITH_OUTPUT(1, "FORWARD(layer 2) violates 'layer out of range' (n = 2)",
                             offerStep(&f, &s, 2, &forwardPastTheModel, NULL));
    rematStep_t backwardFarOut = {.kind = REMAT_STEP_BACKWARD, .layer = UINT16_MAX};
    ASSERT_EXITS_WITH_OUTPUT(1, "BACKWARD(layer 65535) violates 'layer out of range' (n = 2)",
                             offerStep(&f, &s, 0, &backwardFarOut, NULL));
    freeFixture(&f, &s);
}

/* LOSS_* carry layer == n (spec §4.1). */
void testStepExitsOnALossStepWhoseLayerIsNotN(void) {
    fixture_t f;
    buildF1Model(&f);
    rematScheduler_t s = initArena(&f, NULL);
    rematStep_t lossAtLayer1 = {.kind = REMAT_STEP_LOSS_FORWARD, .layer = 1u};
    ASSERT_EXITS_WITH_OUTPUT(1, "LOSS_FORWARD(layer 1) violates 'layer out of range' (n = 2)",
                             offerStep(&f, &s, 2, &lossAtLayer1, NULL));
    freeFixture(&f, &s);
}

/* ---- violations: order and phase (spec §7.4 rule 2, §7.5) ---- */

/* Death-test children only: the §6.1 loop without a test assertion, so a
 * checker that accepts every step simply returns and the child exits 0. */
static void driveCall(fixture_t *f, rematScheduler_t *s) {
    uint32_t producedGen[rematCheckNumWires(s)];
    rematCheck_t c;
    rematCheckInit(&c, s, f->model, f->n, f->lt, producedGen);
    rematBegin(s, f->model, f->n, defaultLossConfig(f->lt), f->x);
    rematStep_t st;
    rematOperands_t op;
    while (rematNext(s, &st)) {
        rematCheckStep(&c, &st, &op);
        rematDone(s, &st);
    }
    rematCheckFinish(&c);
    rematEnd(s);
    rematCheckReleased(&c);
}

/* D24: decorator rows wrap the real ARENA slots and are installed on the
 * test's own instance; their tables are const. */
static void passBegin(rematScheduler_t *s) {
    rematSchedulerFunctions[REMAT_ARENA].begin(s);
}
static bool passNext(rematScheduler_t *s, rematStep_t *st) {
    return rematSchedulerFunctions[REMAT_ARENA].next(s, st);
}
static void passDone(rematScheduler_t *s, const rematStep_t *st) {
    rematSchedulerFunctions[REMAT_ARENA].done(s, st);
}
static void passEnd(rematScheduler_t *s) {
    rematSchedulerFunctions[REMAT_ARENA].end(s);
}
static void passDeinit(rematScheduler_t *s) {
    rematSchedulerFunctions[REMAT_ARENA].deinit(s);
}

/* A row that skips FORWARD(5): it answers the step itself and hands out the
 * next one, so the dispatch's same-step check sees nothing wrong. */
static bool skippingNext(rematScheduler_t *s, rematStep_t *st) {
    if (!passNext(s, st)) {
        return false;
    }
    if (st->kind == REMAT_STEP_FORWARD && st->layer == 5u) {
        passDone(s, st);
        return passNext(s, st);
    }
    return true;
}

static const rematSchedulerFunctions_t g_skippingArena = {.name = "skipping-arena",
                                                          .begin = passBegin,
                                                          .next = skippingNext,
                                                          .done = passDone,
                                                          .end = passEnd,
                                                          .deinit = passDeinit};

/* [ReLU] under MSE: nothing trains (deepest = n = 1), so the stream is
 * FORWARD 0, LOSS_FORWARD and ends; wires ACT 0, ACT 1. */
static void buildAllFrozenModel(fixture_t *f) {
    f->model[0] = reluLayerInit(&(layerQuant_t){.outputQ = &g_floatQ, .propLossQ = &g_floatQ});
    f->n = 1;
    f->lt = MSE;
    f->x = makeInput(&f->in, (size_t[]){1, 4}, 2);
}

/* One F1 ARENA scheduler; `forged` is offered at step k. */
static void assertF1RejectsAt(size_t k, rematStep_t forged, const char *violation) {
    fixture_t f;
    buildF1Model(&f);
    rematScheduler_t s = initArena(&f, NULL);
    ASSERT_EXITS_WITH_OUTPUT(1, violation, offerStep(&f, &s, k, &forged, NULL));
    freeFixture(&f, &s);
}

void testStepExitsOnAForwardAfterTheLossForward(void) {
    assertF1RejectsAt(3, (rematStep_t){.kind = REMAT_STEP_FORWARD, .layer = 1u},
                      "step #3 FORWARD(layer 1) violates 'forward after loss'");
}

void testStepExitsOnAForwardOutOfOrder(void) {
    assertF1RejectsAt(0, (rematStep_t){.kind = REMAT_STEP_FORWARD, .layer = 1u},
                      "violates 'forward order: expected FORWARD(0)'");
}

/* A row that silently drops a step. */
void testStepExitsOnARowThatSkipsAStep(void) {
    fixture_t f;
    buildHarModel(&f);
    rematScheduler_t s = initArena(&f, NULL);
    s.fns = &g_skippingArena;
    ASSERT_EXITS_WITH_OUTPUT(1,
                             "remat[skipping-arena]: step #5 FORWARD(layer 6) violates 'forward "
                             "order: expected FORWARD(5)'",
                             driveCall(&f, &s));
    freeFixture(&f, &s);
}

void testStepExitsOnALossForwardBeforeTheLastForward(void) {
    assertF1RejectsAt(1, (rematStep_t){.kind = REMAT_STEP_LOSS_FORWARD, .layer = 2u},
                      "violates 'loss-forward early' (FORWARD(1) has not run)");
}

void testStepExitsOnADuplicateLossForward(void) {
    assertF1RejectsAt(3, (rematStep_t){.kind = REMAT_STEP_LOSS_FORWARD, .layer = 2u},
                      "violates 'duplicate loss-forward'");
}

void testStepExitsOnALossBackwardBeforeTheLossForward(void) {
    assertF1RejectsAt(0, (rematStep_t){.kind = REMAT_STEP_LOSS_BACKWARD, .layer = 2u},
                      "violates 'loss-backward before loss-forward'");
}

/* The all-frozen stream ends after LOSS_FORWARD; a LOSS_BACKWARD after it
 * would seed a gradient no layer consumes (GRAD 1 does not exist). */
void testStepExitsOnALossBackwardWithoutATrainableLayer(void) {
    fixture_t f;
    buildAllFrozenModel(&f);
    rematScheduler_t s = initArena(&f, NULL);
    rematStep_t lossBackward = {.kind = REMAT_STEP_LOSS_BACKWARD, .layer = 1u};
    ASSERT_EXITS_WITH_OUTPUT(1, "violates 'loss-backward without trainable layer'",
                             offerStep(&f, &s, 2, &lossBackward, NULL));
    freeFixture(&f, &s);
}

void testStepExitsOnADuplicateLossBackward(void) {
    assertF1RejectsAt(4, (rematStep_t){.kind = REMAT_STEP_LOSS_BACKWARD, .layer = 2u},
                      "violates 'duplicate loss-backward'");
}

void testStepExitsOnABackwardBeforeTheLossBackward(void) {
    assertF1RejectsAt(2, (rematStep_t){.kind = REMAT_STEP_BACKWARD, .layer = 1u},
                      "violates 'backward before loss-backward'");
}

/* HAR's BACKWARD(9) sits at step 15; BACKWARD(8) there skips it. */
void testStepExitsOnABackwardOutOfOrder(void) {
    fixture_t f;
    buildHarModel(&f);
    rematScheduler_t s = initArena(&f, NULL);
    rematStep_t skipsNine = {.kind = REMAT_STEP_BACKWARD, .layer = 8u};
    ASSERT_EXITS_WITH_OUTPUT(1, "violates 'backward order: expected BACKWARD(9)'",
                             offerStep(&f, &s, 15, &skipsNine, NULL));
    freeFixture(&f, &s);
}

/* After HAR's last BACKWARD (deepest 0) the cursor is -1: it must stay
 * signed, or it reads as SIZE_MAX, which is not below deepest (spec §7.3). */
void testStepExitsOnABackwardAfterTheLastOne(void) {
    fixture_t f;
    buildHarModel(&f);
    rematScheduler_t s = initArena(&f, NULL);
    rematStep_t againAtZero = {.kind = REMAT_STEP_BACKWARD, .layer = 0u};
    ASSERT_EXITS_WITH_OUTPUT(
        1, "violates 'backward order: expected no further BACKWARD' (deepest = 0)",
        offerStep(&f, &s, 25, &againAtZero, NULL));
    freeFixture(&f, &s);
}

/* The driver must re-init the checker for every call: a stale checker sees the second call's first
 * FORWARD as a forward after the loss. */
static void twoCallsOnOneChecker(fixture_t *f, rematScheduler_t *s) {
    uint32_t producedGen[rematCheckNumWires(s)];
    rematCheck_t c;
    rematCheckInit(&c, s, f->model, f->n, f->lt, producedGen);
    for (int call = 0; call < 2; call++) {
        rematBegin(s, f->model, f->n, defaultLossConfig(f->lt), f->x);
        rematStep_t st;
        rematOperands_t op;
        while (rematNext(s, &st)) {
            rematCheckStep(&c, &st, &op);
            rematDone(s, &st);
        }
        rematEnd(s);
    }
}

void testASecondCallWithoutReInitExitsAtItsFirstStep(void) {
    fixture_t f;
    buildF1Model(&f);
    rematScheduler_t s = initHeap(&f, NULL);
    ASSERT_EXITS_WITH_OUTPUT(1,
                             "remat[heap]: step #5 FORWARD(layer 0) violates 'forward after loss'",
                             twoCallsOnOneChecker(&f, &s));
    freeFixture(&f, &s);
}

/* ---- violations: residency and bind generation (spec §7.3, §7.4 rule 4) ---- */

static void releaseAct1(rematCheck_t *c, rematScheduler_t *s) {
    (void)c;
    rematWireRelease(s->wires, rematActId(s->wires, 1));
}

static void releaseTheSeed(rematCheck_t *c, rematScheduler_t *s) {
    (void)c;
    rematWireRelease(s->wires, rematGradId(s->wires, 2));
}

static void forgetThatAct1WasProduced(rematCheck_t *c, rematScheduler_t *s) {
    c->producedGen[rematActId(s->wires, 1)] = 0u;
}

/* The swap-target bug class (spec §7.6): a row re-binds the seed without its
 * producer running again, so its bytes are not the ones LOSS_BACKWARD wrote. */
static void rebindTheSeed(rematCheck_t *c, rematScheduler_t *s) {
    (void)c;
    uint16_t seed = rematGradId(s->wires, 2);
    uint8_t *bytes = rematWireHdr(s->wires, seed)->data;
    rematWireRelease(s->wires, seed);
    rematWireBind(s->wires, seed, bytes);
}

/* One F1 ARENA scheduler; the step next() hands out at k is offered after
 * `tamper`. */
static void assertF1TamperedRejectsAt(size_t k, tamperFn_t tamper, const char *violation) {
    fixture_t f;
    buildF1Model(&f);
    rematScheduler_t s = initArena(&f, NULL);
    ASSERT_EXITS_WITH_OUTPUT(1, violation, offerStep(&f, &s, k, NULL, tamper));
    freeFixture(&f, &s);
}

void testStepExitsWhenAnInputIsNotResident(void) {
    assertF1TamperedRejectsAt(1, releaseAct1,
                              "step #1 FORWARD(layer 1) violates 'operand not resident: in ACT 1'");
}

/* F1's Linear trains, so its BACKWARD reads ACT 1 (spec §3.7). */
void testStepExitsWhenAReadingBackwardsInputIsNotResident(void) {
    assertF1TamperedRejectsAt(4, releaseAct1,
                              "BACKWARD(layer 1) violates 'operand not resident: in ACT 1'");
}

void testStepExitsWhenGradInIsNotResident(void) {
    assertF1TamperedRejectsAt(4, releaseTheSeed, "violates 'operand not resident: gradIn GRAD 2'");
}

void testStepExitsWhenTheOutputIsNotBound(void) {
    assertF1TamperedRejectsAt(0, releaseAct1,
                              "FORWARD(layer 0) violates 'output not resident: ACT 1'");
}

/* Unreachable through a stream the order rules admit (every read wire's
 * producer ran before it): the state is forged. */
void testStepExitsOnAnOperandNeverProduced(void) {
    assertF1TamperedRejectsAt(1, forgetThatAct1WasProduced,
                              "violates 'operand never produced: ACT 1'");
}

void testStepExitsOnAStaleOperand(void) {
    assertF1TamperedRejectsAt(4, rebindTheSeed,
                              "BACKWARD(layer 1) violates 'operand stale: GRAD 2 rebound since "
                              "produced' (produced at bindGen 1, bound now at bindGen 2)");
}

/* A caller whose input header has no bytes. ACT 0 is borrowed, so it is exempt from the bind
 * generation, not from residency. */
void testStepExitsOnAnInputWithoutBytes(void) {
    fixture_t f;
    buildF1Model(&f);
    f.x->data = NULL;
    rematScheduler_t s = initHeap(&f, NULL);
    ASSERT_EXITS_WITH_OUTPUT(
        1, "remat[heap]: step #0 FORWARD(layer 0) violates 'operand not resident: in ACT 0'",
        driveCall(&f, &s));
    freeFixture(&f, &s);
}

/* [Linear T, LayerNorm frozen, Linear frozen, Linear T] under MSE: n = 4,
 * deepest 0, top 3. Under LIVENESS the frozen Linear's input ACT 2 dies at its
 * forward (its backward does not read it), while the frozen LayerNorm's ACT 1
 * lives to BACKWARD(1) (spec §3.7, §12.2 item 9). */
static void buildFrozenZooModel(fixture_t *f) {
    f->model[0] = makeLinear(4, 4, false);
    f->model[1] = makeLayerNorm(4, true);
    f->model[2] = makeLinear(4, 4, true);
    f->model[3] = makeLinear(4, 2, false);
    f->n = 4;
    f->lt = MSE;
    f->x = makeInput(&f->in, (size_t[]){1, 4}, 2);
}

/* The checker's read-set is the planner's: a checker that demanded every BACKWARD input would
 * reject LIVENESS's dead ACT 2. */
void testCheckAcceptsTheFrozenZooOnBothRows(void) {
    assertEveryStepAccepted(initArena, buildFrozenZooModel);
    assertEveryStepAccepted(initHeap, buildFrozenZooModel);
}

/* ---- violations: operands sharing bytes (spec §7.3, §7.4 rule 5) ---- */

static size_t rangeOf(const rematProgram_t *p, uint16_t wire) {
    for (size_t r = 0; r < p->numRanges; r++) {
        if (p->ranges[r].wire == wire) {
            return r;
        }
    }
    TEST_FAIL_MESSAGE("no range for the wire");
    return 0;
}

/* Spec §12.2 item 3a: ARENA offsets edited after init bypass the init
 * verifier, so the run-time rule is the one that fires. `victim` (a GRAD if
 * victimIsGrad, else an ACT) is placed onto ACT `onto`'s bytes; under
 * STORE_ALL both are co-live at the step that reads or writes them together. */
static void assertSharedBytesRejected(void (*build)(fixture_t *), bool victimIsGrad, size_t victim,
                                      size_t onto, const char *violation) {
    fixture_t f;
    build(&f);
    rematScheduler_t s = initArena(&f, NULL);
    const rematProgram_t *p = &s.plan->train;
    uint16_t victimId = victimIsGrad ? rematGradId(s.wires, victim) : rematActId(s.wires, victim);
    s.row.arena.offsets[rangeOf(p, victimId)] =
        s.row.arena.offsets[rangeOf(p, rematActId(s.wires, onto))];
    ASSERT_EXITS_WITH_OUTPUT(1, violation, driveCall(&f, &s));
    freeFixture(&f, &s);
}

void testStepExitsWhenAForwardsInputAndOutputShareBytes(void) {
    assertSharedBytesRejected(
        buildF1Model, false, 2, 1,
        "step #1 FORWARD(layer 1) violates 'operands share bytes: in/out' (ACT 1 and ACT 2)");
}

/* Pairwise, not only gradIn/out (spec §7.5): F1's trained Linear reads ACT 1
 * in the BACKWARD that reads the seed. */
void testStepExitsWhenABackwardsInputAndGradInShareBytes(void) {
    assertSharedBytesRejected(
        buildF1Model, true, 2, 1,
        "BACKWARD(layer 1) violates 'operands share bytes: in/gradIn' (ACT 1 and GRAD 2)");
}

/* The pair that is not adjacent in {in, gradIn, out}: HAR's BACKWARD(10)
 * reads ACT 10 and the seed and writes GRAD 10. */
void testStepExitsWhenABackwardsInputAndOutputShareBytes(void) {
    assertSharedBytesRejected(
        buildHarModel, true, 10, 10,
        "step #14 BACKWARD(layer 10) violates 'operands share bytes: in/out' (ACT 10 and GRAD 10)");
}

/* The canonical pair (spec §7.5): HAR's BACKWARD(9) does not read its input
 * (Flatten), so its only pair is gradIn/out. */
static void assertGradOntoGradRejected(size_t victim, size_t onto, const char *violation) {
    fixture_t f;
    buildHarModel(&f);
    rematScheduler_t s = initArena(&f, NULL);
    const rematProgram_t *p = &s.plan->train;
    s.row.arena.offsets[rangeOf(p, rematGradId(s.wires, victim))] =
        s.row.arena.offsets[rangeOf(p, rematGradId(s.wires, onto))];
    ASSERT_EXITS_WITH_OUTPUT(1, violation, driveCall(&f, &s));
    freeFixture(&f, &s);
}

void testStepExitsWhenABackwardsGradInAndOutputShareBytes(void) {
    assertGradOntoGradRejected(9, 10,
                               "step #15 BACKWARD(layer 9) violates 'operands share bytes: "
                               "gradIn/out' (GRAD 10 and GRAD 9)");
}

/* LOSS_BACKWARD reads ACT n and writes the seed (spec §7.5). */
void testStepExitsWhenALossBackwardsInputAndOutputShareBytes(void) {
    assertSharedBytesRejected(buildF1Model, true, 2, 2,
                              "step #3 LOSS_BACKWARD(layer 2) violates 'operands share bytes: "
                              "in/out' (ACT 2 and GRAD 2)");
}

static void releaseAct1AndAct2(rematCheck_t *c, rematScheduler_t *s) {
    (void)c;
    rematWireRelease(s->wires, rematActId(s->wires, 1));
    rematWireRelease(s->wires, rematActId(s->wires, 2));
}

/* Both FORWARD(1) operands unbound: residency names the first, before two NULL intervals "share
 * bytes" (spec §7.4: rule 4 before rule 5). */
void testStepExitsOnNonResidencyBeforeSharedBytes(void) {
    assertF1TamperedRejectsAt(1, releaseAct1AndAct2,
                              "step #1 FORWARD(layer 1) violates 'operand not resident: in ACT 1'");
}

/* Only the header the checker sees is aliased: the arena placement stays the verified one. The
 * child exits after the check, before any release could poison the shared bytes twice. */
static void aliasAct9OntoGrad9(rematCheck_t *c, rematScheduler_t *s) {
    (void)c;
    rematWireHdr(s->wires, rematActId(s->wires, 9))->data =
        rematWireHdr(s->wires, rematGradId(s->wires, 9))->data;
}

/* Flatten's BACKWARD(9) (HAR step 15) does not read ACT 9: sharing bytes with its out is legal
 * (spec §7.5). Exit 0 is the acceptance; the empty needle keeps a rejection's banner in the
 * failure message. */
void testStepAcceptsADeadInputSharingBytesWithTheOutput(void) {
    fixture_t f;
    buildHarModel(&f);
    rematScheduler_t s = initArena(&f, NULL);
    ASSERT_EXITS_WITH_OUTPUT(0, "", offerStep(&f, &s, 15, NULL, aliasAct9OntoGrad9));
    freeFixture(&f, &s);
}

/* ---- the stream-level checks (spec §7.6) ---- */

/* Death-test children only: `k` checked steps, then the stream is declared
 * finished. */
static void finishAfter(fixture_t *f, rematScheduler_t *s, size_t k) {
    uint32_t producedGen[rematCheckNumWires(s)];
    rematCheck_t c;
    rematCheckInit(&c, s, f->model, f->n, f->lt, producedGen);
    rematBegin(s, f->model, f->n, defaultLossConfig(f->lt), f->x);
    rematStep_t st;
    rematOperands_t op;
    for (size_t i = 0; i < k; i++) {
        if (!rematNext(s, &st)) {
            return;
        }
        rematCheckStep(&c, &st, &op);
        rematDone(s, &st);
    }
    rematCheckFinish(&c);
}

static void assertF1FinishRejectsAfter(size_t k, const char *violation) {
    fixture_t f;
    buildF1Model(&f);
    rematScheduler_t s = initHeap(&f, NULL);
    ASSERT_EXITS_WITH_OUTPUT(1, violation, finishAfter(&f, &s, k));
    freeFixture(&f, &s);
}

void testFinishExitsOnAMissingForward(void) {
    assertF1FinishRejectsAfter(
        1, "remat[heap]: stream of 1 steps violates 'incomplete stream: missing FORWARD(1)'");
}

void testFinishExitsOnAMissingLossForward(void) {
    assertF1FinishRejectsAfter(2, "violates 'incomplete stream: missing LOSS_FORWARD'");
}

void testFinishExitsOnAMissingLossBackward(void) {
    assertF1FinishRejectsAfter(3, "violates 'incomplete stream: missing LOSS_BACKWARD'");
}

void testFinishExitsOnAMissingBackward(void) {
    assertF1FinishRejectsAfter(4, "violates 'incomplete stream: missing BACKWARD(1)'");
}

/* A row whose stream ends one step early. The driver checks the stream before
 * rematEnd, so the checker names the missing step before the row's own
 * walk-complete check could. */
static bool truncatingNext(rematScheduler_t *s, rematStep_t *st) {
    if (s->walk.step + 1u == s->plan->train.numSteps) {
        return false;
    }
    return passNext(s, st);
}

static const rematSchedulerFunctions_t g_truncatingArena = {.name = "truncating-arena",
                                                            .begin = passBegin,
                                                            .next = truncatingNext,
                                                            .done = passDone,
                                                            .end = passEnd,
                                                            .deinit = passDeinit};

void testFinishExitsWhenARowEndsTheStreamEarly(void) {
    fixture_t f;
    buildF1Model(&f);
    rematScheduler_t s = initArena(&f, NULL);
    s.fns = &g_truncatingArena;
    ASSERT_EXITS_WITH_OUTPUT(1,
                             "remat[truncating-arena]: stream of 4 steps violates 'incomplete "
                             "stream: missing BACKWARD(1)'",
                             driveCall(&f, &s));
    freeFixture(&f, &s);
}

/* A row that consumes the stream's real last step internally -- fetched
 * from the real next(), answered through the real done() -- and then
 * returns false. done() advances the row's own walk.step to numSteps, so
 * rematEnd's rematRequireWalkComplete would NOT fire (unlike truncatingNext
 * above, which never calls the real next() for that step at all); only
 * rematCheckFinish, which this step was never offered to, can catch the
 * missing BACKWARD. */
static bool eatingNext(rematScheduler_t *s, rematStep_t *st) {
    if (s->walk.step + 1u == s->plan->train.numSteps) {
        rematStep_t eaten;
        (void)passNext(s, &eaten);
        passDone(s, &eaten);
        return false;
    }
    return passNext(s, st);
}

static const rematSchedulerFunctions_t g_eatingArena = {.name = "eating-arena",
                                                        .begin = passBegin,
                                                        .next = eatingNext,
                                                        .done = passDone,
                                                        .end = passEnd,
                                                        .deinit = passDeinit};

void testFinishExitsWhenARowEatsTheLastStepInternally(void) {
    fixture_t f;
    buildF1Model(&f);
    rematScheduler_t s = initArena(&f, NULL);
    s.fns = &g_eatingArena;
    ASSERT_EXITS_WITH_OUTPUT(
        1,
        "remat[eating-arena]: stream of 4 steps violates 'incomplete stream: missing BACKWARD(1)'",
        driveCall(&f, &s));
    freeFixture(&f, &s);
}

/* [Linear 4 -> 3] under CE, n = 1 (D20): top = -1 < deepest = 0, so the
 * stream is FORWARD 0, LOSS_FORWARD, LOSS_BACKWARD and no BACKWARD at all;
 * wires ACT 0, ACT 1, the seed GRAD 1. */
static void buildCeSingleLinearModel(fixture_t *f) {
    f->model[0] = makeLinear(4, 3, false);
    f->n = 1;
    f->lt = CROSS_ENTROPY;
    f->x = makeInput(&f->in, (size_t[]){1, 4}, 2);
}

/* Spec §7.6: with n = 1 under CE, LOSS_BACKWARD and zero BACKWARDs is a
 * complete stream, as in today's loop. */
void testFinishAcceptsCeWithOneLayerAndNoBackwardStep(void) {
    assertEveryStepAccepted(initArena, buildCeSingleLinearModel);
    assertEveryStepAccepted(initHeap, buildCeSingleLinearModel);
}

/* No LOSS_BACKWARD and no BACKWARD: complete when nothing trains. */
void testFinishAcceptsAnAllFrozenStream(void) {
    assertEveryStepAccepted(initArena, buildAllFrozenModel);
    assertEveryStepAccepted(initHeap, buildAllFrozenModel);
}

/* A row whose end leaves ACT 1 bound (R8 lifecycle: "end leaves no
 * non-borrowed wire resident"). */
static uint64_t g_strayBytes[1];

static void leakyEnd(rematScheduler_t *s) {
    passEnd(s);
    rematWireBind(s->wires, rematActId(s->wires, 1), (uint8_t *)g_strayBytes);
}

static const rematSchedulerFunctions_t g_leakyArena = {.name = "leaky-arena",
                                                       .begin = passBegin,
                                                       .next = passNext,
                                                       .done = passDone,
                                                       .end = leakyEnd,
                                                       .deinit = passDeinit};

void testReleasedExitsWhenARowLeavesAWireResident(void) {
    fixture_t f;
    buildF1Model(&f);
    rematScheduler_t s = initArena(&f, NULL);
    s.fns = &g_leakyArena;
    ASSERT_EXITS_WITH_OUTPUT(
        1, "remat[leaky-arena]: after rematEnd violates 'wire left resident after end: ACT 1'",
        driveCall(&f, &s));
    freeFixture(&f, &s);
}

/* A driver that checks the release before rematEnd. */
static void releasedBeforeEnd(fixture_t *f, rematScheduler_t *s) {
    uint32_t producedGen[rematCheckNumWires(s)];
    rematCheck_t c;
    rematCheckInit(&c, s, f->model, f->n, f->lt, producedGen);
    rematBegin(s, f->model, f->n, defaultLossConfig(f->lt), f->x);
    rematStep_t st;
    rematOperands_t op;
    while (rematNext(s, &st)) {
        rematCheckStep(&c, &st, &op);
        rematDone(s, &st);
    }
    rematCheckFinish(&c);
    rematCheckReleased(&c);
}

void testReleasedExitsWhileTheInputIsStillBound(void) {
    fixture_t f;
    buildF1Model(&f);
    rematScheduler_t s = initHeap(&f, NULL);
    ASSERT_EXITS_WITH_OUTPUT(1,
                             "violates 'wire left resident after end: ACT 0' (the input is still "
                             "bound)",
                             releasedBeforeEnd(&f, &s));
    freeFixture(&f, &s);
}

#ifdef ODT_MEM_PROFILE
/* Hard rule: the checker reserves nothing; its state is the caller's (spec
 * §2.3). Measured around every checker call on HEAP, whose rows do reserve. */
void testCheckReservesNothing(void) {
    fixture_t f;
    buildHarModel(&f);
    rematScheduler_t s = initHeap(&f, &g_liveness);
    size_t before = memProfileCurrentBytes();
    uint32_t producedGen[rematCheckNumWires(&s)];
    rematCheck_t c;
    rematCheckInit(&c, &s, f.model, f.n, f.lt, producedGen);
    TEST_ASSERT_EQUAL_size_t(before, memProfileCurrentBytes());
    rematBegin(&s, f.model, f.n, defaultLossConfig(f.lt), f.x);
    rematStep_t st;
    while (rematNext(&s, &st)) {
        size_t held = memProfileCurrentBytes();
        rematOperands_t op;
        rematCheckStep(&c, &st, &op);
        TEST_ASSERT_EQUAL_size_t(held, memProfileCurrentBytes());
        rematDone(&s, &st);
    }
    rematCheckFinish(&c);
    rematEnd(&s);
    rematCheckReleased(&c);
    TEST_ASSERT_EQUAL_size_t(before, memProfileCurrentBytes());
    freeFixture(&f, &s);
}
#endif

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(testNumWiresIsTheTablesWireCount);
    RUN_TEST(testNumWiresExitsOnASchedulerThatIsNotInitialised);
    RUN_TEST(testInitSetsTheCursorsFromTheLiveModelAndZeroesProducedGen);
    RUN_TEST(testCheckAcceptsEveryStepOnArenaHar);
    RUN_TEST(testCheckAcceptsEveryStepOnArenaF1);
    RUN_TEST(testCheckAcceptsEveryStepOnHeapHar);
    RUN_TEST(testCheckAcceptsEveryStepOnHeapF1);
    RUN_TEST(testOperandsAreResolvedPositionallyOnHar);
    RUN_TEST(testStepExitsOnAnUnknownStepKind);
    RUN_TEST(testStepExitsOnALayerOutOfRangeBeforeResolvingIt);
    RUN_TEST(testStepExitsOnALossStepWhoseLayerIsNotN);
    RUN_TEST(testStepExitsOnAForwardAfterTheLossForward);
    RUN_TEST(testStepExitsOnAForwardOutOfOrder);
    RUN_TEST(testStepExitsOnARowThatSkipsAStep);
    RUN_TEST(testStepExitsOnALossForwardBeforeTheLastForward);
    RUN_TEST(testStepExitsOnADuplicateLossForward);
    RUN_TEST(testStepExitsOnALossBackwardBeforeTheLossForward);
    RUN_TEST(testStepExitsOnALossBackwardWithoutATrainableLayer);
    RUN_TEST(testStepExitsOnADuplicateLossBackward);
    RUN_TEST(testStepExitsOnABackwardBeforeTheLossBackward);
    RUN_TEST(testStepExitsOnABackwardOutOfOrder);
    RUN_TEST(testStepExitsOnABackwardAfterTheLastOne);
    RUN_TEST(testASecondCallWithoutReInitExitsAtItsFirstStep);
    RUN_TEST(testStepExitsWhenAnInputIsNotResident);
    RUN_TEST(testStepExitsWhenAReadingBackwardsInputIsNotResident);
    RUN_TEST(testStepExitsWhenGradInIsNotResident);
    RUN_TEST(testStepExitsWhenTheOutputIsNotBound);
    RUN_TEST(testStepExitsOnAnOperandNeverProduced);
    RUN_TEST(testStepExitsOnAStaleOperand);
    RUN_TEST(testStepExitsOnAnInputWithoutBytes);
    RUN_TEST(testCheckAcceptsTheFrozenZooOnBothRows);
    RUN_TEST(testStepExitsWhenAForwardsInputAndOutputShareBytes);
    RUN_TEST(testStepExitsWhenABackwardsInputAndGradInShareBytes);
    RUN_TEST(testStepExitsWhenABackwardsInputAndOutputShareBytes);
    RUN_TEST(testStepExitsWhenABackwardsGradInAndOutputShareBytes);
    RUN_TEST(testStepExitsWhenALossBackwardsInputAndOutputShareBytes);
    RUN_TEST(testStepExitsOnNonResidencyBeforeSharedBytes);
    RUN_TEST(testStepAcceptsADeadInputSharingBytesWithTheOutput);
    RUN_TEST(testFinishExitsOnAMissingForward);
    RUN_TEST(testFinishExitsOnAMissingLossForward);
    RUN_TEST(testFinishExitsOnAMissingLossBackward);
    RUN_TEST(testFinishExitsOnAMissingBackward);
    RUN_TEST(testFinishExitsWhenARowEndsTheStreamEarly);
    RUN_TEST(testFinishExitsWhenARowEatsTheLastStepInternally);
    RUN_TEST(testFinishAcceptsCeWithOneLayerAndNoBackwardStep);
    RUN_TEST(testFinishAcceptsAnAllFrozenStream);
    RUN_TEST(testReleasedExitsWhenARowLeavesAWireResident);
    RUN_TEST(testReleasedExitsWhileTheInputIsStillBound);
#ifdef ODT_MEM_PROFILE
    RUN_TEST(testCheckReservesNothing);
#endif
    return UNITY_END();
}
