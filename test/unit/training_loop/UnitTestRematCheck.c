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
    rematEnd(s);
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
    return UNITY_END();
}
