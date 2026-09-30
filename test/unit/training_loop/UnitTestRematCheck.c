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

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(testNumWiresIsTheTablesWireCount);
    RUN_TEST(testNumWiresExitsOnASchedulerThatIsNotInitialised);
    RUN_TEST(testInitSetsTheCursorsFromTheLiveModelAndZeroesProducedGen);
    return UNITY_END();
}
