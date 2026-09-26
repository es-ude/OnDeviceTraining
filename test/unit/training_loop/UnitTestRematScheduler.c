#define SOURCE_FILE "UNIT_TEST_REMAT_SCHEDULER"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
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
#include "RematPlan.h"
#include "RematScheduler.h"
#include "SoftmaxApi.h"
#include "StorageApi.h"
#include "Tensor.h"
#include "unity.h"

void setUp(void) {}
void tearDown(void) {}

/* Fixture helpers copied from UnitTestRematPlan.c (plan Assumption 14).
 * Fixture layers borrow their wire templates, so one FLOAT32 template
 * outlives every fixture model. */
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

static layer_t *makeRelu(quantization_t *q) {
    return reluLayerInit(&(layerQuant_t){.outputQ = q, .propLossQ = q});
}

static layer_t *makeSoftmax(void) {
    layerQuant_t lq;
    layerQuantInitUniform(&lq, &g_floatQ);
    return softmaxLayerInit(&lq);
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

/* A borrowed input header on the caller's stack. Init and bind never read its
 * data, so data stays NULL. */
#define TEST_MAX_RANK 4
typedef struct inputLike {
    size_t dims[TEST_MAX_RANK];
    size_t order[TEST_MAX_RANK];
    shape_t shape;
    tensor_t tensor;
} inputLike_t;

static tensor_t *makeInput(inputLike_t *in, const size_t *dims, size_t rank, quantization_t *q) {
    TEST_ASSERT_TRUE(rank <= TEST_MAX_RANK);
    for (size_t d = 0; d < rank; d++) {
        in->dims[d] = dims[d];
        in->order[d] = d;
    }
    in->shape = (shape_t){
        .numberOfDimensions = rank, .dimensions = in->dims, .orderOfDimensions = in->order};
    in->tensor = (tensor_t){.data = NULL, .shape = &in->shape, .quantization = q, .sparsity = NULL};
    return &in->tensor;
}

/* examples/har_classifier/train_c.c:178-216 (B = 1). */
#define HAR_N 12
static void buildHar(layer_t **model, bool freezeConvs) {
    layerQuant_t lq;
    layerQuantInitUniform(&lq, &g_floatQ);
    trainable_t conv = freezeConvs ? TRAINABLE_FALSE : TRAINABLE_DEFAULT;
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
}

static tensor_t *makeHarInput(inputLike_t *in) {
    return makeInput(in, (size_t[]){1, 9, 128}, 3, &g_floatQ);
}

static uint32_t nextRandom(uint32_t *state) { /* xorshift32, test-local */
    uint32_t v = *state;
    v ^= v << 13;
    v ^= v >> 17;
    v ^= v << 5;
    *state = v;
    return v;
}

static const rematPlanSpec_t g_liveness = {.policy = REMAT_PLAN_LIVENESS};

/* One fixture model and its borrowed input. The BFP template of the F1 model
 * (Task 3) lives here too, because the model's layers borrow it: the fixture
 * must not move while the model is alive. */
typedef struct arenaFixture {
    layer_t *model[HAR_N];
    size_t n;
    lossFuncType_t lt;
    inputLike_t in;
    tensor_t *x;
    uint8_t bfpExponent[1];
    bfpQConfig_t bfpQc;
    quantization_t bfpQ;
} arenaFixture_t;

static void buildHarModel(arenaFixture_t *f) {
    buildHar(f->model, false);
    f->n = HAR_N;
    f->lt = CROSS_ENTROPY;
    f->x = makeHarInput(&f->in);
}

static rematScheduler_t initArena(arenaFixture_t *f, const rematPlanSpec_t *spec) {
    rematScheduler_t s;
    TEST_ASSERT_TRUE(rematArenaInit(&s, f->model, f->n, defaultLossConfig(f->lt), f->x, spec));
    TEST_ASSERT_NOT_NULL(s.wires);
    TEST_ASSERT_NOT_NULL(s.plan);
    return s;
}

static void freeFixture(arenaFixture_t *f, rematScheduler_t *s) {
    rematSchedulerDeinit(s);
    freeModel(f->model, f->n);
}

/* ---- init builds the shared table and plan; the report (spec §5.1, §12.2 item 10) ---- */

void testArenaInitBuildsTheTableAndThePlan(void) {
    arenaFixture_t f;
    buildHarModel(&f);
    rematScheduler_t s = initArena(&f, NULL);
    TEST_ASSERT_EQUAL_INT(REMAT_ARENA, s.type);
    TEST_ASSERT_NOT_NULL(s.wires);
    TEST_ASSERT_NOT_NULL(s.plan);
    TEST_ASSERT_EQUAL_size_t(HAR_N, s.wires->modelSize);
    TEST_ASSERT_EQUAL_INT(REMAT_PLAN_STORE_ALL, s.plan->policy);
    freeFixture(&f, &s);
}

void testReportEchoesTypePolicyAndPlanFacts(void) {
    arenaFixture_t f;
    buildHarModel(&f);
    rematScheduler_t s = initArena(&f, &g_liveness);
    rematReport_t r;
    rematSchedulerReport(&s, &r);
    TEST_ASSERT_EQUAL_INT(REMAT_ARENA, r.type);
    TEST_ASSERT_EQUAL_INT(REMAT_PLAN_LIVENESS, r.policy);
    TEST_ASSERT_TRUE(r.planned);
    TEST_ASSERT_EQUAL_size_t(25, r.numSteps); /* §4.3: 12 + 1 + 1 + 11 */
    freeFixture(&f, &s);
}

static size_t reportedPeakOnHar(const rematPlanSpec_t *spec) {
    arenaFixture_t f;
    buildHarModel(&f);
    rematScheduler_t s = initArena(&f, spec);
    rematReport_t r;
    rematSchedulerReport(&s, &r);
    freeFixture(&f, &s);
    return r.peakLiveBytes;
}

/* Scan-model pins (spec §1, §12.2 item 10), read from the report. */
void testReportPeakLiveBytesHarStoreAllIs74288(void) {
    TEST_ASSERT_EQUAL_size_t(74288, reportedPeakOnHar(NULL));
}

void testReportPeakLiveBytesHarLivenessIs49152(void) {
    TEST_ASSERT_EQUAL_size_t(49152, reportedPeakOnHar(&g_liveness));
}

/* RF3: never initialised, failed before the plan existed, or deinitialised --
 * the report reads nothing through the NULL table or plan. */
void testReportOnAZeroedSchedulerIsEmpty(void) {
    rematScheduler_t s = {0};
    rematReport_t r;
    memset(&r, 0xA5, sizeof r);
    rematSchedulerReport(&s, &r);
    TEST_ASSERT_FALSE(r.planned);
    TEST_ASSERT_EQUAL_size_t(0, r.numSteps);
    TEST_ASSERT_EQUAL_size_t(0, r.peakLiveBytes);
    TEST_ASSERT_EQUAL_size_t(0, r.metadataBytes);
}

void testReportMetadataCountsTheResidentBlocks(void) {
    arenaFixture_t f;
    buildHarModel(&f);
    rematScheduler_t s = initArena(&f, NULL);
    rematReport_t r;
    rematSchedulerReport(&s, &r);
    TEST_ASSERT_EQUAL_size_t(s.wires->slabBytes + s.plan->blockBytes, r.metadataBytes);
    freeFixture(&f, &s);
}

void testDeinitIsNullSafeAndIdempotent(void) {
    rematSchedulerDeinit(NULL);
    arenaFixture_t f;
    buildHarModel(&f);
    rematScheduler_t s = initArena(&f, NULL);
    rematSchedulerDeinit(&s);
    TEST_ASSERT_NULL(s.wires);
    TEST_ASSERT_NULL(s.plan);
    rematSchedulerDeinit(&s);
    freeModel(f.model, f.n);
}

#ifdef ODT_MEM_PROFILE
/* The live-byte counter is real only under ODT_MEM_PROFILE (unit_test_debug,
 * asan, ubsan); the plain unit_test preset compiles this out. */
void testDeinitReturnsEveryInitBlock(void) {
    arenaFixture_t f;
    buildHarModel(&f);
    size_t before = memProfileCurrentBytes();
    rematScheduler_t s = initArena(&f, &g_liveness);
    TEST_ASSERT_TRUE(memProfileCurrentBytes() > before);
    rematSchedulerDeinit(&s);
    TEST_ASSERT_EQUAL_size_t(before, memProfileCurrentBytes());
    freeModel(f.model, f.n);
}
#endif

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(testArenaInitBuildsTheTableAndThePlan);
    RUN_TEST(testReportEchoesTypePolicyAndPlanFacts);
    RUN_TEST(testReportPeakLiveBytesHarStoreAllIs74288);
    RUN_TEST(testReportPeakLiveBytesHarLivenessIs49152);
    RUN_TEST(testReportOnAZeroedSchedulerIsEmpty);
    RUN_TEST(testReportMetadataCountsTheResidentBlocks);
    RUN_TEST(testDeinitIsNullSafeAndIdempotent);
#ifdef ODT_MEM_PROFILE
    RUN_TEST(testDeinitReturnsEveryInitBlock);
#endif
    return UNITY_END();
}
