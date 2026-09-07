#define SOURCE_FILE "UnitTestOdtHook"

#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>

#include "ArithmeticType.h"
#include "BorrowedLayer.h" /* freeLinearLayerShellOnly (static inline, test support) */
#include "CalculateGradsSequential.h"
#include "Common.h"
#include "DataLoaderApi.h"
#include "Dataset.h"
#include "Layer.h"
#include "LayerQuant.h"
#include "LinearApi.h"
#include "LossFunction.h"
#include "OdtHook.h"
#include "Optimizer.h"
#include "OptimizerApi.h"
#include "QuantizationApi.h"
#include "SgdApi.h"
#include "SoftmaxApi.h"
#include "StateDictApi.h"
#include "StorageApi.h"
#include "Tensor.h"
#include "TensorApi.h"
#include "TraceApi.h"
#include "TrainingEpochDefault.h"
#include "unity.h"

_Static_assert(ODT_EVENT_FORWARD_BEGIN == 0 && ODT_EVENT_FORWARD_END == 1 &&
                   ODT_EVENT_BACKWARD_BEGIN == 2 && ODT_EVENT_BACKWARD_END == 3 &&
                   ODT_EVENT_OPTIMIZER_BEGIN == 4 && ODT_EVENT_OPTIMIZER_END == 5,
               "odtEvent_t values 0..5 in this order are a wire contract with external profilers");

void setUp(void) {}
void tearDown(void) {}

/* One shared log for hook events AND trace-sink probes: the interleaving
 * (loss backward inside BACKWARD, layer forwards inside FORWARD) is the
 * contract, not just the event count. Callbacks never assert (Unity longjmps
 * out of a callback would leak the fixture); overflow is caught by the count
 * assertion instead. */
#define MAX_LOG 64
typedef enum { LOG_HOOK, LOG_SINK } logKind_t;
typedef struct {
    logKind_t kind;
    odtEvent_t event; /* LOG_HOOK */
    char phase[16];   /* LOG_SINK */
    void *ctx;        /* LOG_HOOK: the ctx handed back by the hook */
} logEntry_t;
static logEntry_t g_log[MAX_LOG];
static size_t g_logCount;
static int g_ctxToken; /* its address is the opaque ctx under test */

static void resetLog(void) {
    g_logCount = 0;
    memset(g_log, 0, sizeof(g_log));
}

static void recordingHook(void *ctx, odtEvent_t event) {
    if (g_logCount < MAX_LOG) {
        g_log[g_logCount].kind = LOG_HOOK;
        g_log[g_logCount].event = event;
        g_log[g_logCount].ctx = ctx;
    }
    g_logCount++;
}

static void recordingSink(void *ctx, size_t layerIdx, layerType_t type, const char *phase,
                          tensor_t *tensor) {
    (void)ctx;
    (void)layerIdx;
    (void)type;
    (void)tensor;
    if (g_logCount < MAX_LOG) {
        g_log[g_logCount].kind = LOG_SINK;
        snprintf(g_log[g_logCount].phase, sizeof(g_log[g_logCount].phase), "%s", phase);
    }
    g_logCount++;
}

/* Asserts g_log[i] is the hook event `expected` carrying &g_ctxToken. */
static void assertHookAt(size_t i, odtEvent_t expected) {
    TEST_ASSERT_EQUAL_INT_MESSAGE(LOG_HOOK, g_log[i].kind, "expected a hook event at this index");
    TEST_ASSERT_EQUAL_INT(expected, g_log[i].event);
    TEST_ASSERT_EQUAL_PTR(&g_ctxToken, g_log[i].ctx);
}

static void assertSinkAt(size_t i, const char *phase) {
    TEST_ASSERT_EQUAL_INT_MESSAGE(LOG_SINK, g_log[i].kind, "expected a sink probe at this index");
    TEST_ASSERT_EQUAL_STRING(phase, g_log[i].phase);
}

/* The never-installed default: the slot is zero-initialized static state, so
 * a fire before ANY odtHookSet is silent. Must be the FIRST test main() runs
 * -- every later test installs a hook. */
void testFireIsSilentBeforeAnyInstall(void) {
    resetLog();
    odtHookFire(ODT_EVENT_FORWARD_BEGIN);
    TEST_ASSERT_EQUAL_size_t(0, g_logCount);
}

/* [1,2] float32 tensor from a stack buffer (data is COPIED into the tensor). */
static tensor_t *makeRowVec2(float a, float b) {
    size_t *dims = reserveMemory(2 * sizeof(size_t));
    dims[0] = 1;
    dims[1] = 2;
    size_t *order = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, order);
    shape_t *shape = reserveMemory(sizeof(shape_t));
    setShape(shape, dims, 2, order);
    tensor_t *t = initTensor(shape, quantizationInitFloat(), NULL);
    float vals[2] = {a, b};
    tensorFillFromFloatBuffer(t, vals, 2);
    return t;
}

/* Linear(2->2) + Softmax with known weights, CE loss. `q` (caller-owned, freed
 * by the caller AFTER the layers) is the uniform layerQuant template. */
static void buildLinearSoftmaxModel(layer_t *model[2], quantization_t *q, trainable_t trainable) {
    layerQuant_t lq;
    layerQuantInitUniform(&lq, q);
    model[0] = linearLayerInit(
        &(linearInit_t){.inFeatures = 2, .outFeatures = 2, .trainable = trainable}, &lq);
    model[1] = softmaxLayerInit(&lq);
    float W[4] = {0.1f, 0.2f, 0.3f, 0.4f};
    float B[2] = {0.0f, 0.0f};
    modelLoadStateDict(model, 2,
                       (stateDictEntry_t[]){{.name = "fc", .weightData = W, .biasData = B}}, 1);
}

static lossConfig_t ceMeanLoss(void) {
    return (lossConfig_t){
        .funcType = CROSS_ENTROPY, .backwardReduction = REDUCTION_MEAN, .classWeights = NULL};
}

void testCalculateGradsFiresForwardPairThenBackwardPair(void) {
    resetLog();
    quantization_t *q = quantizationInitFloat();
    layer_t *model[2];
    buildLinearSoftmaxModel(model, q, TRAINABLE_DEFAULT);
    tensor_t *x = makeRowVec2(1.0f, 1.0f);
    tensor_t *label = makeRowVec2(1.0f, 0.0f);

    odtHookSet(recordingHook, &g_ctxToken);
    trainingStats_t *stats =
        calculateGradsSequential(model, 2, ceMeanLoss(), REDUCTION_MEAN, x, label);
    odtHookSet(NULL, NULL);

    size_t count = g_logCount;
    freeTrainingStats(stats);
    freeTensor(x);
    freeTensor(label);
    freeLinearLayer(model[0]);
    freeSoftmaxLayer(model[1]);
    freeQuantization(q);

    TEST_ASSERT_EQUAL_size_t(4, count);
    assertHookAt(0, ODT_EVENT_FORWARD_BEGIN);
    assertHookAt(1, ODT_EVENT_FORWARD_END);
    assertHookAt(2, ODT_EVENT_BACKWARD_BEGIN);
    assertHookAt(3, ODT_EVENT_BACKWARD_END);
}

/* tracedGrads shares calculateGradsImpl, so it must fire the same four events;
 * its sink probes pin the phase spans: both layer "fwd" probes inside FORWARD,
 * "lossgrad" (the loss backward) and the one "agrad" inside BACKWARD. With CE
 * the Softmax backward is skipped (combined shortcut), so exactly one agrad
 * fires, for the Linear layer. */
void testTracedGradsInterleavesProbesInsidePhases(void) {
    resetLog();
    quantization_t *q = quantizationInitFloat();
    layer_t *model[2];
    buildLinearSoftmaxModel(model, q, TRAINABLE_DEFAULT);
    tensor_t *x = makeRowVec2(1.0f, 1.0f);
    tensor_t *label = makeRowVec2(1.0f, 0.0f);

    odtHookSet(recordingHook, &g_ctxToken);
    trainingStats_t *stats =
        tracedGrads(model, 2, ceMeanLoss(), REDUCTION_MEAN, x, label, recordingSink, NULL);
    odtHookSet(NULL, NULL);

    size_t count = g_logCount;
    freeTrainingStats(stats);
    freeTensor(x);
    freeTensor(label);
    freeLinearLayer(model[0]);
    freeSoftmaxLayer(model[1]);
    freeQuantization(q);

    TEST_ASSERT_EQUAL_size_t(8, count);
    assertHookAt(0, ODT_EVENT_FORWARD_BEGIN);
    assertSinkAt(1, "fwd");
    assertSinkAt(2, "fwd");
    assertHookAt(3, ODT_EVENT_FORWARD_END);
    assertHookAt(4, ODT_EVENT_BACKWARD_BEGIN);
    assertSinkAt(5, "lossgrad");
    assertSinkAt(6, "agrad");
    assertHookAt(7, ODT_EVENT_BACKWARD_END);
}

/* All-frozen model: backward is skipped entirely (#380 PR2), yet the BACKWARD
 * pair still fires -- the per-call event count is a constant for external
 * occurrence counting. */
void testAllFrozenModelStillFiresBothPairs(void) {
    resetLog();
    quantization_t *q = quantizationInitFloat();
    layer_t *model[2];
    buildLinearSoftmaxModel(model, q, TRAINABLE_FALSE);
    tensor_t *x = makeRowVec2(1.0f, 1.0f);
    tensor_t *label = makeRowVec2(1.0f, 0.0f);

    odtHookSet(recordingHook, &g_ctxToken);
    trainingStats_t *stats =
        tracedGrads(model, 2, ceMeanLoss(), REDUCTION_MEAN, x, label, recordingSink, NULL);
    odtHookSet(NULL, NULL);

    size_t count = g_logCount;
    freeTrainingStats(stats);
    freeTensor(x);
    freeTensor(label);
    freeLinearLayer(model[0]);
    freeSoftmaxLayer(model[1]);
    freeQuantization(q);

    /* fwd, fwd probes only -- no lossgrad/agrad -- plus the four hook events. */
    TEST_ASSERT_EQUAL_size_t(6, count);
    assertHookAt(0, ODT_EVENT_FORWARD_BEGIN);
    assertSinkAt(1, "fwd");
    assertSinkAt(2, "fwd");
    assertHookAt(3, ODT_EVENT_FORWARD_END);
    assertHookAt(4, ODT_EVENT_BACKWARD_BEGIN);
    assertHookAt(5, ODT_EVENT_BACKWARD_END);
}

/* The spec's acceptance test: one calculateGradsSequential + one optimizer
 * step yield the six events, in enum order, each carrying the installed ctx. */
void testSixEventsFireInOrderForOneGradsCallPlusOneStep(void) {
    resetLog();
    quantization_t *q = quantizationInitFloat();
    layer_t *model[2];
    buildLinearSoftmaxModel(model, q, TRAINABLE_DEFAULT);
    tensor_t *x = makeRowVec2(1.0f, 1.0f);
    tensor_t *label = makeRowVec2(1.0f, 0.0f);
    quantization_t *momentumQ = quantizationInitFloat();
    optimizer_t *optim =
        sgdMCreateOptim(0.1f, 0.0f, 0.0f, model, 2, momentumQ,
                        (arithmetic_t){.type = ARITH_FLOAT32, .roundingMode = HALF_AWAY});

    odtHookSet(recordingHook, &g_ctxToken);
    trainingStats_t *stats =
        calculateGradsSequential(model, 2, ceMeanLoss(), REDUCTION_MEAN, x, label);
    optimizerStep(optim);
    odtHookSet(NULL, NULL);

    size_t count = g_logCount;
    freeTrainingStats(stats);
    freeTensor(x);
    freeTensor(label);
    freeOptim(optim); /* frees the collected Linear params -> shell-only below */
    freeLinearLayerShellOnly(model[0]);
    freeSoftmaxLayer(model[1]);
    freeQuantization(momentumQ);
    freeQuantization(q);

    TEST_ASSERT_EQUAL_size_t(6, count);
    for (size_t i = 0; i < 6; i++) {
        assertHookAt(i, (odtEvent_t)i);
    }
}

/* Same step with the hook removed again after a real install: nothing fires
 * -- pins the disable path end to end, not just odtHookFire in isolation. */
void testNothingFiresWhenHookUnset(void) {
    resetLog();
    quantization_t *q = quantizationInitFloat();
    layer_t *model[2];
    buildLinearSoftmaxModel(model, q, TRAINABLE_DEFAULT);
    tensor_t *x = makeRowVec2(1.0f, 1.0f);
    tensor_t *label = makeRowVec2(1.0f, 0.0f);
    quantization_t *momentumQ = quantizationInitFloat();
    optimizer_t *optim =
        sgdMCreateOptim(0.1f, 0.0f, 0.0f, model, 2, momentumQ,
                        (arithmetic_t){.type = ARITH_FLOAT32, .roundingMode = HALF_AWAY});

    odtHookSet(recordingHook, &g_ctxToken);
    odtHookSet(NULL, NULL);
    trainingStats_t *stats =
        calculateGradsSequential(model, 2, ceMeanLoss(), REDUCTION_MEAN, x, label);
    optimizerStep(optim);

    size_t count = g_logCount;
    freeTrainingStats(stats);
    freeTensor(x);
    freeTensor(label);
    freeOptim(optim);
    freeLinearLayerShellOnly(model[0]);
    freeSoftmaxLayer(model[1]);
    freeQuantization(momentumQ);
    freeQuantization(q);

    TEST_ASSERT_EQUAL_size_t(0, count);
}

/* trainingRun's step runs inside trainingEpochDefault: one 2-sample batch
 * must yield two FORWARD/BACKWARD quads and then exactly one OPTIMIZER pair
 * -- i.e. the epoch loop steps through optimizerStep, not the raw vtable. */
static tensor_t *g_epochItems[2];
static tensor_t *g_epochLabels[2];
static tensorArray_t g_epochItemsArr;
static tensorArray_t g_epochLabelsArr;
static dataset_t g_epochDataset;

static sample_t *getEpochSample(size_t id) {
    sample_t *s = reserveMemory(sizeof(sample_t));
    s->item = g_epochDataset.items->array[id];
    s->label = g_epochDataset.labels->array[id];
    return s;
}

static size_t getEpochDatasetSize(void) {
    return g_epochDataset.items->size;
}

void testTrainingEpochDefaultStepsThroughOptimizerStep(void) {
    resetLog();
    g_epochItems[0] = makeRowVec2(5.0f, 1.0f);
    g_epochItems[1] = makeRowVec2(1.0f, 5.0f);
    g_epochLabels[0] = makeRowVec2(1.0f, 0.0f);
    g_epochLabels[1] = makeRowVec2(0.0f, 1.0f);
    g_epochItemsArr.array = g_epochItems;
    g_epochItemsArr.size = 2;
    g_epochLabelsArr.array = g_epochLabels;
    g_epochLabelsArr.size = 2;
    g_epochDataset.items = &g_epochItemsArr;
    g_epochDataset.labels = &g_epochLabelsArr;

    quantization_t *q = quantizationInitFloat();
    layer_t *model[2];
    buildLinearSoftmaxModel(model, q, TRAINABLE_DEFAULT);
    quantization_t *momentumQ = quantizationInitFloat();
    optimizer_t *optim =
        sgdMCreateOptim(0.1f, 0.0f, 0.0f, model, 2, momentumQ,
                        (arithmetic_t){.type = ARITH_FLOAT32, .roundingMode = HALF_AWAY});
    dataLoader_t *dl = dataLoaderInit(getEpochSample, getEpochDatasetSize, /*batchSize*/ 2, NULL,
                                      NULL, /*shuffle*/ false, /*seed*/ 0, /*dropLast*/ true);

    odtHookSet(recordingHook, &g_ctxToken);
    float epochLoss = trainingEpochDefault(model, 2, ceMeanLoss(), dl, optim,
                                           calculateGradsSequential, REDUCTION_MEAN);
    odtHookSet(NULL, NULL);

    size_t count = g_logCount;
    bool lossPositive = epochLoss > 0.0f;
    freeDataLoader(dl);
    freeOptim(optim);
    freeLinearLayerShellOnly(model[0]);
    freeSoftmaxLayer(model[1]);
    freeQuantization(momentumQ);
    freeQuantization(q);
    for (size_t i = 0; i < 2; i++) {
        freeTensor(g_epochItems[i]);
        freeTensor(g_epochLabels[i]);
    }

    TEST_ASSERT_TRUE(lossPositive);
    TEST_ASSERT_EQUAL_size_t(10, count);
    for (size_t s = 0; s < 2; s++) {
        assertHookAt(4 * s + 0, ODT_EVENT_FORWARD_BEGIN);
        assertHookAt(4 * s + 1, ODT_EVENT_FORWARD_END);
        assertHookAt(4 * s + 2, ODT_EVENT_BACKWARD_BEGIN);
        assertHookAt(4 * s + 3, ODT_EVENT_BACKWARD_END);
    }
    assertHookAt(8, ODT_EVENT_OPTIMIZER_BEGIN);
    assertHookAt(9, ODT_EVENT_OPTIMIZER_END);
}

void testFireHandsEventAndCtxToInstalledHook(void) {
    resetLog();
    odtHookSet(recordingHook, &g_ctxToken);
    odtHookFire(ODT_EVENT_OPTIMIZER_END);
    odtHookSet(NULL, NULL);

    TEST_ASSERT_EQUAL_size_t(1, g_logCount);
    assertHookAt(0, ODT_EVENT_OPTIMIZER_END);
}

void testFireIsSilentAfterSetNull(void) {
    resetLog();
    odtHookSet(recordingHook, &g_ctxToken);
    odtHookSet(NULL, NULL); /* the disable path, exercised AFTER a real install */
    odtHookFire(ODT_EVENT_FORWARD_BEGIN);

    TEST_ASSERT_EQUAL_size_t(0, g_logCount);
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(testFireIsSilentBeforeAnyInstall);
    RUN_TEST(testFireHandsEventAndCtxToInstalledHook);
    RUN_TEST(testFireIsSilentAfterSetNull);
    RUN_TEST(testCalculateGradsFiresForwardPairThenBackwardPair);
    RUN_TEST(testTracedGradsInterleavesProbesInsidePhases);
    RUN_TEST(testAllFrozenModelStillFiresBothPairs);
    RUN_TEST(testSixEventsFireInOrderForOneGradsCallPlusOneStep);
    RUN_TEST(testNothingFiresWhenHookUnset);
    RUN_TEST(testTrainingEpochDefaultStepsThroughOptimizerStep);
    return UNITY_END();
}
