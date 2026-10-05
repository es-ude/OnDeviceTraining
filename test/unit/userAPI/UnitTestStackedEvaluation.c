#define SOURCE_FILE "UNIT_TEST_STACKED_EVALUATION"

#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>

#include "ArithmeticType.h"
#include "Conv1d.h"
#include "Conv1dApi.h"
#include "DataLoaderApi.h"
#include "DeathTest.h"
#include "FlattenApi.h"
#include "InferenceApi.h"
#include "LayerQuant.h"
#include "Linear.h"
#include "LinearApi.h"
#include "LossFunction.h"
#include "QuantizationApi.h"
#include "Relu.h"
#include "ReluApi.h"
#include "StorageApi.h"
#include "Tensor.h"
#include "TensorApi.h"
#include "TrainingLoopApi.h"
#include "unity.h"

/* #468: the evaluation entry points take a trailing size_t microBatchSize. */
_Static_assert(_Generic(&evaluationEpoch,
                   float (*)(layer_t **, size_t, lossFuncType_t, dataLoader_t *,
                             inferenceWithLossFn_t, reduction_t, size_t): 1,
                   default: 0),
               "evaluationEpoch must take a trailing size_t microBatchSize (#468)");
_Static_assert(_Generic(&evaluationEpochWithMetrics,
                   epochStats_t (*)(layer_t **, size_t, lossFuncType_t, dataLoader_t *,
                                    inferenceWithLossFn_t, reduction_t, size_t): 1,
                   default: 0),
               "evaluationEpochWithMetrics must take a trailing size_t microBatchSize (#468)");
_Static_assert(_Generic(&evaluationEpochWithReport,
                   classificationReport_t (*)(layer_t **, size_t, lossFuncType_t, dataLoader_t *,
                                              inferenceWithLossFn_t, size_t *, size_t, reduction_t,
                                              size_t): 1,
                   default: 0),
               "evaluationEpochWithReport must take a trailing size_t microBatchSize (#468)");
/* #468: evaluation inherits the training micro-batch unless this is set. */
_Static_assert(_Generic(((trainingRunOptions_t){0}).evalMicroBatchSize, size_t: 1, default: 0),
               "trainingRunOptions_t must carry a size_t evalMicroBatchSize (#468)");

void setUp(void) {}
void tearDown(void) {}

/* ---- fixture: 7 samples, item [4], one-hot label [3] ----------------------- */

#define N_EVAL 7
#define IN_F 4
#define HID 5
#define CLS 3
#define MODEL_SIZE 3

static quantization_t *g_q;
static layerQuant_t g_lq;
static tensor_t *g_items[N_EVAL];
static tensor_t *g_labels[N_EVAL];
static size_t g_datasetSize = N_EVAL;

static tensor_t *buildFloatTensor(const size_t *dims, size_t rank, const float *src) {
    size_t *ownedDims = reserveMemory(rank * sizeof(size_t));
    for (size_t i = 0; i < rank; i++) {
        ownedDims[i] = dims[i];
    }
    size_t *order = reserveMemory(rank * sizeof(size_t));
    setOrderOfDimsForNewTensor(rank, order);
    shape_t *shape = reserveMemory(sizeof(shape_t));
    setShape(shape, ownedDims, rank, order);
    tensor_t *t = initTensor(shape, quantizationInitFloat(), NULL);
    if (src != NULL) {
        tensorFillFromFloatBuffer(t, src, calcNumberOfElementsByTensor(t));
    }
    return t;
}

/* Non-uniform items and labels spread over all classes: uniform data would
 * make the per-row argmax and row-order mutations vacuous. */
static void initData(void) {
    for (size_t i = 0; i < N_EVAL; i++) {
        float x[IN_F];
        for (size_t j = 0; j < IN_F; j++) {
            x[j] = sinf(0.7f * (float)(i * IN_F + j) + 0.3f);
        }
        float y[CLS] = {0.f, 0.f, 0.f};
        y[(i * 2) % CLS] = 1.f;
        g_items[i] = buildFloatTensor((size_t[]){IN_F}, 1, x);
        g_labels[i] = buildFloatTensor((size_t[]){CLS}, 1, y);
    }
    g_datasetSize = N_EVAL;
}

static void freeFixtureData(void) {
    for (size_t i = 0; i < N_EVAL; i++) {
        freeTensor(g_labels[i]);
        freeTensor(g_items[i]);
    }
}

static sample_t *getSample(size_t id) {
    sample_t *s = reserveMemory(sizeof(sample_t));
    s->item = g_items[id];
    s->label = g_labels[id];
    return s;
}

static size_t getDatasetSize(void) {
    return g_datasetSize;
}

static void fillPattern(tensor_t *t, float phase) {
    float *d = (float *)t->data;
    size_t n = calcNumberOfElementsByTensor(t);
    for (size_t i = 0; i < n; i++) {
        d[i] = sinf(phase + 1.3f * (float)i);
    }
}

static void buildModel(layer_t **model) {
    model[0] = linearLayerInit(&(linearInit_t){.inFeatures = IN_F, .outFeatures = HID}, &g_lq);
    model[1] = reluLayerInit(&g_lq);
    model[2] = linearLayerInit(&(linearInit_t){.inFeatures = HID, .outFeatures = CLS}, &g_lq);
    fillPattern(model[0]->config->linear->weights->param, 0.31f);
    fillPattern(model[0]->config->linear->bias->param, 1.7f);
    fillPattern(model[2]->config->linear->weights->param, 2.9f);
    fillPattern(model[2]->config->linear->bias->param, 0.5f);
}

static void freeModel(layer_t **model) {
    freeLinearLayer(model[2]);
    freeReluLayer(model[1]);
    freeLinearLayer(model[0]);
}

/* ---- an eval loader with no batch fails before any getBatch --------------- */

static batch_t *getBatchMustNotRun(dataLoader_t *dl, size_t index) {
    (void)dl;
    (void)index;
    _exit(2);
}

static size_t oneSample(void) {
    return 1;
}

typedef enum { VIA_EPOCH, VIA_METRICS, VIA_REPORT } entryPoint_t;

static void evaluateEmptyLoader(entryPoint_t via) {
    dataLoader_t dl = {.getDatasetSize = oneSample, .batchSize = 2, .getBatch = getBatchMustNotRun};
    layer_t *model[MODEL_SIZE];
    buildModel(model);
    size_t cm[CLS * CLS];
    switch (via) {
    case VIA_EPOCH:
        (void)evaluationEpoch(model, MODEL_SIZE, MSE, &dl, inferenceWithLoss, REDUCTION_MEAN, 0);
        break;
    case VIA_METRICS:
        (void)evaluationEpochWithMetrics(model, MODEL_SIZE, MSE, &dl, inferenceWithLoss,
                                         REDUCTION_MEAN, 0);
        break;
    case VIA_REPORT:
        (void)evaluationEpochWithReport(model, MODEL_SIZE, MSE, &dl, inferenceWithLoss, cm, CLS,
                                        REDUCTION_MEAN, 0);
        break;
    }
}

/* exit 1 (not 2): the guard fires from sizes alone, before the index table
 * (which a dataset smaller than batchSize would overrun) is read. */
void testEmptyEvalLoaderFailsBeforeAnyGetBatch(void) {
    ASSERT_EXITS_WITH_FAILURE(evaluateEmptyLoader(VIA_EPOCH));
    ASSERT_EXITS_WITH_FAILURE(evaluateEmptyLoader(VIA_METRICS));
    ASSERT_EXITS_WITH_FAILURE(evaluateEmptyLoader(VIA_REPORT));
}

/* ---- the report's numClasses must match the label ------------------------ */

static void reportWithNumClasses(size_t numClasses, size_t m) {
    initData();
    layer_t *model[MODEL_SIZE];
    buildModel(model);
    dataLoader_t *dl = dataLoaderInit(getSample, getDatasetSize, 1, NULL, NULL, false, 0, true);
    size_t cm[(CLS + 1) * (CLS + 1)];
    (void)evaluationEpochWithReport(model, MODEL_SIZE, MSE, dl, inferenceWithLoss, cm, numClasses,
                                    REDUCTION_MEAN, m);
}

void testReportRejectsNumClassesNotMatchingTheLabel(void) {
    ASSERT_EXITS_WITH(0, reportWithNumClasses(CLS, 1));
    ASSERT_EXITS_WITH_FAILURE(reportWithNumClasses(CLS + 1, 1));
    ASSERT_EXITS_WITH_FAILURE(reportWithNumClasses(CLS - 1, 1));
    ASSERT_EXITS_WITH_FAILURE(reportWithNumClasses(0, 1));
    ASSERT_EXITS_WITH_FAILURE(reportWithNumClasses(0, 2));
}

/* ---- equivalence: row-independent model, stacked == per-sample ------------ */

static classificationReport_t runReport(size_t m, uint16_t batchSize, size_t *cm) {
    layer_t *model[MODEL_SIZE];
    buildModel(model);
    dataLoader_t *dl =
        dataLoaderInit(getSample, getDatasetSize, batchSize, NULL, NULL, false, 0, true);
    classificationReport_t r = evaluationEpochWithReport(
        model, MODEL_SIZE, MSE, dl, inferenceWithLoss, cm, CLS, REDUCTION_MEAN, m);
    freeDataLoader(dl);
    freeModel(model);
    return r;
}

static float runPlainLoss(size_t m, reduction_t reduction) {
    layer_t *model[MODEL_SIZE];
    buildModel(model);
    dataLoader_t *dl = dataLoaderInit(getSample, getDatasetSize, 1, NULL, NULL, false, 0, true);
    float loss = evaluationEpoch(model, MODEL_SIZE, MSE, dl, inferenceWithLoss, reduction, m);
    freeDataLoader(dl);
    freeModel(model);
    return loss;
}

static bool sameStats(epochStats_t a, epochStats_t b) {
    return memcmp(&a.accuracy, &b.accuracy, sizeof(float)) == 0 &&
           memcmp(&a.precision, &b.precision, sizeof(float)) == 0 &&
           memcmp(&a.recall, &b.recall, sizeof(float)) == 0 &&
           memcmp(&a.f1, &b.f1, sizeof(float)) == 0;
}

/* m in {2, 3, 7, 8} over N = 7: full chunks, a ragged tail, one exact chunk,
 * and m > N (one 7-row chunk). Counts are integers, so CM and the derived
 * rates must match exactly; the loss only by float summation order. */
void testStackedReportMatchesPerSampleForEveryChunkSize(void) {
    initData();
    const size_t ms[4] = {2, 3, 7, 8};
    size_t cm1[CLS * CLS];
    size_t cmM[4][CLS * CLS];
    classificationReport_t r1 = runReport(1, 1, cm1);
    classificationReport_t rM[4];
    for (size_t k = 0; k < 4; k++) {
        rM[k] = runReport(ms[k], 1, cmM[k]);
    }
    float sum1 = runPlainLoss(1, REDUCTION_SUM);
    float sumM = runPlainLoss(3, REDUCTION_SUM);
    freeFixtureData();

    size_t total = 0;
    size_t predictedClasses = 0;
    for (size_t p = 0; p < CLS; p++) {
        size_t rowSum = 0;
        for (size_t a = 0; a < CLS; a++) {
            rowSum += cm1[p * CLS + a];
        }
        total += rowSum;
        predictedClasses += (rowSum > 0);
    }
    TEST_ASSERT_EQUAL_size_t(N_EVAL, total);
    TEST_ASSERT_TRUE_MESSAGE(predictedClasses >= 2, "fixture must predict >= 2 classes");
    for (size_t k = 0; k < 4; k++) {
        TEST_ASSERT_EQUAL_MEMORY(cm1, cmM[k], sizeof(cm1));
        TEST_ASSERT_TRUE(sameStats(r1.stats, rM[k].stats));
        TEST_ASSERT_FLOAT_WITHIN(1e-5f * fabsf(r1.stats.loss) + 1e-6f, r1.stats.loss,
                                 rM[k].stats.loss);
    }
    TEST_ASSERT_FLOAT_WITHIN(1e-5f * fabsf(sum1) + 1e-6f, sum1, sumM);
}

/* Loader batchSize 2 over D = 7 streams N = 6 (dropLast);
 * m = 4 gathers ACROSS batch_t boundaries -> chunks 4 + 2. */
void testStackedSpansLoaderBatchesAndRespectsDropLast(void) {
    initData();
    size_t cm1[CLS * CLS];
    size_t cm4[CLS * CLS];
    classificationReport_t r1 = runReport(1, 2, cm1);
    classificationReport_t r4 = runReport(4, 2, cm4);
    freeFixtureData();
    size_t total = 0;
    for (size_t i = 0; i < CLS * CLS; i++) {
        total += cm4[i];
    }
    TEST_ASSERT_EQUAL_size_t(6, total);
    TEST_ASSERT_EQUAL_MEMORY(cm1, cm4, sizeof(cm1));
    TEST_ASSERT_TRUE(sameStats(r1.stats, r4.stats));
}

/* ---- no entry point gains a getBatch call ------------------------------- */

static size_t g_getBatchCalls;
static getBatchFn_t g_realGetBatch;

static batch_t *countingGetBatch(dataLoader_t *dl, size_t index) {
    g_getBatchCalls++;
    return g_realGetBatch(dl, index);
}

static size_t countGetBatch(entryPoint_t via, size_t m) {
    layer_t *model[MODEL_SIZE];
    buildModel(model);
    dataLoader_t *dl = dataLoaderInit(getSample, getDatasetSize, 1, NULL, NULL, false, 0, true);
    g_realGetBatch = dl->getBatch;
    dl->getBatch = countingGetBatch;
    g_getBatchCalls = 0;
    size_t cm[CLS * CLS];
    switch (via) {
    case VIA_EPOCH:
        (void)evaluationEpoch(model, MODEL_SIZE, MSE, dl, inferenceWithLoss, REDUCTION_MEAN, m);
        break;
    case VIA_METRICS:
        (void)evaluationEpochWithMetrics(model, MODEL_SIZE, MSE, dl, inferenceWithLoss,
                                         REDUCTION_MEAN, m);
        break;
    case VIA_REPORT:
        (void)evaluationEpochWithReport(model, MODEL_SIZE, MSE, dl, inferenceWithLoss, cm, CLS,
                                        REDUCTION_MEAN, m);
        break;
    }
    freeDataLoader(dl);
    freeModel(model);
    return g_getBatchCalls;
}

void testEntryPointsKeepTodaysGetBatchCallCount(void) {
    initData();
    size_t calls[2][3];
    const size_t ms[2] = {1, 3};
    for (size_t k = 0; k < 2; k++) {
        calls[k][0] = countGetBatch(VIA_EPOCH, ms[k]);
        calls[k][1] = countGetBatch(VIA_METRICS, ms[k]);
        calls[k][2] = countGetBatch(VIA_REPORT, ms[k]);
    }
    freeFixtureData();
    for (size_t k = 0; k < 2; k++) {
        TEST_ASSERT_EQUAL_size_t(N_EVAL, calls[k][0]);
        TEST_ASSERT_EQUAL_size_t(N_EVAL + 1, calls[k][1]); /* its existing numClasses peek */
        TEST_ASSERT_EQUAL_size_t(N_EVAL, calls[k][2]);
    }
}

/* ---- MEAN divides by the STREAMED count ---------------------------------- */

/* A replay-like loader: batch 0 carries one extra sample beyond batchSize. */
static batch_t *oneExtraSampleGetBatch(dataLoader_t *dl, size_t index) {
    batch_t *b = g_realGetBatch(dl, index);
    if (index == 0) {
        sample_t **grown = reserveMemory((b->size + 1) * sizeof(sample_t *));
        memcpy(grown, b->samples, b->size * sizeof(sample_t *));
        grown[b->size] = getSample(N_EVAL - 1);
        freeReservedMemory(b->samples);
        b->samples = grown;
        b->size += 1;
    }
    return b;
}

static float runGrownLoss(size_t m, reduction_t reduction) {
    layer_t *model[MODEL_SIZE];
    buildModel(model);
    dataLoader_t *dl = dataLoaderInit(getSample, getDatasetSize, 1, NULL, NULL, false, 0, true);
    g_realGetBatch = dl->getBatch;
    dl->getBatch = oneExtraSampleGetBatch;
    float loss = evaluationEpoch(model, MODEL_SIZE, MSE, dl, inferenceWithLoss, reduction, m);
    freeDataLoader(dl);
    freeModel(model);
    return loss;
}

void testStackedMeanDividesByStreamedCount(void) {
    initData();
    float mean1 = runGrownLoss(1, REDUCTION_MEAN);
    float mean3 = runGrownLoss(3, REDUCTION_MEAN);
    float sum3 = runGrownLoss(3, REDUCTION_SUM);
    freeFixtureData();
    TEST_ASSERT_FLOAT_WITHIN(1e-5f * fabsf(mean1) + 1e-6f, mean1, mean3);
    /* MSE's MEAN is per element (1/(N*F)), so SUM -> MEAN carries the class count. */
    TEST_ASSERT_FLOAT_WITHIN(1e-5f * fabsf(sum3) + 1e-6f, sum3 / (float)((N_EVAL + 1) * CLS),
                             mean3);
}

/* ---- fail-fast: stackability, forward gate, output contract -------------- */

static void metricsWith(inferenceWithLossFn_t fn, size_t m) {
    layer_t *model[MODEL_SIZE];
    buildModel(model);
    dataLoader_t *dl = dataLoaderInit(getSample, getDatasetSize, 1, NULL, NULL, false, 0, true);
    (void)evaluationEpochWithMetrics(model, MODEL_SIZE, MSE, dl, fn, REDUCTION_MEAN, m);
}

static inferenceStats_t *symOutputInference(layer_t **model, size_t n, tensor_t *in,
                                            tensor_t *label, lossFuncType_t f, reduction_t r,
                                            const trainingCall_t *call) {
    inferenceStats_t *s = inferenceWithLoss(model, n, in, label, f, r, call);
    s->output->quantization->type = SYM_INT32;
    return s;
}
static inferenceStats_t *extraRowInference(layer_t **model, size_t n, tensor_t *in, tensor_t *label,
                                           lossFuncType_t f, reduction_t r,
                                           const trainingCall_t *call) {
    inferenceStats_t *s = inferenceWithLoss(model, n, in, label, f, r, call);
    s->output->shape->dimensions[0] += 1;
    return s;
}
/* Same element count, rows moved out of axis 0: [1, rows * C]. */
static inferenceStats_t *flatRowInference(layer_t **model, size_t n, tensor_t *in, tensor_t *label,
                                          lossFuncType_t f, reduction_t r,
                                          const trainingCall_t *call) {
    inferenceStats_t *s = inferenceWithLoss(model, n, in, label, f, r, call);
    size_t *d = s->output->shape->dimensions;
    d[1] *= d[0];
    d[0] = 1;
    return s;
}
static inferenceStats_t *shortRowInference(layer_t **model, size_t n, tensor_t *in, tensor_t *label,
                                           lossFuncType_t f, reduction_t r,
                                           const trainingCall_t *call) {
    inferenceStats_t *s = inferenceWithLoss(model, n, in, label, f, r, call);
    s->output->shape->dimensions[1] -= 1;
    return s;
}
static inferenceStats_t *swappedOrderInference(layer_t **model, size_t n, tensor_t *in,
                                               tensor_t *label, lossFuncType_t f, reduction_t r,
                                               const trainingCall_t *call) {
    inferenceStats_t *s = inferenceWithLoss(model, n, in, label, f, r, call);
    size_t *o = s->output->shape->orderOfDimensions;
    size_t t = o[0];
    o[0] = o[1];
    o[1] = t;
    return s;
}
static inferenceStats_t *nullDataInference(layer_t **model, size_t n, tensor_t *in, tensor_t *label,
                                           lossFuncType_t f, reduction_t r,
                                           const trainingCall_t *call) {
    inferenceStats_t *s = inferenceWithLoss(model, n, in, label, f, r, call);
    s->output->data = NULL; /* leaks in the forked child only */
    return s;
}
static inferenceStats_t *nullOrderInference(layer_t **model, size_t n, tensor_t *in,
                                            tensor_t *label, lossFuncType_t f, reduction_t r,
                                            const trainingCall_t *call) {
    inferenceStats_t *s = inferenceWithLoss(model, n, in, label, f, r, call);
    s->output->shape->orderOfDimensions = NULL; /* leaks in the forked child only */
    return s;
}

static inferenceStats_t *nullStatsInference(layer_t **model, size_t n, tensor_t *in,
                                            tensor_t *label, lossFuncType_t f, reduction_t r,
                                            const trainingCall_t *call) {
    (void)inferenceWithLoss(model, n, in, label, f, r, call); /* leaks in the forked child only */
    return NULL;
}
static inferenceStats_t *nullShapeInference(layer_t **model, size_t n, tensor_t *in,
                                            tensor_t *label, lossFuncType_t f, reduction_t r,
                                            const trainingCall_t *call) {
    inferenceStats_t *s = inferenceWithLoss(model, n, in, label, f, r, call);
    s->output->shape = NULL; /* leaks in the forked child only */
    return s;
}

/* A stream that yields no sample at all (custom getBatch returning empty
 * batches) must fail fast instead of dividing MEAN loss and metrics by 0. */
static batch_t *emptyGetBatch(dataLoader_t *dl, size_t index) {
    (void)dl;
    (void)index;
    batch_t *b = reserveMemory(sizeof(batch_t));
    b->size = 0;
    b->samples = reserveMemory(sizeof(sample_t *));
    return b;
}

static void metricsOverEmptyStream(size_t m) {
    layer_t *model[MODEL_SIZE];
    buildModel(model);
    dataLoader_t *dl = dataLoaderInit(getSample, getDatasetSize, 1, NULL, NULL, false, 0, true);
    dl->getBatch = emptyGetBatch;
    (void)evaluationEpoch(model, MODEL_SIZE, MSE, dl, inferenceWithLoss, REDUCTION_MEAN, m);
}

void testEmptyStreamFailsFast(void) {
    initData();
    ASSERT_EXITS_WITH_FAILURE(metricsOverEmptyStream(1));
    ASSERT_EXITS_WITH_FAILURE(metricsOverEmptyStream(2));
    freeFixtureData();
}

static void metricsWithEmptyFirstBatch(void) {
    layer_t *model[MODEL_SIZE];
    buildModel(model);
    dataLoader_t *dl = dataLoaderInit(getSample, getDatasetSize, 1, NULL, NULL, false, 0, true);
    dl->getBatch = emptyGetBatch;
    (void)evaluationEpochWithMetrics(model, MODEL_SIZE, MSE, dl, inferenceWithLoss, REDUCTION_MEAN,
                                     1);
}

/* The numClasses peek reads samples[0] of getBatch(0): a loader that reports
 * a dataset but hands back an empty first batch must fail fast, not read out
 * of bounds. */
void testEmptyFirstBatchFailsFastInMetricsPeek(void) {
    initData();
    ASSERT_EXITS_WITH_FAILURE(metricsWithEmptyFirstBatch());
    freeFixtureData();
}

void testStackedRejectsMalformedInferenceOutput(void) {
    initData();
    ASSERT_EXITS_WITH(0, metricsWith(inferenceWithLoss, 2)); /* the fixture itself is valid */
    ASSERT_EXITS_WITH_FAILURE(metricsWith(symOutputInference, 2));
    ASSERT_EXITS_WITH_FAILURE(metricsWith(extraRowInference, 2));
    ASSERT_EXITS_WITH_FAILURE(metricsWith(flatRowInference, 2));
    ASSERT_EXITS_WITH_FAILURE(metricsWith(shortRowInference, 2));
    ASSERT_EXITS_WITH_FAILURE(metricsWith(swappedOrderInference, 2));
    ASSERT_EXITS_WITH_FAILURE(metricsWith(nullDataInference, 2));
    ASSERT_EXITS_WITH_FAILURE(metricsWith(nullOrderInference, 2));
    ASSERT_EXITS_WITH_FAILURE(metricsWith(nullStatsInference, 2));
    ASSERT_EXITS_WITH_FAILURE(metricsWith(nullShapeInference, 2));
    freeFixtureData();
}

static void metricsWithMismatchedSample(bool dtype) {
    if (dtype) {
        g_items[4]->quantization->type = SYM_INT32; /* child only */
    } else {
        freeTensor(g_items[4]);
        g_items[4] = buildFloatTensor((size_t[]){IN_F + 1}, 1, NULL);
    }
    metricsWith(inferenceWithLoss, 2);
}

/* The numClasses check on the stacked path: a larger numClasses would not crash there (the
 * counters are sized by it), it would silently mis-shape the matrix. */
void testStackedReportRejectsNumClassesNotMatchingTheLabel(void) {
    ASSERT_EXITS_WITH(0, reportWithNumClasses(CLS, 2));
    ASSERT_EXITS_WITH_FAILURE(reportWithNumClasses(CLS + 1, 2));
}

void testStackedRejectsUnstackableSamples(void) {
    initData();
    ASSERT_EXITS_WITH_FAILURE(metricsWithMismatchedSample(true));
    ASSERT_EXITS_WITH_FAILURE(metricsWithMismatchedSample(false));
    freeFixtureData();
}

/* Allows WithMetrics' numClasses peek (call 0), exits 2 on any later call:
 * proves the gate fires before the stream starts. */
static size_t g_gateCalls;
static batch_t *getBatchMustNotRunAfterGate(dataLoader_t *dl, size_t index) {
    if (g_gateCalls++ > 0) {
        _exit(2);
    }
    return g_realGetBatch(dl, index);
}

static void metricsWithSymForward(size_t m) {
    layer_t *model[MODEL_SIZE];
    buildModel(model);
    model[1]->config->relu->forwardMath.type = ARITH_SYM_INT32; /* child only */
    dataLoader_t *dl = dataLoaderInit(getSample, getDatasetSize, 1, NULL, NULL, false, 0, true);
    g_realGetBatch = dl->getBatch;
    dl->getBatch = getBatchMustNotRunAfterGate;
    (void)evaluationEpochWithMetrics(model, MODEL_SIZE, MSE, dl, inferenceWithLoss, REDUCTION_MEAN,
                                     m);
}

void testStackedForwardGateFiresBeforeTheStream(void) {
    initData();
    g_gateCalls = 0;
    ASSERT_EXITS_WITH_FAILURE(metricsWithSymForward(2));
    freeFixtureData();
}

/* The stacked-evaluation gate is forward-only: a FLOAT32 forward with a SYM
 * prop-loss wire (backward only) is fine. */
static void reportWithSymPropLoss(size_t m, size_t *cm, epochStats_t *out) {
    quantization_t *symQ = quantizationInitSymInt32(HALF_AWAY);
    layer_t *model[MODEL_SIZE];
    buildModel(model);
    quantization_t *saved = model[0]->config->linear->propLossQ;
    model[0]->config->linear->propLossQ = symQ;
    dataLoader_t *dl = dataLoaderInit(getSample, getDatasetSize, 1, NULL, NULL, false, 0, true);
    *out = evaluationEpochWithReport(model, MODEL_SIZE, MSE, dl, inferenceWithLoss, cm, CLS,
                                     REDUCTION_MEAN, m)
               .stats;
    model[0]->config->linear->propLossQ = saved;
    freeDataLoader(dl);
    freeModel(model);
    freeQuantization(symQ);
}

void testStackedAcceptsBackwardOnlyNonFloat32Fields(void) {
    initData();
    size_t cm1[CLS * CLS];
    size_t cm2[CLS * CLS];
    epochStats_t s1;
    epochStats_t s2;
    reportWithSymPropLoss(1, cm1, &s1);
    reportWithSymPropLoss(2, cm2, &s2);
    freeFixtureData();
    TEST_ASSERT_EQUAL_MEMORY(cm1, cm2, sizeof(cm1));
    TEST_ASSERT_TRUE(sameStats(s1, s2));
}

static tensor_t *g_convItems[N_EVAL];
static sample_t *getConvSample(size_t id) {
    sample_t *s = reserveMemory(sizeof(sample_t));
    s->item = g_convItems[id];
    s->label = g_labels[id];
    return s;
}

static void convReport(size_t m, size_t *cm, epochStats_t *out) {
    layer_t *model[4];
    model[0] =
        conv1dLayerInit(&(conv1dInit_t){.inChannels = 1, .outChannels = 2, .kernelSize = 3}, &g_lq);
    model[1] = reluLayerInit(&g_lq);
    model[2] = flattenLayerInit();
    model[3] = linearLayerInit(&(linearInit_t){.inFeatures = 8, .outFeatures = CLS}, &g_lq);
    fillPattern(model[0]->config->conv1d->weights->param, 0.9f);
    fillPattern(model[0]->config->conv1d->bias->param, 0.2f);
    fillPattern(model[3]->config->linear->weights->param, 1.1f);
    fillPattern(model[3]->config->linear->bias->param, 2.3f);
    dataLoader_t *dl = dataLoaderInit(getConvSample, getDatasetSize, 1, NULL, NULL, false, 0, true);
    *out =
        evaluationEpochWithReport(model, 4, MSE, dl, inferenceWithLoss, cm, CLS, REDUCTION_MEAN, m)
            .stats;
    freeDataLoader(dl);
    freeLinearLayer(model[3]);
    freeFlattenLayer(model[2]);
    freeReluLayer(model[1]);
    freeConv1dLayer(model[0]);
}

void testStackedConvFlattenPipelineMatchesPerSample(void) {
    initData();
    for (size_t i = 0; i < N_EVAL; i++) {
        float x[6];
        for (size_t j = 0; j < 6; j++) {
            x[j] = cosf(0.9f * (float)(i * 6 + j));
        }
        g_convItems[i] = buildFloatTensor((size_t[]){1, 6}, 2, x);
    }
    size_t cm1[CLS * CLS];
    size_t cm3[CLS * CLS];
    epochStats_t s1;
    epochStats_t s3;
    convReport(1, cm1, &s1);
    convReport(3, cm3, &s3);
    for (size_t i = 0; i < N_EVAL; i++) {
        freeTensor(g_convItems[i]);
    }
    freeFixtureData();
    TEST_ASSERT_EQUAL_MEMORY(cm1, cm3, sizeof(cm1));
    TEST_ASSERT_TRUE(sameStats(s1, s3));
    TEST_ASSERT_FLOAT_WITHIN(1e-5f * fabsf(s1.loss) + 1e-6f, s1.loss, s3.loss);
}

int main(void) {
    g_q = quantizationInitFloat();
    layerQuantInitUniform(&g_lq, g_q);
    UNITY_BEGIN();
    RUN_TEST(testEmptyEvalLoaderFailsBeforeAnyGetBatch);
    RUN_TEST(testEmptyFirstBatchFailsFastInMetricsPeek);
    RUN_TEST(testReportRejectsNumClassesNotMatchingTheLabel);
    RUN_TEST(testStackedReportMatchesPerSampleForEveryChunkSize);
    RUN_TEST(testStackedSpansLoaderBatchesAndRespectsDropLast);
    RUN_TEST(testEntryPointsKeepTodaysGetBatchCallCount);
    RUN_TEST(testStackedMeanDividesByStreamedCount);
    RUN_TEST(testEmptyStreamFailsFast);
    RUN_TEST(testStackedRejectsMalformedInferenceOutput);
    RUN_TEST(testStackedReportRejectsNumClassesNotMatchingTheLabel);
    RUN_TEST(testStackedRejectsUnstackableSamples);
    RUN_TEST(testStackedForwardGateFiresBeforeTheStream);
    RUN_TEST(testStackedAcceptsBackwardOnlyNonFloat32Fields);
    RUN_TEST(testStackedConvFlattenPipelineMatchesPerSample);
    int result = UNITY_END();
    freeQuantization(g_q);
    return result;
}
