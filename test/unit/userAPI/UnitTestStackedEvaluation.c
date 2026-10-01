#define SOURCE_FILE "UNIT_TEST_STACKED_EVALUATION"

#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>

#include "DataLoaderApi.h"
#include "DeathTest.h"
#include "InferenceApi.h"
#include "LayerQuant.h"
#include "Linear.h"
#include "LinearApi.h"
#include "LossFunction.h"
#include "QuantizationApi.h"
#include "ReluApi.h"
#include "StorageApi.h"
#include "Tensor.h"
#include "TensorApi.h"
#include "TrainingLoopApi.h"
#include "unity.h"

/* #468 D3: the evaluation entry points take a trailing size_t microBatchSize. */
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
/* #468 D1 */
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

/* ---- D9: an eval loader with no batch fails before any getBatch ----------- */

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

/* ---- D11: the report's numClasses must match the label ------------------- */

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
}

int main(void) {
    g_q = quantizationInitFloat();
    layerQuantInitUniform(&g_lq, g_q);
    UNITY_BEGIN();
    RUN_TEST(testEmptyEvalLoaderFailsBeforeAnyGetBatch);
    RUN_TEST(testReportRejectsNumClassesNotMatchingTheLabel);
    int result = UNITY_END();
    freeQuantization(g_q);
    return result;
}
