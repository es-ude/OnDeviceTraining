#define SOURCE_FILE "UNIT_TEST_MICRO_BATCH_STACKING"

#include <ctype.h>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "AdaptivePool1dApi.h"
#include "ArithmeticType.h"
#include "BatchView.h"
#include "BsScheduler.h"
#include "CalculateGradsSequential.h"
#include "Conv1dApi.h"
#include "Conv1dTransposedApi.h"
#include "DataLoaderApi.h"
#include "Dataset.h"
#include "DeathTest.h"
#include "DropoutApi.h"
#include "FlattenApi.h"
#include "GroupNormApi.h"
#include "LayerConfigAccess.h"
#include "LayerNormApi.h"
#include "LayerQuant.h"
#include "LayerWeightsApi.h"
#include "Linear.h"
#include "LinearApi.h"
#include "LossFunction.h"
#include "Optimizer.h"
#include "OptimizerApi.h"
#include "Pool1dApi.h"
#include "QuantLayerApi.h"
#include "QuantizationApi.h"
#include "RNG.h"
#include "ReluApi.h"
#include "SgdApi.h"
#include "SoftmaxApi.h"
#include "StorageApi.h"
#include "Tensor.h"
#include "TensorApi.h"
#include "TrainingBatchDefault.h"
#include "TrainingEpochDefault.h"
#include "TrainingLoopApi.h"
#include "expected_microbatch.h"
#include "unity.h"

/* #152 PR3b (spec §6): stacked micro-batch training. trainingBatchDefault
 * walks the macro batch in b/m chunks of exactly m rows; m > 1 gathers each
 * chunk into two [m, ...] tensors. Compile-time pin of the trailing knob. */
_Static_assert(_Generic(&trainingBatchDefault,
                   float (*)(layer_t **, size_t, lossConfig_t, batch_t *, calculateGradsFn_t,
                             reduction_t, size_t): 1,
                   default: 0),
               "trainingBatchDefault must take a trailing size_t microBatchSize (#152)");

_Static_assert(_Generic(&trainingEpochDefault,
                   float (*)(layer_t **, size_t, lossConfig_t, dataLoader_t *, optimizer_t *,
                             calculateGradsFn_t, reduction_t, size_t): 1,
                   default: 0),
               "trainingEpochDefault must take a trailing size_t microBatchSize (#152)");

_Static_assert(_Generic(((trainingRunOptions_t){0}).microBatchSize, size_t: 1, default: 0),
               "trainingRunOptions_t must carry a size_t microBatchSize (#152)");

void setUp(void) {}
void tearDown(void) {}

/* ---- shared fixture helpers ------------------------------------------------ */

/* Natural-shape FLOAT32 tensor: the loop adds axis 0 itself (spec §5.1). */
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

/* Deterministic, non-uniform values in [-1, 1): uniform data would make the
 * row-order and gather mutations vacuous. */
static void fillRandom(float *dst, size_t n) {
    for (size_t i = 0; i < n; i++) {
        dst[i] = 2.0f * rngNextFloat() - 1.0f;
    }
}

/* trainingBatchDefault frees every sample_t it consumes (#119), so each run
 * gets fresh heap samples over the same dataset-owned tensors. freeBatch
 * releases the samples array + the batch_t afterwards. */
static batch_t *buildBatch(tensor_t **items, tensor_t **labels, size_t n) {
    batch_t *batch = reserveMemory(sizeof(batch_t));
    batch->samples = reserveMemory(n * sizeof(sample_t *));
    batch->size = n;
    for (size_t i = 0; i < n; i++) {
        sample_t *s = reserveMemory(sizeof(sample_t));
        s->item = items[i];
        s->label = labels[i];
        batch->samples[i] = s;
    }
    return batch;
}

#define MAX_PARAMS 16

static void zeroGrads(layer_t **model, size_t modelSize) {
    parameter_t *slots[MAX_PARAMS];
    size_t n = calcTotalNumberOfStates(model, modelSize);
    collectTrainableParameters(model, modelSize, slots);
    for (size_t p = 0; p < n; p++) {
        memset(slots[p]->grad->data, 0, calcBytesPerTensor(slots[p]->grad));
    }
}

/* Every trainable FLOAT32 grad times the mean scale, in model order; returns
 * the element count (callers assert it after teardown). */
static size_t snapshotScaledGrads(layer_t **model, size_t modelSize, float scale, float *out,
                                  size_t capacity) {
    parameter_t *slots[MAX_PARAMS];
    size_t n = calcTotalNumberOfStates(model, modelSize);
    collectTrainableParameters(model, modelSize, slots);
    size_t k = 0;
    for (size_t p = 0; p < n; p++) {
        const float *g = (const float *)slots[p]->grad->data;
        size_t count = calcNumberOfElementsByTensor(slots[p]->grad);
        for (size_t i = 0; i < count; i++) {
            if (k < capacity) {
                out[k] = g[i] * scale;
            }
            k++;
        }
    }
    return k;
}

/* allclose: |ref - got| <= atol + rtol * |ref|. Returns the first failing
 * index, or SIZE_MAX. The summation order differs across m, so bit equality
 * is not expected (spec §9). */
static size_t firstMismatch(const float *ref, const float *got, size_t n, float atol, float rtol) {
    for (size_t i = 0; i < n; i++) {
        if (!(fabsf(ref[i] - got[i]) <= atol + rtol * fabsf(ref[i]))) {
            return i;
        }
    }
    return SIZE_MAX;
}

/* A calculateGradsFn that records what each call received, then delegates. */
#define REC_MAX_CALLS 16
#define REC_MAX_ITEM_FLOATS 32
static size_t g_recCalls;
static size_t g_recItemRows[REC_MAX_CALLS];
static size_t g_recLabelRows[REC_MAX_CALLS];
static size_t g_recLabelRank[REC_MAX_CALLS];
static float g_recItemData[REC_MAX_CALLS][REC_MAX_ITEM_FLOATS];

static void resetRecording(void) {
    g_recCalls = 0;
    memset(g_recItemRows, 0, sizeof g_recItemRows);
    memset(g_recLabelRows, 0, sizeof g_recLabelRows);
    memset(g_recLabelRank, 0, sizeof g_recLabelRank);
    memset(g_recItemData, 0, sizeof g_recItemData);
}

static trainingStats_t *recordingGrads(layer_t **model, size_t modelSize, lossConfig_t lossConfig,
                                       reduction_t forwardReduction, tensor_t *input,
                                       tensor_t *label) {
    if (g_recCalls < REC_MAX_CALLS) {
        g_recItemRows[g_recCalls] = input->shape->dimensions[0];
        g_recLabelRows[g_recCalls] = label->shape->dimensions[0];
        g_recLabelRank[g_recCalls] = label->shape->numberOfDimensions;
        size_t n = calcNumberOfElementsByTensor(input);
        if (n <= REC_MAX_ITEM_FLOATS) {
            memcpy(g_recItemData[g_recCalls], input->data, n * sizeof(float));
        }
    }
    g_recCalls++;
    return calculateGradsSequential(model, modelSize, lossConfig, forwardReduction, input, label);
}

/* ---- model B: Linear(5->4) -> ReLU -> Linear(4->3), MSE --------------------- */

#define B_N 8
#define B_IN 5
#define B_OUT 3
#define B_SIZE 3
#define B_GRADS (B_IN * 4 + 4 + 4 * B_OUT + B_OUT) /* 39 */

static tensor_t *bItems[B_N];
static tensor_t *bLabels[B_N];

static void initModelBData(void) {
    rngSetSeed(1522u);
    for (size_t s = 0; s < B_N; s++) {
        float item[B_IN];
        float label[B_OUT];
        fillRandom(item, B_IN);
        fillRandom(label, B_OUT);
        bItems[s] = buildFloatTensor((size_t[]){B_IN}, 1, item);
        bLabels[s] = buildFloatTensor((size_t[]){B_OUT}, 1, label);
    }
}

static void freeModelBData(void) {
    for (size_t s = 0; s < B_N; s++) {
        freeTensor(bLabels[s]);
        freeTensor(bItems[s]);
    }
}

static void buildModelB(layer_t **model, layerQuant_t *lq) {
    model[0] = linearLayerInit(&(linearInit_t){.inFeatures = B_IN, .outFeatures = 4}, lq);
    model[1] = reluLayerInit(lq);
    model[2] = linearLayerInit(&(linearInit_t){.inFeatures = 4, .outFeatures = B_OUT}, lq);
}

static void freeModelB(layer_t **model) {
    freeLinearLayer(model[2]);
    freeReluLayer(model[1]);
    freeLinearLayer(model[0]);
}

/* ---- model A: every FLOAT32 layer type the loop can stack ------------------- */

/* [B,2,8] conv(2->4,k3) [B,4,6] -> groupNorm(2,4) -> relu -> maxPool(k2,s2)
 * [B,4,3] -> convT(4->3,k2) [B,3,4] -> layerNorm([4]) -> adaptiveAvgPool(3)
 * [B,3,3] -> avgPool(k2,s1) [B,3,2] -> flatten [B,6] -> linear(6->3) ->
 * softmax, CrossEntropy. */
#define A_SIZE 11
#define A_CLASSES 3
#define A_GRADS (24 + 4 + 4 + 4 + 24 + 3 + 4 + 4 + 18 + 3) /* 92 */

static size_t A_LN_SHAPE[1] = {4};

static void buildModelA(layer_t **model, layerQuant_t *lq) {
    model[0] =
        conv1dLayerInit(&(conv1dInit_t){.inChannels = 2, .outChannels = 4, .kernelSize = 3}, lq);
    model[1] = groupNormLayerInit(&(groupNormInit_t){.numGroups = 2, .numChannels = 4}, lq);
    model[2] = reluLayerInit(lq);
    model[3] = maxPool1dLayerInit(
        &(maxPool1dInit_t){.kernelSize = 2, .stride = 2, .inputChannels = 4, .inputLength = 6}, lq);
    model[4] = conv1dTransposedLayerInit(
        &(conv1dTransposedInit_t){.inChannels = 4, .outChannels = 3, .kernelSize = 2}, lq);
    model[5] =
        layerNormLayerInit(&(layerNormInit_t){.normalizedShape = A_LN_SHAPE, .numNormDims = 1}, lq);
    model[6] = adaptiveAvgPool1dLayerInit(&(adaptiveAvgPool1dInit_t){.outputSize = 3}, lq);
    model[7] = avgPool1dLayerInit(&(avgPool1dInit_t){.kernelSize = 2, .stride = 1}, lq);
    model[8] = flattenLayerInit();
    model[9] = linearLayerInit(&(linearInit_t){.inFeatures = 6, .outFeatures = A_CLASSES}, lq);
    model[10] = softmaxLayerInit(lq);
}

static void freeModelA(layer_t **model) {
    freeSoftmaxLayer(model[10]);
    freeLinearLayer(model[9]);
    freeFlattenLayer(model[8]);
    freeAvgPool1dLayer(model[7]);
    freeAdaptiveAvgPool1dLayer(model[6]);
    freeLayerNormLayer(model[5]);
    freeConv1dTransposedLayer(model[4]);
    freeMaxPool1dLayer(model[3]);
    freeReluLayer(model[2]);
    freeGroupNormLayer(model[1]);
    freeConv1dLayer(model[0]);
}

/* ---- chunk walk and gather ---------------------------------------------------- */

void testStackedChunkWalkCallsOncePerChunkWithMRows(void) {
    initModelBData();
    quantization_t *q = quantizationInitFloat();
    layerQuant_t lq;
    layerQuantInitUniform(&lq, q);
    rngSetSeed(11u);
    layer_t *model[B_SIZE];
    buildModelB(model, &lq);

    resetRecording();
    batch_t *batch = buildBatch(bItems, bLabels, B_N);
    trainingBatchDefault(model, B_SIZE, defaultLossConfig(MSE), batch, recordingGrads,
                         REDUCTION_MEAN, 4);
    freeBatch(batch);
    size_t calls = g_recCalls;
    size_t itemRows[2] = {g_recItemRows[0], g_recItemRows[1]};
    size_t labelRows[2] = {g_recLabelRows[0], g_recLabelRows[1]};
    size_t labelRank[2] = {g_recLabelRank[0], g_recLabelRank[1]};
    /* Row r of chunk c must be sample c*4 + r, byte for byte. */
    size_t firstWrongRow = SIZE_MAX;
    for (size_t c = 0; c < 2 && firstWrongRow == SIZE_MAX; c++) {
        for (size_t r = 0; r < 4; r++) {
            if (memcmp(g_recItemData[c] + r * B_IN, bItems[c * 4 + r]->data,
                       B_IN * sizeof(float)) != 0) {
                firstWrongRow = c * 4 + r;
                break;
            }
        }
    }

    freeModelB(model);
    freeQuantization(q);
    freeModelBData();

    TEST_ASSERT_EQUAL_size_t(2, calls);
    TEST_ASSERT_EQUAL_size_t(4, itemRows[0]);
    TEST_ASSERT_EQUAL_size_t(4, itemRows[1]);
    TEST_ASSERT_EQUAL_size_t(4, labelRows[0]);
    TEST_ASSERT_EQUAL_size_t(4, labelRows[1]);
    TEST_ASSERT_EQUAL_size_t(2, labelRank[0]);
    TEST_ASSERT_EQUAL_size_t(2, labelRank[1]);
    TEST_ASSERT_EQUAL_size_t_MESSAGE(SIZE_MAX, firstWrongRow, "gathered rows out of sample order");
}

void testMicroBatchZeroBehavesExactlyLikeOne(void) {
    /* 0 means 1 (spec §6.1): same call pattern, bit-identical loss and grads. */
    initModelBData();
    quantization_t *q = quantizationInitFloat();
    layerQuant_t lq;
    layerQuantInitUniform(&lq, q);
    rngSetSeed(12u);
    layer_t *model[B_SIZE];
    buildModelB(model, &lq);
    float grads[2][B_GRADS];
    float loss[2];
    size_t calls[2];
    size_t rows[2];
    const size_t ms[2] = {0, 1};
    for (size_t k = 0; k < 2; k++) {
        zeroGrads(model, B_SIZE);
        resetRecording();
        batch_t *batch = buildBatch(bItems, bLabels, B_N);
        loss[k] = trainingBatchDefault(model, B_SIZE, defaultLossConfig(MSE), batch, recordingGrads,
                                       REDUCTION_MEAN, ms[k]);
        freeBatch(batch);
        calls[k] = g_recCalls;
        rows[k] = g_recItemRows[0];
        snapshotScaledGrads(model, B_SIZE, 1.0f, grads[k], B_GRADS);
    }

    freeModelB(model);
    freeQuantization(q);
    freeModelBData();

    TEST_ASSERT_EQUAL_size_t(B_N, calls[0]);
    TEST_ASSERT_EQUAL_size_t(B_N, calls[1]);
    TEST_ASSERT_EQUAL_size_t(1, rows[0]);
    TEST_ASSERT_EQUAL_MEMORY(&loss[1], &loss[0], sizeof(float));
    TEST_ASSERT_EQUAL_MEMORY(grads[1], grads[0], sizeof(grads[0]));
}

void testStackedLossIsRowWeightedForMeanAndPlainForSum(void) {
    /* MEAN: Σ(chunkLoss * m) / b equals the per-sample mean; SUM: the plain
     * sum of chunk sums equals the per-sample sum (spec §6.5). */
    initModelBData();
    quantization_t *q = quantizationInitFloat();
    layerQuant_t lq;
    layerQuantInitUniform(&lq, q);
    rngSetSeed(13u);
    layer_t *model[B_SIZE];
    buildModelB(model, &lq);
    float mean[2];
    float sum[2];
    const size_t ms[2] = {1, 4};
    for (size_t k = 0; k < 2; k++) {
        batch_t *batch = buildBatch(bItems, bLabels, B_N);
        mean[k] = trainingBatchDefault(model, B_SIZE, defaultLossConfig(MSE), batch,
                                       calculateGradsSequential, REDUCTION_MEAN, ms[k]);
        freeBatch(batch);
        batch = buildBatch(bItems, bLabels, B_N);
        sum[k] = trainingBatchDefault(model, B_SIZE, defaultLossConfig(MSE), batch,
                                      calculateGradsSequential, REDUCTION_SUM, ms[k]);
        freeBatch(batch);
    }

    freeModelB(model);
    freeQuantization(q);
    freeModelBData();

    TEST_ASSERT_FLOAT_WITHIN(1e-6f + 1e-5f * fabsf(mean[0]), mean[0], mean[1]);
    TEST_ASSERT_FLOAT_WITHIN(1e-6f + 1e-5f * fabsf(sum[0]), sum[0], sum[1]);
    /* Sanity: the two reductions really differ (b * F = 24 apart for MSE). */
    TEST_ASSERT_FLOAT_WITHIN(1e-4f * fabsf(sum[0]), sum[0], mean[0] * (float)(B_N * B_OUT));
}

/* ---- parity across m (spec §9 PR 3b core) --------------------------------- */

void testStackedMatchesPerSampleOnModelA(void) {
    rngSetSeed(1521u);
    quantization_t *q = quantizationInitFloat();
    layerQuant_t lq;
    layerQuantInitUniform(&lq, q);
    layer_t *model[A_SIZE];
    buildModelA(model, &lq);
    static const size_t classOf[B_N] = {0, 2, 1, 1, 0, 2, 2, 0};
    tensor_t *items[B_N];
    tensor_t *labels[B_N];
    for (size_t s = 0; s < B_N; s++) {
        float item[2 * 8];
        fillRandom(item, 2 * 8);
        float label[A_CLASSES] = {0.0f, 0.0f, 0.0f};
        label[classOf[s]] = 1.0f;
        items[s] = buildFloatTensor((size_t[]){2, 8}, 2, item);
        labels[s] = buildFloatTensor((size_t[]){A_CLASSES}, 1, label);
    }
    lossConfig_t cfg = defaultLossConfig(CROSS_ENTROPY);
    batchView_t labelRefView;
    float scale =
        lossFunctions[CROSS_ENTROPY].computeMeanScale(B_N, batchViewOf(&labelRefView, labels[0]));

    float ref[A_GRADS];
    float got[A_GRADS];
    zeroGrads(model, A_SIZE);
    batch_t *batch = buildBatch(items, labels, B_N);
    float refLoss = trainingBatchDefault(model, A_SIZE, cfg, batch, calculateGradsSequential,
                                         REDUCTION_MEAN, 1);
    freeBatch(batch);
    size_t refCount = snapshotScaledGrads(model, A_SIZE, scale, ref, A_GRADS);

    const size_t ms[3] = {2, 4, 8};
    size_t gotCount[3];
    size_t mismatch[3];
    float loss[3];
    for (size_t k = 0; k < 3; k++) {
        zeroGrads(model, A_SIZE);
        batch = buildBatch(items, labels, B_N);
        loss[k] = trainingBatchDefault(model, A_SIZE, cfg, batch, calculateGradsSequential,
                                       REDUCTION_MEAN, ms[k]);
        freeBatch(batch);
        gotCount[k] = snapshotScaledGrads(model, A_SIZE, scale, got, A_GRADS);
        mismatch[k] = firstMismatch(ref, got, A_GRADS, 1e-6f, 1e-4f);
    }

    for (size_t s = 0; s < B_N; s++) {
        freeTensor(labels[s]);
        freeTensor(items[s]);
    }
    freeModelA(model);
    freeQuantization(q);

    TEST_ASSERT_EQUAL_size_t(A_GRADS, refCount);
    for (size_t k = 0; k < 3; k++) {
        TEST_ASSERT_EQUAL_size_t(A_GRADS, gotCount[k]);
        TEST_ASSERT_EQUAL_size_t_MESSAGE(SIZE_MAX, mismatch[k], "stacked grads diverge from m=1");
        TEST_ASSERT_FLOAT_WITHIN(1e-6f + 1e-5f * fabsf(refLoss), refLoss, loss[k]);
    }
}

void testStackedMatchesPerSampleOnMseModelB(void) {
    initModelBData();
    quantization_t *q = quantizationInitFloat();
    layerQuant_t lq;
    layerQuantInitUniform(&lq, q);
    rngSetSeed(14u);
    layer_t *model[B_SIZE];
    buildModelB(model, &lq);
    lossConfig_t cfg = defaultLossConfig(MSE);
    batchView_t labelRefView;
    /* MSE mean scale = 1 / (b * F) with F read off the [1, 3] label view. */
    float scale = lossFunctions[MSE].computeMeanScale(B_N, batchViewOf(&labelRefView, bLabels[0]));

    float ref[B_GRADS];
    float got[B_GRADS];
    zeroGrads(model, B_SIZE);
    batch_t *batch = buildBatch(bItems, bLabels, B_N);
    float refLoss = trainingBatchDefault(model, B_SIZE, cfg, batch, calculateGradsSequential,
                                         REDUCTION_MEAN, 1);
    freeBatch(batch);
    snapshotScaledGrads(model, B_SIZE, scale, ref, B_GRADS);

    const size_t ms[3] = {2, 4, 8};
    size_t mismatch[3];
    float loss[3];
    for (size_t k = 0; k < 3; k++) {
        zeroGrads(model, B_SIZE);
        batch = buildBatch(bItems, bLabels, B_N);
        loss[k] = trainingBatchDefault(model, B_SIZE, cfg, batch, calculateGradsSequential,
                                       REDUCTION_MEAN, ms[k]);
        freeBatch(batch);
        snapshotScaledGrads(model, B_SIZE, scale, got, B_GRADS);
        mismatch[k] = firstMismatch(ref, got, B_GRADS, 1e-6f, 1e-4f);
    }

    freeModelB(model);
    freeQuantization(q);
    freeModelBData();

    TEST_ASSERT_FLOAT_WITHIN(1e-9f, 1.0f / (float)(B_N * B_OUT), scale);
    for (size_t k = 0; k < 3; k++) {
        TEST_ASSERT_EQUAL_size_t_MESSAGE(SIZE_MAX, mismatch[k], "stacked grads diverge from m=1");
        TEST_ASSERT_FLOAT_WITHIN(1e-6f + 1e-5f * fabsf(refLoss), refLoss, loss[k]);
    }
}

/* ---- PyTorch gold: one batched B=4 step (generate_expected_microbatch.py) ---- */

#define GOLD_SIZE 6
#define GOLD_GRADS (18 + 3 + 27 + 3) /* 51 */

void testStackedBatchMatchesPyTorchGold(void) {
    quantization_t *q = quantizationInitFloat();
    layerQuant_t lq;
    layerQuantInitUniform(&lq, q);
    layer_t *model[GOLD_SIZE];
    model[0] = conv1dLayerInit(&(conv1dInit_t){.inChannels = MB_GOLD_C_IN,
                                               .outChannels = MB_GOLD_C_OUT,
                                               .kernelSize = MB_GOLD_K},
                               &lq);
    model[1] = reluLayerInit(&lq);
    model[2] = maxPool1dLayerInit(&(maxPool1dInit_t){.kernelSize = 2,
                                                     .stride = 2,
                                                     .inputChannels = MB_GOLD_C_OUT,
                                                     .inputLength = MB_GOLD_L_IN - MB_GOLD_K + 1},
                                  &lq);
    model[3] = flattenLayerInit();
    model[4] = linearLayerInit(
        &(linearInit_t){.inFeatures = MB_GOLD_FLAT, .outFeatures = MB_GOLD_CLASSES}, &lq);
    model[5] = softmaxLayerInit(&lq);
    float convW[18];
    float convB[3];
    float linW[27];
    float linB[3];
    memcpy(convW, mbGoldConvW, sizeof convW);
    memcpy(convB, mbGoldConvB, sizeof convB);
    memcpy(linW, mbGoldLinW, sizeof linW);
    memcpy(linB, mbGoldLinB, sizeof linB);
    layerLoadWeights(model[0], convW, convB);
    layerLoadWeights(model[4], linW, linB);

    tensor_t *items[MB_GOLD_B];
    tensor_t *labels[MB_GOLD_B];
    for (size_t s = 0; s < MB_GOLD_B; s++) {
        items[s] = buildFloatTensor((size_t[]){MB_GOLD_C_IN, MB_GOLD_L_IN}, 2,
                                    mbGoldItems + s * MB_GOLD_C_IN * MB_GOLD_L_IN);
        labels[s] =
            buildFloatTensor((size_t[]){MB_GOLD_CLASSES}, 1, mbGoldLabels + s * MB_GOLD_CLASSES);
    }
    float gold[GOLD_GRADS];
    memcpy(gold, mbGoldConvWGrad, 18 * sizeof(float));
    memcpy(gold + 18, mbGoldConvBGrad, 3 * sizeof(float));
    memcpy(gold + 21, mbGoldLinWGrad, 27 * sizeof(float));
    memcpy(gold + 48, mbGoldLinBGrad, 3 * sizeof(float));
    batchView_t labelRefView;
    float scale = lossFunctions[CROSS_ENTROPY].computeMeanScale(
        MB_GOLD_B, batchViewOf(&labelRefView, labels[0]));

    /* m = 1 (four single-row calls) and m = 4 (one stacked call). */
    const size_t ms[2] = {1, 4};
    float loss[2];
    size_t count[2];
    size_t mismatch[2];
    for (size_t k = 0; k < 2; k++) {
        float got[GOLD_GRADS];
        zeroGrads(model, GOLD_SIZE);
        batch_t *batch = buildBatch(items, labels, MB_GOLD_B);
        loss[k] = trainingBatchDefault(model, GOLD_SIZE, defaultLossConfig(CROSS_ENTROPY), batch,
                                       calculateGradsSequential, REDUCTION_MEAN, ms[k]);
        freeBatch(batch);
        count[k] = snapshotScaledGrads(model, GOLD_SIZE, scale, got, GOLD_GRADS);
        mismatch[k] = firstMismatch(gold, got, GOLD_GRADS, 1e-6f, 1e-4f);
    }

    for (size_t s = 0; s < MB_GOLD_B; s++) {
        freeTensor(labels[s]);
        freeTensor(items[s]);
    }
    freeSoftmaxLayer(model[5]);
    freeLinearLayer(model[4]);
    freeFlattenLayer(model[3]);
    freeMaxPool1dLayer(model[2]);
    freeReluLayer(model[1]);
    freeConv1dLayer(model[0]);
    freeQuantization(q);

    for (size_t k = 0; k < 2; k++) {
        TEST_ASSERT_EQUAL_size_t(GOLD_GRADS, count[k]);
        TEST_ASSERT_EQUAL_size_t_MESSAGE(SIZE_MAX, mismatch[k], "grads diverge from PyTorch");
        TEST_ASSERT_FLOAT_WITHIN(1e-5f, mbGoldLoss, loss[k]);
    }
}

/* ---- fail-fast contracts ---------------------------------------------------- */

/* A death test that also returns what the child printed: DeathTest.h's
 * ASSERT_EXITS_WITH discards the child's stdout, but the fail-fast messages
 * must name b and m (spec §6.2). PRINT_ERROR writes to stdout in every preset
 * (DEBUG_MODE_ERROR is always defined, src/common/CMakeLists.txt); here the
 * child's stdout is the write end of a pipe, and exit() flushes it. The
 * parent drains the pipe to EOF (a chatty child can never block on a full
 * pipe), keeps the first outCap - 1 bytes NUL-terminated in outBuf, and
 * stores the child's exit code in *codeOut (-1 when it died by a signal). */
#define CAPTURE_EXIT_AND_OUTPUT(statement, outBuf, outCap, codeOut)                                \
    do {                                                                                           \
        int _odtPipe[2];                                                                           \
        TEST_ASSERT_EQUAL_INT_MESSAGE(0, pipe(_odtPipe), "pipe() failed");                         \
        fflush(stdout);                                                                            \
        fflush(stderr);                                                                            \
        pid_t _odtPid = fork();                                                                    \
        TEST_ASSERT_MESSAGE(_odtPid >= 0, "fork() failed");                                        \
        if (_odtPid == 0) {                                                                        \
            close(_odtPipe[0]);                                                                    \
            dup2(_odtPipe[1], STDOUT_FILENO);                                                      \
            close(_odtPipe[1]);                                                                    \
            (void)freopen("/dev/null", "w", stderr);                                               \
            statement;                                                                             \
            _exit(0);                                                                              \
        }                                                                                          \
        close(_odtPipe[1]);                                                                        \
        size_t _odtLen = 0;                                                                        \
        char _odtChunk[256];                                                                       \
        ssize_t _odtN;                                                                             \
        while ((_odtN = read(_odtPipe[0], _odtChunk, sizeof _odtChunk)) > 0) {                     \
            for (ssize_t _odtI = 0; _odtI < _odtN && _odtLen + 1 < (outCap); _odtI++) {            \
                (outBuf)[_odtLen++] = _odtChunk[_odtI];                                            \
            }                                                                                      \
        }                                                                                          \
        (outBuf)[_odtLen] = '\0';                                                                  \
        close(_odtPipe[0]);                                                                        \
        int _odtStatus = 0;                                                                        \
        (void)waitpid(_odtPid, &_odtStatus, 0);                                                    \
        *(codeOut) = WIFEXITED(_odtStatus) ? WEXITSTATUS(_odtStatus) : -1;                         \
    } while (0)

/* Ruling R10 (Leo, 2026-09-23): a message test pins the numbers that matter
 * plus ONE stable keyword, never the wording. True iff `value` appears in
 * `message` as a whole decimal number: a maximal digit run that is not the
 * tail of an identifier, so the 32 of "FLOAT32" or of a function name never
 * counts. PRINT_ERROR's colour codes ("\033[0;31m" ... "\033[0m") always
 * contribute 0 and 31: never assert either of those. */
static bool messageHasNumber(const char *message, size_t value) {
    const char *p = message;
    while (*p != '\0') {
        if (!isdigit((unsigned char)*p)) {
            p++;
            continue;
        }
        bool identifierTail = p > message && (isalpha((unsigned char)p[-1]) || p[-1] == '_');
        size_t parsed = 0;
        bool fits = true;
        while (isdigit((unsigned char)*p)) {
            size_t digit = (size_t)(*p - '0');
            if (parsed > (SIZE_MAX - digit) / 10) {
                fits = false;
            } else {
                parsed = parsed * 10 + digit;
            }
            p++;
        }
        if (!identifierTail && fits && parsed == value) {
            return true;
        }
    }
    return false;
}

static void trainModelBBatchOrDie(layer_t **model, size_t batchSize, size_t microBatchSize) {
    batch_t *batch = buildBatch(bItems, bLabels, batchSize);
    trainingBatchDefault(model, B_SIZE, defaultLossConfig(MSE), batch, calculateGradsSequential,
                         REDUCTION_MEAN, microBatchSize);
}

void testTrainingBatchDefaultRejectsBatchNotDivisibleByMicroBatch(void) {
    /* Check 4 (spec §6.2): b % m == 0 at entry -- m not dividing b (the
     * message must name both: numbers plus the keyword "microBatchSize",
     * ruling R10; a direct call, so checks 1-3 are not on this path), and
     * m > b. */
    initModelBData();
    quantization_t *q = quantizationInitFloat();
    layerQuant_t lq;
    layerQuantInitUniform(&lq, q);
    rngSetSeed(15u);
    layer_t *model[B_SIZE];
    buildModelB(model, &lq);
    char message[512];
    int code = -2;

    CAPTURE_EXIT_AND_OUTPUT(trainModelBBatchOrDie(model, 8, 3), message, sizeof message, &code);
    ASSERT_EXITS_WITH_FAILURE(trainModelBBatchOrDie(model, 4, 8));

    freeModelB(model);
    freeQuantization(q);
    freeModelBData();

    TEST_ASSERT_EQUAL_INT_MESSAGE(1, code, "check 4 must exit(1)");
    TEST_ASSERT_NOT_NULL_MESSAGE(strstr(message, "microBatchSize"), "check 4 must name the knob");
    TEST_ASSERT_TRUE_MESSAGE(messageHasNumber(message, 8), "check 4 must name b (8)");
    TEST_ASSERT_TRUE_MESSAGE(messageHasNumber(message, 3), "check 4 must name m (3)");
}

static void trainEmptyBatchOrDie(layer_t **model, size_t microBatchSize) {
    batch_t empty = {.samples = NULL, .size = 0};
    trainingBatchDefault(model, B_SIZE, defaultLossConfig(MSE), &empty, calculateGradsSequential,
                         REDUCTION_MEAN, microBatchSize);
}

void testStackedRejectsEmptyBatch(void) {
    /* b = 0 passes b % m == 0, but the stacked path sizes its gather buffers
     * from sample 0 and there is none: fail fast instead of reading
     * samples[0]. (m == 1 keeps its pre-existing empty loop.) */
    quantization_t *q = quantizationInitFloat();
    layerQuant_t lq;
    layerQuantInitUniform(&lq, q);
    rngSetSeed(26u);
    layer_t *model[B_SIZE];
    buildModelB(model, &lq);

    ASSERT_EXITS_WITH_FAILURE(trainEmptyBatchOrDie(model, 2));

    freeModelB(model);
    freeQuantization(q);
}

static void trainOneSampleClaimingMRowsOrDie(layer_t **model, sample_t *only, size_t m) {
    sample_t *samples[1] = {only};
    batch_t batch = {.samples = samples, .size = m};
    trainingBatchDefault(model, B_SIZE, defaultLossConfig(MSE), &batch, calculateGradsSequential,
                         REDUCTION_MEAN, m);
}

void testStackedGatherSizeOverflowFailsFast(void) {
    /* m = SIZE_MAX / 32 + 2 with 32-byte items AND labels: m * 32 wraps to
     * exactly 32 (spec §6.4 checked multiply). An unchecked multiply would
     * reserve 32 bytes and gather samples[1..] past the 1-entry array. The
     * batch claims b = m, so the divisibility check passes. */
    quantization_t *q = quantizationInitFloat();
    layerQuant_t lq;
    layerQuantInitUniform(&lq, q);
    rngSetSeed(16u);
    layer_t *model[B_SIZE];
    buildModelB(model, &lq);
    float eight[8] = {0};
    tensor_t *item = buildFloatTensor((size_t[]){8}, 1, eight);
    tensor_t *label = buildFloatTensor((size_t[]){8}, 1, eight);
    sample_t *only = reserveMemory(sizeof(sample_t));
    only->item = item;
    only->label = label;

    ASSERT_EXITS_WITH_FAILURE(trainOneSampleClaimingMRowsOrDie(model, only, SIZE_MAX / 32 + 2));

    freeReservedMemory(only);
    freeTensor(label);
    freeTensor(item);
    freeModelB(model);
    freeQuantization(q);
}

static void trainTwoSamplesOrDie(layer_t **model, tensor_t **items, tensor_t **labels) {
    batch_t *batch = buildBatch(items, labels, 2);
    trainingBatchDefault(model, B_SIZE, defaultLossConfig(MSE), batch, calculateGradsSequential,
                         REDUCTION_MEAN, 2);
}

void testStackedGatherReservationFailureFailsFast(void) {
    /* m * per-sample bytes = 2 * (SIZE_MAX / 8) * 4 = SIZE_MAX - 7 does NOT
     * wrap, but no allocator can serve it: reserveMemory returns NULL (its own
     * size-wrap guard under ODT_MEM_PROFILE, calloc failure otherwise) and the
     * gather must fail fast instead of copying through NULL (spec §6.4). The
     * item is stack-built with that huge shape; its data is never read, and
     * b = m = 2 keeps every sample access in bounds. */
    quantization_t *q = quantizationInitFloat();
    layerQuant_t lq;
    layerQuantInitUniform(&lq, q);
    rngSetSeed(27u);
    layer_t *model[B_SIZE];
    buildModelB(model, &lq);
    float itemData[B_IN] = {0};
    size_t hugeDims[1] = {SIZE_MAX / 8};
    size_t hugeOrder[1] = {0};
    shape_t hugeShape = {
        .numberOfDimensions = 1, .dimensions = hugeDims, .orderOfDimensions = hugeOrder};
    quantization_t floatQ;
    initFloat32Quantization(&floatQ);
    tensor_t hugeItem = {.data = (uint8_t *)itemData, .shape = &hugeShape, .quantization = &floatQ};
    float labelData[B_OUT] = {0};
    tensor_t *label = buildFloatTensor((size_t[]){B_OUT}, 1, labelData);
    tensor_t *items[2] = {&hugeItem, &hugeItem};
    tensor_t *labels[2] = {label, label};

    ASSERT_EXITS_WITH_FAILURE(trainTwoSamplesOrDie(model, items, labels));

    freeTensor(label);
    freeModelB(model);
    freeQuantization(q);
}

static tensor_t *buildBoolMask(size_t n) {
    size_t *dims = reserveMemory(sizeof(size_t));
    dims[0] = n;
    size_t *order = reserveMemory(sizeof(size_t));
    setOrderOfDimsForNewTensor(1, order);
    shape_t *shape = reserveMemory(sizeof(shape_t));
    setShape(shape, dims, 1, order);
    return initTensor(shape, quantizationInitBool(), NULL);
}

void testStackedDropoutFailsFastOnItsMaskCount(void) {
    /* Spec §6.8 known limitation: the caller-allocated mask holds one
     * sample's elements, so at m > 1 Dropout's own count guard exits
     * (Dropout.c forward mask guard) -- a pre-existing fail-fast, pinned. */
    initModelBData();
    quantization_t *q = quantizationInitFloat();
    layerQuant_t lq;
    layerQuantInitUniform(&lq, q);
    rngSetSeed(17u);
    tensor_t *mask = buildBoolMask(4);
    layer_t *model[B_SIZE];
    model[0] = linearLayerInit(&(linearInit_t){.inFeatures = B_IN, .outFeatures = 4}, &lq);
    model[1] = dropoutLayerInit(0.5f, mask, q, q);
    model[2] = linearLayerInit(&(linearInit_t){.inFeatures = 4, .outFeatures = B_OUT}, &lq);

    ASSERT_EXITS_WITH_FAILURE(trainModelBBatchOrDie(model, 4, 2));

    freeLinearLayer(model[2]);
    freeDropoutLayer(model[1]);
    freeLinearLayer(model[0]);
    freeTensor(mask);
    freeQuantization(q);
    freeModelBData();
}

/* ---- per-chunk sample validation (spec §6.3) ------------------------------ */

/* Model C: Flatten -> Linear(8 -> 4), MSE. Items [2, 4], labels [4]. Every
 * defective sample below keeps the reference byte count, so a missing check
 * trains silently (exit 0) instead of crashing: the death tests pin the
 * validation itself, not an incidental out-of-bounds read. */
#define C_N 4
#define C_SIZE 2

static void buildModelC(layer_t **model, layerQuant_t *lq) {
    model[0] = flattenLayerInit();
    model[1] = linearLayerInit(&(linearInit_t){.inFeatures = 8, .outFeatures = 4}, lq);
}

static void freeModelC(layer_t **model) {
    freeLinearLayer(model[1]);
    freeFlattenLayer(model[0]);
}

static void initModelCData(tensor_t **items, tensor_t **labels) {
    rngSetSeed(1523u);
    for (size_t s = 0; s < C_N; s++) {
        float item[8];
        float label[4];
        fillRandom(item, 8);
        fillRandom(label, 4);
        items[s] = buildFloatTensor((size_t[]){2, 4}, 2, item);
        labels[s] = buildFloatTensor((size_t[]){4}, 1, label);
    }
}

static void freeModelCData(tensor_t **items, tensor_t **labels) {
    for (size_t s = 0; s < C_N; s++) {
        freeTensor(labels[s]);
        freeTensor(items[s]);
    }
}

static void trainModelCOrDie(layer_t **model, tensor_t **items, tensor_t **labels) {
    batch_t *batch = buildBatch(items, labels, C_N);
    trainingBatchDefault(model, C_SIZE, defaultLossConfig(MSE), batch, calculateGradsSequential,
                         REDUCTION_MEAN, 2);
}

/* Runs one death check with sample `bad`'s item (or label) replaced. */
static void assertStackingRejects(tensor_t *badTensor, size_t bad, bool isLabel) {
    quantization_t *q = quantizationInitFloat();
    layerQuant_t lq;
    layerQuantInitUniform(&lq, q);
    rngSetSeed(18u);
    layer_t *model[C_SIZE];
    buildModelC(model, &lq);
    tensor_t *items[C_N];
    tensor_t *labels[C_N];
    initModelCData(items, labels);
    tensor_t *goodTensor = isLabel ? labels[bad] : items[bad];
    if (isLabel) {
        labels[bad] = badTensor;
    } else {
        items[bad] = badTensor;
    }

    ASSERT_EXITS_WITH_FAILURE(trainModelCOrDie(model, items, labels));

    if (isLabel) {
        labels[bad] = goodTensor;
    } else {
        items[bad] = goodTensor;
    }
    freeModelCData(items, labels);
    freeModelC(model);
    freeQuantization(q);
}

static const float C_BAD_VALUES[8] = {0.1f, -0.2f, 0.3f, -0.4f, 0.5f, -0.6f, 0.7f, -0.8f};

void testStackedRejectsItemWithDifferentDims(void) {
    tensor_t *bad = buildFloatTensor((size_t[]){4, 2}, 2, C_BAD_VALUES);
    assertStackingRejects(bad, 1, false);
    freeTensor(bad);
}

void testStackedRejectsItemWithDifferentRank(void) {
    /* [2, 4, 1]: the leading dims match sample 0, only the rank differs. */
    tensor_t *bad = buildFloatTensor((size_t[]){2, 4, 1}, 3, C_BAD_VALUES);
    assertStackingRejects(bad, 1, false);
    freeTensor(bad);
}

void testStackedRejectsItemWithDifferentOrder(void) {
    tensor_t *bad = buildFloatTensor((size_t[]){2, 4}, 2, C_BAD_VALUES);
    transposeTensor(bad, 0, 1); /* same dims and bytes, permuted order */
    assertStackingRejects(bad, 1, false);
    freeTensor(bad);
}

static tensor_t *buildSymInt32Item(void) {
    size_t *dims = reserveMemory(2 * sizeof(size_t));
    dims[0] = 2;
    dims[1] = 4;
    size_t *order = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, order);
    shape_t *shape = reserveMemory(sizeof(shape_t));
    setShape(shape, dims, 2, order);
    tensor_t *t = initTensor(shape, quantizationInitSymInt32(HALF_AWAY), NULL);
    tensorFillFromFloatBuffer(t, C_BAD_VALUES, 8);
    return t;
}

void testStackedRejectsNonFloat32Item(void) {
    tensor_t *bad = buildSymInt32Item();
    assertStackingRejects(bad, 1, false);
    freeTensor(bad);
}

void testStackedRejectsNonFloat32FirstSample(void) {
    /* Sample 0 is the reference every other sample is compared against --
     * it must itself be FLOAT32. */
    tensor_t *bad = buildSymInt32Item();
    assertStackingRejects(bad, 0, false);
    freeTensor(bad);
}

void testStackedRejectsLabelWithSparsity(void) {
    sparsity_t sparsity = {.type = SPARSITY_TYPE_1, .config = NULL};
    tensor_t *bad = buildFloatTensor((size_t[]){4}, 1, C_BAD_VALUES);
    bad->sparsity = &sparsity;
    assertStackingRejects(bad, 1, true);
    bad->sparsity = NULL;
    freeTensor(bad);
}

void testStackedRejectsLabelWithDifferentRank(void) {
    /* [2, 2] vs sample 0's [4]: same bytes, the rank differs. (The
     * per-dimension compare is shared with the item path and pinned there,
     * Step 5 (h); a rank-1 label cannot differ in dims at equal bytes.) */
    tensor_t *bad = buildFloatTensor((size_t[]){2, 2}, 2, C_BAD_VALUES);
    assertStackingRejects(bad, 1, true);
    freeTensor(bad);
}

void testStackedRejectsMismatchInALaterChunk(void) {
    /* Sample 3 = row 1 of chunk 1: validation runs for EVERY chunk. */
    tensor_t *bad = buildFloatTensor((size_t[]){4, 2}, 2, C_BAD_VALUES);
    assertStackingRejects(bad, 3, false);
    freeTensor(bad);
}

void testStackedRejectsChunkDifferingFromSampleZero(void) {
    /* Chunk 1 is internally uniform ([4, 2] twice) but differs from sample
     * 0: the gather buffers are sized once per macro batch from sample 0, so
     * every sample must match sample 0, not just its chunk's first sample. */
    tensor_t *badA = buildFloatTensor((size_t[]){4, 2}, 2, C_BAD_VALUES);
    tensor_t *badB = buildFloatTensor((size_t[]){4, 2}, 2, C_BAD_VALUES);
    quantization_t *q = quantizationInitFloat();
    layerQuant_t lq;
    layerQuantInitUniform(&lq, q);
    rngSetSeed(19u);
    layer_t *model[C_SIZE];
    buildModelC(model, &lq);
    tensor_t *items[C_N];
    tensor_t *labels[C_N];
    initModelCData(items, labels);
    tensor_t *good2 = items[2];
    tensor_t *good3 = items[3];
    items[2] = badA;
    items[3] = badB;

    ASSERT_EXITS_WITH_FAILURE(trainModelCOrDie(model, items, labels));

    items[2] = good2;
    items[3] = good3;
    freeModelCData(items, labels);
    freeModelC(model);
    freeQuantization(q);
    freeTensor(badB);
    freeTensor(badA);
}

/* ---- FLOAT32 gate (spec §6.6) --------------------------------------------- */

/* Each gated model below trains cleanly at m = 2 WITHOUT the gate (the funnel
 * converts every non-FLOAT32 operand), so only the gate can make it exit. */
static void trainStackedMseOrDie(layer_t **model, size_t modelSize) {
    batch_t *batch = buildBatch(bItems, bLabels, 4);
    trainingBatchDefault(model, modelSize, defaultLossConfig(MSE), batch, calculateGradsSequential,
                         REDUCTION_MEAN, 2);
}

static void assertStackedGateRejects(layer_t **model, size_t modelSize) {
    initModelBData();
    ASSERT_EXITS_WITH_FAILURE(trainStackedMseOrDie(model, modelSize));
    freeModelBData();
}

void testStackedGateRejectsSymLayer(void) {
    quantization_t *floatQ = quantizationInitFloat();
    quantization_t *symQ = quantizationInitSymInt32(HALF_AWAY);
    arithmetic_t sym = arithmeticFromQuantization(symQ);
    layerQuant_t lq = {.forwardMath = sym,
                       .weightGradMath = sym,
                       .biasGradMath = sym,
                       .propLossMath = sym,
                       .outputQ = symQ,
                       .propLossQ = symQ,
                       .weightStorage = floatQ,
                       .biasStorage = floatQ,
                       .weightGradAccMode = OUT_ACC_DYNAMIC_RESCALE,
                       .biasGradAccMode = OUT_ACC_DYNAMIC_RESCALE};
    rngSetSeed(20u);
    layer_t *model[1] = {
        linearLayerInit(&(linearInit_t){.inFeatures = B_IN, .outFeatures = B_OUT}, &lq)};

    assertStackedGateRejects(model, 1);

    freeLinearLayer(model[0]);
    freeQuantization(symQ);
    freeQuantization(floatQ);
}

void testStackedGateRejectsBfpLayer(void) {
    /* A genuine BFP layer: ARITH_BFP in all four math slots (derived from the
     * BFP wire template) and BFP-stored weights and bias (FLOAT32 init +
     * requantizeTensorInPlace, #270). With FLOAT32 weights an ARITH_BFP
     * forward already dies at ANY m ("requires BFP-stored weights"), which
     * would make this test vacuous; built like this it trains at m = 2
     * without the gate. */
    quantization_t *floatQ = quantizationInitFloat();
    quantization_t *bfpQ = quantizationInitBfp(8, 8, HALF_AWAY);
    layerQuant_t bfpLq;
    layerQuantInitUniform(&bfpLq, bfpQ);
    bfpLq.weightStorage = floatQ;
    bfpLq.biasStorage = floatQ;
    layerQuant_t lq;
    layerQuantInitUniform(&lq, floatQ);
    rngSetSeed(28u);
    layer_t *model[2] = {
        linearLayerInit(&(linearInit_t){.inFeatures = B_IN, .outFeatures = 4}, &bfpLq),
        linearLayerInit(&(linearInit_t){.inFeatures = 4, .outFeatures = B_OUT}, &lq)};
    linearConfig_t *bfpLinear = model[0]->config->linear;
    requantizeTensorInPlace(getParamFromParameter(bfpLinear->weights), bfpQ);
    requantizeTensorInPlace(getParamFromParameter(bfpLinear->bias), bfpQ);

    assertStackedGateRejects(model, 2);

    freeLinearLayer(model[1]);
    freeLinearLayer(model[0]);
    freeQuantization(bfpQ);
    freeQuantization(floatQ);
}

void testStackedGateRejectsBfpWire(void) {
    quantization_t *floatQ = quantizationInitFloat();
    quantization_t *bfpQ = quantizationInitBfp(8, 8, HALF_AWAY);
    layerQuant_t bfpOut;
    layerQuantInitUniform(&bfpOut, floatQ);
    bfpOut.outputQ = bfpQ; /* FLOAT32 math, BFP-stored forward wire */
    layerQuant_t lq;
    layerQuantInitUniform(&lq, floatQ);
    rngSetSeed(21u);
    layer_t *model[2] = {
        linearLayerInit(&(linearInit_t){.inFeatures = B_IN, .outFeatures = 4}, &bfpOut),
        linearLayerInit(&(linearInit_t){.inFeatures = 4, .outFeatures = B_OUT}, &lq)};

    assertStackedGateRejects(model, 2);

    freeLinearLayer(model[1]);
    freeLinearLayer(model[0]);
    freeQuantization(bfpQ);
    freeQuantization(floatQ);
}

void testStackedGateRejectsQuantizationLayer(void) {
    /* Also pins the §6.6 message by its numbers plus ONE keyword, "FLOAT32"
     * (ruling R10): m = 2, the layer index 3 and the layerType_t of
     * QUANTIZATION (8, append-only enum). The Quantization node follows model
     * B, so its index 3 collides with none of the message's other numbers
     * (index 1 would: "microBatchSize 2 > 1"). The offending field is pinned
     * by identity, not wording: whatever layerNonFloat32Field returns for the
     * node must appear in the message (§6.6 "names the offending field"). */
    quantization_t *floatQ = quantizationInitFloat();
    quantization_t *symQ = quantizationInitSymInt32(HALF_AWAY);
    layerQuant_t lq;
    layerQuantInitUniform(&lq, floatQ);
    layerQuant_t toSym;
    layerQuantInitUniform(&toSym, symQ);
    rngSetSeed(22u);
    layer_t *model[B_SIZE + 1];
    buildModelB(model, &lq);
    model[B_SIZE] = quantLayerInit(&toSym);
    initModelBData();
    char message[512];
    int code = -2;

    CAPTURE_EXIT_AND_OUTPUT(trainStackedMseOrDie(model, B_SIZE + 1), message, sizeof message,
                            &code);
    /* A string literal (Task 2), so it stays valid after the frees below. */
    const char *field = layerNonFloat32Field(model[B_SIZE]);

    freeModelBData();
    freeQuantLayer(model[B_SIZE]);
    freeModelB(model);
    freeQuantization(symQ);
    freeQuantization(floatQ);

    TEST_ASSERT_EQUAL_INT_MESSAGE(1, code, "the gate must exit(1)");
    TEST_ASSERT_NOT_NULL_MESSAGE(strstr(message, "FLOAT32"), "the gate must say FLOAT32");
    TEST_ASSERT_TRUE_MESSAGE(messageHasNumber(message, 2), "the gate must name m (2)");
    TEST_ASSERT_TRUE_MESSAGE(messageHasNumber(message, B_SIZE),
                             "the gate must name the layer index (3)");
    TEST_ASSERT_TRUE_MESSAGE(messageHasNumber(message, (size_t)QUANTIZATION),
                             "the gate must name the layerType_t (QUANTIZATION = 8)");
    TEST_ASSERT_NOT_NULL_MESSAGE(field, "the Quantization node must have a non-FLOAT32 field");
    TEST_ASSERT_NOT_NULL_MESSAGE(strstr(message, field), "the gate must name the offending field");
}

void testStackedGateRejectsPerOpSymMathOnFloat32Storage(void) {
    quantization_t *floatQ = quantizationInitFloat();
    layerQuant_t lq;
    layerQuantInitUniform(&lq, floatQ);
    lq.weightGradMath = (arithmetic_t){.type = ARITH_SYM_INT32, .roundingMode = HALF_AWAY};
    rngSetSeed(23u);
    layer_t *model[1] = {
        linearLayerInit(&(linearInit_t){.inFeatures = B_IN, .outFeatures = B_OUT}, &lq)};

    assertStackedGateRejects(model, 1);

    freeLinearLayer(model[0]);
    freeQuantization(floatQ);
}

void testStackedGateRejectsSymGradStorage(void) {
    quantization_t *floatQ = quantizationInitFloat();
    quantization_t *symQ = quantizationInitSymInt32(HALF_AWAY);
    layerQuant_t lq;
    layerQuantInitUniform(&lq, floatQ);
    lq.weightGradStorage = symQ; /* FLOAT32 math and wires, SYM_INT32 grads */
    rngSetSeed(24u);
    layer_t *model[1] = {
        linearLayerInit(&(linearInit_t){.inFeatures = B_IN, .outFeatures = B_OUT}, &lq)};

    assertStackedGateRejects(model, 1);

    freeLinearLayer(model[0]);
    freeQuantization(symQ);
    freeQuantization(floatQ);
}

void testStackedTrainingWithFrozenFirstLayerMatchesPerSample(void) {
    /* A frozen layer has NULL grads (#380): the gate must accept it without
     * dereferencing them, and the trainable layer's grads must match m = 1. */
    initModelBData();
    quantization_t *q = quantizationInitFloat();
    layerQuant_t lq;
    layerQuantInitUniform(&lq, q);
    rngSetSeed(25u);
    layer_t *model[B_SIZE];
    model[0] = linearLayerInit(
        &(linearInit_t){.inFeatures = B_IN, .outFeatures = 4, .trainable = TRAINABLE_FALSE}, &lq);
    model[1] = reluLayerInit(&lq);
    model[2] = linearLayerInit(&(linearInit_t){.inFeatures = 4, .outFeatures = B_OUT}, &lq);
    float ref[4 * B_OUT + B_OUT];
    float got[4 * B_OUT + B_OUT];
    float loss[2];
    const size_t ms[2] = {1, 2};
    for (size_t k = 0; k < 2; k++) {
        zeroGrads(model, B_SIZE);
        batch_t *batch = buildBatch(bItems, bLabels, 4);
        loss[k] = trainingBatchDefault(model, B_SIZE, defaultLossConfig(MSE), batch,
                                       calculateGradsSequential, REDUCTION_MEAN, ms[k]);
        freeBatch(batch);
        snapshotScaledGrads(model, B_SIZE, 1.0f, k == 0 ? ref : got, 4 * B_OUT + B_OUT);
    }
    size_t mismatch = firstMismatch(ref, got, 4 * B_OUT + B_OUT, 1e-6f, 1e-4f);

    freeModelB(model);
    freeQuantization(q);
    freeModelBData();

    TEST_ASSERT_EQUAL_size_t_MESSAGE(SIZE_MAX, mismatch, "trainable grads diverge from m=1");
    TEST_ASSERT_FLOAT_WITHIN(1e-6f + 1e-5f * fabsf(loss[0]), loss[0], loss[1]);
}

/* ---- trainingEpochDefault (spec §6.1, §6.2 check 3) ------------------------ */

/* Classifier: Linear(5->4) -> ReLU -> Linear(4->3) -> Softmax, CrossEntropy,
 * over 8 samples ([5] items, one-hot [3] labels). */
#define CLS_N 8
#define CLS_IN 5
#define CLS_OUT 3
#define CLS_SIZE 4
#define CLS_PARAMS (CLS_IN * 4 + 4 + 4 * CLS_OUT + CLS_OUT) /* 39 */

static tensor_t *clsItems[CLS_N];
static tensor_t *clsLabels[CLS_N];

static void initClassifierData(void) {
    rngSetSeed(1524u);
    for (size_t s = 0; s < CLS_N; s++) {
        float item[CLS_IN];
        fillRandom(item, CLS_IN);
        float label[CLS_OUT] = {0.0f, 0.0f, 0.0f};
        label[s % CLS_OUT] = 1.0f;
        clsItems[s] = buildFloatTensor((size_t[]){CLS_IN}, 1, item);
        clsLabels[s] = buildFloatTensor((size_t[]){CLS_OUT}, 1, label);
    }
}

static void freeClassifierData(void) {
    for (size_t s = 0; s < CLS_N; s++) {
        freeTensor(clsLabels[s]);
        freeTensor(clsItems[s]);
    }
}

static sample_t *getClassifierSample(size_t id) {
    sample_t *s = reserveMemory(sizeof(sample_t));
    s->item = clsItems[id];
    s->label = clsLabels[id];
    return s;
}

static size_t getClassifierDatasetSize(void) {
    return CLS_N;
}

/* A dataset that must never be read: a check that has to fire BEFORE the
 * first batch is drawn exits 1; drawing a sample exits 3 instead, so the
 * death test tells "the early check fired" apart from "a later check caught
 * it after a batch was drawn". */
static sample_t *getTripwireSample(size_t id) {
    (void)id;
    _exit(3);
}

/* A replay-style loader: one sample more than its batchSize per batch. The
 * samples array carries one SPARE valid entry past batch->size, so a missing
 * check 4 trains the ragged tail silently (exit 0) instead of reading past
 * the array -- the backstop death test then pins check 4 itself. */
static batch_t *getBatchWithOneExtraSample(dataLoader_t *dataLoader, size_t index) {
    size_t n = (size_t)dataLoader->batchSize + 1;
    batch_t *batch = reserveMemory(sizeof(batch_t));
    batch->size = n;
    batch->samples = reserveMemory((n + 1) * sizeof(sample_t *));
    for (size_t i = 0; i < n + 1; i++) {
        batch->samples[i] = getClassifierSample((index * dataLoader->batchSize + i) % CLS_N);
    }
    return batch;
}

static void buildClassifier(layer_t **model, layerQuant_t *lq) {
    rngSetSeed(4242u); /* identical initial weights on every build */
    model[0] = linearLayerInit(&(linearInit_t){.inFeatures = CLS_IN, .outFeatures = 4}, lq);
    model[1] = reluLayerInit(lq);
    model[2] = linearLayerInit(&(linearInit_t){.inFeatures = 4, .outFeatures = CLS_OUT}, lq);
    model[3] = softmaxLayerInit(lq);
}

/* freeOptim owns the parameters; the Linear layers are torn down shell-only. */
static void freeClassifierShells(layer_t **model) {
    freeSoftmaxLayer(model[3]);
    freeReservedMemory(model[2]->config->linear);
    freeReservedMemory(model[2]->config);
    freeReservedMemory(model[2]);
    freeReluLayer(model[1]);
    freeReservedMemory(model[0]->config->linear);
    freeReservedMemory(model[0]->config);
    freeReservedMemory(model[0]);
}

static optimizer_t *buildSgd(layer_t **model, quantization_t *momentumQ) {
    optimizer_t *sgd =
        sgdMCreateOptim(0.5f, 0.0f, 0.0f, model, CLS_SIZE, momentumQ,
                        (arithmetic_t){.type = ARITH_FLOAT32, .roundingMode = HALF_AWAY});
    optimizerSetWriteBackRounding(sgd, HALF_AWAY); /* deterministic write-backs */
    return sgd;
}

static size_t snapshotParams(layer_t **model, size_t modelSize, float *out, size_t capacity) {
    parameter_t *slots[MAX_PARAMS];
    size_t n = calcTotalNumberOfStates(model, modelSize);
    collectTrainableParameters(model, modelSize, slots);
    size_t k = 0;
    for (size_t p = 0; p < n; p++) {
        const float *v = (const float *)slots[p]->param->data;
        size_t count = calcNumberOfElementsByTensor(slots[p]->param);
        for (size_t i = 0; i < count; i++) {
            if (k < capacity) {
                out[k] = v[i];
            }
            k++;
        }
    }
    return k;
}

/* One epoch of the classifier at the given batch and micro-batch size. */
static float runClassifierEpoch(uint16_t batchSize, size_t microBatchSize,
                                calculateGradsFn_t calculateGradsFn, float *params) {
    quantization_t *q = quantizationInitFloat();
    layerQuant_t lq;
    layerQuantInitUniform(&lq, q);
    layer_t *model[CLS_SIZE];
    buildClassifier(model, &lq);
    quantization_t *momentumQ = quantizationInitFloat();
    optimizer_t *sgd = buildSgd(model, momentumQ);
    dataLoader_t *dl = dataLoaderInit(getClassifierSample, getClassifierDatasetSize, batchSize,
                                      NULL, NULL, false, 0, true);
    float loss = trainingEpochDefault(model, CLS_SIZE, defaultLossConfig(CROSS_ENTROPY), dl, sgd,
                                      calculateGradsFn, REDUCTION_MEAN, microBatchSize);
    snapshotParams(model, CLS_SIZE, params, CLS_PARAMS);
    freeDataLoader(dl);
    freeOptim(sgd);
    freeQuantization(momentumQ);
    freeClassifierShells(model);
    freeQuantization(q);
    return loss;
}

static void runEpochOnLoaderOrDie(dataLoader_t *dl, size_t microBatchSize) {
    quantization_t *q = quantizationInitFloat();
    layerQuant_t lq;
    layerQuantInitUniform(&lq, q);
    layer_t *model[CLS_SIZE];
    buildClassifier(model, &lq);
    quantization_t *momentumQ = quantizationInitFloat();
    optimizer_t *sgd = buildSgd(model, momentumQ);
    trainingEpochDefault(model, CLS_SIZE, defaultLossConfig(CROSS_ENTROPY), dl, sgd,
                         calculateGradsSequential, REDUCTION_MEAN, microBatchSize);
}

void testTrainingEpochDefaultRejectsBatchNotDivisibleByMicroBatch(void) {
    /* Check 3: loader batch 6, m = 4 -- dies before the first batch is drawn
     * (exit 1, not the tripwire's 3: that is what tells it from check 4,
     * which would print the same 6 and 4) and names b and m (ruling R10:
     * numbers plus the keyword "microBatchSize"). */
    dataLoader_t *dl =
        dataLoaderInit(getTripwireSample, getClassifierDatasetSize, 6, NULL, NULL, false, 0, true);
    char message[512];
    int code = -2;

    CAPTURE_EXIT_AND_OUTPUT(runEpochOnLoaderOrDie(dl, 4), message, sizeof message, &code);

    freeDataLoader(dl);
    TEST_ASSERT_EQUAL_INT_MESSAGE(1, code, "check 3 must exit(1) before any sample is read");
    TEST_ASSERT_NOT_NULL_MESSAGE(strstr(message, "microBatchSize"), "check 3 must name the knob");
    TEST_ASSERT_TRUE_MESSAGE(messageHasNumber(message, 6), "check 3 must name b (6)");
    TEST_ASSERT_TRUE_MESSAGE(messageHasNumber(message, 4), "check 3 must name m (4)");
}

void testTrainingEpochDefaultBackstopsAnIndivisibleReplayBatch(void) {
    /* Loader batchSize 4 passes check 3 at m = 2, but its batches carry 5
     * samples (the replay loader's base + eligible * r shape): check 4 inside
     * trainingBatchDefault must stop it (spec §6.2 known limitation). The
     * message's b is the batch's 5 -- a number check 3, which only sees the
     * loader's 4, can never print. */
    initClassifierData();
    dataLoader_t *dl = dataLoaderInit(getClassifierSample, getClassifierDatasetSize, 4, NULL, NULL,
                                      false, 0, true);
    dl->getBatch = getBatchWithOneExtraSample;
    char message[512];
    int code = -2;

    CAPTURE_EXIT_AND_OUTPUT(runEpochOnLoaderOrDie(dl, 2), message, sizeof message, &code);

    freeDataLoader(dl);
    freeClassifierData();
    TEST_ASSERT_EQUAL_INT_MESSAGE(1, code, "check 4 must stop the ragged replay batch");
    TEST_ASSERT_NOT_NULL_MESSAGE(strstr(message, "microBatchSize"), "check 4 must name the knob");
    TEST_ASSERT_TRUE_MESSAGE(messageHasNumber(message, 5),
                             "check 4 must name the batch's b (5), not the loader's 4");
    TEST_ASSERT_TRUE_MESSAGE(messageHasNumber(message, 2), "check 4 must name m (2)");
}

void testTrainingEpochDefaultMicroBatchZeroEqualsOne(void) {
    initClassifierData();
    float paramsZero[CLS_PARAMS];
    float paramsOne[CLS_PARAMS];
    resetRecording();
    float lossZero = runClassifierEpoch(4, 0, recordingGrads, paramsZero);
    size_t callsZero = g_recCalls;
    size_t rowsZero = g_recItemRows[0];
    float lossOne = runClassifierEpoch(4, 1, calculateGradsSequential, paramsOne);
    freeClassifierData();

    TEST_ASSERT_EQUAL_size_t(CLS_N, callsZero);
    TEST_ASSERT_EQUAL_size_t(1, rowsZero);
    TEST_ASSERT_EQUAL_MEMORY(&lossOne, &lossZero, sizeof(float));
    TEST_ASSERT_EQUAL_MEMORY(paramsOne, paramsZero, sizeof(paramsOne));
}

void testTrainingEpochDefaultStacksAndTracksPerSample(void) {
    /* Two optimizer steps (8 samples, batch 4): m = 4 must really stack
     * (one 4-row call per batch) and land where m = 1 lands. */
    initClassifierData();
    float paramsOne[CLS_PARAMS];
    float paramsFour[CLS_PARAMS];
    float lossOne = runClassifierEpoch(4, 1, calculateGradsSequential, paramsOne);
    resetRecording();
    float lossFour = runClassifierEpoch(4, 4, recordingGrads, paramsFour);
    size_t calls = g_recCalls;
    size_t rows[2] = {g_recItemRows[0], g_recItemRows[1]};
    freeClassifierData();

    TEST_ASSERT_EQUAL_size_t(2, calls);
    TEST_ASSERT_EQUAL_size_t(4, rows[0]);
    TEST_ASSERT_EQUAL_size_t(4, rows[1]);
    TEST_ASSERT_EQUAL_size_t_MESSAGE(SIZE_MAX,
                                     firstMismatch(paramsOne, paramsFour, CLS_PARAMS, 1e-6f, 1e-4f),
                                     "stacked epoch diverges from m=1");
    TEST_ASSERT_FLOAT_WITHIN(1e-6f + 1e-5f * fabsf(lossOne), lossOne, lossFour);
}

/* ---- trainingRun (spec §6.1, §6.2 checks 1 and 2) -------------------------- */

typedef enum {
    NO_SCHEDULER,
    FRESH_SCHEDULER,
    FRESH_SCHEDULER_CAP5,
    SCHEDULER_STEPPED_TWICE,
    NONFINITE_SCHEDULER
} schedulerMode_t;

static float g_epochLoss[8];
static size_t g_epochCount;

static void captureEpochLoss(epochInfo_t info, epochStats_t evalStats) {
    (void)evalStats;
    if (g_epochCount < 8) {
        g_epochLoss[g_epochCount] = info.trainLoss;
    }
    g_epochCount++;
}

/* The scheduler fixtures: b0 = 2, gamma = 2/3, max = 4 gives 2, 3, 4, 4, ...
 * (odd at epoch 1 only); FRESH_SCHEDULER_CAP5 (max = 5) gives 2, 3, 4, 5
 * (odd at epochs 1 AND 3). Both pinned by UnitTestBsScheduler's
 * testBatchSizeAtFollowsTheCappedTwoThirdsSchedule. NONFINITE_SCHEDULER
 * (gamma = 1e20, max = 4) clamps to 1 from epoch 1 and its target leaves the
 * double range at epoch 16 (testBatchSizeAtRejectsNonFiniteTarget there). */
static trainingRunResult_t runClassifier(getSampleFn_t getSample, uint16_t batchSize,
                                         size_t numberOfEpochs, schedulerMode_t mode,
                                         const trainingRunOptions_t *baseOptions,
                                         calculateGradsFn_t calculateGradsFn, float *params) {
    quantization_t *q = quantizationInitFloat();
    layerQuant_t lq;
    layerQuantInitUniform(&lq, q);
    layer_t *model[CLS_SIZE];
    buildClassifier(model, &lq);
    quantization_t *momentumQ = quantizationInitFloat();
    optimizer_t *sgd = buildSgd(model, momentumQ);
    dataLoader_t *trainDl =
        dataLoaderInit(getSample, getClassifierDatasetSize, batchSize, NULL, NULL, false, 0, true);
    dataLoader_t *evalDl =
        dataLoaderInit(getSample, getClassifierDatasetSize, 1, NULL, NULL, false, 0, true);
    bsScheduler_t bs;
    trainingRunOptions_t options = {0};
    if (baseOptions != NULL) {
        options = *baseOptions;
    }
    if (mode != NO_SCHEDULER) {
        float gamma = (mode == NONFINITE_SCHEDULER) ? 1e20f : 0.6666667f;
        exponentialBsInit(&bs, trainDl, NULL, gamma, mode == FRESH_SCHEDULER_CAP5 ? 5 : 4);
        if (mode == SCHEDULER_STEPPED_TWICE) {
            bsSchedulerStep(&bs); /* loader batch 3 */
            bsSchedulerStep(&bs); /* loader batch 4 */
        }
        options.bsScheduler = &bs;
    }
    g_epochCount = 0;
    trainingRunResult_t result =
        trainingRun(model, CLS_SIZE, defaultLossConfig(CROSS_ENTROPY), trainDl, evalDl, sgd,
                    numberOfEpochs, calculateGradsFn, inferenceWithLoss,
                    (baseOptions == NULL && mode == NO_SCHEDULER) ? NULL : &options);
    if (params != NULL) {
        snapshotParams(model, CLS_SIZE, params, CLS_PARAMS);
    }
    freeDataLoader(evalDl);
    freeDataLoader(trainDl);
    freeOptim(sgd);
    freeQuantization(momentumQ);
    freeClassifierShells(model);
    freeQuantization(q);
    return result;
}

static void runClassifierOrDie(uint16_t batchSize, size_t numberOfEpochs, schedulerMode_t mode,
                               size_t microBatchSize) {
    trainingRunOptions_t options = {.microBatchSize = microBatchSize};
    runClassifier(getTripwireSample, batchSize, numberOfEpochs, mode, &options,
                  calculateGradsSequential, NULL);
}

void testTrainingRunRejectsLoaderBatchNotDivisibleByMicroBatch(void) {
    /* Check 1: before epoch 0 -- before even the eval loader's numClasses
     * peek, so the tripwire dataset is never read (exit 1, not 3: that is what
     * tells it from checks 3 and 4, which would print the same 3 and 2) --
     * and the message names b and m (ruling R10: numbers plus the keyword
     * "microBatchSize"). */
    char message[512];
    int code = -2;

    CAPTURE_EXIT_AND_OUTPUT(runClassifierOrDie(3, 2, NO_SCHEDULER, 2), message, sizeof message,
                            &code);

    TEST_ASSERT_EQUAL_INT_MESSAGE(1, code, "check 1 must exit(1) before any sample is read");
    TEST_ASSERT_NOT_NULL_MESSAGE(strstr(message, "microBatchSize"), "check 1 must name the knob");
    TEST_ASSERT_TRUE_MESSAGE(messageHasNumber(message, 3), "check 1 must name b (3)");
    TEST_ASSERT_TRUE_MESSAGE(messageHasNumber(message, 2), "check 1 must name m (2)");
}

void testTrainingRunRejectsScheduleNotDivisibleByMicroBatch(void) {
    /* Check 2: the loader batch (2) divides, but the max=5 schedule 2, 3, 4, 5
     * is odd at epochs 1 AND 3. The walk fires before epoch 0 trains (exit 1,
     * the tripwire dataset is never read) and names the FIRST failing epoch,
     * its batch and m (spec §6.2; ruling R10: numbers plus the keyword
     * "microBatchSize"). Epoch 1 and the absence of epoch 3's batch 5 are the
     * numbers that tell "first" from "any" failing epoch. 5 is also this
     * fixture's maxBatchSize, so the message must not print the cap either. */
    char message[512];
    int code = -2;

    CAPTURE_EXIT_AND_OUTPUT(runClassifierOrDie(2, 4, FRESH_SCHEDULER_CAP5, 2), message,
                            sizeof message, &code);

    TEST_ASSERT_EQUAL_INT_MESSAGE(1, code, "check 2 must exit(1) before any sample is read");
    TEST_ASSERT_NOT_NULL_MESSAGE(strstr(message, "microBatchSize"), "check 2 must name the knob");
    TEST_ASSERT_TRUE_MESSAGE(messageHasNumber(message, 3), "check 2 must name the batch (3)");
    TEST_ASSERT_TRUE_MESSAGE(messageHasNumber(message, 1),
                             "check 2 must name the first failing epoch (1)");
    TEST_ASSERT_FALSE_MESSAGE(messageHasNumber(message, 5),
                              "check 2 must stop at epoch 1, not report epoch 3's batch 5");
    TEST_ASSERT_TRUE_MESSAGE(messageHasNumber(message, 2), "check 2 must name m (2)");
}

void testTrainingRunScheduleWalkSkipsTheUntrainedFinalStep(void) {
    /* One epoch: the step after it writes 3, but no epoch trains at 3, so
     * the walk over [1, numberOfEpochs) is empty and the run completes. */
    initClassifierData();
    trainingRunOptions_t options = {.microBatchSize = 2};
    trainingRunResult_t result = runClassifier(getClassifierSample, 2, 1, FRESH_SCHEDULER, &options,
                                               calculateGradsSequential, NULL);
    freeClassifierData();
    TEST_ASSERT_EQUAL_size_t(1, result.epochsCompleted);
}

void testTrainingRunScheduleWalkStartsAtTheSchedulersLastEpoch(void) {
    /* Stepped twice before the run (lastEpoch 2, loader batch 4): epochs 1
     * and 2 of THIS run train at the batches of lastEpoch 3 and 4 (4, 4),
     * not at those of 1 and 2 (3, 4). */
    initClassifierData();
    trainingRunOptions_t options = {.microBatchSize = 2};
    trainingRunResult_t result = runClassifier(getClassifierSample, 2, 3, SCHEDULER_STEPPED_TWICE,
                                               &options, calculateGradsSequential, NULL);
    freeClassifierData();
    TEST_ASSERT_EQUAL_size_t(3, result.epochsCompleted);
}

void testTrainingRunNeverWalksTheScheduleAtMicroBatch1(void) {
    /* Ruling R9: check 2 walks the schedule only at m > 1. Walked over 17
     * epochs, the NONFINITE_SCHEDULER target would fail fast at epoch 16
     * (exit 1) before epoch 0; unwalked, the run passes every pre-epoch-0
     * check and reads its first sample, where the tripwire dataset exits 3.
     * The default 0 and an explicit 1 must both skip the walk. */
    ASSERT_EXITS_WITH(3, runClassifierOrDie(2, 17, NONFINITE_SCHEDULER, 0));
    ASSERT_EXITS_WITH(3, runClassifierOrDie(2, 17, NONFINITE_SCHEDULER, 1));
}

void testTrainingRunMicroBatchDefaultsAreIdentical(void) {
    /* NULL options, zero-initialised options (microBatchSize 0) and an
     * explicit 1 must train bit-identically (spec §6.1: 0 means 1). */
    initClassifierData();
    float paramsNull[CLS_PARAMS];
    float paramsZero[CLS_PARAMS];
    float paramsOne[CLS_PARAMS];
    trainingRunResult_t rNull = runClassifier(getClassifierSample, 4, 2, NO_SCHEDULER, NULL,
                                              calculateGradsSequential, paramsNull);
    trainingRunOptions_t zero = {0};
    resetRecording();
    trainingRunResult_t rZero =
        runClassifier(getClassifierSample, 4, 2, NO_SCHEDULER, &zero, recordingGrads, paramsZero);
    size_t callsZero = g_recCalls;
    trainingRunOptions_t one = {.microBatchSize = 1};
    trainingRunResult_t rOne = runClassifier(getClassifierSample, 4, 2, NO_SCHEDULER, &one,
                                             calculateGradsSequential, paramsOne);
    freeClassifierData();

    TEST_ASSERT_EQUAL_size_t(2 * CLS_N, callsZero);
    TEST_ASSERT_EQUAL_MEMORY(paramsNull, paramsZero, sizeof(paramsNull));
    TEST_ASSERT_EQUAL_MEMORY(paramsNull, paramsOne, sizeof(paramsNull));
    TEST_ASSERT_EQUAL_MEMORY(&rNull.finalTrainLoss, &rZero.finalTrainLoss, sizeof(float));
    TEST_ASSERT_EQUAL_MEMORY(&rNull.finalTrainLoss, &rOne.finalTrainLoss, sizeof(float));
}

void testTrainingRunSmokeStackedTracksPerSample(void) {
    /* A few epochs of the tiny classifier at m = 4: the loss decreases and
     * ends close to m = 1; every calculateGradsFn call sees 4 rows. */
    initClassifierData();
    trainingRunOptions_t perSample = {.callback = captureEpochLoss, .microBatchSize = 1};
    runClassifier(getClassifierSample, 4, 5, NO_SCHEDULER, &perSample, calculateGradsSequential,
                  NULL);
    float lastPerSample = g_epochLoss[4];
    trainingRunOptions_t stacked = {.callback = captureEpochLoss, .microBatchSize = 4};
    resetRecording();
    runClassifier(getClassifierSample, 4, 5, NO_SCHEDULER, &stacked, recordingGrads, NULL);
    float firstStacked = g_epochLoss[0];
    float lastStacked = g_epochLoss[4];
    size_t calls = g_recCalls;
    size_t minRows = SIZE_MAX;
    for (size_t c = 0; c < REC_MAX_CALLS && c < calls; c++) {
        minRows = g_recItemRows[c] < minRows ? g_recItemRows[c] : minRows;
    }
    freeClassifierData();

    TEST_ASSERT_EQUAL_size_t(5 * 2, calls);
    TEST_ASSERT_EQUAL_size_t(4, minRows);
    TEST_ASSERT_TRUE_MESSAGE(lastStacked < firstStacked, "stacked training must reduce the loss");
    TEST_ASSERT_FLOAT_WITHIN(1e-4f + 1e-3f * fabsf(lastPerSample), lastPerSample, lastStacked);
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(testStackedChunkWalkCallsOncePerChunkWithMRows);
    RUN_TEST(testMicroBatchZeroBehavesExactlyLikeOne);
    RUN_TEST(testStackedLossIsRowWeightedForMeanAndPlainForSum);
    RUN_TEST(testStackedMatchesPerSampleOnModelA);
    RUN_TEST(testStackedMatchesPerSampleOnMseModelB);
    RUN_TEST(testStackedBatchMatchesPyTorchGold);
    RUN_TEST(testTrainingBatchDefaultRejectsBatchNotDivisibleByMicroBatch);
    RUN_TEST(testStackedRejectsEmptyBatch);
    RUN_TEST(testStackedGatherSizeOverflowFailsFast);
    RUN_TEST(testStackedGatherReservationFailureFailsFast);
    RUN_TEST(testStackedDropoutFailsFastOnItsMaskCount);
    RUN_TEST(testStackedRejectsItemWithDifferentDims);
    RUN_TEST(testStackedRejectsItemWithDifferentRank);
    RUN_TEST(testStackedRejectsItemWithDifferentOrder);
    RUN_TEST(testStackedRejectsNonFloat32Item);
    RUN_TEST(testStackedRejectsNonFloat32FirstSample);
    RUN_TEST(testStackedRejectsLabelWithSparsity);
    RUN_TEST(testStackedRejectsLabelWithDifferentRank);
    RUN_TEST(testStackedRejectsMismatchInALaterChunk);
    RUN_TEST(testStackedRejectsChunkDifferingFromSampleZero);
    RUN_TEST(testStackedGateRejectsSymLayer);
    RUN_TEST(testStackedGateRejectsBfpLayer);
    RUN_TEST(testStackedGateRejectsBfpWire);
    RUN_TEST(testStackedGateRejectsQuantizationLayer);
    RUN_TEST(testStackedGateRejectsPerOpSymMathOnFloat32Storage);
    RUN_TEST(testStackedGateRejectsSymGradStorage);
    RUN_TEST(testStackedTrainingWithFrozenFirstLayerMatchesPerSample);
    RUN_TEST(testTrainingEpochDefaultRejectsBatchNotDivisibleByMicroBatch);
    RUN_TEST(testTrainingEpochDefaultBackstopsAnIndivisibleReplayBatch);
    RUN_TEST(testTrainingEpochDefaultMicroBatchZeroEqualsOne);
    RUN_TEST(testTrainingEpochDefaultStacksAndTracksPerSample);
    RUN_TEST(testTrainingRunRejectsLoaderBatchNotDivisibleByMicroBatch);
    RUN_TEST(testTrainingRunRejectsScheduleNotDivisibleByMicroBatch);
    RUN_TEST(testTrainingRunScheduleWalkSkipsTheUntrainedFinalStep);
    RUN_TEST(testTrainingRunScheduleWalkStartsAtTheSchedulersLastEpoch);
    RUN_TEST(testTrainingRunNeverWalksTheScheduleAtMicroBatch1);
    RUN_TEST(testTrainingRunMicroBatchDefaultsAreIdentical);
    RUN_TEST(testTrainingRunSmokeStackedTracksPerSample);
    return UNITY_END();
}
