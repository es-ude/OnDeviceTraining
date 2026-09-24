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
#include "LayerNormApi.h"
#include "LayerQuant.h"
#include "LayerWeightsApi.h"
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
    return UNITY_END();
}
