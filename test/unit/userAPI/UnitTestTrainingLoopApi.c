#define SOURCE_FILE "UNIT_TEST_TRAINING_LOOP_API"

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include "ArithmeticType.h"
#include "BatchView.h"
#include "BorrowedLayer.h"
#include "BsScheduler.h"
#include "CalculateGradsSequential.h"
#include "DataLoaderApi.h"
#include "Dataset.h"
#include "DeathTest.h"
#include "InferenceApi.h"
#include "Linear.h"
#include "LinearApi.h"
#include "LossFunction.h"
#include "LrScheduler.h"
#include "OdtAssert.h"
#include "OptimizerApi.h"
#include "QuantizationApi.h"
#include "RNG.h"
#include "ReluApi.h"
#include "RematPlan.h"
#include "RematScheduler.h"
#include "Sgd.h"
#include "SgdApi.h"
#include "StorageApi.h"
#include "Tensor.h"
#include "TensorApi.h"
#include "TensorConversion.h"
#include "TrainingBatchDefault.h"
#include "TrainingEpochDefault.h"
#include "TrainingLoopApi.h"
#include "unity.h"

/* Batch-size scheduler port: trainingRun's optional inputs (LR scheduler,
 * batch-size scheduler, callback) travel in ONE NULL-able options struct as
 * the last argument; the callback receives epochInfo_t. Compile-time pins. */
_Static_assert(_Generic(&trainingRun,
                   trainingRunResult_t (*)(layer_t **, size_t, lossConfig_t, dataLoader_t *,
                                           dataLoader_t *, optimizer_t *, size_t,
                                           calculateGradsFn_t, inferenceWithLossFn_t,
                                           const trainingRunOptions_t *): 1,
                   default: 0),
               "trainingRun must be (model, modelSize, lossConfig, trainDl, evalDl, optimizer, "
               "numberOfEpochs, calculateGradsFn, inferenceFn, options)");
_Static_assert(_Generic((epochCallbackFn_t)0, void (*)(epochInfo_t, epochStats_t): 1, default: 0),
               "epochCallbackFn_t must be (epochInfo_t info, epochStats_t evalStats)");

/* Build a fresh 2-D float tensor from a literal float buffer. Encapsulates the
 * post-#106 chain so each test stays readable. */
static tensor_t *buildFloatTensor2D(size_t d0, size_t d1, const float *src, size_t count) {
    size_t *dims = reserveMemory(2 * sizeof(size_t));
    dims[0] = d0;
    dims[1] = d1;
    size_t *order = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, order);
    shape_t *shape = reserveMemory(sizeof(shape_t));
    setShape(shape, dims, 2, order);
    tensor_t *t = initTensor(shape, quantizationInitFloat(), NULL);
    tensorFillFromFloatBuffer(t, src, count);
    return t;
}

/* Rank-1 [n] float tensor: the NATURAL shape of a feature-vector sample or a
 * class-vector label. batch_t consumers add the batch axis themselves (#152
 * PR3a); a direct tensor-level call wraps it with batchViewOf. */
static tensor_t *buildFloatTensor1D(size_t n, const float *src) {
    size_t *dims = reserveMemory(sizeof(size_t));
    dims[0] = n;
    size_t *order = reserveMemory(sizeof(size_t));
    setOrderOfDimsForNewTensor(1, order);
    shape_t *shape = reserveMemory(sizeof(shape_t));
    setShape(shape, dims, 1, order);
    tensor_t *t = initTensor(shape, quantizationInitFloat(), NULL);
    tensorFillFromFloatBuffer(t, src, n);
    return t;
}

void testCalculateGradsSequential_MatchesPyTorch() {
    tensor_t *weightsParam = buildFloatTensor2D(2, 3, (float[]){1.f, 1.f, 1.f, 1.f, 1.f, 1.f}, 6);
    tensor_t *weightsGrad = gradInitFloat(weightsParam, NULL);
    parameter_t *weights = parameterInit(weightsParam, weightsGrad);

    tensor_t *biasParam = buildFloatTensor2D(1, 2, (float[]){-1.f, 3.f}, 2);
    tensor_t *biasGrad = gradInitFloat(biasParam, NULL);
    parameter_t *bias = parameterInit(biasParam, biasGrad);

    quantization_t testQ;
    initFloat32Quantization(&testQ);
    layer_t *linear = buildBorrowedLinearLayer(weights, bias, &testQ);

    layer_t *model[] = {linear};
    size_t sizeModel = 1;

    tensor_t *input0 = buildFloatTensor2D(1, 3, (float[]){-4.f, 1.f, 9.f}, 3);
    tensor_t *input1 = buildFloatTensor2D(1, 3, (float[]){5.f, -1.f, 2.f}, 3);
    tensor_t *input2 = buildFloatTensor2D(1, 3, (float[]){-7.f, -5.f, 6.f}, 3);

    tensor_t *label0 = buildFloatTensor2D(1, 2, (float[]){59.f, -23.f}, 2);
    tensor_t *label1 = buildFloatTensor2D(1, 2, (float[]){43.f, 249.f}, 2);
    tensor_t *label2 = buildFloatTensor2D(1, 2, (float[]){23.f, 457.f}, 2);

    quantization_t *momentumQ = quantizationInitFloat();
    optimizer_t *sgd =
        sgdMCreateOptim(0.01f, 0.f, 0.f, model, sizeModel, momentumQ,
                        (arithmetic_t){.type = ARITH_FLOAT32, .roundingMode = HALF_AWAY});
    optimizerFunctions_t sgdFns = optimizerFunctions[SGD_M];
    /* Pre-existing test hack: only step weights, leaving bias unchanged across
     * iterations. freeOptim frees the registered parameters; no state
     * buffers exist at momentum==0 (#308). */
    sgd->sizeStates = 1;

    for (size_t i = 0; i < 23; i++) {
        trainingStats_t *ts0 = calculateGradsSequential(
            model, sizeModel, (lossConfig_t){.funcType = MSE, .backwardReduction = REDUCTION_SUM},
            REDUCTION_SUM, input0, label0, NULL);
        trainingStats_t *ts1 = calculateGradsSequential(
            model, sizeModel, (lossConfig_t){.funcType = MSE, .backwardReduction = REDUCTION_SUM},
            REDUCTION_SUM, input1, label1, NULL);
        trainingStats_t *ts2 = calculateGradsSequential(
            model, sizeModel, (lossConfig_t){.funcType = MSE, .backwardReduction = REDUCTION_SUM},
            REDUCTION_SUM, input2, label2, NULL);

        /* PyTorch reference is MSE-mean-of-features (`2/F * (o-l)` per element). Post-#135
         * backward writes raw `2*(o-l)`; recover the reference trajectory via explicit
         * 1/F = 1/2 scaling here. trainingEpochDefault does this implicitly when
         * backwardReduction == REDUCTION_MEAN; this test exercises calculateGradsSequential
         * directly, so the scaling must be manual. */
        scaleOptimizerGradients(sgd, 1.0f / 2.0f, NULL);

        sgdFns.step(sgd, NULL);
        sgdFns.zero(sgd);

        freeTrainingStats(ts0);
        freeTrainingStats(ts1);
        freeTrainingStats(ts2);
    }

    /* CAPTURE assertion values into stack locals BEFORE any free. */
    float expectedWeights[] = {5.f, -1.f, 9.f, 22.f, -100.f, 18.f};
    linearConfig_t *linearConfig = linear->config->linear;
    float capturedActualWeights[6];
    {
        float *actualWeights = (float *)linearConfig->weights->param->data;
        for (size_t i = 0; i < 6; i++) {
            capturedActualWeights[i] = actualWeights[i];
        }
    }

    /* FREE in reverse-init order. Restore sizeStates so freeOptim cascades
     * to both weights and bias parameters + their state buffers. */
    sgd->sizeStates = 2;
    freeOptim(sgd);
    freeQuantization(momentumQ);
    freeTensor(label2);
    freeTensor(label1);
    freeTensor(label0);
    freeTensor(input2);
    freeTensor(input1);
    freeTensor(input0);
    freeLinearLayerShellOnly(linear);

    /* ASSERT on captured. */
    const float errorPercent = 0.03f;
    for (size_t i = 0; i < 6; i++) {
        float currentThreshold = capturedActualWeights[i] * errorPercent;
        TEST_ASSERT_FLOAT_WITHIN(currentThreshold, expectedWeights[i], capturedActualWeights[i]);
    }
}

void testEvaluationBatch_ReturnsAverageLoss() {
    /* Set up a simple linear model: weights=[1,1,1,1,1,1] shape=[2,3], bias=[-1,3] */
    tensor_t *wParam = buildFloatTensor2D(2, 3, (float[]){1.f, 1.f, 1.f, 1.f, 1.f, 1.f}, 6);
    tensor_t *wGrad = gradInitFloat(wParam, NULL);
    parameter_t *w = parameterInit(wParam, wGrad);

    tensor_t *bParam = buildFloatTensor2D(1, 2, (float[]){-1.f, 3.f}, 2);
    tensor_t *bGrad = gradInitFloat(bParam, NULL);
    parameter_t *b = parameterInit(bParam, bGrad);

    quantization_t testQ;
    initFloat32Quantization(&testQ);
    layer_t *linear = buildBorrowedLinearLayer(w, b, &testQ);
    layer_t *model[] = {linear};

    /* Create 2 natural-shape samples manually */
    tensor_t *input0 = buildFloatTensor1D(3, (float[]){0.f, 1.f, 2.f});
    tensor_t *label0 = buildFloatTensor1D(2, (float[]){10.f, 20.f});
    tensor_t *input1 = buildFloatTensor1D(3, (float[]){1.f, 0.f, 0.f});
    tensor_t *label1 = buildFloatTensor1D(2, (float[]){5.f, 5.f});

    /* Compute expected losses via inferenceWithLoss directly (tensor-level:
     * takes [B, ...], so the samples are wrapped). evaluationBatch returns a
     * pure sum (no division); expected = sum of per-sample MEAN losses. */
    batchView_t in0View, lb0View, in1View, lb1View;
    inferenceStats_t *stats0 =
        inferenceWithLoss(model, 1, batchViewOf(&in0View, input0), batchViewOf(&lb0View, label0),
                          MSE, REDUCTION_MEAN, NULL);
    inferenceStats_t *stats1 =
        inferenceWithLoss(model, 1, batchViewOf(&in1View, input1), batchViewOf(&lb1View, label1),
                          MSE, REDUCTION_MEAN, NULL);
    float expectedSumLoss = stats0->loss + stats1->loss;
    freeInferenceStats(stats0);
    freeInferenceStats(stats1);

    /* Build a batch. Samples must be heap-allocated: evaluationBatch frees
     * each via freeSample (the batch-consumer convention set by #119). */
    sample_t *s0 = reserveMemory(sizeof(sample_t));
    s0->item = input0;
    s0->label = label0;
    sample_t *s1 = reserveMemory(sizeof(sample_t));
    s1->item = input1;
    s1->label = label1;
    sample_t *samples[] = {s0, s1};
    batch_t batch = {.samples = samples, .size = 2};

    float actualSumLoss = evaluationBatch(model, 1, MSE, &batch, inferenceWithLoss, REDUCTION_MEAN);

    /* CAPTURE. */
    float capturedExpected = expectedSumLoss;
    float capturedActual = actualSumLoss;

    /* FREE in reverse-init order. evaluationBatch consumed s0/s1 (freed via
     * freeSample inside the production loop). The tensors they referenced
     * (input0/label0/input1/label1) are still ours to free. */
    freeTensor(label1);
    freeTensor(input1);
    freeTensor(label0);
    freeTensor(input0);
    freeLinearLayerShellOnly(linear);
    freeParameter(b);
    freeParameter(w);

    /* ASSERT on captured. */
    TEST_ASSERT_FLOAT_WITHIN(1e-5f, capturedExpected, capturedActual);
}

/* Test dataset for epoch-level tests: 4 samples, batchSize=2 → 2 batches.
 * File-scope so the dataLoader callback can reach it. Each test that calls
 * initEpochDataset() must call freeEpochDataset() at end so the per-test
 * teardown idiom holds (no shared lazy-init/never-free state).
 *
 * Mirrors the dataset pattern in UnitTestMnistSmoke.c (#120). */
static tensor_t *epochItems[4];
static tensor_t *epochLabels[4];
static dataset_t epochDataset;
static tensorArray_t epochItemsArr;
static tensorArray_t epochLabelsArr;
static bool epochDatasetInit = false;

static const float epochItemDataLiteral[4][2] = {
    {5.f, 1.f},
    {1.f, 5.f},
    {3.f, 1.f},
    {1.f, 3.f},
};
static const float epochLabelDataLiteral[4][2] = {
    {1.f, 0.f},
    {0.f, 1.f},
    {1.f, 0.f},
    {0.f, 1.f},
};

static void initEpochDataset() {
    if (epochDatasetInit) {
        return;
    }
    for (size_t i = 0; i < 4; i++) {
        epochItems[i] = buildFloatTensor1D(2, epochItemDataLiteral[i]);
        epochLabels[i] = buildFloatTensor1D(2, epochLabelDataLiteral[i]);
    }

    epochItemsArr.array = epochItems;
    epochItemsArr.size = 4;
    epochLabelsArr.array = epochLabels;
    epochLabelsArr.size = 4;

    epochDataset.items = &epochItemsArr;
    epochDataset.labels = &epochLabelsArr;
    epochDatasetInit = true;
}

static void freeEpochDataset() {
    if (!epochDatasetInit) {
        return;
    }
    for (size_t i = 0; i < 4; i++) {
        freeTensor(epochItems[i]);
        freeTensor(epochLabels[i]);
    }
    epochDatasetInit = false;
}

static sample_t *getEpochSample(size_t id) {
    sample_t *s = reserveMemory(sizeof(sample_t));
    s->item = epochDataset.items->array[id];
    s->label = epochDataset.labels->array[id];
    return s;
}

static size_t getEpochDatasetSize() {
    return epochDataset.items->size;
}

void testEvaluationEpoch_ReturnsAverageLossAcrossBatches() {
    initEpochDataset();

    /* Identity model */
    tensor_t *wParam = buildFloatTensor2D(2, 2, (float[]){1.f, 0.f, 0.f, 1.f}, 4);
    tensor_t *wGrad = gradInitFloat(wParam, NULL);
    parameter_t *w = parameterInit(wParam, wGrad);

    tensor_t *bParam = buildFloatTensor2D(1, 2, (float[]){0.f, 0.f}, 2);
    tensor_t *bGrad = gradInitFloat(bParam, NULL);
    parameter_t *b = parameterInit(bParam, bGrad);

    quantization_t testQ;
    initFloat32Quantization(&testQ);
    layer_t *linear = buildBorrowedLinearLayer(w, b, &testQ);
    layer_t *model[] = {linear};

    dataLoader_t *dl =
        dataLoaderInit(getEpochSample, getEpochDatasetSize, 1, NULL, NULL, false, true);

    float totalAvg = evaluationEpoch(model, 1, MSE, dl, inferenceWithLoss, REDUCTION_MEAN, 0);

    /* CAPTURE. */
    float capturedTotalAvg = totalAvg;

    /* FREE. */
    freeDataLoader(dl);
    freeLinearLayerShellOnly(linear);
    freeParameter(b);
    freeParameter(w);
    freeEpochDataset();

    /* Identity model output = input, so loss = MSE(input, label)
     * s0: MSE([5,1],[1,0]) = ((5-1)^2+(1-0)^2)/2 = 8.5
     * s1: MSE([1,5],[0,1]) = ((1-0)^2+(5-1)^2)/2 = 8.5
     * s2: MSE([3,1],[1,0]) = ((3-1)^2+(1-0)^2)/2 = 2.5
     * s3: MSE([1,3],[0,1]) = ((1-0)^2+(3-1)^2)/2 = 2.5
     * flat aggregator: totalLoss=22, totalSamples=4, mean=22/4=5.5 */
    TEST_ASSERT_FLOAT_WITHIN(0.01f, 5.5f, capturedTotalAvg);
}

void testEvaluationEpoch_MinibatchMatchesMicrobatchAverage() {
    initEpochDataset();

    /* Identity model — output = input, so loss = MSE(input, label) */
    tensor_t *wParam = buildFloatTensor2D(2, 2, (float[]){1.f, 0.f, 0.f, 1.f}, 4);
    tensor_t *wGrad = gradInitFloat(wParam, NULL);
    parameter_t *w = parameterInit(wParam, wGrad);

    tensor_t *bParam = buildFloatTensor2D(1, 2, (float[]){0.f, 0.f}, 2);
    tensor_t *bGrad = gradInitFloat(bParam, NULL);
    parameter_t *b = parameterInit(bParam, bGrad);

    quantization_t testQ;
    initFloat32Quantization(&testQ);
    layer_t *linear = buildBorrowedLinearLayer(w, b, &testQ);
    layer_t *model[] = {linear};

    /* batchSize=2 → 2 minibatches of 2 samples each
     * Per-sample losses: 8.5, 8.5, 2.5, 2.5
     * flat aggregator: totalLoss=22, totalSamples=4, mean=22/4=5.5 */
    dataLoader_t *dl =
        dataLoaderInit(getEpochSample, getEpochDatasetSize, 2, NULL, NULL, false, true);

    float totalAvg = evaluationEpoch(model, 1, MSE, dl, inferenceWithLoss, REDUCTION_MEAN, 0);

    /* CAPTURE. */
    float capturedTotalAvg = totalAvg;

    /* FREE. */
    freeDataLoader(dl);
    freeLinearLayerShellOnly(linear);
    freeParameter(b);
    freeParameter(w);
    freeEpochDataset();

    /* ASSERT. */
    TEST_ASSERT_FLOAT_WITHIN(0.01f, 5.5f, capturedTotalAvg);
}

void testTrainingBatchDefault_ReturnsAverageLossAndAccumulatesGrads() {
    /* Single linear layer, 2 samples → avg loss should match manually computed value */
    tensor_t *wParam = buildFloatTensor2D(2, 3, (float[]){1.f, 1.f, 1.f, 1.f, 1.f, 1.f}, 6);
    tensor_t *wGrad = gradInitFloat(wParam, NULL);
    parameter_t *w = parameterInit(wParam, wGrad);

    tensor_t *bParam = buildFloatTensor2D(1, 2, (float[]){-1.f, 3.f}, 2);
    tensor_t *bGrad = gradInitFloat(bParam, NULL);
    parameter_t *b = parameterInit(bParam, bGrad);

    quantization_t testQ;
    initFloat32Quantization(&testQ);
    layer_t *linear = buildBorrowedLinearLayer(w, b, &testQ);
    layer_t *model[] = {linear};

    /* Compute expected: run calculateGradsSequential manually per sample
     * (natural-shape samples, wrapped for the tensor-level call). */
    tensor_t *in0 = buildFloatTensor1D(3, (float[]){-4.f, 1.f, 9.f});
    tensor_t *lb0 = buildFloatTensor1D(2, (float[]){59.f, -23.f});
    tensor_t *in1 = buildFloatTensor1D(3, (float[]){5.f, -1.f, 2.f});
    tensor_t *lb1 = buildFloatTensor1D(2, (float[]){43.f, 249.f});
    batchView_t in0View, lb0View, in1View, lb1View;

    /* Get expected losses from individual calculateGrads calls.
     * Use REDUCTION_MEAN + batchSize=2 to match what trainingBatchDefault threads
     * through when called with forwardReduction=REDUCTION_MEAN. */
    trainingStats_t *ts0 = calculateGradsSequential(
        model, 1, (lossConfig_t){.funcType = MSE, .backwardReduction = REDUCTION_SUM},
        REDUCTION_MEAN, batchViewOf(&in0View, in0), batchViewOf(&lb0View, lb0), NULL);
    trainingStats_t *ts1 = calculateGradsSequential(
        model, 1, (lossConfig_t){.funcType = MSE, .backwardReduction = REDUCTION_SUM},
        REDUCTION_MEAN, batchViewOf(&in1View, in1), batchViewOf(&lb1View, lb1), NULL);
    float expectedAvg = (ts0->loss + ts1->loss) / 2.0f;
    freeTrainingStats(ts0);
    freeTrainingStats(ts1);

    /* Reset grads to zero before testing trainingBatchDefault */
    float *gArr = (float *)wGrad->data;
    for (size_t i = 0; i < 6; i++) {
        gArr[i] = 0.f;
    }
    float *bgArr = (float *)bGrad->data;
    bgArr[0] = 0.f;
    bgArr[1] = 0.f;

    /* Build batch. Samples must be heap-allocated: trainingBatchDefault frees
     * each via freeSample (the batch-consumer convention set by #119). */
    sample_t *s0 = reserveMemory(sizeof(sample_t));
    s0->item = in0;
    s0->label = lb0;
    sample_t *s1 = reserveMemory(sizeof(sample_t));
    s1->item = in1;
    s1->label = lb1;
    sample_t *samples[] = {s0, s1};
    batch_t batch = {.samples = samples, .size = 2};

    float actualAvg = trainingBatchDefault(
        model, 1, (lossConfig_t){.funcType = MSE, .backwardReduction = REDUCTION_SUM}, &batch,
        calculateGradsSequential, REDUCTION_MEAN, 1, NULL);

    /* CAPTURE. */
    float capturedExpected = expectedAvg;
    float capturedActual = actualAvg;

    /* FREE. */
    freeTensor(lb1);
    freeTensor(in1);
    freeTensor(lb0);
    freeTensor(in0);
    freeLinearLayerShellOnly(linear);
    freeParameter(b);
    freeParameter(w);

    /* ASSERT. */
    TEST_ASSERT_FLOAT_WITHIN(1e-5f, capturedExpected, capturedActual);
}

void testTrainingBatchDefault_SumAggregatesWithoutDivision() {
    tensor_t *wParam = buildFloatTensor2D(2, 3, (float[]){1.f, 1.f, 1.f, 1.f, 1.f, 1.f}, 6);
    tensor_t *wGrad = gradInitFloat(wParam, NULL);
    parameter_t *w = parameterInit(wParam, wGrad);

    tensor_t *bParam = buildFloatTensor2D(1, 2, (float[]){-1.f, 3.f}, 2);
    tensor_t *bGrad = gradInitFloat(bParam, NULL);
    parameter_t *b = parameterInit(bParam, bGrad);

    quantization_t testQ;
    initFloat32Quantization(&testQ);
    layer_t *linear = buildBorrowedLinearLayer(w, b, &testQ);
    layer_t *model[] = {linear};

    tensor_t *in0 = buildFloatTensor1D(3, (float[]){-4.f, 1.f, 9.f});
    tensor_t *lb0 = buildFloatTensor1D(2, (float[]){59.f, -23.f});
    tensor_t *in1 = buildFloatTensor1D(3, (float[]){5.f, -1.f, 2.f});
    tensor_t *lb1 = buildFloatTensor1D(2, (float[]){43.f, 249.f});
    batchView_t in0View, lb0View, in1View, lb1View;

    trainingStats_t *ts0 = calculateGradsSequential(
        model, 1, (lossConfig_t){.funcType = MSE, .backwardReduction = REDUCTION_SUM},
        REDUCTION_SUM, batchViewOf(&in0View, in0), batchViewOf(&lb0View, lb0), NULL);
    trainingStats_t *ts1 = calculateGradsSequential(
        model, 1, (lossConfig_t){.funcType = MSE, .backwardReduction = REDUCTION_SUM},
        REDUCTION_SUM, batchViewOf(&in1View, in1), batchViewOf(&lb1View, lb1), NULL);
    float expectedSum = ts0->loss + ts1->loss;
    freeTrainingStats(ts0);
    freeTrainingStats(ts1);

    /* Reset grads. */
    float *gArr = (float *)wGrad->data;
    for (size_t i = 0; i < 6; i++) {
        gArr[i] = 0.f;
    }
    float *bgArr = (float *)bGrad->data;
    bgArr[0] = 0.f;
    bgArr[1] = 0.f;

    sample_t *s0 = reserveMemory(sizeof(sample_t));
    s0->item = in0;
    s0->label = lb0;
    sample_t *s1 = reserveMemory(sizeof(sample_t));
    s1->item = in1;
    s1->label = lb1;
    sample_t *samples[] = {s0, s1};
    batch_t batch = {.samples = samples, .size = 2};

    /* SUM forwardReduction: aggregator returns Σ stats->loss, NOT divided by batch->size. */
    float actualSum = trainingBatchDefault(
        model, 1, (lossConfig_t){.funcType = MSE, .backwardReduction = REDUCTION_SUM}, &batch,
        calculateGradsSequential, REDUCTION_SUM, 1, NULL);

    float capturedExpected = expectedSum;
    float capturedActual = actualSum;

    freeTensor(lb1);
    freeTensor(in1);
    freeTensor(lb0);
    freeTensor(in0);
    freeLinearLayerShellOnly(linear);
    freeParameter(b);
    freeParameter(w);

    TEST_ASSERT_FLOAT_WITHIN(1e-3f, capturedExpected, capturedActual);
}

void testTrainingEpochDefault_DoesOptimizerStepPerBatch() {
    /* After one epoch with known data, weights should change.
     * Epoch dataset has 2 input features, 2 output classes. */
    static const float wInitData[4] = {1.f, 0.f, 0.f, 1.f};
    tensor_t *wParam = buildFloatTensor2D(2, 2, wInitData, 4);
    tensor_t *wGrad = gradInitFloat(wParam, NULL);
    parameter_t *w = parameterInit(wParam, wGrad);

    tensor_t *bParam = buildFloatTensor2D(1, 2, (float[]){0.f, 0.f}, 2);
    tensor_t *bGrad = gradInitFloat(bParam, NULL);
    parameter_t *b = parameterInit(bParam, bGrad);

    quantization_t testQ;
    initFloat32Quantization(&testQ);
    layer_t *linear = buildBorrowedLinearLayer(w, b, &testQ);
    layer_t *model[] = {linear};
    size_t sizeModel = 1;

    /* Save initial weights */
    float initWeights[4];
    for (size_t i = 0; i < 4; i++) {
        initWeights[i] = wInitData[i];
    }

    quantization_t *momentumQ = quantizationInitFloat();
    optimizer_t *sgd =
        sgdMCreateOptim(0.01f, 0.f, 0.f, model, sizeModel, momentumQ,
                        (arithmetic_t){.type = ARITH_FLOAT32, .roundingMode = HALF_AWAY});

    /* Use epoch dataset (batchSize=1 → 4 batches) */
    initEpochDataset();
    dataLoader_t *dl =
        dataLoaderInit(getEpochSample, getEpochDatasetSize, 1, NULL, NULL, false, true);

    float epochLoss = trainingEpochDefault(
        model, sizeModel, (lossConfig_t){.funcType = MSE, .backwardReduction = REDUCTION_SUM}, dl,
        sgd, calculateGradsSequential, REDUCTION_SUM, 1, NULL);

    /* CAPTURE assertion values BEFORE any free. */
    bool capturedChanged = false;
    {
        float *curWeights = (float *)wParam->data;
        for (size_t i = 0; i < 4; i++) {
            if (curWeights[i] != initWeights[i]) {
                capturedChanged = true;
                break;
            }
        }
    }
    float capturedEpochLoss = epochLoss;

    /* FREE. freeOptim cascades to w and b parameters; do NOT also free
     * those (would double-free). */
    freeOptim(sgd);
    freeQuantization(momentumQ);
    freeDataLoader(dl);
    freeLinearLayerShellOnly(linear);
    freeEpochDataset();

    /* ASSERT. */
    TEST_ASSERT_TRUE(capturedChanged);
    TEST_ASSERT_TRUE(capturedEpochLoss > 0.0f);
}

void testTrainingEpochDefault_MinibatchStepsOncePerMinibatch() {
    /* Same model + dataset as microbatch training test, but batchSize=2
     * means the optimizer steps twice (once per minibatch) instead of
     * four times, with gradients accumulated across two samples per step.
     * Weight trajectory differs from microbatch; only property tested
     * here is that training runs end-to-end and updates the model. */
    static const float wInitData[4] = {1.f, 0.f, 0.f, 1.f};
    tensor_t *wParam = buildFloatTensor2D(2, 2, wInitData, 4);
    tensor_t *wGrad = gradInitFloat(wParam, NULL);
    parameter_t *w = parameterInit(wParam, wGrad);

    tensor_t *bParam = buildFloatTensor2D(1, 2, (float[]){0.f, 0.f}, 2);
    tensor_t *bGrad = gradInitFloat(bParam, NULL);
    parameter_t *b = parameterInit(bParam, bGrad);

    quantization_t testQ;
    initFloat32Quantization(&testQ);
    layer_t *linear = buildBorrowedLinearLayer(w, b, &testQ);
    layer_t *model[] = {linear};
    size_t sizeModel = 1;

    float initWeights[4];
    for (size_t i = 0; i < 4; i++) {
        initWeights[i] = wInitData[i];
    }

    quantization_t *momentumQ = quantizationInitFloat();
    optimizer_t *sgd =
        sgdMCreateOptim(0.01f, 0.f, 0.f, model, sizeModel, momentumQ,
                        (arithmetic_t){.type = ARITH_FLOAT32, .roundingMode = HALF_AWAY});

    initEpochDataset();
    dataLoader_t *dl =
        dataLoaderInit(getEpochSample, getEpochDatasetSize, 2, NULL, NULL, false, true);

    float epochLoss = trainingEpochDefault(
        model, sizeModel, (lossConfig_t){.funcType = MSE, .backwardReduction = REDUCTION_SUM}, dl,
        sgd, calculateGradsSequential, REDUCTION_SUM, 1, NULL);

    /* CAPTURE. */
    bool capturedChanged = false;
    {
        float *curWeights = (float *)wParam->data;
        for (size_t i = 0; i < 4; i++) {
            if (curWeights[i] != initWeights[i]) {
                capturedChanged = true;
                break;
            }
        }
    }
    float capturedEpochLoss = epochLoss;

    /* FREE. */
    freeOptim(sgd);
    freeQuantization(momentumQ);
    freeDataLoader(dl);
    freeLinearLayerShellOnly(linear);
    freeEpochDataset();

    /* ASSERT. */
    TEST_ASSERT_TRUE(capturedChanged);
    TEST_ASSERT_TRUE(capturedEpochLoss > 0.0f);
}

void testTrainingRun_ReturnsResult() {
    tensor_t *wParam = buildFloatTensor2D(2, 2, (float[]){1.f, 0.f, 0.f, 1.f}, 4);
    tensor_t *wGrad = gradInitFloat(wParam, NULL);
    parameter_t *w = parameterInit(wParam, wGrad);

    tensor_t *bParam = buildFloatTensor2D(1, 2, (float[]){0.f, 0.f}, 2);
    tensor_t *bGrad = gradInitFloat(bParam, NULL);
    parameter_t *b = parameterInit(bParam, bGrad);

    quantization_t testQ;
    initFloat32Quantization(&testQ);
    layer_t *linear = buildBorrowedLinearLayer(w, b, &testQ);
    layer_t *model[] = {linear};

    quantization_t *momentumQ = quantizationInitFloat();
    optimizer_t *sgd =
        sgdMCreateOptim(0.01f, 0.f, 0.f, model, 1, momentumQ,
                        (arithmetic_t){.type = ARITH_FLOAT32, .roundingMode = HALF_AWAY});

    initEpochDataset();
    dataLoader_t *trainDl =
        dataLoaderInit(getEpochSample, getEpochDatasetSize, 1, NULL, NULL, false, true);
    dataLoader_t *evalDl =
        dataLoaderInit(getEpochSample, getEpochDatasetSize, 1, NULL, NULL, false, true);

    trainingRunResult_t result =
        trainingRun(model, 1, (lossConfig_t){.funcType = MSE, .backwardReduction = REDUCTION_SUM},
                    trainDl, evalDl, sgd, 2, calculateGradsSequential, inferenceWithLoss, NULL);

    /* CAPTURE. */
    float capturedFinalTrainLoss = result.finalTrainLoss;
    float capturedEvalLoss = result.finalEvalStats.loss;
    float capturedAccuracy = result.finalEvalStats.accuracy;

    /* FREE. */
    freeOptim(sgd);
    freeQuantization(momentumQ);
    freeDataLoader(evalDl);
    freeDataLoader(trainDl);
    freeLinearLayerShellOnly(linear);
    freeEpochDataset();

    /* ASSERT. */
    TEST_ASSERT_TRUE(capturedFinalTrainLoss > 0.0f);
    TEST_ASSERT_TRUE(capturedEvalLoss > 0.0f);
    TEST_ASSERT_TRUE(capturedAccuracy >= 0.0f);
    TEST_ASSERT_TRUE(capturedAccuracy <= 1.0f);
}

static size_t cbCallCount;
static size_t cbLastEpoch;
static float cbLastTrainLoss;
static epochStats_t cbLastStats;

static void captureCallback(epochInfo_t info, epochStats_t evalStats) {
    cbCallCount++;
    cbLastEpoch = info.epoch;
    cbLastTrainLoss = info.trainLoss;
    cbLastStats = evalStats;
}

void testTrainingRun_CallsCallbackEachEpochWithStats() {
    cbCallCount = 0;
    cbLastEpoch = 0;
    cbLastTrainLoss = 0.0f;
    cbLastStats = (epochStats_t){0};

    tensor_t *wParam = buildFloatTensor2D(2, 2, (float[]){1.f, 0.f, 0.f, 1.f}, 4);
    tensor_t *wGrad = gradInitFloat(wParam, NULL);
    parameter_t *w = parameterInit(wParam, wGrad);

    tensor_t *bParam = buildFloatTensor2D(1, 2, (float[]){0.f, 0.f}, 2);
    tensor_t *bGrad = gradInitFloat(bParam, NULL);
    parameter_t *b = parameterInit(bParam, bGrad);

    quantization_t testQ;
    initFloat32Quantization(&testQ);
    layer_t *linear = buildBorrowedLinearLayer(w, b, &testQ);
    layer_t *model[] = {linear};

    quantization_t *momentumQ = quantizationInitFloat();
    optimizer_t *sgd =
        sgdMCreateOptim(0.01f, 0.f, 0.f, model, 1, momentumQ,
                        (arithmetic_t){.type = ARITH_FLOAT32, .roundingMode = HALF_AWAY});

    initEpochDataset();
    dataLoader_t *trainDl =
        dataLoaderInit(getEpochSample, getEpochDatasetSize, 1, NULL, NULL, false, true);
    dataLoader_t *evalDl =
        dataLoaderInit(getEpochSample, getEpochDatasetSize, 1, NULL, NULL, false, true);

    size_t numberOfEpochs = 3;
    trainingRunResult_t result =
        trainingRun(model, 1, (lossConfig_t){.funcType = MSE, .backwardReduction = REDUCTION_SUM},
                    trainDl, evalDl, sgd, numberOfEpochs, calculateGradsSequential,
                    inferenceWithLoss, &(trainingRunOptions_t){.callback = captureCallback});

    /* CAPTURE. */
    size_t capturedCbCallCount = cbCallCount;
    size_t capturedCbLastEpoch = cbLastEpoch;
    float capturedCbLastTrainLoss = cbLastTrainLoss;
    float capturedCbLastStatsLoss = cbLastStats.loss;
    float capturedCbLastStatsAccuracy = cbLastStats.accuracy;
    float capturedFinalTrainLoss = result.finalTrainLoss;
    float capturedFinalEvalLoss = result.finalEvalStats.loss;
    float capturedFinalEvalAccuracy = result.finalEvalStats.accuracy;

    /* FREE. */
    freeOptim(sgd);
    freeQuantization(momentumQ);
    freeDataLoader(evalDl);
    freeDataLoader(trainDl);
    freeLinearLayerShellOnly(linear);
    freeEpochDataset();

    /* ASSERT. Callback was invoked once per epoch, in order. */
    TEST_ASSERT_EQUAL_UINT(numberOfEpochs, capturedCbCallCount);
    TEST_ASSERT_EQUAL_UINT(numberOfEpochs - 1, capturedCbLastEpoch); /* 0-indexed */

    /* Stats passed to callback are real (not zero-initialized) and match the final result. */
    TEST_ASSERT_TRUE(capturedCbLastTrainLoss > 0.0f);
    TEST_ASSERT_TRUE(capturedCbLastStatsLoss > 0.0f);
    TEST_ASSERT_EQUAL_FLOAT(capturedFinalTrainLoss, capturedCbLastTrainLoss);
    TEST_ASSERT_EQUAL_FLOAT(capturedFinalEvalLoss, capturedCbLastStatsLoss);
    TEST_ASSERT_EQUAL_FLOAT(capturedFinalEvalAccuracy, capturedCbLastStatsAccuracy);
}

/* Build a fresh 2-D SYM_INT32 (int12, HALF_AWAY) tensor quantized from a
 * float buffer — for hand-wired SYM layers (#206 acceptance path). */
static tensor_t *buildSymTensor2D(size_t d0, size_t d1, const float *src, size_t count) {
    size_t *dims = reserveMemory(2 * sizeof(size_t));
    dims[0] = d0;
    dims[1] = d1;
    size_t *order = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, order);
    shape_t *shape = reserveMemory(sizeof(shape_t));
    setShape(shape, dims, 2, order);
    tensor_t *t = initTensor(shape, quantizationInitSymInt32(HALF_AWAY), NULL);
    tensorFillFromFloatBuffer(t, src, count);
    return t;
}

/* Single-sample dataset with NEGATIVE features: on a SYM output wire the
 * mantissas are negative, and reading them as float* reinterprets the int32
 * codes as NaN bit patterns (sign+all-ones-exponent), freezing a naive float
 * argmax at index 0. True argmax is index 1 (-0.5 > -1.0). */
static tensor_t *symMetricsItem;
static tensor_t *symMetricsLabel;
static bool symMetricsDatasetInit = false;

static void initSymMetricsDataset() {
    if (symMetricsDatasetInit) {
        return;
    }
    symMetricsItem = buildFloatTensor1D(2, (float[]){-1.f, -0.5f});
    symMetricsLabel = buildFloatTensor1D(2, (float[]){0.f, 1.f});
    symMetricsDatasetInit = true;
}

static void freeSymMetricsDataset() {
    if (!symMetricsDatasetInit) {
        return;
    }
    freeTensor(symMetricsItem);
    freeTensor(symMetricsLabel);
    symMetricsDatasetInit = false;
}

static sample_t *getSymMetricsSample(size_t id) {
    (void)id;
    sample_t *s = reserveMemory(sizeof(sample_t));
    s->item = symMetricsItem;
    s->label = symMetricsLabel;
    return s;
}

static size_t getSymMetricsDatasetSize() {
    return 1;
}

/* #206 acceptance prerequisite: with a SYM_INT32 output wire (full-SYM
 * classifier head), the metrics argmax must read int32 mantissas — mantissa
 * order IS value order (scale > 0), while a float* cast garbles it. Identity
 * SYM linear reproduces the (negative) input on the wire; the label points at
 * class 1, so a correct argmax yields accuracy 1.0. */
void testEvaluationEpochWithMetrics_SymOutputWire() {
    initSymMetricsDataset();

    tensor_t *wParam = buildSymTensor2D(2, 2, (float[]){1.f, 0.f, 0.f, 1.f}, 4);
    tensor_t *wGrad = gradInitFloat(wParam, NULL);
    parameter_t *w = parameterInit(wParam, wGrad);

    tensor_t *bParam = buildSymTensor2D(1, 2, (float[]){0.f, 0.f}, 2);
    tensor_t *bGrad = gradInitFloat(bParam, NULL);
    parameter_t *b = parameterInit(bParam, bGrad);

    quantization_t *symQ = quantizationInitSymInt32(HALF_AWAY);
    layer_t *linear = buildBorrowedLinearLayer(w, b, symQ);
    layer_t *model[] = {linear};

    dataLoader_t *dl =
        dataLoaderInit(getSymMetricsSample, getSymMetricsDatasetSize, 1, NULL, NULL, false, true);

    epochStats_t stats =
        evaluationEpochWithMetrics(model, 1, MSE, dl, inferenceWithLoss, REDUCTION_MEAN, 0);

    float capturedAccuracy = stats.accuracy;

    freeDataLoader(dl);
    freeLinearLayerShellOnly(linear);
    freeParameter(b);
    freeParameter(w);
    freeQuantization(symQ);
    freeSymMetricsDataset();

    TEST_ASSERT_FLOAT_WITHIN(0.001f, 1.0f, capturedAccuracy);
}

void testEvaluationEpochWithMetrics_AllCorrect() {
    initEpochDataset();

    tensor_t *wParam = buildFloatTensor2D(2, 2, (float[]){1.f, 0.f, 0.f, 1.f}, 4);
    tensor_t *wGrad = gradInitFloat(wParam, NULL);
    parameter_t *w = parameterInit(wParam, wGrad);

    tensor_t *bParam = buildFloatTensor2D(1, 2, (float[]){0.f, 0.f}, 2);
    tensor_t *bGrad = gradInitFloat(bParam, NULL);
    parameter_t *b = parameterInit(bParam, bGrad);

    quantization_t testQ;
    initFloat32Quantization(&testQ);
    layer_t *linear = buildBorrowedLinearLayer(w, b, &testQ);
    layer_t *model[] = {linear};

    dataLoader_t *dl =
        dataLoaderInit(getEpochSample, getEpochDatasetSize, 1, NULL, NULL, false, true);

    /* Identity model: all 4 samples predict correctly (same as testEvaluationEpochAccuracy) */
    epochStats_t stats =
        evaluationEpochWithMetrics(model, 1, MSE, dl, inferenceWithLoss, REDUCTION_MEAN, 0);

    /* CAPTURE. */
    float capturedLoss = stats.loss;
    float capturedAccuracy = stats.accuracy;
    float capturedPrecision = stats.precision;
    float capturedRecall = stats.recall;
    float capturedF1 = stats.f1;

    /* FREE. */
    freeDataLoader(dl);
    freeLinearLayerShellOnly(linear);
    freeParameter(b);
    freeParameter(w);
    freeEpochDataset();

    /* ASSERT. */
    TEST_ASSERT_FLOAT_WITHIN(0.01f, 5.5f, capturedLoss); /* same as testEvaluationEpoch */
    TEST_ASSERT_FLOAT_WITHIN(0.001f, 1.0f, capturedAccuracy);
    TEST_ASSERT_FLOAT_WITHIN(0.001f, 1.0f, capturedPrecision);
    TEST_ASSERT_FLOAT_WITHIN(0.001f, 1.0f, capturedRecall);
    TEST_ASSERT_FLOAT_WITHIN(0.001f, 1.0f, capturedF1);
}

/* Single-sample dataset for fine-grained gradient-magnitude tests. The
 * mirror-image of epochDataset but with exactly one sample, F=4, so
 * computeMeanScaleMSE returns 1/(N*F) = 1/4 — non-trivial scaling that
 * the optimizer-scaling mutation breaks measurably. */
static tensor_t *singleSampleItem;
static tensor_t *singleSampleLabel;
static dataset_t singleSampleDataset;
static tensorArray_t singleSampleItemsArr;
static tensorArray_t singleSampleLabelsArr;
static bool singleSampleDatasetInit = false;

static void initSingleSampleDataset() {
    if (singleSampleDatasetInit) {
        return;
    }
    singleSampleItem = buildFloatTensor1D(4, (float[]){1.f, 1.f, 1.f, 1.f});
    singleSampleLabel = buildFloatTensor1D(4, (float[]){0.f, 0.f, 0.f, 0.f});
    singleSampleItemsArr.array = &singleSampleItem;
    singleSampleItemsArr.size = 1;
    singleSampleLabelsArr.array = &singleSampleLabel;
    singleSampleLabelsArr.size = 1;
    singleSampleDataset.items = &singleSampleItemsArr;
    singleSampleDataset.labels = &singleSampleLabelsArr;
    singleSampleDatasetInit = true;
}

static void freeSingleSampleDataset() {
    if (!singleSampleDatasetInit) {
        return;
    }
    freeTensor(singleSampleItem);
    freeTensor(singleSampleLabel);
    singleSampleDatasetInit = false;
}

static sample_t *getSingleSample(size_t id) {
    sample_t *s = reserveMemory(sizeof(sample_t));
    s->item = singleSampleDataset.items->array[id];
    s->label = singleSampleDataset.labels->array[id];
    return s;
}

static size_t getSingleSampleDatasetSize() {
    return 1;
}

void testTrainingEpochDefault_MeanScalesGradByOneOverNF() {
    /* 4×4 identity-weight linear layer, single sample with input=[1,1,1,1]
     * and label=[0,0,0,0], lr=0.01, momentum=0. F=4 (4 output features),
     * N=1, B=1.
     *
     * Forward: output = W * input + b = [1, 1, 1, 1] (identity passes input).
     * MSE backward raw: 2*(o - l) = [2, 2, 2, 2].
     * Linear backward dL/dW[i,j] = dL/dout[i] * input[j] = 2 * 1 = 2 in
     * every position.
     * scaleOptimizerGradients factor = 1/(N*F) = 1/4.
     * Scaled dL/dW = 0.5 in every position.
     * Step with lr=0.01: each weight changes by 0.01 * 0.5 = 0.005.
     *   Diagonal:  W[i,i] = 1.0 - 0.005 = 0.995.
     *   Off-diag:  W[i,j] = 0.0 - 0.005 = -0.005.
     *
     * Mutation (no scaleOptimizerGradients call): each weight changes by
     * 0.01 * 2.0 = 0.02 instead. W[0,0] = 0.98, W[0,1] = -0.02.
     * Tolerance 1e-4 distinguishes 0.995 from 0.98 cleanly. */

    static const float wInitData[16] = {
        1.f, 0.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 0.f, 1.f,
    };
    tensor_t *wParam = buildFloatTensor2D(4, 4, wInitData, 16);
    tensor_t *wGrad = gradInitFloat(wParam, NULL);
    parameter_t *w = parameterInit(wParam, wGrad);

    tensor_t *bParam = buildFloatTensor2D(1, 4, (float[]){0.f, 0.f, 0.f, 0.f}, 4);
    tensor_t *bGrad = gradInitFloat(bParam, NULL);
    parameter_t *b = parameterInit(bParam, bGrad);

    quantization_t testQ;
    initFloat32Quantization(&testQ);
    layer_t *linear = buildBorrowedLinearLayer(w, b, &testQ);
    layer_t *model[] = {linear};

    /* momentumFactor=0 makes SGD_M behave like plain SGD. */
    quantization_t *momentumQ = quantizationInitFloat();
    optimizer_t *sgd =
        sgdMCreateOptim(0.01f, 0.f, 0.f, model, 1, momentumQ,
                        (arithmetic_t){.type = ARITH_FLOAT32, .roundingMode = HALF_AWAY});

    initSingleSampleDataset();
    dataLoader_t *dl =
        dataLoaderInit(getSingleSample, getSingleSampleDatasetSize, 1, NULL, NULL, false, true);

    /* defaultLossConfig: backwardReduction = REDUCTION_MEAN. */
    lossConfig_t cfg = defaultLossConfig(MSE);

    trainingEpochDefault(model, 1, cfg, dl, sgd, calculateGradsSequential, REDUCTION_MEAN, 1, NULL);

    /* CAPTURE before any free. */
    float capturedW00 = ((float *)wParam->data)[0];
    float capturedW01 = ((float *)wParam->data)[1];

    freeOptim(sgd);
    freeQuantization(momentumQ);
    freeDataLoader(dl);
    freeLinearLayerShellOnly(linear);
    freeSingleSampleDataset();

    TEST_ASSERT_FLOAT_WITHIN(1e-4f, 0.995f, capturedW00);
    TEST_ASSERT_FLOAT_WITHIN(1e-4f, -0.005f, capturedW01);
}

/* Partially-correct dataset: 4 samples, 3 correct, 1 wrong */
static tensor_t *partialItems[4];
static tensor_t *partialLabels[4];
static dataset_t partialDataset;
static tensorArray_t partialItemsArr;
static tensorArray_t partialLabelsArr;
static bool partialDatasetInit = false;

static const float partialItemDataLiteral[4][2] = {
    {5.f, 1.f}, /* pred=0 */
    {3.f, 1.f}, /* pred=0 */
    {4.f, 1.f}, /* pred=0 (wrong!) */
    {1.f, 3.f}, /* pred=1 */
};
static const float partialLabelDataLiteral[4][2] = {
    {1.f, 0.f}, /* true=0 */
    {1.f, 0.f}, /* true=0 */
    {0.f, 1.f}, /* true=1 */
    {0.f, 1.f}, /* true=1 */
};

static void initPartialDataset() {
    if (partialDatasetInit) {
        return;
    }
    for (size_t i = 0; i < 4; i++) {
        partialItems[i] = buildFloatTensor1D(2, partialItemDataLiteral[i]);
        partialLabels[i] = buildFloatTensor1D(2, partialLabelDataLiteral[i]);
    }

    partialItemsArr.array = partialItems;
    partialItemsArr.size = 4;
    partialLabelsArr.array = partialLabels;
    partialLabelsArr.size = 4;

    partialDataset.items = &partialItemsArr;
    partialDataset.labels = &partialLabelsArr;
    partialDatasetInit = true;
}

static void freePartialDataset() {
    if (!partialDatasetInit) {
        return;
    }
    for (size_t i = 0; i < 4; i++) {
        freeTensor(partialItems[i]);
        freeTensor(partialLabels[i]);
    }
    partialDatasetInit = false;
}

static sample_t *getPartialSample(size_t id) {
    sample_t *s = reserveMemory(sizeof(sample_t));
    s->item = partialDataset.items->array[id];
    s->label = partialDataset.labels->array[id];
    return s;
}

static size_t getPartialDatasetSize() {
    return partialDataset.items->size;
}

void testEvaluationEpochWithMetrics_PartiallyCorrect() {
    initPartialDataset();

    tensor_t *wParam = buildFloatTensor2D(2, 2, (float[]){1.f, 0.f, 0.f, 1.f}, 4);
    tensor_t *wGrad = gradInitFloat(wParam, NULL);
    parameter_t *w = parameterInit(wParam, wGrad);

    tensor_t *bParam = buildFloatTensor2D(1, 2, (float[]){0.f, 0.f}, 2);
    tensor_t *bGrad = gradInitFloat(bParam, NULL);
    parameter_t *b = parameterInit(bParam, bGrad);

    quantization_t testQ;
    initFloat32Quantization(&testQ);
    layer_t *linear = buildBorrowedLinearLayer(w, b, &testQ);
    layer_t *model[] = {linear};

    dataLoader_t *dl =
        dataLoaderInit(getPartialSample, getPartialDatasetSize, 1, NULL, NULL, false, true);

    epochStats_t stats =
        evaluationEpochWithMetrics(model, 1, MSE, dl, inferenceWithLoss, REDUCTION_MEAN, 0);

    /* CAPTURE. */
    float capturedAccuracy = stats.accuracy;
    float capturedPrecision = stats.precision;
    float capturedRecall = stats.recall;
    float capturedF1 = stats.f1;

    /* FREE. */
    freeDataLoader(dl);
    freeLinearLayerShellOnly(linear);
    freeParameter(b);
    freeParameter(w);
    freePartialDataset();

    /* ASSERT.
     * tp=[2,1], predCount=[3,1], actualCount=[2,2]
     * accuracy = 3/4 = 0.75 */
    TEST_ASSERT_FLOAT_WITHIN(0.001f, 0.75f, capturedAccuracy);
    /* precision = mean(2/3, 1/1) = 5/6 */
    TEST_ASSERT_FLOAT_WITHIN(0.001f, 5.0f / 6.0f, capturedPrecision);
    /* recall = mean(2/2, 1/2) = 0.75 */
    TEST_ASSERT_FLOAT_WITHIN(0.001f, 0.75f, capturedRecall);
    /* F1 = mean(2*(2/3)*1/(2/3+1), 2*1*0.5/(1+0.5)) = mean(4/5, 2/3) = 11/15 */
    TEST_ASSERT_FLOAT_WITHIN(0.001f, 11.0f / 15.0f, capturedF1);
}

/* Zero-prediction dataset: model never predicts class 1.
 * 3 samples: 2 correct for class 0, 1 wrong (true class 1, predicted 0). */
static tensor_t *zeroPredItems[3];
static tensor_t *zeroPredLabels[3];
static dataset_t zeroPredDataset;
static tensorArray_t zeroPredItemsArr;
static tensorArray_t zeroPredLabelsArr;
static bool zeroPredDatasetInit = false;

static const float zeroPredItemDataLiteral[3][2] = {
    {5.f, 1.f}, /* pred=0 */
    {3.f, 1.f}, /* pred=0 */
    {4.f, 1.f}, /* pred=0 (wrong, true=1) */
};
static const float zeroPredLabelDataLiteral[3][2] = {
    {1.f, 0.f}, /* true=0 */
    {1.f, 0.f}, /* true=0 */
    {0.f, 1.f}, /* true=1 */
};

static void initZeroPredDataset() {
    if (zeroPredDatasetInit) {
        return;
    }
    for (size_t i = 0; i < 3; i++) {
        zeroPredItems[i] = buildFloatTensor1D(2, zeroPredItemDataLiteral[i]);
        zeroPredLabels[i] = buildFloatTensor1D(2, zeroPredLabelDataLiteral[i]);
    }

    zeroPredItemsArr.array = zeroPredItems;
    zeroPredItemsArr.size = 3;
    zeroPredLabelsArr.array = zeroPredLabels;
    zeroPredLabelsArr.size = 3;

    zeroPredDataset.items = &zeroPredItemsArr;
    zeroPredDataset.labels = &zeroPredLabelsArr;
    zeroPredDatasetInit = true;
}

static void freeZeroPredDataset() {
    if (!zeroPredDatasetInit) {
        return;
    }
    for (size_t i = 0; i < 3; i++) {
        freeTensor(zeroPredItems[i]);
        freeTensor(zeroPredLabels[i]);
    }
    zeroPredDatasetInit = false;
}

static sample_t *getZeroPredSample(size_t id) {
    sample_t *s = reserveMemory(sizeof(sample_t));
    s->item = zeroPredDataset.items->array[id];
    s->label = zeroPredDataset.labels->array[id];
    return s;
}

static size_t getZeroPredDatasetSize() {
    return zeroPredDataset.items->size;
}

void testEvaluationEpochWithMetrics_HandlesZeroPredictionClass() {
    initZeroPredDataset();

    tensor_t *wParam = buildFloatTensor2D(2, 2, (float[]){1.f, 0.f, 0.f, 1.f}, 4);
    tensor_t *wGrad = gradInitFloat(wParam, NULL);
    parameter_t *w = parameterInit(wParam, wGrad);

    tensor_t *bParam = buildFloatTensor2D(1, 2, (float[]){0.f, 0.f}, 2);
    tensor_t *bGrad = gradInitFloat(bParam, NULL);
    parameter_t *b = parameterInit(bParam, bGrad);

    quantization_t testQ;
    initFloat32Quantization(&testQ);
    layer_t *linear = buildBorrowedLinearLayer(w, b, &testQ);
    layer_t *model[] = {linear};

    dataLoader_t *dl =
        dataLoaderInit(getZeroPredSample, getZeroPredDatasetSize, 1, NULL, NULL, false, true);

    epochStats_t stats =
        evaluationEpochWithMetrics(model, 1, MSE, dl, inferenceWithLoss, REDUCTION_MEAN, 0);

    /* CAPTURE. */
    float capturedAccuracy = stats.accuracy;
    float capturedPrecision = stats.precision;
    float capturedRecall = stats.recall;
    float capturedF1 = stats.f1;

    /* FREE. */
    freeDataLoader(dl);
    freeLinearLayerShellOnly(linear);
    freeParameter(b);
    freeParameter(w);
    freeZeroPredDataset();

    /* ASSERT.
     * tp=[2,0], predCount=[3,0], actualCount=[2,1]
     * Class 1 has no predictions -> precision guard skips it
     * Class 1 has zero tp -> recall=0 for class 1
     * Class 1 prec+rec=0 -> F1 guard skips it
     * accuracy = 2/3 */
    TEST_ASSERT_FLOAT_WITHIN(0.001f, 2.0f / 3.0f, capturedAccuracy);
    /* precision = (2/3 + 0) / 2 = 1/3 */
    TEST_ASSERT_FLOAT_WITHIN(0.001f, 1.0f / 3.0f, capturedPrecision);
    /* recall = (2/2 + 0/1) / 2 = 0.5 */
    TEST_ASSERT_FLOAT_WITHIN(0.001f, 0.5f, capturedRecall);
    /* F1 class 0: 2*(2/3)*1 / ((2/3)+1) = 4/5; class 1: 0. Mean = 2/5 */
    TEST_ASSERT_FLOAT_WITHIN(0.001f, 2.0f / 5.0f, capturedF1);

    /* No NaN or Inf from division by zero */
    TEST_ASSERT_TRUE(capturedPrecision == capturedPrecision); /* NaN check */
    TEST_ASSERT_TRUE(capturedRecall == capturedRecall);
    TEST_ASSERT_TRUE(capturedF1 == capturedF1);
}

void testEvaluationEpochWithReport_ReturnsConfusionMatrix() {
    initPartialDataset();

    tensor_t *wParam = buildFloatTensor2D(2, 2, (float[]){1.f, 0.f, 0.f, 1.f}, 4);
    tensor_t *wGrad = gradInitFloat(wParam, NULL);
    parameter_t *w = parameterInit(wParam, wGrad);

    tensor_t *bParam = buildFloatTensor2D(1, 2, (float[]){0.f, 0.f}, 2);
    tensor_t *bGrad = gradInitFloat(bParam, NULL);
    parameter_t *b = parameterInit(bParam, bGrad);

    quantization_t testQ;
    initFloat32Quantization(&testQ);
    layer_t *linear = buildBorrowedLinearLayer(w, b, &testQ);
    layer_t *model[] = {linear};

    dataLoader_t *dl =
        dataLoaderInit(getPartialSample, getPartialDatasetSize, 1, NULL, NULL, false, true);

    /* Pre-fill with non-zero to verify WithReport zeroes the caller's buffer before accumulating */
    size_t cm[2 * 2] = {99, 99, 99, 99};
    classificationReport_t report =
        evaluationEpochWithReport(model, 1, MSE, dl, inferenceWithLoss, cm, 2, REDUCTION_MEAN, 0);

    /* CAPTURE. */
    size_t capturedCM[4];
    for (size_t i = 0; i < 4; i++) {
        capturedCM[i] = report.confusionMatrix[i];
    }
    size_t capturedNumClasses = report.numClasses;
    float capturedAccuracy = report.stats.accuracy;

    /* FREE. */
    freeDataLoader(dl);
    freeLinearLayerShellOnly(linear);
    freeParameter(b);
    freeParameter(w);
    freePartialDataset();

    /* ASSERT.
     * Expected CM[predicted][actual]: [[2,1],[0,1]]
     * cm[0*2+0]=2, cm[0*2+1]=1, cm[1*2+0]=0, cm[1*2+1]=1 */
    TEST_ASSERT_EQUAL_UINT(2, capturedCM[0]); /* pred=0, actual=0 */
    TEST_ASSERT_EQUAL_UINT(1, capturedCM[1]); /* pred=0, actual=1 */
    TEST_ASSERT_EQUAL_UINT(0, capturedCM[2]); /* pred=1, actual=0 */
    TEST_ASSERT_EQUAL_UINT(1, capturedCM[3]); /* pred=1, actual=1 */

    TEST_ASSERT_EQUAL_UINT(2, capturedNumClasses);
    TEST_ASSERT_FLOAT_WITHIN(0.001f, 0.75f, capturedAccuracy);
}

void testInferenceWithLoss_PropagatesForwardReductionSum() {
    /* Identity model: output = input. Allows the loss value to be computed
     * independently of the forward chain — only the reduction parameter
     * threading is exercised here. */
    tensor_t *wParam = buildFloatTensor2D(2, 2, (float[]){1.f, 0.f, 0.f, 1.f}, 4);
    tensor_t *wGrad = gradInitFloat(wParam, NULL);
    parameter_t *w = parameterInit(wParam, wGrad);

    tensor_t *bParam = buildFloatTensor2D(1, 2, (float[]){0.f, 0.f}, 2);
    tensor_t *bGrad = gradInitFloat(bParam, NULL);
    parameter_t *b = parameterInit(bParam, bGrad);

    quantization_t testQ;
    initFloat32Quantization(&testQ);
    layer_t *linear = buildBorrowedLinearLayer(w, b, &testQ);
    layer_t *model[] = {linear};

    tensor_t *input = buildFloatTensor2D(1, 2, (float[]){5.f, 1.f}, 2);
    tensor_t *label = buildFloatTensor2D(1, 2, (float[]){1.f, 0.f}, 2);

    /* SUM: (5-1)² + (1-0)² = 17. MEAN: 17 / 2 = 8.5. Different by construction. */
    inferenceStats_t *sumStats =
        inferenceWithLoss(model, 1, input, label, MSE, REDUCTION_SUM, NULL);
    inferenceStats_t *meanStats =
        inferenceWithLoss(model, 1, input, label, MSE, REDUCTION_MEAN, NULL);

    float capturedSum = sumStats->loss;
    float capturedMean = meanStats->loss;

    freeInferenceStats(sumStats);
    freeInferenceStats(meanStats);
    freeTensor(label);
    freeTensor(input);
    freeLinearLayerShellOnly(linear);
    freeParameter(b);
    freeParameter(w);

    TEST_ASSERT_FLOAT_WITHIN(1e-4f, 17.0f, capturedSum);
    TEST_ASSERT_FLOAT_WITHIN(1e-4f, 8.5f, capturedMean);
}

void testTrainingEpochDefault_SumBackwardSkipsOptimizerScaling() {
    /* With backwardReduction = SUM, gradients should land at the optimizer
     * unscaled — multiple steps with SUM should produce a different weight
     * trajectory than MEAN over the same data. */
    static const float wInitData[4] = {1.f, 0.f, 0.f, 1.f};
    static const float bInitData[2] = {0.f, 0.f};

    tensor_t *wParam = buildFloatTensor2D(2, 2, wInitData, 4);
    tensor_t *wGrad = gradInitFloat(wParam, NULL);
    parameter_t *w = parameterInit(wParam, wGrad);
    tensor_t *bParam = buildFloatTensor2D(1, 2, bInitData, 2);
    tensor_t *bGrad = gradInitFloat(bParam, NULL);
    parameter_t *b = parameterInit(bParam, bGrad);

    quantization_t testQ;
    initFloat32Quantization(&testQ);
    layer_t *linear = buildBorrowedLinearLayer(w, b, &testQ);
    layer_t *model[] = {linear};

    /* Use a very small learning rate so SUM doesn't NaN — SUM gradients are
     * larger by N*F than MEAN gradients on the same data. */
    quantization_t *momentumQ = quantizationInitFloat();
    optimizer_t *sgd =
        sgdMCreateOptim(0.001f, 0.f, 0.f, model, 1, momentumQ,
                        (arithmetic_t){.type = ARITH_FLOAT32, .roundingMode = HALF_AWAY});

    initEpochDataset();
    dataLoader_t *dl =
        dataLoaderInit(getEpochSample, getEpochDatasetSize, 1, NULL, NULL, false, true);

    /* SUM backwardReduction skips scaleOptimizerGradients. */
    lossConfig_t cfg = defaultLossConfig(MSE);
    cfg.backwardReduction = REDUCTION_SUM;

    float epochLoss = trainingEpochDefault(model, 1, cfg, dl, sgd, calculateGradsSequential,
                                           REDUCTION_SUM, 1, NULL);

    bool capturedChanged = false;
    {
        float *curWeights = (float *)wParam->data;
        for (size_t i = 0; i < 4; i++) {
            if (curWeights[i] != wInitData[i]) {
                capturedChanged = true;
                break;
            }
        }
    }
    bool capturedFinite = (epochLoss == epochLoss) && (epochLoss != 1.0f / 0.0f);
    float capturedEpochLoss = epochLoss;

    freeOptim(sgd);
    freeQuantization(momentumQ);
    freeDataLoader(dl);
    freeLinearLayerShellOnly(linear);
    freeEpochDataset();

    TEST_ASSERT_TRUE(capturedChanged);
    TEST_ASSERT_TRUE(capturedFinite);
    TEST_ASSERT_TRUE(capturedEpochLoss > 0.0f);
}

void testTrainingEpochDefault_MeanForwardSumBackward_MixedCombination() {
    /* Leo's stated use case: report per-sample MEAN losses (comparable
     * across runs), but step the optimizer with raw SUM gradients (no
     * scaleOptimizerGradients call). */
    static const float wInitData[4] = {1.f, 0.f, 0.f, 1.f};
    static const float bInitData[2] = {0.f, 0.f};

    tensor_t *wParam = buildFloatTensor2D(2, 2, wInitData, 4);
    tensor_t *wGrad = gradInitFloat(wParam, NULL);
    parameter_t *w = parameterInit(wParam, wGrad);
    tensor_t *bParam = buildFloatTensor2D(1, 2, bInitData, 2);
    tensor_t *bGrad = gradInitFloat(bParam, NULL);
    parameter_t *b = parameterInit(bParam, bGrad);

    quantization_t testQ;
    initFloat32Quantization(&testQ);
    layer_t *linear = buildBorrowedLinearLayer(w, b, &testQ);
    layer_t *model[] = {linear};

    quantization_t *momentumQ = quantizationInitFloat();
    optimizer_t *sgd =
        sgdMCreateOptim(0.001f, 0.f, 0.f, model, 1, momentumQ,
                        (arithmetic_t){.type = ARITH_FLOAT32, .roundingMode = HALF_AWAY});

    initEpochDataset();
    dataLoader_t *dl =
        dataLoaderInit(getEpochSample, getEpochDatasetSize, 1, NULL, NULL, false, true);

    lossConfig_t cfg = defaultLossConfig(MSE);
    cfg.backwardReduction = REDUCTION_SUM; /* SUM gradients, no optimizer scaling */

    /* forwardReduction = MEAN: stats->loss is per-sample mean, comparable. */
    float epochLoss = trainingEpochDefault(model, 1, cfg, dl, sgd, calculateGradsSequential,
                                           REDUCTION_MEAN, 1, NULL);

    bool capturedChanged = false;
    {
        float *curWeights = (float *)wParam->data;
        for (size_t i = 0; i < 4; i++) {
            if (curWeights[i] != wInitData[i]) {
                capturedChanged = true;
                break;
            }
        }
    }
    bool capturedFinite = (epochLoss == epochLoss) && (epochLoss != 1.0f / 0.0f);
    float capturedEpochLoss = epochLoss;

    freeOptim(sgd);
    freeQuantization(momentumQ);
    freeDataLoader(dl);
    freeLinearLayerShellOnly(linear);
    freeEpochDataset();

    TEST_ASSERT_TRUE(capturedChanged);
    TEST_ASSERT_TRUE(capturedFinite);
    TEST_ASSERT_TRUE(capturedEpochLoss > 0.0f);
    /* MEAN-reported loss should be per-sample-scale (small), not raw-SUM-scale (large). */
    TEST_ASSERT_TRUE(capturedEpochLoss < 100.0f);
}

void testEvaluationEpoch_FlatAggregator_DivisionByTotalSamples() {
    initEpochDataset();

    tensor_t *wParam = buildFloatTensor2D(2, 2, (float[]){1.f, 0.f, 0.f, 1.f}, 4);
    tensor_t *wGrad = gradInitFloat(wParam, NULL);
    parameter_t *w = parameterInit(wParam, wGrad);
    tensor_t *bParam = buildFloatTensor2D(1, 2, (float[]){0.f, 0.f}, 2);
    tensor_t *bGrad = gradInitFloat(bParam, NULL);
    parameter_t *b = parameterInit(bParam, bGrad);

    quantization_t testQ;
    initFloat32Quantization(&testQ);
    layer_t *linear = buildBorrowedLinearLayer(w, b, &testQ);
    layer_t *model[] = {linear};

    /* batchSize=2 → 2 batches of 2 samples; per-sample MSE losses are
     * 8.5, 8.5, 2.5, 2.5 (computed by hand, as in
     * testEvaluationEpoch_MinibatchMatchesMicrobatchAverage).
     * Flat aggregator: totalLoss = 8.5+8.5+2.5+2.5 = 22; totalSamples = 4;
     * mean = 22/4 = 5.5. */
    dataLoader_t *dl =
        dataLoaderInit(getEpochSample, getEpochDatasetSize, 2, NULL, NULL, false, true);

    /* MEAN: divides by totalSamples. */
    float meanLoss = evaluationEpoch(model, 1, MSE, dl, inferenceWithLoss, REDUCTION_MEAN, 0);

    float capturedMean = meanLoss;

    freeDataLoader(dl);
    freeLinearLayerShellOnly(linear);
    freeParameter(b);
    freeParameter(w);
    freeEpochDataset();

    TEST_ASSERT_FLOAT_WITHIN(0.01f, 5.5f, capturedMean);
}

void testEvaluationEpoch_SumPath_ReturnsRawTotal() {
    initEpochDataset();

    tensor_t *wParam = buildFloatTensor2D(2, 2, (float[]){1.f, 0.f, 0.f, 1.f}, 4);
    tensor_t *wGrad = gradInitFloat(wParam, NULL);
    parameter_t *w = parameterInit(wParam, wGrad);
    tensor_t *bParam = buildFloatTensor2D(1, 2, (float[]){0.f, 0.f}, 2);
    tensor_t *bGrad = gradInitFloat(bParam, NULL);
    parameter_t *b = parameterInit(bParam, bGrad);

    quantization_t testQ;
    initFloat32Quantization(&testQ);
    layer_t *linear = buildBorrowedLinearLayer(w, b, &testQ);
    layer_t *model[] = {linear};

    dataLoader_t *dl =
        dataLoaderInit(getEpochSample, getEpochDatasetSize, 1, NULL, NULL, false, true);

    /* SUM forwardReduction: returns raw totalLoss = sum of per-sample
     * SUM losses. Per-sample SUM: 17, 17, 5, 5 (= per-sample MEAN * F=2). */
    float sumLoss = evaluationEpoch(model, 1, MSE, dl, inferenceWithLoss, REDUCTION_SUM, 0);

    float capturedSum = sumLoss;

    freeDataLoader(dl);
    freeLinearLayerShellOnly(linear);
    freeParameter(b);
    freeParameter(w);
    freeEpochDataset();

    TEST_ASSERT_FLOAT_WITHIN(0.01f, 17.f + 17.f + 5.f + 5.f, capturedSum);
}

void testTrainingRun_HardcodesForwardReductionMean() {
    /* trainingRun should produce comparable train and eval losses (both
     * per-sample MEAN) regardless of lossConfig's other fields. With
     * backwardReduction = MEAN and a near-trivial learnable dataset,
     * train and eval losses should both be small finite floats — and in
     * the same ballpark, since both are MEAN-reduced over the same data
     * shape. */
    tensor_t *wParam = buildFloatTensor2D(2, 2, (float[]){1.f, 0.f, 0.f, 1.f}, 4);
    tensor_t *wGrad = gradInitFloat(wParam, NULL);
    parameter_t *w = parameterInit(wParam, wGrad);
    tensor_t *bParam = buildFloatTensor2D(1, 2, (float[]){0.f, 0.f}, 2);
    tensor_t *bGrad = gradInitFloat(bParam, NULL);
    parameter_t *b = parameterInit(bParam, bGrad);

    quantization_t testQ;
    initFloat32Quantization(&testQ);
    layer_t *linear = buildBorrowedLinearLayer(w, b, &testQ);
    layer_t *model[] = {linear};

    quantization_t *momentumQ = quantizationInitFloat();
    optimizer_t *sgd =
        sgdMCreateOptim(0.01f, 0.f, 0.f, model, 1, momentumQ,
                        (arithmetic_t){.type = ARITH_FLOAT32, .roundingMode = HALF_AWAY});

    initEpochDataset();
    dataLoader_t *trainDl =
        dataLoaderInit(getEpochSample, getEpochDatasetSize, 1, NULL, NULL, false, true);
    dataLoader_t *evalDl =
        dataLoaderInit(getEpochSample, getEpochDatasetSize, 1, NULL, NULL, false, true);

    /* trainingRun hardcodes forwardReduction = MEAN for train/eval comparability.
     * This test uses defaultLossConfig (backwardReduction = MEAN) to act as a
     * regression barrier: if forwardReduction were inadvertently set to SUM,
     * eval loss would be F× larger than train loss, failing the assertion below. */
    lossConfig_t cfg = defaultLossConfig(MSE);

    trainingRunResult_t result = trainingRun(model, 1, cfg, trainDl, evalDl, sgd, 2,
                                             calculateGradsSequential, inferenceWithLoss, NULL);

    float capturedTrain = result.finalTrainLoss;
    float capturedEval = result.finalEvalStats.loss;

    freeOptim(sgd);
    freeQuantization(momentumQ);
    freeDataLoader(evalDl);
    freeDataLoader(trainDl);
    freeLinearLayerShellOnly(linear);
    freeEpochDataset();

    TEST_ASSERT_TRUE(capturedTrain > 0.0f && capturedTrain < 100.0f);
    TEST_ASSERT_TRUE(capturedEval > 0.0f && capturedEval < 100.0f);
    /* MEAN-vs-MEAN comparability: ratio within an order of magnitude. */
    TEST_ASSERT_TRUE(capturedEval / capturedTrain < 10.0f);
    TEST_ASSERT_TRUE(capturedTrain / capturedEval < 10.0f);
}

static float g_lrCapture[8];
static size_t g_lrCaptureCount = 0;
static optimizer_t *g_capturedOptim = NULL;

static void captureLrCallback(epochInfo_t info, epochStats_t evalStats) {
    (void)info;
    (void)evalStats;
    if (g_lrCaptureCount < 8) {
        g_lrCapture[g_lrCaptureCount++] = optimizerFunctions[SGD_M].getLr(g_capturedOptim);
    }
}

void testTrainingRunStepsSchedulerOncePerEpoch(void) {
    tensor_t *wParam = buildFloatTensor2D(2, 2, (float[]){1.f, 0.f, 0.f, 1.f}, 4);
    tensor_t *wGrad = gradInitFloat(wParam, NULL);
    parameter_t *w = parameterInit(wParam, wGrad);

    tensor_t *bParam = buildFloatTensor2D(1, 2, (float[]){0.f, 0.f}, 2);
    tensor_t *bGrad = gradInitFloat(bParam, NULL);
    parameter_t *b = parameterInit(bParam, bGrad);

    quantization_t testQ;
    initFloat32Quantization(&testQ);
    layer_t *linear = buildBorrowedLinearLayer(w, b, &testQ);
    layer_t *model[] = {linear};

    quantization_t *momentumQ = quantizationInitFloat();
    /* base 0.4f halved twice: *0.25 is a power-of-two scale -> exact float */
    optimizer_t *sgd =
        sgdMCreateOptim(0.4f, 0.f, 0.f, model, 1, momentumQ,
                        (arithmetic_t){.type = ARITH_FLOAT32, .roundingMode = HALF_AWAY});

    initEpochDataset();
    dataLoader_t *trainDl =
        dataLoaderInit(getEpochSample, getEpochDatasetSize, 1, NULL, NULL, false, true);
    dataLoader_t *evalDl =
        dataLoaderInit(getEpochSample, getEpochDatasetSize, 1, NULL, NULL, false, true);

    lrScheduler_t sched;
    stepLrInit(&sched, sgd, 1, 0.5f); /* halve every epoch */
    trainingRunResult_t result =
        trainingRun(model, 1, (lossConfig_t){.funcType = MSE, .backwardReduction = REDUCTION_SUM},
                    trainDl, evalDl, sgd, 2, calculateGradsSequential, inferenceWithLoss,
                    &(trainingRunOptions_t){.lrScheduler = &sched});
    (void)result;
    /* base 0.4f halved twice: *0.25 is a power-of-two scale -> exact float */
    TEST_ASSERT_EQUAL_FLOAT(0.4f * 0.25f, optimizerFunctions[SGD_M].getLr(sgd));

    freeOptim(sgd);
    freeQuantization(momentumQ);
    freeDataLoader(evalDl);
    freeDataLoader(trainDl);
    freeLinearLayerShellOnly(linear);
    freeEpochDataset();
}

void testTrainingRunNullSchedulerKeepsLrConstant(void) {
    tensor_t *wParam = buildFloatTensor2D(2, 2, (float[]){1.f, 0.f, 0.f, 1.f}, 4);
    tensor_t *wGrad = gradInitFloat(wParam, NULL);
    parameter_t *w = parameterInit(wParam, wGrad);

    tensor_t *bParam = buildFloatTensor2D(1, 2, (float[]){0.f, 0.f}, 2);
    tensor_t *bGrad = gradInitFloat(bParam, NULL);
    parameter_t *b = parameterInit(bParam, bGrad);

    quantization_t testQ;
    initFloat32Quantization(&testQ);
    layer_t *linear = buildBorrowedLinearLayer(w, b, &testQ);
    layer_t *model[] = {linear};

    quantization_t *momentumQ = quantizationInitFloat();
    optimizer_t *sgd =
        sgdMCreateOptim(0.01f, 0.f, 0.f, model, 1, momentumQ,
                        (arithmetic_t){.type = ARITH_FLOAT32, .roundingMode = HALF_AWAY});

    initEpochDataset();
    dataLoader_t *trainDl =
        dataLoaderInit(getEpochSample, getEpochDatasetSize, 1, NULL, NULL, false, true);
    dataLoader_t *evalDl =
        dataLoaderInit(getEpochSample, getEpochDatasetSize, 1, NULL, NULL, false, true);

    float lrBefore = optimizerFunctions[SGD_M].getLr(sgd);
    trainingRunResult_t result =
        trainingRun(model, 1, (lossConfig_t){.funcType = MSE, .backwardReduction = REDUCTION_SUM},
                    trainDl, evalDl, sgd, 2, calculateGradsSequential, inferenceWithLoss, NULL);
    (void)result;
    TEST_ASSERT_EQUAL_FLOAT(lrBefore, optimizerFunctions[SGD_M].getLr(sgd));

    freeOptim(sgd);
    freeQuantization(momentumQ);
    freeDataLoader(evalDl);
    freeDataLoader(trainDl);
    freeLinearLayerShellOnly(linear);
    freeEpochDataset();
}

void testTrainingRunRejectsSchedulerWiredToDifferentOptimizer(void) {
    tensor_t *wParam = buildFloatTensor2D(2, 2, (float[]){1.f, 0.f, 0.f, 1.f}, 4);
    tensor_t *wGrad = gradInitFloat(wParam, NULL);
    parameter_t *w = parameterInit(wParam, wGrad);

    tensor_t *bParam = buildFloatTensor2D(1, 2, (float[]){0.f, 0.f}, 2);
    tensor_t *bGrad = gradInitFloat(bParam, NULL);
    parameter_t *b = parameterInit(bParam, bGrad);

    quantization_t testQ;
    initFloat32Quantization(&testQ);
    layer_t *linear = buildBorrowedLinearLayer(w, b, &testQ);
    layer_t *model[] = {linear};

    quantization_t *momentumQ = quantizationInitFloat();
    optimizer_t *sgd =
        sgdMCreateOptim(0.01f, 0.f, 0.f, model, 1, momentumQ,
                        (arithmetic_t){.type = ARITH_FLOAT32, .roundingMode = HALF_AWAY});

    initEpochDataset();
    dataLoader_t *trainDl =
        dataLoaderInit(getEpochSample, getEpochDatasetSize, 1, NULL, NULL, false, true);
    dataLoader_t *evalDl =
        dataLoaderInit(getEpochSample, getEpochDatasetSize, 1, NULL, NULL, false, true);

    /* A second, unrelated optimizer the scheduler is (wrongly) wired to. */
    sgd_t otherSgd;
    sgdInit(&otherSgd, 0.1f, 0.0f, 0.0f,
            (arithmetic_t){.type = ARITH_FLOAT32, .roundingMode = HALF_AWAY});
    optimImpl_t otherImpl = {.sgd = &otherSgd};
    optimizer_t otherOptim = {
        .type = SGD_M, .impl = &otherImpl, .parameter = NULL, .states = NULL, .sizeStates = 0};
    lrScheduler_t sched;
    stepLrInit(&sched, &otherOptim, 1, 0.5f);
    ASSERT_EXITS_WITH_FAILURE(
        trainingRun(model, 1, (lossConfig_t){.funcType = MSE, .backwardReduction = REDUCTION_SUM},
                    trainDl, evalDl, sgd, 1, calculateGradsSequential, inferenceWithLoss,
                    &(trainingRunOptions_t){.lrScheduler = &sched}));

    freeOptim(sgd);
    freeQuantization(momentumQ);
    freeDataLoader(evalDl);
    freeDataLoader(trainDl);
    freeLinearLayerShellOnly(linear);
    freeEpochDataset();
}

void testTrainingRunCallbackObservesTheEpochsOwnLr(void) {
    /* API contract (TrainingLoopApi.h): the scheduler steps AFTER the epoch
     * callback, so a callback logging the LR reports the value the epoch
     * TRAINED with. har's per-epoch lr JSON logging is load-bearing on this
     * ordering; stepping before the callback would corrupt experiment logs
     * silently (each epoch reporting the NEXT epoch's LR). */
    tensor_t *wParam = buildFloatTensor2D(2, 2, (float[]){1.f, 0.f, 0.f, 1.f}, 4);
    tensor_t *wGrad = gradInitFloat(wParam, NULL);
    parameter_t *w = parameterInit(wParam, wGrad);

    tensor_t *bParam = buildFloatTensor2D(1, 2, (float[]){0.f, 0.f}, 2);
    tensor_t *bGrad = gradInitFloat(bParam, NULL);
    parameter_t *b = parameterInit(bParam, bGrad);

    quantization_t testQ;
    initFloat32Quantization(&testQ);
    layer_t *linear = buildBorrowedLinearLayer(w, b, &testQ);
    layer_t *model[] = {linear};

    quantization_t *momentumQ = quantizationInitFloat();
    optimizer_t *sgd =
        sgdMCreateOptim(0.4f, 0.f, 0.f, model, 1, momentumQ,
                        (arithmetic_t){.type = ARITH_FLOAT32, .roundingMode = HALF_AWAY});

    initEpochDataset();
    dataLoader_t *trainDl =
        dataLoaderInit(getEpochSample, getEpochDatasetSize, 1, NULL, NULL, false, true);
    dataLoader_t *evalDl =
        dataLoaderInit(getEpochSample, getEpochDatasetSize, 1, NULL, NULL, false, true);

    g_lrCaptureCount = 0;
    g_capturedOptim = sgd;
    lrScheduler_t sched;
    stepLrInit(&sched, sgd, 1, 0.5f);
    trainingRunResult_t result =
        trainingRun(model, 1, (lossConfig_t){.funcType = MSE, .backwardReduction = REDUCTION_SUM},
                    trainDl, evalDl, sgd, 3, calculateGradsSequential, inferenceWithLoss,
                    &(trainingRunOptions_t){.lrScheduler = &sched, .callback = captureLrCallback});
    (void)result;
    TEST_ASSERT_EQUAL_size_t(3, g_lrCaptureCount);
    /* halving by 0.5 is exponent-only: exact floats */
    TEST_ASSERT_EQUAL_FLOAT(0.4f, g_lrCapture[0]);
    TEST_ASSERT_EQUAL_FLOAT(0.4f * 0.5f, g_lrCapture[1]);
    TEST_ASSERT_EQUAL_FLOAT(0.4f * 0.25f, g_lrCapture[2]);

    freeOptim(sgd);
    freeQuantization(momentumQ);
    freeDataLoader(evalDl);
    freeDataLoader(trainDl);
    freeLinearLayerShellOnly(linear);
    freeEpochDataset();
}

/* Batch-size scheduler port: per-epoch facts the callback must see. */
static epochInfo_t g_infoCapture[8];
static size_t g_infoCaptureCount = 0;

static void captureInfoCallback(epochInfo_t info, epochStats_t evalStats) {
    (void)evalStats;
    if (g_infoCaptureCount < 8) {
        g_infoCapture[g_infoCaptureCount++] = info;
    }
}

/* 8-sample VIEW over the 4-sample epoch fixture (id % 4): b0=1, gamma=0.5,
 * max=8 walks the batch 1 -> 2 -> 4 -> 8 across 3 epochs with 8, 4, 2
 * optimizer updates. No new tensors; freeSample releases only the wrapper. */
static sample_t *getEpochSample8(size_t id) {
    return getEpochSample(id % 4);
}

static size_t getEpochDatasetSize8(void) {
    return 8;
}

void testTrainingRunZeroOptionsBehavesLikeNull(void) {
    tensor_t *wParam = buildFloatTensor2D(2, 2, (float[]){1.f, 0.f, 0.f, 1.f}, 4);
    tensor_t *wGrad = gradInitFloat(wParam, NULL);
    parameter_t *w = parameterInit(wParam, wGrad);

    tensor_t *bParam = buildFloatTensor2D(1, 2, (float[]){0.f, 0.f}, 2);
    tensor_t *bGrad = gradInitFloat(bParam, NULL);
    parameter_t *b = parameterInit(bParam, bGrad);

    quantization_t testQ;
    initFloat32Quantization(&testQ);
    layer_t *linear = buildBorrowedLinearLayer(w, b, &testQ);
    layer_t *model[] = {linear};

    quantization_t *momentumQ = quantizationInitFloat();
    optimizer_t *sgd =
        sgdMCreateOptim(0.01f, 0.f, 0.f, model, 1, momentumQ,
                        (arithmetic_t){.type = ARITH_FLOAT32, .roundingMode = HALF_AWAY});

    initEpochDataset();
    dataLoader_t *trainDl =
        dataLoaderInit(getEpochSample, getEpochDatasetSize, 1, NULL, NULL, false, true);
    dataLoader_t *evalDl =
        dataLoaderInit(getEpochSample, getEpochDatasetSize, 1, NULL, NULL, false, true);

    trainingRunOptions_t zero = {0};
    trainingRunResult_t result =
        trainingRun(model, 1, (lossConfig_t){.funcType = MSE, .backwardReduction = REDUCTION_SUM},
                    trainDl, evalDl, sgd, 2, calculateGradsSequential, inferenceWithLoss, &zero);

    /* CAPTURE. */
    float capturedLr = optimizerFunctions[SGD_M].getLr(sgd);
    size_t capturedBatch = trainDl->batchSize;
    float capturedFinalTrainLoss = result.finalTrainLoss;

    /* FREE. */
    freeOptim(sgd);
    freeQuantization(momentumQ);
    freeDataLoader(evalDl);
    freeDataLoader(trainDl);
    freeLinearLayerShellOnly(linear);
    freeEpochDataset();

    /* ASSERT. */
    TEST_ASSERT_EQUAL_FLOAT(0.01f, capturedLr);
    TEST_ASSERT_EQUAL_size_t(1, capturedBatch);
    TEST_ASSERT_TRUE(capturedFinalTrainLoss > 0.0f);
}

void testTrainingRunStepsBsSchedulerOncePerEpochAndCallbackSeesTheEpochsBatch(void) {
    tensor_t *wParam = buildFloatTensor2D(2, 2, (float[]){1.f, 0.f, 0.f, 1.f}, 4);
    tensor_t *wGrad = gradInitFloat(wParam, NULL);
    parameter_t *w = parameterInit(wParam, wGrad);

    tensor_t *bParam = buildFloatTensor2D(1, 2, (float[]){0.f, 0.f}, 2);
    tensor_t *bGrad = gradInitFloat(bParam, NULL);
    parameter_t *b = parameterInit(bParam, bGrad);

    quantization_t testQ;
    initFloat32Quantization(&testQ);
    layer_t *linear = buildBorrowedLinearLayer(w, b, &testQ);
    layer_t *model[] = {linear};

    quantization_t *momentumQ = quantizationInitFloat();
    optimizer_t *sgd =
        sgdMCreateOptim(0.01f, 0.f, 0.f, model, 1, momentumQ,
                        (arithmetic_t){.type = ARITH_FLOAT32, .roundingMode = HALF_AWAY});

    initEpochDataset();
    dataLoader_t *trainDl =
        dataLoaderInit(getEpochSample8, getEpochDatasetSize8, 1, NULL, NULL, false, true);
    dataLoader_t *evalDl =
        dataLoaderInit(getEpochSample, getEpochDatasetSize, 1, NULL, NULL, false, true);

    g_infoCaptureCount = 0;
    bsScheduler_t bsSched;
    exponentialBsInit(&bsSched, trainDl, NULL, 0.5f, 8);
    trainingRunResult_t result = trainingRun(
        model, 1, (lossConfig_t){.funcType = MSE, .backwardReduction = REDUCTION_SUM}, trainDl,
        evalDl, sgd, 3, calculateGradsSequential, inferenceWithLoss,
        &(trainingRunOptions_t){.bsScheduler = &bsSched, .callback = captureInfoCallback});
    (void)result;

    /* CAPTURE. */
    size_t capturedCount = g_infoCaptureCount;
    size_t capturedEpoch[3];
    size_t capturedBatch[3];
    size_t capturedUpdates[3];
    for (size_t i = 0; i < 3; i++) {
        capturedEpoch[i] = g_infoCapture[i].epoch;
        capturedBatch[i] = g_infoCapture[i].batchSize;
        capturedUpdates[i] = g_infoCapture[i].parameterUpdates;
    }
    size_t capturedFinalBatch = trainDl->batchSize;
    size_t capturedLastEpoch = bsSched.lastEpoch;
    float capturedLr = optimizerFunctions[SGD_M].getLr(sgd);

    /* FREE. */
    freeOptim(sgd);
    freeQuantization(momentumQ);
    freeDataLoader(evalDl);
    freeDataLoader(trainDl);
    freeLinearLayerShellOnly(linear);
    freeEpochDataset();

    /* ASSERT. */
    TEST_ASSERT_EQUAL_size_t(3, capturedCount);
    const size_t expectedEpoch[3] = {0, 1, 2};
    const size_t expectedBatch[3] = {1, 2, 4};
    const size_t expectedUpdates[3] = {8, 4, 2};
    ODT_ASSERT_EQUAL_size_t_ARRAY(expectedEpoch, capturedEpoch, 3);
    ODT_ASSERT_EQUAL_size_t_ARRAY(expectedBatch, capturedBatch, 3);
    ODT_ASSERT_EQUAL_size_t_ARRAY(expectedUpdates, capturedUpdates, 3);
    TEST_ASSERT_EQUAL_size_t(8, capturedFinalBatch); /* stepped after epoch 2: 4 -> 8 */
    TEST_ASSERT_EQUAL_size_t(3, capturedLastEpoch);  /* exactly once per epoch */
    TEST_ASSERT_EQUAL_FLOAT(0.01f, capturedLr);      /* optimizer == NULL: LR untouched */
}

void testTrainingRunCallbackObservesTheEpochsOwnLrAndBatchWithBothSchedulers(void) {
    /* BS mode (no compensation) next to an LR scheduler is allowed (guard (d)
     * only rejects a COMPENSATING batch scheduler). The callback must see the
     * LR and the batch the epoch trained with: both schedulers step after it. */
    tensor_t *wParam = buildFloatTensor2D(2, 2, (float[]){1.f, 0.f, 0.f, 1.f}, 4);
    tensor_t *wGrad = gradInitFloat(wParam, NULL);
    parameter_t *w = parameterInit(wParam, wGrad);

    tensor_t *bParam = buildFloatTensor2D(1, 2, (float[]){0.f, 0.f}, 2);
    tensor_t *bGrad = gradInitFloat(bParam, NULL);
    parameter_t *b = parameterInit(bParam, bGrad);

    quantization_t testQ;
    initFloat32Quantization(&testQ);
    layer_t *linear = buildBorrowedLinearLayer(w, b, &testQ);
    layer_t *model[] = {linear};

    quantization_t *momentumQ = quantizationInitFloat();
    optimizer_t *sgd =
        sgdMCreateOptim(0.4f, 0.f, 0.f, model, 1, momentumQ,
                        (arithmetic_t){.type = ARITH_FLOAT32, .roundingMode = HALF_AWAY});

    initEpochDataset();
    dataLoader_t *trainDl =
        dataLoaderInit(getEpochSample, getEpochDatasetSize, 1, NULL, NULL, false, true);
    dataLoader_t *evalDl =
        dataLoaderInit(getEpochSample, getEpochDatasetSize, 1, NULL, NULL, false, true);

    g_infoCaptureCount = 0;
    lrScheduler_t lrSched;
    stepLrInit(&lrSched, sgd, 1, 0.5f);
    bsScheduler_t bsSched;
    exponentialBsInit(&bsSched, trainDl, NULL, 0.5f, 4);
    trainingRunResult_t result = trainingRun(
        model, 1, (lossConfig_t){.funcType = MSE, .backwardReduction = REDUCTION_SUM}, trainDl,
        evalDl, sgd, 3, calculateGradsSequential, inferenceWithLoss,
        &(trainingRunOptions_t){
            .lrScheduler = &lrSched, .bsScheduler = &bsSched, .callback = captureInfoCallback});
    (void)result;

    /* CAPTURE. */
    size_t capturedCount = g_infoCaptureCount;
    float capturedLr[3];
    size_t capturedBatch[3];
    size_t capturedUpdates[3];
    for (size_t i = 0; i < 3; i++) {
        capturedLr[i] = g_infoCapture[i].learningRate;
        capturedBatch[i] = g_infoCapture[i].batchSize;
        capturedUpdates[i] = g_infoCapture[i].parameterUpdates;
    }

    /* FREE. */
    freeOptim(sgd);
    freeQuantization(momentumQ);
    freeDataLoader(evalDl);
    freeDataLoader(trainDl);
    freeLinearLayerShellOnly(linear);
    freeEpochDataset();

    /* ASSERT. halving is exponent-only: exact floats. */
    TEST_ASSERT_EQUAL_size_t(3, capturedCount);
    TEST_ASSERT_EQUAL_FLOAT(0.4f, capturedLr[0]);
    TEST_ASSERT_EQUAL_FLOAT(0.4f * 0.5f, capturedLr[1]);
    TEST_ASSERT_EQUAL_FLOAT(0.4f * 0.25f, capturedLr[2]);
    const size_t expectedBatch[3] = {1, 2, 4};
    const size_t expectedUpdates[3] = {4, 2, 1};
    ODT_ASSERT_EQUAL_size_t_ARRAY(expectedBatch, capturedBatch, 3);
    ODT_ASSERT_EQUAL_size_t_ARRAY(expectedUpdates, capturedUpdates, 3);
}

void testTrainingRunCompensatingBsSchedulerWritesTheLrEachEpoch(void) {
    /* The "BC" arm: a batch scheduler handed the optimizer also writes
     * lr = baseLr * applied / exact after every epoch. b0=1, gamma=0.8, base
     * LR 0.01: after epoch 0 the exact target is 1.25 -> applied 1 -> LR
     * 0.01 * 1 / 1.25 = 0.008, which epoch 1 must train with; after epoch 1
     * the target is 1.5625 -> applied 2 -> LR 0.0128, and the loader ends at 2. */
    tensor_t *wParam = buildFloatTensor2D(2, 2, (float[]){1.f, 0.f, 0.f, 1.f}, 4);
    tensor_t *wGrad = gradInitFloat(wParam, NULL);
    parameter_t *w = parameterInit(wParam, wGrad);

    tensor_t *bParam = buildFloatTensor2D(1, 2, (float[]){0.f, 0.f}, 2);
    tensor_t *bGrad = gradInitFloat(bParam, NULL);
    parameter_t *b = parameterInit(bParam, bGrad);

    quantization_t testQ;
    initFloat32Quantization(&testQ);
    layer_t *linear = buildBorrowedLinearLayer(w, b, &testQ);
    layer_t *model[] = {linear};

    quantization_t *momentumQ = quantizationInitFloat();
    optimizer_t *sgd =
        sgdMCreateOptim(0.01f, 0.f, 0.f, model, 1, momentumQ,
                        (arithmetic_t){.type = ARITH_FLOAT32, .roundingMode = HALF_AWAY});

    initEpochDataset();
    dataLoader_t *trainDl =
        dataLoaderInit(getEpochSample8, getEpochDatasetSize8, 1, NULL, NULL, false, true);
    dataLoader_t *evalDl =
        dataLoaderInit(getEpochSample, getEpochDatasetSize, 1, NULL, NULL, false, true);

    g_infoCaptureCount = 0;
    bsScheduler_t bsSched;
    exponentialBsInit(&bsSched, trainDl, sgd, 0.8f, 8); /* sgd != NULL: compensating */
    trainingRunResult_t result = trainingRun(
        model, 1, (lossConfig_t){.funcType = MSE, .backwardReduction = REDUCTION_SUM}, trainDl,
        evalDl, sgd, 2, calculateGradsSequential, inferenceWithLoss,
        &(trainingRunOptions_t){.bsScheduler = &bsSched, .callback = captureInfoCallback});
    (void)result;

    /* CAPTURE. */
    size_t capturedCount = g_infoCaptureCount;
    float capturedLr0 = g_infoCapture[0].learningRate;
    float capturedLr1 = g_infoCapture[1].learningRate;
    size_t capturedBatch1 = g_infoCapture[1].batchSize;
    float capturedFinalLr = optimizerFunctions[SGD_M].getLr(sgd);
    size_t capturedFinalBatch = trainDl->batchSize;

    /* FREE. */
    freeOptim(sgd);
    freeQuantization(momentumQ);
    freeDataLoader(evalDl);
    freeDataLoader(trainDl);
    freeLinearLayerShellOnly(linear);
    freeEpochDataset();

    /* ASSERT. */
    TEST_ASSERT_EQUAL_size_t(2, capturedCount);
    TEST_ASSERT_EQUAL_FLOAT(0.01f, capturedLr0);
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 0.008f, capturedLr1);      /* 0.01 * 1 / 1.25 */
    TEST_ASSERT_EQUAL_size_t(1, capturedBatch1);               /* round(1.25) */
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 0.0128f, capturedFinalLr); /* 0.01 * 2 / 1.5625 */
    TEST_ASSERT_EQUAL_size_t(2, capturedFinalBatch);
}

void testTrainingEpochDefaultRejectsBatchLargerThanDataset(void) {
    /* 4 samples, batch 8: datasetSize / batchSize == 0 batches. Before the
     * guard this trained nothing and returned 0/0 = NaN silently. */
    tensor_t *wParam = buildFloatTensor2D(2, 2, (float[]){1.f, 0.f, 0.f, 1.f}, 4);
    tensor_t *wGrad = gradInitFloat(wParam, NULL);
    parameter_t *w = parameterInit(wParam, wGrad);

    tensor_t *bParam = buildFloatTensor2D(1, 2, (float[]){0.f, 0.f}, 2);
    tensor_t *bGrad = gradInitFloat(bParam, NULL);
    parameter_t *b = parameterInit(bParam, bGrad);

    quantization_t testQ;
    initFloat32Quantization(&testQ);
    layer_t *linear = buildBorrowedLinearLayer(w, b, &testQ);
    layer_t *model[] = {linear};

    quantization_t *momentumQ = quantizationInitFloat();
    optimizer_t *sgd =
        sgdMCreateOptim(0.01f, 0.f, 0.f, model, 1, momentumQ,
                        (arithmetic_t){.type = ARITH_FLOAT32, .roundingMode = HALF_AWAY});

    initEpochDataset();
    dataLoader_t *trainDl =
        dataLoaderInit(getEpochSample, getEpochDatasetSize, 8, NULL, NULL, false, true);

    ASSERT_EXITS_WITH_FAILURE(trainingEpochDefault(
        model, 1, (lossConfig_t){.funcType = MSE, .backwardReduction = REDUCTION_SUM}, trainDl, sgd,
        calculateGradsSequential, REDUCTION_SUM, 1, NULL));

    freeOptim(sgd);
    freeQuantization(momentumQ);
    freeDataLoader(trainDl);
    freeLinearLayerShellOnly(linear);
    freeEpochDataset();
}

/* Guards (b), (c), (d) of the design: all fire before the first batch. */
void testTrainingRunRejectsBsSchedulerWiredToAnotherLoader(void) {
    tensor_t *wParam = buildFloatTensor2D(2, 2, (float[]){1.f, 0.f, 0.f, 1.f}, 4);
    tensor_t *wGrad = gradInitFloat(wParam, NULL);
    parameter_t *w = parameterInit(wParam, wGrad);

    tensor_t *bParam = buildFloatTensor2D(1, 2, (float[]){0.f, 0.f}, 2);
    tensor_t *bGrad = gradInitFloat(bParam, NULL);
    parameter_t *b = parameterInit(bParam, bGrad);

    quantization_t testQ;
    initFloat32Quantization(&testQ);
    layer_t *linear = buildBorrowedLinearLayer(w, b, &testQ);
    layer_t *model[] = {linear};

    quantization_t *momentumQ = quantizationInitFloat();
    optimizer_t *sgd =
        sgdMCreateOptim(0.01f, 0.f, 0.f, model, 1, momentumQ,
                        (arithmetic_t){.type = ARITH_FLOAT32, .roundingMode = HALF_AWAY});

    initEpochDataset();
    dataLoader_t *trainDl =
        dataLoaderInit(getEpochSample, getEpochDatasetSize, 1, NULL, NULL, false, true);
    dataLoader_t *evalDl =
        dataLoaderInit(getEpochSample, getEpochDatasetSize, 1, NULL, NULL, false, true);

    /* Wired to the EVAL loader — the eval loader must never be resized. */
    bsScheduler_t bsSched;
    exponentialBsInit(&bsSched, evalDl, NULL, 0.5f, 4);
    ASSERT_EXITS_WITH_FAILURE(
        trainingRun(model, 1, (lossConfig_t){.funcType = MSE, .backwardReduction = REDUCTION_SUM},
                    trainDl, evalDl, sgd, 1, calculateGradsSequential, inferenceWithLoss,
                    &(trainingRunOptions_t){.bsScheduler = &bsSched}));

    freeOptim(sgd);
    freeQuantization(momentumQ);
    freeDataLoader(evalDl);
    freeDataLoader(trainDl);
    freeLinearLayerShellOnly(linear);
    freeEpochDataset();
}

void testTrainingRunRejectsBsCompensationWiredToAnotherOptimizer(void) {
    tensor_t *wParam = buildFloatTensor2D(2, 2, (float[]){1.f, 0.f, 0.f, 1.f}, 4);
    tensor_t *wGrad = gradInitFloat(wParam, NULL);
    parameter_t *w = parameterInit(wParam, wGrad);

    tensor_t *bParam = buildFloatTensor2D(1, 2, (float[]){0.f, 0.f}, 2);
    tensor_t *bGrad = gradInitFloat(bParam, NULL);
    parameter_t *b = parameterInit(bParam, bGrad);

    quantization_t testQ;
    initFloat32Quantization(&testQ);
    layer_t *linear = buildBorrowedLinearLayer(w, b, &testQ);
    layer_t *model[] = {linear};

    quantization_t *momentumQ = quantizationInitFloat();
    optimizer_t *sgd =
        sgdMCreateOptim(0.01f, 0.f, 0.f, model, 1, momentumQ,
                        (arithmetic_t){.type = ARITH_FLOAT32, .roundingMode = HALF_AWAY});

    initEpochDataset();
    dataLoader_t *trainDl =
        dataLoaderInit(getEpochSample, getEpochDatasetSize, 1, NULL, NULL, false, true);
    dataLoader_t *evalDl =
        dataLoaderInit(getEpochSample, getEpochDatasetSize, 1, NULL, NULL, false, true);

    /* A second, unrelated optimizer the compensation is (wrongly) wired to. */
    sgd_t otherSgd;
    sgdInit(&otherSgd, 0.1f, 0.0f, 0.0f,
            (arithmetic_t){.type = ARITH_FLOAT32, .roundingMode = HALF_AWAY});
    optimImpl_t otherImpl = {.sgd = &otherSgd};
    optimizer_t otherOptim = {
        .type = SGD_M, .impl = &otherImpl, .parameter = NULL, .states = NULL, .sizeStates = 0};
    bsScheduler_t bsSched;
    exponentialBsInit(&bsSched, trainDl, &otherOptim, 0.5f, 4);
    ASSERT_EXITS_WITH_FAILURE(
        trainingRun(model, 1, (lossConfig_t){.funcType = MSE, .backwardReduction = REDUCTION_SUM},
                    trainDl, evalDl, sgd, 1, calculateGradsSequential, inferenceWithLoss,
                    &(trainingRunOptions_t){.bsScheduler = &bsSched}));

    freeOptim(sgd);
    freeQuantization(momentumQ);
    freeDataLoader(evalDl);
    freeDataLoader(trainDl);
    freeLinearLayerShellOnly(linear);
    freeEpochDataset();
}

void testTrainingRunRejectsLrSchedulerNextToCompensatingBsScheduler(void) {
    tensor_t *wParam = buildFloatTensor2D(2, 2, (float[]){1.f, 0.f, 0.f, 1.f}, 4);
    tensor_t *wGrad = gradInitFloat(wParam, NULL);
    parameter_t *w = parameterInit(wParam, wGrad);

    tensor_t *bParam = buildFloatTensor2D(1, 2, (float[]){0.f, 0.f}, 2);
    tensor_t *bGrad = gradInitFloat(bParam, NULL);
    parameter_t *b = parameterInit(bParam, bGrad);

    quantization_t testQ;
    initFloat32Quantization(&testQ);
    layer_t *linear = buildBorrowedLinearLayer(w, b, &testQ);
    layer_t *model[] = {linear};

    quantization_t *momentumQ = quantizationInitFloat();
    optimizer_t *sgd =
        sgdMCreateOptim(0.01f, 0.f, 0.f, model, 1, momentumQ,
                        (arithmetic_t){.type = ARITH_FLOAT32, .roundingMode = HALF_AWAY});

    initEpochDataset();
    dataLoader_t *trainDl =
        dataLoaderInit(getEpochSample, getEpochDatasetSize, 1, NULL, NULL, false, true);
    dataLoader_t *evalDl =
        dataLoaderInit(getEpochSample, getEpochDatasetSize, 1, NULL, NULL, false, true);

    /* Both correctly wired to `sgd` — and exactly that is the conflict: two
     * writers of the same LR every epoch, last writer wins silently. */
    lrScheduler_t lrSched;
    stepLrInit(&lrSched, sgd, 1, 0.5f);
    bsScheduler_t bsSched;
    exponentialBsInit(&bsSched, trainDl, sgd, 0.5f, 4);
    ASSERT_EXITS_WITH_FAILURE(
        trainingRun(model, 1, (lossConfig_t){.funcType = MSE, .backwardReduction = REDUCTION_SUM},
                    trainDl, evalDl, sgd, 1, calculateGradsSequential, inferenceWithLoss,
                    &(trainingRunOptions_t){.lrScheduler = &lrSched, .bsScheduler = &bsSched}));

    freeOptim(sgd);
    freeQuantization(momentumQ);
    freeDataLoader(evalDl);
    freeDataLoader(trainDl);
    freeLinearLayerShellOnly(linear);
    freeEpochDataset();
}

/* ---- trainingRun's keyed shuffle (#4) ---- */

/* Golds from the Python mirror, tag values spelled as numbers (SHUFFLE 21,
 * SHUFFLE_EVAL 26), for the 4-sample epoch dataset under the run seed 42. From
 * the repository root:
 *   uv run python -c "from examples._shared.keyed_rng import (fold,
 *   key_from_seed as kfs, keyed_shuffle as ks); r = kfs(42); print([ks(4,
 *   fold(r, t, e)) for (t, e) in ((21, 0), (21, 1), (21, 2), (26, 0))])"
 * None of the four orders is the identity and no two are equal. */
#define SHUFFLE_RUN_SEED 42u
static const size_t kTrainOrderGold[3][4] = {{2, 1, 3, 0}, {2, 0, 1, 3}, {1, 3, 0, 2}};
static const size_t kEvalOrderGold[4] = {0, 2, 3, 1};

/* An identity Linear (2 -> 2) under SGD over the 4-sample epoch dataset at
 * batch 1; the train loader shuffles, the eval loader as asked. */
typedef struct {
    quantization_t testQ;
    tensor_t *wParam;
    tensor_t *bParam;
    layer_t *linear;
    layer_t *model[1];
    quantization_t *momentumQ;
    optimizer_t *sgd;
    dataLoader_t *trainDl;
    dataLoader_t *evalDl;
} shuffleRunFixture_t;

static void buildShuffleRunFixture(shuffleRunFixture_t *f, bool reshufflePerEpoch,
                                   bool shuffleEval) {
    f->wParam = buildFloatTensor2D(2, 2, (float[]){1.f, 0.f, 0.f, 1.f}, 4);
    f->bParam = buildFloatTensor2D(1, 2, (float[]){0.f, 0.f}, 2);
    initFloat32Quantization(&f->testQ);
    f->linear = buildBorrowedLinearLayer(parameterInit(f->wParam, gradInitFloat(f->wParam, NULL)),
                                         parameterInit(f->bParam, gradInitFloat(f->bParam, NULL)),
                                         &f->testQ);
    f->model[0] = f->linear;
    f->momentumQ = quantizationInitFloat();
    f->sgd = sgdMCreateOptim(0.01f, 0.f, 0.f, f->model, 1, f->momentumQ,
                             (arithmetic_t){.type = ARITH_FLOAT32, .roundingMode = HALF_AWAY});
    initEpochDataset();
    f->trainDl = dataLoaderInit(getEpochSample, getEpochDatasetSize, 1, NULL, NULL, true, true);
    f->evalDl =
        dataLoaderInit(getEpochSample, getEpochDatasetSize, 1, NULL, NULL, shuffleEval, true);
    dataLoaderSetReshufflePerEpoch(f->trainDl, reshufflePerEpoch);
}

static void freeShuffleRunFixture(shuffleRunFixture_t *f) {
    freeOptim(f->sgd);
    freeQuantization(f->momentumQ);
    freeDataLoader(f->evalDl);
    freeDataLoader(f->trainDl);
    freeLinearLayerShellOnly(f->linear);
    freeEpochDataset();
}

static const dataLoader_t *g_shuffleTrainDl;
static const dataLoader_t *g_shuffleEvalDl;
static size_t g_trainOrders[3][4];
static size_t g_evalOrders[3][4];
static size_t g_shuffleEpochs;

/* The callback runs after an epoch's training and evaluation: both loaders
 * still hold that epoch's orders. */
static void recordShuffleOrders(epochInfo_t info, epochStats_t evalStats) {
    (void)info;
    (void)evalStats;
    if (g_shuffleEpochs < 3) {
        for (size_t i = 0; i < 4; i++) {
            g_trainOrders[g_shuffleEpochs][i] = g_shuffleTrainDl->indices[i];
            g_evalOrders[g_shuffleEpochs][i] = g_shuffleEvalDl->indices[i];
        }
    }
    g_shuffleEpochs++;
}

static void runShuffled(shuffleRunFixture_t *f, size_t epochs, trainingRng_t *rng) {
    g_shuffleTrainDl = f->trainDl;
    g_shuffleEvalDl = f->evalDl;
    (void)trainingRun(f->model, 1, defaultLossConfig(MSE), f->trainDl, f->evalDl, f->sgd, epochs,
                      calculateGradsSequential, inferenceWithLoss,
                      &(trainingRunOptions_t){.callback = recordShuffleOrders, .rng = rng});
}

/* Every epoch trains in the order of its own shuffle key, epoch 0 included,
 * and the shuffle = true eval loader keeps the one order of the run's eval
 * shuffle key in every epoch. */
void testTrainingRunShufflesEachEpochUnderItsOwnKey(void) {
    shuffleRunFixture_t f;
    buildShuffleRunFixture(&f, true, true);
    trainingRng_t rng;
    trainingRngInit(&rng, SHUFFLE_RUN_SEED);
    g_shuffleEpochs = 0;
    runShuffled(&f, 3, &rng);
    freeShuffleRunFixture(&f);

    TEST_ASSERT_EQUAL_size_t(3, g_shuffleEpochs);
    for (size_t e = 0; e < 3; e++) {
        ODT_ASSERT_EQUAL_size_t_ARRAY(kTrainOrderGold[e], g_trainOrders[e], 4);
        ODT_ASSERT_EQUAL_size_t_ARRAY(kEvalOrderGold, g_evalOrders[e], 4);
    }
}

/* A train loader that does not reshuffle per epoch keeps epoch 0's order in
 * every epoch; a shuffle = false eval loader keeps the identity order. */
void testTrainingRunKeepsEpochZerosOrderWithoutPerEpochReshuffle(void) {
    shuffleRunFixture_t f;
    buildShuffleRunFixture(&f, false, false);
    trainingRng_t rng;
    trainingRngInit(&rng, SHUFFLE_RUN_SEED);
    g_shuffleEpochs = 0;
    runShuffled(&f, 2, &rng);
    freeShuffleRunFixture(&f);

    const size_t identity[4] = {0, 1, 2, 3};
    TEST_ASSERT_EQUAL_size_t(2, g_shuffleEpochs);
    for (size_t e = 0; e < 2; e++) {
        ODT_ASSERT_EQUAL_size_t_ARRAY(kTrainOrderGold[0], g_trainOrders[e], 4);
        ODT_ASSERT_EQUAL_size_t_ARRAY(identity, g_evalOrders[e], 4);
    }
}

/* Fills params[0..5] with the fixture's weights, then its biases. */
static void captureShuffleRunParams(const shuffleRunFixture_t *f, float params[6]) {
    for (size_t i = 0; i < 4; i++) {
        params[i] = ((const float *)f->wParam->data)[i];
    }
    for (size_t i = 0; i < 2; i++) {
        params[4 + i] = ((const float *)f->bParam->data)[i];
    }
}

/* The epoch key follows the run's epoch counter, which outlives a trainingRun
 * call: two one-epoch runs train in the orders of epochs 0 and 1 (the eval
 * order staying the run's one order) and end with the parameters of one
 * two-epoch run, bit for bit. */
void testTrainingRunShuffleFollowsTheEpochCounterAcrossCalls(void) {
    shuffleRunFixture_t split;
    buildShuffleRunFixture(&split, true, true);
    trainingRng_t splitRng;
    trainingRngInit(&splitRng, SHUFFLE_RUN_SEED);
    g_shuffleEpochs = 0;
    runShuffled(&split, 1, &splitRng);
    runShuffled(&split, 1, &splitRng);
    float splitParams[6];
    captureShuffleRunParams(&split, splitParams);
    freeShuffleRunFixture(&split);
    size_t splitEpochs = g_shuffleEpochs;
    size_t splitTrain[2][4];
    size_t splitEval[2][4];
    for (size_t e = 0; e < 2; e++) {
        for (size_t i = 0; i < 4; i++) {
            splitTrain[e][i] = g_trainOrders[e][i];
            splitEval[e][i] = g_evalOrders[e][i];
        }
    }

    shuffleRunFixture_t whole;
    buildShuffleRunFixture(&whole, true, true);
    trainingRng_t wholeRng;
    trainingRngInit(&wholeRng, SHUFFLE_RUN_SEED);
    g_shuffleEpochs = 0;
    runShuffled(&whole, 2, &wholeRng);
    float wholeParams[6];
    captureShuffleRunParams(&whole, wholeParams);
    freeShuffleRunFixture(&whole);

    TEST_ASSERT_EQUAL_size_t(2, splitEpochs);
    for (size_t e = 0; e < 2; e++) {
        ODT_ASSERT_EQUAL_size_t_ARRAY(kTrainOrderGold[e], splitTrain[e], 4);
        ODT_ASSERT_EQUAL_size_t_ARRAY(kEvalOrderGold, splitEval[e], 4);
    }
    TEST_ASSERT_EQUAL_MEMORY(wholeParams, splitParams, sizeof splitParams);
}

static void runWithoutAnRngOverAShuffledEvalLoader(void) {
    shuffleRunFixture_t f;
    buildShuffleRunFixture(&f, false, true);
    runShuffled(&f, 1, NULL);
}

static void runWithoutAnRngOverAShuffledTrainLoader(void) {
    shuffleRunFixture_t f;
    buildShuffleRunFixture(&f, false, false);
    runShuffled(&f, 1, NULL);
}

static void evaluateANeverShuffledLoader(void) {
    shuffleRunFixture_t f;
    buildShuffleRunFixture(&f, false, true);
    (void)evaluationEpoch(f.model, 1, MSE, f.evalDl, inferenceWithLoss, REDUCTION_MEAN, 1);
}

/* Without options->rng a shuffle = true loader has no key: the eval loader's
 * shuffle stops before the label probe fetches from it, and the train loader's
 * stops at the start of epoch 0. */
void testTrainingRunWithoutAnRngStopsAtAShuffledEvalLoader(void) {
    ASSERT_EXITS_WITH_OUTPUT(1, "dataLoaderShuffle: shuffle without a key",
                             runWithoutAnRngOverAShuffledEvalLoader());
}

void testTrainingRunWithoutAnRngStopsAtAShuffledTrainLoader(void) {
    ASSERT_EXITS_WITH_OUTPUT(1, "dataLoaderShuffle: shuffle without a key",
                             runWithoutAnRngOverAShuffledTrainLoader());
}

/* The public evaluation entry points never shuffle: a shuffle = true loader
 * must be shuffled by its caller first. */
void testEvaluationEpochOfANeverShuffledLoaderExits(void) {
    ASSERT_EXITS_WITH_OUTPUT(
        1, "getBatch: shuffle = true loader was never shuffled (call dataLoaderShuffle)",
        evaluateANeverShuffledLoader());
}

/* A learning rate of 1e30 blows the 2x2 linear model up on the first step; the
 * next epoch's MSE is inf. Default options (no stop flag): all epochs run. */
void testTrainingRun_DefaultRunsAllEpochsThroughNonFiniteLoss(void) {
    tensor_t *wParam = buildFloatTensor2D(2, 2, (float[]){1.f, 0.f, 0.f, 1.f}, 4);
    tensor_t *wGrad = gradInitFloat(wParam, NULL);
    parameter_t *w = parameterInit(wParam, wGrad);
    tensor_t *bParam = buildFloatTensor2D(1, 2, (float[]){0.f, 0.f}, 2);
    tensor_t *bGrad = gradInitFloat(bParam, NULL);
    parameter_t *b = parameterInit(bParam, bGrad);
    quantization_t testQ;
    initFloat32Quantization(&testQ);
    layer_t *linear = buildBorrowedLinearLayer(w, b, &testQ);
    layer_t *model[] = {linear};
    quantization_t *momentumQ = quantizationInitFloat();
    optimizer_t *sgd =
        sgdMCreateOptim(1e30f, 0.f, 0.f, model, 1, momentumQ,
                        (arithmetic_t){.type = ARITH_FLOAT32, .roundingMode = HALF_AWAY});
    initEpochDataset();
    dataLoader_t *trainDl =
        dataLoaderInit(getEpochSample, getEpochDatasetSize, 1, NULL, NULL, false, true);
    dataLoader_t *evalDl =
        dataLoaderInit(getEpochSample, getEpochDatasetSize, 1, NULL, NULL, false, true);

    trainingRunResult_t result =
        trainingRun(model, 1, (lossConfig_t){.funcType = MSE, .backwardReduction = REDUCTION_SUM},
                    trainDl, evalDl, sgd, 6, calculateGradsSequential, inferenceWithLoss, NULL);

    size_t capturedEpochs = result.epochsCompleted;
    bool capturedStopped = result.stoppedOnNonFiniteLoss;
    float capturedLoss = result.finalTrainLoss;

    freeOptim(sgd);
    freeQuantization(momentumQ);
    freeDataLoader(evalDl);
    freeDataLoader(trainDl);
    freeLinearLayerShellOnly(linear);
    freeEpochDataset();

    TEST_ASSERT_EQUAL_size_t(6, capturedEpochs);
    TEST_ASSERT_FALSE(capturedStopped);
    TEST_ASSERT_FALSE(isfinite(capturedLoss));
}

void testTrainingRun_StopsOnNonFiniteLossWhenRequested(void) {
    tensor_t *wParam = buildFloatTensor2D(2, 2, (float[]){1.f, 0.f, 0.f, 1.f}, 4);
    tensor_t *wGrad = gradInitFloat(wParam, NULL);
    parameter_t *w = parameterInit(wParam, wGrad);
    tensor_t *bParam = buildFloatTensor2D(1, 2, (float[]){0.f, 0.f}, 2);
    tensor_t *bGrad = gradInitFloat(bParam, NULL);
    parameter_t *b = parameterInit(bParam, bGrad);
    quantization_t testQ;
    initFloat32Quantization(&testQ);
    layer_t *linear = buildBorrowedLinearLayer(w, b, &testQ);
    layer_t *model[] = {linear};
    quantization_t *momentumQ = quantizationInitFloat();
    optimizer_t *sgd =
        sgdMCreateOptim(1e30f, 0.f, 0.f, model, 1, momentumQ,
                        (arithmetic_t){.type = ARITH_FLOAT32, .roundingMode = HALF_AWAY});
    initEpochDataset();
    dataLoader_t *trainDl =
        dataLoaderInit(getEpochSample, getEpochDatasetSize, 1, NULL, NULL, false, true);
    dataLoader_t *evalDl =
        dataLoaderInit(getEpochSample, getEpochDatasetSize, 1, NULL, NULL, false, true);
    cbCallCount = 0;

    trainingRunResult_t result = trainingRun(
        model, 1, (lossConfig_t){.funcType = MSE, .backwardReduction = REDUCTION_SUM}, trainDl,
        evalDl, sgd, 6, calculateGradsSequential, inferenceWithLoss,
        &(trainingRunOptions_t){.callback = captureCallback, .stopOnNonFiniteLoss = true});

    size_t capturedEpochs = result.epochsCompleted;
    bool capturedStopped = result.stoppedOnNonFiniteLoss;
    size_t capturedCallbacks = cbCallCount;

    freeOptim(sgd);
    freeQuantization(momentumQ);
    freeDataLoader(evalDl);
    freeDataLoader(trainDl);
    freeLinearLayerShellOnly(linear);
    freeEpochDataset();

    TEST_ASSERT_TRUE(capturedStopped);
    TEST_ASSERT_TRUE(capturedEpochs < 6);
    TEST_ASSERT_TRUE(capturedEpochs >= 1);
    /* The diverged epoch is still reported to the callback before the run ends. */
    TEST_ASSERT_EQUAL_size_t(capturedEpochs, capturedCallbacks);
}

/* Pin: a divergence on the LAST epoch is still a "completed" epoch, not a
 * short run. With numberOfEpochs == 1, the only epoch trains, evaluates,
 * diverges, and the loop breaks -- but epochsCompleted was already set to
 * epoch + 1 == 1 before the stop check runs, so it reads == numberOfEpochs
 * exactly like a non-stopped run would. */
void testTrainingRun_StopOnNonFiniteLoss_LastEpochStillCountsAsCompleted(void) {
    tensor_t *wParam = buildFloatTensor2D(2, 2, (float[]){1.f, 0.f, 0.f, 1.f}, 4);
    tensor_t *wGrad = gradInitFloat(wParam, NULL);
    parameter_t *w = parameterInit(wParam, wGrad);
    tensor_t *bParam = buildFloatTensor2D(1, 2, (float[]){0.f, 0.f}, 2);
    tensor_t *bGrad = gradInitFloat(bParam, NULL);
    parameter_t *b = parameterInit(bParam, bGrad);
    quantization_t testQ;
    initFloat32Quantization(&testQ);
    layer_t *linear = buildBorrowedLinearLayer(w, b, &testQ);
    layer_t *model[] = {linear};
    quantization_t *momentumQ = quantizationInitFloat();
    optimizer_t *sgd =
        sgdMCreateOptim(1e30f, 0.f, 0.f, model, 1, momentumQ,
                        (arithmetic_t){.type = ARITH_FLOAT32, .roundingMode = HALF_AWAY});
    initEpochDataset();
    dataLoader_t *trainDl =
        dataLoaderInit(getEpochSample, getEpochDatasetSize, 1, NULL, NULL, false, true);
    dataLoader_t *evalDl =
        dataLoaderInit(getEpochSample, getEpochDatasetSize, 1, NULL, NULL, false, true);

    trainingRunResult_t result =
        trainingRun(model, 1, (lossConfig_t){.funcType = MSE, .backwardReduction = REDUCTION_SUM},
                    trainDl, evalDl, sgd, 1, calculateGradsSequential, inferenceWithLoss,
                    &(trainingRunOptions_t){.stopOnNonFiniteLoss = true});

    size_t capturedEpochs = result.epochsCompleted;
    bool capturedStopped = result.stoppedOnNonFiniteLoss;

    freeOptim(sgd);
    freeQuantization(momentumQ);
    freeDataLoader(evalDl);
    freeDataLoader(trainDl);
    freeLinearLayerShellOnly(linear);
    freeEpochDataset();

    TEST_ASSERT_TRUE(capturedStopped);
    TEST_ASSERT_EQUAL_size_t(1, capturedEpochs);
}

/* Pin: the loop breaks on the stopping epoch BEFORE lrSchedulerStep runs
 * (TrainingLoopApi.c's stop check precedes the scheduler-step block). An
 * attached lrScheduler must therefore have stepped exactly
 * epochsCompleted - 1 times, mirroring testTrainingRunStepsSchedulerOncePerEpoch's
 * fixture but counting via lrScheduler_t's own lastEpoch (which counts
 * lrSchedulerStep() calls) instead of deriving it from the LR value. */
void testTrainingRun_StopOnNonFiniteLoss_DoesNotStepSchedulersOnTheStoppingEpoch(void) {
    tensor_t *wParam = buildFloatTensor2D(2, 2, (float[]){1.f, 0.f, 0.f, 1.f}, 4);
    tensor_t *wGrad = gradInitFloat(wParam, NULL);
    parameter_t *w = parameterInit(wParam, wGrad);
    tensor_t *bParam = buildFloatTensor2D(1, 2, (float[]){0.f, 0.f}, 2);
    tensor_t *bGrad = gradInitFloat(bParam, NULL);
    parameter_t *b = parameterInit(bParam, bGrad);
    quantization_t testQ;
    initFloat32Quantization(&testQ);
    layer_t *linear = buildBorrowedLinearLayer(w, b, &testQ);
    layer_t *model[] = {linear};
    quantization_t *momentumQ = quantizationInitFloat();
    optimizer_t *sgd =
        sgdMCreateOptim(1e30f, 0.f, 0.f, model, 1, momentumQ,
                        (arithmetic_t){.type = ARITH_FLOAT32, .roundingMode = HALF_AWAY});
    initEpochDataset();
    dataLoader_t *trainDl =
        dataLoaderInit(getEpochSample, getEpochDatasetSize, 1, NULL, NULL, false, true);
    dataLoader_t *evalDl =
        dataLoaderInit(getEpochSample, getEpochDatasetSize, 1, NULL, NULL, false, true);

    lrScheduler_t sched;
    stepLrInit(&sched, sgd, 1, 0.5f); /* halve every epoch */
    trainingRunResult_t result =
        trainingRun(model, 1, (lossConfig_t){.funcType = MSE, .backwardReduction = REDUCTION_SUM},
                    trainDl, evalDl, sgd, 6, calculateGradsSequential, inferenceWithLoss,
                    &(trainingRunOptions_t){.lrScheduler = &sched, .stopOnNonFiniteLoss = true});

    size_t capturedEpochs = result.epochsCompleted;
    size_t capturedSchedulerSteps = sched.lastEpoch;

    freeOptim(sgd);
    freeQuantization(momentumQ);
    freeDataLoader(evalDl);
    freeDataLoader(trainDl);
    freeLinearLayerShellOnly(linear);
    freeEpochDataset();

    TEST_ASSERT_TRUE(result.stoppedOnNonFiniteLoss);
    /* If the LR 1e30 fixture diverges at epoch 0 (the expected case), this is
     * 0 -- asserted structurally against epochsCompleted, not a literal, so
     * the pin holds regardless of exactly which epoch diverges. */
    TEST_ASSERT_EQUAL_size_t(capturedEpochs - 1, capturedSchedulerSteps);
}

/* BFP epic PR2 Task 8 --------------------------------------------------------
 * argmax over a BFP output wire. Mantissa order is NOT value order for BFP:
 * every GROUP carries its own exponent, so a small value in a fine-grained group
 * can hold a LARGER mantissa than a big value in a coarse group. The metrics
 * argmax must therefore DEQUANTIZE and compare floats -- unlike SYM_INT32, where
 * one positive scale makes mantissa order value order.
 *
 * Fixture (m=8 -> qMax=127, e=8 -> bias=127, groups {2, 1}):
 *   class 0: mantissa 122, stored exponent 121 (E=-6) -> 122 * 2^-6 = 1.90625
 *   class 1: mantissa  67, stored exponent 122 (E=-5) ->  67 * 2^-5 = 2.09375
 * Value argmax = 1 (the label's class); mantissa argmax = 0. */
static tensor_t *buildBfpTwoClassOutput(void) {
    size_t *dims = reserveMemory(2 * sizeof(size_t));
    dims[0] = 1;
    dims[1] = 2;
    size_t *order = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, order);
    shape_t *shape = reserveMemory(sizeof(shape_t));
    setShape(shape, dims, 2, order);
    quantization_t *bfpQ = quantizationInitBfpGrouped(8, 8, HALF_AWAY, 2, 1);
    bfpQConfig_t *qc = bfpQ->qConfig;
    qc->exponents[0] = 121;
    qc->exponents[1] = 122;
    tensor_t *t = initTensor(shape, bfpQ, NULL);
    /* 8-bit mantissas: one signed byte per element, written directly so the
     * fixture does not depend on the exponent-derivation path. */
    ((int8_t *)t->data)[0] = 122;
    ((int8_t *)t->data)[1] = 67;
    return t;
}

/* Stand-in for inferenceWithLoss: hands the metrics path a BFP output wire
 * without running a model. The real inference path cannot produce one yet --
 * mseLossForward dispatches on the OUTPUT's dtype and has no BFP arm (loss
 * functions are outside this task's scope) -- but argmaxByTensor consumes
 * whatever wire the inference function returns, so this stub exercises exactly
 * the contract under test. */
static inferenceStats_t *bfpOutputWireInference(layer_t **model, size_t numberOfLayers,
                                                tensor_t *input, tensor_t *label,
                                                lossFuncType_t funcType,
                                                reduction_t forwardReduction,
                                                const trainingCall_t *call) {
    (void)call;
    (void)model;
    (void)numberOfLayers;
    (void)input;
    (void)label;
    (void)funcType;
    (void)forwardReduction;
    inferenceStats_t *stats = reserveMemory(sizeof(inferenceStats_t));
    stats->output = buildBfpTwoClassOutput();
    stats->loss = 0.0f;
    return stats;
}

static tensor_t *bfpMetricsItem;
static tensor_t *bfpMetricsLabel;

static sample_t *getBfpMetricsSample(size_t id) {
    (void)id;
    sample_t *s = reserveMemory(sizeof(sample_t));
    s->item = bfpMetricsItem;
    s->label = bfpMetricsLabel;
    return s;
}

static size_t getBfpMetricsDatasetSize(void) {
    return 1;
}

void testEvaluationEpochWithMetrics_BfpOutputWireDequantCompares() {
    bfpMetricsItem = buildFloatTensor1D(2, (float[]){1.f, 1.f});
    bfpMetricsLabel = buildFloatTensor1D(2, (float[]){0.f, 1.f});

    dataLoader_t *dl =
        dataLoaderInit(getBfpMetricsSample, getBfpMetricsDatasetSize, 1, NULL, NULL, false, true);

    epochStats_t stats =
        evaluationEpochWithMetrics(NULL, 0, MSE, dl, bfpOutputWireInference, REDUCTION_MEAN, 0);

    float capturedAccuracy = stats.accuracy;

    freeDataLoader(dl);
    freeTensor(bfpMetricsLabel);
    freeTensor(bfpMetricsItem);

    TEST_ASSERT_FLOAT_WITHIN_MESSAGE(0.001f, 1.0f, capturedAccuracy,
                                     "BFP argmax must compare DEQUANTIZED values: class 1 "
                                     "(2.09375) wins on value, class 0 wins on raw mantissa");
}

/* ---- #152 PR3a: batch_t consumers add the batch axis -----------------------
 * Datasets deliver samples in their natural shape; trainingBatchDefault,
 * trainingEpochDefault, evaluationBatch and evaluateBatchInternal hand the
 * model [1, ...] batchViewOf views. The probes below stand in for
 * calculateGradsFn / inferenceFn and record what they were handed -- no layer
 * runs, so a natural-shape tensor reaching the callback is a clean assertion
 * failure, whatever a real layer would do with that rank. */

typedef struct {
    size_t calls;
    size_t inputRank;
    size_t inputDims[4];
    size_t inputOrder[4];
    const uint8_t *inputData;
    size_t labelRank;
    size_t labelDims[4];
    const uint8_t *labelData;
} shapeProbe_t;

static shapeProbe_t g_probe;
static tensor_t *g_probeGrad; /* non-NULL: probeGrads adds 1.0f to every element */

static void recordProbe(const tensor_t *input, const tensor_t *label) {
    g_probe.calls++;
    g_probe.inputRank = input->shape->numberOfDimensions;
    for (size_t d = 0; d < g_probe.inputRank && d < 4; d++) {
        g_probe.inputDims[d] = input->shape->dimensions[d];
        g_probe.inputOrder[d] = input->shape->orderOfDimensions[d];
    }
    g_probe.inputData = input->data;
    g_probe.labelRank = label->shape->numberOfDimensions;
    for (size_t d = 0; d < g_probe.labelRank && d < 4; d++) {
        g_probe.labelDims[d] = label->shape->dimensions[d];
    }
    g_probe.labelData = label->data;
}

static trainingStats_t *probeGrads(layer_t **model, size_t modelSize, lossConfig_t lossConfig,
                                   reduction_t forwardReduction, tensor_t *input, tensor_t *label,
                                   const trainingCall_t *call) {
    (void)model;
    (void)modelSize;
    (void)lossConfig;
    (void)forwardReduction;
    (void)call;
    recordProbe(input, label);
    if (g_probeGrad != NULL) {
        float *g = (float *)g_probeGrad->data;
        size_t n = calcNumberOfElementsByTensor(g_probeGrad);
        for (size_t i = 0; i < n; i++) {
            g[i] += 1.0f; /* a raw per-element grad of 1, accumulated like backward */
        }
    }
    trainingStats_t *stats = reserveMemory(sizeof(trainingStats_t));
    stats->output = buildFloatTensor2D(1, 1, (float[]){0.f}, 1);
    stats->loss = 0.25f;
    return stats;
}

/* Echoes the label as a [1, C] output row, so predicted == target. */
static inferenceStats_t *probeInference(layer_t **model, size_t numberOfLayers, tensor_t *input,
                                        tensor_t *label, lossFuncType_t funcType,
                                        reduction_t forwardReduction, const trainingCall_t *call) {
    (void)call;
    (void)model;
    (void)numberOfLayers;
    (void)funcType;
    (void)forwardReduction;
    recordProbe(input, label);
    size_t c = calcNumberOfElementsByTensor(label);
    inferenceStats_t *stats = reserveMemory(sizeof(inferenceStats_t));
    stats->output = buildFloatTensor2D(1, c, (const float *)label->data, c);
    stats->loss = 0.5f;
    return stats;
}

/* A natural [C=1, L=3] item and a natural [1, 2] label (like ECG's [1, 140]
 * target). The leading 1 is deliberate: nothing auto-detects an existing
 * batch axis (data-shape.md) -- a consumer that skips the wrap when dims[0] == 1
 * hands the callback rank 2 instead of rank 3 and fails the asserts below. */
static tensor_t *buildProbeItem(void) {
    return buildFloatTensor2D(1, 3, (float[]){1.f, 2.f, 3.f}, 3);
}

static tensor_t *buildProbeLabel(void) {
    return buildFloatTensor2D(1, 2, (float[]){0.f, 1.f}, 2);
}

/* Asserts on the captured g_probe (static: survives the frees) plus the two
 * zero-copy flags the caller captured before freeing its tensors. */
static void assertProbeSawBatchAxisViews(bool itemDataShared, bool labelDataShared) {
    TEST_ASSERT_EQUAL_size_t(1, g_probe.calls);
    TEST_ASSERT_EQUAL_size_t_MESSAGE(3, g_probe.inputRank, "item must arrive as [1, C, L]");
    TEST_ASSERT_EQUAL_size_t(1, g_probe.inputDims[0]);
    TEST_ASSERT_EQUAL_size_t(1, g_probe.inputDims[1]);
    TEST_ASSERT_EQUAL_size_t(3, g_probe.inputDims[2]);
    TEST_ASSERT_EQUAL_size_t(0, g_probe.inputOrder[0]);
    TEST_ASSERT_EQUAL_size_t(1, g_probe.inputOrder[1]);
    TEST_ASSERT_EQUAL_size_t(2, g_probe.inputOrder[2]);
    TEST_ASSERT_TRUE_MESSAGE(itemDataShared, "the item view must be zero-copy");
    TEST_ASSERT_EQUAL_size_t_MESSAGE(3, g_probe.labelRank, "label must arrive as [1, 1, C]");
    TEST_ASSERT_EQUAL_size_t(1, g_probe.labelDims[0]);
    TEST_ASSERT_EQUAL_size_t(1, g_probe.labelDims[1]);
    TEST_ASSERT_EQUAL_size_t(2, g_probe.labelDims[2]);
    TEST_ASSERT_TRUE_MESSAGE(labelDataShared, "the label view must be zero-copy");
}

void testTrainingBatchDefaultHandsCalculateGradsBatchAxisViews(void) {
    tensor_t *item = buildProbeItem();
    tensor_t *label = buildProbeLabel();
    sample_t *s0 = reserveMemory(sizeof(sample_t)); /* freed by trainingBatchDefault */
    s0->item = item;
    s0->label = label;
    sample_t *samples[] = {s0};
    batch_t batch = {.samples = samples, .size = 1};
    g_probe = (shapeProbe_t){0};
    g_probeGrad = NULL;

    float loss = trainingBatchDefault(
        NULL, 0, (lossConfig_t){.funcType = MSE, .backwardReduction = REDUCTION_SUM}, &batch,
        probeGrads, REDUCTION_MEAN, 1, NULL);

    bool itemDataShared = g_probe.inputData == item->data;
    bool labelDataShared = g_probe.labelData == label->data;
    size_t itemRankAfter = item->shape->numberOfDimensions;
    size_t labelRankAfter = label->shape->numberOfDimensions;

    freeTensor(label);
    freeTensor(item);

    assertProbeSawBatchAxisViews(itemDataShared, labelDataShared);
    TEST_ASSERT_EQUAL_FLOAT(0.25f, loss);
    /* the sample's own shape is untouched: the axis lives only in the view */
    TEST_ASSERT_EQUAL_size_t(2, itemRankAfter);
    TEST_ASSERT_EQUAL_size_t(2, labelRankAfter);
}

void testEvaluationBatchHandsInferenceBatchAxisViews(void) {
    tensor_t *item = buildProbeItem();
    tensor_t *label = buildProbeLabel();
    sample_t *s0 = reserveMemory(sizeof(sample_t)); /* freed by evaluationBatch */
    s0->item = item;
    s0->label = label;
    sample_t *samples[] = {s0};
    batch_t batch = {.samples = samples, .size = 1};
    g_probe = (shapeProbe_t){0};

    float sumLoss = evaluationBatch(NULL, 0, MSE, &batch, probeInference, REDUCTION_SUM);

    bool itemDataShared = g_probe.inputData == item->data;
    bool labelDataShared = g_probe.labelData == label->data;

    freeTensor(label);
    freeTensor(item);

    assertProbeSawBatchAxisViews(itemDataShared, labelDataShared);
    TEST_ASSERT_EQUAL_FLOAT(0.5f, sumLoss);
}

/* Single-sample dataset for the evaluateBatchInternal / trainingEpochDefault
 * probes: file-scope because the dataLoader callbacks carry no context. */
static tensor_t *probeDatasetItem;
static tensor_t *probeDatasetLabel;

static sample_t *getProbeSample(size_t id) {
    (void)id;
    sample_t *s = reserveMemory(sizeof(sample_t));
    s->item = probeDatasetItem;
    s->label = probeDatasetLabel;
    return s;
}

static size_t getProbeDatasetSize(void) {
    return 1;
}

void testEvaluationEpochWithMetricsHandsInferenceBatchAxisViews(void) {
    probeDatasetItem = buildProbeItem();
    probeDatasetLabel = buildProbeLabel();
    dataLoader_t *dl =
        dataLoaderInit(getProbeSample, getProbeDatasetSize, 1, NULL, NULL, false, true);
    g_probe = (shapeProbe_t){0};

    epochStats_t stats =
        evaluationEpochWithMetrics(NULL, 0, MSE, dl, probeInference, REDUCTION_MEAN, 0);

    bool itemDataShared = g_probe.inputData == probeDatasetItem->data;
    bool labelDataShared = g_probe.labelData == probeDatasetLabel->data;
    float capturedAccuracy = stats.accuracy;

    freeDataLoader(dl);
    freeTensor(probeDatasetLabel);
    freeTensor(probeDatasetItem);

    assertProbeSawBatchAxisViews(itemDataShared, labelDataShared);
    /* The metrics path ran on the views: the echoed [1, C] output row and the
     * raw label have the same argmax. (probeInference echoes the label, so this
     * does not pin numClasses; PR3a leaves the numClasses peek untouched.) */
    TEST_ASSERT_FLOAT_WITHIN(0.001f, 1.0f, capturedAccuracy);
}

void testTrainingEpochDefaultMeanScaleSeesLabelBatchAxis(void) {
    /* A natural [C=2, L=3] regression label: computeMeanScaleMSE must see the
     * per-sample F = 6. Read raw, dims[0] = 2 would pass for the batch axis and
     * give F = 3 -- a 2x too large step. probeGrads writes a raw grad of 1 into
     * the one weight; SGD lr 1, momentum 0 => w = 0 - 1 * meanScale = -1/6. */
    probeDatasetItem = buildFloatTensor1D(3, (float[]){1.f, 2.f, 3.f});
    probeDatasetLabel = buildFloatTensor2D(2, 3, (float[]){0.f, 1.f, 2.f, 3.f, 4.f, 5.f}, 6);

    tensor_t *wParam = buildFloatTensor2D(1, 1, (float[]){0.f}, 1);
    tensor_t *wGrad = gradInitFloat(wParam, NULL);
    parameter_t *w = parameterInit(wParam, wGrad);
    tensor_t *bParam = buildFloatTensor2D(1, 1, (float[]){0.f}, 1);
    parameter_t *b = parameterInit(bParam, gradInitFloat(bParam, NULL));
    quantization_t testQ;
    initFloat32Quantization(&testQ);
    layer_t *linear = buildBorrowedLinearLayer(w, b, &testQ);
    layer_t *model[] = {linear};
    quantization_t *momentumQ = quantizationInitFloat();
    optimizer_t *sgd =
        sgdMCreateOptim(1.0f, 0.f, 0.f, model, 1, momentumQ,
                        (arithmetic_t){.type = ARITH_FLOAT32, .roundingMode = HALF_AWAY});
    dataLoader_t *dl =
        dataLoaderInit(getProbeSample, getProbeDatasetSize, 1, NULL, NULL, false, true);
    g_probe = (shapeProbe_t){0};
    g_probeGrad = wGrad;

    trainingEpochDefault(model, 1, defaultLossConfig(MSE), dl, sgd, probeGrads, REDUCTION_MEAN, 1,
                         NULL);
    g_probeGrad = NULL;

    float capturedW = ((float *)wParam->data)[0];

    freeOptim(sgd); /* frees w and b */
    freeQuantization(momentumQ);
    freeDataLoader(dl);
    freeLinearLayerShellOnly(linear);
    freeTensor(probeDatasetLabel);
    freeTensor(probeDatasetItem);

    TEST_ASSERT_FLOAT_WITHIN(1e-6f, -1.0f / 6.0f, capturedW);
}

void setUp() {}
void tearDown() {}

/* ---- trainingRunOptions_t.remat (#4): training and evaluation on it ------- */

typedef struct remRun {
    trainingRunResult_t result;
    rematReport_t report;
    size_t evalPeak; /* the EVAL program's peakLiveBytes */
} remRun_t;

/* testTrainingRun_HardcodesForwardReductionMean's run (two epochs, per
 * sample, the real driver and inferenceWithLoss), with or without a HEAP
 * scheduler in the options. The scheduler is keyed to the [1, 2] batch view
 * the per-sample path hands the driver. */
static remRun_t runTwoEpochs(bool withScheduler, bool deinitBeforeTheRun) {
    tensor_t *wParam = buildFloatTensor2D(2, 2, (float[]){1.f, 0.f, 0.f, 1.f}, 4);
    tensor_t *wGrad = gradInitFloat(wParam, NULL);
    parameter_t *w = parameterInit(wParam, wGrad);
    tensor_t *bParam = buildFloatTensor2D(1, 2, (float[]){0.f, 0.f}, 2);
    tensor_t *bGrad = gradInitFloat(bParam, NULL);
    parameter_t *b = parameterInit(bParam, bGrad);
    quantization_t testQ;
    initFloat32Quantization(&testQ);
    layer_t *linear = buildBorrowedLinearLayer(w, b, &testQ);
    layer_t *model[] = {linear};
    quantization_t *momentumQ = quantizationInitFloat();
    optimizer_t *sgd =
        sgdMCreateOptim(0.01f, 0.f, 0.f, model, 1, momentumQ,
                        (arithmetic_t){.type = ARITH_FLOAT32, .roundingMode = HALF_AWAY});
    initEpochDataset();
    dataLoader_t *trainDl =
        dataLoaderInit(getEpochSample, getEpochDatasetSize, 1, NULL, NULL, false, true);
    dataLoader_t *evalDl =
        dataLoaderInit(getEpochSample, getEpochDatasetSize, 1, NULL, NULL, false, true);
    lossConfig_t cfg = defaultLossConfig(MSE);

    remRun_t run = {0};
    rematScheduler_t s;
    if (withScheduler) {
        batchView_t view;
        TEST_ASSERT_TRUE(rematHeapInit(&s, model, 1, cfg, batchViewOf(&view, epochItems[0]), NULL));
        if (deinitBeforeTheRun) {
            rematSchedulerDeinit(&s);
        }
    }
    run.result =
        trainingRun(model, 1, cfg, trainDl, evalDl, sgd, 2, calculateGradsSequential,
                    inferenceWithLoss, &(trainingRunOptions_t){.remat = withScheduler ? &s : NULL});
    if (withScheduler) {
        rematSchedulerReport(&s, &run.report);
        run.evalPeak = rematPlanProgram(s.plan, REMAT_MODE_EVAL)->peakLiveBytes;
        rematSchedulerDeinit(&s);
    }

    freeOptim(sgd);
    freeQuantization(momentumQ);
    freeDataLoader(evalDl);
    freeDataLoader(trainDl);
    freeLinearLayerShellOnly(linear);
    freeEpochDataset();
    return run;
}

/* The run's last scheduler call was an eval call on the options' scheduler
 * (the peak observable sees only the last bind, and the EVAL program's peak,
 * ACT 1 alone, is below TRAIN's ACT 1 + seed), evaluation completes through
 * the real inferenceWithLoss, and the losses equal a run without one:
 * STORE_ALL HEAP is today's values. */
void testTrainingRunTrainsOnTheOptionsScheduler(void) {
    remRun_t plain = runTwoEpochs(false, false);
    remRun_t onScheduler = runTwoEpochs(true, false);
    TEST_ASSERT_EQUAL_size_t(2, onScheduler.result.epochsCompleted);
    TEST_ASSERT_TRUE(onScheduler.evalPeak > 0);
    TEST_ASSERT_TRUE(onScheduler.evalPeak < onScheduler.report.peakLiveBytes);
    TEST_ASSERT_EQUAL_size_t_MESSAGE(onScheduler.evalPeak, onScheduler.report.observedPeakLiveBytes,
                                     "trainingRun did not evaluate on options->remat");
    TEST_ASSERT_EQUAL_MEMORY(&plain.result.finalTrainLoss, &onScheduler.result.finalTrainLoss,
                             sizeof(float));
    TEST_ASSERT_EQUAL_MEMORY(&plain.result.finalEvalStats.loss,
                             &onScheduler.result.finalEvalStats.loss, sizeof(float));
}

/* A deinitialised scheduler (zeroed by its deinit) in the options exits at the
 * first training call, at the driver's guard, before any bind. */
void testTrainingRunExitsOnADeinitialisedOptionsScheduler(void) {
    ASSERT_EXITS_WITH_OUTPUT(1,
                             "rematCheckNumWires: scheduler not initialised (never initialised, "
                             "or its init failed before the plan was built)",
                             (void)runTwoEpochs(true, true));
}

/* The eval loader's dataset: the first g_evalCount epoch samples. */
static size_t g_evalCount = 4;

static size_t getEvalCountDatasetSize() {
    return g_evalCount;
}

/* NULL, or the eval loader's getBatch for one run (it may delegate to
 * g_baseGetBatch, the loader's own). */
static getBatchFn_t g_evalGetBatchOverride;
static getBatchFn_t g_baseGetBatch;

/* An inferenceWithLossFn_t conformer that wraps the real one: it records the
 * scheduler of every call it gets and forwards the call unchanged. */
static size_t g_forwardedCalls;
static size_t g_forwardedOnScheduler;
static rematScheduler_t *g_expectedRemat;

static inferenceStats_t *forwardingInference(layer_t **model, size_t numberOfLayers,
                                             tensor_t *input, tensor_t *label,
                                             lossFuncType_t funcType, reduction_t forwardReduction,
                                             const trainingCall_t *call) {
    g_forwardedCalls++;
    if (call != NULL && call->remat != NULL && call->remat == g_expectedRemat) {
        g_forwardedOnScheduler++;
    }
    return inferenceWithLoss(model, numberOfLayers, input, label, funcType, forwardReduction, call);
}

/* runTwoEpochs at training micro-batch m, evaluated at mEval over the first
 * evalCount samples (eval batchSize 1, so evalCount is the nominal count),
 * with or without a HEAP scheduler keyed to the [m, 2] training batch. */
static remRun_t runMicroBatched(size_t m, size_t mEval, size_t evalCount,
                                inferenceWithLossFn_t inferenceFn, bool withScheduler) {
    tensor_t *wParam = buildFloatTensor2D(2, 2, (float[]){1.f, 0.f, 0.f, 1.f}, 4);
    tensor_t *wGrad = gradInitFloat(wParam, NULL);
    parameter_t *w = parameterInit(wParam, wGrad);
    tensor_t *bParam = buildFloatTensor2D(1, 2, (float[]){0.f, 0.f}, 2);
    tensor_t *bGrad = gradInitFloat(bParam, NULL);
    parameter_t *b = parameterInit(bParam, bGrad);
    quantization_t testQ;
    initFloat32Quantization(&testQ);
    layer_t *linear = buildBorrowedLinearLayer(w, b, &testQ);
    layer_t *model[] = {linear};
    quantization_t *momentumQ = quantizationInitFloat();
    optimizer_t *sgd =
        sgdMCreateOptim(0.01f, 0.f, 0.f, model, 1, momentumQ,
                        (arithmetic_t){.type = ARITH_FLOAT32, .roundingMode = HALF_AWAY});
    initEpochDataset();
    g_evalCount = evalCount;
    dataLoader_t *trainDl =
        dataLoaderInit(getEpochSample, getEpochDatasetSize, (uint16_t)m, NULL, NULL, false, true);
    dataLoader_t *evalDl =
        dataLoaderInit(getEpochSample, getEvalCountDatasetSize, 1, NULL, NULL, false, true);
    if (g_evalGetBatchOverride != NULL) {
        g_baseGetBatch = evalDl->getBatch;
        evalDl->getBatch = g_evalGetBatchOverride;
        g_evalGetBatchOverride = NULL;
    }
    lossConfig_t cfg = defaultLossConfig(MSE);
    tensor_t *trainBatchLike = buildFloatTensor2D(m, 2, (float[8]){0}, 2 * m);

    remRun_t run = {0};
    rematScheduler_t s;
    if (withScheduler) {
        TEST_ASSERT_TRUE(rematHeapInit(&s, model, 1, cfg, trainBatchLike, NULL));
    }
    g_forwardedCalls = 0;
    g_forwardedOnScheduler = 0;
    g_expectedRemat = withScheduler ? &s : NULL;
    run.result = trainingRun(
        model, 1, cfg, trainDl, evalDl, sgd, 2, calculateGradsSequential, inferenceFn,
        &(trainingRunOptions_t){
            .remat = withScheduler ? &s : NULL, .microBatchSize = m, .evalMicroBatchSize = mEval});
    g_expectedRemat = NULL; /* s ends with this frame */
    if (withScheduler) {
        rematSchedulerReport(&s, &run.report);
        run.evalPeak = rematPlanProgram(s.plan, REMAT_MODE_EVAL)->peakLiveBytes;
        rematSchedulerDeinit(&s);
    }

    freeTensor(trainBatchLike);
    freeOptim(sgd);
    freeQuantization(momentumQ);
    freeDataLoader(evalDl);
    freeDataLoader(trainDl);
    freeLinearLayerShellOnly(linear);
    freeEpochDataset();
    return run;
}

/* A conformer that wraps inferenceWithLoss receives the run's call on every
 * eval call and forwards it: per sample at m = 1 (4 calls per epoch), per
 * chunk of two at m = 2 (2 per epoch). */
void testTrainingRunHandsEveryEvalCallTheOptionsScheduler(void) {
    remRun_t perSample = runMicroBatched(1, 1, 4, forwardingInference, true);
    TEST_ASSERT_EQUAL_size_t(8, g_forwardedCalls);
    TEST_ASSERT_EQUAL_size_t(8, g_forwardedOnScheduler);
    TEST_ASSERT_EQUAL_size_t(perSample.evalPeak, perSample.report.observedPeakLiveBytes);
    remRun_t stacked = runMicroBatched(2, 2, 4, forwardingInference, true);
    TEST_ASSERT_EQUAL_size_t(4, g_forwardedCalls);
    TEST_ASSERT_EQUAL_size_t(4, g_forwardedOnScheduler);
    TEST_ASSERT_EQUAL_size_t(stacked.evalPeak, stacked.report.observedPeakLiveBytes);
}

/* With options->remat set, a run whose eval chunks would not all have the
 * training row count m evaluates without the scheduler: it completes (the
 * death-test wrapper keeps a wrong exit from ending the binary), its last
 * bind is a training call's (TRAIN peak), and its losses equal the
 * scheduler-less run's. */
static void assertEvaluationFallsBackToNoScheduler(size_t m, size_t mEval, size_t evalCount) {
    ASSERT_EXITS_WITH_OUTPUT(0, "",
                             (void)runMicroBatched(m, mEval, evalCount, inferenceWithLoss, true));
    remRun_t plain = runMicroBatched(m, mEval, evalCount, inferenceWithLoss, false);
    remRun_t run = runMicroBatched(m, mEval, evalCount, inferenceWithLoss, true);
    TEST_ASSERT_EQUAL_size_t(2, run.result.epochsCompleted);
    TEST_ASSERT_EQUAL_size_t_MESSAGE(run.report.peakLiveBytes, run.report.observedPeakLiveBytes,
                                     "evaluation ran on options->remat");
    TEST_ASSERT_EQUAL_MEMORY(&plain.result.finalTrainLoss, &run.result.finalTrainLoss,
                             sizeof(float));
    TEST_ASSERT_EQUAL_MEMORY(&plain.result.finalEvalStats.loss, &run.result.finalEvalStats.loss,
                             sizeof(float));
}

void testTrainingRunEvaluatesWithoutTheSchedulerAtAnotherEvalMicroBatch(void) {
    assertEvaluationFallsBackToNoScheduler(1, 2, 4);
}

void testTrainingRunEvaluatesWithoutTheSchedulerOverATailChunk(void) {
    assertEvaluationFallsBackToNoScheduler(2, 2, 3);
}

void testTrainingRunEvaluatesWithoutTheSchedulerBelowOneChunk(void) {
    assertEvaluationFallsBackToNoScheduler(2, 2, 1);
}

/* The control: every chunk is [2, 2], so evaluation runs on the scheduler. */
void testTrainingRunEvaluatesOnTheSchedulerWhenEveryChunkHasTheTrainingRowCount(void) {
    remRun_t plain = runMicroBatched(2, 2, 4, inferenceWithLoss, false);
    remRun_t run = runMicroBatched(2, 2, 4, inferenceWithLoss, true);
    TEST_ASSERT_EQUAL_size_t_MESSAGE(run.evalPeak, run.report.observedPeakLiveBytes,
                                     "evaluation did not run on options->remat");
    TEST_ASSERT_EQUAL_MEMORY(&plain.result.finalTrainLoss, &run.result.finalTrainLoss,
                             sizeof(float));
    TEST_ASSERT_EQUAL_MEMORY(&plain.result.finalEvalStats.loss, &run.result.finalEvalStats.loss,
                             sizeof(float));
}

/* The entry check judges the nominal count, as requireEvaluableChunks does: a
 * loader whose stream is shorter than nominal (here its last batch comes back
 * empty, so 3 of 4 samples stream at m = 2) passes it, and the ragged last
 * chunk meets the scheduler's bind check, which names the row count. */
static batch_t *droppingTheLastSample(dataLoader_t *dataLoader, size_t index) {
    batch_t *batch = g_baseGetBatch(dataLoader, index);
    size_t last = dataLoader->getDatasetSize() / dataLoader->batchSize - 1u;
    if (index == last && batch->size > 0) {
        batch->size--;
        freeSample(batch->samples[batch->size]);
    }
    return batch;
}

/* Runs as the death test's statement, so only the forked child ever sets the
 * override: a failing assertion in the parent cannot leave it set. */
static void runOnAStreamShorterThanItsNominalCount(void) {
    g_evalGetBatchOverride = droppingTheLastSample;
    (void)runMicroBatched(2, 2, 4, inferenceWithLoss, true);
}

void testTrainingRunExitsAtTheRaggedChunkOfAStreamShorterThanItsNominalCount(void) {
    ASSERT_EXITS_WITH_OUTPUT(1,
                             "rematWireTableBind: key mismatch on wire ACT 0, field 'dims[0]': "
                             "built 2, live 1",
                             runOnAStreamShorterThanItsNominalCount());
}

/* ---- training-stream keys: trainingRng_t and its helpers ---- */

/* The expected keys come from the Python mirror, not from the C key chain, with
 * the tag values spelled as numbers (STEP 1, CALL 2, EVAL 6, UPDATE 18). From
 * the repository root:
 *   uv run python -c "from examples._shared.keyed_rng import (fold, key_from_seed
 *   as kfs); r = kfs(0x0123456789abcdef); s1 = fold(r, 1, 1); e1 = fold(r, 6, 1); print([
 *   [hex(w) for w in k] for k in (r, fold(r, 1, 0), s1, fold(s1, 2, 3),
 *   fold(r, 18, 1), fold(r, 6, 0), e1, fold(e1, 2, 5))])"
 * Each test also compares the helper with the explicit rngFold chain, so a wrong
 * tag value and a wrong fold fail differently. */
#define TRAINING_RNG_TEST_SEED 0x0123456789abcdefULL

static void assertTrainingKey(uint32_t w0, uint32_t w1, rngKey_t key) {
    TEST_ASSERT_EQUAL_HEX32(w0, key.w[0]);
    TEST_ASSERT_EQUAL_HEX32(w1, key.w[1]);
}

static void assertSameKey(rngKey_t expected, rngKey_t actual) {
    TEST_ASSERT_EQUAL_HEX32(expected.w[0], actual.w[0]);
    TEST_ASSERT_EQUAL_HEX32(expected.w[1], actual.w[1]);
}

void testTrainingRngInitSetsTheRootAndClearsTheCounters(void) {
    trainingRng_t rng = {.root = {{0xdeadbeefu, 0xdeadbeefu}}, .step = 7u, .epoch = 9u};
    trainingRngInit(&rng, TRAINING_RNG_TEST_SEED);
    assertSameKey(rngKeyFromSeed(TRAINING_RNG_TEST_SEED), rng.root);
    assertTrainingKey(0xd1445d70u, 0x2266ebdeu, rng.root);
    TEST_ASSERT_EQUAL_UINT32(0u, rng.step);
    TEST_ASSERT_EQUAL_UINT32(0u, rng.epoch);
}

void testTrainingRngStepKeyIsTheStepFoldOfTheRoot(void) {
    trainingRng_t rng;
    trainingRngInit(&rng, TRAINING_RNG_TEST_SEED);
    assertSameKey(rngFold(rng.root, RNG_TAG_STEP, 0u), trainingRngStepKey(&rng));
    assertTrainingKey(0x488612fbu, 0x4df9f3a1u, trainingRngStepKey(&rng));
    trainingRngAdvance(&rng);
    assertSameKey(rngFold(rng.root, RNG_TAG_STEP, 1u), trainingRngStepKey(&rng));
    assertTrainingKey(0x925a85f6u, 0xcd30b2f7u, trainingRngStepKey(&rng));
}

void testTrainingRngCallKeyIsTheCallFoldOfItsParent(void) {
    trainingRng_t rng;
    trainingRngInit(&rng, TRAINING_RNG_TEST_SEED);
    trainingRngAdvance(&rng);
    const rngKey_t stepKey = trainingRngStepKey(&rng);
    assertSameKey(rngFold(stepKey, RNG_TAG_CALL, 3u), trainingRngCallKey(stepKey, 3u));
    assertTrainingKey(0xbe960da6u, 0x66f79f1du, trainingRngCallKey(stepKey, 3u));
    const rngKey_t nextCall = trainingRngCallKey(stepKey, 4u);
    TEST_ASSERT_FALSE(nextCall.w[0] == 0xbe960da6u && nextCall.w[1] == 0x66f79f1du);
}

void testTrainingRngUpdateKeyIsTheUpdateFoldOfTheRoot(void) {
    trainingRng_t rng;
    trainingRngInit(&rng, TRAINING_RNG_TEST_SEED);
    trainingRngAdvance(&rng);
    assertSameKey(rngFold(rng.root, RNG_TAG_UPDATE, 1u), trainingRngUpdateKey(&rng));
    assertTrainingKey(0x1722c30du, 0x530de81fu, trainingRngUpdateKey(&rng));
}

void testTrainingRngEvalKeyIsTheEvalFoldOfTheRoot(void) {
    trainingRng_t rng;
    trainingRngInit(&rng, TRAINING_RNG_TEST_SEED);
    trainingRngAdvance(&rng); /* the evaluation key follows the epoch, never the step */
    assertSameKey(rngFold(rng.root, RNG_TAG_EVAL, 0u), trainingRngEvalKey(&rng));
    assertTrainingKey(0x238e4cf9u, 0x06660cf1u, trainingRngEvalKey(&rng));
    trainingRngAdvanceEpoch(&rng);
    const rngKey_t evalKey = trainingRngEvalKey(&rng);
    assertTrainingKey(0xd252cbe9u, 0xbfbe2e3du, evalKey);
    assertTrainingKey(0x0e6cb39du, 0x6d05e020u, trainingRngCallKey(evalKey, 5u));
}

void testTrainingRngAdvanceCountsUpdatesAndAdvanceEpochCountsEpochs(void) {
    trainingRng_t rng;
    trainingRngInit(&rng, TRAINING_RNG_TEST_SEED);
    const rngKey_t root = rng.root;
    trainingRngAdvance(&rng);
    trainingRngAdvance(&rng);
    trainingRngAdvanceEpoch(&rng);
    TEST_ASSERT_EQUAL_UINT32(2u, rng.step);
    TEST_ASSERT_EQUAL_UINT32(1u, rng.epoch);
    assertSameKey(root, rng.root);
}

/* The last step, epoch and call index are still usable: only going past one exits. */
void testTrainingRngReachesTheLastStepEpochAndCallIndexWithoutExiting(void) {
    trainingRng_t rng;
    trainingRngInit(&rng, TRAINING_RNG_TEST_SEED);
    rng.step = UINT32_MAX - 1u;
    rng.epoch = UINT32_MAX - 1u;
    trainingRngAdvance(&rng);
    trainingRngAdvanceEpoch(&rng);
    TEST_ASSERT_EQUAL_UINT32(UINT32_MAX, rng.step);
    TEST_ASSERT_EQUAL_UINT32(UINT32_MAX, rng.epoch);
    const rngKey_t stepKey = trainingRngStepKey(&rng);
    assertSameKey(rngFold(stepKey, RNG_TAG_CALL, UINT32_MAX),
                  trainingRngCallKey(stepKey, (size_t)UINT32_MAX));
}

static void initANullState(void) {
    trainingRngInit(NULL, 1u);
}

static void stepKeyOfANullState(void) {
    (void)trainingRngStepKey(NULL);
}

static void updateKeyOfANullState(void) {
    (void)trainingRngUpdateKey(NULL);
}

static void evalKeyOfANullState(void) {
    (void)trainingRngEvalKey(NULL);
}

static void advanceANullState(void) {
    trainingRngAdvance(NULL);
}

static void advanceEpochOfANullState(void) {
    trainingRngAdvanceEpoch(NULL);
}

void testTrainingRngHelpersExitOnANullState(void) {
    ASSERT_EXITS_WITH_OUTPUT(1, "trainingRngInit: rng is NULL", initANullState());
    ASSERT_EXITS_WITH_OUTPUT(1, "trainingRngStepKey: rng is NULL", stepKeyOfANullState());
    ASSERT_EXITS_WITH_OUTPUT(1, "trainingRngUpdateKey: rng is NULL", updateKeyOfANullState());
    ASSERT_EXITS_WITH_OUTPUT(1, "trainingRngEvalKey: rng is NULL", evalKeyOfANullState());
    ASSERT_EXITS_WITH_OUTPUT(1, "trainingRngAdvance: rng is NULL", advanceANullState());
    ASSERT_EXITS_WITH_OUTPUT(1, "trainingRngAdvanceEpoch: rng is NULL", advanceEpochOfANullState());
}

static void advancePastTheLastStep(void) {
    trainingRng_t rng;
    trainingRngInit(&rng, 1u);
    rng.step = UINT32_MAX;
    trainingRngAdvance(&rng);
}

void testTrainingRngAdvanceExitsWhenTheStepCounterIsExhausted(void) {
    ASSERT_EXITS_WITH_OUTPUT(1, "trainingRngAdvance: step counter exhausted",
                             advancePastTheLastStep());
}

static void advancePastTheLastEpoch(void) {
    trainingRng_t rng;
    trainingRngInit(&rng, 1u);
    rng.epoch = UINT32_MAX;
    trainingRngAdvanceEpoch(&rng);
}

void testTrainingRngAdvanceEpochExitsWhenTheEpochCounterIsExhausted(void) {
    ASSERT_EXITS_WITH_OUTPUT(1, "trainingRngAdvanceEpoch: epoch counter exhausted",
                             advancePastTheLastEpoch());
}

#if SIZE_MAX > UINT32_MAX
static void callKeyPastUint32Max(void) {
    (void)trainingRngCallKey(rngKeyFromSeed(1u), (size_t)UINT32_MAX + 1u);
}

void testTrainingRngCallKeyExitsAboveUint32Max(void) {
    ASSERT_EXITS_WITH_OUTPUT(1, "trainingRngCallKey: call index exceeds UINT32_MAX",
                             callKeyPastUint32Max());
}
#endif

/* ---- shuffle and init keys: the roots outside the step chain ---- */

/* From the Python mirror, tag values spelled as numbers (SHUFFLE 21, INIT 22,
 * SHUFFLE_EVAL 26); same root as the tests above. From the repository root:
 *   uv run python -c "from examples._shared.keyed_rng import (fold,
 *   key_from_seed as kfs); r = kfs(0x0123456789abcdef); print([[hex(w) for w in k]
 *   for k in (fold(r, 21, 0), fold(r, 21, 2), fold(r, 26, 0), fold(r, 22, 0),
 *   fold(r, 22, 3), fold(r, 22, 0xffffffff))])"
 */

/* A per-epoch shuffle follows the epoch counter; a shuffle-once loader keeps
 * epoch 0's key for the whole run. The step counter never enters either. */
void testTrainingRngShuffleKeyFollowsTheEpochOnlyWhenPerEpoch(void) {
    trainingRng_t rng;
    trainingRngInit(&rng, TRAINING_RNG_TEST_SEED);
    assertSameKey(trainingRngShuffleKey(&rng, false), trainingRngShuffleKey(&rng, true));
    trainingRngAdvance(&rng);
    trainingRngAdvanceEpoch(&rng);
    trainingRngAdvanceEpoch(&rng);
    assertSameKey(rngFold(rng.root, RNG_TAG_SHUFFLE, 2u), trainingRngShuffleKey(&rng, true));
    assertTrainingKey(0x0d7b6f6du, 0x8d2a882du, trainingRngShuffleKey(&rng, true));
    assertSameKey(rngFold(rng.root, RNG_TAG_SHUFFLE, 0u), trainingRngShuffleKey(&rng, false));
    assertTrainingKey(0x03ce5c0bu, 0x84d3a650u, trainingRngShuffleKey(&rng, false));
}

/* The evaluation loader is shuffled once per run: its key ignores both counters
 * and differs from the training loader's epoch-0 key. */
void testTrainingRngShuffleEvalKeyIsOneKeyPerRun(void) {
    trainingRng_t rng;
    trainingRngInit(&rng, TRAINING_RNG_TEST_SEED);
    trainingRngAdvance(&rng);
    trainingRngAdvanceEpoch(&rng);
    assertSameKey(rngFold(rng.root, RNG_TAG_SHUFFLE_EVAL, 0u), trainingRngShuffleEvalKey(&rng));
    assertTrainingKey(0xdf93fb35u, 0xcf8b986bu, trainingRngShuffleEvalKey(&rng));
    const rngKey_t trainKey = trainingRngShuffleKey(&rng, false);
    const rngKey_t evalKey = trainingRngShuffleEvalKey(&rng);
    TEST_ASSERT_FALSE(trainKey.w[0] == evalKey.w[0] && trainKey.w[1] == evalKey.w[1]);
}

/* One init key per layer index, independent of the counters; the last 32-bit
 * index is still usable. */
void testTrainingRngInitKeyIsTheInitFoldOfTheLayerIndex(void) {
    trainingRng_t rng;
    trainingRngInit(&rng, TRAINING_RNG_TEST_SEED);
    trainingRngAdvance(&rng);
    trainingRngAdvanceEpoch(&rng);
    assertSameKey(rngFold(rng.root, RNG_TAG_INIT, 0u), trainingRngInitKey(&rng, 0u));
    assertTrainingKey(0x9b266274u, 0xf3f69b89u, trainingRngInitKey(&rng, 0u));
    assertSameKey(rngFold(rng.root, RNG_TAG_INIT, 3u), trainingRngInitKey(&rng, 3u));
    assertTrainingKey(0x5cdda14fu, 0xe704a5f1u, trainingRngInitKey(&rng, 3u));
    assertTrainingKey(0x32cb78bcu, 0xb2b4e2ddu, trainingRngInitKey(&rng, (size_t)UINT32_MAX));
}

static void shuffleKeyOfANullState(void) {
    (void)trainingRngShuffleKey(NULL, true);
}

static void shuffleEvalKeyOfANullState(void) {
    (void)trainingRngShuffleEvalKey(NULL);
}

static void initKeyOfANullState(void) {
    (void)trainingRngInitKey(NULL, 0u);
}

void testTrainingRngShuffleKeyExitsOnANullState(void) {
    ASSERT_EXITS_WITH_OUTPUT(1, "trainingRngShuffleKey: rng is NULL", shuffleKeyOfANullState());
}

void testTrainingRngShuffleEvalKeyExitsOnANullState(void) {
    ASSERT_EXITS_WITH_OUTPUT(1, "trainingRngShuffleEvalKey: rng is NULL",
                             shuffleEvalKeyOfANullState());
}

void testTrainingRngInitKeyExitsOnANullState(void) {
    ASSERT_EXITS_WITH_OUTPUT(1, "trainingRngInitKey: rng is NULL", initKeyOfANullState());
}

#if SIZE_MAX > UINT32_MAX
static void initKeyPastUint32Max(void) {
    trainingRng_t rng;
    trainingRngInit(&rng, 1u);
    (void)trainingRngInitKey(&rng, (size_t)UINT32_MAX + 1u);
}

void testTrainingRngInitKeyExitsAboveUint32Max(void) {
    ASSERT_EXITS_WITH_OUTPUT(1, "trainingRngInitKey: layer index exceeds UINT32_MAX",
                             initKeyPastUint32Max());
}
#endif

/* ---- ODTK: the training-stream state on disk ---- */

/* The expected bytes come from Python's struct module over the mirror's root
 * key, not from the C writer. From the repository root:
 *   uv run python -c "import struct; from examples._shared.keyed_rng import (
 *   key_from_seed as kfs); b = struct.pack('<4sIIIII', b'ODTK', 1,
 *   *kfs(0x0123456789abcdef), 0x11223344, 0x55667788); print(len(b), ', '.join(
 *   '0x%02x' % x for x in b))"
 */
#define ODTK_RECORD_BYTES 24u

static const uint8_t kOdtkRecord[ODTK_RECORD_BYTES] = {
    0x4f, 0x44, 0x54, 0x4b, 0x01, 0x00, 0x00, 0x00, 0x70, 0x5d, 0x44, 0xd1,
    0xde, 0xeb, 0x66, 0x22, 0x44, 0x33, 0x22, 0x11, 0x88, 0x77, 0x66, 0x55};

static trainingRng_t odtkTestState(void) {
    trainingRng_t rng;
    trainingRngInit(&rng, TRAINING_RNG_TEST_SEED);
    rng.step = 0x11223344u;
    rng.epoch = 0x55667788u;
    return rng;
}

/* A stream positioned at the start of the given bytes. */
static FILE *odtkStreamOf(const uint8_t *bytes, size_t numberOfBytes) {
    FILE *f = tmpfile();
    TEST_ASSERT_NOT_NULL(f);
    TEST_ASSERT_EQUAL_UINT((unsigned)numberOfBytes, (unsigned)fwrite(bytes, 1, numberOfBytes, f));
    rewind(f);
    return f;
}

void testTrainingRngSerializeWritesTheOdtkRecord(void) {
    const trainingRng_t rng = odtkTestState();
    FILE *f = tmpfile();
    TEST_ASSERT_NOT_NULL(f);

    trainingRngSerialize(&rng, f);

    TEST_ASSERT_EQUAL_INT((int)ODTK_RECORD_BYTES, (int)ftell(f));
    rewind(f);
    uint8_t bytes[ODTK_RECORD_BYTES];
    TEST_ASSERT_EQUAL_UINT(ODTK_RECORD_BYTES, (unsigned)fread(bytes, 1, sizeof bytes, f));
    fclose(f);
    TEST_ASSERT_EQUAL_HEX8_ARRAY(kOdtkRecord, bytes, ODTK_RECORD_BYTES);
}

/* The reader takes exactly the record's bytes and leaves what follows. */
void testTrainingRngDeserializeReadsTheOdtkRecord(void) {
    const uint8_t recordThenMore[ODTK_RECORD_BYTES + 1u] = {
        0x4f, 0x44, 0x54, 0x4b, 0x01, 0x00, 0x00, 0x00, 0x70, 0x5d, 0x44, 0xd1, 0xde,
        0xeb, 0x66, 0x22, 0x44, 0x33, 0x22, 0x11, 0x88, 0x77, 0x66, 0x55, 0x5a};
    FILE *f = odtkStreamOf(recordThenMore, sizeof recordThenMore);
    trainingRng_t rng = {.root = {{0xdeadbeefu, 0xdeadbeefu}}, .step = 7u, .epoch = 9u};

    trainingRngDeserialize(&rng, f);

    const int next = fgetc(f);
    fclose(f);
    assertTrainingKey(0xd1445d70u, 0x2266ebdeu, rng.root);
    TEST_ASSERT_EQUAL_HEX32(0x11223344u, rng.step);
    TEST_ASSERT_EQUAL_HEX32(0x55667788u, rng.epoch);
    TEST_ASSERT_EQUAL_INT(0x5a, next);
}

/* A restored state hands out the keys of the saved one. */
void testTrainingRngDeserializeRestoresWhatSerializeSaved(void) {
    const trainingRng_t saved = odtkTestState();
    FILE *f = tmpfile();
    TEST_ASSERT_NOT_NULL(f);
    trainingRngSerialize(&saved, f);
    rewind(f);
    trainingRng_t restored;
    trainingRngInit(&restored, 1u);

    trainingRngDeserialize(&restored, f);

    fclose(f);
    assertSameKey(saved.root, restored.root);
    TEST_ASSERT_EQUAL_HEX32(saved.step, restored.step);
    TEST_ASSERT_EQUAL_HEX32(saved.epoch, restored.epoch);
    assertSameKey(trainingRngStepKey(&saved), trainingRngStepKey(&restored));
    assertSameKey(trainingRngShuffleKey(&saved, true), trainingRngShuffleKey(&restored, true));
}

static void deserializeEditedRecord(size_t at, uint8_t value) {
    uint8_t bytes[ODTK_RECORD_BYTES];
    for (size_t i = 0; i < ODTK_RECORD_BYTES; i++) {
        bytes[i] = kOdtkRecord[i];
    }
    bytes[at] = value;
    FILE *f = odtkStreamOf(bytes, ODTK_RECORD_BYTES);
    trainingRng_t rng;
    trainingRngDeserialize(&rng, f);
}

static void deserializeAModelFileMagic(void) {
    deserializeEditedRecord(3u, 'S'); /* "ODTS" */
}

static void deserializeVersion2(void) {
    deserializeEditedRecord(4u, 0x02u);
}

static void deserializeATruncatedRecord(void) {
    FILE *f = odtkStreamOf(kOdtkRecord, ODTK_RECORD_BYTES - 4u);
    trainingRng_t rng;
    trainingRngDeserialize(&rng, f);
}

static void serializeANullState(void) {
    FILE *f = tmpfile();
    trainingRngSerialize(NULL, f);
}

static void deserializeIntoANullState(void) {
    FILE *f = odtkStreamOf(kOdtkRecord, ODTK_RECORD_BYTES);
    trainingRngDeserialize(NULL, f);
}

void testTrainingRngDeserializeExitsOnABadMagic(void) {
    ASSERT_EXITS_WITH_OUTPUT(1, "trainingRngDeserialize: bad magic (not an ODTK record)",
                             deserializeAModelFileMagic());
}

void testTrainingRngDeserializeExitsOnAnUnsupportedVersion(void) {
    ASSERT_EXITS_WITH_OUTPUT(1, "trainingRngDeserialize: unsupported version 2",
                             deserializeVersion2());
}

/* A short stream stops in the shared wire reader, with its text. */
void testTrainingRngDeserializeExitsOnATruncatedRecord(void) {
    ASSERT_EXITS_WITH_OUTPUT(1, "short read (4 bytes): truncated or corrupt stream",
                             deserializeATruncatedRecord());
}

void testTrainingRngSerializeExitsOnANullState(void) {
    ASSERT_EXITS_WITH_OUTPUT(1, "trainingRngSerialize: rng is NULL", serializeANullState());
}

void testTrainingRngDeserializeExitsOnANullState(void) {
    ASSERT_EXITS_WITH_OUTPUT(1, "trainingRngDeserialize: rng is NULL", deserializeIntoANullState());
}

/* ---- the keys the training and evaluation loops hand down (#4) ---- */

#define KEYED_RUN_SEED 0x5eedu
#define KEY_RECORDS 16

/* A NULL call stays NULL: nothing to copy, no key to derive. */
void testTrainingCallForIndexKeepsANullCallNull(void) {
    trainingCall_t next;
    rngKey_t callKey = {{0xdeadbeefu, 0xdeadbeefu}};

    TEST_ASSERT_NULL(trainingCallForIndex(NULL, 3u, &next, &callKey));
    TEST_ASSERT_EQUAL_HEX32(0xdeadbeefu, callKey.w[0]);
}

/* A call without a key is copied with .key NULL; every other field carries over. */
void testTrainingCallForIndexCopiesACallWithoutAKey(void) {
    trainingRng_t rng;
    rematScheduler_t s;
    const trainingCall_t call = {.remat = &s, .rng = &rng};
    trainingCall_t next;
    rngKey_t callKey;

    const trainingCall_t *got = trainingCallForIndex(&call, 3u, &next, &callKey);

    TEST_ASSERT_EQUAL_PTR(&next, got);
    TEST_ASSERT_EQUAL_PTR(&s, next.remat);
    TEST_ASSERT_EQUAL_PTR(&rng, next.rng);
    TEST_ASSERT_NULL(next.key);
}

/* A keyed call is copied with .key replaced by the key of the call index
 * under it, written into the caller's storage. */
void testTrainingCallForIndexDerivesTheCallKeyUnderTheParentKey(void) {
    trainingRng_t rng;
    trainingRngInit(&rng, KEYED_RUN_SEED);
    rematScheduler_t s;
    const rngKey_t parent = trainingRngStepKey(&rng);
    const trainingCall_t call = {.remat = &s, .key = &parent, .rng = &rng};
    trainingCall_t next;
    rngKey_t callKey;

    const trainingCall_t *got = trainingCallForIndex(&call, 3u, &next, &callKey);

    TEST_ASSERT_EQUAL_PTR(&next, got);
    TEST_ASSERT_EQUAL_PTR(&callKey, next.key);
    assertSameKey(trainingRngCallKey(parent, 3u), callKey);
    TEST_ASSERT_EQUAL_PTR(&s, next.remat);
    TEST_ASSERT_EQUAL_PTR(&rng, next.rng);
}

/* next may be the call itself: the key is still derived from the parent key
 * the call held before the copy. */
void testTrainingCallForIndexDerivesTheKeyInPlace(void) {
    trainingRng_t rng;
    trainingRngInit(&rng, KEYED_RUN_SEED);
    rematScheduler_t s;
    const rngKey_t parent = trainingRngStepKey(&rng);
    trainingCall_t call = {.remat = &s, .key = &parent, .rng = &rng};
    rngKey_t callKey;

    const trainingCall_t *got = trainingCallForIndex(&call, 3u, &call, &callKey);

    TEST_ASSERT_EQUAL_PTR(&call, got);
    TEST_ASSERT_EQUAL_PTR(&callKey, call.key);
    assertSameKey(trainingRngCallKey(parent, 3u), callKey);
    TEST_ASSERT_EQUAL_PTR(&s, call.remat);
    TEST_ASSERT_EQUAL_PTR(&rng, call.rng);
}

typedef struct {
    bool hasKey;
    rngKey_t key;
} keyRecord_t;

static keyRecord_t g_gradKeys[KEY_RECORDS];
static size_t g_gradKeyCount;
static keyRecord_t g_evalKeys[KEY_RECORDS];
static size_t g_evalKeyCount;

static void recordCallKey(keyRecord_t *records, size_t *count, const trainingCall_t *call) {
    if (*count < KEY_RECORDS) {
        records[*count].hasKey = call != NULL && call->key != NULL;
        if (records[*count].hasKey) {
            records[*count].key = *call->key;
        }
    }
    (*count)++;
}

static trainingStats_t *keyRecordingGrads(layer_t **model, size_t modelSize,
                                          lossConfig_t lossConfig, reduction_t forwardReduction,
                                          tensor_t *input, tensor_t *label,
                                          const trainingCall_t *call) {
    recordCallKey(g_gradKeys, &g_gradKeyCount, call);
    return calculateGradsSequential(model, modelSize, lossConfig, forwardReduction, input, label,
                                    call);
}

static inferenceStats_t *keyRecordingInference(layer_t **model, size_t numberOfLayers,
                                               tensor_t *input, tensor_t *label,
                                               lossFuncType_t funcType,
                                               reduction_t forwardReduction,
                                               const trainingCall_t *call) {
    recordCallKey(g_evalKeys, &g_evalKeyCount, call);
    return inferenceWithLoss(model, numberOfLayers, input, label, funcType, forwardReduction, call);
}

/* runMicroBatched's one-Linear model and SGD over the 4-sample dataset:
 * training batches of trainBatch, evaluation over the first evalCount
 * samples in batches of evalBatch. */
typedef struct {
    quantization_t testQ;
    layer_t *linear;
    layer_t *model[1];
    quantization_t *momentumQ;
    optimizer_t *sgd;
    dataLoader_t *trainDl;
    dataLoader_t *evalDl;
} keyedFixture_t;

static void buildKeyedFixture(keyedFixture_t *f, uint16_t trainBatch, uint16_t evalBatch,
                              size_t evalCount) {
    tensor_t *wParam = buildFloatTensor2D(2, 2, (float[]){1.f, 0.f, 0.f, 1.f}, 4);
    parameter_t *w = parameterInit(wParam, gradInitFloat(wParam, NULL));
    tensor_t *bParam = buildFloatTensor2D(1, 2, (float[]){0.f, 0.f}, 2);
    parameter_t *b = parameterInit(bParam, gradInitFloat(bParam, NULL));
    initFloat32Quantization(&f->testQ);
    f->linear = buildBorrowedLinearLayer(w, b, &f->testQ);
    f->model[0] = f->linear;
    f->momentumQ = quantizationInitFloat();
    f->sgd = sgdMCreateOptim(0.01f, 0.f, 0.f, f->model, 1, f->momentumQ,
                             (arithmetic_t){.type = ARITH_FLOAT32, .roundingMode = HALF_AWAY});
    initEpochDataset();
    g_evalCount = evalCount;
    f->trainDl =
        dataLoaderInit(getEpochSample, getEpochDatasetSize, trainBatch, NULL, NULL, false, true);
    f->evalDl =
        dataLoaderInit(getEpochSample, getEvalCountDatasetSize, evalBatch, NULL, NULL, false, true);
    g_gradKeyCount = 0;
    g_evalKeyCount = 0;
}

static void freeKeyedFixture(keyedFixture_t *f) {
    freeOptim(f->sgd);
    freeQuantization(f->momentumQ);
    freeDataLoader(f->evalDl);
    freeDataLoader(f->trainDl);
    freeLinearLayerShellOnly(f->linear);
    freeEpochDataset();
}

/* Two epochs of trainingRun at training micro-batch m and evaluation micro-batch mEval. */
static void runKeyed(uint16_t trainBatch, size_t m, uint16_t evalBatch, size_t evalCount,
                     size_t mEval, trainingRng_t *rng) {
    keyedFixture_t f;
    buildKeyedFixture(&f, trainBatch, evalBatch, evalCount);
    (void)trainingRun(
        f.model, 1, defaultLossConfig(MSE), f.trainDl, f.evalDl, f.sgd, 2, keyRecordingGrads,
        keyRecordingInference,
        &(trainingRunOptions_t){.microBatchSize = m, .evalMicroBatchSize = mEval, .rng = rng});
    freeKeyedFixture(&f);
}

static void assertCallKey(rngKey_t parent, size_t callIndex, keyRecord_t record) {
    TEST_ASSERT_TRUE(record.hasKey);
    assertSameKey(trainingRngCallKey(parent, callIndex), record.key);
}

/* Each training call gets the key of its index under the step key of its
 * update; the step counts every update of the run, across epochs. */
void testTrainingRunHandsEachTrainingCallTheCallKeyOfItsUpdate(void) {
    trainingRng_t rng;
    trainingRngInit(&rng, KEYED_RUN_SEED);

    runKeyed(2, 1, 1, 4, 1, &rng);

    TEST_ASSERT_EQUAL_UINT32(4u, rng.step);
    TEST_ASSERT_EQUAL_UINT32(2u, rng.epoch);
    TEST_ASSERT_EQUAL_size_t(8, g_gradKeyCount);
    trainingRng_t expected;
    trainingRngInit(&expected, KEYED_RUN_SEED);
    for (size_t u = 0; u < 4; u++) {
        const rngKey_t stepKey = trainingRngStepKey(&expected);
        for (size_t i = 0; i < 2; i++) {
            assertCallKey(stepKey, i, g_gradKeys[2 * u + i]);
        }
        trainingRngAdvance(&expected);
    }
}

/* At m > 1 one call covers m samples, so the index counts chunks: a batch of
 * four at m = 2 makes calls 0 and 1. */
void testTrainingRunNumbersStackedTrainingCallsByChunk(void) {
    trainingRng_t rng;
    trainingRngInit(&rng, KEYED_RUN_SEED);

    runKeyed(4, 2, 1, 4, 1, &rng);

    TEST_ASSERT_EQUAL_UINT32(2u, rng.step);
    TEST_ASSERT_EQUAL_size_t(4, g_gradKeyCount);
    trainingRng_t expected;
    trainingRngInit(&expected, KEYED_RUN_SEED);
    for (size_t u = 0; u < 2; u++) {
        const rngKey_t stepKey = trainingRngStepKey(&expected);
        assertCallKey(stepKey, 0, g_gradKeys[2 * u]);
        assertCallKey(stepKey, 1, g_gradKeys[2 * u + 1]);
        trainingRngAdvance(&expected);
    }
}

/* Each evaluation call gets the key of its position in the epoch under that
 * epoch's evaluation key; positions run on across the loader's batches. */
void testTrainingRunKeysEachEvaluationCallByItsPositionInTheEpoch(void) {
    trainingRng_t rng;
    trainingRngInit(&rng, KEYED_RUN_SEED);

    runKeyed(2, 1, 2, 4, 1, &rng);

    TEST_ASSERT_EQUAL_size_t(8, g_evalKeyCount);
    trainingRng_t expected;
    trainingRngInit(&expected, KEYED_RUN_SEED);
    for (size_t e = 0; e < 2; e++) {
        const rngKey_t evalKey = trainingRngEvalKey(&expected);
        for (size_t j = 0; j < 4; j++) {
            assertCallKey(evalKey, j, g_evalKeys[4 * e + j]);
        }
        trainingRngAdvanceEpoch(&expected);
    }
}

/* At m > 1 an evaluation call covers a chunk, so the index counts chunks:
 * three samples at m = 2 make a full chunk and the shorter last one, four make
 * two full chunks; both are calls 0 and 1. */
void testTrainingRunNumbersStackedEvaluationCallsByChunk(void) {
    for (size_t evalCount = 3; evalCount <= 4; evalCount++) {
        trainingRng_t rng;
        trainingRngInit(&rng, KEYED_RUN_SEED);

        runKeyed(2, 2, 1, evalCount, 2, &rng);

        TEST_ASSERT_EQUAL_size_t(4, g_evalKeyCount);
        trainingRng_t expected;
        trainingRngInit(&expected, KEYED_RUN_SEED);
        for (size_t e = 0; e < 2; e++) {
            const rngKey_t evalKey = trainingRngEvalKey(&expected);
            assertCallKey(evalKey, 0, g_evalKeys[2 * e]);
            assertCallKey(evalKey, 1, g_evalKeys[2 * e + 1]);
            trainingRngAdvanceEpoch(&expected);
        }
    }
}

/* Without options->rng no training and no evaluation call gets a key. */
void testTrainingRunWithoutAnRngHandsNoKeys(void) {
    runKeyed(2, 1, 2, 4, 1, NULL);

    TEST_ASSERT_EQUAL_size_t(8, g_gradKeyCount);
    TEST_ASSERT_EQUAL_size_t(8, g_evalKeyCount);
    for (size_t r = 0; r < 8; r++) {
        TEST_ASSERT_FALSE(g_gradKeys[r].hasKey);
        TEST_ASSERT_FALSE(g_evalKeys[r].hasKey);
    }
}

static const trainingRng_t *g_callbackRng;
static uint32_t g_callbackRngEpochs[2];
static size_t g_callbackCount;

static void recordRngEpochCallback(epochInfo_t info, epochStats_t evalStats) {
    (void)info;
    (void)evalStats;
    if (g_callbackCount < 2) {
        g_callbackRngEpochs[g_callbackCount] = g_callbackRng->epoch;
    }
    g_callbackCount++;
}

/* The epoch counter advances right after the evaluation, so the callback of
 * an epoch already sees that epoch counted. */
void testTrainingRunCountsTheEpochBeforeTheCallback(void) {
    trainingRng_t rng;
    trainingRngInit(&rng, KEYED_RUN_SEED);
    g_callbackRng = &rng;
    g_callbackCount = 0;
    keyedFixture_t f;
    buildKeyedFixture(&f, 2, 2, 4);

    (void)trainingRun(f.model, 1, defaultLossConfig(MSE), f.trainDl, f.evalDl, f.sgd, 2,
                      calculateGradsSequential, inferenceWithLoss,
                      &(trainingRunOptions_t){.callback = recordRngEpochCallback, .rng = &rng});
    freeKeyedFixture(&f);

    TEST_ASSERT_EQUAL_size_t(2, g_callbackCount);
    TEST_ASSERT_EQUAL_UINT32(1u, g_callbackRngEpochs[0]);
    TEST_ASSERT_EQUAL_UINT32(2u, g_callbackRngEpochs[1]);
}

#define SR_EVAL_HIDDEN 16

/* Linear(2 -> 16) rounding its forward SR into an 8-bit SYM_INT32 wire, then
 * Linear(16 -> 2), trained at lr 0 so every epoch evaluates the same weights. */
typedef struct {
    quantization_t testQ;
    quantization_t *wireQ;
    quantization_t *momentumQ;
    layer_t *model[2];
    optimizer_t *sgd;
    dataLoader_t *trainDl;
    dataLoader_t *evalDl;
} srEvalFixture_t;

static parameter_t *buildFloatParam(size_t d0, size_t d1, float step, float offset) {
    float values[2 * SR_EVAL_HIDDEN];
    for (size_t i = 0; i < d0 * d1; i++) {
        values[i] = step * (float)((i * 7) % 13) + offset;
    }
    tensor_t *t = buildFloatTensor2D(d0, d1, values, d0 * d1);
    return parameterInit(t, gradInitFloat(t, NULL));
}

static void buildSrEvalFixture(srEvalFixture_t *f) {
    initFloat32Quantization(&f->testQ);
    f->wireQ = quantizationInitSymInt32WithBits(HALF_AWAY, 8);
    f->model[0] = buildBorrowedLinearLayer(buildFloatParam(SR_EVAL_HIDDEN, 2, 0.1f, -0.6f),
                                           buildFloatParam(1, SR_EVAL_HIDDEN, 0.f, 0.f), &f->testQ);
    f->model[0]->config->linear->forwardMath =
        (arithmetic_t){.type = ARITH_FLOAT32, .roundingMode = SR_HALF_AWAY};
    f->model[0]->config->linear->outputQ = f->wireQ;
    f->model[1] = buildBorrowedLinearLayer(buildFloatParam(2, SR_EVAL_HIDDEN, 0.05f, -0.3f),
                                           buildFloatParam(1, 2, 0.f, 0.f), &f->testQ);
    f->momentumQ = quantizationInitFloat();
    f->sgd = sgdMCreateOptim(0.f, 0.f, 0.f, f->model, 2, f->momentumQ,
                             (arithmetic_t){.type = ARITH_FLOAT32, .roundingMode = HALF_AWAY});
    initEpochDataset();
    f->trainDl = dataLoaderInit(getEpochSample, getEpochDatasetSize, 1, NULL, NULL, false, true);
    f->evalDl = dataLoaderInit(getEpochSample, getEpochDatasetSize, 1, NULL, NULL, false, true);
}

static void freeSrEvalFixture(srEvalFixture_t *f) {
    freeOptim(f->sgd);
    freeQuantization(f->momentumQ);
    freeDataLoader(f->evalDl);
    freeDataLoader(f->trainDl);
    freeLinearLayerShellOnly(f->model[1]);
    freeLinearLayerShellOnly(f->model[0]);
    freeQuantization(f->wireQ);
    freeEpochDataset();
}

static float g_srEvalLosses[2];

static void recordSrEvalLoss(epochInfo_t info, epochStats_t evalStats) {
    g_srEvalLosses[info.epoch] = evalStats.loss;
}

/* Two epochs of trainingRun on the SR fixture; losses[e] = epoch e's evaluation. */
static void runSrEvaluation(uint64_t seed, float losses[2]) {
    srEvalFixture_t f;
    buildSrEvalFixture(&f);
    trainingRng_t rng;
    trainingRngInit(&rng, seed);
    (void)trainingRun(f.model, 2, defaultLossConfig(MSE), f.trainDl, f.evalDl, f.sgd, 2,
                      calculateGradsSequential, inferenceWithLoss,
                      &(trainingRunOptions_t){.callback = recordSrEvalLoss, .rng = &rng});
    losses[0] = g_srEvalLosses[0];
    losses[1] = g_srEvalLosses[1];
    freeSrEvalFixture(&f);
}

/* The evaluation draws under the run's EVAL key of its epoch: one seed repeats
 * every epoch's evaluation, and two epochs over the same weights evaluate
 * under different keys. */
void testTrainingRunEvaluatesEachEpochUnderItsOwnEvalKey(void) {
    float first[2];
    float again[2];

    runSrEvaluation(KEYED_RUN_SEED, first);
    runSrEvaluation(KEYED_RUN_SEED, again);

    TEST_ASSERT_EQUAL_MEMORY_MESSAGE(first, again, sizeof(first),
                                     "one seed repeats every epoch's evaluation");
    TEST_ASSERT_FALSE_MESSAGE(first[0] == first[1],
                              "the two epochs evaluate the same weights under their own keys");
}

/* The epoch level reads only .rng: a .key on its own call never reaches a
 * training call. */
void testTrainingEpochDoesNotHandItsOwnKeyDown(void) {
    keyedFixture_t f;
    buildKeyedFixture(&f, 2, 1, 4);
    const rngKey_t stray = rngKeyFromSeed(KEYED_RUN_SEED);

    (void)trainingEpochDefault(f.model, 1, defaultLossConfig(MSE), f.trainDl, f.sgd,
                               keyRecordingGrads, REDUCTION_MEAN, 1,
                               &(trainingCall_t){.key = &stray});

    freeKeyedFixture(&f);
    TEST_ASSERT_EQUAL_size_t(4, g_gradKeyCount);
    for (size_t r = 0; r < 4; r++) {
        TEST_ASSERT_FALSE(g_gradKeys[r].hasKey);
    }
}

static stepFn_t g_savedSgdStep;
static keyRecord_t g_updateKeys[KEY_RECORDS];
static size_t g_updateKeyCount;

/* Records the update key of each optimizer step, then runs the real step. */
static void keyRecordingStep(optimizer_t *optim, const rngKey_t *updateKey) {
    if (g_updateKeyCount < KEY_RECORDS) {
        g_updateKeys[g_updateKeyCount].hasKey = updateKey != NULL;
        if (updateKey != NULL) {
            g_updateKeys[g_updateKeyCount].key = *updateKey;
        }
    }
    g_updateKeyCount++;
    g_savedSgdStep(optim, updateKey);
}

/* One epoch of two updates with the SGD step recorded. */
static void epochWithRecordedSteps(const trainingCall_t *call) {
    keyedFixture_t f;
    buildKeyedFixture(&f, 2, 1, 4);
    g_savedSgdStep = optimizerFunctions[SGD_M].step;
    optimizerFunctions[SGD_M].step = keyRecordingStep;
    g_updateKeyCount = 0;

    (void)trainingEpochDefault(f.model, 1, defaultLossConfig(MSE), f.trainDl, f.sgd,
                               keyRecordingGrads, REDUCTION_MEAN, 1, call);

    optimizerFunctions[SGD_M].step = g_savedSgdStep;
    freeKeyedFixture(&f);
}

/* Each update steps the optimizer under that update's key; the step counter
 * advances after the step, so the next update gets the next key. */
void testTrainingEpochStepsEachUpdateUnderItsUpdateKey(void) {
    trainingRng_t rng;
    trainingRngInit(&rng, KEYED_RUN_SEED);

    epochWithRecordedSteps(&(trainingCall_t){.rng = &rng});

    TEST_ASSERT_EQUAL_UINT32(2u, rng.step);
    TEST_ASSERT_EQUAL_size_t(2, g_updateKeyCount);
    trainingRng_t expected;
    trainingRngInit(&expected, KEYED_RUN_SEED);
    for (size_t u = 0; u < 2; u++) {
        TEST_ASSERT_TRUE(g_updateKeys[u].hasKey);
        assertSameKey(trainingRngUpdateKey(&expected), g_updateKeys[u].key);
        trainingRngAdvance(&expected);
    }
}

/* Without an rng every update steps without a key. */
void testTrainingEpochStepsWithoutAKeyWithoutAnRng(void) {
    epochWithRecordedSteps(NULL);

    TEST_ASSERT_EQUAL_size_t(2, g_updateKeyCount);
    TEST_ASSERT_FALSE(g_updateKeys[0].hasKey);
    TEST_ASSERT_FALSE(g_updateKeys[1].hasKey);
}

/* A Linear whose forward rounds stochastically into a SYM_INT32 output,
 * evaluated through the public batch entry point, which passes no call. */
static void evaluateAnSrForwardWithoutAKey(void) {
    tensor_t *wParam = buildFloatTensor2D(2, 3, (float[]){1.f, 1.f, 1.f, 1.f, 1.f, 1.f}, 6);
    parameter_t *w = parameterInit(wParam, gradInitFloat(wParam, NULL));
    tensor_t *bParam = buildFloatTensor2D(1, 2, (float[]){-1.f, 3.f}, 2);
    parameter_t *b = parameterInit(bParam, gradInitFloat(bParam, NULL));
    quantization_t floatQ;
    initFloat32Quantization(&floatQ);
    layer_t *linear = buildBorrowedLinearLayer(w, b, &floatQ);
    linear->config->linear->forwardMath =
        (arithmetic_t){.type = ARITH_FLOAT32, .roundingMode = SR_HALF_AWAY};
    linear->config->linear->outputQ = quantizationInitSymInt32WithBits(HALF_AWAY, 8);
    layer_t *model[] = {linear};
    sample_t *s = reserveMemory(sizeof(sample_t));
    s->item = buildFloatTensor1D(3, (float[]){0.5f, 1.f, 2.f});
    s->label = buildFloatTensor1D(2, (float[]){10.f, 20.f});
    sample_t *samples[] = {s};
    batch_t batch = {.samples = samples, .size = 1};
    (void)evaluationBatch(model, 1, MSE, &batch, inferenceWithLoss, REDUCTION_MEAN);
}

/* No key, no randomness: the public evaluation entry points are key-less, so
 * a model whose forward draws stops at its first draw. */
void testEvaluationOfADrawingForwardWithoutAKeyExits(void) {
    ASSERT_EXITS_WITH_OUTPUT(1, "roundByMode: SR rounding without a stream",
                             evaluateAnSrForwardWithoutAKey());
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(testTrainingBatchDefaultHandsCalculateGradsBatchAxisViews);
    RUN_TEST(testEvaluationBatchHandsInferenceBatchAxisViews);
    RUN_TEST(testEvaluationEpochWithMetricsHandsInferenceBatchAxisViews);
    RUN_TEST(testTrainingEpochDefaultMeanScaleSeesLabelBatchAxis);
    RUN_TEST(testCalculateGradsSequential_MatchesPyTorch);
    RUN_TEST(testEvaluationBatch_ReturnsAverageLoss);
    RUN_TEST(testEvaluationEpoch_ReturnsAverageLossAcrossBatches);
    RUN_TEST(testEvaluationEpoch_MinibatchMatchesMicrobatchAverage);
    RUN_TEST(testTrainingBatchDefault_ReturnsAverageLossAndAccumulatesGrads);
    RUN_TEST(testTrainingBatchDefault_SumAggregatesWithoutDivision);
    RUN_TEST(testTrainingEpochDefault_DoesOptimizerStepPerBatch);
    RUN_TEST(testTrainingEpochDefault_MinibatchStepsOncePerMinibatch);
    RUN_TEST(testTrainingEpochDefault_MeanScalesGradByOneOverNF);
    RUN_TEST(testTrainingEpochDefault_SumBackwardSkipsOptimizerScaling);
    RUN_TEST(testTrainingEpochDefault_MeanForwardSumBackward_MixedCombination);
    RUN_TEST(testTrainingRun_ReturnsResult);
    RUN_TEST(testEvaluationEpochWithMetrics_AllCorrect);
    RUN_TEST(testEvaluationEpochWithMetrics_SymOutputWire);
    RUN_TEST(testEvaluationEpochWithMetrics_BfpOutputWireDequantCompares);
    RUN_TEST(testEvaluationEpochWithMetrics_PartiallyCorrect);
    RUN_TEST(testEvaluationEpochWithMetrics_HandlesZeroPredictionClass);
    RUN_TEST(testEvaluationEpochWithReport_ReturnsConfusionMatrix);
    RUN_TEST(testTrainingRun_CallsCallbackEachEpochWithStats);
    RUN_TEST(testInferenceWithLoss_PropagatesForwardReductionSum);
    RUN_TEST(testEvaluationEpoch_FlatAggregator_DivisionByTotalSamples);
    RUN_TEST(testEvaluationEpoch_SumPath_ReturnsRawTotal);
    RUN_TEST(testTrainingRun_HardcodesForwardReductionMean);
    RUN_TEST(testTrainingRunStepsSchedulerOncePerEpoch);
    RUN_TEST(testTrainingRunNullSchedulerKeepsLrConstant);
    RUN_TEST(testTrainingRunRejectsSchedulerWiredToDifferentOptimizer);
    RUN_TEST(testTrainingRunCallbackObservesTheEpochsOwnLr);
    RUN_TEST(testTrainingRunZeroOptionsBehavesLikeNull);
    RUN_TEST(testTrainingRunStepsBsSchedulerOncePerEpochAndCallbackSeesTheEpochsBatch);
    RUN_TEST(testTrainingRunCallbackObservesTheEpochsOwnLrAndBatchWithBothSchedulers);
    RUN_TEST(testTrainingRunCompensatingBsSchedulerWritesTheLrEachEpoch);
    RUN_TEST(testTrainingEpochDefaultRejectsBatchLargerThanDataset);
    RUN_TEST(testTrainingRunRejectsBsSchedulerWiredToAnotherLoader);
    RUN_TEST(testTrainingRunRejectsBsCompensationWiredToAnotherOptimizer);
    RUN_TEST(testTrainingRunRejectsLrSchedulerNextToCompensatingBsScheduler);
    RUN_TEST(testTrainingRunShufflesEachEpochUnderItsOwnKey);
    RUN_TEST(testTrainingRunKeepsEpochZerosOrderWithoutPerEpochReshuffle);
    RUN_TEST(testTrainingRunShuffleFollowsTheEpochCounterAcrossCalls);
    RUN_TEST(testTrainingRunWithoutAnRngStopsAtAShuffledEvalLoader);
    RUN_TEST(testTrainingRunWithoutAnRngStopsAtAShuffledTrainLoader);
    RUN_TEST(testEvaluationEpochOfANeverShuffledLoaderExits);
    RUN_TEST(testTrainingRun_DefaultRunsAllEpochsThroughNonFiniteLoss);
    RUN_TEST(testTrainingRun_StopsOnNonFiniteLossWhenRequested);
    RUN_TEST(testTrainingRun_StopOnNonFiniteLoss_LastEpochStillCountsAsCompleted);
    RUN_TEST(testTrainingRun_StopOnNonFiniteLoss_DoesNotStepSchedulersOnTheStoppingEpoch);
    RUN_TEST(testTrainingRunTrainsOnTheOptionsScheduler);
    RUN_TEST(testTrainingRunExitsOnADeinitialisedOptionsScheduler);
    RUN_TEST(testTrainingRunHandsEveryEvalCallTheOptionsScheduler);
    RUN_TEST(testTrainingRunEvaluatesWithoutTheSchedulerAtAnotherEvalMicroBatch);
    RUN_TEST(testTrainingRunEvaluatesWithoutTheSchedulerOverATailChunk);
    RUN_TEST(testTrainingRunEvaluatesWithoutTheSchedulerBelowOneChunk);
    RUN_TEST(testTrainingRunEvaluatesOnTheSchedulerWhenEveryChunkHasTheTrainingRowCount);
    RUN_TEST(testTrainingRunExitsAtTheRaggedChunkOfAStreamShorterThanItsNominalCount);
    RUN_TEST(testTrainingRngInitSetsTheRootAndClearsTheCounters);
    RUN_TEST(testTrainingRngStepKeyIsTheStepFoldOfTheRoot);
    RUN_TEST(testTrainingRngCallKeyIsTheCallFoldOfItsParent);
    RUN_TEST(testTrainingRngUpdateKeyIsTheUpdateFoldOfTheRoot);
    RUN_TEST(testTrainingRngEvalKeyIsTheEvalFoldOfTheRoot);
    RUN_TEST(testTrainingRngAdvanceCountsUpdatesAndAdvanceEpochCountsEpochs);
    RUN_TEST(testTrainingRngReachesTheLastStepEpochAndCallIndexWithoutExiting);
    RUN_TEST(testTrainingRngHelpersExitOnANullState);
    RUN_TEST(testTrainingRngAdvanceExitsWhenTheStepCounterIsExhausted);
    RUN_TEST(testTrainingRngAdvanceEpochExitsWhenTheEpochCounterIsExhausted);
#if SIZE_MAX > UINT32_MAX
    RUN_TEST(testTrainingRngCallKeyExitsAboveUint32Max);
#endif
    RUN_TEST(testTrainingRngShuffleKeyFollowsTheEpochOnlyWhenPerEpoch);
    RUN_TEST(testTrainingRngShuffleEvalKeyIsOneKeyPerRun);
    RUN_TEST(testTrainingRngInitKeyIsTheInitFoldOfTheLayerIndex);
    RUN_TEST(testTrainingRngShuffleKeyExitsOnANullState);
    RUN_TEST(testTrainingRngShuffleEvalKeyExitsOnANullState);
    RUN_TEST(testTrainingRngInitKeyExitsOnANullState);
#if SIZE_MAX > UINT32_MAX
    RUN_TEST(testTrainingRngInitKeyExitsAboveUint32Max);
#endif
    RUN_TEST(testTrainingRngSerializeWritesTheOdtkRecord);
    RUN_TEST(testTrainingRngDeserializeReadsTheOdtkRecord);
    RUN_TEST(testTrainingRngDeserializeRestoresWhatSerializeSaved);
    RUN_TEST(testTrainingRngDeserializeExitsOnABadMagic);
    RUN_TEST(testTrainingRngDeserializeExitsOnAnUnsupportedVersion);
    RUN_TEST(testTrainingRngDeserializeExitsOnATruncatedRecord);
    RUN_TEST(testTrainingRngSerializeExitsOnANullState);
    RUN_TEST(testTrainingRngDeserializeExitsOnANullState);
    RUN_TEST(testTrainingCallForIndexKeepsANullCallNull);
    RUN_TEST(testTrainingCallForIndexCopiesACallWithoutAKey);
    RUN_TEST(testTrainingCallForIndexDerivesTheCallKeyUnderTheParentKey);
    RUN_TEST(testTrainingCallForIndexDerivesTheKeyInPlace);
    RUN_TEST(testTrainingRunHandsEachTrainingCallTheCallKeyOfItsUpdate);
    RUN_TEST(testTrainingRunCountsTheEpochBeforeTheCallback);
    RUN_TEST(testTrainingRunNumbersStackedTrainingCallsByChunk);
    RUN_TEST(testTrainingRunKeysEachEvaluationCallByItsPositionInTheEpoch);
    RUN_TEST(testTrainingRunNumbersStackedEvaluationCallsByChunk);
    RUN_TEST(testTrainingRunWithoutAnRngHandsNoKeys);
    RUN_TEST(testTrainingRunEvaluatesEachEpochUnderItsOwnEvalKey);
    RUN_TEST(testTrainingEpochDoesNotHandItsOwnKeyDown);
    RUN_TEST(testTrainingEpochStepsEachUpdateUnderItsUpdateKey);
    RUN_TEST(testTrainingEpochStepsWithoutAKeyWithoutAnRng);
    RUN_TEST(testEvaluationOfADrawingForwardWithoutAKeyExits);
    return UNITY_END();
}
