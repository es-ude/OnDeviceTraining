#define SOURCE_FILE "UNIT_TEST_TRAINING_LOOP_API"

#include <stddef.h>
#include <stdint.h>

#include "ArithmeticType.h"
#include "BorrowedLayer.h"
#include "CalculateGradsSequential.h"
#include "DataLoaderApi.h"
#include "Dataset.h"
#include "DeathTest.h"
#include "InferenceApi.h"
#include "Linear.h"
#include "LinearApi.h"
#include "LossFunction.h"
#include "LrScheduler.h"
#include "OptimizerApi.h"
#include "QuantizationApi.h"
#include "ReluApi.h"
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

/* #327 contract: trainingRun takes the NULL-able scheduler directly after
 * the optimizer (arg 7). Compile-time pin. */
_Static_assert(_Generic(&trainingRun,
                   trainingRunResult_t (*)(layer_t **, size_t, lossConfig_t, dataLoader_t *,
                                           dataLoader_t *, optimizer_t *, lrScheduler_t *, size_t,
                                           calculateGradsFn_t, inferenceWithLossFn_t,
                                           epochCallbackFn_t): 1,
                   default: 0),
               "#327: trainingRun must be (model, modelSize, lossConfig, trainDl, evalDl, "
               "optimizer, scheduler, numberOfEpochs, calculateGradsFn, inferenceFn, callback)");

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
            REDUCTION_SUM, input0, label0);
        trainingStats_t *ts1 = calculateGradsSequential(
            model, sizeModel, (lossConfig_t){.funcType = MSE, .backwardReduction = REDUCTION_SUM},
            REDUCTION_SUM, input1, label1);
        trainingStats_t *ts2 = calculateGradsSequential(
            model, sizeModel, (lossConfig_t){.funcType = MSE, .backwardReduction = REDUCTION_SUM},
            REDUCTION_SUM, input2, label2);

        /* PyTorch reference is MSE-mean-of-features (`2/F * (o-l)` per element). Post-#135
         * backward writes raw `2*(o-l)`; recover the reference trajectory via explicit
         * 1/F = 1/2 scaling here. trainingEpochDefault does this implicitly when
         * backwardReduction == REDUCTION_MEAN; this test exercises calculateGradsSequential
         * directly, so the scaling must be manual. */
        scaleOptimizerGradients(sgd, 1.0f / 2.0f);

        sgdFns.step(sgd);
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

    /* Create 2 samples manually */
    tensor_t *input0 = buildFloatTensor2D(1, 3, (float[]){0.f, 1.f, 2.f}, 3);
    tensor_t *label0 = buildFloatTensor2D(1, 2, (float[]){10.f, 20.f}, 2);
    tensor_t *input1 = buildFloatTensor2D(1, 3, (float[]){1.f, 0.f, 0.f}, 3);
    tensor_t *label1 = buildFloatTensor2D(1, 2, (float[]){5.f, 5.f}, 2);

    /* Compute expected losses via inferenceWithLoss directly.
     * evaluationBatch now returns a pure sum (no division); expected = sum of
     * per-sample MEAN losses. */
    inferenceStats_t *stats0 = inferenceWithLoss(model, 1, input0, label0, MSE, REDUCTION_MEAN);
    inferenceStats_t *stats1 = inferenceWithLoss(model, 1, input1, label1, MSE, REDUCTION_MEAN);
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
        epochItems[i] = buildFloatTensor2D(1, 2, epochItemDataLiteral[i], 2);
        epochLabels[i] = buildFloatTensor2D(1, 2, epochLabelDataLiteral[i], 2);
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
        dataLoaderInit(getEpochSample, getEpochDatasetSize, 1, NULL, NULL, false, 0, true);

    float totalAvg = evaluationEpoch(model, 1, MSE, dl, inferenceWithLoss, REDUCTION_MEAN);

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
        dataLoaderInit(getEpochSample, getEpochDatasetSize, 2, NULL, NULL, false, 0, true);

    float totalAvg = evaluationEpoch(model, 1, MSE, dl, inferenceWithLoss, REDUCTION_MEAN);

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

    /* Compute expected: run calculateGradsSequential manually per sample */
    tensor_t *in0 = buildFloatTensor2D(1, 3, (float[]){-4.f, 1.f, 9.f}, 3);
    tensor_t *lb0 = buildFloatTensor2D(1, 2, (float[]){59.f, -23.f}, 2);
    tensor_t *in1 = buildFloatTensor2D(1, 3, (float[]){5.f, -1.f, 2.f}, 3);
    tensor_t *lb1 = buildFloatTensor2D(1, 2, (float[]){43.f, 249.f}, 2);

    /* Get expected losses from individual calculateGrads calls.
     * Use REDUCTION_MEAN + batchSize=2 to match what trainingBatchDefault threads
     * through when called with forwardReduction=REDUCTION_MEAN. */
    trainingStats_t *ts0 = calculateGradsSequential(
        model, 1, (lossConfig_t){.funcType = MSE, .backwardReduction = REDUCTION_SUM},
        REDUCTION_MEAN, in0, lb0);
    trainingStats_t *ts1 = calculateGradsSequential(
        model, 1, (lossConfig_t){.funcType = MSE, .backwardReduction = REDUCTION_SUM},
        REDUCTION_MEAN, in1, lb1);
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
        calculateGradsSequential, REDUCTION_MEAN);

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

    tensor_t *in0 = buildFloatTensor2D(1, 3, (float[]){-4.f, 1.f, 9.f}, 3);
    tensor_t *lb0 = buildFloatTensor2D(1, 2, (float[]){59.f, -23.f}, 2);
    tensor_t *in1 = buildFloatTensor2D(1, 3, (float[]){5.f, -1.f, 2.f}, 3);
    tensor_t *lb1 = buildFloatTensor2D(1, 2, (float[]){43.f, 249.f}, 2);

    trainingStats_t *ts0 = calculateGradsSequential(
        model, 1, (lossConfig_t){.funcType = MSE, .backwardReduction = REDUCTION_SUM},
        REDUCTION_SUM, in0, lb0);
    trainingStats_t *ts1 = calculateGradsSequential(
        model, 1, (lossConfig_t){.funcType = MSE, .backwardReduction = REDUCTION_SUM},
        REDUCTION_SUM, in1, lb1);
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
        calculateGradsSequential, REDUCTION_SUM);

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
        dataLoaderInit(getEpochSample, getEpochDatasetSize, 1, NULL, NULL, false, 0, true);

    float epochLoss = trainingEpochDefault(
        model, sizeModel, (lossConfig_t){.funcType = MSE, .backwardReduction = REDUCTION_SUM}, dl,
        sgd, calculateGradsSequential, REDUCTION_SUM);

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
        dataLoaderInit(getEpochSample, getEpochDatasetSize, 2, NULL, NULL, false, 0, true);

    float epochLoss = trainingEpochDefault(
        model, sizeModel, (lossConfig_t){.funcType = MSE, .backwardReduction = REDUCTION_SUM}, dl,
        sgd, calculateGradsSequential, REDUCTION_SUM);

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
        dataLoaderInit(getEpochSample, getEpochDatasetSize, 1, NULL, NULL, false, 0, true);
    dataLoader_t *evalDl =
        dataLoaderInit(getEpochSample, getEpochDatasetSize, 1, NULL, NULL, false, 0, true);

    trainingRunResult_t result = trainingRun(
        model, 1, (lossConfig_t){.funcType = MSE, .backwardReduction = REDUCTION_SUM}, trainDl,
        evalDl, sgd, NULL, 2, calculateGradsSequential, inferenceWithLoss, NULL);

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

static void captureCallback(size_t epoch, float trainLoss, epochStats_t evalStats) {
    cbCallCount++;
    cbLastEpoch = epoch;
    cbLastTrainLoss = trainLoss;
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
        dataLoaderInit(getEpochSample, getEpochDatasetSize, 1, NULL, NULL, false, 0, true);
    dataLoader_t *evalDl =
        dataLoaderInit(getEpochSample, getEpochDatasetSize, 1, NULL, NULL, false, 0, true);

    size_t numberOfEpochs = 3;
    trainingRunResult_t result =
        trainingRun(model, 1, (lossConfig_t){.funcType = MSE, .backwardReduction = REDUCTION_SUM},
                    trainDl, evalDl, sgd, NULL, numberOfEpochs, calculateGradsSequential,
                    inferenceWithLoss, captureCallback);

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
    symMetricsItem = buildFloatTensor2D(1, 2, (float[]){-1.f, -0.5f}, 2);
    symMetricsLabel = buildFloatTensor2D(1, 2, (float[]){0.f, 1.f}, 2);
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

    dataLoader_t *dl = dataLoaderInit(getSymMetricsSample, getSymMetricsDatasetSize, 1, NULL, NULL,
                                      false, 0, true);

    epochStats_t stats =
        evaluationEpochWithMetrics(model, 1, MSE, dl, inferenceWithLoss, REDUCTION_MEAN);

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
        dataLoaderInit(getEpochSample, getEpochDatasetSize, 1, NULL, NULL, false, 0, true);

    /* Identity model: all 4 samples predict correctly (same as testEvaluationEpochAccuracy) */
    epochStats_t stats =
        evaluationEpochWithMetrics(model, 1, MSE, dl, inferenceWithLoss, REDUCTION_MEAN);

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
    singleSampleItem = buildFloatTensor2D(1, 4, (float[]){1.f, 1.f, 1.f, 1.f}, 4);
    singleSampleLabel = buildFloatTensor2D(1, 4, (float[]){0.f, 0.f, 0.f, 0.f}, 4);
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
        dataLoaderInit(getSingleSample, getSingleSampleDatasetSize, 1, NULL, NULL, false, 0, true);

    /* defaultLossConfig: backwardReduction = REDUCTION_MEAN. */
    lossConfig_t cfg = defaultLossConfig(MSE);

    trainingEpochDefault(model, 1, cfg, dl, sgd, calculateGradsSequential, REDUCTION_MEAN);

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
        partialItems[i] = buildFloatTensor2D(1, 2, partialItemDataLiteral[i], 2);
        partialLabels[i] = buildFloatTensor2D(1, 2, partialLabelDataLiteral[i], 2);
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
        dataLoaderInit(getPartialSample, getPartialDatasetSize, 1, NULL, NULL, false, 0, true);

    epochStats_t stats =
        evaluationEpochWithMetrics(model, 1, MSE, dl, inferenceWithLoss, REDUCTION_MEAN);

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
        zeroPredItems[i] = buildFloatTensor2D(1, 2, zeroPredItemDataLiteral[i], 2);
        zeroPredLabels[i] = buildFloatTensor2D(1, 2, zeroPredLabelDataLiteral[i], 2);
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
        dataLoaderInit(getZeroPredSample, getZeroPredDatasetSize, 1, NULL, NULL, false, 0, true);

    epochStats_t stats =
        evaluationEpochWithMetrics(model, 1, MSE, dl, inferenceWithLoss, REDUCTION_MEAN);

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
        dataLoaderInit(getPartialSample, getPartialDatasetSize, 1, NULL, NULL, false, 0, true);

    /* Pre-fill with non-zero to verify WithReport zeroes the caller's buffer before accumulating */
    size_t cm[2 * 2] = {99, 99, 99, 99};
    classificationReport_t report =
        evaluationEpochWithReport(model, 1, MSE, dl, inferenceWithLoss, cm, 2, REDUCTION_MEAN);

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
    inferenceStats_t *sumStats = inferenceWithLoss(model, 1, input, label, MSE, REDUCTION_SUM);
    inferenceStats_t *meanStats = inferenceWithLoss(model, 1, input, label, MSE, REDUCTION_MEAN);

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
        dataLoaderInit(getEpochSample, getEpochDatasetSize, 1, NULL, NULL, false, 0, true);

    /* SUM backwardReduction skips scaleOptimizerGradients. */
    lossConfig_t cfg = defaultLossConfig(MSE);
    cfg.backwardReduction = REDUCTION_SUM;

    float epochLoss =
        trainingEpochDefault(model, 1, cfg, dl, sgd, calculateGradsSequential, REDUCTION_SUM);

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
        dataLoaderInit(getEpochSample, getEpochDatasetSize, 1, NULL, NULL, false, 0, true);

    lossConfig_t cfg = defaultLossConfig(MSE);
    cfg.backwardReduction = REDUCTION_SUM; /* SUM gradients, no optimizer scaling */

    /* forwardReduction = MEAN: stats->loss is per-sample mean, comparable. */
    float epochLoss =
        trainingEpochDefault(model, 1, cfg, dl, sgd, calculateGradsSequential, REDUCTION_MEAN);

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
        dataLoaderInit(getEpochSample, getEpochDatasetSize, 2, NULL, NULL, false, 0, true);

    /* MEAN: divides by totalSamples. */
    float meanLoss = evaluationEpoch(model, 1, MSE, dl, inferenceWithLoss, REDUCTION_MEAN);

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
        dataLoaderInit(getEpochSample, getEpochDatasetSize, 1, NULL, NULL, false, 0, true);

    /* SUM forwardReduction: returns raw totalLoss = sum of per-sample
     * SUM losses. Per-sample SUM: 17, 17, 5, 5 (= per-sample MEAN * F=2). */
    float sumLoss = evaluationEpoch(model, 1, MSE, dl, inferenceWithLoss, REDUCTION_SUM);

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
        dataLoaderInit(getEpochSample, getEpochDatasetSize, 1, NULL, NULL, false, 0, true);
    dataLoader_t *evalDl =
        dataLoaderInit(getEpochSample, getEpochDatasetSize, 1, NULL, NULL, false, 0, true);

    /* trainingRun hardcodes forwardReduction = MEAN for train/eval comparability.
     * This test uses defaultLossConfig (backwardReduction = MEAN) to act as a
     * regression barrier: if forwardReduction were inadvertently set to SUM,
     * eval loss would be F× larger than train loss, failing the assertion below. */
    lossConfig_t cfg = defaultLossConfig(MSE);

    trainingRunResult_t result = trainingRun(model, 1, cfg, trainDl, evalDl, sgd, NULL, 2,
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

static void captureLrCallback(size_t epoch, float trainLoss, epochStats_t evalStats) {
    (void)epoch;
    (void)trainLoss;
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
        dataLoaderInit(getEpochSample, getEpochDatasetSize, 1, NULL, NULL, false, 0, true);
    dataLoader_t *evalDl =
        dataLoaderInit(getEpochSample, getEpochDatasetSize, 1, NULL, NULL, false, 0, true);

    lrScheduler_t sched;
    stepLrInit(&sched, sgd, 1, 0.5f); /* halve every epoch */
    trainingRunResult_t result = trainingRun(
        model, 1, (lossConfig_t){.funcType = MSE, .backwardReduction = REDUCTION_SUM}, trainDl,
        evalDl, sgd, &sched, 2, calculateGradsSequential, inferenceWithLoss, NULL);
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
        dataLoaderInit(getEpochSample, getEpochDatasetSize, 1, NULL, NULL, false, 0, true);
    dataLoader_t *evalDl =
        dataLoaderInit(getEpochSample, getEpochDatasetSize, 1, NULL, NULL, false, 0, true);

    float lrBefore = optimizerFunctions[SGD_M].getLr(sgd);
    trainingRunResult_t result = trainingRun(
        model, 1, (lossConfig_t){.funcType = MSE, .backwardReduction = REDUCTION_SUM}, trainDl,
        evalDl, sgd, NULL, 2, calculateGradsSequential, inferenceWithLoss, NULL);
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
        dataLoaderInit(getEpochSample, getEpochDatasetSize, 1, NULL, NULL, false, 0, true);
    dataLoader_t *evalDl =
        dataLoaderInit(getEpochSample, getEpochDatasetSize, 1, NULL, NULL, false, 0, true);

    /* A second, unrelated optimizer the scheduler is (wrongly) wired to. */
    sgd_t otherSgd;
    sgdInit(&otherSgd, 0.1f, 0.0f, 0.0f,
            (arithmetic_t){.type = ARITH_FLOAT32, .roundingMode = HALF_AWAY});
    optimImpl_t otherImpl = {.sgd = &otherSgd};
    optimizer_t otherOptim = {
        .type = SGD_M, .impl = &otherImpl, .parameter = NULL, .states = NULL, .sizeStates = 0};
    lrScheduler_t sched;
    stepLrInit(&sched, &otherOptim, 1, 0.5f);
    ASSERT_EXITS_WITH_FAILURE(trainingRun(
        model, 1, (lossConfig_t){.funcType = MSE, .backwardReduction = REDUCTION_SUM}, trainDl,
        evalDl, sgd, &sched, 1, calculateGradsSequential, inferenceWithLoss, NULL));

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
        dataLoaderInit(getEpochSample, getEpochDatasetSize, 1, NULL, NULL, false, 0, true);
    dataLoader_t *evalDl =
        dataLoaderInit(getEpochSample, getEpochDatasetSize, 1, NULL, NULL, false, 0, true);

    g_lrCaptureCount = 0;
    g_capturedOptim = sgd;
    lrScheduler_t sched;
    stepLrInit(&sched, sgd, 1, 0.5f);
    trainingRunResult_t result = trainingRun(
        model, 1, (lossConfig_t){.funcType = MSE, .backwardReduction = REDUCTION_SUM}, trainDl,
        evalDl, sgd, &sched, 3, calculateGradsSequential, inferenceWithLoss, captureLrCallback);
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

/* #381: trainingRun's epoch loop must gate dataLoaderReshuffle() on the
 * TRAIN loader's reshufflePerEpoch flag, calling it only for epoch > 0 (epoch
 * 0 already got its permutation from dataLoaderInit's init-shuffle), and must
 * NEVER touch the eval loader — even when the eval loader's own flag is set,
 * which would catch an accidental reshuffle-both regression. */
void testTrainingRunReshuffleFlagOffKeepsTrainIndices(void) {
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
    /* shuffle=true so the loader actually owns a live permutation to protect;
     * reshufflePerEpoch stays at its default (false) — not set here. */
    dataLoader_t *trainDl =
        dataLoaderInit(getEpochSample, getEpochDatasetSize, 1, NULL, NULL, true, 123, true);
    dataLoader_t *evalDl =
        dataLoaderInit(getEpochSample, getEpochDatasetSize, 1, NULL, NULL, false, 0, true);

    size_t snapshot[4];
    for (size_t i = 0; i < 4; i++) {
        snapshot[i] = trainDl->indices[i];
    }

    trainingRunResult_t result = trainingRun(
        model, 1, (lossConfig_t){.funcType = MSE, .backwardReduction = REDUCTION_SUM}, trainDl,
        evalDl, sgd, NULL, 2, calculateGradsSequential, inferenceWithLoss, NULL);
    (void)result;

    /* CAPTURE. */
    size_t capturedIndices[4];
    for (size_t i = 0; i < 4; i++) {
        capturedIndices[i] = trainDl->indices[i];
    }

    /* FREE. */
    freeOptim(sgd);
    freeQuantization(momentumQ);
    freeDataLoader(evalDl);
    freeDataLoader(trainDl);
    freeLinearLayerShellOnly(linear);
    freeEpochDataset();

    /* ASSERT. */
    TEST_ASSERT_EQUAL_size_t_ARRAY(snapshot, capturedIndices, 4);
}

void testTrainingRunReshuffleFlagOnChangesTrainButNotEvalIndices(void) {
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
        dataLoaderInit(getEpochSample, getEpochDatasetSize, 1, NULL, NULL, true, 123, true);
    /* eval loader ALSO flagged on: proves trainingRun never reshuffles eval,
     * not merely that it happens to leave a never-flagged loader alone. */
    dataLoader_t *evalDl =
        dataLoaderInit(getEpochSample, getEpochDatasetSize, 1, NULL, NULL, true, 456, true);
    dataLoaderSetReshufflePerEpoch(trainDl, true);
    dataLoaderSetReshufflePerEpoch(evalDl, true);

    size_t trainSnapshot[4];
    size_t evalSnapshot[4];
    for (size_t i = 0; i < 4; i++) {
        trainSnapshot[i] = trainDl->indices[i];
        evalSnapshot[i] = evalDl->indices[i];
    }

    trainingRunResult_t result = trainingRun(
        model, 1, (lossConfig_t){.funcType = MSE, .backwardReduction = REDUCTION_SUM}, trainDl,
        evalDl, sgd, NULL, 2, calculateGradsSequential, inferenceWithLoss, NULL);
    (void)result;

    /* CAPTURE. */
    size_t capturedTrainIndices[4];
    size_t capturedEvalIndices[4];
    for (size_t i = 0; i < 4; i++) {
        capturedTrainIndices[i] = trainDl->indices[i];
        capturedEvalIndices[i] = evalDl->indices[i];
    }

    /* FREE. */
    freeOptim(sgd);
    freeQuantization(momentumQ);
    freeDataLoader(evalDl);
    freeDataLoader(trainDl);
    freeLinearLayerShellOnly(linear);
    freeEpochDataset();

    /* ASSERT. */
    bool trainChanged = false;
    for (size_t i = 0; i < 4; i++) {
        if (capturedTrainIndices[i] != trainSnapshot[i]) {
            trainChanged = true;
            break;
        }
    }
    TEST_ASSERT_TRUE(trainChanged);
    TEST_ASSERT_EQUAL_size_t_ARRAY(evalSnapshot, capturedEvalIndices, 4);
}

/* Pins down the "epoch 0 uses the init shuffle" half of the #381 contract
 * specifically: a single-epoch run must NEVER call dataLoaderReshuffle (the
 * two tests above only prove the flag gates something across >=2 epochs;
 * they'd stay green even if trainingRun mistakenly reshuffled epoch 0 too,
 * since "changed at all" doesn't pin down WHICH epoch boundary caused it). */
void testTrainingRunReshuffleFlagOn_SingleEpochNeverReshuffles(void) {
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
        dataLoaderInit(getEpochSample, getEpochDatasetSize, 1, NULL, NULL, true, 123, true);
    dataLoader_t *evalDl =
        dataLoaderInit(getEpochSample, getEpochDatasetSize, 1, NULL, NULL, false, 0, true);
    dataLoaderSetReshufflePerEpoch(trainDl, true);

    size_t snapshot[4];
    for (size_t i = 0; i < 4; i++) {
        snapshot[i] = trainDl->indices[i];
    }

    trainingRunResult_t result = trainingRun(
        model, 1, (lossConfig_t){.funcType = MSE, .backwardReduction = REDUCTION_SUM}, trainDl,
        evalDl, sgd, NULL, 1, calculateGradsSequential, inferenceWithLoss, NULL);
    (void)result;

    /* CAPTURE. */
    size_t capturedIndices[4];
    for (size_t i = 0; i < 4; i++) {
        capturedIndices[i] = trainDl->indices[i];
    }

    /* FREE. */
    freeOptim(sgd);
    freeQuantization(momentumQ);
    freeDataLoader(evalDl);
    freeDataLoader(trainDl);
    freeLinearLayerShellOnly(linear);
    freeEpochDataset();

    /* ASSERT. */
    TEST_ASSERT_EQUAL_size_t_ARRAY(snapshot, capturedIndices, 4);
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
                                                reduction_t forwardReduction) {
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
    bfpMetricsItem = buildFloatTensor2D(1, 2, (float[]){1.f, 1.f}, 2);
    bfpMetricsLabel = buildFloatTensor2D(1, 2, (float[]){0.f, 1.f}, 2);

    dataLoader_t *dl = dataLoaderInit(getBfpMetricsSample, getBfpMetricsDatasetSize, 1, NULL, NULL,
                                      false, 0, true);

    epochStats_t stats =
        evaluationEpochWithMetrics(NULL, 0, MSE, dl, bfpOutputWireInference, REDUCTION_MEAN);

    float capturedAccuracy = stats.accuracy;

    freeDataLoader(dl);
    freeTensor(bfpMetricsLabel);
    freeTensor(bfpMetricsItem);

    TEST_ASSERT_FLOAT_WITHIN_MESSAGE(0.001f, 1.0f, capturedAccuracy,
                                     "BFP argmax must compare DEQUANTIZED values: class 1 "
                                     "(2.09375) wins on value, class 0 wins on raw mantissa");
}

void setUp() {}
void tearDown() {}

int main(void) {
    UNITY_BEGIN();
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
    RUN_TEST(testTrainingRunReshuffleFlagOffKeepsTrainIndices);
    RUN_TEST(testTrainingRunReshuffleFlagOnChangesTrainButNotEvalIndices);
    RUN_TEST(testTrainingRunReshuffleFlagOn_SingleEpochNeverReshuffles);
    return UNITY_END();
}
