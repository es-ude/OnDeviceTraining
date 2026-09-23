#define SOURCE_FILE "UNIT_TEST_MNIST_SMOKE"

#include <stddef.h>
#include <stdio.h>
#include <time.h>

#include "unity.h"

#include "BorrowedLayer.h"
#include "CalculateGradsSequential.h"
#include "DataLoaderApi.h"
#include "Dataset.h"
#include "InferenceApi.h"
#include "LayerQuant.h"
#include "LinearApi.h"
#include "LossFunction.h"
#include "OptimizerApi.h"
#include "QuantizationApi.h"
#include "ReluApi.h"
#include "SgdApi.h"
#include "SoftmaxApi.h"
#include "StorageApi.h"
#include "Tensor.h"
#include "TensorApi.h"
#include "TrainingLoopApi.h"

void setUp() {}
void tearDown() {}

#define INPUT_DIM 4
#define HIDDEN_DIM 8
#define NUM_CLASSES 3
#define DATASET_SIZE 6
#define MODEL_SIZE 4

/* Tiny trivially-learnable dataset: each input is a noisy one-hot prototype of
 * its class. The arrays are file-scope because the getSample callback (fired
 * during trainingRun) must reach them; freeDataset releases them at end. */
static tensor_t *items[DATASET_SIZE];
static tensor_t *labels[DATASET_SIZE];

static const float itemDataLiteral[DATASET_SIZE][INPUT_DIM] = {
    {1.0f, 0.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 1.0f, 0.0f},
    {0.9f, 0.1f, 0.0f, 0.0f}, {0.1f, 0.9f, 0.0f, 0.0f}, {0.0f, 0.1f, 0.9f, 0.0f},
};
static const float labelDataLiteral[DATASET_SIZE][NUM_CLASSES] = {
    {1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f, 1.0f},
    {1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f, 1.0f},
};

static void initDataset() {
    for (size_t i = 0; i < DATASET_SIZE; i++) {
        /* Item tensor: natural shape (INPUT_DIM); trainingRun adds the batch axis. */
        size_t *itemDims = reserveMemory(sizeof(size_t));
        itemDims[0] = INPUT_DIM;
        size_t *itemOrder = reserveMemory(sizeof(size_t));
        setOrderOfDimsForNewTensor(1, itemOrder);
        shape_t *itemShape = reserveMemory(sizeof(shape_t));
        setShape(itemShape, itemDims, 1, itemOrder);
        items[i] = initTensor(itemShape, quantizationInitFloat(), NULL);
        tensorFillFromFloatBuffer(items[i], itemDataLiteral[i], INPUT_DIM);

        /* Label tensor: natural shape (NUM_CLASSES). */
        size_t *labelDims = reserveMemory(sizeof(size_t));
        labelDims[0] = NUM_CLASSES;
        size_t *labelOrder = reserveMemory(sizeof(size_t));
        setOrderOfDimsForNewTensor(1, labelOrder);
        shape_t *labelShape = reserveMemory(sizeof(shape_t));
        setShape(labelShape, labelDims, 1, labelOrder);
        labels[i] = initTensor(labelShape, quantizationInitFloat(), NULL);
        tensorFillFromFloatBuffer(labels[i], labelDataLiteral[i], NUM_CLASSES);
    }
}

static void freeDataset() {
    for (size_t i = 0; i < DATASET_SIZE; i++) {
        freeTensor(items[i]);
        freeTensor(labels[i]);
    }
}

static sample_t *getSample(size_t id) {
    sample_t *sample = reserveMemory(sizeof(sample_t));
    sample->item = items[id];
    sample->label = labels[id];
    return sample;
}

static size_t getDatasetSize() {
    return DATASET_SIZE;
}

/* Builds a 4-layer Linear -> ReLU -> Linear -> Softmax model. The shared
 * layer-level quantization q is exposed via q_out so the test can free it
 * after the model layers are freed. Each parameter tensor takes its own
 * fresh quantization (initTensor takes ownership), distinct from q. */
static void buildModel(layer_t **model, quantization_t **q_out) {
    quantization_t *q = quantizationInitFloat();
    *q_out = q;
    layerQuant_t lq;
    layerQuantInitUniform(&lq, q);

    /* XAVIER_UNIFORM weights (gain 1); bias defaults to the factory's PyTorch
     * uniform(+/- 1/sqrt(fan_in)) rather than the pre-migration ZEROS — this
     * test only asserts that training reduces the loss, not specific
     * parameter values, so the divergence is harmless. */
    model[0] = linearLayerInit(
        &(linearInit_t){.inFeatures = INPUT_DIM,
                        .outFeatures = HIDDEN_DIM,
                        .bias = BIAS_TRUE,
                        .weightInit = {.scheme = INIT_XAVIER_UNIFORM, .gain = 1.0f}},
        &lq);
    model[1] = reluLayerInit(&lq);

    model[2] = linearLayerInit(
        &(linearInit_t){.inFeatures = HIDDEN_DIM,
                        .outFeatures = NUM_CLASSES,
                        .bias = BIAS_TRUE,
                        .weightInit = {.scheme = INIT_XAVIER_UNIFORM, .gain = 1.0f}},
        &lq);
    model[3] = softmaxLayerInit(&lq);
}

static size_t cbInvocations;
static float firstTrainLoss;
static float lastTrainLoss;

static void captureEpoch(epochInfo_t info, epochStats_t evalStats) {
    (void)evalStats;
    if (cbInvocations == 0) {
        firstTrainLoss = info.trainLoss;
    }
    lastTrainLoss = info.trainLoss;
    cbInvocations++;
}

/* End-to-end smoke test: runs the same pipeline as MnistExperiment at miniature
 * scale. Catches regressions in the Linear->ReLU->Linear->Softmax + CrossEntropy
 * path, trainingRun()'s epoch/batch orchestration, and the epochCallback
 * signature. Runs in well under a second and is suitable for CI.
 */
void testMnistSmoke_FullTrainingPipelineReducesLoss() {
    initDataset();

    dataLoader_t *trainDl =
        dataLoaderInit(getSample, getDatasetSize, 2, NULL, NULL, false, 0, true);
    dataLoader_t *evalDl = dataLoaderInit(getSample, getDatasetSize, 1, NULL, NULL, false, 0, true);

    layer_t *model[MODEL_SIZE];
    quantization_t *q = NULL;
    buildModel(model, &q);

    quantization_t *momentumQ = quantizationInitFloat();
    optimizer_t *sgd =
        sgdMCreateOptim(0.1f, 0.0f, 0.0f, model, MODEL_SIZE, momentumQ,
                        (arithmetic_t){.type = ARITH_FLOAT32, .roundingMode = HALF_AWAY});

    cbInvocations = 0;
    size_t numberOfEpochs = 20;
    trainingRunResult_t result =
        trainingRun(model, MODEL_SIZE,
                    (lossConfig_t){.funcType = CROSS_ENTROPY, .backwardReduction = REDUCTION_MEAN},
                    trainDl, evalDl, sgd, numberOfEpochs, calculateGradsSequential,
                    inferenceWithLoss, &(trainingRunOptions_t){.callback = captureEpoch});

    /* CAPTURE all assertion values into stack locals BEFORE any free. */
    size_t capturedCbInvocations = cbInvocations;
    float capturedFinalTrainLoss = result.finalTrainLoss;
    float capturedAccuracy = result.finalEvalStats.accuracy;
    float capturedFirstTrainLoss = firstTrainLoss;
    float capturedLastTrainLoss = lastTrainLoss;

    /* FREE in reverse-init order.
     * NOTE: freeOptim cascades to all model parameters via freeParameter.
     * Do NOT also call freeParameter on w0/b0/w1/b1 — would be a double-free. */
    freeOptim(sgd);
    freeSoftmaxLayer(model[3]);
    freeLinearLayerShellOnly(model[2]);
    freeReluLayer(model[1]);
    freeLinearLayerShellOnly(model[0]);
    freeDataLoader(evalDl);
    freeDataLoader(trainDl);
    freeQuantization(momentumQ);
    freeQuantization(q);
    freeDataset();

    /* ASSERT on captured. */
    TEST_ASSERT_EQUAL_UINT(numberOfEpochs, capturedCbInvocations);
    TEST_ASSERT_TRUE(capturedFinalTrainLoss > 0.0f);
    TEST_ASSERT_TRUE(capturedAccuracy >= 0.0f);
    TEST_ASSERT_TRUE(capturedAccuracy <= 1.0f);
    TEST_ASSERT_TRUE(capturedLastTrainLoss < capturedFirstTrainLoss);
}

/* Regression test for #94. The original audit observed a silent exit(1) on
 * macOS when an snprintf or gmtime_r call sat between init() and trainingRun()
 * in the base project; #94 hypothesised an uninitialized-read or sized-buffer
 * overrun in a static-init path whose visible symptom was gated on stack/dyld
 * layout. This test mirrors the reproducer pattern (full FLOAT32 setup,
 * gmtime_r + snprintf, then trainingRun) on the in-tree smoke pipeline. If the
 * underlying memory bug is still present, trainingRun terminates with exit(1)
 * before reaching any assertion; reaching the final TEST_ASSERT_TRUE proves
 * the run completed normally. */
void testMnistSmoke_SnprintfGmtimeRBetweenSetupAndTrainingRun_NoSilentExit() {
    initDataset();

    dataLoader_t *trainDl =
        dataLoaderInit(getSample, getDatasetSize, 2, NULL, NULL, false, 0, true);
    dataLoader_t *evalDl = dataLoaderInit(getSample, getDatasetSize, 1, NULL, NULL, false, 0, true);

    layer_t *model[MODEL_SIZE];
    quantization_t *q = NULL;
    buildModel(model, &q);

    quantization_t *momentumQ = quantizationInitFloat();
    optimizer_t *sgd =
        sgdMCreateOptim(0.1f, 0.0f, 0.0f, model, MODEL_SIZE, momentumQ,
                        (arithmetic_t){.type = ARITH_FLOAT32, .roundingMode = HALF_AWAY});

    /* The poison block from #94's reproducer: gmtime_r + snprintf wedged
     * between setup and trainingRun. Pre-fix this triggered silent exit(1). */
    char buf[64];
    time_t t = time(NULL);
    struct tm tmStruct;
    gmtime_r(&t, &tmStruct);
    snprintf(buf, sizeof(buf), "%04d-%02d-%02d", tmStruct.tm_year + 1900, tmStruct.tm_mon + 1,
             tmStruct.tm_mday);

    cbInvocations = 0;
    trainingRunResult_t result =
        trainingRun(model, MODEL_SIZE,
                    (lossConfig_t){.funcType = CROSS_ENTROPY, .backwardReduction = REDUCTION_MEAN},
                    trainDl, evalDl, sgd, 1, calculateGradsSequential, inferenceWithLoss,
                    &(trainingRunOptions_t){.callback = captureEpoch});

    /* CAPTURE before free. */
    size_t capturedCbInvocations = cbInvocations;
    float capturedFinalTrainLoss = result.finalTrainLoss;
    char capturedFirstChar = buf[0];

    freeOptim(sgd);
    freeSoftmaxLayer(model[3]);
    freeLinearLayerShellOnly(model[2]);
    freeReluLayer(model[1]);
    freeLinearLayerShellOnly(model[0]);
    freeDataLoader(evalDl);
    freeDataLoader(trainDl);
    freeQuantization(momentumQ);
    freeQuantization(q);
    freeDataset();

    /* Reaching these at all proves trainingRun did not silent-exit. */
    TEST_ASSERT_EQUAL_UINT(1, capturedCbInvocations);
    TEST_ASSERT_TRUE(capturedFinalTrainLoss > 0.0f);
    TEST_ASSERT_NOT_EQUAL('\0', capturedFirstChar);
}

int main() {
    UNITY_BEGIN();
    RUN_TEST(testMnistSmoke_FullTrainingPipelineReducesLoss);
    RUN_TEST(testMnistSmoke_SnprintfGmtimeRBetweenSetupAndTrainingRun_NoSilentExit);
    return UNITY_END();
}
