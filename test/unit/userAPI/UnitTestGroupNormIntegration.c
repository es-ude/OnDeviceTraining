#define SOURCE_FILE "UNIT_TEST_GROUPNORM_INTEGRATION"

#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdlib.h>

#include "AdaptivePool1dApi.h"
#include "CalculateGradsSequential.h"
#include "Conv1d.h"
#include "Conv1dApi.h"
#include "DataLoaderApi.h"
#include "Dataset.h"
#include "FlattenApi.h"
#include "GroupNorm.h"
#include "GroupNormApi.h"
#include "InferenceApi.h"
#include "Layer.h"
#include "LayerQuant.h"
#include "Linear.h"
#include "LinearApi.h"
#include "LossFunction.h"
#include "Optimizer.h"
#include "QuantizationApi.h"
#include "ReluApi.h"
#include "SgdApi.h"
#include "SoftmaxApi.h"
#include "StateDictApi.h"
#include "StorageApi.h"
#include "Tensor.h"
#include "TensorApi.h"
#include "TrainingLoopApi.h"
#include "unity.h"

void setUp(void) {}
void tearDown(void) {}

/* Model geometry: conv1d(2->4, k3, SAME, bias-less) -> groupNorm(1,4) -> ReLU
 * -> adaptiveAvgPool1d(1) -> Flatten -> linear(4->3) -> Softmax. */
#define IN_CHANNELS 2
#define CONV_CHANNELS 4
#define SEQ_LEN 8
#define NUM_CLASSES 3
#define MODEL_SIZE 7
#define NUM_SAMPLES 8

/* Trainable param element counts (all FLOAT32 storage). */
#define CONV_W_COUNT (CONV_CHANNELS * IN_CHANNELS * 3) /* [Cout,Cin,K] = 24 */
#define GAMMA_COUNT CONV_CHANNELS                      /* [C] = 4 */
#define BETA_COUNT CONV_CHANNELS                       /* [C] = 4 */
#define LIN_W_COUNT (NUM_CLASSES * CONV_CHANNELS)      /* [out,in] = 12 */
#define LIN_B_COUNT NUM_CLASSES                        /* 3 */

/* Deterministic, linearly separable 2-class fixture — NO RNG. Samples 0..3 are
 * class 0 (channel-0 positive, channel-1 negative), 4..7 are class 1 (signs
 * flipped). Per-sample and per-timestep offsets keep the 8 samples distinct
 * without breaking the sign-based separation the model must learn. Each item is
 * a rank-3 [1, IN_CHANNELS, SEQ_LEN] tensor (GroupNorm requires rank-3 [B,C,T]);
 * the DataLoader passes items one at a time and grads accumulate across the
 * batch. File-scope so the getSample callback (fired during trainingRun) reaches
 * them; freeDataset releases them at the end. */
static tensor_t *items[NUM_SAMPLES];
static tensor_t *labels[NUM_SAMPLES];

static void fillSampleData(size_t s, float *itemBuf, float *labelBuf) {
    int cls = (s < 4) ? 0 : 1;
    float sign = (cls == 0) ? 1.0f : -1.0f;
    float amp = 1.5f + 0.1f * (float)(s % 4);
    for (size_t t = 0; t < SEQ_LEN; t++) {
        float ramp = 0.05f * (float)t;
        itemBuf[0 * SEQ_LEN + t] = sign * amp + ramp;
        itemBuf[1 * SEQ_LEN + t] = -sign * amp - ramp;
    }
    for (size_t c = 0; c < NUM_CLASSES; c++) {
        labelBuf[c] = ((int)c == cls) ? 1.0f : 0.0f;
    }
}

static tensor_t *build3DFloat(size_t d0, size_t d1, size_t d2, const float *data, size_t n) {
    size_t *dims = reserveMemory(3 * sizeof(size_t));
    dims[0] = d0;
    dims[1] = d1;
    dims[2] = d2;
    size_t *order = reserveMemory(3 * sizeof(size_t));
    setOrderOfDimsForNewTensor(3, order);
    shape_t *shape = reserveMemory(sizeof(shape_t));
    setShape(shape, dims, 3, order);
    tensor_t *t = initTensor(shape, quantizationInitFloat(), NULL);
    tensorFillFromFloatBuffer(t, (float *)data, n);
    return t;
}

static tensor_t *build2DFloat(size_t d0, size_t d1, const float *data, size_t n) {
    size_t *dims = reserveMemory(2 * sizeof(size_t));
    dims[0] = d0;
    dims[1] = d1;
    size_t *order = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, order);
    shape_t *shape = reserveMemory(sizeof(shape_t));
    setShape(shape, dims, 2, order);
    tensor_t *t = initTensor(shape, quantizationInitFloat(), NULL);
    tensorFillFromFloatBuffer(t, (float *)data, n);
    return t;
}

static void initDataset(void) {
    for (size_t s = 0; s < NUM_SAMPLES; s++) {
        float itemBuf[IN_CHANNELS * SEQ_LEN];
        float labelBuf[NUM_CLASSES];
        fillSampleData(s, itemBuf, labelBuf);
        items[s] = build3DFloat(1, IN_CHANNELS, SEQ_LEN, itemBuf, IN_CHANNELS * SEQ_LEN);
        labels[s] = build2DFloat(1, NUM_CLASSES, labelBuf, NUM_CLASSES);
    }
}

static void freeDataset(void) {
    for (size_t s = 0; s < NUM_SAMPLES; s++) {
        freeTensor(items[s]);
        freeTensor(labels[s]);
    }
}

static sample_t *getSample(size_t id) {
    sample_t *sample = reserveMemory(sizeof(sample_t));
    sample->item = items[id];
    sample->label = labels[id];
    return sample;
}

static size_t getDatasetSize(void) {
    return NUM_SAMPLES;
}

static size_t cbInvocations;
static float firstTrainLoss;
static float lastTrainLoss;

static void captureEpoch(size_t epoch, float trainLoss, epochStats_t evalStats) {
    (void)epoch;
    (void)evalStats;
    if (cbInvocations == 0) {
        firstTrainLoss = trainLoss;
    }
    lastTrainLoss = trainLoss;
    cbInvocations++;
}

/* After freeOptimSgdM frees every registered parameter_t (conv weight,
 * groupNorm gamma/beta, linear weight/bias), the owning layers must be torn
 * down SHELL-ONLY — a factory free would freeParameter the same objects again
 * (double-free). All three layers come from Borrowing factories
 * (ownsQuantizations=false), so the shared math quantization is freed once by
 * the test, not by the shells. Pattern: UnitTestBiaslessConvOptim.c /
 * UnitTestLayerNormIntegration.c. */
static void freeConv1dLayerShell(layer_t *layer) {
    freeReservedMemory(layer->config->conv1d->kernel);
    freeReservedMemory(layer->config->conv1d);
    freeReservedMemory(layer->config);
    freeReservedMemory(layer);
}

static void freeGroupNormLayerShell(layer_t *layer) {
    /* GroupNorm carries no normalizedShape array (unlike LayerNorm): the config
     * struct + wrappers are all that remain once gamma/beta are optimizer-owned. */
    freeReservedMemory(layer->config->groupNorm);
    freeReservedMemory(layer->config);
    freeReservedMemory(layer);
}

static void freeLinearLayerShell(layer_t *layer) {
    freeReservedMemory(layer->config->linear);
    freeReservedMemory(layer->config);
    freeReservedMemory(layer);
}

static void buildModel(layer_t **model, quantization_t *q) {
    layerQuant_t lq;
    layerQuantInitUniform(&lq, q);

    model[0] = conv1dLayerInit(&(conv1dInit_t){.inChannels = IN_CHANNELS,
                                               .outChannels = CONV_CHANNELS,
                                               .kernelSize = 3,
                                               .padding = SAME,
                                               .bias = BIAS_FALSE},
                               &lq);
    model[1] =
        groupNormLayerInit(&(groupNormInit_t){.numGroups = 1, .numChannels = CONV_CHANNELS}, &lq);
    model[2] = reluLayerInit(&lq);
    model[3] = adaptiveAvgPool1dLayerInit(&(adaptiveAvgPool1dInit_t){.outputSize = 1}, &lq);
    model[4] = flattenLayerInit();
    model[5] = linearLayerInit(
        &(linearInit_t){.inFeatures = CONV_CHANNELS, .outFeatures = NUM_CLASSES, .bias = BIAS_TRUE},
        &lq);
    model[6] = softmaxLayerInit(&lq);
}

static void copyParamOut(parameter_t *p, float *dst, size_t count) {
    const float *src = (const float *)p->param->data;
    for (size_t i = 0; i < count; i++) {
        dst[i] = src[i];
    }
}

static void fillParamConstant(parameter_t *p, float value) {
    size_t count = calcNumberOfElementsByTensor(p->param);
    float *dst = (float *)p->param->data;
    for (size_t i = 0; i < count; i++) {
        dst[i] = value;
    }
}

/* End-to-end training exercise for the full GroupNorm wiring: state-count,
 * multi-epoch training-loss reduction (via trainingRun's canonical
 * accumulate/scale/step/zero pipeline), and a hand-built state-dict round trip.
 * Every GROUPNORM userApi switch arm this commit adds is reached here except
 * the trace-only layerParameters arm (trace facility, unexercised by the plain
 * training path). */
void testGroupNormClassifierTrainsAndRoundTrips(void) {
    initDataset();

    quantization_t *q = quantizationInitFloat();
    layer_t *model[MODEL_SIZE];
    buildModel(model, q);

    /* (a) 5 optimizer states: conv (bias-less) 1 + groupNorm 2 + linear 2. */
    size_t numStates = calcTotalNumberOfStates(model, MODEL_SIZE);

    /* Full-batch (batchSize == dataset size): one deterministic step per epoch,
     * no shuffling. */
    dataLoader_t *trainDl =
        dataLoaderInit(getSample, getDatasetSize, NUM_SAMPLES, NULL, NULL, false, 0, true);
    dataLoader_t *evalDl = dataLoaderInit(getSample, getDatasetSize, 1, NULL, NULL, false, 0, true);

    quantization_t *momentumQ = quantizationInitFloat();
    /* lr 0.1 / momentum 0.5: the separable fixture drives the loss ~30x below
     * the halving threshold across the whole hyperparameter neighborhood, so the
     * assertion is deterministic and far from the knife-edge (weight init is
     * RNG-seeded to a fixed constant, so every run is bit-identical). */
    optimizer_t *optim = sgdMCreateOptim(0.1f, 0.5f, 0.0f, model, MODEL_SIZE, FLOAT32, momentumQ);

    /* (b) 30 epochs; captureEpoch records the first and last training loss. */
    cbInvocations = 0;
    size_t numberOfEpochs = 30;
    trainingRun(model, MODEL_SIZE,
                (lossConfig_t){.funcType = CROSS_ENTROPY, .backwardReduction = REDUCTION_MEAN},
                trainDl, evalDl, optim, numberOfEpochs, calculateGradsSequential, inferenceWithLoss,
                captureEpoch);

    /* (c) State-dict round trip. Probe two batch-1 inputs (one per class),
     * capture predictions, copy the trained params out, perturb the live params,
     * restore via modelLoadStateDict, and re-probe. */
    float probeBufA[IN_CHANNELS * SEQ_LEN];
    float probeBufB[IN_CHANNELS * SEQ_LEN];
    float dummyLabel[NUM_CLASSES];
    fillSampleData(0, probeBufA, dummyLabel);
    fillSampleData(4, probeBufB, dummyLabel);
    tensor_t *probeA = build3DFloat(1, IN_CHANNELS, SEQ_LEN, probeBufA, IN_CHANNELS * SEQ_LEN);
    tensor_t *probeB = build3DFloat(1, IN_CHANNELS, SEQ_LEN, probeBufB, IN_CHANNELS * SEQ_LEN);

    tensor_t *predA0 = inference(model, MODEL_SIZE, probeA);
    tensor_t *predB0 = inference(model, MODEL_SIZE, probeB);
    float predA0v[NUM_CLASSES], predB0v[NUM_CLASSES];
    for (size_t c = 0; c < NUM_CLASSES; c++) {
        predA0v[c] = ((float *)predA0->data)[c];
        predB0v[c] = ((float *)predB0->data)[c];
    }
    freeTensor(predA0);
    freeTensor(predB0);

    float savedConvW[CONV_W_COUNT], savedGamma[GAMMA_COUNT], savedBeta[BETA_COUNT];
    float savedLinW[LIN_W_COUNT], savedLinB[LIN_B_COUNT];
    copyParamOut(model[0]->config->conv1d->weights, savedConvW, CONV_W_COUNT);
    copyParamOut(model[1]->config->groupNorm->gamma, savedGamma, GAMMA_COUNT);
    copyParamOut(model[1]->config->groupNorm->beta, savedBeta, BETA_COUNT);
    copyParamOut(model[5]->config->linear->weights, savedLinW, LIN_W_COUNT);
    copyParamOut(model[5]->config->linear->bias, savedLinB, LIN_B_COUNT);

    /* Perturb every trainable param so restore is genuinely observable. */
    fillParamConstant(model[0]->config->conv1d->weights, 0.5f);
    fillParamConstant(model[1]->config->groupNorm->gamma, 0.5f);
    fillParamConstant(model[1]->config->groupNorm->beta, 0.5f);
    fillParamConstant(model[5]->config->linear->weights, 0.5f);
    fillParamConstant(model[5]->config->linear->bias, 0.5f);

    tensor_t *predAperturbed = inference(model, MODEL_SIZE, probeA);
    float predAperturbedv[NUM_CLASSES];
    for (size_t c = 0; c < NUM_CLASSES; c++) {
        predAperturbedv[c] = ((float *)predAperturbed->data)[c];
    }
    freeTensor(predAperturbed);

    tensor_t *predBperturbed = inference(model, MODEL_SIZE, probeB);
    float predBperturbedv[NUM_CLASSES];
    for (size_t c = 0; c < NUM_CLASSES; c++) {
        predBperturbedv[c] = ((float *)predBperturbed->data)[c];
    }
    freeTensor(predBperturbed);

    /* conv entry .biasData = NULL (bias-less, loader sanctions it); groupNorm
     * entry weight->gamma, bias->beta (beta mandatory). */
    stateDictEntry_t entries[] = {
        {.name = "conv", .weightData = savedConvW, .biasData = NULL},
        {.name = "groupNorm", .weightData = savedGamma, .biasData = savedBeta},
        {.name = "linear", .weightData = savedLinW, .biasData = savedLinB},
    };
    modelLoadStateDict(model, MODEL_SIZE, entries, 3);

    tensor_t *predA1 = inference(model, MODEL_SIZE, probeA);
    tensor_t *predB1 = inference(model, MODEL_SIZE, probeB);
    float predA1v[NUM_CLASSES], predB1v[NUM_CLASSES];
    for (size_t c = 0; c < NUM_CLASSES; c++) {
        predA1v[c] = ((float *)predA1->data)[c];
        predB1v[c] = ((float *)predB1->data)[c];
    }
    freeTensor(predA1);
    freeTensor(predB1);

    /* Did perturbation actually move the predictions (round trip is non-vacuous)?
     * Probe BOTH classes so the check can't pass on a class-A-only fluke. */
    bool perturbChanged = false;
    for (size_t c = 0; c < NUM_CLASSES; c++) {
        if (fabsf(predAperturbedv[c] - predA0v[c]) > 1e-4f) {
            perturbChanged = true;
        }
        if (fabsf(predBperturbedv[c] - predB0v[c]) > 1e-4f) {
            perturbChanged = true;
        }
    }

    float capturedFirst = firstTrainLoss;
    float capturedLast = lastTrainLoss;
    size_t capturedCbInvocations = cbInvocations;

    freeTensor(probeB);
    freeTensor(probeA);
    freeOptimSgdM(optim); /* frees conv weight, gamma/beta, linear weight/bias */
    freeSoftmaxLayer(model[6]);
    freeLinearLayerShell(model[5]);
    freeFlattenLayer(model[4]);
    freeAdaptiveAvgPool1dLayer(model[3]);
    freeReluLayer(model[2]);
    freeGroupNormLayerShell(model[1]);
    freeConv1dLayerShell(model[0]);
    freeDataLoader(evalDl);
    freeDataLoader(trainDl);
    freeQuantization(momentumQ);
    freeQuantization(q);
    freeDataset();

    /* (a) */
    TEST_ASSERT_EQUAL_UINT(5, numStates);
    /* (b) */
    TEST_ASSERT_EQUAL_UINT(numberOfEpochs, capturedCbInvocations);
    TEST_ASSERT_TRUE_MESSAGE(isfinite(capturedFirst), "first-epoch loss must be finite");
    TEST_ASSERT_TRUE_MESSAGE(isfinite(capturedLast), "final loss must be finite");
    TEST_ASSERT_TRUE_MESSAGE(capturedLast <= 0.5f * capturedFirst,
                             "30 epochs must at least halve the training loss");
    /* (c) */
    TEST_ASSERT_TRUE_MESSAGE(perturbChanged, "perturbing params must change predictions");
    for (size_t c = 0; c < NUM_CLASSES; c++) {
        TEST_ASSERT_FLOAT_WITHIN_MESSAGE(1e-6f, predA0v[c], predA1v[c],
                                         "class-A prediction must be restored by state-dict load");
        TEST_ASSERT_FLOAT_WITHIN_MESSAGE(1e-6f, predB0v[c], predB1v[c],
                                         "class-B prediction must be restored by state-dict load");
    }
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(testGroupNormClassifierTrainsAndRoundTrips);
    return UNITY_END();
}
