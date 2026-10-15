#define SOURCE_FILE "UNIT_TEST_DROPOUT_INTEGRATION"

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

#include <math.h>

#include "Bernoulli.h"
#include "BorrowedLayer.h"
#include "CalculateGradsSequential.h"
#include "DataLoaderApi.h"
#include "Deserialize.h"
#include "Dropout.h"
#include "DropoutApi.h"
#include "InferenceApi.h"
#include "Layer.h"
#include "LayerQuant.h"
#include "Linear.h"
#include "LinearApi.h"
#include "LossFunction.h"
#include "Optimizer.h"
#include "OptimizerApi.h"
#include "QuantizationApi.h"
#include "RNG.h"
#include "Serialize.h"
#include "SgdApi.h"
#include "StateDictApi.h"
#include "StorageApi.h"
#include "Tensor.h"
#include "TensorApi.h"
#include "TrainingLoopApi.h"
#include "unity.h"

void setUp(void) {}
void tearDown(void) {}

static void stubKeepEven(tensor_t *mask, float probTrue, rngStream_t *rs) {
    (void)probTrue;
    (void)rs;
    size_t n = calcNumberOfElementsByTensor(mask);
    for (size_t i = 0; i < n; i++) {
        tensorBoolSet(mask, i, (i % 2) == 0);
    }
}

static tensor_t *build2DFloat(size_t b, size_t f, const float *data, size_t n) {
    size_t *dims = reserveMemory(2 * sizeof(size_t));
    dims[0] = b;
    dims[1] = f;
    size_t *order = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, order);
    shape_t *shape = reserveMemory(sizeof(shape_t));
    setShape(shape, dims, 2, order);
    tensor_t *t = initTensor(shape, quantizationInitFloat(), NULL);
    tensorFillFromFloatBuffer(t, (float *)data, n);
    return t;
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

// Exercises the DROPOUT wire of calculateGradsSequential's wire table + the
// training-flag lifecycle: training-mode forward (output != input), flag
// restored to false afterward.
void testTrainingStep_DropoutActiveThenFlagRestored(void) {
    quantization_t *fq = quantizationInitFloat();
    quantization_t *bq = quantizationInitFloat();
    tensor_t *mask = buildBoolMask(4);
    layer_t *drop = dropoutLayerInit(0.5f, mask, fq, bq);
    layer_t *model[1] = {drop};

    tensor_t *input = build2DFloat(1, 4, (float[]){1.f, 2.f, 3.f, 4.f}, 4);
    tensor_t *label = build2DFloat(1, 4, (float[]){0.f, 0.f, 0.f, 0.f}, 4);

    bernoulliFillMaskFn_t saved = bernoulliGetFillMaskFn();
    bernoulliSetFillMaskFn(stubKeepEven);
    trainingStats_t *stats = calculateGradsSequential(
        model, 1, (lossConfig_t){.funcType = MSE, .backwardReduction = REDUCTION_SUM},
        REDUCTION_SUM, input, label, NULL);
    bernoulliSetFillMaskFn(saved);

    bool statsNotNull = (stats != NULL);
    float o0 = (stats && stats->output) ? ((float *)stats->output->data)[0] : -1.f;
    float o1 = (stats && stats->output) ? ((float *)stats->output->data)[1] : -1.f;
    float o2 = (stats && stats->output) ? ((float *)stats->output->data)[2] : -1.f;
    bool flagRestored = (drop->config->dropout->training == false);

    freeTrainingStats(stats);
    freeTensor(label);
    freeTensor(input);
    freeDropoutLayer(drop);
    freeTensor(mask);
    freeQuantization(bq);
    freeQuantization(fq);

    TEST_ASSERT_TRUE(statsNotNull);
    TEST_ASSERT_FLOAT_WITHIN(1e-5f, 2.f, o0);
    TEST_ASSERT_FLOAT_WITHIN(1e-5f, 0.f, o1);
    TEST_ASSERT_FLOAT_WITHIN(1e-5f, 6.f, o2);
    TEST_ASSERT_TRUE(flagRestored);
}

// Exercises InferenceApi initBufferOutput DROPOUT case + vtable forward.
// Inference => training flag false => identity.
void testInference_DropoutIsIdentity(void) {
    quantization_t *fq = quantizationInitFloat();
    quantization_t *bq = quantizationInitFloat();
    tensor_t *mask = buildBoolMask(4);
    layer_t *drop = dropoutLayerInit(0.5f, mask, fq, bq);
    layer_t *model[1] = {drop};

    tensor_t *input = build2DFloat(1, 4, (float[]){1.f, 2.f, 3.f, 4.f}, 4);
    tensor_t *output = inference(model, 1, input);

    bool outNotNull = (output != NULL);
    float captured[4];
    for (size_t i = 0; i < 4; i++) {
        captured[i] = (output && output->data) ? ((float *)output->data)[i] : -1.f;
    }

    freeTensor(output);
    freeTensor(input);
    freeDropoutLayer(drop);
    freeTensor(mask);
    freeQuantization(bq);
    freeQuantization(fq);

    TEST_ASSERT_TRUE(outNotNull);
    float expected[] = {1.f, 2.f, 3.f, 4.f};
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(expected, captured, 4);
}

// Exercises Optimizer.calcNumberOfStatesByLayer + SgdApi state-alloc:
// Dropout is parameter-less => zero optimizer state.
void testOptimizer_ZeroStatesForDropout(void) {
    quantization_t *fq = quantizationInitFloat();
    quantization_t *bq = quantizationInitFloat();
    tensor_t *mask = buildBoolMask(4);
    layer_t *drop = dropoutLayerInit(0.5f, mask, fq, bq);
    layer_t *model[1] = {drop};

    size_t numStates = calcTotalNumberOfStates(model, 1);
    quantization_t *momentumQ = quantizationInitFloat();
    optimizer_t *optim =
        sgdMCreateOptim(0.01f, 0.9f, 0.0f, model, 1, momentumQ,
                        (arithmetic_t){.type = ARITH_FLOAT32, .roundingMode = HALF_AWAY});
    size_t sizeStates = optim->sizeStates;

    freeOptim(optim);
    freeDropoutLayer(drop);
    freeTensor(mask);
    freeQuantization(momentumQ);
    freeQuantization(bq);
    freeQuantization(fq);

    TEST_ASSERT_EQUAL_UINT(0, numStates);
    TEST_ASSERT_EQUAL_UINT(0, sizeStates);
}

// Exercises StateDictApi.layerHasParameters: a parameter-less model with zero
// entries must load without error.
void testStateDictLoad_DropoutNoParams(void) {
    quantization_t *fq = quantizationInitFloat();
    quantization_t *bq = quantizationInitFloat();
    tensor_t *mask = buildBoolMask(4);
    layer_t *drop = dropoutLayerInit(0.5f, mask, fq, bq);
    layer_t *model[1] = {drop};

    modelLoadStateDict(model, 1, NULL, 0); // must return without exiting

    freeDropoutLayer(drop);
    freeTensor(mask);
    freeQuantization(bq);
    freeQuantization(fq);
    TEST_PASS();
}

// Exercises a Linear(4→4) → Dropout(0.5) → Linear(4→4) model through one MSE
// training step, verifying that dropout's propLoss is correctly consumed by
// the upstream Linear backward pass.
void testMultiLayer_LinearDropoutLinear_BackwardCompletes(void) {
    quantization_t *q = quantizationInitFloat();
    layerQuant_t lq;
    layerQuantInitUniform(&lq, q);

    const rngKey_t initKey0 = rngFold(rngKeyFromSeed(1u), RNG_TAG_INIT, 0u);
    const rngKey_t initKey2 = rngFold(rngKeyFromSeed(1u), RNG_TAG_INIT, 2u);
    layer_t *linear0 = linearLayerInit(
        &(linearInit_t){
            .inFeatures = 4, .outFeatures = 4, .bias = BIAS_TRUE, .weightInit.key = &initKey0},
        &lq);

    tensor_t *mask = buildBoolMask(4);
    layer_t *drop = dropoutLayerInit(0.5f, mask, q, q);

    layer_t *linear1 = linearLayerInit(
        &(linearInit_t){
            .inFeatures = 4, .outFeatures = 4, .bias = BIAS_TRUE, .weightInit.key = &initKey2},
        &lq);

    layer_t *model[3] = {linear0, drop, linear1};

    tensor_t *input = build2DFloat(1, 4, (float[]){1.f, 2.f, 3.f, 4.f}, 4);
    tensor_t *label = build2DFloat(1, 4, (float[]){0.f, 0.f, 0.f, 0.f}, 4);

    bernoulliFillMaskFn_t saved = bernoulliGetFillMaskFn();
    bernoulliSetFillMaskFn(stubKeepEven);
    trainingStats_t *stats = calculateGradsSequential(
        model, 3, (lossConfig_t){.funcType = MSE, .backwardReduction = REDUCTION_SUM},
        REDUCTION_SUM, input, label, NULL);
    bernoulliSetFillMaskFn(saved);

    bool statsNotNull = (stats != NULL);
    float loss = stats ? stats->loss : -1.f;
    bool lossFinite = (loss == loss) && (fabsf(loss) < 1e30f);
    size_t outElems = (stats && stats->output) ? calcNumberOfElementsByTensor(stats->output) : 0;

    freeTrainingStats(stats);
    freeTensor(label);
    freeTensor(input);
    freeLinearLayer(linear1);
    freeDropoutLayer(drop);
    freeTensor(mask);
    freeLinearLayer(linear0);
    freeQuantization(q);

    TEST_ASSERT_TRUE(statsNotNull);
    TEST_ASSERT_TRUE(lossFinite);
    TEST_ASSERT_EQUAL_UINT(4, outElems);
}

/* ---- resuming a run at an epoch boundary (#4) ---- */

#define RESUME_SAMPLES 4
#define RESUME_PARAMS 23 /* Linear(4 -> 3): 12 + 3, Linear(3 -> 2): 6 + 2 */

static tensor_t *g_resumeItems[RESUME_SAMPLES];
static tensor_t *g_resumeLabels[RESUME_SAMPLES];

static tensor_t *build1DFloat(size_t n, const float *data) {
    size_t *dims = reserveMemory(sizeof(size_t));
    dims[0] = n;
    size_t *order = reserveMemory(sizeof(size_t));
    setOrderOfDimsForNewTensor(1, order);
    shape_t *shape = reserveMemory(sizeof(shape_t));
    setShape(shape, dims, 1, order);
    tensor_t *t = initTensor(shape, quantizationInitFloat(), NULL);
    tensorFillFromFloatBuffer(t, (float *)data, n);
    return t;
}

static void buildResumeDataset(void) {
    for (size_t i = 0; i < RESUME_SAMPLES; i++) {
        const float x[4] = {0.5f * (float)i - 1.0f, 1.0f, -0.5f, 0.25f * (float)i};
        const float y[2] = {(float)(i % 2), (float)(1 - i % 2)};
        g_resumeItems[i] = build1DFloat(4, x);
        g_resumeLabels[i] = build1DFloat(2, y);
    }
}

static void freeResumeDataset(void) {
    for (size_t i = 0; i < RESUME_SAMPLES; i++) {
        freeTensor(g_resumeItems[i]);
        freeTensor(g_resumeLabels[i]);
    }
}

static sample_t *getResumeSample(size_t id) {
    sample_t *s = reserveMemory(sizeof(sample_t));
    s->item = g_resumeItems[id];
    s->label = g_resumeLabels[id];
    return s;
}

static size_t resumeDatasetSize(void) {
    return RESUME_SAMPLES;
}

/* Linear(4 -> 3) -> Dropout(0.5) -> Linear(3 -> 2) under SGD without momentum,
 * initialised from the init keys of initSeed; the train loader reshuffles per
 * epoch. */
typedef struct {
    quantization_t *q;
    layerQuant_t lq;
    tensor_t *mask;
    layer_t *model[3];
    quantization_t *momentumQ;
    optimizer_t *sgd;
    dataLoader_t *trainDl;
    dataLoader_t *evalDl;
} resumeRun_t;

static void buildResumeRun(resumeRun_t *r, uint64_t initSeed) {
    r->q = quantizationInitFloat();
    layerQuantInitUniform(&r->lq, r->q);
    const rngKey_t initKey0 = rngFold(rngKeyFromSeed(initSeed), RNG_TAG_INIT, 0u);
    const rngKey_t initKey2 = rngFold(rngKeyFromSeed(initSeed), RNG_TAG_INIT, 2u);
    r->model[0] = linearLayerInit(
        &(linearInit_t){
            .inFeatures = 4, .outFeatures = 3, .bias = BIAS_TRUE, .weightInit.key = &initKey0},
        &r->lq);
    r->mask = buildBoolMask(3);
    r->model[1] = dropoutLayerInit(0.5f, r->mask, r->q, r->q);
    r->model[2] = linearLayerInit(
        &(linearInit_t){
            .inFeatures = 3, .outFeatures = 2, .bias = BIAS_TRUE, .weightInit.key = &initKey2},
        &r->lq);
    r->momentumQ = quantizationInitFloat();
    r->sgd = sgdMCreateOptim(0.05f, 0.f, 0.f, r->model, 3, r->momentumQ,
                             (arithmetic_t){.type = ARITH_FLOAT32, .roundingMode = HALF_AWAY});
    r->trainDl = dataLoaderInit(getResumeSample, resumeDatasetSize, 1, NULL, NULL, true, true);
    dataLoaderSetReshufflePerEpoch(r->trainDl, true);
    r->evalDl = dataLoaderInit(getResumeSample, resumeDatasetSize, 1, NULL, NULL, false, true);
}

static void runResume(resumeRun_t *r, size_t epochs, trainingRng_t *rng) {
    (void)trainingRun(r->model, 3,
                      (lossConfig_t){.funcType = MSE, .backwardReduction = REDUCTION_MEAN},
                      r->trainDl, r->evalDl, r->sgd, epochs, calculateGradsSequential,
                      inferenceWithLoss, &(trainingRunOptions_t){.rng = rng});
}

/* Fills params[0..22] with both Linears' weights, then their biases. */
static void captureResumeParams(resumeRun_t *r, float params[RESUME_PARAMS]) {
    size_t at = 0;
    for (size_t l = 0; l < 3; l += 2) {
        tensor_t *w = r->model[l]->config->linear->weights->param;
        tensor_t *b = r->model[l]->config->linear->bias->param;
        for (size_t i = 0; i < calcNumberOfElementsByTensor(w); i++) {
            params[at++] = ((const float *)w->data)[i];
        }
        for (size_t i = 0; i < calcNumberOfElementsByTensor(b); i++) {
            params[at++] = ((const float *)b->data)[i];
        }
    }
}

/* freeOptim frees every model parameter, so the Linears go as shells. */
static void freeResumeRun(resumeRun_t *r) {
    freeOptim(r->sgd);
    freeQuantization(r->momentumQ);
    freeDataLoader(r->evalDl);
    freeDataLoader(r->trainDl);
    freeLinearLayerShellOnly(r->model[2]);
    freeDropoutLayer(r->model[1]);
    freeTensor(r->mask);
    freeLinearLayerShellOnly(r->model[0]);
    freeQuantization(r->q);
}

/* Two epochs in one go, and one epoch + save (the model as ODTS, the training
 * state as ODTK) + load into a fresh skeleton with other initial values and
 * another training state + one more epoch, end bit for bit alike: the second
 * epoch's sample order follows the restored epoch counter, its Dropout masks
 * the restored step counter. Resume only between epochs (no batch cursor),
 * with SGD without momentum and no scheduler (neither state is saved). */
void testResumingAtAnEpochBoundaryContinuesTheRunBitForBit(void) {
    buildResumeDataset();

    resumeRun_t whole;
    buildResumeRun(&whole, 1u);
    trainingRng_t wholeRng;
    trainingRngInit(&wholeRng, 7u);
    runResume(&whole, 2, &wholeRng);
    float wholeParams[RESUME_PARAMS];
    captureResumeParams(&whole, wholeParams);
    freeResumeRun(&whole);

    resumeRun_t first;
    buildResumeRun(&first, 1u);
    trainingRng_t firstRng;
    trainingRngInit(&firstRng, 7u);
    runResume(&first, 1, &firstRng);
    FILE *f = tmpfile();
    serializeModel(first.model, 3, f);
    trainingRngSerialize(&firstRng, f);
    freeResumeRun(&first);

    resumeRun_t resumed;
    buildResumeRun(&resumed, 999u);
    trainingRng_t resumedRng;
    trainingRngInit(&resumedRng, 999u);
    rewind(f);
    deserializeModel(resumed.model, 3, f);
    trainingRngDeserialize(&resumedRng, f);
    fclose(f);
    runResume(&resumed, 1, &resumedRng);
    float resumedParams[RESUME_PARAMS];
    captureResumeParams(&resumed, resumedParams);
    freeResumeRun(&resumed);
    freeResumeDataset();

    TEST_ASSERT_EQUAL_MEMORY(wholeParams, resumedParams, sizeof wholeParams);
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(testTrainingStep_DropoutActiveThenFlagRestored);
    RUN_TEST(testInference_DropoutIsIdentity);
    RUN_TEST(testOptimizer_ZeroStatesForDropout);
    RUN_TEST(testStateDictLoad_DropoutNoParams);
    RUN_TEST(testMultiLayer_LinearDropoutLinear_BackwardCompletes);
    RUN_TEST(testResumingAtAnEpochBoundaryContinuesTheRunBitForBit);
    return UNITY_END();
}
