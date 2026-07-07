#define SOURCE_FILE "UNIT_TEST_DROPOUT_INTEGRATION"

#include <stdbool.h>
#include <stdlib.h>

#include <math.h>

#include "Bernoulli.h"
#include "CalculateGradsSequential.h"
#include "Dropout.h"
#include "DropoutApi.h"
#include "InferenceApi.h"
#include "Layer.h"
#include "LayerQuant.h"
#include "LinearApi.h"
#include "LossFunction.h"
#include "Optimizer.h"
#include "QuantizationApi.h"
#include "SgdApi.h"
#include "StateDictApi.h"
#include "StorageApi.h"
#include "Tensor.h"
#include "TensorApi.h"
#include "unity.h"

void setUp(void) {}
void tearDown(void) {}

static void stubKeepEven(tensor_t *mask, float probTrue) {
    (void)probTrue;
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

// Exercises CalculateGradsSequential initLayerOutputs DROPOUT case + the
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
        REDUCTION_SUM, input, label);
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
    optimizer_t *optim = sgdMCreateOptim(0.01f, 0.9f, 0.0f, model, 1, FLOAT32, momentumQ);
    size_t sizeStates = optim->sizeStates;

    freeOptimSgdM(optim);
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

    layer_t *linear0 =
        linearLayerInit(&(linearInit_t){.inFeatures = 4, .outFeatures = 4, .bias = BIAS_TRUE}, &lq);

    tensor_t *mask = buildBoolMask(4);
    layer_t *drop = dropoutLayerInit(0.5f, mask, q, q);

    layer_t *linear1 =
        linearLayerInit(&(linearInit_t){.inFeatures = 4, .outFeatures = 4, .bias = BIAS_TRUE}, &lq);

    layer_t *model[3] = {linear0, drop, linear1};

    tensor_t *input = build2DFloat(1, 4, (float[]){1.f, 2.f, 3.f, 4.f}, 4);
    tensor_t *label = build2DFloat(1, 4, (float[]){0.f, 0.f, 0.f, 0.f}, 4);

    bernoulliFillMaskFn_t saved = bernoulliGetFillMaskFn();
    bernoulliSetFillMaskFn(stubKeepEven);
    trainingStats_t *stats = calculateGradsSequential(
        model, 3, (lossConfig_t){.funcType = MSE, .backwardReduction = REDUCTION_SUM},
        REDUCTION_SUM, input, label);
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

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(testTrainingStep_DropoutActiveThenFlagRestored);
    RUN_TEST(testInference_DropoutIsIdentity);
    RUN_TEST(testOptimizer_ZeroStatesForDropout);
    RUN_TEST(testStateDictLoad_DropoutNoParams);
    RUN_TEST(testMultiLayer_LinearDropoutLinear_BackwardCompletes);
    return UNITY_END();
}
