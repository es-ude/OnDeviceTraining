#define SOURCE_FILE "UNIT_TEST_ADAPTIVE_AVG_POOL_1D_INTEGRATION"

#include <stdbool.h>
#include <stdlib.h>

#include "AdaptivePool1dApi.h"
#include "CalculateGradsSequential.h"
#include "InferenceApi.h"
#include "Layer.h"
#include "LayerQuant.h"
#include "LossFunction.h"
#include "Optimizer.h"
#include "QuantizationApi.h"
#include "SgdApi.h"
#include "StateDictApi.h"
#include "StorageApi.h"
#include "TensorApi.h"
#include "unity.h"

void setUp(void) {}
void tearDown(void) {}

static layer_t *buildPool(quantization_t *q, size_t outputSize) {
    layerQuant_t lq;
    layerQuantInitUniform(&lq, q);
    return adaptiveAvgPool1dLayerInit(&(adaptiveAvgPool1dInit_t){.outputSize = outputSize}, &lq);
}

static tensor_t *build3DFloat(size_t d0, size_t d1, size_t d2, float const *data, size_t n) {
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

// Exercises CalculateGradsSequential initLayerOutputs ADAPTIVE_AVGPOOL1D case +
// vtable forward/backward. Missing case -> initLayerOutputs hits default exit(1).
void testTrainingStep_PoolsCorrectly(void) {
    quantization_t *q = quantizationInitFloat();
    layer_t *pool = buildPool(q, 2);
    layer_t *model[1] = {pool};

    tensor_t *input = build3DFloat(1, 1, 4, (float[]){1.f, 2.f, 3.f, 4.f}, 4);
    tensor_t *label = build3DFloat(1, 1, 2, (float[]){0.f, 0.f}, 2);

    trainingStats_t *stats = calculateGradsSequential(
        model, 1, (lossConfig_t){.funcType = MSE, .backwardReduction = REDUCTION_SUM},
        REDUCTION_SUM, input, label);

    bool statsNotNull = (stats != NULL);
    size_t d2 = (stats && stats->output) ? stats->output->shape->dimensions[2] : 0;
    float out0 =
        (stats && stats->output && stats->output->data) ? ((float *)stats->output->data)[0] : 0.f;
    float out1 =
        (stats && stats->output && stats->output->data) ? ((float *)stats->output->data)[1] : 0.f;

    freeTrainingStats(stats);
    freeTensor(label);
    freeTensor(input);
    freeAdaptiveAvgPool1dLayer(pool);
    freeQuantization(q);

    TEST_ASSERT_TRUE(statsNotNull);
    TEST_ASSERT_EQUAL_UINT(2, d2);
    TEST_ASSERT_FLOAT_WITHIN(1e-5f, 1.5f, out0);
    TEST_ASSERT_FLOAT_WITHIN(1e-5f, 3.5f, out1);
}

// Exercises InferenceApi initBufferOutput ADAPTIVE_AVGPOOL1D case + vtable forward.
void testInference_ProducesAdaptiveMeans(void) {
    quantization_t *q = quantizationInitFloat();
    layer_t *pool = buildPool(q, 2);
    layer_t *model[1] = {pool};

    tensor_t *input = build3DFloat(1, 1, 4, (float[]){1.f, 2.f, 3.f, 4.f}, 4);
    tensor_t *output = inference(model, 1, input);

    bool outNotNull = (output != NULL);
    float out0 = (output && output->data) ? ((float *)output->data)[0] : 0.f;
    float out1 = (output && output->data) ? ((float *)output->data)[1] : 0.f;

    freeTensor(output);
    freeTensor(input);
    freeAdaptiveAvgPool1dLayer(pool);
    freeQuantization(q);

    TEST_ASSERT_TRUE(outNotNull);
    TEST_ASSERT_FLOAT_WITHIN(1e-5f, 1.5f, out0);
    TEST_ASSERT_FLOAT_WITHIN(1e-5f, 3.5f, out1);
}

// Exercises Optimizer.c calcNumberOfStatesByLayer + SgdApi.c state-alloc:
// the layer is parameter-less, so both must classify it as zero-state.
void testOptimizer_ZeroStatesForParameterlessPool(void) {
    quantization_t *q = quantizationInitFloat();
    layer_t *pool = buildPool(q, 2);
    layer_t *model[1] = {pool};

    size_t numStates = calcTotalNumberOfStates(model, 1);
    quantization_t *momentumQ = quantizationInitFloat();
    optimizer_t *optim = sgdMCreateOptim(0.01f, 0.9f, 0.0f, model, 1, FLOAT32, momentumQ);
    size_t sizeStates = optim->sizeStates;

    freeOptimSgdM(optim);
    freeAdaptiveAvgPool1dLayer(pool);
    freeQuantization(momentumQ);
    freeQuantization(q);

    TEST_ASSERT_EQUAL_UINT(0, numStates);
    TEST_ASSERT_EQUAL_UINT(0, sizeStates);
}

// Exercises StateDictApi.c layerHasParameters: a parameter-less model with zero
// entries must load without error. Missing case -> layerHasParameters exit(1).
void testStateDictLoad_NoParamLayerZeroEntries(void) {
    quantization_t *q = quantizationInitFloat();
    layer_t *pool = buildPool(q, 2);
    layer_t *model[1] = {pool};

    modelLoadStateDict(model, 1, NULL, 0); // must return without exiting

    freeAdaptiveAvgPool1dLayer(pool);
    freeQuantization(q);
    TEST_PASS();
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(testTrainingStep_PoolsCorrectly);
    RUN_TEST(testInference_ProducesAdaptiveMeans);
    RUN_TEST(testOptimizer_ZeroStatesForParameterlessPool);
    RUN_TEST(testStateDictLoad_NoParamLayerZeroEntries);
    return UNITY_END();
}
