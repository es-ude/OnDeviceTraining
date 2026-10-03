#define SOURCE_FILE "UNIT_TEST_FLATTEN_INTEGRATION"

#include <stdbool.h>
#include <stdlib.h>

#include "CalculateGradsSequential.h"
#include "FlattenApi.h"
#include "Layer.h"
#include "LayerQuant.h"
#include "LinearApi.h"
#include "LossFunction.h"
#include "QuantizationApi.h"
#include "SoftmaxApi.h"
#include "StorageApi.h"
#include "TensorApi.h"
#include "unity.h"

// Trains [Flatten -> Linear(6,2) -> Softmax] for one step with MSE loss.
// Main point: the wire table's FLATTEN rule must derive output-Q from the input tensor
// (Flatten has no outputQ of its own).
void testCalculateGradsSequential_WithFlattenFirst_DoesNotCrash(void) {
    quantization_t *q = quantizationInitFloat();
    layerQuant_t lq;
    layerQuantInitUniform(&lq, q);

    layer_t *flatten = flattenLayerInit();
    layer_t *linear = linearLayerInit(
        &(linearInit_t){.inFeatures = 6,
                        .outFeatures = 2,
                        .bias = BIAS_TRUE,
                        .weightInit = {.scheme = INIT_XAVIER_UNIFORM, .gain = 1.0f}},
        &lq);
    layer_t *softmax = softmaxLayerInit(&lq);
    layer_t *model[3] = {flatten, linear, softmax};

    /* Input [1, 2, 3] = 6 elements. */
    size_t *inputDims = reserveMemory(3 * sizeof(size_t));
    inputDims[0] = 1;
    inputDims[1] = 2;
    inputDims[2] = 3;
    size_t *inputOrder = reserveMemory(3 * sizeof(size_t));
    setOrderOfDimsForNewTensor(3, inputOrder);
    shape_t *inputShape = reserveMemory(sizeof(shape_t));
    setShape(inputShape, inputDims, 3, inputOrder);
    tensor_t *input = initTensor(inputShape, quantizationInitFloat(), NULL);
    tensorFillFromFloatBuffer(input, (float[]){0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f}, 6);

    /* Label [1, 2]. */
    size_t *labelDims = reserveMemory(2 * sizeof(size_t));
    labelDims[0] = 1;
    labelDims[1] = 2;
    size_t *labelOrder = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, labelOrder);
    shape_t *labelShape = reserveMemory(sizeof(shape_t));
    setShape(labelShape, labelDims, 2, labelOrder);
    tensor_t *label = initTensor(labelShape, quantizationInitFloat(), NULL);
    tensorFillFromFloatBuffer(label, (float[]){1.0f, 0.0f}, 2);

    trainingStats_t *stats = calculateGradsSequential(
        model, 3, (lossConfig_t){.funcType = MSE, .backwardReduction = REDUCTION_SUM},
        REDUCTION_SUM, input, label);

    /* CAPTURE before frees. */
    bool capturedStatsNotNull = (stats != NULL);
    bool capturedOutputNotNull = (stats && stats->output != NULL);
    size_t capturedNumDims =
        (stats && stats->output) ? stats->output->shape->numberOfDimensions : 0;
    size_t capturedDim0 = (stats && stats->output) ? stats->output->shape->dimensions[0] : 0;
    size_t capturedDim1 = (stats && stats->output) ? stats->output->shape->dimensions[1] : 0;

    /* FREE in reverse-init order. */
    freeTrainingStats(stats);
    freeTensor(label);
    freeTensor(input);
    freeSoftmaxLayer(softmax);
    freeLinearLayer(linear);
    freeFlattenLayer(flatten);
    freeQuantization(q);

    /* ASSERT. */
    TEST_ASSERT_TRUE(capturedStatsNotNull);
    TEST_ASSERT_TRUE(capturedOutputNotNull);
    TEST_ASSERT_EQUAL_UINT(2, capturedNumDims);
    TEST_ASSERT_EQUAL_UINT(1, capturedDim0);
    TEST_ASSERT_EQUAL_UINT(2, capturedDim1);
}

void testCalculateGradsSequential_FlattenRank1_DoesNotOOB(void) {
    // Regression: the training wire allocator sized output-shape buffers by INPUT rank, but
    // flattenCalcOutputShape always writes 2 slots. For rank-1 input this is OOB.
    size_t *inputDims = reserveMemory(1 * sizeof(size_t));
    inputDims[0] = 5;
    size_t *inputOrder = reserveMemory(1 * sizeof(size_t));
    setOrderOfDimsForNewTensor(1, inputOrder);
    shape_t *inputShape = reserveMemory(sizeof(shape_t));
    setShape(inputShape, inputDims, 1, inputOrder);
    tensor_t *input = initTensor(inputShape, quantizationInitFloat(), NULL);
    tensorFillFromFloatBuffer(input, (float[]){1.f, 2.f, 3.f, 4.f, 5.f}, 5);

    size_t *labelDims = reserveMemory(2 * sizeof(size_t));
    labelDims[0] = 5;
    labelDims[1] = 1;
    size_t *labelOrder = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, labelOrder);
    shape_t *labelShape = reserveMemory(sizeof(shape_t));
    setShape(labelShape, labelDims, 2, labelOrder);
    tensor_t *label = initTensor(labelShape, quantizationInitFloat(), NULL);
    tensorFillFromFloatBuffer(label, (float[]){0.f, 0.f, 0.f, 0.f, 0.f}, 5);

    layer_t *flatten = flattenLayerInit();
    layer_t *model[1] = {flatten};

    trainingStats_t *stats = calculateGradsSequential(
        model, 1, (lossConfig_t){.funcType = MSE, .backwardReduction = REDUCTION_SUM},
        REDUCTION_SUM, input, label);

    /* CAPTURE before frees. */
    bool capturedStatsNotNull = (stats != NULL);
    size_t capturedNumDims =
        (stats && stats->output) ? stats->output->shape->numberOfDimensions : 0;
    size_t capturedDim0 = (stats && stats->output) ? stats->output->shape->dimensions[0] : 0;
    size_t capturedDim1 = (stats && stats->output) ? stats->output->shape->dimensions[1] : 0;

    /* FREE in reverse-init order. */
    freeTrainingStats(stats);
    freeTensor(label);
    freeTensor(input);
    freeFlattenLayer(flatten);

    /* ASSERT. */
    TEST_ASSERT_TRUE(capturedStatsNotNull);
    TEST_ASSERT_EQUAL_UINT(2, capturedNumDims);
    TEST_ASSERT_EQUAL_UINT(5, capturedDim0);
    TEST_ASSERT_EQUAL_UINT(1, capturedDim1);
}

void testCalculateGradsSequential_FlattenOnly_PreservesValues(void) {
    // FLATTEN is identity on values; stats->output must equal input, reshaped.
    float inputData[] = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f};

    size_t *inputDims = reserveMemory(3 * sizeof(size_t));
    inputDims[0] = 1;
    inputDims[1] = 2;
    inputDims[2] = 3;
    size_t *inputOrder = reserveMemory(3 * sizeof(size_t));
    setOrderOfDimsForNewTensor(3, inputOrder);
    shape_t *inputShape = reserveMemory(sizeof(shape_t));
    setShape(inputShape, inputDims, 3, inputOrder);
    tensor_t *input = initTensor(inputShape, quantizationInitFloat(), NULL);
    tensorFillFromFloatBuffer(input, inputData, 6);

    size_t *labelDims = reserveMemory(2 * sizeof(size_t));
    labelDims[0] = 1;
    labelDims[1] = 6;
    size_t *labelOrder = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, labelOrder);
    shape_t *labelShape = reserveMemory(sizeof(shape_t));
    setShape(labelShape, labelDims, 2, labelOrder);
    tensor_t *label = initTensor(labelShape, quantizationInitFloat(), NULL);
    tensorFillFromFloatBuffer(label, (float[]){0.f, 0.f, 0.f, 0.f, 0.f, 0.f}, 6);

    layer_t *flatten = flattenLayerInit();
    layer_t *model[1] = {flatten};

    trainingStats_t *stats = calculateGradsSequential(
        model, 1, (lossConfig_t){.funcType = MSE, .backwardReduction = REDUCTION_SUM},
        REDUCTION_SUM, input, label);

    /* CAPTURE before frees. */
    bool capturedStatsNotNull = (stats != NULL);
    size_t capturedNumDims =
        (stats && stats->output) ? stats->output->shape->numberOfDimensions : 0;
    size_t capturedDim0 = (stats && stats->output) ? stats->output->shape->dimensions[0] : 0;
    size_t capturedDim1 = (stats && stats->output) ? stats->output->shape->dimensions[1] : 0;
    float capturedValues[6] = {0};
    if (stats && stats->output && stats->output->data) {
        for (size_t i = 0; i < 6; i++) {
            capturedValues[i] = ((float *)stats->output->data)[i];
        }
    }

    /* FREE in reverse-init order. */
    freeTrainingStats(stats);
    freeTensor(label);
    freeTensor(input);
    freeFlattenLayer(flatten);

    /* ASSERT. */
    TEST_ASSERT_TRUE(capturedStatsNotNull);
    TEST_ASSERT_EQUAL_UINT(2, capturedNumDims);
    TEST_ASSERT_EQUAL_UINT(1, capturedDim0);
    TEST_ASSERT_EQUAL_UINT(6, capturedDim1);
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(inputData, capturedValues, 6);
}

void setUp(void) {}
void tearDown(void) {}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(testCalculateGradsSequential_WithFlattenFirst_DoesNotCrash);
    RUN_TEST(testCalculateGradsSequential_FlattenRank1_DoesNotOOB);
    RUN_TEST(testCalculateGradsSequential_FlattenOnly_PreservesValues);
    return UNITY_END();
}
