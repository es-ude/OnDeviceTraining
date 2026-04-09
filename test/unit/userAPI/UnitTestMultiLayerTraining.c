#define SOURCE_FILE "UNIT_TEST_MULTI_LAYER_TRAINING"

#include <stddef.h>

#include "LossFunction.h"
#include "TensorApi.h"
#include "LinearApi.h"
#include "ReluApi.h"
#include "SoftmaxApi.h"
#include "SgdApi.h"
#include "unity.h"
#include "TrainingLoopApi.h"
#include "CalculateGradsSequential.h"
#include "TrainingBatchDefault.h"
#include "QuantizationApi.h"
#include "Tensor.h"
#include "StorageApi.h"
#include "InferenceApi.h"
#include "DataLoaderApi.h"
#include "Dataset.h"

void setUp() {}
void tearDown() {}

/*! Integration test: multi-layer model (Linear→ReLU→Linear→Softmax) with CrossEntropy.
 *  Reproduces the MnistExperiment structure at small scale (3→4→2).
 *  Uses tensorInitWithDistribution to init bias with ZEROS — exposes the += vs *= bug.
 */
void testMultiLayerBackward_WithCrossEntropy_DoesNotCrash() {
    quantization_t *q = quantizationInitFloat();

    // Layer 0: Linear 3→4
    float w0Data[4 * 3] = {0};
    size_t w0Dims[] = {4, 3};
    tensor_t *w0Param = tensorInitWithDistribution(ZEROS, w0Data, w0Dims, 2, q, NULL, 3, 4);
    tensor_t *w0Grad = gradInitFloat(w0Param, NULL);
    parameter_t *w0 = parameterInit(w0Param, w0Grad);

    float b0Data[4] = {0};
    size_t b0Dims[] = {1, 4};
    tensor_t *b0Param = tensorInitWithDistribution(ZEROS, b0Data, b0Dims, 2, q, NULL, 1, 4);
    tensor_t *b0Grad = gradInitFloat(b0Param, NULL);
    parameter_t *b0 = parameterInit(b0Param, b0Grad);

    layer_t *linear0 = linearLayerInit(w0, b0, q, q, q, q);
    layer_t *relu = reluLayerInit(q, q);

    // Layer 1: Linear 4→2
    float w1Data[2 * 4] = {0};
    size_t w1Dims[] = {2, 4};
    tensor_t *w1Param = tensorInitWithDistribution(ZEROS, w1Data, w1Dims, 2, q, NULL, 4, 2);
    tensor_t *w1Grad = gradInitFloat(w1Param, NULL);
    parameter_t *w1 = parameterInit(w1Param, w1Grad);

    float b1Data[2] = {0};
    size_t b1Dims[] = {1, 2};
    tensor_t *b1Param = tensorInitWithDistribution(ZEROS, b1Data, b1Dims, 2, q, NULL, 1, 2);
    tensor_t *b1Grad = gradInitFloat(b1Param, NULL);
    parameter_t *b1 = parameterInit(b1Param, b1Grad);

    layer_t *linear1 = linearLayerInit(w1, b1, q, q, q, q);
    layer_t *softmax = softmaxLayerInit(q, q);

    layer_t *model[] = {linear0, relu, linear1, softmax};
    size_t sizeModel = 4;

    // Input: [1, 3], Label: [1, 2] (one-hot)
    float inputData[] = {1.0f, 2.0f, 3.0f};
    size_t inputDims[] = {1, 3};
    tensor_t *input = tensorInitFloat(inputData, inputDims, 2, NULL);

    float labelData[] = {1.0f, 0.0f};
    size_t labelDims[] = {1, 2};
    tensor_t *label = tensorInitFloat(labelData, labelDims, 2, NULL);

    // This is the call that crashes in the MnistExperiment
    trainingStats_t *stats = calculateGradsSequential(model, sizeModel, CROSS_ENTROPY,
                                                       input, label);

    TEST_ASSERT_NOT_NULL(stats);
    // Loss should be finite and non-negative
    TEST_ASSERT_TRUE(stats->loss >= 0.0f);

    freeTrainingStats(stats);
}

/*! Integration test: same as above but using tensorInitFloat (no distribution).
 *  This should always work — validates the backward pass logic itself is correct.
 */
void testMultiLayerBackward_WithManualInit_DoesNotCrash() {
    quantization_t testQ;
    initFloat32Quantization(&testQ);

    // Layer 0: Linear 3→4
    float w0Data[] = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f,
                      0.7f, 0.8f, 0.9f, 1.0f, 1.1f, 1.2f};
    size_t w0Dims[] = {4, 3};
    tensor_t *w0Param = tensorInitFloat(w0Data, w0Dims, 2, NULL);
    float w0GradData[12] = {0};
    tensor_t *w0Grad = tensorInitFloat(w0GradData, w0Dims, 2, NULL);
    parameter_t *w0 = parameterInit(w0Param, w0Grad);

    float b0Data[] = {0.0f, 0.0f, 0.0f, 0.0f};
    size_t b0Dims[] = {1, 4};
    tensor_t *b0Param = tensorInitFloat(b0Data, b0Dims, 2, NULL);
    float b0GradData[] = {0.0f, 0.0f, 0.0f, 0.0f};
    tensor_t *b0Grad = tensorInitFloat(b0GradData, b0Dims, 2, NULL);
    parameter_t *b0 = parameterInit(b0Param, b0Grad);

    layer_t *linear0 = linearLayerInit(w0, b0, &testQ, &testQ, &testQ, &testQ);
    layer_t *relu = reluLayerInit(&testQ, &testQ);

    // Layer 1: Linear 4→2
    float w1Data[] = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f};
    size_t w1Dims[] = {2, 4};
    tensor_t *w1Param = tensorInitFloat(w1Data, w1Dims, 2, NULL);
    float w1GradData[8] = {0};
    tensor_t *w1Grad = tensorInitFloat(w1GradData, w1Dims, 2, NULL);
    parameter_t *w1 = parameterInit(w1Param, w1Grad);

    float b1Data[] = {0.0f, 0.0f};
    size_t b1Dims[] = {1, 2};
    tensor_t *b1Param = tensorInitFloat(b1Data, b1Dims, 2, NULL);
    float b1GradData[] = {0.0f, 0.0f};
    tensor_t *b1Grad = tensorInitFloat(b1GradData, b1Dims, 2, NULL);
    parameter_t *b1 = parameterInit(b1Param, b1Grad);

    layer_t *linear1 = linearLayerInit(w1, b1, &testQ, &testQ, &testQ, &testQ);
    layer_t *softmax = softmaxLayerInit(&testQ, &testQ);

    layer_t *model[] = {linear0, relu, linear1, softmax};
    size_t sizeModel = 4;

    float inputData[] = {1.0f, 2.0f, 3.0f};
    size_t inputDims[] = {1, 3};
    tensor_t *input = tensorInitFloat(inputData, inputDims, 2, NULL);

    float labelData[] = {1.0f, 0.0f};
    size_t labelDims[] = {1, 2};
    tensor_t *label = tensorInitFloat(labelData, labelDims, 2, NULL);

    trainingStats_t *stats = calculateGradsSequential(model, sizeModel, CROSS_ENTROPY,
                                                       input, label);

    TEST_ASSERT_NOT_NULL(stats);
    TEST_ASSERT_TRUE(stats->loss >= 0.0f);

    // Verify bias grads were accumulated (not zero after backward)
    float *b1GradValues = (float *)b1Grad->data;
    bool anyNonZero = false;
    for (size_t i = 0; i < 2; i++) {
        if (b1GradValues[i] != 0.0f) {
            anyNonZero = true;
            break;
        }
    }
    TEST_ASSERT_TRUE(anyNonZero);

    freeTrainingStats(stats);
}

/*! Integration test: run multiple training steps to verify grad accumulation is stable. */
void testMultiLayerTraining_MultipleSteps_GradsAccumulate() {
    quantization_t testQ;
    initFloat32Quantization(&testQ);

    float w0Data[] = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f,
                      0.7f, 0.8f, 0.9f, 1.0f, 1.1f, 1.2f};
    size_t w0Dims[] = {4, 3};
    tensor_t *w0Param = tensorInitFloat(w0Data, w0Dims, 2, NULL);
    float w0GradData[12] = {0};
    tensor_t *w0Grad = tensorInitFloat(w0GradData, w0Dims, 2, NULL);
    parameter_t *w0 = parameterInit(w0Param, w0Grad);

    float b0Data[] = {0.0f, 0.0f, 0.0f, 0.0f};
    size_t b0Dims[] = {1, 4};
    tensor_t *b0Param = tensorInitFloat(b0Data, b0Dims, 2, NULL);
    float b0GradData[] = {0.0f, 0.0f, 0.0f, 0.0f};
    tensor_t *b0Grad = tensorInitFloat(b0GradData, b0Dims, 2, NULL);
    parameter_t *b0 = parameterInit(b0Param, b0Grad);

    layer_t *linear0 = linearLayerInit(w0, b0, &testQ, &testQ, &testQ, &testQ);
    layer_t *relu = reluLayerInit(&testQ, &testQ);

    float w1Data[] = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f};
    size_t w1Dims[] = {2, 4};
    tensor_t *w1Param = tensorInitFloat(w1Data, w1Dims, 2, NULL);
    float w1GradData[8] = {0};
    tensor_t *w1Grad = tensorInitFloat(w1GradData, w1Dims, 2, NULL);
    parameter_t *w1 = parameterInit(w1Param, w1Grad);

    float b1Data[] = {0.0f, 0.0f};
    size_t b1Dims[] = {1, 2};
    tensor_t *b1Param = tensorInitFloat(b1Data, b1Dims, 2, NULL);
    float b1GradData[] = {0.0f, 0.0f};
    tensor_t *b1Grad = tensorInitFloat(b1GradData, b1Dims, 2, NULL);
    parameter_t *b1 = parameterInit(b1Param, b1Grad);

    layer_t *linear1 = linearLayerInit(w1, b1, &testQ, &testQ, &testQ, &testQ);
    layer_t *softmax = softmaxLayerInit(&testQ, &testQ);

    layer_t *model[] = {linear0, relu, linear1, softmax};
    size_t sizeModel = 4;

    optimizer_t *sgd = sgdMCreateOptim(0.01f, 0.f, 0.f, model, sizeModel, FLOAT32);
    optimizerFunctions_t sgdFns = optimizerFunctions[SGD_M];

    float inputData[] = {1.0f, 2.0f, 3.0f};
    size_t inputDims[] = {1, 3};
    tensor_t *input = tensorInitFloat(inputData, inputDims, 2, NULL);

    float labelData[] = {1.0f, 0.0f};
    size_t labelDims[] = {1, 2};
    tensor_t *label = tensorInitFloat(labelData, labelDims, 2, NULL);

    // Run 3 training steps
    for (size_t step = 0; step < 3; step++) {
        trainingStats_t *stats = calculateGradsSequential(model, sizeModel, CROSS_ENTROPY,
                                                           input, label);
        TEST_ASSERT_NOT_NULL(stats);
        TEST_ASSERT_TRUE(stats->loss >= 0.0f);
        freeTrainingStats(stats);

        sgdFns.step(sgd);
        sgdFns.zero(sgd);
    }
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(testMultiLayerBackward_WithCrossEntropy_DoesNotCrash);
    RUN_TEST(testMultiLayerBackward_WithManualInit_DoesNotCrash);
    RUN_TEST(testMultiLayerTraining_MultipleSteps_GradsAccumulate);
    return UNITY_END();
}
