#define SOURCE_FILE "UNIT_TEST_MULTI_LAYER_TRAINING"

#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "ArithmeticType.h"
#include "BorrowedLayer.h"
#include "CalculateGradsSequential.h"
#include "DataLoaderApi.h"
#include "Dataset.h"
#include "DeathTest.h"
#include "InferenceApi.h"
#include "LayerQuant.h"
#include "Linear.h"
#include "LinearApi.h"
#include "LossFunction.h"
#include "OptimizerApi.h"
#include "QuantizationApi.h"
#include "RNG.h"
#include "ReluApi.h"
#include "SgdApi.h"
#include "SoftmaxApi.h"
#include "StorageApi.h"
#include "Tensor.h"
#include "TensorApi.h"
#include "TraceApi.h"
#include "TrainingBatchDefault.h"
#include "TrainingLoopApi.h"
#include "unity.h"

void setUp() {}
void tearDown() {}

/*! Integration test: multi-layer model (Linear→ReLU→Linear→Softmax) with CrossEntropy.
 *  Reproduces the MnistExperiment structure at small scale (3→4→2).
 *  Uses initDistribution to init weights/biases with ZEROS — exposes the += vs *= bug.
 */
void testMultiLayerBackward_WithCrossEntropy_DoesNotCrash() {
    quantization_t *q = quantizationInitFloat();
    layerQuant_t lq;
    layerQuantInitUniform(&lq, q);
    distribution_t zeros = {.type = ZEROS};

    /* Layer 0 weights w0 (4x3, ZEROS). */
    size_t *w0Dims = reserveMemory(2 * sizeof(size_t));
    w0Dims[0] = 4;
    w0Dims[1] = 3;
    size_t *w0Order = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, w0Order);
    shape_t *w0Shape = reserveMemory(sizeof(shape_t));
    setShape(w0Shape, w0Dims, 2, w0Order);
    tensor_t *w0Param = initTensor(w0Shape, quantizationInitFloat(), NULL);
    initDistribution(w0Param, &zeros);
    tensor_t *w0Grad = gradInitFloat(w0Param, NULL);
    parameter_t *w0 = parameterInit(w0Param, w0Grad);

    /* Layer 0 bias b0 (1x4, ZEROS). */
    size_t *b0Dims = reserveMemory(2 * sizeof(size_t));
    b0Dims[0] = 1;
    b0Dims[1] = 4;
    size_t *b0Order = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, b0Order);
    shape_t *b0Shape = reserveMemory(sizeof(shape_t));
    setShape(b0Shape, b0Dims, 2, b0Order);
    tensor_t *b0Param = initTensor(b0Shape, quantizationInitFloat(), NULL);
    initDistribution(b0Param, &zeros);
    tensor_t *b0Grad = gradInitFloat(b0Param, NULL);
    parameter_t *b0 = parameterInit(b0Param, b0Grad);

    layer_t *linear0 = buildBorrowedLinearLayer(w0, b0, q);
    layer_t *relu = reluLayerInit(&lq);

    /* Layer 1 weights w1 (2x4, ZEROS). */
    size_t *w1Dims = reserveMemory(2 * sizeof(size_t));
    w1Dims[0] = 2;
    w1Dims[1] = 4;
    size_t *w1Order = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, w1Order);
    shape_t *w1Shape = reserveMemory(sizeof(shape_t));
    setShape(w1Shape, w1Dims, 2, w1Order);
    tensor_t *w1Param = initTensor(w1Shape, quantizationInitFloat(), NULL);
    initDistribution(w1Param, &zeros);
    tensor_t *w1Grad = gradInitFloat(w1Param, NULL);
    parameter_t *w1 = parameterInit(w1Param, w1Grad);

    /* Layer 1 bias b1 (1x2, ZEROS). */
    size_t *b1Dims = reserveMemory(2 * sizeof(size_t));
    b1Dims[0] = 1;
    b1Dims[1] = 2;
    size_t *b1Order = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, b1Order);
    shape_t *b1Shape = reserveMemory(sizeof(shape_t));
    setShape(b1Shape, b1Dims, 2, b1Order);
    tensor_t *b1Param = initTensor(b1Shape, quantizationInitFloat(), NULL);
    initDistribution(b1Param, &zeros);
    tensor_t *b1Grad = gradInitFloat(b1Param, NULL);
    parameter_t *b1 = parameterInit(b1Param, b1Grad);

    layer_t *linear1 = buildBorrowedLinearLayer(w1, b1, q);
    layer_t *softmax = softmaxLayerInit(&lq);

    layer_t *model[] = {linear0, relu, linear1, softmax};
    size_t sizeModel = 4;

    /* Input (1x3). */
    size_t *inputDims = reserveMemory(2 * sizeof(size_t));
    inputDims[0] = 1;
    inputDims[1] = 3;
    size_t *inputOrder = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, inputOrder);
    shape_t *inputShape = reserveMemory(sizeof(shape_t));
    setShape(inputShape, inputDims, 2, inputOrder);
    tensor_t *input = initTensor(inputShape, quantizationInitFloat(), NULL);
    tensorFillFromFloatBuffer(input, (float[]){1.0f, 2.0f, 3.0f}, 3);

    /* Label (1x2 one-hot). */
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
        model, sizeModel,
        (lossConfig_t){.funcType = CROSS_ENTROPY, .backwardReduction = REDUCTION_SUM},
        REDUCTION_SUM, input, label);

    /* CAPTURE. */
    bool capturedNotNull = (stats != NULL);
    float capturedLoss = stats ? stats->loss : -1.0f;

    /* FREE in reverse-init order. */
    freeTrainingStats(stats);
    freeTensor(label);
    freeTensor(input);
    freeSoftmaxLayer(softmax);
    freeLinearLayerShellOnly(linear1);
    freeParameter(b1);
    freeParameter(w1);
    freeReluLayer(relu);
    freeLinearLayerShellOnly(linear0);
    freeParameter(b0);
    freeParameter(w0);
    freeQuantization(q);

    /* ASSERT on captured. */
    TEST_ASSERT_TRUE(capturedNotNull);
    TEST_ASSERT_TRUE(capturedLoss >= 0.0f);
}

/*! Integration test: same as above but with manually filled weights.
 *  Validates the backward pass logic itself is correct.
 */
void testMultiLayerBackward_WithManualInit_DoesNotCrash() {
    quantization_t *q = quantizationInitFloat();
    layerQuant_t lq;
    layerQuantInitUniform(&lq, q);

    /* Layer 0 weights w0 (4x3, manual values). */
    size_t *w0Dims = reserveMemory(2 * sizeof(size_t));
    w0Dims[0] = 4;
    w0Dims[1] = 3;
    size_t *w0Order = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, w0Order);
    shape_t *w0Shape = reserveMemory(sizeof(shape_t));
    setShape(w0Shape, w0Dims, 2, w0Order);
    tensor_t *w0Param = initTensor(w0Shape, quantizationInitFloat(), NULL);
    tensorFillFromFloatBuffer(
        w0Param, (float[]){0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.0f, 1.1f, 1.2f},
        12);
    tensor_t *w0Grad = gradInitFloat(w0Param, NULL);
    parameter_t *w0 = parameterInit(w0Param, w0Grad);

    /* Layer 0 bias b0 (1x4, zeros). */
    size_t *b0Dims = reserveMemory(2 * sizeof(size_t));
    b0Dims[0] = 1;
    b0Dims[1] = 4;
    size_t *b0Order = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, b0Order);
    shape_t *b0Shape = reserveMemory(sizeof(shape_t));
    setShape(b0Shape, b0Dims, 2, b0Order);
    tensor_t *b0Param = initTensor(b0Shape, quantizationInitFloat(), NULL);
    /* initTensor zero-initializes data per TensorApi.c:81-92, so no explicit fill. */
    tensor_t *b0Grad = gradInitFloat(b0Param, NULL);
    parameter_t *b0 = parameterInit(b0Param, b0Grad);

    layer_t *linear0 = buildBorrowedLinearLayer(w0, b0, q);
    layer_t *relu = reluLayerInit(&lq);

    /* Layer 1 weights w1 (2x4, manual). */
    size_t *w1Dims = reserveMemory(2 * sizeof(size_t));
    w1Dims[0] = 2;
    w1Dims[1] = 4;
    size_t *w1Order = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, w1Order);
    shape_t *w1Shape = reserveMemory(sizeof(shape_t));
    setShape(w1Shape, w1Dims, 2, w1Order);
    tensor_t *w1Param = initTensor(w1Shape, quantizationInitFloat(), NULL);
    tensorFillFromFloatBuffer(w1Param, (float[]){0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f},
                              8);
    tensor_t *w1Grad = gradInitFloat(w1Param, NULL);
    parameter_t *w1 = parameterInit(w1Param, w1Grad);

    /* Layer 1 bias b1 (1x2, zeros). */
    size_t *b1Dims = reserveMemory(2 * sizeof(size_t));
    b1Dims[0] = 1;
    b1Dims[1] = 2;
    size_t *b1Order = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, b1Order);
    shape_t *b1Shape = reserveMemory(sizeof(shape_t));
    setShape(b1Shape, b1Dims, 2, b1Order);
    tensor_t *b1Param = initTensor(b1Shape, quantizationInitFloat(), NULL);
    tensor_t *b1Grad = gradInitFloat(b1Param, NULL);
    parameter_t *b1 = parameterInit(b1Param, b1Grad);

    layer_t *linear1 = buildBorrowedLinearLayer(w1, b1, q);
    layer_t *softmax = softmaxLayerInit(&lq);

    layer_t *model[] = {linear0, relu, linear1, softmax};
    size_t sizeModel = 4;

    /* Input (1x3). */
    size_t *inputDims = reserveMemory(2 * sizeof(size_t));
    inputDims[0] = 1;
    inputDims[1] = 3;
    size_t *inputOrder = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, inputOrder);
    shape_t *inputShape = reserveMemory(sizeof(shape_t));
    setShape(inputShape, inputDims, 2, inputOrder);
    tensor_t *input = initTensor(inputShape, quantizationInitFloat(), NULL);
    tensorFillFromFloatBuffer(input, (float[]){1.0f, 2.0f, 3.0f}, 3);

    /* Label (1x2). */
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
        model, sizeModel,
        (lossConfig_t){.funcType = CROSS_ENTROPY, .backwardReduction = REDUCTION_SUM},
        REDUCTION_SUM, input, label);

    /* CAPTURE. The original test checks that b1Grad has at least one nonzero
     * value AFTER the backward pass; we capture that boolean before frees so
     * the post-free freeParameter(b1) doesn't zero or invalidate b1Grad. */
    bool capturedNotNull = (stats != NULL);
    float capturedLoss = stats ? stats->loss : -1.0f;
    bool capturedAnyNonZero = false;
    if (b1Grad && b1Grad->data) {
        float *vals = (float *)b1Grad->data;
        for (size_t i = 0; i < 2; i++) {
            if (vals[i] != 0.0f) {
                capturedAnyNonZero = true;
                break;
            }
        }
    }

    /* FREE in reverse-init order. */
    freeTrainingStats(stats);
    freeTensor(label);
    freeTensor(input);
    freeSoftmaxLayer(softmax);
    freeLinearLayerShellOnly(linear1);
    freeParameter(b1);
    freeParameter(w1);
    freeReluLayer(relu);
    freeLinearLayerShellOnly(linear0);
    freeParameter(b0);
    freeParameter(w0);
    freeQuantization(q);

    /* ASSERT on captured. */
    TEST_ASSERT_TRUE(capturedNotNull);
    TEST_ASSERT_TRUE(capturedLoss >= 0.0f);
    TEST_ASSERT_TRUE(capturedAnyNonZero);
}

/*! Integration test: run multiple training steps to verify grad accumulation is stable. */
void testMultiLayerTraining_MultipleSteps_GradsAccumulate() {
    quantization_t *q = quantizationInitFloat();
    layerQuant_t lq;
    layerQuantInitUniform(&lq, q);

    /* Layer 0 weights w0 (4x3). */
    size_t *w0Dims = reserveMemory(2 * sizeof(size_t));
    w0Dims[0] = 4;
    w0Dims[1] = 3;
    size_t *w0Order = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, w0Order);
    shape_t *w0Shape = reserveMemory(sizeof(shape_t));
    setShape(w0Shape, w0Dims, 2, w0Order);
    tensor_t *w0Param = initTensor(w0Shape, quantizationInitFloat(), NULL);
    tensorFillFromFloatBuffer(
        w0Param, (float[]){0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.0f, 1.1f, 1.2f},
        12);
    tensor_t *w0Grad = gradInitFloat(w0Param, NULL);
    parameter_t *w0 = parameterInit(w0Param, w0Grad);

    /* Layer 0 bias b0 (1x4, zeros). */
    size_t *b0Dims = reserveMemory(2 * sizeof(size_t));
    b0Dims[0] = 1;
    b0Dims[1] = 4;
    size_t *b0Order = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, b0Order);
    shape_t *b0Shape = reserveMemory(sizeof(shape_t));
    setShape(b0Shape, b0Dims, 2, b0Order);
    tensor_t *b0Param = initTensor(b0Shape, quantizationInitFloat(), NULL);
    tensor_t *b0Grad = gradInitFloat(b0Param, NULL);
    parameter_t *b0 = parameterInit(b0Param, b0Grad);

    layer_t *linear0 = buildBorrowedLinearLayer(w0, b0, q);
    layer_t *relu = reluLayerInit(&lq);

    /* Layer 1 weights w1 (2x4). */
    size_t *w1Dims = reserveMemory(2 * sizeof(size_t));
    w1Dims[0] = 2;
    w1Dims[1] = 4;
    size_t *w1Order = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, w1Order);
    shape_t *w1Shape = reserveMemory(sizeof(shape_t));
    setShape(w1Shape, w1Dims, 2, w1Order);
    tensor_t *w1Param = initTensor(w1Shape, quantizationInitFloat(), NULL);
    tensorFillFromFloatBuffer(w1Param, (float[]){0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f},
                              8);
    tensor_t *w1Grad = gradInitFloat(w1Param, NULL);
    parameter_t *w1 = parameterInit(w1Param, w1Grad);

    /* Layer 1 bias b1 (1x2). */
    size_t *b1Dims = reserveMemory(2 * sizeof(size_t));
    b1Dims[0] = 1;
    b1Dims[1] = 2;
    size_t *b1Order = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, b1Order);
    shape_t *b1Shape = reserveMemory(sizeof(shape_t));
    setShape(b1Shape, b1Dims, 2, b1Order);
    tensor_t *b1Param = initTensor(b1Shape, quantizationInitFloat(), NULL);
    tensor_t *b1Grad = gradInitFloat(b1Param, NULL);
    parameter_t *b1 = parameterInit(b1Param, b1Grad);

    layer_t *linear1 = buildBorrowedLinearLayer(w1, b1, q);
    layer_t *softmax = softmaxLayerInit(&lq);

    layer_t *model[] = {linear0, relu, linear1, softmax};
    size_t sizeModel = 4;

    /* Optimizer takes references to w0/b0/w1/b1 — its free will cascade. */
    quantization_t *momentumQ = quantizationInitFloat();
    optimizer_t *sgd =
        sgdMCreateOptim(0.01f, 0.f, 0.f, model, sizeModel, momentumQ,
                        (arithmetic_t){.type = ARITH_FLOAT32, .roundingMode = HALF_AWAY});
    optimizerFunctions_t sgdFns = optimizerFunctions[SGD_M];

    /* Input (1x3). */
    size_t *inputDims = reserveMemory(2 * sizeof(size_t));
    inputDims[0] = 1;
    inputDims[1] = 3;
    size_t *inputOrder = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, inputOrder);
    shape_t *inputShape = reserveMemory(sizeof(shape_t));
    setShape(inputShape, inputDims, 2, inputOrder);
    tensor_t *input = initTensor(inputShape, quantizationInitFloat(), NULL);
    tensorFillFromFloatBuffer(input, (float[]){1.0f, 2.0f, 3.0f}, 3);

    /* Label (1x2). */
    size_t *labelDims = reserveMemory(2 * sizeof(size_t));
    labelDims[0] = 1;
    labelDims[1] = 2;
    size_t *labelOrder = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, labelOrder);
    shape_t *labelShape = reserveMemory(sizeof(shape_t));
    setShape(labelShape, labelDims, 2, labelOrder);
    tensor_t *label = initTensor(labelShape, quantizationInitFloat(), NULL);
    tensorFillFromFloatBuffer(label, (float[]){1.0f, 0.0f}, 2);

    /* Run 3 training steps. CAPTURE per-step assertions into a per-step
     * tracking array; assert at end after all frees. */
    bool capturedNotNull[3];
    float capturedLoss[3];
    for (size_t step = 0; step < 3; step++) {
        trainingStats_t *stats = calculateGradsSequential(
            model, sizeModel,
            (lossConfig_t){.funcType = CROSS_ENTROPY, .backwardReduction = REDUCTION_SUM},
            REDUCTION_SUM, input, label);
        capturedNotNull[step] = (stats != NULL);
        capturedLoss[step] = stats ? stats->loss : -1.0f;
        freeTrainingStats(stats);

        sgdFns.step(sgd);
        sgdFns.zero(sgd);
    }

    /* FREE in reverse-init order.
     * NOTE: freeOptim cascades to w0, b0, w1, b1 via freeParameter (per
     * SgdApi.c:85-93). Do NOT also call freeParameter(w0/b0/w1/b1) here — it
     * would be a double-free. */
    freeTensor(label);
    freeTensor(input);
    freeOptim(sgd);
    freeSoftmaxLayer(softmax);
    freeLinearLayerShellOnly(linear1);
    freeReluLayer(relu);
    freeLinearLayerShellOnly(linear0);
    freeQuantization(momentumQ);
    freeQuantization(q);

    /* ASSERT on captured. */
    for (size_t step = 0; step < 3; step++) {
        TEST_ASSERT_TRUE(capturedNotNull[step]);
        TEST_ASSERT_TRUE(capturedLoss[step] >= 0.0f);
    }
}

/*! BFP epic PR1 capstone: Linear(3->4)->Relu->Linear(4->2)+MSE, the same
 *  harness idioms as testMultiLayerTraining_MultipleSteps_GradsAccumulate
 *  above (raw shape/tensor allocation, buildBorrowedLinearLayer, plain SGD),
 *  but BOTH Linear weight PARAMs are requantized in place (FLOAT32-init +
 *  requantizeTensorInPlace, the #270 pattern) to grouped BFP -- one group
 *  per output row (numGroups=outFeatures, groupSize=inFeatures), so
 *  numGroups*groupSize equals the weight's element count exactly (the
 *  Task-6 validateBfpQConfigShape gate). Forward dequantizes the BFP weight
 *  through the float bridge (arithmeticFromQuantization(BFP)==ARITH_FLOAT32,
 *  Task 8 Arm 1); backward computes FLOAT32 grads untouched; the optimizer's
 *  OUT_WRITE write-back re-quantizes the updated weight fresh into BFP via
 *  the conversionMatrix diagonal, honoring writeBackRounding through the
 *  target's storage slot (Task 8 Arm 2) -- textbook fake-quant training. */
void testBfpFakeQuantTrainingLossDecreasesAndGridMoves(void) {
    /* BFP epic PR2 Task 8 carry-over: the SR_HALF_AWAY configs below draw from
     * the module-global RNG -- seed it so the run is reproducible. */
    rngSetSeed(20250811u);
    quantization_t *q = quantizationInitFloat();
    layerQuant_t lq;
    layerQuantInitUniform(&lq, q);

    /* Layer 0 weights w0 (4x3, outFeatures=4, inFeatures=3). */
    size_t *w0Dims = reserveMemory(2 * sizeof(size_t));
    w0Dims[0] = 4;
    w0Dims[1] = 3;
    size_t *w0Order = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, w0Order);
    shape_t *w0Shape = reserveMemory(sizeof(shape_t));
    setShape(w0Shape, w0Dims, 2, w0Order);
    tensor_t *w0Param = initTensor(w0Shape, quantizationInitFloat(), NULL);
    tensorFillFromFloatBuffer(
        w0Param, (float[]){0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.0f, 1.1f, 1.2f},
        12);
    tensor_t *w0Grad = gradInitFloat(w0Param, NULL);
    parameter_t *w0 = parameterInit(w0Param, w0Grad);

    /* Layer 0 bias b0 (1x4, zeros via initTensor). */
    size_t *b0Dims = reserveMemory(2 * sizeof(size_t));
    b0Dims[0] = 1;
    b0Dims[1] = 4;
    size_t *b0Order = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, b0Order);
    shape_t *b0Shape = reserveMemory(sizeof(shape_t));
    setShape(b0Shape, b0Dims, 2, b0Order);
    tensor_t *b0Param = initTensor(b0Shape, quantizationInitFloat(), NULL);
    tensor_t *b0Grad = gradInitFloat(b0Param, NULL);
    parameter_t *b0 = parameterInit(b0Param, b0Grad);

    layer_t *linear0 = buildBorrowedLinearLayer(w0, b0, q);
    layer_t *relu = reluLayerInit(&lq);

    /* Layer 1 weights w1 (2x4, outFeatures=2, inFeatures=4). */
    size_t *w1Dims = reserveMemory(2 * sizeof(size_t));
    w1Dims[0] = 2;
    w1Dims[1] = 4;
    size_t *w1Order = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, w1Order);
    shape_t *w1Shape = reserveMemory(sizeof(shape_t));
    setShape(w1Shape, w1Dims, 2, w1Order);
    tensor_t *w1Param = initTensor(w1Shape, quantizationInitFloat(), NULL);
    tensorFillFromFloatBuffer(w1Param, (float[]){0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f},
                              8);
    tensor_t *w1Grad = gradInitFloat(w1Param, NULL);
    parameter_t *w1 = parameterInit(w1Param, w1Grad);

    /* Layer 1 bias b1 (1x2, zeros via initTensor). */
    size_t *b1Dims = reserveMemory(2 * sizeof(size_t));
    b1Dims[0] = 1;
    b1Dims[1] = 2;
    size_t *b1Order = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, b1Order);
    shape_t *b1Shape = reserveMemory(sizeof(shape_t));
    setShape(b1Shape, b1Dims, 2, b1Order);
    tensor_t *b1Param = initTensor(b1Shape, quantizationInitFloat(), NULL);
    tensor_t *b1Grad = gradInitFloat(b1Param, NULL);
    parameter_t *b1 = parameterInit(b1Param, b1Grad);

    layer_t *linear1 = buildBorrowedLinearLayer(w1, b1, q);

    layer_t *model[] = {linear0, relu, linear1};
    size_t sizeModel = 3;

    /* Requantize BOTH weight PARAMs to grouped BFP -- per-output-row blocks
     * (numGroups=outFeatures, groupSize=inFeatures); bias/grads stay FLOAT32. */
    quantization_t *w0BfpQ = quantizationInitBfpGrouped(8, 8, SR_HALF_AWAY, 4, 3);
    requantizeTensorInPlace(w0Param, w0BfpQ);
    freeQuantization(w0BfpQ); /* getQLike deep-clones -- template unused after */
    quantization_t *w1BfpQ = quantizationInitBfpGrouped(8, 8, SR_HALF_AWAY, 2, 4);
    requantizeTensorInPlace(w1Param, w1BfpQ);
    freeQuantization(w1BfpQ);

    uint8_t w0ExpBefore[4];
    uint8_t w1ExpBefore[2];
    memcpy(w0ExpBefore, ((bfpQConfig_t *)w0Param->quantization->qConfig)->exponents, 4);
    memcpy(w1ExpBefore, ((bfpQConfig_t *)w1Param->quantization->qConfig)->exponents, 2);

    /* Optimizer takes references to w0/b0/w1/b1 — its free will cascade. */
    quantization_t *momentumQ = quantizationInitFloat();
    optimizer_t *sgd =
        sgdMCreateOptim(0.002f, 0.f, 0.f, model, sizeModel, momentumQ,
                        (arithmetic_t){.type = ARITH_FLOAT32, .roundingMode = HALF_AWAY});
    optimizerFunctions_t sgdFns = optimizerFunctions[SGD_M];

    /* Input (1x3). */
    size_t *inputDims = reserveMemory(2 * sizeof(size_t));
    inputDims[0] = 1;
    inputDims[1] = 3;
    size_t *inputOrder = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, inputOrder);
    shape_t *inputShape = reserveMemory(sizeof(shape_t));
    setShape(inputShape, inputDims, 2, inputOrder);
    tensor_t *input = initTensor(inputShape, quantizationInitFloat(), NULL);
    tensorFillFromFloatBuffer(input, (float[]){1.0f, 2.0f, 3.0f}, 3);

    /* Label (1x2, MSE regression target). */
    size_t *labelDims = reserveMemory(2 * sizeof(size_t));
    labelDims[0] = 1;
    labelDims[1] = 2;
    size_t *labelOrder = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, labelOrder);
    shape_t *labelShape = reserveMemory(sizeof(shape_t));
    setShape(labelShape, labelDims, 2, labelOrder);
    tensor_t *label = initTensor(labelShape, quantizationInitFloat(), NULL);
    tensorFillFromFloatBuffer(label, (float[]){0.2f, -0.3f}, 2);

    /* Run STEPS training steps. CAPTURE per-step loss; assert after frees. */
    size_t STEPS = 20;
    float firstLoss = NAN;
    float lastLoss = NAN;
    for (size_t step = 0; step < STEPS; step++) {
        trainingStats_t *stats = calculateGradsSequential(model, sizeModel, defaultLossConfig(MSE),
                                                          REDUCTION_MEAN, input, label);
        if (step == 0) {
            firstLoss = stats->loss;
        }
        lastLoss = stats->loss;
        freeTrainingStats(stats);

        sgdFns.step(sgd);
        sgdFns.zero(sgd);
    }

    /* CAPTURE post-training state before frees. */
    bool stillBfp = w0Param->quantization->type == BFP && w1Param->quantization->type == BFP;
    bfpQConfig_t *w0QC = w0Param->quantization->qConfig;
    bfpQConfig_t *w1QC = w1Param->quantization->qConfig;
    bool geometryUnchanged =
        w0QC->numGroups == 4 && w0QC->groupSize == 3 && w0QC->mantissaBits == 8 &&
        w0QC->exponentBits == 8 && w1QC->numGroups == 2 && w1QC->groupSize == 4 &&
        w1QC->mantissaBits == 8 && w1QC->exponentBits == 8 && w0Param->shape->dimensions[0] == 4 &&
        w0Param->shape->dimensions[1] == 3 && w1Param->shape->dimensions[0] == 2 &&
        w1Param->shape->dimensions[1] == 4;
    bool anyExponentChanged = false;
    for (size_t i = 0; i < 4; i++) {
        if (w0QC->exponents[i] != w0ExpBefore[i]) {
            anyExponentChanged = true;
        }
    }
    for (size_t i = 0; i < 2; i++) {
        if (w1QC->exponents[i] != w1ExpBefore[i]) {
            anyExponentChanged = true;
        }
    }
    bool gradsStillFloat =
        w0Grad->quantization->type == FLOAT32 && w1Grad->quantization->type == FLOAT32;

    /* FREE in reverse-init order.
     * NOTE: freeOptim cascades to w0, b0, w1, b1 via freeParameter (per
     * SgdApi.c:85-93). Do NOT also call freeParameter(w0/b0/w1/b1) here — it
     * would be a double-free. freeParameter->freeTensor->freeQuantization
     * already has a BFP arm (Task 6), so the requantized weight tensors free
     * cleanly through the ordinary cascade. */
    freeTensor(label);
    freeTensor(input);
    freeOptim(sgd);
    freeLinearLayerShellOnly(linear1);
    freeReluLayer(relu);
    freeLinearLayerShellOnly(linear0);
    freeQuantization(momentumQ);
    freeQuantization(q);

    /* ASSERT on captured. */
    TEST_ASSERT_TRUE_MESSAGE(isfinite(firstLoss) && isfinite(lastLoss),
                             "BFP fake-quant training losses must be finite");
    TEST_ASSERT_TRUE_MESSAGE(lastLoss < firstLoss,
                             "BFP fake-quant training must converge (loss must decrease)");
    TEST_ASSERT_TRUE_MESSAGE(stillBfp, "weight params must remain BFP after training");
    TEST_ASSERT_TRUE_MESSAGE(geometryUnchanged,
                             "weight geometry/widths must be unchanged after training");
    TEST_ASSERT_TRUE_MESSAGE(anyExponentChanged,
                             "the optimizer's OUT_WRITE requant must re-derive at least one "
                             "group's exponent (the grid must move)");
    TEST_ASSERT_TRUE_MESSAGE(gradsStillFloat, "grad storage must stay FLOAT32 (default, #261)");
}

/* ===========================================================================
 * BFP epic PR2 Task 8 capstone: BFP *WIRES* (not just params).
 * ======================================================================== */

#define BFP_WIRE_MAX_GROUPS 16

typedef struct bfpWireCapture {
    bool seen;
    int type;
    size_t numElements;
    size_t numGroups;
    size_t groupSize;
    uint8_t mantissaBits;
    uint8_t exponentBits;
    uint8_t exponents[BFP_WIRE_MAX_GROUPS];
} bfpWireCapture_t;

/* The hidden wire lives ONLY inside calculateGradsImpl -- initLayerOutputs
 * allocates it and deInitLayerOutputs frees it before the call returns, and
 * trainingStats->output carries the FINAL (FLOAT32) wire. The layer-0 "fwd"
 * probe is therefore the only place the BFP wire is observable, and it fires
 * right after the OUT_WRITE epilogue derived the exponents. Captures the FIRST
 * forward only (the zero-state comparison must see step 1's grid). */
static void captureLayer0ForwardWire(void *ctx, size_t layerIdx, layerType_t layerType,
                                     const char *phase, tensor_t *tensor) {
    (void)layerType;
    bfpWireCapture_t *cap = ctx;
    if (cap->seen || layerIdx != 0 || strcmp(phase, "fwd") != 0) {
        return;
    }
    cap->seen = true;
    cap->type = (int)tensor->quantization->type;
    cap->numElements = calcNumberOfElementsByTensor(tensor);
    if (tensor->quantization->type != BFP) {
        return;
    }
    bfpQConfig_t *qc = tensor->quantization->qConfig;
    cap->numGroups = qc->numGroups;
    cap->groupSize = qc->groupSize;
    cap->mantissaBits = qc->mantissaBits;
    cap->exponentBits = qc->exponentBits;
    size_t copyGroups = qc->numGroups < BFP_WIRE_MAX_GROUPS ? qc->numGroups : BFP_WIRE_MAX_GROUPS;
    memcpy(cap->exponents, qc->exponents, copyGroups);
}

typedef struct bfpWireFixture {
    quantization_t *floatQ;
    quantization_t *bfpWireQ;
    quantization_t *momentumQ;
    parameter_t *w0;
    parameter_t *b0;
    parameter_t *w1;
    parameter_t *b1;
    layer_t *linear0;
    layer_t *linear1;
    layer_t *model[2];
    optimizer_t *sgd;
    tensor_t *input;
    tensor_t *label;
} bfpWireFixture_t;

/* Deterministic FLOAT32 parameter, values base, base+step, base+2*step, ... */
static parameter_t *buildRampParam2D(size_t d0, size_t d1, float base, float step) {
    size_t *dims = reserveMemory(2 * sizeof(size_t));
    dims[0] = d0;
    dims[1] = d1;
    size_t *order = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, order);
    shape_t *shape = reserveMemory(sizeof(shape_t));
    setShape(shape, dims, 2, order);
    tensor_t *param = initTensor(shape, quantizationInitFloat(), NULL);
    size_t n = d0 * d1;
    float values[n];
    for (size_t i = 0; i < n; i++) {
        values[i] = base + step * (float)i;
    }
    tensorFillFromFloatBuffer(param, values, n);
    return parameterInit(param, gradInitFloat(param, NULL));
}

static tensor_t *buildFloatTensor2D(size_t d0, size_t d1, const float *values) {
    size_t *dims = reserveMemory(2 * sizeof(size_t));
    dims[0] = d0;
    dims[1] = d1;
    size_t *order = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, order);
    shape_t *shape = reserveMemory(sizeof(shape_t));
    setShape(shape, dims, 2, order);
    tensor_t *t = initTensor(shape, quantizationInitFloat(), NULL);
    tensorFillFromFloatBuffer(t, (float *)values, d0 * d1);
    return t;
}

/*! Linear(3->hidden) -> Linear(hidden->2) + MSE, with layer 0's FORWARD WIRE
 *  declared grouped BFP. NO Relu in the model: Relu/Dropout/Flatten are guarded
 *  against BFP storage until epic PR4, so a BFP wire may only ever land between
 *  two funnel layers.
 *
 *  Pre-flip (arithmeticFromQuantization still maps BFP -> ARITH_FLOAT32) this is
 *  textbook fake-quant: layer 0's GEMM runs in float, the funnel's OUT_WRITE
 *  epilogue packs the activations into the BFP wire and DERIVES its exponents,
 *  and layer 1's IN_READ dequantizes them back. propLossQ, both weight/bias
 *  params and all grads stay FLOAT32.
 *
 *  `templateNumGroups` is the numGroups the caller declares in the template --
 *  deliberately decoupled from the truth: the allocator DERIVES
 *  numGroups = wireElements / wireGroupSize (plan Decision 5). */
static void buildBfpWireFixture(bfpWireFixture_t *f, size_t hidden, size_t templateNumGroups,
                                size_t wireGroupSize) {
    f->floatQ = quantizationInitFloat();
    f->bfpWireQ = quantizationInitBfpGrouped(6, 8, SR_HALF_AWAY, templateNumGroups, wireGroupSize);

    f->w0 = buildRampParam2D(hidden, 3, 0.1f, 0.05f);
    f->b0 = buildRampParam2D(1, hidden, 0.0f, 0.0f);
    f->w1 = buildRampParam2D(2, hidden, 0.1f, 0.05f);
    f->b1 = buildRampParam2D(1, 2, 0.0f, 0.0f);

    f->linear0 = buildBorrowedLinearLayer(f->w0, f->b0, f->floatQ);
    /* Only the forward wire goes BFP; propLossQ and the grad math stay FLOAT32.
     * forwardMath is DERIVED from the wire template (pre-flip: ARITH_FLOAT32),
     * so this test tracks the derivation rather than hardcoding it. */
    f->linear0->config->linear->outputQ = f->bfpWireQ;
    f->linear0->config->linear->forwardMath = arithmeticFromQuantization(f->bfpWireQ);
    f->linear1 = buildBorrowedLinearLayer(f->w1, f->b1, f->floatQ);
    f->model[0] = f->linear0;
    f->model[1] = f->linear1;

    f->momentumQ = quantizationInitFloat();
    f->sgd = sgdMCreateOptim(0.002f, 0.f, 0.f, f->model, 2, f->momentumQ,
                             (arithmetic_t){.type = ARITH_FLOAT32, .roundingMode = HALF_AWAY});

    f->input = buildFloatTensor2D(1, 3, (float[]){1.0f, 2.0f, 3.0f});
    f->label = buildFloatTensor2D(1, 2, (float[]){0.2f, -0.3f});
}

/* Reverse-init order; freeOptim cascades into w0/b0/w1/b1 (SgdApi), so the
 * layers are torn down shell-only. */
static void freeBfpWireFixture(bfpWireFixture_t *f) {
    freeTensor(f->label);
    freeTensor(f->input);
    freeOptim(f->sgd);
    freeLinearLayerShellOnly(f->linear1);
    freeLinearLayerShellOnly(f->linear0);
    freeQuantization(f->momentumQ);
    freeQuantization(f->bfpWireQ);
    freeQuantization(f->floatQ);
}

/*! THE Task 8 capstone. Hidden wire is [1, 6] -> 6 elements; the template
 *  declares numGroups=2, which is WRONG for this wire -- the allocator derives
 *  6/2 = 3 groups (Decision 5). Trains 20 fake-quant SGD steps. */
void testBfpWireFakeQuantTrainingLossDecreasesAndWirePacks(void) {
    rngSetSeed(4242u);
    bfpWireFixture_t f;
    buildBfpWireFixture(&f, /*hidden=*/6, /*templateNumGroups=*/2, /*wireGroupSize=*/2);

    bfpWireCapture_t cap = {0};
    optimizerFunctions_t sgdFns = optimizerFunctions[SGD_M];
    float firstLoss = NAN;
    float lastLoss = NAN;
    for (size_t step = 0; step < 20; step++) {
        trainingStats_t *stats = tracedGrads(f.model, 2, defaultLossConfig(MSE), REDUCTION_MEAN,
                                             f.input, f.label, captureLayer0ForwardWire, &cap);
        if (step == 0) {
            firstLoss = stats->loss;
        }
        lastLoss = stats->loss;
        freeTrainingStats(stats);
        sgdFns.step(f.sgd);
        sgdFns.zero(f.sgd);
    }

    /* CAPTURE (cap is already a value copy) then FREE, assert last. */
    bool anyExponentMoved = false;
    uint8_t zeroState = (uint8_t)((1 << (8 - 1)) - 1); /* exponentBits=8 -> bias 127 */
    for (size_t g = 0; g < cap.numGroups && g < BFP_WIRE_MAX_GROUPS; g++) {
        if (cap.exponents[g] != zeroState) {
            anyExponentMoved = true;
        }
    }

    freeBfpWireFixture(&f);

    TEST_ASSERT_TRUE_MESSAGE(cap.seen, "layer-0 forward probe must have fired");
    TEST_ASSERT_EQUAL_INT_MESSAGE(BFP, cap.type, "the hidden wire tensor must be BFP-stored");
    TEST_ASSERT_EQUAL_UINT_MESSAGE(6, cap.numElements, "hidden wire is [1, 6]");
    TEST_ASSERT_EQUAL_UINT_MESSAGE(3, cap.numGroups,
                                   "wire numGroups must be DERIVED (6 elements / groupSize 2 = 3), "
                                   "not taken from the template's numGroups=2");
    TEST_ASSERT_EQUAL_UINT_MESSAGE(2, cap.groupSize, "groupSize comes from the template");
    TEST_ASSERT_EQUAL_UINT8_MESSAGE(6, cap.mantissaBits, "mantissa width comes from the template");
    TEST_ASSERT_EQUAL_UINT8_MESSAGE(8, cap.exponentBits, "exponent width comes from the template");
    TEST_ASSERT_TRUE_MESSAGE(anyExponentMoved,
                             "the forward OUT_WRITE must derive the wire's grid: at least one "
                             "group exponent must leave the zero state");
    TEST_ASSERT_TRUE_MESSAGE(isfinite(firstLoss) && isfinite(lastLoss),
                             "BFP-wire fake-quant training losses must be finite");
    TEST_ASSERT_TRUE_MESSAGE(lastLoss < firstLoss,
                             "BFP-wire fake-quant training must converge (loss must decrease)");
}

/*! Decision 5, pinned hard: a template numGroups that cannot possibly describe
 *  the wire (7 groups of 2 = 14 elements, wire has 6) is IGNORED -- geometry is
 *  derived from the wire's own element count. Without the derivation the
 *  allocator would build a 7-group config over a 6-element buffer. */
void testBfpWireGeometryIgnoresTemplateNumGroups(void) {
    rngSetSeed(4242u);
    bfpWireFixture_t f;
    buildBfpWireFixture(&f, /*hidden=*/6, /*templateNumGroups=*/7, /*wireGroupSize=*/2);

    bfpWireCapture_t cap = {0};
    trainingStats_t *stats = tracedGrads(f.model, 2, defaultLossConfig(MSE), REDUCTION_MEAN,
                                         f.input, f.label, captureLayer0ForwardWire, &cap);
    freeTrainingStats(stats);
    freeBfpWireFixture(&f);

    TEST_ASSERT_TRUE(cap.seen);
    TEST_ASSERT_EQUAL_INT(BFP, cap.type);
    TEST_ASSERT_EQUAL_UINT_MESSAGE(3, cap.numGroups,
                                   "derived 6/2 = 3 must win over the template's numGroups=7");
    TEST_ASSERT_EQUAL_UINT(2, cap.groupSize);
}

/*! A groupSize that does not divide the wire has no valid derived geometry --
 *  fail fast with a guided message instead of silently truncating. Wire is
 *  [1, 9] with groupSize 2: an un-guarded floor division would yield a {4, 2}
 *  config covering only 8 of the 9 elements, and the packer would index
 *  exponents[4] past the array. */
void testInitLayerOutputsBfpGroupSizeMismatchDies(void) {
    bfpWireFixture_t f;
    buildBfpWireFixture(&f, /*hidden=*/9, /*templateNumGroups=*/2, /*wireGroupSize=*/2);

    ASSERT_EXITS_WITH_FAILURE(freeTrainingStats(calculateGradsSequential(
        f.model, 2, defaultLossConfig(MSE), REDUCTION_MEAN, f.input, f.label)));

    freeBfpWireFixture(&f);
}

/* Twin of captureLayer0ForwardWire for the BACKWARD wire: the "agrad" probe at
 * layer 0 hands over the dx tensor `initGradTensor` built from layer 1's
 * propLossQ, after layer 1's backward wrote into it. Same short-lived-tensor
 * argument as the forward sink -- deInitGradTensor frees it before the call
 * returns. */
static void captureLayer0BackwardWire(void *ctx, size_t layerIdx, layerType_t layerType,
                                      const char *phase, tensor_t *tensor) {
    (void)layerType;
    bfpWireCapture_t *cap = ctx;
    if (cap->seen || layerIdx != 0 || strcmp(phase, "agrad") != 0) {
        return;
    }
    cap->seen = true;
    cap->type = (int)tensor->quantization->type;
    cap->numElements = calcNumberOfElementsByTensor(tensor);
    if (tensor->quantization->type != BFP) {
        return;
    }
    bfpQConfig_t *qc = tensor->quantization->qConfig;
    cap->numGroups = qc->numGroups;
    cap->groupSize = qc->groupSize;
    cap->mantissaBits = qc->mantissaBits;
    cap->exponentBits = qc->exponentBits;
    size_t copyGroups = qc->numGroups < BFP_WIRE_MAX_GROUPS ? qc->numGroups : BFP_WIRE_MAX_GROUPS;
    memcpy(cap->exponents, qc->exponents, copyGroups);
}

/* Move the BFP template off the forward wire and onto layer 1's dx wire: the
 * forward then runs entirely FLOAT32 (so the loss, which has no BFP arm, is
 * reachable) and the BFP allocation happens in initGradTensor instead --
 * initGradTensor(gradCurr, layerOutputs[1], backwardWireQ(linear1)). */
static void moveBfpTemplateToDxWire(bfpWireFixture_t *f) {
    f->linear0->config->linear->outputQ = f->floatQ;
    f->linear0->config->linear->forwardMath = arithmeticFromQuantization(f->floatQ);
    f->linear1->config->linear->propLossQ = f->bfpWireQ;
    f->linear1->config->linear->propLossMath = arithmeticFromQuantization(f->bfpWireQ);
}

/*! initGradTensor's BFP arm, live: the dx wire between the two Linears is
 *  [1, 6] -> 6 elements, groupSize 2 -> derived numGroups 3 (the template's
 *  numGroups=2 is ignored, same Decision 5 rule as the forward allocators).
 *  Pre-flip this is the dx-side fake-quant bridge: layer 1's backward OUT_WRITEs
 *  its dx into the BFP wire, layer 0's weight-grad GEMM IN_READs it back. */
void testBfpDxWireAllocatesThroughInitGradTensor(void) {
    rngSetSeed(4242u);
    bfpWireFixture_t f;
    buildBfpWireFixture(&f, /*hidden=*/6, /*templateNumGroups=*/2, /*wireGroupSize=*/2);
    moveBfpTemplateToDxWire(&f);

    bfpWireCapture_t cap = {0};
    trainingStats_t *stats = tracedGrads(f.model, 2, defaultLossConfig(MSE), REDUCTION_MEAN,
                                         f.input, f.label, captureLayer0BackwardWire, &cap);
    float loss = stats->loss;
    freeTrainingStats(stats);
    freeBfpWireFixture(&f);

    TEST_ASSERT_TRUE_MESSAGE(cap.seen, "layer-0 agrad probe must have fired");
    TEST_ASSERT_EQUAL_INT_MESSAGE(BFP, cap.type, "the dx wire tensor must be BFP-stored");
    TEST_ASSERT_EQUAL_UINT(6, cap.numElements);
    TEST_ASSERT_EQUAL_UINT_MESSAGE(3, cap.numGroups,
                                   "dx-wire numGroups must be DERIVED (6 / 2), not the template's");
    TEST_ASSERT_EQUAL_UINT(2, cap.groupSize);
    TEST_ASSERT_TRUE_MESSAGE(isfinite(loss), "the dx-wire BFP round trip must stay finite");
}

/*! initGradTensor's divisibility fail-fast (the dx-wire twin of
 *  testInitLayerOutputsBfpGroupSizeMismatchDies). Same discriminating fixture
 *  shape: a 9-element dx wire with groupSize 2, so that floor division yields
 *  the CONSTRUCTIBLE shape {4, 2} -- initBfpQConfigGrouped's own guard does not
 *  fire, and without this check the packer would index exponents[4] past a
 *  4-entry array. (A groupSize that floors to {1, n} would be caught by
 *  initBfpQConfigGrouped anyway and would make this test vacuous.) */
void testInitGradTensorBfpGroupSizeMismatchDies(void) {
    rngSetSeed(4242u);
    bfpWireFixture_t f;
    buildBfpWireFixture(&f, /*hidden=*/9, /*templateNumGroups=*/2, /*wireGroupSize=*/2);
    moveBfpTemplateToDxWire(&f);

    ASSERT_EXITS_WITH_FAILURE(freeTrainingStats(calculateGradsSequential(
        f.model, 2, defaultLossConfig(MSE), REDUCTION_MEAN, f.input, f.label)));

    freeBfpWireFixture(&f);
}

/*! Owning factories deep-copy outputQ/propLossQ, and for BFP that copy owns a
 *  fresh exponents block -- so the teardown must be freeQuantization: the old
 *  freeReservedMemory(qConfig) + freeReservedMemory(q) pair leaked exactly that
 *  block, once per Owning layer. Two independent assertions: reserveMemory's
 *  live-byte counter returns to its pre-factory mark (the leak itself -- a real
 *  check under ODT_MEM_PROFILE, which the unit_test_debug and unit_test_asan
 *  presets both enable; vacuously 0 == 0 without it, the UnitTestPpcaReplay
 *  precedent), and the caller's template survives untouched (the copy is
 *  independent -- freeing the layer must not reach into it). */
void testOwningFactoryBfpOutputQFreesExponents(void) {
    quantization_t *floatQ = quantizationInitFloat();
    quantization_t *bfpQ = quantizationInitBfpGrouped(8, 8, HALF_AWAY, 2, 2);
    layerQuant_t lq;
    layerQuantInitUniform(&lq, floatQ);
    lq.outputQ = bfpQ;

    size_t liveBytesBefore = memProfileMark();
    layer_t *layer = linearLayerInitOwning(
        &(linearInit_t){.inFeatures = 3, .outFeatures = 4, .bias = BIAS_TRUE}, &lq);
    freeLinearLayer(layer);
    size_t liveBytesAfter = memProfileMark();

    bfpQConfig_t *qc = bfpQ->qConfig;
    size_t capturedNumGroups = qc->numGroups;
    size_t capturedGroupSize = qc->groupSize;
    uint8_t capturedExponent0 = qc->exponents[0];

    freeQuantization(bfpQ);
    freeQuantization(floatQ);

    TEST_ASSERT_EQUAL_size_t_MESSAGE(liveBytesBefore, liveBytesAfter,
                                     "an Owning layer with a BFP outputQ must free every block it "
                                     "allocated -- including the deep-copied exponents");
    TEST_ASSERT_EQUAL_UINT_MESSAGE(2, capturedNumGroups,
                                   "the caller's BFP template must survive the layer's teardown");
    TEST_ASSERT_EQUAL_UINT(2, capturedGroupSize);
    TEST_ASSERT_EQUAL_UINT8_MESSAGE(127, capturedExponent0,
                                    "template exponents must be untouched (zero state, bias 127)");
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(testMultiLayerBackward_WithCrossEntropy_DoesNotCrash);
    RUN_TEST(testMultiLayerBackward_WithManualInit_DoesNotCrash);
    RUN_TEST(testMultiLayerTraining_MultipleSteps_GradsAccumulate);
    RUN_TEST(testBfpFakeQuantTrainingLossDecreasesAndGridMoves);
    RUN_TEST(testBfpWireFakeQuantTrainingLossDecreasesAndWirePacks);
    RUN_TEST(testBfpWireGeometryIgnoresTemplateNumGroups);
    RUN_TEST(testInitLayerOutputsBfpGroupSizeMismatchDies);
    RUN_TEST(testBfpDxWireAllocatesThroughInitGradTensor);
    RUN_TEST(testInitGradTensorBfpGroupSizeMismatchDies);
    RUN_TEST(testOwningFactoryBfpOutputQFreesExponents);
    return UNITY_END();
}
