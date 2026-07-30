#define SOURCE_FILE "UNIT_TEST_MULTI_LAYER_TRAINING"

#include <math.h>
#include <stddef.h>
#include <string.h>

#include "ArithmeticType.h"
#include "BorrowedLayer.h"
#include "CalculateGradsSequential.h"
#include "DataLoaderApi.h"
#include "Dataset.h"
#include "InferenceApi.h"
#include "LayerQuant.h"
#include "Linear.h"
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

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(testMultiLayerBackward_WithCrossEntropy_DoesNotCrash);
    RUN_TEST(testMultiLayerBackward_WithManualInit_DoesNotCrash);
    RUN_TEST(testMultiLayerTraining_MultipleSteps_GradsAccumulate);
    RUN_TEST(testBfpFakeQuantTrainingLossDecreasesAndGridMoves);
    return UNITY_END();
}
