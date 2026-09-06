#define SOURCE_FILE "UNIT_TEST_MULTI_LAYER_TRAINING"

#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "ArithmeticType.h"
#include "BorrowedLayer.h"
#include "CalculateGradsSequential.h"
#include "Conv1dApi.h"
#include "DataLoaderApi.h"
#include "Dataset.h"
#include "DeathTest.h"
#include "FlattenApi.h"
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
#include "TrainingEpochDefault.h"
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
 *  through the float bridge: every math slot here derives from the FLOAT32 `q`,
 *  so the Task 9 flip (BFP now derives ARITH_BFP) leaves this fake-quant
 *  profile untouched -- BFP is storage only, the compute is declared FLOAT32
 *  (Task 8 Arm 1); backward computes FLOAT32 grads untouched; the optimizer's
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
 *  Textbook fake-quant: layer 0's GEMM runs in float, the funnel's OUT_WRITE
 *  epilogue packs the activations into the BFP wire and DERIVES its exponents,
 *  and layer 1's IN_READ dequantizes them back. propLossQ, both weight/bias
 *  params and all grads stay FLOAT32.
 *
 *  Since the Task 9 derivation flip, fake-quant is EXPLICIT: forwardMath is
 *  pinned to {ARITH_FLOAT32, SR_HALF_AWAY} -- bit-identical to what deriving
 *  from the BFP template used to yield -- because deriving now selects the
 *  native ARITH_BFP arm, which fail-fasts on this fixture's FLOAT32-stored
 *  weights (Task 7 rule 1). Native forward has its own capstone
 *  (testBfpNativeForwardTrainingLossDecreasesAndGridMoves); the subject HERE
 *  is the wire ALLOCATOR's derived geometry, which is arithmetic-agnostic.
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
     * forwardMath is PINNED to the float bridge (see the fake-quant note in the
     * doc comment above): {ARITH_FLOAT32, SR_HALF_AWAY} is exactly what
     * arithmeticFromQuantization(bfpWireQ) returned before the Task 9 flip. */
    f->linear0->config->linear->outputQ = f->bfpWireQ;
    f->linear0->config->linear->forwardMath =
        (arithmetic_t){.type = ARITH_FLOAT32, .roundingMode = SR_HALF_AWAY};
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
 * initGradTensor(gradCurr, layerOutputs[1], backwardWireQ(linear1)).
 *
 * pinPropLossMath == true: the fake-quant bridge (both existing callers) --
 * propLossMath stays the FLOAT32 arithmeticFromQuantization(floatQ) already
 * set by buildBorrowedLinearLayer, re-pinned here only to match the wire's
 * SR_HALF_AWAY rounding.
 * pinPropLossMath == false (Task 9): deriving ARITH_BFP for propLossMath
 * selects Linear's native backward arm, which invokes the SAME width-anchor
 * rule as the forward (linearForward's rule 1): ANY ARITH_BFP math slot on a
 * layer requires that layer's OWN weights to be BFP-stored, not just the dx
 * wire's storage config. FLOAT32-init + requantizeTensorInPlace (#270), same
 * recipe buildBfpNativeFixture uses for layer 0's weights above -- one group
 * per output row ([2, 6] -> numGroups=2, groupSize=6). */
static void moveBfpTemplateToDxWire(bfpWireFixture_t *f, bool pinPropLossMath) {
    f->linear0->config->linear->outputQ = f->floatQ;
    f->linear0->config->linear->forwardMath = arithmeticFromQuantization(f->floatQ);
    f->linear1->config->linear->propLossQ = f->bfpWireQ;
    if (pinPropLossMath) {
        f->linear1->config->linear->propLossMath =
            (arithmetic_t){.type = ARITH_FLOAT32, .roundingMode = SR_HALF_AWAY};
    } else {
        f->linear1->config->linear->propLossMath = arithmeticFromQuantization(f->bfpWireQ);
        quantization_t *w1BfpQ = quantizationInitBfpGrouped(8, 8, SR_HALF_AWAY, 2, 6);
        requantizeTensorInPlace(getParamFromParameter(f->linear1->config->linear->weights), w1BfpQ);
        freeQuantization(w1BfpQ);
    }
}

/*! initGradTensor's BFP arm, live: the dx wire between the two Linears is
 *  [1, 6] -> 6 elements, groupSize 2 -> derived numGroups 3 (the template's
 *  numGroups=2 is ignored, same Decision 5 rule as the forward allocators).
 *  The dx-side fake-quant bridge (propLossMath pinned ARITH_FLOAT32, see
 *  moveBfpTemplateToDxWire): layer 1's backward OUT_WRITEs its dx into the BFP
 *  wire, layer 0's weight-grad GEMM IN_READs it back. */
void testBfpDxWireAllocatesThroughInitGradTensor(void) {
    rngSetSeed(4242u);
    bfpWireFixture_t f;
    buildBfpWireFixture(&f, /*hidden=*/6, /*templateNumGroups=*/2, /*wireGroupSize=*/2);
    moveBfpTemplateToDxWire(&f, /*pinPropLossMath=*/true);

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

/*! Boundary of the Decision-5 derivation: a template groupSize EQUAL to the
 *  wire's element count derives numGroups == 1 -- and one group spanning the
 *  whole tensor IS per-tensor blocking, whose only grammatical spelling is
 *  {1,0} (initBfpQConfigGrouped rejects {1,N}). The allocator must normalize
 *  to the per-tensor config instead of dying: the divisibility guard has
 *  passed, so its "pick a divisor" guidance was already followed. */
void testBfpWireGroupSizeEqualToWireElementsNormalizesToPerTensor(void) {
    rngSetSeed(4242u);
    bfpWireFixture_t f;
    buildBfpWireFixture(&f, /*hidden=*/6, /*templateNumGroups=*/2, /*wireGroupSize=*/6);

    bfpWireCapture_t cap = {0};
    trainingStats_t *stats = tracedGrads(f.model, 2, defaultLossConfig(MSE), REDUCTION_MEAN,
                                         f.input, f.label, captureLayer0ForwardWire, &cap);
    float loss = stats->loss;
    freeTrainingStats(stats);
    freeBfpWireFixture(&f);

    TEST_ASSERT_TRUE_MESSAGE(cap.seen, "layer-0 forward probe must have fired");
    TEST_ASSERT_EQUAL_INT_MESSAGE(BFP, cap.type, "the hidden wire tensor must be BFP-stored");
    TEST_ASSERT_EQUAL_UINT_MESSAGE(1, cap.numGroups,
                                   "groupSize == wire elements must derive ONE group");
    TEST_ASSERT_EQUAL_UINT_MESSAGE(
        0, cap.groupSize, "one whole-tensor group is per-tensor blocking -- canonical {1,0}");
    TEST_ASSERT_TRUE_MESSAGE(isfinite(loss), "the normalized wire must stay trainable");
}

/*! initGradTensor twin of the normalization above: the dx wire has 6 elements,
 *  template groupSize 6 -> per-tensor {1,0}, not a {1,6} grammar death. */
void testBfpDxWireGroupSizeEqualToWireElementsNormalizesToPerTensor(void) {
    rngSetSeed(4242u);
    bfpWireFixture_t f;
    buildBfpWireFixture(&f, /*hidden=*/6, /*templateNumGroups=*/2, /*wireGroupSize=*/6);
    moveBfpTemplateToDxWire(&f, /*pinPropLossMath=*/true);

    bfpWireCapture_t cap = {0};
    trainingStats_t *stats = tracedGrads(f.model, 2, defaultLossConfig(MSE), REDUCTION_MEAN,
                                         f.input, f.label, captureLayer0BackwardWire, &cap);
    float loss = stats->loss;
    freeTrainingStats(stats);
    freeBfpWireFixture(&f);

    TEST_ASSERT_TRUE_MESSAGE(cap.seen, "layer-0 agrad probe must have fired");
    TEST_ASSERT_EQUAL_INT_MESSAGE(BFP, cap.type, "the dx wire tensor must be BFP-stored");
    TEST_ASSERT_EQUAL_UINT_MESSAGE(1, cap.numGroups,
                                   "groupSize == wire elements must derive ONE group");
    TEST_ASSERT_EQUAL_UINT_MESSAGE(
        0, cap.groupSize, "one whole-tensor group is per-tensor blocking -- canonical {1,0}");
    TEST_ASSERT_TRUE_MESSAGE(isfinite(loss), "the normalized dx wire must stay trainable");
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
    moveBfpTemplateToDxWire(&f, /*pinPropLossMath=*/true);

    ASSERT_EXITS_WITH_FAILURE(freeTrainingStats(calculateGradsSequential(
        f.model, 2, defaultLossConfig(MSE), REDUCTION_MEAN, f.input, f.label)));

    freeBfpWireFixture(&f);
}

/*! Task 9 capstone: the last uncovered wire combination -- native ARITH_BFP
 *  backward WRITING into a BFP-stored dx wire through OUT_WRITE, not just
 *  reading one back (testBfpDxWireAllocatesThroughInitGradTensor's pinned-
 *  FLOAT32 bridge covers the read side). moveBfpTemplateToDxWire(&f,
 *  pinPropLossMath=false) derives layer 1's propLossMath as ARITH_BFP and
 *  requantizes its weights to BFP (the width-anchor rule 1 requires, see the
 *  helper's doc comment); layer 1's backward then dispatches the native
 *  propLossKernelBfp arm and its OUT_WRITE derives the dx wire's grid
 *  directly, no float bridge.
 *
 *  RED-by-construction: commenting out Linear.c's
 *  `case ARITH_BFP: return bfpKernel;` in linearBackwardKernelForArithmetic
 *  falls through to the dispatch default and kills the whole binary the
 *  instant this test's first backward pass reaches layer 1's propLoss call
 *  (PRINT_ERROR "Linear backward (propLoss): quantization type not
 *  implemented") -- verified, then restored, before this test was accepted
 *  green. */
void testBfpDxWireNativeBackwardTrains(void) {
    rngSetSeed(4242u);
    bfpWireFixture_t f;
    buildBfpWireFixture(&f, /*hidden=*/6, /*templateNumGroups=*/2, /*wireGroupSize=*/2);
    moveBfpTemplateToDxWire(&f, /*pinPropLossMath=*/false);

    bfpWireCapture_t cap = {0};
    optimizerFunctions_t sgdFns = optimizerFunctions[SGD_M];
    float firstLoss = NAN;
    float lastLoss = NAN;
    for (size_t step = 0; step < 5; step++) {
        trainingStats_t *stats = tracedGrads(f.model, 2, defaultLossConfig(MSE), REDUCTION_MEAN,
                                             f.input, f.label, captureLayer0BackwardWire, &cap);
        if (step == 0) {
            firstLoss = stats->loss;
        }
        lastLoss = stats->loss;
        freeTrainingStats(stats);
        sgdFns.step(f.sgd);
        sgdFns.zero(f.sgd);
    }

    /* CAPTURE, then FREE, then assert (Unity longjmps out of the first failure).
     * Guarded on cap.seen: an unfired probe leaves exponentBits at its zero-init
     * 0, and 1 << (0 - 1) is a negative shift (UB) -- the cap.seen assert below
     * already fails that case, so this loop is skipped rather than risking it. */
    bool wireExponentMoved = false;
    if (cap.seen) {
        uint8_t zeroState = (uint8_t)((1 << (cap.exponentBits - 1)) - 1);
        for (size_t g = 0; g < cap.numGroups && g < BFP_WIRE_MAX_GROUPS; g++) {
            if (cap.exponents[g] != zeroState) {
                wireExponentMoved = true;
            }
        }
    }

    freeBfpWireFixture(&f);

    TEST_ASSERT_TRUE_MESSAGE(cap.seen, "layer-0 agrad probe must have fired");
    TEST_ASSERT_EQUAL_INT_MESSAGE(BFP, cap.type,
                                  "the dx wire must be BFP-stored (native OUT_WRITE)");
    TEST_ASSERT_EQUAL_UINT(6, cap.numElements);
    TEST_ASSERT_EQUAL_UINT_MESSAGE(3, cap.numGroups,
                                   "dx-wire numGroups must be DERIVED (6 / 2), not the template's");
    TEST_ASSERT_EQUAL_UINT(2, cap.groupSize);
    TEST_ASSERT_TRUE_MESSAGE(wireExponentMoved,
                             "the native propLoss OUT_WRITE must derive the wire's grid: at "
                             "least one group exponent must leave the zero state");
    TEST_ASSERT_TRUE_MESSAGE(isfinite(firstLoss) && isfinite(lastLoss),
                             "native BFP dx-wire training losses must be finite");
    TEST_ASSERT_TRUE_MESSAGE(lastLoss < firstLoss,
                             "native BFP dx-wire training must converge (loss must decrease)");
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

/* ===========================================================================
 * BFP epic PR2 Task 9 capstone: NATIVE ARITH_BFP forward.
 * ======================================================================== */

typedef struct bfpNativeFixture {
    quantization_t *floatQ;
    quantization_t *bfpWireQ;
    quantization_t *momentumQ;
    layer_t *linear0;
    layer_t *linear1;
    layer_t *model[2];
    optimizer_t *sgd;
    tensor_t *input;
    tensor_t *label;
    /* What layerQuantInitUniform(bfpWireQ) DERIVED, captured before the
     * backward slots are pinned -- the flip-sensitive observable (every other
     * assertion in the capstone is storage-side and holds pre-flip too). */
    arithmetic_t derivedForward;
    arithmeticType_t derivedWeightGrad;
    arithmeticType_t derivedBiasGrad;
    arithmeticType_t derivedPropLoss;
} bfpNativeFixture_t;

/*! Linear(3->4) -> Linear(4->2) + MSE with layer 0 running NATIVE ARITH_BFP
 *  forward: its forward wire, weights and bias are all BFP, and the GEMM is
 *  matmulBfpTensors (block partials folded per same-exponent segment), not a
 *  float bridge over dequantized operands.
 *
 *  Layer 0's whole profile DERIVES from one grouped BFP template via
 *  layerQuantInitUniform -- which since the Task 9 flip yields ARITH_BFP in all
 *  FOUR math slots. Since epic PR3's native Linear backward,
 *  `pinWeightGradMath == false` leaves all four slots derived (fully native);
 *  `true` is the fake-quant-backward variant: all THREE backward slots pinned
 *  to ARITH_FLOAT32 + a FLOAT32 propLossQ.
 *
 *  Storage slots follow #270: parameters are FLOAT32-init (the factory rejects
 *  anything else) and reach BFP storage through requantizeTensorInPlace --
 *  mandatory here, since Task 7's rule 1 fail-fasts an ARITH_BFP forward with
 *  non-BFP weights (a FLOAT32 weight has no width source to stage at).
 *
 *  Layer 1 is entirely FLOAT32 (Decision 9: the loss-facing wire stays FLOAT32
 *  -- no loss function has a BFP arm before epic PR4); it consumes the BFP
 *  hidden wire through the funnel's IN_READ dequantization. No Relu: BFP
 *  storage is guarded out of Relu/Dropout/Flatten until epic PR4.
 *
 *  `weightGradStorage` (Task 6, #300 axis): NULL keeps the pre-Task-6 default
 *  (grads stay FLOAT32, #261); a non-NULL per-tensor BFP template opts layer
 *  0's weight grad into BFP storage end-to-end -- the load-bearing e2e knob
 *  both existing callers below leave unexercised. */
static void buildBfpNativeFixture(bfpNativeFixture_t *f, bool pinWeightGradMath,
                                  quantization_t *weightGradStorage) {
    f->floatQ = quantizationInitFloat();
    /* groupSize 2 over the [1, 4] hidden wire -> derived numGroups 2. */
    f->bfpWireQ = quantizationInitBfpGrouped(6, 8, SR_HALF_AWAY, 2, 2);

    layerQuant_t lq0;
    layerQuantInitUniform(&lq0, f->bfpWireQ);
    f->derivedForward = lq0.forwardMath;
    f->derivedWeightGrad = lq0.weightGradMath.type;
    f->derivedBiasGrad = lq0.biasGradMath.type;
    f->derivedPropLoss = lq0.propLossMath.type;

    if (pinWeightGradMath) {
        lq0.weightGradMath = (arithmetic_t){.type = ARITH_FLOAT32, .roundingMode = HALF_AWAY};
        lq0.biasGradMath = (arithmetic_t){.type = ARITH_FLOAT32, .roundingMode = HALF_AWAY};
        lq0.propLossMath = (arithmetic_t){.type = ARITH_FLOAT32, .roundingMode = HALF_AWAY};
        lq0.propLossQ = f->floatQ;
    }
    lq0.weightStorage = f->floatQ; /* #270: FLOAT32 init, then requantize below */
    lq0.biasStorage = f->floatQ;
    lq0.weightGradStorage = weightGradStorage;
    f->linear0 = linearLayerInit(
        &(linearInit_t){.inFeatures = 3, .outFeatures = 4, .bias = BIAS_TRUE}, &lq0);

    /* Weights: one group per output row (4 rows x 3 in-features == the element
     * count, the validateBfpQConfigShape gate). Bias: per-tensor {1, 0} -- the
     * matmul dequantizes the bias seed through its own group scale, so its
     * widths need not match the weights'. */
    quantization_t *w0BfpQ = quantizationInitBfpGrouped(8, 8, SR_HALF_AWAY, 4, 3);
    requantizeTensorInPlace(getParamFromParameter(f->linear0->config->linear->weights), w0BfpQ);
    freeQuantization(w0BfpQ);
    quantization_t *b0BfpQ = quantizationInitBfp(8, 8, SR_HALF_AWAY);
    requantizeTensorInPlace(getParamFromParameter(f->linear0->config->linear->bias), b0BfpQ);
    freeQuantization(b0BfpQ);

    layerQuant_t lq1;
    layerQuantInitUniform(&lq1, f->floatQ);
    f->linear1 = linearLayerInit(
        &(linearInit_t){.inFeatures = 4, .outFeatures = 2, .bias = BIAS_TRUE}, &lq1);

    f->model[0] = f->linear0;
    f->model[1] = f->linear1;

    f->momentumQ = quantizationInitFloat();
    f->sgd = sgdMCreateOptim(0.002f, 0.f, 0.f, f->model, 2, f->momentumQ,
                             (arithmetic_t){.type = ARITH_FLOAT32, .roundingMode = HALF_AWAY});

    f->input = buildFloatTensor2D(1, 3, (float[]){1.0f, 2.0f, 3.0f});
    f->label = buildFloatTensor2D(1, 2, (float[]){0.2f, -0.3f});
}

/* Reverse-init order; freeOptim cascades into every parameter the factories
 * allocated (SgdApi), so the layers are torn down shell-only. Both layers
 * BORROW their wire configs (linearLayerInit, ownsQuantizations == false), so
 * the templates are freed here exactly once. */
static void freeBfpNativeFixture(bfpNativeFixture_t *f) {
    freeTensor(f->label);
    freeTensor(f->input);
    freeOptim(f->sgd);
    freeLinearLayerShellOnly(f->linear1);
    freeLinearLayerShellOnly(f->linear0);
    freeQuantization(f->momentumQ);
    freeQuantization(f->bfpWireQ);
    freeQuantization(f->floatQ);
}

/*! THE capstone (PR2 Task 9, uniform-native since epic PR3): 25 training
 *  steps with layer 0 fully derived -- forward AND all three backward slots
 *  run native ARITH_BFP (no pins).
 *  Asserts, in one run, that (a) the derivation flipped -- one BFP template
 *  yields ARITH_BFP in all four slots, (b) the native forward trains: finite,
 *  decreasing loss, (c) the hidden wire is BFP with the DERIVED geometry and a
 *  grid that left the zero state, (d) the weights stay BFP with their own
 *  geometry and a grid the optimizer's OUT_WRITE requant moved, and (e) grads
 *  stay FLOAT32 (#261). */
void testBfpNativeForwardTrainingLossDecreasesAndGridMoves(void) {
    rngSetSeed(1717u);
    bfpNativeFixture_t f;
    buildBfpNativeFixture(&f, /*pinWeightGradMath=*/false, /*weightGradStorage=*/NULL);

    tensor_t *w0Param = getParamFromParameter(f.linear0->config->linear->weights);
    tensor_t *w0Grad = getGradFromParameter(f.linear0->config->linear->weights);
    tensor_t *b0Grad = getGradFromParameter(f.linear0->config->linear->bias);
    uint8_t w0ExpBefore[4];
    memcpy(w0ExpBefore, ((bfpQConfig_t *)w0Param->quantization->qConfig)->exponents, 4);

    bfpWireCapture_t cap = {0};
    optimizerFunctions_t sgdFns = optimizerFunctions[SGD_M];
    float firstLoss = NAN;
    float lastLoss = NAN;
    for (size_t step = 0; step < 25; step++) {
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

    /* CAPTURE, then FREE, then assert (Unity longjmps out of the first failure). */
    bool wireExponentMoved = false;
    uint8_t zeroState = (uint8_t)((1 << (8 - 1)) - 1); /* exponentBits=8 -> bias 127 */
    for (size_t g = 0; g < cap.numGroups && g < BFP_WIRE_MAX_GROUPS; g++) {
        if (cap.exponents[g] != zeroState) {
            wireExponentMoved = true;
        }
    }
    int derivedForwardType = (int)f.derivedForward.type;
    int derivedForwardRounding = (int)f.derivedForward.roundingMode;
    bool allFourSlotsDerivedBfp = f.derivedWeightGrad == ARITH_BFP &&
                                  f.derivedBiasGrad == ARITH_BFP && f.derivedPropLoss == ARITH_BFP;
    int configuredForwardType = (int)f.linear0->config->linear->forwardMath.type;
    int weightStorageType = (int)w0Param->quantization->type;
    bfpQConfig_t *w0QC = w0Param->quantization->qConfig;
    bool weightGeometryUnchanged = w0QC->numGroups == 4 && w0QC->groupSize == 3 &&
                                   w0QC->mantissaBits == 8 && w0QC->exponentBits == 8;
    bool weightExponentMoved = false;
    for (size_t i = 0; i < 4; i++) {
        if (w0QC->exponents[i] != w0ExpBefore[i]) {
            weightExponentMoved = true;
        }
    }
    bool gradsStillFloat =
        w0Grad->quantization->type == FLOAT32 && b0Grad->quantization->type == FLOAT32;

    freeBfpNativeFixture(&f);

    TEST_ASSERT_EQUAL_INT_MESSAGE(ARITH_BFP, derivedForwardType,
                                  "BFP storage must DERIVE native ARITH_BFP (the epic PR2 flip)");
    TEST_ASSERT_EQUAL_INT_MESSAGE(SR_HALF_AWAY, derivedForwardRounding,
                                  "the derived arithmetic carries the config's own roundingMode");
    TEST_ASSERT_TRUE_MESSAGE(allFourSlotsDerivedBfp,
                             "layerQuantInitUniform over a BFP template must derive ARITH_BFP in "
                             "ALL FOUR math slots -- and since epic PR3 all four RUN native");
    TEST_ASSERT_EQUAL_INT_MESSAGE(ARITH_BFP, configuredForwardType,
                                  "layer 0's forward must have RUN native ARITH_BFP");
    TEST_ASSERT_TRUE_MESSAGE(cap.seen, "layer-0 forward probe must have fired");
    TEST_ASSERT_EQUAL_INT_MESSAGE(BFP, cap.type, "the hidden wire tensor must be BFP-stored");
    TEST_ASSERT_EQUAL_UINT_MESSAGE(4, cap.numElements, "hidden wire is [1, 4]");
    TEST_ASSERT_EQUAL_UINT_MESSAGE(2, cap.numGroups, "wire numGroups is DERIVED: 4 elements / 2");
    TEST_ASSERT_EQUAL_UINT_MESSAGE(2, cap.groupSize, "groupSize comes from the template");
    TEST_ASSERT_EQUAL_UINT8_MESSAGE(6, cap.mantissaBits, "mantissa width comes from the template");
    TEST_ASSERT_EQUAL_UINT8_MESSAGE(8, cap.exponentBits, "exponent width comes from the template");
    TEST_ASSERT_TRUE_MESSAGE(wireExponentMoved,
                             "the forward OUT_WRITE must derive the wire's grid: at least one "
                             "group exponent must leave the zero state");
    TEST_ASSERT_TRUE_MESSAGE(isfinite(firstLoss) && isfinite(lastLoss),
                             "native BFP forward training losses must be finite");
    TEST_ASSERT_TRUE_MESSAGE(lastLoss < firstLoss,
                             "native BFP forward training must converge (loss must decrease)");
    TEST_ASSERT_EQUAL_INT_MESSAGE(BFP, weightStorageType,
                                  "weight params must remain BFP after training");
    TEST_ASSERT_TRUE_MESSAGE(weightGeometryUnchanged,
                             "weight geometry/widths must be unchanged after training");
    TEST_ASSERT_TRUE_MESSAGE(weightExponentMoved,
                             "the optimizer's OUT_WRITE requant must re-derive at least one "
                             "weight group's exponent (the grid must move)");
    TEST_ASSERT_TRUE_MESSAGE(gradsStillFloat, "grad storage must stay FLOAT32 (default, #261)");
}

/*! The fixture's OTHER variant (pinWeightGradMath == true): native ARITH_BFP
 *  forward + all three backward slots explicitly pinned ARITH_FLOAT32 with a
 *  FLOAT32 dx wire -- the documented fake-quant-backward recipe
 *  (docs/conventions/arithmetic-bfp.md §5.1). Post-PR3 this stays a supported
 *  configuration, not just a stopgap, so it keeps end-to-end coverage: the
 *  pins must actually land on the config (flag-branch sensitivity -- a
 *  fully-derived model would also train, so loss alone cannot detect a broken
 *  flag) and training must converge. */
void testBfpPinnedFloat32BackwardTrainingLossDecreases(void) {
    rngSetSeed(1717u);
    bfpNativeFixture_t f;
    buildBfpNativeFixture(&f, /*pinWeightGradMath=*/true, /*weightGradStorage=*/NULL);

    optimizerFunctions_t sgdFns = optimizerFunctions[SGD_M];
    float firstLoss = NAN;
    float lastLoss = NAN;
    for (size_t step = 0; step < 25; step++) {
        trainingStats_t *stats = calculateGradsSequential(f.model, 2, defaultLossConfig(MSE),
                                                          REDUCTION_MEAN, f.input, f.label);
        if (step == 0) {
            firstLoss = stats->loss;
        }
        lastLoss = stats->loss;
        freeTrainingStats(stats);
        sgdFns.step(f.sgd);
        sgdFns.zero(f.sgd);
    }

    /* CAPTURE, then FREE, then assert. */
    linearConfig_t *cfg0 = f.linear0->config->linear;
    int configuredForwardType = (int)cfg0->forwardMath.type;
    bool backwardSlotsPinnedFloat = cfg0->weightGradMath.type == ARITH_FLOAT32 &&
                                    cfg0->biasGradMath.type == ARITH_FLOAT32 &&
                                    cfg0->propLossMath.type == ARITH_FLOAT32;

    freeBfpNativeFixture(&f);

    TEST_ASSERT_EQUAL_INT_MESSAGE(ARITH_BFP, configuredForwardType,
                                  "layer 0's forward must still run native ARITH_BFP");
    TEST_ASSERT_TRUE_MESSAGE(backwardSlotsPinnedFloat,
                             "pinWeightGradMath == true must pin ALL THREE backward slots to "
                             "ARITH_FLOAT32 (the fake-quant-backward variant)");
    TEST_ASSERT_TRUE_MESSAGE(isfinite(firstLoss) && isfinite(lastLoss),
                             "pinned-FLOAT32-backward training losses must be finite");
    TEST_ASSERT_TRUE_MESSAGE(lastLoss < firstLoss,
                             "native BFP forward + pinned FLOAT32 backward must converge "
                             "(loss must decrease)");
}

/* ===========================================================================
 * BFP epic PR3 Task 6 capstone: per-tensor BFP GRAD storage, load-bearing e2e.
 * ======================================================================== */

/*! Task 6's own load-bearing e2e: same native-BFP-forward fixture as above,
 *  but layer 0's weight grad ALSO opts into per-tensor BFP storage via the
 *  weightGradStorage knob (gradInit's grouped-only gate, Step 1). Exercises,
 *  in one 5-step run: the accumulateOut BFP-target arm (Task 5) writing every
 *  backward pass's weight grad, the optimizer's read of that grad through
 *  conversionMatrix[BFP][FLOAT32] (unmodified, PR3 groundwork), and the
 *  zeroGrad BFP arm (Step 3) resetting codes+exponents to the canonical
 *  zero state after every step. The exponent half of that reset is
 *  SYM/ASYM-parity hygiene, not accumulate-correctness: FixedGrid's
 *  fresh-vs-carry decision is a codes-only scan and the memset already
 *  zeroes every code -- the final exponent assertion below pins the
 *  hygiene contract itself.
 *
 *  RED before Steps 1-4 land: gradInit's then-unconditional BFP reject
 *  (TensorApi.c) kills the whole binary the instant this fixture builds
 *  layer 0's weight grad tensor -- written first in this task per the
 *  brief's Step 5 ordering note, this is that RED. */
void testBfpGradStorageTrainingAccumulatesAndSteps(void) {
    rngSetSeed(1717u);
    quantization_t *gradKnob = quantizationInitBfp(8, 8, HALF_AWAY);
    bfpNativeFixture_t f;
    buildBfpNativeFixture(&f, /*pinWeightGradMath=*/false, gradKnob);
    freeQuantization(gradKnob); /* gradInit clones via getQLike -- template unused after */

    tensor_t *w0Grad = getGradFromParameter(f.linear0->config->linear->weights);
    TEST_ASSERT_EQUAL_INT_MESSAGE(BFP, w0Grad->quantization->type,
                                  "guard: weightGradStorage knob must land BFP grad storage");

    optimizerFunctions_t sgdFns = optimizerFunctions[SGD_M];
    float firstLoss = NAN;
    float lastLoss = NAN;
    uint8_t gradExponentAfterBackward = (uint8_t)bfpExponentBias(w0Grad->quantization->qConfig);
    for (size_t step = 0; step < 5; step++) {
        trainingStats_t *stats = calculateGradsSequential(f.model, 2, defaultLossConfig(MSE),
                                                          REDUCTION_MEAN, f.input, f.label);
        if (step == 0) {
            firstLoss = stats->loss;
        }
        lastLoss = stats->loss;
        freeTrainingStats(stats);
        if (step == 4) {
            /* Capture BEFORE the optimizer step/zero on the last iteration --
             * zeroGrad resets exponents back to bias every step, so this is
             * the only point where the accumulate arm's moved grid is
             * observable. */
            bfpQConfig_t *gradQC = w0Grad->quantization->qConfig;
            gradExponentAfterBackward = gradQC->exponents[0];
        }
        sgdFns.step(f.sgd);
        sgdFns.zero(f.sgd);
    }

    /* CAPTURE post-loop (post-zero) state, then FREE, then assert. */
    bfpQConfig_t *gradQCAfter = w0Grad->quantization->qConfig;
    int gradTypeAfter = (int)w0Grad->quantization->type;
    size_t gradNumGroupsAfter = gradQCAfter->numGroups;
    uint8_t gradExponentAfterZero = gradQCAfter->exponents[0];
    uint8_t zeroStateBias = (uint8_t)bfpExponentBias(gradQCAfter);

    freeBfpNativeFixture(&f);

    TEST_ASSERT_TRUE_MESSAGE(isfinite(firstLoss) && isfinite(lastLoss),
                             "BFP grad-storage training losses must be finite");
    TEST_ASSERT_TRUE_MESSAGE(lastLoss < firstLoss,
                             "BFP grad-storage training must converge (loss must decrease)");
    TEST_ASSERT_EQUAL_INT_MESSAGE(BFP, gradTypeAfter,
                                  "weight grad must stay BFP-stored after training (Step 1/2)");
    TEST_ASSERT_EQUAL_size_t_MESSAGE(1, gradNumGroupsAfter,
                                     "grads are per-tensor-only (#300 axis, Step 1)");
    TEST_ASSERT_NOT_EQUAL_MESSAGE(
        zeroStateBias, gradExponentAfterBackward,
        "the accumulateOut BFP-target arm (Task 5) must have moved the grad's exponent "
        "off the zero state during backward, read through conversionMatrix[BFP][FLOAT32] "
        "by the optimizer step");
    TEST_ASSERT_EQUAL_UINT8_MESSAGE(
        zeroStateBias, gradExponentAfterZero,
        "the zeroGrad BFP arm (Step 3) must reset every exponent back to bias after the step");
}

/* ===========================================================================
 * BFP epic PR3 Task 8 capstone: REDUCTION_MEAN through the DEFAULT epoch path.
 * ======================================================================== */

/* Two-sample dataset for the REDUCTION_MEAN e2e below -- file-scope because
 * the dataLoader callbacks carry no context pointer (the epochDataset pattern
 * in UnitTestTrainingLoopApi.c). Built/freed inside the one test that uses it. */
static tensor_t *bfpMeanEpochItems[2];
static tensor_t *bfpMeanEpochLabels[2];

static sample_t *getBfpMeanEpochSample(size_t id) {
    sample_t *s = reserveMemory(sizeof(sample_t));
    s->item = bfpMeanEpochItems[id];
    s->label = bfpMeanEpochLabels[id];
    return s;
}

static size_t getBfpMeanEpochDatasetSize() {
    return 2;
}

/*! Task 8's load-bearing e2e: the Task 6 fixture (native BFP forward + BFP
 *  weight-grad storage) driven through the DEFAULT TrainingLoopApi epoch path
 *  (trainingEpochDefault) with defaultLossConfig's backwardReduction ==
 *  REDUCTION_MEAN -- so every batch runs TrainingEpochDefault.c's mean-scale
 *  branch: computeMeanScale -> scaleOptimizerGradients over a MIXED optimizer
 *  (layer 0's weight grad BFP, every other grad FLOAT32) -> step -> zero.
 *  Before Task 8's BFP arm, scaleOptimizerGradients's default arm exit(1)s on
 *  the BFP grad the moment the first batch completes -- that process death is
 *  this test's RED, and the finite decreasing loss is the proof the last gap
 *  in the default epoch loop is closed. */
void testBfpGradStorageTrainsUnderReductionMean(void) {
    rngSetSeed(1717u);
    quantization_t *gradKnob = quantizationInitBfp(8, 8, HALF_AWAY);
    bfpNativeFixture_t f;
    buildBfpNativeFixture(&f, /*pinWeightGradMath=*/false, gradKnob);
    freeQuantization(gradKnob);

    bfpMeanEpochItems[0] = buildFloatTensor2D(1, 3, (float[]){1.0f, 2.0f, 3.0f});
    bfpMeanEpochLabels[0] = buildFloatTensor2D(1, 2, (float[]){0.2f, -0.3f});
    bfpMeanEpochItems[1] = buildFloatTensor2D(1, 3, (float[]){0.5f, -1.0f, 2.0f});
    bfpMeanEpochLabels[1] = buildFloatTensor2D(1, 2, (float[]){-0.1f, 0.4f});
    dataLoader_t *dl = dataLoaderInit(getBfpMeanEpochSample, getBfpMeanEpochDatasetSize, 1, NULL,
                                      NULL, false, 0, true);

    tensor_t *w0Grad = getGradFromParameter(f.linear0->config->linear->weights);
    float firstEpochLoss = NAN;
    float lastEpochLoss = NAN;
    for (size_t epoch = 0; epoch < 8; epoch++) {
        float epochLoss = trainingEpochDefault(f.model, 2, defaultLossConfig(MSE), dl, f.sgd,
                                               calculateGradsSequential, REDUCTION_MEAN);
        if (epoch == 0) {
            firstEpochLoss = epochLoss;
        }
        lastEpochLoss = epochLoss;
    }

    /* CAPTURE, then FREE (reverse init order), then assert. */
    int gradTypeAfter = (int)w0Grad->quantization->type;
    size_t gradNumGroupsAfter = ((bfpQConfig_t *)w0Grad->quantization->qConfig)->numGroups;

    freeDataLoader(dl);
    freeTensor(bfpMeanEpochLabels[1]);
    freeTensor(bfpMeanEpochItems[1]);
    freeTensor(bfpMeanEpochLabels[0]);
    freeTensor(bfpMeanEpochItems[0]);
    freeBfpNativeFixture(&f);

    TEST_ASSERT_TRUE_MESSAGE(isfinite(firstEpochLoss) && isfinite(lastEpochLoss),
                             "REDUCTION_MEAN epoch losses must stay finite with BFP grad storage");
    TEST_ASSERT_TRUE_MESSAGE(lastEpochLoss < firstEpochLoss,
                             "the default epoch path (mean-scale -> step -> zero per batch) must "
                             "converge with BFP-stored weight grads");
    TEST_ASSERT_EQUAL_INT_MESSAGE(BFP, gradTypeAfter,
                                  "weight grad must stay BFP-stored after epoch training");
    TEST_ASSERT_EQUAL_size_t_MESSAGE(1, gradNumGroupsAfter,
                                     "grads are per-tensor-only (#300 axis)");
}

/* ===========================================================================
 * #420 C2: CONV-FAMILY BFP grad-storage capstone.
 *
 * The two capstones above pin BFP weight-grad storage on a LINEAR layer only,
 * so the conv weightGrad/biasGrad accumulate route -- the Conv1d kernels'
 * FLOAT32 raw intermediate flowing into accumulateOut's BFP-target arm -- ships
 * with no e2e coverage at all. This capstone closes that: Conv1d -> Flatten ->
 * Linear with per-tensor BFP weightGradStorage AND biasGradStorage on the conv
 * layer (the Linear capstones exercise the weight knob only), trained through
 * the default epoch path.
 *
 * Everything else stays FLOAT32 on purpose: BFP STORAGE is guarded out of
 * Flatten until epic PR4, so a BFP forward wire could not reach the Linear head
 * at all -- and it is irrelevant here, since the claim under test is about grad
 * STORAGE, not about ARITH_BFP math. Bias is BIAS_TRUE so the biasGrad route is
 * live.
 * ======================================================================== */

#define BFP_CONV_IN_CHANNELS 1
#define BFP_CONV_OUT_CHANNELS 2
#define BFP_CONV_KERNEL_SIZE 2
#define BFP_CONV_SEQ_LEN 4
#define BFP_CONV_OUT_LEN 3 /* VALID, stride 1: 4 - 2 + 1 */
#define BFP_CONV_FLAT_FEATURES (BFP_CONV_OUT_CHANNELS * BFP_CONV_OUT_LEN)
#define BFP_CONV_NUM_CLASSES 2
#define BFP_CONV_MODEL_SIZE 3

static tensor_t *buildFloatTensor3D(size_t d0, size_t d1, size_t d2, const float *values) {
    size_t *dims = reserveMemory(3 * sizeof(size_t));
    dims[0] = d0;
    dims[1] = d1;
    dims[2] = d2;
    size_t *order = reserveMemory(3 * sizeof(size_t));
    setOrderOfDimsForNewTensor(3, order);
    shape_t *shape = reserveMemory(sizeof(shape_t));
    setShape(shape, dims, 3, order);
    tensor_t *t = initTensor(shape, quantizationInitFloat(), NULL);
    tensorFillFromFloatBuffer(t, (float *)values, d0 * d1 * d2);
    return t;
}

/* File-scope for the same reason the REDUCTION_MEAN fixture above is: the
 * dataLoader callbacks carry no context pointer. */
static tensor_t *bfpConvEpochItems[2];
static tensor_t *bfpConvEpochLabels[2];

static sample_t *getBfpConvEpochSample(size_t id) {
    sample_t *s = reserveMemory(sizeof(sample_t));
    s->item = bfpConvEpochItems[id];
    s->label = bfpConvEpochLabels[id];
    return s;
}

static size_t getBfpConvEpochDatasetSize() {
    return 2;
}

/* Conv1d comes from a BORROWING factory but its parameters are registered with
 * the optimizer, which frees them in freeOptim's cascade -- so the layer is
 * torn down shell-only (the GroupNorm/BiaslessConv integration-test pattern).
 * The kernel_t is factory-allocated and optimizer-invisible, so it is freed
 * here explicitly. */
static void freeConv1dLayerShellOnly(layer_t *layer) {
    freeReservedMemory(layer->config->conv1d->kernel);
    freeReservedMemory(layer->config->conv1d);
    freeReservedMemory(layer->config);
    freeReservedMemory(layer);
}

void testBfpConvGradStorageTrainsUnderDefaultEpoch(void) {
    rngSetSeed(1717u);
    quantization_t *floatQ = quantizationInitFloat();
    quantization_t *gradKnob = quantizationInitBfp(8, 8, HALF_AWAY);

    layerQuant_t lqConv;
    layerQuantInitUniform(&lqConv, floatQ);
    lqConv.weightGradStorage = gradKnob;
    lqConv.biasGradStorage = gradKnob;
    layer_t *conv = conv1dLayerInit(&(conv1dInit_t){.inChannels = BFP_CONV_IN_CHANNELS,
                                                    .outChannels = BFP_CONV_OUT_CHANNELS,
                                                    .kernelSize = BFP_CONV_KERNEL_SIZE,
                                                    .bias = BIAS_TRUE},
                                    &lqConv);
    freeQuantization(gradKnob); /* gradInit deep-clones via getQLike */

    layerQuant_t lqPlain;
    layerQuantInitUniform(&lqPlain, floatQ);
    layer_t *flat = flattenLayerInit();
    layer_t *head = linearLayerInit(&(linearInit_t){.inFeatures = BFP_CONV_FLAT_FEATURES,
                                                    .outFeatures = BFP_CONV_NUM_CLASSES,
                                                    .bias = BIAS_TRUE},
                                    &lqPlain);
    layer_t *model[BFP_CONV_MODEL_SIZE] = {conv, flat, head};

    tensor_t *wGrad = getGradFromParameter(conv->config->conv1d->weights);
    tensor_t *bGrad = getGradFromParameter(conv->config->conv1d->bias);
    /* (a) the knob landed BFP grad storage on BOTH conv parameters. */
    int wGradType = (int)wGrad->quantization->type;
    int bGradType = (int)bGrad->quantization->type;

    bfpConvEpochItems[0] = buildFloatTensor3D(1, BFP_CONV_IN_CHANNELS, BFP_CONV_SEQ_LEN,
                                              (float[]){1.0f, 2.0f, 3.0f, 1.5f});
    bfpConvEpochLabels[0] = buildFloatTensor2D(1, BFP_CONV_NUM_CLASSES, (float[]){0.2f, -0.3f});
    bfpConvEpochItems[1] = buildFloatTensor3D(1, BFP_CONV_IN_CHANNELS, BFP_CONV_SEQ_LEN,
                                              (float[]){0.5f, -1.0f, 2.0f, -0.25f});
    bfpConvEpochLabels[1] = buildFloatTensor2D(1, BFP_CONV_NUM_CLASSES, (float[]){-0.1f, 0.4f});
    dataLoader_t *dl = dataLoaderInit(getBfpConvEpochSample, getBfpConvEpochDatasetSize, 1, NULL,
                                      NULL, false, 0, true);

    quantization_t *momentumQ = quantizationInitFloat();
    optimizer_t *sgd =
        sgdMCreateOptim(0.02f, 0.f, 0.f, model, BFP_CONV_MODEL_SIZE, momentumQ,
                        (arithmetic_t){.type = ARITH_FLOAT32, .roundingMode = HALF_AWAY});

    /* (a, strong form) ONE backward before the epoch loop, so the grads can be
     * inspected between backward and the optimizer's zero: the accumulate arm
     * must have moved BOTH grids off the zero state. zeroGrad resets exponents
     * to bias every step, so this is the only point where that is observable. */
    trainingStats_t *seedStats =
        calculateGradsSequential(model, BFP_CONV_MODEL_SIZE, defaultLossConfig(MSE), REDUCTION_MEAN,
                                 bfpConvEpochItems[0], bfpConvEpochLabels[0]);
    freeTrainingStats(seedStats);
    /* Sentinels keep the CAPTURE phase crash-free if the grad-storage knob
     * ever regresses to the FLOAT32 default -- a FLOAT32 grad carries a NULL
     * qConfig, and a mutation-time null deref here would replace the clean
     * dtype assertion below with a segfault nobody can attribute to a case. */
    uint8_t zeroStateBias = 0;
    uint8_t wGradExponent = 0;
    uint8_t bGradExponent = 0;
    if (wGradType == BFP && bGradType == BFP) {
        zeroStateBias = (uint8_t)bfpExponentBias(wGrad->quantization->qConfig);
        wGradExponent = ((bfpQConfig_t *)wGrad->quantization->qConfig)->exponents[0];
        bGradExponent = ((bfpQConfig_t *)bGrad->quantization->qConfig)->exponents[0];
    }
    optimizerFunctions[SGD_M].zero(sgd);

    /* (b) params must move: capture the conv weights before training. */
    float weightsBefore[BFP_CONV_OUT_CHANNELS * BFP_CONV_IN_CHANNELS * BFP_CONV_KERNEL_SIZE];
    tensor_t *wParam = getParamFromParameter(conv->config->conv1d->weights);
    memcpy(weightsBefore, wParam->data, sizeof(weightsBefore));

    /* (c) loss decreases across 10 epochs of the DEFAULT epoch path
     * (mean-scale -> scaleOptimizerGradients -> step -> zero per batch). */
    float firstEpochLoss = NAN;
    float lastEpochLoss = NAN;
    for (size_t epoch = 0; epoch < 10; epoch++) {
        float epochLoss = trainingEpochDefault(model, BFP_CONV_MODEL_SIZE, defaultLossConfig(MSE),
                                               dl, sgd, calculateGradsSequential, REDUCTION_MEAN);
        if (epoch == 0) {
            firstEpochLoss = epochLoss;
        }
        lastEpochLoss = epochLoss;
    }

    float weightsAfter[BFP_CONV_OUT_CHANNELS * BFP_CONV_IN_CHANNELS * BFP_CONV_KERNEL_SIZE];
    memcpy(weightsAfter, wParam->data, sizeof(weightsAfter));
    int wGradTypeAfter = (int)wGrad->quantization->type;
    int bGradTypeAfter = (int)bGrad->quantization->type;
    size_t wGradNumGroups =
        wGradTypeAfter == BFP ? ((bfpQConfig_t *)wGrad->quantization->qConfig)->numGroups : 0;

    /* CAPTURE -> FREE (reverse init order) -> assert. */
    freeOptim(sgd);
    freeLinearLayerShellOnly(head);
    freeFlattenLayer(flat);
    freeConv1dLayerShellOnly(conv);
    freeQuantization(momentumQ);
    freeDataLoader(dl);
    freeTensor(bfpConvEpochLabels[1]);
    freeTensor(bfpConvEpochItems[1]);
    freeTensor(bfpConvEpochLabels[0]);
    freeTensor(bfpConvEpochItems[0]);
    freeQuantization(floatQ);

    TEST_ASSERT_EQUAL_INT_MESSAGE(
        BFP, wGradType, "weightGradStorage must land BFP storage on the conv weight grad");
    TEST_ASSERT_EQUAL_INT_MESSAGE(BFP, bGradType,
                                  "biasGradStorage must land BFP storage on the conv bias grad");
    TEST_ASSERT_NOT_EQUAL_MESSAGE(zeroStateBias, wGradExponent,
                                  "the conv weightGrad accumulate route must move the BFP grad's "
                                  "exponent off the zero state during backward");
    TEST_ASSERT_NOT_EQUAL_MESSAGE(zeroStateBias, bGradExponent,
                                  "the conv biasGrad accumulate route must move the BFP grad's "
                                  "exponent off the zero state during backward");
    bool moved = false;
    for (size_t i = 0; i < sizeof(weightsBefore) / sizeof(weightsBefore[0]); i++) {
        if (weightsBefore[i] != weightsAfter[i]) {
            moved = true;
        }
    }
    TEST_ASSERT_TRUE_MESSAGE(moved, "a training step must move the conv weights read back through "
                                    "the BFP grad");
    TEST_ASSERT_TRUE_MESSAGE(isfinite(firstEpochLoss) && isfinite(lastEpochLoss),
                             "conv BFP grad-storage epoch losses must stay finite");
    TEST_ASSERT_TRUE_MESSAGE(lastEpochLoss < firstEpochLoss,
                             "the default epoch path must converge with BFP-stored conv grads");
    TEST_ASSERT_EQUAL_INT_MESSAGE(BFP, wGradTypeAfter,
                                  "conv weight grad must stay BFP-stored after training");
    TEST_ASSERT_EQUAL_INT_MESSAGE(BFP, bGradTypeAfter,
                                  "conv bias grad must stay BFP-stored after training");
    TEST_ASSERT_EQUAL_size_t_MESSAGE(1, wGradNumGroups, "grads are per-tensor-only (#300 axis)");
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
    RUN_TEST(testBfpWireGroupSizeEqualToWireElementsNormalizesToPerTensor);
    RUN_TEST(testBfpDxWireGroupSizeEqualToWireElementsNormalizesToPerTensor);
    RUN_TEST(testInitGradTensorBfpGroupSizeMismatchDies);
    RUN_TEST(testBfpDxWireNativeBackwardTrains);
    RUN_TEST(testOwningFactoryBfpOutputQFreesExponents);
    RUN_TEST(testBfpNativeForwardTrainingLossDecreasesAndGridMoves);
    RUN_TEST(testBfpPinnedFloat32BackwardTrainingLossDecreases);
    RUN_TEST(testBfpGradStorageTrainingAccumulatesAndSteps);
    RUN_TEST(testBfpGradStorageTrainsUnderReductionMean);
    RUN_TEST(testBfpConvGradStorageTrainsUnderDefaultEpoch);
    return UNITY_END();
}
