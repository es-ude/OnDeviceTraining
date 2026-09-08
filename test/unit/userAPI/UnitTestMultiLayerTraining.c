#define SOURCE_FILE "UNIT_TEST_MULTI_LAYER_TRAINING"

#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "AdaptivePool1dApi.h"
#include "ArithmeticType.h"
#include "BorrowedLayer.h"
#include "CalculateGradsSequential.h"
#include "Conv1dApi.h"
#include "DataLoaderApi.h"
#include "Dataset.h"
#include "DeathTest.h"
#include "FlattenApi.h"
#include "GroupNorm.h"
#include "GroupNormApi.h"
#include "InferenceApi.h"
#include "LayerNorm.h"
#include "LayerNormApi.h"
#include "LayerQuant.h"
#include "Linear.h"
#include "LinearApi.h"
#include "LossFunction.h"
#include "OptimizerApi.h"
#include "Pool1dApi.h"
#include "QuantizationApi.h"
#include "RNG.h"
#include "ReluApi.h"
#include "SgdApi.h"
#include "Softmax.h"
#include "SoftmaxApi.h"
#include "StorageApi.h"
#include "Tensor.h"
#include "TensorApi.h"
#include "TraceApi.h"
#include "TrainingBatchDefault.h"
#include "TrainingEpochDefault.h"
#include "TrainingLoopApi.h"
#include "expected_softmax.h"
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
    /* "Moved off the zero state" is a proxy: the zero state IS exponent == bias
     * (scale 1.0), so a grad whose absmax happened to derive exactly that exponent
     * would false-fail here. rngSetSeed(1717u) above pins the draw that makes this
     * deterministic -- the exponents are not otherwise asserted, so a seed change
     * must re-confirm both of these. */
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

/* ===========================================================================
 * BFP epic PR4 capstone: the whole non-GEMM topology on ONE uniform BFP wire.
 * ======================================================================== */

/* buildRampParam2D's rank-3 twin — the conv weight is [Cout, Cin, K]. */
static parameter_t *buildRampParam3D(size_t d0, size_t d1, size_t d2, float base, float step) {
    size_t *dims = reserveMemory(3 * sizeof(size_t));
    dims[0] = d0;
    dims[1] = d1;
    dims[2] = d2;
    size_t *order = reserveMemory(3 * sizeof(size_t));
    setOrderOfDimsForNewTensor(3, order);
    shape_t *shape = reserveMemory(sizeof(shape_t));
    setShape(shape, dims, 3, order);
    tensor_t *param = initTensor(shape, quantizationInitFloat(), NULL);
    size_t n = d0 * d1 * d2;
    float values[n];
    for (size_t i = 0; i < n; i++) {
        values[i] = base + step * (float)i;
    }
    tensorFillFromFloatBuffer(param, values, n);
    return parameterInit(param, gradInitFloat(param, NULL));
}

/* Flatten's exponent carry (Task 3) is NOT observable through the capstone's
 * loss assertions: dropping it leaves the flatten wire at the zero state
 * (scale 1.0) with UNCHANGED codes, and training still converges — the
 * distortion is consistent across steps and the fused (p - y) gradient stays
 * bounded. Verified empirically (mutant applied, capstone still PASSed).
 * So the carry gets its own deterministic observable, taken from the live
 * training run through the TraceApi probes:
 *   fwd@2   = MaxPool's output wire  -> fwd@3   = Flatten's output wire
 *   agrad@3 = grad ENTERING Flatten  -> agrad@2 = grad entering MaxPool
 *             (i.e. exactly what Flatten's backward produced)
 * Both pairs must carry IDENTICAL per-group exponents; the non-vacuity
 * assertions pin that the source grids actually left the zero state, so an
 * all-127 == all-127 comparison cannot pass by accident. */
#define BFP_PR4_MAX_GROUPS 8

typedef struct pr4CarryCapture {
    bool seenFwdPool;
    bool seenFwdFlat;
    bool seenAgradFlat;
    bool seenAgradPool;
    size_t nFwdPool;
    size_t nFwdFlat;
    size_t nAgradFlat;
    size_t nAgradPool;
    uint8_t fwdPool[BFP_PR4_MAX_GROUPS];
    uint8_t fwdFlat[BFP_PR4_MAX_GROUPS];
    uint8_t agradFlat[BFP_PR4_MAX_GROUPS];
    uint8_t agradPool[BFP_PR4_MAX_GROUPS];
} pr4CarryCapture_t;

/* First occurrence only — the carry must hold on step 1, before ten SGD steps
 * can wash the two grids into coincidence. */
static void pr4SnapExponents(bool *seen, size_t *nOut, uint8_t *dst, const tensor_t *t) {
    if (*seen || t->quantization->type != BFP) {
        return;
    }
    const bfpQConfig_t *qc = t->quantization->qConfig;
    size_t n = qc->numGroups < BFP_PR4_MAX_GROUPS ? qc->numGroups : BFP_PR4_MAX_GROUPS;
    *seen = true;
    *nOut = n;
    memcpy(dst, qc->exponents, n);
}

static void capturePr4FlattenCarry(void *ctx, size_t layerIdx, layerType_t layerType,
                                   const char *phase, tensor_t *tensor) {
    (void)layerType;
    pr4CarryCapture_t *cap = ctx;
    if (strcmp(phase, "fwd") == 0) {
        if (layerIdx == 2) {
            pr4SnapExponents(&cap->seenFwdPool, &cap->nFwdPool, cap->fwdPool, tensor);
        } else if (layerIdx == 3) {
            pr4SnapExponents(&cap->seenFwdFlat, &cap->nFwdFlat, cap->fwdFlat, tensor);
        }
    } else if (strcmp(phase, "agrad") == 0) {
        if (layerIdx == 3) {
            pr4SnapExponents(&cap->seenAgradFlat, &cap->nAgradFlat, cap->agradFlat, tensor);
        } else if (layerIdx == 2) {
            pr4SnapExponents(&cap->seenAgradPool, &cap->nAgradPool, cap->agradPool, tensor);
        }
    }
}

/* BFP epic PR4 capstone (R-P7e): the whole non-GEMM topology on ONE uniform
 * BFP wire config — conv1d -> relu -> maxpool -> flatten -> linear -> softmax,
 * CrossEntropy loss. Every arm PR4 added is on the critical path:
 *   - conv1d and linear declare native ARITH_BFP (PR2/PR3) in all four math
 *     slots, so their weights are FLOAT32-init + requantizeTensorInPlace
 *     (#270 requireFloat32 gate). Which of those slots actually EXECUTES
 *     differs per layer -- see the #423-item-4 block near the assertions;
 *   - relu is packed-domain transparent (Tasks 1/2);
 *   - maxpool compares dequantized values and scatters to argmax (Tasks 8/9);
 *   - flatten carries the exponent array (Task 3);
 *   - softmax runs its NATIVE forward (P6-8): layerQuantInitUniform derives
 *     ARITH_BFP forwardMath from the shared BFP wire and Softmax's forward
 *     dispatches on it (Task 4's i-exp kernel). Its propLossMath derives
 *     ARITH_BFP too (Task 5's funnel arm), but CrossEntropy's FUSED backward
 *     still skips the softmax layer entirely (CalculateGradsSequential.c:
 *     backwardIndex -= 1 for CROSS_ENTROPY), so that native backward is not
 *     reached HERE -- see unitTestUniformBfpSoftmaxMseTrains (MSE, non-CE)
 *     for that coverage;
 *   - the loss reaches its BFP fake-quant arm (Task 10) because the model
 *     output wire is BFP.
 * Wire element counts are 12 / 12 / 6 / 6 / 2 / 2 — all divisible by the
 * template groupSize 2, which the allocators require (Decision 5); the last
 * two normalize to per-tensor {1, 0} (groupSize == wire elements). The PARAMS
 * use a per-tensor {1, 0} BFP config: the conv weight's reduction run is
 * ic*k = 3, and the §2 param rule demands groupSize divide it. */
void testBfpUniformPoolActivationModelTrains(void) {
    rngSetSeed(4242u);
    quantization_t *bfpWireQ = quantizationInitBfpGrouped(8, 8, SR_HALF_AWAY, 2, 2);
    layerQuant_t lq;
    layerQuantInitUniform(&lq, bfpWireQ);

    /* conv1d [1, 1, 8] -> [1, 2, 6] (K=3, VALID, stride 1), no bias. The
     * kernel_t is heap-allocated because freeConv1dLayerShellOnly (above)
     * frees it; buildBorrowedConv1dLayer only borrows the pointer. */
    parameter_t *convW = buildRampParam3D(2, 1, 3, 0.20f, 0.05f);
    quantization_t *convWQ = quantizationInitBfp(8, 8, SR_HALF_AWAY);
    requantizeTensorInPlace(getParamFromParameter(convW), convWQ);
    freeQuantization(convWQ); /* requantizeTensorInPlace clones the template */
    kernel_t *convKernel = reserveMemory(sizeof(kernel_t));
    initKernel(convKernel, 3, VALID, /*dilation=*/1, /*stride=*/1);
    layer_t *conv = buildBorrowedConv1dLayer(convW, NULL, convKernel, bfpWireQ);

    layer_t *relu = reluLayerInit(&lq);

    /* maxpool K=2 stride 2 over [1, 2, 6] -> [1, 2, 3]. */
    layer_t *pool = maxPool1dLayerInit(
        &(maxPool1dInit_t){.kernelSize = 2, .stride = 2, .inputChannels = 2, .inputLength = 6},
        &lq);

    layer_t *flatten = flattenLayerInit(); /* [1, 2, 3] -> [1, 6] */

    /* linear 6 -> 2. */
    parameter_t *linW = buildRampParam2D(2, 6, 0.10f, 0.03f);
    parameter_t *linB = buildRampParam2D(1, 2, 0.05f, 0.05f);
    quantization_t *linWQ = quantizationInitBfp(8, 8, SR_HALF_AWAY);
    quantization_t *linBQ = quantizationInitBfp(8, 8, SR_HALF_AWAY);
    requantizeTensorInPlace(getParamFromParameter(linW), linWQ);
    requantizeTensorInPlace(getParamFromParameter(linB), linBQ);
    freeQuantization(linBQ);
    freeQuantization(linWQ);
    layer_t *linear = buildBorrowedLinearLayer(linW, linB, bfpWireQ);

    layer_t *softmax = softmaxLayerInit(&lq);

    layer_t *model[6] = {conv, relu, pool, flatten, linear, softmax};

    quantization_t *momentumQ = quantizationInitFloat();
    optimizer_t *sgd =
        sgdMCreateOptim(0.05f, 0.9f, 0.f, model, 6, momentumQ,
                        (arithmetic_t){.type = ARITH_FLOAT32, .roundingMode = HALF_AWAY});

    float inputValues[8] = {0.9f, -0.4f, 1.3f, 0.2f, -1.1f, 0.7f, 0.3f, -0.6f};
    tensor_t *input = buildFloatTensor3D(1, 1, 8, inputValues);
    tensor_t *label = buildFloatTensor2D(1, 2, (float[]){1.0f, 0.0f});

    /* Snapshot the conv weight's packed payload: the SGD write-back must land
     * in BFP storage, so the codes have to change. */
    tensor_t *convWTensor = getParamFromParameter(convW);
    size_t convWBytes = calcNumberOfBytesForData(convWTensor->quantization, 6);
    uint8_t before[16];
    memcpy(before, convWTensor->data, convWBytes);

    optimizerFunctions_t sgdFns = optimizerFunctions[SGD_M];
    pr4CarryCapture_t carry = {0};
    float firstLoss = NAN;
    float lastLoss = NAN;
    for (size_t step = 0; step < 10; step++) {
        trainingStats_t *stats =
            tracedGrads(model, 6, defaultLossConfig(CROSS_ENTROPY), REDUCTION_MEAN, input, label,
                        capturePr4FlattenCarry, &carry);
        if (step == 0) {
            firstLoss = stats->loss;
        }
        lastLoss = stats->loss;
        freeTrainingStats(stats);
        sgdFns.step(sgd);
        sgdFns.zero(sgd);
    }

    /* CAPTURE -> FREE (reverse init order) -> assert (Unity longjmps out of
     * the first failure, so nothing may be read after the teardown). */
    bool codesMoved = memcmp(before, convWTensor->data, convWBytes) != 0;
    conv1dConfig_t *convCfg = conv->config->conv1d;
    /* #423 item 4 -- read this before ticking the item, because ASSERTED and
     * EXECUTED are not the same set here.
     * ASSERTED: all four of the conv's math slots are ARITH_BFP by CONFIG, and
     * its weights are still BFP-stored after training.
     * EXECUTED in this loop: forwardMath (the native BFP conv GEMM) and
     * weightGradMath (conv1dCalcWeightGradsBfp, over a BFP lossGrad that came
     * down through Flatten's, MaxPool's and Relu's BFP backwards) -- both
     * confirmed by distinct-exit-code probes.
     * NOT executed: biasGradMath, skipped by Conv1d's `if (cfg->bias)` because
     * this fixture's conv is bias-less; and propLossMath, skipped by
     * `if (propLoss != NULL)` because conv is the DEEPEST trainable layer and
     * #380 PR2 truncation hands it propLoss == NULL. The dx skip is
     * structural, not a fixture choice: no topology reaches a conv's dx arm
     * unless a trainable layer sits below it. That arm is covered by
     * UnitTestConv1d's PR3 gold tests (testConv1dBackwardBfpDx*) and, in this
     * same binary, by Linear's structurally identical dx arm at index 4. */
    bool convAllSlotsBfp =
        convCfg->forwardMath.type == ARITH_BFP && convCfg->weightGradMath.type == ARITH_BFP &&
        convCfg->biasGradMath.type == ARITH_BFP && convCfg->propLossMath.type == ARITH_BFP;
    int convWeightStorage = (int)convWTensor->quantization->type;
    int linearForwardType = (int)linear->config->linear->forwardMath.type;
    /* P6-8: the pin removal must be observable, not just declared -- both
     * softmax math slots derive ARITH_BFP through the ordinary layerQuant
     * path now (forward already executed native before this task; backward
     * derives it too, though CE's skip means it is not the layer that
     * exercises it here). */
    int softmaxForwardType = (int)softmax->config->softmax->forwardMath.type;
    int softmaxPropLossType = (int)softmax->config->softmax->propLossMath.type;
    bool allProbesFired =
        carry.seenFwdPool && carry.seenFwdFlat && carry.seenAgradFlat && carry.seenAgradPool;
    bool carryGroupCountsMatch =
        carry.nFwdPool == carry.nFwdFlat && carry.nAgradFlat == carry.nAgradPool;
    const uint8_t zeroState = 127; /* exponentBits 8 -> bias 127 */
    bool fwdSourceGridMoved = false;
    bool agradSourceGridMoved = false;
    for (size_t g = 0; g < carry.nFwdPool; g++) {
        if (carry.fwdPool[g] != zeroState) {
            fwdSourceGridMoved = true;
        }
    }
    for (size_t g = 0; g < carry.nAgradFlat; g++) {
        if (carry.agradFlat[g] != zeroState) {
            agradSourceGridMoved = true;
        }
    }

    freeTensor(label);
    freeTensor(input);
    freeOptim(sgd);
    freeSoftmaxLayer(softmax);
    freeLinearLayerShellOnly(linear);
    freeFlattenLayer(flatten);
    freeMaxPool1dLayer(pool);
    freeReluLayer(relu);
    freeConv1dLayerShellOnly(conv);
    freeQuantization(momentumQ);
    freeQuantization(bfpWireQ);

    TEST_ASSERT_TRUE_MESSAGE(convAllSlotsBfp,
                             "the capstone's conv must DECLARE native ARITH_BFP in all four math "
                             "slots (#423 item 4; forward + weightGrad execute here, biasGrad is "
                             "bias-less and dx is #380-truncated -- see the block above)");
    TEST_ASSERT_EQUAL_INT_MESSAGE(ARITH_BFP, linearForwardType,
                                  "the capstone's linear must run native ARITH_BFP too");
    TEST_ASSERT_EQUAL_INT_MESSAGE(ARITH_BFP, softmaxForwardType,
                                  "the capstone's softmax must derive native ARITH_BFP forward "
                                  "(P6-8 flip)");
    TEST_ASSERT_EQUAL_INT_MESSAGE(ARITH_BFP, softmaxPropLossType,
                                  "the capstone's softmax must derive native ARITH_BFP "
                                  "propLossMath too, even though CE's skip means this layer's "
                                  "backward never runs here (P6-8 flip)");
    TEST_ASSERT_EQUAL_INT_MESSAGE(BFP, convWeightStorage,
                                  "conv weights must stay BFP-stored after training");
    TEST_ASSERT_TRUE_MESSAGE(isfinite(firstLoss) && isfinite(lastLoss),
                             "uniform-BFP training must stay finite through every PR4 arm");
    TEST_ASSERT_TRUE_MESSAGE(lastLoss < firstLoss,
                             "uniform-BFP conv->relu->pool->flatten->linear->softmax+CE must "
                             "converge (the vision-gate acceptance)");
    TEST_ASSERT_TRUE_MESSAGE(codesMoved,
                             "the SGD write-back must land in the conv weight's BFP storage");
    TEST_ASSERT_TRUE_MESSAGE(allProbesFired,
                             "all four BFP wire probes around Flatten must have fired");
    TEST_ASSERT_TRUE_MESSAGE(carryGroupCountsMatch,
                             "a reshape preserves the element count, so both wires of each pair "
                             "must derive the same numGroups");
    /* Two seed-dependent assertions follow, both pinned by rngSetSeed(4242u)
     * above (the C2 test makes the same disclosure for its own proxy):
     * (a) "moved off the zero state" is a PROXY -- the zero state IS
     *     exponent == bias == 127, so a grid whose absmax happened to derive
     *     exactly 127 would false-fail. At mantissaBits 8 (qMax 127) that
     *     needs a per-group absmax in [64, 127]; this fixture's values are of
     *     order 0.5, three binades away, so it is robust but not immune.
     * (b) lastLoss < firstLoss is the genuinely seed-sensitive one: the wires
     *     round SR_HALF_AWAY (stochastic), so the trajectory depends on the
     *     draw. A seed change -- or an RNG-consumption change anywhere
     *     upstream -- must re-confirm both. The four STRUCTURAL assertions
     *     (arith slots, weight dtype, both exponent carries) do not depend on
     *     the seed. */
    TEST_ASSERT_TRUE_MESSAGE(fwdSourceGridMoved,
                             "non-vacuity: MaxPool's output grid must have left the zero state");
    TEST_ASSERT_TRUE_MESSAGE(agradSourceGridMoved,
                             "non-vacuity: the grad entering Flatten must have left the zero "
                             "state");
    TEST_ASSERT_EQUAL_UINT8_ARRAY_MESSAGE(carry.fwdPool, carry.fwdFlat, carry.nFwdPool,
                                          "Flatten forward must carry the per-group exponents "
                                          "verbatim (Task 3)");
    TEST_ASSERT_EQUAL_UINT8_ARRAY_MESSAGE(carry.agradFlat, carry.agradPool, carry.nAgradFlat,
                                          "Flatten backward must carry the per-group exponents "
                                          "verbatim (Task 3)");
}

/* ===========================================================================
 * BFP epic PR5 Task 7 capstones: the NORM topology end to end.
 * ======================================================================== */

/* Geometry of the norm capstone below. conv1d(2->4, K=3, SAME, bias-less) ->
 * groupNorm(G=1, C=4) -> relu -> adaptiveAvgPool1d(1) -> flatten ->
 * layerNorm({4}) -> linear(4->2) -> softmax, CrossEntropy + REDUCTION_MEAN.
 * The produced-wire element counts along that chain are
 *   32 / 32 / 32 / 4 / 4 / 4 / 2 / 2
 * -- every one divisible by the template groupSize 2, which the wire allocator
 * REQUIRES (Decision 5; a non-divisor aborts the process, so a green run is the
 * proof). The two 2-element wires normalize to per-tensor {1, 0} (groupSize ==
 * wire elements). The GEMM PARAMS use per-tensor {1, 0} BFP: the §2 param rule
 * demands groupSize divide the reduction run (conv: Cin*K = 6, linear:
 * inFeatures = 4), and per-tensor satisfies it unconditionally. The NORM params
 * come straight from the Task 6 factories, which derive {2, 2} from the
 * 4-element gamma/beta and the template's groupSize. */
#define BFP_NORM_IN_CHANNELS 2
#define BFP_NORM_CONV_CHANNELS 4
#define BFP_NORM_SEQ_LEN 8
#define BFP_NORM_NUM_CLASSES 2
#define BFP_NORM_MODEL_SIZE 8
#define BFP_NORM_LAYERNORM_IDX 5 /* model index of the LayerNorm -- the probe key */
#define BFP_NORM_CONV_W_COUNT (BFP_NORM_CONV_CHANNELS * BFP_NORM_IN_CHANNELS * 3)
#define BFP_NORM_LIN_W_COUNT (BFP_NORM_NUM_CLASSES * BFP_NORM_CONV_CHANNELS)

/* The LayerNorm's OUTPUT wire is the observable for "the norms' OUT_WRITE
 * derived a live BFP grid": it is allocated fresh by the training loop every
 * step, so the only way to read its exponents is from inside the run. First
 * occurrence only -- the claim is that the grid is derived at all, and step 1
 * is the strictest point at which to make it. */
typedef struct pr5NormCapture {
    bool seenFwdLayerNorm;
    size_t nFwdLayerNorm;
    uint8_t fwdLayerNorm[BFP_PR4_MAX_GROUPS];
} pr5NormCapture_t;

static void capturePr5LayerNormWire(void *ctx, size_t layerIdx, layerType_t layerType,
                                    const char *phase, tensor_t *tensor) {
    (void)layerType;
    pr5NormCapture_t *cap = ctx;
    if (layerIdx == BFP_NORM_LAYERNORM_IDX && strcmp(phase, "fwd") == 0) {
        pr4SnapExponents(&cap->seenFwdLayerNorm, &cap->nFwdLayerNorm, cap->fwdLayerNorm, tensor);
    }
}

/* FLOAT32 parameter_t from explicit values (buildRampParam2D/3D's twin -- a
 * ramp gives every conv output channel the SAME cross-channel weight
 * difference, which makes the four conv channels near-degenerate on a
 * two-channel sign-flipped fixture). The §5.2 recipe then requantizes the
 * param tensor in place. */
static parameter_t *buildFloatParam3D(size_t d0, size_t d1, size_t d2, const float *values) {
    tensor_t *param = buildFloatTensor3D(d0, d1, d2, values);
    return parameterInit(param, gradInitFloat(param, NULL));
}

static parameter_t *buildFloatParam2D(size_t d0, size_t d1, const float *values) {
    tensor_t *param = buildFloatTensor2D(d0, d1, values);
    return parameterInit(param, gradInitFloat(param, NULL));
}

/* Both norm factories ALWAYS free gamma/beta, and freeOptim already freed every
 * parameter it registered -- so after freeOptim the layers must come down
 * shell-only (the UnitTestGroupNormIntegration.c / UnitTestLayerNormIntegration.c
 * pattern). Both are Borrowing factories (ownsQuantizations == false), so the
 * shared wire config is freed once by the test, not by the shells. */
static void freeLayerNormLayerShellOnly(layer_t *layer) {
    freeReservedMemory(layer->config->layerNorm->normalizedShape);
    freeReservedMemory(layer->config->layerNorm);
    freeReservedMemory(layer->config);
    freeReservedMemory(layer);
}

static void freeGroupNormLayerShellOnly(layer_t *layer) {
    freeReservedMemory(layer->config->groupNorm);
    freeReservedMemory(layer->config);
    freeReservedMemory(layer);
}

/*! BFP epic PR5 capstone: the whole NORM topology on ONE uniform BFP wire
 *  profile (`quantizationInitBfpGrouped(8, 8, SR_HALF_AWAY, 2, 2)` through
 *  layerQuantInitUniform), trained with SGD+momentum through CrossEntropy.
 *  Every PR5 arm is on the critical path:
 *    - GroupNorm (Tasks 4/5) and LayerNorm (Tasks 2/3) run native ARITH_BFP
 *      forward AND backward, both anchored on the uniform wire config;
 *    - their gamma/beta are BFP-STORED, allocated by the Task 6 factories
 *      (constant-fill on the BFP grid: 1.0 is exact at m = 8, 0.0 is the
 *      zero state) with the {2, 2} geometry derived from the param length;
 *    - the Reduce BFP arms (Task 1) supply both norms' mean/variance;
 *    - the optimizer's OUT_WRITE requant writes the trained gammas back INTO
 *      BFP storage.
 *  Everything around them is PR2-PR4 machinery re-exercised in a topology
 *  those PRs never ran: conv1d/linear native BFP GEMMs over BFP params
 *  (FLOAT32-init + requantizeTensorInPlace, the §5.2 recipe -- the #270
 *  requireFloat32 gate keeps random-init factories FLOAT32-only), relu's
 *  packed-domain transparency, AdaptiveAvgPool1d's BFP forward/backward,
 *  Flatten's exponent carry, and softmax's NATIVE forward (P6-8: derived
 *  ARITH_BFP dispatches into Task 4's i-exp kernel) whose native backward
 *  (Task 5) CrossEntropy's fused gradient still skips entirely.
 *
 *  Seed discipline: rngSetSeed(4242u) pins the SR_HALF_AWAY draws, so the
 *  trajectory -- and with it `lastLoss < firstLoss` -- is deterministic but
 *  seed-sensitive; the structural assertions (arith slots, wire dtypes) are
 *  not. 4242u is the FIRST seed tried and it passed, so no adjacent seed was
 *  needed. */
void testBfpUniformNormModelTrainsAndGridsMove(void) {
    rngSetSeed(4242u);
    quantization_t *bfpWireQ = quantizationInitBfpGrouped(8, 8, SR_HALF_AWAY, 2, 2);
    layerQuant_t lq;
    layerQuantInitUniform(&lq, bfpWireQ);

    /* conv1d [1, 2, 8] -> [1, 4, 8]; SAME padding, stride 1, bias-less. The
     * kernel_t is heap-allocated because freeConv1dLayerShellOnly frees it. */
    static const float convWValues[BFP_NORM_CONV_W_COUNT] = {
        0.35f,  -0.20f, 0.15f,  -0.30f, 0.25f, -0.10f, -0.25f, 0.40f,
        -0.15f, 0.20f,  -0.35f, 0.30f,  0.10f, 0.30f,  -0.45f, -0.15f,
        -0.20f, 0.40f,  -0.40f, -0.10f, 0.25f, 0.35f,  0.15f,  -0.30f};
    parameter_t *convW =
        buildFloatParam3D(BFP_NORM_CONV_CHANNELS, BFP_NORM_IN_CHANNELS, 3, convWValues);
    quantization_t *convWQ = quantizationInitBfp(8, 8, SR_HALF_AWAY);
    requantizeTensorInPlace(getParamFromParameter(convW), convWQ);
    freeQuantization(convWQ); /* requantizeTensorInPlace clones the template */
    kernel_t *convKernel = reserveMemory(sizeof(kernel_t));
    initKernel(convKernel, 3, SAME, /*dilation=*/1, /*stride=*/1);
    layer_t *conv = buildBorrowedConv1dLayer(convW, NULL, convKernel, bfpWireQ);

    layer_t *groupNorm = groupNormLayerInit(
        &(groupNormInit_t){.numGroups = 1, .numChannels = BFP_NORM_CONV_CHANNELS}, &lq);
    layer_t *relu = reluLayerInit(&lq);
    layer_t *pool = adaptiveAvgPool1dLayerInit(&(adaptiveAvgPool1dInit_t){.outputSize = 1}, &lq);
    layer_t *flatten = flattenLayerInit(); /* [1, 4, 1] -> [1, 4] */
    size_t normShape[1] = {BFP_NORM_CONV_CHANNELS};
    layer_t *layerNorm =
        layerNormLayerInit(&(layerNormInit_t){.normalizedShape = normShape, .numNormDims = 1}, &lq);

    static const float linWValues[BFP_NORM_LIN_W_COUNT] = {0.30f,  -0.20f, 0.15f,  -0.35f,
                                                           -0.25f, 0.40f,  -0.10f, 0.20f};
    static const float linBValues[BFP_NORM_NUM_CLASSES] = {0.05f, -0.05f};
    parameter_t *linW = buildFloatParam2D(BFP_NORM_NUM_CLASSES, BFP_NORM_CONV_CHANNELS, linWValues);
    parameter_t *linB = buildFloatParam2D(1, BFP_NORM_NUM_CLASSES, linBValues);
    quantization_t *linWQ = quantizationInitBfp(8, 8, SR_HALF_AWAY);
    quantization_t *linBQ = quantizationInitBfp(8, 8, SR_HALF_AWAY);
    requantizeTensorInPlace(getParamFromParameter(linW), linWQ);
    requantizeTensorInPlace(getParamFromParameter(linB), linBQ);
    freeQuantization(linBQ);
    freeQuantization(linWQ);
    layer_t *linear = buildBorrowedLinearLayer(linW, linB, bfpWireQ);

    /* Same flip as the PR4 capstone (P6-8): softmax now runs its NATIVE
     * forward (derived ARITH_BFP), while CrossEntropy's fused backward still
     * skips the layer entirely -- see unitTestUniformBfpSoftmaxMseTrains for
     * the native-backward-through-the-loop coverage. */
    layer_t *softmax = softmaxLayerInit(&lq);

    layer_t *model[BFP_NORM_MODEL_SIZE] = {conv,    groupNorm, relu,   pool,
                                           flatten, layerNorm, linear, softmax};

    /* lr 0.01 deliberately, not the PR4 capstone's 0.05: at 0.05 this fixture
     * converges so hard that the CE loss hits EXACTLY 0.0 by step 9 -- the
     * 2-element softmax wire quantizes p = 0.998 to code 64 at the group's own
     * exponent, i.e. exactly 1.0, so -log(p) == 0 and the last three steps see
     * a zero gradient. A saturated endpoint costs the test its REGRESSION
     * SENSITIVITY: `lastLoss < firstLoss` still holds after a kernel
     * regression that only halves the learning signal, because a 2x slower
     * descent still reaches exactly 0.0 inside 12 steps. At 0.01 the curve is
     * a monotone 0.219 -> 0.056 with every step non-degenerate, so a partial
     * regression moves the endpoint and stays observable. */
    quantization_t *momentumQ = quantizationInitFloat();
    optimizer_t *sgd =
        sgdMCreateOptim(0.01f, 0.9f, 0.f, model, BFP_NORM_MODEL_SIZE, momentumQ,
                        (arithmetic_t){.type = ARITH_FLOAT32, .roundingMode = HALF_AWAY});

    /* Two-sample, two-class fixture: channel 0 positive / channel 1 negative is
     * class 0, signs flipped is class 1 (the UnitTestGroupNormIntegration.c
     * fixture at rank 3, which GroupNorm REQUIRES: [B, C, T]). Deterministic --
     * the only RNG consumer in this test is the SR_HALF_AWAY rounding. */
    float itemA[BFP_NORM_IN_CHANNELS * BFP_NORM_SEQ_LEN];
    float itemB[BFP_NORM_IN_CHANNELS * BFP_NORM_SEQ_LEN];
    for (size_t t = 0; t < BFP_NORM_SEQ_LEN; t++) {
        float ramp = 0.05f * (float)t;
        itemA[0 * BFP_NORM_SEQ_LEN + t] = 1.5f + ramp;
        itemA[1 * BFP_NORM_SEQ_LEN + t] = -1.5f - ramp;
        itemB[0 * BFP_NORM_SEQ_LEN + t] = -1.6f - ramp;
        itemB[1 * BFP_NORM_SEQ_LEN + t] = 1.6f + ramp;
    }
    tensor_t *inputs[2] = {buildFloatTensor3D(1, BFP_NORM_IN_CHANNELS, BFP_NORM_SEQ_LEN, itemA),
                           buildFloatTensor3D(1, BFP_NORM_IN_CHANNELS, BFP_NORM_SEQ_LEN, itemB)};
    tensor_t *labels[2] = {buildFloatTensor2D(1, BFP_NORM_NUM_CLASSES, (float[]){1.0f, 0.0f}),
                           buildFloatTensor2D(1, BFP_NORM_NUM_CLASSES, (float[]){0.0f, 1.0f})};

    /* Snapshot the LayerNorm gamma's packed payload: the SGD write-back must
     * land IN BFP STORAGE, so the codes have to leave the all-ones seed. */
    tensor_t *lnGammaTensor = getParamFromParameter(layerNorm->config->layerNorm->gamma);
    size_t lnGammaBytes =
        calcNumberOfBytesForData(lnGammaTensor->quantization, BFP_NORM_CONV_CHANNELS);
    uint8_t gammaCodesBefore[16];
    memcpy(gammaCodesBefore, lnGammaTensor->data, lnGammaBytes);

    optimizerFunctions_t sgdFns = optimizerFunctions[SGD_M];
    pr5NormCapture_t cap = {0};
    float firstLoss = NAN;
    float lastLoss = NAN;
    for (size_t step = 0; step < 12; step++) {
        float stepLoss = 0.f;
        for (size_t s = 0; s < 2; s++) {
            trainingStats_t *stats =
                tracedGrads(model, BFP_NORM_MODEL_SIZE, defaultLossConfig(CROSS_ENTROPY),
                            REDUCTION_MEAN, inputs[s], labels[s], capturePr5LayerNormWire, &cap);
            stepLoss += stats->loss;
            freeTrainingStats(stats);
        }
        if (step == 0) {
            firstLoss = 0.5f * stepLoss;
        }
        lastLoss = 0.5f * stepLoss;
        sgdFns.step(sgd);
        sgdFns.zero(sgd);
    }

    /* CAPTURE -> FREE (reverse init order) -> assert (Unity longjmps out of the
     * first failure, so nothing may be read after the teardown). */
    bool gammaCodesMoved = memcmp(gammaCodesBefore, lnGammaTensor->data, lnGammaBytes) != 0;
    layerNormConfig_t *lnCfg = layerNorm->config->layerNorm;
    groupNormConfig_t *gnCfg = groupNorm->config->groupNorm;
    bool normsDeclareBfpMath =
        lnCfg->forwardMath.type == ARITH_BFP && lnCfg->propLossMath.type == ARITH_BFP &&
        gnCfg->forwardMath.type == ARITH_BFP && gnCfg->propLossMath.type == ARITH_BFP;
    bool normPropLossWiresBfp = lnCfg->propLossQ != NULL && lnCfg->propLossQ->type == BFP &&
                                gnCfg->propLossQ != NULL && gnCfg->propLossQ->type == BFP;
    bool normParamsBfpStored = lnCfg->gamma->param->quantization->type == BFP &&
                               lnCfg->beta->param->quantization->type == BFP &&
                               gnCfg->gamma->param->quantization->type == BFP &&
                               gnCfg->beta->param->quantization->type == BFP;
    /* P6-8: the pin removal must be observable, not just declared. */
    softmaxConfig_t *smCfg = softmax->config->softmax;
    bool softmaxDeclaresBfpMath =
        smCfg->forwardMath.type == ARITH_BFP && smCfg->propLossMath.type == ARITH_BFP;
    const uint8_t zeroState = 127; /* exponentBits 8 -> bias 127 */
    bool lnWireGridMoved = false;
    for (size_t g = 0; g < cap.nFwdLayerNorm; g++) {
        if (cap.fwdLayerNorm[g] != zeroState) {
            lnWireGridMoved = true;
        }
    }
    bool lnProbeFired = cap.seenFwdLayerNorm;
    size_t lnWireGroups = cap.nFwdLayerNorm;

    freeTensor(labels[1]);
    freeTensor(labels[0]);
    freeTensor(inputs[1]);
    freeTensor(inputs[0]);
    freeOptim(sgd); /* frees conv weight, both norms' gamma/beta, linear w/b */
    freeSoftmaxLayer(softmax);
    freeLinearLayerShellOnly(linear);
    freeLayerNormLayerShellOnly(layerNorm);
    freeFlattenLayer(flatten);
    freeAdaptiveAvgPool1dLayer(pool);
    freeReluLayer(relu);
    freeGroupNormLayerShellOnly(groupNorm);
    freeConv1dLayerShellOnly(conv);
    freeQuantization(momentumQ);
    freeQuantization(bfpWireQ);

    TEST_ASSERT_TRUE_MESSAGE(normsDeclareBfpMath,
                             "layerQuantInitUniform over ONE BFP profile must DERIVE ARITH_BFP "
                             "forwardMath AND propLossMath on BOTH norms (asserted, not hand-set)");
    TEST_ASSERT_TRUE_MESSAGE(normPropLossWiresBfp,
                             "both norms' dx wires must be BFP-typed by config -- the whole "
                             "backward chain ran on packed grads");
    TEST_ASSERT_TRUE_MESSAGE(normParamsBfpStored,
                             "the Task 6 factories must have allocated BFP-stored gamma AND beta "
                             "for both norms");
    TEST_ASSERT_TRUE_MESSAGE(softmaxDeclaresBfpMath,
                             "the capstone's softmax must derive native ARITH_BFP in both math "
                             "slots too (P6-8 flip)");
    TEST_ASSERT_TRUE_MESSAGE(isfinite(firstLoss) && isfinite(lastLoss),
                             "uniform-BFP norm training must stay finite through every PR5 arm");
    TEST_ASSERT_TRUE_MESSAGE(lastLoss < firstLoss,
                             "uniform-BFP conv->groupNorm->relu->pool->flatten->layerNorm->linear"
                             "->softmax+CE must converge (the vision-gate acceptance)");
    TEST_ASSERT_TRUE_MESSAGE(lnProbeFired, "the LayerNorm forward-wire probe must have fired");
    TEST_ASSERT_EQUAL_UINT_MESSAGE(2, lnWireGroups,
                                   "the LayerNorm output wire is 4 elements / groupSize 2");
    /* Two seed-dependent claims, both pinned by rngSetSeed(4242u): "moved off
     * the zero state" is a PROXY (the zero state IS exponent == bias == 127, so
     * a grid whose absmax happened to derive exactly 127 would false-fail), and
     * lastLoss < firstLoss depends on the SR_HALF_AWAY draw. A seed change --
     * or an RNG-consumption change anywhere upstream -- must re-confirm both. */
    TEST_ASSERT_TRUE_MESSAGE(lnWireGridMoved,
                             "the LayerNorm forward's OUT_WRITE must DERIVE the output wire's "
                             "exponent grid: at least one group must leave the zero state");
    TEST_ASSERT_TRUE_MESSAGE(gammaCodesMoved,
                             "the SGD write-back must land in the LayerNorm gamma's BFP storage: "
                             "the packed codes must leave the all-ones seed");
}

/*! Task 7's second capstone -- the norm twin of
 *  testBfpGradStorageTrainingAccumulatesAndSteps (which pins the knob on a
 *  LINEAR weight only). A single factory-built LayerNorm on the same uniform
 *  BFP profile, with per-tensor BFP {8, 8, SR_HALF_AWAY} storage on BOTH
 *  gamma's and beta's GRAD (Task 6's rule-8 positive half). One backward pass
 *  through the training loop's grad calculation exercises, in order:
 *    - the norms' ARITH_BFP dgamma/dbeta ops writing through accumulateOut's
 *      BFP-target arm (accumulateFloatIntoBfpTensorRescale) into packed
 *      storage -- codes AND a derived exponent;
 *    - the optimizer's read of those grads through conversionMatrix[BFP]
 *      [FLOAT32] and its OUT_WRITE requant back into gamma's BFP storage;
 *    - optimizerZeroGrad's BFP arm resetting codes to zero AND exponents to
 *      bias (the SYM/ASYM-parity hygiene contract).
 *  The LayerNorm is the deepest -- and only -- trainable layer, so #380 PR2
 *  truncation hands its backward propLoss == NULL: the dgamma/dbeta ops run,
 *  the dx op does not. That is the point; dx has its own PR5 coverage. */
void testBfpNormGradStorageAccumulatesAndSteps(void) {
    rngSetSeed(4242u);
    quantization_t *bfpWireQ = quantizationInitBfpGrouped(8, 8, SR_HALF_AWAY, 2, 2);
    quantization_t *gradKnob = quantizationInitBfp(8, 8, SR_HALF_AWAY);
    layerQuant_t lq;
    layerQuantInitUniform(&lq, bfpWireQ);
    lq.weightGradStorage = gradKnob;
    lq.biasGradStorage = gradKnob;

    size_t normShape[1] = {BFP_NORM_CONV_CHANNELS};
    layer_t *layerNorm =
        layerNormLayerInit(&(layerNormInit_t){.normalizedShape = normShape, .numNormDims = 1}, &lq);
    freeQuantization(gradKnob); /* gradInit deep-clones via getQLike */

    layer_t *model[1] = {layerNorm};
    quantization_t *momentumQ = quantizationInitFloat();
    optimizer_t *sgd =
        sgdMCreateOptim(0.2f, 0.f, 0.f, model, 1, momentumQ,
                        (arithmetic_t){.type = ARITH_FLOAT32, .roundingMode = HALF_AWAY});

    tensor_t *input =
        buildFloatTensor2D(1, BFP_NORM_CONV_CHANNELS, (float[]){0.9f, -0.4f, 1.3f, 0.2f});
    tensor_t *label =
        buildFloatTensor2D(1, BFP_NORM_CONV_CHANNELS, (float[]){0.5f, -0.5f, 1.0f, -1.0f});

    layerNormConfig_t *cfg = layerNorm->config->layerNorm;
    tensor_t *gammaGrad = getGradFromParameter(cfg->gamma);
    tensor_t *betaGrad = getGradFromParameter(cfg->beta);
    /* (a) the knob landed BFP grad storage on BOTH norm parameters. */
    int gammaGradType = (int)gammaGrad->quantization->type;
    int betaGradType = (int)betaGrad->quantization->type;

    /* Sentinels keep the CAPTURE phase crash-free if the knob ever regresses to
     * the FLOAT32 default -- a FLOAT32 grad carries a NULL qConfig, and a
     * null deref here would replace the clean dtype assertion with a segfault. */
    uint8_t zeroStateBias = 0;
    size_t gradBytes = 0;
    size_t gradNumGroups = 0;
    if (gammaGradType == BFP && betaGradType == BFP) {
        zeroStateBias = (uint8_t)bfpExponentBias(gammaGrad->quantization->qConfig);
        gradNumGroups = ((bfpQConfig_t *)gammaGrad->quantization->qConfig)->numGroups;
        gradBytes = calcNumberOfBytesForData(gammaGrad->quantization, BFP_NORM_CONV_CHANNELS);
    }

    tensor_t *gammaParam = getParamFromParameter(cfg->gamma);
    size_t paramBytes = calcNumberOfBytesForData(gammaParam->quantization, BFP_NORM_CONV_CHANNELS);
    uint8_t paramBefore[16];
    memcpy(paramBefore, gammaParam->data, paramBytes);

    /* A fresh BFP grad is the canonical zero state: all-zero codes, exponent ==
     * bias. Pinned BEFORE the backward so the post-zeroGrad assertions below
     * are a genuine round trip and not a restatement of the initial state. */
    bool freshCodesZero = true;
    for (size_t i = 0; i < gradBytes; i++) {
        if (((const uint8_t *)gammaGrad->data)[i] != 0u) {
            freshCodesZero = false;
        }
    }

    trainingStats_t *stats =
        calculateGradsSequential(model, 1, defaultLossConfig(MSE), REDUCTION_MEAN, input, label);
    float loss = stats->loss;
    freeTrainingStats(stats);

    /* (b) the ACC arm wrote packed codes AND derived a grid. Read BEFORE the
     * step/zero -- zeroGrad resets the exponent to bias, so this is the only
     * point where the accumulate arm's moved grid is observable. */
    bool gammaGradCodesNonZero = false;
    bool betaGradCodesNonZero = false;
    for (size_t i = 0; i < gradBytes; i++) {
        if (((const uint8_t *)gammaGrad->data)[i] != 0u) {
            gammaGradCodesNonZero = true;
        }
        if (((const uint8_t *)betaGrad->data)[i] != 0u) {
            betaGradCodesNonZero = true;
        }
    }
    uint8_t gammaGradExponentAfterBackward =
        gradBytes > 0 ? ((bfpQConfig_t *)gammaGrad->quantization->qConfig)->exponents[0] : 0;

    /* (c) the step must move the BFP-stored gamma, read back through the BFP
     * grad. gamma's seed is EXACTLY 1.0 (code 64 at stored exponent bias - 6),
     * so one code is 2^-6 == 0.015625 and the update has to clear half of that
     * to be observable at all -- which is why lr is 0.2 here and not the 0.01
     * the model capstone above uses: |dgamma| runs ~0.1-2.4 on this fixture, so
     * every one of the four codes moves by tens of LSBs, far from the knife
     * edge. Hand-checked against the kernel math: the four post-step gammas
     * (0.972 / 0.513 / 0.889 / 1.099) reproduce codes 125, 65 | 57, 71 at the
     * two groups' re-derived exponents. */
    optimizerFunctions_t sgdFns = optimizerFunctions[SGD_M];
    sgdFns.step(sgd);
    bool paramMoved = memcmp(paramBefore, gammaParam->data, paramBytes) != 0;

    /* (d) zeroGrad's BFP arm: codes back to zero AND exponents back to bias. */
    sgdFns.zero(sgd);
    bool zeroedCodes = true;
    for (size_t i = 0; i < gradBytes; i++) {
        if (((const uint8_t *)gammaGrad->data)[i] != 0u ||
            ((const uint8_t *)betaGrad->data)[i] != 0u) {
            zeroedCodes = false;
        }
    }
    uint8_t gammaGradExponentAfterZero =
        gradBytes > 0 ? ((bfpQConfig_t *)gammaGrad->quantization->qConfig)->exponents[0] : 0;
    int gammaGradTypeAfter = (int)gammaGrad->quantization->type;

    /* CAPTURE -> FREE (reverse init order) -> assert. */
    freeTensor(label);
    freeTensor(input);
    freeOptim(sgd); /* frees gamma/beta + their BFP grads */
    freeLayerNormLayerShellOnly(layerNorm);
    freeQuantization(momentumQ);
    freeQuantization(bfpWireQ);

    TEST_ASSERT_EQUAL_INT_MESSAGE(
        BFP, gammaGradType, "weightGradStorage must land BFP storage on the LayerNorm gamma grad");
    TEST_ASSERT_EQUAL_INT_MESSAGE(
        BFP, betaGradType, "biasGradStorage must land BFP storage on the LayerNorm beta grad");
    TEST_ASSERT_EQUAL_size_t_MESSAGE(1, gradNumGroups, "grads are per-tensor-only (#300 axis)");
    TEST_ASSERT_TRUE_MESSAGE(freshCodesZero,
                             "guard: a fresh BFP grad must start with all-zero codes -- otherwise "
                             "the post-backward and post-zeroGrad claims below are vacuous");
    TEST_ASSERT_TRUE_MESSAGE(isfinite(loss), "the BFP norm backward's loss must be finite");
    TEST_ASSERT_TRUE_MESSAGE(gammaGradCodesNonZero,
                             "the ARITH_BFP dgamma op must have accumulated non-zero packed codes "
                             "into the BFP grad (accumulateFloatIntoBfpTensorRescale)");
    TEST_ASSERT_TRUE_MESSAGE(betaGradCodesNonZero,
                             "the ARITH_BFP dbeta op must have accumulated non-zero packed codes "
                             "into the BFP grad");
    /* Seed-dependent proxy, same disclosure as the PR3/PR4 grad-storage
     * capstones: the zero state IS exponent == bias, so a grad whose absmax
     * happened to derive exactly that would false-fail. rngSetSeed(4242u) pins
     * it; a seed change must re-confirm. */
    TEST_ASSERT_NOT_EQUAL_MESSAGE(zeroStateBias, gammaGradExponentAfterBackward,
                                  "the accumulate arm must have moved the gamma grad's exponent "
                                  "off the zero state during backward");
    TEST_ASSERT_TRUE_MESSAGE(paramMoved,
                             "an SGD step must move the BFP-stored gamma, read back through the "
                             "BFP grad via conversionMatrix[BFP][FLOAT32]");
    TEST_ASSERT_TRUE_MESSAGE(zeroedCodes,
                             "optimizerZeroGrad's BFP arm must reset every packed code to zero");
    TEST_ASSERT_EQUAL_UINT8_MESSAGE(zeroStateBias, gammaGradExponentAfterZero,
                                    "optimizerZeroGrad's BFP arm must reset the exponent back to "
                                    "bias (the SYM/ASYM-parity hygiene contract)");
    TEST_ASSERT_EQUAL_INT_MESSAGE(BFP, gammaGradTypeAfter,
                                  "the gamma grad must stay BFP-stored across step + zero");
}

/* ===========================================================================
 * BFP epic PR6 Task 2 (P6-1) capstone: loop-contract e2e.
 * ======================================================================== */

/*! THE strongest pin that P6-1's root fix is wired correctly end to end: the
 *  training loop hands softmaxBackward the layer's INPUT (layerOutputs[i]),
 *  and a wrong dx through softmax poisons the UPSTREAM linear layer's weight
 *  grad. Model: Linear(3->3, ramp weights) -> Softmax, MSE loss, ONE
 *  calculateGradsSequential call. softmaxMseE2eExpectedWeightGrad is
 *  goldgen'd (generate_expected_softmax.py) via torch.autograd through the
 *  WHOLE chain (F.linear -> softmax -> mse_loss(reduction='sum'), matching
 *  the repo's raw-per-element MSE backward convention -- docs/conventions/
 *  loss.md).
 *
 *  This test does NOT independently fail against pre-fix code: Steps 3-5 of
 *  this task already fixed both softmax backward arms, so by the time this
 *  test exists the fix is already live. Its RED/GREEN evidence is instead a
 *  mutation check: temporarily revert softmaxBackwardFloat's recompute (use
 *  the layer input `x` as `s` again, matching the pre-fix bug), rerun, and
 *  confirm THIS test fails -- proof the loop contract (calculateGradsSequential
 *  handing softmax the LOGITS, not probabilities) is covered end to end, not
 *  just at the unit level. See the task-2 report for the transcript. */
void unitTestSoftmaxMseBackwardThroughLoop(void) {
    quantization_t *q = quantizationInitFloat();

    parameter_t *w0 = buildRampParam2D(3, 3, 0.1f, 0.05f);
    parameter_t *b0 = buildRampParam2D(1, 3, 0.0f, 0.0f);
    layer_t *linear0 = buildBorrowedLinearLayer(w0, b0, q);

    layerQuant_t lq;
    layerQuantInitUniform(&lq, q);
    layer_t *softmax = softmaxLayerInit(&lq);

    layer_t *model[] = {linear0, softmax};

    tensor_t *input = buildFloatTensor2D(1, 3, (float[]){1.0f, -0.5f, 2.0f});
    tensor_t *label = buildFloatTensor2D(1, 3, (float[]){0.2f, 0.5f, 0.3f});

    trainingStats_t *stats =
        calculateGradsSequential(model, 2, defaultLossConfig(MSE), REDUCTION_MEAN, input, label);

    /* CAPTURE. */
    tensor_t *w0GradTensor = getGradFromParameter(w0);
    float capturedWeightGrad[9];
    for (size_t i = 0; i < 9; i++) {
        capturedWeightGrad[i] = ((float *)w0GradTensor->data)[i];
    }

    /* FREE (reverse init order). */
    freeTrainingStats(stats);
    freeTensor(label);
    freeTensor(input);
    freeSoftmaxLayer(softmax);
    freeLinearLayerShellOnly(linear0);
    freeParameter(b0);
    freeParameter(w0);
    freeQuantization(q);

    /* ASSERT. */
    for (size_t i = 0; i < 9; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-4f, softmaxMseE2eExpectedWeightGrad[i], capturedWeightGrad[i]);
    }
}

/* ===========================================================================
 * BFP epic PR6 Task 6 (P6-8) capstone: uniform-BFP softmax+MSE e2e.
 * ======================================================================== */

/*! The non-CE twin of the two capstones above: MSE does not special-case
 *  softmax the way CalculateGradsSequential.c's CROSS_ENTROPY branch does
 *  (backwardIndex -= 1), so this IS the model that drives
 *  softmaxBackwardKernelBfp (Task 5's native funnel arm) through the
 *  training loop -- linear(4->3) -> softmax, ONE uniform BFP wire profile
 *  (m=8/e=8) through layerQuantInitUniform, SGD+momentum.
 *
 *  This test pins the WIRING (loss decreases, the linear weight's packed
 *  codes move off their seed), not exact values -- a wrong-gradient mutant
 *  that still decreases loss is caught by the Task-2/Task-5 gold tests, not
 *  here. REACHABILITY evidence (verified during development, not committed):
 *  pinning propLossMath back to ARITH_FLOAT32 does NOT make this test
 *  silently pass on a fake-quant substitute -- softmaxBackward's FLOAT32 arm
 *  runs its own bfpRequireNoBfpWire guard (Task 5) against a BFP-typed wire
 *  and exit(1)s before any compute, because this fixture's propLossQ stays
 *  BFP regardless of propLossMath (a fake-quant softmax backward needs BOTH
 *  the math AND the wire declared non-BFP; this uniform fixture never gives
 *  it that). A second, independent check confirms it is specifically the
 *  NATIVE kernel that runs on the UNMUTATED config: a temporary exit(3)
 *  poison of softmaxBackwardKernelBfp made this test die with that exact
 *  code (see the task-6 report for the transcript -- a permanent poison
 *  would defeat its own purpose). */
void unitTestUniformBfpSoftmaxMseTrains(void) {
    rngSetSeed(4242u);
    quantization_t *bfpWireQ = quantizationInitBfp(8, 8, SR_HALF_AWAY);
    layerQuant_t lq;
    layerQuantInitUniform(&lq, bfpWireQ);

    /* linear 4 -> 3. */
    parameter_t *linW = buildRampParam2D(3, 4, 0.10f, 0.03f);
    parameter_t *linB = buildRampParam2D(1, 3, 0.05f, 0.05f);
    quantization_t *linWQ = quantizationInitBfp(8, 8, SR_HALF_AWAY);
    quantization_t *linBQ = quantizationInitBfp(8, 8, SR_HALF_AWAY);
    requantizeTensorInPlace(getParamFromParameter(linW), linWQ);
    requantizeTensorInPlace(getParamFromParameter(linB), linBQ);
    freeQuantization(linBQ);
    freeQuantization(linWQ);
    layer_t *linear = buildBorrowedLinearLayer(linW, linB, bfpWireQ);

    layer_t *softmax = softmaxLayerInit(&lq);

    layer_t *model[2] = {linear, softmax};

    quantization_t *momentumQ = quantizationInitFloat();
    optimizer_t *sgd =
        sgdMCreateOptim(0.05f, 0.9f, 0.f, model, 2, momentumQ,
                        (arithmetic_t){.type = ARITH_FLOAT32, .roundingMode = HALF_AWAY});

    float inputValues[4] = {0.9f, -0.4f, 1.3f, 0.2f};
    tensor_t *input = buildFloatTensor2D(1, 4, inputValues);
    tensor_t *label = buildFloatTensor2D(1, 3, (float[]){1.0f, 0.0f, 0.0f});

    /* Snapshot the linear weight's packed payload: the SGD write-back must
     * land in BFP storage, so the codes have to leave the ramp seed. */
    tensor_t *linWTensor = getParamFromParameter(linW);
    size_t linWBytes = calcNumberOfBytesForData(linWTensor->quantization, 12);
    uint8_t before[16];
    memcpy(before, linWTensor->data, linWBytes);

    optimizerFunctions_t sgdFns = optimizerFunctions[SGD_M];
    float firstLoss = NAN;
    float lastLoss = NAN;
    for (size_t step = 0; step < 10; step++) {
        trainingStats_t *stats = calculateGradsSequential(model, 2, defaultLossConfig(MSE),
                                                          REDUCTION_MEAN, input, label);
        if (step == 0) {
            firstLoss = stats->loss;
        }
        lastLoss = stats->loss;
        freeTrainingStats(stats);
        sgdFns.step(sgd);
        sgdFns.zero(sgd);
    }

    /* CAPTURE -> FREE (reverse init order) -> assert (Unity longjmps out of
     * the first failure, so nothing may be read after the teardown). No
     * config-level ARITH_BFP type assertion here on purpose (unlike the two
     * capstones above): that would only confirm the fixture didn't override
     * the derived value, not that softmaxBackwardKernelBfp actually RAN --
     * see the doc comment's mutation-check/reachability-probe note. */
    bool codesMoved = memcmp(before, linWTensor->data, linWBytes) != 0;

    freeTensor(label);
    freeTensor(input);
    freeOptim(sgd);
    freeSoftmaxLayer(softmax);
    freeLinearLayerShellOnly(linear);
    freeQuantization(momentumQ);
    freeQuantization(bfpWireQ);

    TEST_ASSERT_TRUE_MESSAGE(isfinite(firstLoss) && isfinite(lastLoss),
                             "uniform-BFP linear->softmax+MSE training must stay finite");
    TEST_ASSERT_TRUE_MESSAGE(lastLoss < firstLoss,
                             "uniform-BFP linear->softmax+MSE must converge (the non-CE vision-"
                             "gate acceptance)");
    TEST_ASSERT_TRUE_MESSAGE(codesMoved,
                             "the SGD write-back must land in the linear weight's BFP storage");
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
    RUN_TEST(testBfpUniformPoolActivationModelTrains);
    RUN_TEST(testBfpUniformNormModelTrainsAndGridsMove);
    RUN_TEST(testBfpNormGradStorageAccumulatesAndSteps);
    RUN_TEST(unitTestSoftmaxMseBackwardThroughLoop);
    RUN_TEST(unitTestUniformBfpSoftmaxMseTrains);
    return UNITY_END();
}
