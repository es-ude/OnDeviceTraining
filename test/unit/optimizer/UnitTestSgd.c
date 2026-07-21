#define SOURCE_FILE "SGD-UTEST"
#include <stdlib.h>
#include <string.h>

#include "ArithmeticType.h"
#include "BorrowedLayer.h"
#include "DeathTest.h"
#include "Layer.h"
#include "LayerNorm.h"
#include "LayerQuant.h"
#include "Linear.h"
#include "LinearApi.h"
#include "Optimizer.h"
#include "OptimizerApi.h"
#include "QuantizationApi.h"
#include "RNG.h"
#include "Sgd.h"
#include "SgdApi.h"
#include "StorageApi.h"
#include "Tensor.h"
#include "TensorApi.h"
#include "TensorConversion.h"
#include "unity.h"

#include <ReluApi.h>

/* #310 contract: sgdMCreateOptim takes a by-value arithmetic_t updateMath
 * (mirroring the layer-side forwardMath/weightGradMath knobs) as its last
 * parameter -- the arithmetic the three SGD update ops run in. _Generic
 * picks 1 only for the exact 7-arg signature; anything else fails at
 * compile time. */
_Static_assert(_Generic(&sgdMCreateOptim,
                   optimizer_t *(*)(float, float, float, layer_t **, size_t, quantization_t *,
                                    arithmetic_t): 1,
                   default: 0),
               "#310: sgdMCreateOptim must be (lr, momentumFactor, weightDecay, "
               "model, sizeModel, momentumQuant, updateMath)");

/* #328 groundwork: the shared factory helpers PR C's adamWCreateOptim reuses. */
_Static_assert(_Generic(&collectTrainableParameters,
                   void (*)(layer_t **, size_t, parameter_t **): 1,
                   default: 0),
               "#328: collectTrainableParameters must be (model, sizeModel, slots)");
_Static_assert(_Generic(&validateOptimizerGradStorage,
                   void (*)(optimizer_t *, const char *): 1,
                   default: 0),
               "#328: validateOptimizerGradStorage must be (optim, factoryName)");
_Static_assert(_Generic(&modelHasFrozenLayer, bool (*)(layer_t **, size_t): 1, default: 0),
               "#380: modelHasFrozenLayer must be (model, sizeModel)");
_Static_assert(_Generic(&freeOptim, void (*)(optimizer_t *): 1, default: 0),
               "#328: freeOptim must be (optim)");

void setUp() {}
void tearDown() {}

/* Local copy of the Task-1 helper (test/unit/layer/UnitTestLinear.c:355) --
 * tests are independent binaries, so this file builds its own frozen/
 * trainable Linear layer instead of including across test directories. */
static layer_t *buildFloatLinearWithTrainable(trainable_t trainable) {
    quantization_t *q = quantizationInitFloat();
    layerQuant_t lq;
    layerQuantInitUniform(&lq, q);
    layer_t *layer = linearLayerInitOwning(
        &(linearInit_t){.inFeatures = 3, .outFeatures = 2, .trainable = trainable}, &lq);
    freeQuantization(q);
    return layer;
}

/* #380 PR1 Task 4: frozen layers must contribute zero states and never be
 * collected -- a 2-layer model (frozen + trainable) must count/collect as if
 * only the trainable layer existed. */
void testOptimizerSkipsFrozenLayerInCountAndCollection(void) {
    layer_t *frozenL = buildFloatLinearWithTrainable(TRAINABLE_FALSE);
    layer_t *trainL = buildFloatLinearWithTrainable(TRAINABLE_DEFAULT);
    layer_t *model[] = {frozenL, trainL};
    size_t count = calcTotalNumberOfStates(model, 2);

    quantization_t *momentumQ = quantizationInitFloat();
    optimizer_t *optim = sgdMCreateOptim(0.1f, 0.9f, 0.0f, model, 2, momentumQ,
                                         (arithmetic_t){.type = ARITH_FLOAT32});
    size_t sizeStates = optim->sizeStates;
    bool slot0IsTrainWeights = optim->parameter[0] == trainL->config->linear->weights;
    bool slot1IsTrainBias = optim->parameter[1] == trainL->config->linear->bias;
    bool statesAllocated = optim->states != NULL;

    freeOptim(optim);                 /* frees ONLY the collected (trainable) params */
    freeLinearLayerShellOnly(trainL); /* BorrowedLayer.h helper — params already freed */
    freeLinearLayer(frozenL);         /* frozen params NOT collected — full free */
    freeQuantization(momentumQ);

    TEST_ASSERT_EQUAL_size_t(2, count);
    TEST_ASSERT_EQUAL_size_t(2, sizeStates);
    TEST_ASSERT_TRUE(slot0IsTrainWeights);
    TEST_ASSERT_TRUE(slot1IsTrainBias);
    TEST_ASSERT_TRUE(statesAllocated);
}

/* #380 PR1 Task 4: an all-frozen model has zero trainable states -- the
 * factory must fail fast (exit(1)) rather than build a degenerate optimizer
 * (also forecloses the reserveMemory(0)/#160 N=0 edge). */
void testSgdCreateAllFrozenModelExits(void) {
    ASSERT_EXITS_WITH(1, {
        layer_t *frozenL = buildFloatLinearWithTrainable(TRAINABLE_FALSE);
        layer_t *model[] = {frozenL};
        quantization_t *momentumQ = quantizationInitFloat();
        sgdMCreateOptim(0.1f, 0.0f, 0.0f, model, 1, momentumQ,
                        (arithmetic_t){.type = ARITH_FLOAT32});
    });
}

void testSgdMCreateOptim() {
    /* Shared layer-config quantization (caller-owned). */
    quantization_t *layerQ = quantizationInitFloat();

    /* linear0 weights (heap shape, FLOAT32, dims {1, 3}, ndims=2). */
    size_t *w0Dims = reserveMemory(2 * sizeof(size_t));
    w0Dims[0] = 1;
    w0Dims[1] = 3;
    size_t *w0Order = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, w0Order);
    shape_t *w0Shape = reserveMemory(sizeof(shape_t));
    setShape(w0Shape, w0Dims, 2, w0Order);
    tensor_t *w0Param = initTensor(w0Shape, quantizationInitFloat(), NULL);
    tensorFillFromFloatBuffer(w0Param, (float[]){0.f, 1.f, 2.f}, 3);
    tensor_t *w0Grad = gradInitFloat(w0Param, NULL);
    tensorFillFromFloatBuffer(w0Grad, (float[]){0.f, 0.f, 0.f}, 3);
    parameter_t *weights0 = parameterInit(w0Param, w0Grad);

    /* linear0 bias (heap shape, FLOAT32, dims {1, 3}, ndims=1 -> 1 element). */
    size_t *b0Dims = reserveMemory(1 * sizeof(size_t));
    b0Dims[0] = 1;
    size_t *b0Order = reserveMemory(1 * sizeof(size_t));
    setOrderOfDimsForNewTensor(1, b0Order);
    shape_t *b0Shape = reserveMemory(sizeof(shape_t));
    setShape(b0Shape, b0Dims, 1, b0Order);
    tensor_t *b0Param = initTensor(b0Shape, quantizationInitFloat(), NULL);
    tensorFillFromFloatBuffer(b0Param, (float[]){0.f}, 1);
    tensor_t *b0Grad = gradInitFloat(b0Param, NULL);
    tensorFillFromFloatBuffer(b0Grad, (float[]){0.f}, 1);
    parameter_t *bias0 = parameterInit(b0Param, b0Grad);

    /* linear1 needs its OWN weights/bias parameters (Bug 1 fix): freeOptim
     * will walk optim->parameter[] and freeParameter each entry once. Sharing
     * the same parameter_t pointers across linear0 and linear1 would cause a
     * double-free. The pointer-equality assertions below still hold because
     * they compare layer config slots against optim->parameter slots, both
     * of which now point to whichever parameter object the layer was given. */
    size_t *w1Dims = reserveMemory(2 * sizeof(size_t));
    w1Dims[0] = 1;
    w1Dims[1] = 3;
    size_t *w1Order = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, w1Order);
    shape_t *w1Shape = reserveMemory(sizeof(shape_t));
    setShape(w1Shape, w1Dims, 2, w1Order);
    tensor_t *w1Param = initTensor(w1Shape, quantizationInitFloat(), NULL);
    tensorFillFromFloatBuffer(w1Param, (float[]){0.f, 1.f, 2.f}, 3);
    tensor_t *w1Grad = gradInitFloat(w1Param, NULL);
    tensorFillFromFloatBuffer(w1Grad, (float[]){0.f, 0.f, 0.f}, 3);
    parameter_t *weights1 = parameterInit(w1Param, w1Grad);

    size_t *b1Dims = reserveMemory(1 * sizeof(size_t));
    b1Dims[0] = 1;
    size_t *b1Order = reserveMemory(1 * sizeof(size_t));
    setOrderOfDimsForNewTensor(1, b1Order);
    shape_t *b1Shape = reserveMemory(sizeof(shape_t));
    setShape(b1Shape, b1Dims, 1, b1Order);
    tensor_t *b1Param = initTensor(b1Shape, quantizationInitFloat(), NULL);
    tensorFillFromFloatBuffer(b1Param, (float[]){0.f}, 1);
    tensor_t *b1Grad = gradInitFloat(b1Param, NULL);
    tensorFillFromFloatBuffer(b1Grad, (float[]){0.f}, 1);
    parameter_t *bias1 = parameterInit(b1Param, b1Grad);

    layerQuant_t lq;
    layerQuantInitUniform(&lq, layerQ);
    layer_t *linear0 = buildBorrowedLinearLayer(weights0, bias0, layerQ);
    layer_t *relu0 = reluLayerInit(&lq);
    layer_t *linear1 = buildBorrowedLinearLayer(weights1, bias1, layerQ);

    layer_t *model[] = {linear0, relu0, linear1};
    size_t sizeModel = sizeof(model) / sizeof(model[0]);
    float lr = 0.1f;
    float momentumFactor = 0.9f;
    float weightDecay = 0.5f;

    quantization_t *momentumQ = quantizationInitFloat();
    optimizer_t *optim =
        sgdMCreateOptim(lr, momentumFactor, weightDecay, model, sizeModel, momentumQ,
                        (arithmetic_t){.type = ARITH_FLOAT32, .roundingMode = HALF_AWAY});
    sgd_t *sgd = optim->impl->sgd;

    linearConfig_t *linear0Conf = linear0->config->linear;
    linearConfig_t *linear1Conf = linear1->config->linear;

    /* CAPTURE before frees. */
    float capturedLr = sgd->learningRate;
    float capturedMomentum = sgd->momentumFactor;
    float capturedWeightDecay = sgd->weightDecay;
    size_t capturedSizeStates = optim->sizeStates;

    parameter_t *capturedL0Weights = linear0Conf->weights;
    parameter_t *capturedL0Bias = linear0Conf->bias;
    parameter_t *capturedL1Weights = linear1Conf->weights;
    parameter_t *capturedL1Bias = linear1Conf->bias;
    parameter_t *capturedOptimP0 = optim->parameter[0];
    parameter_t *capturedOptimP1 = optim->parameter[1];
    parameter_t *capturedOptimP2 = optim->parameter[2];
    parameter_t *capturedOptimP3 = optim->parameter[3];

    size_t l0WeightsN = calcNumberOfElementsByParameter(linear0Conf->weights);
    size_t l0BiasN = calcNumberOfElementsByParameter(linear0Conf->bias);
    size_t l1WeightsN = calcNumberOfElementsByParameter(linear1Conf->weights);
    size_t l1BiasN = calcNumberOfElementsByParameter(linear1Conf->bias);

    /* l0WeightsN=3, l0BiasN=1, l1WeightsN=3, l1BiasN=1 (per shape semantics above). */
    float capturedL0WGrad[3];
    float capturedL0WState[3];
    for (size_t i = 0; i < l0WeightsN; i++) {
        capturedL0WGrad[i] = ((float *)linear0Conf->weights->grad->data)[i];
        capturedL0WState[i] = ((float *)optim->states[0]->stateBuffers[0]->data)[i];
    }

    float capturedL0BGrad[1];
    float capturedL0BState[1];
    for (size_t i = 0; i < l0BiasN; i++) {
        capturedL0BGrad[i] = ((float *)linear0Conf->bias->grad->data)[i];
        capturedL0BState[i] = ((float *)optim->states[1]->stateBuffers[0]->data)[i];
    }

    float capturedL1WGrad[3];
    float capturedL1WState[3];
    for (size_t i = 0; i < l1WeightsN; i++) {
        capturedL1WGrad[i] = ((float *)linear1Conf->weights->grad->data)[i];
        capturedL1WState[i] = ((float *)optim->states[2]->stateBuffers[0]->data)[i];
    }

    float capturedL1BGrad[1];
    float capturedL1BState[1];
    for (size_t i = 0; i < l1BiasN; i++) {
        capturedL1BGrad[i] = ((float *)linear1Conf->bias->grad->data)[i];
        capturedL1BState[i] = ((float *)optim->states[3]->stateBuffers[0]->data)[i];
    }

    /* FREE in reverse-init order. freeOptim cascades to all parameters
     * (and their grads) registered with the optimizer (per SgdApi.c:85-93,
     * mirroring the post-#110 ownership contract). */
    freeOptim(optim);
    freeLinearLayerShellOnly(linear1);
    freeReluLayer(relu0);
    freeLinearLayerShellOnly(linear0);
    freeQuantization(momentumQ);
    freeQuantization(layerQ);

    /* ASSERT. */
    TEST_ASSERT_EQUAL_FLOAT(lr, capturedLr);
    TEST_ASSERT_EQUAL_FLOAT(momentumFactor, capturedMomentum);
    TEST_ASSERT_EQUAL_FLOAT(weightDecay, capturedWeightDecay);
    TEST_ASSERT_EQUAL_size_t(4, capturedSizeStates);

    TEST_ASSERT_EQUAL_PTR(capturedL0Weights, capturedOptimP0);
    TEST_ASSERT_EQUAL_PTR(capturedL0Bias, capturedOptimP1);
    TEST_ASSERT_EQUAL_PTR(capturedL1Weights, capturedOptimP2);
    TEST_ASSERT_EQUAL_PTR(capturedL1Bias, capturedOptimP3);

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(capturedL0WGrad, capturedL0WState, l0WeightsN);
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(capturedL0BGrad, capturedL0BState, l0BiasN);
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(capturedL1WGrad, capturedL1WState, l1WeightsN);
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(capturedL1BGrad, capturedL1BState, l1BiasN);
}

void testSGDStep() {
    quantization_t *layerQ = quantizationInitFloat();

    /* weights (heap shape, FLOAT32, dims {3, 1}, ndims=1 -> 3 elements). */
    size_t *wDims = reserveMemory(1 * sizeof(size_t));
    wDims[0] = 3;
    size_t *wOrder = reserveMemory(1 * sizeof(size_t));
    setOrderOfDimsForNewTensor(1, wOrder);
    shape_t *wShape = reserveMemory(sizeof(shape_t));
    setShape(wShape, wDims, 1, wOrder);
    tensor_t *weightParam = initTensor(wShape, quantizationInitFloat(), NULL);
    tensorFillFromFloatBuffer(weightParam, (float[]){1.f, 2.f, -3.f}, 3);
    tensor_t *weightGrad = gradInitFloat(weightParam, NULL);
    /* Bug 2 fix: populate the heap-allocated grad buffer rather than
     * overwriting weightGrad->data with a stack pointer (which would leak
     * the heap buffer and make freeTensor UB on the dangling stack ptr). */
    tensorFillFromFloatBuffer(weightGrad, (float[]){1.f, -1.f, 2.f}, 3);
    parameter_t *weights = parameterInit(weightParam, weightGrad);

    /* bias (heap shape, FLOAT32, dims {2, 1}, ndims=1 -> 2 elements). */
    size_t *bDims = reserveMemory(1 * sizeof(size_t));
    bDims[0] = 2;
    size_t *bOrder = reserveMemory(1 * sizeof(size_t));
    setOrderOfDimsForNewTensor(1, bOrder);
    shape_t *bShape = reserveMemory(sizeof(shape_t));
    setShape(bShape, bDims, 1, bOrder);
    tensor_t *biasParam = initTensor(bShape, quantizationInitFloat(), NULL);
    tensorFillFromFloatBuffer(biasParam, (float[]){-1.f, 3.f}, 2);
    tensor_t *biasGrad = gradInitFloat(biasParam, NULL);
    tensorFillFromFloatBuffer(biasGrad, (float[]){1.f, 3.f}, 2);
    parameter_t *bias = parameterInit(biasParam, biasGrad);

    layer_t *linear = buildBorrowedLinearLayer(weights, bias, layerQ);

    layer_t *model[] = {linear};
    size_t modelSize = 1;
    float lr = 0.1f;
    float momentumFactor = 0.9f;
    float weightDecay = 0.01f;

    quantization_t *momentumQ = quantizationInitFloat();
    optimizer_t *sgd =
        sgdMCreateOptim(lr, momentumFactor, weightDecay, model, modelSize, momentumQ,
                        (arithmetic_t){.type = ARITH_FLOAT32, .roundingMode = HALF_AWAY});

    optimizerFunctions_t sgdFns = optimizerFunctions[sgd->type];
    sgdFns.step(sgd);

    /* CAPTURE first-step values before second step. */
    float capturedWAfterStep1[3];
    for (size_t i = 0; i < 3; i++) {
        capturedWAfterStep1[i] = ((float *)linear->config->linear->weights->param->data)[i];
    }
    float capturedBAfterStep1[2];
    for (size_t i = 0; i < 2; i++) {
        capturedBAfterStep1[i] = ((float *)linear->config->linear->bias->param->data)[i];
    }

    sgdFns.step(sgd);

    /* CAPTURE second-step values before frees. */
    float capturedWAfterStep2[3];
    for (size_t i = 0; i < 3; i++) {
        capturedWAfterStep2[i] = ((float *)linear->config->linear->weights->param->data)[i];
    }
    float capturedBAfterStep2[2];
    for (size_t i = 0; i < 2; i++) {
        capturedBAfterStep2[i] = ((float *)linear->config->linear->bias->param->data)[i];
    }

    /* FREE in reverse-init order. freeOptim already frees the registered
     * weights/bias — freeLinearLayer would double-free them. */
    freeOptim(sgd);
    freeLinearLayerShellOnly(linear);
    freeQuantization(momentumQ);
    freeQuantization(layerQ);

    /* ASSERT. */
    float wPExpected[] = {0.899f, 2.098f, -3.197f};
    float bPExpected[] = {-1.099f, 2.697f};
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(wPExpected, capturedWAfterStep1, 3);
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(bPExpected, capturedBAfterStep1, 2);

    float wPExpected2[] = {0.707201f, 2.284102f, -3.571103f};
    float bPExpected2[] = {-1.287001f, 2.121603f};
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(wPExpected2, capturedWAfterStep2, 3);
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(bPExpected2, capturedBAfterStep2, 2);
}

void testSGDZeroGrad() {
    quantization_t *layerQ = quantizationInitFloat();

    /* weights (heap shape, FLOAT32, dims {3, 1}, ndims=2 -> 3 elements). */
    size_t *wDims = reserveMemory(2 * sizeof(size_t));
    wDims[0] = 3;
    wDims[1] = 1;
    size_t *wOrder = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, wOrder);
    shape_t *wShape = reserveMemory(sizeof(shape_t));
    setShape(wShape, wDims, 2, wOrder);
    tensor_t *weightParam = initTensor(wShape, quantizationInitFloat(), NULL);
    tensorFillFromFloatBuffer(weightParam, (float[]){1.f, 2.f, -3.f}, 3);
    tensor_t *weightGrad = gradInitFloat(weightParam, NULL);
    /* Bug 2 fix: same as testSGDStep. */
    tensorFillFromFloatBuffer(weightGrad, (float[]){1.f, -1.f, 2.f}, 3);
    parameter_t *weights = parameterInit(weightParam, weightGrad);

    /* bias (heap shape, FLOAT32, dims {2, 1}, ndims=2 -> 2 elements). */
    size_t *bDims = reserveMemory(2 * sizeof(size_t));
    bDims[0] = 2;
    bDims[1] = 1;
    size_t *bOrder = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, bOrder);
    shape_t *bShape = reserveMemory(sizeof(shape_t));
    setShape(bShape, bDims, 2, bOrder);
    tensor_t *biasParam = initTensor(bShape, quantizationInitFloat(), NULL);
    tensorFillFromFloatBuffer(biasParam, (float[]){-1.f, 3.f}, 2);
    /* Bug 3 fix: gradInitFloat must take biasParam (not weightParam) so the
     * grad's shape matches bias (2 elements), not weight (3 elements). */
    tensor_t *biasGrad = gradInitFloat(biasParam, NULL);
    tensorFillFromFloatBuffer(biasGrad, (float[]){1.f, 3.f}, 2);
    parameter_t *bias = parameterInit(biasParam, biasGrad);

    layer_t *linear = buildBorrowedLinearLayer(weights, bias, layerQ);

    layer_t *model[] = {linear};
    size_t modelSize = 1;
    float lr = 0.1f;
    float momentumFactor = 0.9f;
    float weightDecay = 0.01f;

    quantization_t *momentumQ = quantizationInitFloat();
    optimizer_t *sgd =
        sgdMCreateOptim(lr, momentumFactor, weightDecay, model, modelSize, momentumQ,
                        (arithmetic_t){.type = ARITH_FLOAT32, .roundingMode = HALF_AWAY});

    optimizerZeroGrad(sgd);

    /* CAPTURE before frees. */
    float capturedWGrad[3];
    for (size_t i = 0; i < 3; i++) {
        capturedWGrad[i] = ((float *)weights->grad->data)[i];
    }
    float capturedBGrad[2];
    for (size_t i = 0; i < 2; i++) {
        capturedBGrad[i] = ((float *)bias->grad->data)[i];
    }

    /* FREE in reverse-init order. freeOptim already frees the registered
     * weights/bias — freeLinearLayer would double-free them. */
    freeOptim(sgd);
    freeLinearLayerShellOnly(linear);
    freeQuantization(momentumQ);
    freeQuantization(layerQ);

    /* ASSERT. */
    float wGradExpected[] = {0.f, 0.f, 0.f};
    float bGradExpected[] = {0.f, 0.f};
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(wGradExpected, capturedWGrad, 3);
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(bGradExpected, capturedBGrad, 2);
}

void testSgdZeroGradOnSymInt32GradZeroesMantissasAndResetsScale(void) {
    /* PR1c: default grads are FLOAT32; SYM via knob. weightGradStorage/
     * biasGradStorage opt this layer's grads back into SYM_INT32 so this test
     * keeps exercising the SYM zero+scale-reset path it's named for. */
    quantization_t *fwd = quantizationInitFloat();
    quantization_t *bwd = quantizationInitSymInt32(HALF_AWAY);
    layerQuant_t lq = {.forwardMath = arithmeticFromQuantization(fwd),
                       .weightGradMath = arithmeticFromQuantization(bwd),
                       .biasGradMath = arithmeticFromQuantization(bwd),
                       .propLossMath = arithmeticFromQuantization(bwd),
                       .outputQ = fwd,
                       .propLossQ = bwd,
                       .weightStorage = fwd,
                       .biasStorage = fwd,
                       .weightGradStorage = bwd,
                       .biasGradStorage = bwd};
    layer_t *layer =
        linearLayerInit(&(linearInit_t){.inFeatures = 3, .outFeatures = 2, .bias = BIAS_TRUE}, &lq);

    tensor_t *wGrad = layer->config->linear->weights->grad;
    TEST_ASSERT_EQUAL_INT(SYM_INT32, wGrad->quantization->type); /* guard */

    /* Pre-load non-zero mantissas and a non-1.0 scale to prove the reset. */
    size_t nW = calcNumberOfElementsByTensor(wGrad);
    for (size_t i = 0; i < nW; i++) {
        ((int32_t *)wGrad->data)[i] = 123 + (int32_t)i;
    }
    ((symInt32QConfig_t *)wGrad->quantization->qConfig)->scale = 0.07f;

    layer_t *model[] = {layer};
    quantization_t *momentumQ = quantizationInitFloat();
    optimizer_t *optim =
        sgdMCreateOptim(0.1f, 0.9f, 0.0f, model, 1, momentumQ,
                        (arithmetic_t){.type = ARITH_FLOAT32, .roundingMode = HALF_AWAY});

    optimizerZeroGrad(optim);

    /* CAPTURE post-zero state. */
    int allZero = 1;
    for (size_t i = 0; i < nW; i++) {
        if (((int32_t *)wGrad->data)[i] != 0) {
            allZero = 0;
        }
    }
    float scaleAfterZero = ((symInt32QConfig_t *)wGrad->quantization->qConfig)->scale;

    /* Prove subsequent accumulation works: load a fresh int mantissa. */
    ((int32_t *)wGrad->data)[0] = 5;
    int accumWorks = (((int32_t *)wGrad->data)[0] == 5);

    freeOptim(optim); /* frees the layer's parameters */
    freeReservedMemory(layer->config->linear);
    freeReservedMemory(layer->config);
    freeReservedMemory(layer);
    freeQuantization(momentumQ);
    freeQuantization(bwd);
    freeQuantization(fwd);

    TEST_ASSERT_TRUE_MESSAGE(allZero, "optimizerZeroGrad must zero SYM_INT32 grad mantissas");
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 1.0f, scaleAfterZero);
    TEST_ASSERT_TRUE(accumWorks);
}

/* TDD RED for #277 Task 2: momentum state must carry its OWN quantization
 * config, decoupled from the parameter's dtype. Before this change,
 * sgdMCreateOptim built each state via getTensorLike(param->param), so a
 * SYM_INT32 weight/bias produced a SYM_INT32 momentum accumulator -- this
 * pins a FLOAT32 momentumQuant against a SYM_INT32 param and asserts the
 * state buffer lands FLOAT32 regardless. */
void testSgdMCreateOptimMomentumStateIsFloatIndependentOfSymInt32ParamDtype(void) {
    quantization_t *symQ = quantizationInitSymInt32(HALF_AWAY);

    /* weights: SYM_INT32 param + SYM_INT32 grad, shape [2, 3]. */
    size_t *wDims = reserveMemory(2 * sizeof(size_t));
    wDims[0] = 2;
    wDims[1] = 3;
    size_t *wOrder = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, wOrder);
    shape_t *wShape = reserveMemory(sizeof(shape_t));
    setShape(wShape, wDims, 2, wOrder);
    tensor_t *wParam = initTensor(wShape, quantizationInitSymInt32(HALF_AWAY), NULL);
    tensorFillFromFloatBuffer(wParam, (float[]){0.f, 0.f, 0.f, 0.f, 0.f, 0.f}, 6);
    tensor_t *wGrad = gradInitSymInt32(wParam, HALF_AWAY, NULL);
    parameter_t *weights = parameterInit(wParam, wGrad);

    /* bias: SYM_INT32 param + SYM_INT32 grad, shape [2]. */
    size_t *bDims = reserveMemory(1 * sizeof(size_t));
    bDims[0] = 2;
    size_t *bOrder = reserveMemory(1 * sizeof(size_t));
    setOrderOfDimsForNewTensor(1, bOrder);
    shape_t *bShape = reserveMemory(sizeof(shape_t));
    setShape(bShape, bDims, 1, bOrder);
    tensor_t *bParam = initTensor(bShape, quantizationInitSymInt32(HALF_AWAY), NULL);
    tensorFillFromFloatBuffer(bParam, (float[]){0.f, 0.f}, 2);
    tensor_t *bGrad = gradInitSymInt32(bParam, HALF_AWAY, NULL);
    parameter_t *bias = parameterInit(bParam, bGrad);

    layer_t *layer = buildBorrowedLinearLayer(weights, bias, symQ);
    layer_t *model[1] = {layer};

    quantization_t *momentumQ = quantizationInitFloat();
    optimizer_t *optim =
        sgdMCreateOptim(0.1f, 0.9f, 0.0f, model, 1, momentumQ,
                        (arithmetic_t){.type = ARITH_FLOAT32, .roundingMode = HALF_AWAY});

    /* CAPTURE before frees. */
    qtype_t capturedWeightParamType =
        weights->param->quantization->type; /* guard: stays SYM_INT32 */
    qtype_t capturedBiasParamType = bias->param->quantization->type;
    qtype_t capturedWeightStateType = optim->states[0]->stateBuffers[0]->quantization->type;
    qtype_t capturedBiasStateType = optim->states[1]->stateBuffers[0]->quantization->type;

    freeOptim(optim);
    freeLinearLayerShellOnly(layer);
    freeQuantization(momentumQ);
    freeQuantization(symQ);

    TEST_ASSERT_EQUAL_INT(SYM_INT32, capturedWeightParamType);
    TEST_ASSERT_EQUAL_INT(SYM_INT32, capturedBiasParamType);
    TEST_ASSERT_EQUAL_INT(FLOAT32, capturedWeightStateType);
    TEST_ASSERT_EQUAL_INT(FLOAT32, capturedBiasStateType);
}

void testSgdMCreateOptimRegistersLayerNormGammaAndBeta(void) {
    /* Build gamma parameter: shape [3], FLOAT32. */
    size_t *gDims = reserveMemory(1 * sizeof(size_t));
    gDims[0] = 3;
    size_t *gOrder = reserveMemory(1 * sizeof(size_t));
    setOrderOfDimsForNewTensor(1, gOrder);
    shape_t *gShape = reserveMemory(sizeof(shape_t));
    setShape(gShape, gDims, 1, gOrder);
    tensor_t *gParam = initTensor(gShape, quantizationInitFloat(), NULL);
    tensorFillFromFloatBuffer(gParam, (float[]){1.f, 1.f, 1.f}, 3);
    tensor_t *gGrad = gradInitFloat(gParam, NULL);
    tensorFillFromFloatBuffer(gGrad, (float[]){0.f, 0.f, 0.f}, 3);
    parameter_t *gamma = parameterInit(gParam, gGrad);

    /* Build beta parameter: shape [3], FLOAT32. */
    size_t *bDims = reserveMemory(1 * sizeof(size_t));
    bDims[0] = 3;
    size_t *bOrder = reserveMemory(1 * sizeof(size_t));
    setOrderOfDimsForNewTensor(1, bOrder);
    shape_t *bShape = reserveMemory(sizeof(shape_t));
    setShape(bShape, bDims, 1, bOrder);
    tensor_t *bParam = initTensor(bShape, quantizationInitFloat(), NULL);
    tensorFillFromFloatBuffer(bParam, (float[]){0.f, 0.f, 0.f}, 3);
    tensor_t *bGrad = gradInitFloat(bParam, NULL);
    tensorFillFromFloatBuffer(bGrad, (float[]){0.f, 0.f, 0.f}, 3);
    parameter_t *beta = parameterInit(bParam, bGrad);

    /* Build LayerNorm config + layer. */
    quantization_t *fq = quantizationInitFloat();
    quantization_t *bq = quantizationInitFloat();

    size_t *normShape = reserveMemory(sizeof(size_t));
    normShape[0] = 3;

    layerNormConfig_t *lnCfg = reserveMemory(sizeof(layerNormConfig_t));
    initLayerNormConfig(lnCfg, gamma, beta, normShape, 1, 1e-5f, fq, bq);

    layerConfig_t *lcfg = reserveMemory(sizeof(layerConfig_t));
    lcfg->layerNorm = lnCfg;

    layer_t *lnLayer = reserveMemory(sizeof(layer_t));
    lnLayer->type = LAYERNORM;
    lnLayer->config = lcfg;

    layer_t *model[] = {lnLayer};
    size_t sizeModel = 1;
    float lr = 0.1f;
    float momentumFactor = 0.9f;
    float weightDecay = 0.0f;

    quantization_t *momentumQ = quantizationInitFloat();
    optimizer_t *optim =
        sgdMCreateOptim(lr, momentumFactor, weightDecay, model, sizeModel, momentumQ,
                        (arithmetic_t){.type = ARITH_FLOAT32, .roundingMode = HALF_AWAY});

    /* CAPTURE before frees. */
    size_t capturedSizeStates = optim->sizeStates;
    parameter_t *capturedP0 = optim->parameter[0];
    parameter_t *capturedP1 = optim->parameter[1];

    size_t gammaN = calcNumberOfElementsByParameter(lnCfg->gamma);
    size_t betaN = calcNumberOfElementsByParameter(lnCfg->beta);

    size_t capturedGammaStateN = calcNumberOfElementsByTensor(optim->states[0]->stateBuffers[0]);
    size_t capturedBetaStateN = calcNumberOfElementsByTensor(optim->states[1]->stateBuffers[0]);

    /* FREE: freeOptim frees the two registered parameter_t* (gamma, beta)
     * and their state buffers. Then free the layer wrapper structs. */
    freeOptim(optim);
    freeReservedMemory(lnLayer);
    freeReservedMemory(lcfg);
    freeReservedMemory(normShape);
    freeQuantization(momentumQ);
    freeQuantization(bq);
    freeQuantization(fq);
    freeReservedMemory(lnCfg);

    /* ASSERT. */
    TEST_ASSERT_EQUAL_size_t(2, capturedSizeStates);
    TEST_ASSERT_EQUAL_PTR(gamma, capturedP0);
    TEST_ASSERT_EQUAL_PTR(beta, capturedP1);
    TEST_ASSERT_EQUAL_size_t(gammaN, capturedGammaStateN);
    TEST_ASSERT_EQUAL_size_t(betaN, capturedBetaStateN);
}

void testLayerNormContributesTwoOptimizerStates(void) {
    /* #380: calcNumberOfStatesByLayer now checks layerIsFrozen(layer) first,
     * which for LAYERNORM dereferences layer->config->layerNorm->frozen -- a
     * bare .config = NULL fixture (the pre-#380 shortcut) would crash. A
     * stack-local layerNormConfig_t with frozen left at its zero-init default
     * (false, the trainable case) keeps this a minimal, config-only fixture. */
    layerNormConfig_t lnCfg = {0};
    layerConfig_t lcfg = {.layerNorm = &lnCfg};
    layer_t layer = {.type = LAYERNORM, .config = &lcfg};
    layer_t *model[] = {&layer};
    size_t sizeModel = 1;

    size_t nStates = calcTotalNumberOfStates(model, sizeModel);

    TEST_ASSERT_EQUAL_size_t(2, nStates);
}

void testSgdStepSymInt32DoesNotRoundTripGrad(void) {
    /* Manual SYM param+grad (low-level path — heap shape per file convention). */
    size_t *pDims = reserveMemory(1 * sizeof(size_t));
    pDims[0] = 3;
    size_t *pOrder = reserveMemory(1 * sizeof(size_t));
    setOrderOfDimsForNewTensor(1, pOrder);
    shape_t *pShape = reserveMemory(sizeof(shape_t));
    setShape(pShape, pDims, 1, pOrder);
    tensor_t *p = initTensor(pShape, quantizationInitSymInt32(HALF_AWAY), NULL);
    tensorFillFromFloatBuffer(p, (float[]){0.1f, -0.2f, 0.3f}, 3);
    tensor_t *g = gradInitSymInt32(p, HALF_AWAY, NULL);
    parameter_t *param = parameterInit(p, g);

    /* Pre-load known grad mantissas + a non-1.0 scale. */
    ((int32_t *)g->data)[0] = 10;
    ((int32_t *)g->data)[1] = -20;
    ((int32_t *)g->data)[2] = 30;
    ((symInt32QConfig_t *)g->quantization->qConfig)->scale = 0.05f;

    /* Minimal stack optimizer around one parameter; hand-assembled momentum==0
     * optimizer; SYM_INT32 param. sgdStepM's unified momentum==0 fast path reads only
     * impl/sizeStates/parameter (+ sgd learningRate/weightDecay). */
    sgd_t sgd;
    sgdInit(&sgd, 0.1f, 0.0f, 0.0f,
            (arithmetic_t){.type = ARITH_FLOAT32, .roundingMode = HALF_AWAY});
    parameter_t *params[1] = {param};
    optimImpl_t impl;
    impl.sgd = &sgd;
    optimizer_t optim;
    optim.parameter = params;
    optim.states = NULL;
    optim.sizeStates = 1;
    optim.impl = &impl;
    optim.writeBackRounding =
        HALF_AWAY; /* hand-built: field is otherwise uninitialized stack (#279) */

    sgdStepM(&optim);

    /* CAPTURE -> free -> assert: grad mantissas + scale must be untouched. */
    int32_t m0 = ((int32_t *)g->data)[0];
    int32_t m1 = ((int32_t *)g->data)[1];
    int32_t m2 = ((int32_t *)g->data)[2];
    float gScale = ((symInt32QConfig_t *)g->quantization->qConfig)->scale;

    freeParameter(param);

    TEST_ASSERT_EQUAL_INT(10, m0);
    TEST_ASSERT_EQUAL_INT(-20, m1);
    TEST_ASSERT_EQUAL_INT(30, m2);
    TEST_ASSERT_FLOAT_WITHIN(1e-9f, 0.05f, gScale);
}

void testSgdStepMMomentumZeroIgnoresStatesAndUpdatesParam(void) {
    /* #308 contract: momentumFactor == 0 makes the momentum state
     * semantically nonexistent -- sgdStepM must not touch optim->states
     * (explicitly NULL here; the factory allocates none in this mode) and
     * must apply the plain update param -= lr*(grad + wd*param) in a single
     * op. RED before the fast path: sgdStepM unconditionally derefs
     * optim->states[i] -> NULL-deref crash. */
    size_t *pDims = reserveMemory(1 * sizeof(size_t));
    pDims[0] = 2;
    size_t *pOrder = reserveMemory(1 * sizeof(size_t));
    setOrderOfDimsForNewTensor(1, pOrder);
    shape_t *pShape = reserveMemory(sizeof(shape_t));
    setShape(pShape, pDims, 1, pOrder);
    tensor_t *p = initTensor(pShape, quantizationInitFloat(), NULL);
    tensorFillFromFloatBuffer(p, (float[]){1.f, -2.f}, 2);
    tensor_t *g = gradInitFloat(p, NULL);
    tensorFillFromFloatBuffer(g, (float[]){0.5f, 0.25f}, 2);
    parameter_t *param = parameterInit(p, g);

    sgd_t sgd;
    sgdInit(&sgd, 0.1f, 0.0f, 0.5f,
            (arithmetic_t){.type = ARITH_FLOAT32,
                           .roundingMode = HALF_AWAY}); /* lr=0.1, momentum=0, wd=0.5 */
    parameter_t *params[1] = {param};
    optimImpl_t impl;
    impl.sgd = &sgd;
    optimizer_t optim;
    optim.parameter = params;
    optim.states = NULL; /* the #308 contract under test */
    optim.sizeStates = 1;
    optim.impl = &impl;
    optim.writeBackRounding =
        HALF_AWAY; /* hand-built: field is otherwise uninitialized stack (#279) */

    sgdStepM(&optim);

    /* CAPTURE -> free -> assert (file convention). */
    float p0 = ((float *)p->data)[0];
    float p1 = ((float *)p->data)[1];
    freeParameter(param);

    /* param -= lr*(grad + wd*param):
     * p0: 1.0  - 0.1*(0.5  + 0.5*1.0)  = 1.0 - 0.1*1.0    = 0.9
     * p1: -2.0 - 0.1*(0.25 + 0.5*-2.0) = -2.0 - 0.1*-0.75 = -1.925 */
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 0.9f, p0);
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, -1.925f, p1);
}

void testSgdZeroGradAsymSubByteZeroesAllPackedBytes(void) {
    /* ASYM qBits=3, N=10 grad -> packed ceil(30/8) = 4 bytes. Pre-fix sizing
     * (size_t division inside ceil()) truncated to 3, leaving stale grad bits in
     * the last packed byte. Mutation guard: reverting to the old formula leaves
     * byte 3 == 0xFF -> RED. */
    size_t *pDims = reserveMemory(2 * sizeof(size_t));
    pDims[0] = 2;
    pDims[1] = 5;
    size_t *pOrder = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, pOrder);
    shape_t *pShape = reserveMemory(sizeof(shape_t));
    setShape(pShape, pDims, 2, pOrder);
    tensor_t *p = initTensor(pShape, quantizationInitFloat(), NULL);
    tensorFillFromFloatBuffer(p, (float[]){0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f}, 10);
    tensor_t *g = gradInitAsym(p, 3, HALF_AWAY, NULL);
    parameter_t *param = parameterInit(p, g);
    memset(g->data, 0xFF, 4); /* poison all packed grad bytes */
    /* Poison the config too (PR3, spec §5.3): byte-zero alone is not
     * VALUE-zero for ASYM (code 0 decodes to zeroPoint*scale) — the config
     * reset asserted below is the fix for the PR2 watch-list item. */
    asymQConfig_t *gAsymQ = g->quantization->qConfig;
    gAsymQ->scale = 0.42f;
    gAsymQ->zeroPoint = 5;

    /* Hand-built optimizer: sgdMCreateOptim rejects sub-byte grad storage until
     * PR3, but optimizerZeroGrad reads only sizeStates/parameter — this characterizes
     * the primitive PR3 will rely on. */
    sgd_t sgd;
    sgdInit(&sgd, 0.1f, 0.0f, 0.0f,
            (arithmetic_t){.type = ARITH_FLOAT32, .roundingMode = HALF_AWAY});
    parameter_t *params[1] = {param};
    optimImpl_t impl;
    impl.sgd = &sgd;
    optimizer_t optim;
    optim.parameter = params;
    optim.sizeStates = 1;
    optim.impl = &impl;
    optim.writeBackRounding =
        HALF_AWAY; /* hand-built: field is otherwise uninitialized stack (#279) */

    optimizerZeroGrad(&optim);

    /* CAPTURE -> free -> assert (file convention). */
    uint8_t b0 = g->data[0];
    uint8_t b1 = g->data[1];
    uint8_t b2 = g->data[2];
    uint8_t b3 = g->data[3];
    float scaleAfter = gAsymQ->scale;
    int32_t zeroPointAfter = gAsymQ->zeroPoint;
    freeParameter(param);

    TEST_ASSERT_EQUAL_UINT8(0, b0);
    TEST_ASSERT_EQUAL_UINT8(0, b1);
    TEST_ASSERT_EQUAL_UINT8(0, b2);
    TEST_ASSERT_EQUAL_UINT8(0, b3);
    /* ADDITIVE (PR3): value-zero, not just byte-zero — code 0 must decode to
     * exactly 0.0f, which requires zeroPoint==0 (scale is irrelevant to that
     * particular identity, but reset for hygiene/consistency with SYM/SYM_INT32). */
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 1.0f, scaleAfter);
    TEST_ASSERT_EQUAL_INT32(0, zeroPointAfter);
}

void testSgdZeroGradSymSubByteZeroesAllPackedBytesAndResetsScale(void) {
    /* SYM sibling of the ASYM test above (PR3, spec §5.3). qBits=3, N=10 grad
     * -> packed ceil(30/8)=4 bytes. SYM's all-zero-mantissa state is already
     * the first-store trigger for accumulateFloatIntoSymTensorFixedGrid, so
     * correctness does not depend on the scale reset — this asserts the
     * hygiene/consistency reset the spec calls for anyway. */
    size_t *pDims = reserveMemory(2 * sizeof(size_t));
    pDims[0] = 2;
    pDims[1] = 5;
    size_t *pOrder = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, pOrder);
    shape_t *pShape = reserveMemory(sizeof(shape_t));
    setShape(pShape, pDims, 2, pOrder);
    tensor_t *p = initTensor(pShape, quantizationInitFloat(), NULL);
    tensorFillFromFloatBuffer(p, (float[]){0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f}, 10);
    tensor_t *g = gradInitSym(p, 3, HALF_AWAY, NULL);
    parameter_t *param = parameterInit(p, g);
    memset(g->data, 0xFF, 4); /* poison all packed grad bytes */
    symQConfig_t *gSymQ = g->quantization->qConfig;
    gSymQ->scale = 0.42f; /* poison the scale too */

    sgd_t sgd;
    sgdInit(&sgd, 0.1f, 0.0f, 0.0f,
            (arithmetic_t){.type = ARITH_FLOAT32, .roundingMode = HALF_AWAY});
    parameter_t *params[1] = {param};
    optimImpl_t impl;
    impl.sgd = &sgd;
    optimizer_t optim;
    optim.parameter = params;
    optim.sizeStates = 1;
    optim.impl = &impl;
    optim.writeBackRounding =
        HALF_AWAY; /* hand-built: field is otherwise uninitialized stack (#279) */

    optimizerZeroGrad(&optim);

    /* CAPTURE -> free -> assert (file convention). */
    uint8_t b0 = g->data[0];
    uint8_t b1 = g->data[1];
    uint8_t b2 = g->data[2];
    uint8_t b3 = g->data[3];
    float scaleAfter = gSymQ->scale;
    freeParameter(param);

    TEST_ASSERT_EQUAL_UINT8(0, b0);
    TEST_ASSERT_EQUAL_UINT8(0, b1);
    TEST_ASSERT_EQUAL_UINT8(0, b2);
    TEST_ASSERT_EQUAL_UINT8(0, b3);
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 1.0f, scaleAfter);
}

void testSgdMCreateOptimAdmitsPackedSymGradStorage(void) {
    /* CONTRACT FLIP (second of exactly two sanctioned in this PR — spec §5.1;
     * pr3-autonomous-decisions.md item 5. The other is Task 3's ExecuteOp
     * `testAccIntoSubByteTargetAborts` -> `testAccFixedIntoPackedSymDerivesThenCarriesGrid`).
     * This test used to be `testSgdMCreateOptimRejectsSubByteGradStorage`,
     * pinning "packed SYM grad storage aborts optimizer construction". PR3
     * deliberately admits SYM (and ASYM) grads at the optimizer gate, so
     * construction of the exact same fixture must now SUCCEED. The BOOL-grad
     * rejection sibling below takes over this test's former negative-coverage
     * role (BOOL remains unsupported). */
    size_t *wDims = reserveMemory(2 * sizeof(size_t));
    wDims[0] = 2;
    wDims[1] = 3;
    size_t *wOrder = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, wOrder);
    shape_t *wShape = reserveMemory(sizeof(shape_t));
    setShape(wShape, wDims, 2, wOrder);
    tensor_t *wParam = initTensor(wShape, quantizationInitFloat(), NULL);
    tensorFillFromFloatBuffer(wParam, (float[]){0.f, 0.f, 0.f, 0.f, 0.f, 0.f}, 6);
    tensor_t *wGrad = gradInitSym(wParam, 8, HALF_AWAY, NULL);
    parameter_t *weights = parameterInit(wParam, wGrad);

    size_t *bDims = reserveMemory(1 * sizeof(size_t));
    bDims[0] = 2;
    size_t *bOrder = reserveMemory(1 * sizeof(size_t));
    setOrderOfDimsForNewTensor(1, bOrder);
    shape_t *bShape = reserveMemory(sizeof(shape_t));
    setShape(bShape, bDims, 1, bOrder);
    tensor_t *bParam = initTensor(bShape, quantizationInitFloat(), NULL);
    tensorFillFromFloatBuffer(bParam, (float[]){0.f, 0.f}, 2);
    tensor_t *bGrad = gradInitFloat(bParam, NULL);
    parameter_t *bias = parameterInit(bParam, bGrad);

    quantization_t *fQ = quantizationInitFloat();
    layer_t *layer = buildBorrowedLinearLayer(weights, bias, fQ);
    layer_t *model[1] = {layer};

    quantization_t *momentumQ = quantizationInitFloat();
    optimizer_t *optim =
        sgdMCreateOptim(0.1f, 0.0f, 0.0f, model, 1, momentumQ,
                        (arithmetic_t){.type = ARITH_FLOAT32, .roundingMode = HALF_AWAY});

    /* CAPTURE before frees. */
    qtype_t capturedWGradType = weights->grad->quantization->type;
    size_t capturedSizeStates = optim->sizeStates;

    /* freeOptim cascades to the registered parameters (weights, bias). */
    freeOptim(optim);
    freeLinearLayerShellOnly(layer);
    freeQuantization(momentumQ);
    freeQuantization(fQ);

    TEST_ASSERT_EQUAL_INT(SYM, capturedWGradType);
    TEST_ASSERT_EQUAL_size_t(2, capturedSizeStates);
}

void testSgdMCreateOptimRejectsBoolGradStorage(void) {
    /* New negative fixture (replaces the flipped test's old rejection role):
     * FLOAT32 weight param whose grad is allocated BOOL — never a legitimate
     * grad dtype (no meaningful gradient value fits one bit). Optimizer
     * construction must abort. Hand-built per the old fixture's pattern
     * (initTensor + explicit quantization, no factory). */
    size_t *wDims = reserveMemory(2 * sizeof(size_t));
    wDims[0] = 2;
    wDims[1] = 3;
    size_t *wOrder = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, wOrder);
    shape_t *wShape = reserveMemory(sizeof(shape_t));
    setShape(wShape, wDims, 2, wOrder);
    tensor_t *wParam = initTensor(wShape, quantizationInitFloat(), NULL);
    tensorFillFromFloatBuffer(wParam, (float[]){0.f, 0.f, 0.f, 0.f, 0.f, 0.f}, 6);
    tensor_t *wGrad = initTensor(getShapeLike(wParam->shape), quantizationInitBool(), NULL);
    parameter_t *weights = parameterInit(wParam, wGrad);

    size_t *bDims = reserveMemory(1 * sizeof(size_t));
    bDims[0] = 2;
    size_t *bOrder = reserveMemory(1 * sizeof(size_t));
    setOrderOfDimsForNewTensor(1, bOrder);
    shape_t *bShape = reserveMemory(sizeof(shape_t));
    setShape(bShape, bDims, 1, bOrder);
    tensor_t *bParam = initTensor(bShape, quantizationInitFloat(), NULL);
    tensorFillFromFloatBuffer(bParam, (float[]){0.f, 0.f}, 2);
    tensor_t *bGrad = gradInitFloat(bParam, NULL);
    parameter_t *bias = parameterInit(bParam, bGrad);

    quantization_t *fQ = quantizationInitFloat();
    layer_t *layer = buildBorrowedLinearLayer(weights, bias, fQ);
    layer_t *model[1] = {layer};

    quantization_t *momentumQ = quantizationInitFloat();
    ASSERT_EXITS_WITH_FAILURE(
        sgdMCreateOptim(0.1f, 0.0f, 0.0f, model, 1, momentumQ,
                        (arithmetic_t){.type = ARITH_FLOAT32, .roundingMode = HALF_AWAY}));

    /* Teardown (parent continues after the fork-based assert; file convention). */
    freeLinearLayerShellOnly(layer);
    freeParameter(weights);
    freeParameter(bias);
    freeQuantization(momentumQ);
    freeQuantization(fQ);
}

void testSgdMCreateOptimRejectsNullGradSlot(void) {
    /* PR #366 hardening: every factory-collected trainable parameter must
     * carry an allocated grad -- no freeze mechanism exists, and step/zeroGrad
     * dereference grad unconditionally, so a NULL grad in a slot is a
     * mis-built model that would otherwise crash mid-training. Fail fast at
     * create instead of silently skipping the slot (old behavior). */
    size_t *wDims = reserveMemory(2 * sizeof(size_t));
    wDims[0] = 2;
    wDims[1] = 3;
    size_t *wOrder = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, wOrder);
    shape_t *wShape = reserveMemory(sizeof(shape_t));
    setShape(wShape, wDims, 2, wOrder);
    tensor_t *wParam = initTensor(wShape, quantizationInitFloat(), NULL);
    tensorFillFromFloatBuffer(wParam, (float[]){0.f, 0.f, 0.f, 0.f, 0.f, 0.f}, 6);
    parameter_t weights = {.param = wParam, .grad = NULL};

    quantization_t *fQ = quantizationInitFloat();
    layer_t *layer = buildBorrowedLinearLayer(&weights, NULL, fQ);
    layer_t *model[1] = {layer};

    quantization_t *momentumQ = quantizationInitFloat();
    ASSERT_EXITS_WITH_FAILURE(
        sgdMCreateOptim(0.1f, 0.0f, 0.0f, model, 1, momentumQ,
                        (arithmetic_t){.type = ARITH_FLOAT32, .roundingMode = HALF_AWAY}));

    /* Teardown (parent continues after the fork-based assert; weights is
     * stack-local, only the heap tensor + shells need freeing). */
    freeLinearLayerShellOnly(layer);
    freeTensor(wParam);
    freeQuantization(momentumQ);
    freeQuantization(fQ);
}

void testSgdMCreateOptimRejectsNonFloat32UpdateMath(void) {
    /* #310: ARITH_SYM_INT32 update arithmetic is not implemented -- the
     * factory must fail fast at creation, not corrupt at step time. Also
     * pins that the factory actually forwards updateMath into sgdInit
     * (a factory that dropped the arg would NOT exit -> RED). */
    quantization_t *layerQ = quantizationInitFloat();

    size_t *wDims = reserveMemory(2 * sizeof(size_t));
    wDims[0] = 1;
    wDims[1] = 2;
    size_t *wOrder = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, wOrder);
    shape_t *wShape = reserveMemory(sizeof(shape_t));
    setShape(wShape, wDims, 2, wOrder);
    tensor_t *wParam = initTensor(wShape, quantizationInitFloat(), NULL);
    tensorFillFromFloatBuffer(wParam, (float[]){1.f, 2.f}, 2);
    tensor_t *wGrad = gradInitFloat(wParam, NULL);
    parameter_t *weights = parameterInit(wParam, wGrad);
    layer_t *linear = buildBorrowedLinearLayer(weights, NULL, layerQ);
    layer_t *model[] = {linear};
    quantization_t *momentumQ = quantizationInitFloat();

    ASSERT_EXITS_WITH_FAILURE(
        sgdMCreateOptim(0.1f, 0.0f, 0.0f, model, 1, momentumQ,
                        (arithmetic_t){.type = ARITH_SYM_INT32, .roundingMode = HALF_AWAY}));

    freeParameter(weights);
    freeLinearLayerShellOnly(linear);
    freeQuantization(momentumQ);
    freeQuantization(layerQ);
}

void testSgdStepMRejectsNonFloat32UpdateMath(void) {
    /* #310 airtightness: the update kernels raw-cast operand data to
     * float*, so a non-FLOAT32 prologue (e.g. int32 scratch codes) would be
     * silently misread as float bit patterns. A hand-assembled optimizer
     * bypasses the factory guard -- the step itself must also fail fast. */
    size_t *pDims = reserveMemory(1 * sizeof(size_t));
    pDims[0] = 1;
    size_t *pOrder = reserveMemory(1 * sizeof(size_t));
    setOrderOfDimsForNewTensor(1, pOrder);
    shape_t *pShape = reserveMemory(sizeof(shape_t));
    setShape(pShape, pDims, 1, pOrder);
    tensor_t *p = initTensor(pShape, quantizationInitFloat(), NULL);
    tensorFillFromFloatBuffer(p, (float[]){1.f}, 1);
    tensor_t *g = gradInitFloat(p, NULL);
    parameter_t *param = parameterInit(p, g);

    sgd_t sgd;
    sgdInit(&sgd, 0.1f, 0.0f, 0.0f,
            (arithmetic_t){.type = ARITH_FLOAT32, .roundingMode = HALF_AWAY});
    sgd.updateMath.type = ARITH_SYM_INT32; /* bypass the sgdInit guard */
    parameter_t *params[1] = {param};
    optimImpl_t impl;
    impl.sgd = &sgd;
    optimizer_t optim;
    optim.parameter = params;
    optim.states = NULL;
    optim.sizeStates = 1;
    optim.impl = &impl;
    optim.writeBackRounding =
        HALF_AWAY; /* hand-built: field is otherwise uninitialized stack (#279) */

    ASSERT_EXITS_WITH_FAILURE(sgdStepM(&optim));

    freeParameter(param);
}

void testSgdStepMFloatReadsPackedSymGradGeneric(void) {
    /* Packed-SYM grad through the fast path (generic-read guard, PR3 #261).
     * Hand-built optimizer (UnitTestSgd.c:545-556 stack pattern): FLOAT32
     * param, grad = SYM@8 seeded via ONE accumulateFloatIntoSymTensorFixedGrid
     * call. The grad starts fresh (all-zero mantissa from initTensor's
     * zero-fill), so the call triggers first-store grid derivation (spec
     * §4.1): scale = absmax(inc)/qMax, qMax = 2^(8-1)-1 = 127. inc = {2.0,
     * -1.5, 0.5} -> absmax = 2.0 -> scale = 2.0/127. codes = round(inc/scale):
     *   2.0  / scale =  127.0   (exact) -> code  127
     *  -1.5  / scale =  -95.25          -> code  -95
     *   0.5  / scale =   31.75          -> code   32
     * (no rounding ties: HALF_AWAY only matters at x.5, none of these land
     * on one). momentumFactor=0 routes through the #308 fast path (single op,
     * optim->states untouched): param -= lr*dequant(grad).
     * Mutation: forcing the executeOp prologue to raw-cast the packed SYM bytes
     * instead of dequantizing them (its per-dtype operand conversion) misreads
     * the storage as raw floats -> garbage grad values -> RED. */
    size_t *pDims = reserveMemory(1 * sizeof(size_t));
    pDims[0] = 3;
    size_t *pOrder = reserveMemory(1 * sizeof(size_t));
    setOrderOfDimsForNewTensor(1, pOrder);
    shape_t *pShape = reserveMemory(sizeof(shape_t));
    setShape(pShape, pDims, 1, pOrder);
    tensor_t *p = initTensor(pShape, quantizationInitFloat(), NULL);
    tensorFillFromFloatBuffer(p, (float[]){1.f, 2.f, -3.f}, 3);
    tensor_t *g = gradInitSym(p, 8, HALF_AWAY, NULL);
    parameter_t *param = parameterInit(p, g);

    float inc[] = {2.0f, -1.5f, 0.5f};
    accumulateFloatIntoSymTensorFixedGrid(g, inc, 3);

    /* Minimal stack optimizer around one parameter (file convention,
     * UnitTestSgd.c:545-556): momentumFactor==0 -> sgdStepM never reads
     * optim->states (#308). */
    sgd_t sgd;
    sgdInit(&sgd, 0.1f, 0.0f, 0.0f,
            (arithmetic_t){.type = ARITH_FLOAT32, .roundingMode = HALF_AWAY});
    parameter_t *params[1] = {param};
    optimImpl_t impl;
    impl.sgd = &sgd;
    optimizer_t optim;
    optim.parameter = params;
    optim.states = NULL;
    optim.sizeStates = 1;
    optim.impl = &impl;
    optim.writeBackRounding =
        HALF_AWAY; /* hand-built: field is otherwise uninitialized stack (#279) */

    sgdStepM(&optim);

    /* CAPTURE -> free -> assert: derive expected with the SAME formula the
     * comment above walks through (float scale = 2.0f/127.f), not hardcoded
     * decimals, so the tolerance below is only absorbing FP-order noise, not
     * a hand-rounding mismatch. */
    float p0 = ((float *)p->data)[0];
    float p1 = ((float *)p->data)[1];
    float p2 = ((float *)p->data)[2];

    freeParameter(param);

    float lr = 0.1f;
    float scale = 2.0f / 127.f;
    float expected0 = 1.f - lr * (127.f * scale);
    float expected1 = 2.f - lr * (-95.f * scale);
    float expected2 = -3.f - lr * (32.f * scale);

    TEST_ASSERT_FLOAT_WITHIN(1e-6f, expected0, p0);
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, expected1, p1);
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, expected2, p2);
}

void testSgdStepMMixedDtypeMovesBothParamsCorrectly(void) {
    /* Momentum==0 fast path (#308) over a mixed-dtype model: a FLOAT32 and a
     * SYM_INT32 parameter must both update correctly from the SAME
     * momentum-free sgdStepM call, with optim->states left NULL throughout.
     * Per-target dtype dispatch happens in the executeOp funnel's epilogue
     * (conversionMatrix lookup on OUT_WRITE), not via any qtype switch in
     * sgdStepM itself -- this is what lets one code path serve both dtypes. */

    /* Parameter A: FLOAT32 param + grad, shape [2]. */
    size_t *aDims = reserveMemory(1 * sizeof(size_t));
    aDims[0] = 2;
    size_t *aOrder = reserveMemory(1 * sizeof(size_t));
    setOrderOfDimsForNewTensor(1, aOrder);
    shape_t *aShape = reserveMemory(sizeof(shape_t));
    setShape(aShape, aDims, 1, aOrder);
    tensor_t *paramA = initTensor(aShape, quantizationInitFloat(), NULL);
    tensorFillFromFloatBuffer(paramA, (float[]){1.f, 2.f}, 2);
    tensor_t *gradA = gradInitFloat(paramA, NULL);
    tensorFillFromFloatBuffer(gradA, (float[]){1.f, 1.f}, 2);
    parameter_t *A = parameterInit(paramA, gradA);

    /* Parameter B: SYM_INT32 param + grad, shape [2]. */
    size_t *bDims = reserveMemory(1 * sizeof(size_t));
    bDims[0] = 2;
    size_t *bOrder = reserveMemory(1 * sizeof(size_t));
    setOrderOfDimsForNewTensor(1, bOrder);
    shape_t *bShape = reserveMemory(sizeof(shape_t));
    setShape(bShape, bDims, 1, bOrder);
    tensor_t *paramB = initTensor(bShape, quantizationInitSymInt32(HALF_AWAY), NULL);
    tensorFillFromFloatBuffer(paramB, (float[]){0.5f, -0.5f}, 2);
    tensor_t *gradB = gradInitSymInt32(paramB, HALF_AWAY, NULL);
    tensorFillFromFloatBuffer(gradB, (float[]){1.f, 1.f}, 2);
    parameter_t *B = parameterInit(paramB, gradB);

    /* Minimal stack optimizer around two heterogeneous-dtype parameters (file
     * convention, UnitTestSgd.c:545-556 pattern extended to 2 params).
     * momentumFactor==0 -> sgdStepM never reads optim->states (#308). */
    sgd_t sgd;
    sgdInit(&sgd, 0.1f, 0.0f, 0.0f,
            (arithmetic_t){.type = ARITH_FLOAT32, .roundingMode = HALF_AWAY});
    parameter_t *params[2] = {A, B};
    optimImpl_t impl;
    impl.sgd = &sgd;
    optimizer_t optim;
    optim.parameter = params;
    optim.states = NULL;
    optim.sizeStates = 2;
    optim.impl = &impl;
    optim.writeBackRounding =
        HALF_AWAY; /* hand-built: field is otherwise uninitialized stack (#279) */

    sgdStepM(&optim);

    /* CAPTURE -> free -> assert (file convention). */
    float aAfter[2];
    aAfter[0] = ((float *)paramA->data)[0];
    aAfter[1] = ((float *)paramA->data)[1];

    float bBefore[2] = {0.5f, -0.5f};
    tensor_t bAfterFloat;
    quantization_t bAfterFloatQ;
    initFloat32Quantization(&bAfterFloatQ);
    uint8_t bAfterFloatData[2 * sizeof(float)];
    setTensorValuesForConversion(bAfterFloatData, &bAfterFloatQ, paramB, &bAfterFloat);
    convertTensor(paramB, &bAfterFloat);
    float bAfter[2];
    bAfter[0] = ((float *)bAfterFloat.data)[0];
    bAfter[1] = ((float *)bAfterFloat.data)[1];

    freeParameter(A);
    freeParameter(B);

    /* A: pure FLOAT32 arithmetic, no quantization noise -- fast path collapses
     * to param -= lr*grad (weightDecay=0) = param - 0.1*1.0. */
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 0.9f, aAfter[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 1.9f, aAfter[1]);

    /* B: SYM_INT32 param must move in the gradient-descent direction (param
     * decreases where grad > 0), within int12 quantization noise
     * (qMaxBits=12 -> scale ~ 0.6/2047, error << 0.01). */
    TEST_ASSERT_TRUE_MESSAGE(bAfter[0] < bBefore[0], "SYM_INT32 param[0] must decrease");
    TEST_ASSERT_TRUE_MESSAGE(bAfter[1] < bBefore[1], "SYM_INT32 param[1] must decrease");
    TEST_ASSERT_FLOAT_WITHIN(0.01f, 0.4f, bAfter[0]);
    TEST_ASSERT_FLOAT_WITHIN(0.01f, -0.6f, bAfter[1]);
}

/* #279 SRHTE dead-zone fixture (optimizer epic #277, Task 3): shape-[2]
 * SYM_INT32 param. Index 0 is a large "anchor" (grad=0, weightDecay=0) whose
 * decoded value always equals the tensor's absmax by construction -- writeOut's
 * FLOAT32->SYM_INT32 requant (convertFloatTensorToSymInt32Tensor) RE-DERIVES
 * the per-tensor scale from the current decoded values on every OUT_WRITE
 * (there is no hidden full-precision accumulator), so pinning the absmax
 * pins the scale essentially constant across iterations too. Index 1 is the
 * "target": its constant grad produces a per-step update of exactly
 * fraction*scale (fraction=0.25, comfortably sub-ULP: < the 0.5 half-grid
 * boundary round-to-nearest needs to move a code). Caller owns freeing the
 * returned parameter_t (freeParameter cascades param+grad). */
static parameter_t *buildSymDeadZoneFixture(roundingMode_t roundingMode, float lr) {
    size_t *pDims = reserveMemory(1 * sizeof(size_t));
    pDims[0] = 2;
    size_t *pOrder = reserveMemory(1 * sizeof(size_t));
    setOrderOfDimsForNewTensor(1, pOrder);
    shape_t *pShape = reserveMemory(sizeof(shape_t));
    setShape(pShape, pDims, 1, pOrder);
    tensor_t *p = initTensor(pShape, quantizationInitSymInt32(roundingMode), NULL);

    const float anchorVal = 100.f;
    tensorFillFromFloatBuffer(p, (float[]){anchorVal, 0.f}, 2);

    /* qMax for the default 12-bit SYM_INT32 operand width (ODT_SYM_OPERAND_QMAXBITS,
     * #227): 2^(12-1) - 1. Mirrors convertFloatTensorToSymInt32Tensor's own
     * derivation so `scale` here matches what the OUT_WRITE epilogue computes. */
    const float qMax = 2047.f;
    const float scale = anchorVal / qMax;
    const float fraction = 0.25f;
    const float delta = fraction * scale;
    const float targetGrad = delta / lr; /* so that lr*targetGrad == delta */

    tensor_t *g = gradInitFloat(p, NULL);
    tensorFillFromFloatBuffer(g, (float[]){0.f, targetGrad}, 2);

    return parameterInit(p, g);
}

void testSgdStepHalfAwayNeverEscapesSymDeadZone(void) {
    /* #279 control: deterministic HALF_AWAY write-back (the explicit opt-out
     * on the OPTIMIZER, matching the param's own qConfig). Every sgdStepM
     * momentum==0 step re-decodes the target from its (unchanged) integer
     * code, applies the same sub-ULP delta, and requants -- landing back on
     * the exact same code every time. Mutation guard: if writeOut ever
     * stopped re-deriving the target's value from its stored code (e.g. a
     * latent float accumulator were added), this would go RED (code would
     * drift and eventually move), so this test also pins that absence of
     * hidden state. */
    const float lr = 0.1f;
    parameter_t *param = buildSymDeadZoneFixture(HALF_AWAY, lr);

    int32_t initialTargetCode = ((int32_t *)param->param->data)[1];
    TEST_ASSERT_EQUAL_INT(0, initialTargetCode); /* fixture guard */

    sgd_t sgd;
    sgdInit(&sgd, lr, 0.f, 0.f, (arithmetic_t){.type = ARITH_FLOAT32, .roundingMode = HALF_AWAY});
    parameter_t *params[1] = {param};
    optimImpl_t impl;
    impl.sgd = &sgd;
    optimizer_t optim;
    optim.parameter = params;
    optim.states = NULL;
    optim.sizeStates = 1;
    optim.impl = &impl;
    optim.writeBackRounding = HALF_AWAY;

    int codeEverMoved = 0;
    for (int i = 0; i < 500; i++) {
        sgdStepM(&optim);
        if (((int32_t *)param->param->data)[1] != initialTargetCode) {
            codeEverMoved = 1;
        }
    }
    int32_t finalTargetCode = ((int32_t *)param->param->data)[1];

    freeParameter(param);

    TEST_ASSERT_FALSE_MESSAGE(codeEverMoved, "#279 dead-zone: HALF_AWAY must never move a sub-ULP "
                                             "SYM_INT32 code across 500 identical updates");
    TEST_ASSERT_EQUAL_INT(0, finalTargetCode);
}

void testSgdStepDetWriteBackOverridesSrParamQConfig(void) {
    /* #279 inverse direction: the param's own qConfig says SR_HALF_AWAY, but
     * the OPTIMIZER opts out to deterministic HALF_AWAY -- the write-back
     * must stay frozen. Together with the SR-over-HALF_AWAY test below this
     * pins that the optimizer's writeBackRounding wins in BOTH directions
     * (the storage qConfig never leaks into the training write-back), which
     * is what makes HALF_AWAY-for-storage + SR-for-training composable. */
    rngSetSeed(12345u);

    const float lr = 0.1f;
    parameter_t *param = buildSymDeadZoneFixture(SR_HALF_AWAY, lr);

    int32_t initialTargetCode = ((int32_t *)param->param->data)[1];
    TEST_ASSERT_EQUAL_INT(0, initialTargetCode); /* fixture guard */

    sgd_t sgd;
    sgdInit(&sgd, lr, 0.f, 0.f, (arithmetic_t){.type = ARITH_FLOAT32, .roundingMode = HALF_AWAY});
    parameter_t *params[1] = {param};
    optimImpl_t impl;
    impl.sgd = &sgd;
    optimizer_t optim;
    optim.parameter = params;
    optim.states = NULL;
    optim.sizeStates = 1;
    optim.impl = &impl;
    optim.writeBackRounding = HALF_AWAY;

    int codeEverMoved = 0;
    for (int i = 0; i < 500; i++) {
        sgdStepM(&optim);
        if (((int32_t *)param->param->data)[1] != initialTargetCode) {
            codeEverMoved = 1;
        }
    }
    roundingMode_t storageModeAfter =
        ((symInt32QConfig_t *)param->param->quantization->qConfig)->roundingMode;

    freeParameter(param);

    TEST_ASSERT_FALSE_MESSAGE(codeEverMoved,
                              "#279 opt-out: deterministic writeBackRounding must keep a "
                              "sub-ULP update frozen even when the param's own qConfig "
                              "says SR_HALF_AWAY");
    TEST_ASSERT_EQUAL_INT_MESSAGE(SR_HALF_AWAY, storageModeAfter,
                                  "write-back swap must restore the param's own storage "
                                  "roundingMode");
}

void testSgdStepOptimizerSrWriteBackOverridesHalfAwayParamQConfig(void) {
    /* #279 default policy: the OPTIMIZER owns the training write-back rounding.
     * A param whose storage qConfig says HALF_AWAY (the obvious construction,
     * and what serialization persists) must still escape the dead-zone when
     * optim->writeBackRounding is SR_HALF_AWAY -- and the swap must be
     * transient: after stepping, the storage config still reads HALF_AWAY. */
    rngSetSeed(24680u);

    const float lr = 0.1f;
    parameter_t *param = buildSymDeadZoneFixture(HALF_AWAY, lr);

    int32_t initialTargetCode = ((int32_t *)param->param->data)[1];
    TEST_ASSERT_EQUAL_INT(0, initialTargetCode); /* fixture guard */

    sgd_t sgd;
    sgdInit(&sgd, lr, 0.f, 0.f, (arithmetic_t){.type = ARITH_FLOAT32, .roundingMode = HALF_AWAY});
    parameter_t *params[1] = {param};
    optimImpl_t impl;
    impl.sgd = &sgd;
    optimizer_t optim;
    optim.parameter = params;
    optim.states = NULL;
    optim.sizeStates = 1;
    optim.impl = &impl;
    optim.writeBackRounding = SR_HALF_AWAY;

    int codeEverMoved = 0;
    for (int i = 0; i < 500; i++) {
        sgdStepM(&optim);
        if (((int32_t *)param->param->data)[1] != initialTargetCode) {
            codeEverMoved = 1;
        }
    }
    roundingMode_t storageModeAfter =
        ((symInt32QConfig_t *)param->param->quantization->qConfig)->roundingMode;

    freeParameter(param);

    TEST_ASSERT_TRUE_MESSAGE(codeEverMoved, "#279 default: optimizer-owned SR_HALF_AWAY write-back "
                                            "must escape the dead-zone of a HALF_AWAY-configured "
                                            "SYM_INT32 param within 500 iterations");
    TEST_ASSERT_EQUAL_INT_MESSAGE(HALF_AWAY, storageModeAfter,
                                  "write-back swap must restore the param's own storage "
                                  "roundingMode (it is serialized into checkpoints)");
}

void testSgdStepMMomentumPathHonorsOptimizerSrWriteBack(void) {
    /* #279: the momentum>0 param write-back is the same training write-back
     * seam as the momentum==0 fast path -- the optimizer's SR_HALF_AWAY must
     * escape the dead-zone of a HALF_AWAY-configured param through THIS path
     * too. momentumFactor=0.25 keeps the asymptotic per-step delta at
     * fraction/(1-m) = 0.333*scale -- still sub-ULP (< the 0.5 half-grid
     * boundary a deterministic round needs), so HALF_AWAY would stay frozen
     * forever while SR escapes with per-step probability >= 0.25. FLOAT32
     * momentum state (the shipped #277 decoupling) accumulates exactly. */
    rngSetSeed(97531u);

    const float lr = 0.1f;
    parameter_t *param = buildSymDeadZoneFixture(HALF_AWAY, lr);

    int32_t initialTargetCode = ((int32_t *)param->param->data)[1];
    TEST_ASSERT_EQUAL_INT(0, initialTargetCode); /* fixture guard */

    sgd_t sgd;
    sgdInit(&sgd, lr, 0.25f, 0.f, (arithmetic_t){.type = ARITH_FLOAT32, .roundingMode = HALF_AWAY});
    parameter_t *params[1] = {param};
    optimImpl_t impl;
    impl.sgd = &sgd;

    tensor_t *state = initTensor(getShapeLike(param->param->shape), quantizationInitFloat(), NULL);
    tensor_t *stateBuffers[1] = {state};
    states_t paramStates = {.stateBuffers = stateBuffers, .statesPerParameter = 1};
    states_t *states[1] = {&paramStates};

    optimizer_t optim;
    optim.parameter = params;
    optim.states = states;
    optim.sizeStates = 1;
    optim.impl = &impl;
    optim.writeBackRounding = SR_HALF_AWAY;

    int codeEverMoved = 0;
    for (int i = 0; i < 500; i++) {
        sgdStepM(&optim);
        if (((int32_t *)param->param->data)[1] != initialTargetCode) {
            codeEverMoved = 1;
        }
    }

    freeTensor(state);
    freeParameter(param);

    TEST_ASSERT_TRUE_MESSAGE(codeEverMoved, "#279: optimizer SR write-back must escape the "
                                            "dead-zone through the momentum param path");
}

void testSgdStepMStateWriteBackHonorsOptimizerSrRounding(void) {
    /* #279: momentum-STATE storage is a training write-back too (Leo's call:
     * params + states). A quantized state accumulating sub-ULP increments has
     * the identical dead-zone -- the running velocity freezes instead of the
     * weight. Fixture: SYM_INT32 state with a 100.0 anchor pinning the scale
     * (same construction as buildSymDeadZoneFixture); momentumFactor=1 makes
     * the state kernel out = state + grad, so a grad of 0.25*scale is a pure
     * sub-ULP state increment. HALF_AWAY on the state's own qConfig would
     * freeze it; the optimizer's SR_HALF_AWAY must move it. The param is
     * FLOAT32 so the param write-back stays out of the picture. */
    rngSetSeed(86420u);

    size_t *pDims = reserveMemory(1 * sizeof(size_t));
    pDims[0] = 2;
    size_t *pOrder = reserveMemory(1 * sizeof(size_t));
    setOrderOfDimsForNewTensor(1, pOrder);
    shape_t *pShape = reserveMemory(sizeof(shape_t));
    setShape(pShape, pDims, 1, pOrder);
    tensor_t *p = initTensor(pShape, quantizationInitFloat(), NULL);
    tensorFillFromFloatBuffer(p, (float[]){0.f, 0.f}, 2);

    const float anchorVal = 100.f;
    const float qMax = 2047.f;
    const float scale = anchorVal / qMax;
    const float subUlpIncrement = 0.25f * scale;

    tensor_t *g = gradInitFloat(p, NULL);
    tensorFillFromFloatBuffer(g, (float[]){0.f, subUlpIncrement}, 2);
    parameter_t *param = parameterInit(p, g);

    tensor_t *state = initTensor(getShapeLike(p->shape), quantizationInitSymInt32(HALF_AWAY), NULL);
    tensorFillFromFloatBuffer(state, (float[]){anchorVal, 0.f}, 2);
    int32_t initialStateCode = ((int32_t *)state->data)[1];
    TEST_ASSERT_EQUAL_INT(0, initialStateCode); /* fixture guard */

    sgd_t sgd;
    sgdInit(&sgd, 0.1f, 1.0f, 0.f,
            (arithmetic_t){.type = ARITH_FLOAT32, .roundingMode = HALF_AWAY});
    parameter_t *params[1] = {param};
    optimImpl_t impl;
    impl.sgd = &sgd;
    tensor_t *stateBuffers[1] = {state};
    states_t paramStates = {.stateBuffers = stateBuffers, .statesPerParameter = 1};
    states_t *states[1] = {&paramStates};
    optimizer_t optim;
    optim.parameter = params;
    optim.states = states;
    optim.sizeStates = 1;
    optim.impl = &impl;
    optim.writeBackRounding = SR_HALF_AWAY;

    int stateCodeEverMoved = 0;
    for (int i = 0; i < 500; i++) {
        sgdStepM(&optim);
        if (((int32_t *)state->data)[1] != initialStateCode) {
            stateCodeEverMoved = 1;
        }
    }
    roundingMode_t stateStorageModeAfter =
        ((symInt32QConfig_t *)state->quantization->qConfig)->roundingMode;

    freeTensor(state);
    freeParameter(param);

    TEST_ASSERT_TRUE_MESSAGE(stateCodeEverMoved,
                             "#279: optimizer SR write-back must move a sub-ULP increment "
                             "on quantized momentum-state storage within 500 iterations");
    TEST_ASSERT_EQUAL_INT_MESSAGE(HALF_AWAY, stateStorageModeAfter,
                                  "state write-back swap must restore the state's own "
                                  "storage roundingMode");
}

void testSgdMCreateOptimMomentumZeroAllocatesNoStates(void) {
    /* #308: momentumFactor == 0 must not allocate momentum-state buffers
     * (real RAM on an MCU for a configuration that has no state). Contract:
     * optim->states == NULL; freeOptim still frees all parameters.
     * RED today: the factory unconditionally allocates states. */
    quantization_t *layerQ = quantizationInitFloat();

    size_t *wDims = reserveMemory(2 * sizeof(size_t));
    wDims[0] = 1;
    wDims[1] = 3;
    size_t *wOrder = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, wOrder);
    shape_t *wShape = reserveMemory(sizeof(shape_t));
    setShape(wShape, wDims, 2, wOrder);
    tensor_t *wParam = initTensor(wShape, quantizationInitFloat(), NULL);
    tensorFillFromFloatBuffer(wParam, (float[]){1.f, 2.f, 3.f}, 3);
    tensor_t *wGrad = gradInitFloat(wParam, NULL);
    parameter_t *weights = parameterInit(wParam, wGrad);

    layer_t *linear = buildBorrowedLinearLayer(weights, NULL, layerQ);
    layer_t *model[] = {linear};

    quantization_t *momentumQ = quantizationInitFloat();
    optimizer_t *optim =
        sgdMCreateOptim(0.1f, 0.0f, 0.0f, model, 1, momentumQ,
                        (arithmetic_t){.type = ARITH_FLOAT32, .roundingMode = HALF_AWAY});

    /* CAPTURE -> free -> assert. */
    states_t **capturedStates = optim->states;
    size_t capturedSizeStates = optim->sizeStates;

    freeOptim(optim);
    freeLinearLayerShellOnly(linear);
    freeQuantization(momentumQ);
    freeQuantization(layerQ);

    TEST_ASSERT_EQUAL_size_t(1, capturedSizeStates);
    TEST_ASSERT_NULL(capturedStates);
}

void testSgdMCreateOptimDefaultsWriteBackRoundingToSr(void) {
    /* #279 ratified default: a factory-built optimizer trains with seeded
     * SR_HALF_AWAY write-backs (the sweep-proven dead-zone escape), so a
     * quantized model built the obvious way cannot silently sit frozen.
     * Deterministic HALF_AWAY is the explicit opt-out. */
    quantization_t *layerQ = quantizationInitFloat();

    size_t *wDims = reserveMemory(2 * sizeof(size_t));
    wDims[0] = 1;
    wDims[1] = 3;
    size_t *wOrder = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, wOrder);
    shape_t *wShape = reserveMemory(sizeof(shape_t));
    setShape(wShape, wDims, 2, wOrder);
    tensor_t *wParam = initTensor(wShape, quantizationInitFloat(), NULL);
    tensorFillFromFloatBuffer(wParam, (float[]){1.f, 2.f, 3.f}, 3);
    tensor_t *wGrad = gradInitFloat(wParam, NULL);
    parameter_t *weights = parameterInit(wParam, wGrad);

    layer_t *linear = buildBorrowedLinearLayer(weights, NULL, layerQ);
    layer_t *model[] = {linear};

    quantization_t *momentumQ = quantizationInitFloat();
    optimizer_t *optim =
        sgdMCreateOptim(0.1f, 0.0f, 0.0f, model, 1, momentumQ,
                        (arithmetic_t){.type = ARITH_FLOAT32, .roundingMode = HALF_AWAY});

    /* CAPTURE -> free -> assert. */
    roundingMode_t capturedDefault = optim->writeBackRounding;

    freeOptim(optim);
    freeLinearLayerShellOnly(linear);
    freeQuantization(momentumQ);
    freeQuantization(layerQ);

    TEST_ASSERT_EQUAL_INT_MESSAGE(SR_HALF_AWAY, capturedDefault,
                                  "#279: sgdMCreateOptim must default writeBackRounding "
                                  "to seeded SR_HALF_AWAY");
}

void testOptimizerSetWriteBackRoundingOptsOutToDeterministic(void) {
    /* #279: the explicit opt-out -- callers wanting deterministic write-backs
     * (bit-parity twins, storage round-trip tests) state it in one call
     * instead of silently inheriting the non-learning default. */
    quantization_t *layerQ = quantizationInitFloat();

    size_t *wDims = reserveMemory(2 * sizeof(size_t));
    wDims[0] = 1;
    wDims[1] = 3;
    size_t *wOrder = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, wOrder);
    shape_t *wShape = reserveMemory(sizeof(shape_t));
    setShape(wShape, wDims, 2, wOrder);
    tensor_t *wParam = initTensor(wShape, quantizationInitFloat(), NULL);
    tensorFillFromFloatBuffer(wParam, (float[]){1.f, 2.f, 3.f}, 3);
    tensor_t *wGrad = gradInitFloat(wParam, NULL);
    parameter_t *weights = parameterInit(wParam, wGrad);

    layer_t *linear = buildBorrowedLinearLayer(weights, NULL, layerQ);
    layer_t *model[] = {linear};

    quantization_t *momentumQ = quantizationInitFloat();
    optimizer_t *optim =
        sgdMCreateOptim(0.1f, 0.0f, 0.0f, model, 1, momentumQ,
                        (arithmetic_t){.type = ARITH_FLOAT32, .roundingMode = HALF_AWAY});

    optimizerSetWriteBackRounding(optim, HALF_AWAY);

    /* CAPTURE -> free -> assert. */
    roundingMode_t capturedAfterOptOut = optim->writeBackRounding;

    freeOptim(optim);
    freeLinearLayerShellOnly(linear);
    freeQuantization(momentumQ);
    freeQuantization(layerQ);

    TEST_ASSERT_EQUAL_INT_MESSAGE(HALF_AWAY, capturedAfterOptOut,
                                  "optimizerSetWriteBackRounding must write through to the "
                                  "field the step write-backs read");
}

void testOptimizerVtableGetSetLrRoundTripsSgdLearningRate(void) {
    /* #327: the scheduler reaches the LR only through the vtable. Pin that
     * the SGD row exposes working accessors and that setLr writes through to
     * the struct field the step kernels actually read. */
    sgd_t sgd;
    sgdInit(&sgd, 0.5f, 0.9f, 0.0f,
            (arithmetic_t){.type = ARITH_FLOAT32, .roundingMode = HALF_AWAY});
    optimImpl_t impl = {.sgd = &sgd};
    optimizer_t optim = {
        .type = SGD_M, .impl = &impl, .parameter = NULL, .states = NULL, .sizeStates = 0};

    TEST_ASSERT_EQUAL_FLOAT(0.5f, optimizerFunctions[SGD_M].getLr(&optim));

    optimizerFunctions[SGD_M].setLr(&optim, 0.125f);

    TEST_ASSERT_EQUAL_FLOAT(0.125f, optimizerFunctions[SGD_M].getLr(&optim));
    TEST_ASSERT_EQUAL_FLOAT(0.125f, sgd.learningRate);
}

int main() {
    UNITY_BEGIN();
    RUN_TEST(testOptimizerSkipsFrozenLayerInCountAndCollection);
    RUN_TEST(testSgdCreateAllFrozenModelExits);
    RUN_TEST(testSgdMCreateOptim);
    RUN_TEST(testSGDStep);
    RUN_TEST(testSGDZeroGrad);
    RUN_TEST(testSgdZeroGradOnSymInt32GradZeroesMantissasAndResetsScale);
    RUN_TEST(testSgdMCreateOptimMomentumStateIsFloatIndependentOfSymInt32ParamDtype);
    RUN_TEST(testLayerNormContributesTwoOptimizerStates);
    RUN_TEST(testSgdMCreateOptimRegistersLayerNormGammaAndBeta);
    RUN_TEST(testSgdStepSymInt32DoesNotRoundTripGrad);
    RUN_TEST(testSgdStepMMomentumZeroIgnoresStatesAndUpdatesParam);
    RUN_TEST(testSgdMCreateOptimAdmitsPackedSymGradStorage);
    RUN_TEST(testSgdMCreateOptimRejectsBoolGradStorage);
    RUN_TEST(testSgdMCreateOptimRejectsNullGradSlot);
    RUN_TEST(testSgdMCreateOptimRejectsNonFloat32UpdateMath);
    RUN_TEST(testSgdStepMRejectsNonFloat32UpdateMath);
    RUN_TEST(testSgdStepMFloatReadsPackedSymGradGeneric);
    RUN_TEST(testSgdZeroGradAsymSubByteZeroesAllPackedBytes);
    RUN_TEST(testSgdZeroGradSymSubByteZeroesAllPackedBytesAndResetsScale);
    RUN_TEST(testSgdStepMMixedDtypeMovesBothParamsCorrectly);
    RUN_TEST(testSgdStepHalfAwayNeverEscapesSymDeadZone);
    RUN_TEST(testSgdStepDetWriteBackOverridesSrParamQConfig);
    RUN_TEST(testSgdStepOptimizerSrWriteBackOverridesHalfAwayParamQConfig);
    RUN_TEST(testSgdStepMMomentumPathHonorsOptimizerSrWriteBack);
    RUN_TEST(testSgdStepMStateWriteBackHonorsOptimizerSrRounding);
    RUN_TEST(testSgdMCreateOptimMomentumZeroAllocatesNoStates);
    RUN_TEST(testSgdMCreateOptimDefaultsWriteBackRoundingToSr);
    RUN_TEST(testOptimizerSetWriteBackRoundingOptsOutToDeterministic);
    RUN_TEST(testOptimizerVtableGetSetLrRoundTripsSgdLearningRate);
    return UNITY_END();
}
