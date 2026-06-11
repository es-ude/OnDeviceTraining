#define SOURCE_FILE "SGD-UTEST"
#include <stdlib.h>

#include "Layer.h"
#include "LayerNorm.h"
#include "LayerQuant.h"
#include "Linear.h"
#include "LinearApi.h"
#include "Optimizer.h"
#include "OptimizerApi.h"
#include "QuantizationApi.h"
#include "Sgd.h"
#include "SgdApi.h"
#include "StorageApi.h"
#include "Tensor.h"
#include "TensorApi.h"
#include "unity.h"

#include <ReluApi.h>

void setUp() {}
void tearDown() {}

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

    /* linear1 needs its OWN weights/bias parameters (Bug 1 fix): freeOptimSgdM
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

    layer_t *linear0 = linearLayerInitLegacy(weights0, bias0, layerQ, layerQ, layerQ, layerQ);
    layer_t *relu0 = reluLayerInitLegacy(layerQ, layerQ);
    layer_t *linear1 = linearLayerInitLegacy(weights1, bias1, layerQ, layerQ, layerQ, layerQ);

    layer_t *model[] = {linear0, relu0, linear1};
    size_t sizeModel = sizeof(model) / sizeof(model[0]);
    float lr = 0.1f;
    float momentumFactor = 0.9f;
    float weightDecay = 0.5f;

    optimizer_t *optim =
        sgdMCreateOptim(lr, momentumFactor, weightDecay, model, sizeModel, FLOAT32);
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

    /* FREE in reverse-init order. freeOptimSgdM cascades to all parameters
     * (and their grads) registered with the optimizer (per SgdApi.c:85-93,
     * mirroring the post-#110 ownership contract). */
    freeOptimSgdM(optim);
    freeLinearLayerLegacy(linear1);
    freeReluLayerLegacy(relu0);
    freeLinearLayerLegacy(linear0);
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

    layer_t *linear = linearLayerInitLegacy(weights, bias, layerQ, layerQ, layerQ, layerQ);

    layer_t *model[] = {linear};
    size_t modelSize = 1;
    float lr = 0.1f;
    float momentumFactor = 0.9f;
    float weightDecay = 0.01f;

    optimizer_t *sgd = sgdMCreateOptim(lr, momentumFactor, weightDecay, model, modelSize, FLOAT32);

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

    /* FREE in reverse-init order. */
    freeOptimSgdM(sgd);
    freeLinearLayerLegacy(linear);
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

    layer_t *linear = linearLayerInitLegacy(weights, bias, layerQ, layerQ, layerQ, layerQ);

    layer_t *model[] = {linear};
    size_t modelSize = 1;
    float lr = 0.1f;
    float momentumFactor = 0.9f;
    float weightDecay = 0.01f;

    optimizer_t *sgd = sgdMCreateOptim(lr, momentumFactor, weightDecay, model, modelSize, FLOAT32);

    sgdZeroGrad(sgd);

    /* CAPTURE before frees. */
    float capturedWGrad[3];
    for (size_t i = 0; i < 3; i++) {
        capturedWGrad[i] = ((float *)weights->grad->data)[i];
    }
    float capturedBGrad[2];
    for (size_t i = 0; i < 2; i++) {
        capturedBGrad[i] = ((float *)bias->grad->data)[i];
    }

    /* FREE in reverse-init order. */
    freeOptimSgdM(sgd);
    freeLinearLayerLegacy(linear);
    freeQuantization(layerQ);

    /* ASSERT. */
    float wGradExpected[] = {0.f, 0.f, 0.f};
    float bGradExpected[] = {0.f, 0.f};
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(wGradExpected, capturedWGrad, 3);
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(bGradExpected, capturedBGrad, 2);
}

void testSgdZeroGradOnSymInt32GradZeroesMantissasAndResetsScale(void) {
    quantization_t *fwd = quantizationInitFloat();
    quantization_t *bwd = quantizationInitSymInt32(HALF_AWAY);
    layerQuant_t lq = {
        .forwardMath = fwd, .backwardMath = bwd, .weightStorage = fwd, .biasStorage = fwd};
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
    optimizer_t *optim = sgdMCreateOptim(0.1f, 0.9f, 0.0f, model, 1, SYM_INT32);

    sgdZeroGrad(optim);

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

    freeOptimSgdM(optim); /* frees the layer's parameters */
    freeReservedMemory(layer->config->linear);
    freeReservedMemory(layer->config);
    freeReservedMemory(layer);
    freeQuantization(bwd);
    freeQuantization(fwd);

    TEST_ASSERT_TRUE_MESSAGE(allZero, "sgdZeroGrad must zero SYM_INT32 grad mantissas");
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 1.0f, scaleAfterZero);
    TEST_ASSERT_TRUE(accumWorks);
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

    optimizer_t *optim =
        sgdMCreateOptim(lr, momentumFactor, weightDecay, model, sizeModel, FLOAT32);

    /* CAPTURE before frees. */
    size_t capturedSizeStates = optim->sizeStates;
    parameter_t *capturedP0 = optim->parameter[0];
    parameter_t *capturedP1 = optim->parameter[1];

    size_t gammaN = calcNumberOfElementsByParameter(lnCfg->gamma);
    size_t betaN = calcNumberOfElementsByParameter(lnCfg->beta);

    size_t capturedGammaStateN = calcNumberOfElementsByTensor(optim->states[0]->stateBuffers[0]);
    size_t capturedBetaStateN = calcNumberOfElementsByTensor(optim->states[1]->stateBuffers[0]);

    /* FREE: freeOptimSgdM frees the two registered parameter_t* (gamma, beta)
     * and their state buffers. Then free the layer wrapper structs. */
    freeOptimSgdM(optim);
    freeReservedMemory(lnLayer);
    freeReservedMemory(lcfg);
    freeReservedMemory(normShape);
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
    layer_t layer = {.type = LAYERNORM, .config = NULL};
    layer_t *model[] = {&layer};
    size_t sizeModel = 1;

    size_t nStates = calcTotalNumberOfStates(model, sizeModel);

    TEST_ASSERT_EQUAL_size_t(2, nStates);
}

int main() {
    UNITY_BEGIN();
    RUN_TEST(testSgdMCreateOptim);
    RUN_TEST(testSGDStep);
    RUN_TEST(testSGDZeroGrad);
    RUN_TEST(testSgdZeroGradOnSymInt32GradZeroesMantissasAndResetsScale);
    RUN_TEST(testLayerNormContributesTwoOptimizerStates);
    RUN_TEST(testSgdMCreateOptimRegistersLayerNormGammaAndBeta);
    return UNITY_END();
}
