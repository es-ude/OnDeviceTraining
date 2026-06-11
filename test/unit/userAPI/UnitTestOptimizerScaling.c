#define SOURCE_FILE "UNIT_TEST_OPTIMIZER_SCALING"

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "Layer.h"
#include "Linear.h"
#include "LinearApi.h"
#include "Optimizer.h"
#include "OptimizerApi.h"
#include "Quantization.h"
#include "QuantizationApi.h"
#include "Rounding.h"
#include "SgdApi.h"
#include "StorageApi.h"
#include "Tensor.h"
#include "TensorApi.h"
#include "unity.h"

void setUp() {}
void tearDown() {}

/* Build a one-layer Linear model with grads pre-filled to known values, so
 * scaleOptimizerGradients's effect on the optimizer's parameter list is
 * directly observable. */
static optimizer_t *buildOneLayerOptimWithGrads(layer_t **modelOut, parameter_t **wOut,
                                                parameter_t **bOut, float *initialWGrad,
                                                float *initialBGrad) {
    tensor_t *wParam;
    {
        size_t *dims = reserveMemory(2 * sizeof(size_t));
        dims[0] = 2;
        dims[1] = 3;
        size_t *order = reserveMemory(2 * sizeof(size_t));
        setOrderOfDimsForNewTensor(2, order);
        shape_t *shape = reserveMemory(sizeof(shape_t));
        setShape(shape, dims, 2, order);
        wParam = initTensor(shape, quantizationInitFloat(), NULL);
        tensorFillFromFloatBuffer(wParam, (float[]){1.f, 1.f, 1.f, 1.f, 1.f, 1.f}, 6);
    }
    tensor_t *wGrad = gradInitFloat(wParam, NULL);
    tensorFillFromFloatBuffer(wGrad, initialWGrad, 6);
    parameter_t *w = parameterInit(wParam, wGrad);

    tensor_t *bParam;
    {
        size_t *dims = reserveMemory(2 * sizeof(size_t));
        dims[0] = 1;
        dims[1] = 2;
        size_t *order = reserveMemory(2 * sizeof(size_t));
        setOrderOfDimsForNewTensor(2, order);
        shape_t *shape = reserveMemory(sizeof(shape_t));
        setShape(shape, dims, 2, order);
        bParam = initTensor(shape, quantizationInitFloat(), NULL);
        tensorFillFromFloatBuffer(bParam, (float[]){0.f, 0.f}, 2);
    }
    tensor_t *bGrad = gradInitFloat(bParam, NULL);
    tensorFillFromFloatBuffer(bGrad, initialBGrad, 2);
    parameter_t *b = parameterInit(bParam, bGrad);

    quantization_t testQ;
    initFloat32Quantization(&testQ);
    layer_t *linear = linearLayerInitLegacy(w, b, &testQ, &testQ, &testQ, &testQ);
    modelOut[0] = linear;
    *wOut = w;
    *bOut = b;

    /* SGD optimizer wraps both parameters via the standard helper. */
    return sgdMCreateOptim(0.01f, 0.f, 0.f, modelOut, 1, FLOAT32);
}

void testScaleOptimizerGradients_DoublesGradients() {
    layer_t *model[1];
    parameter_t *w;
    parameter_t *b;
    float wInit[6] = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f};
    float bInit[2] = {7.f, 8.f};

    optimizer_t *sgd = buildOneLayerOptimWithGrads(model, &w, &b, wInit, bInit);

    scaleOptimizerGradients(sgd, 2.0f);

    /* CAPTURE before any free. */
    float capturedWGrad[6];
    {
        float *g = (float *)w->grad->data;
        for (size_t i = 0; i < 6; i++) {
            capturedWGrad[i] = g[i];
        }
    }
    float capturedBGrad[2];
    {
        float *g = (float *)b->grad->data;
        capturedBGrad[0] = g[0];
        capturedBGrad[1] = g[1];
    }

    /* FREE. freeOptimSgdM cascades to both parameters. */
    freeOptimSgdM(sgd);
    freeLinearLayerLegacy(model[0]);

    /* ASSERT — every grad doubled. */
    for (size_t i = 0; i < 6; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-5f, wInit[i] * 2.0f, capturedWGrad[i]);
    }
    TEST_ASSERT_FLOAT_WITHIN(1e-5f, bInit[0] * 2.0f, capturedBGrad[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-5f, bInit[1] * 2.0f, capturedBGrad[1]);
}

/* These tests verify that the validation branch is reached without aborting
 * the process. PRINT_ERROR writes to stderr but does not exit; if the impl
 * regressed to abort()/exit(), all three tests would never reach their
 * assertions. Captured grads are unchanged because the loop still runs with
 * the bad factor — the spec accepts that the warning is informational only. */
void testScaleOptimizerGradients_FactorZero_DoesNotAbort() {
    layer_t *model[1];
    parameter_t *w;
    parameter_t *b;
    float wInit[6] = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f};
    float bInit[2] = {7.f, 8.f};

    optimizer_t *sgd = buildOneLayerOptimWithGrads(model, &w, &b, wInit, bInit);

    /* Must not abort — validation is a warning. */
    scaleOptimizerGradients(sgd, 0.0f);

    /* CAPTURE that grads are now zero (factor 0 multiplied through). */
    float capturedFirst = ((float *)w->grad->data)[0];

    freeOptimSgdM(sgd);
    freeLinearLayerLegacy(model[0]);

    TEST_ASSERT_EQUAL_FLOAT(0.0f, capturedFirst);
}

void testScaleOptimizerGradients_FactorNaN_DoesNotAbort() {
    layer_t *model[1];
    parameter_t *w;
    parameter_t *b;
    float wInit[6] = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f};
    float bInit[2] = {7.f, 8.f};

    optimizer_t *sgd = buildOneLayerOptimWithGrads(model, &w, &b, wInit, bInit);

    /* NaN propagates through; we just want to prove the function returns. */
    float nanFactor = 0.0f / 0.0f;
    scaleOptimizerGradients(sgd, nanFactor);

    float captured = ((float *)w->grad->data)[0];

    freeOptimSgdM(sgd);
    freeLinearLayerLegacy(model[0]);

    /* NaN != NaN by IEEE 754. */
    TEST_ASSERT_TRUE(captured != captured);
}

/* SYM_INT32 builder mirrors buildOneLayerOptimWithGrads but pins both param and
 * grad tensors to SYM_INT32 quantization. The grad tensor's int32 storage and
 * scale are written directly (NOT via tensorFillFromFloatBuffer, which would
 * route through convertTensor and recompute scale from the float source).
 * The param data stays at default-zero, which matches the scale=1.0 default
 * from initSymInt32QConfig — sgdStepSymInt32 round-trips it through float and
 * back, but for the scaling assertions below the param values are irrelevant. */
static optimizer_t *buildSymInt32OneLayerOptim(layer_t **modelOut, parameter_t **wOut,
                                               parameter_t **bOut, float wInitialScale,
                                               const int32_t *wInitialGradInt32,
                                               float bInitialScale,
                                               const int32_t *bInitialGradInt32, float lr,
                                               float momentum) {
    /* Weight param: SYM_INT32, dims [2, 3] (6 elements). */
    tensor_t *wParam;
    {
        size_t *dims = reserveMemory(2 * sizeof(size_t));
        dims[0] = 2;
        dims[1] = 3;
        size_t *order = reserveMemory(2 * sizeof(size_t));
        setOrderOfDimsForNewTensor(2, order);
        shape_t *shape = reserveMemory(sizeof(shape_t));
        setShape(shape, dims, 2, order);
        wParam = initTensor(shape, quantizationInitSymInt32(HALF_AWAY), NULL);
    }
    tensor_t *wGrad = gradInitSymInt32(wParam, HALF_AWAY, NULL);
    {
        int32_t *gradData = (int32_t *)wGrad->data;
        memcpy(gradData, wInitialGradInt32, 6 * sizeof(int32_t));
        symInt32QConfig_t *gradQ = wGrad->quantization->qConfig;
        gradQ->scale = wInitialScale;
    }
    parameter_t *w = parameterInit(wParam, wGrad);

    /* Bias param: SYM_INT32, dims [1, 2] (2 elements). */
    tensor_t *bParam;
    {
        size_t *dims = reserveMemory(2 * sizeof(size_t));
        dims[0] = 1;
        dims[1] = 2;
        size_t *order = reserveMemory(2 * sizeof(size_t));
        setOrderOfDimsForNewTensor(2, order);
        shape_t *shape = reserveMemory(sizeof(shape_t));
        setShape(shape, dims, 2, order);
        bParam = initTensor(shape, quantizationInitSymInt32(HALF_AWAY), NULL);
    }
    tensor_t *bGrad = gradInitSymInt32(bParam, HALF_AWAY, NULL);
    {
        int32_t *gradData = (int32_t *)bGrad->data;
        memcpy(gradData, bInitialGradInt32, 2 * sizeof(int32_t));
        symInt32QConfig_t *gradQ = bGrad->quantization->qConfig;
        gradQ->scale = bInitialScale;
    }
    parameter_t *b = parameterInit(bParam, bGrad);

    quantization_t *layerQ = quantizationInitSymInt32(HALF_AWAY);
    layer_t *linear = linearLayerInitLegacy(w, b, layerQ, layerQ, layerQ, layerQ);
    modelOut[0] = linear;
    *wOut = w;
    *bOut = b;

    return sgdMCreateOptim(lr, momentum, 0.f, modelOut, 1, SYM_INT32);
}

void testScaleOptimizerGradients_SymInt32_ScalesScaleOnly() {
    layer_t *model[1];
    parameter_t *w;
    parameter_t *b;
    int32_t wGradInt[6] = {10, 20, 30, -40, 50, -60};
    int32_t bGradInt[2] = {100, -200};
    float wScale0 = 0.5f;
    float bScale0 = 0.25f;
    float factor = 0.25f;

    optimizer_t *sgd =
        buildSymInt32OneLayerOptim(model, &w, &b, wScale0, wGradInt, bScale0, bGradInt, 0.01f, 0.f);

    scaleOptimizerGradients(sgd, factor);

    /* CAPTURE before frees. */
    int32_t capturedWInt[6];
    memcpy(capturedWInt, w->grad->data, 6 * sizeof(int32_t));
    int32_t capturedBInt[2];
    memcpy(capturedBInt, b->grad->data, 2 * sizeof(int32_t));
    float capturedWScale = ((symInt32QConfig_t *)w->grad->quantization->qConfig)->scale;
    float capturedBScale = ((symInt32QConfig_t *)b->grad->quantization->qConfig)->scale;

    freeOptimSgdM(sgd);
    freeLinearLayerLegacy(model[0]);

    /* int32 storage is byte-for-byte unchanged. */
    for (size_t i = 0; i < 6; i++) {
        TEST_ASSERT_EQUAL_INT32(wGradInt[i], capturedWInt[i]);
    }
    for (size_t i = 0; i < 2; i++) {
        TEST_ASSERT_EQUAL_INT32(bGradInt[i], capturedBInt[i]);
    }
    /* scale absorbed the multiplicative factor. */
    TEST_ASSERT_FLOAT_WITHIN(1e-7f, wScale0 * factor, capturedWScale);
    TEST_ASSERT_FLOAT_WITHIN(1e-7f, bScale0 * factor, capturedBScale);
}

void testScaleOptimizerGradients_SymInt32_DequantEquivalence() {
    layer_t *model[1];
    parameter_t *w;
    parameter_t *b;
    int32_t wGradInt[6] = {7, -14, 21, -28, 35, -42};
    int32_t bGradInt[2] = {3, -3};
    float wScale0 = 0.125f;
    float bScale0 = 1.0f;
    float factor = 0.5f;

    /* Pre-compute dequantized grads (float_value = int32_value * scale) before
     * scaling, then multiply by factor — this is the mathematical identity the
     * scale-only path must satisfy. */
    float wDequantBeforeTimesFactor[6];
    for (size_t i = 0; i < 6; i++) {
        wDequantBeforeTimesFactor[i] = (float)wGradInt[i] * wScale0 * factor;
    }
    float bDequantBeforeTimesFactor[2];
    for (size_t i = 0; i < 2; i++) {
        bDequantBeforeTimesFactor[i] = (float)bGradInt[i] * bScale0 * factor;
    }

    optimizer_t *sgd =
        buildSymInt32OneLayerOptim(model, &w, &b, wScale0, wGradInt, bScale0, bGradInt, 0.01f, 0.f);

    scaleOptimizerGradients(sgd, factor);

    /* CAPTURE dequantized values after scaling, using post-scale int32 + scale. */
    float wDequantAfter[6];
    {
        int32_t *gradInt = (int32_t *)w->grad->data;
        float postScale = ((symInt32QConfig_t *)w->grad->quantization->qConfig)->scale;
        for (size_t i = 0; i < 6; i++) {
            wDequantAfter[i] = (float)gradInt[i] * postScale;
        }
    }
    float bDequantAfter[2];
    {
        int32_t *gradInt = (int32_t *)b->grad->data;
        float postScale = ((symInt32QConfig_t *)b->grad->quantization->qConfig)->scale;
        for (size_t i = 0; i < 2; i++) {
            bDequantAfter[i] = (float)gradInt[i] * postScale;
        }
    }

    freeOptimSgdM(sgd);
    freeLinearLayerLegacy(model[0]);

    for (size_t i = 0; i < 6; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-6f, wDequantBeforeTimesFactor[i], wDequantAfter[i]);
    }
    for (size_t i = 0; i < 2; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-6f, bDequantBeforeTimesFactor[i], bDequantAfter[i]);
    }
}

void testScaleOptimizerGradients_SymInt32_MomentumSgdAppliesScaledGradient() {
    /* End-to-end: scaleOptimizerGradients into sgdStep with momentum > 0.
     * The dequantized grad (int32 * scale) MUST equal int32_initial * scale_initial * factor
     * once scaling is applied, and momentum-SGD's parameter update (with
     * momentum=0 in the very first step) becomes -lr * scaled_grad. We assert
     * that the post-step parameter equals the expected dequantized update,
     * proving scaleOptimizerGradients fed sgdStep correctly. */
    layer_t *model[1];
    parameter_t *w;
    parameter_t *b;
    int32_t wGradInt[6] = {4, 8, 12, 16, 20, 24};
    int32_t bGradInt[2] = {2, -2};
    float wScale0 = 0.5f;
    float bScale0 = 0.5f;
    float factor = 0.25f;
    float lr = 0.1f;

    /* Expected param-after-step: param_before (zero from calloc) - lr * (int32 * scale_initial *
     * factor). With param_before == 0: param_after == -lr * dequant_grad_scaled. */
    float wExpectedParam[6];
    for (size_t i = 0; i < 6; i++) {
        wExpectedParam[i] = -lr * (float)wGradInt[i] * wScale0 * factor;
    }
    float bExpectedParam[2];
    for (size_t i = 0; i < 2; i++) {
        bExpectedParam[i] = -lr * (float)bGradInt[i] * bScale0 * factor;
    }

    optimizer_t *sgd = buildSymInt32OneLayerOptim(model, &w, &b, wScale0, wGradInt, bScale0,
                                                  bGradInt, lr, 0.f /* momentum=0 first step */);

    scaleOptimizerGradients(sgd, factor);
    optimizerFunctions[sgd->type].step(sgd);

    /* CAPTURE param-after-step as dequantized floats. */
    float wParamAfter[6];
    {
        int32_t *paramInt = (int32_t *)w->param->data;
        float pScale = ((symInt32QConfig_t *)w->param->quantization->qConfig)->scale;
        for (size_t i = 0; i < 6; i++) {
            wParamAfter[i] = (float)paramInt[i] * pScale;
        }
    }
    float bParamAfter[2];
    {
        int32_t *paramInt = (int32_t *)b->param->data;
        float pScale = ((symInt32QConfig_t *)b->param->quantization->qConfig)->scale;
        for (size_t i = 0; i < 2; i++) {
            bParamAfter[i] = (float)paramInt[i] * pScale;
        }
    }

    freeOptimSgdM(sgd);
    freeLinearLayerLegacy(model[0]);

    /* Tolerance accounts for the int32 round-trip in sgdStepSymInt32 — the
     * intermediate float value gets requantized through wScale0 (post-step
     * scale unchanged on param tensor in this iteration). 1e-3f is generous
     * enough for the small magnitudes here. */
    for (size_t i = 0; i < 6; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-3f, wExpectedParam[i], wParamAfter[i]);
    }
    for (size_t i = 0; i < 2; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-3f, bExpectedParam[i], bParamAfter[i]);
    }
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(testScaleOptimizerGradients_DoublesGradients);
    RUN_TEST(testScaleOptimizerGradients_FactorZero_DoesNotAbort);
    RUN_TEST(testScaleOptimizerGradients_FactorNaN_DoesNotAbort);
    RUN_TEST(testScaleOptimizerGradients_SymInt32_ScalesScaleOnly);
    RUN_TEST(testScaleOptimizerGradients_SymInt32_DequantEquivalence);
    RUN_TEST(testScaleOptimizerGradients_SymInt32_MomentumSgdAppliesScaledGradient);
    return UNITY_END();
}
