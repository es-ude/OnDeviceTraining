#define SOURCE_FILE "UNIT_TEST_OPTIMIZER_SCALING"

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "ArithmeticType.h"
#include "BorrowedLayer.h"
#include "DeathTest.h"
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
#include "TensorConversion.h"
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
    layer_t *linear = buildBorrowedLinearLayer(w, b, &testQ);
    modelOut[0] = linear;
    *wOut = w;
    *bOut = b;

    /* SGD optimizer wraps both parameters via the standard helper. Momentum
     * config is a transient template -- sgdMCreateOptim clones it per state
     * via getQLike, so it's safe to free right after the call. */
    quantization_t *momentumQ = quantizationInitFloat();
    optimizer_t *optim =
        sgdMCreateOptim(0.01f, 0.f, 0.f, modelOut, 1, momentumQ,
                        (arithmetic_t){.type = ARITH_FLOAT32, .roundingMode = HALF_AWAY});
    freeQuantization(momentumQ);
    return optim;
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

    /* FREE. freeOptim cascades to both parameters. */
    freeOptim(sgd);
    freeLinearLayerShellOnly(model[0]);

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

    freeOptim(sgd);
    freeLinearLayerShellOnly(model[0]);

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

    freeOptim(sgd);
    freeLinearLayerShellOnly(model[0]);

    /* NaN != NaN by IEEE 754. */
    TEST_ASSERT_TRUE(captured != captured);
}

/* SYM_INT32 builder mirrors buildOneLayerOptimWithGrads but pins both param and
 * grad tensors to SYM_INT32 quantization. The grad tensor's int32 storage and
 * scale are written directly (NOT via tensorFillFromFloatBuffer, which would
 * route through convertTensor and recompute scale from the float source).
 * The param data stays at default-zero, which matches the scale=1.0 default
 * from initSymInt32QConfig — the executeOp funnel round-trips it through float
 * and back, but for the scaling assertions below the param values are irrelevant. */
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
    layer_t *linear = buildBorrowedLinearLayer(w, b, layerQ);
    modelOut[0] = linear;
    *wOut = w;
    *bOut = b;

    quantization_t *momentumQ = quantizationInitFloat();
    optimizer_t *optim =
        sgdMCreateOptim(lr, momentum, 0.f, modelOut, 1, momentumQ,
                        (arithmetic_t){.type = ARITH_FLOAT32, .roundingMode = HALF_AWAY});
    freeQuantization(momentumQ);
    return optim;
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

    freeOptim(sgd);
    freeLinearLayerShellOnly(model[0]);

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

    freeOptim(sgd);
    freeLinearLayerShellOnly(model[0]);

    for (size_t i = 0; i < 6; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-6f, wDequantBeforeTimesFactor[i], wDequantAfter[i]);
    }
    for (size_t i = 0; i < 2; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-6f, bDequantBeforeTimesFactor[i], bDequantAfter[i]);
    }
}

void testScaleOptimizerGradients_SymInt32_MomentumSgdAppliesScaledGradient() {
    /* End-to-end: scaleOptimizerGradients into sgdStepM with momentum > 0.
     * The dequantized grad (int32 * scale) MUST equal int32_initial * scale_initial * factor
     * once scaling is applied, and momentum-SGD's parameter update (with
     * momentum=0 in the very first step) becomes -lr * scaled_grad. We assert
     * that the post-step parameter equals the expected dequantized update,
     * proving scaleOptimizerGradients fed sgdStepM correctly. */
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

    freeOptim(sgd);
    freeLinearLayerShellOnly(model[0]);

    /* Tolerance accounts for the int32 round-trip in the executeOp funnel — the
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

/* Dequantizes `grad` into a caller-owned float buffer via the same
 * convertTensor read path the optimizer's grad-dtype-generic step reads use
 * (PR3, Sgd.c). Works for any admitted grad dtype whose conversionMatrix cell
 * to FLOAT32 exists (SYM_INT32/SYM/ASYM all do). */
static void dequantGradToFloat(tensor_t *grad, float *out, size_t n) {
    tensor_t floatView;
    quantization_t floatQ;
    initFloat32Quantization(&floatQ);
    uint8_t floatBytes[n * sizeof(float)];
    setTensorValuesForConversion(floatBytes, &floatQ, grad, &floatView);
    convertTensor(grad, &floatView);
    memcpy(out, floatView.data, n * sizeof(float));
}

/* SYM builder mirrors buildSymInt32OneLayerOptim but pins the grad tensors to
 * packed sub-byte SYM (param stays FLOAT32 — packed grad storage is
 * independent of param dtype, #261). Grad values are seeded via
 * tensorFillFromFloatBuffer, which routes through convertFloatTensorToSymTensor
 * (fresh absmax-derived scale + fit-guarded pack) — deterministic, and the
 * exact codes/scale don't matter to these tests (they only check the fold's
 * effect: bytes untouched, scale scaled, dequant equivalence). */
static optimizer_t *buildSymOneLayerOptim(layer_t **modelOut, parameter_t **wOut,
                                          parameter_t **bOut, const float *wInitialGradFloat,
                                          const float *bInitialGradFloat, uint8_t qBits, float lr,
                                          float momentum) {
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
        tensorFillFromFloatBuffer(wParam, (float[]){0.f, 0.f, 0.f, 0.f, 0.f, 0.f}, 6);
    }
    tensor_t *wGrad = gradInitSym(wParam, qBits, HALF_AWAY, NULL);
    tensorFillFromFloatBuffer(wGrad, wInitialGradFloat, 6);
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
    tensor_t *bGrad = gradInitSym(bParam, qBits, HALF_AWAY, NULL);
    tensorFillFromFloatBuffer(bGrad, bInitialGradFloat, 2);
    parameter_t *b = parameterInit(bParam, bGrad);

    quantization_t *layerQ = quantizationInitFloat();
    layer_t *linear = buildBorrowedLinearLayer(w, b, layerQ);
    modelOut[0] = linear;
    *wOut = w;
    *bOut = b;

    quantization_t *momentumQ = quantizationInitFloat();
    optimizer_t *optim =
        sgdMCreateOptim(lr, momentum, 0.f, modelOut, 1, momentumQ,
                        (arithmetic_t){.type = ARITH_FLOAT32, .roundingMode = HALF_AWAY});
    freeQuantization(momentumQ);
    return optim;
}

void testScaleOptimizerGradients_Sym_ScalesScaleOnly(void) {
    layer_t *model[1];
    parameter_t *w;
    parameter_t *b;
    float wGradInit[6] = {1.f, -2.f, 3.f, -4.f, 5.f, -6.f};
    float bGradInit[2] = {0.5f, -0.5f};
    uint8_t qBits = 8;
    float factor = 0.25f;

    optimizer_t *sgd =
        buildSymOneLayerOptim(model, &w, &b, wGradInit, bGradInit, qBits, 0.01f, 0.f);

    size_t wBytes = calcNumberOfBytesForData(w->grad->quantization, 6);
    size_t bBytes = calcNumberOfBytesForData(b->grad->quantization, 2);
    uint8_t wBytesBefore[wBytes];
    uint8_t bBytesBefore[bBytes];
    memcpy(wBytesBefore, w->grad->data, wBytes);
    memcpy(bBytesBefore, b->grad->data, bBytes);
    float wScaleBefore = ((symQConfig_t *)w->grad->quantization->qConfig)->scales[0];
    float bScaleBefore = ((symQConfig_t *)b->grad->quantization->qConfig)->scales[0];

    scaleOptimizerGradients(sgd, factor);

    /* CAPTURE before frees. */
    uint8_t wBytesAfter[wBytes];
    uint8_t bBytesAfter[bBytes];
    memcpy(wBytesAfter, w->grad->data, wBytes);
    memcpy(bBytesAfter, b->grad->data, bBytes);
    float wScaleAfter = ((symQConfig_t *)w->grad->quantization->qConfig)->scales[0];
    float bScaleAfter = ((symQConfig_t *)b->grad->quantization->qConfig)->scales[0];

    freeOptim(sgd);
    freeLinearLayerShellOnly(model[0]);

    /* packed codes byte-for-byte unchanged. */
    TEST_ASSERT_EQUAL_UINT8_ARRAY(wBytesBefore, wBytesAfter, wBytes);
    TEST_ASSERT_EQUAL_UINT8_ARRAY(bBytesBefore, bBytesAfter, bBytes);
    /* scale absorbed the multiplicative factor. */
    TEST_ASSERT_FLOAT_WITHIN(1e-7f, wScaleBefore * factor, wScaleAfter);
    TEST_ASSERT_FLOAT_WITHIN(1e-7f, bScaleBefore * factor, bScaleAfter);
}

void testScaleOptimizerGradients_Sym_DequantEquivalence(void) {
    layer_t *model[1];
    parameter_t *w;
    parameter_t *b;
    float wGradInit[6] = {1.f, -2.f, 3.f, -4.f, 5.f, -6.f};
    float bGradInit[2] = {0.5f, -0.5f};
    uint8_t qBits = 8;
    float factor = 0.5f;

    optimizer_t *sgd =
        buildSymOneLayerOptim(model, &w, &b, wGradInit, bGradInit, qBits, 0.01f, 0.f);

    float wDequantBefore[6];
    float bDequantBefore[2];
    dequantGradToFloat(w->grad, wDequantBefore, 6);
    dequantGradToFloat(b->grad, bDequantBefore, 2);

    scaleOptimizerGradients(sgd, factor);

    float wDequantAfter[6];
    float bDequantAfter[2];
    dequantGradToFloat(w->grad, wDequantAfter, 6);
    dequantGradToFloat(b->grad, bDequantAfter, 2);

    freeOptim(sgd);
    freeLinearLayerShellOnly(model[0]);

    for (size_t i = 0; i < 6; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-5f, wDequantBefore[i] * factor, wDequantAfter[i]);
    }
    for (size_t i = 0; i < 2; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-5f, bDequantBefore[i] * factor, bDequantAfter[i]);
    }
}

/* ASYM builder mirrors buildSymOneLayerOptim (gradInitAsym instead of
 * gradInitSym) — see that function's comment for the seeding rationale. */
static optimizer_t *buildAsymOneLayerOptim(layer_t **modelOut, parameter_t **wOut,
                                           parameter_t **bOut, const float *wInitialGradFloat,
                                           const float *bInitialGradFloat, uint8_t qBits, float lr,
                                           float momentum) {
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
        tensorFillFromFloatBuffer(wParam, (float[]){0.f, 0.f, 0.f, 0.f, 0.f, 0.f}, 6);
    }
    tensor_t *wGrad = gradInitAsym(wParam, qBits, HALF_AWAY, NULL);
    tensorFillFromFloatBuffer(wGrad, wInitialGradFloat, 6);
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
    tensor_t *bGrad = gradInitAsym(bParam, qBits, HALF_AWAY, NULL);
    tensorFillFromFloatBuffer(bGrad, bInitialGradFloat, 2);
    parameter_t *b = parameterInit(bParam, bGrad);

    quantization_t *layerQ = quantizationInitFloat();
    layer_t *linear = buildBorrowedLinearLayer(w, b, layerQ);
    modelOut[0] = linear;
    *wOut = w;
    *bOut = b;

    quantization_t *momentumQ = quantizationInitFloat();
    optimizer_t *optim =
        sgdMCreateOptim(lr, momentum, 0.f, modelOut, 1, momentumQ,
                        (arithmetic_t){.type = ARITH_FLOAT32, .roundingMode = HALF_AWAY});
    freeQuantization(momentumQ);
    return optim;
}

void testScaleOptimizerGradients_Asym_ScalesScaleOnly(void) {
    layer_t *model[1];
    parameter_t *w;
    parameter_t *b;
    float wGradInit[6] = {1.f, -2.f, 3.f, -4.f, 5.f, -6.f};
    float bGradInit[2] = {0.5f, -0.5f};
    uint8_t qBits = 8;
    float factor = 0.25f;

    optimizer_t *sgd =
        buildAsymOneLayerOptim(model, &w, &b, wGradInit, bGradInit, qBits, 0.01f, 0.f);

    size_t wBytes = calcNumberOfBytesForData(w->grad->quantization, 6);
    size_t bBytes = calcNumberOfBytesForData(b->grad->quantization, 2);
    uint8_t wBytesBefore[wBytes];
    uint8_t bBytesBefore[bBytes];
    memcpy(wBytesBefore, w->grad->data, wBytes);
    memcpy(bBytesBefore, b->grad->data, bBytes);
    asymQConfig_t *wQ = w->grad->quantization->qConfig;
    asymQConfig_t *bQ = b->grad->quantization->qConfig;
    float wScaleBefore = wQ->scales[0];
    float bScaleBefore = bQ->scales[0];
    uint16_t wZeroPointBefore = wQ->zeroPoints[0];
    uint16_t bZeroPointBefore = bQ->zeroPoints[0];

    scaleOptimizerGradients(sgd, factor);

    /* CAPTURE before frees. */
    uint8_t wBytesAfter[wBytes];
    uint8_t bBytesAfter[bBytes];
    memcpy(wBytesAfter, w->grad->data, wBytes);
    memcpy(bBytesAfter, b->grad->data, bBytes);
    float wScaleAfter = wQ->scales[0];
    float bScaleAfter = bQ->scales[0];
    uint16_t wZeroPointAfter = wQ->zeroPoints[0];
    uint16_t bZeroPointAfter = bQ->zeroPoints[0];

    freeOptim(sgd);
    freeLinearLayerShellOnly(model[0]);

    /* packed codes byte-for-byte unchanged. */
    TEST_ASSERT_EQUAL_UINT8_ARRAY(wBytesBefore, wBytesAfter, wBytes);
    TEST_ASSERT_EQUAL_UINT8_ARRAY(bBytesBefore, bBytesAfter, bBytes);
    /* scale absorbed the multiplicative factor; zeroPoint is untouched (an
     * additive offset on the code axis, not part of the multiplicative fold). */
    TEST_ASSERT_FLOAT_WITHIN(1e-7f, wScaleBefore * factor, wScaleAfter);
    TEST_ASSERT_FLOAT_WITHIN(1e-7f, bScaleBefore * factor, bScaleAfter);
    TEST_ASSERT_EQUAL_INT32(wZeroPointBefore, wZeroPointAfter);
    TEST_ASSERT_EQUAL_INT32(bZeroPointBefore, bZeroPointAfter);
}

void testScaleOptimizerGradients_Asym_DequantEquivalence(void) {
    layer_t *model[1];
    parameter_t *w;
    parameter_t *b;
    float wGradInit[6] = {1.f, -2.f, 3.f, -4.f, 5.f, -6.f};
    float bGradInit[2] = {0.5f, -0.5f};
    uint8_t qBits = 8;
    float factor = 0.5f;

    optimizer_t *sgd =
        buildAsymOneLayerOptim(model, &w, &b, wGradInit, bGradInit, qBits, 0.01f, 0.f);

    float wDequantBefore[6];
    float bDequantBefore[2];
    dequantGradToFloat(w->grad, wDequantBefore, 6);
    dequantGradToFloat(b->grad, bDequantBefore, 2);

    scaleOptimizerGradients(sgd, factor);

    float wDequantAfter[6];
    float bDequantAfter[2];
    dequantGradToFloat(w->grad, wDequantAfter, 6);
    dequantGradToFloat(b->grad, bDequantAfter, 2);

    freeOptim(sgd);
    freeLinearLayerShellOnly(model[0]);

    for (size_t i = 0; i < 6; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-5f, wDequantBefore[i] * factor, wDequantAfter[i]);
    }
    for (size_t i = 0; i < 2; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-5f, bDequantBefore[i] * factor, bDequantAfter[i]);
    }
}

/* Group-quant PR3 Task 4: defensive belt-and-suspenders fail-fast on the
 * SYM grad arm. gradInit's own carrier gate already rejects a grouped SYM
 * template before a grad tensor is ever built through the sanctioned API
 * (grads are per-tensor unconditionally, symQConfig_t's carrier-gate doc
 * comment, #300 axis) -- so this fixture bypasses gradInit entirely and
 * hand-builds a grouped SYM grad tensor directly (initTensor +
 * quantizationInitSymGrouped), exactly the kind of hand-assembled optimizer
 * this file's other builders (buildSymOneLayerOptim etc.) and the SGD/AdamW
 * hand-built optimizer_t tests already exercise elsewhere in this tree.
 * Folding `factor` into scales[0] alone (the per-tensor SYM arm's existing
 * code) would silently scale ONLY group 0 and leave every other group
 * untouched -- scaleOptimizerGradients must fail fast instead. */
void testScaleOptimizerGradientsRejectsGroupedSymGrad(void) {
    ASSERT_EXITS_WITH_FAILURE({
        size_t *pDims = reserveMemory(1 * sizeof(size_t));
        pDims[0] = 6;
        size_t *pOrder = reserveMemory(1 * sizeof(size_t));
        setOrderOfDimsForNewTensor(1, pOrder);
        shape_t *pShape = reserveMemory(sizeof(shape_t));
        setShape(pShape, pDims, 1, pOrder);
        tensor_t *wParam = initTensor(pShape, quantizationInitFloat(), NULL);
        tensorFillFromFloatBuffer(wParam, (float[]){0.f, 0.f, 0.f, 0.f, 0.f, 0.f}, 6);

        size_t *gDims = reserveMemory(1 * sizeof(size_t));
        gDims[0] = 6;
        size_t *gOrder = reserveMemory(1 * sizeof(size_t));
        setOrderOfDimsForNewTensor(1, gOrder);
        shape_t *gShape = reserveMemory(sizeof(shape_t));
        setShape(gShape, gDims, 1, gOrder);
        tensor_t *wGrad = initTensor(gShape, quantizationInitSymGrouped(8, HALF_AWAY, 2, 3), NULL);

        parameter_t *w = parameterInit(wParam, wGrad);
        parameter_t *parArr[1] = {w};
        /* Extra parens (not just the usual designated-initializer literal):
         * the preprocessor's macro-argument comma scan only tracks real
         * parentheses, not braces, so a bare `{a, b}` initializer here would
         * split ASSERT_EXITS_WITH_FAILURE's single block argument in two --
         * wrapping the whole compound literal in one more `()` pair keeps
         * the comma enclosed. */
        optimizer_t optim = ((optimizer_t){.parameter = parArr, .sizeStates = 1});

        scaleOptimizerGradients(&optim, 2.0f);
    });
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(testScaleOptimizerGradients_DoublesGradients);
    RUN_TEST(testScaleOptimizerGradients_FactorZero_DoesNotAbort);
    RUN_TEST(testScaleOptimizerGradients_FactorNaN_DoesNotAbort);
    RUN_TEST(testScaleOptimizerGradients_SymInt32_ScalesScaleOnly);
    RUN_TEST(testScaleOptimizerGradients_SymInt32_DequantEquivalence);
    RUN_TEST(testScaleOptimizerGradients_SymInt32_MomentumSgdAppliesScaledGradient);
    RUN_TEST(testScaleOptimizerGradients_Sym_ScalesScaleOnly);
    RUN_TEST(testScaleOptimizerGradients_Sym_DequantEquivalence);
    RUN_TEST(testScaleOptimizerGradients_Asym_ScalesScaleOnly);
    RUN_TEST(testScaleOptimizerGradients_Asym_DequantEquivalence);
    RUN_TEST(testScaleOptimizerGradientsRejectsGroupedSymGrad);
    return UNITY_END();
}
