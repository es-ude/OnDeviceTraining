#define SOURCE_FILE "UNIT_TEST_LAYERNORM_INTEGRATION"

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

#include "CalculateGradsSequential.h"
#include "InferenceApi.h"
#include "Layer.h"
#include "LayerNorm.h"
#include "LayerNormApi.h"
#include "LayerQuant.h"
#include "Linear.h"
#include "LinearApi.h"
#include "LossFunction.h"
#include "Optimizer.h"
#include "OptimizerApi.h"
#include "Quantization.h"
#include "QuantizationApi.h"
#include "SgdApi.h"
#include "StorageApi.h"
#include "Tensor.h"
#include "TensorApi.h"
#include "unity.h"

void setUp(void) {}
void tearDown(void) {}

static tensor_t *build2DFloat(size_t b, size_t f, const float *data, size_t n) {
    size_t *dims = reserveMemory(2 * sizeof(size_t));
    dims[0] = b;
    dims[1] = f;
    size_t *order = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, order);
    shape_t *shape = reserveMemory(sizeof(shape_t));
    setShape(shape, dims, 2, order);
    tensor_t *t = initTensor(shape, quantizationInitFloat(), NULL);
    tensorFillFromFloatBuffer(t, (float *)data, n);
    return t;
}

static tensor_t *build2DSym(size_t b, size_t f, const float *data, size_t n) {
    size_t *dims = reserveMemory(2 * sizeof(size_t));
    dims[0] = b;
    dims[1] = f;
    size_t *order = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, order);
    shape_t *shape = reserveMemory(sizeof(shape_t));
    setShape(shape, dims, 2, order);
    tensor_t *t = initTensor(shape, quantizationInitSymInt32(HALF_AWAY), NULL);
    tensorFillFromFloatBuffer(t, (float *)data, n);
    return t;
}

/* freeOptim cascades into freeParameter for every registered parameter_t
 * (Linear weights/bias, LayerNorm gamma/beta) — the SAME objects the layers
 * own. The factory frees (freeLinearLayer / freeLayerNormLayer) would call
 * freeParameter again, so after freeOptim the layers must be torn down
 * shell-only. Pattern from UnitTestSgd.c
 * (testSgdZeroGradOnSymInt32GradZeroesMantissasAndResetsScale). Both layers
 * here come from the Borrowing factories (ownsQuantizations=false), so the
 * shared math quantization is freed by the caller, not by the shells. */
static void freeLinearLayerShell(layer_t *layer) {
    freeReservedMemory(layer->config->linear);
    freeReservedMemory(layer->config);
    freeReservedMemory(layer);
}

static void freeLayerNormLayerShell(layer_t *layer) {
    freeReservedMemory(layer->config->layerNorm->normalizedShape);
    freeReservedMemory(layer->config->layerNorm);
    freeReservedMemory(layer->config);
    freeReservedMemory(layer);
}

/* One FLOAT32 training step through Linear(4→3) → LayerNorm([3]) → Linear(3→2):
 * forward + MSE loss + backward + SGD-M step. Verifies the loss is finite and
 * that LayerNorm's gamma AND beta actually trained (changed from their 1/0
 * factory init after the optimizer step). */
void testLinearLayerNormLinearOneTrainingStep(void) {
    quantization_t *q = quantizationInitFloat();
    layerQuant_t lq;
    layerQuantInitUniform(&lq, q);

    layer_t *linear0 =
        linearLayerInit(&(linearInit_t){.inFeatures = 4, .outFeatures = 3, .bias = BIAS_TRUE}, &lq);
    layer_t *norm = layerNormLayerInit(
        &(layerNormInit_t){.normalizedShape = (size_t[]){3}, .numNormDims = 1}, &lq);
    layer_t *linear1 =
        linearLayerInit(&(linearInit_t){.inFeatures = 3, .outFeatures = 2, .bias = BIAS_TRUE}, &lq);
    layer_t *model[3] = {linear0, norm, linear1};

    tensor_t *input =
        build2DFloat(2, 4, (float[]){0.5f, -1.f, 2.f, 1.5f, -0.5f, 1.f, -2.f, 0.25f}, 8);
    tensor_t *label = build2DFloat(2, 2, (float[]){1.f, -1.f, 0.5f, 2.f}, 4);

    float gammaBefore = ((float *)norm->config->layerNorm->gamma->param->data)[0];
    float betaBefore = ((float *)norm->config->layerNorm->beta->param->data)[0];

    quantization_t *momentumQ = quantizationInitFloat();
    optimizer_t *optim =
        sgdMCreateOptim(0.1f, 0.9f, 0.0f, model, 3, momentumQ,
                        (arithmetic_t){.type = ARITH_FLOAT32, .roundingMode = HALF_AWAY});

    trainingStats_t *stats = calculateGradsSequential(model, 3, defaultLossConfig(MSE),
                                                      REDUCTION_MEAN, input, label, NULL);
    sgdStepM(optim);

    bool statsNotNull = (stats != NULL);
    float loss = stats ? stats->loss : NAN;
    bool lossFinite = isfinite(loss);
    float gammaAfter = ((float *)norm->config->layerNorm->gamma->param->data)[0];
    float betaAfter = ((float *)norm->config->layerNorm->beta->param->data)[0];

    /* Post-step forward through the InferenceApi path: the training step above
     * routes layer outputs through CalculateGradsSequential's own switch, so
     * this is the only coverage of initBufferOutput's LAYERNORM case (mirrors
     * UnitTestDropoutIntegration). */
    tensor_t *predicted = inference(model, 3, input);
    bool predictedNotNull = (predicted != NULL);
    bool predictedFinite = predictedNotNull;
    for (size_t i = 0; predictedNotNull && i < 4; i++) {
        predictedFinite = predictedFinite && isfinite(((float *)predicted->data)[i]);
    }

    freeTensor(predicted);
    freeTrainingStats(stats);
    freeTensor(label);
    freeTensor(input);
    freeOptim(optim); /* frees w0/b0, gamma/beta, w1/b1 parameter_t */
    freeLinearLayerShell(linear1);
    freeLayerNormLayerShell(norm);
    freeLinearLayerShell(linear0);
    freeQuantization(momentumQ);
    freeQuantization(q);

    TEST_ASSERT_TRUE(statsNotNull);
    TEST_ASSERT_TRUE_MESSAGE(lossFinite, "loss must be finite after one training step");
    TEST_ASSERT_TRUE_MESSAGE(gammaAfter != gammaBefore,
                             "LayerNorm gamma[0] must change after the SGD-M step");
    TEST_ASSERT_TRUE_MESSAGE(betaAfter != betaBefore,
                             "LayerNorm beta[0] must change after the SGD-M step");
    TEST_ASSERT_TRUE(predictedNotNull);
    TEST_ASSERT_TRUE_MESSAGE(predictedFinite, "inference output must be finite");
}

/* Full-SYM single-LayerNorm training step through the public training APIs:
 * calculateGradsSequential (MSE — the only SYM-capable loss backward) + SGD-M
 * step on SYM gamma/beta with SYM grads. Single-layer model: LayerNorm sits at
 * the loss boundary, so there's no downstream layer to feed regardless of
 * whether its forward wire is producer-restored (PR1b.2) or not — a chained
 * Quantization layer was never required here even before that migration.
 * Test behavior unchanged: validates the full SYM end-to-end path.
 * A FLOAT32 twin trained on the same data is the reference; 5e-3 absolute
 * dequant tolerance absorbs input quantization + strategy-A grad noise
 * (PR-0 E2E precedent tolerance). */
void testLayerNormSymInt32SingleLayerTrainingStep(void) {
    const float inputVals[8] = {1.f, 2.f, 3.f, 4.f, 10.f, 20.f, 30.f, 40.f};
    const float labelVals[8] = {0.5f, -1.f, 2.f, 1.5f, -0.5f, 1.f, -2.f, 0.25f};
    size_t normShape[] = {4};

    /* ---- SYM model (under test) ---- */
    /* PR1c: default grads are FLOAT32; SYM via knob. weightGradStorage/
     * biasGradStorage opt gamma/beta grads back into SYM_INT32 so this test
     * keeps exercising the full SYM training step it's named for. */
    layerNormInit_t initSym = {.normalizedShape = normShape, .numNormDims = 1, .eps = 1e-5f};
    quantization_t *symQ = quantizationInitSymInt32(HALF_AWAY);
    layerQuant_t lqSym = {.forwardMath = arithmeticFromQuantization(symQ),
                          .propLossMath = arithmeticFromQuantization(symQ),
                          .outputQ = symQ,
                          .propLossQ = symQ,
                          .weightStorage = symQ,
                          .biasStorage = symQ,
                          .weightGradStorage = symQ,
                          .biasGradStorage = symQ,
                          /* LayerNorm's true hardcode is DYNAMIC_RESCALE for
                           * BOTH gamma and beta -- see LayerNorm.c
                           * initLayerNormConfig. */
                          .weightGradAccMode = OUT_ACC_DYNAMIC_RESCALE,
                          .biasGradAccMode = OUT_ACC_DYNAMIC_RESCALE};
    layer_t *lnSym = layerNormLayerInit(&initSym, &lqSym);
    layer_t *modelSym[1] = {lnSym};

    tensor_t *inSym = build2DSym(2, 4, inputVals, 8);
    tensor_t *labelSym = build2DSym(2, 4, labelVals, 8);

    bool gradSymDtype = (lnSym->config->layerNorm->gamma->grad->quantization->type == SYM_INT32);

    quantization_t *momentumQSym = quantizationInitFloat();
    optimizer_t *optimSym =
        sgdMCreateOptim(0.1f, 0.0f, 0.0f, modelSym, 1, momentumQSym,
                        (arithmetic_t){.type = ARITH_FLOAT32, .roundingMode = HALF_AWAY});
    trainingStats_t *statsSym = calculateGradsSequential(modelSym, 1, defaultLossConfig(MSE),
                                                         REDUCTION_MEAN, inSym, labelSym, NULL);
    sgdStepM(optimSym);

    float gammaSym[4], betaSym[4];
    {
        layerNormConfig_t *cfg = lnSym->config->layerNorm;
        float gs = ((symInt32QConfig_t *)cfg->gamma->param->quantization->qConfig)->scale;
        float bs = ((symInt32QConfig_t *)cfg->beta->param->quantization->qConfig)->scale;
        for (size_t i = 0; i < 4; i++) {
            gammaSym[i] = (float)((int32_t *)cfg->gamma->param->data)[i] * gs;
            betaSym[i] = (float)((int32_t *)cfg->beta->param->data)[i] * bs;
        }
    }
    bool symStatsNotNull = (statsSym != NULL);
    float symLoss = statsSym ? statsSym->loss : NAN;

    /* ---- FLOAT32 twin (reference) ---- */
    layerNormInit_t initF = {.normalizedShape = normShape, .numNormDims = 1, .eps = 1e-5f};
    quantization_t *fQ = quantizationInitFloat();
    layerQuant_t lqF;
    layerQuantInitUniform(&lqF, fQ);
    layer_t *lnF = layerNormLayerInit(&initF, &lqF);
    layer_t *modelF[1] = {lnF};

    tensor_t *inF = build2DFloat(2, 4, inputVals, 8);
    tensor_t *labelF = build2DFloat(2, 4, labelVals, 8);

    quantization_t *momentumQF = quantizationInitFloat();
    optimizer_t *optimF =
        sgdMCreateOptim(0.1f, 0.0f, 0.0f, modelF, 1, momentumQF,
                        (arithmetic_t){.type = ARITH_FLOAT32, .roundingMode = HALF_AWAY});
    trainingStats_t *statsF = calculateGradsSequential(modelF, 1, defaultLossConfig(MSE),
                                                       REDUCTION_MEAN, inF, labelF, NULL);
    sgdStepM(optimF);

    float gammaF[4], betaF[4];
    for (size_t i = 0; i < 4; i++) {
        gammaF[i] = ((float *)lnF->config->layerNorm->gamma->param->data)[i];
        betaF[i] = ((float *)lnF->config->layerNorm->beta->param->data)[i];
    }

    freeTrainingStats(statsF);
    freeTrainingStats(statsSym);
    freeTensor(labelF);
    freeTensor(inF);
    freeTensor(labelSym);
    freeTensor(inSym);
    freeOptim(optimF); /* frees gamma/beta parameter_t of the float twin */
    freeOptim(optimSym);
    freeLayerNormLayerShell(lnF);
    freeLayerNormLayerShell(lnSym);
    freeQuantization(momentumQF);
    freeQuantization(momentumQSym);
    freeQuantization(fQ);
    freeQuantization(symQ);

    TEST_ASSERT_TRUE(symStatsNotNull);
    TEST_ASSERT_TRUE_MESSAGE(isfinite(symLoss), "SYM loss must be finite");
    TEST_ASSERT_TRUE_MESSAGE(
        gradSymDtype, "explicit weightGradStorage/biasGradStorage knob must yield SYM_INT32 grads");
    for (size_t i = 0; i < 4; i++) {
        /* gamma init 1.0 / beta init 0.0: both twins must move, and the SYM
         * twin must land near the float twin. */
        TEST_ASSERT_TRUE_MESSAGE(fabsf(gammaF[i] - 1.0f) > 1e-6f,
                                 "reference gamma must actually train");
        TEST_ASSERT_FLOAT_WITHIN(5e-3f, gammaF[i], gammaSym[i]);
        TEST_ASSERT_FLOAT_WITHIN(5e-3f, betaF[i], betaSym[i]);
    }
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(testLinearLayerNormLinearOneTrainingStep);
    RUN_TEST(testLayerNormSymInt32SingleLayerTrainingStep);
    return UNITY_END();
}
