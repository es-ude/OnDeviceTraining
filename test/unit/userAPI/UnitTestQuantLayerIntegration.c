#define SOURCE_FILE "UNIT_TEST_QUANT_LAYER_INTEGRATION"

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
#include "ModelValidationApi.h"
#include "Optimizer.h"
#include "QuantLayerApi.h"
#include "Quantization.h"
#include "QuantizationApi.h"
#include "QuantizationLayer.h"
#include "SgdApi.h"
#include "StorageApi.h"
#include "Tensor.h"
#include "TensorApi.h"
#include "unity.h"

void setUp(void) {}
void tearDown(void) {}

/* Deterministic fixtures shared by the micro tests (D.4-D.6) and chain test (D.9). */
static const float W0_VALS[12] = {0.5f, -0.3f, 0.8f, 0.1f,  -0.7f, 0.4f,
                                  0.2f, -0.9f, 0.6f, -0.2f, 0.3f,  0.7f};
static const float B0_VALS[3] = {0.1f, -0.2f, 0.3f};
static const float INPUT_VALS[8] = {1.f, 2.f, 3.f, 4.f, 10.f, 20.f, 30.f, 40.f};

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

/* Trainable Linear with manually built parameters: the new-factory KAIMING
 * init requires FLOAT32 weight storage (LinearApi.c guard), so SYM Linear
 * layers use the Legacy factory + explicit parameter_t (UnitTestLinear SYM
 * precedent), extended with grad tensors for sgdMCreateOptim / sgdStepM. */
static layer_t *buildTrainableLinear(size_t inF, size_t outF, const float *w, const float *b,
                                     quantization_t *mathQ, bool sym) {
    size_t *wDims = reserveMemory(2 * sizeof(size_t));
    wDims[0] = outF;
    wDims[1] = inF;
    size_t *wOrder = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, wOrder);
    shape_t *wShape = reserveMemory(sizeof(shape_t));
    setShape(wShape, wDims, 2, wOrder);
    tensor_t *wParam = initTensor(
        wShape, sym ? quantizationInitSymInt32(HALF_AWAY) : quantizationInitFloat(), NULL);
    tensorFillFromFloatBuffer(wParam, (float *)w, outF * inF);
    parameter_t *weights = parameterInit(wParam, gradInit(wParam, mathQ, NULL));

    size_t *bDims = reserveMemory(1 * sizeof(size_t));
    bDims[0] = outF;
    size_t *bOrder = reserveMemory(1 * sizeof(size_t));
    setOrderOfDimsForNewTensor(1, bOrder);
    shape_t *bShape = reserveMemory(sizeof(shape_t));
    setShape(bShape, bDims, 1, bOrder);
    tensor_t *bParam = initTensor(
        bShape, sym ? quantizationInitSymInt32(HALF_AWAY) : quantizationInitFloat(), NULL);
    tensorFillFromFloatBuffer(bParam, (float *)b, outF);
    parameter_t *bias = parameterInit(bParam, gradInit(bParam, mathQ, NULL));

    return linearLayerInitLegacy(weights, bias, mathQ, mathQ, mathQ, mathQ);
}

/* Manual Quant-layer builder: documents the quantizationConfig_t contract
 * without depending on the factory (Task D.7). Borrowed quantizations. */
static layer_t *buildQuantLayer(quantization_t *forwardQ, quantization_t *backwardQ) {
    quantizationConfig_t *cfg = reserveMemory(sizeof(quantizationConfig_t));
    cfg->forwardQ = forwardQ;
    cfg->backwardQ = backwardQ;
    cfg->ownsQuantizations = false;
    layerConfig_t *lc = reserveMemory(sizeof(layerConfig_t));
    lc->quantization = cfg;
    layer_t *layer = reserveMemory(sizeof(layer_t));
    initLayer(layer, QUANTIZATION, lc);
    return layer;
}

static void freeQuantLayerShell(layer_t *layer) {
    freeReservedMemory(layer->config->quantization);
    freeReservedMemory(layer->config);
    freeReservedMemory(layer);
}

/* freeOptimSgdM cascades into freeParameter for every registered parameter_t —
 * the SAME objects the layers reference; layers are torn down shell-only
 * afterwards (UnitTestLayerNormIntegration pattern). */
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

void testSgdMCreateOptimSkipsQuantizationLayer(void) {
    quantization_t *symQ = quantizationInitSymInt32(HALF_AWAY);
    layer_t *linear = buildTrainableLinear(4, 3, W0_VALS, B0_VALS, symQ, true);
    layer_t *quant = buildQuantLayer(symQ, symQ);
    layer_t *model[2] = {linear, quant};

    optimizer_t *optim = sgdMCreateOptim(0.1f, 0.0f, 0.0f, model, 2, SYM_INT32);
    size_t sizeStates = optim->sizeStates;

    freeOptimSgdM(optim); /* frees the Linear weights/bias parameter_t */
    freeQuantLayerShell(quant);
    freeLinearLayerShell(linear);
    freeQuantization(symQ);

    TEST_ASSERT_EQUAL_UINT(2, sizeStates); /* Linear w+b; QUANTIZATION contributes 0 states */
}

void testTrainingStepLinearSymThenQuantRequantsOutputAndFiniteLoss(void) {
    quantization_t *symQ = quantizationInitSymInt32(HALF_AWAY);
    layer_t *linear = buildTrainableLinear(4, 3, W0_VALS, B0_VALS, symQ, true);
    layer_t *quant = buildQuantLayer(symQ, symQ);
    layer_t *model[2] = {linear, quant};

    tensor_t *input = build2DSym(2, 4, INPUT_VALS, 8);
    tensor_t *label = build2DSym(2, 3, (float[]){0.5f, -1.f, 2.f, 1.5f, -0.5f, 1.f}, 6);

    trainingStats_t *stats =
        calculateGradsSequential(model, 2, defaultLossConfig(MSE), REDUCTION_MEAN, input, label);
    bool statsNotNull = (stats != NULL);
    float loss = stats ? stats->loss : NAN;

    int32_t maxAbsMantissa = 0;
    for (size_t i = 0; statsNotNull && i < 6; i++) {
        int32_t m = ((int32_t *)stats->output->data)[i];
        if (m < 0) {
            m = -m;
        }
        if (m > maxAbsMantissa) {
            maxAbsMantissa = m;
        }
    }

    freeTrainingStats(stats);
    freeTensor(label);
    freeTensor(input);
    freeParameter(linear->config->linear->weights); /* no optimizer in this test */
    freeParameter(linear->config->linear->bias);
    freeQuantLayerShell(quant);
    freeLinearLayerShell(linear);
    freeQuantization(symQ);

    TEST_ASSERT_TRUE(statsNotNull);
    TEST_ASSERT_TRUE_MESSAGE(isfinite(loss), "loss must be finite after the training step");
    TEST_ASSERT_TRUE_MESSAGE(maxAbsMantissa > 0, "quant output must be non-trivial");
    TEST_ASSERT_TRUE_MESSAGE(maxAbsMantissa <= 32767,
                             "quant layer output must obey the int16 inter-layer contract");
}

void testInferenceLinearSymThenQuantOutputsInt16RangeMantissas(void) {
    quantization_t *symQ = quantizationInitSymInt32(HALF_AWAY);
    layer_t *linear = buildTrainableLinear(4, 3, W0_VALS, B0_VALS, symQ, true);
    layer_t *quant = buildQuantLayer(symQ, symQ);
    layer_t *model[2] = {linear, quant};

    tensor_t *input = build2DSym(2, 4, INPUT_VALS, 8);
    tensor_t *predicted = inference(model, 2, input);

    bool predictedNotNull = (predicted != NULL);
    bool dequantFinite = true;
    int32_t maxAbsMantissa = 0;
    float scale =
        predictedNotNull ? ((symInt32QConfig_t *)predicted->quantization->qConfig)->scale : NAN;
    for (size_t i = 0; predictedNotNull && i < 6; i++) {
        int32_t m = ((int32_t *)predicted->data)[i];
        dequantFinite = dequantFinite && isfinite((float)m * scale);
        if (m < 0) {
            m = -m;
        }
        if (m > maxAbsMantissa) {
            maxAbsMantissa = m;
        }
    }

    freeTensor(predicted);
    freeTensor(input);
    freeParameter(linear->config->linear->weights);
    freeParameter(linear->config->linear->bias);
    freeQuantLayerShell(quant);
    freeLinearLayerShell(linear);
    freeQuantization(symQ);

    TEST_ASSERT_TRUE(predictedNotNull);
    TEST_ASSERT_TRUE_MESSAGE(dequantFinite, "dequantized inference output must be finite");
    TEST_ASSERT_TRUE_MESSAGE(maxAbsMantissa > 0, "inference output must be non-trivial");
    TEST_ASSERT_TRUE_MESSAGE(maxAbsMantissa <= 32767,
                             "quant layer output must obey the int16 inter-layer contract");
}

static const float W1_VALS[6] = {0.6f, -0.4f, 0.2f, -0.5f, 0.3f, 0.9f};
static const float B1_VALS[2] = {0.05f, -0.1f};
static const float LABEL_VALS[4] = {1.f, -1.f, 0.5f, 2.f};

/* Acceptance criterion of #192 (spec D3): one full-SYM training step through
 * Linear(4->3) -> Quant -> LayerNorm([3]) -> Quant -> Linear(3->2) + MSE +
 * SGD-M, vs a FLOAT32 twin (Linear -> LayerNorm -> Linear; a FLOAT->FLOAT
 * Quant layer is a config error by design, so the twin omits them) trained
 * on identical data.
 *
 * Tolerance: the single-layer SYM precedent is 5e-3 absolute
 * (UnitTestLayerNormIntegration PR-0/PR-3). Here error compounds: weight/
 * input/label quantization (<= absMax/2/32767 ~ 1.5e-5 * absMax per tensor),
 * TWO dynamic requant hops in forward AND backward (<= scale/2 = 0.5 LSB per
 * element per hop), LayerNorm strategy-A grads, and two Linear grad paths
 * consuming requantized dy. Treating each of the ~4 noisy stages as one
 * 5e-3-class error source in series gives 4 * 5e-3 = 2e-2 as the chain
 * bound. */
#define CHAIN_TOL 2e-2f

/* Linear SYM bias grads accumulate at fixed scale (init 1.0); one SGD step
 * moves them in whole LSBs of size lr, so the worst-case deviation from the
 * float twin is lr * 0.5 = 0.1 * 0.5 = 0.05. This is the documented Deutel-
 * style fixed-scale scheme (docs/CONVENTIONS.md "Two accumulation schemes";
 * scheme choice tracked in #218), NOT a Quantization-layer error — all
 * weight/gamma/beta paths through both requant hops stay <= CHAIN_TOL. */
#define BIAS_TOL 5e-2f

static void dequantParam(parameter_t *p, float *out, size_t n, bool sym) {
    if (sym) {
        float scale = ((symInt32QConfig_t *)p->param->quantization->qConfig)->scale;
        for (size_t i = 0; i < n; i++) {
            out[i] = (float)((int32_t *)p->param->data)[i] * scale;
        }
    } else {
        for (size_t i = 0; i < n; i++) {
            out[i] = ((float *)p->param->data)[i];
        }
    }
}

void testFullSymChainTrainingStepMatchesFloatTwin(void) {
    size_t normShape[1] = {3};

    /* ---- SYM model (under test) ---- */
    quantization_t *symQ = quantizationInitSymInt32(HALF_AWAY);
    layerQuant_t lqSym = {
        .forwardMath = symQ, .backwardMath = symQ, .weightStorage = symQ, .biasStorage = symQ};
    layer_t *lin0S = buildTrainableLinear(4, 3, W0_VALS, B0_VALS, symQ, true);
    layer_t *quant1 = quantLayerInit(&lqSym);
    layer_t *lnS = layerNormLayerInit(
        &(layerNormInit_t){.normalizedShape = normShape, .numNormDims = 1, .eps = 1e-5f}, &lqSym);
    layer_t *quant2 = quantLayerInit(&lqSym);
    layer_t *lin1S = buildTrainableLinear(3, 2, W1_VALS, B1_VALS, symQ, true);
    layer_t *modelSym[5] = {lin0S, quant1, lnS, quant2, lin1S};

    /* Setup assert: the chain satisfies the int16 inter-layer contract. */
    TEST_ASSERT_TRUE_MESSAGE(validateModelQuantization(modelSym, 5),
                             "chain with Quant layers must pass the validator");

    tensor_t *inS = build2DSym(2, 4, INPUT_VALS, 8);
    tensor_t *labelS = build2DSym(2, 2, LABEL_VALS, 4);

    optimizer_t *optimS = sgdMCreateOptim(0.1f, 0.0f, 0.0f, modelSym, 5, SYM_INT32);
    trainingStats_t *statsS =
        calculateGradsSequential(modelSym, 5, defaultLossConfig(MSE), REDUCTION_MEAN, inS, labelS);
    sgdStepM(optimS);

    bool symStatsNotNull = (statsS != NULL);
    float symLoss = statsS ? statsS->loss : NAN;

    float w0S[12], b0S[3], gammaS[3], betaS[3], w1S[6], b1S[2];
    dequantParam(lin0S->config->linear->weights, w0S, 12, true);
    dequantParam(lin0S->config->linear->bias, b0S, 3, true);
    dequantParam(lnS->config->layerNorm->gamma, gammaS, 3, true);
    dequantParam(lnS->config->layerNorm->beta, betaS, 3, true);
    dequantParam(lin1S->config->linear->weights, w1S, 6, true);
    dequantParam(lin1S->config->linear->bias, b1S, 2, true);

    /* ---- FLOAT32 twin (reference) ---- */
    quantization_t *fQ = quantizationInitFloat();
    layerQuant_t lqF;
    layerQuantInitUniform(&lqF, fQ);
    layer_t *lin0F = buildTrainableLinear(4, 3, W0_VALS, B0_VALS, fQ, false);
    layer_t *lnF = layerNormLayerInit(
        &(layerNormInit_t){.normalizedShape = normShape, .numNormDims = 1, .eps = 1e-5f}, &lqF);
    layer_t *lin1F = buildTrainableLinear(3, 2, W1_VALS, B1_VALS, fQ, false);
    layer_t *modelF[3] = {lin0F, lnF, lin1F};

    TEST_ASSERT_TRUE_MESSAGE(validateModelQuantization(modelF, 3),
                             "FLOAT-only twin must pass the validator");

    tensor_t *inF = build2DFloat(2, 4, INPUT_VALS, 8);
    tensor_t *labelF = build2DFloat(2, 2, LABEL_VALS, 4);

    optimizer_t *optimF = sgdMCreateOptim(0.1f, 0.0f, 0.0f, modelF, 3, FLOAT32);
    trainingStats_t *statsF =
        calculateGradsSequential(modelF, 3, defaultLossConfig(MSE), REDUCTION_MEAN, inF, labelF);
    sgdStepM(optimF);

    float w0F[12], b0F[3], gammaF[3], betaF[3], w1F[6], b1F[2];
    dequantParam(lin0F->config->linear->weights, w0F, 12, false);
    dequantParam(lin0F->config->linear->bias, b0F, 3, false);
    dequantParam(lnF->config->layerNorm->gamma, gammaF, 3, false);
    dequantParam(lnF->config->layerNorm->beta, betaF, 3, false);
    dequantParam(lin1F->config->linear->weights, w1F, 6, false);
    dequantParam(lin1F->config->linear->bias, b1F, 2, false);

    freeTrainingStats(statsF);
    freeTrainingStats(statsS);
    freeTensor(labelF);
    freeTensor(inF);
    freeTensor(labelS);
    freeTensor(inS);
    freeOptimSgdM(optimF); /* frees w0F/b0F, gammaF/betaF, w1F/b1F parameter_t */
    freeOptimSgdM(optimS);
    freeLinearLayerShell(lin1F);
    freeLayerNormLayerShell(lnF);
    freeLinearLayerShell(lin0F);
    freeLinearLayerShell(lin1S);
    freeQuantLayer(quant2);
    freeLayerNormLayerShell(lnS);
    freeQuantLayer(quant1);
    freeLinearLayerShell(lin0S);
    freeQuantization(fQ);
    freeQuantization(symQ);

    TEST_ASSERT_TRUE(symStatsNotNull);
    TEST_ASSERT_TRUE_MESSAGE(isfinite(symLoss), "SYM chain loss must be finite");
    for (size_t i = 0; i < 3; i++) {
        TEST_ASSERT_TRUE_MESSAGE(fabsf(gammaF[i] - 1.0f) > 1e-6f,
                                 "reference gamma must actually train");
    }
    for (size_t i = 0; i < 12; i++) {
        TEST_ASSERT_FLOAT_WITHIN(CHAIN_TOL, w0F[i], w0S[i]);
    }
    for (size_t i = 0; i < 3; i++) {
        TEST_ASSERT_FLOAT_WITHIN(BIAS_TOL, b0F[i], b0S[i]);
        TEST_ASSERT_FLOAT_WITHIN(CHAIN_TOL, gammaF[i], gammaS[i]);
        TEST_ASSERT_FLOAT_WITHIN(CHAIN_TOL, betaF[i], betaS[i]);
    }
    for (size_t i = 0; i < 6; i++) {
        TEST_ASSERT_FLOAT_WITHIN(CHAIN_TOL, w1F[i], w1S[i]);
    }
    for (size_t i = 0; i < 2; i++) {
        TEST_ASSERT_FLOAT_WITHIN(BIAS_TOL, b1F[i], b1S[i]);
    }
}

void testValidatorRejectsChainWithoutQuantLayers(void) {
    quantization_t *symQ = quantizationInitSymInt32(HALF_AWAY);
    layerQuant_t lqSym = {
        .forwardMath = symQ, .backwardMath = symQ, .weightStorage = symQ, .biasStorage = symQ};
    size_t normShape[1] = {3};
    layer_t *lin0 = buildTrainableLinear(4, 3, W0_VALS, B0_VALS, symQ, true);
    layer_t *ln = layerNormLayerInit(
        &(layerNormInit_t){.normalizedShape = normShape, .numNormDims = 1, .eps = 1e-5f}, &lqSym);
    layer_t *lin1 = buildTrainableLinear(3, 2, W1_VALS, B1_VALS, symQ, true);
    layer_t *model[3] = {lin0, ln, lin1};

    /* Linear-SYM raw accumulator would feed LayerNorm directly — do NOT train. */
    bool valid = validateModelQuantization(model, 3);

    freeParameter(lin1->config->linear->weights);
    freeParameter(lin1->config->linear->bias);
    freeLinearLayerShell(lin1);
    freeLayerNormLayer(ln); /* factory free: gamma/beta not registered with any optimizer */
    freeParameter(lin0->config->linear->weights);
    freeParameter(lin0->config->linear->bias);
    freeLinearLayerShell(lin0);
    freeQuantization(symQ);

    TEST_ASSERT_FALSE_MESSAGE(valid, "missing Quant layers must be rejected by the validator");
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(testSgdMCreateOptimSkipsQuantizationLayer);
    RUN_TEST(testTrainingStepLinearSymThenQuantRequantsOutputAndFiniteLoss);
    RUN_TEST(testInferenceLinearSymThenQuantOutputsInt16RangeMantissas);
    RUN_TEST(testFullSymChainTrainingStepMatchesFloatTwin);
    RUN_TEST(testValidatorRejectsChainWithoutQuantLayers);
    return UNITY_END();
}
